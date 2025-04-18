'''
Functions for running linear and nonlinear control parameter optimizations

- Run full set of DLCs
- Process and find Worst Case
- Nonlinear
    - Tune ROSCO, update
    - Run single, worst case DLC
- Linear  (currently: only doing this!)
    - Generate linear model from nonlinear simulation
    - Tune linear ROSCO
    - Run linear simulation
- Process DEL, other measures for cost function

'''
from openfast_io.FAST_reader   import InputReader_OpenFAST as reader

from pCrunch import Crunch,FatigueParams, AeroelasticOutput



from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox.utilities import write_DISCON,read_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco import discon_lib_path
from openfast_io.turbsim_file   import TurbSimFile


# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
import numpy as np
import sys, os, platform, yaml
import fnmatch
import pickle
from scipy.interpolate import CubicSpline

from .run_dfsm import run_dfsm

def valid_extension(fp,ext):
    return any([fnmatch.fnmatch(fp,ext_) for ext_ in [ext]])

def compute_rot_avg(u,y,z,t,R,HubHt):
    ''' 
    Compute rotor average wind speed, where R is the rotor radius
    '''

    rot_avg = np.zeros((3,len(t)))
    
    for i in range(3):
        u_      = u[i,:,:,:]
        yy, zz = np.meshgrid(y,z)
        rotor_ind = np.sqrt(yy**2 + (zz - HubHt)**2) < R

        u_rot = []
        for u_plane in u_:
            u_rot.append(u_plane[rotor_ind].mean())

        rot_avg[i,:] = u_rot

    return rot_avg

class MF_Turbine(object):
    '''
    Multifidelity turbine object:
    - Level 2 linear openfast model
    - Level 3 full nonlinear openfast simulation

    Both models use the same wind inputs, via case_inputs, iec attributes

    '''

    def __init__(self,dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,ode_method = 'RK4',transition_time = 0,mpi_options = None):


        self.dfsm_file = dfsm_file
        self.ode_method = ode_method
        self.transition_time = transition_time

        if mpi_options == None:
            self.mpi_options = {'mpi_run':False}
        else:
            self.mpi_options = mpi_options

        with open(dfsm_file,'rb') as handle:
            self.dfsm = pickle.load(handle)

        self.reqd_states = reqd_states
        self.nx = 2*len(reqd_states)
        self.reqd_controls = reqd_controls
        self.reqd_outputs = reqd_outputs
        self.ny = len(reqd_outputs)

        self.channels = reqd_states + reqd_controls + reqd_outputs

        # save openfast directory
        self.OF_dir = OF_dir

        # get the list of .fst files
        fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]
        fst_files = sorted(fst_files)

        self.fst_files = fst_files
        self.n_cases = len(fst_files)


        # get a list of cp_ct_cq files
        cp_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*_Cp_Ct_Cq.txt')]
        cp_files = sorted(cp_files)

        self.cp_files = cp_files

        # get a list of wave elev files
        wv_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*_Wave.Elev')]
        wv_files = sorted(wv_files)

        self.wv_files = wv_files

        # get a list of DISCON files
        discon_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*_DISCON.IN')]
        discon_files = sorted(discon_files)

        self.discon_files = discon_files

        # Setup the controller object

        inputs = load_rosco_yaml(rosco_yaml)

        path_params         = inputs['path_params']
        turbine_params      = inputs['turbine_params']
        controller_params   = inputs['controller_params']

        turbine = ROSCO_turbine.Turbine(turbine_params)

        turbine.load_from_fast(
        fst_files[0],
        OF_dir,
        rot_source='txt',txt_filename=cp_files[0]
        )

        self.turbine = turbine
        self.controller_params = controller_params

        # Tune controller
        self.tune_and_write_files() 

        self.generate_case_data()

    def tune_and_write_files(self,omega_pc = None):

        turbine = self.turbine
        controller_params = self.controller_params

        if not(omega_pc == None):
            controller_params['omega_pc'] = omega_pc

        discon_files = self.discon_files
        cp_files = self.cp_files

        controller      = ROSCO_controller.Controller(controller_params)
        controller.tune_controller(turbine)

        for i_case in range(self.n_cases):
            write_DISCON(
            turbine,controller,
            param_file=discon_files[i_case], 
            txt_filename=cp_files[i_case]
            )

        self.controller = controller

    def generate_case_data(self):

        n_cases  = self.n_cases
        case_data_all = []

        for i_case in range(n_cases):

            case_data = {}
            case_data['case'] = i_case

            fst_file = self.fst_files[i_case]

            reader_case = reader()
            reader_case.FAST_InputFile = fst_file
            reader_case.FAST_directory = ''

            reader_case.read_MainInput()

            ed_file = self.OF_dir + os.sep + reader_case.fst_vt['Fst']['EDFile']
            
            reader_case.FAST_directory = self.OF_dir
            reader_case.read_ElastoDyn(ed_file)
            reader_case.read_InflowWind()
            reader_case.read_ServoDyn()
            reader_case.read_DISCON_in()

            ts_file = reader_case.fst_vt['InflowWind']['FileName_BTS']
            wv_file = self.wv_files[i_case]

            

            dt = reader_case.fst_vt['Fst']['DT']
            t0 = 0
            tf = reader_case.fst_vt['Fst']['TMax']

            VS_GenEff = reader_case.fst_vt['DISCON_in']['VS_GenEff']
            self.GB_ratio = reader_case.fst_vt['DISCON_in']['WE_GearboxRatio']
            VS_RtPwr = reader_case.fst_vt['DISCON_in']['VS_RtPwr']
            n_blades = reader_case.fst_vt['ElastoDyn']['NumBl']
            rotorD = 2*reader_case.fst_vt['ElastoDyn']['TipRad']
            hub_height = reader_case.fst_vt['ElastoDyn']['HubRad']  +reader_case.fst_vt['ElastoDyn']['TowerHt']

            wind_fun,wave_fun = self.load_datasets(ts_file,wv_file,rotorD,hub_height)
            bp0 = reader_case.fst_vt['ElastoDyn']['BlPitch1']

            x0 = np.zeros((self.nx,))

            for state in self.reqd_states:
                ind = self.reqd_states.index(state)

                try:
                    x0[ind] = reader_case.fst_vt['ElastoDyn'][state]
                except:
                    x0[ind] = 0

            ind = self.reqd_states.index('GenSpeed')
            x0[ind] = reader_case.fst_vt['ElastoDyn']['RotSpeed']*self.GB_ratio

            args = {'DT':dt,'num_blade':n_blades,'pitch':bp0}

            param = {}
            param['VS_GenEff'] = VS_GenEff
            param['WE_GearboxRatio'] = self.GB_ratio
            param['VS_RtPwr'] = VS_RtPwr
            param['time'] = [t0]
            param['dt']= dt
            param['blade_pitch'] = [bp0]
            param['gen_torque'] = [19000]
            param['t0'] = t0
            param['tf'] = tf 
            param['gen_speed_scaling'] = 1
            param['lib_name'] = discon_lib_path
            param['num_blade'] = n_blades
            param['ny'] = self.ny
            param['args'] = args
            param['param_filename'] = self.discon_files[i_case]
            param['w_fun'] = wind_fun
            param['wave_fun'] = wave_fun

            

            case_data = {}
            case_data['case'] = i_case
            case_data['param'] = param 
            case_data['dt'] = dt 
            case_data['x0'] = x0 
            case_data['tspan'] = [t0,tf]
            case_data['dfsm'] = self.dfsm
            case_data['ode_method'] = self.ode_method

            case_data_all.append(case_data)
            

        self.case_data_all = case_data_all

    

    def load_datasets(self,ts_file,wv_file,rotorD,hub_height):

        ts_file_     = TurbSimFile(ts_file)
        rot_avg = compute_rot_avg(ts_file_['u'],ts_file_['y'],ts_file_['z'],ts_file_['t'],rotorD,hub_height)
        u_h         = rot_avg[0,:]
        t_wind          = ts_file_['t']

        wind_fun = CubicSpline(t_wind,u_h)

        # Load data from the file
        loaded_time = []
        loaded_wave = []
        
        with open(wv_file, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                t, elev = map(float, line.strip().split())
                loaded_time.append(t)
                loaded_wave.append(elev)

        # Convert lists to NumPy arrays
        t_wv = np.array(loaded_time)
        wv_elev = np.array(loaded_wave)

        wave_fun = CubicSpline(t_wv,wv_elev)

        return wind_fun,wave_fun

    def run_dfsm(self):

        case_data_all = self.case_data_all

        cruncher,ae_output_list,chan_time_list = run_dfsm(case_data_all,self.reqd_states,self.reqd_controls,self.reqd_outputs,self.mpi_options,self.GB_ratio,self.transition_time)

        return cruncher,ae_output_list,chan_time_list




        
    def compare(self,dofs,overwrite=True):
        ''' 
        Compare level 2 and 3 timeseries, for debugging purposes

        '''
        self.gen_level2_model(dofs, overwrite)

        # Extract disturbance
        dist = []
        for case in self.case_list:
            ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
            ts_file.compute_rot_avg(self.iec.D/2)
            u_h         = ts_file['rot_avg'][0,:]
            tt          = ts_file['t']
            dist.append({'Time':tt, 'Wind': u_h})

        # Run level 2
        self.run_level2(self.controller,dist)

        # Run level 3
        self.run_level3(self.controller, overwrite)

        # comparison plot, used to be in Level 2, not sure if information is here
        if True:
            comp_channels = ['RtVAvgxh','GenSpeed','BldPitch1','TwrBsMyt','PtfmPitch']
            fig = [None] * len(self.level3_out)
            ax = [None] * len(comp_channels)
            
            for iFig, (l2_out, l3_out) in enumerate(zip(self.level2_out,self.level3_out)):
                fig[iFig] = plt.figure()

                for iPlot, chan in enumerate(comp_channels):
                    ax[iPlot] = plt.subplot(len(comp_channels),1,iPlot+1)
                    # level 3 output
                    try:
                        ax[iPlot].plot(l3_out['Time'],l3_out[chan])
                    except:
                        print(chan + ' is not in OpenFAST OutList')

                    # level 2 output
                    try:
                        ax[iPlot].plot(l2_out['Time'],l2_out[chan])
                    except:
                        print(chan + ' is not in Linearization OutList')
                    ax[iPlot].set_ylabel(chan)
                    ax[iPlot].grid(True)
                    if not iPlot == (len(comp_channels) - 1):
                        ax[iPlot].set_xticklabels([])
                        
                ax[iPlot].set_xlabel('Time, seconds')

                fig[iFig].legend(('Level 3','Level 2'),ncol=2,loc=9)

            plt.savefig('L2_L3.png', bbox_inches='tight', dpi=450)
            
            return fig, ax


    def gen_level2_model(self,dofs,overwrite=True):
        '''
            dofs: list of strings representing ElastoDyn DOFs that will be linearized, including:
                    - FlapDOF1, FlapDOF2, EdgeDOF, TeetDOF, DrTrDOF, GenDOF, YawDOF
                    - TwFADOF1, TwFADOF2, TwSSDOF1, TwSSDOF2,
                    - PtfmSgDOF, PtfmSwDOF, PtfmHvDOF, PtfmRDOF, PtfmPDOF, PtfmYDOF
        '''
        lin_fast = LinearFAST(FAST_ver='OpenFAST', dev_branch=True)

        # fast info
        lin_fast.weis_dir                 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
        
        lin_fast.FAST_InputFile           = self.FAST_InputFile   # FAST input file (ext=.fst)
        lin_fast.FAST_directory           = self.FAST_directory
        lin_fast.FAST_runDirectory        = self.FAST_level2_directory
        lin_fast.FAST_linearDirectory     = self.FAST_level2_directory
        lin_fast.debug_level              = 2
        lin_fast.dev_branch               = True
        lin_fast.write_yaml               = True
        
        lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
        lin_fast.wind_speeds                 = self.level2_wind_speeds
        lin_fast.DOFs                       = dofs  # enable with 
        lin_fast.TMax                       = 1600   # should be 1000-2000 sec or more with hydrodynamic states
        lin_fast.NLinTimes                  = 12

        lin_fast.overwrite_outfiles       = overwrite

        # simulation setup
        lin_fast.cores                      = self.n_cores

        # overwrite steady & linearizations
        lin_fast.overwrite        = overwrite           # for debugging only
        
        # run OpenFAST linearizations
        lin_fast.gen_linear_model()

        self.LinearTurbine = lin_mod.LinearTurbineModel(
            self.FAST_level2_directory,
            lin_fast.case_name_list,
            lin_fast.NLinTimes,
            )

    def run_level2(self,controller,disturbance):
        controller.tune_controller(self.turbine)
        linCont             = lin_mod.LinearControlModel(controller)
        self.level2_out     = []
        for dist in disturbance:
            l2_out, _, P_cl = self.LinearTurbine.solve(dist,Plot=False,open_loop=False,controller=linCont)
            self.level2_out.append(l2_out)


    def run_level3(self,controller,overwrite=True):
        controller.tune_controller(self.turbine)
        # Run FAST cases
        fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)
        
        # Select Turbine Model
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.FAST_InputFile    = self.FAST_InputFile  # FAST input file (ext=.fst)

        fastBatch.debug_level       = 2
        fastBatch.overwrite_outfiles = overwrite        # for debugging purposes
        
        # Set control parameters
        discon_vt = ROSCO_utilities.DISCON_dict(self.turbine,controller)
        for discon_input in discon_vt:
            self.case_inputs[('DISCON_in',discon_input)] = {'vals': [discon_vt[discon_input]], 'group': 0}
            
        self.case_list, self.case_name_list, self.dlc_list = self.iec.execute(case_inputs=self.case_inputs)
        
        fastBatch.case_list         = self.case_list
        fastBatch.case_name_list    = self.case_name_list
        fastBatch.channels          = self.channels
        fastBatch.FAST_runDirectory = self.FAST_level3_directory
        fastBatch.post              = FAST_IO_timeseries

        if self.n_cores == 1:
            out = fastBatch.run_serial()
        else:
            out = fastBatch.run_multi(cores=self.n_cores)

        self.level3_batch   = fastBatch
        self.level3_out     = out


class Level3_Turbine(object):
    
    def __init__(self,mf_turb):
        self.mf_turb = mf_turb

    def compute(self,omega_pc):
        self.mf_turb.controller.omega_pc = omega_pc
        self.mf_turb.run_level3(self.mf_turb.controller)

        return compute_outputs(self.mf_turb.level3_out)



class DFSM_Turbine(object):

    def __init__(self,mf_turb):
        self.mf_turb        = mf_turb

    def compute(self,omega_pc):
        
        self.mf_turb.controller.omega_pc = omega_pc
        self.mf_turb.run_dfsm(self.mf_turb.controller,self.disturbance)

        outputs = compute_outputs(self.mf_turb.level2_out)

        return outputs

def compute_outputs(levelX_out):
    # compute Tower Base Myt DEL
    for lx_out in levelX_out:
        lx_out['meta'] = {}
        lx_out['meta']['name'] = 'placeholder'
    chan_info = [('TwrBsMyt',4)]
    la = Loads_Analysis()
    TwrBsMyt_DEL = la.get_DEL(levelX_out,chan_info)['TwrBsMyt'].tolist()

    # Generator Speed Measures
    GenSpeed_Max = [lx_out['GenSpeed'].max() for lx_out in levelX_out]
    GenSpeed_Std = [lx_out['GenSpeed'].std() for lx_out in levelX_out]

    # Platform pitch measures
    PtfmPitch_Max = [lx_out['PtfmPitch'].max() for lx_out in levelX_out]
    PtfmPitch_Std = [lx_out['PtfmPitch'].std() for lx_out in levelX_out]

    # save outputs
    outputs = {}
    outputs['TwrBsMyt_DEL']     = TwrBsMyt_DEL[0]
    outputs['GenSpeed_Max']     = GenSpeed_Max[0]
    outputs['GenSpeed_Std']     = GenSpeed_Std[0]
    outputs['PtfmPitch_Max']    = PtfmPitch_Max[0]
    outputs['PtfmPitch_Std']    = PtfmPitch_Std[0]

    return outputs

# if __name__ == '__main__':
#     # 0. Set up Model, using default input files
#     import time
#     s = time.time()

#     mf_turb = MF_Turbine()
#     mf_turb.n_cores = 4

#     dofs = ['GenDOF','TwFADOF1','PtfmPDOF']
#     # mf_turb.compare(dofs=dofs)

#     l2_turb = Level2_Turbine(mf_turb, dofs)
    
#     print('Time to train L2 model', time.time() - s)
#     s = time.time()
    
#     l2_outs = l2_turb.compute(.15)
    
#     print('l2_outs')
#     print(l2_outs)
    
#     print()
    
#     l2_outs = l2_turb.compute(.16)
    
#     print('l2_outs changed')
#     print(l2_outs)
    
    
#     print('Time to compute two L2 models', time.time() - s)
    
    
#     s = time.time()
#     l3_turb = Level3_Turbine(mf_turb)
    
#     print('Time to train L3 model', time.time() - s)
#     s = time.time()
    
#     l3_outs = l3_turb.compute(0.10082741)
    
#     print('Time to compute L3 model', time.time() - s)
#     s = time.time()
    
#     print('l3_outs')
#     print(l3_outs)
    
    # l3_turb = Level3_Turbine(mf_turb)
    # 
    # print()
    # 
    # l3_outs = l3_turb.compute(.16)
    # 
    # print('l3_outs changed')
    # print(l3_outs)
    # 
    # 
    
