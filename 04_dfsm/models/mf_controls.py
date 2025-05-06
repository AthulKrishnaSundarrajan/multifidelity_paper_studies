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
from weis.aeroelasticse.FAST_reader    import InputReader_OpenFAST as reader

from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox.utilities import write_DISCON,read_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco import discon_lib_path
from weis.aeroelasticse.turbsim_file   import TurbSimFile


# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
import numpy as np
import sys, os, platform, yaml
import fnmatch
import pickle
from scipy.interpolate import CubicSpline

from .run_dfsm import run_dfsm
from .run_openfast import run_openfast

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
        
        inputs = load_rosco_yaml(rosco_yaml,rank_0 = True)
        
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
    
    def run_openfast(self,overwrite_flag = False):

        cruncher,ae_output_list,chan_time_list = run_openfast(self.fst_files,self.reqd_states,self.reqd_controls,self.reqd_outputs,self.mpi_options,self.GB_ratio,self.transition_time,overwrite_flag = overwrite_flag)

        return cruncher,ae_output_list,chan_time_list


class Level3_Turbine(object):
    
    def __init__(self,mf_turb):
        self.mf_turb = mf_turb

    def compute(self,omega_pc):

        self.mf_turb.tune_and_write_files(omega_pc)
        summary_stats,DELs,_ = self.mf_turb.run_openfast(overwrite_flag = True)
        outputs = compute_outputs(summary_stats,DELs)

        return outputs



class DFSM_Turbine(object):

    def __init__(self,mf_turb):
        self.mf_turb        = mf_turb

    def compute(self,omega_pc):
        
        self.mf_turb.tune_and_write_files(omega_pc)
        summary_stats,DELs,chan_time_list = self.mf_turb.run_dfsm()

        outputs = compute_outputs(summary_stats,DELs)

        return outputs

def compute_outputs(summary_stats,DELs):
    
    w_mean = summary_stats['RtVAvgxh']['mean']
    n_cases  = len(w_mean)

    from pCrunch import PowerProduction
    pp = PowerProduction('I')
    prob = pp.prob_WindDist(w_mean, disttype='pdf')
    prob/=prob.sum()

    # save outputs
    if n_cases == 1:
        outputs = {}
        outputs['TwrBsMyt_DEL']     = DELs['TwrBsMyt'].iloc[0]
        outputs['GenSpeed_Max']     = summary_stats['GenSpeed']['max'].iloc[0]/7.5

    else:

        outputs = {}
        outputs['TwrBsMyt_DEL'] = np.sum(np.array(DELs['TwrBsMyt'])*prob)
        outputs['GenSpeed_Max'] = np.max(np.array(summary_stats['GenSpeed']['max']))/7.5
        
        

    return outputs

    
