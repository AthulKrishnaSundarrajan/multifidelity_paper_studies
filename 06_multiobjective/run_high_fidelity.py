import numpy as np
from models.prod_functions import LFTurbine,HFTurbine
from models.mf_controls import MF_Turbine,valid_extension
from time import time
from os import path
import openmdao.api as om
import os,sys
from weis.glue_code.mpi_tools import MPI
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver
import pickle


if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    bounds = {'omega_pc' : np.array([0.1, 0.3]),'zeta_pc' : np.array([0.6, 3.0]),'Kp_float': np.array([-40,0]),'ptfm_freq':np.array([0,0.4])}
    desvars = {'omega_pc':np.array([0.18]),'zeta_pc' : np.array([2.2]),'Kp_float':np.array([-15]),'ptfm_freq':np.array([0.2])}
    nvar = len(desvars.keys());print(nvar)

    scaling_dict = None #{'omega_pc':10}#{'omega_pc':10,'Kp_float':0.1,'ptfm_freq':10}
    obj1 = 'TwrBsMyt_DEL'
    obj2 = 'GenSpeed_Std'

    n_pts = 10
    dt_ = 1/n_pts

    w2 = np.linspace(0,1,n_pts)
    
    w1 = 1-w2

    print(w1)

    objs = np.zeros((n_pts,2))
    iter_list = np.zeros((n_pts,))
    opt_pts = np.zeros((n_pts,nvar))

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/rated_p05_51' + os.sep + 'openfast_runs'
    wind_dataset = OF_dir + os.sep + 'wind_dataset.pkl'
    mhk = False

    results_folder = OF_dir + os.sep + 'high_fid_results_4var'

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


    fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]

    n_OF_runs = len(fst_files)

    if MPI:
        
        # set number of design variables and finite difference variables as 1
        n_DV = 1; n_FD = 1

        # get maximum available cores
        max_cores = MPI.COMM_WORLD.Get_size()

        # get number of cases we will be running
        max_parallel_OF_runs = max([int(np.floor((max_cores - n_DV) / n_DV)), 1])
        n_OF_runs_parallel = min([int(n_OF_runs), max_parallel_OF_runs])

        olaf = False

        # get mapping
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(n_FD, n_OF_runs_parallel)

        rank    = MPI.COMM_WORLD.Get_rank()

        if rank < len(color_map):
            try:
                color_i = color_map[rank]
            except IndexError:
                raise ValueError('The number of finite differencing variables is {} and the correct number of cores were not allocated'.format(n_FD))
        else:
            color_i = max(color_map) + 1
        
        comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    else:
        color_i = 0
        rank = 0


    if rank == 0:

        if mhk:
            # 1. DFSM file and the model detials
            dfsm_file = this_dir + os.sep + 'dfsm_mhk.pkl'

            with open(dfsm_file,'rb') as handle:
                dfsm = pickle.load(handle) 

            # required states
            reqd_states = ['PtfmPitch','PtfmHeave','GenSpeed']
            
            # required controls
            reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
            
            # required outputs
            reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt'] 

            
            # 3. ROSCO yaml file
            rosco_yaml = this_dir + os.sep + 'RM1_MHK.rosco.yaml'

            bounds = np.array([[0.1, 1.5],[0.1,3.0],[0,40],[0,1]])
            bounds = {'omega_pc' : np.array([0.1, 1.5]),'zeta_pc' : np.array([0.1,3.0]),'Kp_float': np.array([0,40]),'ptfm_freq':np.array([0,1])}
            desvars = {'omega_pc':np.array([0.9]),'zeta_pc' : np.array([0.7]),'Kp_float':np.array([9.6]),'ptfm_freq':np.array([0.6613])}
            scaling_dict = None
            n_dims = len(desvars.keys())

            t_transition = 50


        else:

            # 1. DFSM file and the model detials
            dfsm_file = this_dir + os.sep + 'dfsm_iea15_51.pkl'

            reqd_states = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed']
            reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
            reqd_outputs = ['TwrBsFxt','TwrBsMyt','GenPwr','YawBrTAxp','NcIMURAys','RtFldCp','RtFldCt']

            
            # 3. ROSCO yaml file
            rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

            bounds = np.array([[1, 3],[0.6,3.0],[-4,0],[0,4]])
            bounds = {'omega_pc' : np.array([0.1, 0.3]),'zeta_pc' : np.array([0.6,3.0]),'Kp_float': np.array([-40,0]),'ptfm_freq':np.array([0,0.4])}
            desvars = {'omega_pc':np.array([0.12]),'zeta_pc' : np.array([2.25]),'Kp_float':np.array([-15]),'ptfm_freq':np.array([0.3])}
            scaling_dict = None #{'omega_pc':10,'Kp_float':0.1,'ptfm_freq':10}
            n_dims = len(desvars.keys())

            t_transition = 200


    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None


        for i_pt in range(n_pts):

            multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}
        

            class Model(om.ExplicitComponent):
                def initialize(self):
                    self.options.declare('desvars')
                    
                def setup(self):
                    desvars = self.options["desvars"]
                    for key in desvars:
                        self.add_input(key, val=desvars[key])
                    
                    self.add_output('TwrBsMyt_DEL', val=0.)
                    self.add_output('GenSpeed_Std', val=0.)
                    self.add_output('GenSpeed_Max', val=0.)
                    
                    self.multi_fid_dict = multi_fid_dict

                    if not(multi_fid_dict == None):
                        self.add_output('wt_objectives',val = 0.)
                    
                    mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=t_transition,wind_dataset=wind_dataset,mhk = mhk)
                    self.model = HFTurbine(desvars, mf_turb,multi_fid_dict = multi_fid_dict, scaling_dict = scaling_dict)

                def compute(self, inputs, outputs):
                    # desvars = self.options["desvars"]
                    model_outputs = self.model.compute(inputs)
                    outputs['TwrBsMyt_DEL'] = model_outputs['TwrBsMyt_DEL']
                    outputs['GenSpeed_Std'] = model_outputs['GenSpeed_Std']
                    outputs['GenSpeed_Max'] = model_outputs['GenSpeed_Max']

                    if not(multi_fid_dict == None):
                        outputs['wt_objectives'] = model_outputs['wt_objectives']

            
            
            p = om.Problem(model=om.Group(num_par_fd = 1),comm = comm_i,reports = False)
            model = p.model
            # model.approx_totals(method='fd', step=1e-3, form='central')
            comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

            for key in desvars:
                model.set_input_defaults(key, val=desvars[key])
                
            s = time()
            
            p.driver = NLoptDriver()
            p.driver.options['optimizer'] = "LN_COBYLA"
            p.driver.options['tol'] = 1e-2

            ikey = 0
            for key in desvars:
                model.add_design_var(key, lower=bounds[key][0], upper=bounds[key][1])
                ikey+=1

            
            model.add_objective('wt_objectives', ref=1.e0)
            #model.add_constraint('GenSpeed_Max',upper = 1.2)
            p.driver.recording_options['includes'] = ['*']
            p.driver.recording_options['record_objectives'] = True
            p.driver.recording_options['record_constraints'] = True
            p.driver.recording_options['record_desvars'] = True
            p.driver.recording_options['record_inputs'] = True
            p.driver.recording_options['record_outputs'] = True
            p.driver.recording_options['record_residuals'] = True

            p.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

            recorder = om.SqliteRecorder("cases.sql")
            p.driver.add_recorder(recorder)

            p.setup()
            p.run_driver()

            cr = om.CaseReader(p.get_outputs_dir() / "cases.sql")
            driver_cases = cr.list_cases('driver')

            n_iter = len(driver_cases)


            for i,key in enumerate(desvars.keys()):
                opt_pts[i_pt,i] = p.get_val(key)
            objs[i_pt,0] = p.get_val(obj1)
            objs[i_pt,1] = p.get_val(obj2)

            iter_list[i_pt] = n_iter

        print(opt_pts)
        print(objs)
        print(n_iter)

        results_dict = {'objs':objs,'opt_pts':opt_pts,'w1':w1,'w2':w2,'iter_list':iter_list}

        with open(results_folder+os.sep+'hf_results.pkl','wb') as handle:
            pickle.dump(results_dict,handle)



    #---------------------------------------------------
    # More MPI stuff
    #---------------------------------------------------

    if MPI and color_i < 1000000:
        sys.stdout.flush()
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)
        sys.stdout.flush()

        # close signal to subprocessors
        subprocessor_stop(comm_map_down)
        sys.stdout.flush()

    if MPI and color_i < 1000000:
        MPI.COMM_WORLD.Barrier()
