
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from models.prod_functions import LFTurbine,HFTurbine
from models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from weis.glue_code.mpi_tools import MPI
import pickle
import time as timer

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/nearrated_5' + os.sep + 'openfast_runs'
    wind_dataset = OF_dir + os.sep + 'wind_dataset.pkl'

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

        # 1. DFSM file and the model detials
        dfsm_file = this_dir + os.sep + 'dfsm_fowt_1p6.pkl'

        reqd_states = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed']
        reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
        reqd_outputs = ['TwrBsFxt','TwrBsMyt','GenPwr','YawBrTAxp','NcIMURAys','RtFldCp','RtFldCt']

        
        # 3. ROSCO yaml file
        rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=200,wind_dataset=wind_dataset)
        bounds = {'omega_pc' : np.array([[1, 3]]),'zeta_pc' : np.array([[0.10, 3.0]])}
        desvars = {'omega_pc' : np.array([2.5]),'zeta_pc': np.array([2.5])}
        scaling_dict = {'omega_pc':10}

        model_low = LFTurbine(desvars,  mf_turb, scaling_dict = scaling_dict)
        model_high = HFTurbine(desvars, mf_turb, scaling_dict = scaling_dict)

        np.random.seed(123)

        trust_region = SimpleTrustRegion(
            model_low,
            model_high,
            bounds,
            disp=2,
            trust_radius=0.5,
            num_initial_points=2,
            radius_tol = 1e-3,
            optimization_log = True,
            log_filename = 'MO_DEL_JMD_all.txt'
        )

        trust_region.add_objective("TwrBsMyt_DEL", scaler = 1e-0)
        trust_region.add_constraint("GenSpeed_Max", upper=1.2)
        #trust_region.add_constraint("TwrBsMyt_DEL",upper = 3e6)
        trust_region.set_initial_point(np.array([2.5,2.5]))
        # trust_region.construct_approximations(interp_method = 'smt')
        # approx_functions = trust_region.approximation_functions


        # dv = trust_region.design_vectors
        # outputs_low = trust_region.outputs_low
        # outputs_high = trust_region.outputs_high

        # results_dict = {'dv':dv,'outputs_low':outputs_low,'outputs_high':outputs_high}

        # with open('approx_func_test.pkl','wb') as handle:
        #     pickle.dump(results_dict,handle)


        t1 = timer.time()
        trust_region.optimize(plot=False, num_basinhop_iterations=0,num_iterations = 40)
        t2 = timer.time()

        




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

    