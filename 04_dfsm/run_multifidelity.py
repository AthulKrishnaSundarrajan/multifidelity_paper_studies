
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
    OF_dir = this_dir + os.sep + 'transition' + os.sep + 'openfast_runs'

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

        reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
        reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
        reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt']

        
        # 3. ROSCO yaml file
        rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options)
        bounds = {'pc_omega' : np.array([[0.10, 0.3]])}
        desvars = {'pc_omega' : np.array([0.25])}

        model_low = LFTurbine(desvars,  mf_turb)
        model_high = HFTurbine(desvars, mf_turb)

        np.random.seed(123)

        trust_region = SimpleTrustRegion(
            model_low,
            model_high,
            bounds,
            disp=2,
            trust_radius=0.5,
            num_initial_points=3,
            radius_tol = 1e-3,
            optimization_log = True
        )

        trust_region.add_objective("TwrBsMyt_DEL", scaler=1e-5)
        trust_region.add_constraint("GenSpeed_Max", upper=1.2)
        # trust_region.construct_approximations(interp_method = 'smt')
        # approx_functions = trust_region.approximation_functions


        # dv = trust_region.design_vectors
        # outputs_low = trust_region.outputs_low
        # outputs_high = trust_region.outputs_high

        # results_dict = {'dv':dv,'outputs_low':outputs_low,'outputs_high':outputs_high}

        # with open('approx_func_test.pkl','wb') as handle:
        #     pickle.dump(results_dict,handle)


        t1 = timer.time()
        trust_region.optimize(plot=False, num_basinhop_iterations=0,num_iterations = 1)
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

    