import numpy as np
import os,sys
import matplotlib.pyplot as plt
from models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from models.prod_functions import LFTurbine,HFTurbine
from weis.glue_code.mpi_tools import MPI
import pickle

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/test' + os.sep + 'openfast_runs'
    wind_dataset = OF_dir + os.sep + 'wind_dataset.pkl'

    fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]
    n_OF_runs = len(fst_files)

    run_sens_study = False
    
    bounds = np.array([[0.10, 3.0],[0,-40]])
    desvars = {'zeta_pc' : np.array([2]),'Kp_float': np.array([-9])}
    npts = 5
    

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
        
        mf_controls = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=00,wind_dataset=wind_dataset)

        model_low = LFTurbine(desvars,  mf_controls)
        model_high = HFTurbine(desvars, mf_controls)

        fig_fol = OF_dir + os.sep + 'plot_comp'
        if not os.path.exists(fig_fol):
            os.mkdir(fig_fol)

        if run_sens_study:

            n_samples = npts**2

            twrbsmyt_del = np.zeros((n_samples,2))
            pitch_travel = np.zeros((n_samples,2))
            genspeed_max = np.zeros((n_samples,2))
            genspeed_std = np.zeros((n_samples,2))
            ptfmpitch_max = np.zeros((n_samples,2))
            ptfmpitch_std = np.zeros((n_samples,2))
            p_avg = np.zeros((n_samples,2))

            omega_pc = np.linspace(bounds[0,0],bounds[0,1],npts)
            zeta_pc = np.linspace(bounds[1,0],bounds[1,1],npts)
            O,Z = np.meshgrid(omega_pc,zeta_pc)

            OZ = np.hstack([O,Z])

            OZ = np.reshape(OZ,[n_samples,2],order = 'F')



            

            for ipt in range(n_samples):
                dvar = {'zeta_pc':OZ[ipt,0],'Kp_float':OZ[ipt,1]}

                print(dvar)
                outputs_low = model_low.compute(dvar)
                outputs_high = model_high.compute(dvar)

                twrbsmyt_del[ipt,0] = outputs_high['TwrBsMyt_DEL']
                genspeed_max[ipt,0] = outputs_high['GenSpeed_Max']
                pitch_travel[ipt,0] = outputs_high['avg_pitch_travel']
                genspeed_std[ipt,0] = outputs_high['GenSpeed_Std']
                ptfmpitch_max[ipt,0] = outputs_high['PtfmPitch_Max']
                ptfmpitch_std[ipt,0] = outputs_high['PtfmPitch_Std']
                p_avg[ipt,0] = outputs_high['P_avg']

                twrbsmyt_del[ipt,1] = outputs_low['TwrBsMyt_DEL']
                genspeed_max[ipt,1] = outputs_low['GenSpeed_Max']
                pitch_travel[ipt,1] = outputs_low['avg_pitch_travel']
                genspeed_std[ipt,1] = outputs_low['GenSpeed_Std']
                ptfmpitch_max[ipt,1] = outputs_low['PtfmPitch_Max']
                ptfmpitch_std[ipt,1] = outputs_low['PtfmPitch_Std']
                p_avg[ipt,1] = outputs_low['P_avg']


            print(model_high.n_count)
            print(model_low.n_count)
            print(twrbsmyt_del)
            

            results_dict = {'OZ':OZ,'twrbsmyt_del':twrbsmyt_del,'genspeed_max':genspeed_max,'pitch_travel':pitch_travel,'genspeed_std':genspeed_std,'ptfmpitch_max':ptfmpitch_max,'ptfmpitch_std':ptfmpitch_std,'P_avg':p_avg}
            with open('sensstudy_results_ZK.pkl','wb') as handle:
                pickle.dump(results_dict,handle)
                
        else:

            cruncher_dfsm,ae_output_list_dfsm,chan_time_list_dfsm = mf_controls.run_dfsm()

            outputs_dfsm = compute_outputs(cruncher_dfsm)
            print(outputs_dfsm)

            cruncher_of,ae_output_list_of,chan_time_list_of = mf_controls.run_openfast()
            outputs_of = compute_outputs(cruncher_of)
            print(outputs_of)

            i_fig = 0
            
            for ct_of,ct_dfsm in zip(chan_time_list_of,chan_time_list_dfsm):

                channels = mf_controls.channels

                for chan in channels:

                    fig,ax = plt.subplots(1)
                    fig.subplots_adjust(hspace = 0.65)

                    ax.plot(ct_of['Time'],ct_of[chan],label = 'OpenFAST')
                    ax.plot(ct_dfsm['Time'],ct_dfsm[chan],label = 'DFSM')
                    ax.legend(ncol = 2)

                    fig.savefig(fig_fol + os.sep +chan +'_comp'+str(i_fig)+'.pdf')
                    plt.close(fig)
                i_fig+=1

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

    


