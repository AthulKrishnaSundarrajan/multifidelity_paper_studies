import numpy as np
import os,sys
import matplotlib.pyplot as plt
from models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from models.prod_functions import LFTurbine,HFTurbine
from weis.glue_code.mpi_tools import MPI

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'transition2' + os.sep + 'openfast_runs'

    fst_files = [os.path.join(OF_dir,f) for f in os.listdir(OF_dir) if valid_extension(f,'*.fst')]

    n_OF_runs = len(fst_files)

    run_sens_study = True
    
    bounds = np.array([0.10, 0.3])
    desvars = {'pc_omega' : np.array([0.2])}
    

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
        
        mf_controls = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options)

        model_low = LFTurbine(desvars,  mf_controls)
        model_high = HFTurbine(desvars, mf_controls)

        fig_fol = OF_dir + os.sep + 'plot_comp'
        if not os.path.exists(fig_fol):
            os.mkdir(fig_fol)

        if run_sens_study:

            npts = 1

            twrbsmyt_del = np.zeros((npts,2))
            genspeed_max = np.zeros((npts,2))

            pc_omega = [0.106] #np.linspace(bounds[0],bounds[-1],npts)
            print(pc_omega)

            for ipt in range(npts):
                dvar = {'pc_omega':pc_omega[ipt]}
                print(dvar)
                outputs_low = model_low.compute(dvar)
                outputs_high = model_high.compute(dvar)

                twrbsmyt_del[ipt,0] = outputs_high['TwrBsMyt_DEL']
                genspeed_max[ipt,0] = outputs_high['GenSpeed_Max']

                twrbsmyt_del[ipt,1] = outputs_low['TwrBsMyt_DEL']
                genspeed_max[ipt,1] = outputs_low['GenSpeed_Max']

            print(model_high.n_count)
            print(model_low.n_count)

            fig,ax = plt.subplots(1)

            ax.plot(pc_omega,twrbsmyt_del[:,0],'.-',markersize = 10,label = 'HF')
            ax.plot(pc_omega,twrbsmyt_del[:,1],'.-',markersize = 10,label = 'LF')
            ax.set_xlabel('PC_Omega')
            ax.set_ylabel('TwrBsMyt')
            ax.legend(ncol = 2)


            fig.savefig(fig_fol + os.sep +'sens_DEL'+'.pdf')
            plt.close(fig)

            fig,ax = plt.subplots(1)

            ax.plot(pc_omega,genspeed_max[:,0],'.-',markersize = 10,label = 'HF')
            ax.plot(pc_omega,genspeed_max[:,1],'.-',markersize = 10,label = 'LF')
            ax.set_xlabel('PC_Omega')
            ax.set_ylabel('Genspeed Max')
            ax.legend(ncol = 2)


            fig.savefig(fig_fol + os.sep +'sens_GS'+'.pdf')
            plt.close(fig)

        else:

            sum_stats_dfsm,DELs_dfsm,chan_time_list_dfsm = mf_controls.run_dfsm()

            outputs_dfsm = compute_outputs(sum_stats_dfsm,DELs_dfsm)
            print(outputs_dfsm)

            sum_stats_of,DELs_of,chan_time_list_of = mf_controls.run_openfast()
            outputs_of = compute_outputs(sum_stats_of,DELs_of)
            print(outputs_of)

            i_fig = 0
            
            for ct_of,ct_dfsm in zip(chan_time_list_of,chan_time_list_dfsm):
            
                fig,ax = plt.subplots(3,1)
                fig.subplots_adjust(hspace = 0.65)

                ax[0].plot(ct_of['Time'],ct_of['RtVAvgxh'])
                ax[0].plot(ct_dfsm['Time'],ct_dfsm['RtVAvgxh'])

                ax[1].plot(ct_of['Time'],ct_of['TwrBsMyt'])
                ax[1].plot(ct_dfsm['Time'],ct_dfsm['TwrBsMyt'])

                ax[2].plot(ct_of['Time'],ct_of['BldPitch1'])
                ax[2].plot(ct_dfsm['Time'],ct_dfsm['BldPitch1'])

                fig.savefig(fig_fol + os.sep +'comp_'+str(i_fig)+'.pdf')
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

    


