
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from models.prod_functions import LFTurbine,HFTurbine
from models.mf_controls import MF_Turbine,compute_outputs,valid_extension
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from weis.glue_code.mpi_tools import MPI
import pickle,dill,copy
import time as timer

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15
format = '.pdf'

if __name__ == '__main__':

    if MPI:
        from weis.glue_code.mpi_tools import map_comm_heirarchical,subprocessor_loop, subprocessor_stop

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/RM1_300' + os.sep + 'openfast_runs'
    wind_dataset = OF_dir + os.sep + 'wind_dataset.pkl'
    mhk = True

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

            bounds = {'omega_pc' : np.array([0.1, 1.5]),'zeta_pc' : np.array([0.1,3.0]),'Kp_float': np.array([0,4]),'ptfm_freq':np.array([0,1])}
            desvars = {'omega_pc':np.array([0.9]),'zeta_pc' : np.array([0.7]),'Kp_float':np.array([0.96]),'ptfm_freq':np.array([0.6613])}
            scaling_dict = {'Kp_float':0.1}
            nvar = len(desvars.keys())

            t_transition = 50


        else:

            # 1. DFSM file and the model detials
            dfsm_file = this_dir + os.sep + 'dfsm_iea15_65.pkl'

            reqd_states = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed']
            reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
            reqd_outputs = ['TwrBsFxt','TwrBsMyt','GenPwr','YawBrTAxp','NcIMURAys','RtFldCp','RtFldCt']

            
            # 3. ROSCO yaml file
            rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

            bounds = np.array([[1, 3],[0.6,3.0],[-4,0],[0,4]])
            bounds = {'omega_pc' : np.array([1, 3]),'zeta_pc' : np.array([0.6,3.0]),'Kp_float': np.array([-4,0]),'ptfm_freq':np.array([0,4])}
            desvars = {'omega_pc':np.array([1]),'zeta_pc' : np.array([2.61]),'Kp_float':np.array([-4.9]),'ptfm_freq':np.array([0.2])}
            scaling_dict = {'omega_pc':10,'Kp_float':0.1,'ptfm_freq':10}
            nvar = len(desvars.keys())

            t_transition = 200

    if color_i == 0:

        if MPI:
            mpi_options = {}
            mpi_options['mpi_run'] = True 
            mpi_options['mpi_comm_map_down'] = comm_map_down

        else:

            mpi_options = None
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,mpi_options=mpi_options,transition_time=t_transition,wind_dataset=wind_dataset,mhk = mhk)
        nvar = len(desvars.keys());print(nvar)

        # w1 = np.linspace(1,0,n_pts)
        # w2 = 1-w1

        obj1 = 'TwrBsMyt_DEL'
        obj2 = 'GenSpeed_Std'

        results_folder = OF_dir + os.sep + 'multi_fid_results_4var_cobyla2'

        lf_results_folder = OF_dir + os.sep + 'low_fid_results_4var'

        use_lf_results = False
        use_prev_pt = False

        if use_lf_results:
            lf_results_file = lf_results_folder + os.sep +'lf_results.pkl'

            with open(lf_results_file,'rb') as handle:
                lf_results = pickle.load(handle)

            opt_pts_lf = lf_results['opt_pts']
            w1 = lf_results['w1']
            w2 = lf_results['w2']

            n_pts = len(w1)

        else:
            n_pts = 10
            dt_ = 1/n_pts

            w2 = np.linspace(0,1,n_pts)
            w1 = 1-w2

        objs = np.zeros((n_pts,2))
        opt_pts = np.zeros((n_pts,nvar))

        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        lf_warmstart_file = OF_dir + os.sep +'lf_ws_file_LHC_4.dill'
        hf_warmstart_file = OF_dir + os.sep +'hf_ws_file_LHC_4.dill'

        for i_pt in range(n_pts):

            if mhk:
                multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}
                lf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_lf_'+ str(i_pt)+'.dill'
                hf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_hf_'+ str(i_pt)+'.dill'

                lf_warmstart_file_ = lf_warmstart_file
                hf_warmstart_file_ = hf_warmstart_file


                with open(lf_warmstart_file_,'rb') as handle:
                    lf_res = dill.load(handle)

                with open(hf_warmstart_file_,'rb') as handle:
                    hf_res = dill.load(handle)

                lf_res_iter = copy.deepcopy(lf_res)
                hf_res_iter = copy.deepcopy(hf_res)

                DV0 = np.array(hf_res['desvars'])[-1,:]

                print(DV0)

                
                for i in range(len(lf_res_iter['outputs'])):
                    lf_res_iter['outputs'][i]['wt_objectives'] = lf_res_iter['outputs'][i][obj1]*w1[i_pt] + lf_res_iter['outputs'][i][obj2]*w2[i_pt]

                for i in range(len(hf_res_iter['outputs'])):
                    hf_res_iter['outputs'][i]['wt_objectives'] = hf_res_iter['outputs'][i][obj1]*w1[i_pt] + hf_res_iter['outputs'][i][obj2]*w2[i_pt]

                with open(lf_warmstart_file_iter,'wb') as handle:
                    dill.dump(lf_res_iter,handle)

                with open(hf_warmstart_file_iter,'wb') as handle:
                    dill.dump(hf_res_iter,handle)


                num_init_pts = 0

            else:
                multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}

                lf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_lf_'+ str(i_pt)+'.dill'
                hf_warmstart_file_iter = results_folder + os.sep+'multiobj_iter_hf_'+ str(i_pt)+'.dill'

                lf_warmstart_file_ = lf_warmstart_file
                hf_warmstart_file_ = hf_warmstart_file


                with open(lf_warmstart_file_,'rb') as handle:
                    lf_res = dill.load(handle)

                with open(hf_warmstart_file_,'rb') as handle:
                    hf_res = dill.load(handle)

                lf_res_iter = copy.deepcopy(lf_res)
                hf_res_iter = copy.deepcopy(hf_res)

                DV0 = np.array(hf_res['desvars'])[-1,:]

                
                for i in range(len(lf_res_iter['outputs'])):
                    lf_res_iter['outputs'][i]['wt_objectives'] = lf_res_iter['outputs'][i][obj1]*w1[i_pt] + lf_res_iter['outputs'][i][obj2]*w2[i_pt]

                for i in range(len(hf_res_iter['outputs'])):
                    hf_res_iter['outputs'][i]['wt_objectives'] = hf_res_iter['outputs'][i][obj1]*w1[i_pt] + hf_res_iter['outputs'][i][obj2]*w2[i_pt]

                with open(lf_warmstart_file_iter,'wb') as handle:
                    dill.dump(lf_res_iter,handle)

                with open(hf_warmstart_file_iter,'wb') as handle:
                    dill.dump(hf_res_iter,handle)

                num_init_pts = 0


            np.random.seed(123)

            model_low = LFTurbine(desvars,  mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict,warmstart_file = lf_warmstart_file_iter)
            model_high = HFTurbine(desvars, mf_turb, scaling_dict = scaling_dict,multi_fid_dict=multi_fid_dict, warmstart_file = hf_warmstart_file_iter)

            trust_region = SimpleTrustRegion(
                model_low,
                model_high,
                bounds = bounds,
                disp=2,
                trust_radius=0.5,
                num_initial_points=num_init_pts,
                radius_tol = 1e-2,
                optimization_log = True,
                log_filename = results_folder + os.sep +'MO_DEL_STD_'+str(i_pt)+'.txt'
            )

            if mhk:

                
                trust_region.add_objective("wt_objectives", scaler = 1e-0)
                trust_region.design_vectors = np.array(hf_res_iter['desvars'])
                trust_region.set_initial_point(np.array([0.9,0.7,0.96,0.66]))



            else:

                trust_region.design_vectors = np.array(hf_res_iter['desvars'])
                trust_region.add_objective("wt_objectives", scaler = 1e-0)


                if use_lf_results:
                    init_pt = opt_pts_lf[i_pt,:]

                    for i,key in enumerate(desvars.keys()):
                        if key in scaling_dict:
                            init_pt[i] = init_pt[i]*scaling_dict[key]

                    print(init_pt)
                    trust_region.set_initial_point(init_pt)

                else:

                    if use_prev_pt and i_pt > 0:
                        trust_region.set_initial_point(opt_pts[i_pt-1,:])
                    else:
                        trust_region.set_initial_point(np.array([0.9,0.7,0.96,0.66]))


            t1 = timer.time()
            trust_region.optimize(plot=False, num_basinhop_iterations=0,num_iterations = 40)
            t2 = timer.time()

            opt_pts[i_pt,:] = trust_region.design_vectors[-1,:]
            objs[i_pt,0] = trust_region.model_high.run(opt_pts[i_pt,:])[obj1]
            objs[i_pt,1] = trust_region.model_high.run(opt_pts[i_pt,:])[obj2]

            
        fig,ax = plt.subplots()
        ax.plot(objs[:,0],objs[:,1],'.',color = 'k',markersize = markersize)
        ax.set_xlabel(obj1,fontsize = fontsize_axlabel)
        ax.set_ylabel(obj2,fontsize = fontsize_axlabel)
        ax.tick_params(labelsize=fontsize_tick)
        ax.grid()

        fig.savefig(results_folder + os.sep +'DELvsSTD.pdf')




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

    