import numpy as np
import matplotlib.pyplot as plt
import os,dill
import matplotlib.tri as tri
import smt.surrogate_models as smt


if __name__ == '__main__':

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/rated_p05_35' + os.sep + 'openfast_runs'

    # get folder for multifid results
    MF_results_fol = OF_dir + os.sep + 'multi_fid_results'

    # 
    iter = 3

    lf_file = MF_results_fol + os.sep + 'multiobj_iter_lf_'+str(iter)+'.dill'
    hf_file = MF_results_fol + os.sep + 'multiobj_iter_hf_'+str(iter)+'.dill'

    with open(hf_file,'rb') as handle:
        hf_results_iter = dill.load(handle)

    with open(lf_file,'rb') as handle:
        lf_results_iter = dill.load(handle)

    desvars_hf = np.array(hf_results_iter['desvars'])
    desvars_lf = np.array(lf_results_iter['desvars'])
    
    n_start = 15+1

    desvars_init = desvars_hf[:n_start,:]

    outputs_iter = np.zeros((n_start,2))

    for i in range(n_start):
        outputs_iter[i,0] = hf_results_iter['outputs'][i]['TwrBsMyt_DEL']
        outputs_iter[i,1] = lf_results_iter['outputs'][i]['TwrBsMyt_DEL']

    sm = smt.KRG(print_global=False, theta0=[1e-1])
    
    error = outputs_iter[:,0] - outputs_iter[:,1]
    sm.set_training_values(desvars_init, error)
    sm.train()

    error_pred = np.squeeze(sm.predict_values(desvars_hf[:n_start,:]))

    save_dir = this_dir + os.sep + 'debug_surrogate'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fig1,ax1 = plt.subplots(1)

    cbar = ax1.tricontourf(desvars_init[:,0],desvars_init[:,1],outputs_iter[:,0])
    ax1.plot(desvars_init[:,0],desvars_init[:,1],'.r')

    ax1.set_xlabel('Omega PC')
    ax1.set_ylabel('Zeta PC')
    fig1.colorbar(cbar)
    fig1.savefig(save_dir+os.sep+'obj_hf.pdf')

    fig1,ax1 = plt.subplots(1)

    cbar = ax1.tricontourf(desvars_init[:,0],desvars_init[:,1],outputs_iter[:,1])
    ax1.plot(desvars_init[:,0],desvars_init[:,1],'.r')

    ax1.set_xlabel('Omega PC')
    ax1.set_ylabel('Zeta PC')
    fig1.colorbar(cbar)
    fig1.savefig(save_dir+os.sep+'obj_lf.pdf')


    fig1,ax1 = plt.subplots(1)
    error_c0 = outputs_iter[:,1]+error_pred
    cbar = ax1.tricontourf(desvars_init[:,0],desvars_init[:,1],error_c0)
    ax1.plot(desvars_init[:,0],desvars_init[:,1],'.r')

    ax1.set_xlabel('Omega PC')
    ax1.set_ylabel('Zeta PC')
    fig1.colorbar(cbar)
    fig1.savefig(save_dir+os.sep+'obj_c.pdf')

    iter_0 = desvars_hf[n_start-1,:]
    iter_1 = desvars_hf[n_start,:]

    ind_lf_0 = ((desvars_lf == iter_0))
    ind_lf_0 = np.logical_and(ind_lf_0[:,0],ind_lf_0[:,1])
    ind_lf_0 = np.argwhere(ind_lf_0)[0,0]

    ind_lf_1 = ((desvars_lf == iter_1))
    ind_lf_1 = np.logical_and(ind_lf_1[:,0],ind_lf_1[:,1])
    ind_lf_1 = np.argwhere(ind_lf_1)[0,0]

    op_lf_iter0 = lf_results_iter['outputs'][n_start]['TwrBsMyt_DEL'] 
    op_hf_iter0 = hf_results_iter['outputs'][n_start]['TwrBsMyt_DEL'] 

    ind_lf_01 = desvars_lf[ind_lf_0:ind_lf_1,:]

    len_01 = len(ind_lf_01)
    op_lf_01 = np.zeros((len_01))
    ind = 0
    for i in range(ind_lf_0,ind_lf_1):
        op_lf_01[ind] = lf_results_iter['outputs'][i]['TwrBsMyt_DEL']
        ind+=1

    error_c = op_lf_01 + np.squeeze(sm.predict_values(ind_lf_01))

    fig1,ax1 = plt.subplots(1)
    
    cbar = ax1.tricontourf(desvars_lf[:ind_lf_1,0],desvars_lf[:ind_lf_1,1],np.hstack([error_c0[:-1],error_c]))
    ax1.plot(desvars_init[:,0],desvars_init[:,1],'.r')

    ax1.set_xlabel('Omega PC')
    ax1.set_ylabel('Zeta PC')
    fig1.colorbar(cbar)
    fig1.savefig(save_dir+os.sep+'obj_c1.pdf')

    breakpoint()
