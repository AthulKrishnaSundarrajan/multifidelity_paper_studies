import numpy as np
import matplotlib.pyplot as plt
import pickle
import smt.surrogate_models as smt

if __name__ == '__main__':

    pklfile = 'approx_func_test.pkl'

    with open(pklfile,'rb') as handle:
        approx_dict = pickle.load(handle)

    dv = approx_dict['dv']
    outputs_high = approx_dict['outputs_high']
    outputs_low = approx_dict['outputs_low']
    dv_low = np.min(dv);dv_high = np.max(dv)
    omega_ = np.linspace(dv_low,dv_high,50)

    markersize = 20

    fig,ax = plt.subplots(1)

    sm = smt.KPLS(print_global=False, theta0=[1e-1])
    sm.set_training_values(dv, outputs_high['TwrBsMyt_DEL'] - outputs_low['TwrBsMyt_DEL'])
    sm.train()

    sm_low = smt.KPLS(print_global=False, theta0=[1e-1])
    sm_low.set_training_values(dv, outputs_low['TwrBsMyt_DEL'])
    sm_low.train()

    diff = sm.predict_values(omega_)
    lf = sm_low.predict_values(omega_)

    ax.plot(dv,outputs_high['TwrBsMyt_DEL'],'.',markersize = markersize,label = 'OpenFAST') 
    ax.plot(dv,outputs_low['TwrBsMyt_DEL'],'.',markersize = markersize,label = 'DFSM') 
    ax.plot(omega_,lf+diff,'-',color = 'k',label = 'diff+LF')
    ax.set_xlabel("PC_Omega");ax.set_ylabel('DEL')
    ax.legend(ncol = 3)

    fig,ax = plt.subplots(1)

    sm = smt.KPLS(print_global=False, theta0=[1e-1])
    sm.set_training_values(dv, outputs_high['GenSpeed_Max'] - outputs_low['GenSpeed_Max'])
    sm.train()
    

    sm_low = smt.KPLS(print_global=False, theta0=[1e-1])
    sm_low.set_training_values(dv, outputs_low['GenSpeed_Max'])
    sm_low.train()

    diff = sm.predict_values(omega_)
    lf = sm_low.predict_values(omega_)

    ax.plot(dv,outputs_high['GenSpeed_Max'],'.',markersize = markersize,label = 'OpenFAST') 
    ax.plot(dv,outputs_low['GenSpeed_Max'],'.',markersize = markersize,label = 'DFSM') 
    ax.set_xlabel("PC_Omega");ax.set_ylabel('GS MAX')
    ax.plot(omega_,lf + diff,'-',color = 'k',label = 'diff + LF')
    ax.legend(ncol = 3)

    plt.show()

