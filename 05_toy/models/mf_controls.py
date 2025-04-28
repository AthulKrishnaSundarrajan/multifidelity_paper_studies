# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
import pickle
import smt.surrogate_models as smt
from matplotlib import colors


import numpy as np
import sys, os, platform, yaml

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 12


key_metrics = ['DEL','AEP','GenSpeed_std','PtfmPitch_std','PtfmPitch_max_mean','PtfmPitch_max','rotor_overspeed',
               'pitch_travel','pitch_duty_cycle','max_pitch_rate']

class MF_Turbine(object):

    def __init__(self,results_file,obj= 'DEL',const = 'rotor_overspeed'):
        self.results_file = results_file
        self.obj = obj
        self.const = const

        with open(results_file,'rb') as handle:
            results = pickle.load(handle)

        key_metrics_of = results['key_metrics_openfast']
        key_metrics_dfsm = results['key_metrics_dfsm']
        DV = results['DV']
        self.DV = DV

        obj_ind = key_metrics.index(obj)
        const_ind = key_metrics.index(const)

        obj_of = key_metrics_of[:,obj_ind]
        obj_dfsm = key_metrics_dfsm[:,obj_ind]

        con_of = key_metrics_of[:,const_ind]
        con_dfsm = key_metrics_dfsm[:,const_ind]

        print('Training surrogate models')

        sm_obj_of = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj_of.set_training_values(DV,obj_of)
        sm_obj_of.train()

        sm_obj_dfsm = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj_dfsm.set_training_values(DV,obj_dfsm)
        sm_obj_dfsm.train()

        sm_con_of = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_con_of.set_training_values(DV,con_of)
        sm_con_of.train()

        sm_con_dfsm = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_con_dfsm.set_training_values(DV,con_dfsm)
        sm_con_dfsm.train()

        self.sm_obj_of = sm_obj_of
        self.sm_obj_dfsm = sm_obj_dfsm

        self.sm_con_of = sm_con_of
        self.sm_con_dfsm = sm_con_dfsm


    def compare(self):
        
        DV = self.DV
        n_pts = len(DV)

        omega_pc = np.unique(DV[:,0])
        zeta_pc = np.unique(DV[:,1])

        O,Z = np.meshgrid(omega_pc,zeta_pc)

        n_samples = len(omega_pc)

        metric_of = np.zeros((n_pts,))
        metric_dfsm = np.zeros((n_pts,))

        for i_pt in range(n_pts):

            metric_of[i_pt] = self.sm_obj_of.predict_values(np.atleast_2d(DV[i_pt,:]))
            metric_dfsm[i_pt] = self.sm_obj_dfsm.predict_values(np.atleast_2d(DV[i_pt,:]))

        max_dfsm = np.max(metric_dfsm)
        max_of = np.max(metric_of)

        metric_dfsm = np.reshape(metric_dfsm,[n_samples,n_samples])
        metric_of = np.reshape(metric_of,[n_samples,n_samples])

        datasets = [metric_dfsm/max_dfsm,metric_of/max_of]

        # create a single norm to be shared across all images
        norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

        fig,axs = plt.subplots(1,2,constrained_layout = True)
        fig.suptitle('Obj',fontsize = fontsize_axlabel)

        images = []
        for ax, data in zip(axs, datasets):
            images.append(ax.contourf(O,Z,data, norm=norm))

        
        axs[0].set_xlabel('Omega PC',fontsize = fontsize_axlabel)
        axs[0].set_ylabel('Zeta PC',fontsize = fontsize_axlabel)
        axs[0].set_title('DFSM',fontsize = fontsize_axlabel)   
        
        axs[1].set_xlabel('Omega PC',fontsize = fontsize_axlabel)
        axs[1].set_title('OpenFAST',fontsize = fontsize_axlabel)

        fig.colorbar(images[0],ax = axs,orientation='horizontal')

        plt.show()


    def run_level2(self,dv):

        outputs = {}
        outputs[self.obj] = self.sm_obj_dfsm.predict_values(np.atleast_2d(dv))[0,0]
        outputs[self.const] = self.sm_con_dfsm.predict_values(np.atleast_2d(dv))[0,0]

        return outputs
        


    def run_level3(self,dv):

        outputs = {}
        
        outputs[self.obj] = self.sm_obj_of.predict_values(np.atleast_2d(dv))[0,0]
        outputs[self.const] = self.sm_con_of.predict_values(np.atleast_2d(dv))[0,0]
        
        return outputs
        


class Level3_Turbine(object):
    
    def __init__(self,mf_turb):
        self.mf_turb = mf_turb

    def compute(self,dv):
        outputs = self.mf_turb.run_level3(dv)

        return outputs



class Level2_Turbine(object):

    def __init__(self,mf_turb):
        self.mf_turb = mf_turb

    def compute(self,dv):
        outputs = self.mf_turb.run_level2(dv)

        return outputs

