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

    def __init__(self,results_file,obj1= 'twrbsmyt_del',obj2 = 'genspeed_std',const = 'genspeed_max'):
        self.results_file = results_file
        self.obj1 = obj1
        self.obj2 = obj2
        self.const = const

        with open(results_file,'rb') as handle:
            results = pickle.load(handle)

        objective_1 = results[obj1]
        objective_2 = results[obj2]
        constraint = results[const]
        

        DV = results['OZ']
        self.DV = DV

        obj1_of = objective_1[:,0]
        obj1_dfsm = objective_1[:,1]

        obj2_of = objective_2[:,0]
        obj2_dfsm = objective_2[:,1]

        const_of = constraint[:,0]
        const_dfsm = constraint[:,1]


        print('Training surrogate models')

        sm_obj1_of = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj1_of.set_training_values(DV,obj1_of)
        sm_obj1_of.train()

        sm_obj1_dfsm = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj1_dfsm.set_training_values(DV,obj1_dfsm)
        sm_obj1_dfsm.train()

        sm_obj2_of = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj2_of.set_training_values(DV,obj2_of)
        sm_obj2_of.train()

        sm_obj2_dfsm = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_obj2_dfsm.set_training_values(DV,obj2_dfsm)
        sm_obj2_dfsm.train()

        sm_const_of = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_const_of.set_training_values(DV,const_of)
        sm_const_of.train()

        sm_const_dfsm = smt.KPLS(print_global = False, theta0 = [1e-1])
        sm_const_dfsm.set_training_values(DV,const_dfsm)
        sm_const_dfsm.train()

        self.sm_obj1_of = sm_obj1_of
        self.sm_obj1_dfsm = sm_obj1_dfsm

        self.sm_obj2_of = sm_obj2_of
        self.sm_obj2_dfsm = sm_obj2_dfsm

        self.sm_con_of = sm_const_of
        self.sm_con_dfsm = sm_const_dfsm


    def compare(self,qty):
        
        DV = self.DV
        n_pts = len(DV)

        omega_pc = np.unique(DV[:,0])
        zeta_pc = np.unique(DV[:,1])

        O,Z = np.meshgrid(omega_pc,zeta_pc)

        n_samples = len(omega_pc)

        metric_of = np.zeros((n_pts,))
        metric_dfsm = np.zeros((n_pts,))

        for i_pt in range(n_pts):

            if qty == 'twrbsmyt_del':

                metric_of[i_pt] = self.sm_obj1_of.predict_values(np.atleast_2d(DV[i_pt,:]))
                metric_dfsm[i_pt] = self.sm_obj1_dfsm.predict_values(np.atleast_2d(DV[i_pt,:]))

            elif qty == 'genspeed_std':

                metric_of[i_pt] = self.sm_obj2_of.predict_values(np.atleast_2d(DV[i_pt,:]))
                metric_dfsm[i_pt] = self.sm_obj2_dfsm.predict_values(np.atleast_2d(DV[i_pt,:]))

            elif qty == 'genspeed_max':

                metric_of[i_pt] = self.sm_const_of.predict_values(np.atleast_2d(DV[i_pt,:]))
                metric_dfsm[i_pt] = self.sm_const_dfsm.predict_values(np.atleast_2d(DV[i_pt,:]))


        max_dfsm = np.max(metric_dfsm)
        max_of = np.max(metric_of)

        metric_dfsm = np.reshape(metric_dfsm,[n_samples,n_samples],order = 'F')
        metric_of = np.reshape(metric_of,[n_samples,n_samples],order = 'F')

        datasets = [metric_dfsm/1,metric_of/1]

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

        #plt.show()

        return O,Z,metric_of


    def run_level2(self,dv):


        outputs = {}
        outputs[self.obj1] = self.sm_obj1_dfsm.predict_values(np.atleast_2d(dv))[0,0]
        outputs[self.obj2] = self.sm_obj2_dfsm.predict_values(np.atleast_2d(dv))[0,0]
        outputs[self.const] = self.sm_con_dfsm.predict_values(np.atleast_2d(dv))[0,0]
        
        return outputs
        


    def run_level3(self,dv):

        outputs = {}
        
        outputs[self.obj1] = self.sm_obj1_of.predict_values(np.atleast_2d(dv))[0,0]
        outputs[self.obj2] = self.sm_obj2_of.predict_values(np.atleast_2d(dv))[0,0]
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

