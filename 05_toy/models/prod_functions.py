from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine
import numpy as np


class L2Turbine(BaseModel):
    
    def __init__(self,desvars_init, mf_turb,multi_fid_dict = None,warmstart_file = None):
        super(L2Turbine, self).__init__(desvars_init)
        
        self.l2_turb = Level2_Turbine(mf_turb)
        self.multi_fid_dict = multi_fid_dict
        self.warmstart_file = warmstart_file
        self.n_count = 0

    def compute(self, desvars):
        

        o = desvars['omega_pc'][0]
        z = desvars['zeta_pc'][0]
        
        dv = np.array([o,z])
        outputs = self.l2_turb.compute(dv)
        self.n_count+=1

        if not(self.multi_fid_dict == None):

            obj1 = self.multi_fid_dict['obj1']
            obj2 = self.multi_fid_dict['obj2']
            wt_obj1 = self.multi_fid_dict['w1']
            wt_obj2 = self.multi_fid_dict['w2']

            o1 = outputs[obj1]
            o2 = outputs[obj2]

            outputs['wt_objectives'] = wt_obj1*o1 + wt_obj2*o2
        
        
        return outputs
        
        
class L3Turbine(BaseModel):
    
    def __init__(self,desvars_init, mf_turb,multi_fid_dict = None,warmstart_file = None):
        super(L3Turbine, self).__init__(desvars_init)

        self.l3_turb = Level3_Turbine(mf_turb)
        self.multi_fid_dict = multi_fid_dict
        self.warmstart_file = warmstart_file
        self.n_count = 0

    def compute(self, desvars):
        o = desvars['omega_pc'][0]
        z = desvars['zeta_pc'][0]

        dv = np.array([o,z])
        

        outputs = self.l3_turb.compute(dv)
        self.n_count+=1

        if not(self.multi_fid_dict == None):

            obj1 = self.multi_fid_dict['obj1']
            obj2 = self.multi_fid_dict['obj2']
            wt_obj1 = self.multi_fid_dict['w1']
            wt_obj2 = self.multi_fid_dict['w2']

            o1 = outputs[obj1]
            o2 = outputs[obj2]

            outputs['wt_objectives'] = wt_obj1*o1 + wt_obj2*o2

        return outputs
        
        

