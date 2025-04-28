from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine
import numpy as np


class L2Turbine(BaseModel):
    
    def __init__(self,desvars_init, mf_turb):
        super(L2Turbine, self).__init__(desvars_init)
        
        self.l2_turb = Level2_Turbine(mf_turb)
        self.n_count = 0

    def compute(self, desvars):
        

        o = desvars['pc_omega']
        z = desvars['pc_zeta']
        
        dv = np.array([o,z])
        outputs = self.l2_turb.compute(dv)
        self.n_count+=1
        
        return outputs
        
        
class L3Turbine(BaseModel):
    
    def __init__(self,desvars_init, mf_turb):
        super(L3Turbine, self).__init__(desvars_init)

        self.l3_turb = Level3_Turbine(mf_turb)
        self.n_count = 0

    def compute(self, desvars):
        o = desvars['pc_omega']
        z = desvars['pc_zeta']

        dv = np.array([o,z])
        

        outputs = self.l3_turb.compute(dv)
        self.n_count+=1

        return outputs
        
        

