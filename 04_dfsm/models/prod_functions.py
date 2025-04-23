from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, DFSM_Turbine, Level3_Turbine


class LFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb):

        super(LFTurbine, self).__init__(desvars_init)
        self.n_count = 0
        self.lf_turb = DFSM_Turbine(mf_turb)

    def compute(self, desvars):
        outputs = self.lf_turb.compute(desvars['pc_omega'])
        self.n_count+=1
        
        return outputs
        
        
class HFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb):
        super(HFTurbine, self).__init__(desvars_init)
        self.l3_turb = Level3_Turbine(mf_turb)
        self.n_count = 0

    def compute(self, desvars):
        outputs = self.l3_turb.compute(desvars['pc_omega'])
        self.n_count+=1
        
        return outputs
        
        

