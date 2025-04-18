from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, DFSM_Turbine, Level3_Turbine


class LFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb):

        super(LFTurbine, self).__init__(desvars_init)
        
        self.lf_turb = DFSM_Turbine(mf_turb)

    def compute(self, desvars):
        outputs = self.l2_turb.compute(desvars['pc_omega'])
        
        return outputs
        
        
class HFTurbine(BaseModel):
    
    def __init__(self, desvars_init, warmstart_file, mf_turb):
        super(HFTurbine, self).__init__(desvars_init)
        self.l3_turb = Level3_Turbine(mf_turb)

    def compute(self, desvars):
        outputs = self.l3_turb.compute(desvars['pc_omega'])
        
        return outputs
        
        

