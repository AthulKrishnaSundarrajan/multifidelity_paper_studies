from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, DFSM_Turbine, Level3_Turbine


class LFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb,warmstart_file = None,scaling_dict = None):

        super(LFTurbine, self).__init__(desvars_init)
        self.n_count = 0
        self.warmstart_file = warmstart_file
        self.scaling_dict = scaling_dict
        self.lf_turb = DFSM_Turbine(mf_turb)

    def compute(self, desvars):
        scaling_dict = self.scaling_dict
        outputs = self.lf_turb.compute(desvars,scaling_dict)
        self.n_count+=1
        
        return outputs
        
        
class HFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb,warmstart_file = None,scaling_dict = None):

        super(HFTurbine, self).__init__(desvars_init)
        self.l3_turb = Level3_Turbine(mf_turb)
        self.warmstart_file = warmstart_file
        self.scaling_dict = scaling_dict
        self.n_count = 0

    def compute(self, desvars):
        scaling_dict = self.scaling_dict
        
        outputs = self.l3_turb.compute(desvars,scaling_dict)
        self.n_count+=1
        
        return outputs
        
        

