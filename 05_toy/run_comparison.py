import numpy as np
from models.mf_controls import MF_Turbine
import os
from models.prod_functions import L3Turbine

this_dir = os.path.dirname(os.path.realpath(__file__))
results_file = this_dir + os.sep + 'sensstudy_results_OZ.pkl'

mf_turbine = MF_Turbine(results_file=results_file,obj1= 'twrbsmyt_del',obj2 = 'genspeed_std',const = 'genspeed_max')

mf_turbine.compare('genspeed_std')

desvars = {'omega_pc' : 0.2,'zeta_pc': 1.0}

model = L3Turbine(desvars,mf_turbine)

outputs = model.compute(desvars)

breakpoint()
    