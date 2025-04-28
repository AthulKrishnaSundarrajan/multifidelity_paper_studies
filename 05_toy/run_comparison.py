import numpy as np
from models.mf_controls import MF_Turbine
import os
from models.prod_functions import L3Turbine

this_dir = os.path.dirname(os.path.realpath(__file__))
results_file = this_dir + os.sep + 'doe_comparison_results.pkl'

mf_turbine = MF_Turbine(results_file=results_file,obj= 'DEL',const = 'rotor_overspeed')

mf_turbine.compare()

desvars = {'pc_omega' : 0.2,'pc_zeta': 1.0}

model = L3Turbine(desvars,mf_turbine)

outputs = model.compute(desvars)

breakpoint()
    