import numpy as np
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from models.prod_functions import L2Turbine, L3Turbine
from models.mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine
import os


bounds = {'pc_omega':np.array([0.1,0.3]),'pc_zeta':np.array([0.1,3.0])}#np.array([[0.10, 0.3],[0.1,3.0]])
desvars = {'pc_omega' : 0.3,'pc_zeta': 3.0}

this_dir = os.path.dirname(os.path.realpath(__file__))
results_file = this_dir + os.sep + 'doe_comparison_results.pkl'

mf_turb = MF_Turbine(results_file=results_file,obj = 'DEL',const = 'rotor_overspeed')

model_low = L2Turbine(desvars,mf_turb)
model_high = L3Turbine(desvars,mf_turb)

np.random.seed(123)

trust_region = SimpleTrustRegion(
    model_low,
    model_high,
    bounds,
    disp=1,
    trust_radius=0.5,
    num_initial_points=10,
    radius_tol = 1e-3,
    optimization_log = True
)
trust_region.set_initial_point([0.2,2.0])

trust_region.add_objective("DEL", scaler=1e-5)
trust_region.add_constraint("rotor_overspeed", upper=0.2)


trust_region.optimize(plot=False, num_basinhop_iterations=5)
