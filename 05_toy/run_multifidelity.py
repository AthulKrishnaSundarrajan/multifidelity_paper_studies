import numpy as np
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from models.prod_functions import L2Turbine, L3Turbine
from models.mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine
import os
import matplotlib.pyplot as plt
import pickle

bounds = bounds = {'omega_pc' : np.array([[0.10, 0.3]]),'zeta_pc' : np.array([[0.10, 3.0]])}
desvars = {'omega_pc' : np.array([0.1]),'zeta_pc': np.array([3.0])}

results_file = 'sensstudy_results_OZ.pkl'

n_pts = 15

objs = np.zeros((n_pts,2))
opt_pts = np.zeros((n_pts,2))

w1 = np.linspace(1,0,n_pts)
w2 = 1-w1

obj1 = 'twrbsmyt_del'
obj2 = 'genspeed_std'

mf_turbine = MF_Turbine(results_file=results_file,obj1= obj1,obj2 = obj2,const = 'genspeed_max')

low_warmstart_file = None #'model_low_outputs.dill'
high_warmstart_file = None #'model_high_outputs.dill'

for i_pt in range(n_pts):


    multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}


    model_low = L2Turbine(desvars,mf_turbine,multi_fid_dict,warmstart_file = low_warmstart_file)
    model_high = L3Turbine(desvars,mf_turbine,multi_fid_dict,warmstart_file = high_warmstart_file)


    np.random.seed(123)

    trust_region = SimpleTrustRegion(
        model_low,
        model_high,
        bounds,
        disp=1,
        trust_radius=0.5,
        num_initial_points=5,
        radius_tol = 1e-6,
        optimization_log = True
    )
    trust_region.set_initial_point([0.2,1.0])

    trust_region.add_objective("wt_objectives", scaler=1e0)
    #trust_region.add_constraint("genspeed_max", upper=1.2)


    trust_region.optimize(plot=False, num_basinhop_iterations=0)

    # model_low.warmstart_file = low_warmstart_file
    # model_high.warmstart_file = high_warmstart_file

    # model_low.save_results(model_low.saved_desvars,model_low.saved_outputs)
    # model_high.save_results(model_high.saved_desvars,model_high.saved_outputs)
    
    opt_pts[i_pt,:] = trust_region.design_vectors[-1,:]

    objs[i_pt,0] = trust_region.model_high.run(opt_pts[i_pt,:])[obj1]
    objs[i_pt,1] = trust_region.model_high.run(opt_pts[i_pt,:])[obj2]
    
fig,ax = plt.subplots(1)

ax.plot(objs[:,0],objs[:,1],'.',markersize = 8)
ax.set_xlabel(obj1)
ax.set_ylabel(obj2)
print(objs)
print(opt_pts)

with open('multi-fid-res.pkl','wb') as handle:
    pickle.dump(objs,handle)

plt.show()