import numpy as np
from models.prod_functions import L2Turbine,L3Turbine
from models.mf_controls import MF_Turbine
from time import time
from os import path
import os
import openmdao.api as om
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver
import matplotlib.pyplot as plt
import pickle


bounds = np.array([[0.10, 0.3],[0.1,3.0]])
desvars = {'omega_pc' : 0.1,'zeta_pc': 3.0}

results_file = 'sensstudy_results_OZ.pkl'



n_pts = 15

objs = np.zeros((n_pts,2))
opt_pts = np.zeros((n_pts,2))

w1 = np.linspace(1,0,n_pts)
w2 = 1-w1

obj1 = 'twrbsmyt_del'
obj2 = 'genspeed_std'

mf_turbine = MF_Turbine(results_file=results_file,obj1= obj1,obj2 = obj2,const = 'genspeed_max')

for i_pt in range(n_pts):


    multi_fid_dict = {'obj1':obj1,'obj2':obj2,'w1':w1[i_pt],'w2':w2[i_pt]}


    class Model(om.ExplicitComponent):
        def initialize(self):
            self.options.declare('desvars')
            
        def setup(self):
            desvars = self.options["desvars"]
            for key in desvars:
                self.add_input(key, val=desvars[key])
            
            self.add_output('genspeed_std', val=0.)
            self.add_output('genspeed_max', val=0.)
            self.add_output('twrbsmyt_del',val = 0)

            if not(multi_fid_dict == None):
                self.add_output('wt_objectives',val = 0)
            
            
        
            self.model = L2Turbine(desvars,mf_turbine,multi_fid_dict)

        def compute(self, inputs, outputs):

            desvars = self.options["desvars"]
            
            op = self.model.compute(inputs)
            
            for key in op.keys():
            #outputs['genspeed_std'] = op['genspeed_std']
                outputs[key] = op[key]
            
            
            
            
    p = om.Problem(model=om.Group())
    model = p.model
    model.approx_totals(method='fd', step=1e-3, form='central')
    comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

    for key in desvars:
        model.set_input_defaults(key, val=desvars[key])

    s = time()

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = "SLSQP"
    p.driver.options['tol'] = 1e-10



    idv = 0
    for key in desvars:
        model.add_design_var(key, lower=bounds[idv,0], upper=bounds[idv,1])
        idv+= 1

    #model.add_constraint('genspeed_max', upper=1.2)

    if multi_fid_dict == None:
        model.add_objective('genspeed_std', ref=1.e0)
    else:
        model.add_objective('wt_objectives', ref=1.e0)


    p.driver.recording_options['includes'] = ['*']
    p.driver.recording_options['record_objectives'] = True
    p.driver.recording_options['record_constraints'] = True
    p.driver.recording_options['record_desvars'] = True
    p.driver.recording_options['record_inputs'] = True
    p.driver.recording_options['record_outputs'] = True
    p.driver.recording_options['record_residuals'] = True

    p.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

    recorder = om.SqliteRecorder("cases.sql")
    p.driver.add_recorder(recorder)

    p.setup(mode = 'fwd')
    p.driver.options['maxiter'] = 100
    p.run_driver()

    objs[i_pt,0] = p.get_val('Model.'+obj1)
    objs[i_pt,1] = p.get_val('Model.'+obj2)

    opt_pts[i_pt,0] = p.get_val('Model.omega_pc')
    opt_pts[i_pt,1] = p.get_val('Model.zeta_pc')


fig,ax = plt.subplots(1)

ax.plot(objs[:,0],objs[:,1],'.',markersize = 8)
ax.set_xlabel(obj1)
ax.set_ylabel(obj2)
print(objs)
print(opt_pts)

with open('low-fid-res.pkl','wb') as handle:
    pickle.dump(objs,handle)

# O,Z,obj1_c = mf_turbine.compare(obj1)
# O,Z,obj2_c = mf_turbine.compare(obj2)

# fig,ax = plt.subplots(1)

# ax.contourf(O,Z,obj1_c)
# ax.plot(opt_pts[:,0],opt_pts[:,1],'r.',markersize = 8)
# ax.set_title(obj1)

# fig,ax = plt.subplots(1)

# ax.contourf(O,Z,obj2_c)
# ax.plot(opt_pts[:,0],opt_pts[:,1],'r.',markersize = 8)
# ax.set_title(obj2)

plt.show()
#breakpoint()
