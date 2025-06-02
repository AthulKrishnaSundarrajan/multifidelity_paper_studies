import numpy as np
from models.prod_functions import L3Turbine
from models.mf_controls import MF_Turbine
from time import time
from os import path
import os
import openmdao.api as om
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver


bounds = np.array([[0.10, 0.3],[0.1,3.0]])
desvars = {'omega_pc' : 0.2,'zeta_pc': 2.0}

results_file = 'sensstudy_results_OZ.pkl'


class Model(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            self.add_input(key, val=desvars[key])
        
        self.add_output('genspeed_std', val=0.)
        self.add_output('genspeed_max', val=0.)
        
        mf_turbine = MF_Turbine(results_file=results_file,obj1= 'twrbsmyt_del',obj2 = 'genspeed_std',const = 'genspeed_max')
    
        self.model = L3Turbine(desvars,mf_turbine)

    def compute(self, inputs, outputs):

        desvars = self.options["desvars"]
        
        op = self.model.compute(inputs)
        outputs['genspeed_std'] = op['genspeed_std']
        outputs['genspeed_max'] = op['genspeed_max']
        
        
        
        
p = om.Problem(model=om.Group())
model = p.model
model.approx_totals(method='fd', step=1e-3, form='central')
comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

for key in desvars:
    model.set_input_defaults(key, val=desvars[key])

s = time()

p.driver = NLoptDriver()
p.driver.options['optimizer'] = "LN_COBYLA"
p.driver.options['tol'] = 1e-5


idv = 0
for key in desvars:
    model.add_design_var(key, lower=bounds[idv,0], upper=bounds[idv,1])
    idv+= 1

model.add_constraint('genspeed_max', upper=1.2)
model.add_objective('genspeed_std', ref=1.e0)

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
