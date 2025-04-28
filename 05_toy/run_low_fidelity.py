import numpy as np
from models.prod_functions import L2Turbine
from models.mf_controls import MF_Turbine
from time import time
from os import path
import os
import openmdao.api as om


bounds = np.array([[0.10, 0.3],[0.1,3.0]])
desvars = {'pc_omega' : 0.2,'pc_zeta': 1.0}

this_dir = os.path.dirname(os.path.realpath(__file__))
results_file = this_dir + os.sep + 'doe_comparison_results.pkl'

class Model(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            self.add_input(key, val=desvars[key])
        
        self.add_output('DEL', val=0.)
        self.add_output('rotor_overspeed', val=0.)
        
        mf_turb = MF_Turbine(results_file=results_file,obj = 'DEL',const = 'rotor_overspeed')
    
        self.model = L2Turbine(desvars,mf_turb)

    def compute(self, inputs, outputs):

        desvars = self.options["desvars"]
        
        op = self.model.compute(inputs)
        outputs['DEL'] = op['DEL']
        outputs['rotor_overspeed'] = op['rotor_overspeed']
        
        
        
        
p = om.Problem(model=om.Group())
model = p.model
model.approx_totals(method='fd', step=1e-3, form='central')
comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

for key in desvars:
    model.set_input_defaults(key, val=desvars[key])

s = time()

p.driver = om.ScipyOptimizeDriver()
p.driver.options['optimizer'] = "SLSQP"



idv = 0
for key in desvars:
    model.add_design_var(key, lower=bounds[idv,0], upper=bounds[idv,1])
    idv+= 1

model.add_constraint('rotor_overspeed', upper=0.2)
model.add_objective('DEL', ref=1.e5)

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

p.setup(mode='fwd')
p.driver.options['maxiter'] = 100
p.run_driver()
