import numpy as np
from models.prod_functions import LFTurbine
from models.mf_controls import MF_Turbine
from time import time
from os import path
import openmdao.api as om
import os


bounds = np.array([0.10, 0.3])
desvars = {'omega_pc' : np.array([0.2])}

 # get path to this directory
this_dir = os.path.dirname(os.path.realpath(__file__))


 # 1. DFSM file and the model detials
dfsm_file = this_dir + os.sep + 'dfsm_fowt_1p6.pkl'

reqd_states = ['PtfmSurge','PtfmPitch','TTDspFA','GenSpeed']
reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt']

# 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
OF_dir = this_dir + os.sep + 'outputs/near_rated_test' + os.sep + 'openfast_runs'

# 3. ROSCO yaml file
rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

class Model(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            self.add_input(key, val=desvars[key])
        
        self.add_output('TwrBsMyt_DEL', val=0.)
        self.add_output('GenSpeed_Max', val=0.)
        
        
        mf_turb = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml,)
        
        self.model = LFTurbine(desvars, mf_turb)

    def compute(self, inputs, outputs):
        desvars = self.options["desvars"]
        model_outputs = self.model.compute(inputs)
        outputs['TwrBsMyt_DEL'] = model_outputs['TwrBsMyt_DEL']
        outputs['GenSpeed_Max'] = model_outputs['GenSpeed_Max']
        
        
        
        
p = om.Problem(model=om.Group())
model = p.model
model.approx_totals(method='fd', step=1e-6, form='central')
comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

for key in desvars:
    model.set_input_defaults(key, val=desvars[key])
    
s = time()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = "SLSQP"

for key in desvars:
    model.add_design_var(key, lower=bounds[0], upper=bounds[1])
model.add_constraint('GenSpeed_Max', upper=9.)
model.add_objective('TwrBsMyt_DEL', ref=1.e5)

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
p.run_driver()
