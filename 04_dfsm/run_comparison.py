import numpy as np
import os
import matplotlib.pyplot as plt
from models.mf_controls import MF_Turbine

if __name__ == '__main__':

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Main inputs that mf_controls need

    # 1. DFSM file and the model detials
    dfsm_file = this_dir + os.sep + 'dfsm_fowt_1p6.pkl'

    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr','RtFldCp','RtFldCt']

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'rated-16' + os.sep + 'openfast_runs'

    # 3. ROSCO yaml file
    rosco_yaml = this_dir + os.sep + 'IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'

    mf_controls = MF_Turbine(dfsm_file,reqd_states,reqd_controls,reqd_outputs,OF_dir,rosco_yaml)

    cruncher,ae_output_list,chan_time_list = mf_controls.run_dfsm()

    breakpoint()


