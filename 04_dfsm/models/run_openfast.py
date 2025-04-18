import os
import numpy as np
import shutil

from weis.aeroelasticse.FAST_wrapper import FAST_wrapper
from weis.dfsm.dfsm_utilities import compile_dfsm_results
from pCrunch import Crunch,FatigueParams, AeroelasticOutput

of_path = os.path.realpath( shutil.which('openfast') )



def run_openfast(case_data_all,reqd_states,reqd_controls,reqd_outputs,mpi_options,GB_ratio = 1, TStart = 0):

    if mpi_options['mpi_run']:

        # evaluate the closed loop simulations in parallel using MPI

        sim_outputs = run_mpi(case_data_all,mpi_options)

    else:

        # evaluate the closed loop simulations serially
        
        sim_outputs = run_serial(case_data_all)

    chan_time_list = []
    ae_output_list = []
    fatigue_channels = {
                    'TwrBsMyt': FatigueParams(slope=4),
                                        }

    for i_case,sim_result in enumerate(sim_outputs):

        # extract results
        T_dfsm = sim_result['T_dfsm']

        states_dfsm = sim_result['states_dfsm']

        controls_dfsm = sim_result['controls_dfsm']

        outputs_dfsm = sim_result['outputs_dfsm']

        # compile results from DFSM
        OutData = compile_dfsm_results(T_dfsm,states_dfsm,controls_dfsm,outputs_dfsm,reqd_states,
                                        reqd_controls,reqd_outputs,GB_ratio,TStart)
        
        chan_time_list.append(OutData)

        # get output
        ae_output = AeroelasticOutput(OutData, dlc = 'dfsm_'+str(i_case),  fatigue_channels = fatigue_channels )

        ae_output_list.append(ae_output)

    cruncher = Crunch(outputs = [],lean = True)

    for output in ae_output_list:
        cruncher.add_output(output)

    return cruncher,ae_output_list,chan_time_list