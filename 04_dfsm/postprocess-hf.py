import numpy as np
import openmdao.api as om

high_fid_results = 'run_high_fidelity_out/cases.sql'

cr = om.CaseReader(high_fid_results)

driver_cases = cr.get_cases('driver')

for case in driver_cases:
    print(case.get_design_vars())

#dv = driver_cases[-1].get_design_vars()

breakpoint()