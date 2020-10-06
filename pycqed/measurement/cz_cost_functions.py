from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from collections import deque
import numpy as np
import time
from pycqed.measurement import optimization as opt
from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis.analysis_toolbox import normalize_TD_data
from pycqed.measurement.openql_experiments import multi_qubit_oql as mqo
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.measurement.openql_experiments import clifford_rb_oql as cl_oql

counter_param = ManualParameter('counter', unit='#')
counter_param(0)

def conventional_CZ_cost_func(device, FL_LutMan_QR, MC,
                              prepare_for_timedomain=True,
                              disable_metadata=True,
                              qubits=['X', 'D4'],
                              flux_codeword_park=None,
                              reduced_swp_points=True,
                              flux_codeword='cz',
                              include_single_qubit_phase_in_cost=False,
                              include_leakage_in_cost=True,
                              measure_two_conditional_oscillations=False,
                              fixed_max_nr_of_repeated_gates=None,
                              target_single_qubit_phase=360,
                              parked_qubit_seq='ground',
                              CZ_duration=40,
                              extract_only=False,
                              target_phase=180,
                              wait_time_ns=0, waveform_name='cz_SE',
                              cond_phase_weight_factor=1):

    counter_param(counter_param()+1)
    FL_LutMan_QR.AWG.get_instr().stop()
    FL_LutMan_QR.generate_standard_waveforms()
    FL_LutMan_QR.load_waveforms_onto_AWG_lookuptable(waveform_name)
    FL_LutMan_QR.AWG.get_instr().start()
    q0,q1 = qubits[0], qubits[1]
    if len(qubits)>2:
        q2 = qubits[2]
    else:
        q2 = None
    if len(qubits)>3:
        q3 = qubits[3]
    else:
        q3 = None
    a = device.measure_conditional_oscillation(
        q0,q1,q2,q3,
        MC=MC,
        wait_time_ns=wait_time_ns,
        prepare_for_timedomain=prepare_for_timedomain,
        reduced_swp_points=reduced_swp_points,
        flux_codeword=flux_codeword,
        flux_codeword_park=flux_codeword_park,
        parked_qubit_seq=parked_qubit_seq,
        # nr_of_repeated_gates=FL_LutMan_QR.mcz_nr_of_repeated_gates(),
        label=counter_param(),
        disable_metadata=disable_metadata,
        extract_only=extract_only)
    delta_phi_a = a.proc_data_dict['quantities_of_interest']['phi_cond'].n % 360
    missing_frac_a = a.proc_data_dict['quantities_of_interest']['missing_fraction'].n
    offset_difference_a = a.proc_data_dict['quantities_of_interest']['offs_diff'].n
    phi_0_a = (a.proc_data_dict['quantities_of_interest']['phi_0'].n+180) % 360 - 180
    phi_1_a = (a.proc_data_dict['quantities_of_interest']['phi_1'].n+180) % 360 - 180

    if measure_two_conditional_oscillations:
        b = device.measure_conditional_oscillation(
            q1,q0,q2,q3,
            MC=MC,
            wait_time_ns=wait_time_ns,
            prepare_for_timedomain=prepare_for_timedomain,
            reduced_swp_points=reduced_swp_points,
            flux_codeword=flux_codeword,
            flux_codeword_park=flux_codeword_park,
            parked_qubit_seq=parked_qubit_seq,
            # nr_of_repeated_gates=FL_LutMan_QR.mcz_nr_of_repeated_gates(),
            label=counter_param(),
            disable_metadata=disable_metadata,
            extract_only=extract_only)
        delta_phi_b = b.proc_data_dict['quantities_of_interest']['phi_cond'].n % 360
        missing_frac_b = b.proc_data_dict['quantities_of_interest']['missing_fraction'].n
        offset_difference_b = b.proc_data_dict['quantities_of_interest']['offs_diff'].n
        phi_0_b = (b.proc_data_dict['quantities_of_interest']['phi_0'].n+180) % 360 - 180
        phi_1_b = (b.proc_data_dict['quantities_of_interest']['phi_1'].n+180) % 360 - 180

    # HERE substitute contribution with multi_targets_phase_offset
    # cost_function_val = abs(delta_phi_a-target_phase)
    func_weight_angles = opt.multi_targets_phase_offset(target_phase, 360)
    func_weight_angles_phase = opt.multi_targets_phase_offset(target_single_qubit_phase, 360)
    cost_function_val = func_weight_angles(delta_phi_a) * cond_phase_weight_factor

    if include_leakage_in_cost:
        cost_function_val += abs(offset_difference_a)*100
        cost_function_val += abs(a.proc_data_dict['quantities_of_interest']['missing_fraction'].n)*200

    if measure_two_conditional_oscillations:
        # HERE substitute contribution with multi_targets_phase_offset
        cost_function_val += func_weight_angles(delta_phi_b) * cond_phase_weight_factor
        if include_leakage_in_cost:
            cost_function_val += abs(offset_difference_b)*100
            cost_function_val += abs(b.proc_data_dict['quantities_of_interest']['missing_fraction'].n)*200

    if include_single_qubit_phase_in_cost:
        cost_function_val += func_weight_angles_phase(phi_0_a)
        if measure_two_conditional_oscillations:
            cost_function_val += func_weight_angles_phase(phi_0_b)
    if measure_two_conditional_oscillations:
        cost_function_val /= 2
        # HERE substitute contribution with multi_targets_phase_offset
        # cost_function_val += abs(phi_0_a)
        # if measure_two_conditional_oscillations:
        #     cost_function_val += abs(phi_0_b)

    if measure_two_conditional_oscillations:
        return {
            'cost_function_val': cost_function_val,
            'delta_phi': (delta_phi_a+delta_phi_b)/2,
            'single_qubit_phase_0_a': phi_0_a,
            'single_qubit_phase_0_b': phi_0_b,
            'missing_fraction': (missing_frac_a+missing_frac_b)*100,
            'offset_difference': (offset_difference_a+offset_difference_b)*100,
            'park_phase_off': (a.proc_data_dict['quantities_of_interest']['park_phase_off'].n+180)%360-180,
            'park_phase_on': (a.proc_data_dict['quantities_of_interest']['park_phase_on'].n+180)%360-180
        }
    else:
        return {
            'cost_function_val': cost_function_val,
            'delta_phi': delta_phi_a, # conditional phase
            'single_qubit_phase_0': phi_0_a, # phase_corr (phase_off)
            'single_qubit_phase_1': phi_1_a,
            'osc_amp_0': a.proc_data_dict['quantities_of_interest']['osc_amp_0'].n,
            'osc_amp_1': a.proc_data_dict['quantities_of_interest']['osc_amp_1'].n,
            'missing_fraction': missing_frac_a*100,
            'offset_difference': offset_difference_a*100,  # to convert to %
            'park_phase_off': (a.proc_data_dict['quantities_of_interest']['park_phase_off'].n+180)%360-180,
            'park_phase_on': (a.proc_data_dict['quantities_of_interest']['park_phase_on'].n+180)%360-180
        }
