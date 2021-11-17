# import numpy as np
# import time
import logging as log
from typing import List, Union

# from pycqed.measurement import detector_functions as det
# from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import optimization as opt

from qcodes.instrument.parameter import ManualParameter
# from pycqed.analysis.analysis_toolbox import normalize_TD_data
# from pycqed.measurement.openql_experiments import multi_qubit_oql as mqo
# from pycqed.analysis_v2 import measurement_analysis as ma2
# from pycqed.measurement.openql_experiments import clifford_rb_oql as cl_oql

counter_param = ManualParameter('counter', unit='#')
counter_param(0)

def conventional_CZ_cost_func(device, FL_LutMan_QR, MC,
                              prepare_for_timedomain=True,
                              disable_metadata=True,
                              qubits=['X', 'D4'],
                              flux_codeword_park=None,
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
                              waveform_name='cz_SE',
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
        prepare_for_timedomain=prepare_for_timedomain,
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
            prepare_for_timedomain=prepare_for_timedomain,
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



def conventional_CZ_cost_func2(device, MC,
                              prepare_for_timedomain=True,
                              disable_metadata=True,
                              pairs=[['X', 'D4']],
                              parked_qbs=None,
                              flux_codeword='cz',
                              wait_time_before_flux_ns: int = 0,
                              wait_time_after_flux_ns: int = 0,
                              include_single_qubit_phase_in_cost=False,
                              include_leakage_in_cost=True,
                              measure_two_conditional_oscillations=False,
                              fixed_max_nr_of_repeated_gates=None,
                              target_single_qubit_phase=360,
                              parked_qubit_seq='ground',
                              CZ_duration=40,
                              extract_only=False,
                              target_phase=180,
                              cond_phase_weight_factor=1):

    counter_param(counter_param()+1)
    for pair in pairs:
        QR = device.find_instrument(pair[0])
        FL_LutMan_QR = QR.instr_LutMan_Flux.get_instr()
        # FL_LutMan_QR.AWG.get_instr().stop() # do we really need it, it is done in load_wfs 
        FL_LutMan_QR.generate_cz_waveforms()
        FL_LutMan_QR.load_waveforms_onto_AWG_lookuptable()
        # FL_LutMan_QR.AWG.get_instr().start() # do we really need it, it is done in load_wfs 

    result_dict = device.measure_conditional_oscillation_multi(
                    pairs=pairs,
                    parked_qbs=parked_qbs,
                    MC=MC,
                    prepare_for_timedomain=prepare_for_timedomain,
                    wait_time_before_flux_ns=wait_time_before_flux_ns,
                    wait_time_after_flux_ns=wait_time_after_flux_ns,
                    flux_codeword=flux_codeword,
                    parked_qubit_seq=parked_qubit_seq,
                    label=counter_param(),
                    disable_metadata=disable_metadata,
                    extract_only=extract_only)
    
    n_pairs = len(pairs)
    
    Delta_phi_a = [ result_dict[f'pair_{i+1}_delta_phi_a'] for i in range(n_pairs) ]
    Missing_frac_a = [ result_dict[f'pair_{i+1}_missing_frac_a'] for i in range(n_pairs) ]
    Offset_difference_a = [ result_dict[f'pair_{i+1}_offset_difference_a'] for i in range(n_pairs) ]
    Phi_0_a = [ result_dict[f'pair_{i+1}_phi_0_a'] for i in range(n_pairs) ]
    Phi_1_a = [ result_dict[f'pair_{i+1}_phi_1_a'] for i in range(n_pairs) ]

    # HERE substitute contribution with multi_targets_phase_offset
    # cost_function_val = abs(delta_phi_a-target_phase)
    # NOTE added cost function normalization (now the max value is 3 if cond_phase_weight_factor is 1)
    func_weight_angles = opt.multi_targets_phase_offset(target_phase, 360)
    func_weight_angles_phase = opt.multi_targets_phase_offset(target_single_qubit_phase, 360)
    # Create multiple cost functions
    Cost_function_val = [ None for i in range(n_pairs) ]
    Result_dict = {}
    for i in range(n_pairs):
        Cost_function_val[i] = func_weight_angles(Delta_phi_a[i]) / target_phase * cond_phase_weight_factor
        if include_leakage_in_cost:
            Cost_function_val[i] += abs(Offset_difference_a[i]) #* 100
            Cost_function_val[i] += abs(Missing_frac_a[i]) #* 200
        if include_single_qubit_phase_in_cost:
            Cost_function_val[i] += func_weight_angles_phase(Phi_0_a[i]) / target_single_qubit_phase

        Result_dict[f'cost_function_val_{pairs[i]}'] = Cost_function_val[i]
        Result_dict[f'delta_phi_{pairs[i]}'] = Delta_phi_a[i]
        Result_dict[f'single_qubit_phase_0_{pairs[i]}'] = Phi_0_a[i]
        Result_dict[f'single_qubit_phase_1_{pairs[i]}'] = Phi_1_a[i]
        Result_dict[f'missing_fraction_{pairs[i]}'] = Missing_frac_a[i]*100
        Result_dict[f'offset_difference_{pairs[i]}'] = Offset_difference_a[i]*100

    return Result_dict


def parity_check_cost(
        phase_diff: float, 
        missing_fraction: float=None, 
        phase_weight: float=1, 
        target_phase: float=180
    ):
    phi_dist_func = opt.multi_targets_phase_offset(target_phase, 360)
    phi_distance_from_target = phi_dist_func(phase_diff)
    cost = phase_weight * phi_distance_from_target/target_phase

    if missing_fraction:
        if missing_fraction > 1:
            log.warning(f"missing_fraction {missing_fraction} was probably given in percent, but raw value required instead!")
        cost += missing_fraction

    return cost


def parity_check_cost_function(
        device,
        MC,
        flux_lm, # lutman of fluxed qubit that needs to upload new pulses
        target_qubits: List[str],
        control_qubits: List[str], # needs to be given in order of the UHF
        flux_dance_steps: List[int],
        flux_codeword: str='flux-dance',
        ramsey_qubits: Union[list, bool]=True,
        refocusing: bool=True,
        phase_offsets: List[float]=None,
        phase_weight_factor: float=1,
        include_missing_frac_cost: bool=False,
        wait_time_before_flux_ns: int=0,
        wait_time_after_flux_ns: int=0,
        prepare_for_timedomain: bool=False,
        disable_metadata: bool=True,
        plotting: bool=False
    ):

    counter_param(counter_param()+1)

    # waveforms are already uploaded by sweep functions, so no need to do any extra preparation here
    # # TODO find high qubits and prepare only those
    # for qubit in ancilla_qubit + data_qubits:
    #     qb = device.find_instrument(qubit)
    #     FL_LutMan_qb = qb.instr_LutMan_Flux.get_instr()
    #     # FL_LutMan_qb.AWG.get_instr().stop() # do we really need it, it is done in load_wfs 
    #     # NOTE new waveform generator function, should be tested
    #     FL_LutMan_qb.generate_cz_waveforms()
    #     FL_LutMan_qb.load_waveforms_onto_AWG_lookuptable()
    #     # FL_LutMan_qb.AWG.get_instr().start() # do we really need it, it is done in load_wfs 

    # flux_lm.generate_cz_waveforms()
    # flux_lm.load_waveforms_onto_AWG_lookuptable()

    result_dict = device.measure_parity_check_flux_dance(
        MC=MC,
        target_qubits=target_qubits,
        control_qubits=control_qubits,
        ramsey_qubits=ramsey_qubits,
        flux_dance_steps=flux_dance_steps,
        flux_codeword=flux_codeword,
        refocusing=refocusing,
        phase_offsets=phase_offsets,
        prepare_for_timedomain=prepare_for_timedomain,
        wait_time_before_flux_ns=wait_time_before_flux_ns,
        wait_time_after_flux_ns=wait_time_after_flux_ns,
        label_suffix=counter_param(),
        disable_metadata=disable_metadata,
        plotting=plotting
        )

    phi_diff = (result_dict['phi_osc'][result_dict['cases'][0]] \
                - result_dict['phi_osc'][result_dict['cases'][-1]]) % 360
    
    cost = parity_check_cost(phase_diff=phi_diff, 
                            missing_fraction=result_dict['missing_frac'][control_qubits[0]] 
                                            if include_missing_frac_cost else None,
                            phase_weight=phase_weight_factor)

    result_dict[f'missing_frac_{control_qubits[0]}'] = 100 * result_dict['missing_frac'][control_qubits[0]]
    result_dict['cost_function_val'] = cost
    result_dict['phi_diff'] = phi_diff

    return result_dict

