import logging
import os
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
import qcodes as qc
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.waveform_control import pulsar
from pycqed.measurement.waveform_control.element import calculate_time_correction
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.standard_elements import distort_and_compensate
import pycqed.measurement.waveform_control.fluxpulse_predistortion as fluxpulse_predistortion
from pycqed.utilities.general import get_required_upload_information

from importlib import reload
reload(pulse)
from pycqed.measurement.waveform_control import pulse_library
reload(pulse_library)

station = qc.station


def single_pulse_seq(pulse_pars=None,
                     comp_pulse=True,
                     verbose=False,
                     distortion_dict=None,
                     return_seq=False):


    if pulse_pars is None:
        pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6,
                      'dead_time_length': 10e-6}
    minus_pulse_pars = {'pulse_type': 'SquarePulse',
                        'pulse_delay': 3e-6 + pulse_pars['length'] + \
                                       pulse_pars['pulse_delay'],
                        'channel': pulse_pars['channel'],
                        'amplitude': -pulse_pars['amplitude'],
                        'length': pulse_pars['length'],
                        'dead_time_length': 10e-6}

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_pulse_pars['length']),
                       'channel': pulse_pars['channel'],
                       'amplitude': 0,
                       'length': pulse_pars['dead_time_length']}

    trig_marker = {'pulse_type': 'SquarePulse',
                   'pulse_delay': 0.,
                   'channel': 'ch1_marker1',
                   'amplitude': 1.,
                   'length': .1e-6}
    # 'length': 5e-6}
    seq_name = 'Square_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    for i, iter in enumerate([0, 1]):  # seq has to have at least 2 elts

        if comp_pulse:
            pulse_list = [
                pulse_pars, trig_marker, minus_pulse_pars, dead_time_pulse]
        else:
            pulse_list = [pulse_pars, trig_marker, dead_time_pulse]
        # pulse_list = [pulse_pars, dead_time_pulse]

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    else:
        preloaded_kernels = []
    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def Ram_Z_seq(operation_dict, q0, distortion_dict,
              times,
              recovery_phase=0,
              RO_delay=3e-6,
              artificial_detuning=None,
              operation_name='Z',
              cal_points=True,
              verbose=False, upload=True):
    '''
    Performs a Ram-Z sequence similar to a conventional echo sequence.

    Timings of sequence
              <---  tau   --->
        |mX90| |   Z          |  |recPi2|---|RO|
    '''
    seq_name = 'Ram-Z-seq_{}'.format(q0)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']
    # Allows using some other flux pulse to perform the RamZ with
    if (('Z '+q0) not in operation_dict) or (operation_name != 'Z'):
        operation_dict['Z ' + q0] = deepcopy(
            operation_dict[operation_name + ' ' + q0])

    recPi2 = deepcopy(operation_dict['X90 '+q0])

    operation_dict['recPi2 ' + q0] = recPi2

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        if artificial_detuning is not None:
            D_phase = ((tau-times[0]) * artificial_detuning * 360) % 360
            operation_dict['recPi2 ' + q0]['phase'] = D_phase
        operation_dict['Z ' + q0]['length'] = tau
        pulse_list = ['mX90 '+q0, 'Z '+q0, 'recPi2 ' + q0, 'RO ' + q0]

        # calibration points overwrite the pulse_combinations list
        if cal_points and ((i == (len(times)-4) or i == (len(times)-3))):
            pulse_list = ['I '+q0, 'RO ' + q0]

        elif cal_points and ((i == (len(times)-2) or i == (len(times)-1))):
            pulse_list = ['X180 '+q0, 'RO ' + q0]
        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
        if distortion_dict is not None:
            print('\r Distorting element {}/{} '.format(i+1, len(times)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def Echo_Z_seq(operation_dict, q0, distortion_dict,
               times,
               recovery_phases=[0],
               RO_delay=3e-6,
               operation_name='Z ',
               Z_signs=[+1, +1],
               echo_MW_pulse=True,
               artificial_detuning=None,
               cal_points=True,
               verbose=False, upload=True):
    '''
    Performs a Ram-Z sequence similar to a conventional echo sequence.

    Timings of sequence
              <---  tau   --->
        |mX90| |   Z          |  |recPi2|---|RO|
    '''
    seq_name = 'Echo_Z_seq_{}'.format(q0)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']
    ########################################################
    ########################################################
    pulse_list = ['mX90 '+q0,
                  'Z0 '+q0, 'mid_pulse '+q0, 'Z1 '+q0,
                  'recPi2 ' + q0, 'RO ' + q0]
    ########################################################
    ########################################################
    operation_dict['Z0 ' + q0] = deepcopy(
        operation_dict[operation_name + ' ' + q0])
    operation_dict['Z1 ' + q0] = deepcopy(
        operation_dict[operation_name + ' ' + q0])

    operation_dict['Z0 ' + q0]['amplitude'] *= Z_signs[0]
    operation_dict['Z1 ' + q0]['amplitude'] *= Z_signs[1]
    if echo_MW_pulse:
        pulse_list[2] = 'X180 '+q0
    else:
        pulse_list[2] = 'I '+q0

    recPi2 = deepcopy(operation_dict['X90 '+q0])

    operation_dict['recPi2 ' + q0] = recPi2

    # allows sweeping any variable
    if len(recovery_phases) == 1:  # assumes it is a list
        recovery_phases *= len(times)
    if len(times) == 1:  # assumes it is a list
        times *= len(recovery_phases)

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        operation_dict['Z0 ' + q0]['length'] = tau/2
        operation_dict['Z1 ' + q0]['length'] = tau/2
        D_phase = recovery_phases[i]
        if artificial_detuning is not None:
            D_phase += ((tau-times[0]) * artificial_detuning * 360) % 360
        operation_dict['recPi2 ' + q0]['phase'] = D_phase % 360

        # calibration points overwrite the pulse_combinations list
        if cal_points and ((i == (len(times)-4) or i == (len(times)-3))):
            pulse_list = ['I '+q0, 'RO ' + q0]

        elif cal_points and ((i == (len(times)-2) or i == (len(times)-1))):
            pulse_list = ['X180 '+q0, 'RO ' + q0]
        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
        if distortion_dict is not None:
            print('\r Distorting element {}/{} '.format(i+1, len(times)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def Ram_Z_delay_seq(operation_dict, q0, distortion_dict,
                    inter_pulse_delay=300e-9, recovery_phase=0,
                    RO_delay=3e-6,
                    operation_name='Z',
                    times=np.arange(-100e-9, 400e-9, 25e-9),
                    cal_points=True,
                    verbose=False, upload=True,
                    return_seq=False):
    '''
    Performs a Ram-Z sequence useful for calibrating timings of flux pulses

    Timings of sequence
              <-tau_inter_pulse->
        |mX90|  ---   |Z|  ---   |recPi2|---|RO|
              <- t -> <-- dt1 -->

    '''
    seq_name = 'Ram-Z-seq_{}'.format(q0)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']
    # Allows using some other flux pulse to perform the RamZ with
    if (('Z '+q0) not in operation_dict) or (operation_name != 'Z'):
        operation_dict['Z ' + q0] = deepcopy(
            operation_dict[operation_name + ' ' + q0])

    # Setting the RO very high to ensure no overlap when moving the flux pulse
    operation_dict['RO ' + q0]['pulse_delay'] = RO_delay
    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        # Calibration points
        if (i == (len(times)-4) or i == (len(times)-3)):
            pulse_list = ['I '+q0, 'RO ' + q0]
        elif (i == (len(times)-2) or i == (len(times)-1)):
            pulse_list = ['X180 '+q0, 'RO ' + q0]
        else:
            operation_dict['Z '+q0]['pulse_delay'] = tau
            t1 = inter_pulse_delay - tau  # ref_point is start of flux pulse

            recPi2 = deepcopy(operation_dict['X90 '+q0])
            recPi2['ref_point'] = 'start'
            recPi2['phase'] = recovery_phase
            recPi2['pulse_delay'] = t1
            operation_dict['recPi2 ' + q0] = recPi2

            pulse_list = ['mX90 '+q0, 'Z '+q0, 'recPi2 ' + q0, 'RO ' + q0]

        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
        if distortion_dict is not None:
            print('\r Distorting element {}/{}'.format(i+1, len(times)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def chevron_seq(operation_dict, q0,
                pulse_lengths=np.arange(0, 120e-9, 2e-9),
                verbose=False,
                distortion_dict=None,
                upload=True,
                cal_points=True):
    '''
    Chevron sequence where length of the "SWAP" operation is varied
        X180 - SWAP(l) - RO


    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''

    seq_name = 'Chevron_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']

    SWAP_amp = operation_dict['SWAP '+q0]['amplitude']
    # seq has to have at least 2 elts
    for i, pulse_length in enumerate(pulse_lengths):
        # this converts negative pulse lenghts to negative pulse amplitudes
        if pulse_length < 0:
            operation_dict['SWAP '+q0]['amplitude'] = -SWAP_amp
        else:
            operation_dict['SWAP '+q0]['amplitude'] = SWAP_amp

        if cal_points and (i == (len(pulse_lengths)-4) or
                           i == (len(pulse_lengths)-3)):
            pulse_combinations = ['RO '+q0]
        elif cal_points and (i == (len(pulse_lengths)-2) or
                             i == (len(pulse_lengths)-1)):
            pulse_combinations = ['X180 ' + q0, 'RO ' + q0]
        else:
            operation_dict[
                'SWAP '+q0]['square_pulse_length'] = abs(pulse_length)
            pulse_combinations = ['X180 '+q0, 'SWAP ' + q0, 'RO '+q0]

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]

        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\r Distorting element {}/{}'.format(i+1, len(pulse_lengths)),
                  end='')
            if i == len(pulse_lengths):
                print()
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def SwapN(operation_dict, q0,
          nr_pulses_list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
          alpha=1.,
          verbose=False,
          distortion_dict=None,
          upload=True,
          cal_points=True,
          inter_swap_wait=10e-9):
    '''
    Sequence of N swap operations
        (N_max-N)* FluxID - X180 - N*SWAP - X180 RO

    pulse_dict:           (dict) dictionary containing the pulse parameters
    q0                    (str)  name of the target qubit
    nr_pulses_list        (list) nr of swaps gates for each element
    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''
    seq_name = 'SWAPN_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    sequencer_config = operation_dict['sequencer_config']

    # Create the correction pulses
    operation_dict['FluxId '+q0] = deepcopy(operation_dict['SWAP ' + q0])
    # Flux identity
    operation_dict['FluxId '+q0]['amplitude'] = 0
    n_max = nr_pulses_list[-1]
    for j in range(n_max):
        SWAP_pulse_j = deepcopy(operation_dict['SWAP '+q0])
        SWAP_pulse_j['amplitude'] = SWAP_pulse_j['amplitude']*(alpha**j)
        # SWAP_pulse_train.append(SWAP_pulse_j)
        operation_dict['SWAP_{} {}'.format(j, q0)] = SWAP_pulse_j

    for i, n in enumerate(nr_pulses_list):
        # SWAP_pulse_train = []
        pulse_combinations = (['X180 ' + q0] +
                              ['FluxId '+q0]*(n_max-n))
        for j in range(n):
            pulse_combinations += ['SWAP_{} {}'.format(j, q0)]
        pulse_combinations += ['RO '+q0]

        # calibration points overwrite the pulse_combinations list
        # All pulses are replaced with identities.
        if cal_points and (i == (len(nr_pulses_list)-4) or
                           i == (len(nr_pulses_list)-3)):
            pulse_combinations = (['RO '+q0])
        elif cal_points and (i == (len(nr_pulses_list)-2) or
                             i == (len(nr_pulses_list)-1)):
            pulse_combinations = (['X180 ' + q0] + ['RO '+q0])

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]

        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} '.format(i+1,
                                                       len(nr_pulses_list)),
                  end='')
            if i == len(nr_pulses_list):
                print()
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def swap_swap_wait(mw_pulse_pars, RO_pars,
                   flux_pulse_pars,
                   phases=np.linspace(0, 720, 41),
                   inter_swap_wait=100e-9,
                   verbose=False,
                   distortion_dict=None,
                   upload=True,
                   cal_points=True):
    '''
    Sequence of 2 swap operations with varying recovery pulse
        mY90 - swap - idle- swap - rphi90 - RO

    mw_pulse_pars:        (dict) qubit control pulse pars
    RO_pars:              (dict) qubit RO pars
    flux_pulse_pars:      (dict) flux puplse pars
    inter_swap_wait       (float) wait time in seconds between the two swaps
    phases                (list) phases used for the recovery pulse
    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''

    # To be merged with swap-CP-swap
    logging.warning('Do not use, I have marked this for deletion -MAR')

    preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    # renamed as the dict contains the pulse directly
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    minus_flux_pulse_pars['amplitude'] = -flux_pulse_pars['amplitude']

    # Pulse is used to set the starting ref_point for the compensation pulses
    dead_time_pulse_pars = {'pulse_type': 'SquarePulse',
                            'pulse_delay': flux_pulse_pars['dead_time'],
                            'channel': flux_pulse_pars['channel'],
                            'amplitude': 0,
                            'length': 0.}

    seq_name = 'swap_swap_wait_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)

    # seq has to have at least 2 elts
    for i, phase in enumerate(phases):
        if cal_points and (i == (len(phases)-4) or i == (len(phases)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(phases)-2) or i == (len(phases)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            # correcting timings
            recovery_pi2 = deepcopy(pulses['X90'])
            recovery_pi2['phase'] = phase
            second_flux_pulse = deepcopy(flux_pulse_pars)
            second_flux_pulse['pulse_delay'] = inter_swap_wait

            pulse_list = [pulses['mY90']] + [flux_pulse_pars] + [second_flux_pulse] \
                + [recovery_pi2]+[RO_pars] + \
                [dead_time_pulse_pars] + [minus_flux_pulse_pars]*2

            el = multi_pulse_elt(i, station, pulse_list)
            if distortion_dict is not None:
                el = distort_and_compensate(
                    el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def chevron_with_excited_bus_2Qubits(mw_pulse_pars_qCP, mw_pulse_pars_qS,
                                     flux_pulse_pars_qCP, flux_pulse_pars_qS,
                                     RO_pars,
                                     distortion_dict,
                                     chevron_pulse_lengths=np.arange(
                                         0, 120e-9, 2e-9),
                                     excitations=1,
                                     verbose=False,
                                     upload=True,
                                     cal_points=True):
    '''
    Sequence that swaps an excitation from qS into the bus and does chevron
    type measurement between qCPhase qubit and the bus. In the end, X180 gate
    to qCP is applied to invert population 0 and 1 to maximize 1-2 discrimination.

        X180 qS - swap qS,B - X180 qCP - chevron pulse qCP,B -
        - X180 qCP - RO

    qS is the "swap qubit"
    qCP is the "CPhase qubit"

    mw_pulse_pars qCP:    (dict) qubit control pulse pars
    mw_pulse_pars qS:     (dict) qubit control pulse pars
    flux_pulse_pars qCP:  (dict) flux puplse pars
    flux_pulse_pars qS:   (dict) flux puplse pars
    RO_pars:              (dict) qubit RO pars, ideally a multiplexed readout
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    excitations:          (enum) [0, 1, 'both'] whether to put an excitation in
                          the swap qubit, both does the sequence both ways.
    chevron_pulse_lengths (list) amplitudes for the chevron pulse
    verbose=False:        (bool) used for verbosity printing in the pulsar
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points

    TODO:
        - move getting the pulse dict to a single function
    '''

    # ############ This getting pulse dict should be a single function
    mw_pulses_qCP = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(mw_pulse_pars_qCP), ' qCP')
    mw_pulses_qS = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(mw_pulse_pars_qS), ' qS')
    # This should come out of this dict in a smarter way
    swap_qCP = {'swap qCP': flux_pulse_pars_qCP}
    swap_qS = {'swap qS': flux_pulse_pars_qS}
    RO_dict = {'RO': RO_pars}
    pulse_dict = {}
    pulse_dict.update(mw_pulses_qCP)
    pulse_dict.update(mw_pulses_qS)
    pulse_dict.update(swap_qCP)
    pulse_dict.update(swap_qS)
    pulse_dict.update(RO_dict)
    # End of the getting pulse dict

    # Getting the minus flux pulses should also be in the get pulse dict
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars_qCP)
    pulse_dict['mswap qCP'] = deepcopy(pulse_dict['swap qCP'])
    pulse_dict['mswap qS'] = deepcopy(pulse_dict['swap qS'])
    pulse_dict['mswap qCP']['amplitude'] = -pulse_dict['swap qCP']['amplitude']
    pulse_dict['mswap qS']['amplitude'] = -pulse_dict['swap qS']['amplitude']

    pulse_dict.update({'mFlux_pulse': minus_flux_pulse_pars})
    pulse_dict.update({'dead_time_pulse':
                       {'pulse_type': 'SquarePulse',
                        'pulse_delay': flux_pulse_pars_qCP['dead_time'],
                        'channel': flux_pulse_pars_qCP['channel'],
                        'amplitude': 0,
                        'length': 0.}})

    # Pulse is used to set the starting ref_point for the compensation pulses

    seq_name = 'chevron_with_excited_bus_2Qubits'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    # seq has to have at least 2 elts
    for i, chevron_pulse_length in enumerate(chevron_pulse_lengths):
        pulse_dict['swap qCP']['square_pulse_length'] = chevron_pulse_length
        pulse_dict['mswap qCP']['square_pulse_length'] = chevron_pulse_length
        if excitations == 'both':
            if (i < (len(chevron_pulse_lengths)-4*cal_points)/2):
                excitation = False
            else:
                excitation = True
        elif excitations == 0:
            excitation = False
        elif excitations == 1:
            excitation = True
        else:
            raise ValueError(
                'excitations {} not recognized'.format(excitations))
        if cal_points and (i == (len(chevron_pulse_lengths)-4)):
            pulse_combinations = ['I qCP', 'I qS', 'RO']
        elif cal_points and (i == (len(chevron_pulse_lengths)-3)):
            pulse_combinations = ['I qCP', 'X180 qS', 'RO']
        elif cal_points and ((i == len(chevron_pulse_lengths)-2)):
            pulse_combinations = ['X180 qCP', 'I qS', 'RO']
        elif cal_points and (i == (len(chevron_pulse_lengths)-1)):
            pulse_combinations = ['X180 qCP', 'X180 qS', 'RO']
        else:
            if excitation:
                pulse_combinations = ['X180 qCP', 'X180 qS'] +\
                    [ 'swap qS'] +\
                    ['swap qCP'] +\
                    ['swap qS'] +\
                    ['X180 qCP', 'X180 qS'] +\
                    ['RO'] +\
                    ['dead_time_pulse']+['mswap qS']*2+['mswap qCP']
            else:
                pulse_combinations = ['X180 qCP', 'I qS'] +\
                    ['swap qS'] +\
                    ['swap qCP'] +\
                    ['swap qS'] +\
                    ['X180 qCP', 'I qS'] +\
                    ['RO'] +\
                    ['dead_time_pulse']+['mswap qS']*2+['mswap qCP']
                # correcting timings
        pulses = []
        for p in pulse_combinations:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(i, station, pulses)
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def XSWAPxy(phis, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
            excite=True,
            verbose=False,
            distortion_dict=None,
            upload=True,
            return_seq=False):
    '''

    '''
    if flux_pulse_pars is None:
        flux_pulse_pars = {'pulse_type': 'SquarePulse',
                           'pulse_delay': .1e-6,
                           'channel': 'ch3',
                           'amplitude': 0.5,
                           'length': .1e-6}
    # flux_pulse_pars['amplitude'] = 0.
    minus_flux_pulse_pars = {'pulse_type': 'SquarePulse',
                             'pulse_delay': 0.,  # will be overwritten
                             'channel': flux_pulse_pars['channel'],
                             'amplitude': -flux_pulse_pars['amplitude'],
                             'length': flux_pulse_pars['length']}
    original_delay = deepcopy(RO_pars)['pulse_delay']

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_flux_pulse_pars['length']),
                       'channel': flux_pulse_pars['channel'],
                       'amplitude': 0,
                       'length': 0.}

    seq_name = 'Chevron_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    lngt = flux_pulse_pars['length']
    minus_flux_pulse_pars['length'] = lngt
    for i in range(rep_max):  # seq has to have at least 2 elts
        # correcting timings
        pulse_buffer = 50e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])
        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + lngt
        dead_time_pulse['pulse_delay'] = RO_pars['pulse_delay']

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']
        if excite:
            init_pulse = pulses['X180']
        else:
            init_pulse = pulses['I']

        buffer_swap = 50e-9

        sec_flux_pulse_pars = deepcopy(flux_pulse_pars)
        flux_pulse_pars['pulse_delay'] = flux_pulse_pars[
            'length'] + buffer_swap
        sec_minus_flux_pulse_pars = deepcopy(minus_flux_pulse_pars)
        sec_minus_flux_pulse_pars['pulse_delay'] = minus_flux_pulse_pars[
            'length'] + buffer_swap
        if i == 0:
            pulse_list = [init_pulse,
                          flux_pulse_pars,
                          RO_pars,
                          minus_flux_pulse_pars,
                          dead_time_pulse]
        else:
            pulse_list = [init_pulse,
                          flux_pulse_pars] + [sec_flux_pulse_pars]*(2*i) + [RO_pars,
                                                                            minus_flux_pulse_pars] + [sec_minus_flux_pulse_pars]*(2*i) + [dead_time_pulse]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        # + ((-int(lngt*1e9)) % 50)*1e-9
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(np.arange(rep_max))+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(
            len(np.arange(rep_max))+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def chevron_seq_cphase(lengths, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
                       cphase_pulse_pars=None, artificial_detuning=None,
                       phase_2=None,
                       verbose=False,
                       distortion_dict=None,
                       upload=True,
                       return_seq=False):
    '''

    '''
    if flux_pulse_pars is None:
        flux_pulse_pars = {'pulse_type': 'SquarePulse',
                           'pulse_delay': .1e-6,
                           'channel': 'ch3',
                           'amplitude': 0.5,
                           'length': .1e-6}
    # flux_pulse_pars['amplitude'] = 0.
    minus_flux_pulse_pars = {'pulse_type': 'SquarePulse',
                             'pulse_delay': 0.,  # will be overwritten
                             'channel': flux_pulse_pars['channel'],
                             'amplitude': -flux_pulse_pars['amplitude'],
                             'length': flux_pulse_pars['length']}
    original_delay = deepcopy(RO_pars)['pulse_delay']

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_flux_pulse_pars['length']),
                       'channel': flux_pulse_pars['channel'],
                       'amplitude': 0,
                       'length': 0.}

    seq_name = 'Chevron_seq'
    minus_cphase_pulse_pars = deepcopy(cphase_pulse_pars)
    minus_cphase_pulse_pars['amplitude'] = - \
        minus_cphase_pulse_pars['amplitude']
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    for i, lngt in enumerate(lengths):  # seq has to have at least 2 elts
        cphase_pulse_pars['length'] = lngt
        minus_cphase_pulse_pars['length'] = lngt
        cphase_pulse_pars['frequency'] = 0.5/lngt
        minus_cphase_pulse_pars['frequency'] = 0.5/lngt
        # cphase_pulse_pars['phase'] = -(90./np.pi)*(cphase_pulse_pars['pulse_delay'])/lngt
        # minus_cphase_pulse_pars['phase'] = -(90./np.pi)*(minus_cphase_pulse_pars['pulse_delay'])/lngt
        # correcting timings
        pulse_buffer = 50e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])
        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + lngt
        dead_time_pulse['pulse_delay'] = RO_pars['pulse_delay']

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']
        firstY90m = deepcopy(pulses['Y90'])
        firstY90m['pulse_delay'] = flux_pulse_pars['length'] + 30e-9
        secondY90m = deepcopy(pulses['X90'])

        if phase_2 is not None:
            secondY90m['phase'] = phase_2
        elif artificial_detuning is not None:
            secondY90m['phase'] = (lngt-lengths[0]) * artificial_detuning * 360
        secondY90m['pulse_delay'] = lngt + 20e-9
        pulse_list = [pulses['X180'],
                      flux_pulse_pars,
                      firstY90m,
                      cphase_pulse_pars,
                      secondY90m,
                      RO_pars,
                      minus_flux_pulse_pars,
                      pulses['I'],
                      minus_cphase_pulse_pars,
                      dead_time_pulse]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6 + ((-int(lngt*1e9)) % 50)*1e-9

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(lengths)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(
            len(lengths)+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def BusT2(times, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
          verbose=False, distortion_dict=None,
          upload=True, return_seq=False):
    '''

    '''
    if flux_pulse_pars is None:
        raise ValueError('Need flux parameters for the gate.')
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    minus_flux_pulse_pars['amplitude'] = -minus_flux_pulse_pars['amplitude']

    original_delay = deepcopy(RO_pars)['pulse_delay']

    seq_name = 'BusT2_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_flux_pulse_pars['length']),
                       'channel': flux_pulse_pars['channel'],
                       'amplitude': 0,
                       'length': 0.}
    for i, tt in enumerate(times):
        # correcting timings
        pulse_buffer = 50e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])

        flux_pulse_pars_2 = deepcopy(flux_pulse_pars)
        # flux_pulse_pars_2['amplitude'] = 0.
        flux_pulse_pars_2['pulse_delay'] = tt + flux_pulse_pars['length']

        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + flux_pulse_pars['length']

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']

        minus_flux_pulse_pars_2 = deepcopy(flux_pulse_pars_2)
        minus_flux_pulse_pars_2['amplitude'] = - \
            minus_flux_pulse_pars_2['amplitude']

        dead_time_pulse['pulse_delay'] = RO_pars['pulse_delay']

        pulse_list = [pulses['Y90'], flux_pulse_pars, flux_pulse_pars_2, pulses['Y90'],
                      RO_pars, minus_flux_pulse_pars, minus_flux_pulse_pars_2,
                      dead_time_pulse]

        # This ensures fixed point
        pulse_list[0]['pulse_delay'] += 0.01e-6  # + ((-int(tt*1e9)) % 50)*1e-9

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(times)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(
            len(times)+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def BusEcho(times, mw_pulse_pars, RO_pars, artificial_detuning=None, flux_pulse_pars=None,
            verbose=False, distortion_dict=None,
            upload=True, return_seq=False):
    '''

    '''
    if flux_pulse_pars is None:
        raise ValueError('Need flux parameters for the gate.')
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    minus_flux_pulse_pars['amplitude'] = -minus_flux_pulse_pars['amplitude']

    original_delay = deepcopy(RO_pars)['pulse_delay']

    seq_name = 'BusEcho_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    pulse_pars_x2 = deepcopy(pulses['X90'])

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_flux_pulse_pars['length']),
                       'channel': flux_pulse_pars['channel'],
                       'amplitude': 0,
                       'length': 0.}
    for i, tt in enumerate(times):
        # correcting timings
        pulse_buffer = 50e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])

        flux_pulse_pars_2 = deepcopy(flux_pulse_pars)
        # flux_pulse_pars_2['amplitude'] = 0.
        flux_pulse_pars_2['pulse_delay'] = tt*0.5 + flux_pulse_pars['length']

        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + flux_pulse_pars['length']

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']

        minus_flux_pulse_pars_2 = deepcopy(flux_pulse_pars_2)
        minus_flux_pulse_pars_2['amplitude'] = - \
            minus_flux_pulse_pars_2['amplitude']

        dead_time_pulse['pulse_delay'] = RO_pars['pulse_delay']
        if artificial_detuning is not None:
            pulse_pars_x2['phase'] = (tt-times[0]) * artificial_detuning * 360

        pulse_list = [pulses['Y90'], flux_pulse_pars, flux_pulse_pars_2, pulses['X180'],
                      flux_pulse_pars, flux_pulse_pars_2, pulse_pars_x2,
                      RO_pars, minus_flux_pulse_pars, minus_flux_pulse_pars_2, pulses[
                          'I'],
                      minus_flux_pulse_pars, minus_flux_pulse_pars_2, dead_time_pulse]

        # This ensures fixed point
        pulse_list[0]['pulse_delay'] += 0.01e-6  # + ((-int(tt*1e9)) % 50)*1e-9

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(times)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(
            len(times)+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def preload_kernels_func(distortion_dict):
    logging.warning('deprecated. you should use the kernel object directly')
    # output_dict = {ch: [] for ch in distortion_dict['ch_list']}
    # for ch in distortion_dict['ch_list']:
    #     for kernel in distortion_dict[ch]:
    #         if kernel is not '':
    #             if kernel in cached_kernels.keys():
    #                 print('Cached {}'.format(kernel_dir+kernel))
    #                 output_dict[ch].append(cached_kernels[kernel])
    #             else:
    #                 print('Loading {}'.format(kernel_dir+kernel))
    #                 # print(os.path.isfile('kernels/'+kernel))
    #                 kernel_vec = np.loadtxt(kernel_dir+kernel)
    #                 output_dict[ch].append(kernel_vec)
    #                 cached_kernels.update({kernel: kernel_vec})
    # return output_dict
    return None


def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing the input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude']*pulse_pars['amp90_scale']

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] = 90
    pulses['mY180']['phase'] = 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] = 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] = 90

    return pulses


def SWAP_CZ_SWAP_phase_corr_swp(operation_dict, qS, qCZ,
                                sweep_qubit,
                                RO_target='all',
                                rec_phases=None,
                                phase_corr_amps=None,
                                distortion_dict=None,
                                CZ_disabled=False,
                                excitations='both_cases',
                                # 0 in 1st half and 1 in 2nd
                                cal_points_with_flux_pulses=True,
                                verbose=False,
                                upload=True):
    '''
    Sequence that swaps qS with the bus and does CPhase between qCZ and the bus
        X180 qS - Ym90 qCZ - swap qS,B - CPhase qCZ,B - swap qS,B - fphi90 qCZ
        - X180 qS - RO

    the keyword swap target and control reverses the
    qubit roles during a second sweep:

        X180 qCZ - Ym90 qS - swap qS,B - CPhase qCZ,B - swap qS,B - fphi90 qS
        - X180 qCZ - RO

    qS is the "SWAP qubit"
    qCZ is the "C-Phase qubit"
    '''
    sequencer_config = operation_dict['sequencer_config']

    seq_name = 'SWAP_CZ_SWAP_phase_corr_swp_{}_{}'.format(qS, qCZ)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    if CZ_disabled:
        operation_dict['CZ '+qCZ]['amplitude'] = 0
        operation_dict['CZ '+qCZ]['phase_corr_pulse_amp'] = 0

    ################################################
    # Creating additional pulses for this sequence #
    ################################################
    # the recovery SWAP is identical to the regular SWAP operation, unless
    # an rSWAP is explicitly contained in the operation dict

    operation_dict['phi90 ' + qCZ] = deepcopy(operation_dict['Y90 ' + qCZ])
    operation_dict['phi90 ' + qS] = deepcopy(operation_dict['Y90 ' + qS])
    # operation_dict['rSWAP ' + qS] = deepcopy(operation_dict['SWAP ' + qS])
    if ('rSWAP ' + qS) not in operation_dict.keys():
        operation_dict['rSWAP ' + qS] = deepcopy(operation_dict['SWAP ' + qS])
    operation_dict['CZ_corr ' + qCZ]['ref_point'] = 'simultaneous'

    # seq has to have at least 2 elts
    # mid_point_phase_amp = phase_corr_amps[len(phase_corr_amps[:-4])//4]

    if (rec_phases is None) and (phase_corr_amps is None):
        raise Exception('Must sweep either recovery phase or phase corr amp')
    if rec_phases is None:
        rec_phases = [90]*len(phase_corr_amps)
    if phase_corr_amps is None:
        if sweep_qubit == qCZ:
            phase_corr_amp = operation_dict[
                'CZ_corr ' + qCZ]['amplitude']
        else:
            phase_corr_amp = operation_dict[
                'SWAP_corr ' + qS]['amplitude']
        phase_corr_amps = [phase_corr_amp]*len(rec_phases)

    ############################################
    # Generating the elements  #
    ############################################
    for i in range(len(phase_corr_amps)):
        operation_dict['phi90 ' + qCZ]['phase'] = rec_phases[i]
        operation_dict['phi90 ' + qS]['phase'] = rec_phases[i]
        if sweep_qubit == qCZ:
            operation_dict['CZ_corr ' + qCZ]['amplitude'] \
                = phase_corr_amps[i]
        else:
            operation_dict['SWAP_corr ' + qS]['amplitude'] \
                = phase_corr_amps[i]
        ######################
        # The base seqeunce  #
        ######################
        if sweep_qubit == qCZ:
            pulse_combinations = (
                ['I ' + qS, 'mY90 ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'phi90 ' + qCZ, 'I '+qCZ, 'I '+qS, 'RO '+RO_target])
            if (excitations == 1 or (excitations == 'both_cases' and
                                     i >= (len(phase_corr_amps)-4)/2)):
                # Put a single excitation in the Swap qubit by replacing Id
                pulse_combinations[0] = 'X180 ' + qS
                pulse_combinations[-2] = 'X180 ' + qS

        elif sweep_qubit == qS:
            pulse_combinations = (
                ['mY90 ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'mY90 ' + qCZ, 'Y90 '+qCZ, 'phi90 '+qS, 'RO '+RO_target])
            # Two pulses on the CZ qubit are to emulate tomo pulses
            if (excitations == 1 or (excitations == 'both_cases' and
                                     (i >= len(phase_corr_amps)-4)/2)):
                # Put a single excitation in the CZ qubit by replacing Id
                pulse_combinations[1] = 'X180 ' + qCZ
                pulse_combinations[-3] = 'X180 ' + qCZ
        else:
            raise ValueError('Sweep qubit "{}" must be either "{}" or "{}"'.format(
                sweep_qubit, qS, qCZ))
        ############################################
        #             calibration points           #
        ############################################
        if i == (len(rec_phases) - 4):
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'I '+qCZ, 'I '+qS, 'RO '+RO_target])
        elif i == (len(rec_phases) - 3):
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'I '+qCZ, 'X180 '+qS, 'RO '+RO_target])
        elif i == (len(rec_phases) - 2):
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'X180 '+qCZ, 'I '+qS, 'RO '+RO_target])
        elif i == (len(rec_phases) - 1):
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'X180 '+qCZ, 'X180 '+qS, 'RO '+RO_target])

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} '.format(i+1,
                                                       len(phase_corr_amps)),
                  end='')
            if i == len(phase_corr_amps):
                print()
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def rSWAP_amp_sweep(operation_dict, qS, qCZ,
                    RO_target='all',
                    recovery_swap_amps=np.arange(0.3, 0.55, 0.01),
                    distortion_dict=None,
                    CZ_disabled=False,
                    emulate_cross_driving=False,
                    cal_points_with_flux_pulses=True,
                    verbose=False,
                    upload=True, **kw):
    '''
    Sequence that swaps qS with the bus and does CPhase between qCZ and the bus
        X180 qS - Ym90 qCZ - swap qS,B - CPhase qCZ,B - swap qS,B - fphi90 qCZ
        - X180 qS - RO

    kw is not used, but implemented to avoid crashing when passed argument name

    the keyword swap target and control reverses the
    qubit roles during a second sweep:

        X180 qCZ - Ym90 qS - swap qS,B - CPhase qCZ,B - swap qS,B - fphi90 qS
        - X180 qCZ - RO

    qS is the "SWAP qubit"
    qCZ is the "C-Phase qubit"
    '''
    sequencer_config = operation_dict['sequencer_config']

    seq_name = 'SWAP_CZ_SWAP_phase_corr_swp_{}_{}'.format(qS, qCZ)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    if CZ_disabled:
        operation_dict['CZ '+qCZ]['amplitude'] = 0
        operation_dict['CZ '+qCZ]['phase_corr_pulse_amp'] = 0

    ################################################
    # Creating additional pulses for this sequence #
    ################################################
    # the recovery SWAP is identical to the regular SWAP operation, unless
    # an rSWAP is explicitly contained in the operation dict
    if ('rSWAP ' + qS) not in operation_dict.keys():
        operation_dict['rSWAP ' + qS] = deepcopy(operation_dict['SWAP ' + qS])
    operation_dict['CZ_corr ' + qCZ]['ref_point'] = 'simultaneous'

    rSWAP_cals = np.mean(recovery_swap_amps[:-4])

    ############################################
    # Generating the elements  #
    ############################################
    for i in range(len(recovery_swap_amps)):
        operation_dict['rSWAP ' + qS]['amplitude'] = recovery_swap_amps[i]
        ######################
        # The base sequence  #
        ######################
        pulse_combinations = (
            ['X180 ' + qS, 'I ' + qCZ,
             'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
             'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
             'I ' + qCZ, 'I '+qCZ, 'I '+qS, 'RO '+RO_target])
        if emulate_cross_driving is True:
            pulse_combinations[1] = 'Y90 ' + qCZ
            pulse_combinations[7] = 'mY90 ' + qCZ
        ############################################
        #             calibration points           #
        ############################################
        if i == (len(recovery_swap_amps) - 4):
            operation_dict['rSWAP ' + qS]['amplitude'] = rSWAP_cals
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'I '+qCZ, 'I '+qS, 'RO '+RO_target])
        elif i == (len(recovery_swap_amps) - 3):
            operation_dict['rSWAP ' + qS]['amplitude'] = rSWAP_cals
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'I '+qCZ, 'X180 '+qS, 'RO '+RO_target])
        elif i == (len(recovery_swap_amps) - 2):
            operation_dict['rSWAP ' + qS]['amplitude'] = rSWAP_cals
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'X180 '+qCZ, 'I '+qS, 'RO '+RO_target])
        elif i == (len(recovery_swap_amps) - 1):
            operation_dict['rSWAP ' + qS]['amplitude'] = rSWAP_cals
            pulse_combinations = (
                ['I ' + qS, 'I ' + qCZ,
                 'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
                 'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
                 'I ' + qCZ, 'X180 '+qCZ, 'X180 '+qS, 'RO '+RO_target])

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} '.format(i+1,
                                                       len(recovery_swap_amps)),
                  end='')
            if i == len(recovery_swap_amps):
                print()
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def BusT1(operation_dict, q0,
          times,
          distortion_dict=None,
          verbose=False,
          upload=True):
    '''

    '''
    sequencer_config = operation_dict['sequencer_config']

    seq_name = 'BusT1_{}'.format(q0)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    base_sequence = ['X180 ' + q0, 'SWAP '+q0, 'rSWAP '+q0, 'RO '+q0]

    for i, tau in enumerate(times):
        operation_dict['rSWAP '+q0] = deepcopy(operation_dict['SWAP '+q0])
        operation_dict['rSWAP '+q0]['pulse_delay'] = tau
        pulse_combinations = base_sequence
        ############################################
        #             calibration points           #
        ############################################
        if (i == (len(times) - 4)) or (i == (len(times)-3)):
            pulse_combinations = (['RO ' + q0])
        elif i == (len(times) - 2) or i == (len(times) - 1):
            pulse_combinations = (['X180 '+q0, 'RO ' + q0])
        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} \t'.format(i+1, len(times)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def FluxTrack(operation_dict, q0,
              pulse_lengths=np.arange(0, 120e-9, 2e-9),
              verbose=False,
              distortion_dict=None,
              upload=True,
              cal_points=True):
    '''
    FluxTrack sequence where a poitive and negative SWAP are implemented
    amplitude and length are not varied.
        X180 - SWAP(l) - RO


    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''

    seq_name = 'FluxTrack_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']

    # SWAP_amp = operation_dict['SWAP '+q0]['amplitude']
    mSWAP = deepcopy(operation_dict['SWAP ' + q0])
    mSWAP['amplitude'] *= -1
    operation_dict['mSWAP ' + q0] = mSWAP
    # seq has to have at least 2 elts
    total_elts = 2 + cal_points*4
    for i in range(total_elts):
        if i == 0:
            pulse_combinations = ['X180 ' + q0, 'SWAP ' + q0, 'RO '+q0]
        elif i == 1:
            pulse_combinations = ['X180 ' + q0, 'mSWAP ' + q0, 'RO '+q0]
        # Calibration points
        elif i == 2 or i == 3:
            pulse_combinations = ['RO '+q0]
        elif i == 4 or i == 5:
            pulse_combinations = ['X180 ' + q0, 'RO '+q0]
        else:
            raise Exception('larger index than expected')

        # # this converts negative pulse lenghts to negative pulse amplitudes
        # operation_dict[
        #     'SWAP '+q0]['amplitude'] = np.abs(operation_dict['SWAP '+q0]['amplitude'])*(-1)**i
        # if cal_points and (i == (len(pulse_lengths)-4) or
        #                    i == (len(pulse_lengths)-3)):
        #     pulse_combinations = ['RO '+q0]
        # elif cal_points and (i == (len(pulse_lengths)-2) or
        #                      i == (len(pulse_lengths)-1)):
        #     pulse_combinations = ['X180 ' + q0, 'RO ' + q0]
        # else:
        #     pulse_combinations = ['X180 '+q0, 'SWAP ' + q0, 'RO '+q0]

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]

        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\r Distorting element {}/{} '.format(i+1, total_elts),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list



def Ramsey_with_flux_pulse_meas_seq(thetas, qb, X90_separation, verbose=False,
                                    upload=True, return_seq=False,
                                    cal_points=False):
    '''
    Performs a Ramsey with interleaved Flux pulse

    Timings of sequence
           <----- |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase

    timing of the flux pulse relative to the center of the first X90 pulse

    Args:
        thetas: numpy array of phase shifts for the second pi/2 pulse
        qb: qubit object (must have the methods get_operation_dict(),
        get_drive_pars() etc.
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''

    qb_name = qb.name
    operation_dict = qb.get_operation_dict()
    pulse_pars = qb.get_drive_pars()
    RO_pars = qb.get_RO_pars()
    seq_name = 'Measurement_Ramsey_sequence_with_Flux_pulse'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    flux_pulse = operation_dict["flux "+qb_name]
    # Used for checking dynamic phase compensation
    # if flux_pulse['amplitude'] != 0:
    #     flux_pulse['basis_rotation'] = {qb_name: -80.41028958782647}

    flux_pulse['ref_point'] = 'end'
    X90_2 = deepcopy(pulses['X90'])
    X90_2['pulse_delay'] = X90_separation - flux_pulse['pulse_delay'] \
                            - X90_2['nr_sigma']*X90_2['sigma']
    X90_2['ref_point'] = 'start'

    for i, theta in enumerate(thetas):
        X90_2['phase'] = theta*180/np.pi
        if cal_points and (i == (len(thetas)-4) or i == (len(thetas)-3)):
            el = multi_pulse_elt(i, station, [RO_pars])
        elif cal_points and (i == (len(thetas)-2) or i == (len(thetas)-1)):
            flux_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def dynamic_phase_meas_seq(thetas, qb_name, CZ_pulse_name,
                           operation_dict, verbose=False,
                           upload=True, return_seq=False,
                           cal_points=True):
    '''
    Performs a Ramsey with interleaved Flux pulse

    Timings of sequence
           <----- |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase

    timing of the flux pulse relative to the center of the first X90 pulse

    Args:
        thetas: numpy array of phase shifts for the second pi/2 pulse
        qb: qubit object (must have the methods get_operation_dict(),
            get_drive_pars() etc.
        CZ_pulse_name: str of the form
            'CZ ' + qb_target.name + ' ' + qb_control.name
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''

    # qb_name = qb.name
    # operation_dict = qb.get_operation_dict()
    # pulse_pars = qb.get_drive_pars()
    # RO_pars = qb.get_RO_pars()
    seq_name = 'Measurement_dynamic_phase'
    seq = sequence.Sequence(seq_name)
    el_list = []

    # pulses = get_pulse_dict_from_pars(pulse_pars)
    flux_pulse = deepcopy(operation_dict[CZ_pulse_name])
    X90_2 = deepcopy(operation_dict['X90 ' + qb_name])
    RO_pars = deepcopy(operation_dict['RO ' + qb_name])
    #
    # # putting control qb in |e> state:
    # qbc_name = operation_dict[CZ_pulse_name]['target_qubit']
    # X180_qbc = operation_dict['X180s ' + qbc_name]

    # Used for checking dynamic phase compensation
    # if flux_pulse['amplitude'] != 0:
    #     flux_pulse['basis_rotation'] = {qb_name: -80.41028958782647}

    # for j, amp in enumerate([0, flux_pulse_amp]):
    #     flux_pulse['amplitude'] = amp
    #     elt_name_offset = j*len(thetas)

    # flux_pulse['amplitude'] = flux_pulse_amp
    pulse_list = [operation_dict['X90 ' + qb_name], flux_pulse]
    for i, theta in enumerate(thetas):
        if theta == thetas[-4]:
            # after this point, we do not want the flux pulse
            pulse_list = [operation_dict['X90 ' + qb_name]]
        #     flux_pulse['amplitude'] = 0
        #     if 'aux_channels_dict' in flux_pulse:
        #         for ch in flux_pulse['aux_channels_dict']:
        #             flux_pulse['aux_channels_dict'][ch] = 0

        if cal_points and (theta == thetas[-4] or theta == thetas[-3]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qb_name],
                                  RO_pars])
        elif cal_points and (theta == thetas[-2] or theta == thetas[-1]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qb_name],
                                  RO_pars])
        else:
            X90_2['phase'] = theta * 180 / np.pi
            pulse_list_complete = pulse_list + [X90_2, RO_pars]
            el = multi_pulse_elt(i, station, pulse_list_complete)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Chevron_flux_pulse_length_seq(lengths, qb_control, qb_target, spacing=50e-9,
                                  verbose=False, cal_points=False,
                                  upload=True, return_seq=False):

    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|
                         <------>
                         spacing
    args:
        lengths: np.array containing the lengths of the fluxpulses
        qb_control: instance of the qubit class
        qb_control: instance of the qubit class
        spacing: float

    '''
    qb_name_control = qb_control.name
    operation_dict_control = qb_control.get_operation_dict()
    pulse_pars_control = qb_control.get_drive_pars()
    pulse_pars_target = qb_target.get_drive_pars()
    RO_pars_target = qb_target.get_RO_pars()
    seq_name = 'Chevron_sequence_flux_ampl_sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses_control = get_pulse_dict_from_pars(pulse_pars_control)
    pulses_target = get_pulse_dict_from_pars(pulse_pars_target)

    X180_control = pulses_control['X180']
    X180_control['ref_point'] = 'start'
    X180_target = pulses_target['X180']
    X180_target['ref_point'] = 'start'

    flux_pulse_control = operation_dict_control["flux "+qb_name_control]
    flux_pulse_control['ref_point'] = 'end'
    flux_pulse_control['pulse_delay'] = spacing

    max_length = np.max(lengths)

    RO_pars_target['ref_point'] = 'start'
    RO_pars_target['pulse_delay'] = max_length + spacing

    if flux_pulse_control['pulse_type'] == 'GaussFluxPulse':
        flux_pulse_control['pulse_delay'] -= flux_pulse_control['buffer']
        RO_pars_target['pulse_delay'] -= 2* flux_pulse_control['buffer']


    for i, length in enumerate(lengths):
        flux_pulse_control['length'] = length

        if cal_points and (i == (len(lengths)-4) or i == (len(lengths)-3)):
            el = multi_pulse_elt(i, station, [RO_pars_target])
        elif cal_points and (i == (len(lengths)-2) or i == (len(lengths)-1)):
            flux_pulse_control['amplitude'] = 0
            el = multi_pulse_elt(i, station, [X180_control, X180_target,
                                              flux_pulse_control,
                                              RO_pars_target])
        else:
            el = multi_pulse_elt(i, station, [X180_control,X180_target,
                                              flux_pulse_control,
                                              RO_pars_target])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Chevron_length_seq_new(hard_sweep_dict, soft_sweep_dict,
                       qbc_name, qbt_name, qbr_name,
                       CZ_pulse_name, operation_dict,
                       upload_all=True,
                       verbose=False, cal_points=False,
                       upload=True, return_seq=False):

    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|

    args:
        lengths: np.array containing the lengths of the fluxpulses
        flux_pulse_amp: namplitude of the fluxpulse
        qb_name_c: control qubit name
        qb_name_t: target qubit name
        CZ_pulse_name: name of CZ pulse in the pulse dict
        operation_dict: contains operation dicts of both qubits

    '''

    seq_name = 'Chevron_hard_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    RO_pulse = deepcopy(operation_dict['RO ' + qbr_name])
    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])

    for param_name, param_value in soft_sweep_dict.items():
        CZ_pulse[param_name] = param_value

    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
    else:
        upload_AWGs = [station.pulsar.get(CZ_pulse['channel'] + '_AWG')] + \
                      [station.pulsar.get(ch + '_AWG') for ch in
                       CZ_pulse['aux_channels_dict']]
        upload_channels = [CZ_pulse['channel']] + \
                          list(CZ_pulse['aux_channels_dict'])

    sweep_points = hard_sweep_dict['values']
    max_swpts = np.max(sweep_points)
    for i, sp in enumerate(sweep_points):
        if hard_sweep_dict['parameter_name'] == 'pulse_length':
            RO_pulse['pulse_delay'] = max_swpts - sp
        CZ_pulse[hard_sweep_dict['parameter_name']] = sp

        if cal_points and (i == (len(sweep_points)-4) or
                           i == (len(sweep_points)-3)):
            el = multi_pulse_elt(i, station, [RO_pulse])
        elif cal_points and (i == (len(sweep_points)-2) or
                             i == (len(sweep_points)-1)):
            CZ_pulse['amplitude'] = 0
            for ch in CZ_pulse['aux_channels_dict']:
                CZ_pulse['aux_channels_dict'][ch] = 0
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbc_name],
                                  operation_dict['X180s ' + qbt_name],
                                  CZ_pulse,
                                  RO_pulse])
        else:
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbc_name],
                                  operation_dict['X180s ' + qbt_name],
                                  CZ_pulse,
                                  RO_pulse])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Chevron_frequency_seq(frequencies, length, flux_pulse_amp,
                           qbc_name, qbt_name, qbr_name,
                           CZ_pulse_name, operation_dict,
                           upload_all=True,
                           verbose=False, cal_points=False,
                           upload=True, return_seq=False):

    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|

    args:
        lengths: np.array containing the lengths of the fluxpulses
        flux_pulse_amp: namplitude of the fluxpulse
        qb_name_c: control qubit name
        qb_name_t: target qubit name
        CZ_pulse_name: name of CZ pulse in the pulse dict
        operation_dict: contains operation dicts of both qubits

    '''

    seq_name = 'Chevron_frequency_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    RO_pulse = deepcopy(operation_dict['RO ' + qbr_name])
    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])

    print(upload_all)
    print(flux_pulse_amp)
    CZ_pulse['amplitude'] = flux_pulse_amp
    CZ_pulse['pulse_length'] = length

    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
    else:
        upload_AWGs = [station.pulsar.get(CZ_pulse['channel'] + '_AWG')]
        upload_channels = [CZ_pulse['channel']]

    for i, frequency in enumerate(frequencies):
        CZ_pulse['frequency'] = frequency

        if cal_points and (i == (len(frequencies)-4) or i == (len(frequencies)-3)):
            el = multi_pulse_elt(i, station, [RO_pulse])
        elif cal_points and (i == (len(frequencies)-2) or i == (len(frequencies)-1)):
            CZ_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbc_name],
                                  operation_dict['X180s ' + qbt_name],
                                  CZ_pulse,
                                  RO_pulse])
        else:
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbc_name],
                                  operation_dict['X180s ' + qbt_name],
                                  CZ_pulse,
                                  RO_pulse])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Chevron_flux_pulse_ampl_seq(ampls, qb_control,
                                qb_target, spacing=50e-9,
                                cal_points=False, verbose=False,
                                upload=True, return_seq=False,
                                ):
    '''
    chevron sequence (sweep of the flux pulse amplitude)

    Timings of sequence

    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  -----------------------------  |RO|
                         <------>               <------>
                         spacing                 spacing
    args:
        lengths: np.array containing the lengths of the fluxpulses
        qb_control: instance of the qubit class
        qb_control: instance of the qubit class
        spacing: float

    '''
    qb_name_control = qb_control.name
    qb_name_target = qb_target.name
    operation_dict_control = qb_control.get_operation_dict()
    operation_dict_target = qb_target.get_operation_dict()
    pulse_pars_control = qb_control.get_drive_pars()
    pulse_pars_target = qb_target.get_drive_pars()
    RO_pars_target = qb_target.get_RO_pars()
    seq_name = 'Chevron_sequence_flux_ampl_sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses_control = get_pulse_dict_from_pars(pulse_pars_control)
    pulses_target = get_pulse_dict_from_pars(pulse_pars_target)

    X180_control = pulses_control['X180']
    X180_control['ref_point'] = 'start'
    X180_target = pulses_target['X180']
    X180_target['ref_point'] = 'start'


    flux_pulse_control = operation_dict_control["flux "+qb_name_control]
    flux_pulse_control['ref_point'] = 'end'
    flux_pulse_control['pulse_delay'] = spacing


    flux_pulse_length = flux_pulse_control['length']

    RO_pars_target['ref_point'] = 'start'
    RO_pars_target['pulse_delay'] = flux_pulse_length + spacing

    for i, ampl in enumerate(ampls):
        flux_pulse_control['amplitude'] = ampl

        if cal_points and (i == (len(ampls)-4) or i == (len(ampls)-3)):
            el = multi_pulse_elt(i, station, [RO_pars_target])
        elif cal_points and (i == (len(ampls)-2) or i == (len(ampls)-1)):
            flux_pulse_control['amplitude'] = 0
            el = multi_pulse_elt(i, station, [X180_control, X180_target,
                                              flux_pulse_control,
                                              RO_pars_target])
        else:
            el = multi_pulse_elt(i, station, [X180_control,X180_target,
                                              flux_pulse_control,
                                              RO_pars_target])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def flux_pulse_CPhase_seq(sweep_points, qb_control, qb_target,
                          sweep_mode='length',
                          X90_phase=0,
                          spacing=50e-9,
                          verbose=False,cal_points=False,
                          upload=True, return_seq=False,
                          measurement_mode='excited_state',
                          reference_measurements=False,
                          upload_AWGs='all',
                          upload_channels='all'
                          ):

    '''
    chevron like sequence (sweep of the flux pulse length)
    where the phase is measures.
    The sequence function is programmed, such that it either can take lengths, amplitudes or phases to sweep.

    Timings of sequence
                                         <-- length -->
                                          or amplitude
    qb_control:    |X180|  ----------   |  fluxpulse   |

    qb_target:     ------- |X90|  ------------------------------|X90|--------  |RO|
                                <------>                       X90_phase
                                spacing
    args:
        sweep_points: np.array containing the lengths of the fluxpulses
        qb_control: instance of the qubit class
        qb_control: instance of the qubit class
        sweep_mode: str, either 'length', 'amplitude' or amplitude
        X90_phase: float, phase of the second X90 pulse in rad
        spacing: float
        measurement_mode (str): either 'excited_state', 'ground_state'
        reference_measurement (bool): if True, a reference measurement with the
                                      control qubit in ground state is added in the
                                      same hard sweep. IMPORTANT: you need to double
                                      the hard sweep points!
                                      e.g. thetas = np.concatenate((thetas,thetas))
    '''

    qb_name_control = qb_control.name
    operation_dict_control = qb_control.get_operation_dict()
    pulse_pars_control = qb_control.get_drive_pars()
    pulse_pars_target = qb_target.get_drive_pars()
    RO_pars_target = qb_target.get_RO_pars()
    seq_name = 'Chevron_sequence_flux_{}_sweep_rel_phase_meas'.format(sweep_mode)
    seq = sequence.Sequence(seq_name)
    el_list = []

    if reference_measurements:
        measurement_mode = 'excited_state'
    pulses_control = get_pulse_dict_from_pars(pulse_pars_control)
    pulses_target = get_pulse_dict_from_pars(pulse_pars_target)

    X180_control = deepcopy(pulses_control['X180'])
    X180_control['ref_point'] = 'start'
    X90_target = pulses_target['X90']
    X90_target['ref_point'] = 'end'
    if measurement_mode == 'ground_state':
        X180_control['amplitude'] = 0.

    X90_target_2 = deepcopy(X90_target)
    X90_target_2['phase'] = X90_phase*180/np.pi
    #     X90_target_2['phase'] = 1

    flux_pulse_control = operation_dict_control["flux "+qb_name_control]
    flux_pulse_control['ref_point'] = 'end'
    flux_pulse_control['delay'] = spacing

    if sweep_mode == 'length':
        max_length = np.max(sweep_points)
    else:
        max_length = flux_pulse_control['length']

    X90_target_2['ref_point'] = 'start'
    X90_target_2['pulse_delay'] = max_length + spacing

    RO_pars_target['ref_point'] = 'end'

    for i, sweep_point in enumerate(sweep_points):

        if reference_measurements:
            if i >= int(len(sweep_points)/2):
                X180_control['amplitude'] = 0.
        if sweep_mode == 'length':
            flux_pulse_control['length'] = sweep_point
        elif sweep_mode == 'amplitude':
            flux_pulse_control['amplitude'] = sweep_point
        elif sweep_mode == 'phase':
            X90_target_2['phase'] = sweep_point*180/np.pi

        if cal_points and (i == (len(sweep_points)-4) or i == (len(sweep_points)-3)):
            el = multi_pulse_elt(i, station, [RO_pars_target])
        elif cal_points and (i == (len(sweep_points)-2) or i == (len(sweep_points)-1)):
            flux_pulse_control['amplitude'] = 0
            el = multi_pulse_elt(i, station, [X180_control, X90_target,
                                              flux_pulse_control, X90_target_2,
                                              RO_pars_target])
        else:
            print(X180_control,'\n',X90_target,'\n',flux_pulse_control,'\n',X90_target_2,'\n',RO_pars_target)
            el = multi_pulse_elt(i, station, [X180_control, X90_target,
                                              flux_pulse_control, X90_target_2,
                                              RO_pars_target])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)


    print(upload_AWGs)      ##################################################
    print(upload_channels)
    for it,el in enumerate(el_list):
        print('it: ',it,' , ',el)
    print(seq)     ###########################################################
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def fluxpulse_scope_sequence(delays, qb, verbose=False,
                             cal_points=False,
                             upload=True,
                             return_seq=False,
                             spacing=30e-9):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ----------------------------  |RO|
       |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->

    args:
        delays: array of delays of the pi pulse w.r.t the flux pulse
        qb: qubit object
        spacing: float, spacing between pulse and RO
    '''
    qb_name = qb.name
    operation_dict = qb.get_operation_dict()
    pulse_pars = qb.get_drive_pars()
    RO_pars = qb.get_RO_pars()
    seq_name = 'Fluxpulse_scope_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    flux_pulse = operation_dict["flux "+qb_name]

    X180_2 = deepcopy(pulses['X180'])
    X180_2['ref_point'] = 'start'
    pulse_delay_offset = -X180_2['sigma']*X180_2['nr_sigma']/2. - \
                         flux_pulse['pulse_delay']
    RO_pars['pulse_delay'] = flux_pulse['length'] + flux_pulse[
        'pulse_delay'] - delays[0] + spacing

    for i, delay in enumerate(delays):
        X180_2['pulse_delay'] = delay + pulse_delay_offset
        if cal_points and (i == (len(delays)-4) or i == (len(delays)-3)):
            el = multi_pulse_elt(i, station, [RO_pars])
        elif cal_points and (i == (len(delays)-2) or i == (len(delays)-1)):
            flux_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station, [flux_pulse, X180_2, RO_pars])
        else:
            el = multi_pulse_elt(i, station, [flux_pulse, X180_2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def fluxpulse_scope_alpha_sequence(delays, qb_name, nzcz_alpha,
                                   CZ_pulse_name, operation_dict,
                                   verbose=False, cal_points=False,
                                   upload=True, upload_all=True,
                                   return_seq=False,
                                   spacing=30e-9):

    seq_name = 'Fluxpulse_scope_alpha_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    cz_pulse = operation_dict[CZ_pulse_name]
    cz_pulse['alpha'] = nzcz_alpha
    X180_2 = deepcopy(operation_dict['X180 ' + qb_name])
    X180_2['ref_point'] = 'start'
    pulse_delay_offset = -X180_2['sigma']*X180_2['nr_sigma']/2. - \
                         cz_pulse['pulse_delay']
    RO_pars = operation_dict['RO ' + qb_name]
    RO_pars['pulse_delay'] = cz_pulse['pulse_length'] + \
                             cz_pulse['buffer_length_start'] + \
                             cz_pulse['buffer_length_end'] + \
                             cz_pulse['pulse_delay'] - delays[0] + spacing
    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
    else:
        upload_AWGs = [station.pulsar.get(cz_pulse['channel'] + '_AWG')] + \
                      [station.pulsar.get(ch + '_AWG') for ch in
                       cz_pulse['aux_channels_dict']]
        upload_channels = [cz_pulse['channel']] + \
                          list(cz_pulse['aux_channels_dict'])

    for i, delay in enumerate(delays):
        X180_2['pulse_delay'] = delay + pulse_delay_offset
        if cal_points and (i == (len(delays)-4) or i == (len(delays)-3)):
            el = multi_pulse_elt(i, station, [RO_pars])
        elif cal_points and (i == (len(delays)-2) or i == (len(delays)-1)):
            el = multi_pulse_elt(i, station, [operation_dict['X180 ' + qb_name],
                                              RO_pars])
        else:
            el = multi_pulse_elt(i, station, [cz_pulse, X180_2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def flux_pulse_CPhase_seq_new(phases, flux_params, max_flux_length,
                              qbc_name, qbt_name, qbr_name,
                              operation_dict,
                              CZ_pulse_name,
                              CZ_pulse_channel,
                              verbose=False, cal_points=False,
                              upload=True, return_seq=False,
                              reference_measurements=False,
                              first_data_point=True
                              ):

    '''
    chevron like sequence (sweep of the flux pulse length)
    where the phase is measures.
    The sequence function is programmed, such that it either can take lengths,
    amplitudes or phases to sweep.

    Timings of sequence
                                         <-- length -->
                                          or amplitude
    qb_control:    |X180|  ----------   |  fluxpulse   |

    qb_target:     ------- |X90|  ------------------------------|X90|------|RO|
                                                             sweep phase

    args:
        sweep_points: np.array containing the lengths of the fluxpulses
        qb_control: instance of the qubit class
        qb_control: instance of the qubit class
        reference_measurement (bool):
            if True, a reference measurement with the
            control qubit in ground state is added in the
            same hard sweep. IMPORTANT: you need to double
            the hard sweep points!
            e.g. thetas = np.concatenate((thetas,thetas))
    '''

    seq_name = 'cphase_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_list = []

    flux_amplitude = flux_params[1]
    flux_length = flux_params[0]
    frequency = 0
    if len(flux_params) > 2:
        frequency = flux_params[2]

    RO_pulse = deepcopy(operation_dict['RO ' + qbr_name])
    RO_pulse['pulse_delay'] = max_flux_length

    X180_control = deepcopy(operation_dict['X180 ' + qbc_name])

    buffer_pulse = deepcopy(operation_dict['flux ' + qbc_name])
    buffer_pulse['length'] = max_flux_length-flux_length
    buffer_pulse['amplitude'] = 0.
    #The virtual flux pulse is uploaded to the I_channel of control qb
    buffer_pulse['channel'] = CZ_pulse_channel

    X90_target_2 = deepcopy(operation_dict['X90 ' + qbt_name])
    X90_target = deepcopy(operation_dict['X90s '+ qbt_name])

    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])
    CZ_pulse['amplitude'] = flux_amplitude
    CZ_pulse['pulse_length'] = flux_length
    CZ_pulse['channel'] = CZ_pulse_channel
    if frequency > 0:
        CZ_pulse['frequency'] = frequency

    pulse_list.append(X180_control)
    pulse_list.append(X90_target)
    pulse_list.append(buffer_pulse)#Buffer pulse in order to fix the X90 separation
    pulse_list.append(CZ_pulse)
    pulse_list.append(X90_target_2)
    pulse_list.append(RO_pulse)

    if not first_data_point:
        reduced_pulse_list = [buffer_pulse, CZ_pulse]
        upload_channels,upload_AWGs = get_required_upload_information\
                                            (reduced_pulse_list, station)
        if X90_target['I_channel'].split('_')[0] in upload_AWGs:
            upload_channels.append(X90_target['I_channel'])
            upload_channels.append(X90_target['Q_channel'])


    else:
        upload_channels, upload_AWGs = get_required_upload_information(
            pulse_list, station)

    for i, phase in enumerate(phases):

        X90_target_2['phase'] = phase*180/np.pi
        if reference_measurements and i >= int(len(phases)/2):
            X180_control['amplitude'] = 0

        if cal_points and (i == (len(phases)-4)
                           or i == (len(phases)-3)):
            el = multi_pulse_elt(i, station, [RO_pulse])
        elif cal_points and (i == (len(phases)-2)
                             or i == (len(phases)-1)):
            CZ_pulse['amplitude'] = 0
            for ch in CZ_pulse['aux_channels_dict']:
                CZ_pulse['aux_channels_dict'][ch] = 0
            el = multi_pulse_elt(i, station,
                                 [X180_control,
                                  X90_target, buffer_pulse,
                                  CZ_pulse, X90_target_2,
                                  RO_pulse])
        else:
            el = multi_pulse_elt(i, station,
                                 [X180_control,
                                  X90_target, buffer_pulse,
                                  CZ_pulse, X90_target_2,
                                  RO_pulse])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        print('uploading channels: ', upload_channels)
        print('of AWGs: ', upload_AWGs)
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def cphase_nz_seq(phases, flux_params_dict,
                  qbc_name, qbt_name,
                  operation_dict,
                  CZ_pulse_name,
                  CZ_pulse_channel,
                  num_cz_gates=1,
                  max_flux_length=None,
                  verbose=False, cal_points=False,
                  num_cal_points=4,
                  upload=True, return_seq=False,
                  first_data_point=True
                  ):

    assert num_cz_gates % 2 != 0

    seq_name = 'cphase_nz_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_list = []
    print(flux_params_dict)
    RO_pulse = deepcopy(operation_dict['RO mux'])
    X180_control = deepcopy(operation_dict['X180 ' + qbc_name])
    X90_target = deepcopy(operation_dict['X90s ' + qbt_name])
    X180_control_2 = deepcopy(operation_dict['X180 ' + qbc_name])
    X90_target_2 = deepcopy(operation_dict['X90 ' + qbt_name])

    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])
    for param_name, param_value in flux_params_dict.items():
        CZ_pulse[param_name] = param_value
    CZ_pulse['channel'] = CZ_pulse_channel

    if max_flux_length is not None:
        max_flux_length += (CZ_pulse['buffer_length_start'] +
                            CZ_pulse['buffer_length_end'])
        max_flux_length *= num_cz_gates
        # RO_pulse['pulse_delay'] = max_flux_length
    print(max_flux_length)

    pulse_list.append(X180_control)
    pulse_list.append(X90_target)

    reduced_pulse_list = []
    if 'pulse_length' in flux_params_dict:
        if max_flux_length is None:
            raise ValueError('Specify "max_flux_length."')
        buffer_pulse = deepcopy(operation_dict['flux ' + qbc_name])
        buffer_pulse['length'] = max_flux_length - flux_params_dict[
            'pulse_length']
        buffer_pulse['amplitude'] = 0.
        #The virtual flux pulse is uploaded to the I_channel of control qb
        buffer_pulse['channel'] = CZ_pulse_channel
        #Buffer pulse in order to fix the X90 separation
        pulse_list.append(buffer_pulse)
        reduced_pulse_list += [buffer_pulse]

    pulse_list.append(num_cz_gates*[CZ_pulse])
    # pulse_list.append(X180_control)
    pulse_list.append(X90_target_2)
    pulse_list.append(RO_pulse)

    if not first_data_point:
        reduced_pulse_list += num_cz_gates*[CZ_pulse]
        upload_channels, upload_AWGs = get_required_upload_information(
            reduced_pulse_list, station)
        if X90_target['I_channel'].split('_')[0] in upload_AWGs:
            upload_channels.append(X90_target['I_channel'])
            upload_channels.append(X90_target['Q_channel'])
    else:
        upload_channels = 'all'
        upload_AWGs = 'all'
        # upload_channels, upload_AWGs = get_required_upload_information(
        #     pulse_list, station)

    unique_phases = np.unique(phases)
    for pipulse in [True, False]:
        if not pipulse:
            X180_control['amplitude'] = 0
        for i, phase in enumerate(unique_phases):
            el_iter = i if pipulse else len(unique_phases)+i
            if cal_points and num_cal_points == 3 and \
                    i == (len(unique_phases) - 3):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['RO mux']])
            elif cal_points and num_cal_points == 3 and \
                    i == (len(unique_phases) - 2):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['X180 ' + qbc_name],
                                      operation_dict['X180s ' + qbt_name],
                                      operation_dict['RO mux']])
            elif cal_points and num_cal_points == 3 and \
                    i == (len(unique_phases) - 1):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['X180 ' + qbc_name],
                                      operation_dict['X180s ' + qbt_name],
                                      operation_dict['X180_ef ' + qbc_name],
                                      operation_dict['X180s_ef ' + qbt_name],
                                      operation_dict['RO mux']])
            elif cal_points and num_cal_points == 4 and \
                    (i == (len(unique_phases) - 4) or
                     i == (len(unique_phases) - 3)):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['RO mux']])
            elif cal_points and num_cal_points == 4 and \
                    (i == (len(unique_phases) - 2) or
                     i == (len(unique_phases) - 1)):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['X180 ' + qbc_name],
                                      operation_dict['X180s ' + qbt_name],
                                      operation_dict['RO mux']])
            elif cal_points and num_cal_points == 2 and \
                    (i == (len(unique_phases) - 2) or
                     i == (len(unique_phases) - 1)):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['RO mux']])
            else:
                X90_target_2['phase'] = phase*180/np.pi
                pulse_list = [X180_control, X90_target]
                if 'pulse_length' in flux_params_dict:
                    pulse_list += [buffer_pulse]
                if not pipulse:
                    X90_target_2['ref_point'] = 'start'
                    pulse_list += num_cz_gates*[CZ_pulse]
                    pulse_list += [X180_control_2,
                                   X90_target_2, RO_pulse]
                else:
                    pulse_list += num_cz_gates * [CZ_pulse]
                    pulse_list += [X90_target_2, RO_pulse]
                el = multi_pulse_elt(el_iter, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
    if upload:
        print('uploading channels: ', upload_channels)
        print('of AWGs: ', upload_AWGs)
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def cphase_nz_seq_old(phases, flux_params, max_flux_length,
                  qbc_name, qbt_name,
                  operation_dict,
                  CZ_pulse_name,
                  CZ_pulse_channel,
                  verbose=False, cal_points=False,
                  upload=True, return_seq=False,
                  reference_measurements=False,
                  first_data_point=True
                  ):

    seq_name = 'cphase_nz_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_list = []

    flux_length = flux_params[0]
    flux_amplitude = flux_params[1]
    alpha = flux_params[2]

    RO_pulse = deepcopy(operation_dict['RO mux'])
    RO_pulse['pulse_delay'] = max_flux_length

    X180_control = deepcopy(operation_dict['X180 ' + qbc_name])

    buffer_pulse = deepcopy(operation_dict['flux ' + qbc_name])
    buffer_pulse['length'] = max_flux_length-flux_length
    buffer_pulse['amplitude'] = 0.
    #The virtual flux pulse is uploaded to the I_channel of control qb
    buffer_pulse['channel'] = CZ_pulse_channel

    X90_target_2 = deepcopy(operation_dict['X90s ' + qbt_name])
    X90_target = deepcopy(operation_dict['X90s ' + qbt_name])

    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])
    CZ_pulse['pulse_length'] = flux_length
    CZ_pulse['amplitude'] = flux_amplitude
    CZ_pulse['alpha'] = alpha
    CZ_pulse['channel'] = CZ_pulse_channel

    pulse_list.append(X180_control)
    pulse_list.append(X90_target)
    pulse_list.append(buffer_pulse)#Buffer pulse in order to fix the X90 separation
    pulse_list.append(CZ_pulse)
    pulse_list.append(X180_control)
    pulse_list.append(X90_target_2)
    pulse_list.append(RO_pulse)

    if not first_data_point:
        reduced_pulse_list = [buffer_pulse, CZ_pulse]
        upload_channels, upload_AWGs = get_required_upload_information(
            reduced_pulse_list, station)
        if X90_target['I_channel'].split('_')[0] in upload_AWGs:
            upload_channels.append(X90_target['I_channel'])
            upload_channels.append(X90_target['Q_channel'])
    else:
        upload_channels, upload_AWGs = get_required_upload_information(
            pulse_list, station)

    unique_phases = np.unique(phases)
    for pipulse in [True, False]:
        if not pipulse:
            X180_control['amplitude'] = 0
        for i, phase in enumerate(unique_phases):
            el_iter = i if pipulse else len(unique_phases)+i
            if cal_points and (i == (len(unique_phases)-4)
                               or i == (len(unique_phases)-3)):
                el = multi_pulse_elt(el_iter, station, [RO_pulse])
            elif cal_points and (i == (len(unique_phases)-2)
                                 or i == (len(unique_phases)-1)):
                el = multi_pulse_elt(el_iter, station,
                                     [operation_dict['X180 ' + qbc_name],
                                      operation_dict['X180s ' + qbt_name],
                                      RO_pulse])
            else:
                X90_target_2['phase'] = phase*180/np.pi
                el = multi_pulse_elt(el_iter, station,
                                     [X180_control,
                                      X90_target, buffer_pulse,
                                      CZ_pulse, X180_control,
                                      X90_target_2,
                                      RO_pulse])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
    if upload:
        print('uploading channels: ', upload_channels)
        print('of AWGs: ', upload_AWGs)
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def CZ_bleed_through_phase_seq(phases, qb_name, CZ_pulse_name, CZ_separation,
                               operation_dict, oneCZ_msmt=False, nr_cz_gates=1,
                               verbose=False, upload=True, return_seq=False,
                               upload_all=True, cal_points=True):
    '''
    Performs a Ramsey-like with interleaved Flux pulse

    Timings of sequence
                           CZ_separation
    |CZ|-|CZ|- ... -|CZ| <---------------> |X90|-|CZ|-|X90|-|RO|

    Args:
        end_times: numpy array of delays after second CZ pulse
        qb: qubit object (must have the methods get_operation_dict(),
            get_drive_pars() etc.
        CZ_pulse_name: str of the form
            'CZ ' + qb_target.name + ' ' + qb_control.name
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''
    # if maximum_CZ_separation is None:
    #     maximum_CZ_separation = CZ_separation

    seq_name = 'CZ Bleed Through phase sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    X90_1 = deepcopy(operation_dict['X90 ' + qb_name])
    X90_2 = deepcopy(operation_dict['X90 ' + qb_name])
    RO_pars = deepcopy(operation_dict['RO ' + qb_name])
    CZ_pulse1 = deepcopy(operation_dict[CZ_pulse_name])
    CZ_pulse_len = CZ_pulse1['pulse_length']
    drag_pulse_len = deepcopy(X90_1['sigma']*X90_1['nr_sigma'])
    spacerpulse = {'pulse_type': 'SquarePulse',
                   'channel': X90_1['I_channel'],
                   'amplitude': 0.0,
                   'length': CZ_separation - drag_pulse_len,
                   'ref_point': 'end',
                   'pulse_delay': 0}
    if oneCZ_msmt:
        spacerpulse_X90 = {'pulse_type': 'SquarePulse',
                           'channel': X90_1['I_channel'],
                           'amplitude': 0.0,
                           'length': CZ_pulse1['buffer_length_start'] +
                                     CZ_pulse1['buffer_length_end'] +
                                     CZ_pulse_len,
                           'ref_point': 'end',
                           'pulse_delay': 0}
        main_pulse_list = [CZ_pulse1, spacerpulse,
                           X90_1, spacerpulse_X90]
    else:
        CZ_pulse2 = deepcopy(operation_dict[CZ_pulse_name])
        main_pulse_list = int(nr_cz_gates)*[CZ_pulse1]
        main_pulse_list += [spacerpulse, X90_1, CZ_pulse2]
    el_main = multi_pulse_elt(0, station,  main_pulse_list,
                              trigger=True, name='el_main')
    el_list.append(el_main)

    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
        # else:
        # upload_AWGs = [station.pulsar.get(CZ_pulse1['channel'] + '_AWG')] + \
        #               [station.pulsar.get(ch + '_AWG') for ch in
        #                CZ_pulse1['aux_channels_dict']]
        # upload_channels = [CZ_pulse1['channel']] + \
        #                   list(CZ_pulse1['aux_channels_dict'])

    for i, theta in enumerate(phases):
        if cal_points and (theta == phases[-4] or theta == phases[-3]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        elif cal_points and (theta == phases[-2] or theta == phases[-1]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        else:
            X90_2['phase'] = theta*180/np.pi
            el = multi_pulse_elt(i+1, station, [X90_2, RO_pars], trigger=False,
                                 name='el_{}'.format(i+1),
                                 previous_element=el_main)
            el_list.append(el)

            seq.append('m_{}'.format(i), 'el_main',  trigger_wait=True)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=False)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def fgge_cphase_seq(phases, flux_params, max_flux_length,
                    qbc_name, qbt_name, qbr_name,
                    operation_dict,
                    CZ_pulse_name,
                    verbose=False,cal_points=False,
                    upload=True, return_seq=False,
                    reference_measurements=False,
                    first_data_point = True):

    seq_name = 'FGGE cphase seq phases sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    CZ_pulse = deepcopy(operation_dict[CZ_pulse_name])
    X90_target_2 = deepcopy(operation_dict['X90 ' + qbt_name])

    if not first_data_point:
        reduced_pulse_list = [buffer_pulse,CZ_pulse]
        upload_channels,upload_AWGs = get_required_upload_information \
            (reduced_pulse_list,station)
        if X90_target['I_channel'].split('_')[0] in upload_AWGs:
            upload_channels.append(X90_target['I_channel'])
            upload_channels.append(X90_target['Q_channel'])


    else:
        upload_channels,upload_AWGs = get_required_upload_information(
            pulse_list, station)

    for i, phase in enumerate(phases):
        if cal_points and (i == (len(phases)-4)
                           or i == (len(phases)-3)):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qbr_name],
                                  operation_dict['RO ' + qbr_name]])
        elif cal_points and (i == (len(phases)-2)
                             or i == (len(phases)-1)):
            CZ_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbr_name],
                                  operation_dict['RO ' + qbr_name]])
        else:
            X90_target_2['phase'] = phase*180/np.pi
            if reference_measurements and i >= int(len(phases)/2):
                X180_target['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbt_name],
                                  operation_dict['X90 ' + qbt_name],
                                  CZ_pulse, X90_target_2,
                                  operation_dict['RO ' + qbr_name]])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        print('uploading channels: ', upload_channels)
        print('of AWGs: ', upload_AWGs)
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name
