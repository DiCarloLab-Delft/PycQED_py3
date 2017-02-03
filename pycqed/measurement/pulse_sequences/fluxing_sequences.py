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
    '''

    '''
    if pulse_pars is None:
        pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6,
                      'dead_time_length': 10e-6}
    minus_pulse_pars = {'pulse_type': 'SquarePulse',
                        'pulse_delay': 3e-6 + pulse_pars['length'] + pulse_pars['pulse_delay'],
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
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def Ram_Z_seq(pulse_dict, q0, distortion_dict,
              inter_pulse_delay=300e-9, recovery_phase=0,
              RO_delay=3e-6,
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

    # Flux correction pulse, should be added automatically
    mZ = deepcopy(pulse_dict['Z '+q0])
    mZ['pulse_delay'] = 1e-6
    mZ['amplitude'] = - mZ['amplitude']
    pulse_dict['mZ '+q0] = mZ

    # Setting the RO very high to ensure no overlap when moving the flux pulse
    pulse_dict['RO ' + q0]['pulse_delay'] = RO_delay
    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        # Calibration points
        if (i == (len(times)-4) or i == (len(times)-3)):
            pulse_list = ['I '+q0, 'RO ' + q0]
        elif (i == (len(times)-2) or i == (len(times)-1)):
            pulse_list = ['X180 '+q0, 'RO ' + q0]
        else:
            pulse_dict['Z '+q0]['pulse_delay'] = tau
            t1 = inter_pulse_delay - tau  # refpoint is start of flux pulse

            recPi2 = deepcopy(pulse_dict['X90 '+q0])
            recPi2['refpoint'] = 'start'
            recPi2['phase'] = recovery_phase
            recPi2['pulse_delay'] = t1
            pulse_dict['recPi2 ' + q0] = recPi2
            pulse_list = ['mX90 '+q0, 'Z '+q0, 'recPi2 ' + q0, 'RO ' + q0,
                          'mZ ' + q0]
        pulses = []
        for p in pulse_list:
            pulses += [pulse_dict[p]]
        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)

    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

    return seq, el_list


def SwapN(operation_dict, q0,
          nr_pulses_list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
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
    fluxing_channel = operation_dict['SWAP '+q0]['channel']
    sequencer_config = operation_dict['sequencer_config']

    # Create the correction pulses
    operation_dict['FluxId '+q0] = deepcopy(operation_dict['SWAP ' + q0])
    # Flux identity
    operation_dict['FluxId '+q0]['amplitude'] = 0

    seq_name = 'SWAPN_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    n_max = nr_pulses_list[-1]
    # seq has to have at least 2 elts
    for i, n in enumerate(nr_pulses_list):
        pulse_combinations = (['X180 ' + q0] +
                              ['FluxId '+q0]*(n_max-n) +
                              ['SWAP '+q0]*(n) +
                              ['X180 ' + q0] + ['RO '+q0])

        # calibration points overwrite the pulse_combinations list
        # All pulses are replaced with identities.
        if cal_points and (i == (len(nr_pulses_list)-4) or
                           i == (len(nr_pulses_list)-3)):
            pulse_combinations = (['I ' + q0] +
                                  ['FluxId '+q0]*(n_max) +
                                  ['I ' + q0] + ['RO '+q0])
        elif cal_points and (i == (len(nr_pulses_list)-2) or
                             i == (len(nr_pulses_list)-1)):
            pulse_combinations = (['I ' + q0] +
                                  ['FluxId '+q0]*(n_max) +
                                  ['X180 ' + q0] + ['RO '+q0])
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

    return seq, el_list


def SwapN_alpha(operation_dict, q0,
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
    fluxing_channel = operation_dict['SWAP '+q0]['channel']
    sequencer_config = operation_dict['sequencer_config']

    # Create the correction pulses
    operation_dict['FluxId '+q0] = deepcopy(operation_dict['SWAP ' + q0])
    # Flux identity
    operation_dict['FluxId '+q0]['amplitude'] = 0

    seq_name = 'SWAPN_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    n_max = nr_pulses_list[-1]
    # seq has to have at least 2 elts
    for i, n in enumerate(nr_pulses_list):
        SWAP_pulse_train = []


        for j in range(n):
            SWAP_pulse = deepcopy(operation_dict['SWAP '+q0])
            # print(SWAP_pulse)
            SWAP_pulse['amplitude'] = SWAP_pulse['amplitude']*(alpha**j)
            SWAP_pulse_train.append(SWAP_pulse)
        pulse_combinations = (['X180 ' + q0] +
                              ['FluxId '+q0]*(n_max-n) +
                              SWAP_pulse_train +
                              ['X180 ' + q0] + ['RO '+q0])

        # calibration points overwrite the pulse_combinations list
        # All pulses are replaced with identities.
        if cal_points and (i == (len(nr_pulses_list)-4) or
                           i == (len(nr_pulses_list)-3)):
            pulse_combinations = (['I ' + q0] +
                                  ['FluxId '+q0]*(n_max) +
                                  ['I ' + q0] + ['RO '+q0])
        elif cal_points and (i == (len(nr_pulses_list)-2) or
                             i == (len(nr_pulses_list)-1)):
            pulse_combinations = (['I ' + q0] +
                                  ['FluxId '+q0]*(n_max) +
                                  ['X180 ' + q0] + ['RO '+q0])
        pulses = []
        if cal_points and (i > (len(nr_pulses_list)-5)):
            for p in pulse_combinations:
                pulses += [operation_dict[p]]
        else:
            for j, p in enumerate(pulse_combinations):
                if j < (n_max-n):
                    pulses += [operation_dict[p]]
                elif j == (n_max-n+1):
                    pulses += SWAP_pulse_train
                elif j > n_max:
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

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

    # Pulse is used to set the starting refpoint for the compensation pulses
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

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

    # Pulse is used to set the starting refpoint for the compensation pulses

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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
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
    #                 print('Cached {}'.format(kernel_dir_path+kernel))
    #                 output_dict[ch].append(cached_kernels[kernel])
    #             else:
    #                 print('Loading {}'.format(kernel_dir_path+kernel))
    #                 # print(os.path.isfile('kernels/'+kernel))
    #                 kernel_vec = np.loadtxt(kernel_dir_path+kernel)
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
    operation_dict['CZ_corr ' + qCZ]['refpoint'] = 'simultaneous'

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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

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
    operation_dict['CZ_corr ' + qCZ]['refpoint'] = 'simultaneous'

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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

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
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

    return seq, el_list
