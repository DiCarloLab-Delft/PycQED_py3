import logging
import os
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd

from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.waveform_control import pulsar
from pycqed.measurement.waveform_control.element import calculate_time_correction

from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt

from importlib import reload
reload(pulse)
from pycqed.measurement.waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
kernel_dir_path = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)
cached_kernels = {}


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


def chevron_seq(mw_pulse_pars, RO_pars,
                flux_pulse_pars,
                pulse_lengths=np.arange(0, 120e-9, 2e-9),
                verbose=False,
                distortion_dict=None,
                upload=True,
                cal_points=True):
    '''
    Chevron sequence where length of the "swap" operation is varied
        X180 - swap(l) - RO

    mw_pulse_pars:        (dict) qubit control pulse pars
    RO_pars:              (dict) qubit RO pars
    flux_pulse_pars:      (dict) flux puplse pars
    nr_pulses_list        (list) nr of swaps gates for each element
    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''
    # renamed as the dict contains the pulse directly
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    # Pulse is used to set the starting refpoint for the compensation pulses
    dead_time_pulse_pars = {'pulse_type': 'SquarePulse',
                            'pulse_delay': flux_pulse_pars['dead_time'],
                            'channel': flux_pulse_pars['channel'],
                            'amplitude': 0,
                            'length': 0.}

    seq_name = 'Chevron_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    normal_flux_amp = flux_pulse_pars['amplitude']
    # seq has to have at least 2 elts
    for i, pulse_length in enumerate(pulse_lengths):
        if pulse_length<0: # this converts negative pulse lenghts to negative pulse amplitudes
                flux_pulse_pars['amplitude'] = -normal_flux_amp
                minus_flux_pulse_pars['amplitude'] = normal_flux_amp

        else:
                flux_pulse_pars['amplitude'] = normal_flux_amp
                minus_flux_pulse_pars['amplitude'] = -normal_flux_amp

        if cal_points and (i == (len(pulse_lengths)-4) or i == (len(pulse_lengths)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(pulse_lengths)-2) or i == (len(pulse_lengths)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            flux_pulse_pars['square_pulse_length'] = abs(pulse_length)
            minus_flux_pulse_pars['square_pulse_length'] = abs(pulse_length)
            pulse_list = [pulses['X180']] + [flux_pulse_pars]+[RO_pars] + \
                [dead_time_pulse_pars] + [minus_flux_pulse_pars]

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




def SwapN(mw_pulse_pars, RO_pars,
          flux_pulse_pars,
          nr_pulses_list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
          verbose=False,
          distortion_dict=None,
          upload=True,
          cal_points=True,
          inter_swap_wait=10e-9):
    '''
    Sequence of N swap operations
        X180 - N*swap - X180 RO

    mw_pulse_pars:        (dict) qubit control pulse pars
    RO_pars:              (dict) qubit RO pars
    flux_pulse_pars:      (dict) flux puplse pars
    nr_pulses_list        (list) nr of swaps gates for each element
    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''
    # renamed as the dict contains the pulse directly
    swap = {'swap': flux_pulse_pars}
    mswap = {'mswap': flux_pulse_pars}
    RO_dict = {'RO': RO_pars}
    pulse_dict = {}
    pulse_dict.update(RO_dict)
    pulse_dict.update(swap)
    pulse_dict.update(mswap)

    pulse_dict['mswap'] = deepcopy(pulse_dict['swap'])
    pulse_dict['mswap']['amplitude'] = -pulse_dict['swap']['amplitude']
    pulse_dict['swap']['pulse_delay'] = inter_swap_wait
    # Pulse is used to set the starting refpoint for the compensation pulses
    pulse_dict.update({'dead_time_pulse':
                       {'pulse_type': 'SquarePulse',
                        'pulse_delay': flux_pulse_pars['dead_time'],
                        'channel': flux_pulse_pars['channel'],
                        'amplitude': 0,
                        'length': 0.}})

    seq_name = 'SWAPN_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    mw_pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    pulse_dict.update(mw_pulses)

    # seq has to have at least 2 elts
    for i, nr_pulses in enumerate(nr_pulses_list):
        if cal_points and (i == (len(nr_pulses_list)-4) or i == (len(nr_pulses_list)-3)):
            pulse_combinations = ['I'] + ['RO']
        elif cal_points and (i == (len(nr_pulses_list)-2) or i == (len(nr_pulses_list)-1)):
            pulse_combinations = ['X180'] + ['RO']
        else:
            # correcting timings
            pulse_combinations = ['X180'] + ['swap']*(nr_pulses) \
                + ['X180']+['RO'] + \
                ['dead_time_pulse'] + ['mswap']*(nr_pulses)

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



def swap_CP_swap_2Qubits(mw_pulse_pars_qCP, mw_pulse_pars_qS,
                         flux_pulse_pars_qCP, flux_pulse_pars_qS,
                         RO_pars,
                         distortion_dict,
                         CPhase=True,
                         excitations='both',
                         inter_swap_wait=100e-9,
                         phases=np.linspace(0, 720, 41),
                         verbose=False,
                         upload=True,
                         reverse_control_target=False,
                         cal_points=True):
    '''
    Sequence that swaps qS with the bus and does CPhase between qCP and the bus
        X180 qS - Ym90 qCP - swap qS,B - CPhase qCP,B - swap qS,B - fphi90 qCP
        - X180 qS - RO

    the keyword swap target and control reverses the
    qubit roles during a second sweep:

        X180 qCP - Ym90 qS - swap qS,B - CPhase qCP,B - swap qS,B - fphi90 qS
        - X180 qCP - RO

    qS is the "swap qubit"
    qCP is the "CPhase qubit"

    mw_pulse_pars qCP:    (dict) qubit control pulse pars
    mw_pulse_pars qS:     (dict) qubit control pulse pars
    flux_pulse_pars qCP:  (dict) flux puplse pars
    flux_pulse_pars qS:   (dict) flux puplse pars
    RO_pars:              (dict) qubit RO pars, ideally a multiplexed readout
    distortion_dict=None: (dict) flux_pulse predistortion kernels

    CPhase:               (bool) if False replaces CPhase with an identity
    excitations:          (enum) [0, 1, 'both'] whether to put an excitation in
                          the swap qubit, both does the sequence both ways.
    phases                (list) phases used for the recovery pulse
    inter_swap_wait       (float) wait time in seconds between the two swaps
    verbose=False:        (bool) used for verbosity printing in the pulsar
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points

    TODO:
        - move getting the pulse dict to a single function
        - default to the single qubit swap-wait-swap sequence if no pulse pars
          for qCP
        - Add all four calibration points
    '''

    # ############ This getting pulse dict should be a single function
    mw_pulses_qCP = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(mw_pulse_pars_qCP), ' qCP')
    mw_pulses_qS = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(mw_pulse_pars_qS), ' qS')
    # This should come out of this dict in a smarter way
    CPhase_qCP = {'CPhase qCP': flux_pulse_pars_qCP}# pulse incluse single qubit phase correction
    swap_qS = {'swap qS': flux_pulse_pars_qS}
    RO_dict = {'RO': RO_pars}
    pulse_dict = {}
    pulse_dict.update(mw_pulses_qCP)
    pulse_dict.update(mw_pulses_qS)
    pulse_dict.update(CPhase_qCP)
    pulse_dict.update(swap_qS)
    pulse_dict.update(RO_dict)
    ## End of the getting pulse dict


    # Preloading should be almost instant (no preloading here)
    #preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    # renamed as the dict contains the pulse directly

    # Getting the minus flux pulses should also be in the get pulse dict
    # minus_flux_pulse_pars = deepcopy(flux_pulse_pars_qCP) not used
    pulse_dict['mCPhase qCP'] = deepcopy(pulse_dict['CPhase qCP'])
    pulse_dict['mswap qS'] = deepcopy(pulse_dict['swap qS'])
    pulse_dict['mCPhase qCP']['amplitude'] = -pulse_dict['CPhase qCP']['amplitude']
    pulse_dict['mswap qS']['amplitude'] = -pulse_dict['swap qS']['amplitude']

    # pulse_dict.update({'mFlux_pulse': minus_flux_pulse_pars}) not used
    pulse_dict.update({'dead_time_pulse':
                       {'pulse_type': 'SquarePulse',
                        'pulse_delay': flux_pulse_pars_qCP['dead_time'],
                        'channel': flux_pulse_pars_qCP['channel'],
                        'amplitude': 0,
                        'length': 0.}})

    # Pulse is used to set the starting refpoint for the compensation pulses

    seq_name = 'swap_CP_swap_2Qubits'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    recovery_swap = deepcopy(pulse_dict['swap qS'])

    #pulse to correct single-qubit phase error in the swap qubit
    pulse_dict['phase corr qS'] = deepcopy(pulse_dict['swap qS'])
    pulse_dict['phase corr qS']['square_pulse_length'] = flux_pulse_pars_qS['phase_corr_pulse_length']
    pulse_dict['phase corr qS']['amplitude'] = flux_pulse_pars_qS['phase_corr_pulse_amp']
    pulse_dict['mphase corr qS'] = deepcopy(pulse_dict['swap qS'])
    pulse_dict['mphase corr qS']['square_pulse_length'] = flux_pulse_pars_qS['phase_corr_pulse_length']
    pulse_dict['mphase corr qS']['amplitude'] = -flux_pulse_pars_qS['phase_corr_pulse_amp']

    if not CPhase:
        pulse_dict['CPhase qCP']['amplitude'] = 0
        pulse_dict['mCPhase qCP']['amplitude'] = 0
        pulse_dict['phase corr qS']['amplitude'] = 0
        pulse_dict['CPhase qCP']['phase_corr_pulse_amp'] = 0

    recovery_swap['pulse_delay'] = inter_swap_wait/2
    pulse_dict['recovery swap qS'] = recovery_swap
    pulse_dict['CPhase qCP']['pulse_delay'] = inter_swap_wait/2

    # seq has to have at least 2 elts
    for i, phase in enumerate(phases):
        if reverse_control_target:
            if (i < (len(phases)-4*cal_points)/4):
                reverse = False
                excitation = False
            elif (i < (len(phases)-4*cal_points)/2):
                reverse = False
                excitation = True
            elif (i < (len(phases)-4*cal_points)*3/4):
                reverse = True
                excitation = False
            else:
                reverse = True
                excitation = True
        else:
            reverse = False
            if excitations == 'both':
                reverse=False
                if (i < (len(phases)-4*cal_points)/2):
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


        if cal_points and (i == (len(phases)-4)):
            pulse_combinations = ['I qCP', 'I qS', 'RO']
        elif cal_points and (i == (len(phases)-3)):
            pulse_combinations = ['I qCP', 'X180 qS', 'RO']
        elif cal_points and ((i == len(phases)-2)):
            pulse_combinations = ['X180 qCP', 'I qS', 'RO']
        elif cal_points and (i == (len(phases)-1)):
            pulse_combinations = ['X180 qCP', 'X180 qS', 'RO']
        else:
            if not reverse:
                rphi90_qCP = deepcopy(pulse_dict['X90 qCP'])
                rphi90_qCP['phase'] = phase
                pulse_dict['rphi90 qCP'] = rphi90_qCP

                if excitation:
                    pulse_combinations = ['X180 qS', 'mY90 qCP', 'swap qS'] + \
                        ['CPhase qCP'] + \
                        ['recovery swap qS','phase corr qS', 'rphi90 qCP', 'X180 qS', 'RO'] + \
                        ['dead_time_pulse']+['mswap qS']*2+['mCPhase qCP']+['mphase corr qS']
                else:
                    pulse_combinations = ['I qS', 'mY90 qCP', 'swap qS'] + \
                        ['CPhase qCP'] + \
                        ['recovery swap qS','phase corr qS', 'rphi90 qCP', 'I qS', 'RO'] + \
                        ['dead_time_pulse']+['mswap qS']*2+['mCPhase qCP']+['mphase corr qS']

            else:
                rphi90_qS = deepcopy(pulse_dict['X90 qS'])
                rphi90_qS['phase'] = phase
                pulse_dict['rphi90 qS'] = rphi90_qS
                if excitation:
                    pulse_combinations = ['X180 qCP', 'mY90 qS', 'swap qS'] + \
                        ['CPhase qCP'] + \
                        ['recovery swap qS','phase corr qS', 'rphi90 qS', 'X180 qCP', 'RO'] + \
                        ['dead_time_pulse']+['mswap qS']*2+['mCPhase qCP']+['mphase corr qS']
                else:
                    pulse_combinations = ['I qCP', 'mY90 qS', 'swap qS'] + \
                        ['CPhase qCP'] + \
                        ['recovery swap qS','phase corr qS', 'rphi90 qS', 'I qCP', 'RO'] + \
                        ['dead_time_pulse']+['mswap qS']*2+['mCPhase qCP']+['mphase corr qS']


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




def chevron_with_excited_bus_2Qubits(mw_pulse_pars_qCP, mw_pulse_pars_qS,
                         flux_pulse_pars_qCP, flux_pulse_pars_qS,
                         RO_pars,
                         distortion_dict,
                         chevron_pulse_lengths=np.arange(0, 120e-9, 2e-9),
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
        - Add all four calibration points
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
    ## End of the getting pulse dict

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
        pulse_dict['swap qCP']['square_pulse_length']= chevron_pulse_length
        pulse_dict['mswap qCP']['square_pulse_length']= chevron_pulse_length
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
                pulse_combinations = ['X180 qCP', 'I qS']+\
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


def BusT1(times, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
          verbose=False, distortion_dict=None,
          upload=True, return_seq=False):
    '''

    '''
    if flux_pulse_pars is None:
        raise ValueError('Need flux parameters for the gate.')
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    minus_flux_pulse_pars['amplitude'] = -minus_flux_pulse_pars['amplitude']

    original_delay = deepcopy(RO_pars)['pulse_delay']

    seq_name = 'BusT1_seq'
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

        pulse_list = [pulses['X180'], flux_pulse_pars, flux_pulse_pars_2,
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


def distort_and_compensate(element, distortion_dict):
    """
    Distorts an element using the contenst of a distortion dictionary.
    The distortion dictionary should be formatted as follows.

    dist_dict{'ch_list': ['chx', 'chy'],
              'chx': np.array(.....),
              'chy': np.array(.....)}
    """

    t_vals, outputs_dict = element.waveforms()
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        kernelvec = distortion_dict[ch]
        outputs_dict[ch] = np.convolve(
            outputs_dict[ch], kernelvec)[:length]
        element.distorted_wfs[ch] = outputs_dict[ch][:len(t_vals)]
    return element


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





