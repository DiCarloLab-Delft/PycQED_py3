import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
import qcodes as qc
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.standard_elements import distort_and_compensate
from pycqed.utilities.general import get_required_upload_information
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    sweep_pulse_params, add_preparation_pulses, pulse_list_list_seq


from importlib import reload
reload(pulse)
from pycqed.measurement.waveform_control import pulse_library
reload(pulse_library)

import logging
log = logging.getLogger(__name__)


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


def chevron_seqs(hard_sweep_dict, soft_sweep_dict, qbc_name, qbt_name, qbr_name,
                 cz_pulse_name, operation_dict, prep_params=dict(),
                 cal_points=None, upload=True):

    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|

   '''

    seq_name = 'Chevron_sequence'

    ge_pulse_qbc = deepcopy(operation_dict['X180 ' + qbc_name])
    ge_pulse_qbc['name'] = 'chevron_pi_qbc'
    ge_pulse_qbt = deepcopy(operation_dict['X180s ' + qbt_name])
    ge_pulse_qbt['name'] = 'chevron_pi_qbt'
    for ge_pulse in [ge_pulse_qbc, ge_pulse_qbt]:
        ge_pulse['element_name'] = 'chevron_pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'chevron_flux'

    ro_pulse = deepcopy(operation_dict['RO ' + qbr_name])
    if 'pulse_length' in hard_sweep_dict:
        max_flux_length = max(hard_sweep_dict['pulse_length']['values'])
        ro_pulse['ref_pulse'] = 'chevron_pi_qbc'
        ro_pulse['pulse_delay'] = max_flux_length + \
                                  flux_pulse.get('buffer_length_start', 0) + \
                                  flux_pulse.get('buffer_length_end', 0)

    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        flux_p = deepcopy(flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        pulses = [ge_pulse_qbc, ge_pulse_qbt, flux_p, ro_pulse]
        swept_pulses = sweep_pulse_params(
            pulses, {f'chevron_flux.{k}': v['values']
                        for k, v in hard_sweep_dict.items()})
        swept_pulses_soft_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_soft_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.extend(seq)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


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


def fluxpulse_scope_sequence(
              delays, freqs, qb_name, operation_dict, cz_pulse_name,
              cal_points=None, prep_params=dict(), upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ----------------------------  |RO|
       |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->
    '''
    seq_name = 'Fluxpulse_scope_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'FPS_Pi'
    ge_pulse['element_name'] = 'FPS_Pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'FPS_Flux'
    flux_pulse['ref_pulse'] = 'FPS_Pi'
    flux_pulse['ref_point'] = 'middle'
    flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start', 0)

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPS_Ro'
    ro_pulse['ref_pulse'] = 'FPS_Pi'
    ro_pulse['ref_point'] = 'middle'
    ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - np.min(delays) + \
                              flux_pulse.get('buffer_length_end', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(pulses,
                                      {'FPS_Flux.pulse_delay': -delays})

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements()), freqs

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


def cz_bleed_through_phase_seq(phases, qb_name, CZ_pulse_name, CZ_separation,
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

