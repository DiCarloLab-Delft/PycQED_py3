import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import copy
import datetime
import os
import lmfit
from copy import deepcopy
import pygsti
from pycqed.utilities.general import temporary_value

import logging
log = logging.getLogger(__name__)

import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.detector_functions as det
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration_points import CalibrationPoints
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    one_qubit_reset
from pycqed.measurement.waveform_control import pulsar as ps
import pycqed.measurement.composite_detector_functions as cdet
import pycqed.analysis.measurement_analysis as ma
import pycqed.analysis.randomized_benchmarking_analysis as rbma
from pycqed.analysis_v3 import pipeline_analysis as pla
import pycqed.analysis_v2.readout_analysis as ra
import pycqed.analysis_v2.timedomain_analysis as tda
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.utilities.general import temporary_value
import pycqed.analysis.analysis_toolbox as a_tools

try:
    import \
        pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as uhfqc
except ModuleNotFoundError:
    log.warning('"UHFQuantumController" not imported.')

from pycqed.measurement.optimization import nelder_mead, \
    generate_new_training_set
from pygsti import construction as constr


def multiplexed_pulse(readouts, f_LO, upload=True):
    """
    Sets up a frequency-multiplexed pulse on the awg-sequencer of the UHFQC.
    Updates the qubit ro_pulse_type parameter. This needs to be reverted if
    thq qubit object is to update its readout pulse later on.

    Args:
        readouts: A list of different readouts. For each readout the list
                  contains the qubit objects that are read out in that readout.
        f_LO: The LO frequency that will be used.
        upload: Whether to update the hardware instrument settings.
        plot_filename: The file to save the plot of the multiplexed pulse PSD.
            If `None` or `True`, plot is only shown, and not saved. If `False`,
            no plot is generated.

    Returns:
        The generated pulse waveform.
    """
    if not hasattr(readouts[0], '__iter__'):
        readouts = [readouts]
    fs = 1.8e9

    readout_pulses = []
    for qubits in readouts:
        qb_pulses = {}
        maxlen = 0

        for qb in qubits:
            # qb.RO_pulse_type('Multiplexed_pulse_UHFQC')
            qb.f_RO_mod(qb.f_RO() - f_LO)
            samples = int(qb.RO_pulse_length() * fs)

            pulse = qb.RO_amp() * np.ones(samples)
            tbase = np.linspace(0, len(pulse) / fs, len(pulse), endpoint=False)

            if qb.ro_pulse_shape() == 'gaussian_filtered':
                filter_sigma = qb.ro_pulse_filter_sigma()
                nr_sigma = qb.ro_pulse_nr_sigma()
                filter_samples = int(filter_sigma * nr_sigma * fs)
                filter_sample_idxs = np.arange(filter_samples)
                filter = np.exp(
                    -0.5 * (filter_sample_idxs - filter_samples / 2) ** 2 /
                    (filter_sigma * fs) ** 2)
                filter /= filter.sum()
                pulse = np.convolve(pulse, filter, mode='full')
            elif qb.ro_pulse_shape() == 'gaussian_filtered_flip':
                pulse = pulse * (
                            np.ones(samples) - np.cos(2 * np.pi * 200e6 * tbase))

                filter_sigma = qb.ro_pulse_filter_sigma()
                nr_sigma = qb.ro_pulse_nr_sigma()
                filter_samples = int(filter_sigma * nr_sigma * fs)
                filter_sample_idxs = np.arange(filter_samples)
                filter = np.exp(
                    -0.5 * (filter_sample_idxs - filter_samples / 2) ** 2 /
                    (filter_sigma * fs) ** 2)
                filter /= filter.sum()
                pulse = np.convolve(pulse, filter, mode='full')
            elif qb.ro_pulse_shape() == 'CLEAR':
                filter_sigma = qb.ro_pulse_filter_sigma()
                nr_sigma = qb.ro_pulse_nr_sigma()
                filter_samples = int(filter_sigma * nr_sigma * fs)
                filter_sample_idxs = np.arange(filter_samples)
                filter = np.exp(
                    -0.5 * (filter_sample_idxs - filter_samples / 2) ** 2 /
                    (filter_sigma * fs) ** 2)
                filter /= filter.sum()
                pulse = uhfqc.CLEAR_shape(qb.RO_amp(), qb.RO_pulse_length(),
                                          qb.ro_CLEAR_delta_amp_segment(),
                                          qb.ro_CLEAR_segment_length(),
                                          sampling_rate=fs)

                pulse = np.convolve(pulse, filter, mode='full')
            elif qb.ro_pulse_shape() == 'square':
                pass
            else:
                raise ValueError('Unsupported pulse type for {}: {}' \
                                 .format(qb.name, qb.ro_pulse_shape()))

            tbase = np.linspace(0, len(pulse) / fs, len(pulse), endpoint=False)
            pulse = pulse * np.exp(-2j * np.pi * qb.f_RO_mod() * tbase)

            qb_pulses[qb.name] = pulse
            if pulse.size > maxlen:
                maxlen = pulse.size

        pulse = np.zeros(maxlen, dtype=np.complex)
        for p in qb_pulses.values():
            pulse += np.pad(p, (0, maxlen - p.size), mode='constant',
                            constant_values=0)
        readout_pulses.append(pulse)

    if upload:
        UHFQC = readouts[0][0].UHFQC
        if len(readout_pulses) == 1:
            UHFQC.awg_sequence_acquisition_and_pulse(
                Iwave=np.real(pulse).copy(), Qwave=np.imag(pulse).copy())
        else:
            UHFQC.awg_sequence_acquisition_and_pulse_multi_segment(readout_pulses)
        DC_LO = readouts[0][0].readout_DC_LO
        UC_LO = readouts[0][0].readout_UC_LO
        DC_LO.frequency(f_LO)
        UC_LO.frequency(f_LO)


def get_operation_dict(qubits):
    operation_dict = dict()
    for qb in qubits:
        operation_dict.update(qb.get_operation_dict())
    return operation_dict


def get_multiplexed_readout_detector_functions(qubits, nr_averages=2**10,
                                               nr_shots=4095,
                                               used_channels=None,
                                               correlations=None,
                                               add_channels=None,
                                               det_get_values_kws=None,
                                               **kw):
    uhfs = set()
    uhf_instances = {}
    max_int_len = {}
    channels = {}
    acq_classifier_params = {}
    acq_state_prob_mtxs = {}
    for qb in qubits:
        uhf = qb.instr_uhf()
        uhfs.add(uhf)
        uhf_instances[uhf] = qb.instr_uhf.get_instr()

        if uhf not in max_int_len:
            max_int_len[uhf] = 0
        max_int_len[uhf] = max(max_int_len[uhf], qb.acq_length())

        if uhf not in channels:
            channels[uhf] = []
        channels[uhf] += [qb.acq_I_channel()]
        if qb.acq_weights_type() in ['SSB', 'DSB', 'optimal_qutrit']:
            if qb.acq_Q_channel() is not None:
                channels[uhf] += [qb.acq_Q_channel()]

        if uhf not in acq_classifier_params:
            acq_classifier_params[uhf] = []
        acq_classifier_params[uhf] += [qb.acq_classifier_params()]
        if uhf not in acq_state_prob_mtxs:
            acq_state_prob_mtxs[uhf] = []
        acq_state_prob_mtxs[uhf] += [qb.acq_state_prob_mtx()]

    if det_get_values_kws is None:
        det_get_values_kws = {}
        det_get_values_kws_in = None
    else:
        det_get_values_kws_in = deepcopy(det_get_values_kws)
        for uhf in acq_state_prob_mtxs:
            det_get_values_kws_in.pop(uhf, False)
    for uhf in acq_state_prob_mtxs:
        if uhf not in det_get_values_kws:
            det_get_values_kws[uhf] = {}
        det_get_values_kws[uhf].update({
            'classifier_params': acq_classifier_params[uhf],
            'state_prob_mtx': acq_state_prob_mtxs[uhf]})
        if det_get_values_kws_in is not None:
            det_get_values_kws[uhf].update(det_get_values_kws_in)
    if add_channels is None:
        add_channels = {uhf: [] for uhf in uhfs}
    elif isinstance(add_channels, list):
        add_channels = {uhf: add_channels for uhf in uhfs}
    else:  # is a dict
        pass
    for uhf in add_channels:
        channels[uhf] += add_channels[uhf]

    if correlations is None:
        correlations = {uhf: [] for uhf in uhfs}
    elif isinstance(correlations, list):
        correlations = {uhf: correlations for uhf in uhfs}
    else:  # is a dict
        for uhf in uhfs:
            if uhf not in correlations:
                correlations[uhf] = []

    if used_channels is None:
        used_channels = {uhf: None for uhf in uhfs}
    elif isinstance(used_channels, list):
        used_channels = {uhf: used_channels for uhf in uhfs}
    else:  # is a dict
        for uhf in uhfs:
            if uhf not in used_channels:
                used_channels[uhf] = None

    AWG = None
    for qb in qubits:
        qbAWG = qb.instr_pulsar.get_instr()
        if AWG is not None and qbAWG is not AWG:
            raise Exception('Multi qubit detector can not be created with '
                            'multiple pulsar instances')
        AWG = qbAWG

    individual_detectors = {uhf: {
            'int_log_det': det.UHFQC_integration_logging_det(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                result_logging_mode='raw', **kw),
            'dig_log_det': det.UHFQC_integration_logging_det(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                result_logging_mode='digitized', **kw),
            'int_avg_det': det.UHFQC_integrated_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages, **kw),
            'int_avg_classif_det': det.UHFQC_classifier_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                get_values_function_kwargs=det_get_values_kws[uhf],
                result_logging_mode='raw', **kw),
            'dig_avg_det': det.UHFQC_integrated_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                result_logging_mode='digitized', **kw),
            'inp_avg_det': det.UHFQC_input_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, nr_averages=nr_averages,
                nr_samples=4096,
                **kw),
            'int_corr_det': det.UHFQC_correlation_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                used_channels=used_channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                correlations=correlations[uhf], **kw),
            'dig_corr_det': det.UHFQC_correlation_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                used_channels=used_channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                correlations=correlations[uhf], thresholding=True, **kw),
        } for uhf in uhfs}

    combined_detectors = {det_type: det.UHFQC_multi_detector([
        individual_detectors[uhf][det_type] for uhf in uhfs])
        for det_type in ['int_log_det', 'dig_log_det',
                         'int_avg_det', 'dig_avg_det', 'inp_avg_det',
                         'int_avg_classif_det', 'int_corr_det', 'dig_corr_det']}

    return combined_detectors


def get_multi_qubit_prep_params(prep_params_list):
    if len(prep_params_list) == 0:
        raise ValueError('prep_params_list is empty.')

    thresh_map = {}
    for prep_params in prep_params_list:
        if 'threshold_mapping' in prep_params:
            thresh_map.update(prep_params['threshold_mapping'])

    prep_params = deepcopy(prep_params_list[0])
    prep_params['threshold_mapping'] = thresh_map
    return prep_params


def get_meas_obj_value_names_map(qubits, det_type):
    if det_type == 'int_avg_classif_det':
        meas_obj_value_names_map = {
            qb.name: [vn + ' ' + qb.instr_uhf() for
                      vn in qb.int_avg_classif_det.value_names]
            for qb in qubits}
    elif det_type == 'inp_avg_det':
        meas_obj_value_names_map = {
            qb.name: [vn + ' ' + qb.instr_uhf() for
                      vn in qb.inp_avg_det.value_names]
            for qb in qubits}
    else:
        meas_obj_value_names_map = {
            qb.name: [vn + ' ' + qb.instr_uhf() for
                      vn in qb.int_avg_det.value_names]
            for qb in qubits}
    return meas_obj_value_names_map


def calculate_minimal_readout_spacing(qubits, ro_slack=10e-9, drive_pulses=0):
    """

    Args:
        qubits:
        ro_slack: minimal time needed between end of wint and next RO trigger
        drive_pulses:

    Returns:

    """
    UHFQC = None
    for qb in qubits:
        UHFQC = qb.UHFQC
        break
    drive_pulse_len = None
    max_ro_len = 0
    max_int_length = 0
    for qb in qubits:
        if drive_pulse_len is not None:
            if drive_pulse_len != qb.gauss_sigma() * qb.nr_sigma() and \
                    drive_pulses != 0:
                log.warning('Caution! Not all qubit drive pulses are the '
                            'same length. This might cause trouble in the '
                            'sequence.')
            drive_pulse_len = max(drive_pulse_len,
                                  qb.gauss_sigma() * qb.nr_sigma())
        else:
            drive_pulse_len = qb.gauss_sigma() * qb.nr_sigma()
        max_ro_len = max(max_ro_len, qb.RO_pulse_length())
        max_int_length = max(max_int_length, qb.RO_acq_integration_length())

    ro_spacing = 2 * UHFQC.qas_0_delay() / 1.8e9
    ro_spacing += max_int_length
    ro_spacing += ro_slack
    ro_spacing -= drive_pulse_len
    ro_spacing -= max_ro_len
    return ro_spacing


def measure_multiplexed_readout(qubits, liveplot=False,
                                shots=5000,
                                RO_spacing=None, preselection=True,
                                thresholds=None, thresholded=False,
                                analyse=True):
    for qb in qubits:
        MC = qb.instr_mc.get_instr()

    for qb in qubits:
        qb.prepare(drive='timedomain')

    if RO_spacing is None:
        UHFQC = qubits[0].instr_uhf.get_instr()
        RO_spacing = UHFQC.qas_0_delay() * 2 / 1.8e9
        RO_spacing += UHFQC.qas_0_integration_length() / 1.8e9
        RO_spacing += 50e-9  # for slack
        RO_spacing = np.ceil(RO_spacing * 225e6 / 3) / 225e6 * 3

    sf = awg_swf2.n_qubit_off_on(
        [qb.get_ge_pars() for qb in qubits],
        [qb.get_ro_pars() for qb in qubits],
        preselection=preselection,
        parallel_pulses=True,
        RO_spacing=RO_spacing)

    m = 2 ** (len(qubits))
    if preselection:
        m *= 2
    if thresholded:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'int_log_det']

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(m))
    MC.set_detector_function(df)
    MC.run('{}_multiplexed_ssro'.format('-'.join(
        [qb.name for qb in qubits])))

    if analyse and thresholds is not None:
        channel_map = {qb.name: qb.int_log_det.value_names[0]+' '+qb.instr_uhf() for qb in qubits}
        ra.Multiplexed_Readout_Analysis(options_dict=dict(
            n_readouts=(2 if preselection else 1) * 2 ** len(qubits),
            thresholds=thresholds,
            channel_map=channel_map,
            use_preselection=preselection
        ))


def measure_active_reset(qubits, shots=5000,
                         qutrit=False, upload=True, label=None):
    MC = qubits[0].instr_mc.get_instr()
    trig = qubits[0].instr_trigger.get_instr()

    # combine operations and preparation dictionaries
    operation_dict = get_operation_dict(qubits)
    qb_names = [qb.name for qb in qubits]
    prep_params = \
        get_multi_qubit_prep_params([qb.preparation_params() for qb in qubits])

    # sequence
    seq, swp = mqs.n_qubit_reset(qb_names, operation_dict, prep_params,
                                upload=False, states='gef' if qutrit else 'ge')
    # create sweep points
    sp = SweepPoints('reset_reps', swp, '', 'Nr. Reset Repetitions')

    df = get_multiplexed_readout_detector_functions(qubits,
                                                    nr_shots=shots)['int_log_det']

    for qb in qubits:
        qb.prepare(drive='timedomain')

    MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq, upload=upload))
    MC.set_sweep_points(swp)
    MC.set_detector_function(df)
    if label is None:
        label = 'active_reset_{}_x{}_{}'.format('ef' if qutrit else 'e',
                                                prep_params['reset_reps'],
                                                ','.join(qb_names))
    exp_metadata = {'preparation_params': prep_params,
                    'sweep_points': sp,
                    'shots': shots}
    temp_values = [(qb.acq_shots, shots) for qb in qubits]
    temp_values += [(MC.soft_avg, 1)]
    with temporary_value(*temp_values):
        MC.run(name=label,  exp_metadata=exp_metadata)

def measure_arbitrary_sequence(qubits, sequence=None, sequence_function=None,
                               sequence_args=dict(), drive='timedomain', label=None,
                               detector_function=None, df_kwargs=dict(),
                               sweep_function=awg_swf.SegmentHardSweep,
                               sweep_points=None, temporary_values=(),
                               exp_metadata=None, upload=True,
                               analyze=True):
    """
    Measures arbitrary sequence provided in input.
    Args:
        qubits (list): qubits on which the sequence is performed
        sequence (Sequence): sequence to measure. Optionally,
            the path of the sequence can be provided (eg. sqs.active_reset) as
            sequence_function.
        sequence_function (callable): sequence function which creates a sequences using
            sequence_args. Should return (sequence, sweep_points).
        sequence_args (dict): arguments used to build the sequence
        drive (string): drive method. Defaults to timedomain
        label (string): measurement label. Defaults to sequence.name.
        detector_function (string): detector function string. eg.
            'int_avg_detector'. Built using multi_uhf get_multiplexed_readout_detector_functions
        df_kwargs (dict): detector function kwargs
        sweep_function (callable): sweep function. Defaults to segment hard sweep.
        sweep_points (list or array): list of sweep points. Required only if
            argument sequence is used.
        temporary_values (tuple): list of tuple pairs with qcode param and its
            temporary value. eg [(qb1.acq_shots, 10000),(MC.soft_avg, 1)]
        exp_metadata:
        upload:
        analyze:

    Returns:

    """
    if sequence is None and sequence_function is None:
        raise ValueError("Either Sequence or sequence name must be given.")

    MC = qubits[0].instr_mc.get_instr()

    # combine preparation dictionaries
    qb_names = [qb.name for qb in qubits]
    prep_params = \
        get_multi_qubit_prep_params([qb.preparation_params() for qb in qubits])

    # sequence
    if sequence is not None:
        if sweep_points is None:
            raise ValueError("Sweep points must be specified if sequence object"
                             "is given")
    else:
        sequence, sweep_points = sequence_function(**sequence_args)

    # create sweep points
    df = get_multiplexed_readout_detector_functions(qubits, **df_kwargs)[
        detector_function]

    for qb in qubits:
        qb.prepare(drive=drive)

    MC.set_sweep_function(sweep_function(sequence=sequence, upload=upload))
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(df)

    if label is None:
        label = f'{sequence.name}_{",".join(qb_names)}'

    if exp_metadata is None:
        exp_metadata = {}

    exp_metadata.update({'preparation_params': prep_params,
                    # 'sweep_points': ,
                    })
    if len(temporary_values) > 0:
        with temporary_value(*temporary_values):
            MC.run(name=label, exp_metadata=exp_metadata)
    else:
        MC.run(name=label, exp_metadata=exp_metadata)

    if analyze:
        return ma.MeasurementAnalysis()

def measure_parity_correction(qb0, qb1, qb2, feedback_delay, f_LO,
                              CZ_pulses, nreps=1, parity_op='ZZ',
                              upload=True, MC=None, prep_sequence=None,
                              nr_dd_pulses=0, dd_scheme=None,
                              nr_shots=5000, nr_parity_measurements=1,
                              tomography_basis=(
                                      'I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                              reset=True, preselection=False, ro_spacing=1e-6,
                              skip_n_initial_parity_checks=0, skip_elem='RO',
                              add_channels=None):
    """
    Important things to check when running the experiment:
        Is the readout separation commensurate with 225 MHz?

    Args:
        parity_op: 'ZZ', 'XX', 'XX,ZZ' or 'ZZ,XX' specifies the type of parity
                   measurement
    """
    exp_metadata = {'feedback_delay': feedback_delay,
                    'CZ_pulses': CZ_pulses,
                    'nr_parity_measurements': nr_parity_measurements,
                    'ro_spacing': ro_spacing,
                    'nr_dd_pulses': nr_dd_pulses,
                    'dd_scheme': dd_scheme,
                    'parity_op': parity_op,
                    'prep_sequence': prep_sequence,
                    'skip_n_initial_parity_checks':
                        skip_n_initial_parity_checks,
                    'skip_elem': skip_elem}

    if reset == 'simple':
        nr_parity_measurements = 1

    nr_ancilla_readouts = nr_parity_measurements
    if skip_elem == 'RO':
        nr_ancilla_readouts -= skip_n_initial_parity_checks
    if preselection:
        if prep_sequence == 'mixed':
            multiplexed_pulse([(qb0, qb1, qb2), (qb0, qb2)] +
                              [(qb1,)] * nr_ancilla_readouts +
                              [(qb0, qb1, qb2)], f_LO)
        else:
            multiplexed_pulse([(qb0, qb1, qb2)] +
                              [(qb1,)] * nr_ancilla_readouts +
                              [(qb0, qb1, qb2)], f_LO)
    else:
        if prep_sequence == 'mixed':
            multiplexed_pulse([(qb0, qb2)] +
                              [(qb1,)] * nr_ancilla_readouts +
                              [(qb0, qb1, qb2)], f_LO)
        else:
            multiplexed_pulse([(qb1,)] * nr_ancilla_readouts +
                              [(qb0, qb1, qb2)], f_LO)

    qubits = [qb0, qb1, qb2]
    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    sf = awg_swf2.parity_correction(
        qb0.name, qb1.name, qb2.name,
        operation_dict=get_operation_dict(qubits), CZ_pulses=CZ_pulses,
        feedback_delay=feedback_delay, prep_sequence=prep_sequence,
        reset=reset, nr_parity_measurements=nr_parity_measurements,
        parity_op=parity_op,
        tomography_basis=tomography_basis,
        preselection=preselection,
        ro_spacing=ro_spacing,
        dd_scheme=dd_scheme,
        nr_dd_pulses=nr_dd_pulses,
        skip_n_initial_parity_checks=skip_n_initial_parity_checks,
        skip_elem=skip_elem,
        upload=upload, verbose=False)

    nr_readouts = 1 + nr_ancilla_readouts + (1 if preselection else 0) \
                  + (1 if prep_sequence == 'mixed' else 0)
    nr_readouts *= len(tomography_basis) ** 2

    nr_shots *= nr_readouts
    df = get_multiplexed_readout_detector_functions(
        qubits, nr_shots=nr_shots, add_channels=add_channels)['int_log_det']

    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(nr_shots))
    MC.set_sweep_function_2D(swf.Delayed_None_Sweep(mode='set_delay', delay=5))
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)

    if skip_n_initial_parity_checks == 0:
        skip_str = ''
    else:
        skip_str = 'skip' + str(skip_n_initial_parity_checks)
        skip_str += skip_elem.replace(' ', '')

    MC.run_2D(name='two_qubit_parity_{}_x{}{}{}{}-{}'.format(
        parity_op, nr_parity_measurements, skip_str,
        prep_sequence if prep_sequence == 'mixed' else '',
        '' if reset else '_noreset', '_'.join([qb.name for qb in qubits])),
        exp_metadata=exp_metadata)

def measure_parity_single_round(ancilla_qubit, data_qubits, CZ_map, 
                                preps=None, upload=True, prep_params=None, 
                                cal_points=None, analyze=True,
                                exp_metadata=None, label=None, 
                                detector='int_avg_det'):
    """

    :param ancilla_qubit:
    :param data_qubits:
    :param CZ_map: example:
        {'CZ qb1 qb2': ['Y90 qb1', 'CX qb1 qb2', 'mY90 qb1'],
         'CZ qb3 qb4': ['CZ qb4 qb3']}
    :param preps:
    :param upload:
    :param prep_params:
    :param cal_points:
    :param analyze:
    :param exp_metadata:
    :param label:
    :param detector:
    :return:
    """

    qubits = [ancilla_qubit] + data_qubits
    qb_names = [qb.name for qb in qubits]
    for qb in qubits:
        qb.prepare(drive='timedomain')
    
    if label is None:
        label = 'Parity-1-round_'+'-'.join([qb.name for qb in qubits])
    
    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in qubits])

    if cal_points is None:
        cal_points = CalibrationPoints.multi_qubit(qb_names, 'ge')

    if preps is None:
        preps = [''.join(s) 
            for s in itertools.product(*len(data_qubits)*['ge'])]

    MC = ancilla_qubit.instr_mc.get_instr()

    seq, sweep_points = mqs.parity_single_round_seq(
            ancilla_qubit.name, [qb.name for qb in data_qubits], CZ_map,
            preps=preps, cal_points=cal_points, prep_params=prep_params,
            operation_dict=get_operation_dict(qubits), upload=False)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Preparation'))
    MC.set_sweep_points(sweep_points)

    MC.set_detector_function(
        get_multiplexed_readout_detector_functions(
            qubits, 
            nr_averages=ancilla_qubit.acq_averages(), 
            nr_shots=ancilla_qubit.acq_shots(),
        )[detector])
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update(
        {'sweep_name': 'Preparation',
         'preparations': preps,
         'cal_points': repr(cal_points),
         'rotate': True,
         'cal_states_rotations':
             {qbn: {'g': 0, 'e': 1} for qbn in qb_names},
         'data_to_fit': {qbn: 'pe' for qbn in qb_names},
         'preparation_params': prep_params,
         'hard_sweep_params': {'preps': {'values': np.arange(0, len(preps)),
                                         'unit': ''}}
        })

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        channel_map = {
            qb.name: qb.int_log_det.value_names[0] + ' ' + qb.instr_uhf() for qb in
            qubits}
        tda.MultiQubit_TimeDomain_Analysis(qb_names=qb_names, options_dict=dict(
                channel_map=channel_map
            ))


def measure_parity_single_round_phases(ancilla_qubit, data_qubits, CZ_map,
                                       phases = np.linspace(0,2*np.pi,7),
                                       prep_anc='g', upload=True,
                                       prep_params=None,
                                       cal_points=None, analyze=True,
                                       exp_metadata=None, label=None,
                                       detector='int_avg_det'):
    """

    :param ancilla_qubit:
    :param data_qubits:
    :param CZ_map: example:
        {'CZ qb1 qb2': ['Y90 qb1', 'CX qb1 qb2', 'mY90 qb1'],
         'CZ qb3 qb4': ['CZ qb4 qb3']}
    :param preps:
    :param upload:
    :param prep_params:
    :param cal_points:
    :param analyze:
    :param exp_metadata:
    :param label:
    :param detector:
    :return:
    """

    qubits = [ancilla_qubit] + data_qubits
    qb_names = [qb.name for qb in qubits]
    for qb in qubits:
        qb.prepare(drive='timedomain')

    if label is None:
        label = 'Parity-1-round_phases_' + '-'.join([qb.name for qb in qubits])

    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in qubits])

    if cal_points is None:
        cal_points = CalibrationPoints.multi_qubit(qb_names, 'ge')

    MC = ancilla_qubit.instr_mc.get_instr()

    seq, sweep_points = mqs.parity_single_round__phases_seq(
        ancilla_qubit.name, [qb.name for qb in data_qubits], CZ_map,
        phases=phases,
        prep_anc=prep_anc, cal_points=cal_points, prep_params=prep_params,
        operation_dict=get_operation_dict(qubits), upload=False)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload, parameter_name='Preparation'))
    MC.set_sweep_points(sweep_points)

    MC.set_detector_function(
        get_multiplexed_readout_detector_functions(
            qubits,
            nr_averages=ancilla_qubit.acq_averages(),
            nr_shots=ancilla_qubit.acq_shots(),
        )[detector])
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update(
        {'sweep_name': 'Phases',
         'preparations': phases,
         'cal_points': repr(cal_points),
         'rotate': True,
         'cal_states_rotations':
             {qbn: {'g': 0, 'e': 1} for qbn in qb_names},
         'data_to_fit': {qbn: 'pe' for qbn in qb_names},
         'preparation_params': prep_params,
         'hard_sweep_params': {'phases': {'values': phases,
                                         'unit': 'rad'}}
         })

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        channel_map = {
            qb.name: qb.int_log_det.value_names[0] + ' ' + qb.instr_uhf() for qb in
            qubits}
        tda.MultiQubit_TimeDomain_Analysis(qb_names=qb_names, options_dict=dict(
            channel_map=channel_map
        ))

def measure_tomography(qubits, prep_sequence, state_name,
                       rots_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                       use_cal_points=True,
                       preselection=True,
                       rho_target=None,
                       shots=None,
                       ro_spacing=1e-6,
                       ro_slack=10e-9,
                       thresholded=False,
                       liveplot=True,
                       nreps=1, run=True,
                       upload=True):
    exp_metadata = {}

    for qb in qubits:
        MC = qb.instr_mc.get_instr()

    for qb in qubits:
        qb.prepare(drive='timedomain')

    if ro_spacing is None:
        ro_spacing = calculate_minimal_readout_spacing(qubits, ro_slack,
                                                       drive_pulses=1)

    qubit_names = [qb.name for qb in qubits]
    if preselection:
        label = '{}_tomography_ssro_preselection_{}'.format(state_name, '-'.join(
            [qb.name for qb in qubits]))
    else:
        label = '{}_tomography_ssro_{}'.format(state_name, '-'.join(
            [qb.name for qb in qubits]))
    seq = sequence.Sequence(label)

    seq_tomo, seg_list_tomo = mqs.n_qubit_tomo_seq(qubit_names,
                                                   get_operation_dict(qubits),
                                                   prep_sequence=prep_sequence,
                                                   rots_basis=rots_basis,
                                                   return_seq=True,
                                                   upload=False,
                                                   preselection=preselection,
                                                   ro_spacing=ro_spacing)
    seg_list = seg_list_tomo

    if use_cal_points:
        seq_cal, seg_list_cal = mqs.n_qubit_ref_all_seq(qubit_names,
                                                        get_operation_dict(qubits),
                                                        return_seq=True,
                                                        upload=False,
                                                        preselection=preselection,
                                                        ro_spacing=ro_spacing)
        # seq += seq_cal
        seg_list += seg_list_cal
    seq = sequence.Sequence(label)
    for seg in seg_list:
        seq.add(seg)

    n_segments = len(seg_list)
    if preselection:
        n_segments *= 2

    # from this point on number of segments is fixed
    sf = awg_swf2.n_qubit_seq_sweep(seq_len=n_segments)
    shots *= n_segments
    if shots > 1048576:
        shots = 1048576 - 1048576 % n_segments
    # if shots is None:
    #     shots = 4094 - 4094 % n_segments
    # # shots = 600000

    if thresholded:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'int_log_det']

    # make a channel map
    # fixme - channels and qubits are not always in the same order
    # getting a channel map should be a nice function, but where ?
    channel_map = {}
    for qb, channel_name in zip(qubits, df.value_names):
        channel_map[qb.name] = channel_name

    # todo Calibration point description code should be a reusable function
    #   but where?
    if use_cal_points:
        # calibration definition for all combinations
        cal_defs = []
        for i, name in enumerate(itertools.product("ge", repeat=len(qubits))):
            name = ''.join(name)  # tuple to string
            cal_defs.append({})
            for qb in qubits:
                if preselection:
                    cal_defs[i][channel_map[qb.name]] = \
                        [2 * len(seg_list) + 2 * i + 1]
                else:
                    cal_defs[i][channel_map[qb.name]] = \
                        [len(seg_list) + i]
    else:
        cal_defs = None

    exp_metadata["n_segments"] = n_segments
    exp_metadata["rots_basis"] = rots_basis
    if rho_target is not None:
        exp_metadata["rho_target"] = rho_target
    exp_metadata["cal_points"] = cal_defs
    exp_metadata["channel_map"] = channel_map
    exp_metadata["use_preselection"] = preselection

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)
    if run:
        MC.run_2D(label, exp_metadata=exp_metadata)

    return 'Tomo'


def measure_two_qubit_randomized_benchmarking(
        qb1, qb2, cliffords, nr_seeds, cz_pulse_name,
        character_rb=False, net_clifford=0,
        clifford_decomposition_name='HZ', interleaved_gate=None,
        n_cal_points_per_state=2, cal_states=tuple(),
        label=None, prep_params=None, upload=True, analyze_RB=True,
        classified=True, correlated=False, thresholded=True, averaged=True):

    qb1n = qb1.name
    qb2n = qb2.name
    qubits = [qb1, qb2]

    if label is None:
        if interleaved_gate is None:
            label = 'RB_{}_{}_seeds_{}_cliffords_{}{}'.format(
                clifford_decomposition_name, nr_seeds, cliffords[-1],
                qb1n, qb2n)
        else:
            label = 'IRB_{}_{}_{}_seeds_{}_cliffords_{}{}'.format(
                interleaved_gate, clifford_decomposition_name, nr_seeds,
                cliffords[-1], qb1n, qb2n)

    for qb in qubits:
        MC = qb.instr_mc.get_instr()
        qb.prepare(drive='timedomain')

    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in [qb1, qb2]])

    cal_states = CalibrationPoints.guess_cal_states(cal_states)
    cp = CalibrationPoints.multi_qubit([qb1n, qb2n], cal_states,
                                       n_per_state=n_cal_points_per_state)

    operation_dict = get_operation_dict(qubits)
    sequences, hard_sweep_points, soft_sweep_points = \
        mqs.two_qubit_randomized_benchmarking_seqs(
            qb1n=qb1n, qb2n=qb2n, operation_dict=operation_dict,
            cliffords=cliffords, nr_seeds=np.arange(nr_seeds),
            max_clifford_idx=24**2 if character_rb else 11520,
            cz_pulse_name=cz_pulse_name, net_clifford=net_clifford,
            clifford_decomposition_name=clifford_decomposition_name,
            interleaved_gate=interleaved_gate, upload=False,
            cal_points=cp, prep_params=prep_params)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=sequences[0], upload=upload,
        parameter_name='Nr. Cliffords', unit='')
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points if classified else
                        hard_sweep_points*max(qb.acq_shots() for qb in qubits))

    MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
        hard_sweep_func, sequences, 'Nr. Seeds', ''))
    MC.set_sweep_points_2D(soft_sweep_points)
    det_get_values_kws = {'classified': classified,
                          'correlated': correlated,
                          'thresholded': thresholded,
                          'averaged': averaged}
    if classified:
        det_type = 'int_avg_classif_det'
        nr_shots = max(qb.acq_averages() for qb in qubits)
    else:
        det_type = 'int_log_det'
        nr_shots = max(qb.acq_shots() for qb in qubits)
    det_func = get_multiplexed_readout_detector_functions(
        qubits, nr_averages=max(qb.acq_averages() for qb in qubits),
        nr_shots=nr_shots, det_get_values_kws=det_get_values_kws)[det_type]
    MC.set_detector_function(det_func)

    # create sweep points
    sp = SweepPoints('nr_seeds', np.arange(nr_seeds), '', 'Nr. Seeds')
    sp.add_sweep_dimension()
    sp.add_sweep_parameter('cliffords', cliffords, '',
                           'Number of applied Cliffords, $m$')

    qb_names = [qb1n, qb2n] + (['corr'] if (classified and correlated) else [])
    # create analysis pipeline object
    meas_obj_value_names_map = get_meas_obj_value_names_map(qubits, det_type)
    if classified and correlated:
        # FIXME: do the proper thing (SEPT)
        meas_obj_value_names_map['corr'] = ['correlation ' + qb1.instr_uhf()]
    pp = ProcessingPipeline()
    for i, mobjn in enumerate(qb_names):
        num_avg_bins = [len(cliffords)]*len(meas_obj_value_names_map[mobjn])
        pp.add_node(
            'get_std_deviation', keys_in=meas_obj_value_names_map[mobjn],
            num_bins=num_avg_bins)
        pp.add_node(
            'average', keys_in=meas_obj_value_names_map[mobjn],
            num_bins=num_avg_bins)
        pp.add_node(
            'SingleQubitRBAnalysis', keys_in='previous',
            std_keys=[k+' std' for k in meas_obj_value_names_map[mobjn]],
            meas_obj_name=mobjn, do_plotting=(i == len(qb_names)-1))
    # create experimental metadata
    exp_metadata = {'preparation_params': prep_params,
                    'cal_points': repr(cp),
                    'sweep_points': sp,
                    'meas_obj_sweep_points_map':
                       {qbn: ['nr_seeds', 'cliffords'] for qbn in qb_names},
                    'meas_obj_value_names_map': meas_obj_value_names_map,
                    'processing_pipe': pp}
    MC.run_2D(name=label, exp_metadata=exp_metadata)

    if analyze_RB:
        pla.PipelineDataAnalysis()


def measure_n_qubit_simultaneous_randomized_benchmarking(
        qubits, f_LO,
        nr_cliffords=None, nr_seeds=50,
        gate_decomp='HZ', interleaved_gate=None,
        cal_points=False, nr_averages=None,
        thresholded=True,
        experiment_channels=None,
        soft_avgs=1, analyze_RB=True,
        MC=None, UHFQC=None, pulsar=None,
        label=None, verbose=False, run=True):
    '''
    Performs a simultaneous randomized benchmarking experiment on n qubits.
    type(nr_cliffords) == array
    type(nr_seeds) == int

    Args:
        qubits (list): list of qubit objects to perfomr RB on
        f_LO (float): readout LO frequency
        nr_cliffords (numpy.ndarray): numpy.arange(max_nr_cliffords), where
            max_nr_cliffords is the number of Cliffords in the longest seqeunce
            in the RB experiment
        nr_seeds (int): the number of times to repeat each Clifford sequence of
            length nr_cliffords[i]
        gate_decomposition (str): 'HZ' or 'XY'
        interleaved_gate (str): used for regular single qubit Clifford IRB
            string referring to one of the gates in the single qubit
            Clifford group
        thresholded (bool): whether to use the thresholding feature
            of the UHFQC
        experiment_channels (list or tuple): all the qb UHFQC RO channels used
            in the experiment. Not always just the RO channels for the qubits
            passed in to this function. The user might be running an n qubit
            experiment but is now only measuring a subset of them. This function
            should not use the channels for the unused qubits as correlation
            channels because this will change the settings of that channel.
        soft_avgs (int): number of soft averages to use
        MC: MeasurementControl object
        UHFQC: UHFQC object
        pulsar: pulsar object or AWG object
        label (str): measurement label
        verbose (bool): print runtime info
    '''

    if nr_cliffords is None:
        raise ValueError("Unspecified nr_cliffords.")
    if UHFQC is None:
        UHFQC = qubits[0].UHFQC
        log.warning("Unspecified UHFQC instrument. Using qubits[0].UHFQC.")
    if pulsar is None:
        pulsar = qubits[0].AWG
        log.warning("Unspecified pulsar instrument. Using qubits[0].AWG.")
    if MC is None:
        MC = qubits[0].MC
        log.warning("Unspecified MC object. Using qubits[0].MC.")
    if experiment_channels is None:
        experiment_channels = []
        for qb in qubits:
            experiment_channels += [qb.RO_acq_weight_function_I()]
        log.warning('experiment_channels is None. Using only the channels '
                    'in the qubits RO_acq_weight_function_I parameters.')
    if label is None:
        label = 'SRB_{}_{}_seeds_{}_cliffords_qubits{}'.format(
            gate_decomp, nr_seeds, nr_cliffords[-1] if
            hasattr(nr_cliffords, '__iter__') else nr_cliffords,
            ''.join([qb.name[-1] for qb in qubits]))

    key = 'int'
    if thresholded:
        key = 'dig'
        log.warning('Make sure you have set them!.')
        label += '_thresh'

    if nr_averages is None:
        nr_averages = max(qb.RO_acq_averages() for qb in qubits)
    operation_dict = get_operation_dict(qubits)
    qubit_names_list = [qb.name for qb in qubits]
    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)
    multiplexed_pulse(qubits, f_LO, upload=True)

    if len(qubits) == 2:
        if not hasattr(nr_cliffords, '__iter__'):
            raise ValueError('For a two qubit experiment, nr_cliffords must '
                             'be an array of sequence lengths.')

        correlations = [(qubits[0].RO_acq_weight_function_I(),
                         qubits[1].RO_acq_weight_function_I())]
        det_func = get_multiplexed_readout_detector_functions(
            qubits, nr_averages=nr_averages, used_channels=experiment_channels,
            correlations=correlations)[key + '_corr_det']
        hard_sweep_points = np.arange(nr_seeds)
        hard_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_fixed_length(
                qubit_names_list=qubit_names_list,
                operation_dict=operation_dict,
                nr_cliffords_value=nr_cliffords[0],
                nr_seeds_array=np.arange(nr_seeds),
                # clifford_sequence_list=clifford_sequence_list,
                upload=False,
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate,
                verbose=verbose, cal_points=cal_points)
        soft_sweep_points = nr_cliffords
        soft_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_sequence_lengths(
                n_qubit_RB_sweepfunction=hard_sweep_func)

    else:
        nr_shots = nr_averages * nr_seeds
        det_func = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=nr_shots)[key + '_log_det']

        hard_sweep_points = np.tile(np.arange(nr_seeds), nr_averages)
        # hard_sweep_points = np.arange(nr_shots)
        hard_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_fixed_length(
                qubit_names_list=qubit_names_list,
                operation_dict=operation_dict,
                nr_cliffords_value=nr_cliffords[0],
                nr_seeds_array=np.arange(nr_seeds),
                # clifford_sequence_list=clifford_sequence_list,
                upload=False,
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate,
                verbose=verbose, cal_points=cal_points)
        soft_sweep_points = nr_cliffords
        soft_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_sequence_lengths(
                n_qubit_RB_sweepfunction=hard_sweep_func)

    if cal_points:
        step = np.abs(hard_sweep_points[-1] - hard_sweep_points[-2])
        hard_sweep_points_to_use = np.concatenate(
            [hard_sweep_points,
             [hard_sweep_points[-1] + step, hard_sweep_points[-1] + 2 * step]])
    else:
        hard_sweep_points_to_use = hard_sweep_points

    MC.soft_avg(soft_avgs)
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points_to_use)

    MC.set_sweep_function_2D(soft_sweep_func)
    MC.set_sweep_points_2D(soft_sweep_points)

    MC.set_detector_function(det_func)
    if run:
        MC.run_2D(label)

    if len(qubits) == 2:
        ma.MeasurementAnalysis(label=label, TwoD=True, close_file=True)

        if analyze_RB:
            rbma.Simultaneous_RB_Analysis(
                qb_names=[qb.name for qb in qubits],
                use_latest_data=True,
                gate_decomp=gate_decomp,
                add_correction=True)

    return MC


def cphase_gate_tuneup_predictive(qbc, qbt, qbr, initial_values: list,
                                  std_deviations: list = [20e-9, 0.02],
                                  phases=None, MC=None,
                                  estimator='GRNN_neupy',
                                  hyper_parameter_dict: dict = None,
                                  sampling_numbers: list = [70, 30],
                                  max_measurements=2,
                                  tol=[0.016, 0.05],
                                  timestamps: list = None,
                                  update=False, full_output=True,
                                  fine_tune=True, fine_tune_minmax=None):
    '''
    Args:
        qb_control (QuDev_Transmon): control qubit (with flux pulses)
        qb_target (QuDev_Transmon): target qubit
        phases (numpy array): phases used in the Ramsey measurement
        timestamps (list): measurement history. Enables collecting
                           datapoints from existing measurements and add them
                           to the training set. If there are existing timestamps,
                           cphases and population losses will be extracted from
                           all timestamps in the list. It will be optimized for
                           the combined data then and a cphase measurement will
                           be run with the optimal parameters. Possibly more data
                           will be taken after the first optimization round.
    Returns:
        pulse_length_best_value, pulse_amplitude_best_value

    '''
    ############## CHECKING INPUT #######################
    if not update:
        log.warning("Does not automatically update the CZ pulse length "
                    "and amplitude. "
                    "Set update=True if you want this!")
    if not (isinstance(sampling_numbers, list) or
            isinstance(sampling_numbers, np.ndarray)):
        sampling_numbers = [sampling_numbers]
    if max_measurements != len(sampling_numbers):
        log.warning('Did not provide sampling number for each iteration '
                    'step! Additional iterations will be carried out with the'
                    'last value in sampling numbers ')
    if len(initial_values) != 2:
        logging.error('Incorrect number of input mean values for Gaussian '
                      'sampling provided!')
    if len(std_deviations) != 2:
        logging.error('Incorrect number of standard deviations for Gaussian '
                      'sampling provided!')
    if hyper_parameter_dict is None:
        log.warning('\n No hyperparameters passed to predictive mixer '
                    'calibration routine. Default values for the estimator'
                    'will be used!\n')

        hyper_parameter_dict = {'cv_n_fold': 10,
                                'std_scaling': [0.4, 0.4]}

    if phases is None:
        phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    phases = np.concatenate((phases, phases))

    if MC is None:
        MC = qbc.MC

    if not isinstance(timestamps, list):
        if timestamps is None:
            timestamps = []
        else:
            timestamps = [timestamps]
    timestamps_iter = copy.deepcopy(timestamps)
    target_value_names = [r"$|\phi_c/\pi - 1| [a.u]$", 'Population Loss [%]']
    std_factor = 0.2

    ################## START ROUTINE ######################

    pulse_length_best = initial_values[0]
    pulse_amplitude_best = initial_values[1]
    std_length = std_deviations[0]
    std_amp = std_deviations[1]
    iteration = 0

    cphase_testing_agent = Averaged_Cphase_Measurement(qbc, qbt, qbr, 32, MC,
                                                       n_average=5, tol=tol)

    while not cphase_testing_agent.converged:
        training_grid = None
        target_values = None

        for i, t in enumerate(timestamps_iter):

            flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis_Predictive(
                timestamp=t,
                label='CPhase_measurement_{}_{}'.format(qbc.name, qbt.name),
                qb_name=qbc.name, cal_points=False, plot=False,
                save_plot=False,
                reference_measurements=True, only_cos_fits=True)

            cphases = flux_pulse_ma.cphases
            population_losses = flux_pulse_ma.population_losses

            target_phases = np.abs(np.abs(cphases / np.pi) - 1.)
            target_pops = np.abs(population_losses)

            new_train_values = np.array([flux_pulse_ma.sweep_points_2D[0][::2],
                                         flux_pulse_ma.sweep_points_2D[1][::2]]).T
            new_target_values = np.array([target_phases, target_pops]).T
            training_grid, target_values = generate_new_training_set(
                new_train_values,
                new_target_values,
                training_grid=training_grid,
                target_values=target_values)

            if iteration == 0:
                log.info('Added {} training samples from timestamp {}!' \
                      .format(np.shape(new_train_values)[0], t))

        data_size = 0 if training_grid is None else np.shape(training_grid)[0]

        # if not (iteration == 0 and timestamps_iter):
        log.info('\n{} samples before Iteration {}'.format(data_size,
                                                        iteration))
        if iteration >= len(sampling_numbers):
            sampling_number = sampling_numbers[-1]
        else:
            sampling_number = sampling_numbers[iteration]
        if iteration > 0:
            std_length *= std_factor  # rescale std deviations for next round
            std_amp *= std_factor

        new_flux_lengths = np.random.normal(pulse_length_best,
                                            std_length,
                                            sampling_number)
        new_flux_lengths = np.abs(new_flux_lengths)
        new_flux_amps = np.random.normal(pulse_amplitude_best,
                                         std_amp,
                                         sampling_number)
        log.info('measuring {} samples in iteration {} \n'. \
              format(sampling_number, iteration))

        cphases, population_losses, flux_pulse_ma = \
            measure_cphase(qbc, qbt, qbr,
                           new_flux_lengths, new_flux_amps,
                           phases=phases,
                           plot=False,
                           MC=MC)

        target_phases = np.abs(np.abs(cphases / np.pi) - 1.)
        target_pops = np.abs(population_losses)
        new_train_values = np.array([flux_pulse_ma.sweep_points_2D[0][::2],
                                     flux_pulse_ma.sweep_points_2D[1][::2]]).T
        new_target_values = np.array([target_phases, target_pops]).T

        training_grid, target_values = generate_new_training_set(
            new_train_values,
            new_target_values,
            training_grid=training_grid,
            target_values=target_values)
        new_timestamp = flux_pulse_ma.timestamp_string

        # train and test
        target_norm = np.sqrt(target_values[:, 0] ** 2 + target_values[:, 1] ** 2)
        min_ind = np.argmin(target_norm)
        x_init = [training_grid[min_ind, 0], training_grid[min_ind, 1]]
        a_pred = ma.OptimizationAnalysis_Predictive2D(training_grid,
                                                      target_values,
                                                      flux_pulse_ma,
                                                      x_init=x_init,
                                                      estimator=estimator,
                                                      hyper_parameter_dict=hyper_parameter_dict,
                                                      target_value_names=target_value_names)
        pulse_length_best = a_pred.optimization_result[0]
        pulse_amplitude_best = a_pred.optimization_result[1]
        cphase_testing_agent.lengths_opt.append(pulse_length_best)
        cphase_testing_agent.amps_opt.append(pulse_amplitude_best)

        # Get cphase with optimized values
        cphase_opt, population_loss_opt = cphase_testing_agent. \
            yield_new_measurement()

        if fine_tune:
            log.info('optimized flux parameters good enough for finetuning.\n'
                  'Finetuning amplitude with 6 values at fixed flux length!')
            if fine_tune_minmax is None:
                lower_amp = pulse_amplitude_best - std_amp
                higher_amp = pulse_amplitude_best + std_amp
            else:
                lower_amp = pulse_amplitude_best - fine_tune_minmax
                higher_amp = pulse_amplitude_best + fine_tune_minmax

            finetune_amps = np.linspace(lower_amp, higher_amp, 6)
            pulse_amplitude_best = cphase_finetune_parameters(
                qbc, qbt, qbr,
                pulse_length_best,
                finetune_amps, phases, MC)
            cphase_testing_agent.lengths_opt.append(pulse_length_best)
            cphase_testing_agent.amps_opt.append(pulse_amplitude_best)
            cphase_testing_agent.yield_new_measurement()

        # check success of iteration step
        if cphase_testing_agent.converged:
            log.info('Cphase optimization converged in iteration {}.'. \
                  format(iteration))

        elif iteration + 1 >= max_measurements:
            cphase_testing_agent.converged = True
            log.warning('\n maximum iterations exceeded without hitting'
                        ' specified tolerance levels for optimization!\n')
        else:
            log.info('Iteration {} finished. Not converged with cphase {}*pi and '
                  'population recovery {} %' \
                  .format(iteration, cphase_testing_agent.cphases[-1],
                          np.abs(1. - cphase_testing_agent.pop_losses[-1]) * 100))

            log.info('Running Iteration {} of {} ...'.format(iteration + 1,
                                                          max_measurements))

        if len(cphase_testing_agent.cphases) >= 2:
            cphases1 = cphase_testing_agent.cphases[-1]
            cphases2 = cphase_testing_agent.cphases[-2]
            if cphases1 > cphases2:
                std_factor = 1.5

        if new_timestamp is not None:
            timestamps_iter.append(new_timestamp)
        iteration += 1

    cphase_opt = cphase_testing_agent.cphases[-1]
    population_recovery_opt = np.abs(
        1. - cphase_testing_agent.pop_losses[-1]) * 100
    pulse_length_best = cphase_testing_agent.lengths_opt[-1]
    pulse_amplitude_best = cphase_testing_agent.amps_opt[-1]
    std_cphase = cphase_testing_agent.cphase_std

    log.info('CPhase optimization finished with optimal values: \n',
          'Controlled Phase QBc={} Qb Target={}: '.format(qbc.name, qbt.name),
          cphase_opt, r" ($ \pm $", std_cphase, ' )', r"$\pi$", '\n',
          'Population Recovery |e> Qb Target: {}% \n' \
          .format(population_recovery_opt),
          '@ flux pulse Paramters: \n',
          'Pulse Length: {:0.1f} ns \n'.format(pulse_length_best * 1e9),
          'Pulse Ampllitude: {:0.4f} V \n'.format(pulse_amplitude_best))
    if update:
        qbc.set('CZ_{}_amp'.format(qbt.name), pulse_amplitude_best)
        qbc.set('CZ_{}_length'.format(qbt.name), pulse_length_best)
    if full_output:
        return pulse_length_best, pulse_amplitude_best, \
               [population_recovery_opt, cphase_opt], [std_cphase]
    else:
        return pulse_length_best, pulse_amplitude_best


def cphase_finetune_parameters(qbc, qbt, qbr, flux_length, flux_amplitudes,
                               phases, MC, save_fig=True, show=True):
    """
    measures cphases for a single slice of chevron with fixed flux length.
    Returns the best amplitude in flux_amplitudes for a cphase of pi.
    """
    flux_lengths = len(flux_amplitudes) * [flux_length]
    cphases, population_losses, ma_ram2D = \
        measure_cphase(qbc, qbt, qbr,
                       flux_lengths,
                       flux_amplitudes,
                       phases=phases,
                       plot=True,
                       MC=MC,
                       fit_statistics=False)
    cphases %= 2 * np.pi
    fit_res = lmfit.Model(lambda x, m, b: m * np.tan(x / 2 - np.pi / 2) + b).fit(
        x=cphases, data=flux_amplitudes, m=1, b=np.mean(flux_amplitudes))
    best_amp = fit_res.model.func(np.pi, **fit_res.best_values)
    amps_model = fit_res.model.func(cphases, **fit_res.best_values)
    fig, ax = plt.subplots()
    ax.plot(cphases * 180 / np.pi, flux_amplitudes / 1e-3, 'o-')
    ax.plot(cphases * 180 / np.pi, amps_model / 1e-3, '-r')
    ax.hlines(best_amp / 1e-3, cphases[0] * 180 / np.pi, cphases[-1] * 180 / np.pi)
    ax.vlines(180, flux_amplitudes.min() / 1e-3, flux_amplitudes.max() / 1e-3)
    ax.set_ylabel('Flux pulse amplitude (mV)')
    ax.set_xlabel('Conditional phase (rad)')
    ax.set_title('CZ {}-{}'.format(qbc.name, qbt.name))

    ax.text(0.5, 0.95, 'Best amp = {:.6f} V'.format(best_amp),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)

    if save_fig:
        fig_title = 'CPhase_amp_sweep_{}_{}'.format(qbc.name, qbt.name)
        fig_title = '{}--{:%Y%m%d_%H%M%S}'.format(
            fig_title, datetime.datetime.now())
        save_folder = ma_ram2D.folder
        filename = os.path.abspath(os.path.join(save_folder, fig_title + '.png'))
        fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

    return best_amp


def measure_measurement_induced_dephasing(qb_dephased, qb_targeted, phases, amps,
                                          readout_separation, nr_readouts=1,
                                          label=None, n_cal_points_per_state=1,
                                          cal_states='auto', prep_params=None,
                                          exp_metadata=None, analyze=True,
                                          upload=True, **kw):
    classified = kw.get('classified', False)
    predictive_label = kw.pop('predictive_label', False)
    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in qb_dephased])

    if label is None:
        label = 'measurement_induced_dephasing_x{}_{}_{}'.format(
            nr_readouts,
            ''.join([qb.name for qb in qb_dephased]),
            ''.join([qb.name for qb in qb_targeted]))

    hard_sweep_params = {
        'phase': {'unit': 'deg',
            'values': np.tile(phases, len(amps))},
        'ro_amp_scale': {'unit': 'deg',
            'values': np.repeat(amps, len(phases))}
    }

    for qb in set(qb_targeted) | set(qb_dephased):
        MC = qb.instr_mc.get_instr()
        qb.prepare(drive='timedomain')

    cal_states = CalibrationPoints.guess_cal_states(cal_states)
    cp = CalibrationPoints.multi_qubit([qb.name for qb in qb_dephased], cal_states,
                                       n_per_state=n_cal_points_per_state)

    operation_dict = get_operation_dict(list(set(qb_dephased + qb_targeted)))
    seq, sweep_points = mqs.measurement_induced_dephasing_seq(
        [qb.name for qb in qb_targeted], [qb.name for qb in qb_dephased], operation_dict,
        amps, phases, pihalf_spacing=readout_separation, prep_params=prep_params,
        cal_points=cp, upload=False, sequence_name=label)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload,
        parameter_name='readout_idx', unit='')
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(sweep_points)

    det_name = 'int_avg{}_det'.format('_classif' if classified else '')
    det_func = get_multiplexed_readout_detector_functions(
        qb_dephased, nr_averages=max(qb.acq_averages() for qb in qb_dephased)
    )[det_name]
    MC.set_detector_function(det_func)

    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({'qb_dephased': [qb.name for qb in qb_dephased],
                         'qb_targeted': [qb.name for qb in qb_targeted],
                         'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'classified_ro': classified,
                         'rotate': len(cal_states) != 0 and not classified,
                         'data_to_fit': {qb.name: 'pe' for qb in qb_dephased},
                         'hard_sweep_params': hard_sweep_params})

    MC.run(label, exp_metadata=exp_metadata)

    tda.MeasurementInducedDephasingAnalysis(qb_names=[qb.name for qb in qb_dephased])


def calibrate_n_qubits(qubits, f_LO, sweep_points_dict, sweep_params=None,
                       artificial_detuning=None,
                       cal_points=True, no_cal_points=4, upload=True,
                       MC=None, soft_avgs=1, n_rabi_pulses=1,
                       thresholded=False,  # analyses can't do thresholded=True!
                       analyze=True, update=False,
                       UHFQC=None, pulsar=None, **kw):
    """
    Args:
        qubits: list of qubits
        f_LO: multiplexed RO LO freq
        sweep_points_dict:  dict of the form {msmt_name: sweep_points_array}
            where msmt_name must be one of the following:
            ['rabi', 'n_rabi', 'ramsey', 'qscale', 'T1', 'T2'}
        sweep_params: this function defines this variable for each msmt. But
            see the seqeunce function mqs.general_multi_qubit_seq for details
        artificial_detuning: for ramsey and T2 (echo) measurements. It is
            ignored for the other measurements
        cal_points: whether to prepare cal points or not
        no_cal_points: how many cal points to prepare
        upload: whether to upload to AWGs
        MC: MC object
        soft_avgs:  soft averages
        n_rabi_pulses: for the n_rabi measurement
        thresholded: whether to threshold the results (NOT IMPLEMENTED)
        analyze: whether to analyze
        update: whether to update relevant parameters based on analysis
        UHFQC: UHFQC object
        pulsar: pulsar

    Kwargs:
        This function can also add dynamical decoupling (DD) pulses with the
        following parameters:

        nr_echo_pulses (int, default=0): number of DD pulses; if 0 then this
            function will not add DD pulses
        UDD_scheme (bool, default=True): if True, it uses the Uhrig DD scheme,
            else it uses the CPMG scheme
        idx_DD_start (int, default:-1): index of the first DD pulse in the
            waveform for a single qubit. For example, is we have n=3 qubits,
            and have 4 pulses per qubit, and we want to inset DD pulses
            between the first and second pulse, then idx_DD_start = 1.
            For a Ramsey experiment (2 pulses per qubit), idx_DD_start = -1
            (default value) and the DD pulses are inserted between the
            two pulses.

        You can also add the kwargs used in the standard TD analysis functions.
    """

    if MC is None:
        MC = qubits[0].MC
    if UHFQC is None:
        UHFQC = qubits[0].UHFQC
    if pulsar is None:
        pulsar = qubits[0].AWG
    artificial_detuning_echo = kw.pop('artificial_detuning_echo', None)

    qubit_names = [qb.name for qb in qubits]
    if len(qubit_names) == 1:
        msmt_suffix = qubits[0].msmt_suffix
    elif len(qubit_names) > 5:
        msmt_suffix = '_{}qubits'.format(len(qubit_names))
    else:
        msmt_suffix = '_qbs{}'.format(''.join([i[-1] for i in qubit_names]))

    sweep_points_dict = deepcopy(sweep_points_dict)
    for key, spts in sweep_points_dict.items():
        if spts is None:
            if key == 'n_rabi':
                sweep_points_dict[key] = {}
                for qb in qubits:
                    sweep_points_dict[key][qb.name] = \
                        np.linspace(
                            (n_rabi_pulses - 1) * qb.amp180() / n_rabi_pulses,
                            min((n_rabi_pulses + 1) * qb.amp180() / n_rabi_pulses,
                                0.9), 41)
            else:
                raise ValueError('Sweep points for {} measurement are not '
                                 'defined.'.format(key))

    if cal_points:
        sweep_points_dict = deepcopy(sweep_points_dict)
        for key, spts in sweep_points_dict.items():
            if not isinstance(spts, dict):
                if key == 'qscale':
                    temp_array = np.zeros(3 * spts.size)
                    np.put(temp_array, list(range(0, temp_array.size, 3)), spts)
                    np.put(temp_array, list(range(1, temp_array.size, 3)), spts)
                    np.put(temp_array, list(range(2, temp_array.size, 3)), spts)
                    spts = temp_array
                    step = np.abs(spts[-1] - spts[-4])
                else:
                    step = np.abs(spts[-1] - spts[-2])
                if no_cal_points == 4:
                    sweep_points_dict[key] = np.concatenate(
                        [spts, [spts[-1] + step, spts[-1] + 2 * step,
                                spts[-1] + 3 * step, spts[-1] + 4 * step]])
                elif no_cal_points == 2:
                    sweep_points_dict[key] = np.concatenate(
                        [spts, [spts[-1] + step, spts[-1] + 2 * step]])
                else:
                    sweep_points_dict[key] = spts
            else:
                for k in spts:
                    if key == 'qscale':
                        temp_array = np.zeros(3 * spts[k].size)
                        np.put(temp_array, list(range(0, temp_array.size, 3)),
                               spts[k])
                        np.put(temp_array, list(range(1, temp_array.size, 3)),
                               spts[k])
                        np.put(temp_array, list(range(2, temp_array.size, 3)),
                               spts[k])
                        spts[k] = temp_array
                        step = np.abs(spts[k][-1] - spts[k][-4])
                    else:
                        step = np.abs(spts[k][-1] - spts[k][-2])
                    if no_cal_points == 4:
                        sweep_points_dict[key][k] = np.concatenate(
                            [spts[k], [spts[k][-1] + step, spts[k][-1] + 2 * step,
                                       spts[k][-1] + 3 * step,
                                       spts[k][-1] + 4 * step]])
                    elif no_cal_points == 2:
                        sweep_points_dict[key][k] = np.concatenate(
                            [spts[k],
                             [spts[k][-1] + step, spts[k][-1] + 2 * step]])
                    else:
                        sweep_points_dict[key][k] = spts[k]

    # set up multiplexed readout
    multiplexed_pulse(qubits, f_LO, upload=True)
    operation_dict = get_operation_dict(qubits)
    if thresholded:
        key = 'dig'
    else:
        key = 'int'

    nr_averages = max([qb.RO_acq_averages() for qb in qubits])
    df = get_multiplexed_readout_detector_functions(
        qubits, nr_averages=nr_averages)[key + '_avg_det']

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    # Do measurements
    # RABI
    if 'rabi' in sweep_points_dict:
        exp_metadata = {}
        sweep_points = sweep_points_dict['rabi']

        if sweep_params is None:
            sweep_params = (
                ('X180', {'pulse_pars': {'amplitude': (lambda sp: sp)},
                          'repeat': 1}),
            )

        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='amplitude',
                                         unit='V', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(df)
        label = 'Rabi' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            rabi_ana = tda.RabiAnalysis(qb_names=qubit_names)
            if update:
                for qb in qubits:
                    try:
                        qb.amp180(rabi_ana.proc_data_dict[
                                      'analysis_params_dict'][qb.name]['piPulse'])
                        qb.amp90_scale(0.5)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)

    # N-RABI
    if 'n_rabi' in sweep_points_dict:
        exp_metadata = {}
        sweep_points = sweep_points_dict['n_rabi']
        if sweep_params is None:
            sweep_params = (
                ('X180', {'pulse_pars': {'amplitude': (lambda sp: sp)},
                          'repeat': n_rabi_pulses}),
            )

        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='amplitude',
                                         unit='V', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points[list(sweep_points)[0]])
        MC.set_detector_function(df)
        label = 'Rabi-n{}'.format(n_rabi_pulses) + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            rabi_ana = tda.RabiAnalysis(qb_names=qubit_names)
            if update:
                for qb in qubits:
                    try:
                        qb.amp180(rabi_ana.proc_data_dict[
                                      'analysis_params_dict'][qb.name]['piPulse'])
                        qb.amp90_scale(0.5)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)

    # RAMSEY
    if 'ramsey' in sweep_points_dict:
        exp_metadata = {}
        if artificial_detuning is None:
            raise ValueError('Specify an artificial_detuning for the Ramsey '
                             'measurement.')
        sweep_points = sweep_points_dict['ramsey']
        drag_pulse_length = qubits[0].nr_sigma() * qubits[0].gauss_sigma()
        zz_coupling = 470e3
        if sweep_params is None:
            sweep_params = (
                ('X90', {}),
                ('X90', {
                    'pulse_pars': {
                        'refpoint': 'start',
                        'pulse_delay': (lambda sp: sp),
                        'phase': (lambda sp:
                                  ((sp - sweep_points[0]) * artificial_detuning *
                                   360) % 360),
                        # 'basis_rotation': (lambda sp: 2*np.pi*zz_coupling *
                        #                   (sp+drag_pulse_length)*180/np.pi)
                    }}),

            )
        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='time',
                                         unit='s', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(df)
        label = 'Ramsey' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        exp_metadata['artificial_detuning'] = artificial_detuning
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            ramsey_ana = tda.RamseyAnalysis(qb_names=qubit_names)
            if update:
                for qb in qubits:
                    try:
                        qb.f_qubit(ramsey_ana.proc_data_dict[
                                       'analysis_params_dict'][qb.name][
                                       'exp_decay_' + qb.name]['new_qb_freq'])
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)
                    try:
                        qb.T2_star(ramsey_ana.proc_data_dict[
                                       'analysis_params_dict'][qb.name][
                                       'exp_decay_' + qb.name]['T2_star'])
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)

    # QSCALE
    if 'qscale' in sweep_points_dict:
        exp_metadata = {}
        sweep_points = sweep_points_dict['qscale']

        if sweep_params is None:
            sweep_params = (
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i % 3 == 0)}),
                ('X180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                          'condition': (lambda i: i % 3 == 0)}),
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i % 3 == 1)}),
                ('Y180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                          'condition': (lambda i: i % 3 == 1)}),
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i % 3 == 2)}),
                ('mY180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                           'condition': (lambda i: i % 3 == 2)}),
                ('RO', {})
            )

        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='qscale_factor',
                                         unit='', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(df)
        label = 'QScale' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            qscale_ana = tda.QScaleAnalysis(qb_names=qubit_names)
            if update:
                for qb in qubits:
                    try:
                        qb.motzoi(qscale_ana.proc_data_dict[
                                      'analysis_params_dict'][qb.name]['qscale'])
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)

    # T1
    if 'T1' in sweep_points_dict:
        exp_metadata = {}
        sweep_points = sweep_points_dict['T1']
        if sweep_params is None:
            sweep_params = (
                ('X180', {}),
                ('RO mux', {'pulse_pars': {'pulse_delay': (lambda sp: sp)}})
            )

        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='time',
                                         unit='s', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(df)
        label = 'T1' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            T1_ana = tda.T1Analysis(qb_names=qubit_names)
            if update:
                for qb in qubits:
                    try:
                        qb.T1(T1_ana.proc_data_dict['analysis_params_dict'][
                                  qb.name]['T1'])
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)
    # T2 ECHO
    if 'T2' in sweep_points_dict:
        exp_metadata = {}
        sweep_points = sweep_points_dict['T2']
        if sweep_params is None:
            sweep_params = (
                ('X90', {}),
                ('X180', {'pulse_pars': {'refpoint': 'start',
                                         'pulse_delay': (lambda sp: sp / 2)}}),
                ('X90', {'pulse_pars': {
                    'refpoint': 'start',
                    'pulse_delay': (lambda sp: sp / 2)}})
            )

        if artificial_detuning_echo is not None:
            sweep_params[-1][1]['pulse_pars']['phase'] = \
                lambda sp: ((sp - sweep_points[0]) *
                            artificial_detuning_echo * 360) % 360

        sf = awg_swf2.calibrate_n_qubits(sweep_params=sweep_params,
                                         sweep_points=sweep_points,
                                         qubit_names=qubit_names,
                                         operation_dict=operation_dict,
                                         cal_points=cal_points,
                                         upload=upload,
                                         parameter_name='time',
                                         unit='s', **kw)

        MC.soft_avg(soft_avgs)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(df)
        label = 'T2_echo' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points,
                            'artificial_detuning': artificial_detuning_echo}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            echo_ana = tda.EchoAnalysis(
                qb_names=qubit_names,
                options_dict={'artificial_detuning': artificial_detuning_echo})
            if update:
                for qb in qubits:
                    try:
                        qb.T2(echo_ana.proc_data_dict[
                                  'analysis_params_dict'][qb.name]['T2_echo'])
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                    'updated.' % e)


def measure_chevron(qbc, qbt, qbr, hard_sweep_params, soft_sweep_params,
                    cz_pulse_name, upload=True, label=None,
                    classified=False, n_cal_points_per_state=2,
                    cal_states='auto', prep_params=None,
                    exp_metadata=None, analyze=True):

    if len(list(soft_sweep_params)) > 1:
        log.warning('There is more than one soft sweep parameter.')
    if label is None:
        label = 'Chevron_{}{}'.format(qbc.name, qbt.name)
    MC = qbr.instr_mc.get_instr()
    for qb in [qbc, qbt, qbr]:
        qb.prepare(drive='timedomain')

    cal_states = CalibrationPoints.guess_cal_states(cal_states)
    cp = CalibrationPoints.single_qubit(qbr.name, cal_states,
                                        n_per_state=n_cal_points_per_state)

    if prep_params is None:
        prep_params = \
            get_multi_qubit_prep_params([qb.preparation_params()
                                         for qb in [qbc, qbt]])
    sequences, hard_sweep_points, soft_sweep_points = \
        fsqs.chevron_seqs(
            qbc_name=qbc.name, qbt_name=qbt.name, qbr_name=qbr.name,
            hard_sweep_dict=hard_sweep_params,
            soft_sweep_dict=soft_sweep_params,
            operation_dict=get_operation_dict([qbc, qbt, qbr]),
            cz_pulse_name=cz_pulse_name,
            cal_points=cp, upload=False, prep_params=prep_params)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=sequences[0], upload=upload,
        parameter_name=list(hard_sweep_params)[0],
        unit=list(hard_sweep_params.values())[0]['unit'])
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)

    channels_to_upload = [qbc.get_operation_dict()[cz_pulse_name]['channel']]
    MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
        hard_sweep_func, sequences,
        list(soft_sweep_params)[0], list(soft_sweep_params.values())[0]['unit'],
        channels_to_upload=channels_to_upload))
    MC.set_sweep_points_2D(soft_sweep_points)
    MC.set_detector_function(qbr.int_avg_classif_det if classified
                             else qbr.int_avg_det)
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'rotate': len(cal_states) != 0,
                         'data_to_fit': {qbr.name: 'pe'},
                         'hard_sweep_params': hard_sweep_params,
                         'soft_sweep_params': soft_sweep_params})
    MC.run_2D(name=label, exp_metadata=exp_metadata)

    if analyze:
        tda.MultiQubit_TimeDomain_Analysis(qb_names=[qbr.name],
                                           options_dict={'TwoD': True})


def measure_cphase_nn(qbc, qbt, qbr, lengths, amps, alphas=None,
                   CZ_pulse_name=None,
                   phases=None, MC=None,
                   cal_points=False, plot=False,
                   save_plot=True,
                   prepare_for_timedomain=True,
                   output_measured_values=False,
                   analyze=True, upload=True, **kw):
    '''
    method to measure the phase acquired during a flux pulse conditioned on the
    state of the control qubit (self).
    In this measurement, the phase from two Ramsey type measurements
    on qb_target is measured, once with the control qubit in the excited state
    and once in the ground state. The conditional phase is calculated as the
    difference.


    Args:
        qb_target (QuDev_transmon): target qubit / non-fluxed qubit
        amps (list): list or array of flux pulse amplitudes
        lengths (list):  list or array of flux pulse lengths (must have same
                         dimension as amps)
        phases (array): phases used for the Ramsey type phase sweep
        spacing (float): spacing between flux pulse and Ramsey pulses in s
        MC (optional): measurement control
        cal_points (bool): if True, calibration points are measured
        plot (bool): if true, the phase fit is shown
        return_population_loss: if true, the population loss (loss of contrast
                                when having the control qubit in the excited
                                state is returned)
        upload_AWGs (list): list of the AWGs to be uploaded
        upload_channels (list): list of channels to be uploaded
        prepare_for_timedomain (bool): if False, the
                                       self.prepare_for_timedomain()
                                       is NOT called

    Returns:
        cphases (numpy array): array of the conditional phases measured at
                              (amps[i], lengths[i])
    '''
    if len(amps) != len(lengths):
        raise ValueError('amps and lengths must have the same '
                         'dimension.')

    if MC is None:
        MC = qbc.MC

    if phases is None:
        phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        phases = np.concatenate((phases, phases))

    operation_dict = get_operation_dict([qbc, qbt, qbr])
    if CZ_pulse_name is None:
        CZ_pulse_name = 'CZ ' + qbt.name + ' ' + qbc.name

    optimize_nz_pulse = False

    CZ_pulse_channel = operation_dict[CZ_pulse_name]['channel']
    max_flux_length = np.max(lengths)
    if optimize_nz_pulse:
        s1 = awg_swf.CPhase_NZ_hard_swf(phases,
                                        qbc.name,
                                        qbt.name,
                                        CZ_pulse_name,
                                        CZ_pulse_channel,
                                        operation_dict,
                                        max_flux_length,
                                        cal_points=cal_points,
                                        reference_measurements=True,
                                        upload=upload)
        s2 = awg_swf.Flux_pulse_CPhase_soft_swf(s1, sweep_param='length',
                                                upload=upload)
        s3 = awg_swf.Flux_pulse_CPhase_soft_swf(s1, sweep_param='amplitude',
                                                upload=upload)
        s4 = awg_swf.Flux_pulse_CPhase_soft_swf(s1, sweep_param='alpha',
                                                upload=upload)
        MC.set_sweep_functions([s1, s2, s3, s4])
        MC.set_sweep_points(phases)
        # Here the order of the parameters matters! Paramters must be
        # set in the same order as their sweepfunctions!
        MC.set_sweep_points_2D(np.array([lengths, amps, alphas]).T)
        MC.set_detector_function(qbr.int_avg_det)
    else:
        s1 = awg_swf.Flux_pulse_CPhase_hard_swf_new(phases,
                                                    qbc.name,
                                                    qbt.name,
                                                    qbr.name,
                                                    CZ_pulse_name,
                                                    CZ_pulse_channel,
                                                    operation_dict,
                                                    max_flux_length,
                                                    cal_points=cal_points,
                                                    reference_measurements=True,
                                                    upload=upload)
        s2 = awg_swf.Flux_pulse_CPhase_soft_swf(s1, sweep_param='length',
                                                upload=upload)
        s3 = awg_swf.Flux_pulse_CPhase_soft_swf(s1, sweep_param='amplitude',
                                                upload=upload)

        MC.set_sweep_functions([s1, s2, s3])
        MC.set_sweep_points(phases)
        # Here the order of the parameters matters! Paramters must be
        # set in the same order as their sweepfunctions!
        MC.set_sweep_points_2D(np.array([lengths, amps]).T)
        MC.set_detector_function(qbr.int_avg_det)

    t0 = time.time()
    if prepare_for_timedomain:
        for qb in [qbc, qbt, qbr]:
            qb.prepare_for_timedomain()
    MC.run_2D('CPhase_measurement_{}_{}'.format(qbc.name, qbt.name))

    t1 = time.time()
    log.info('Measured Cphases with ',
          len(amps) * len(phases),
          ' sweeppoints in T=', t1 - t0, ' s.')

    if analyze:
        flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis_Predictive(
            label='CPhase_measurement_{}_{}'.format(qbc.name, qbt.name),
            qb_name=qbc.name, cal_points=cal_points, plot=plot,
            save_plot=save_plot, reference_measurements=True,
            only_cos_fits=True, **kw)
        cphases = flux_pulse_ma.cphases
        population_losses = flux_pulse_ma.population_losses
        if output_measured_values:
            log.info('fitted phases: ', cphases)
            log.info('pop loss: ', population_losses)
        return cphases, population_losses, flux_pulse_ma
    else:
        return


def measure_cphase(qbc, qbt, soft_sweep_params, cz_pulse_name,
                   hard_sweep_params=None, max_flux_length=None,
                   num_cz_gates=1, n_cal_points_per_state=1, cal_states='auto',
                   prep_params=None, exp_metadata=None, label=None,
                   analyze=True, upload=True, for_ef=True, **kw):
    '''
    method to measure the leakage and the phase acquired during a flux pulse
    conditioned on the state of the control qubit (self).
    In this measurement, the phase from two Ramsey type measurements
    on qb_target is measured, once with the control qubit in the excited state
    and once in the ground state. The conditional phase is calculated as the
    difference.

    Args:
        qbc (QuDev_transmon): control qubit / fluxed qubit
        qbt (QuDev_transmon): target qubit / non-fluxed qubit
    '''
    plot_all_traces = kw.get('plot_all_traces', True)
    plot_all_probs = kw.get('plot_all_probs', True)
    classified = kw.get('classified', False)
    predictive_label = kw.pop('predictive_label', False)
    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in [qbc, qbt]])

    if label is None:
        if predictive_label:
            label = 'Predictive_cphase_nz_measurement'
        else:
            label = 'CPhase_nz_measurement'
        if classified:
            label += '_classified'
        if 'active' in prep_params['preparation_type']:
            label += '_reset'
        if num_cz_gates > 1:
            label += f'_{num_cz_gates}_gates'
        label += f'_{qbc.name}_{qbt.name}'

    if hard_sweep_params is None:
        hard_sweep_params = {
            'phase': {'values': np.tile(np.linspace(0, 2*np.pi, 6)*180/np.pi, 2),
                      'unit': 'deg'}
        }

    for qb in [qbc, qbt]:
        MC = qb.instr_mc.get_instr()
        qb.prepare(drive='timedomain')

    cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                    for_ef=for_ef)
    cp = CalibrationPoints.multi_qubit([qbc.name, qbt.name], cal_states,
                                        n_per_state=n_cal_points_per_state)

    if max_flux_length is not None:
        log.debug(f'max_flux_length = {max_flux_length*1e9:.2f} ns, set by user')
    operation_dict = get_operation_dict([qbc, qbt])
    sequences, hard_sweep_points, soft_sweep_points = \
        fsqs.cphase_seqs(
            hard_sweep_dict=hard_sweep_params,
            soft_sweep_dict=soft_sweep_params,
            qbc_name=qbc.name, qbt_name=qbt.name,
            cz_pulse_name=cz_pulse_name,
            operation_dict=operation_dict,
            cal_points=cp, upload=False, prep_params=prep_params,
            max_flux_length=max_flux_length,
            num_cz_gates=num_cz_gates)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=sequences[0], upload=upload,
        parameter_name=list(hard_sweep_params)[0],
        unit=list(hard_sweep_params.values())[0]['unit'])
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)

    channels_to_upload = [operation_dict[cz_pulse_name]['channel']]
    MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
        hard_sweep_func, sequences,
        list(soft_sweep_params)[0], list(soft_sweep_params.values())[0]['unit'],
        channels_to_upload=channels_to_upload))
    MC.set_sweep_points_2D(soft_sweep_points)

    det_get_values_kws = {'classified': classified,
                          'correlated': False,
                          'thresholded': True,
                          'averaged': True}
    det_name = 'int_avg{}_det'.format('_classif' if classified else '')
    det_func = get_multiplexed_readout_detector_functions(
        [qbc, qbt], nr_averages=max(qb.acq_averages() for qb in [qbc, qbt]),
        det_get_values_kws=det_get_values_kws)[det_name]
    MC.set_detector_function(det_func)

    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({'leakage_qbname': qbc.name,
                         'cphase_qbname': qbt.name,
                         'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'classified_ro': classified,
                         'rotate': len(cal_states) != 0 and not classified,
                         'cal_states_rotations':
                             {qbc.name: {'g': 0, 'f': 1},
                              qbt.name: {'g': 0, 'e': 1}} if
                             (len(cal_states) != 0 and not classified) else None,
                         'data_to_fit': {qbc.name: 'pf', qbt.name: 'pe'},
                         'hard_sweep_params': hard_sweep_params,
                         'soft_sweep_params': soft_sweep_params})
    MC.run_2D(label, exp_metadata=exp_metadata)
    if analyze:
        if classified:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in [qbc, qbt]}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_det.value_names]
                           for qb in [qbc, qbt]}
        flux_pulse_tdma = tda.CPhaseLeakageAnalysis(
            qb_names=[qbc.name, qbt.name],
            options_dict={'TwoD': True, 'plot_all_traces': plot_all_traces,
                          'plot_all_probs': plot_all_probs,
                          'channel_map': channel_map})
        cphases = flux_pulse_tdma.proc_data_dict[
            'analysis_params_dict']['cphase']['val']
        population_losses = flux_pulse_tdma.proc_data_dict[
            'analysis_params_dict']['population_loss']['val']
        leakage = flux_pulse_tdma.proc_data_dict[
            'analysis_params_dict']['leakage']['val']
        return cphases, population_losses, leakage, flux_pulse_tdma
    else:
        return

def measure_arbitrary_phase(qbc, qbt, target_phases, phase_func, cz_pulse_name,
        soft_sweep_params=dict(), measure_dynamic_phase=False,
        measure_conditional_phase=True, hard_sweep_params=None,
        num_cz_gates=1, n_cal_points_per_state=1, cal_states='auto',
        classified_ro=True, prep_params=None, exp_metadata=dict(), label=None,
        analyze=True, upload=True, for_ef=True, **kw):
    '''
    method to measure the leakage and the phase acquired during a flux pulse
    conditioned on the state of the control qubit (self).
    In this measurement, the phase from two Ramsey type measurements
    on qb_target is measured, once with the control qubit in the excited state
    and once in the ground state. The conditional phase is calculated as the
    difference.

    Args:
        qbc (QuDev_transmon): control qubit / fluxed qubit
        qbt (QuDev_transmon): target qubit / non-fluxed qubit
        phase_func (callable): function with input the target phase, returning
         (flux pulse amplitude, dyn_phase)
    '''

    if label is None:
        label = 'Arbitrary_Phase_{}_{}'.format(qbc.name, qbt.name)
    assert qbc.get_operation_dict()[cz_pulse_name]['pulse_type'] == \
        'BufferedCZPulseEffectiveTime', "Arbritrary phase measurement requires" \
            "'BufferedCZPulseEffectiveTime' pulse type but pulse type is '{}'" \
        .format(qbc.get_operation_dict()[cz_pulse_name]['pulse_type'])
    results = dict() #dictionary to store measurement results
    amplitudes, predicted_dyn_phase = phase_func(target_phases)
    soft_sweep_params['amplitude'] = dict(values=amplitudes, unit='V')
    exp_metadata.update(dict(target_phases=target_phases))
    if measure_conditional_phase:
        cphases, population_losses, leakage, flux_pulse_tdma = \
            measure_cphase(qbc=qbc, qbt=qbt, soft_sweep_params=soft_sweep_params,
                           cz_pulse_name=cz_pulse_name,
                           hard_sweep_params=hard_sweep_params,
                           num_cz_gates=num_cz_gates,
                           n_cal_points_per_state=n_cal_points_per_state,
                           cal_states=cal_states,
                           prep_params=prep_params, exp_metadata=exp_metadata,
                           label=label, analyze=True, upload=upload, for_ef=for_ef,
                           classified= classified_ro,
                           **kw)
        if analyze:
            # get folder to save figures.
            # FIXME: temporary while no proper analysis class is made
            a = ma.MeasurementAnalysis(auto=False)
            a.get_naming_and_values()
            save_folder = a.folder
            if kw.get("wrap_phase", True):
                tol = kw.get("wrap_tol", 0)
                cphases[cphases < 0 - tol] = cphases[cphases < 0 - tol] + 2 * np.pi
                cphases[cphases > 2 * np.pi + tol] = cphases[cphases > 2 * np.pi + tol] + \
                                                     2 * np.pi
                target_phases[target_phases < 0 - tol] = \
                    target_phases[target_phases < 0 - tol] + 2 * np.pi
                target_phases[target_phases > 2 * np.pi + tol] = \
                    target_phases[target_phases > 2 * np.pi + tol] + 2 * np.pi

            for param_name, values in soft_sweep_params.items():
                fig, ax = plt.subplots(2, sharex=True, figsize=(7, 8))
                ax[0].scatter(values['values'], target_phases * 180 / np.pi,
                              label='Target phase', marker='x')
                ax[0].scatter(values['values'], cphases * 180 / np.pi,
                              label='Measured phase', marker='x')
                ax[0].set_ylabel(f"Conditional Phase (deg)")
                ax[0].legend(prop=dict(size=12))

                diff_phases = ((cphases - target_phases + np.pi) %
                               (2 * np.pi) - np.pi) * 180 / np.pi

                ax[1].scatter(values['values'], diff_phases, label='Target - Measured')
                ax[1].set_xlabel(f"{param_name} ({values.get('unit', '')})")
                ax[1].set_ylabel(f"Conditional Phase (deg)")
                ax[1].plot([], [], color='w',
                           label=f"mean err: {np.mean(diff_phases):0.2f} $\pm$ "
                           f"{np.std(diff_phases):0.2f}°\n" \
                           f"median err: {np.median(diff_phases): 0.2f}°")
                ax[1].legend(prop=dict(size=12))
                fig.savefig(
                    os.path.join(save_folder,
                                 f"cphase_and_target_phase_vs_{param_name}.png"))
            results['cphases'] = cphases
            results['cphases_diff'] = diff_phases

    if measure_dynamic_phase:
        dyn_phases = []
        # FIXME: infering amplitude parameter from pulse name, but if naming
        #  protocol changes this might fail
        ampl_param_name = "_".join(cz_pulse_name.split(" ")[:-1] + ["amplitude"])
        for amp in amplitudes:
            with temporary_value(
                    (getattr(qbc, ampl_param_name), amp)):
                dyn_phases.append(
                    measure_dynamic_phases(qbc, qbt, cz_pulse_name, update=False,
                                           qubits_to_measure=[qbc],
                                           reset_phases_before_measurement=True))

        if analyze:
            a = ma.MeasurementAnalysis(auto=False)
            a.get_naming_and_values()
            save_folder = a.folder
            dyn_phases = np.array([d[qbc.name] for d in dyn_phases])

            if kw.get("wrap_phase", True):
                dyn_phases[dyn_phases < 0 ] = dyn_phases[dyn_phases < 0] + 360
                dyn_phases[dyn_phases > 360] = dyn_phases[dyn_phases > 360] + 360
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].scatter(amplitudes, predicted_dyn_phase,
                          label='Predicted dynamic phase', marker='x')
            ax[0].scatter(amplitudes, dyn_phases, marker='x',
                          label='Measured dynamic phase')
            ax[0].set_ylabel(f"Dynamic Phase (deg)")
            ax[0].legend(prop=dict(size=12))

            # wrapping to get difference around 0 degree
            diff_dyn_phases = \
                (dyn_phases - predicted_dyn_phase + 180) % 360 - 180
            ax[1].scatter(amplitudes, diff_dyn_phases, label='Target - Measured')
            ax[1].set_xlabel(f"Amplitude (V)")
            ax[1].set_ylabel(f"Dynamic Phase (deg)")
            ax[1].legend(prop=dict(size=12))
            fig.savefig(os.path.join(save_folder, "dynamic_phase.png"))
            results['dphases'] = dyn_phases
            results['dphases_diff'] = diff_dyn_phases

    return results


def measure_dynamic_phases(qbc, qbt, cz_pulse_name, hard_sweep_params=None,
                           qubits_to_measure=None, cal_points=True,
                           analyze=True, upload=True, n_cal_points_per_state=1,
                           cal_states='auto', prep_params=None,
                           exp_metadata=None, classified=False, update=False,
                           reset_phases_before_measurement=True,
                           basis_rot_par=None, prepend_n_cz=0):

    if qubits_to_measure is None:
        qubits_to_measure = [qbc, qbt]
    if hard_sweep_params is None:
        hard_sweep_params = {
            'phase': {
                'values': np.tile(np.linspace(0, 2*np.pi, 6)*180/np.pi, 2),
                'unit': 'deg'}}

    if basis_rot_par is None:
        basis_rot_par = f'{cz_pulse_name[:-8]}_{qbt.name}_basis_rotation'

    if reset_phases_before_measurement:
        dyn_phases = {qb.name: 0 for qb in qubits_to_measure}
        qbc.set(basis_rot_par, dyn_phases)
    else:
        dyn_phases = qbc.get(basis_rot_par)
    for qb in qubits_to_measure:
        label = f'Dynamic_phase_measurement_CZ{qbt.name}{qbc.name}-{qb.name}'
        qb.prepare(drive='timedomain')
        MC = qbc.instr_mc.get_instr()

        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                qb.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None

        prep_params = qb.preparation_params()
        seq, hard_sweep_points = \
            fsqs.dynamic_phase_seq(
                qb_name=qb.name, hard_sweep_dict=hard_sweep_params,
                operation_dict=get_operation_dict(qubits_to_measure),
                cz_pulse_name=cz_pulse_name, cal_points=cp,
                prepend_n_cz=prepend_n_cz,
                upload=False, prep_params=prep_params)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload,
            parameter_name=list(hard_sweep_params)[0],
            unit=list(hard_sweep_params.values())[0]['unit']))
        MC.set_sweep_points(hard_sweep_points)
        MC.set_detector_function(qb.int_avg_classif_det if classified
                                 else qb.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'use_cal_points': cal_points,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': cal_points,
                             'data_to_fit': {qb.name: 'pe'},
                             'cal_states_rotations':
                                 {qb.name: {'g': 0, 'e': 1}},
                             'hard_sweep_params': hard_sweep_params})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            flux_pulse_amp = None
            if 'amplitude' in qbc.get_operation_dict()[cz_pulse_name]:
                flux_pulse_amp = qbc.get_operation_dict()[cz_pulse_name]['amplitude']
            MA = tda.CZDynamicPhaseAnalysis(qb_names=[qb.name], options_dict={
                'flux_pulse_length': qbc.get_operation_dict()[cz_pulse_name][
                    'pulse_length'],
                'flux_pulse_amp': flux_pulse_amp})
            dyn_phases[qb.name] = \
                MA.proc_data_dict['analysis_params_dict'][qb.name][
                    'dynamic_phase']['val']*180/np.pi
    if update and reset_phases_before_measurement:
        qbc.set(basis_rot_par, dyn_phases)
    return dyn_phases


def measure_ramsey_with_flux_pulse(qb, cz_pulse_name, hard_sweep_params=None,
                           qb_freq=None,
                           artificial_detunings=20e6,
                           cal_points=True,
                           analyze=True, upload=True, n_cal_points_per_state=2,
                           cal_states='auto', prep_params=None,
                           exp_metadata=None, classified=False):

    if hard_sweep_params is None:
        hard_sweep_params = {
            'Delay': {
                'values': np.linspace(0, 200e-9, 31),
                'unit': 's'}}
    label = f'Ramsey_flux_pulse-{qb.name}'


    if qb_freq is None:
        qb_freq = qb.ge_freq()

    with temporary_value(qb.ge_freq, qb_freq):
        qb.prepare(drive='timedomain')
        MC = qb.instr_mc.get_instr()

        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                qb.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None

        if prep_params is None:
            prep_params = qb.preparation_params()

        operation_dict = get_operation_dict([qb])

        seq, hard_sweep_points = fsqs.Ramsey_time_with_flux_seq(qb.name,
                                                                hard_sweep_params,
                                                                operation_dict,
                                                                cz_pulse_name,
                                                                artificial_detunings=artificial_detunings,
                                                                cal_points=cp,
                                                                upload=False,
                                                                prep_params=prep_params)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload,
            parameter_name=list(hard_sweep_params)[0],
            unit=list(hard_sweep_params.values())[0]['unit']))
        MC.set_sweep_points(hard_sweep_points)
        MC.set_detector_function(qb.int_avg_classif_det if classified
                                 else qb.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {qb.name: hard_sweep_params[
            'Delay']['values']},
                             'sweep_name': 'Delay',
                             'sweep_unit': 's',
                             'artificial_detuning': artificial_detunings,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': True,
                             'cal_states_rotations': {qb.name: []},
                             'data_to_fit': {qb.name: 'pe'},
                            })
        MC.run(label, exp_metadata=exp_metadata)

    ramsey_ana = None
    if analyze:
        ramsey_ana = tda.RamseyAnalysis(
            qb_names=[qb.name], options_dict=dict(
                fit_gaussian_decay=True))
        new_qubit_freq = ramsey_ana.proc_data_dict[
            'analysis_params_dict'][qb.name]['exp_decay_' + qb.name][
            'new_qb_freq']
        T2_star = ramsey_ana.proc_data_dict[
            'analysis_params_dict'][qb.name]['exp_decay_' + qb.name][
            'T2_star']
    return new_qubit_freq, T2_star


def measure_cz_bleed_through(qb, CZ_separation_times, phases, CZ_pulse_name,
                             label=None, upload=True, cal_points=True,
                             oneCZ_msmt=False, soft_avgs=1, nr_cz_gates=1,
                             TwoD=True, analyze=True, MC=None):
    if MC is None:
        MC = qb.MC

    exp_metadata = {'CZ_pulse_name': CZ_pulse_name,
                    'cal_points': cal_points,
                    'oneCZ_msmt': oneCZ_msmt,
                    'nr_cz_gates': nr_cz_gates}

    qb.prepare_for_timedomain()

    if cal_points:
        step = np.abs(phases[-1] - phases[-2])
        phases = np.concatenate(
            [phases, [phases[-1] + step, phases[-1] + 2 * step,
                      phases[-1] + 3 * step, phases[-1] + 4 * step]])

    operation_dict = qb.get_operation_dict()

    s1 = awg_swf.CZ_bleed_through_phase_hard_sweep(
        qb_name=qb.name,
        CZ_pulse_name=CZ_pulse_name,
        CZ_separation=CZ_separation_times[0],
        operation_dict=operation_dict,
        oneCZ_msmt=oneCZ_msmt,
        nr_cz_gates=nr_cz_gates,
        verbose=False,
        upload=(False if TwoD else upload),
        return_seq=False,
        cal_points=cal_points)

    MC.set_sweep_function(s1)
    MC.set_sweep_points(phases)
    if TwoD:
        if len(CZ_separation_times) != 1:
            s2 = awg_swf.CZ_bleed_through_separation_time_soft_sweep(
                s1, upload=upload)
            MC.set_sweep_function_2D(s2)
            MC.set_sweep_points_2D(CZ_separation_times)
        elif nr_cz_gates > 1:
            exp_metadata['CZ_separation_time'] = CZ_separation_times[0]
            s2 = awg_swf.CZ_bleed_through_nr_cz_gates_soft_sweep(
                s1, upload=upload)
            MC.set_sweep_function_2D(s2)
            MC.set_sweep_points_2D(1 + np.arange(0, nr_cz_gates, 2))
    MC.set_detector_function(qb.int_avg_det)
    MC.soft_avg(soft_avgs)

    if label is None:
        idx = CZ_pulse_name.index('q')
        if oneCZ_msmt:
            label = 'CZ_phase_msmt_{}{}'.format(CZ_pulse_name[idx:idx + 3],
                                                qb.msmt_suffix)
        else:
            label = str(nr_cz_gates)
            label += 'CZ_bleed_through_{}{}'.format(CZ_pulse_name[idx:idx + 3],
                                                    qb.msmt_suffix)
    MC.run(label, mode=('2D' if TwoD else '1D'), exp_metadata=exp_metadata)

    if analyze:
        tda.MultiQubit_TimeDomain_Analysis(qb_names=[qb.name],
                                           options_dict={'TwoD': TwoD})


def measure_ramsey_add_pulse(measured_qubit, pulsed_qubit, times=None,
                             artificial_detuning=0, label='', analyze=True,
                             cal_states="auto", n_cal_points_per_state=2,
                             n=1, upload=True,  last_ge_pulse=False, for_ef=False,
                             classified_ro=False, prep_params=None,
                             exp_metadata=None):
    if times is None:
        raise ValueError("Unspecified times for measure_ramsey")
    if artificial_detuning is None:
        log.warning('Artificial detuning is 0.')
    if np.abs(artificial_detuning) < 1e3:
        log.warning('The artificial detuning is too small. The units'
                    'should be Hz.')
    if np.any(times > 1e-3):
        log.warning('The values in the times array might be too large.'
                    'The units should be seconds.')

    for qb in [pulsed_qubit, measured_qubit]:
        qb.prepare(drive='timedomain')
    MC = measured_qubit.instr_mc.get_instr()
    if prep_params is None:
        prep_params = measured_qubit.preparation_params()

    # Define the measurement label
    if label == '':
        label = 'Ramsey_add_pulse_{}'.format(pulsed_qubit.name) + \
                measured_qubit.msmt_suffix

    # create cal points
    cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
    cp = CalibrationPoints.single_qubit(measured_qubit.name, cal_states,
                                        n_per_state=n_cal_points_per_state)
    # create sequence
    seq, sweep_points = mqs.ramsey_add_pulse_seq_active_reset(
        times=times, measured_qubit_name=measured_qubit.name,
        pulsed_qubit_name=pulsed_qubit.name,
        operation_dict=get_operation_dict([measured_qubit, pulsed_qubit]),
        cal_points=cp, n=n, artificial_detunings=artificial_detuning,
        upload=False, for_ef=for_ef, last_ge_pulse=False, prep_params=prep_params)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload, parameter_name='Delay', unit='s'))
    MC.set_sweep_points(sweep_points)

    MC.set_detector_function(
        measured_qubit.int_avg_classif_det if classified_ro else
        measured_qubit.int_avg_det)

    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update(
        {'sweep_points_dict': {measured_qubit.name: times},
         'sweep_name': 'Delay',
         'sweep_unit': 's',
         'cal_points': repr(cp),
         'preparation_params': prep_params,
         'last_ge_pulses': [last_ge_pulse],
         'artificial_detuning': artificial_detuning,
         'rotate': len(cp.states) != 0,
         'data_to_fit': {measured_qubit.name: 'pf' if for_ef else 'pe'},
         'measured_qubit': measured_qubit.name,
         'pulsed_qubit': pulsed_qubit.name})

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        tda.RamseyAddPulseAnalysis(qb_names=[measured_qubit.name])


def measure_ramsey_add_pulse_sweep_phase(
        measured_qubit, pulsed_qubit, phases,
        interleave=True, label='', MC=None,
        analyze=True, close_fig=True,
        cal_points=True, upload=True):
    for qb in [pulsed_qubit, measured_qubit]:
        qb.prepare_for_timedomain()
    if MC is None:
        MC = measured_qubit.MC

    # Define the measurement label
    if label == '':
        label = 'Ramsey_add_pulse_{}_Sweep_phases'.format(pulsed_qubit.name) + \
                measured_qubit.msmt_suffix

    step = np.abs(phases[1] - phases[0])
    if interleave:
        phases = np.repeat(phases, 2)

    if cal_points:
        sweep_points = np.concatenate(
            [phases, [phases[-1] + step, phases[-1] + 2 * step,
                      phases[-1] + 3 * step, phases[-1] + 4 * step]])
    else:
        sweep_points = phases

    Rams_swf = awg_swf2.Ramsey_add_pulse_sweep_phase_swf(
        measured_qubit_name=measured_qubit.name,
        pulsed_qubit_name=pulsed_qubit.name,
        operation_dict=get_operation_dict([measured_qubit, pulsed_qubit]),
        cal_points=cal_points,
        upload=upload)
    MC.set_sweep_function(Rams_swf)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(measured_qubit.int_avg_det)
    MC.run(label, exp_metadata={'measured_qubit': measured_qubit.name,
                                'pulsed_qubit': pulsed_qubit.name})

    if analyze:
        ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                               qb_name=measured_qubit.name)


def measure_pygsti(qubits, f_LO, pygsti_gateset=None,
                   upload=True, nr_shots_per_seg=2 ** 12,
                   thresholded=True, analyze_shots=True, analyze_pygsti=True,
                   preselection=True, ro_spacing=1e-6, label=None,
                   MC=None, UHFQC=None, pulsar=None, run=True, **kw):
    if UHFQC is None:
        UHFQC = qubits[0].UHFQC
        log.warning("Unspecified UHFQC instrument. Using {}.UHFQC.".format(
            qubits[0].name))
    if pulsar is None:
        pulsar = qubits[0].AWG
        log.warning("Unspecified pulsar instrument. Using {}.AWG.".format(
            qubits[0].name))
    if MC is None:
        MC = qubits[0].MC
        log.warning("Unspecified MC object. Using {}.MC.".format(
            qubits[0].name))

    if len(qubits) == 2:
        log.warning('Make sure the first qubit in the list is the '
                    'control qubit!')
    # Generate list of experiments with pyGSTi
    qb_names = [qb.name for qb in qubits]

    maxLengths = kw.pop('maxLengths', [1, 2])
    linear_GST = kw.pop('linear_GST', True)
    if pygsti_gateset is not None:
        prep_fiducials = pygsti_gateset.prepStrs
        meas_fiducials = pygsti_gateset.effectStrs
        germs = pygsti_gateset.germs
        gs_target = pygsti_gateset.gs_target
        if linear_GST:
            listOfExperiments = pygsti.construction.list_lgst_gatestrings(
                prep_fiducials, meas_fiducials, gs_target)
        else:
            listOfExperiments = constr.make_lsgst_experiment_list(
                gs_target, prep_fiducials, meas_fiducials, germs, maxLengths)
    else:
        prep_fiducials = kw.pop('prep_fiducials', None)
        meas_fiducials = kw.pop('meas_fiducials', None)
        germs = kw.pop('germs', None)
        gs_target = kw.pop('gs_target', None)
        listOfExperiments = kw.pop('listOfExperiments', None)
        # if np.any(np.array([prep_fiducials, meas_fiducials, germs,
        #                     gs_target]) == None):
        #     raise ValueError('Please provide either pyGSTi gate set or the '
        #                      'kwargs "prep_fiducials", "meas_fiducials", '
        #                      '"germs", "gs_target".')
        # listOfExperiments = constr.make_lsgst_experiment_list(
        #     gs_target, prep_fiducials, meas_fiducials, germs, maxLengths)

    nr_exp = len(listOfExperiments)

    # Set label
    if label is None:
        label = ''
        if pygsti_gateset is not None:
            if linear_GST:
                label += 'Linear'
            else:
                label += 'LongSeq'
        if len(qubits) == 1:
            label += 'GST_{}{}'.format(
                '-'.join([s[1::] for s in gs_target.gates]),
                qubits[0].msmt_suffix)
        else:
            label += 'GST_{}_qbs{}'.format(
                '-'.join([s[1::] for s in gs_target.gates]),
                ''.join([qb.name[-1] for qb in qubits]))

    # Set detector function
    key = 'int'
    if thresholded:
        key = 'dig'
        log.warning('This is a thresholded measurement. Make sure you '
              'have set the threshold values!')
        label += '_thresh'

    # Prepare qubits and readout pulse
    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)
    multiplexed_pulse(qubits, f_LO, upload=True)

    MC_run_mode = '1D'
    # Check if there are too many experiments to do
    max_exp_len = kw.pop('max_exp_len', 800)
    if nr_exp > max_exp_len:
        nr_subexp = nr_exp // max_exp_len
        pygsti_sublistOfExperiments = [listOfExperiments[
                                       i * max_exp_len:(i + 1) * max_exp_len] for
                                       i in range(nr_subexp)]
        remaining_exps = nr_exp - max_exp_len * nr_subexp
        if remaining_exps > 0:
            pygsti_sublistOfExperiments += [listOfExperiments[-remaining_exps::]]
        # Set detector function
        nr_shots = nr_shots_per_seg * max_exp_len * (2 if preselection else 1)
        det_func = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=nr_shots, values_per_point=2,
            values_per_point_suffex=['_presel', '_measure'])[key + '_log_det']

        # Define hard sweep
        # hard_sweep_points = np.repeat(np.arange(max_exp_len),
        #                             nr_shots_per_seg*(2 if preselection else 1))
        hard_sweep_points = np.arange(max_exp_len * nr_shots_per_seg *
                                      (2 if preselection else 1))

        hard_sweep_func = \
            awg_swf2.GST_swf(qb_names,
                             pygsti_listOfExperiments=
                             pygsti_sublistOfExperiments[0],
                             operation_dict=get_operation_dict(qubits),
                             preselection=preselection,
                             ro_spacing=ro_spacing,
                             upload=False)

        # Define hard sweep
        soft_sweep_points = np.arange(len(pygsti_sublistOfExperiments))
        soft_sweep_func = awg_swf2.GST_experiment_sublist_swf(
            hard_sweep_func,
            pygsti_sublistOfExperiments)
        MC_run_mode = '2D'
    else:
        # Set detector function
        nr_shots = nr_shots_per_seg * nr_exp * (2 if preselection else 1)
        det_func = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=nr_shots, values_per_point=2,
            values_per_point_suffex=['_presel', '_measure'])[key + '_log_det']
        # Define hard sweep
        # hard_sweep_points = np.repeat(np.arange(nr_exp),
        #                             nr_shots_per_seg*(2 if preselection else 1))
        hard_sweep_points = np.arange(max_exp_len * nr_shots_per_seg *
                                      (2 if preselection else 1))
        hard_sweep_func = \
            awg_swf2.GST_swf(qb_names,
                             pygsti_listOfExperiments=listOfExperiments,
                             operation_dict=get_operation_dict(qubits),
                             preselection=preselection,
                             ro_spacing=ro_spacing,
                             upload=upload)

    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)
    if MC_run_mode == '2D':
        MC.set_sweep_function_2D(soft_sweep_func)
        MC.set_sweep_points_2D(soft_sweep_points)
    MC.set_detector_function(det_func)

    exp_metadata = {'pygsti_gateset': pygsti_gateset,
                    'linear_GST': linear_GST,
                    'preselection': preselection,
                    'thresholded': thresholded,
                    'nr_shots_per_seg': nr_shots_per_seg,
                    'nr_exp': nr_exp}
    reduction_type = kw.pop('reduction_type', None)
    if reduction_type is not None:
        exp_metadata.update({'reduction_type': reduction_type})
    if nr_exp > max_exp_len:
        exp_metadata.update({'max_exp_len': max_exp_len})
    if not linear_GST:
        exp_metadata.update({'maxLengths': maxLengths})
    if preselection:
        exp_metadata.update({'ro_spacing': ro_spacing})
    if run:
        MC.run(name=label, mode=MC_run_mode, exp_metadata=exp_metadata)

    # Analysis
    if analyze_shots:
        if thresholded:
            MA = ma.MeasurementAnalysis(TwoD=(MC_run_mode == '2D'))
        else:
            thresholds = {qb.name: 1.5 * UHFQC.get(
                'qas_0_thresholds_{}_level'.format(
                    qb.RO_acq_weight_function_I())) for qb in qubits}
            channel_map = {qb.name: det_func.value_names[0] for qb in qubits}
            MA = ra.MultiQubit_SingleShot_Analysis(options_dict=dict(
                TwoD=(MC_run_mode == '2D'),
                n_readouts=(2 if preselection else 1) * nr_exp,
                thresholds=thresholds,
                channel_map=channel_map
            ))

        if analyze_pygsti:
            # Create experiment dataset
            basis_states = [''.join(s) for s in
                            list(itertools.product(['0', '1'],
                                                   repeat=len(qubits)))]
            dataset = pygsti.objects.DataSet(outcomeLabels=basis_states)
            if thresholded:
                if len(qubits) == 1:
                    shots = MA.measured_values[0]
                    if preselection:
                        shots = shots[1::2]
                    for i, gs in enumerate(listOfExperiments):
                        gs_shots = shots[i::nr_exp]
                        dataset[gs] = {'0': len(gs_shots[gs_shots == 0]),
                                       '1': len(gs_shots[gs_shots == 1])}
            else:
                nr_shots_MA = len(MA.proc_data_dict[
                                      'shots_thresholded'][qb_names[0]])
                shots = MA.proc_data_dict['probability_table'] * nr_shots_MA
                if preselection:
                    shots = shots[1::2]
                for i, gs in enumerate(listOfExperiments):
                    for j, state in enumerate(basis_states):
                        dataset[gs].update({basis_states[j]: shots[i, j]})

            dataset.done_adding_data()
            # Get results
            if linear_GST:
                results = pygsti.do_linear_gst(dataset, gs_target,
                                               prep_fiducials, meas_fiducials,
                                               verbosity=3)
            else:
                results = pygsti.do_long_sequence_gst(dataset, gs_target,
                                                      prep_fiducials,
                                                      meas_fiducials,
                                                      germs, maxLengths,
                                                      verbosity=3)
            # Save analysis report
            filename = os.path.abspath(os.path.join(
                MA.folder, label))
            pygsti.report.create_standard_report(
                results,
                filename=filename,
                title=label, verbosity=2)

    return MC


def measure_multi_parity_multi_round(ancilla_qubits, data_qubits,
                                     parity_map, CZ_map,
                                     prep=None, upload=True, prep_params=None,
                                     mode='tomo',
                                     parity_seperation=1100e-9,
                                     rots_basis=('I', 'Y90', 'X90'),
                                     parity_loops = 1,
                                     cal_points=None, analyze=True,
                                     exp_metadata=None, label=None,
                                     detector='int_log_det'):
    """

    :param ancilla_qubit:
    :param data_qubits:
    :param CZ_map: example:
        {'CZ qb1 qb2': ['Y90 qb1', 'CX qb1 qb2', 'mY90 qb1'],
         'CZ qb3 qb4': ['CZ qb4 qb3']}
    :param preps:
    :param upload:
    :param prep_params:
    :param cal_points:
    :param analyze:
    :param exp_metadata:
    :param label:
    :param detector:
    :return:
    """

    qubits = ancilla_qubits + data_qubits
    qb_names = [qb.name for qb in qubits]
    for qb in qubits:
        qb.prepare(drive='timedomain')

    if label is None:
        label = 'S7-rounds_' + str(parity_loops) + '_' + '-'.join(rots_basis) + \
                '_' + '-'.join([qb.name for qb in qubits])

    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(
            [qb.preparation_params() for qb in qubits])

    # if cal_points is None:
    #     cal_points = CalibrationPoints.multi_qubit(qb_names, 'ge')

    if prep is None:
        prep = 'g'*len(data_qubits)

    MC = ancilla_qubits[0].instr_mc.get_instr()

    seq, sweep_points = mqs.multi_parity_multi_round_seq(
                                 [qb.name for qb in ancilla_qubits],
                                 [qb.name for qb in data_qubits],
                                 parity_map,
                                 CZ_map,
                                 prep,
                                 operation_dict=get_operation_dict(qubits),
                                 mode=mode,
                                 parity_seperation=parity_seperation,
                                 rots_basis=rots_basis,
                                 parity_loops=parity_loops,
                                 cal_points=cal_points,
                                 prep_params=prep_params,
                                 upload=upload)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
        sequence=seq, upload=False, parameter_name='Tomography'))
    MC.set_sweep_points(sweep_points)

    rounds = 0
    for k in range(len(parity_map)):
        if parity_map[k]['round'] > rounds:
            rounds = parity_map[k]['round']
    rounds += 1

    MC.set_detector_function(
        get_multiplexed_readout_detector_functions(
            qubits,
            nr_averages=ancilla_qubits[0].acq_averages(),
            nr_shots=ancilla_qubits[0].acq_shots(),
        )[detector])
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update(
        {'sweep_name': 'Tomography',
         'preparation_params': prep_params,
         'hard_sweep_params': {'tomo': {'values': np.arange(0, len(sweep_points)),
                                         'unit': ''}},
         'parity_map': parity_map
         })

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        tda.MultiQubit_TimeDomain_Analysis(qb_names=qb_names)


def measure_ro_dynamic_phases(pulsed_qubit, measured_qubits,
                              hard_sweep_params=None, exp_metadata=None,
                              pulse_separation=None, init_state='g',
                              upload=True, n_cal_points_per_state=1,
                              cal_states='auto', classified=False,
                              prep_params=None):
    if not hasattr(measured_qubits, '__iter__'):
        measured_qubits = [measured_qubits]
    qubits = measured_qubits + [pulsed_qubit]

    if pulse_separation is None:
        pulse_separation = max([qb.acq_length() for qb in qubits])

    for qb in qubits:
        MC = qb.instr_mc.get_instr()
        qb.prepare(drive='timedomain')

    if hard_sweep_params is None:
        hard_sweep_params = {
            'phase': {'values': np.tile(np.linspace(0, 2*np.pi, 6)*180/np.pi, 2),
                      'unit': 'deg'}
        }

    cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                    for_ef=False)
    cp = CalibrationPoints.multi_qubit([qb.name for qb in qubits], cal_states,
                                       n_per_state=n_cal_points_per_state)

    if prep_params is None:
        prep_params = measured_qubits[0].preparation_params()
    operation_dict = get_operation_dict(qubits)
    sequences, hard_sweep_points = \
        mqs.ro_dynamic_phase_seq(
            hard_sweep_dict=hard_sweep_params,
            qbp_name=pulsed_qubit.name, qbr_names=[qb.name for qb in
                                                   measured_qubits],
            operation_dict=operation_dict,
            pulse_separation=pulse_separation,
            init_state = init_state, prep_params=prep_params,
            cal_points=cp, upload=False)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=sequences, upload=upload,
        parameter_name=list(hard_sweep_params)[0],
        unit=list(hard_sweep_params.values())[0]['unit'])
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)

    det_name = 'int_avg{}_det'.format('_classif' if classified else '')
    det_func = get_multiplexed_readout_detector_functions(
        measured_qubits, nr_averages=max(qb.acq_averages() for qb in measured_qubits)
    )[det_name]
    MC.set_detector_function(det_func)

    if exp_metadata is None:
        exp_metadata = {}

    hard_sweep_params = {
        'phase': {'values': np.repeat(np.tile(np.linspace(0, 2 * np.pi, 6) * 180 /
                                         np.pi,
                                     2), 2),
                  'unit': 'deg'}
    }
    exp_metadata.update({'qbnames': [qb.name for qb in qubits],
                         'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'rotate': len(cal_states) != 0,
                         'cal_states_rotations':
                             {qb.name: {'g': 0, 'e': 1} for qb in qubits} if
                             len(cal_states) != 0 else None,
                         'data_to_fit': {qb.name: 'pe' for qb in qubits},
                         'hard_sweep_params': hard_sweep_params})
    MC.run('RO_DynamicPhase_{}{}'.format(
        pulsed_qubit.name, ''.join([qb.name for qb in qubits])),
        exp_metadata=exp_metadata)
    tda.MultiQubit_TimeDomain_Analysis(qb_names=[qb.name for qb in measured_qubits])