import numpy as np
import matplotlib.pyplot as plt
import logging
import itertools
import time
import copy
import datetime
import os
import lmfit

import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.detector_functions as det
import pycqed.measurement.composite_detector_functions as cdet
import pycqed.analysis.measurement_analysis as ma
import pycqed.analysis.randomized_benchmarking_analysis as rbma
import pycqed.analysis_v2.readout_analysis as ra
import pycqed.analysis.tomography as tomo
from pycqed.measurement.optimization import nelder_mead, \
                                            generate_new_training_set
import qcodes as qc
station = qc.station

def measure_two_qubit_AllXY(device, q0_name, q1_name,
                            sequence_type='sequential', MC=None):
    if MC is None:
        MC = qc.station.components['MC']
    q0 = station.components[q0_name]
    q1 = station.components[q1_name]
    # AllXY on Data top
    # FIXME: multiplexed RO should come from the device
    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=q0._acquisition_instr, AWG=q0.AWG,
        channels=[q1.RO_acq_weight_function_I(),
                  q0.RO_acq_weight_function_I()],
        nr_averages=q0.RO_acq_averages(),
        integration_length=q0.RO_acq_integration_length(),
        result_logging_mode='lin_trans')

    operation_dict = device.get_operation_dict()

    two_qubit_AllXY_sweep = awg_swf.awg_seq_swf(
        mqs.two_qubit_AllXY,
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'q0': q0_name,
                             'q1': q1_name,
                             'RO_target': q0_name,
                             'sequence_type': sequence_type})

    MC.set_sweep_function(two_qubit_AllXY_sweep)
    MC.set_sweep_points(np.arange(42))
    MC.set_detector_function(int_avg_det)
    MC.run('2 qubit AllXY sequential')
    ma.MeasurementAnalysis()


def measure_SWAP_CZ_SWAP(device, qS_name, qCZ_name,
                         CZ_phase_corr_amps,
                         sweep_qubit,
                         excitations='both_cases',
                         MC=None, upload=True):
    if MC is None:
        MC = station.components['MC']
    if excitations == 'both_cases':
        CZ_phase_corr_amps = np.tile(CZ_phase_corr_amps, 2)
    amp_step = CZ_phase_corr_amps[1]-CZ_phase_corr_amps[0]
    swp_pts = np.concatenate([CZ_phase_corr_amps,
                              np.arange(4)*amp_step+CZ_phase_corr_amps[-1]])

    qS = device.qubits()[qS_name]
    qCZ = device.qubits()[qCZ_name]

    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=qS._acquisition_instr, AWG=qS.AWG,
        channels=[qCZ.RO_acq_weight_function_I(),
                  qS.RO_acq_weight_function_I()],
        nr_averages=qS.RO_acq_averages(),
        integration_length=qS.RO_acq_integration_length(),
        result_logging_mode='lin_trans')
    operation_dict = device.get_operation_dict()
    S_CZ_S_swf = awg_swf.awg_seq_swf(
        fsqs.SWAP_CZ_SWAP_phase_corr_swp,
        parameter_name='phase_corr_amps',
        unit='V',
        AWG=qS.AWG,
        fluxing_channels=[qS.fluxing_channel(), qCZ.fluxing_channel()],
        upload=upload,
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'qS': qS.name,
                             'qCZ': qCZ.name,
                             'sweep_qubit': sweep_qubit,
                             'RO_target': qCZ.name,
                             'excitations': excitations,
                             'upload': upload,
                             'distortion_dict': qS.dist_dict()})

    MC.set_sweep_function(S_CZ_S_swf)
    MC.set_detector_function(int_avg_det)
    MC.set_sweep_points(swp_pts)
    # MC.set_sweep_points(phases)

    MC.run('SWAP_CP_SWAP_{}_{}'.format(qS_name, qCZ_name))
    ma.MeasurementAnalysis()  # N.B. you may want to run different analysis


def resonant_cphase(phases, low_qubit, high_qubit,
                    timings_dict, mmt_label='', MC=None, run=True):
    """
    Performs the fringe measurements of a resonant cphase gate between two qubits.
    low_qubit is gonna be swapped with the bus
    high_qubit is gonna be adiabatically pulsed
    """
    if MC is None:
        MC = station.MC
    cal_points = 8
    lengths_cal = phases[-1] + np.arange(1, 1+cal_points)*(phases[1]-phases[0])
    lengths_vec = np.concatenate((np.repeat(phases, 2), lengths_cal))

    q0_pulse_pars, RO_pars = low_qubit.get_pulse_pars()
    q1_pulse_pars, RO_pars = high_qubit.get_pulse_pars()
    swap_pars_q0 = low_qubit.get_flux_pars()[0]
    cphase_pars_q1 = high_qubit._cphase_pars
    swap_pars_q0.update({'pulse_type': 'SquarePulse'})
    # print(phases)
    cphase = awg_swf.cphase_fringes(phases=phases,
                                    q0_pulse_pars=q0_pulse_pars,
                                    q1_pulse_pars=q1_pulse_pars,
                                    RO_pars=RO_pars,
                                    swap_pars_q0=swap_pars_q0,
                                    cphase_pars_q1=cphase_pars_q1,
                                    timings_dict=timings_dict,
                                    dist_dict=dist_dict,
                                    upload=False,
                                    return_seq=True)

    station.AWG.ch3_amp(2.)
    station.AWG.ch4_amp(2.)
    cphase.pre_upload()

    MC.set_sweep_function(cphase)
    MC.set_sweep_points(lengths_vec)

    p = 'ch%d_amp' % high_qubit.fluxing_channel()
    station.AWG.set(p, high_qubit.SWAP_amp())
    p = 'ch%d_amp' % low_qubit.fluxing_channel()
    station.AWG.set(p, low_qubit.SWAP_amp())
    MC.set_detector_function(high_qubit.int_avg_det)
    if run:
        MC.run('CPHASE_Fringes_%s_%s_%s' %
               (low_qubit.name, high_qubit.name, mmt_label))
        # ma.TD_Analysis(auto=True,label='CPHASE_Fringes')
    return cphase.seq


def tomo2Q_cardinal(cardinal, qubit0, qubit1, timings_dict,
                    nr_shots=512, nr_rep=10, mmt_label='', MC=None, run=True):
    """
    Performs the fringe measurements of a resonant cphase gate between two qubits.
    low_qubit is gonna be swapped with the bus
    high_qubit is gonna be adiabatically pulsed
    """
    if MC is None:
        MC = station.MC
    cal_points = 28
    # sweep_points = np.arange(cal_points+36)
    sweep_points = np.arange(nr_shots*nr_rep*(36+cal_points))

    q0_pulse_pars, RO_pars = qubit0.get_pulse_pars()
    q1_pulse_pars, RO_pars = qubit1.get_pulse_pars()
    # print(phases)
    tomo = awg_swf.two_qubit_tomo_cardinal(cardinal=cardinal,
                                           q0_pulse_pars=q0_pulse_pars,
                                           q1_pulse_pars=q1_pulse_pars,
                                           RO_pars=RO_pars,
                                           timings_dict=timings_dict,
                                           upload=True,
                                           return_seq=False)

    # detector = det.UHFQC_integrated_average_detector(
    #     UHFQC=qubit0._acquisition_instr,
    #     AWG=station.AWG,
    #     channels=[qubit0.RO_acq_weight_function_I(),
    #               qubit1.RO_acq_weight_function_I()],
    #     nr_averages=qubit0.RO_acq_averages(),
    #     integration_length=qubit0.RO_acq_integration_length(),
    #     result_logging_mode='lin_trans')

    detector = det.UHFQC_integration_logging_det(
        UHFQC=qubit0._acquisition_instr,
        AWG=station.AWG,
        channels=[qubit0.RO_acq_weight_function_I(),
                  qubit1.RO_acq_weight_function_I()],
        nr_shots=256,
        integration_length=qubit0.RO_acq_integration_length(),
        result_logging_mode='lin_trans')

    MC.set_sweep_function(tomo)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(detector)
    if run:
        MC.run('Tomo_%s_%s_%s_%s' % (cardinal,
                                     qubit0.name,
                                     qubit1.name,
                                     mmt_label))
    return tomo.seq


def tomo2Q_bell(bell_state, device, qS_name, qCZ_name, CPhase=True,
                nr_shots=256, nr_rep=1, mmt_label='',
                MLE=False,
                MC=None, run=True):
    """
    Performs the fringe measurements of a resonant cphase gate between two qubits.
    low_qubit is gonna be swapped with the bus
    high_qubit is gonna be adiabatically pulsed
    """
    if MC is None:
        MC = station.MC
    cal_points = 28
    sweep_points = np.arange(nr_shots*nr_rep*(36+cal_points))

    operation_dict = device.get_operation_dict()
    qS = device.qubits()[qS_name]
    qCZ = device.qubits()[qCZ_name]

    detector = det.UHFQC_integration_logging_det(
        UHFQC=qS._acquisition_instr,
        AWG=qS.AWG,
        channels=[qCZ.RO_acq_weight_function_I(),
                  qS.RO_acq_weight_function_I()],
        nr_shots=nr_shots,
        integration_length=qS.RO_acq_integration_length(),
        result_logging_mode='lin_trans')

    tomo_swf = awg_swf.awg_seq_swf(
        mqs.two_qubit_tomo_bell,
        # parameter_name='Pre-rotation',
        AWG=qS.AWG,
        fluxing_channels=[qS.fluxing_channel(), qCZ.fluxing_channel()],
        awg_seq_func_kwargs={'bell_state': bell_state,
                             'operation_dict': operation_dict,
                             'qS': qS.name,
                             'qCZ': qCZ.name,
                             'RO_target': qCZ.name,
                             'distortion_dict': qS.dist_dict()})
    MC.soft_avg(1) # Single shots cannot be averaged.
    MC.set_sweep_function(tomo_swf)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(detector)
    if run:
        MC.run('BellTomo_%s_%s_%s_%s' % (bell_state,
                                         qS_name,
                                         qCZ_name,
                                         mmt_label))
    tomo.analyse_tomo(MLE=MLE, target_bell=bell_state%10)
    # return tomo_swf.seq


def rSWAP_scan(device, qS_name, qCZ_name,
               recovery_swap_amps, emulate_cross_driving=False,
               MC=None, upload=True):
    if MC is None:
        MC = station.components['MC']
    amp_step = recovery_swap_amps[1]-recovery_swap_amps[0]
    swp_pts = np.concatenate([recovery_swap_amps,
                              np.arange(4)*amp_step+recovery_swap_amps[-1]])

    qS = device.qubits()[qS_name]
    qCZ = device.qubits()[qCZ_name]

    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=qS._acquisition_instr, AWG=qS.AWG,
        channels=[qCZ.RO_acq_weight_function_I(),
                  qS.RO_acq_weight_function_I()],
        nr_averages=qS.RO_acq_averages(),
        integration_length=qS.RO_acq_integration_length(),
        result_logging_mode='lin_trans')
    operation_dict = device.get_operation_dict()
    rSWAP_swf = awg_swf.awg_seq_swf(
        fsqs.rSWAP_amp_sweep,
        parameter_name='rSWAP amplitude',
        unit=r'% dac resolution',
        AWG=qS.AWG,
        fluxing_channels=[qS.fluxing_channel(), qCZ.fluxing_channel()],
        upload=upload,
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'qS': qS.name,
                             'qCZ': qCZ.name,
                             'recovery_swap_amps': swp_pts,
                             'RO_target': qCZ.name,
                             'emulate_cross_driving': emulate_cross_driving,
                             'upload': upload,
                             'distortion_dict': qS.dist_dict()})

    MC.set_sweep_function(rSWAP_swf)
    MC.set_detector_function(int_avg_det)
    MC.set_sweep_points(swp_pts)
    # MC.set_sweep_points(phases)

    MC.run('rSWAP_{}_{}'.format(qS_name, qCZ_name))
    ma.MeasurementAnalysis()  # N.B. you may want to run different analysis

def tomo2Q_cphase_cardinal(cardinal_state, device, qS_name, qCZ_name, CPhase=True,
                nr_shots=256, nr_rep=1, mmt_label='',
                MLE=False,
                MC=None, run=True):
    """
    Performs the fringe measurements of a resonant cphase gate between two qubits.
    low_qubit is gonna be swapped with the bus
    high_qubit is gonna be adiabatically pulsed
    """
    if MC is None:
        MC = station.MC
    cal_points = 28
    sweep_points = np.arange(nr_shots*nr_rep*(36+cal_points))

    operation_dict = device.get_operation_dict()
    qS = device.qubits()[qS_name]
    qCZ = device.qubits()[qCZ_name]

    detector = det.UHFQC_integration_logging_det(
        UHFQC=qS._acquisition_instr,
        AWG=qS.AWG,
        channels=[qCZ.RO_acq_weight_function_I(),
                  qS.RO_acq_weight_function_I()],
        nr_shots=nr_shots,
        integration_length=qS.RO_acq_integration_length(),
        result_logging_mode='lin_trans')

    tomo_swf = awg_swf.awg_seq_swf(
        mqs.two_qubit_tomo_cphase_cardinal,
        # parameter_name='Pre-rotation',
        AWG=qS.AWG,
        fluxing_channels=[qS.fluxing_channel(), qCZ.fluxing_channel()],
        awg_seq_func_kwargs={'cardinal_state': cardinal_state,
                             'operation_dict': operation_dict,
                             'qS': qS.name,
                             'qCZ': qCZ.name,
                             'RO_target': qCZ.name,
                             'distortion_dict': qS.dist_dict()})
    MC.soft_avg(1) # Single shots cannot be averaged.
    MC.set_sweep_function(tomo_swf)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(detector)
    if run:
        MC.run('CPhaseTomo_%s_%s_%s_%s' % (cardinal_state,
                                         qS_name,
                                         qCZ_name,
                                         mmt_label))
    # return tomo_swf.seq



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
            #qb.RO_pulse_type('Multiplexed_pulse_UHFQC')
            qb.f_RO_mod(qb.f_RO() - f_LO)
            samples = int(qb.RO_pulse_length() * fs)
            pulse = qb.RO_amp()*np.ones(samples)

            if qb.ro_pulse_shape() == 'gaussian_filtered':
                filter_sigma = qb.ro_pulse_filter_sigma()
                nr_sigma = qb.ro_pulse_nr_sigma()
                filter_samples = int(filter_sigma*nr_sigma*fs)
                filter_sample_idxs = np.arange(filter_samples)
                filter = np.exp(-0.5*(filter_sample_idxs - filter_samples/2)**2 /
                                (filter_sigma*fs)**2)
                filter /= filter.sum()
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


def get_multiplexed_readout_pulse_dictionary(qubits):
    """Takes the readout pulse parameters from the first qubit in `qubits`"""
    maxlen = 0
    for qb in qubits:
        if qb.RO_pulse_length() > maxlen:
            maxlen = qb.RO_pulse_length()
    return {'RO_pulse_marker_channel': qubits[0].RO_acq_marker_channel(),
            'acq_marker_channel': qubits[0].RO_acq_marker_channel(),
            'acq_marker_delay': qubits[0].RO_acq_marker_delay(),
            'amplitude': 0.0,
            'length': maxlen,
            'operation_type': 'RO',
            'phase': 0,
            'pulse_delay': qubits[0].RO_pulse_delay(),
            'pulse_type': 'Multiplexed_UHFQC_pulse',
            'target_qubit': ','.join([qb.name for qb in qubits])}


def get_operation_dict(qubits):
    operation_dict = {'RO mux':
                      get_multiplexed_readout_pulse_dictionary(qubits)}
    for qb in qubits:
        operation_dict.update(qb.get_operation_dict())
    return operation_dict


def get_multiplexed_readout_detector_functions(qubits, nr_averages=2**10,
                                               nr_shots=4095, UHFQC=None,
                                               pulsar=None,
                                               used_channels=None,
                                               correlations=None,
                                               **kw):
    max_int_len = 0
    for qb in qubits:
        if qb.RO_acq_integration_length() > max_int_len:
            max_int_len = qb.RO_acq_integration_length()

    channels = []
    for qb in qubits:
        channels += [qb.RO_acq_weight_function_I()]
        if qb.ro_acq_weight_type() in ['SSB', 'DSB']:
            if qb.RO_acq_weight_function_Q() is not None:
                channels += [qb.RO_acq_weight_function_Q()]

    if correlations is None:
        correlations = []
    if used_channels is None:
        used_channels = channels

    for qb in qubits:
        if UHFQC is None:
            UHFQC = qb.UHFQC
        if pulsar is None:
            pulsar = qb.AWG
        break
    return {
        'int_log_det': det.UHFQC_integration_logging_det(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_shots=nr_shots,
            result_logging_mode='raw', **kw),
        'dig_log_det': det.UHFQC_integration_logging_det(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_shots=nr_shots,
            result_logging_mode='digitized', **kw),
        'int_avg_det': det.UHFQC_integrated_average_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_averages=nr_averages, **kw),
        'dig_avg_det': det.UHFQC_integrated_average_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_averages=nr_averages,
            result_logging_mode='digitized', **kw),
        'inp_avg_det': det.UHFQC_input_average_detector(
            UHFQC=UHFQC, AWG=pulsar, nr_averages=nr_averages, nr_samples=4096,
            **kw),
        'int_corr_det': det.UHFQC_correlation_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            used_channels=used_channels,
            integration_length=max_int_len, nr_averages=nr_averages,
            correlations=correlations, **kw),
        'dig_corr_det': det.UHFQC_correlation_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            used_channels=used_channels,
            integration_length=max_int_len, nr_averages=nr_averages,
            correlations=correlations, thresholding=True, **kw),
    }


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
            if drive_pulse_len != qb.gauss_sigma()*qb.nr_sigma() and \
                            drive_pulses != 0:
                logging.warning('Caution! Not all qubit drive pulses are the '
                                'same length. This might cause trouble in the '
                                'sequence.')
            drive_pulse_len = max(drive_pulse_len,
                                  qb.gauss_sigma()*qb.nr_sigma())
        else:
            drive_pulse_len = qb.gauss_sigma()*qb.nr_sigma()
        max_ro_len = max(max_ro_len, qb.RO_pulse_length())
        max_int_length = max(max_int_length, qb.RO_acq_integration_length())

    ro_spacing = 2 * UHFQC.quex_wint_delay() / 1.8e9
    ro_spacing += max_int_length
    ro_spacing += ro_slack
    ro_spacing -= drive_pulse_len
    ro_spacing -= max_ro_len
    return ro_spacing


def measure_multiplexed_readout(qubits, f_LO, nreps=4, liveplot=False,
                                RO_spacing=None, preselection=True, MC=None,
                                thresholds=None, thresholded=False,
                                analyse=True):

    multiplexed_pulse(qubits, f_LO, upload=True)

    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    UHFQC = qubits[0].UHFQC

    if RO_spacing is None:
        RO_spacing = UHFQC.quex_wint_delay()*2/1.8e9
        RO_spacing += UHFQC.quex_wint_length()/1.8e9
        RO_spacing += 50e-9  # for slack
        RO_spacing = np.ceil(RO_spacing*225e6/3)/225e6*3

    sf = awg_swf2.n_qubit_off_on(
        [qb.get_drive_pars() for qb in qubits],
        get_multiplexed_readout_pulse_dictionary(qubits),
        preselection=preselection,
        parallel_pulses=True,
        RO_spacing=RO_spacing)

    m = 2 ** (len(qubits))
    if preselection:
        m *= 2
    shots = 4094 - 4094 % m
    if thresholded:
        df = get_multiplexed_readout_detector_functions(qubits,
                 nr_shots=shots)['dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(qubits,
                 nr_shots=shots)['int_log_det']



    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)
    MC.run_2D('{}_multiplexed_ssro'.format('-'.join(
        [qb.name for qb in qubits])))

    if analyse and thresholds is not None:
        channel_map = {qb.name: df.value_names[0] for qb in qubits}

        ra.Multiplexed_Readout_Analysis(options_dict=dict(
            n_readouts=(2 if preselection else 1)*2**len(qubits),
            thresholds=thresholds,
            channel_map=channel_map
        ))

def measure_active_reset(qubits, reset_cycle_time, nr_resets=1, nreps=1,
                         MC=None, upload=True, sequence='reset_g'):
    """possible sequences: 'reset_g', 'reset_e', 'idle', 'flip'"""
    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    operation_dict = {
        'RO': get_multiplexed_readout_pulse_dictionary(qubits)}
    qb_names = []
    for qb in qubits:
        qb_names.append(qb.name)
        operation_dict.update(qb.get_operation_dict())

    sf = awg_swf2.n_qubit_reset(
        qubit_names=qb_names,
        operation_dict=operation_dict,
        reset_cycle_time=reset_cycle_time,
        #sequence=sequence,
        nr_resets=nr_resets,
        upload=upload)

    m = 2 ** (len(qubits))
    m *= (nr_resets + 1)
    shots = 4094 - 4094 % m
    df = get_multiplexed_readout_detector_functions(qubits,
             nr_shots=shots)['int_log_det']

    prev_avg = MC.soft_avg()
    MC.soft_avg(1)

    for qb in qubits:
        qb.prepare_for_timedomain()

    f_LO = qubits[0].f_RO() - qubits[0].f_RO_mod()
    multiplexed_pulse(qubits, f_LO)

    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)

    MC.run_2D(name='active_reset_x{}_{}'.format(nr_resets, ','.join(qb_names)))

    MC.soft_avg(prev_avg)


def measure_parity_correction(qb0, qb1, qb2, feedback_delay, f_LO, nreps=1,
                             upload=True, MC=None, prep_sequence=None,
                             nr_echo_pulses=4, cpmg_scheme=True,
                             tomography_basis=(
                                 'I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                             reset=True, preselection=False, ro_spacing=1e-6):
    """
    Important things to check when running the experiment:
        Is the readout separation commensurate with 225 MHz?
    """

    if preselection:
        multiplexed_pulse([(qb0, qb1, qb2), (qb1,), (qb0, qb1, qb2)], f_LO)
    else:
        multiplexed_pulse([(qb1,), (qb0, qb1, qb2)], f_LO)

    qubits = [qb0, qb1, qb2]
    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    sf = awg_swf2.parity_correction(qb0.name, qb1.name, qb2.name,
                                    operation_dict=get_operation_dict(qubits),
                                    feedback_delay=feedback_delay,
                                    prep_sequence=prep_sequence, reset=reset,
                                    nr_echo_pulses=nr_echo_pulses,
                                    cpmg_scheme=cpmg_scheme,
                                    tomography_basis=tomography_basis,
                                    upload=upload, verbose=False,
                                    preselection=preselection,
                                    ro_spacing=ro_spacing)

    nr_readouts = (3 if preselection else 2)*len(tomography_basis)**2
    nr_shots = 4095 - 4095 % nr_readouts
    df = get_multiplexed_readout_detector_functions(
        qubits, nr_shots=nr_shots)['int_log_det']

    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.tile(np.arange(nr_readouts)/2,
                                [nr_shots//nr_readouts]))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)

    MC.run_2D(name='two_qubit_parity{}-{}'.format(
        '' if reset else '_noreset', '_'.join([qb.name for qb in qubits])))


def measure_tomography(qubits, prep_sequence, state_name, f_LO,
                       rots_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                       use_cal_points=True,
                       preselection=True,
                       rho_target=None,
                       shots=None,
                       MC=None,
                       ro_spacing=1e-6,
                       ro_slack=10e-9,
                       thresholded=False,
                       liveplot=True,
                       nreps=1, run=True,
                       upload=True):
    exp_metadata = {}

    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    # set up multiplexed readout
    multiplexed_pulse(qubits, f_LO, upload=True)

    if ro_spacing is None:
        ro_spacing = calculate_minimal_readout_spacing(qubits, ro_slack,
                                                           drive_pulses=1)

    qubit_names = [qb.name for qb in qubits]

    seq_tomo, elts_tomo = mqs.n_qubit_tomo_seq(qubit_names,
                                               get_operation_dict(qubits),
                                               prep_sequence=prep_sequence,
                                               rots_basis=rots_basis,
                                               return_seq=True,
                                               upload=False,
                                               preselection=preselection,
                                               ro_spacing=ro_spacing)
    seq = seq_tomo
    elts = elts_tomo

    if use_cal_points:
        seq_cal, elts_cal = mqs.n_qubit_ref_all_seq(qubit_names,
                                                    get_operation_dict(qubits),
                                                    return_seq=True,
                                                    upload=False,
                                                    preselection=preselection,
                                                    ro_spacing=ro_spacing)
        seq += seq_cal
        elts += elts_cal
    n_segments = len(seq.elements)
    if preselection:
        n_segments *= 2

    # from this point on number of segments is fixed
    sf = awg_swf2.n_qubit_seq_sweep(seq_len=n_segments)
    if shots is None:
        shots = 4094 - 4094 % n_segments
    # shots = 600000

    if thresholded:
        df = get_multiplexed_readout_detector_functions(qubits,
                                            nr_shots=shots)['dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(qubits,
                                            nr_shots=shots)['int_log_det']
    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

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
                        [2*len(seq_tomo.elements) + 2*i + 1]
                else:
                    cal_defs[i][channel_map[qb.name]] = \
                        [len(seq_tomo.elements) + i]
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
        station.pulsar.program_awgs(seq, *elts)

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)
    if run:
        if preselection:
            label = '{}_tomography_ssro_preselection_{}'.format(state_name, '-'.join(
                [qb.name for qb in qubits]))
        else:
            label = '{}_tomography_ssro_{}'.format(state_name, '-'.join(
                [qb.name for qb in qubits]))

        MC.run_2D(label, exp_metadata=exp_metadata)

    return elts


def measure_two_qubit_randomized_benchmarking(qb1, qb2, f_LO,
                                              nr_cliffords_array,
                                              nr_seeds_value,
                                              CZ_pulse_name=None,
                                              net_clifford=0,
                                              nr_averages=4096,
                                              clifford_decomposition_name='HZ',
                                              interleaved_gate=None,
                                              MC=None, UHFQC=None,
                                              pulsar=None, label=None, run=True,
                                              analyze_RB=True):

    qb1n = qb1.name
    qb2n = qb2.name
    qubits = [qb1, qb2]

    if label is None:
        if interleaved_gate is None:
            label = 'RB_{}_{}_seeds_{}_cliffords_{}{}'.format(
                clifford_decomposition_name, nr_seeds_value,
                nr_cliffords_array[-1], qb1n, qb2n)
        else:
            label = 'IRB_{}_{}_{}_seeds_{}_cliffords'.format(
                interleaved_gate, clifford_decomposition_name, nr_seeds_value,
                nr_cliffords_array[-1], qb1n, qb2n)

    if UHFQC is None:
        UHFQC = qb1.UHFQC
        logging.warning("Unspecified UHFQC instrument. Using qb1.UHFQC.")
    if pulsar is None:
        pulsar = qb1.AWG
        logging.warning("Unspecified pulsar instrument. Using qb1.AWG.")
    if MC is None:
        MC = qb1.MC
        logging.warning("Unspecified MC object. Using qb1.MC.")

    if CZ_pulse_name is None:
        logging.warning('"CZ_pulse_name" is not specified. Using '
                        '"CZ_pulse_name = CZ {} {}".'.format(qb2n, qb1n))
        CZ_pulse_name = 'CZ ' + qb2n + ' ' + qb1n

    for qb in qubits:
        qb.RO_acq_averages(nr_averages)
        qb.prepare_for_timedomain(multiplexed=True)

    multiplexed_pulse(qubits, f_LO, upload=True)
    operation_dict = get_operation_dict(qubits)

    hard_sweep_points = np.arange(nr_seeds_value)
    hard_sweep_func = awg_swf2.two_qubit_randomized_benchmarking_one_length(
        qb1n=qb1n, qb2n=qb2n, operation_dict=operation_dict,
        nr_cliffords_value=nr_cliffords_array[0],
        CZ_pulse_name=CZ_pulse_name,
        net_clifford=net_clifford,
        clifford_decomposition_name=clifford_decomposition_name,
        interleaved_gate=interleaved_gate,
        upload=False)

    soft_sweep_points = nr_cliffords_array
    soft_sweep_func = awg_swf2.two_qubit_randomized_benchmarking_nr_cliffords(
        two_qubit_RB_sweepfunction=hard_sweep_func)

    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)
    MC.set_sweep_function_2D(soft_sweep_func)
    MC.set_sweep_points_2D(soft_sweep_points)

    correlations = [(qb1.RO_acq_weight_function_I(),
                     qb2.RO_acq_weight_function_I())]

    det_func = get_multiplexed_readout_detector_functions(
        qubits, nr_averages=nr_averages, UHFQC=UHFQC, pulsar=pulsar,
        correlations=correlations)['dig_corr_det']

    MC.set_detector_function(det_func)
    if run:
        MC.run(label, mode='2D')
        ma.MeasurementAnalysis(label=label, TwoD=True, close_file=True)

        if analyze_RB:
            rbma.Simultaneous_RB_Analysis(
                qb_names=[qb1n, qb2n],
                use_latest_data=True,
                gate_decomp=clifford_decomposition_name,
                add_correction=True)


def measure_n_qubit_simultaneous_randomized_benchmarking(
        qubits, f_LO,
        nr_cliffords=None, nr_seeds=50,
        gate_decomp='HZ', interleaved_gate=None,
        cal_points=False,
        thresholded=True,
        experiment_channels=None,
        soft_avgs=1,
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
        logging.warning("Unspecified UHFQC instrument. Using qubits[0].UHFQC.")
    if pulsar is None:
        pulsar = qubits[0].AWG
        logging.warning("Unspecified pulsar instrument. Using qubits[0].AWG.")
    if MC is None:
        MC = qubits[0].MC
        logging.warning("Unspecified MC object. Using qubits[0].MC.")
    if experiment_channels is None:
        experiment_channels = []
        for qb in qubits:
            experiment_channels += [qb.RO_acq_weight_function_I()]
        logging.warning('experiment_channels is None. Using only the channels '
                        'in the qubits RO_acq_weight_function_I parameters.')
    print(experiment_channels)
    if label is None:
        label = 'SRB_{}_{}_seeds_{}_cliffords_qubits{}'.format(
            gate_decomp, nr_seeds, nr_cliffords[-1] if
            hasattr(nr_cliffords, '__iter__') else nr_cliffords,
            ''.join([qb.name[-1] for qb in qubits]))

    key = 'int'
    if thresholded:
        key = 'dig'
        logging.warning('Make sure you have set them!.')
        label += '_thresh'

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
            qubits, nr_averages=nr_averages, UHFQC=UHFQC,
            pulsar=pulsar, used_channels=experiment_channels,
            correlations=correlations)[key+'_corr_det']
        hard_sweep_points = np.arange(nr_seeds)
        hard_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_fixed_length(
                qubit_names_list=qubit_names_list,
                operation_dict=operation_dict,
                nr_cliffords_value=nr_cliffords[0],
                nr_seeds_array=np.arange(nr_seeds),
                upload=False,
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate,
                verbose=verbose, cal_points=cal_points)
        soft_sweep_points = nr_cliffords
        soft_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_sequence_lengths(
            n_qubit_RB_sweepfunction=hard_sweep_func)

    else:
        if hasattr(nr_cliffords, '__iter__'):
                    raise ValueError('For an experiment with more than two '
                                     'qubits, nr_cliffords must int or float.')

        k = 4095//nr_seeds
        nr_shots = 4095 - 4095 % nr_seeds
        det_func = get_multiplexed_readout_detector_functions(
            qubits, UHFQC=UHFQC, pulsar=pulsar,
            nr_shots=nr_shots)[key+'_log_det']

        hard_sweep_points = np.tile(np.arange(nr_seeds), k)
        hard_sweep_func = \
            awg_swf2.n_qubit_Simultaneous_RB_fixed_length(
                qubit_names_list=qubit_names_list,
                operation_dict=operation_dict,
                nr_cliffords_value=nr_cliffords,
                nr_seeds_array=np.arange(nr_seeds),
                upload=True,
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate,
                verbose=verbose, cal_points=cal_points)
        soft_sweep_points = np.arange(nr_averages//k)
        soft_sweep_func = swf.None_Sweep()

    if cal_points:
        step = np.abs(hard_sweep_points[-1] - hard_sweep_points[-2])
        hard_sweep_points_to_use = np.concatenate(
            [hard_sweep_points,
             [hard_sweep_points[-1]+step, hard_sweep_points[-1]+2*step]])
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

    # # reset all correlation channels in the UHFQC
    # for ch in range(5):
    #     UHFQC.set('quex_corr_{}_mode'.format(ch), 0)
    #     UHFQC.set('quex_corr_{}_source'.format(ch), 0)
    #
    # for qb in qubits:
    #     if thresholding and V_th_a is not None:
    #         # set back the original threshold values
    #         UHFQC.set('quex_thres_{}_level'.format(
    #             qb.RO_acq_weight_function_I()), th_vals[qb.name])


    return MC


def measure_two_qubit_tomo_Bell(bell_state, qb_c, qb_t, f_LO,
                                basis_pulses=None,
                                cal_state_repeats=7, spacing=100e-9,
                                CZ_disabled=False,
                                num_flux_pulses=0, verbose=False,
                                nr_averages=1024, soft_avgs=1,
                                MC=None, UHFQC=None, pulsar=None,
                                upload=True, label=None, run=True,
                                used_channels=None):
    """
                 |spacing|spacing|
    |qCZ> --gate1--------*--------after_pulse-----| tomo |
                         |
    |qS > --gate2--------*------------------------| tomo |

        qb_c (qCZ) is the control qubit (pulsed)
        qb_t (qS) is the target qubit

    Args:
        bell_state (int): which Bell state to prepare according to:
            0 -> phi_Minus
            1 -> phi_Plus
            2 -> psi_Minus
            3 -> psi_Plus
        qb_c (qubit object): control qubit
        qb_t (qubit object): target qubit
        f_LO (float): RO LO frequency
        basis_pulses (tuple): tomo pulses to be applied on each qubit
            default: ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        cal_state_repeats (int): number of times to repeat each cal state
        spacing (float): spacing before and after CZ pulse; see diagram above
        CZ_disabled (bool): True -> same bell state preparation but without
            the CZ pulse; False -> normal bell state preparation
        num_flux_pulses (int): number of times to apply a flux pulse with the
            same length as the one used in the CZ gate before the entire
            sequence (before bell state prep).
        verbose (bool): print runtime info
        nr_averages (float): number of averages to use
        soft_avgs (int): number of soft averages to use
        MC: MeasurementControl object
        UHFQC: UHFQC object
        pulsar: pulsar object or AWG object
        upload (bool): whether to upload sequence to AWG or not
        label (str): measurement label
        run (bool): whether to execute MC.run() or not
    """
    if basis_pulses is None:
        basis_pulses = ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        logging.warning('basis_pulses not specified. Using the'
                        'following basis:\n{}'.format(basis_pulses))

    qubits = [qb_c, qb_t]
    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    if UHFQC is None:
        UHFQC = qb_c.UHFQC
        logging.warning("Unspecified UHFQC instrument. Using qb_c.UHFQC.")
    if pulsar is None:
        pulsar = qb_c.AWG
        logging.warning("Unspecified pulsar instrument. Using qb_c.AWG.")
    if MC is None:
        MC = qb_c.MC
        logging.warning("Unspecified MC object. Using qb_c.MC.")

    Bell_state_dict = {'0': 'phiMinus', '1': 'phiPlus',
                       '2': 'psiMinus', '3': 'psiPlus'}
    if label is None:
        label = '{}_tomo_Bell_{}_{}'.format(
            Bell_state_dict[str(bell_state)], qb_c.name, qb_t.name)

    if num_flux_pulses != 0:
        label += '_' + str(num_flux_pulses) + 'preFluxPulses'

    multiplexed_pulse(qubits, f_LO, upload=True)
    RO_pars = get_multiplexed_readout_pulse_dictionary(qubits)


    correlations = [(qubits[0].RO_acq_weight_function_I(),
                     qubits[1].RO_acq_weight_function_I())]

    corr_det = get_multiplexed_readout_detector_functions(
        qubits, nr_averages=nr_averages, UHFQC=UHFQC,
        used_channels=used_channels,
        pulsar=pulsar, correlations=correlations)['int_corr_det']

    tomo_Bell_swf_func = awg_swf2.tomo_Bell(
        bell_state=bell_state, qb_c=qb_c, qb_t=qb_t,
        RO_pars=RO_pars, num_flux_pulses=num_flux_pulses, spacing=spacing,
        basis_pulses=basis_pulses, cal_state_repeats=cal_state_repeats,
        verbose=verbose, upload=upload, CZ_disabled=CZ_disabled)

    MC.soft_avg(soft_avgs)
    MC.set_sweep_function(tomo_Bell_swf_func)
    MC.set_sweep_points(np.arange(64))
    MC.set_detector_function(corr_det)
    if run:
        MC.run(label)

    ma.MeasurementAnalysis(close_file=True)


def measure_three_qubit_tomo_GHZ(qubits, f_LO,
                                 basis_pulses=None,
                                 CZ_qubit_dict=None,
                                 cal_state_repeats=2,
                                 spacing=100e-9,
                                 thresholding=True, V_th_a=None,
                                 verbose=False,
                                 nr_averages=1024, soft_avgs=1,
                                 MC=None, UHFQC=None, pulsar=None,
                                 upload=True, label=None, run=True):
    """
              |spacing|spacing|
    |q0> --Y90--------*--------------------------------------|======|
                      |
    |q1> --Y90s-------*--------mY90--------*-----------------| tomo |
                                           |
    |q2> --Y90s----------------------------*--------mY90-----|======|
                                   |spacing|spacing|
    Args:
        qubits (list or tuple): list of 3 qubits
        f_LO (float): RO LO frequency
        CZ_qubit_dict(dict):  dict of the following form:
            {'qbc0': qb_control_name_CZ0,
             'qbt0': qb_target_name_CZ0,
             'qbc1': qb_control_name_CZ1,
             'qbt1': qb_target_name_CZ1}
            where CZ0, and CZ1 refer to the first and second CZ applied.
            (We start counting from zero because we are programmers.)
        basis_pulses (tuple): tomo pulses to be applied on each qubit
            default: ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        cal_state_repeats (int): number of times to repeat each cal state
        spacing (float): spacing before and after CZ pulse; see diagram above
        thresholding (bool): whether to use the thresholding feature
            of the UHFQC
        V_th_a (list or tuple): contains the SSRO assignment thresholds for
            each qubit in qubits. These values must be correctly scaled!
        verbose (bool): print runtime info
        nr_averages (float): number of averages to use
        soft_avgs (int): number of soft averages to use
        MC: MeasurementControl object
        UHFQC: UHFQC object
        pulsar: pulsar object or AWG object
        upload (bool): whether to upload sequence to AWG or not
        label (str): measurement label
        run (bool): whether to execute MC.run() or not
    """
    if CZ_qubit_dict is None:
        raise ValueError('CZ_qubit_dict is None.')
    if basis_pulses is None:
        basis_pulses = ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        logging.warning('basis_pulses not specified. Using the'
                        'following basis:\n{}'.format(basis_pulses))
    if UHFQC is None:
        UHFQC = qubits[0].UHFQC
        logging.warning("Unspecified UHFQC instrument. Using {}.UHFQC.".format(
            qubits[0].name))
    if pulsar is None:
        pulsar = qubits[0].AWG
        logging.warning("Unspecified pulsar instrument. Using {}.AWG.".format(
            qubits[0].name))
    if MC is None:
        MC = qubits[0].MC
        logging.warning("Unspecified MC object. Using {}.MC.".format(
            qubits[0].name))

    if label is None:
        label = '{}-{}-{}_GHZ_tomo'.format(*[qb.name for qb in qubits])

    key = 'int'
    if thresholding:
        if V_th_a is None:
            raise ValueError('Unknown threshold values.')
        else:
            label += '_thresh'
            key = 'dig'
            for thresh_level, qb in zip(V_th_a, qubits):
                UHFQC.set('quex_thres_{}_level'.format(
                    qb.RO_acq_weight_function_I()), thresh_level)

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    multiplexed_pulse(qubits, f_LO, upload=True)
    RO_pars = get_multiplexed_readout_pulse_dictionary(qubits)

    det_func = get_multiplexed_readout_detector_functions(
        qubits, UHFQC=UHFQC, pulsar=pulsar,
        nr_shots=nr_averages)[key+'_log_det']

    hard_sweep_points = np.arange(nr_averages)
    hard_sweep_func = \
        awg_swf2.three_qubit_GHZ_tomo(qubits=qubits, RO_pars=RO_pars,
                                      CZ_qubit_dict=CZ_qubit_dict,
                                      basis_pulses=basis_pulses,
                                      cal_state_repeats=cal_state_repeats,
                                      spacing=spacing,
                                      verbose=verbose,
                                      upload=upload)

    total_nr_segments = len(basis_pulses)**len(qubits) + \
                        cal_state_repeats*2**len(qubits)
    soft_sweep_points = np.arange(total_nr_segments)
    soft_sweep_func = swf.None_Sweep()

    MC.soft_avg(soft_avgs)
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)

    MC.set_sweep_function_2D(soft_sweep_func)
    MC.set_sweep_points_2D(soft_sweep_points)

    MC.set_detector_function(det_func)
    if run:
        MC.run_2D(label)

    ma.TwoD_Analysis(label=label, close_file=True)


def cphase_gate_tuneup(qb_control, qb_target,
                       initial_values_dict=None,
                       initial_step_dict=None,
                       MC_optimization=None,
                       MC_detector=None,
                       maxiter=50,
                       spacing=20e-9,
                       ramsey_phases=None):

    '''
    function that runs the nelder mead algorithm to optimize the CPhase gate
    parameters (pulse lengths and pulse amplitude)

    Args:
        qb_control (QuDev_Transmon): control qubit (with flux pulses)
        qb_target (QuDev_Transmon): target qubit
        initial_values_dict (dict): dictionary containing the initial flux
                                    pulse amp and length (keys are 'amplitude'
                                    and 'length')
        initial_step_dict (dict): dictionary containing the initial step size
                                  of the flux pulse amp and length (keys are
                                  'step_amplitude' and 'step_length')
        MC_optimization (MeasurementControl): measurement control for the
                                              adaptive optimization sweep
        MC_detector (MeasurementControl): measurement control used in the detector
                                            function to run the actual experiment
        maxiter (int): maximum optimization steps passed to the nelder mead function
        name (str): measurement name
        spacing (float): safety spacing between drive pulses and flux pulse
        ramsey_phases (numpy array): phases used in the Ramsey measurement

    Returns:
        pulse_length_best_value, pulse_amplitude_best_value

    '''

    if MC_optimization is None:
        MC_optimization = qb_control.MC

    if initial_values_dict is None:
        pulse_length_init = qb_control.flux_pulse_length()
        pulse_amplitude_init = qb_control.flux_pulse_amp()
    else:
        pulse_length_init = initial_values_dict['length']
        pulse_amplitude_init = initial_values_dict['amplitude']

    if initial_step_dict is None:
        init_step_length = 1.0e-9
        init_step_amplitude = 1.0e-3
    else:
        init_step_length = initial_step_dict['step_length']
        init_step_amplitude = initial_step_dict['step_amplitude']

    # initial measurement so set up all instruments
    qb_control.measure_cphase(qb_target,
                              amps=[pulse_amplitude_init],
                              lengths=[pulse_length_init],
                              cal_points=False, plot=False,
                              phases=ramsey_phases, spacing=spacing,
                              prepare_for_timedomain=False
                              )

    d = cdet.CPhase_optimization(qb_control=qb_control, qb_target=qb_target,
                                 MC=MC_detector,
                                 ramsey_phases=ramsey_phases,
                                 spacing=spacing)

    S1 = d.flux_pulse_length
    S2 = d.flux_pulse_amp

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [pulse_length_init, pulse_amplitude_init],
                    'initial_step': [init_step_length, init_step_amplitude],
                    'no_improv_break': 12,
                    'minimize': True,
                    'maxiter': maxiter}
    MC_optimization.set_sweep_functions([S1, S2])
    MC_optimization.set_detector_function(d)
    MC_optimization.set_adaptive_function_parameters(ad_func_pars)
    MC_optimization.run(name=name, mode='adaptive')
    a1 = ma.OptimizationAnalysis(label=name)
    a2 = ma.OptimizationAnalysis_v2(label=name)
    pulse_length_best_value = a1.optimization_result[0][0]
    pulse_amplitude_best_value = a1.optimization_result[0][1]

    return pulse_length_best_value, pulse_amplitude_best_value


def cphase_gate_tuneup_predictive(qbc, qbt, qbr, initial_values: list,
                                  std_deviations: list = [20e-9,0.03],
                                  phases = None, MC = None,
                                  estimator = 'GRNN_neupy',
                                  hyper_parameter_dict : dict = None,
                                  sampling_numbers: list = [70,30],
                                  max_measurements = 2,
                                  tol = [0.016,0.05],
                                  timestamps : list = None,
                                  update = False,
                                  full_output = True,
                                  fine_tune = True, fine_tune_minmax=None):
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
        logging.warning("Does not automatically update the CZ pulse length "
                        "and amplitude. "
                        "Set update=True if you want this!")
    if not (isinstance(sampling_numbers,list) or
            isinstance(sampling_numbers,np.ndarray)):
        sampling_numbers = [sampling_numbers]
    if max_measurements != len(sampling_numbers):
        logging.warning('Did not provide sampling number for each iteration '
                        'step! Additional iterations will be carried out with the'
                        'last value in sampling numbers ')
    if len(initial_values) != 2:
        logging.error('Incorrect number of input mean values for Gaussian '
                      'sampling provided!')
    if len(std_deviations) != 2:
        logging.error('Incorrect number of standard deviations for Gaussian '
                      'sampling provided!')
    if hyper_parameter_dict is None:
        logging.warning('\n No hyperparameters passed to predictive mixer '
                        'calibration routine. Default values for the estimator'
                        'will be used!\n')

        hyper_parameter_dict = {'cv_n_fold': 10,
                                'std_scaling': [0.4, 0.4]}

    if phases is None:
        phases = np.linspace(0, 2*np.pi, 16, endpoint=False)
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

        for i,t in enumerate(timestamps_iter):

            flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis_Predictive(
                timestamp=t,
                label='CPhase_measurement_{}_{}'.format(qbc.name, qbt.name),
                qb_name=qbc.name, cal_points=False, plot=False,
                save_plot=False,
                reference_measurements=True, only_cos_fits=True)

            cphases = flux_pulse_ma.cphases
            population_losses = flux_pulse_ma.population_losses

            target_phases = np.abs(np.abs(cphases/np.pi) - 1.)
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
                print('Added {} training samples from timestamp {}!'\
                      .format(np.shape(new_train_values)[0], t))

        data_size = 0 if training_grid is None else np.shape(training_grid)[0]

        # if not (iteration == 0 and timestamps_iter):
        print('\n{} samples before Iteration {}'.format(data_size,
                                                       iteration+1))
        if iteration >= len(sampling_numbers):
            sampling_number = sampling_numbers[-1]
        else:
            sampling_number = sampling_numbers[iteration]
        if iteration > 0:
            std_length *= std_factor #rescale std deviations for next round
            std_amp *= std_factor

        new_flux_lengths = np.random.normal(pulse_length_best,
                                            std_length,
                                            sampling_number)
        new_flux_lengths = np.abs(new_flux_lengths)
        new_flux_amps = np.random.normal(pulse_amplitude_best,
                                         std_amp,
                                         sampling_number)
        print('measuring {} samples in iteration {} \n'.\
              format(sampling_number, iteration+1))

        cphases, population_losses, flux_pulse_ma = \
                    measure_cphase(qbc, qbt, qbr,
                                     new_flux_lengths, new_flux_amps,
                                     phases = phases,
                                     plot = False,
                                     MC = MC)

        target_phases = np.abs(np.abs(cphases/np.pi) - 1.)
        target_pops = np.abs(population_losses)
        new_train_values = np.array([flux_pulse_ma.sweep_points_2D[0][::2],
                                   flux_pulse_ma.sweep_points_2D[1][::2]]).T
        new_target_values = np.array([target_phases, target_pops]).T

        training_grid, target_values = generate_new_training_set(
                                              new_train_values,
                                              new_target_values,
                                              training_grid = training_grid,
                                              target_values = target_values)
        new_timestamp = flux_pulse_ma.timestamp_string

    #train and test
        target_norm = np.sqrt(target_values[:, 0]**2+target_values[:, 1]**2)
        min_ind = np.argmin(target_norm)
        x_init = [training_grid[min_ind, 0], training_grid[min_ind,1]]
        a_pred = ma.OptimizationAnalysis_Predictive2D(training_grid,
                                    target_values,
                                    flux_pulse_ma,
                                    x_init = x_init,
                                    estimator = estimator,
                                    hyper_parameter_dict = hyper_parameter_dict,
                                    target_value_names = target_value_names)
        pulse_length_best = a_pred.optimization_result[0]
        pulse_amplitude_best = a_pred.optimization_result[1]
        cphase_testing_agent.lengths_opt.append(pulse_length_best)
        cphase_testing_agent.amps_opt.append(pulse_amplitude_best)

        #Get cphase with optimized values
        cphase_opt, population_loss_opt = cphase_testing_agent. \
                                                         yield_new_measurement()

        if fine_tune:
            print('optimized flux parameters good enough for finetuning.\n'
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


        #check success of iteration step
        if cphase_testing_agent.converged:
            print('Cphase optimization converged in iteration {}.'.\
                  format(iteration+1))

        elif iteration+1 >= max_measurements:
            cphase_testing_agent.converged = True
            logging.warning('\n maximum iterations exceeded without hitting'
                            ' specified tolerance levels for optimization!\n')
        else:
            print('Iteration {} finished. Not converged with cphase {}*pi and '
                  'population recovery {} %'\
                  .format(iteration,cphase_testing_agent.cphases[-1],
                           np.abs(1.- cphase_testing_agent.pop_losses[-1])*100))

            print('Running Iteration {} of {} ...'.format(iteration+1,
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
    population_recovery_opt = np.abs(1.-cphase_testing_agent.pop_losses[-1])*100
    pulse_length_best = cphase_testing_agent.lengths_opt[-1]
    pulse_amplitude_best = cphase_testing_agent.amps_opt[-1]
    std_cphase = cphase_testing_agent.cphase_std

    print('CPhase optimization finished with optimal values: \n',
          'Controlled Phase QBc={} Qb Target={}: '.format(qbc.name,qbt.name),
          cphase_opt,r" ($ \pm $",std_cphase,' )',r"$\pi$",'\n',
          'Population Recovery |e> Qb Target: {}% \n' \
          .format(population_recovery_opt),
          '@ flux pulse Paramters: \n',
          'Pulse Length: {:0.1f} ns \n'.format(pulse_length_best*1e9),
          'Pulse Length: {:0.4f} V \n'.format(pulse_amplitude_best))
    if update:
        qbc.set('CZ_{}_amp'.format(qbt.name), pulse_amplitude_best)
        qbc.set('CZ_{}_length'.format(qbt.name), pulse_length_best)
    if full_output:
        return pulse_length_best, pulse_amplitude_best,\
               [population_recovery_opt,cphase_opt],[std_cphase]
    else:
        return pulse_length_best, pulse_amplitude_best


def cphase_finetune_parameters(qbc, qbt, qbr, flux_length, flux_amplitudes,
                               phases, MC, save_fig=True, show=True):
    """
    measures cphases for a single slice of chevron with fixed flux length.
    Returns the best amplitude in flux_amplitudes for a cphase of pi.
    """
    flux_lengths = len(flux_amplitudes)*[flux_length]
    cphases, population_losses, ma_ram2D = \
        measure_cphase(qbc, qbt, qbr,
                       flux_lengths,
                       flux_amplitudes,
                       phases=phases,
                       plot=True,
                       MC=MC,
                       fit_statistics=False)
    cphases %= 2*np.pi
    fit_res = lmfit.Model(lambda x, m, b: m*np.tan(x/2-np.pi/2) + b).fit(
        x=cphases, data=flux_amplitudes, m=1, b=np.mean(flux_amplitudes))
    best_amp = fit_res.model.func(np.pi, **fit_res.best_values)
    amps_model = fit_res.model.func(cphases, **fit_res.best_values)
    fig, ax = plt.subplots()
    ax.plot(cphases*180/np.pi, flux_amplitudes/1e-3, 'o-')
    ax.plot(cphases*180/np.pi, amps_model/1e-3, '-r')
    ax.hlines(best_amp/1e-3, cphases[0]*180/np.pi, cphases[-1]*180/np.pi)
    ax.vlines(180, flux_amplitudes.min()/1e-3, flux_amplitudes.max()/1e-3)
    ax.set_ylabel('Flux pulse amplitude (mV)')
    ax.set_xlabel('Conditional phase (rad)')
    ax.set_title('CZ {}-{}'.format(qbc.name, qbt.name ))

    ax.text(0.5, 0.95, 'Best amp = {:.6f} V'.format(best_amp),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)

    if save_fig:

        fig_title = 'CPhase_amp_sweep_{}_{}'.format(qbc.name, qbt.name)
        fig_title = '{}--{:%Y%m%d_%H%M%S}'.format(
            fig_title, datetime.datetime.now())
        save_folder = ma_ram2D.folder
        filename = os.path.abspath(os.path.join(save_folder, fig_title+'.png'))
        fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

    return best_amp

class Averaged_Cphase_Measurement():

    def __init__(self, qbc, qbt, qbr, n_phases, MC, n_average=5,
                 tol= [0.016, 0.05]):
        if MC is None:
            self.MC = qbc.MC
        self.qbc = qbc
        self.qbt = qbt
        self.qbr = qbr
        self.MC = MC
        phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)
        self.phases = np.concatenate((phases, phases))
        self.cphases = []
        self.pop_losses = []
        self.lengths_opt = []
        self.amps_opt = []
        self.n_average = n_average
        self.cphase_std = None
        self.tol = tol
        self.converged = False

    def yield_new_measurement(self):

        if not self.lengths_opt or not self.amps_opt:
            logging.error('called averaged cphase measurement generator without'
                          'and optimization result for flux parameters!')
        test_lengths = np.repeat([self.lengths_opt[-1]], self.n_average)
        test_amps = np.repeat([self.amps_opt[-1]], self.n_average)

        cphases_opt, population_losses_opt, ma_ram2D_opt = \
                                    measure_cphase(self.qbc,
                                                    self.qbt,
                                                    self.qbr,
                                                    test_lengths,
                                                    test_amps,
                                                    phases=self.phases,
                                                    plot=True,
                                                    MC=self.MC,
                                                    fit_statistics=True)
        cphases_opt = np.abs(cphases_opt/np.pi)
        self.cphase_std = np.std(cphases_opt)
        self.cphases.append(np.mean(cphases_opt))
        self.pop_losses.append(np.mean(population_losses_opt))
        self.check_convergence()

        return self.cphases[-1],self.pop_losses[-1]

    def check_convergence(self):

        if np.abs(self.cphases[-1] - 1.) < self.tol[0] \
              and self.pop_losses[-1] < self.tol[1]:

            self.converged = True


def calibrate_n_qubits(qubits, f_LO, sweep_points_dict, sweep_params=None,
                       artificial_detuning=None,
                       cal_points=True, no_cal_points=4, upload=True,
                       MC=None, soft_avgs=1, n_rabi_pulses=1,
                       thresholded=False, #analyses can't handle it!
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

    exp_metadata = None

    # set up multiplexed readout
    multiplexed_pulse(qubits, f_LO, upload=True)
    operation_dict = get_operation_dict(qubits)
    if thresholded:
        key = 'dig'
    else:
        key = 'int'

    nr_averages = max([qb.RO_acq_averages() for qb in qubits])
    df = get_multiplexed_readout_detector_functions(
        qubits, UHFQC=UHFQC, pulsar=pulsar,
        nr_averages=nr_averages)[key + '_avg_det']

    RO_channels_dict = {}
    for qb in qubits:
        RO_channels_dict[qb.name] = [qb.RO_acq_weight_function_I()]
        if qb.ro_acq_weight_type() in ['SSB', 'DSB']:
            RO_channels_dict[qb.name] += [qb.RO_acq_weight_function_Q()]
        qb.prepare_for_timedomain(multiplexed=True)

    qubit_names = [qb.name for qb in qubits]

    if len(qubit_names) > 5:
        msmt_suffix = '_{}qubits'.format(len(qubit_names))
    else:
        msmt_suffix = '_qbs{}'.format(''.join([i[-1] for i in qubit_names]))

    if cal_points:
        for key, spts in sweep_points_dict.items():
            if spts is None:
                if key == 'n_rabi':
                    sweep_points_dict[key] = {}
                    for qb in qubits:
                        sweep_points_dict[key][qb.name] = \
                            np.linspace(
                                (n_rabi_pulses-1)*qb.amp180()/n_rabi_pulses,
                                 min((n_rabi_pulses+1)*qb.amp180()/n_rabi_pulses,
                                 0.95), 41)
                else:
                   raise ValueError('Sweep points for {} measurement are not '
                                    'defined.'.format(key))
            else:
                if key != 'qscale':
                    step = np.abs(spts[-1]-spts[-2])
                    if no_cal_points == 4:
                        sweep_points_dict[key] = np.concatenate(
                            [spts, [spts[-1]+step, spts[-1]+2*step,
                                    spts[-1]+3*step, spts[-1]+4*step]])
                    elif no_cal_points == 2:
                        sweep_points_dict[key] = np.concatenate(
                            [spts, [spts[-1]+step, spts[-1]+2*step]])
                    else:
                        sweep_points_dict[key] = spts

    # Do measurements
    # RABI
    if 'rabi' in sweep_points_dict:
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
            for qb in qubits:
                RabiA = ma.Rabi_Analysis(
                    label=label, qb_name=qb.name,
                    RO_channels=RO_channels_dict[qb.name],
                    NoCalPoints=4, close_fig=True,
                    plot_title_suffix='_'+qb.name, **kw)

                print(qb.name,  RabiA.rabi_amplitudes['piPulse'])
                if update:
                    rabi_amps = RabiA.rabi_amplitudes
                    amp180 = rabi_amps['piPulse']
                    amp90 = rabi_amps['piHalfPulse']
                    try:
                        qb.amp180(amp180)
                        qb.amp90_scale(0.5)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.' % e)

    # N-RABI
    if 'n_rabi' in sweep_points_dict:
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
            for qb in qubits:
                RabiA = ma.Rabi_Analysis(
                    label=label, qb_name=qb.name,
                    RO_channels=RO_channels_dict[qb.name],
                    NoCalPoints=4, close_fig=True,
                    plot_title_suffix='_'+qb.name,
                    new_sweep_points=sweep_points[qb.name], **kw)

                if update:
                    rabi_amps = RabiA.rabi_amplitudes
                    amp180 = rabi_amps['piPulse']
                    amp90 = rabi_amps['piHalfPulse']
                    try:
                        qb.amp180(amp180)
                        qb.amp90_scale(0.5)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.' % e)

    # RAMSEY
    if 'ramsey' in sweep_points_dict:
        if artificial_detuning is None:
            raise ValueError('Specify an artificial_detuning for the Ramsey '
                             'measurement.')
        sweep_points = sweep_points_dict['ramsey']
        if sweep_params is None:
            sweep_params = (
                ('X90', {}),
                ('X90', {
                    'pulse_pars': {
                        'refpoint': 'start',
                        'pulse_delay': (lambda sp: sp),
                        'phase': (lambda sp:
                                  ((sp-sweep_points[0]) * artificial_detuning *
                                   360) % 360)}})
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
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            for qb in qubits:
                RamseyA = ma.Ramsey_Analysis(
                    label=label, qb_name=qb.name,
                    NoCalPoints=4, RO_channels=RO_channels_dict[qb.name],
                    artificial_detuning=artificial_detuning,
                    plot_title_suffix='_'+qb.name, **kw)

                if update:
                    new_qubit_freq = RamseyA.qubit_frequency
                    T2_star = RamseyA.T2_star
                    try:
                        qb.f_qubit(new_qubit_freq)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)
                    try:
                        qb.T2_star(T2_star['T2_star'])
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)

    # QSCALE
    if 'qscale' in sweep_points_dict:
        sweep_points = sweep_points_dict['qscale']
        temp_array = np.zeros(3*sweep_points.size)
        np.put(temp_array,list(range(0,temp_array.size,3)),sweep_points)
        np.put(temp_array,list(range(1,temp_array.size,3)),sweep_points)
        np.put(temp_array,list(range(2,temp_array.size,3)),sweep_points)
        sweep_points = temp_array

        if cal_points:
            step = np.abs(sweep_points[-1]-sweep_points[-2])
            if no_cal_points == 4:
                sweep_points = np.concatenate(
                    [sweep_points, [sweep_points[-1]+step,
                                    sweep_points[-1]+2*step,
                                    sweep_points[-1]+3*step,
                                    sweep_points[-1]+4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [sweep_points, [sweep_points[-1]+step,
                                    sweep_points[-1]+2*step]])
            else:
                pass

        if sweep_params is None:
            sweep_params = (
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i%3==0)}),
                ('X180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                          'condition': (lambda i: i%3==0)}),
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i%3==1)}),
                ('Y180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                          'condition': (lambda i: i%3==1)}),
                ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                         'condition': (lambda i: i%3==2)}),
                ('mY180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                           'condition': (lambda i: i%3==2)}),
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
            for qb in qubits:
                qscaleA = ma.QScale_Analysis(
                    label=label, qb_name=qb.name,
                    NoCalPoints=4,
                    RO_channels=RO_channels_dict[qb.name],
                    plot_title_suffix='_'+qb.name, **kw)

                if update:
                    qscale_dict = qscaleA.optimal_qscale #dictionary of value, stderr
                    qscale_value = qscale_dict['qscale']

                    try:
                        qb.motzoi(qscale_value)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)

    # T1
    if 'T1' in sweep_points_dict:
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
            for qb in qubits:
                T1_Analysis = ma.T1_Analysis(
                    label=label, qb_name=qb.name, NoCalPoints=4,
                    RO_channels=RO_channels_dict[qb.name],
                    plot_title_suffix='_'+qb.name, **kw)

                if update:
                    try:
                        qb.T1(T1_Analysis.T1_dict['T1'])
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)
    # T2 ECHO
    if 'T2' in sweep_points_dict:
        if artificial_detuning is None:
            raise ValueError('Specify an artificial_detuning for the Ramsey '
                             'measurement.')
        sweep_points = sweep_points_dict['T2']
        if sweep_params is None:
            sweep_params = (
                ('X90', {}),
                ('X180', {'pulse_pars': {'refpoint': 'start',
                                         'pulse_delay': (lambda sp: sp/2)}}),
                ('X90', {'pulse_pars': {
                    'refpoint': 'start',
                    'pulse_delay': (lambda sp: sp/2),
                    'phase': (lambda sp:
                              ((sp-sweep_points[0]) * artificial_detuning *
                               360) % 360)}})

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
        label = 'T2_echo' + msmt_suffix
        if isinstance(sweep_points, dict):
            exp_metadata = {'sweep_points_dict': sweep_points}
        MC.run(label, exp_metadata=exp_metadata)
        sweep_params = None

        if analyze:
            for qb in qubits:
                RamseyA = ma.Ramsey_Analysis(
                    label=label, qb_name=qb.name,
                    NoCalPoints=4, RO_channels=RO_channels_dict[qb.name],
                    artificial_detuning=artificial_detuning,
                    plot_title_suffix='_'+qb.name, **kw)

                if update:
                    T2_star = RamseyA.T2_star
                    try:
                        qb.T2(T2_star['T2_star'])
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)


def measure_cz_frequency_sweep(qbc, qbt, qbr, frequencies, length, amplitude,
                    cal_points=True, upload=True,
                    verbose=False, return_seq=False,
                    MC=None, soft_averages=1):

    if MC is None:
        MC = qbc.MC

    operation_dict = get_operation_dict([qbc, qbt, qbr])
    CZ_pulse_name = 'CZ ' + qbt.name + ' ' + qbc.name

    for qb in [qbc, qbt, qbr]:
        qb.prepare_for_timedomain()

    sf1 = awg_swf.Chevron_frequency_hard_swf(
        frequencies=frequencies,
        length=length,
        flux_pulse_amp=amplitude,
        qbc_name=qbc.name,
        qbt_name=qbt.name,
        qbr_name=qbr.name,
        readout_qbt=qbr.name,
        CZ_pulse_name=CZ_pulse_name,
        operation_dict=operation_dict,
        verbose=verbose, cal_points=cal_points,
        upload=upload, return_seq=return_seq)
    MC.soft_avg(soft_averages)
    MC.set_sweep_function(sf1)
    MC.set_sweep_points(frequencies)

    MC.set_detector_function(qbr.int_avg_det)
    MC.run('CZ_Frequency_Sweep_{}{}'.format(qbc.name, qbt.name))

    ma.MeasurementAnalysis()

def measure_chevron(qbc, qbt, qbr, lengths, amplitudes, frequencies,
                    cal_points=True, upload=True,
                    verbose=False, return_seq=False,
                    MC=None, soft_averages=1):

    if MC is None:
        MC = qbc.MC

    operation_dict = get_operation_dict([qbc, qbt, qbr])
    CZ_pulse_name = 'CZ ' + qbt.name + ' ' + qbc.name

    for qb in [qbc, qbt, qbr]:
        qb.prepare_for_timedomain()

    flux_channel = operation_dict[CZ_pulse_name]['channel']

    sf1 = awg_swf.Chevron_length_swf_new(
                lengths=lengths,
                flux_pulse_amp=amplitudes[0],
                frequency=frequencies[0],
                qbc_name=qbc.name,
                qbt_name=qbt.name,
                qbr_name=qbr.name,
                readout_qbt=qbr.name,
                CZ_pulse_name=CZ_pulse_name,
                operation_dict=operation_dict,
                verbose=verbose, cal_points=cal_points,
                upload=False, return_seq=return_seq)
    MC.soft_avg(soft_averages)
    MC.set_sweep_function(sf1)
    MC.set_sweep_points(lengths)

    if len(amplitudes) > 1:
        sf2 = awg_swf.Chevron_ampl_swf_new(hard_sweep=sf1)
        sweep_points_2D = amplitudes
    elif len(frequencies) > 1:
        sf2 = awg_swf.Chevron_freq_swf_new(hard_sweep=sf1)
        sweep_points_2D = frequencies
    else:
        raise ValueError('At least amplitudes or frequencies must have len > 1')


    MC.set_sweep_function_2D(sf2)
    MC.set_sweep_points_2D(sweep_points_2D)
    MC.set_detector_function(qbr.int_avg_det)
    MC.run_2D('Chevron_{}{}'.format(qbc.name, qbt.name))

    ma.MeasurementAnalysis(TwoD=True)

def measure_cphase(qbc, qbt, qbr, lengths, amps,
                       phases=None, MC=None,
                       cal_points=False, plot=False,
                       save_plot=True,
                       prepare_for_timedomain=True,
                       output_measured_values=False,
                       upload=True,**kw):
    '''
    method to measure the phase acquired during a flux pulse conditioned on the state
    of the control qubit (self).
    In this measurement, the phase from two Ramsey type measurements
    on qb_target is measured, once with the control qubit in the excited state and once
    in the ground state. The conditional phase is calculated as the difference.


    Args:
        qb_target (QuDev_transmon): target qubit / non-fluxed qubit
        amps (list): list or array of flux pulse amplitudes
        lengths (list):  list or array of flux pulse lengths (must have same dimension as
                         amps)
        phases (array): phases used for the Ramsey type phase sweep
        spacing (float): spacing between flux pulse and Ramsey pulses in s
        MC (optional): measurement control
        cal_points (bool): if True, calibration points are measured
        plot (bool): if true, the phase fit is shown
        return_population_loss: if true, the population loss (loss of contrast when having
                                the control qubit in the excited state is returned)
        upload_AWGs (list): list of the AWGs to be uploaded
        upload_channels (list): list of channels to be uploaded
        prepare_for_timedomain (bool): if False, the self.prepare_for_timedomain()
                                       is NOT called

    Returns:
        cphases (numpy array): array of the conditional phases measured at
                              (amps[i], lengths[i])
    '''
    if len(amps) != len(lengths):
        logging.warning('amps and lengths must have the same '
                        'dimension.')

    if MC is None:
        MC = qbc.MC

    if phases is None:
        phases = np.linspace(0, 2*np.pi, 16, endpoint=False)
        phases = np.concatenate((phases,phases))


    operation_dict = get_operation_dict([qbc, qbt, qbr])
    CZ_pulse_name = 'CZ ' + qbt.name + ' ' + qbc.name
    CZ_pulse_channel = qbc.flux_pulse_channel()
    max_flux_length = np.max(lengths)

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
    s2 = awg_swf.Flux_pulse_CPhase_soft_swf(s1,sweep_param='length',
                                                  upload=upload)
    s3 = awg_swf.Flux_pulse_CPhase_soft_swf(s1,sweep_param='amplitude',
                                    upload=upload)

    t0 = time.time()
    if prepare_for_timedomain:
        for qb in [qbc, qbt, qbr]:
             qb.prepare_for_timedomain()
    MC.set_sweep_functions([s1,s2,s3])
    MC.set_sweep_points(phases)
    #Here the order of the parameters matters! Paramters must be
    #set in the same order as their sweepfunctions!
    MC.set_sweep_points_2D(np.array([lengths,amps]).T)
    MC.set_detector_function(qbr.int_avg_det)
    MC.run_2D('CPhase_measurement_{}_{}'.format(qbc.name,qbt.name))

    t1 = time.time()
    print('Measured Cphases with ',
          len(amps)*len(phases),
          ' sweeppoints in T=',t1-t0,' s.')

    # ma.TwoD_Analysis(close_file=True)
    flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis_Predictive(
        label='CPhase_measurement_{}_{}'.format(qbc.name, qbt.name),
        qb_name=qbc.name, cal_points=cal_points, plot=plot, save_plot=save_plot,
        reference_measurements=True, only_cos_fits=True, **kw)
    cphases = flux_pulse_ma.cphases
    population_losses = flux_pulse_ma.population_losses
    if output_measured_values:
        print('fitted phases: ', cphases)
        print('pop loss: ', population_losses)
    return cphases, population_losses, flux_pulse_ma


def measure_cphase_frequency(qbc, qbt, qbr, frequencies, length, amp,
                       phases=None, MC=None, cal_points=False, plot=False,
                       save_plot=True,
                       prepare_for_timedomain=True,
                       output_measured_values=False,
                       upload=True,**kw):
    '''
    method to measure the phase acquired during a flux pulse conditioned on the state
    of the control qubit (self).
    In this measurement, the phase from two Ramsey type measurements
    on qb_target is measured, once with the control qubit in the excited state and once
    in the ground state. The conditional phase is calculated as the difference.


    Args:
        qb_target (QuDev_transmon): target qubit / non-fluxed qubit
        amps (list): list or array of flux pulse amplitudes
        lengths (list):  list or array of flux pulse lengths (must have same dimension as
                         amps)
        phases (array): phases used for the Ramsey type phase sweep
        spacing (float): spacing between flux pulse and Ramsey pulses in s
        MC (optional): measurement control
        cal_points (bool): if True, calibration points are measured
        plot (bool): if true, the phase fit is shown
        return_population_loss: if true, the population loss (loss of contrast when having
                                the control qubit in the excited state is returned)
        upload_AWGs (list): list of the AWGs to be uploaded
        upload_channels (list): list of channels to be uploaded
        prepare_for_timedomain (bool): if False, the self.prepare_for_timedomain()
                                       is NOT called

    Returns:
        cphases (numpy array): array of the conditional phases measured at
                              (amps[i], lengths[i])
    '''

    if MC is None:
        MC = qbc.MC

    if phases is None:
        phases = np.linspace(0, 2*np.pi, 16, endpoint=False)
        phases = np.concatenate((phases,phases))


    operation_dict = get_operation_dict([qbc, qbt, qbr])
    CZ_pulse_name = 'CZ ' + qbt.name + ' ' + qbc.name
    CZ_pulse_channel = qbc.flux_pulse_channel()
    max_flux_length = length

    s1 = awg_swf.Flux_pulse_CPhase_hard_swf_frequency(phases,
                                                qbc.name,
                                                qbt.name,
                                                qbr.name,
                                                CZ_pulse_name,
                                                length, amp,
                                                CZ_pulse_channel,
                                                operation_dict,
                                                max_flux_length,
                                                cal_points=cal_points,
                                                reference_measurements=True,
                                                upload=upload)
    s2 = awg_swf.Flux_pulse_CPhase_soft_swf(s1,sweep_param='frequency',
                                            upload=upload)

    if prepare_for_timedomain:
        for qb in [qbc, qbt, qbr]:
            qb.prepare_for_timedomain()
    MC.set_sweep_function(s1)
    MC.set_sweep_points(phases)
    #Here the order of the parameters matters! Paramters must be
    #set in the same order as their sweepfunctions!
    MC.set_sweep_function_2D(s2)
    MC.set_sweep_points_2D(frequencies)
    MC.set_detector_function(qbr.int_avg_det)
    MC.run_2D('CPhase_measurement_{}_{}'.format(qbc.name,qbt.name))

    flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis(
        label='CPhase_measurement_{}_{}'.format(qbc.name, qbt.name),
        qb_name=qbc.name, cal_points=False, save_plot=False,
        reference_measurements=True, only_cos_fits=True)

    fitted_phases_exited = flux_pulse_ma.fitted_phases[:: 2]
    fitted_phases_ground = flux_pulse_ma.fitted_phases[1:: 2]

    cphases = fitted_phases_exited - fitted_phases_ground
    if output_measured_values:
        print('fitted phases: ', cphases)
    return cphases, flux_pulse_ma


def measure_CZ_bleed_through(qb, CZ_separation_times, phases,
                             CZ_pulse_name, upload=True, cal_points=True,
                             MC=None):

    if MC is None:
        MC = qb.MC

    qb.prepare_for_timedomain()

    if cal_points:
        step = np.abs(phases[-1]-phases[-2])
        phases = np.concatenate(
            [phases, [phases[-1]+step,  phases[-1]+2*step,
                      phases[-1]+3*step, phases[-1]+4*step]])

    operation_dict = qb.get_operation_dict()
    CZ_channel = operation_dict[CZ_pulse_name]['channel']

    s1 = awg_swf.CZ_bleed_through_phase_hard_sweep(
        qb_name=qb.name,
        CZ_pulse_name=CZ_pulse_name,
        CZ_separation=CZ_separation_times[0],
        operation_dict=operation_dict,
        maximum_CZ_separation=np.max(CZ_separation_times),
        verbose=False,
        upload=True,
        return_seq=False,
        cal_points=cal_points)

    s2 = awg_swf.CZ_bleed_through_separation_time_soft_sweep(
        s1, upload=upload, upload_channels=[CZ_channel])

    MC.set_sweep_function(s1)
    MC.set_sweep_points(phases)
    MC.set_sweep_function_2D(s2)
    MC.set_sweep_points_2D(CZ_separation_times)
    MC.set_detector_function(qb.int_avg_det)
    MC.run_2D('CZ_bleed_through{}'.format(qb.msmt_suffix))

    ma.MeasurementAnalysis(TwoD=True)



