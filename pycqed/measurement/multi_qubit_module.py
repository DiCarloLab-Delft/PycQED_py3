

import numpy as np
import matplotlib.pyplot as plt
import logging

import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.detector_functions as det
import pycqed.analysis.measurement_analysis as ma
import pycqed.analysis.tomography as tomo

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


def multiplexed_pulse(qubits, f_LO, upload=True, plot_filename=None):
    """
    Sets up a frequency-multiplexed pulse on the awg-sequencer of the UHFQC.
    Updates the qubit ro_pulse_type parameter. This needs to be reverted if
    thq qubit object is to update its readout pulse later on.

    Args:
        qubits: A list of qubits to do a pulse for.
        f_LO: The LO frequency that will be used.
        upload: Whether to update the hardware instrument settings.
        plot_filename: The file to save the plot of the multiplexed pulse PSD. 
            If `None` or `True`, plot is only shown, and not saved. If `False`,
            no plot is generated.

    Returns:
        The generated pulse waveform.
    """

    fs = 1.8e9

    pulses = {}
    maxlen = 0

    for qb in qubits:
        qb.RO_pulse_type('Multiplexed_pulse_UHFQC')
        qb.f_RO_mod(qb.f_RO() - f_LO)
        samples = int(qb.RO_pulse_length() * fs)
        tbase = np.linspace(0, samples / fs, samples, endpoint=False)
        pulse = qb.RO_amp() * np.exp(
            -2j * np.pi * qb.f_RO_mod() * tbase + 0.25j * np.pi)
        pulses[qb.name] = pulse
        if pulse.size > maxlen:
            maxlen = pulse.size

    pulse = np.zeros(maxlen, dtype=np.complex)
    for p in pulses.values():
        pulse += np.pad(p, (0, maxlen - p.size), mode='constant',
                        constant_values=0)

    if plot_filename is not False:
        pulse_fft = np.fft.fft(
            np.pad(pulse, (1000, 1000), mode='constant', constant_values=0))
        pulse_fft = np.fft.fftshift(pulse_fft)
        fbase = np.fft.fftfreq(pulse_fft.size, 1 / fs)
        fbase = np.fft.fftshift(fbase)
        fbase = f_LO - fbase
        y = 20 * np.log10(np.abs(pulse_fft))
        plt.plot(fbase / 1e9, y, '-', lw=0.7)
        ymin, ymax = np.max(y) - 40, np.max(y) + 5
        plt.ylim(ymin, ymax)
        plt.vlines(f_LO / 1e9, ymin, ymax, colors='0.5', linestyles='dotted',
                   lw=0.7, )
        plt.text(f_LO / 1e9, ymax, 'LO', rotation=90, va='top', ha='right')
        for qb in qubits:
            plt.vlines([qb.f_RO() / 1e9, 2 * f_LO / 1e9 - qb.f_RO() / 1e9],
                       ymin, ymax, colors='0.5', linestyles=['solid', 'dotted'],
                       lw=0.7)
            plt.text(qb.f_RO() / 1e9, ymax, qb.name, rotation=90, va='top',
                     ha='right')
            plt.text(2 * f_LO / 1e9 - qb.f_RO() / 1e9, ymax, qb.name,
                     rotation=90, va='top', ha='right')

        plt.ylabel('power (dB)')
        plt.xlabel('frequency (GHz)')
        if isinstance(plot_filename, str):
            plt.savefig(plot_filename, bbox_inches='tight')
        plt.show()

    if upload:
        UHFQC = qubits[0].UHFQC
        UHFQC.awg_sequence_acquisition_and_pulse(np.real(pulse).copy(),
                                                 np.imag(pulse).copy(),
                                                 acquisition_delay=0)
        LO = qubits[0].readout_DC_LO
        LO.frequency(f_LO)


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

def get_multiplexed_readout_detector_functions(qubits, nr_averages=2**10,
                                               nr_shots=4095, UHFQC=None,
                                               pulsar=None, correlations=None):
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
            result_logging_mode='raw'),
        'dig_log_det': det.UHFQC_integration_logging_det(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_shots=nr_shots,
            result_logging_mode='digitized'),
        'int_avg_det': det.UHFQC_integrated_average_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_averages=nr_averages),
        'inp_avg_det': det.UHFQC_input_average_detector(
            UHFQC=UHFQC, AWG=pulsar, nr_averages=nr_averages, nr_samples=4096),
        'int_corr_det': det.UHFQC_correlation_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_averages=nr_averages,
            correlations=correlations),
        'dig_corr_det': det.UHFQC_correlation_detector(
            UHFQC=UHFQC, AWG=pulsar, channels=channels,
            integration_length=max_int_len, nr_averages=nr_averages,
            correlations=correlations, thresholding=True),
    }

def calculate_minimal_readout_spacing(qubits, ro_slack=10e-9, drive_pulses=0):
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
                                ro_spacing=None, preselection=True, MC=None,
                                plot_filename=False, ro_slack=10e-9):

    multiplexed_pulse(qubits, f_LO, upload=True, plot_filename=plot_filename)

    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    if ro_spacing is None:
        ro_spacing = calculate_minimal_readout_spacing(qubits, ro_slack,
                                                       drive_pulses=1)

    sf = awg_swf2.n_qubit_off_on(
        [qb.get_drive_pars() for qb in qubits],
        get_multiplexed_readout_pulse_dictionary(qubits),
        preselection=preselection,
        parallel_pulses=True,
        RO_spacing=ro_spacing)

    m = 2 ** (len(qubits))
    if preselection:
        m *= 2
    shots = 4094 - 4094 % m
    df = get_multiplexed_readout_detector_functions(qubits, nr_shots=shots) \
        ['int_log_det']

    for qb in qubits:
        qb.prepare_for_timedomain()

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)
    MC.run_2D('{}_multiplexed_ssro'.format('-'.join(
        [qb.name for qb in qubits])))


def measure_active_reset(qubits, feedback_delay, nr_resets=1, nreps=1,
                         MC=None, upload=True, sequence='reset_g',
                         readout_delay=None):
    """possible sequences: 'reset_g', 'reset_e', 'idle', 'flip'"""
    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    # if readout_delay is None:
    #     readout_delay = calculate_minimal_readout_spacing(qubits,
    #                                                       drive_pulses=0)

    sf = awg_swf2.n_qubit_reset(
        pulse_pars_list=[qb.get_drive_pars() for qb in qubits],
        RO_pars=get_multiplexed_readout_pulse_dictionary(qubits),
        feedback_delay=feedback_delay,
        readout_delay=readout_delay,
        #sequence=sequence,
        nr_resets=nr_resets,
        upload=upload)

    m = 2 ** (len(qubits))
    m *= (nr_resets + 1)
    shots = 4094 - 4094 % m
    df = get_multiplexed_readout_detector_functions(qubits, nr_shots=shots) \
        ['int_log_det']

    prev_avg = MC.soft_avg()
    MC.soft_avg(1)

    for qb in qubits:
        qb.prepare_for_timedomain()

    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(shots))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)

    MC.run_2D(name='active_reset_{}_{}'.format(sequence, '-'.join(
        [qb.name for qb in qubits])))

    MC.soft_avg(prev_avg)
