import numpy as np
import matplotlib.pyplot as plt
import logging

import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.detector_functions as det
import pycqed.measurement.composite_detector_functions as cdet
import pycqed.analysis.measurement_analysis as ma
import pycqed.analysis.tomography as tomo
from pycqed.measurement.optimization import nelder_mead
import pycqed.instrument_drivers.meta_instrument.device_object as device

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
        #qb.RO_pulse_type('Multiplexed_pulse_UHFQC')
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

    device.multiplexed_pulse(qubits, f_LO, upload=True,
                              plot_filename=plot_filename)

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
        device.get_multiplexed_readout_pulse_dictionary(qubits),
        preselection=preselection,
        parallel_pulses=True,
        RO_spacing=ro_spacing)

    m = 2 ** (len(qubits))
    if preselection:
        m *= 2
    shots = 4094 - 4094 % m
    df = device.get_multiplexed_readout_detector_functions(qubits,
             nr_shots=shots)['int_log_det']

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

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
                         MC=None, upload=True, sequence='reset_g'):
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
        RO_pars=device.get_multiplexed_readout_pulse_dictionary(qubits),
        feedback_delay=feedback_delay,
        #sequence=sequence,
        nr_resets=nr_resets,
        upload=upload)

    m = 2 ** (len(qubits))
    m *= (nr_resets + 1)
    shots = 4094 - 4094 % m
    df = device.get_multiplexed_readout_detector_functions(qubits,
             nr_shots=shots)['int_log_det']

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


def measure_two_qubit_parity(qb0, qb1, qb2, feedback_delay, f_LO, upload=True,
                             MC=None, nr_averages=2**10, prep_sequence=None,
                             tomography_basis=(
                                 'I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')):
    """
    Important things to check when running the experiment:
        Is the readout separation commensurate with 225 MHz?
    """

    device.multiplexed_pulse([(qb1,), (qb0, qb1, qb2)], f_LO)


    qubits = [qb0, qb1, qb2]
    for qb in qubits:
        if MC is None:
            MC = qb.MC
        else:
            break

    for qb in qubits:
        qb.prepare_for_timedomain(multiplexed=True)

    sf = awg_swf2.two_qubit_parity(qb0, qb1, qb2, feedback_delay=feedback_delay,
                                   prep_sequence=prep_sequence,
                                   tomography_basis=tomography_basis,
                                   upload=upload, verbose=False)

    df = device.get_multiplexed_readout_detector_functions(
        qubits, nr_averages=nr_averages, correlations=[(
            qb0.RO_acq_weight_function_I(), qb2.RO_acq_weight_function_I()
        )])['int_corr_det']

    MC.set_sweep_function(sf)
    MC.set_sweep_points(2*len(tomography_basis)**2)
    MC.set_detector_function(df)

    MC.run(name='two_qubit_parity-{}'.format('_'.join([qb.name for qb in
                                                       qubits])))



def measure_n_qubit_simultaneous_randomized_benchmarking(
        qubits, f_LO,
        nr_cliffords=None, nr_seeds=50,
        CxC_RB=True, idx_for_RB=0,
        gate_decomp='HZ', interleaved_gate=None,
        CZ_info_dict=None, interleave_CZ=True,
        thresholding=True,  V_th_a=None,
        nr_averages=1024, soft_avgs=1,
        MC=None, UHFQC=None, pulsar=None,
        label=None, verbose=False):

    '''
    Performs a simultaneous randomized benchmarking experiment on n qubits.
    type(nr_cliffords) == array
    type(nr_seeds) == int

    Args:
        qubit_list (list): list of qubit objects to perfomr RB on
        f_LO (float): readout LO frequency
        nr_cliffords (numpy.ndarray): numpy.arange(max_nr_cliffords), where
            max_nr_cliffords is the number of Cliffords in the longest seqeunce
            in the RB experiment
        nr_seeds (int): the number of times to repeat each Clifford sequence of
            length nr_cliffords[i]
        CxC_RB (bool): whether to perform CxCxCx..xC RB or
            (CxIx..xI, IxCx..XI, ..., IxIx..xC) RB
        idx_for_RB (int): if CxC_RB==False, refers to the index of the
            pulse_pars in pulse_pars_list which will undergo the RB protocol
            (i.e. the position of the Z operator when we measure
            ZIII..I, IZII..I, IIZI..I etc.)
        gate_decomposition (str): 'HZ' or 'XY'
        interleaved_gate (str): used for regular single qubit Clifford IRB
            string referring to one of the gates in the single qubit
            Clifford group
        CZ_info_dict (dict): dict indicating which qbs in the CZ gates are the
            control and the target. Can have the following forms:
            either:    {'qbc': qb_control_name_CZ,
                        'qbt': qb_target_name_CZ}
            if only one CZ gate is interleaved;
            or:       {'CZi': {'qbc': qb_control_name_CZ,
                               'qbt': qb_target_name_CZ}}
            if multiple CZ gates are interleaved; CZi = [CZ0, CZ1, ...] where
            CZi is the c-phase gate between qbc->qbt.
            (We start counting from zero because we are programmers.)
        interleave_CZ (bool): Only used if CZ_info_dict != None
            True -> interleave the CZ gate
            False -> interleave the ICZ gate
        thresholding (bool): whether to use the thresholding feature
            of the UHFQC
        V_th_a (list or tuple): contains the SSRO assignment thresholds for
            each qubit in qubits. These values must be correctly scaled!
        nr_averages (float): number of averages to use
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

    if label is None:
        if CxC_RB:
            label = '{}-qubit_CxC_RB_{}_{}_seeds_{}_cliffords'.format(
                len(qubits), gate_decomp, nr_seeds, nr_cliffords[-1])
        else:
            label = '{}-qubit_CxI_IxC_{}_RB_{}_{}_seeds_{}_cliffords'.format(
                len(qubits), idx_for_RB, gate_decomp,
                nr_seeds, nr_cliffords[-1])

    if CZ_info_dict is not None:
        if interleave_CZ:
            label += '_CZ'
        else:
            label += '_noCZ'

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

    multiplexed_pulse(qubits, f_LO, upload=True, plot_filename=True)
    RO_pars = get_multiplexed_readout_pulse_dictionary(qubits)

    if len(qubits) == 2:
        if len(nr_cliffords)==1:
            raise ValueError('For a two qubit experiment, nr_cliffords must '
                             'be an array of sequence lengths.')

        correlations = [(qubits[0].RO_acq_weight_function_I(),
                         qubits[1].RO_acq_weight_function_I())]

        det_func = get_multiplexed_readout_detector_functions(
            qubits, nr_averages=nr_averages, UHFQC=UHFQC,
            pulsar=pulsar, correlations=correlations)[key+'_corr_det']

        hard_sweep_points = np.arange(nr_seeds)
        hard_sweep_func = \
            awg_swf2.two_qubit_Simultaneous_RB_fixed_length(
                qubit_list=qubits, RO_pars=RO_pars,
                nr_cliffords_value=nr_cliffords[0], CxC_RB=CxC_RB,
                idx_for_RB=idx_for_RB, upload=False,
                gate_decomposition=gate_decomp,  CZ_info_dict=CZ_info_dict,
                interleaved_gate=interleaved_gate,
                verbose=verbose, interleave_CZ=interleave_CZ)

        soft_sweep_points = nr_cliffords
        soft_sweep_func = \
            awg_swf2.two_qubit_Simultaneous_RB_sequence_lengths(
            n_qubit_RB_sweepfunction=hard_sweep_func)

    else:
        if not isinstance(nr_cliffords, float) or \
                not isinstance(nr_cliffords, int):
                    raise ValueError('For an experiment with more than two '
                                     'qubits, nr_cliffords must int or float.')

        det_func = get_multiplexed_readout_detector_functions(
            qubits, UHFQC=UHFQC, pulsar=pulsar,
            nr_shots=nr_averages)[key+'_log_det']

        hard_sweep_points = np.arange(nr_averages)
        hard_sweep_func = \
            awg_swf2.two_qubit_Simultaneous_RB_fixed_length(
                qubit_list=qubits, RO_pars=RO_pars,
                nr_cliffords_value=nr_cliffords, CxC_RB=CxC_RB,
                idx_for_RB=idx_for_RB, upload=True,
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate, interleave_CZ=interleave_CZ,
                verbose=verbose, CZ_info_dict=CZ_info_dict)

        soft_sweep_points = np.arange(nr_seeds)
        soft_sweep_func = swf.None_Sweep()

    MC.soft_avg(soft_avgs)
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(hard_sweep_points)

    MC.set_sweep_function_2D(soft_sweep_func)
    MC.set_sweep_points_2D(soft_sweep_points)

    MC.set_detector_function(det_func)
    MC.run_2D(label)

    ma.TwoD_Analysis(close_file=True)

def measure_two_qubit_tomo_Bell(bell_state, qb_c, qb_t, f_LO,
                                basis_pulses=None,
                                cal_state_repeats=7, spacing=100e-9,
                                CZ_disabled=False,
                                num_flux_pulses=0, verbose=False,
                                nr_averages=1024, soft_avgs=1,
                                MC=None, UHFQC=None, pulsar=None,
                                upload=True, label=None, run=True):
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
            CURRENTLY NOT USED!
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

    Bell_state_dict = {'0':'phiMinus', '1':'phiPlus',
                       '2':'psiMinus', '3':'psiPlus'}
    if label is None:
        label = '{}_tomo_Bell_{}_{}'.format(
            Bell_state_dict[str(bell_state)], qb_c.name, qb_t.name)

    if num_flux_pulses!=0:
        label += '_' + str(num_flux_pulses) + 'preFluxPulses'

    multiplexed_pulse(qubits, f_LO, upload=True, plot_filename=True)
    RO_pars = get_multiplexed_readout_pulse_dictionary(qubits)

    correlations = [(qubits[0].RO_acq_weight_function_I(),
                     qubits[1].RO_acq_weight_function_I())]

    corr_det = get_multiplexed_readout_detector_functions(
        qubits, nr_averages=nr_averages, UHFQC=UHFQC,
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

    multiplexed_pulse(qubits, f_LO, upload=True, plot_filename=True)
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

    ma.TwoD_Analysis(close_file=True)

def cphase_gate_tuneup(qb_control, qb_target,
                       initial_values_dict=None,
                       initial_step_dict=None,
                       MC_optimization=None,
                       MC_detector=None,
                       maxiter=50,
                       name='cphase_tuneup',
                       cal_points=False,
                       spacing=10e-9,
                       ramsey_phases=None,
                       distorted=False,
                       distortion_dict=None):

    '''
    function that runs the nelder mead algorithm to optimize the CPhase gate
    parameters (pulse lengths and pulse amplitude)

    Args:
        qb_control: control qubit (with flux pulses)
        qb_target: target qubit
        initial_values_dict:
        initial_step_dict:
        MC_optimization:
        MC_detector:
        name:


    Returns:

    '''
    if MC_optimization is None:
        MC_optimization = station.MC

    if initial_values_dict is None:
        pulse_length_init = qb_control.CZ_pulse_length()
        pulse_amplitude_init = qb_control.CZ_pulse_amp()
    else:
        pulse_length_init = initial_values_dict['length']
        pulse_amplitude_init = initial_values_dict['amplitude']

    if initial_step_dict is None:
        init_step_length = 1.0e-9
        init_step_amplitude = 1.0e-3
    else:
        init_step_length = initial_step_dict['step_length']
        init_step_amplitude = initial_step_dict['step_length']

    d = cdet.CPhase_optimization(qb_control=qb_control,qb_target=qb_target,
                                 MC=MC_detector,
                                 ramsey_phases=ramsey_phases)


    S1 = d.flux_pulse_length
    S2 = d.flux_pulse_amp

    # S1_dummy = awg_swf.Flux_pulse_CPhase_meas_hard_swf(qb_control,qb_target,
    #                                              sweep_mode='length',
    #                                              spacing=spacing,
    #                                              cal_points=False,
    #                                              distorted=False,
    #                                              distortion_dict=None,
    #                                              upload=False)
    #
    # S1 = awg_swf.Flux_pulse_CPhase_meas_2D(qb_control,qb_target, S1_dummy,
    #                                        sweep_mode='length',
    #                                        measurement_mode='excited_state')
    # S2 = awg_swf.Flux_pulse_CPhase_meas_2D(qb_control,qb_target, S1_dummy,
    #                                        sweep_mode='amplitude',
    #                                        measurement_mode='excited_state')

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [pulse_length_init,pulse_amplitude_init],
                    'initial_step': [init_step_length, init_step_amplitude],
                    'no_improv_break': 12,
                    'minimize': True,
                    'maxiter': maxiter}

    MC_optimization.set_sweep_functions([S2, S1])
    MC_optimization.set_detector_function(d)
    MC_optimization.set_adaptive_function_parameters(ad_func_pars)
    MC_optimization.run(name=name, mode='adaptive')
    a = ma.OptimizationAnalysis_v2()
    pulse_length_best_value = a.a.optimization_result[0][0]
    pulse_amplitude_best_value = a.a.optimization_result[0][1]

    return pulse_length_best_value, pulse_amplitude_best_value