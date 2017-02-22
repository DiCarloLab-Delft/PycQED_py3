import numpy as np
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
import qcodes as qc
from pycqed.analysis import tomography as tomo

from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
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
        cross_talk_suppression=True)

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
        cross_talk_suppression=True)
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
    #     cross_talk_suppression=True)

    detector = det.UHFQC_integration_logging_det(
        UHFQC=qubit0._acquisition_instr,
        AWG=station.AWG,
        channels=[qubit0.RO_acq_weight_function_I(),
                  qubit1.RO_acq_weight_function_I()],
        nr_shots=256,
        integration_length=qubit0.RO_acq_integration_length(),
        cross_talk_suppression=True)

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
        cross_talk_suppression=True)

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
        cross_talk_suppression=True)
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
        cross_talk_suppression=True)

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
