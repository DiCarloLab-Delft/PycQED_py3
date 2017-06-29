from pycqed.measurement.waveform_control_CC import multi_qubit_qasm_seqs as mqqs
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

import numpy as np
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import multiplexed_RO_analysis as mra
import qcodes as qc


def measure_two_qubit_AllXY(device, q0_name, q1_name,
                            sequence_type='sequential', MC=None,
                            result_logging_mode='lin_trans'):
    if MC is None:
        MC = qc.station.components['MC']

    q0 = device._qubits[q0_name]
    q1 = device._qubits[q1_name]

    q0.prepare_for_timedomain()
    q1.prepare_for_timedomain()

    # device.prepare_multiplexed_RO() # <--- not implemented yet
    double_points = True
    AllXY = mqqs.two_qubit_AllXY(q0_name, q1_name,
                                 RO_target=q0_name,
                                 sequence_type=sequence_type,
                                 replace_q1_pulses_X180=False,
                                 double_points=double_points)

    op_dict = device.get_operation_dict()
    for q in device.qubits():
        op_dict['I ' + q]['instruction'] = ''

    s = swf.QASM_Sweep(AllXY.name, device.seq_contr.get_instr(), op_dict)
    d = det.UHFQC_integrated_average_detector(
        device.acquisition_instrument.get_instr(),
        AWG=device.seq_contr.get_instr(),
        nr_averages=q0.RO_acq_averages(),
        integration_length=q0.RO_acq_integration_length(),
        result_logging_mode=result_logging_mode,
        channels=[q0.RO_acq_weight_function_I(),
                  q1.RO_acq_weight_function_I()])
    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(21*(1+double_points)))
    MC.set_detector_function(d)
    MC.run('AllXY_{}_{}'.format(q0_name, q1_name))
    ma.MeasurementAnalysis()


def measure_two_qubit_ssro(device, q0_name, q1_name, nr_shots=4092*4,
                           MC=None, no_scaling=False,
                           result_logging_mode='lin_trans'):
    # N.B. this function can be replaced with a more general multi-qubit ssro
    if MC is None:
        MC = qc.station.components['MC']

    q0 = device._qubits[q0_name]
    q1 = device._qubits[q1_name]

    q0.prepare_for_timedomain()
    q1.prepare_for_timedomain()

    # device.prepare_multiplexed_RO() # <--- not implemented yet
    two_q_ssro = mqqs.two_qubit_off_on(q0_name, q1_name,
                                       RO_target=q0_name)
    # FIXME: to be replaced with all

    op_dict = device.get_operation_dict()
    for q in device.qubits():
        op_dict['I ' + q]['instruction'] = ''

    s = swf.QASM_Sweep(two_q_ssro.name, device.seq_contr.get_instr(), op_dict)
    d = det.UHFQC_integration_logging_det(
        device.acquisition_instrument.get_instr(),
        AWG=device.seq_contr.get_instr(),
        nr_shots=4092,
        integration_length=q0.RO_acq_integration_length(),
        result_logging_mode=result_logging_mode,
        channels=[q0.RO_acq_weight_function_I(),
                  q1.RO_acq_weight_function_I()])
    if no_scaling:
        d.scaling_factor = 1

    old_soft_avg = MC.soft_avg()
    old_live_plot_enabled = MC.live_plot_enabled()
    MC.soft_avg(1)
    MC.live_plot_enabled(False)

    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(nr_shots))
    MC.set_detector_function(d)
    MC.run('SSRO_{}_{}'.format(q0_name, q1_name))
    MC.soft_avg(old_soft_avg)
    MC.live_plot_enabled(old_live_plot_enabled)
    return mra.two_qubit_ssro_fidelity('SSRO_{}_{}'.format(q0_name, q1_name))


def measure_Ram_Z(device, q0_name, q1_name):
    '''
    Measure Ram-Z sequence with different wait times after triggering the QWG
    and between the two pi-half pulses.
    '''
    raise NotImplementedError()


def measure_chevron(device, q0_name, q1_name):
    '''
    Measure chevron pattern, sweeping amplitude and duration of the flux
    pulse.
    '''
    raise NotImplementedError()

def measure_chevron(device, q0_name, q1_name,
                    CC, MC,
                    QWG_flux_lutman, #operation_dict,
                    amps=np.arange(0.482, .491, .0010),
                    lengths=np.arange(50e-9, 500e-9, 5e-9),
                    wait_during_flux=400e-8,
                    wait_after_trigger=40e-9,
                    excite_q1=True):
    '''
    Measure chevron for the qubits q0 and q1.

    Args:
        device:  device object
        q0_name : name of the first qubit. q0 will be flux pulsed
        q1_name : name of the second qubit (what is this needed for? )

    '''
    CC = device.central_controller.get_instr()
    operation_dict = Starmon.get_operation_dict()
    single_pulse_elt = mqqs.chevron_seq(q0.name, q1.name, excite_q1=excite_q1,
                                        RO_target=q0.name,
                                        wait_during_flux=wait_during_flux,
                                        wait_after_trigger=wait_after_trigger)
    # single_pulse_elt = flux_pulse_seq('QR', 'QL')
    single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name, operation_dict)
    qumis_file = single_pulse_asm
    CC.load_instructions(qumis_file.name)
    CC.start()

    s1 = swf.QWG_lutman_par(QWG_flux_lutman, QWG_flux_lutman.F_amp)
    s2 = swf.QWG_lutman_par(QWG_flux_lutman, QWG_flux_lutman.F_length)
    MC.soft_avg(1)
    MC.set_sweep_function(s1)
    MC.set_sweep_function_2D(s2)
    MC.set_sweep_points(amps)
    MC.set_sweep_points_2D(lengths)
    d = det.UHFQC_integrated_average_detector(
        UHFQC=q0._acquisition_instrument, AWG=CC,
        channels=[
            q0.RO_acq_weight_function_I(),
            q1.RO_acq_weight_function_I()],
        nr_averages=q0.RO_acq_averages(),
        real_imag=True, single_int_avg=True,
        result_logging_mode='lin_trans',
        integration_length=q0.RO_acq_integration_length(),
        seg_per_point=1)

    d._set_real_imag(True)

    MC.set_detector_function(d)
    MC.run('Chevron_{}_{}'.format(q0.name, q1.name), mode='2D')




def measure_CZ_calibration(device, q0_name, q1_name):
    '''
    Measure calibration sequence for C-phase gate, sweeping the phase of the
    recovery pulse.
    '''
    raise NotImplementedError()
