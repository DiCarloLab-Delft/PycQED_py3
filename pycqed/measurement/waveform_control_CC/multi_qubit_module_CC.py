from pycqed.measurement.waveform_control_CC import multi_qubit_qasm_seqs as mqqs
from pycqed.measurement.waveform_control_CC import qasm_helpers as qh
from pycqed.measurement import detector_functions as det

import numpy as np
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import multiplexed_RO_analysis as mra
import qcodes as qc

##################
# This is the prepare for multiplexed RO snippet !!
##################

# LutManMan.acquisition_delay(q0.RO_acq_marker_delay())
# # Generate multiplexed pulse
# multiplexed_wave = [['RO_LutMan_QR', 'M_square'], ['RO_LutMan_QL', 'M_square']]
# LutManMan.generate_multiplexed_pulse(multiplexed_wave)
# LutManMan.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')


def measure_two_qubit_AllXY(device, q0_name, q1_name,
                            sequence_type='sequential', MC=None):
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

    s = qh.QASM_Sweep(AllXY.name, device.seq_contr.get_instr(), op_dict)
    d = det.UHFQC_integrated_average_detector(
        device.acquisition_instrument.get_instr(),
        AWG=device.seq_contr.get_instr(),
        nr_averages=q0.RO_acq_averages(),
        integration_length=q0.RO_acq_integration_length(),
        channels=[0, 1])
    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(21*(1+double_points)))
    MC.set_detector_function(d)
    MC.run('AllXY_{}_{}'.format(q0_name, q1_name))
    ma.MeasurementAnalysis()


def measure_two_qubit_ssro(device, q0_name, q1_name, nr_shots=4092*4,
                           MC=None, no_scaling=False,
                           crosstalk_suppression=False):
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

    s = qh.QASM_Sweep(two_q_ssro.name, device.seq_contr.get_instr(), op_dict)
    d = det.UHFQC_integration_logging_det(
        device.acquisition_instrument.get_instr(),
        AWG=device.seq_contr.get_instr(),
        nr_shots=4092,
        integration_length=q0.RO_acq_integration_length(),
        crosstalk_suppression=crosstalk_suppression,
        channels=[q0.RO_acq_weight_function_I(),
                  q1.RO_acq_weight_function_I(), 2])
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
    mra.two_qubit_ssro_fidelity('SSRO_{}_{}'.format(q0_name, q1_name))
    MC.soft_avg(old_soft_avg)
    MC.live_plot_enabled(old_live_plot_enabled)


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


def measure_CZ_calibration(device, q0_name, q1_name):
    '''
    Measure calibration sequence for C-phase gate, sweeping the phase of the
    recovery pulse.
    '''
    raise NotImplementedError()
