import numpy as np
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.analysis import measurement_analysis as ma
import qcodes as qc
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
station = qc.station


def measure_chevron(device, q0_name, amps, length, MC=None):
    # Might belong better in some fluxing module but that does not exist yet
    if MC is None:
        MC = qc.station.components['MC']

    if len(amps) == 1:
        slice_scan = True
    else:
        slice_scan = False

    operation_dict = device.get_operation_dict()

    # preparation of sweep points and cal points
    cal_points = 4
    lengths_cal = length[-1] + \
        np.arange(1, 1+cal_points)*(length[1]-length[0])
    lengths_vec = np.concatenate((length, lengths_cal))

    # start preparations
    q0 = device.qubits()[q0_name]
    q0.prepare_for_timedomain()
    dist_dict = q0.dist_dict()

    chevron_swf = awg_swf.awg_seq_swf(
        fsqs.chevron_seq,
        parameter_name='pulse_lengths',
        AWG=q0.AWG,
        fluxing_channels=[q0.fluxing_channel()],
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'q0': q0_name,
                             'distortion_dict': q0.dist_dict()})

    MC.set_sweep_function(chevron_swf)
    MC.set_sweep_points(lengths_vec)
    if not slice_scan:
        MC.set_sweep_function_2D(
            q0.AWG.parameters[q0.fluxing_channel()+'_amp'])
        MC.set_sweep_points_2D(amps)

    MC.set_detector_function(q0.int_avg_det_rot)
    if slice_scan:
        q0.AWG.parameters[q0.fluxing_channel()+'_amp'].set(amps[0])
        MC.run('Chevron_slice_%s' % q0.name)
        ma.TD_Analysis(auto=True)
    else:
        MC.run('Chevron_2D_%s' % q0.name, mode='2D')
        ma.Chevron_2D(auto=True)


def measure_SWAPN(device, q0_name, swap_amps,
                  number_of_swaps=30, MC=None):
    if MC is None:
        MC = qc.station.components['MC']
    q0 = device.qubits()[q0_name]
    # These are the sweep points
    swap_vec = np.arange(number_of_swaps)*2
    cal_points = 4
    lengths_cal = swap_vec[-1] + \
        np.arange(1, 1+cal_points)*(swap_vec[1]-swap_vec[0])
    swap_vec = np.concatenate((swap_vec, lengths_cal))

    operation_dict = device.get_operation_dict()
    AWG = q0.AWG

    repSWAP = awg_swf.awg_seq_swf(
        fsqs.SwapN,
        parameter_name='nr_pulses_list',
        unit='#',
        AWG=q0.AWG,
        fluxing_channels=[q0.fluxing_channel()],
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'q0': q0_name,
                             'distortion_dict': q0.dist_dict()})

    MC.set_sweep_function(repSWAP)
    MC.set_sweep_points(swap_vec)

    MC.set_sweep_function_2D(AWG.ch4_amp)
    MC.set_sweep_points_2D(swap_amps)

    MC.set_detector_function(q0.int_avg_det_rot)
    MC.run('SWAPN_%s' % q0.name, mode='2D')
    ma.TwoD_Analysis(auto=True)


def measure_SWAPN_alpha(device, q0_name, swap_amps, alpha,
                        number_of_swaps=30, MC=None):
    if MC is None:
        MC = qc.station.components['MC']
    q0 = device.qubits()[q0_name]
    # These are the sweep points
    swap_vec = np.arange(number_of_swaps)*2
    cal_points = 4
    lengths_cal = swap_vec[-1] + \
        np.arange(1, 1+cal_points)*(swap_vec[1]-swap_vec[0])
    swap_vec = np.concatenate((swap_vec, lengths_cal))

    operation_dict = device.get_operation_dict()
    AWG = q0.AWG

    repSWAP = awg_swf.awg_seq_swf(
        fsqs.SwapN_alpha,
        parameter_name='nr_pulses_list',
        unit='#',
        AWG=q0.AWG,
        fluxing_channels=[q0.fluxing_channel()],
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'q0': q0_name,
                             'alpha': alpha,
                             'distortion_dict': q0.dist_dict()})

    MC.set_sweep_function(repSWAP)
    MC.set_sweep_points(swap_vec)

    MC.set_sweep_function_2D(AWG.ch4_amp)
    MC.set_sweep_points_2D(swap_amps)

    MC.set_detector_function(q0.int_avg_det_rot)
    MC.run('SWAPN_%s' % q0.name, mode='2D')
    ma.TwoD_Analysis(auto=True)


def measure_BusT1(device, q0_name, times, MC=None):
    if MC is None:
        MC = qc.station.components['MC']
    q0 = device.qubits()[q0_name]
    cal_points = 4
    cal_pts = times[-1] + \
        np.arange(1, 1+cal_points)*(times[1]-times[0])
    times = np.concatenate((times, cal_pts))

    operation_dict = device.get_operation_dict()
    AWG = q0.AWG

    busT1swf = awg_swf.awg_seq_swf(
        fsqs.BusT1,
        parameter_name='times',
        unit='s',
        AWG=q0.AWG,
        fluxing_channels=[q0.fluxing_channel()],
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'q0': q0_name,
                             'distortion_dict': q0.dist_dict()})

    MC.set_sweep_function(busT1swf)
    MC.set_sweep_points(times)

    MC.set_detector_function(q0.int_avg_det_rot)
    MC.run('BusT1_{}'.format(q0.name))
    ma.T1_Analysis(label='BusT1')


def measure_FluxTrack(device, q0_name, amps, fluxes, MC=None):
    if MC is None:
        MC = qc.station.components['MC']
    q0 = device.qubits()[q0_name]

    operation_dict = device.get_operation_dict()
    AWG = q0.AWG

    FluxTrack_det = cdet.FluxTrack(qubit=q0,
                                   MC=qc.station.components['MC_nested'],
                                   AWG=AWG)

    MC.set_sweep_function(AWG.ch4_amp)
    MC.set_sweep_points(amps)

    MC.set_sweep_function_2D(Flux_Control)
    MC.set_sweep_points_2D(fluxes)

    MC.set_detector_function(q0.int_avg_det_rot)
    MC.run('FluxTrack_%s' % q0.name, mode='2D')
    ma.TwoD_Analysis(auto=True)
