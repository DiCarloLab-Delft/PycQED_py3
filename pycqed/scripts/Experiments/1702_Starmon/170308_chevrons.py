import numpy as np
import qcodes as qc
from pycqed.measurement.waveform_control import sequence
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.standard_elements import distort_and_compensate

station = qc.station


def chevron_seq(operation_dict,
                pulse_qubit,
                swap_qubit,
                RO_target,
                pulse_lengths=np.arange(0, 120e-9, 2e-9),
                verbose=False,
                distortion_dict=None,
                upload=True,
                cal_points=True):
    '''
    Chevron sequence where length of the "SWAP" operation is varied
        X180 - SWAP(l) - RO


    verbose=False:        (bool) used for verbosity printing in the pulsar
    distortion_dict=None: (dict) flux_pulse predistortion kernels
    upload=True:          (bool) uploads to AWG, set False for testing purposes
    cal_points=True:      (bool) wether to use calibration points
    '''

    seq_name = 'Chevron_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']

    SWAP_amp = operation_dict['SWAP '+swap_qubit]['amplitude']
    # seq has to have at least 2 elts
    for i, pulse_length in enumerate(pulse_lengths):
        # this converts negative pulse lenghts to negative pulse amplitudes
        if pulse_length < 0:
            operation_dict['SWAP '+swap_qubit]['amplitude'] = -SWAP_amp
        else:
            operation_dict['SWAP '+swap_qubit]['amplitude'] = SWAP_amp

        if cal_points and (i == (len(pulse_lengths)-4) or
                           i == (len(pulse_lengths)-3)):
            pulse_combinations = ['RO '+RO_target]
        elif cal_points and (i == (len(pulse_lengths)-2) or
                             i == (len(pulse_lengths)-1)):
            pulse_combinations = ['X180 ' + pulse_qubit, 'RO ' + RO_target]
        else:
            operation_dict[
                'SWAP '+swap_qubit]['square_pulse_length'] = abs(pulse_length)
            pulse_combinations = ['X180 '+pulse_qubit, 'SWAP ' + swap_qubit,
                                  'RO '+RO_target]

        pulses = []
        for p in pulse_combinations:
            pulses += [operation_dict[p]]

        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\r Distorting element {}/{}'.format(i+1, len(pulse_lengths)),
                  end='')
            if i == len(pulse_lengths):
                print()
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)

    return seq, el_list





def measure_chevron(device, pulse_qubit, swap_qubit, RO_target, amps, length,
                    source=None,
                    MC=None):
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
    q0 = device.qubits()[pulse_qubit]
    q1 = device.qubits()[swap_qubit]
    q2 = device.qubits()[RO_target]

    q0.prepare_for_timedomain()
    q1.prepare_for_timedomain()
    q2.prepare_for_timedomain()

    q0.td_source.frequency(q0.f_qubit()-q0.f_pulse_mod())
    q0.td_source.on()
    if source is not None:
        source.on()
    dist_dict = q1.dist_dict()

    chevron_swf = awg_swf.awg_seq_swf(
        chevron_seq,
        parameter_name='pulse_lengths',
        AWG=q0.AWG,
        fluxing_channels=[q1.fluxing_channel()],
        awg_seq_func_kwargs={'operation_dict': operation_dict,
                             'pulse_qubit': pulse_qubit,
                             'swap_qubit': swap_qubit,
                             'RO_target': RO_target,
                             'distortion_dict': q1.dist_dict()})

    MC.set_sweep_function(chevron_swf)
    MC.set_sweep_points(lengths_vec)
    if not slice_scan:
        MC.set_sweep_function_2D(
            q1.AWG.parameters[q1.fluxing_channel()+'_amp'])
        MC.set_sweep_points_2D(amps)

    MC.set_detector_function(q2.int_avg_det_rot)
    if slice_scan:
        q1.AWG.parameters[q1.fluxing_channel()+'_amp'].set(amps[0])
        MC.run('Chevron_slice_%s' % q1.name)
        ma.TD_Analysis(auto=True)
    else:
        MC.run('Chevron_2D_%s' % q1.name, mode='2D')
        ma.Chevron_2D(auto=True)



###########
QL.RO_acq_averages(512)
# Running the experiment
QL_LO.frequency(QR.f_qubit())
length=np.arange(0, .3e-6, 5e-9)
amps = np.arange(1.285, 1.31, 0.001)
measure_chevron(Starmon, pulse_qubit='QL', swap_qubit='QR', RO_target='QL',
                source=QL_LO,
                amps=amps,
                length=length,
                MC=station.MC)



length=np.arange(0, .3e-6, 5e-9)
amps = np.arange(1.285, 1.31, 0.001)
measure_chevron(Starmon, pulse_qubit='QL', swap_qubit='QR', RO_target='QL',
                source=QL_LO,
                amps=amps,
                length=length,
                MC=station.MC)

 k0.kernel_list(['RT_Compiled_170308.txt'])
QR.dist_dict({'ch_list': ['ch2'], 'ch2': k0.kernel()})


length=np.arange(0, .6e-6, 5e-9)
amps = np.arange(1.265, 1.30, 0.001)

MC.soft_avg(2)
for i in range(100):
    measure_chevron(Starmon, pulse_qubit='QL', swap_qubit='QR', RO_target='QL',
                    source=QL_LO,
                    amps=amps,
                    length=length,
                    MC=station.MC)