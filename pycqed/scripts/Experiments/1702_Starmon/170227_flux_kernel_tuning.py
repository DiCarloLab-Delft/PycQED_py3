import numpy as np
from pycqed.measurement.waveform_control_CC import waveform as wf
import logging
import qcodes as qc
station = qc.station
from pycqed.measurement.waveform_control import viewer



k0 = station.components['k0']
QWG = station.components['QWG']

QWG.run_mode('CODeword')
QWG.stop()
QWG.deleteWaveformAll()

for ch in range(4):
    QWG.set('ch{}_state'.format(ch+1), True)

for tr in range(8):
    QWG.set('tr{}_trigger_level'.format(tr+1), 1.5)


block_I, block_Q = wf.block_pulse(1, 10e-9, sampling_rate=1e9)
block_I = np.array(block_I)
block_Q = np.array(block_Q)
for i in range(8):
    QWG.createWaveformReal('wf_I_{}'.format(i), block_I*(i+1)/10)
    QWG.createWaveformReal('wf_Q_{}'.format(i), block_Q*(i+1)/10)
    print('CW: ', i)
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 1), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 2), 'wf_Q_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 3), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 4), 'wf_Q_{}'.format(i))




square_pulse_cw = 5

block_I, block_Q = wf.block_pulse(1, 4e-6, sampling_rate=1e9)
block_I = np.array(block_I)
block_Q = np.array(block_Q)

# # distorted_I = k0.convolve_kernel([block_I, k0.kernel()], length=5e-6)

QWG.createWaveformReal('wf_I_{}'.format(square_pulse_cw), block_I)
print('QWG_square_pulse to CW: ', square_pulse_cw)
QWG.set('codeword_{}_ch{}_waveform'.format(square_pulse_cw, 1),
        'wf_I_{}'.format(square_pulse_cw))
QWG.set('codeword_{}_ch{}_waveform'.format(square_pulse_cw, 2),
        'wf_Q_{}'.format(square_pulse_cw))


QWG.start()
QWG.getOperationComplete()


for i in range(QWG.getSystemErrorCount()):
    logging.warning(QWG.getError())

from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import sequence
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt


def QWG_square_flux_pulse_seq(operation_dict, verbose=False):
    seq_name = 'QWG_square_flux_pulse_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']
    pulse_combinations = ['Start_pulse', 'CW_5']
    # for i in range(8):
    #     pulse_combinations += ['CW_{}'.format(i)]
    pulses = []
    for p in pulse_combinations:
        pulses += [operation_dict[p]]

    el = multi_pulse_elt(0, station, pulses, sequencer_config)
    el_list.append(el)
    seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq, el_list


operation_dict = {'Start_pulse': {'operation_type': 'other',
                                  'pulse_type': 'SquarePulse',
                                  'channels': ['ch3_marker1'],
                                  'amplitude': 1,
                                  'length': 40e-9,
                                  'pulse_delay': 20e-9},
                  'sequencer_config': None}
for i in range(10):
    operation_dict['CW_{}'.format(i)] = {'pulse_type': 'QWG_Codeword',
                                         'operation_type': 'other',
                                         'cw_trigger_channel': 'ch1_marker1',
                                         # 'cw_channels': ['ch1_marker2', 'ch2_marker1',
                                         #                 'ch2_marker2'],
                                         'cw_channels': ['ch2_marker2', 'ch2_marker1',
                                                         'ch1_marker2'],
                                         'amplitude': 1,
                                         'codeword': i,
                                         'length': 40e-9,
                                         'pulse_delay': 20e-9}

QWG_square_flux_pulse_seq(operation_dict)
AWG.start()

try:
    vw.clear()
except:
    vw = None

channels = ['ch1_marker1', 'ch1_marker2',
            'ch2_marker1', 'ch2_marker2',
            'ch3_marker1']
for i, elt_idx in enumerate([0]):
    vw = viewer.show_element_pyqt(
        station.pulsar.last_elements[elt_idx], vw,
        channels=channels,
        color_idx=i % len(viewer.color_cycle))


##############