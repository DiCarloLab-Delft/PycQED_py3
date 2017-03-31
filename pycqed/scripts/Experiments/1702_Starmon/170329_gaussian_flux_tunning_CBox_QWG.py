import numpy as np
from pycqed.measurement.waveform_control_CC import waveform as wf
import logging
import qcodes as qc
station = qc.station
from pycqed.measurement.waveform_control import viewer


k0 = station.components['k0']
QWG = station.components['QWG']
CBox = station.components['CBox']

QWG.run_mode('CODeword')
QWG.stop()
QWG.deleteWaveformAll()

for ch in range(4):
    QWG.set('ch{}_state'.format(ch+1), True)


def set_trigger_level(val):
    for tr in range(8):
        QWG.set('tr{}_trigger_level'.format(tr+1), val)
set_trigger_level(.5)

block_I, block_Q = wf.block_pulse(1, 10e-9, sampling_rate=1e9)
block_I = np.array(block_I)
block_Q = np.array(block_Q)

amp = 1
for i in range(8):
    QWG.createWaveformReal('wf_I_{}'.format(i), amp * block_I*(i+1)/10)
    QWG.createWaveformReal('wf_Q_{}'.format(i), amp * block_Q*(i+1)/10)
    print('CW: ', i)
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 1), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 2), 'wf_Q_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 3), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 4), 'wf_Q_{}'.format(i))


# block_I, block_Q = wf.block_pulse(1, 10e-6, sampling_rate=1e9)

sigma = 600e-9
mod = 1e9
block_I, block_Q = wf.mod_gauss(1., sigma, mod, nr_sigma=8, sampling_rate=1e9,
                axis='x', motzoi=0, delay=0)
block_I = np.array(block_I)
block_Q = np.array(block_Q)

t0 = time.time()
block_I *= .5
# distorted_I = k0.convolve_kernel([k0.kernel(), block_I], length_samples=30e3)
distorted_I = block_I

block_cw = 5
comp_cw = 6
QWG.createWaveformReal('Distorted_block', distorted_I)
QWG.createWaveformReal('Distorted_block_compensation', -1*distorted_I)
QWG.getOperationComplete()
t1 = time.time()

print(t1-t0)

QWG.set('codeword_{}_ch{}_waveform'.format(block_cw, 3),
        'Distorted_block')

QWG.set('codeword_{}_ch{}_waveform'.format(comp_cw, 3),
        'Distorted_block_compensation')


QWG.start()
QWG.getOperationComplete()


for i in range(QWG.getSystemErrorCount()):
    logging.warning(QWG.getError())


from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs
reload(sqqs)


from os.path import join
from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib
reload(ins_lib)
base_qasm_path = sqqs.base_qasm_path

from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.utilities.general import mopen

operation_dict = {'Wait_5':
                  {'duration': 5, 'instruction': 'wait 5\n'},
                  'Wait_4':
                  {'duration': 4, 'instruction': 'wait 4\n'},
                  'Wait_20':
                  {'duration': 20, 'instruction': 'wait 20\n'},
                  'Wait_10000':
                  {'duration': 10000, 'instruction': 'wait 10000\n'},
                  'Wait_20000':
                  {'duration': 20000, 'instruction': 'wait 20000\n'},
                  'Start_marker':
                  {'duration': 20000,
                   'instruction': ins_lib.trigg_ch_to_instr(6, duration=10)}}
for i in range(8):
    operation_dict['CW_{}'.format(i)] = {
        'duration': 5,
        'instruction': ins_lib.qwg_cw_trigger(i)}


def Flux_cal_seq(block_cw, comp_cw):
    filename = join(base_qasm_path, 'Flux_cal_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('Start_marker \n')
    qasm_file.writelines('CW_{} \n'.format(block_cw))
    qasm_file.writelines('Wait_20000 \n') # hardcoded delay to ensure enough wait
    qasm_file.writelines('Wait_20 \n')
    qasm_file.writelines('CW_{} \n'.format(comp_cw))
    qasm_file.writelines('Wait_20000 \n')
    qasm_file.writelines('Wait_20000 \n')

    qasm_file.close()
    return qasm_file


single_pulse_elt = Flux_cal_seq(5, 6)
single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name, operation_dict)
qumis_file = single_pulse_asm
CBox.load_instructions(qumis_file.name)
CBox.start()
