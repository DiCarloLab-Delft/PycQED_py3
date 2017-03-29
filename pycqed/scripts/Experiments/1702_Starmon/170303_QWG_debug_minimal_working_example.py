from pycqed.measurement.waveform_control_CC import waveform as wf
iport logging
import qcodes as qc
station = qc.station

QWG = station.components['QWG']
# CBox = station.components['CBox']

QWG.stop()
QWG.deleteWaveformAll()


block_I, block_Q = wf.block_pulse(1, 10e-9, sampling_rate=1e9)
block_I=np.array(block_I)
block_Q = np.array(block_Q)
for i in range(8):
    QWG.createWaveformReal('wf_I_{}'.format(i), block_I*(i+1)/10)
    QWG.createWaveformReal('wf_Q_{}'.format(i), block_Q*(i+1)/10)


    QWG.set('codeword_{}_ch{}_waveform'.format(i, 1), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 2), 'wf_Q_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 3), 'wf_I_{}'.format(i))
    QWG.set('codeword_{}_ch{}_waveform'.format(i, 4), 'wf_Q_{}'.format(i))

QWG.start()
QWG.getOperationComplete()


for i in range(QWG.getSystemErrorCount()):
    logging.warning(QWG.getError())


CBox.load_instructions(r'D:\GitHubRepos\PycQED_py3\pycqed\measurement\waveform_control_CC\micro_instruction_files\qwg_test.qumis')
CBox.core_state('active')
CBox.run_mode('run')
