
#!/usr/bin/python
import unittest
from pycqed.instrument_drivers.physical_instruments.QuTech_AWG_Module \
    import QuTech_AWG_Module
from pycqed.measurement.waveform_control_CC.waveform import Waveform
import time
import numpy as np
from socket import timeout
from qcodes.utils import validators as vals
import matplotlib.pyplot as plt
import pycqed as pq

# create waveforms
sampleCnt = 96
fs = 1e9  # sampling frequency
f = fs/32
# mu=30, sigma=10, dirAmpl=1.0 fits 64 samples nicely
mu = 30e-9
sigma = 10e-9
dirAmpl = 1.0
# mu2=15, sigma2=5, dirAmpl2=1.0 fits 64 samples nicely
mu2 = 15e-9
sigma2 = 5e-9
dirAmpl2 = 1.0

wvCos = Waveform.cos(fs, sampleCnt, f)
wvSin = Waveform.sin(fs, sampleCnt, f)
wvZero = Waveform.DC(fs, sampleCnt)
wvHi = Waveform.DC(fs, sampleCnt, 1.0)
wvLo = Waveform.DC(fs, sampleCnt, -1.0)
wvGauss = Waveform.gauss(fs, sampleCnt, mu, sigma)
wvDerivGauss = Waveform.derivGauss(fs, sampleCnt, mu, sigma, dirAmpl)
wvGauss2 = Waveform.gauss(fs, sampleCnt, mu2, sigma2)
wvDerivGauss2 = Waveform.derivGauss(fs, sampleCnt, mu2, sigma2, dirAmpl2)

try:
    qwg1 = pq.station['QWG']
except:
    qwg1 = QuTech_AWG_Module(
        'QWG', address='192.168.0.10',
        port=5025, server_name=None)
qwg1.reset()

def run(continuous=True):

    if continuous:
        qwg1.createWaveformReal('cos', wvCos)
        qwg1.createWaveformReal('sin', wvSin)
        qwg1.createWaveformReal('zero', wvZero)
        qwg1.createWaveformReal('hi', wvHi)
        qwg1.createWaveformReal('lo', wvLo)
        qwg1.createWaveformReal('gauss', wvGauss)
        qwg1.createWaveformReal('derivGauss', wvDerivGauss)

        # qwg1.set('ch1_default_waveform', 'hi')
        # qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'hi')
        # qwg1.set('ch4_default_waveform', 'zero')

        qwg1.run_mode('CONt')

    else:  # codeword based
        qwg1.createWaveformReal('zero', wvZero)
        qwg1.createWaveformReal('hi', wvHi)
        qwg1.createWaveformReal('lo', wvLo)
        qwg1.createWaveformReal('gauss', wvGauss)
        qwg1.createWaveformReal('derivGauss', wvDerivGauss)
        qwg1.createWaveformReal('gauss2', wvGauss2)
        qwg1.createWaveformReal('derivGauss2', wvDerivGauss2)

        qwg1.createWaveformReal('gaussNeg', -wvGauss)

        # segment 0: idle
        qwg1.set('ch1_default_waveform', 'zero')
        qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'zero')
        qwg1.set('ch4_default_waveform', 'zero')

        # set some standard waveform to all codewords
        for seg in range(8):
            qwg1.set('codeword_{}_ch{}_waveform'.format(seg, 1), 'gauss')
            qwg1.set('codeword_{}_ch{}_waveform'.format(seg, 2), 'derivGauss')
            qwg1.set('codeword_{}_ch{}_waveform'.format(seg, 3), 'gauss2')
            qwg1.set('codeword_{}_ch{}_waveform'.format(seg, 4), 'derivGauss2')

        qwg1.run_mode('CODeword')

    qwg1.ch_pair1_sideband_frequency.set(100e6)
    qwg1.ch_pair3_sideband_frequency.set(100e6)
    qwg1.syncSidebandGenerators()

    qwg1.ch1_state.set(True)
    qwg1.ch2_state.set(True)
    qwg1.ch3_state.set(True)
    qwg1.ch4_state.set(True)

    qwg1.start()

    # read back
    qwg1.getOperationComplete()
    # wvCosReadBack = qwg1.getWaveformDataFloat('cos')
    # plt.plot(wvCosReadBack)
    # plt.ylabel('cos')
    # plt.show()

    # # waveform upload performance
    # sizes = [100, 500, 1000, 1500, 2000, 2500]
    # nrIter = 50
    # durations = []
    # megaBytesPerSecond = []
    # for size in sizes:
    #     wvTest = Waveform.sin(fs, size, f)
    #     qwg1.getOperationComplete()
    #     markStart = time.perf_counter()
    #     for i in range(nrIter):
    #         qwg1.createWaveformReal('testSize{}Nr{}'.format(size, i), wvTest)
    #     qwg1.getOperationComplete()
    #     markEnd = time.perf_counter()
    #     duration = (markEnd-markStart)/nrIter
    #     durations.append(duration*1e3)
    #     megaBytesPerSecond.append(size*4/duration/1e6)
    # print(sizes)
    # print(durations)
    # print(megaBytesPerSecond)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(sizes, durations, 'bs')
    # plt.xlabel('upload size [samples]')
    # plt.ylabel('duration per upload [ms]')
    # plt.axis([0, 2600, 0, 5])
    # plt.subplot(212)
    # plt.plot(sizes, megaBytesPerSecond, 'g^')
    # plt.xlabel('upload size [samples]')
    # plt.ylabel('performance [MB/s]')
    # plt.axis([0, 2600, 0, 20])
    # plt.show()

    # # list waveforms
    # wlistSize = qwg1.WlistSize()
    # print('WLIST size: ', wlistSize)
    # print('WLIST: ', qwg1.Wlist())

    # # show some info
    # print('Identity: ', qwg1.getIdentity())
    print('Error messages: ')
    for i in range(qwg1.getSystemErrorCount()):
        print(qwg1.getError())


if __name__ == '__main__':
    run()
