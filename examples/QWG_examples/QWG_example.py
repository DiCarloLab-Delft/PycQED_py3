#!/usr/bin/python

import pycqed as pq # FIXME: must be before qcodes

import time
import numpy as np
import matplotlib.pyplot as plt

from pycqed.instrument_drivers.library.Transport import IPTransport
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG,QWGMultiDevices
#from pycqed.measurement.waveform_control_CC.waveform import Waveform


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

if 0: # functions disappeared
    wvCos = Waveform.cos(fs, sampleCnt, f)
    wvSin = Waveform.sin(fs, sampleCnt, f)
    wvZero = Waveform.DC(fs, sampleCnt)
    wvHi = Waveform.DC(fs, sampleCnt, 1.0)
    wvLo = Waveform.DC(fs, sampleCnt, -1.0)
    wvGauss = Waveform.gauss_pulse(fs, sampleCnt, mu, sigma)
    wvDerivGauss = Waveform.derivGauss(fs, sampleCnt, mu, sigma, dirAmpl)
    wvGauss2 = Waveform.gauss(fs, sampleCnt, mu2, sigma2)
    wvDerivGauss2 = Waveform.derivGauss(fs, sampleCnt, mu2, sigma2, dirAmpl2)
else:
    wvZero = np.array(np.arange(0, 1, 1/sampleCnt))
    wvGauss = 0.7 * np.array(np.arange(0, 1, 1/sampleCnt))
    wvDerivGauss = 0.7 * np.array(np.arange(0, 1, 1/sampleCnt))
    wvGauss2 = 0.7 * np.array(np.arange(0, 1, 1/sampleCnt))
    wvDerivGauss2 = 0.7 * np.array(np.arange(0, 1, 1/sampleCnt))


qwg1 = QWG('qwg_21', IPTransport('192.168.0.179'))
qwg1.init()

def run(continuous=True):

    if continuous:
        qwg1.create_waveform_real('cos', wvCos)
        qwg1.create_waveform_real('sin', wvSin)
        qwg1.create_waveform_real('zero', wvZero)
        qwg1.create_waveform_real('hi', wvHi)
        qwg1.create_waveform_real('lo', wvLo)
        qwg1.create_waveform_real('gauss', wvGauss)
        qwg1.create_waveform_real('derivGauss', wvDerivGauss)

        # qwg1.set('ch1_default_waveform', 'hi')
        # qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'hi')
        # qwg1.set('ch4_default_waveform', 'zero')

        qwg1.run_mode('CONt')

    else:  # codeword based
        if 0:
            qwg1.create_waveform_real('zero', wvZero)
            # qwg1.create_waveform_real('hi', wvHi)
            # qwg1.create_waveform_real('lo', wvLo)
            qwg1.create_waveform_real('gauss', wvGauss)
            qwg1.create_waveform_real('derivGauss', wvDerivGauss)
            qwg1.create_waveform_real('gauss2', wvGauss2)
            qwg1.create_waveform_real('derivGauss2', wvDerivGauss2)
            qwg1.create_waveform_real('gaussNeg', wvGauss) # FIXME: -wvGauss

        # segment 0: idle
        qwg1.set('ch1_default_waveform', 'zero')
        qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'zero')
        qwg1.set('ch4_default_waveform', 'zero')

        # set some standard waveform to all codewords
        for seg in range(8):
            qwg1.set('wave_ch{}_cw{:03}'.format(1, seg), wvGauss)
            qwg1.set('wave_ch{}_cw{:03}'.format(2, seg), wvDerivGauss)
            qwg1.set('wave_ch{}_cw{:03}'.format(3, seg), wvGauss2)
            qwg1.set('wave_ch{}_cw{:03}'.format(4, seg), wvDerivGauss2)

        qwg1.run_mode('CODeword')

    qwg1.ch_pair1_sideband_frequency.set(100e6)
    qwg1.ch_pair3_sideband_frequency.set(100e6)
    qwg1.sync_sideband_generators()

    qwg1.ch1_state.set(True)
    qwg1.ch2_state.set(True)
    qwg1.ch3_state.set(True)
    qwg1.ch4_state.set(True)

    qwg1.start()

    # read back
    qwg1.get_operation_complete()


    if 1:
        wvCosReadBack = qwg1.get_waveform_data_float('cos')
        plt.plot(wvCosReadBack)
        plt.ylabel('cos')
        plt.show()

        # waveform upload performance
        sizes = [100, 500, 1000, 1500, 2000, 2500]
        nrIter = 50
        durations = []
        megaBytesPerSecond = []
        for size in sizes:
            wvTest = Waveform.sin(fs, size, f)
            qwg1.get_operation_complete()
            markStart = time.perf_counter()
            for i in range(nrIter):
                qwg1.create_waveform_real(f'testSize{size}Nr{i}', wvTest)
            qwg1.getOperationComplete()
            markEnd = time.perf_counter()
            duration = (markEnd-markStart)/nrIter
            durations.append(duration*1e3)
            megaBytesPerSecond.append(size*4/duration/1e6)
        print(sizes)
        print(durations)
        print(megaBytesPerSecond)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(sizes, durations, 'bs')
        plt.xlabel('upload size [samples]')
        plt.ylabel('duration per upload [ms]')
        plt.axis([0, 2600, 0, 5])
        plt.subplot(212)
        plt.plot(sizes, megaBytesPerSecond, 'g^')
        plt.xlabel('upload size [samples]')
        plt.ylabel('performance [MB/s]')
        plt.axis([0, 2600, 0, 20])
        plt.show()

        # list waveforms
        print('WLIST size: ', qwg1.get_wlist_size())
        print('WLIST: ', qwg1.get_wlist())

    print('Identity: ', qwg1.get_identity())
    qwg1.check_errors()


if __name__ == '__main__':
    run(False)
