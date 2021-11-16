#!/usr/bin/python

import pycqed as pq # FIXME: must be before qcodes

import time
#import numpy as np
import matplotlib.pyplot as plt

from pycqed.instrument_drivers.library.Transport import IPTransport
#import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG,QWGMultiDevices

from waveform import Waveform


# create waveforms
sample_cnt = 96
fs = 1e9  # sampling frequency
f = fs/32
# mu=30, sigma=10, dirAmpl=1.0 fits 64 samples nicely
mu = 30e-9
sigma = 10e-9
dir_ampl = 1.0
# mu2=15, sigma2=5, dirAmpl2=1.0 fits 64 samples nicely
mu2 = 15e-9
sigma2 = 5e-9
dir_ampl2 = 1.0

wv_cos = Waveform.cos(fs, sample_cnt, f)
wv_sin = Waveform.sin(fs, sample_cnt, f)
wv_zero = Waveform.DC(fs, sample_cnt)
wv_hi = Waveform.DC(fs, sample_cnt, 1.0)
wv_lo = Waveform.DC(fs, sample_cnt, -1.0)
wv_gauss = Waveform.gauss(fs, sample_cnt, mu, sigma)
wv_deriv_gauss = Waveform.derivGauss(fs, sample_cnt, mu, sigma, dir_ampl)
wv_gauss2 = Waveform.gauss(fs, sample_cnt, mu2, sigma2)
wv_deriv_gauss2 = Waveform.derivGauss(fs, sample_cnt, mu2, sigma2, dir_ampl2)


qwg1 = QWG('qwg_21', IPTransport('192.168.0.179'))
qwg1.init()

def run(continuous=True):

    if continuous:
        qwg1.create_waveform_real('cos', wv_cos)
        qwg1.create_waveform_real('sin', wv_sin)
        qwg1.create_waveform_real('zero', wv_zero)
        qwg1.create_waveform_real('hi', wv_hi)
        qwg1.create_waveform_real('lo', wv_lo)
        qwg1.create_waveform_real('gauss', wv_gauss)
        qwg1.create_waveform_real('derivGauss', wv_deriv_gauss)

        # qwg1.set('ch1_default_waveform', 'hi')
        # qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'hi')
        # qwg1.set('ch4_default_waveform', 'zero')

        qwg1.run_mode('CONt')

    else:  # codeword based
        qwg1.create_waveform_real('zero', wv_zero)
        qwg1.create_waveform_real('hi', wv_hi)
        qwg1.create_waveform_real('lo', wv_lo)
        qwg1.create_waveform_real('gauss', wv_gauss)
        qwg1.create_waveform_real('derivGauss', wv_deriv_gauss)
        qwg1.create_waveform_real('gauss2', wv_gauss2)
        qwg1.create_waveform_real('derivGauss2', wv_deriv_gauss2)
        qwg1.create_waveform_real('gaussNeg', -wv_gauss)

        # segment 0: idle
        qwg1.set('ch1_default_waveform', 'zero')
        qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'zero')
        qwg1.set('ch4_default_waveform', 'zero')

        # set some standard waveform to all codewords
        for seg in range(8):
            qwg1.set('wave_ch{}_cw{:03}'.format(1, seg), wv_gauss)
            qwg1.set('wave_ch{}_cw{:03}'.format(2, seg), wv_deriv_gauss)
            qwg1.set('wave_ch{}_cw{:03}'.format(3, seg), wv_gauss2)
            qwg1.set('wave_ch{}_cw{:03}'.format(4, seg), wv_deriv_gauss2)

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
        wv_cos_read_back = qwg1.get_waveform_data_float('cos')
        plt.plot(wv_cos_read_back)
        plt.ylabel('cos')
        plt.show()

        # waveform upload performance
        sizes = [100, 500, 1000, 1500, 2000, 2500]
        nr_iter = 50
        durations = []
        mega_bytes_per_second = []
        for size in sizes:
            wv_test = Waveform.sin(fs, size, f)
            qwg1.get_operation_complete()
            mark_start = time.perf_counter()
            for i in range(nr_iter):
                qwg1.create_waveform_real(f'testSize{size}Nr{i}', wv_test)
            qwg1.getOperationComplete()
            markEnd = time.perf_counter()
            duration = (markEnd-mark_start)/nr_iter
            durations.append(duration*1e3)
            mega_bytes_per_second.append(size*4/duration/1e6)
        print(sizes)
        print(durations)
        print(mega_bytes_per_second)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(sizes, durations, 'bs')
        plt.xlabel('upload size [samples]')
        plt.ylabel('duration per upload [ms]')
        plt.axis([0, 2600, 0, 5])
        plt.subplot(212)
        plt.plot(sizes, mega_bytes_per_second, 'g^')
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
