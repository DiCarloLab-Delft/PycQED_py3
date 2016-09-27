#!/usr/bin/python
import unittest
from instrument_drivers.physical_instruments.QuTech_AWG_Module \
    import QuTech_AWG_Module
from measurement.waveform_control_CC.waveform import Waveform
import time
import numpy as np
from socket import timeout
from qcodes.utils  import validators as vals


class QWG_tests(unittest.TestCase):
    '''
    This is a test suite for testing the HeterodyneSource Instrument.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        try:
            self.qwg = QWG
        except:
            self.qwg = QuTech_AWG_Module(
                'QWG', address='192.168.42.10',
                port=5025, server_name=None)

        self.qwg.reset()
        qwg1._socket.settimeout(.3)  # set low timeout for tests

    def test_IDN(self):
        IDN = self.qwg.IDN()
        self.assertIsInstance(IDN, dict)
        self.assertEqual(IDN['vendor'], 'QuTech')
        self.assertEqual(IDN['model'], 'QWG')

    def test_Error(self):
        err_msg = self.qwg.getError()
        self.assertEqual(err_msg, '0,"No error"')

    def bool_get_set(self, par):
        old_val = par.get()
        par.set(True)
        self.assertTrue(par.get(), msg=par.name)
        par.set(False)
        self.assertFalse(par.get(), msg=par.name)
        par.set(old_val)

    def array_get_set(self, par):
        old_val = par.get()
        shape = par._vals._shape
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100
        v = (np.zeros(shape)+mn)+(mx-mn)/2
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn))
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn))
        par.set(old_val)

    def floating_get_set(self, par):
        old_val = par.get()
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100.5
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100.5
        v = (mn)+(mx-mn)/2
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn))
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn))
        par.set(old_val)

    def integer_get_set(self, par):
        old_val = par.get()
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100
        v = (mn)+(mx-mn)//2
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn)*10)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn)*10)
        par.set(old_val)

    def string_get_set(self, par):
        old_val = par.get()
        min_len = par._vals._min_length
        max_len = par._vals._max_length

        v = 'test_string'
        if len(v) > max_len:
            v = v[:max_len]
        if len(v) < min_len:
            par.set(v)  # expect failure and fix it
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        par.set(old_val)

    def enum_get_set(self, par):
        old_val = par.get()
        for v in par._vals._values:
            par.set(v)
            self.assertEqual(v, par.get(), msg=par.name)
        par.set(old_val)

    def test_parameters(self):
        for parname, par in sorted(self.qwg.parameters.items()):
            failing_pars = []
            #for i in range(16):
                #failing_pars.append('ch{}_amp'.format(i))
                #failing_pars.append('ch{}_offset'.format(i))
                #failing_pars.append('tr{}_trigger_level'.format(i))

                # Sideband phase always returns 0 when get
                #failing_pars.append('ch_pair{}_sideband_phase'.format(i))
                # transformation matrix get returns garbage
                #failing_pars.append('ch_pair{}_transform_matrix'.format(i))

            # Error messages:  -113,"Undefined header;AWGC:RMOD?"
            #failing_pars.append('run_mode')

            if par.name not in ['IDN'] and par.name not in failing_pars:
                # print('parname:', par.name)
                old_value = par.get()
                old_value2 = par.get()
                self.assertEqual(old_value2, old_value, msg=par.name)
                if hasattr(par, '_vals'):
                    validator = par._vals
                    if isinstance(validator, vals.Ints):
                        self.integer_get_set(par)
                    elif isinstance(validator, vals.Numbers):
                        self.floating_get_set(par)
                    elif isinstance(validator, vals.Arrays):
                        self.array_get_set(par)
                    elif isinstance(validator, vals.Bool):
                        self.bool_get_set(par)
                    elif isinstance(validator, vals.Strings):
                        self.string_get_set(par)
                    elif isinstance(validator, vals.Enum):
                        self.enum_get_set(par)

                    else:
                        print('{} validator "{}" not recognized'.format(
                            par.name, par._vals))
                else:
                    print(par.name, 'does not have validator, not testing')
            else:
                print('Not in pars to be tested: "{}"'.format(par.name))

    def tearDown(self):
        self.qwg._socket.settimeout(5)
        # set timeout back to default

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
marker1 = []
marker2 = []


# if 1:
qwg1 = QuTech_AWG_Module('QWG-1', '192.168.42.10', 5025, server_name=None)
# else:
#     # local variant, in combination with 'nc -l 5025' run locally from a
#     # terminal
#     qwg1 = QuTech_AWG_Module('QWG-1', '127.0.0.1', 5025, server_name=None)
#qwg1 = QWG
qwg1.reset()

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        QWG_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)

    if 1:  # continuous
        qwg1.createWaveformReal('cos', wvCos, marker1, marker2)
        qwg1.createWaveformReal('sin', wvSin, marker1, marker2)
        qwg1.createWaveformReal('zero', wvZero, marker1, marker2)
        qwg1.createWaveformReal('hi', wvHi, marker1, marker2)
        qwg1.createWaveformReal('lo', wvLo, marker1, marker2)
        qwg1.createWaveformReal('gauss', wvGauss, marker1, marker2)
        qwg1.createWaveformReal('derivGauss', wvDerivGauss, marker1, marker2)

        qwg1.set('ch1_default_waveform', 'gauss')
        qwg1.set('ch2_default_waveform', 'derivGauss')
        qwg1.set('ch3_default_waveform', 'gauss')
        qwg1.set('ch4_default_waveform', 'zero')

        qwg1.run_mode('CONt')

    else:  # codeword based
        qwg1.createWaveformReal('zero', wvZero, marker1, marker2)
        qwg1.createWaveformReal('hi', wvHi, marker1, marker2)
        qwg1.createWaveformReal('lo', wvLo, marker1, marker2)
        qwg1.createWaveformReal('gauss', wvGauss, marker1, marker2)
        qwg1.createWaveformReal('derivGauss', wvDerivGauss, marker1, marker2)
        qwg1.createWaveformReal('gauss2', wvGauss2, marker1, marker2)
        qwg1.createWaveformReal('derivGauss2', wvDerivGauss2, marker1, marker2)

        qwg1.createWaveformReal('gaussNeg', -wvGauss, marker1, marker2)

        # segment 0: idle
        qwg1.set('ch1_default_waveform', 'zero')
        qwg1.set('ch2_default_waveform', 'zero')
        qwg1.set('ch3_default_waveform', 'zero')
        qwg1.set('ch4_default_waveform', 'zero')

        seg = 0

        # segment 1:
        seg = seg+1
        # Seg corresponds to codeword
        qwg1.setSeqElemWaveform(seg, 1, 'hi')
        qwg1.setSeqElemWaveform(seg, 2, 'hi')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss')
        qwg1.setSeqElemWaveform(seg, 4, 'derivGauss')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'gauss')
        qwg1.setSeqElemWaveform(seg, 2, 'gaussNeg')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
        qwg1.setSeqElemWaveform(seg, 4, 'derivGauss2')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'gauss')
        qwg1.setSeqElemWaveform(seg, 2, 'derivGauss')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
        qwg1.setSeqElemWaveform(seg, 4, 'derivGauss2')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'gauss2')
        qwg1.setSeqElemWaveform(seg, 2, 'derivGauss2')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss')
        qwg1.setSeqElemWaveform(seg, 4, 'derivGauss')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'zero')
        qwg1.setSeqElemWaveform(seg, 2, 'zero')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss')
        qwg1.setSeqElemWaveform(seg, 4, 'gaussNeg')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'zero')
        qwg1.setSeqElemWaveform(seg, 2, 'zero')
        qwg1.setSeqElemWaveform(seg, 3, 'derivGauss')
        qwg1.setSeqElemWaveform(seg, 4, 'zero')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'zero')
        qwg1.setSeqElemWaveform(seg, 2, 'gauss2')
        qwg1.setSeqElemWaveform(seg, 3, 'zero')
        qwg1.setSeqElemWaveform(seg, 4, 'zero')

        seg = seg+1
        qwg1.setSeqElemWaveform(seg, 1, 'zero')
        qwg1.setSeqElemWaveform(seg, 2, 'zero')
        qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
        qwg1.setSeqElemWaveform(seg, 4, 'zero')

        qwg1.setRunModeCodeword()


    qwg1.ch_pair1_sideband_frequency.set(100e6)
    qwg1.ch_pair3_sideband_frequency.set(100e6)
    qwg1.syncSidebandGenerators()

    qwg1.ch1_state.set(True)
    qwg1.ch2_state.set(True)
    qwg1.ch3_state.set(True)
    qwg1.ch4_state.set(True)

    qwg1.run()

    print('Identity: ', qwg1.getIdentity())
    print('Error messages: ')
    for i in range(qwg1.getSystemErrorCount()):
        print(qwg1.getError())
