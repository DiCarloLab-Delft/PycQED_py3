#!/usr/bin/python
import unittest
from instrument_drivers.physical_instruments.QuTech_AWG_Module \
    import QuTech_AWG_Module
from measurement.waveform_control_CC.waveform import Waveform
import time
from socket import timeout


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
        try:
            qwg1._socket.recv(1000)
        except timeout:
            pass

    def test_IDN(self):
        IDN = self.qwg.IDN()
        self.assertIsInstance(IDN, dict)
        self.assertEqual(IDN['vendor'], 'QuTech')
        self.assertEqual(IDN['model'], 'QWG')

    def test_Error(self):
        err_msg = self.qwg.getError()
        self.assertEqual(err_msg, '0,"No error"\n')

    def test_parameters(self):
        # Clearing comms should not be an issue
        try:
            qwg1._socket.recv(1000)
        except timeout:
            pass
        for parname, par in self.qwg.parameters.items():

            failing_pars = []
            for i in [1, 3]:
                failing_pars.append('ch_pair{}_sideband_frequency'.format(i))
                failing_pars.append('ch_pair{}_sideband_phase'.format(i))
            for i in range(self.qwg.device_descriptor.numTriggers):
                failing_pars.append('ch{}_offset'.format(i+1))
                failing_pars.append('ch{}_trigger_level'.format(i+1))
            failing_pars.append('run_mode')
            failing_pars.append('trigger_level')

            for i in range(self.qwg.device_descriptor.numChannels):
                failing_pars.append('ch{}_amp'.format(i+1))

            if par.name not in ['IDN']+failing_pars:
                # print('parname:', par.name)
                try:
                    old_value = par.get()
                    old_value2 = par.get()
                    self.assertEqual(old_value2, old_value)
                except timeout:
                    raise(TimeoutError(' could not read {}'.format(par.name)))
                if hasattr(par, '_vals'):
                    vals = par._vals
                    if vals.is_numeric:
                        min_val = vals._min_value
                        max_val = vals._max_value
                        if max_val != float("inf"):
                            with self.assertRaises(
                                    ValueError,
                                    msg='{} max_val+1 {}'.format(par.name, (
                                    max_val+1))):
                                par.set(max_val+1)
                        if min_val != -float("inf"):
                            with self.assertRaises(
                                    ValueError,
                                    msg='{} min_val-1: {}'.format(
                                    par.name, (min_val-1))):
                                par.set(min_val-1)
                    else:
                        print(par.name, ' is not numeric, not testing')
                else:
                    print(par.name, 'does not have validator')
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
#     qwg1 = QuTech_AWG_Module('QWG-1', '192.168.42.10', 5025, server_name=None)
# else:
#     # local variant, in combination with 'nc -l 5025' run locally from a
#     # terminal
#     qwg1 = QuTech_AWG_Module('QWG-1', '127.0.0.1', 5025, server_name=None)
qwg1 = QWG
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

        if 1:  # dc
            qwg1.setWaveform(1, 'gauss')  # 'hi')
            qwg1.setWaveform(2, 'derivGauss')  # 'zero')
            qwg1.setWaveform(3, 'gauss')  # 'hi')
            qwg1.setWaveform(4, 'zero')  # 'zero')
        else:
            qwg1.setWaveform(1, 'derivGauss')
            qwg1.setWaveform(2, 'zero')
            qwg1.setWaveform(3, 'zero')
            qwg1.setWaveform(4, 'gauss')
        qwg1.setRunModeContinuous()

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
        qwg1.setWaveform(1, 'zero')
        qwg1.setWaveform(2, 'zero')
        qwg1.setWaveform(3, 'zero')
        qwg1.setWaveform(4, 'zero')

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

    qwg1.ch1_state.set(1)
    qwg1.ch2_state.set(1)
    qwg1.ch3_state.set(1)
    qwg1.ch4_state.set(1)

    qwg1.run()

    print('Identity: ', qwg1.getIdentity())
    print('Error messages: ', qwg1.getError())

