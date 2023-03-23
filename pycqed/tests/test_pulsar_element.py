import numpy as np
import unittest
import qcodes as qc
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
import time


class Test_Element(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.station = qc.Station()
        print(str(time.time()))
        self.AWG = VirtualAWG5014('AWG')
        self.AWG.clock_freq(1e9)
        self.pulsar = Pulsar('Pulsar', self.AWG.name)
        self.station.pulsar = self.pulsar
        for i in range(4):
            self.pulsar.define_channel(id='ch{}'.format(i+1),
                                          name='ch{}'.format(i+1),
                                          type='analog',
                                          # max safe IQ voltage
                                          high=.7, low=-.7,
                                          offset=0.0, delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                          name='ch{}_marker1'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                          name='ch{}_marker2'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)

    def test_basic_element(self):
        test_elt = element.Element('test_elt', pulsar=self.pulsar)
        test_elt.add(SquarePulse(name='dummy_square',
                                 channel='ch1',
                                 amplitude=.3, length=20e-9))
        min_samples = 960
        ch1_wf = test_elt.waveforms()[1]['ch1']
        self.assertEqual(len(ch1_wf), min_samples)

        expected_wf = np.zeros(960)
        expected_wf[:20] = .3
        np.testing.assert_array_almost_equal(ch1_wf, expected_wf)

    def test_timing(self):
        test_elt = element.Element('test_elt', pulsar=self.pulsar)
        refpulse = SquarePulse(name='dummy_square',
                               channel='ch1',
                               amplitude=0, length=20e-9)
        test_elt.add(refpulse, start=100e-9, name='dummy_square')

        test_elt.add(SquarePulse(name='dummy_square',
                                 channel='ch1',
                                 amplitude=.3, length=20e-9),
                     refpulse='dummy_square', start=100e-9, refpoint='start')
        min_samples = 960
        ch1_wf = test_elt.waveforms()[1]['ch1']
        self.assertEqual(len(ch1_wf), min_samples)

        expected_wf = np.zeros(960)
        expected_wf[100:120] = .3
        np.testing.assert_array_almost_equal(ch1_wf, expected_wf)

    def test_fixpoint(self):
        # Fixed point should shift both elements by 2 ns
        test_elt = element.Element('test_elt', pulsar=self.pulsar)
        refpulse = SquarePulse(name='dummy_square',
                               channel='ch1',
                               amplitude=.5, length=20e-9)
        test_elt.add(refpulse,  name='dummy_square')

        test_elt.add(SquarePulse(name='dummy_square',
                                 channel='ch1',
                                 amplitude=.3, length=20e-9),
                     operation_type='RO',
                     refpulse='dummy_square', start=98e-9, refpoint='start')
        min_samples = 1020
        ch1_wf = test_elt.waveforms()[1]['ch1']
        self.assertEqual(len(ch1_wf), min_samples)

        expected_wf = np.zeros(1020)
        expected_wf[902:922] = .5
        expected_wf[1000:1020] = .3
        np.testing.assert_array_almost_equal(ch1_wf, expected_wf)

    def tearDown(self):
        qc.Instrument.close_all()
