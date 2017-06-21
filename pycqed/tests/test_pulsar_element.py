import numpy as np
import unittest
import qcodes as qc
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
import time

class Test_Element(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.station = qc.Station()
        self.AWG = VirtualAWG5014('AWG'+str(time.time()))
        self.AWG.clock_freq(1e9)
        self.pulsar = Pulsar('Pulsar' + str(time.time()), self.AWG.name)
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

    def test_operation_dependent_buffers_and_compensation(self):
        # Fixed point should shift both elements by 2 ns

        RO_amp = 0.1
        MW_amp = 0.3
        Flux_amp = 0.5
        operation_dict = {'RO q0': {'amplitude': RO_amp,
                                    'length': 300e-9,
                                    'operation_type': 'RO',
                                    'channel': 'ch1',
                                    'pulse_delay': 0,
                                    'pulse_type': 'SquarePulse'},
                          'MW q0': {'amplitude': MW_amp,
                                    'length': 20e-9,
                                    'operation_type': 'MW',
                                    'channel': 'ch1',
                                    'pulse_delay': 0,
                                    'pulse_type': 'SquarePulse'},
                          'Flux q0': {'amplitude': Flux_amp,
                                      'length': 40e-9,
                                      'operation_type': 'Flux',
                                      'channel': 'ch1',
                                      'pulse_delay': 0,
                                      'pulse_type': 'SquarePulse'},

                          'sequencer_config': {'Buffer_Flux_Flux': 1e-9,
                                               'Buffer_Flux_MW': 2e-9,
                                               'Buffer_Flux_RO': 3e-9,
                                               'Buffer_MW_Flux': 4e-9,
                                               'Buffer_MW_MW': 5e-9,
                                               'Buffer_MW_RO': 6e-9,
                                               'Buffer_RO_Flux': 7e-9,
                                               'Buffer_RO_MW': 8e-9,
                                               'Buffer_RO_RO': 10e-9,
                                               'RO_fixed_point': 1e-06,
                                               'Flux_comp_dead_time': 3e-6}}
        sequencer_config = operation_dict['sequencer_config']

        fake_seq = ['MW q0', 'MW q0', 'Flux q0', 'MW q0', 'RO q0']
        pulses = []
        for p in fake_seq:
            pulses += [operation_dict[p]]
        test_elt = multi_pulse_elt(0, self.station, pulses, sequencer_config)

        min_samples = 1800+3040  # 1us fixpoint, 300ns RO pulse and 4ns zeros
        ch1_wf = test_elt.waveforms()[1]['ch1']
        self.assertEqual(len(ch1_wf), min_samples)

        expected_wf = np.zeros(min_samples)
        expected_wf[1000:1300] = RO_amp
        expected_wf[974:994] = MW_amp
        expected_wf[932:972] = Flux_amp
        expected_wf[908:928] = MW_amp
        expected_wf[883:903] = MW_amp
        expected_wf[4300:4340] = -Flux_amp

        np.testing.assert_array_almost_equal(ch1_wf, expected_wf)

    # def test_distorted_attribute(self):

    #     test_elt = element.Element('test_elt', pulsar=self.pulsar)

    #     self.assertTrue((len(test_elt._channels)) != 0)

    #     for ch, item in test_elt._channels.items():
    #         self.assertFalse(item['distorted'])
    #     self.assertEqual(len(test_elt.distorted_wfs), 0)

    #     test_elt.add(SquarePulse(name='dummy_square',
    #                              channel='ch1',
    #                              amplitude=.3, length=20e-9))

    #     dist_dict = {'ch_list': ['ch1'],
    #                  'ch1': self.kernel_list}
    #     test_elt = fsqs.distort(test_elt, dist_dict)
    #     self.assertEqual(len(test_elt.distorted_wfs), 1)
    #     for ch, item in test_elt._channels.items():
    #         if ch == 'ch1':
    #             self.assertTrue(item['distorted'])
    #             self.assertTrue(ch in test_elt.distorted_wfs.keys())
    #         else:
    #             self.assertFalse(item['distorted'])
