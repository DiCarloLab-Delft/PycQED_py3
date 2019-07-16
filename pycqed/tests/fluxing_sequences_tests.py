
import numpy as np
import os
import unittest
import sys
# assumes pycqed already imported, not a safe import
# will work in a better way after we pip install pycqed
sys.path.append(r'D:\GitHubRepos\PycQED_py3')
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
import qcodes as qc

class element_distortion(unittest.TestCase):
    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.pulsar = Pulsar()

        for i in range(4):
            self.pulsar.define_channel(id='ch{}'.format(i+1),
                                       name='ch{}'.format(i+1))
            self.pulsar.define_channel(id='ch{}_m1'.format(i+1),
                                       name='ch{}_marker1'.format(i+1))
            self.pulsar.define_channel(id='ch{}_m2'.format(i+1),
                                          name='ch{}_marker2'.format(i+1))
            # max safe IQ voltage
            self.pulsar.set('ch{}_amp'.format(i + 1), 0.7)
            self.pulsar.set('ch{}_marker1_amp'.format(i+1), 2.0)
            self.pulsar.set('ch{}_marker2_amp'.format(i+1), 2.0)

        # We need to discuss where to store this stuff

        fsqs.kernel_dir = 'testing/kernels/'
        self.kernel_list = ['kernels_rabi/kernel_160208_220046_it1.txt']


    def test_distorted_attribute(self):

        test_elt = element.Element('test_elt', pulsar=self.pulsar)


        self.assertTrue((len(test_elt._channels))!=0)

        for ch, item in test_elt._channels.items():
            self.assertFalse(item['distorted'])
        self.assertEqual(len(test_elt.distorted_wfs), 0 )

        test_elt.add(SquarePulse(name='dummy_sqaure',
                                 channel='ch1',
                                 amplitude=.3, length=20e-9))

        dist_dict = {'ch_list': ['ch1'],
                     'ch1': self.kernel_list}
        test_elt = fsqs.distort(test_elt, dist_dict)
        self.assertEqual(len(test_elt.distorted_wfs), 1)
        for ch, item in test_elt._channels.items():
            if ch == 'ch1':
                self.assertTrue(item['distorted'])
                self.assertTrue(ch in test_elt.distorted_wfs.keys())
            else:
                self.assertFalse(item['distorted'])





# if __name__ is '__main__':
test_classes_to_run = [element_distortion]

suites_list = []
for test_class in test_classes_to_run:
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    suites_list.append(suite)

combined_test_suite = unittest.TestSuite(suites_list)
runner = unittest.TextTestRunner(verbosity=2).run(combined_test_suite)
