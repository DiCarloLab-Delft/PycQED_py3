import unittest
import numpy as np
from . import defHeaders_CBox_v3 as defHeaders
from . import test_suite


class CBox_tests_v3(test_suite.CBox_tests):

    def test_firmware_version(self):
        v = CBox.get('firmware_version')
        print(v)
        self.assertTrue(int(v[1]) == 3)  # major version
        self.assertTrue(int(int(v[3:5])) > 13)  # minor version

    def test_setting_mode(self):
        for i in range(5):
            self.CBox.set('acquisition_mode', i)
            self.assertEqual(self.CBox.get('acquisition_mode'),
                             defHeaders.acquisition_modes[i])
        self.CBox.set('acquisition_mode', 0)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[0])

        for i in range(2):
            self.CBox.set('run_mode', i)
            self.assertEqual(self.CBox.get('run_mode'),
                             defHeaders.run_modes[i])
        self.CBox.set('run_mode', 0)

        for i in range(3):
            self.CBox.set('trigger_source', i)
            self.assertEqual(self.CBox.get('trigger_source'),
                             defHeaders.trigger_sources[i])
        self.CBox.set('trigger_source', 0)

        for j in range(3):
            for i in range(3):
                self.CBox.set('AWG{}_mode'.format(j), i)
                self.assertEqual(self.CBox.get('AWG{}_mode'.format(j)),
                                 defHeaders.awg_modes[i])
                self.CBox.set('AWG{}_mode'.format(j), 0)

    def test_integration_average_mode(self):
        self.CBox.set('acquisition_mode', 0)
        NoSamples = 60


        weights0 = np.ones(512) * 1
        weights1 = np.ones(512) * 0

        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        self.CBox.set('nr_averages', 4)
        self.CBox.set('nr_samples', NoSamples)

        self.CBox.set('acquisition_mode', 2)
        [IntAvgRes0, IntAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 0)

        # Test signal lengths set correctly
        self.assertEqual(len(InputAvgRes0), NoSamples)
        # Test if setting weights to zero functions correctly
        self.assertTrue((IntAvgRes1 == np.zeros(NoSamples)).all())

        weights1 = np.ones(512) * 1
        self.CBox.set('sig1_integration_weights', weights1)
        self.CBox.set('lin_trans_coeffs', [0, 0, 0, 1])
        self.CBox.set('acquisition_mode', 4)
        [IntAvgRes0, IntAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 0)

        # Test if setting lin trans coeff to zero functions correctly
        self.assertTrue((IntAvgRes0 == np.zeros(NoSamples)).all())
        self.assertFalse((IntAvgRes1 == np.zeros(NoSamples)).all())
