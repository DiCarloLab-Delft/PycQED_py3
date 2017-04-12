import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools

class Test_qubitspec_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.a_pk = ma.Qubit_Spectroscopy_Analysis(timestamp='20170412_163359')





    def test_spectroscopy_find_peak(self):
        peaks_dict = self.a_pk.peaks
        self.assertEqual(peaks_dict['dips'], [])

        first_peak = peaks_dict['peaks'][0]
        peak_low = 4.55e9
        peak_high = 4.555e9

        self.assertGreaterEqual(first_peak, peak_low)
        self.assertGreaterEqual(peak_high, first_peak)



    def test_spectroscopy_find_dip(self):
        # Test the correct file is loaded
        pass
        # self.assertEqual(self.a_acq_delay.folder,
        #                  os.path.join(self.datadir, '20170227',
        #                               '115118_acquisition_delay_scan_qubit'))

        # id = a_tools.nearest_idx(self.a_acq_delay.sweep_points,
        #                          self.a_acq_delay.max_delay)

        # self.assertGreaterEqual(self.a_acq_delay.measured_values[0][id],
        #                         0.95*max(self.a_acq_delay.measured_values[0]))