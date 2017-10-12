import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_RTE_Analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_syndrome_analsyis_single_file(self):
        # Test running it once with showing the initial fit
        ma.FlippingAnalysis(t_start='20170726_164901',
                            options_dict={'plot_init': True})

    def test_syndrome_analysis_multi_file(self):
        pass

    def _check_scaling(self, timestamp, known_detuning, places):
        a = ma.FlippingAnalysis(t_start=timestamp)
        s = a.get_scale_factor()
        self.assertAlmostEqual(s*known_detuning, 1, places=places)
        print('Scale factor {:.4f} known detuning {:.4f}'.format(
            s, known_detuning))
