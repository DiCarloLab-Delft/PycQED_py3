import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_flipping_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_flipping_analysis(self):

        # this test is based on an experiment with a known
        # added detuning in the amplitude. The test tests that the analysis
        # works for a range of known scale factors.

        # 20% detuning only works for coarse
        self._check_scaling('20170726_164507', 0.8, 1)

        self._check_scaling('20170726_164536', 0.9, 1)
        self._check_scaling('20170726_164550', 0.9, 1)
        self._check_scaling('20170726_164605', 0.95, 2)
        self._check_scaling('20170726_164619', 0.95, 2)
        self._check_scaling('20170726_164635', 0.99, 2)
        self._check_scaling('20170726_164649', 0.99, 2)
        self._check_scaling('20170726_164704', 1, 2)
        self._check_scaling('20170726_164718', 1, 2)
        self._check_scaling('20170726_164733', 1.01, 2)
        self._check_scaling('20170726_164747', 1.01, 2)
        self._check_scaling('20170726_164802', 1.05, 1)
        self._check_scaling('20170726_164816', 1.05, 1)
        self._check_scaling('20170726_164831', 1.1, 1)
        self._check_scaling('20170726_164845', 1.1, 1)

        # 20% detuning only works for coarse
        self._check_scaling('20170726_164901', 1.2, 1)

        # Test running it once with showing the initial fit
        ma.FlippingAnalysis(t_start='20170726_164901',
                            options_dict={'plot_init': True})

    def _check_scaling(self, timestamp, known_detuning, places):
        a = ma.FlippingAnalysis(t_start=timestamp)
        s = a.get_scale_factor()
        self.assertAlmostEqual(s*known_detuning, 1, places=places)
        print('Scale factor {:.4f} known detuning {:.4f}'.format(
            s, known_detuning))
