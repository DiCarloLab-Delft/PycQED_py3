import numpy as np
import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma

@unittest.expectedFailure
class Test_flipping_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_flipping_analysis(self):

        timestamps = ['20170726_164507', '20170726_164521', '20170726_164536',
                      '20170726_164550', '20170726_164605', '20170726_164619',
                      '20170726_164635', '20170726_164649', '20170726_164704',
                      '20170726_164718', '20170726_164733', '20170726_164747',
                      '20170726_164802', '20170726_164816', '20170726_164831',
                      '20170726_164845', '20170726_164901', '20170726_164915']

        # these scale factors are based on an experiment with a known
        # added detuning in the amplitude
        expected_scaling_factors = np.array([1/.8, 1/.8, 1/.9, 1/.9,
                                             1/.95, 1/.95, 1/.99, 1/.99, 1, 1,
                                             1/1.01, 1/1.01, 1/1.05, 1/1.05,
                                             1/1.1, 1/1.1, 1/1.2, 1/1.2])

        for i, ts in enumerate(timestamps):
            fa = ma.FlippingAnalysis(t_start=ts)
            s = fa.get_scaling_factor()
            print(fa.timestamps[0], s)
            self.assertEqual(s, expected_scaling_factors[i])
