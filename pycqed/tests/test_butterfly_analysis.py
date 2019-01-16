import pycqed as pq
import matplotlib.pyplot as plt
import os
from pycqed.analysis import measurement_analysis as ma
from numpy.testing import assert_almost_equal


class TestSSRODiscriminationAnalysis:

    @classmethod
    def setup_class(cls):
        cls.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = cls.datadir

    def test_butterfly_postselected(self):
        # Test the correct file is loaded
        a = ma.butterfly_analysis(timestamp='20170710_180002',
                                  close_main_fig=False, initialize=True,
                                  threshold=0.5,
                                  digitize=False, case=True)

        assert_almost_equal(a.butterfly_coeffs['F_a_butterfly'], 0.7998,
                            decimal=3)

    def test_butterfly_simple(self):
        # Test the correct file is loaded
        a = ma.butterfly_analysis(timestamp='20170710_182949',
                                  close_main_fig=False, initialize=False,
                                  threshold=0.5,
                                  digitize=False, case=True)

        assert_almost_equal(a.butterfly_coeffs['F_a_butterfly'], 0.819,
                            decimal=3)

    @classmethod
    def teardown_class(cls):
        plt.close('all')
