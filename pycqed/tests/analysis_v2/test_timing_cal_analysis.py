import unittest
import pycqed as pq
import os
import matplotlib.pyplot as plt
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_Timing_Cal_Flux_Coarse(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_Timing_Cal_Flux_Coarse(self):
        ma.Timing_Cal_Flux_Coarse(t_start='20180409_221408',
                                  ch_idx=1, close_figs=True,
                                  ro_latency=-200e-9,
                                  flux_latency=10e-9,
                                  mw_pulse_separation=100e-9)
