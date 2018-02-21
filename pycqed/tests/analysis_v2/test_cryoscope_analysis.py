import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_Cryoscope_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_RamZFluxArc(self):
        a=ma.RamZFluxArc(t_start='20180205_105633', t_stop='20180205_120210')

        # test dac arc conversion
        # For this to work all other parts have to work
        amps = a.freq_to_amp([.5e9, .6e9, .8e9])
        exp_amps = np.array([ 0.5357841 ,  0.58333725,  0.66727005])
        np.testing.assert_array_almost_equal(amps, exp_amps)