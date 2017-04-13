import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools

class Test_Rabi_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.rabis = [ma.Rabi_Analysis(timestamp='20170412_185618'),
            ma.Rabi_Analysis(timestamp='20170412_183928'),
            ma.Rabi_Analysis(timestamp='20170413_134244')]





    def test_Rabi_analysis(self):
        for ii in range(len(self.rabis)):
            rabi_an = self.rabis[ii]
            for tt in range(2):
                rabi_amp = rabi_an.fit_res[tt].values['period']/2.
                amp_low = 0.63
                amp_high = 0.8

                self.assertGreaterEqual(rabi_amp, amp_low)
                self.assertGreaterEqual(amp_high, rabi_amp)