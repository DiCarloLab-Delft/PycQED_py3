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
        self.rabis = [self.rabis.append(ma.Rabi_Analysis(timestamp='20170412_185618')),
            self.rabis.append(ma.Rabi_Analysis(timestamp='20170412_183928'))]





    def test_Rabi_analysis(self):
        for ii in range(len(self.rabis)):
            rabi_an = self.rabis[ii]
            for tt in range(2):
                rabi_freq = rabi_an.fit_res[tt].values['frequency']
                freq_low = 0.69
                freq_high = 0.8

                self.assertGreaterEqual(rabi_freq, freq_low)
                self.assertGreaterEqual(freq_high, rabi_freq)