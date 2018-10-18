'''
Hacked together by Rene Vollmer
'''

import unittest
import matplotlib.pyplot as plt
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma

dac = 'VFCQ6'
qubit = 'Q2'
t_start = '20180414_121500'
t_stop = '20180414_141000'


class Test_DAC_scan_analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')


    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_FluxFrequency_Qubit(self):
        dic = {
            'fitparams_key': 'Fitted Params distance.f0.value',
            's21_normalize_per_dac': True,
            's21_normalize_global': True, 's21_percentile': 70,
            'verbose': True,
            'dac_key': 'Instrument settings.fluxcurrent.' + dac,
        }
        d = ma.FluxFrequency(t_start=t_start, t_stop=t_stop, label='_spectroscopy',
                             options_dict=dic, is_spectroscopy=True, temp_keys={'T_mc': 'Fridge monitor.T_MC'},
                             auto=True, do_fitting=True, close_figs=False)

        # todo: test fit results, loaded and processed data

    @unittest.expectedFailure
    def test_FluxFrequency_Resonator(self):
        dic = {
            'fitparams_key': 'Fitted Params HM.f0.value',
            'fitparams_corr_fact': 1e9,
            'qubit_freq': 6.95e9,
            'verbose': True,
            'dac_key': 'Instrument settings.fluxcurrent.' + dac,
        }
        d = ma.FluxFrequency(t_start=t_start, t_stop=t_stop, label='_Resonator',
                             options_dict=dic, is_spectroscopy=False, extract_fitparams=True,
                             auto=True, do_fitting=True, close_figs=False)

        # todo: test fit results, loaded and processed data
