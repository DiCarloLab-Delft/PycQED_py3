'''
Hacked together by Rene Vollmer
'''

import unittest
import matplotlib.pyplot as plt
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma

tau_keys = {
    ma.CoherenceTimesAnalysis.T1: 'Analysis.Fitted Params F|1>.tau.value',
    ma.CoherenceTimesAnalysis.T2: 'Analysis.Fitted Params corr_data.tau.value',
    ma.CoherenceTimesAnalysis.T2_star: 'Analysis.Fitted Params raw w0.tau.value',
}
tau_std_keys = {
    ma.CoherenceTimesAnalysis.T1: 'Analysis.Fitted Params F|1>.tau.stderr',
    ma.CoherenceTimesAnalysis.T2: 'Analysis.Fitted Params corr_data.tau.stderr',
    ma.CoherenceTimesAnalysis.T2_star: 'Analysis.Fitted Params raw w0.tau.stderr',
}
labels = {
    ma.CoherenceTimesAnalysis.T1: '_T1-VFC_res1_dac_channel_VFCQ6_',
    ma.CoherenceTimesAnalysis.T2: '_echo-VFC_res1_dac_channel_VFCQ6_',
    ma.CoherenceTimesAnalysis.T2_star: '_Ramsey-VFC_res1_dac_channel_VFCQ6_',
}

dac = 'VFCQ6'
qubit = 'Q2'
t_start = '20180412_190000'
t_stop = '20180412_210000'


class Test_Cryoscope_analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_CoherenceTimesAnalysisSingle(self):
        key = ma.CoherenceTimesAnalysis.T2_star

        a = ma.CoherenceTimesAnalysisSingle(t_start=t_start, t_stop=t_stop, label=labels[key],
                                            auto=True, extract_only=False,
                                            tau_key=tau_keys[key], tau_std_key=tau_std_keys[key],
                                            plot_versus_dac=True, dac_key='Instrument settings.fluxcurrent.' + dac,
                                            plot_versus_frequency=True,
                                            frequency_key='Instrument settings.' + qubit + '.freq_qubit')

        #np.testing.assert_('dac_arc_fitfct' in a.fit_res.keys())
        #np.testing.assert_('flux_values' in a.fit_res.keys())
        #np.testing.assert_('Ec' in a.fit_res.keys())
        #np.testing.assert_('Ej' in a.fit_res.keys())
        #np.testing.assert_('sensitivity_values' in a.fit_res.keys())

    def test_CoherenceTimesAnalysis(self):
        b = ma.CoherenceTimesAnalysis(dac_instr_names=[dac], qubit_instr_names=[qubit],
                                      t_start=t_start, t_stop=t_stop, labels=labels,
                                      tau_keys=tau_keys, tau_std_keys=tau_std_keys,
                                      plot_versus_dac=True,
                                      dac_key_pattern='Instrument settings.fluxcurrent.{DAC}',
                                      plot_versus_frequency=True,
                                      frequency_key_pattern='Instrument settings.{Q}.freq_qubit',
                                      auto=True, extract_only=False, close_figs=False, options_dict={'verbose': False})

