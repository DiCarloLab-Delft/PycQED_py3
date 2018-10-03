'''
Hacked together by Rene Vollmer
Edited by Adriaan
'''

import unittest
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


class Test_CoherenceTimesAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_CoherenceTimesAnalysisSingle(self):
        key = ma.CoherenceTimesAnalysis.T2_star

        a = ma.CoherenceTimesAnalysisSingle(
            t_start=t_start, t_stop=t_stop, label=labels[key],
            auto=True, extract_only=False,
            tau_key=tau_keys[key], tau_std_key=tau_std_keys[key],
            plot_versus_dac=True, dac_key='Instrument settings.fluxcurrent.' + dac,
            plot_versus_frequency=True,
            frequency_key='Instrument settings.' + qubit + '.freq_qubit')

    def test_CoherenceTimesAnalysis(self):
        a = ma.CoherenceTimesAnalysis(
            t_start='20181002_190542', t_stop='20181002_203700',
            dac_instr_names=['FBL_D1'], qubit_instr_names=['D1'],
            use_chisqr=False)

        expected_fig_keys = {'D1_coherence_gamma',
                             'D1_coherence_times_dac_relation',
                             'D1_coherence_times_flux_relation',
                             'D1_coherence_times_freq_relation',
                             'D1_coherence_times_sensitivity_relation',
                             'D1_coherence_times_time_stability',
                             'coherence_ratios_flux',
                             'coherence_ratios_sensitivity'}

        self.assertTrue(set(a.figs.keys()).issubset(expected_fig_keys))

        self.assertAlmostEqual(a.fit_res['D1']['gamma_intercept'],
                               62032.192, places=2)
        self.assertAlmostEqual(a.fit_res['D1']['sqrtA_echo'],
                               -0.058470, places=4)




class Test_AliasedCoherenceTimesAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_AliasedCoherenceTimesAnalysisSingle(self):
        a = ma.AliasedCoherenceTimesAnalysisSingle(t_start='20181002_153626',
                                                   ch_idxs=[2, 3])
        pars = a.fit_res['coherence_decay'].params
        self.assertAlmostEqual(pars['tau'], 3.43e-6, places=7)
        self.assertAlmostEqual(pars['A'], 0.741, places=3)
        self.assertAlmostEqual(pars['n'], 1.93, places=2)
        self.assertAlmostEqual(pars['o'], 0.0254, places=3)

        self.assertEqual(a.raw_data_dict['detuning'][0], -800e6)
