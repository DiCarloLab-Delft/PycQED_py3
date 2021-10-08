'''
Hacked together by Rene Vollmer
Edited by Adriaan
'''
import json
import numpy as np
import unittest

import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma
import pycqed.analysis_v2.coherence_analysis as ca

tau_keys = {
    ca.CoherenceTimesAnalysis_old.T1: 'Analysis.Fitted Params F|1>.tau.value',
    ca.CoherenceTimesAnalysis_old.T2: 'Analysis.Fitted Params corr_data.tau.value',
    ca.CoherenceTimesAnalysis_old.T2_star: 'Analysis.Fitted Params raw w0.tau.value',
}
tau_std_keys = {
    ca.CoherenceTimesAnalysis_old.T1: 'Analysis.Fitted Params F|1>.tau.stderr',
    ca.CoherenceTimesAnalysis_old.T2: 'Analysis.Fitted Params corr_data.tau.stderr',
    ca.CoherenceTimesAnalysis_old.T2_star: 'Analysis.Fitted Params raw w0.tau.stderr',
}
labels = {
    ca.CoherenceTimesAnalysis_old.T1: '_T1-VFC_res1_dac_channel_VFCQ6_',
    ca.CoherenceTimesAnalysis_old.T2: '_echo-VFC_res1_dac_channel_VFCQ6_',
    ca.CoherenceTimesAnalysis_old.T2_star: '_Ramsey-VFC_res1_dac_channel_VFCQ6_',
}

dac = 'VFCQ6'
qubit = 'Q2'
t_start = '20180412_190000'
t_stop = '20180412_210000'


class Test_CoherenceAnalysis_Helpers(unittest.TestCase):
    def test_calculate_n_avg(self):
        pass

    def test_arch(self):
        freq = ca.arch(0, 300e6, 10e9, 0, .5)
        self.assertAlmostEqual(freq, 4598979485.566357, places=2)

    def test_partial_omega_over_flux(self):
        deriv = ca.partial_omega_over_flux(0, 300e6, 10e9)
        self.assertAlmostEqual(deriv, 0)

        deriv = ca.partial_omega_over_flux(0.3, 300e6, 10e9)
        self.assertLess(deriv, 0)

        deriv = ca.partial_omega_over_flux(-0.3, 300e6, 10e9)
        self.assertGreater(deriv, 0)

        deriv = ca.partial_omega_over_flux(0.75, 300e6, 10e9)
        self.assertGreater(deriv, 0)

    def test_fit_frequencies(self):
        dac_vals = np.linspace(-.3, .4, 41)
        Ec = 300e6
        Ej = 10e9
        offset = 0
        dac0 = .5
        freqs = ca.arch(dac_vals, Ec, Ej, offset, dac0)

        fit_res = ca.fit_frequencies(dac_vals, freqs)
        # fit_res.plot_fit(show_init=True) # For debug purposes
        pars = fit_res.params

        self.assertAlmostEqual(Ec, pars['Ec'].value, places=-4)
        self.assertAlmostEqual(Ej, pars['Ej'].value, places=-4)
        self.assertAlmostEqual(offset, pars['offset'].value, places=2)
        self.assertAlmostEqual(dac0, pars['dac0'].value, places=2)

    def test_residual_Gamma(self):
        pass

    def test_fit_gammas(self):
        pass


class Test_PSD_Analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

        with open(os.path.join(self.datadir, 'coherence_table.json')) as f:
            self.testdata_table = np.array(json.load(f))

    def test_PSD_Analysis_gamma_intercept(self):
        a = ca.PSD_Analysis(self.testdata_table)

        # fixme: this value is based on a hardcoded constant
        sqrtA_echo = a[0]
        self.assertAlmostEqual(sqrtA_echo,
                               -0.059, places=3)


class Test_CoherenceAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

        with open(os.path.join(self.datadir, 'coherence_table.json')) as f:
            self.testdata_table = np.array(json.load(f))
        self.a = ca.CoherenceAnalysis(
            self.testdata_table,
            t_start='20181002_190542', t_stop='20181002_203700',
            options_dict={'tag_tstamp': False, 'save_figs': False})

    def test_CoherenceAnalysis_quantities(self):
        self.assertAlmostEqual(
            self.a.proc_data_dict['sqrtA_echo']*1e6,
            -0.059, places=3)

    def test_CoherenceAnalysis_figs(self):
        expected_fig_keys = {'coherence_times', 'coherence_ratios',
                             'dac_arc', 'gamma_fit'}
        self.assertTrue(set(self.a.figs.keys()).issubset(expected_fig_keys))


class Test_CoherenceTimesAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_CoherenceTimesAnalysisSingle(self):
        key = ca.CoherenceTimesAnalysis_old.T2_star

        a = ma.CoherenceTimesAnalysisSingle(
            t_start=t_start, t_stop=t_stop, label=labels[key],
            auto=True, extract_only=False,
            tau_key=tau_keys[key], tau_std_key=tau_std_keys[key],
            plot_versus_dac=True, dac_key='Instrument settings.fluxcurrent.' + dac,
            plot_versus_frequency=True,
            frequency_key='Instrument settings.' + qubit + '.freq_qubit')

        a = ca.CoherenceTimesAnalysisSingle(t_start='20190702_160000', t_stop='20190702_235900',
                                            label='T1_D1',
                                            dac_key='Instrument settings.fluxcurrent.FBL_D1',
                                            frequency_key='Instrument settings.D1.freq_qubit',
                                            fit_T1_vs_freq=True,
                                            options_dict={'guess_mode_frequency': [6.5e9, 7.1e9]})

        self.assertAlmostEqual(a.fit_res['Q_qubit'],
                               669405.3002364064, places=-2)
        self.assertAlmostEqual(a.fit_res['fres1'],
                               6478982350.367288, places=-4)
        self.assertAlmostEqual(a.fit_res['gres1'],
                               15719845.798823722, places=-3)
        self.assertAlmostEqual(a.fit_res['kappares1'],
                               777397213.3098346, places=-4)
        self.assertAlmostEqual(a.fit_res['fres2'],
                               7105933947.128368, places=-4)
        self.assertAlmostEqual(a.fit_res['gres2'],
                               15860920.61850709, places=-3)
        self.assertAlmostEqual(a.fit_res['kappares2'],
                               452876320.52273417, places=-4)

    def test_CoherenceTimesAnalysis_old(self):
        a = ca.CoherenceTimesAnalysis_old(
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
                               62032.192, places=-2)
        self.assertAlmostEqual(a.fit_res['D1']['sqrtA_echo'],
                               -0.059, places=3)


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
