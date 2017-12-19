import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


class Test_Rabi_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_Rabi_analysis(self):
        rabis = [ma.Rabi_Analysis(timestamp='20170412_185618'),
                 ma.Rabi_Analysis(timestamp='20170412_183928'),
                 ma.Rabi_Analysis(timestamp='20170413_134244')]
        for rabi_an in rabis:
            for tt in range(2):
                rabi_amp = rabi_an.fit_res[tt].values['period']/2.
                amp_low = 0.63
                amp_high = 0.8

                self.assertGreaterEqual(rabi_amp, amp_low)
                self.assertGreaterEqual(amp_high, rabi_amp)

        a = ma.Rabi_Analysis(timestamp='20170607_160504')
        self.assertAlmostEqual(a.fit_res[0].values['period'], 0.3839, places=2)
        self.assertAlmostEqual(a.fit_res[1].values['period'], 0.3839, places=2)

        a = ma.Rabi_Analysis(timestamp='20170607_160504', auto=False)
        self.assertAlmostEqual(a.get_measured_amp180(), 0.3839/2, places=2)

    def test_Rabi_single_weight(self):
        a = ma.Rabi_Analysis(timestamp='20170607_211203')
        self.assertAlmostEqual(a.fit_res[0].values['period'], 0.3839, places=2)

        a = ma.Rabi_Analysis(timestamp='20170607_211203', auto=False)
        self.assertAlmostEqual(a.get_measured_amp180(), 0.3839/2, places=2)

    def test_Rabi_other_params(self):

        a = ma.Rabi_Analysis(timestamp='20170607_160504')
        self.assertAlmostEqual(1/a.fit_result.values['frequency'], 0.3839, places=2)

        a = ma.Rabi_Analysis(timestamp='20170607_160504')
        self.assertAlmostEqual(a.rabi_amplitudes['piPulse'], 0.3839/2, places=2)


class Test_DoubleFreq_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_doublefreq_fit(self):
        a = ma.DoubleFrequency(timestamp='20170201_122018')
        self.assertEqual(
            a.folder,
            os.path.join(self.datadir, '20170201', '122018_Ramsey_AncT'))
        fit_res = a.fit_res.best_values

        # Test if the fit gives the expected means
        self.assertAlmostEqual(fit_res['osc_offset'], 0.507, places=3)
        self.assertAlmostEqual(fit_res['phase_1'], -0.006, places=3)
        self.assertAlmostEqual(fit_res['phase_2'], 6.293, places=3)
        self.assertAlmostEqual(fit_res['freq_1']*1e-6, 0.286, places=3)
        self.assertAlmostEqual(fit_res['freq_2']*1e-6, 0.235, places=3)
        self.assertAlmostEqual(fit_res['tau_1']*1e6, 23.8, places=1)
        self.assertAlmostEqual(fit_res['tau_2']*1e6, 15.1, places=1)
        self.assertAlmostEqual(fit_res['amp_1'], 0.25, places=2)
        self.assertAlmostEqual(fit_res['amp_2'], 0.25, places=2)


class test_ramsey_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_ramsey_IQ_data(self):
        a = ma.Ramsey_Analysis(timestamp='20170607_145645')
        fpar = a.fit_res.best_values
        self.assertAlmostEqual(fpar['tau']*1e6, 19.577, places=2)
        self.assertAlmostEqual(fpar['exponential_offset'], 0.5057, places=2)
        self.assertAlmostEqual(fpar['frequency'], 250349.150, places=2)

    def test_ramsey_single_weight(self):
        a = ma.Ramsey_Analysis(timestamp='20170607_211144')
        fpar = a.fit_res.best_values
        self.assertAlmostEqual(fpar['tau']*1e6, 8.9793, places=2)
        self.assertAlmostEqual(fpar['exponential_offset'], 0.5057, places=2)
        self.assertAlmostEqual(fpar['frequency'], 61135.024, places=2)


class test_echo_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_echo_single_weight(self):
        a = ma.Echo_analysis(timestamp='20170607_211611')
        fpar = a.fit_res.best_values
        self.assertAlmostEqual(fpar['tau']*1e6, 12.7, places=2)


class test_allxy_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_allxy_IQ_data(self):
        a = ma.AllXY_Analysis(timestamp='20170607_161456')
        self.assertAlmostEqual(a.deviation_total, 0.02855964154)

    def test_allxy_single_weight(self):
        a = ma.AllXY_Analysis(timestamp='20170607_211630')
        self.assertAlmostEqual(a.deviation_total, 0.0335, places=3)


class test_t1_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_T1_IQ_data(self):
        a = ma.T1_Analysis(timestamp='20170607_152324')
        self.assertAlmostEqual(a.T1*1e6, 35.0788, places=3)

    def test_T1_single_weight(self):
        a = ma.T1_Analysis(timestamp='20170607_152324')
        self.assertAlmostEqual(a.T1*1e6, 35.0788, places=3)

    def test_loading_T1_fit_res_from_file(self):
        a = ma.T1_Analysis(timestamp='20170607_210448', auto=False)
        T1 = a.get_measured_T1()[0]
        self.assertAlmostEqual(T1*1e6, 18.0505, places=3)


class test_RB_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_RB_IQ_data(self):
        a = ma.RandomizedBenchmarking_Analysis(timestamp='20170607_210655')
        fpar = a.fit_res.best_values
        self.assertAlmostEqual(fpar['p'], 0.99289, places=3)
        F = a.fit_res.params['fidelity_per_Clifford'].value*100
        self.assertAlmostEqual(F, 99.64495, places=3)

    @unittest.expectedFailure
    def test_RB_single_weight(self):
        raise NotImplementedError


class test_motzoi_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_motzoi_IQ_no_cal_pts(self):
        a = ma.Motzoi_XY_analysis(timestamp='20170607_161234')
        self.assertAlmostEqual(a.optimal_motzoi, -0.3202, places=2)

    def test_motzoi_single_weight(self):
        a = ma.Motzoi_XY_analysis(timestamp='20170607_210555')
        self.assertAlmostEqual(a.optimal_motzoi, -0.2856, places=2)


class Test_qscale_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_qscale_analysis(self):
        a = ma.QScale_Analysis(timestamp='20170929_165005')
        self.assertAlmostEqual(a.optimal_qscale['qscale'], 0.03178, places=2)

        a = ma.QScale_Analysis(timestamp='20170929_205730')
        self.assertAlmostEqual(a.optimal_qscale['qscale'], 0.07973, places=2)
