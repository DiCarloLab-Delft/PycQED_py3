import unittest
import numpy as np
import pycqed as pq
import os
import matplotlib.pyplot as plt
from pycqed.analysis_v2 import measurement_analysis as ma
from pycqed.analysis_v2 import readout_analysis as ra

# Add test: 20180508\182642 - 183214


class Test_SSRO_auto_angle(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_angles(self):
        tp = 2*np.pi
        ro_amp_high_factor = 0.1
        ts = '20180508_182642'
        te = '20180508_183214'
        for angle in np.arange(0, 360, 30):
            options_dict = {
                'verbose': True,
                'rotation_angle': angle*np.pi/180,
                'auto_rotation_angle': True,
                'fixed_p01': 0,
                'fixed_p10': 0,
                'nr_bins': 100,
            }
            label = 'SSRO_%d_%.2f' % (angle, ro_amp_high_factor*100)
            aut = ma.Singleshot_Readout_Analysis(t_start=ts, t_stop=te,
                                                 label=label,
                                                 extract_only=True,
                                                 do_fitting=True,
                                                 options_dict=options_dict)
            aut_angle = aut.proc_data_dict['raw_offset'][2]
            aut_angle = aut_angle % tp
            aut_snr = aut.fit_res['shots_all'].params['SNR'].value

            options_dict['auto_rotation_angle'] = False
            opt = ma.Singleshot_Readout_Analysis(t_start=ts, t_stop=te,
                                                 label=label,
                                                 extract_only=True,
                                                 do_fitting=True,
                                                 options_dict=options_dict)
            opt_angle = opt.proc_data_dict['raw_offset'][2]
            opt_angle = opt_angle % tp
            opt_snr = opt.fit_res['shots_all'].params['SNR'].value
            da = min(abs(aut_angle-opt_angle),
                     abs(aut_angle-opt_angle+2*np.pi),
                     abs(aut_angle-opt_angle-2*np.pi))

            # Check if the angle was found within a few degrees
            self.assertLess(da*180/np.pi, 9)

            # Check if the SNRs roughly make sense
            self.assertLess(aut_snr, 1.1)
            self.assertLess(opt_snr, 1.1)
            self.assertGreater(aut_snr, 0.55)
            self.assertGreater(opt_snr, 0.55)


class Test_SSRO_discrimination_analysis(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def assertBetween(self, value, min_v, max_v):
        """Fail if value is not between min and max_v (inclusive)."""
        self.assertGreaterEqual(value, min_v)
        self.assertLessEqual(value, max_v)

    def test_SSRO_analysis_basic_1D(self):
        t_start = '20171016_135112'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           options_dict={'plot_init': True})
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw'],
                                       -3.66, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.922, decimal=3)
        self.assertBetween(a.proc_data_dict['threshold_fit'], -3.69, -3.62)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.920, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_discr'],
                                       -3.64, decimal=1)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       0.996, decimal=2)

    def test_SSRO_analysis_basic_1D_wrong_peak_selected(self):
        # This fit failed when I made a typo in the peak selection part
        t_start = '20171016_171715'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           extract_only=True)
        self.assertBetween(a.proc_data_dict['threshold_raw'], -3.3, -3.2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.944, decimal=2)
        self.assertBetween(a.proc_data_dict['threshold_fit'], -3.3, -3.2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.944, decimal=2)
        self.assertBetween(a.proc_data_dict['threshold_discr'], -3.3, -3.2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       0.99, decimal=2)

    def test_SSRO_analysis_basic_1D_misfit(self):
        # This dataset failed before I added additional constraints to the
        # guess
        t_start = '20171016_181021'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           extract_only=True)
        self.assertBetween(a.proc_data_dict['threshold_raw'], -1, -0.7)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.949, decimal=2)
        self.assertBetween(a.proc_data_dict['threshold_fit'], -1, -.7)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.945, decimal=2)
        self.assertBetween(a.proc_data_dict['threshold_discr'], -1, -.7)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       1.000, decimal=2)
        self.assertLess(a.proc_data_dict['residual_excitation'], 0.09)
        np.testing.assert_almost_equal(
            a.proc_data_dict['measurement_induced_relaxation'], 0.1,
            decimal=1)


class Test_readout_analysis_functions(unittest.TestCase):
    def test_get_arb_comb_xx_label(self):

        labels = ra.get_arb_comb_xx_label(2, qubit_idx=0)
        np.testing.assert_equal(labels[0], 'x0')
        np.testing.assert_equal(labels[1], 'x1')
        np.testing.assert_equal(labels[2], 'x2')

        labels = ra.get_arb_comb_xx_label(2, qubit_idx=1)
        np.testing.assert_equal(labels[0], '0x')
        np.testing.assert_equal(labels[1], '1x')
        np.testing.assert_equal(labels[2], '2x')

        labels = ra.get_arb_comb_xx_label(nr_of_qubits=4, qubit_idx=1)
        np.testing.assert_equal(labels[0], 'xx0x')
        np.testing.assert_equal(labels[1], 'xx1x')
        np.testing.assert_equal(labels[2], 'xx2x')

        labels = ra.get_arb_comb_xx_label(nr_of_qubits=4, qubit_idx=3)

        np.testing.assert_equal(labels[0], '0xxx')
        np.testing.assert_equal(labels[1], '1xxx')
        np.testing.assert_equal(labels[2], '2xxx')

    def test_get_assignment_fid_from_cumhist(self):
        chist_0 = np.array([0, 0, 0, 0, 0, .012,
                            .068, .22, .43, .59, .78, 1, 1])
        chist_1 = np.array([0, 0.01, .05, .16, .21,
                            .24, .38, .62, .81, 1, 1, 1, 1])
        centers = np.linspace(0, 1, len(chist_0))

        # Fidelity for identical distributions should be 0.5 (0.5 is random)
        fid, th = ra.get_assignement_fid_from_cumhist(
            chist_0, chist_0, centers)
        self.assertEqual(fid, 0.5)

        # Test on the fake distribution
        fid, threshold = ra.get_assignement_fid_from_cumhist(
            chist_0, chist_1)
        np.testing.assert_almost_equal(fid, 0.705)
        np.testing.assert_almost_equal(threshold, 9)
        # Test on the fake distribution
        fid, threshold = ra.get_assignement_fid_from_cumhist(
            chist_0, chist_1, centers)
        np.testing.assert_almost_equal(fid, 0.705)
        np.testing.assert_almost_equal(threshold, 0.75)


class Test_multiplexed_readout_analysis(unittest.TestCase):


    def test_multiplexed_readout_analysis(self):
        timestamp='20190916_184929'
        
    #     t_start = '20180323_150203'
    #     t_stop = t_start
    #     a = ma.Multiplexed_Readout_Analysis(t_start=t_start, t_stop=t_stop,
    #                                         qubit_names=['QR', 'QL'])
    #     np.testing.assert_almost_equal(a.proc_data_dict['F_ass_raw QL'],
    #                                    0.72235812133072408)

    #     np.testing.assert_almost_equal(a.proc_data_dict['F_ass_raw QR'],
    #                                    0.81329500978473579)

    #     np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw QL'],
    #                                    1.9708007812500004)
    #     np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw QR'],
    #                                    -7.1367667055130006)

    # def test_name_assignement(self):
    #     t_start = '20180323_150203'
    #     t_stop = t_start
    #     a = ma.Multiplexed_Readout_Analysis(t_start=t_start, t_stop=t_stop)
    #     np.testing.assert_equal(a.proc_data_dict['qubit_names'], ['q1', 'q0'])

    #     a = ma.Multiplexed_Readout_Analysis(t_start=t_start, t_stop=t_stop,
    #                                         qubit_names=['QR', 'QL'])
    #     np.testing.assert_equal(a.proc_data_dict['qubit_names'], ['QR', 'QL'])
