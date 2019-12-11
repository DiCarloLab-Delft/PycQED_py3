import unittest
import h5py
import json
import numpy as np
import os
import pycqed as pq
import matplotlib.pyplot as plt
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.measurement_analysis as ma2


class Test_base_analysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        a_tools.datadir = self.datadir

    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @unittest.skip("FIXME: test dataset has wrong channel convention. See tests/analysis_v2/test_timedomain_analysis_v2.py::Test_Conditional_Oscillation_Analysis")
    def test_save_fit_results(self):
        # strictly speaking an integration test as it relies on the cond
        # oscillation analysis, but the only thing tested here is
        # if the value of the fit_result is saved.
        a = ma2.Conditional_Oscillation_Analysis(t_start='20181126_131143',
                                                 cal_points='gef')

        exp_val = a.fit_res['cos_fit_off'].params['amplitude'].value

        fn = a_tools.measurement_filename(a_tools.data_from_time(ts))
        with h5py.File(fn, 'r') as file:
            saved_val = float(file['Analysis']['cos_fit_off']['params']
                              ['amplitude'].attrs['value'])

        a.fit_res = {}
        a.save_fit_results()
        np.testing.assert_almost_equal(exp_val, saved_val, decimal=3)

    def test_save_quantities_of_interest(self):
        # Test based on test below to get a dummy dataset
        ts = '20161124_162604'
        a = ba.BaseDataAnalysis()
        a.proc_data_dict['quantities_of_interest'] = {'a': 5}
        a.timestamps = [ts]
        a.save_quantities_of_interest()

        fn = a_tools.measurement_filename(a_tools.data_from_time(ts))
        with h5py.File(fn, 'r') as file:
            saved_val = float(file['Analysis']['quantities_of_interest'].attrs['a'])

        assert saved_val == 5


    def test_save_load_json(self):
        # Load data from file
        a = ba.BaseDataAnalysis(
            data_file_path=os.path.join(self.datadir, '20170808',
                                        '010101_analysis_v2_json',
                                        'fake_data_20170731_010040.json'))
        a.extract_data()

        # Test if it loaded correctly
        np.testing.assert_equal(
            a.raw_data_dict, {
                'datetime': ['2017-07-31 01:00:40', '2017-07-31 01:37:24',
                             '2017-07-31 02:14:20', '2017-07-31 02:51:24',
                             '2017-07-31 03:28:44', '2017-07-31 04:05:34',
                             '2017-07-31 04:42:20', '2017-07-31 05:19:06',
                             '2017-07-31 05:55:51', '2017-07-31 06:33:05',
                             '2017-07-31 07:10:06', '2017-07-31 07:47:14',
                             '2017-07-31 08:23:59', '2017-07-31 09:00:44',
                             '2017-07-31 09:37:29', '2017-07-31 10:14:28'],
                'detuning': [0.0, -49320761.24079269, -22225390.9111607,
                             -13568964.579852352, 21392223.595482837,
                             13822537.166975206, -2590995.820591142,
                             -8951076.834515035, 6067740.757130785,
                             10811363.072895428],
                'folder': 'D:\\Experiments\\1702_Starmon\\data\\20170731'
                          '\\010040_CZ_phase_ripple_sin_QR_ker_RT_cryo1',
                'phase': [65.00124000990506, -6.851530218234939,
                          -6.020656176836422, -38.856093130306355,
                          -25.559965171823812, -8.05129115281107,
                          -5.655511651379513, -11.782325134462313,
                          -18.545062293081163, -3.0447784441939847],
                'plot_times': [0.0, 2e-09, 4e-09, 6.000000000000001e-09,
                               8e-09, 1e-08, 1.2000000000000002e-08,
                               1.4000000000000001e-08, 1.6e-08,
                               1.8000000000000002e-08],
                'timestamps': ['20170731_010040', '20170731_013724',
                               '20170731_021420', '20170731_025124',
                               '20170731_032844', '20170731_040534',
                               '20170731_044220', '20170731_051906',
                               '20170731_055551', '20170731_063305',
                               '20170731_071006', '20170731_074714',
                               '20170731_082359', '20170731_090044',
                               '20170731_093729', '20170731_101428']}
        )

        # Try saving the data again.
        a.save_data(
            savedir=os.path.join(self.datadir, '20170808',
                                 '010101_analysis_v2_json'),
            savebase='saved_by_test_all',
            fmt='json',
            tag_tstamp=False,
            key_list='auto')

        # This time use default argument for savedir
        oldFolder = a.raw_data_dict['folder']
        a.raw_data_dict['folder'] = [os.path.join(self.datadir, '20170808',
                                              '010101_analysis_v2_json')]
        a.save_data(
            savebase='saved_by_test_select_keys',
            fmt='json',
            tag_tstamp=False,
            key_list=['detuning', 'phase'])
        # Need to set old folder again to pass assertions below
        a.raw_data_dict['folder'] = oldFolder

        # Reload the data
        with open(os.path.join(self.datadir, '20170808',
                               '010101_analysis_v2_json',
                               'saved_by_test_all.json'), 'r') as file:
            loaded_dict_all = json.load(file)

        with open(os.path.join(self.datadir, '20170808',
                               '010101_analysis_v2_json',
                               'saved_by_test_select_keys.json'),
                  'r') as file:
            loaded_dict_select_keys = json.load(file)

        np.testing.assert_equal(loaded_dict_all, a.raw_data_dict)
        np.testing.assert_equal(loaded_dict_select_keys, {
                'detuning': [0.0, -49320761.24079269, -22225390.9111607,
                             -13568964.579852352, 21392223.595482837,
                             13822537.166975206, -2590995.820591142,
                             -8951076.834515035, 6067740.757130785,
                             10811363.072895428],
                'phase': [65.00124000990506, -6.851530218234939,
                          -6.020656176836422, -38.856093130306355,
                          -25.559965171823812, -8.05129115281107,
                          -5.655511651379513, -11.782325134462313,
                          -18.545062293081163, -3.0447784441939847]
                })
