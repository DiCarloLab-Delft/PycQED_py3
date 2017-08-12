import unittest
import json
import numpy as np
import os
import pycqed as pq
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba


class Test_base_analysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        a_tools.datadir = self.datadir

    def test_save_load_json(self):
        # Load data from file
        a = ba.BaseDataAnalysis(
            data_file_path=os.path.join(self.datadir, '20170808',
                                        '010101_analysis_v2_json',
                                        'fake_data_20170731_010040.json'))
        a.extract_data()

        # Test if it loaded correctly
        np.testing.assert_equal(
            a.data_dict, {
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
        oldFolder = a.data_dict['folder']
        a.data_dict['folder'] = [os.path.join(self.datadir, '20170808',
                                              '010101_analysis_v2_json')]
        a.save_data(
            savebase='saved_by_test_select_keys',
            fmt='json',
            tag_tstamp=False,
            key_list=['detuning', 'phase'])
        # Need to set old folder again to pass assertions below
        a.data_dict['folder'] = oldFolder

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

        np.testing.assert_equal(loaded_dict_all, a.data_dict)
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
