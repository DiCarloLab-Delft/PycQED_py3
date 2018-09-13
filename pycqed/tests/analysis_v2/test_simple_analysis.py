import unittest
import pycqed as pq
import os
import matplotlib.pyplot as plt
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_SimpleAnalysis(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_1D_analysis_multi_file(self):
        a = ma.Basic1DAnalysis(t_start='20170726_164507',
                               t_stop='20170726_164845',
                               options_dict={'scan_label': 'flipping'})
        self.assertTrue(len(a.timestamps) > 5)

    def test_1D_analysis_single_file(self):
        # giving only a single file
        a = ma.Basic1DAnalysis(t_start='20170726_164845',
                               options_dict={'scan_label': 'flipping'})
        self.assertEqual(a.timestamps, ['20170726_164845'])

    def test_2D_analysis_multi_file(self):

        # N.B. by setting x2, x2_label and x2_unit in the options dict
        # the values can be plotted versus the varied parameter between
        # the linecuts
        a = ma.Basic2DAnalysis(t_start='20170726_164521',
                               t_stop='20170726_164845',
                               options_dict={'scan_label': 'flipping'})
        self.assertTrue(len(a.timestamps) > 5)

    def test_2D_interpolated(self):
        a=ma.Basic2DInterpolatedAnalysis(t_start='20180522_030206')
        fig_keys = list(a.figs.keys())
        exp_list_keys = ['Cost function value', 'Conditional phase',
                         'offset difference']
        self.assertEqual(fig_keys, exp_list_keys)

    def test_1D_binned_analysis(self):
        a=ma.Basic1DBinnedAnalysis(label='120543_Single_qubit_GST_QL')
