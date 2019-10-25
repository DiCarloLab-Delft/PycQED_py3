import unittest
import pycqed as pq
import os
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import alignment_analysis as aa
import matplotlib.pyplot as plt


class Test_Alignment_Analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        a_tools.datadir = self.datadir

    def test_alignment_analysis(self):
        opt_dict = {'fit_options': {'model': 'complex'}}
        a = aa.AlignmentAnalysis(t_start='20190724_122351',
                                 t_stop='20190724_123047')
