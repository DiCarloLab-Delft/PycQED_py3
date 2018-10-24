import unittest
import matplotlib.pyplot as plt
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_GST_data_extraction_analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_GST_SingleQubit_DataExtraction(self):
        a = ma.GST_SingleQubit_DataExtraction(label='131808_Single_qubit_GST')
        ds = a.proc_data_dict['dataset']

        exp_val_m1 = {('0',): 226.0, ('1',): 274.0}
        val_m1 = list(ds.values())[-2].allcounts
        self.assertDictEqual(exp_val_m1, val_m1)

    def test_GST_TwoQubit_DataExtraction(self):
        a = ma.GST_TwoQubit_DataExtraction(label='155752_Two_qubit_GST')
        ds = a.proc_data_dict['dataset']
        exp_val_0 = {('00',): 100.0, ('01',): 26.0, ('10',): 4.0, ('11',): 2.0}
        val_0 = list(ds.values())[0].allcounts
        self.assertDictEqual(exp_val_0, val_0)

        exp_val_m1 = {('00',): 32.0, ('01',): 40.0, ('10',): 21.0, ('11',): 39.0}
        val_m1 = list(ds.values())[-2].allcounts
        self.assertDictEqual(exp_val_m1, val_m1)
