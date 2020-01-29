import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import residual_ZZ_analysis as rza

class Test_residual_ZZ_analysis(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
		ma.a_tools.datadir = self.datadir

	def test_matrix_diagonal(self):
		# Test if diagonal matrix elements are nan
		qubit_labels = ['D1', 'Z1', 'X', 'D3']
		ts_list = ma.a_tools.get_timestamps_in_range('20200124_182258','20200124_183814', '')
		a = rza.Residual_Crosstalk(qubits=qubit_labels, t_start=ts_list[0], t_stop=ts_list[-1])
		for i in range(len(qubit_labels)):
			self.assertTrue(np.isnan(a.proc_data_dict['quantities_of_interest']['matrix'][i,i]))

	def test_matrix_non_diagonal(self):
		# Test if non-diagonal matrix elements are close to expected data values
		qubit_labels = ['D1', 'Z1', 'X', 'D3']
		ts_list = ma.a_tools.get_timestamps_in_range('20200124_182258','20200124_183814', '')
		a = rza.Residual_Crosstalk(qubits=qubit_labels, t_start=ts_list[0], t_stop=ts_list[-1])
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][0,1], -221985, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][0,2], -482802, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][0,3], 0, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][1,0], -221897, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][1,2], 0, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][1,3], -51330, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][2,0], -495339, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][2,1], 0, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][2,3], -40532, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][3,0], 0, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][3,1], -47308, -1)
		self.assertAlmostEqual(a.proc_data_dict['quantities_of_interest']['matrix'][3,2], -41571, -1)