import unittest
import pycqed as pq
import os
import numpy as np
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import tomography_execute as tomography_execute
import qutip as qt

ma.a_tools.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')


class Test_tomography_execute(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir


    def test_tomo_analysis_cardinal_state(self):
      
      #The dataset corresponds to the 00 cardinal state.
      tomo_object = tomography_execute.TomographyExecute(timestamp='20161124_162604',tomography_type = "MLE")
      # Get the dm for 00 cardinal state  
      # returned rho is a quantum object
      rho_tomo = (tomo_object.get_density_matrix()).full()
      # get a rho target corresponding to the 00 state
      rho_target = (qt.ket2dm(qt.basis(4, 0))).full()
      #get it's fidelity to a 00 state
      #This is not the correct fidelity 
      #The correct method shhould be pauli labels
      benchmark_fidelity = np.real_if_close(np.dot(rho_tomo.flatten(),rho_target.flatten()))
      self.assertAlmostEqual(benchmark_fidelity, 0.9679030, places=6)

    



    
