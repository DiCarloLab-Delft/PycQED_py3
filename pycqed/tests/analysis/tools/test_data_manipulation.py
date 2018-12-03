import unittest
import pycqed as pq
import os
import numpy as np
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.tools.data_manipulation import \
    populations_using_rate_equations

from uncertainties import ufloat

class Test_AnalysisToolsDataManipulation():
    def test_populations_using_rate_equations(self):
        V0 = 0
        V1 = 1
        V2 = .8 
        SI = np.array([.5, .4, .6, .7])
        SX = np.array([.5, .6, .55, .7])
        P0, P1, P2, M_inv = populations_using_rate_equations(SI=SI, SX=SX, 
            V0=V0, V1=V1, V2=V2)

        # First two elements all population is inverted -> no P2
        np.testing.assert_array_almost_equal(P0, [.5, .6, 0.35, 1/6])
        np.testing.assert_array_almost_equal(P1, [.5, .4, 0.4, 1/6])
        np.testing.assert_array_almost_equal(P2, [0, 0, 0.25, 4/6])

    def test_populations_using_rate_equations_float(self):
        V0 = 0
        V1 = 1
        V2 = .8 
        SI = np.array([ufloat(.5,0), ufloat(.6, .01), ufloat(.7, .2)])
        SX = np.array([ufloat(.5,0), ufloat(.55, .01), ufloat(.7, .2)])
        P0, P1, P2, M_inv = populations_using_rate_equations(SI=SI, SX=SX, 
            V0=V0, V1=V1, V2=V2)

        print('P0:', P0)
        print('P1:', P1)
        print('P2:', P2)
        P0_nom = np.array([p0.nominal_value for p0 in P0])
        P1_nom = np.array([p1.nominal_value for p1 in P1])
        P2_nom = np.array([p2.nominal_value for p2 in P2])
        # First two elements all population is inverted -> no P2
        np.testing.assert_array_almost_equal(P0_nom, 
            [.5, 0.35, 1/6])
        np.testing.assert_array_almost_equal(P1_nom, 
            [.5, .4, 1/6])
        # np.testing.assert_array_almost_equal(P2, [0, 0, 0.25, 4/6])
