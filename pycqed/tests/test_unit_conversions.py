import unittest
import numpy as np

from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor


class Test_SI_prefix_scale_factor(unittest.TestCase):

    def test_non_SI(self):
        unit = 'arb.unit.'
        scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
        self.assertEqual(scale_factor, 1)
        self.assertEqual(unit, post_unit)

    def test_SI_scale_factors(self):
        unit = 'V'
        scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
        self.assertEqual(scale_factor, 1)
        self.assertEqual(' '+unit, post_unit)

        scale_factor, post_unit = SI_prefix_and_scale_factor(val=5000,
                                                             unit=unit)
        self.assertEqual(scale_factor, 1/1000)
        self.assertEqual('k'+unit, post_unit)

        scale_factor, post_unit = SI_prefix_and_scale_factor(val=0.05,
                                                             unit=unit)
        self.assertEqual(scale_factor, 1000)
        self.assertEqual('m'+unit, post_unit)
