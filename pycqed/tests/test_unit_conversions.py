import unittest
import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel


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


class test_SI_unit_aware_labels(unittest.TestCase):

    def test_label_scaling(self):
        """
        This test creates a dummy plot and checks if the tick labels are
        rescaled correctly
        """
        f, ax = plt.subplots()
        x = np.linspace(-6, 6, 101)
        y = np.cos(x)
        ax.plot(x*1000, y/1e5)

        set_xlabel(ax, 'Distance', 'm')
        set_ylabel(ax, 'Amplitude', 'V')

        xticks = ax.get_xticks()
        xtick_labels = ax.get_xticklabels()
        xtick_label_vals = [float(xl.get_text()) for xl in xtick_labels]
        for xt, xl in zip(xticks, xtick_label_vals):
            self.assertEqual(xt*1e-3, xl)

        yticks = ax.get_yticks()
        ytick_labels = ax.get_yticklabels()
        ytick_label_vals = [float(yl.get_text()) for yl in ytick_labels]
        for yt, yl in zip(yticks, ytick_label_vals):
            self.assertAlmostEqual(yt*1e6, yl)
