import unittest
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.plotting import SI_val_to_msg_str

from pycqed.analysis.tools.plotting import format_lmfit_par, plot_lmfit_res


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
        self.assertEqual(''+unit, post_unit)

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

        xlab = ax.get_xlabel()
        ylab = ax.get_ylabel()
        self.assertEqual(xlab, 'Distance (km)')
        self.assertEqual(ylab, 'Amplitude (μV)')

    def test_SI_val_to_msg_str(self):
        val, unit = SI_val_to_msg_str(1030, 'm')
        self.assertEqual(val, str(1.03))
        self.assertEqual(unit, 'km')


class test_format_lmfit_par(unittest.TestCase):
    def test_format_lmfit_par(self):
        p = lmfit.Parameter('p')
        p.value = 5.12
        p.stderr = 0.024
        test_str = format_lmfit_par('test_par', p, end_char='\n')
        self.assertEqual(test_str, 'test_par: 5.1200$\\pm$0.0240\n')

    def test_format_lmfit_par_missing_stderr(self):
        p = lmfit.Parameter('p')
        p.value = 5.12
        test_str = format_lmfit_par('test_par', p, end_char='')
        self.assertEqual(test_str, 'test_par: 5.1200$\\pm$NaN')


class test_plot_lmfit_res(unittest.TestCase):

    def test_plot_model_result(self):
        def line(a, b, x):
            return a*x+b

        a = .1
        b = 5
        x = np.linspace(0, 20, 31)
        y = line(a, b, x)

        line_model = lmfit.Model(line, independent_vars='x')
        line_model.set_param_hint('a', value=a)
        line_model.set_param_hint('b', value=b)
        params = line_model.make_params()
        fit_res = line_model.fit(y, x=x, params=params)

        f, ax = plt.subplots()
        plot_lmfit_res(fit_res, ax=ax, plot_kws={'color': 'C1'},
                       plot_init=True, plot_init_kws={'ls': '--'})
