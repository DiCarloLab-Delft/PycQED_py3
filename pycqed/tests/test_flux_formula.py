import numpy as np
import unittest

from pycqed.analysis import fitting_models as fit_mods


class Test_Flux_formula(unittest.TestCase):

    def test_freq_to_dac(self):
        f_max = 6.01e9
        dac_sweet_spot = .02
        E_c = 300e6
        dac_flux_coefficient = 1/.2
        for asym in [0, .5, .9]:
            dac_voltages = np.linspace(-.100, .100, 201)
            freqs = fit_mods.Qubit_dac_to_freq(
                dac_voltages, f_max=f_max, E_c=E_c,
                dac_sweet_spot=dac_sweet_spot,
                dac_flux_coefficient=dac_flux_coefficient,
                asymmetry=asym)

            self.assertEqual(np.max(freqs), f_max)
            self.assertAlmostEqual(dac_voltages[np.argmax(freqs)],
                                   dac_sweet_spot)

            # Simple model excluding assymetry
            simple_freq_model = (f_max + E_c)*np.sqrt(abs(np.cos(
                dac_flux_coefficient*(dac_voltages-dac_sweet_spot))))-E_c

            freqs = fit_mods.Qubit_dac_to_freq(
                dac_voltages, f_max=f_max, E_c=E_c,
                dac_sweet_spot=dac_sweet_spot,
                dac_flux_coefficient=dac_flux_coefficient,
                asymmetry=0)
            np.testing.assert_array_almost_equal(freqs, simple_freq_model)

    def test_dac_to_freq(self):
        f_max = 6.01e9
        dac_sweet_spot = .02
        dac_voltages_pos_branch = np.linspace(
            dac_sweet_spot, 3*dac_sweet_spot, 201)
        freqs = fit_mods.Qubit_dac_to_freq(
            dac_voltages_pos_branch, f_max=f_max, E_c=300e6,
            dac_sweet_spot=dac_sweet_spot, dac_flux_coefficient=1/.2,
            asymmetry=0)

        recovered_dac_voltages = fit_mods.Qubit_freq_to_dac(
            freqs,
            f_max=f_max, E_c=300e6,
            dac_sweet_spot=dac_sweet_spot, dac_flux_coefficient=1/.2,
            asymmetry=0, branch='positive')

        np.testing.assert_array_almost_equal(dac_voltages_pos_branch,
                                             recovered_dac_voltages)

        dac_voltages_neg_branch = np.linspace(
            dac_sweet_spot, -3*dac_sweet_spot, 201)
        freqs = fit_mods.Qubit_dac_to_freq(
            dac_voltages_neg_branch, f_max=f_max, E_c=300e6,
            dac_sweet_spot=dac_sweet_spot, dac_flux_coefficient=1/.2,
            asymmetry=0)

        recovered_dac_voltages = fit_mods.Qubit_freq_to_dac(
            freqs,
            f_max=f_max, E_c=300e6,
            dac_sweet_spot=dac_sweet_spot, dac_flux_coefficient=1/.2,
            asymmetry=0, branch='negative')

        np.testing.assert_array_almost_equal(dac_voltages_neg_branch,
                                             recovered_dac_voltages)
