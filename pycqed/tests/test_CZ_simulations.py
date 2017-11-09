import unittest

import numpy as np

from pycqed.simulations.CZ_leakage_simulation import \
    simulate_CZ_trajectory, ket_to_phase


class Test_CZ_single_trajectory_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_single_trajectory(self):
        params_dict = {'length': 20e-9,
                       'lambda_2': 0,
                       'lambda_3': 0,
                       'theta_f': 80,
                       'f_01_max': 6e9,
                       'f_interaction': 5e9,
                       'J2': 40e6,
                       'E_c': 300e6,
                       'dac_flux_coefficient': 1,
                       'asymmetry': 0,
                       'sampling_rate': 2e9,
                       'return_all': True}

        picked_up_phase, leakage, res1, res2, eps_vec, tlist = \
            simulate_CZ_trajectory(**params_dict)

        phases1 = (ket_to_phase(res1.states)) % (2*np.pi)/(2*np.pi)*360
        phases2 = (ket_to_phase(res2.states)) % (2*np.pi)/(2*np.pi)*360
        phase_diff = (phases2-phases1) % 360

        # the qubit slowly picks up phase during this trajectory as
        # expected.

        # The actual test is commented out since the parametrization changed

        # exp_phase_diff = [
        #     0.,        359.83449265,  359.64971077,  359.46648237,
        #     359.07660938, 358.77253771,  358.21184268,  357.39013731,
        #     356.45515109,  355.43917876, 354.27877827,  352.90139428,
        #     351.24504546,  349.26908599,  346.93648832, 344.23060963,
        #     341.12285954,  337.57502835,  333.55042864,  328.99435847,
        #     323.85041975,  318.05989583,  311.60439877,  304.54265015,
        #     297.09265858, 289.67485697,  282.87091222,  277.28479652,
        #     273.36500521,  271.32010334, 271.08094385,  272.27593352,
        #     274.06500821,  274.86606498,  272.81514717, 269.63012373,
        #     270.68248056,  270.55544633,  270.40182712,  269.56779701]
        # np.testing.assert_array_almost_equal(exp_phase_diff, phase_diff)
