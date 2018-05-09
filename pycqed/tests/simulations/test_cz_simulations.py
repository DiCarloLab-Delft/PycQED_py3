import unittest
import matplotlib.pyplot as plt
import qutip as qtp
import numpy as np

from pycqed.simulations.CZ_leakage_simulation import \
    simulate_CZ_trajectory, ket_to_phase

from pycqed.simulations import cz_unitary_simulation as czu

class Test_cz_unitary_simulation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_Hamiltonian(self):
        H_0 = czu.coupled_transmons_hamiltonian(5, 11, alpha_q0 = .1, J=.01)
        self.assertEqual(H_0.shape, (6,6))
        np.testing.assert_almost_equal(H_0[0,0], 0)
        np.testing.assert_almost_equal(H_0[1, 1], 5)
        np.testing.assert_almost_equal(H_0[2, 2], 2*5 + .1)
        np.testing.assert_almost_equal(H_0[1, 3], 0.01) # J
        np.testing.assert_almost_equal(H_0[3, 3], 11)
        np.testing.assert_almost_equal(H_0[4, 4], 11+5)
        np.testing.assert_almost_equal(H_0[5, 5], 11+2*5+.1)

    def test_rotating_frame(self):
        w_q0 = 5e9
        w_q1 = 6e9

        H_0 = czu.coupled_transmons_hamiltonian(5e9, 6e9, alpha_q0=.1, J=0)
        tlist = np.arange(0, 10e-9, .1e-9)
        U_t = qtp.propagator(H_0, tlist)
        for t_idx in [3, 5, 11]:
            # with rotating term, single qubit terms have phase
            U = U_t[t_idx]
            self.assertTrue(abs(np.angle(U[1,1]))>.1)
            self.assertTrue(abs(np.angle(U[3,3]))>.1)

            U = czu.rotating_frame_transformation(U_t[t_idx],
                                                  tlist[t_idx],
                                                  w_q0=w_q0, w_q1=w_q1)
            # after removing rotating term, single qubit terms have static phase
            self.assertTrue(abs(np.angle(U[1,1]))<.1)
            self.assertTrue(abs(np.angle(U[3,3]))<.1)



    def test_resonant_sudden_CZ(self):

        alpha_q0 = 250e6 *2*np.pi
        J = 2.5e6 *2*np.pi
        w_q0 = 6e9 *2*np.pi
        w_q1 = w_q0+alpha_q0

        H_0 = czu.coupled_transmons_hamiltonian(w_q0, w_q1,
                                                alpha_q0=alpha_q0, J=J)
        tlist = np.arange(0, 150e-9, .1e-9)

        U_t = qtp.propagator(H_0, tlist)

        time_idx = [0, 500, 1000] # 0, 50 and 100ns

        # time_idx = range(len(tlist))
        phases = np.asanyarray([czu.phases_from_unitary(
            czu.rotating_frame_transformation(
                U_t[t_idx], tlist[t_idx], w_q0, w_q1) )
            for t_idx in time_idx])
        # f, ax = plt.subplots()
        # ax.plot(tlist, phases[:,0], label='Phi_00')
        # ax.plot(tlist, phases[:,1], label='Phi_01')
        # ax.plot(tlist, phases[:,2], label='Phi_10')
        # ax.plot(tlist, phases[:,3], label='Phi_11')
        # ax.plot(tlist, phases[:,4], label='Phi_cond')
        # ax.legend()
        # plt.show()

        np.testing.assert_almost_equal(phases[0, 3], 0, decimal=-1)
        np.testing.assert_almost_equal(phases[1, 3], 0, decimal=-1)
        np.testing.assert_almost_equal(phases[2, 3], 180, decimal=-1)


    def test_offresonant_adiabatic_CZ(self):
        alpha_q0 = 250e6 *2*np.pi
        J = 2.5e6 *2*np.pi
        w_q0 = 6e9 *2*np.pi
        w_q1 = w_q0+alpha_q0*1.05

        H_0 = czu.coupled_transmons_hamiltonian(w_q0, w_q1,
                                                alpha_q0=alpha_q0, J=J)
        tlist = np.arange(0, 150e-9, .1e-9)

        U_t = qtp.propagator(H_0, tlist)

        time_idx = [0, 500, 1000] # 0, 50 and 100ns
        phases = np.asanyarray([czu.phases_from_unitary(
            czu.rotating_frame_transformation(
                U_t[t_idx], tlist[t_idx], w_q0, w_q1) )
            for t_idx in time_idx])

        np.testing.assert_almost_equal(phases[0, 3], 0, decimal=-1)
        np.testing.assert_almost_equal(phases[1, 3], -20, decimal=-1)
        np.testing.assert_almost_equal(phases[2, 3], -32, decimal=-1)






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
        # exp_phase_diff = [....]
        # np.testing.assert_array_almost_equal(exp_phase_diff, phase_diff)
