import numpy as np
from unittest import expectedFailure
from unittest import TestCase

from pycqed.simulations.pauli_transfer_matrices import(
    X,Y,Z, H, S, S2, CZ, X_theta, Y_theta, Z_theta,
    process_fidelity, average_gate_fidelity)


class TestPauliTransferProps(TestCase):

    """
    Only the most basic test for the transfer matrices.
    Intended to catch silly things like typos
    """

    def test_single_qubit_Paulis(self):
        for pauli in [X, Y, Z]:
            pauli2 = np.dot(pauli,pauli)
            np.testing.assert_array_equal(pauli2,
                                          np.eye(4, dtype=int))

    def test_basic_pauli_ops(self):
        exp_Z = np.dot(X, Y)
        np.testing.assert_array_equal(Z, exp_Z)

        exp_mZ = np.dot(Y, X)
        np.testing.assert_array_equal(Z, exp_mZ)

    def test_Hadamard(self):
        np.testing.assert_array_equal(np.dot(H, H),
                                      np.eye(4, dtype=int))
    def test_S_gate(self):
        np.testing.assert_array_equal(np.dot(S, S), S2)
        np.testing.assert_array_equal(np.dot(S, S2), np.eye(4,dtype=int))

    def test_cphase(self):
            CZ2 = np.linalg.multi_dot([CZ, CZ])
            CZ2 = np.dot(CZ, CZ)
            np.testing.assert_array_equal(CZ2, np.eye(16, dtype=int))


    def test_cphase(self):
        CZ2 = np.linalg.multi_dot([CZ, CZ])
        CZ2 = np.dot(CZ, CZ)
        np.testing.assert_array_equal(CZ2, np.eye(16, dtype=int))


    def test_angle_rotation_static(self):
        np.testing.assert_array_almost_equal(X_theta(180), X)
        np.testing.assert_array_almost_equal(Y_theta(180), Y)
        np.testing.assert_array_almost_equal(Z_theta(180), Z)


    def test_angle_rotation_unit(self):
        np.testing.assert_array_almost_equal(X_theta(32, unit='deg'),
                                      X_theta(np.deg2rad(32), unit='rad'))
        np.testing.assert_array_almost_equal(Y_theta(18, unit='deg'),
                                      Y_theta(np.deg2rad(18), unit='rad'))
        np.testing.assert_array_almost_equal(Z_theta(180, unit='deg'),
                                      Z_theta(np.deg2rad(180), unit='rad'))


    def test_angle_rotations(self):

        # Expressed in the Pauli basis
        X_state = np.array([0, 1, 0, 0])
        Y_state = np.array([0, 0, 1, 0])
        Z_state = np.array([0, 0, 0, 1])

        np.testing.assert_array_almost_equal(np.dot(Z_theta(90), X_state), Y_state)
        np.testing.assert_array_almost_equal(np.dot(Z_theta(-90), X_state), -Y_state)

        np.testing.assert_array_almost_equal(np.dot(Z_theta(90), Z_state), Z_state)

        np.testing.assert_array_almost_equal(np.dot(X_theta(90), Y_state), Z_state)
        np.testing.assert_array_almost_equal(np.dot(X_theta(-90), Y_state), -Z_state)

    def test_fidelity_calculation(self):

        Z_160 = Z_theta(160)
        F_pro = process_fidelity(Z_160, Z)
        self.assertAlmostEqual(F_pro, 0.9698463)

        F_avg = average_gate_fidelity(Z_160, Z)
        self.assertAlmostEqual(F_avg, 0.979897540)