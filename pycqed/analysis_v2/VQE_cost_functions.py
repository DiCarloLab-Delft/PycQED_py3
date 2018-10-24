import numpy as np
import sys
sys.path.append('D:/repository/PycQED_py3')


class VQE_cost_functions(object):

    def __init__(self,
                 path_file,
                 expect_values):
        """
        Class is instantiated with the hydrogen data path file and measured expectation values.
        The expectation values are in the following order II, IZ, ZI, ZZ, XX, YY

        """
        self.path_file = path_file
        self.hydrogen_data = np.loadtxt(self.path_file, unpack=False)
        self.interatomic_distances = self.hydrogen_data[:, 0]
        self.weight_of_pauli_terms = self.hydrogen_data[:, 1:7]
        self.expect_values = expect_values

    def cost_function_bare_VQE(self, distance_index):
        cost_func_bare = np.dot(
            self.expect_values, self.weight_of_pauli_terms[distance_index, :])
        return cost_func_bare

    def get_pauli_ops(self):
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.identity(2)
        II = np.kron(I, I)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)
        ZZ = np.kron(Z, Z)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        return II, IZ, ZI, ZZ, XX, YY

    def get_hamiltonian(self, distance_index):
        terms = self.get_pauli_ops()
        gs = self.weight_of_pauli_terms[distance_index, :]
        ham = np.zeros((4,4), dtype=np.complex128)
        for i,g in enumerate(gs):
            ham += g*terms[i]
        return ham