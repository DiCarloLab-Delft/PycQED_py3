import numpy as np
from copy import deepcopy
import unittest
import scipy

# For keeping self contained only
import sys
import os
PycQEDdir = (os.path.abspath('../..'))
sys.path.append(PycQEDdir)
print('PycQEDdir:', PycQEDdir)

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)

from pycqed.measurement.randomized_benchmarking.clifford_group \
    import(clifford_lookuptable)
import measurement.randomized_benchmarking.randomized_benchmarking \
    as rb



# Basic states
# desnity matrices in pauli basis
X0 = 1/np.sqrt(2) * np.matrix([1,  1, 0, 0]).T
X1 = 1/np.sqrt(2) * np.matrix([1,  -1, 0, 0]).T
Y0 = 1/np.sqrt(2) * np.matrix([1, 0, 1, 0]).T
Y1 = 1/np.sqrt(2) * np.matrix([1, 0, -1, 0]).T
Z0 = 1/np.sqrt(2) * np.matrix([1, 0, 0, 1]).T
Z1 = 1/np.sqrt(2) * np.matrix([1, 0, 0, -1]).T

polar_states = [X0, X1, Y0, Y1, Z0, Z1]

# Superoperators for our gate set
Gi = np.eye(4)
Gx90 = np.matrix([[1.0, 0, 0, 0],
                 [0, 1.0, 0, 0],
                 [0, 0, 0, -1],
                 [0, 0, 1.0, 0]])

Gx180 = np.matrix([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, -1., 0],
                  [0, 0, 0, -1.]])

Gy90 = np.matrix([[1.0, 0, 0, 0],
                 [0, 0, 0, 1.],
                 [0, 0, 1., 0],
                 [0, -1, 0, 0]])


Gy180 = np.matrix([[1.0, 0, 0, 0],
                  [0, -1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, -1.0]])


# Test inner product between states

def invert_unitary_component_PTM(PTM):
    """
    inverts only the unitary part of a superoperator in the Pauli basis
    uses property that the first column corresponds to the non-unitary
    (T1, T2) errors.
    """
    assert(np.shape(PTM) == (4, 4))
    # return PTM
    # for comparing under assumption that X180 affects FRB same as mX180
    unitary_part = PTM[1:, 1:]
    newPTM = deepcopy(PTM)
    newPTM[1:, 1:] = unitary_part.T
    return np.matrix(newPTM)

Ideal_gates = {'I': Gi,
               'X90': Gx90,
               'X180': Gx180,
               'Y90': Gy90,
               'Y180': Gy180,
               'mX90': invert_unitary_component_PTM(Gx90),
               'mX180': invert_unitary_component_PTM(Gx180),
               'mY90': invert_unitary_component_PTM(Gy90),
               'mY180': invert_unitary_component_PTM(Gy180),
               }


def generate_clifford_operators(gateset,
                                clifford_decomposition=gate_decomposition):
    clifford_operators = []
    for i, cl in enumerate(gate_decomposition):
        gate = np.eye(4)
        for gate_Id in cl:
            gate = gateset[gate_Id]*gate

        clifford_operators.append(gate)
    return clifford_operators


def calc_p_depolarizing(gate, target_gate, input_states=polar_states):
    p = []
    for i, state in enumerate(input_states):
        target_state = target_gate*state

        p.append(target_state.T*gate*state)
    # geometric mean
    return np.prod(np.array(p))**(1/len(p))


def calculate_RB_fid(gateset, target_gateset,
                     clifford_decomposition=gate_decomposition):
    clifford_ops = generate_clifford_operators(gateset,
                                               clifford_decomposition)
    target_cl_ops = generate_clifford_operators(gateset,
                                                clifford_decomposition)
    probs = []
    for i in range(len(clifford_ops)):
        probs.append(calc_p_depolarizing(gate=clifford_ops[i],
                                         target_gate=target_cl_ops[i]))
    # geometric mean
    return np.prod(np.array(probs))**(1/len(probs))


class Test_density_vecs(unittest.TestCase):
        def test_overlap_with_self(self):
            for vec in polar_states:
                self.assertAlmostEqual((vec.T * vec), 1)

        def test_overlap_with_orthogonal(self):
            for s0, s1 in zip(polar_states[:-1:2], polar_states[1::2]):
                self.assertAlmostEqual((s0.T * s1), 0)

        def test_overlap_with_different_bases(self):
            for i, s0 in enumerate(polar_states):
                if i % 2 == 0:
                    for j in range(len(polar_states)):
                        if j != i and j != (i+1):
                            self.assertAlmostEqual(
                                (s0.T * polar_states[j]), 0.5)
                else:
                    for j in range(len(polar_states)):
                        if j != i and j != (i-1):
                            self.assertAlmostEqual(
                                (s0.T * polar_states[j]), 0.5)


class Test_basic_operations(unittest.TestCase):
    def test_valid(self):
        g = Ideal_gates
        np.testing.assert_almost_equal(g['X90'], g['X90'])
        np.testing.assert_almost_equal(g['X180'], g['X180'])
        np.testing.assert_almost_equal(g['Y90'], g['Y90'])
        np.testing.assert_almost_equal(g['Y180'], g['Y180'])
        np.testing.assert_almost_equal(g['I'], g['I'])

        # Test some basic operations
    def test_identity(self):
        g = Ideal_gates
        for vec in polar_states:
            np.testing.assert_almost_equal(vec, g['I']*vec)

    def test_basic_rotations(self):
        g = Ideal_gates
        np.testing.assert_almost_equal(X0, g['X180']*X0)
        np.testing.assert_almost_equal(X1, g['Y180']*X0)
        np.testing.assert_almost_equal(X0, g['X90']*X0)
        np.testing.assert_almost_equal(Z1, g['Y90']*X0)
        np.testing.assert_almost_equal(Z0, g['Y90']*X1)

        np.testing.assert_almost_equal(Y1.T*(g['X180']*Y0), 1)
        np.testing.assert_almost_equal(Y0, g['Y180']*Y0)
        np.testing.assert_almost_equal(Z0, g['X90']*Y0)
        np.testing.assert_almost_equal(Z1, g['X90']*Y1)
        np.testing.assert_almost_equal(Y0, g['Y90']*Y0)

        np.testing.assert_almost_equal(Z1, g['X180']*Z0)
        np.testing.assert_almost_equal(Z1, g['Y180']*Z0)
        np.testing.assert_almost_equal(Y1, g['X90']*Z0)
        np.testing.assert_almost_equal(X0, g['Y90']*Z0)

    def test_inverses(self):
        g = Ideal_gates
        np.testing.assert_almost_equal(g['X90']*g['mX90'], g['I'])
        np.testing.assert_almost_equal(g['Y90']*g['mY90'], g['I'])
        np.testing.assert_almost_equal(g['X180']*g['mX180'], g['I'])
        np.testing.assert_almost_equal(g['Y180']*g['mY180'], g['I'])


class Test_clifford_composition(unittest.TestCase):
    def test_case(self):

        cl_ops = generate_clifford_operators(gateset=Ideal_gates)
        self.assertTrue(len(cl_ops), 24)

        for i in range(24):
            rb_seq = rb.randomized_benchmarking_sequence(100, desired_net_cl=i)
            net_cliff = np.eye(4)
            for rb_idx in rb_seq:
                net_cliff = cl_ops[rb_idx]*net_cliff
            np.testing.assert_almost_equal(net_cliff, cl_ops[i])


class Test_clifford_fidelity(unittest.TestCase):
    def test_depolarizing_probability(self):
        for name, gate in Ideal_gates.items():
            p = calc_p_depolarizing(gate, gate)
            np.testing.assert_almost_equal(p, 1)

        cl_ops = generate_clifford_operators(gateset=Ideal_gates)
        for i in range(len(cl_ops)):
            for j in range(len(cl_ops)):
                p = calc_p_depolarizing(cl_ops[i], cl_ops[j])
                if i == j:
                    np.testing.assert_almost_equal(p, 1)
                else:
                    np.testing.assert_array_less(p, 1)

    def test_clifford_fidelity(self):
        F_rb = calculate_RB_fid(Ideal_gates, Ideal_gates)
        F_rb = 1
        np.testing.assert_almost_equal(F_rb, 1)

if __name__ == '__main__':
    test_classes_to_run = [Test_density_vecs,
                           Test_basic_operations,
                           Test_clifford_composition,
                           Test_clifford_fidelity
                           ]

    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    combined_test_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner(verbosity=1).run(combined_test_suite)
