print('hello')
import numpy as np
import unittest

# Basic states
# desnity matrices in pauli basis
X0 = 1/np.sqrt(2) * np.array([1,  1, 0, 0])
X1 = 1/np.sqrt(2) * np.array([1,  -1, 0, 0])
Y0 = 1/np.sqrt(2) * np.array([1, 0, 1, 0])
Y1 = 1/np.sqrt(2) * np.array([1, 0, -1, 0])
Z0 = 1/np.sqrt(2) * np.array([1, 0, 0, 1])
Z1 = 1/np.sqrt(2) * np.array([1, 0, 0, -1])

polar_states = [X0, X1, Y0, Y1, Z0, Z1]

# Superoperators for our gate set
I = np.ones(4)

Gi = np.array([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0],
               [0, 0, 1.0, 0],
               [0, 0, 0, 1.]])

Gx90 = np.array([[1.0, 0, 0, 0],
                 [0, 1.0, 0, 0],
                 [0, 0, 0, -1, 0],
                 [0, 0, 1.0, 0]])

Gx180 = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, -1.0]])

Gy90 = np.array([[1.0, 0, 0, 0],
                 [0, 0, 0, 1.0],
                 [0, 0, 1.0, 0],
                 [0, 1, 0, 0, 0]])


Gy180 = np.array([[1.0, 0, 0, 0],
                  [0, -1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, -1.0]])


# Test inner product between states


Ideal_gates = {'I': np.ones(4),
               'x90':None}

# Define all the tests


class Test_density_vecs(unittest.TestCase):
        def test_overlap_with_self(self):
            for vec in polar_states:
                self.assertAlmostEqual(np.dot(vec.T, vec), 1)

        def test_overlap_with_orthogonal(self):
            for s0, s1 in zip(polar_states[:-1:2], polar_states[1::2]):
                self.assertAlmostEqual(np.dot(s0.T, s1), 0)

        def test_overlap_with_different_bases(self):
            for i, s0 in enumerate(polar_states):
                if i % 2 == 0:
                    for j in range(len(polar_states)):
                        if j != i and j != (i+1):
                            self.assertAlmostEqual(
                                np.dot(s0.T, polar_states[j]), 0.5)
                else:
                    for j in range(len(polar_states)):
                        if j != i and j != (i-1):
                            self.assertAlmostEqual(
                                np.dot(s0.T, polar_states[j]), 0.5)

test_classes_to_run = [Test_density_vecs]

suites_list = []
for test_class in test_classes_to_run:
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    suites_list.append(suite)

combined_test_suite = unittest.TestSuite(suites_list)
runner = unittest.TextTestRunner(verbosity=2).run(combined_test_suite)
