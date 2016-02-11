import numpy as np
from unittest import TestCase

from modules.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable, Clifford_group)

import modules.measurement.randomized_benchmarking.randomized_benchmarking as rb


class TestLookuptable(TestCase):
    def test_unique_mapping(self):
        for row in clifford_lookuptable:
            self.assertFalse(len(row) > len(set(row)))

    def test_sum_of_rows(self):
        expected_sum = np.sum(range(len(Clifford_group)))
        for row in clifford_lookuptable:
            self.assertEqual(np.sum(row), expected_sum)

    def test_element_index_in_group(self):
        for row in clifford_lookuptable:
            for el in row:
                self.assertTrue(el < len(Clifford_group))


class TestCalculateNetClifford(TestCase):
    def test_dummy(self):
        pass

    def test_identity_does_nothing(self):
        id_seq = np.zeros(5)
        net_cl = rb.calculate_net_clifford(id_seq)
        self.assertEqual(net_cl, 0)

        for i in range(len(Clifford_group)):
            id_seq[3] = i
            net_cl = rb.calculate_net_clifford(id_seq)
            self.assertEqual(net_cl, i)

    def test_pauli_squared_is_ID(self):
        for cl in [0, 3, 6, 9, 12]:  # 12 is Hadamard
            net_cl = rb.calculate_net_clifford([cl, cl])
            self.assertEqual(net_cl, 0)


