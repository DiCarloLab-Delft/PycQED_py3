import numpy as np
from unittest import expectedFailure
from unittest import TestCase
from zlib import crc32

from pycqed.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable, clifford_group_single_qubit,
    X,Y,Z, H, S, S2, CZ)

import pycqed.measurement.randomized_benchmarking.randomized_benchmarking \
    as rb

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)

from pycqed.measurement.randomized_benchmarking import \
    two_qubit_clifford_group as tqc
from pycqed.measurement.randomized_benchmarking.generate_clifford_hash_tables import construct_clifford_lookuptable

class TestLookuptable(TestCase):
    def test_unique_mapping(self):
        for row in clifford_lookuptable:
            self.assertFalse(len(row) > len(set(row)))

    def test_sum_of_rows(self):
        expected_sum = np.sum(range(len(clifford_group_single_qubit)))
        for row in clifford_lookuptable:
            self.assertEqual(np.sum(row), expected_sum)

    def test_element_index_in_group(self):
        for row in clifford_lookuptable:
            for el in row:
                self.assertTrue(el < len(clifford_group_single_qubit))


class TestCalculateNetClifford(TestCase):
    def test_identity_does_nothing(self):
        id_seq = np.zeros(5)
        net_cl = rb.calculate_net_clifford(id_seq)
        self.assertEqual(net_cl, 0)

        for i in range(len(clifford_group_single_qubit)):
            id_seq[3] = i
            net_cl = rb.calculate_net_clifford(id_seq)
            self.assertEqual(net_cl, i)

    def test_pauli_squared_is_ID(self):
        for cl in [0, 3, 6, 9, 12]:  # 12 is Hadamard
            net_cl = rb.calculate_net_clifford([cl, cl])
            self.assertEqual(net_cl, 0)


class TestRecoveryClifford(TestCase):
    def testInversionRandomSequence(self):
        random_cliffords = np.random.randint(0, len(clifford_group_single_qubit), 100)
        net_cl = rb.calculate_net_clifford(random_cliffords)

        for des_cl in range(len(clifford_group_single_qubit)):
            rec_cliff = rb.calculate_recovery_clifford(net_cl, des_cl)
            comb_seq = np.append(random_cliffords, rec_cliff)

            comb_net_cl_simple = rb.calculate_net_clifford([net_cl, rec_cliff])
            comb_net_cl = rb.calculate_net_clifford(comb_seq)

            self.assertEqual(comb_net_cl, des_cl)
            self.assertEqual(comb_net_cl_simple, des_cl)


class TestRB_sequence(TestCase):
    def test_net_cliff(self):
        for i in range(len(clifford_group_single_qubit)):
            rb_seq = rb.randomized_benchmarking_sequence(500, desired_net_cl=i)
            net_cliff = rb.calculate_net_clifford(rb_seq)
            self.assertEqual(net_cliff, i)

    def test_seed_reproduces(self):
        rb_seq_a = rb.randomized_benchmarking_sequence(500, seed=5)
        rb_seq_b = rb.randomized_benchmarking_sequence(500, seed=None)
        rb_seq_c = rb.randomized_benchmarking_sequence(500, seed=5)
        rb_seq_d = rb.randomized_benchmarking_sequence(500, seed=None)
        self.assertTrue((rb_seq_a == rb_seq_c).all())
        self.assertTrue((rb_seq_a != rb_seq_b).any)
        self.assertTrue((rb_seq_c != rb_seq_b).any)
        self.assertTrue((rb_seq_b != rb_seq_d).any)


class TestGateDecomposition(TestCase):
    def test_unique_elements(self):
        for gate in gate_decomposition:
            self.assertEqual(gate_decomposition.count(gate), 1)

    def test_average_number_of_gates(self):
        from itertools import chain
        avg_nr_gates = len(list(chain(*gate_decomposition)))/24
        self.assertEqual(avg_nr_gates, 1.875)


######################################################################
# Two qubit clifford group below
######################################################################

class TestHashedLookuptables(TestCase):
    def test_single_qubit_hashtable_constructed(self):
        hash_table = construct_clifford_lookuptable(tqc.SingleQubitClifford,
                                                    np.arange(24))
        for i in range(24):
            Cl = tqc.SingleQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_single_qubit_hashtable_file(self):
        hash_table = tqc.get_single_qubit_clifford_hash_table()

        for i in range(24):
            Cl = tqc.SingleQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_two_qubit_hashtable_constructed(self):
        hash_table = construct_clifford_lookuptable(tqc.TwoQubitClifford,
                                                    np.arange(11520))
        for i in range(11520):
            Cl = tqc.TwoQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_two_qubit_hashtable_file(self):
        hash_table = tqc.get_two_qubit_clifford_hash_table()
        for i in range(11520):
            Cl = tqc.TwoQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_get_clifford_id(self):
        for i in range(24):
            Cl = tqc.SingleQubitClifford(i)
            idx = tqc.get_clifford_id(Cl.pauli_transfer_matrix)
            self.assertEqual(idx, Cl.idx)

        for i in range(11520):
            Cl = tqc.TwoQubitClifford(i)
            idx = tqc.get_clifford_id(Cl.pauli_transfer_matrix)
            self.assertEqual(idx, Cl.idx)

class Test_CliffordGroupProperties(TestCase):

    def test_single_qubit_group(self):
        hash_table = tqc.get_single_qubit_clifford_hash_table()
        self.assertEqual(len(hash_table), 24)
        self.assertEqual(len(np.unique(hash_table)), 24)

    # Testing the subgroups of the Clifford group
    def test_single_qubit_like_PTM(self):
        hash_table = []
        for idx in np.arange(24**2):
            clifford = tqc.single_qubit_like_PTM(idx)
            hash_val = crc32(clifford.round().astype(int))
            hash_table.append(hash_val)
        self.assertEqual(len(hash_table), 24**2)
        self.assertEqual(len(np.unique(hash_table)), 24**2)
        with self.assertRaises(AssertionError):
            clifford = tqc.single_qubit_like_PTM(24**2+1)

    def test_CNOT_like_PTM(self):
        hash_table = []
        for idx in np.arange(5184):
            clifford = tqc.CNOT_like_PTM(idx)
            hash_val = crc32(clifford.round().astype(int))
            hash_table.append(hash_val)
        self.assertEqual(len(hash_table), 5184)
        self.assertEqual(len(np.unique(hash_table)), 5184)
        with self.assertRaises(AssertionError):
            clifford = tqc.CNOT_like_PTM(5184**2+1)

    def test_iSWAP_like_PTM(self):
        hash_table = []
        for idx in np.arange(5184):
            clifford = tqc.iSWAP_like_PTM(idx)
            hash_val = crc32(clifford.round().astype(int))
            hash_table.append(hash_val)
        self.assertEqual(len(hash_table), 5184)
        self.assertEqual(len(np.unique(hash_table)), 5184)
        with self.assertRaises(AssertionError):
            clifford = tqc.iSWAP_like_PTM(5184+1)

    def test_SWAP_like_PTM(self):
        hash_table = []
        for idx in np.arange(24**2):
            clifford = tqc.SWAP_like_PTM(idx)
            hash_val = crc32(clifford.round().astype(int))
            hash_table.append(hash_val)
        self.assertEqual(len(hash_table), 24**2)
        self.assertEqual(len(np.unique(hash_table)), 24**2)
        with self.assertRaises(AssertionError):
            clifford = tqc.SWAP_like_PTM(24**2+1)

    def test_two_qubit_group(self):
        hash_table = tqc.get_two_qubit_clifford_hash_table()
        self.assertEqual(len(hash_table), 11520)
        self.assertEqual(len(np.unique(hash_table)), 11520)

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


class TestCliffordCalculus(TestCase):

    def test_products(self):
        Cl_3 = tqc.SingleQubitClifford(3)
        Cl_3*Cl_3
        self.assertEqual(Cl_3.idx, 3) # Pauli X
        self.assertEqual((Cl_3*Cl_3).idx, 0) # The identity

        Cl_3 = tqc.TwoQubitClifford(3)
        self.assertEqual(Cl_3.idx, 3) # Pauli X on q0
        self.assertEqual((Cl_3*Cl_3).idx, 0) # The identity
        product_hash = crc32((Cl_3*Cl_3).pauli_transfer_matrix.round().astype(int))
        target_hash = crc32(tqc.TwoQubitClifford(0).pauli_transfer_matrix.round().astype(int))
        self.assertEqual(product_hash, target_hash)

    def test_inverse_single_qubit_clifford(self):
        for i in range(24):
            print(i)
            Cl = tqc.SingleQubitClifford(i)
            Cl_inv = Cl.get_inverse()

            self.assertEqual((Cl_inv*Cl).idx, 0)





    # def test_inverse_two_qubit_clifford(self):
    #     for i in range(11520):
    #         Cl = tqc.TwoQubitClifford(i)
    #         Cl_inv = Cl.get_inverse()

    #         self.assertEqual((Cl_inv*Cl).idx, 0)




