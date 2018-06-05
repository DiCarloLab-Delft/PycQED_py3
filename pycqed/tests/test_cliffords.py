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
    import(gate_decomposition, epstein_fixed_length_decomposition)

from pycqed.measurement.randomized_benchmarking import \
    two_qubit_clifford_group as tqc
from pycqed.measurement.randomized_benchmarking.generate_clifford_hash_tables import construct_clifford_lookuptable


np.random.seed(0)
test_indices_2Q = np.random.randint(0, high=11520, size=50)
# To test all elements of the 2 qubit clifford group use:
# test_indices_2Q = np.arange(11520)

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

    def test_average_number_of_gates_epst_efficient(self):
        from itertools import chain
        avg_nr_gates = len(list(chain(*gate_decomposition)))/24
        self.assertEqual(avg_nr_gates, 1.875)

    def test_average_number_of_gates_epst_fixed_length(self):
        from itertools import chain
        avg_nr_gates = len(list(chain(*epstein_fixed_length_decomposition)))/24
        self.assertEqual(avg_nr_gates, 3)


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
        for i in test_indices_2Q:
            Cl = tqc.TwoQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_two_qubit_hashtable_file(self):
        hash_table = tqc.get_two_qubit_clifford_hash_table()
        for i in test_indices_2Q:
            Cl = tqc.TwoQubitClifford(i)
            target_hash = crc32(Cl.pauli_transfer_matrix.round().astype(int))
            table_idx = hash_table.index(target_hash)
            self.assertEqual(table_idx, i)

    def test_get_clifford_id(self):
        for i in range(24):
            Cl = tqc.SingleQubitClifford(i)
            idx = tqc.get_clifford_id(Cl.pauli_transfer_matrix)
            self.assertEqual(idx, Cl.idx)

        for i in test_indices_2Q:
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

    def test_product_order(self):
        """
        Tests that the order of multiplying matrices is the same as what is
        defined in numpy.dot
        """
        Cl_528 = tqc.TwoQubitClifford(528)
        Cl_9230 = tqc.TwoQubitClifford(9230)

        Cliff_prod = Cl_528*Cl_9230
        dot_prod = np.dot(Cl_528.pauli_transfer_matrix,
                          Cl_9230.pauli_transfer_matrix)
        np.testing.assert_array_equal(Cliff_prod.pauli_transfer_matrix,
                                      dot_prod)


    def test_inverse_single_qubit_clifford(self):
        for i in range(24):
            Cl = tqc.SingleQubitClifford(i)
            Cl_inv = Cl.get_inverse()
            self.assertEqual((Cl_inv*Cl).idx, 0)

    def test_inverse_two_qubit_clifford(self):
        for i in test_indices_2Q:
            Cl = tqc.TwoQubitClifford(i)
            Cl_inv = Cl.get_inverse()
            self.assertEqual((Cl_inv*Cl).idx, 0)

class TestCliffordGateDecomposition(TestCase):
    def test_single_qubit_gate_decomposition(self):
        for i in range(24):
            CL = tqc.SingleQubitClifford(i)
            gate_dec = CL.gate_decomposition
            self.assertIsInstance(gate_dec, list)
            for g in gate_dec:
                self.assertIsInstance(g[0], str)
                self.assertEqual(g[1], 'q0')

    def test_two_qubit_gate_decomposition(self):
        for idx in (test_indices_2Q):
            CL = tqc.TwoQubitClifford(idx)
            gate_dec = CL.gate_decomposition
            print(idx, gate_dec)
            self.assertIsInstance(gate_dec, list)
            for g in gate_dec:
                self.assertIsInstance(g[0], str)
                if g[0] == 'CZ':
                    self.assertEqual(g[1], ['q0', 'q1'])
                else:
                    self.assertIn(g[1], ['q0', 'q1'])

    def test_gate_decomposition_unique_single_qubit(self):
        hash_table = []
        for i in range(24):
            CL = tqc.SingleQubitClifford(i)
            gate_dec = CL.gate_decomposition
            hash_table.append(crc32(bytes(str(gate_dec), 'utf-8')))
        self.assertEqual(len(hash_table),24)
        self.assertEqual(len(np.unique(hash_table)),24)

    def test_gate_decomposition_unique_two_qubit(self):
        hash_table = []
        for i in range(11520):
            CL = tqc.TwoQubitClifford(i)
            gate_dec = CL.gate_decomposition
            hash_table.append(crc32(bytes(str(gate_dec), 'utf-8')))
        self.assertEqual(len(hash_table), 11520)
        self.assertEqual(len(np.unique(hash_table)), 11520)




class TestCliffordClassRBSeqs(TestCase):
    """

    """
    def test_single_qubit_randomized_benchmarking_sequence(self):
        """
        """
        seeds = [0, 100, 200, 300, 400]
        net_cliffs = np.arange(len(seeds))
        for seed, net_cl in zip(seeds, net_cliffs):
            cliffords_single_qubit_class = rb.randomized_benchmarking_sequence(
                n_cl=20, desired_net_cl=0,  number_of_qubits=1, seed=0)
            cliffords = rb.randomized_benchmarking_sequence_old(
                n_cl=20, desired_net_cl=0, seed=0)
            np.testing.assert_array_equal(cliffords_single_qubit_class, cliffords)

    def test_interleaved_randomized_benchmarking_sequence_1Q(self):
        seeds = [0, 100, 200, 300, 400]
        net_cliffs = np.arange(len(seeds))
        for seed, net_cl in zip(seeds, net_cliffs):
            intl_cliffords = rb.randomized_benchmarking_sequence(
                n_cl=20, desired_net_cl=0, number_of_qubits=1, seed=0,
                interleaving_cl=0)
            cliffords = rb.randomized_benchmarking_sequence(
                n_cl=20, desired_net_cl=0, seed=0)

            new_cliff = np.empty(cliffords.size*2-1, dtype=int)
            new_cliff[0::2] = cliffords
            new_cliff[1::2] = 0
            np.testing.assert_array_equal(intl_cliffords,
                                          new_cliff)


    def test_interleaved_randomized_benchmarking_sequence_2Q(self):
        seeds = [0, 100, 200, 300, 400]
        net_cliffs = np.arange(len(seeds))
        for seed, net_cl in zip(seeds, net_cliffs):
            intl_cliffords = rb.randomized_benchmarking_sequence(
                n_cl=20, desired_net_cl=0, number_of_qubits=2, seed=0,
                interleaving_cl=0)
            cliffords = rb.randomized_benchmarking_sequence(
                n_cl=20, number_of_qubits=2, desired_net_cl=0, seed=0)

            new_cliff = np.empty(cliffords.size*2-1, dtype=int)
            new_cliff[0::2] = cliffords
            new_cliff[1::2] = 0
            np.testing.assert_array_equal(intl_cliffords,
                                          new_cliff)


    def test_two_qubit_randomized_benchmarking_sequence(self):
        """
        """
        seeds = [0, 100, 200, 300, 400]
        net_cliffs = np.arange(len(seeds))
        for seed, net_cl in zip(seeds, net_cliffs):
            rb.randomized_benchmarking_sequence(
                n_cl=20, desired_net_cl=0, number_of_qubits=2, seed=0)


            # rb.two_qubit_randomized_benchmarking_sequence(
            #     n_cl=20, desired_net_cl=0, seed=0)
            # no test for correctness here. Corectness depend on the fact
            # that it implements code very similar to the Single qubit version
            # and has components that are all tested.


