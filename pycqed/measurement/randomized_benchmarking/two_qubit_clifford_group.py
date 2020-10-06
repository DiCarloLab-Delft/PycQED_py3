import numpy as np
from zlib import crc32
from os.path import join, dirname, abspath
from pycqed.measurement.randomized_benchmarking.clifford_group import clifford_group_single_qubit as C1, CZ, S1
from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(epstein_efficient_decomposition)

hash_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')

"""
This file contains Clifford decompositions for the two qubit Clifford group.

The Clifford decomposition closely follows two papers:
Corcoles et al. Process verification ... Phys. Rev. A. 2013
    http://journals.aps.org/pra/pdf/10.1103/PhysRevA.87.030301
for the different classes of two-qubit Cliffords.

and
Barends et al. Superconducting quantum circuits at the ... Nature 2014
    https://www.nature.com/articles/nature13171?lang=en
for writing the cliffords in terms of CZ gates.


###########################################################################
2-qubit clifford decompositions

The two qubit clifford group (C2) consists of 11520 two-qubit cliffords
These gates can be subdivided into four classes.
    1. The Single-qubit like class  | 576 elements  (24^2)
    2. The CNOT-like class          | 5184 elements (24^2 * 3^2)
    3. The iSWAP-like class         | 5184 elements (24^2 * 3^2)
    4. The SWAP-like class          | 576  elements (24^2)
    --------------------------------|------------- +
    Two-qubit Clifford group C2     | 11520 elements


1. The Single-qubit like class
    -- C1 --
    -- C1 --

2. The CNOT-like class
    --C1--•--S1--      --C1--•--S1------
          |        ->        |
    --C1--⊕--S1--      --C1--•--S1^Y90--

3. The iSWAP-like class
    --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
          |       ->        |        |
    --C1--*--S1--     --C1--•--mY90--•--S1^X90--

4. The SWAP-like class
    --C1--x--     --C1--•-mY90--•--Y90--•-------
          |   ->        |       |       |
    --C1--x--     --C1--•--Y90--•-mY90--•--Y90--

C1: element of the single qubit Clifford group
    N.B. we use the decomposition defined in Epstein et al. here

S1: element of the S1 group, a subgroup of the single qubit Clifford group

S1[0] = I
S1[1] = rY90, rX90
S1[2] = rXm90, rYm90

Important clifford indices:

        I    : Cl 0
        X90  : Cl 16
        Y90  : Cl 21
        X180 : Cl 3
        Y180 : Cl 6
        Z180 : Cl 9
        CZ   : 4368

"""
# set as a module wide variable instead of argument to function for speed
# reasons
gate_decomposition = epstein_efficient_decomposition

# used to transform the S1 subgroup
X90 = C1[16]
Y90 = C1[21]
mY90 = C1[15]

# A dict containing clifford IDs with common names.
common_cliffords = {'I':  0, 'X':  3, 'Y':  6, 'Z':  9,
                    'II':  0, 'IX':  3, 'IY':  6, 'IZ':  9,

                    'XI': 24*3 + 0, 'XX': 24*3 + 3,
                    'XY': 24*3 + 6, 'XZ': 24*3 + 9,

                    'YI':  24*6 + 0, 'YX':  24*6 + 3,
                    'YY':  24*6 + 6, 'YZ':  24*6 + 9,

                    'ZI':  24*9 + 0, 'ZX':  24*9 + 3,
                    'ZY':  24*9 + 6, 'ZZ':  24*9 + 9,

                    'X90':  16,
                    'Y90':  21,
                    'X180':  3,
                    'Y180':  6,
                    'Z180':  9,
                    'CZ': 104368,
                    }


class Clifford(object):

    def __mul__(self, other):
        """
        Product of two clifford gates.
        returns a new Clifford object that performs the net operation
        that is the product of both operations.
        """
        net_op = np.dot(self.pauli_transfer_matrix,
                        other.pauli_transfer_matrix)
        idx = get_clifford_id(net_op)
        return self.__class__(idx)

    def __repr__(self):
        return '{}(idx={})'.format(self.__class__.__name__, self.idx)

    def __str__(self):
        return '{} idx {}\n Gates: {}\n'.format(self.__class__.__name__, self.idx,
                                                self.gate_decomposition.__str__(),
                                                )

    def get_inverse(self):
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = get_clifford_id(inverse_ptm)
        return self.__class__(idx)


class SingleQubitClifford(Clifford):

    def __init__(self, idx: int):
        assert(idx < 24)
        self.idx = idx
        self.pauli_transfer_matrix = C1[idx]

    @property
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the single qubit Clifford group
        according to the decomposition by Epstein et al.
        """
        if not hasattr(self, '_gate_decomposition'):
            self._gate_decomposition = [(g, 'q0') for g in
                                        gate_decomposition[self.idx]]
        return self._gate_decomposition


class TwoQubitClifford(Clifford):

    def __init__(self, idx: int):
        assert(idx < 11520)
        self.idx = idx

        if idx < 576:
            self.pauli_transfer_matrix = single_qubit_like_PTM(idx)
        elif idx < 576 + 5184:
            self.pauli_transfer_matrix = CNOT_like_PTM(idx-576)
        elif idx < 576 + 2*5184:
            self.pauli_transfer_matrix = iSWAP_like_PTM(idx-(576+5184))
        elif idx < 11520:
            self.pauli_transfer_matrix = SWAP_like_PTM(idx-(576+2*5184))

    @property
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the two qubit Clifford group.

        Single qubit Cliffords are decompesed according to Epstein et al.

        Using the method to get this avoids expensive function calls
        whenever the Clifford is instantiated
        """
        if not hasattr(self, '_gate_decomposition'):
            if self.idx < 576:
                self._gate_decomposition = single_qubit_like_gates(self.idx)
            elif self.idx < 576 + 5184:
                self._gate_decomposition = CNOT_like_gates(self.idx-576)
            elif self.idx < 576 + 2*5184:
                self._gate_decomposition = iSWAP_like_gates(
                    self.idx-(576+5184))
            elif self.idx < 11520:
                self._gate_decomposition = SWAP_like_gates(
                    self.idx-(576+2*5184))

        return self._gate_decomposition


def single_qubit_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the single qubit like class
        (q0)  -- C1 --
        (q1)  -- C1 --
    """
    assert(idx < 24**2)
    idx_q0 = idx % 24
    idx_q1 = idx//24
    pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
    return pauli_transfer_matrix


def single_qubit_like_gates(idx):
    """
    Returns the gates for Cliffords of the single qubit like class
        (q0)  -- C1 --
        (q1)  -- C1 --
    """
    assert(idx < 24**2)
    idx_q0 = idx % 24
    idx_q1 = idx//24

    g_q0 = [(g, 'q0') for g in gate_decomposition[idx_q0]]
    g_q1 = [(g, 'q1') for g in gate_decomposition[idx_q1]]
    gates = g_q0 + g_q1
    return gates


def CNOT_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the cnot like class
        (q0)  --C1--•--S1--      --C1--•--S1------
                    |        ->        |
        (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
    """
    assert(idx < 5184)
    idx_0 = idx % 24
    idx_1 = (idx // 24) % 24
    idx_2 = (idx // 576) % 3
    idx_3 = (idx // 1728)

    C1_q0 = np.kron(np.eye(4), C1[idx_0])
    C1_q1 = np.kron(C1[idx_1], np.eye(4))
    CZ
    S1_q0 = np.kron(np.eye(4), S1[idx_2])
    S1y_q1 = np.kron(np.dot(C1[idx_3], Y90), np.eye(4))
    return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1, CZ, S1_q0, S1y_q1])))


def CNOT_like_gates(idx):
    """
    Returns the gates for Cliffords of the cnot like class
        (q0)  --C1--•--S1--      --C1--•--S1------
                    |        ->        |
        (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
    """
    assert(idx < 5184)
    idx_0 = idx % 24
    idx_1 = (idx // 24) % 24
    idx_2 = (idx // 576) % 3
    idx_3 = (idx // 1728)

    C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_0]]
    C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_1]]
    CZ = [('CZ', ['q0', 'q1'])]

    idx_2s = get_clifford_id(S1[idx_2])
    S1_q0 = [(g, 'q0') for g in gate_decomposition[idx_2s]]
    idx_3s = get_clifford_id(np.dot(C1[idx_3], Y90))
    S1_yq1 = [(g, 'q1') for g in gate_decomposition[idx_3s]]

    gates = C1_q0 + C1_q1 + CZ + S1_q0 + S1_yq1
    return gates


def iSWAP_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the iSWAP like class
        (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                    |       ->        |        |
        (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
    """
    assert(idx < 5184)
    idx_0 = idx % 24
    idx_1 = (idx // 24) % 24
    idx_2 = (idx // 576) % 3
    idx_3 = (idx // 1728)

    C1_q0 = np.kron(np.eye(4), C1[idx_0])
    C1_q1 = np.kron(C1[idx_1], np.eye(4))
    CZ
    sq_swap_gates = np.kron(mY90, Y90)
    CZ
    S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
    S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

    return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1,
                                              CZ, sq_swap_gates, CZ,
                                              S1_q0, S1y_q1])))


def iSWAP_like_gates(idx):
    """
    Returns the gates for Cliffords of the iSWAP like class
        (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                    |       ->        |        |
        (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
    """
    assert(idx < 5184)
    idx_0 = idx % 24
    idx_1 = (idx // 24) % 24
    idx_2 = (idx // 576) % 3
    idx_3 = (idx // 1728)

    C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_0]]
    C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_1]]
    CZ = [('CZ', ['q0', 'q1'])]

    sqs_idx_q0 = get_clifford_id(Y90)
    sqs_idx_q1 = get_clifford_id(mY90)
    sq_swap_gates_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
    sq_swap_gates_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

    # S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
    # S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

    idx_2s = get_clifford_id(np.dot(S1[idx_2], Y90))
    S1_q0 = [(g, 'q0') for g in gate_decomposition[idx_2s]]
    idx_3s = get_clifford_id(np.dot(C1[idx_3], X90))
    S1y_q1 = [(g, 'q1') for g in gate_decomposition[idx_3s]]

    gates = (C1_q0 + C1_q1 + CZ +
             sq_swap_gates_q0 + sq_swap_gates_q1 + CZ +
             S1_q0 + S1y_q1)
    return gates


def SWAP_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the SWAP like class

    (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                |   ->        |       |       |
    (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
    """
    assert(idx < 24**2)
    idx_q0 = idx % 24
    idx_q1 = idx//24
    sq_like_cliff = np.kron(C1[idx_q1], C1[idx_q0])
    sq_swap_gates_0 = np.kron(Y90, mY90)
    sq_swap_gates_1 = np.kron(mY90, Y90)
    sq_swap_gates_2 = np.kron(Y90, np.eye(4))

    return np.linalg.multi_dot(list(reversed([sq_like_cliff, CZ,
                                              sq_swap_gates_0, CZ,
                                              sq_swap_gates_1, CZ,
                                              sq_swap_gates_2])))


def SWAP_like_gates(idx):
    """
    Returns the gates for Cliffords of the SWAP like class

    (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                |   ->        |       |       |
    (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
    """
    assert(idx < 24**2)
    idx_q0 = idx % 24
    idx_q1 = idx//24
    C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_q0]]
    C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_q1]]
    CZ = [('CZ', ['q0', 'q1'])]

    sq_swap_gates_0 = np.kron(Y90, mY90)

    sqs_idx_q0 = get_clifford_id(mY90)
    sqs_idx_q1 = get_clifford_id(Y90)
    sq_swap_gates_0_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
    sq_swap_gates_0_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

    sqs_idx_q0 = get_clifford_id(Y90)
    sqs_idx_q1 = get_clifford_id(mY90)
    sq_swap_gates_1_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
    sq_swap_gates_1_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

    sqs_idx_q1 = get_clifford_id(Y90)
    sq_swap_gates_2_q0 = [(g, 'q0') for g in gate_decomposition[0]]
    sq_swap_gates_2_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

    gates = (C1_q0 + C1_q1 + CZ +
             sq_swap_gates_0_q0 + sq_swap_gates_0_q1 + CZ +
             sq_swap_gates_1_q0 + sq_swap_gates_1_q1 + CZ +
             sq_swap_gates_2_q0 + sq_swap_gates_2_q1)
    return gates


##############################################################################
# It is important that this check is after the Clifford objects as otherwise
# it is impossible to generate the hash tables
##############################################################################
try:
    open(join(hash_dir, 'single_qubit_hash_lut.txt'), 'r')
except FileNotFoundError:
    print("Clifford group hash tables not detected.")
    from pycqed.measurement.randomized_benchmarking.generate_clifford_hash_tables import generate_hash_tables
    generate_hash_tables()


def get_single_qubit_clifford_hash_table():
    """
    Get's the single qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    with open(join(hash_dir, 'single_qubit_hash_lut.txt'),
              'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table


def get_two_qubit_clifford_hash_table():
    """
    Get's the two qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    with open(join(hash_dir, 'two_qubit_hash_lut.txt'),
              'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table


def get_clifford_id(pauli_transfer_matrix):
    """
    returns the unique Id of a Clifford.
    """
    unique_hash = crc32(pauli_transfer_matrix.astype(int))
    if np.array_equal(np.shape(pauli_transfer_matrix), (4, 4)):
        hash_table = get_single_qubit_clifford_hash_table()
    elif np.array_equal(np.shape(pauli_transfer_matrix), (16, 16)):
        hash_table = get_two_qubit_clifford_hash_table()
    else:
        raise NotImplementedError()
    idx = hash_table.index(unique_hash)
    return idx
