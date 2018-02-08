import numpy as np
from zlib import crc32
from os.path import join, dirname, abspath
from pycqed.measurement.randomized_benchmarking.clifford_group import clifford_group_single_qubit as C1, CZ, S1
from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)


"""
This file contains Clifford decompositions for the two qubit Clifford group.

The Clifford decomposition follows closely two papers:
Corcoles et al. Process verification .... Phys. Rev. A. 2013
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
    N.B. we do not use the decomposition defined in Epstein et al. here
    but we follow the decomposition according to Barends et al.
S1: element of the S1 group, a subgroup of the single qubit Clifford group

S1[0] = I
S1[1] = rY90, rX90
S1[2] = rXm90, rYm90


"""

# used to transform the S1 subgroup
X90 = C1[16]
Y90 = C1[21]
mY90 = C1[15]

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
        return '{} idx {}\n PTM: {}\n'.format(self.__class__.__name__, self.idx,
                                            self.pauli_transfer_matrix.__str__,
                                            )

    def get_inverse(self):
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = get_clifford_id(inverse_ptm)
        return self.__class__(idx)


class SingleQubitClifford(Clifford):
    def __init__(self, idx: int, decomposition: str='Epstein'):
        assert(idx<24)
        self.idx = idx
        self.pauli_transfer_matrix = C1[idx]
        if decomposition =='Epstein':
            self.gate_decomposition  = gate_decomposition[idx]


class TwoQubitClifford(Clifford):
    def __init__(self, idx: int, decomposition: str='Epstein'):
        assert(idx<11520)
        self.idx = idx

        if idx < 576:
            self.pauli_transfer_matrix = single_qubit_like_PTM(idx)
        elif idx < 576 + 5184:
            self.pauli_transfer_matrix = CNOT_like_PTM(idx-576)
        elif idx< 576 + 2*5184:
            self.pauli_transfer_matrix = iSWAP_like_PTM(idx-(576+5184))
        elif idx<11520:
            self.pauli_transfer_matrix = SWAP_like_PTM(idx-(576+2*5184))

def single_qubit_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the single qubit like class
        (q0)  -- C1 --
        (q1)  -- C1 --
    """
    assert(idx<24**2)
    idx_q0 = idx%24
    idx_q1 = idx//24
    pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
    return pauli_transfer_matrix

def CNOT_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the cnot like class
        (q0)  --C1--•--S1--      --C1--•--S1------
                    |        ->        |
        (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
    """
    assert(idx<5184)
    idx_0 = idx % 24
    idx_1 = (idx // 24) % 24
    idx_2 = (idx // 576) % 3
    idx_3 = (idx // 1728)

    C1_q0 = np.kron(np.eye(4), C1[idx_0])
    C1_q1 = np.kron(C1[idx_1], np.eye(4))
    CZ
    S1_q0 = np.kron(np.eye(4), S1[idx_2])
    S1y_q1 = np.kron(np.dot(C1[idx_3], Y90), np.eye(4))
    return np.linalg.multi_dot([C1_q0, C1_q1, CZ, S1_q0, S1y_q1])

def iSWAP_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the iSWAP like class
        (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                    |       ->        |        |
        (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
    """
    assert(idx<5184)
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

    return np.linalg.multi_dot([C1_q0, C1_q1,
                              CZ, sq_swap_gates, CZ,
                              S1_q0, S1y_q1])


def SWAP_like_PTM(idx):
    """
    Returns the pauli transfer matrix for gates of the SWAP like class

    (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                |   ->        |       |       |
    (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
    """
    assert(idx<24**2)
    idx_q0 = idx%24
    idx_q1 = idx//24
    sq_like_cliff = np.kron(C1[idx_q1], C1[idx_q0])
    sq_swap_gates_0 = np.kron(Y90, mY90)
    sq_swap_gates_1 = np.kron(mY90, Y90)
    sq_swap_gates_2 = np.kron(Y90, np.eye(4))

    return np.linalg.multi_dot([sq_like_cliff, CZ,
                               sq_swap_gates_0, CZ,
                               sq_swap_gates_1, CZ,
                               sq_swap_gates_2])



def get_single_qubit_clifford_hash_table():
    """
    Get's the single qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    hash_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')

    with open(join(hash_dir, 'single_qubit_hash_lut.txt'),
              'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table

def get_two_qubit_clifford_hash_table():
    """
    Get's the two qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    hash_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')

    with open(join(hash_dir, 'two_qubit_hash_lut.txt'),
              'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table


def get_clifford_id(pauli_transfer_matrix):
    """
    returns the unique Id of a Clifford.
    """
    unique_hash = crc32(pauli_transfer_matrix.round().astype(int))
    if np.array_equal(np.shape(pauli_transfer_matrix), (4, 4)):
        hash_table = get_single_qubit_clifford_hash_table()
    elif np.array_equal(np.shape(pauli_transfer_matrix), (16, 16)):
        hash_table = get_two_qubit_clifford_hash_table()
    else:
        raise NotImplementedError()
    idx = hash_table.index(unique_hash)
    return idx
