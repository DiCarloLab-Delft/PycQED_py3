"""
FIXME: handling of cliffords is spread all over the place:
- this file provides classes Clifford, SingleQubitClifford and TwoQubitClifford
- pycqed.measurement.randomized_benchmarking.clifford_group defines clifford_group_single_qubit, and generate_clifford_lookuptable
- pycqed.measurement.randomized_benchmarking.generate_clifford_hash_tables provides generate_hash_tables
- pycqed.measurement.randomized_benchmarking.clifford_decompositions provides decompositions
- pycqed.simulations.pauli_transfer_matrices provides transfer matrices

And then there are hardcoded Clifford IDs and group sizes overywhere
"""

import numpy as np
from zlib import crc32
from os.path import join, dirname, abspath

from pycqed.measurement.randomized_benchmarking.clifford_group import clifford_group_single_qubit as C1  # the full group
from pycqed.measurement.randomized_benchmarking.clifford_group import S1  # the S1 subgroup of C1
from pycqed.simulations.pauli_transfer_matrices import CZ
from pycqed.measurement.randomized_benchmarking.clifford_decompositions import epstein_efficient_decomposition


hash_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')

# set as a module wide variable instead of argument to function for speed reasons
gate_decomposition = epstein_efficient_decomposition

# matrices used to transform the S1 subgroup
# FIXME: handle indices vs name in single place only
X90 = C1[16]
Y90 = C1[21]
mY90 = C1[15]

# A dict containing clifford IDs with common names.
# FIXME: should be separate per Clifford class. Hardly used
# FIXME: handle indices vs name in single place only
common_cliffords = {
    # in SingleQubitClifford:
    'I':  0, 'X':  3, 'Y':  6, 'Z':  9,

    'X90':  16,
    'Y90':  21,
    'X180':  3,
    'Y180':  6,
    'Z180':  9,

    # in TwoQubitClifford:
    'II':  0, 'IX':  3, 'IY':  6, 'IZ':  9,

    'XI': 24*3 + 0, 'XX': 24*3 + 3,
    'XY': 24*3 + 6, 'XZ': 24*3 + 9,

    'YI':  24*6 + 0, 'YX':  24*6 + 3,
    'YY':  24*6 + 6, 'YZ':  24*6 + 9,

    'ZI':  24*9 + 0, 'ZX':  24*9 + 3,
    'ZY':  24*9 + 6, 'ZZ':  24*9 + 9,

    # single qubit gates (hack) when using TwoQubitClifford
    'CZ': 104368 # 100000 + 576 + 14*24 + 2*1728
}


class Clifford(object):
    # class variables
    _hash_table = None

    def __mul__(self, other):
        """
        Product of two clifford gates.
        returns a new Clifford object that performs the net operation
        that is the product of both operations.
        """
        net_op = np.dot(self.pauli_transfer_matrix,
                        other.pauli_transfer_matrix)
        idx = self._get_clifford_id(net_op)
        return self.__class__(idx)

    def __repr__(self):
        return f'{self.__class__.__name__}(idx={self.idx})'

    def __str__(self):
        return f'{self.__class__.__name__} idx {self.idx}\n Gates: {self.gate_decomposition.__str__()}\n'

    def get_inverse(self):
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = self._get_clifford_id(inverse_ptm)
        return self.__class__(idx)

    ##########################################################################
    # Abstract class methods
    ##########################################################################

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix):
        pass


class SingleQubitClifford(Clifford):
    # class constants
    GRP_SIZE = 24

    # class variables
    _gate_decompositions = [None] * GRP_SIZE

    def __init__(self, idx: int, i: int=0):
        assert(idx < self.GRP_SIZE)
        self.idx = idx
        self.pauli_transfer_matrix = C1[idx]
        self.i = i

    @property  # FIXME: remove
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the single qubit Clifford group
        according to the decomposition by Epstein et al.
        """
        if self._gate_decompositions[self.idx] is None:
            _gate_decomposition = [(g, f'q{self.i}') for g in gate_decomposition[self.idx]]
            self._gate_decompositions[self.idx] = _gate_decomposition
        return self._gate_decompositions[self.idx]

    ##########################################################################
    # Class methods
    ##########################################################################

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix):
        """
        returns the unique Id of a Clifford.
        """
        unique_hash = crc32(pauli_transfer_matrix.astype(int))

        if cls._hash_table is None:
            cls._hash_table = get_single_qubit_clifford_hash_table()

        idx = cls._hash_table.index(unique_hash)
        return idx


"""
This class contains Clifford decompositions for the two qubit Clifford group.

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

class TwoQubitClifford(Clifford):
    # class constants
    GRP_SIZE_CLIFFORD = SingleQubitClifford.GRP_SIZE
    GRP_SIZE_SINGLE_QUBIT = GRP_SIZE_CLIFFORD**2
    GRP_SIZE_S1 = 3  # the S1 subgroup of SingleQubitClifford
    GRP_SIZE_CNOT = GRP_SIZE_SINGLE_QUBIT * GRP_SIZE_S1**2
    GRP_SIZE_ISWAP = GRP_SIZE_CNOT
    GRP_SIZE_SWAP = GRP_SIZE_SINGLE_QUBIT
    GRP_SIZE = GRP_SIZE_SINGLE_QUBIT + GRP_SIZE_CNOT + GRP_SIZE_ISWAP + GRP_SIZE_SWAP

    assert(GRP_SIZE_SINGLE_QUBIT == 576)
    assert(GRP_SIZE_CNOT == 5184)
    assert(GRP_SIZE == 11520)

    # FIXME: fix remaining magic constants below, and handle common code blocks as such

    # class variables
    _gate_decompositions = [None] * GRP_SIZE
    _pauli_transfer_matrices = [None] * GRP_SIZE

    def __init__(self, idx: int):
        assert(idx < self.GRP_SIZE)
        self.idx = idx

    @property  # FIXME: remove
    def pauli_transfer_matrix(self):
        # check cache
        if self._pauli_transfer_matrices[self.idx] is None:
            # compute
            if self.idx < 576:
                _pauli_transfer_matrix = self.single_qubit_like_PTM(self.idx)
            elif self.idx < 576 + 5184:
                _pauli_transfer_matrix = self.CNOT_like_PTM(self.idx-576)
            elif self.idx < 576 + 2*5184:
                _pauli_transfer_matrix = self.iSWAP_like_PTM(self.idx-(576+5184))
            else:  # NB: GRP_SIZE checked upon construction
                _pauli_transfer_matrix = self.SWAP_like_PTM(self.idx-(576+2*5184))

            # store in cache
            self._pauli_transfer_matrices[self.idx] = _pauli_transfer_matrix

        return self._pauli_transfer_matrices[self.idx]

    @property  # FIXME: remove
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the two qubit Clifford group.

        Single qubit Cliffords are decomposed according to Epstein et al.
        """

        # check cache
        if self._gate_decompositions[self.idx] is None:
            # compute
            if self.idx < 576:
                _gate_decomposition = self.single_qubit_like_gates(self.idx)
            elif self.idx < 576 + 5184:
                _gate_decomposition = self.CNOT_like_gates(self.idx-576)
            elif self.idx < 576 + 2*5184:
                _gate_decomposition = self.iSWAP_like_gates(self.idx-(576+5184))
            else:  # NB: GRP_SIZE checked upon construction
                _gate_decomposition = self.SWAP_like_gates(self.idx-(576+2*5184))

            # store in cache
            self._gate_decompositions[self.idx] = _gate_decomposition

        return self._gate_decompositions[self.idx]

    ##########################################################################
    # Class methods
    ##########################################################################

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix) -> int:
        """
        returns the unique Id of a Clifford.
        """
        unique_hash = crc32(pauli_transfer_matrix.astype(int))

        if cls._hash_table is None:
            cls._hash_table = get_two_qubit_clifford_hash_table()

        idx = cls._hash_table.index(unique_hash)
        return idx

    @classmethod
    def single_qubit_like_PTM(cls, idx):
        """
        Returns the pauli transfer matrix for gates of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert(idx < cls.GRP_SIZE_SINGLE_QUBIT)
        idx_q0 = idx % 24
        idx_q1 = idx//24
        pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
        return pauli_transfer_matrix

    @classmethod
    def single_qubit_like_gates(cls, idx):
        """
        Returns the gates for Cliffords of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert(idx < cls.GRP_SIZE_SINGLE_QUBIT)
        idx_q0 = idx % 24
        idx_q1 = idx//24

        g_q0 = [(g, 'q0') for g in gate_decomposition[idx_q0]]
        g_q1 = [(g, 'q1') for g in gate_decomposition[idx_q1]]
        gates = g_q0 + g_q1
        return gates

    @classmethod
    def CNOT_like_PTM(cls, idx):
        """
        Returns the pauli transfer matrix for gates of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert(idx < cls.GRP_SIZE_CNOT)
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = (idx // 1728)

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # CZ
        S1_q0 = np.kron(np.eye(4), S1[idx_2])
        S1y_q1 = np.kron(np.dot(C1[idx_3], Y90), np.eye(4))
        return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1, CZ, S1_q0, S1y_q1])))

    @classmethod
    def CNOT_like_gates(cls, idx):
        """
        Returns the gates for Cliffords of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert(idx < cls.GRP_SIZE_CNOT)
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = (idx // 1728)

        C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_0]]
        C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_1]]
        CZ = [('CZ', ['q0', 'q1'])]  # FIXME: shadows 'CZ' from outer scope, more occurrences below

        idx_2s = SingleQubitClifford._get_clifford_id(S1[idx_2])
        S1_q0 = [(g, 'q0') for g in gate_decomposition[idx_2s]]
        # FIXME: precomputation of these 3 entries would be more efficient (more similar occurrences in this file):
        idx_3s = SingleQubitClifford._get_clifford_id(np.dot(C1[idx_3], Y90))
        S1_yq1 = [(g, 'q1') for g in gate_decomposition[idx_3s]]

        gates = C1_q0 + C1_q1 + CZ + S1_q0 + S1_yq1
        return gates

    @classmethod
    def iSWAP_like_PTM(cls, idx):
        """
        Returns the pauli transfer matrix for gates of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert(idx < cls.GRP_SIZE_ISWAP)
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = (idx // 1728)

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # CZ
        sq_swap_gates = np.kron(mY90, Y90)
        # CZ
        S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
        S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

        return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1,
                                                  CZ, sq_swap_gates, CZ,
                                                  S1_q0, S1y_q1])))

    @classmethod
    def iSWAP_like_gates(cls, idx):
        """
        Returns the gates for Cliffords of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert(idx < cls.GRP_SIZE_ISWAP)
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = (idx // 1728)

        C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_0]]
        C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_1]]
        CZ = [('CZ', ['q0', 'q1'])]

        sqs_idx_q0 = SingleQubitClifford._get_clifford_id(Y90)
        sqs_idx_q1 = SingleQubitClifford._get_clifford_id(mY90)
        sq_swap_gates_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

        # S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
        # S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

        idx_2s = SingleQubitClifford._get_clifford_id(np.dot(S1[idx_2], Y90))
        S1_q0 = [(g, 'q0') for g in gate_decomposition[idx_2s]]
        idx_3s = SingleQubitClifford._get_clifford_id(np.dot(C1[idx_3], X90))
        S1y_q1 = [(g, 'q1') for g in gate_decomposition[idx_3s]]

        gates = (C1_q0 + C1_q1 + CZ +
                 sq_swap_gates_q0 + sq_swap_gates_q1 + CZ +
                 S1_q0 + S1y_q1)
        return gates

    @classmethod
    def SWAP_like_PTM(cls, idx:int) -> np.ndarray:
        """
        Returns the pauli transfer matrix for gates of the SWAP like class

        (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
        """
        assert(idx < cls.GRP_SIZE_SWAP)
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

    @classmethod
    def SWAP_like_gates(cls, idx):
        """
        Returns the gates for Cliffords of the SWAP like class

        (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
        """
        assert(idx < cls.GRP_SIZE_SWAP)
        idx_q0 = idx % 24
        idx_q1 = idx//24
        C1_q0 = [(g, 'q0') for g in gate_decomposition[idx_q0]]
        C1_q1 = [(g, 'q1') for g in gate_decomposition[idx_q1]]
        CZ = [('CZ', ['q0', 'q1'])]

        # sq_swap_gates_0 = np.kron(Y90, mY90)

        sqs_idx_q0 = SingleQubitClifford._get_clifford_id(mY90)
        sqs_idx_q1 = SingleQubitClifford._get_clifford_id(Y90)
        sq_swap_gates_0_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_0_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

        sqs_idx_q0 = SingleQubitClifford._get_clifford_id(Y90)
        sqs_idx_q1 = SingleQubitClifford._get_clifford_id(mY90)
        sq_swap_gates_1_q0 = [(g, 'q0') for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_1_q1 = [(g, 'q1') for g in gate_decomposition[sqs_idx_q1]]

        sqs_idx_q1 = SingleQubitClifford._get_clifford_id(Y90)
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

# FIXME: handle all hash table handling to single file/class
try:
    open(join(hash_dir, 'single_qubit_hash_lut.txt'), 'r')
    # FIXME: also check 'two_qubit_hash_lut.txt'
except FileNotFoundError:
    print("Clifford group hash tables not detected.")
    from pycqed.measurement.randomized_benchmarking.generate_clifford_hash_tables import generate_hash_tables
    generate_hash_tables()


def get_single_qubit_clifford_hash_table():
    """
    Get's the single qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    with open(join(hash_dir, 'single_qubit_hash_lut.txt'), 'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table


def get_two_qubit_clifford_hash_table():
    """
    Get's the two qubit clifford hash table. Requires this to be generated
    first. To generate, execute "generate_clifford_hash_tables.py".
    """
    with open(join(hash_dir, 'two_qubit_hash_lut.txt'), 'r') as f:
        hash_table = [int(line.rstrip('\n')) for line in f]
    return hash_table

# FIXME: replace by class methods _get_clifford_id()
# def get_clifford_id(pauli_transfer_matrix):
#     """
#     returns the unique Id of a Clifford.
#     """
#     # FIXME: opens file on every call
#     unique_hash = crc32(pauli_transfer_matrix.astype(int))
#     if np.array_equal(np.shape(pauli_transfer_matrix), (4, 4)):
#         hash_table = get_single_qubit_clifford_hash_table()
#     elif np.array_equal(np.shape(pauli_transfer_matrix), (16, 16)):
#         hash_table = get_two_qubit_clifford_hash_table()
#     else:
#         raise NotImplementedError()
#     idx = hash_table.index(unique_hash)
#     return idx
