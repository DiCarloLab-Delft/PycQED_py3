from os.path import join, dirname, abspath
from pycqed.measurement.randomized_benchmarking.clifford_group import clifford_group_single_qubit
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

###################################
# The single qubit Clifford group #
###################################

# S1 Clifford subgroup
# S1 = [C1[0], C1[8], C1[7]]
# S1_rX90 = [C1[12], C1[22], C1[15]]
# S1_rY90 = [C1[14], C1[20], C1[17]]

# 1. encoding gates/paulis
# 2. How to multiply operators
# 3. How to go from operator to Clifford decomposed into gates (11520 lookuptable)
    # Lookuptable for inversion using hashes


"""
Encoding into the extended Pauli set
We encode very Clifford into 3 2 bit integers
    P = {phase_bit , \sigma_i, \sigma_j | i,j element {0, X, Y, Z}}
    Encoding:
        sign_bit:
             1 + 0i -> 00
             1j -> 01
            -1 + 0i -> 10
            -1j -> 11



Example: rZ90 -> {
"""





class Clifford(object):
    # def __init__(self):
    #     self.operator = operator
    #     self.idx = None # int denoting the id of the Clifford
    #     self.gate_decomposition = None

    def __mul__(self, other):
        """
        Product of two clifford gates.
        returns a Clifford gate with
        """
        # FIXME: Puali multiplication must be implemented here
        net_op = self.operator*other.operator
        return Clifford(net_op)

    def __repr__(self):
        return '{}(idx={})'.format(self.__class__.__name__, self.idx)

    def __str__(self):
        return str(self.gate_decomposition)

class SingleQubitClifford(Clifford):
    def __init__(self, idx: int, decomposition: str='Epstein'):
        assert(idx<24)
        self.idx = idx
        self.pauli_transfer_matrix = clifford_group_single_qubit[idx]
        if decomposition =='Epstein':
            self.gate_decomposition  = gate_decomposition[idx]


class TwoQubitClifford(Clifford):
    def __init__(self, idx: int, decomposition: str='Epstein'):
        assert(idx<11520)
        self.idx = idx
        raise NotImplemented()
        # self.pauli_transfer_matrix = clifford_group_single_qubit[idx]
        # if decomposition =='Epstein':
        #     self.gate_decomposition  = gate_decomposition[idx]


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



# class Single_qubit_like(Clifford):
#     def __init__(self, idx: int):
#         assert(idx<24**2)
#         self.idx = idx
#         idx_q0 = idx%24
#         idx_q1 = idx//24
#         # FIXME: proper way of defining the operator
#         self.operator = C1_op[idx_q0], C1_op[idx_q1]
#         self.gates = C1[idx_q0], C1[idx_q1]

# class CNOT_like(Clifford):
#     def __init__(self, j:int):
#         assert(j<5184)
#         j1 = j % 24
#         j2 = (j // 24) % 24
#         j3 = S1_convert[(j // 576) % 3]
#         j4 = S1Y_convert[(j // 1728)]
#         self.operator = Pauli_prod([sqc1[j1],sqc2[j2],CX,sqc1[j3],sqc2[j4]])

# class iSWAP_like(Cliffordt):
#     def __init__(self, j:int):
#         assert(j<5184)
#         j1 = j % 24
#         j2 = (j // 24) % 24
#         j3 = S1Y_convert[(j // 576) % 3]
#         j4 = S1X_convert[(j // 1728)]
#         self.operator = Pauli_prod([sqc1[j1],sqc2[j2],iSWAP,sqc1[j3],sqc2[j4]])

# class SWAP_like(Clifford):
#     def __init__(self, j:int):
#         assert(j<576)
#         j1 = j % 24
#         j2 = j // 24
#         self.operator = Pauli_prod([sqc1[j1],sqc2[j2],SWAP])





# def C2_operator(j: int):
#     """
#     Returns the pauli operator for element j of the two qubit
#     Clifford group C2.


#     """
#     if j < 576:
#         return SQ_class_operator(j)
#     elif j < 576 + 5184:
#         return CNOT_class_operator(j-576)
#     elif j< 576 + 2*5184:
#         return iSWAP_class_operator(j-576+5184)
#     elif j<11520:
#         return SWAP_class_operator(j-576+2*5184)
#     else:
#         raise ValueError('j ({}) must be smaller than {}')
