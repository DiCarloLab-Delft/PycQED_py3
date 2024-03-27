import numpy as np
from pycqed.simulations.pauli_transfer_matrices import I, X, Y, Z, S, S2, H, CZ
'''
Decomposition of the single qubit clifford group as per
Eptstein et al. Phys. Rev. A 89, 062321 (2014)
'''

clifford_group_single_qubit = [np.empty([4, 4])]*(24)
# explictly reversing order because order of operators is order in time
clifford_group_single_qubit[0] = np.linalg.multi_dot([I, I, I][::-1])
clifford_group_single_qubit[1] = np.linalg.multi_dot([I, I, S][::-1])
clifford_group_single_qubit[2] = np.linalg.multi_dot([I, I, S2][::-1])
clifford_group_single_qubit[3] = np.linalg.multi_dot([X, I, I][::-1])
clifford_group_single_qubit[4] = np.linalg.multi_dot([X, I, S][::-1])
clifford_group_single_qubit[5] = np.linalg.multi_dot([X, I, S2][::-1])
clifford_group_single_qubit[6] = np.linalg.multi_dot([Y, I, I][::-1])
clifford_group_single_qubit[7] = np.linalg.multi_dot([Y, I, S][::-1])
clifford_group_single_qubit[8] = np.linalg.multi_dot([Y, I, S2][::-1])
clifford_group_single_qubit[9] = np.linalg.multi_dot([Z, I, I][::-1])
clifford_group_single_qubit[10] = np.linalg.multi_dot([Z, I, S][::-1])
clifford_group_single_qubit[11] = np.linalg.multi_dot([Z, I, S2][::-1])

clifford_group_single_qubit[12] = np.linalg.multi_dot([I, H, I][::-1])
clifford_group_single_qubit[13] = np.linalg.multi_dot([I, H, S][::-1])
clifford_group_single_qubit[14] = np.linalg.multi_dot([I, H, S2][::-1])
clifford_group_single_qubit[15] = np.linalg.multi_dot([X, H, I][::-1])
clifford_group_single_qubit[16] = np.linalg.multi_dot([X, H, S][::-1])
clifford_group_single_qubit[17] = np.linalg.multi_dot([X, H, S2][::-1])
clifford_group_single_qubit[18] = np.linalg.multi_dot([Y, H, I][::-1])
clifford_group_single_qubit[19] = np.linalg.multi_dot([Y, H, S][::-1])
clifford_group_single_qubit[20] = np.linalg.multi_dot([Y, H, S2][::-1])
clifford_group_single_qubit[21] = np.linalg.multi_dot([Z, H, I][::-1])
clifford_group_single_qubit[22] = np.linalg.multi_dot([Z, H, S][::-1])
clifford_group_single_qubit[23] = np.linalg.multi_dot([Z, H, S2][::-1])

# S1 is a subgroup of C1 (single qubit Clifford group) used when generating C2
S1 = [clifford_group_single_qubit[0],
      clifford_group_single_qubit[1],
      clifford_group_single_qubit[2]]


def generate_clifford_lookuptable(clifford_group_single_qubit):
    '''
    mapping in the lookuptable goes as follows

    Using the lookuptable:
    Row "i" corresponds to the clifford you have; Cl_A
    Column "j" corresponds to the clifford that is applied; Cl_B
    The value in (i,j) corresponds to the index of the resulting clifford
         Cl_C = np.dot(Cl_B, cl_A)

    Note: this function should work for generating a lookuptable for any closed
    group by taking a list of all elements in the group in matrix form as input
    arguments. However it has only been tested with the 24 element single qubit
    Clifford group.
    '''
    len_cl_grp = len(clifford_group_single_qubit)
    clifford_lookuptable = np.empty((len_cl_grp, len_cl_grp), dtype=int)
    for i in range(len_cl_grp):
        for j in range(len_cl_grp):
            # Reversed because column j is applied to row i
            net_cliff = (np.dot(clifford_group_single_qubit[
                         j], clifford_group_single_qubit[i]))
            net_cliff_id = [(net_cliff == cliff).all() for cliff in
                            clifford_group_single_qubit].index(True)
            clifford_lookuptable[i, j] = net_cliff_id
    return clifford_lookuptable

# Lookuptable based on representation of the clifford group used in this file
clifford_lookuptable = generate_clifford_lookuptable(
    clifford_group_single_qubit)
