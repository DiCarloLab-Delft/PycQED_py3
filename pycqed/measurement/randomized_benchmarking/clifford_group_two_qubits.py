import numpy as np
'''
Decomposition of the single qubit clifford group as per
Eptstein et al. Phys. Rev. A 89, 062321 (2014)
'''


# Clifford group decomposition maps
I = np.eye(4)
# Pauli group
X = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, -1]], dtype=int)
Y = np.array([[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, -1]], dtype=int)
Z = np.array([[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, 1]], dtype=int)
# Exchange group
S = np.array([[1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0]], dtype=int)
S2 = np.dot(S, S)
# Hadamard group
H = np.array([[1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, -1, 0],
              [0, 1, 0, 0]], dtype=int)


Clifford_group = [np.empty([4, 4])]*(24)
# explictly reversing order because order of operators is order in time
Clifford_group[0] = np.linalg.multi_dot([I, I, I][::-1])
Clifford_group[1] = np.linalg.multi_dot([I, I, S][::-1])
Clifford_group[2] = np.linalg.multi_dot([I, I, S2][::-1])
Clifford_group[3] = np.linalg.multi_dot([X, I, I][::-1])
Clifford_group[4] = np.linalg.multi_dot([X, I, S][::-1])
Clifford_group[5] = np.linalg.multi_dot([X, I, S2][::-1])
Clifford_group[6] = np.linalg.multi_dot([Y, I, I][::-1])
Clifford_group[7] = np.linalg.multi_dot([Y, I, S][::-1])
Clifford_group[8] = np.linalg.multi_dot([Y, I, S2][::-1])
Clifford_group[9] = np.linalg.multi_dot([Z, I, I][::-1])
Clifford_group[10] = np.linalg.multi_dot([Z, I, S][::-1])
Clifford_group[11] = np.linalg.multi_dot([Z, I, S2][::-1])

Clifford_group[12] = np.linalg.multi_dot([I, H, I][::-1])
Clifford_group[13] = np.linalg.multi_dot([I, H, S][::-1])
Clifford_group[14] = np.linalg.multi_dot([I, H, S2][::-1])
Clifford_group[15] = np.linalg.multi_dot([X, H, I][::-1])
Clifford_group[16] = np.linalg.multi_dot([X, H, S][::-1])
Clifford_group[17] = np.linalg.multi_dot([X, H, S2][::-1])
Clifford_group[18] = np.linalg.multi_dot([Y, H, I][::-1])
Clifford_group[19] = np.linalg.multi_dot([Y, H, S][::-1])
Clifford_group[20] = np.linalg.multi_dot([Y, H, S2][::-1])
Clifford_group[21] = np.linalg.multi_dot([Z, H, I][::-1])
Clifford_group[22] = np.linalg.multi_dot([Z, H, S][::-1])
Clifford_group[23] = np.linalg.multi_dot([Z, H, S2][::-1])


def generate_clifford_lookuptable(Clifford_group):
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
    len_cl_grp = len(Clifford_group)
    clifford_lookuptable = np.empty((len_cl_grp, len_cl_grp), dtype=int)
    for i in range(len_cl_grp):
        for j in range(len_cl_grp):
            # Reversed because column j is applied to row i
            net_cliff = (np.dot(Clifford_group[j], Clifford_group[i]))
            net_cliff_id = [(net_cliff == cliff).all() for cliff in
                            Clifford_group].index(True)
            clifford_lookuptable[i, j] = net_cliff_id
    return clifford_lookuptable

# Lookuptable based on representation of the clifford group used in this file
clifford_lookuptable = generate_clifford_lookuptable(Clifford_group)
