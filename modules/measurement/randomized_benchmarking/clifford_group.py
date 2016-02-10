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
              [0, 0, 0, -1]])
Y = np.array([[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, -1]])
Z = np.array([[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, 1]])
# Exchange group
S = np.array([[1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])
S2 = np.dot(S, S)
# Hadamard group
H = np.array([[1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, -1, 0],
              [0, 1, 0, 0]])


Clifford_group = [np.empty([4, 4])]*(26)
Clifford_group[0] = np.linalg.multi_dot([I, I, I])
Clifford_group[1] = np.linalg.multi_dot([I, I, S])
Clifford_group[2] = np.linalg.multi_dot([I, I, S2])
Clifford_group[3] = np.linalg.multi_dot([X, I, I])
Clifford_group[4] = np.linalg.multi_dot([X, I, S])
Clifford_group[5] = np.linalg.multi_dot([X, I, S2])
Clifford_group[6] = np.linalg.multi_dot([Y, I, I])
Clifford_group[7] = np.linalg.multi_dot([Y, I, S])
Clifford_group[8] = np.linalg.multi_dot([Y, I, S2])
Clifford_group[9] = np.linalg.multi_dot([Z, I, I])
Clifford_group[10] = np.linalg.multi_dot([Z, I, S])
Clifford_group[11] = np.linalg.multi_dot([Z, I, S2])

Clifford_group[12] = np.linalg.multi_dot([I, H, I])
Clifford_group[13] = np.linalg.multi_dot([I, H, S])
Clifford_group[14] = np.linalg.multi_dot([I, H, S2])
Clifford_group[15] = np.linalg.multi_dot([X, H, I])
Clifford_group[16] = np.linalg.multi_dot([X, H, S])
Clifford_group[17] = np.linalg.multi_dot([X, H, S2])
Clifford_group[18] = np.linalg.multi_dot([Y, H, I])
Clifford_group[19] = np.linalg.multi_dot([Y, H, S])
Clifford_group[20] = np.linalg.multi_dot([Y, H, S2])
Clifford_group[21] = np.linalg.multi_dot([Z, H, I])
Clifford_group[22] = np.linalg.multi_dot([Z, H, S])
Clifford_group[23] = np.linalg.multi_dot([Z, H, S2])
