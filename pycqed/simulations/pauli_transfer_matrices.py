import numpy as np
"""
This file contains pauli transfer matrices for all basic qubit operations.
"""


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

CZ = np.array([
    [1,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 1,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  1,  0],
    [0,  0,  0,  1,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],

    [0,  0,  0,  0,   0,  0,  0,  1,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  1,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   0,  -1,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   1,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],

    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  1,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  -1,  0,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  1,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   1,  0,  0,  0,   0, 0,  0,  0],

    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   1, 0,  0,  0],
    [0,  1,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  1,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  0],
    [0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   0, 0,  0,  1]],
    dtype=int)


def X_theta(theta:float, unit='deg'):
    """
    PTM of rotation of theta degrees along the X axis
    """
    if unit=='deg':
        theta = np.deg2rad(theta)

    X = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, np.cos(theta), -np.sin(theta)],
                  [0, 0, np.sin(theta), np.cos(theta)]], dtype=float)
    return X


def Y_theta(theta:float, unit='deg'):
    """
    PTM of rotation of theta degrees along the X axis
    """
    if unit=='deg':
        theta = np.deg2rad(theta)

    Y = np.array([[1, 0, 0, 0],
                  [0, np.cos(theta), 0, np.sin(theta)],
                  [0, 0, 1, 0],
                  [0, -np.sin(theta), 0, np.cos(theta)]], dtype=float)
    return Y


def Z_theta(theta:float, unit='deg'):
    """
    PTM of rotation of theta degrees along the X axis
    """
    if unit=='deg':
        theta = np.deg2rad(theta)

    Z = np.array([[1, 0, 0, 0],
                  [0, np.cos(theta), -np.sin(theta), 0],
                  [0, np.sin(theta), np.cos(theta), 0],
                  [0, 0, 0, 1]], dtype=float)
    return Z



##############################################################################
#
##############################################################################

def process_fidelity(ptm_0, ptm_1, d: int=None):
    """
    Calculates the average process fidelity between two pauli transfer matrices
    Args:
        ptm_0 (array) : n*n array specifying the first pauli transfer matrix
        ptm_1 (array) : n*n array specifying the second pauli transfer matrix
        d    (int)    : dimension of the Hilbert space
    returns:
        F (float)     : Process fidelity
    """
    if d == None:
        d = np.shape(ptm_0)[0]**0.5

    return np.dot(ptm_0.T, ptm_1).trace()/(d**2)


def average_gate_fidelity(ptm_0, ptm_1, d: int=None):

    """
    Calculates the average average gate fidelity between two pauli transfer
        matrices
    Args:
        ptm_0 (array) : n*n array specifying the first pauli transfer matrix
        ptm_1 (array) : n*n array specifying the second pauli transfer matrix
        d    (int)    : dimension of the Hilbert space
    returns:
        F_gate (float): Average gate fidelity
    """

    if d == None:
        d = np.shape(ptm_0)[0]**0.5
    F_pro = process_fidelity(ptm_0, ptm_1, d)
    F_avg_gate = process_fid_to_avg_gate_fid(F_pro, d)
    return F_avg_gate

def process_fid_to_avg_gate_fid(F_pro: float, d:int):
    """
    Converts
    """
    F_avg_gate = (d*F_pro+1)/(d+1)
    return F_avg_gate
