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
                  [0, 0, np.sin(theta), np.cos(theta)]], dtype=int)
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
                  [0, -np.sin(theta), 0, np.cos(theta)]], dtype=int)
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
                  [0, 0, 0, 1]], dtype=int)
    return Z
