import numpy as np


def normalize(v):
    """
    Normalizes 1D (possibly complex) vector
    :param v: vector to normalize
    :return: normalized vector
    """
    return v / np.sqrt(v.dot(v.conjugate()))

def gram_schmidt(B):
    """
    Gram Schmidt algorithm used to orthonormalize matrix B.
    :param B: complex matrix. e.g. \n
              B = [[b_{0,0}],[b_{0,1}],
                  [[b_{1,0}],[b_{1,1}]]\n where b_{i,j} corresponds
              to the i-th element of column basis vector j
              and the goal is to orthonormalize the basis spanned by
              the columns of B
    :return: Orthonormalized matrix with same dimensions as B
    """
    B[:, 0] = normalize(B[:, 0])
    for i in range(1, B.shape[1]):
        Ai = B[:, i]
        for j in range(0, i):
            Aj = B[:, j]
            t = Ai.dot(Aj.conjugate())
            Ai = Ai - t * Aj
        B[:, i] = normalize(Ai)
    return B