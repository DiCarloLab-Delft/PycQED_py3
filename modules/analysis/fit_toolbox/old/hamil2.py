import numpy as n
from numpy import *
import numpy
from numpy import matrix
import scipy
from scipy.linalg import inv, det, eig
import scipy.linalg as la
'''
This file containts different hamiltonians and matrices.
'''

def hamilB(B0):
    '''
    Calculate hamiltonians for all 4 orientations for specific B field of the 
    P1 defec in diamond.
    creates hamiltonians of the form:
    H = alfa B @ S + A @ S @ I
    with @ the tensor inproduct operation. 
    input: B-field vector
    outputs: 4 hamiltonians
    '''

    # constants.
    pi = n.pi
    beta = 1  # Bohr magneton
    g = 1  # spectroscopic splitting factor
    alfa = 28.04
    angle = pi -2 * arcsin(1 / sqrt(3))  #109.5 degrees angle between tetrahedral bonds.

    # matrices.
    Sx, Sy, Sz = spinhalf()
    Ix, Iy, Iz = spin1()
    R1 = transpose(rot(angle, axis = 'y'))
    R2 = transpose(rot(pi * 2 / 3, axis = 'z') * rot(angle, axis = 'y'))
    R3 = transpose(rot(-pi * 2 / 3, axis = 'z') * rot(angle, axis = 'y'))

    B1_ = B0
    B2_ = R1 * B1_
    B3_ = R2 * B1_
    B4_ = R3 * B1_

    #hyperfine tensor
    A = matrix([[81.33], [81.33], [114.03]])
    
    # hyperfine interaction hamiltonian
    H_ASI = A.item(0) * la.kron(Sx, Ix) + A.item(1) * la.kron(Sy, Iy) + A.item(2) * la.kron(Sz, Iz)

    # hamiltonians work in electron spin half + nuclear spin 1 base, 6D
    H1 = alfa * la.kron((B1_.item(0) * Sx + B1_.item(1) * Sy + B1_.item(2) * Sz), eye(3)) + H_ASI 
    H2 = alfa * la.kron((B2_.item(0) * Sx + B2_.item(1) * Sy + B2_.item(2) * Sz), eye(3)) + H_ASI
    H3 = alfa * la.kron((B3_.item(0) * Sx + B3_.item(1) * Sy + B3_.item(2) * Sz), eye(3)) + H_ASI
    H4 = alfa * la.kron((B4_.item(0) * Sx + B4_.item(1) * Sy + B4_.item(2) * Sz), eye(3)) + H_ASI


    return H1, H2, H3, H4

def evalcalc(H):
    '''
    calculates all energy differences from the lowest one in the P1 spectrum, 
    for all 4 different orrientations.
    '''
    f = n.zeros([len(H[0]) - 1, len(H)])

    for k in range(len(H)):
        Eval = eig(H[k], left = False, right = False)
        Eval = sort(Eval)
        for i in range(len(Eval)-1):
            f[i,k] = abs(Eval[i+1] - Eval[0])         

    return f

def evalcalc2(H):
    '''
    calculates all energy differences from the lowest one in the P1 spectrum, 
    for all 4 different orrientations.
    '''
    f = n.zeros([len(H[0]), len(H)])

    for k in range(len(H)):
        Eval = eig(H[k], left = False, right = False)
        f[:,k] = real(Eval)

    return f

def evallvls(f):
    y = n.zeros([3, 4])
    for k in range(4):
        y[0, k] = f[4,k]
        y[1, k] = f[3,k] - f[0,k]
        y[2, k] = f[2,k] - f[1,k]

    return y


def rot(theta, **kw):
    '''
    Returns rotation matrix around specific axis defined by axis of angle theta.

    Known KW:
    -axis: specify 'x','y','z'.
    '''
    a = numpy.sin(theta)
    b = numpy.cos(theta)
    axis = kw.pop('axis',True)

    if axis=='z':
        R = matrix([[b, -a, 0], [a, b, 0], [0, 0, 1]])
    else:
        if axis=='y':
            R = matrix([[b, 0, -a],[0, 1, 0],[a, 0, b]])
        else:
            R = matrix([[1, 0, 0],[0, b, -a],[0, a, b]])

    return R

def spinhalf(**kw):
    '''
    Returns Spin matrices for spin half system. The value of h_bar can be set,
    otherwise h_bar = 1. Can either return 1 of the spin matrices or tensor 
    S=[Sx Sy Sz].
    '''
    h = kw.pop('h',True)
    if kw.pop('h',False):
        h = 1

    axis = kw.pop('axis',True)
    a = h / 2.
    Sx = matrix([[0, a],[a, 0]])
    Sy = matrix([[0, -1j*a],[1j*a, 0]])
    Sz = matrix([[a, 0], [0, -a]])

    if axis=='z':
        S = Sz
    else:
        if axis=='y':
            S = Sy
        else:
            if axis=='x':
                S = Sx 
            else:
                S = [Sx, Sy, Sz]
    return S


def spin1(**kw):
    '''
    Returns Spin matrices for spin 1 system. The value of h_bar can be set, otherwise 
    h_bar = 1. Can either return 1 of the spin matrices or tensor S=[Sx Sy Sz].
    '''
    h = kw.pop('h',True)
    if kw.pop('h',False):
        h = 1

    axis = kw.pop('axis',True)
    a = h / numpy.sqrt(2)
    Sx = matrix([[0, a, 0],[a, 0, a],[0, a, 0]])
    Sy = matrix([[0, -1j*a, 0],[1j*a, 0, -1j*a],[0, 1j*a, 0]])
    Sz =matrix([[h, 0, 0], [0, 0, 0], [0, 0, -h]])

    if axis=='z':
        S = Sz
    else:
        if axis=='y':
            S = Sy
        else:
            if axis=='x':
                S = Sx 
            else:
                S = [Sx, Sy, Sz]
    return S


def dzf1(Dvalue): 
    '''
    Returns the zero field splitting matrix D for spin 1 system. (not most general case)
    '''
    D = matrix([[Dvalue,0,0],[0,0,0],[0,0,Dvalue]])
    return D
