import logging
import numpy as np
from numpy.linalg import inv
from typing import List, Optional, Tuple
import scipy as sp
try:
    import qutip as qtp
except ImportError as e:
    logging.warning('Could not import qutip, tomo code will not work')

# General state tomography functions

def least_squares_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                             Omega: Optional[np.ndarray]=None) -> qtp.Qobj:
    """
    Executes generalized linear least squares fit of the density matrix to
    the measured observables.

    The resulting density matrix could be unphysical, as the positive
    semi-definiteness is not ensured. The resulting density matrix has unit
    trace.

    To enforce the unit trace of the resulting density matrix, only the
    traceless part of the density matrix is fitted. For this we also need to
    subtract the trace of the measurement operators from the expectation values
    before fitting.

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus.
        Omega: The covariance matrix of the expectation values mu.
               If a 1-dimensional array is passed, the values are interpreted
               as the variations of the mus and the correlations are assumed to
               be zero.
               If `None` is passed, then all measurements are assumed to have
               equal variances.
    Returns: The found density matrix as a qutip operator.
    """
    d = Fs[0].shape[0]
    if Omega is None:
        Omega = np.diag(np.ones(len(mus)))
    elif len(Omega.shape) == 1:
        Omega = np.diag(Omega)
    OmegaInv = inv(Omega)

    Os = hermitian_traceless_basis(d)
    Fmtx = np.array([[(F * O).tr().real for O in Os] for F in Fs])
    F0s = np.array([F.tr().real / d for F in Fs])

    cs = inv((Fmtx.T).dot(OmegaInv).dot(Fmtx)).dot(Fmtx.T).dot(OmegaInv).dot(
        mus - F0s)

    rho = qtp.qeye(d) / d + sum([c * O for c, O in zip(cs, Os)])
    return rho


def hermitian_traceless_basis(d: int) -> List[qtp.Qobj]:
    """
    Generates a list of basis-vector matrices for the group of traceless
    Hermitian matrices.
    Args:
        d: dimension of the Hilbert space.
    Returns: A list of basis-vector matrices.
    """
    Os = []
    for i in range(d):
        for j in range(d):
            if i == d - 1 and j == d - 1:
                continue
            elif i == j:
                O = np.diag(-np.ones(d, dtype=np.complex) / d)
                O[i, j] += 1
            elif i > j:
                O = np.zeros((d, d), dtype=np.complex)
                O[i, j] = 1
                O[j, i] = 1
            else:
                O = np.zeros((d, d), dtype=np.complex)
                O[i, j] = 1j
                O[j, i] = -1j
            Os.append(qtp.Qobj(O))
    return Os


def mle_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                   Omega: Optional[np.ndarray]=None,
                   rho_guess: Optional[qtp.Qobj]=None) -> qtp.Qobj:
    """
    Executes a maximum likelihood fit to the measured observables, respecting
    the physicality constraint of the density matrix.

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus.
        Omega: The covariance matrix of the expectation values mu.
               If a 1-dimensional array is passed, the values are interpreted
               as the variations of the mus and the correlations are assumed to
               be zero.
               If `None` is passed, then all measurements are assumed to have
               equal variances.
        rho_guess: The initial value for the density matrix.
    Returns: The found density matrix as a qutip operator.
    """
    d = Fs[0].shape[0]
    if Omega is None:
        Omega = np.diag(np.ones(len(mus)))
    elif len(Omega.shape) == 1:
        Omega = np.diag(Omega)
    OmegaInv = inv(Omega)

    if rho_guess is None:
        params0 = np.ones(d * d)
    else:
        rho_positive = rho_guess - min(rho_guess.eigenenergies().min(), 0)
        rho_positive += 1e-9
        T = np.linalg.cholesky(rho_positive.data.toarray())
        params0 = np.real(np.concatenate((T[np.tril_indices(d)],
                                          -1j*T[np.tril_indices(d, -1)])))

    def min_func(params, mus, Fs, OmegaInv, d):
        T = ltriag_matrix(params, d)
        rho = T * T.dag()
        dmus = mus - np.array([(rho * F).tr().real for F in Fs])
        return (dmus.T.dot(OmegaInv).dot(dmus)).real

    def constr_func(params, d):
        T = ltriag_matrix(params, d)
        rho = T * T.dag()
        return rho.tr().real - 1

    params = sp.optimize.minimize(min_func, params0, (mus, Fs, OmegaInv, d),
                                  method='SLSQP', constraints={'type': 'eq',
                                      'fun': constr_func, 'args': (d, )}).x
    T = ltriag_matrix(params, d)
    return T * T.dag()


def ltriag_matrix(params: np.ndarray, d: int):
    """
    Creates a lower-triangular matrix of dimension `d` from an array of d**2
    real numbers with the elements on the diagonal being real and complex on
    the off-diagonal.

    The Cholesky decomposition of a positive-definite matrix is of this form.
    """
    T = np.zeros((d, d), dtype=np.complex)
    T[np.tril_indices(d)] += params[:((d * d + d) // 2)]
    T[np.tril_indices(d, -1)] += 1j * params[((d * d + d) // 2):]
    return qtp.Qobj(T)

# PycQED specific functions


def measurement_operator_from_calpoints(
        calpoints: np.ndarray, repetitions:int=1) -> Tuple[qtp.Qobj, float]:
    """
    Calculates the measurement operator and its expected variation from
    a list of calibration points corresponding to the preparations of the
    diagonal elements of the density matrix.

    For example for a two-qubit measurement, the calibration points would
    correspond to the prepared states gg, ge, eg, ee.
    """
    d = len(calpoints) // repetitions
    means = np.array([np.mean(calpoints[-repetitions*i:][:repetitions])
                      for i in range(d, 0, -1)])
    F = qtp.Qobj(np.diag(means))
    variation = np.std(calpoints - means.repeat(repetitions))**2 \
        if repetitions > 1 else 1
    return F, variation


def rotated_measurement_operators(rotations: List[qtp.Qobj],
                                  Fs: List[qtp.Qobj]) -> List[List[qtp.Qobj]]:
    """
    For each measurement operator in Fs, calculates the measurement operators
    when first applying the rotations in the rotations parameter to the system.
    """
    return [[U.dag() * F * U for U in rotations] for F in Fs]


def standard_qubit_pulses_to_rotations(pulse_list: List[Tuple]) \
        -> List[qtp.Qobj]:
    """
    Converts lists of n-tuples of standard PycQED single-qubit pulse names to
    the corresponding rotation matrices on the n-qubit Hilbert space.
    """
    standard_pulses = {
        'I': qtp.qeye(2),
        'X180': qtp.rotation(qtp.sigmax(), np.pi),
        'mX180': qtp.rotation(qtp.sigmax(), -np.pi),
        'Y180': qtp.rotation(qtp.sigmay(), np.pi),
        'mY180': qtp.rotation(qtp.sigmay(), -np.pi),
        'X90': qtp.rotation(qtp.sigmax(), np.pi/2),
        'mX90': qtp.rotation(qtp.sigmax(), -np.pi/2),
        'Y90': qtp.rotation(qtp.sigmay(), np.pi/2),
        'mY90': qtp.rotation(qtp.sigmay(), -np.pi/2),
    }
    rotations = [qtp.tensor(*[standard_pulses[pulse] for pulse in qb_pulses])
                 for qb_pulses in pulse_list]
    for i in range(len(rotations)):
        rotations[i].dims = [[d] for d in rotations[i].shape]
    return rotations


def fidelity(rho1: qtp.Qobj, rho2: qtp.Qobj) -> float:
    """ Returns the fidelity between the two quantum states rho1 and rho2. """
    if rho1.type == 'ket':
        rho1 = rho1 * rho1.dag()
    elif rho1.type == 'bra':
        rho1 = rho1.dag() * rho1
    if rho2.type == 'ket':
        rho2 = rho2 * rho2.dag()
    elif rho2.type == 'bra':
        rho2 = rho2.dag() * rho2
    rho1.dims = rho2.dims
    return (rho1.sqrtm()*rho2*rho1.sqrtm()).sqrtm().tr().real ** 2


def purity(rho: qtp.Qobj) -> float:
    if rho.type == 'ket' or rho.type == 'bra':
        return 1
    else:
        return (rho*rho).tr().real
