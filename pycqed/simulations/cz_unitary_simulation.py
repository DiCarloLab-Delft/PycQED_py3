"""
April 2018
Simulates the trajectory implementing a CZ gate.
"""
import time
import numpy as np
import qutip as qtp
from pycqed.measurement import detector_functions as det
from scipy.interpolate import interp1d
from pycqed.measurement.waveform_control_CC import waveform as wf

alpha_q0 = 250e6 * 2*np.pi
w_q0 = 6e9 * 2 * np.pi  # Lower frequency qubit
w_q1 = 7e9 * 2 * np.pi  # Upper frequency (fluxing) qubit

J = 2.5e6 * 2 * np.pi  # coupling strength

tlist = np.arange(0, 250e-9, .05e-9)

# operators
b = qtp.tensor(qtp.destroy(2), qtp.qeye(3))  # LSB is static qubit
a = qtp.tensor(qtp.qeye(2), qtp.destroy(3))
n_q0 = a.dag() * a
n_q1 = b.dag() * b


# Hamiltonian
def coupled_transmons_hamiltonian(w_q0, w_q1, alpha_q0, J):
    """
    Hamiltonian of two coupled anharmonic resonators.
    Because the intention is to tune one qubit into resonance with the other,
    the number of levels is limited.
        q1 -> fluxing qubit, 2-levels
        q0 -> static qubit, 3-levels

    intended avoided crossing:
        11 <-> 02

    N.B. the frequency of q1 is expected to be larger than that of q0
        w_q1 > w_q1
    """

    H_0 = w_q0 * n_q0 + w_q1 * n_q1 +  \
        1/2*alpha_q0*(a.dag()*a.dag()*a*a) +\
        J * (a.dag() + a) * (b + b.dag())
    return H_0


H_0 = coupled_transmons_hamiltonian(w_q0=w_q0, w_q1=w_q1, alpha_q0=alpha_q0,
                                    J=J)

#
U_target = qtp.Qobj([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 0, 1]],
                    type='oper',
                    dims=[[2, 3], [3, 2]])
U_target._type = 'oper'


def rotating_frame_transformation(U, t: float,
                                  w_q0: float=0, w_q1: float =0):
    """
    Transforms the frame of the unitary according to
        U' = U_{RF}*U*U_{RF}^dag
    with
        U_{RF} = e^{-i w_q0 a^dag a t } otimes e^{-i w_q1 b^dag b t }

    Args:
        U (QObj): Unitary to be transformed
        t (float): time at which to transform
        w_q0 (float): freq of frame for q0
        w_q1 (float): freq of frame for q1

    """
    U_RF = (1j*w_q0*n_q0*t).expm() * (1j*w_q1*n_q1*t).expm()

    U_prime = U_RF * U  # * U_RF(t=0).dag()
    return U_prime


def phases_from_unitary(U):
    """
    Returns the phases from the unitary
    """
    phi_00 = np.rad2deg(np.angle(U[0, 0]))  # expected to equal 0
    phi_01 = np.rad2deg(np.angle(U[1, 1]))
    phi_10 = np.rad2deg(np.angle(U[3, 3]))
    phi_11 = np.rad2deg(np.angle(U[4, 4]))

    phi_cond = (phi_11 - phi_01 - phi_10 + phi_00) % 360

    return phi_00, phi_01, phi_10, phi_11, phi_cond


def pro_fid_unitary_compsubspace(U, U_target):
    # TODO: verify if this is correct (it seems correct)
    """
    Process fidelity in the computational subspace for a qubit and qutrit


    """
    inner = U.dag()*U_target
    part_idx = [0, 1, 3, 4]  # only computational subspace
    ptrace = 0
    for i in part_idx:
        ptrace += inner[i, i]
    dim = 4  # 2 qubits comp subspace

    return ptrace/dim


def leakage_from_unitary(U):
    """
    Calculates leakage by summing over all in and output states in the
    computational subspace.
        L1 = 1- sum_i sum_j abs(|<phi_i|U|phi_j>|)**2
    """
    sump = 0
    for i in range(4):
        for j in range(4):
            bra_i = qtp.tensor(qtp.ket([i//2], dim=[2]),
                               qtp.ket([i % 2], dim=[3])).dag()
            ket_j = qtp.tensor(qtp.ket([j//2], dim=[2]),
                               qtp.ket([j % 2], dim=[3]))
            p = np.abs((bra_i*U*ket_j).data[0, 0])**2
            sump += p
    sump /= 4  # divide by dimension of comp subspace
    L1 = 1-sump
    return L1


def seepage_from_unitary(U):
    """
    Calculates leakage by summing over all in and output states in the
    computational subspace.
        L1 = 1- sum_i sum_j abs(|<phi_i|U|phi_j>|)**2
    """
    sump = 0
    for i in range(2):
        for j in range(2):
            bra_i = qtp.tensor(qtp.ket([i], dim=[2]),
                               qtp.ket([2], dim=[3])).dag()
            ket_j = qtp.tensor(qtp.ket([j], dim=[2]),
                               qtp.ket([2], dim=[3]))
            p = np.abs((bra_i*U*ket_j).data[0, 0])**2
            sump += p
    sump /= 2  # divide by dimension of comp subspace
    L1 = 1-sump
    return L1


def pro_fid_from_unitary(U, U_target):
    """
    i and j sum over the states that span the full computation subspace.
        F = sum_i sum_j abs(|<phi_i| U^dag * U_t |phi_j>|)**2
    """
    raise NotImplementedError

    # sump = 0
    # for i in range(36):
    #     for j in range(36):
    #         bra_i = pauli_bases_states[i].dag()
    #         ket_j = pauli_bases_states[j]
    #         inner = U.dag()*U_target
    #         p = np.abs((bra_i*inner*ket_j).data[0,0])**2
    #         sump+=p
    #         print(sump, p)

    # F = sump # normalize for dimension
    # return F


def simulate_quantities_of_interest(H_0, tlist, eps_vec,
                                    sim_step: float=0.1e-9,
                                    verbose: bool=True):
    """
    Calculates the quantities of interest from the propagator U

    Args:
        H_0 (Qobj): static hamiltonian, see "coupled_transmons_hamiltonian"
            for the expected form of the Hamiltonian.
        tlist (array): times in s, describes the x component of the
            trajectory to simulate
        eps_vec(array): detuning describes the y-component of the trajectory
            to simulate.

    Returns
        phi_cond (float):   conditional phase (deg)
        L1      (float):    leakage
        L2      (float):    seepage

    # TODO:
        return the Fidelity in the comp subspace with and without correcting
        for phase errors.
    """

    eps_interp = interp1d(tlist, eps_vec, fill_value='extrapolate')

    # function only exists to wrap
    def eps_t(t, args=None):
        return eps_interp(t)

    H_c = n_q0
    H_t = [H_0, [H_c, eps_t]]

    tlist_sim = (np.arange(0, np.max(tlist), sim_step))
    t0 = time.time()
    U_t = qtp.propagator(H_t, tlist_sim)
    t1 = time.time()
    if verbose:
        print('simulation took {:.2f}s'.format(t1-t0))

    U_final = U_t[-1]
    phases = phases_from_unitary(U_final)
    phi_cond = phases[-1]
    L1 = leakage_from_unitary(U_final)
    L2 = seepage_from_unitary(U_final)
    return {'phi_cond': phi_cond, 'L1': L1, 'L2': L2}


class CZ_trajectory(det.Soft_Detector):
    def __init__(self, H_0, fluxlutman):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                required to generate the waveform for the trajectory.
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2']
        self.value_units = ['a.u.', 'deg', '%', '%']
        self.fluxlutman = fluxlutman
        self.H_0 = H_0

    def acquire_data_point(self, **kw):
        tlist = (np.arange(0, self.fluxlutman.cz_length(),
                           1/self.fluxlutman.sampling_rate()))
        if not self.fluxlutman.czd_double_sided():
            f_pulse = wf.martinis_flux_pulse(
                length=self.fluxlutman.cz_length(),
                lambda_2=self.fluxlutman.cz_lambda_2(),
                lambda_3=self.fluxlutman.cz_lambda_3(),
                theta_f=self.fluxlutman.cz_theta_f(),
                f_01_max=self.fluxlutman.cz_freq_01_max(),
                J2=self.fluxlutman.cz_J2(),
                f_interaction=self.fluxlutman.cz_freq_interaction(),
                sampling_rate=self.fluxlutman.sampling_rate(),
                return_unit='f01')
        else:
            f_pulse = self.get_f_pulse_double_sided()

        # extract base frequency from the Hamiltonian
        w_q0 = np.real(self.H_0[1,1])
        eps_vec = f_pulse - w_q0

        qoi = simulate_quantities_of_interest(
            H_0=self.H_0,
            tlist=tlist, eps_vec=eps_vec,
            sim_step=1e-9, verbose=False)

        cost_func_val = abs(qoi['phi_cond']-180) + qoi['L1']*100 * 5
        return cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100

    def get_f_pulse_double_sided(self):
        half_CZ_A = wf.martinis_flux_pulse(
            length=self.fluxlutman.cz_length()*self.fluxlutman.czd_length_ratio(),
            lambda_2=self.fluxlutman.cz_lambda_2(),
            lambda_3=self.fluxlutman.cz_lambda_3(),
            theta_f=self.fluxlutman.cz_theta_f(),
            f_01_max=self.fluxlutman.cz_freq_01_max(),
            J2=self.fluxlutman.cz_J2(),
            # E_c=self.fluxlutman.cz_E_c(),
            f_interaction=self.fluxlutman.cz_freq_interaction(),
            sampling_rate=self.fluxlutman.sampling_rate(),
            return_unit='f01')

        # Generate the second CZ pulse. If the params are np.nan, default
        # to the main parameter
        if not np.isnan(self.fluxlutman.czd_theta_f()):
            d_theta_f = self.fluxlutman.czd_theta_f()
        else:
            d_theta_f = self.fluxlutman.cz_theta_f()

        if not np.isnan(self.fluxlutman.czd_lambda_2()):
            d_lambda_2 = self.fluxlutman.czd_lambda_2()
        else:
            d_lambda_2 = self.fluxlutman.cz_lambda_2()
        if not np.isnan(self.fluxlutman.czd_lambda_3()):
            d_lambda_3 = self.fluxlutman.czd_lambda_3()
        else:
            d_lambda_3 = self.fluxlutman.cz_lambda_3()

        half_CZ_B = wf.martinis_flux_pulse(
            length=self.fluxlutman.cz_length()*(1-self.fluxlutman.czd_length_ratio()),
            lambda_2=d_lambda_2,
            lambda_3=d_lambda_3,
            theta_f=d_theta_f,
            f_01_max=self.fluxlutman.cz_freq_01_max(),
            J2=self.fluxlutman.cz_J2(),
            f_interaction=self.fluxlutman.cz_freq_interaction(),
            sampling_rate=self.fluxlutman.sampling_rate(),
            return_unit='f01')

        # N.B. No amp scaling and offset present
        f_pulse = np.concatenate([half_CZ_A, half_CZ_B])
        return f_pulse
