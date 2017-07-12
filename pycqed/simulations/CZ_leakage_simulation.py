"""
This is a translation of Leo's Martinis Pulses simulation.
For those parts that it pulseshape comes from the waveform module.
"""
import numpy as np
import qutip as qtp
from pycqed.measurement.waveform_control_CC.waveform import \
    martinis_flux_pulse_v2 as martinis_flux_pulse
import logging


def ket_to_phase(states):
    """
    Returns the absolute phase picked up by the first eigenstate in the Z
    basis. Note that this phase calculation is specific due to a trick
    used in this model where two different scenarios of the same state are
    compared.
    """
    phases = np.zeros(len(states))
    for i in range(len(states)):
        phase_a = np.arctan2(float(np.imag(states[i][0])),
                             float(np.real(states[i][0])))
        phases[i] = phase_a

    return phases


def simulate_CZ_trajectory(length, lambda_2, lambda_3, theta_f,
                           f_01_max,
                           f_interaction,
                           J2,
                           E_c,
                           V_per_phi0=1,
                           dac_flux_coefficient=None,
                           asymmetry=0,
                           sampling_rate=2e9, return_all=False,
                           verbose=False):
    """
    Input args:
        length               (float) :parameter for waveform (s)
        lambda_2             (float): parameter for waveform
        lambda_3             (float): parameter for waveform
        theta_f              (float): parameter for waveform (deg)
        f_01_max             (float): parameter for waveform (Hz)
        f_interaction        (float): parameter for waveform (Hz)
        J2                   (float): parameter for waveform (Hz)
        E_c                  (float): parameter for waveform (Hz)
        V_per_phi0           (float): parameter for waveform (V)
        asymmetry            (float): parameter for waveform
        sampling_rate        (float): sampling rate used in simulation (Hz)
        verbose              (bool) : enables optional print statements
        return_all           (bool) : if True returns internal params of the
                                      simulation (see below).
    Returns:
        picked_up_phase in degree
        leakage         population leaked to the 02 state
    N.B. the function only returns the quantities below if return_all is True
        res1:           vector of qutip results for 1 excitation
        res2:           vector of qutip results for 1 excitation
        eps_vec :       vector of detunings as function of time
        tlist:          vector of times used in the simulation
    """
    if dac_flux_coefficient is not None:
        loggin.warning('dac_flux_coefficient deprecated. Please use the '
                       'physically meaningful V_per_phi0 instead.')
        V_per_phi0 = np.pi/dac_flux_coefficient

    Hx = qtp.sigmax()*2*np.pi  # so that freqs are real and not radial
    Hz = qtp.sigmaz()*2*np.pi  # so that freqs are real and not radial

    def J1_t(t, args=None):
        # if there is only one excitation, there is no interaction
        return 0

    def J2_t(t, args=None):
        return J2

    tlist = (np.arange(0, length, 1/sampling_rate))

    f_pulse = martinis_flux_pulse(
        length, lambda_2=lambda_2, lambda_3=lambda_3,
        theta_f=theta_f, f_01_max=f_01_max, E_c=E_c,
        V_per_phi0=V_per_phi0,
        f_interaction=f_interaction, J2=J2,
        return_unit='eps',
        sampling_rate=sampling_rate)

    eps_vec = f_pulse

    # function used for qutip simulation
    def eps_t(t, args=None):
        idx = np.argmin(abs(tlist-t))
        return float(eps_vec[idx])

    H1_t = [[Hx, J1_t], [Hz, eps_t]]
    H2_t = [[Hx, J2_t], [Hz, eps_t]]
    try:
        psi0 = qtp.basis(2, 0)
        res1 = qtp.mesolve(H1_t, psi0, tlist, [], [])  # ,progress_bar=False )
        res2 = qtp.mesolve(H2_t, psi0, tlist, [], [])  # ,progress_bar=False )
    except Exception as e:
        logging.warning(e)
        # This is exception handling to be able to use this in a loop
        return 0, 0

    phases1 = (ket_to_phase(res1.states) % (2*np.pi))/(2*np.pi)*360
    phases2 = (ket_to_phase(res2.states) % (2*np.pi))/(2*np.pi)*360
    phase_diff_deg = (phases2 - phases1) % 360
    leakage_vec = np.zeros(len(res2.states))
    for i in range(len(res2.states)):
        leakage_vec[i] = 1-(psi0.dag()*res2.states[i]).norm()

    leakage = leakage_vec[-1]
    picked_up_phase = phase_diff_deg[-1]
    if verbose:
        print('Picked up phase: {:.1f} (deg) \nLeakage: \t {:.3f} (%)'.format(
            picked_up_phase, leakage*100))
    if return_all:
        return picked_up_phase, leakage, res1, res2, eps_vec, tlist
    else:
        return picked_up_phase, leakage
