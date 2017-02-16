"""
This is a translation of Leo's Martinis Pulses simulation.
For those parts that it is possible I will use the conversions
and pulses as defined in "waveform".

The goal of translating is to get understanding of this.
A thing I want to explore if this works is the sensitivity to
parameters, specifically those in the conversion from theta to Volts,
more so than the individual lambda coefficients.
"""
import numpy as np
import qutip as qtp
from pycqed.measurement.waveform_control_CC.waveform import martinis_flux_pulse
import logging

def SetupFluxPulsing(manifold=1, g1=250e6):
    """
    manifold (int) excitation manifold that the calculation operates in
                1-exciation or 2-excitation
    g1  (float) qubit bus coupling in Hz
    """
    global f01_max
    f01_max = 5.94*1e9
    global Fbus
    Fbus = 4.8*1e9
    global Ec
    Ec = 0.367*1e9
    global g
    g = g1*np.sqrt(manifold)


manifold = 2
SetupFluxPulsing(manifold=manifold)

# Note MAR: I think this formula for V0 is wrong
if(manifold == 1):
    #  F_01(V)=(f01_max+Ec)*sqrt(Cos(V/Vo))-Ec;
    # The scaling factor Vo is chosen so that the 01-10 avoided crossing
    # happens at V=1;
    V0 = 1/np.arccos(((Fbus+Ec)/(f01_max+Ec))**2)
else:
    #  F_12(V)=(f01_max+Ec)*sqrt(Cos(V/V0))-2Ec;
    # The scaling factor V0 is chosen so that the 11-02 aV0ided crossing
    # happens at V=1;
    V0 = 1/np.arccos(((Fbus+2*Ec)/(f01_max+Ec))**2)

# estimate 2Xi
TwoXi = (np.sqrt(2)*g)**2*Ec/(f01_max-Fbus)/(f01_max-Fbus-Ec)


def theta_from_V(V):
    """
    Calculates theta from V given the module wide variable definitions.
    This should be replaced with the waveform functions I use for
    pulse generation.
    """

    # Note, using the default parameters
    if(manifold == 1):
        f01 = (f01_max+Ec)*np.sqrt(np.cos(V/V0))-Ec
    else:
        f01 = (f01_max+Ec)*np.sqrt(np.cos(V/V0))-2*Ec

    eps = f01-Fbus  # detuning to bus
    theta = np.arctan2(2*g, eps)
    return theta
    # //print eps, f01


def V_from_theta(theta, manifold):
    eps = 2*g/np.tan(theta)
    if(manifold == 1):
        V = V0*np.acos(((Fbus+eps+Ec)/(f01_max+Ec))**2)
    else:
        V = V0*np.acos(((Fbus+eps+2*Ec)/(f01_max+Ec))**2)
    return V


def eps_from_V(V, manifold):
    # detuning from V
    if(manifold == 1):
        eps = ((f01_max+Ec)*np.sqrt(np.cos(V/V0))-Ec)-Fbus
    else:
        eps = ((f01_max+Ec)*np.sqrt(np.cos(V/V0))-2*Ec)-Fbus
    return eps


def getFlux_pulse(tau_pulse, thetamax,
                  frac,  # numerical overhead?
                  manifold=2,
                  lambda1=1, lambda2=0, lambda3=0, lambda4=0,
                  V_max=1):
    """
    Input args:
        tau_pulse: time pulse in ns
        thetamax: final angle
        frac: numerical overhead?
        manifold: 1 or 2 excitation manifold

    Replace this with waveform
    """
    SetupFluxPulsing(manifold=manifold)
    numt = tau_pulse*frac+1
    ThetaVec = np.zeros(numt)
    x = np.arange(numt)  # hacked in here to get what I think is x
    lambda0 = 1-lambda1

    thetamin = theta_from_V(0, manifold=manifold)
    # this formula seems wrong

    # Martinis version
    dThetavec = lambda0 + lambda1*(1-np.cos(1*2*np.pi*x/tau_pulse))/2 + \
        lambda2*(1-np.cos(2*2*np.pi*x/tau_pulse))/2 // + \
        lambda2*(1-np.cos(2*2*np.pi*x/tau_pulse)) // + \
        lambda3*(1-np.cos(3*2*np.pi*x/tau_pulse)) + \
        lambda4*(1-np.cos(4*2*np.pi*x/tau_pulse))

    # Scaling the dTheta vector
    ThetaVec = thetamin+dThetavec*(thetamax-thetamin)/V_max
    # For debugging only: square pulse
    # ThetaVec=ThetaMax;

    VVec = V_from_theta(ThetaVec, manifold=manifold)
    eps_vec = eps_from_V(VVec, manifold=manifold)

    return eps_vec


def evolveH(tau_pulse, ThetaMax, frac, manifold=2, lambda1=1, lambda2=0):
    """
    Evolves the time-dependent hamiltonian as defined by the detuning vector.

    This is the single most important function of the simulation.
    I have to get this to work and then I'm good.
    """
    # Setup pulsing basics, and import global variables
    SetupFluxPulsing(manifold=manifold)

    # Get the detuning vector
    eps_vec = getFlux_pulse(
        tau_pulse, ThetaMax, frac,
        manifold=manifold, lambda1=lambda1, lambda2=lambda2)
    numT = len(eps_vec)
    dT = tau_pulse/numT

    thisH = np.empty(2, 2)
    thisHm = np.empty(2, 2)
    thisHp = np.empty(2, 2)
    Eye = np.eye(2, 2)

    # make/o/n = (numT+1)  Pop11Vec = NaN
    # setscale/P x, 0, dt, Pop11Vec
    # make/o/n = (numT+1)  PhaseVec = Nan
    # setscale/P x, 0, dt, PhaseVec
    # make/o/n = (numT+1)  ReA11Vec = Nan
    # setscale/P x, 0, dt, ReA11Vec

    # make/o/n = (2)/C PsiT = 0
    Delta = eps_from_V(0, manifold=manifold)
    Em = Delta/2-np.sqrt(g ^ 2+(Delta/2) ^ 2)
    Pnorm = np.sqrt(1+(Em/g) ^ 2)

    # get eigenbasis in OFF state
    PsiT[0] = Em/g
    PsiT[1] = 1
    PsiT /= Pnorm
    # duplicate/o PsiT PsiStart

    # matrixop/o/c  overlap = PsiStart ^ h x PsiT
    # Pop11vec[0] = cabs(overlap[0]) ^ 2
    # PhaseVec[0] = atan2(imag(PsiT[1]), real(PsiT[1]))/pi
    # ReA11Vec[0] = real(PsiT[1])

    for i in range(numT):
        # construct the hamiltonian
        thisH[0][0] = eps_vec[i]
        thisH[0][1] = g
        thisH[1][0] = g
        thisH[1][1] = 0  # Why is there a zero here?

    # Jos Thijssen's trick: see his book
    thisHm = (Eye + (0 + 1j*-1)*thisH/2*dt*2*pi)
    thisHp = (Eye + (0 + 1j*1) * thisH/2*dt*2*pi)
    # matrixop/o/c PsiT = inv(thisHp) x thisHm x PsiT

    # normalize
    # matrixop/o  foo = PsiT ^ h x PsiT
    # PsiT = sqrt(foo[0])

    # matrixop/o/c  overlap = PsiStart ^ h x PsiT
    # Pop11vec[i] = cabs(overlap[0]) ^ 2
    # PhaseVec[i] = atan2(imag(PsiT[1]), real(PsiT[1]))
    # ReA11Vec[i] = real(PsiT[1])

    # phaseunwrap(PhaseVec)
    # PhaseVec = pi

    # make/o/n = (2) latestValues
    # latestValues[0] = 1-Pop11vec[numT]
    # latestValues[1] = PhaseVec[numT]
#
    # return latestValues[1]


print('g: {:.4g} (MHz)'.format(g*1e-6))
print('V0: {:.4g} (V)'.format(V0))
print('Two Xi: {:.4g} (MHz)'.format(TwoXi*1e-6))

print('Theta0: {:.4g} (rad)'.format(theta_from_V(0)))
print('Theta90?: {:.4g} (rad)'.format(theta_from_V(V0)))
# NB: the values for theta0 and theta90 do not make sense using Leo's formulae

# print TwoXi

# Below here is helper functions for my own model using qutip


def ket_to_phase(states):
    phases = np.zeros(len(states))
    for i in range(len(states)):
        phases[i] = np.arctan2(float(np.imag(states[i][0])),
                               float(np.real(states[i][0])))
    return phases


def simulate_CZ_trajectory(length, lambda_coeffs, theta_f,
                           f_01_max,
                           f_interaction,
                           J_2,
                           E_c,
                           dac_flux_coefficient=1,
                           asymmetry=0,
                           sampling_rate=2e9, return_all=False,
                           verbose=False):
    """
    Input args:
        tp: pulse length
    Returns:
        picked_up_phase in degree
        leakage         population leaked to the 02 state
    N.B. the function only returns the quantities below if return_all is True
        res1:           vector of qutip results for 1 excitation
        res2:           vector of qutip results for 1 excitation
        eps_vec :       vector of detunings as function of time
        tlist:          vector of times used in the simulation
    """

    psi0 = qtp.basis(2, 0)

    Hx = qtp.sigmax()*2*np.pi  # so that freqs are real and not radial
    Hz = qtp.sigmaz()*2*np.pi  # so that freqs are real and not radial

    def J1_t(t, args=None):
        # if there is only one excitation, there is no interaction
        return 0

    def J_2_t(t, args=None):
        return J_2

    tlist = (np.arange(0, length, 1/sampling_rate))

    f_pulse = martinis_flux_pulse(length, lambda_coeffs, theta_f,
                                  f01_max, f_interaction, J_2,
                                  Ec, dac_flux_coefficient,
                                  return_unit='eps',
                                  sampling_rate=sampling_rate)

    eps_vec = np.ones(len(tlist))*f_pulse[0]
    eps_vec[0:len(f_pulse)] = f_pulse

    # function used for qutip simulation
    def eps_t(t, args=None):
        idx = np.argmin(abs(tlist-t))
        return float(eps_vec[idx])

    H1_t = [[Hx, J1_t], [Hz, eps_t]]
    H2_t = [[Hx, J_2_t], [Hz, eps_t]]
    try:
        res1 = qtp.mesolve(H1_t, psi0, tlist, [], [])  # ,progress_bar=False )
        res2 = qtp.mesolve(H2_t, psi0, tlist, [], [])  # ,progress_bar=False )
    except Exception as e:
        logging.warning(e)
        # This is exception handling to be able to use this in a loop
        return 0, 0

    phases1 = (ket_to_phase(res1.states) % (2*np.pi))/(2*np.pi)*360
    phases2 = (ket_to_phase(res2.states) % (2*np.pi))/(2*np.pi)*360
    phase_diff_deg = (phases2 - phases1) % 360
    leakage_vec = 1-(qtp.expect(qtp.sigmaz(), res2.states[:])+1)/2

    leakage = leakage_vec[-1]
    picked_up_phase = phase_diff_deg[-1]
    if verbose:
        print('Picked up phase: {:.1f} (deg) \nLeakage: \t {:.3f} (%)'.format(
            picked_up_phase, leakage*100))
    if return_all:
        return picked_up_phase, leakage, res1, res2, eps_vec, tlist
    else:
        return picked_up_phase, leakage
