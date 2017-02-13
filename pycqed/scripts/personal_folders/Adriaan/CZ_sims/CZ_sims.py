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


def V_from_theta(theta):
    eps = 2*g/np.tan(theta)
    if(manifold == 1):
        V = V0*np.acos(((Fbus+eps+Ec)/(f01_max+Ec))**2)
    else:
        V = V0*np.acos(((Fbus+eps+2*Ec)/(f01_max+Ec))**2)
    return V


def eps_from_V(V):
    if(manifold == 1):
        eps = ((f01_max+Ec)*np.sqrt(np.cos(V/V0))-Ec)-Fbus
    else:
        eps = ((f01_max+Ec)*np.sqrt(np.cos(V/V0))-2*Ec)-Fbus
    return eps


def getFlux_pulse(tp, thetamax,
                  frac,  # numerical overhead?
                  manifold=2,
                  lambda1=1, lambda2=0, lambda3=0):
    """
    Replace this with waveform
    """
    SetupFluxPulsing(manifold=manifold)
    numt = tp*frac+1
    ThetaVec = np.zeros(numt)
    lambda0 = 1-lambda1
    Thetamin = theta_from_V(0, manifold=manifold)
    # this formula seems wrong

    # Martinis version
    dThetavec = lambda0+lambda1*(1-cos(1*2*pi*x/tp))/2+lambda2*(1-cos(2*2*pi*x/tp))/2 // + lambda2*(
        1-cos(2*2*pi*x/tp)) // + lambda3*(1-cos(3*2*pi*x/tp)) + lambda4*(1-cos(4*2*pi*x/tp))
    # My version
    # dThetavec=lambda0+lambda1*sin(2*pi*x/tp/2)//+lambda2*sin(2*pi*x/tp*3/2)+lambda3*sin(2*pi*x/tp*5/2)

    # Scaling the dTheta vector
    # Wavestats/q dThetavec
    Thetavec = ThetaMin+dThetaVec*(ThetaMax-ThetaMin)/V_max

    # For debugging only: square pulse
    # ThetaVec=ThetaMax;

    # duplicate/o ThetaVec Vvec
    # duplicate/o ThetaVec DetVec
    # Vvec = Nan
    # Detvec = Nan

    # Vvec[] = getVfromTheta(ThetaVec[p], manifold=manifold)


def evolveH(tp, ThetaMax, frac, manifold=2, lambda1=1, lambda2=0):
    """
    Evolves the time-dependent hamiltonian as defined by the detuning vector.

    This is the single most important function of the simulation.
    I have to get this to work and then I'm good.
    """
    # Setup pulsing basics, and import global variables
    SetupFluxPulsing(manifold=manifold)

    # Get the detuning vector
    eps_vec = getFluxPulse(
        tp, ThetaMax, frac, manifold=manifold, lambda1=lambda1, lambda2=lambda2)
    numT = len(eps_vec)
    dT = taup/numT

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
    Delta = getDetFromV(0, manifold=manifold)
    Em = Delta/2-sqrt(g ^ 2+(Delta/2) ^ 2)
    # Pnorm = sqrt(1+(Em/g) ^ 2)

    # get eigenbasis in OFF state
    # PsiT[0] = Em/g
    # PsiT[1] = 1
    # PsiT /= Pnorm
    # duplicate/o PsiT PsiStart

    # matrixop/o/c  overlap = PsiStart ^ h x PsiT
    # Pop11vec[0] = cabs(overlap[0]) ^ 2
    # PhaseVec[0] = atan2(imag(PsiT[1]), real(PsiT[1]))/pi
    # ReA11Vec[0] = real(PsiT[1])

    # variable i = 1
    for i in range(numT):
        # construct the hamiltonian
        thisH[0][0] = DetVec[i-1]
        thisH[0][1] = g
        thisH[1][0] = g
        thisH[1][1] = 0 # Why is there a zero here?

        # Jos Thijssen's trick: see his book
        # thisHm = (Eye+cmplx(0, -1)*thisH/2*dt*2*pi)
        # thisHp = (Eye+cmplx(0, 1) * thisH/2*dt*2*pi)
        # matrixop/o/c PsiT = inv(thisHp) x thisHm x PsiT

        # normalize
        # matrixop/o  foo = PsiT ^ h x PsiT
        # PsiT = sqrt(foo[0])

        # matrixop/o/c  overlap = PsiStart ^ h x PsiT
        # Pop11vec[i] = cabs(overlap[0]) ^ 2
        # PhaseVec[i] = atan2(imag(PsiT[1]), real(PsiT[1]))
        # ReA11Vec[i] = real(PsiT[1])
x
    phaseunwrap(PhaseVec)
    # PhaseVec = pi

    # make/o/n = (2) latestValues
    # latestValues[0] = 1-Pop11vec[numT]
    # latestValues[1] = PhaseVec[numT]

    return latestValues[1]


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
        phases[i] = np.arctan2(float(np.imag(states[i][1])),
                               float(np.real(states[i][1])))
    return phases
