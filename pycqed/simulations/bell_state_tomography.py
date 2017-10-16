import quantumsim
import numpy as np


def bell_circuit(t1_flux: int=np.inf, t2_flux: int=np.inf,
                 t1_idle: int=np.inf, t2_idle: int=np.inf,
                 CZ_duration: int=240,
                 CZ_buffer: int=40,
                 mw_duration: int=40,
                 target_bell: int=0,
                 error_conditional_phase: float=0):
    '''
    Construct a quantumsim circuit that makes one of the four Bell states
    using a CZ gate.
    "fluxing qubit" refers to the qubit that is moved in frequency during the
    CZ pulse.
    "idle qubit" refers to the qubit that stays at its sweet spot during the
    CZ pulse.
    The basis used is (idle qubit, fluxing qubit).

    Args:
        t1_flux, t2_flux (int):
            T1 and T2echo for the fluxing qubit at its sweet spot (in ns).
        t1_idle, t2_idle (int):
            T1 and T2echo for the idle qubit at its sweet spot (in ns).
        CZ_duration (int):
            Duration of the CZ gate in nanoseconds.
        CZ_buffer (int):
            Buffer after the CZ gate in nanoseconds.
        mw_duration (int):
            Duration microwave gates in nanoseconds.
        target_bell (int):
            Target bell state for the circuit.
            0: |00> - |11>
            1: |00> + |11>
            2: |01> - |10>
            3: |01> + |10>
        error_conditional_phase (float):
            Phase error on the conditional phase of the CZ gate (in deg).

    Returns
        c: Quantumsim circuit.
    '''
    phase_error = np.deg2rad(error_conditional_phase)

    c = quantumsim.circuit.Circuit("Bell state generation")
    qb0 = quantumsim.circuit.Qubit('flux', t1_flux, t2_flux)
    qb1 = quantumsim.circuit.Qubit('idle', t1_idle, t2_idle)
    c.add_qubit(qb0)
    c.add_qubit(qb1)

    if target_bell in [0, 2]:
        angle_0 = np.pi/2
    else:
        angle_0 = -np.pi/2
    if target_bell in [0, 1]:
        angle_1 = np.pi/2
    else:
        angle_1 = -np.pi/2

    t = mw_duration/2

    gate1 = quantumsim.circuit.RotateY('flux', angle=angle_0, time=t)
    gate1.label = '$G_1$'
    c.add_gate(gate1)
    # gates are simultaneous
    # t += mw_duration/2
    # t += mw_duration/2

    gate2 = quantumsim.circuit.RotateY('idle', angle=angle_1, time=t)
    gate2.label = '$G_2$'
    c.add_gate(gate2)

    t += mw_duration/2
    t += CZ_duration/2

    c.add_gate(quantumsim.circuit.CPhaseRotation('flux', 'idle',
                                                 angle=np.pi + phase_error/2,
                                                 time=t))

    if phase_error != 0:
        c.add_gate(quantumsim.circuit.RotateZ('flux', angle=phase_error,
                                              time=t+0.01))

    t += CZ_duration/2

    z_gate = c.add_gate(quantumsim.circuit.RotateZ('flux', time=t, angle=0))
    z_gate.label = '$R_Z^{\\theta}$'
    z_gate = c.add_gate(quantumsim.circuit.RotateZ('idle', time=t, angle=0))
    z_gate.label = '$R_Z^{\\theta}$'

    t += CZ_buffer
    t += mw_duration/2

    rec_gate = c.add_gate(quantumsim.circuit.RotateY('idle', angle=-np.pi/2,
                                                     time=t))
    rec_gate.label = '$Y^{-90}$'
    t += mw_duration/2

    c.add_waiting_gates(tmin=0, tmax=t)
    c.order()

    return c


def get_phase_error_rms(fmax: float, fint: float, E_c: float,
                        t_run: float=1e-6, t_averaging: float=1.0,
                        t_gate: float=240e-9, A_flux_noise: float=6e-6):
    '''
    Calculate rms of phase error (in deg) due to flux noise at the
    interaction point of a CZ gate.

    Args:
        fmax (float):
            Sweet spot frequency of the qubit in Hz.
        fint (float):
            Frequency of the qubit at the interaction point of the CZ gate
            in Hz.
        E_c (float):
            Charging energy of the qubit in Hz.
        t_run (float):
            Duration of a single run of the circuit in s.
        t_averaging (float):
            Averaging time of the whole experiment in s.
        t_gate (float):
            Duration of the CZ gate in s.
        A_flux_noise (float):
            1/f noise magnitude.

    Returns:
        phase_rms: rms of the conditional phase error in deg.
    '''
    cosine = ((fint + E_c) / (fmax + E_c))**2
    beta = np.pi / 2 * (fint + E_c) * np.sqrt(1 - cosine**2) / cosine
    return (A_flux_noise * np.sqrt(np.log(t_averaging / t_run)) * t_gate *
            360 * beta)
