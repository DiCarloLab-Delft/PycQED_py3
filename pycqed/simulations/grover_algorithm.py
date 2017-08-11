import quantumsim as qsim
import numpy as np
import qutip


def grover_circuit(base_t1_flux: int=np.inf, base_t2_flux: int=np.inf,
                   detun_t1_flux: int=np.inf, detun_t2_flux: int=np.inf,
                   t1_idle: int=np.inf, t2_idle: int=np.inf,
                   CZ_duration: int=240,
                   CZ_buffer: int=40,
                   mw_duration: int=40,
                   err_flux: float=0, err_idle: float=0,
                   err_conditional: float=0,
                   flux_noise_CZ_phase: float=0,
                   omega: int=0):
    '''
    Construct a quantumsim circuit that runs Grover's algorithm for the oracle
    state given by omega. The circuit uses VariableDecoherenceQubits.
    "fluxing qubit" refers to the qubit that is moved in frequency during the
    CZ pulse.
    "idle qubit" refers to the qubit that stays at its sweet spot during the
    CZ pulse.
    The basis used is (idle qubit, fluxing qubit)

    Args:
        base_t1_flux, base_t2_flux (int):
                T1 and T2echo for the fluxing qubit at its sweet spot
                (in ns).
        detun_t1_flux, detun_t2_flux (int):
                T1 and T2echo for the fluxing qubit when it is detuned to the
                interaction point (in ns).
        t1_idle, t2_idle (int):
                T1 and T2echo for the idle qubit at its sweet spot (in ns).
        CZ_duration (int):
                Duration of the CZ gate in nanoseconds.
        CZ_buffer (int):
                Buffer after the CZ gate in nanoseconds.
        mw_duration (int):
                Duration microwave gates in nanoseconds.
        err_flux, err_idle (float):
                Single-qubit phase errors of the second CZ pulse in degrees.
        err_conditional (float):
                Conditional phase error of the second CZ pulse in degrees.
        flux_noise_CZ_phase (float):
                Error on the conditional phase added by flux noise.
        omega (int):
                Oracle for Grover's algorithm. The binary representation of
                this integer represents the expected output state of the
                circuit.

    Returns
        c: Quantumsim circuit.
    '''
    c = qsim.circuit.Circuit('Grover')
    q_flux = qsim.circuit.VariableDecoherenceQubit(
        'q_flux', base_t1=base_t1_flux, base_t2=base_t2_flux,
        t1s=[(mw_duration, mw_duration + CZ_duration, detun_t1_flux),
             (2*mw_duration + CZ_buffer + CZ_duration,
              2*mw_duration + CZ_buffer + 2*CZ_duration, detun_t1_flux)],
        t2s=[(mw_duration, mw_duration + CZ_duration, detun_t2_flux),
             (2*mw_duration + CZ_buffer + CZ_duration,
              2*mw_duration + CZ_buffer + 2*CZ_duration, detun_t2_flux)])
    q_idle = qsim.circuit.Qubit('q_idle', t1=t1_idle, t2=t2_idle)

    c.add_qubit(q_idle)
    c.add_qubit(q_flux)

    if omega == 0:
        flux_sign = 1
        idle_sign = 1
    elif omega == 1:
        flux_sign = 1
        idle_sign = -1
    elif omega == 2:
        flux_sign = -1
        idle_sign = 1
    elif omega == 3:
        flux_sign = -1
        idle_sign = -1

    t = mw_duration/2

    c.add_gate(qsim.circuit.RotateY('q_flux',
                                    angle=flux_sign * np.pi/2,
                                    time=t))
    c.add_gate(qsim.circuit.RotateY('q_idle',
                                    angle=idle_sign * np.pi/2,
                                    time=t))

    t += mw_duration/2
    t += CZ_duration/2

    c.add_gate(qsim.circuit.CPhaseRotation(
        'q_flux', 'q_idle',
        angle=np.pi + np.deg2rad(flux_noise_CZ_phase/2),
        time=t))
    t += 0.01
    c.add_gate(qsim.circuit.RotateZ(
        'q_flux',
        angle=np.deg2rad(flux_noise_CZ_phase),
        time=t))

    t += CZ_duration/2
    t += CZ_buffer
    t += mw_duration/2

    c.add_gate(qsim.circuit.RotateY('q_flux', angle=np.pi/2, time=t))
    c.add_gate(qsim.circuit.RotateY('q_idle', angle=np.pi/2, time=t))

    t += mw_duration/2
    t += CZ_duration/2

    c.add_gate(qsim.circuit.CPhaseRotation(
        'q_flux', 'q_idle',
        angle=(np.pi + np.deg2rad(err_conditional + flux_noise_CZ_phase)),
        time=t))

    t += CZ_duration/2

    if err_flux != 0:
        c.add_gate(qsim.circuit.RotateZ(
            'q_flux',
            angle=np.deg2rad(err_flux + flux_noise_CZ_phase),
            time=t))
    if err_idle != 0:
        c.add_gate(qsim.circuit.RotateZ(
            'q_idle',
            angle=np.deg2rad(err_idle),
            time=t))

    t += CZ_buffer
    t += mw_duration/2

    c.add_gate(qsim.circuit.RotateY('q_flux', angle=np.pi/2, time=t))
    c.add_gate(qsim.circuit.RotateY('q_idle', angle=np.pi/2, time=t))

    c.add_waiting_gates(tmin=0,
                        tmax=3*mw_duration + 2*CZ_duration + 2*CZ_buffer)
    c.order()
    return c


def run_circuit(c):
    '''
    Run circuit c on qubits initialized in the 0^n state and return the
    density matrix of the output state as qutip Qobj.
    '''
    state = qsim.sparsedm.SparseDM(c.get_qubit_names())
    c.apply_to(state)
    return qutip.Qobj(state.full_dm.to_array(), dims=[[2, 2], [2, 2]])
