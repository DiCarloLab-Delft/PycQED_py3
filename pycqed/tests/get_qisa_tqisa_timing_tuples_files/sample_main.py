# EXAMPLE PYTHON FILE
import openql.openql as ql
from openql.openql import Program, Kernel
import quantumInfinity as qi
import numpy as np 

def two_qubit_AllXY(q0: int, q1: int, 
                    sequence_type='sequential',
                    replace_q1_pulses_X180: bool=False,
                    double_points: bool=True):
    platf = qi.platform
    p = Program(pname="two_qubit_AllXY", nqubits=platf.get_qubit_number(),
                p=platf)

    pulse_combinations = [ ['rx90', 'rx90'],
                          ['ry90', 'ry90']]

    pulse_combinations_tiled = pulse_combinations + pulse_combinations
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]

    if replace_q1_pulses_X180:
        pulse_combinations_q1 = ['rx180' for val in pulse_combinations]

    pulse_combinations_q0 = pulse_combinations
    pulse_combinations_q1 = pulse_combinations_tiled

    i = 0
    for pulse_comb_q0, pulse_comb_q1 in zip(pulse_combinations_q0,
                                            pulse_combinations_q1):
        i += 1
        k = Kernel('AllXY_{}'.format(i), p=platf)
        k.prepz(q0)
        k.prepz(q1)
        k.prepz(q0)
        k.gate(pulse_comb_q0[0], q0)
        k.gate('i', q1)
        k.gate('fl_cw_01',2,0)
        k.gate(pulse_comb_q0[1], q0)
        k.gate('i', q1)
        k.gate('i', q0)
        k.prepz(q1)
        k.gate(pulse_comb_q1[0], q1)
        k.gate('i', q0)
        k.gate(pulse_comb_q1[1], q1)
        k.gate('fl_cw_01',2,0)
        k.measure(q0)
        k.measure(q1)
        p.add_kernel(k)
        
    sweep_points = np.arange(i)
    p.set_sweep_points(sweep_points, len(sweep_points))
    p.compile()
    return p
  
two_qubit_AllXY(0, 1, double_points=True)