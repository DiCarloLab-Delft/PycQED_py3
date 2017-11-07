import numpy as np
from os.path import join, dirname
import openql.openql as ql
from openql.openql import Program, Kernel, Platform

base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)


def single_flux_pulse_seq(qubit_indices: tuple,
                          platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="single_flux_pulse_seq",
                nqubits=platf.get_qubit_number(),
                p=platf)
    print(platf_cfg)
    k = Kernel("main", p=platf)
    for idx in qubit_indices:
        k.prepz(idx)  # to ensure enough separation in timing
    for i in range(7):
        k.gate('CW_00', i)
    k.gate('fl_cw_02', qubit_indices[0], qubit_indices[1])
    p.add_kernel(k)
    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def flux_staircase_seq(platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="flux_staircase_seq",
                nqubits=platf.get_qubit_number(),
                p=platf)
    print(platf_cfg)
    k = Kernel("main", p=platf)
    for i in range(1):
        k.prepz(i)  # to ensure enough separation in timing
    for i in range(1):
        k.gate('CW_00', i)
    k.gate('CW_00', 6)
    for cw in range(8):
        k.gate('fl_cw_{:02d}'.format(cw), 2, 0)
        k.gate('fl_cw_{:02d}'.format(cw), 3, 1)
        k.gate("wait", [0, 1, 2, 3], 200) # because scheduling is wrong.

    p.add_kernel(k)
    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p
