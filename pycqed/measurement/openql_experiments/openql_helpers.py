"""

"""
import re
import numpy as np
import json
import matplotlib.pyplot as plt
from pycqed.measurement.openql_experiments.get_qisa_tqisa_timing_tuples import (
    get_qisa_tqisa_timing_tuples
)


from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches


def infer_tqisa_filename(qisa_fn: str):
    """
    Get's the expected tqisa filename based on the qisa filename.
    """
    return qisa_fn[:-4]+'tqisa'


def get_start_time(line: str):
    """
    Takes in a line of a tqisa file and returns the starting time.
    This corrects for the timing in the "bs" instruction.

    Time is in units of clocks.

    Example tqsia line:
        "   76014:    bs 4    cw_03 s0 | cw_05 s2"
        -> would return 76018
    """

    start_time = int(line.split(':')[0])
    if 'bs' in line:
        # Takes the second character after "bs"
        pre_interval = int(line.split('bs')[1][1])
        start_time += pre_interval

    return start_time


def get_register_map(qisa_fn: str):
    """
    Extracts the map for the smis and smit qubit registers from a qisa file
    """
    reg_map = {}
    with open(qisa_fn, 'r') as q_file:
        linenum = 0
        for line in q_file:
            if 'start' in line:
                break
            if 'smis' in line or 'smit' in line:
                reg_key = line[5:line.find(',')]
                start_reg_idx = line.find('{')
                reg_val = (line[start_reg_idx:].strip())
                reg_map[reg_key] = eval(reg_val)
    return reg_map


def split_instr_to_op_targ(instr: str, reg_map: dict):
    """
    Takes part of an instruction and splits it into a tuple of
    codeword, target

    e.g.:
        "cw_03 s2" -> "cw_03", {2}
    """
    cw, sreg = instr.split(' ')
    target_qubits = reg_map[sreg]
    return (cw, target_qubits)


def get_timetuples(qisa_fn: str):
    """


    Returns time tuples of the form
        (start_time, operation, target_qubits)
    """
    reg_map = get_register_map(qisa_fn)

    tqisa_fn = infer_tqisa_filename(qisa_fn)
    time_tuples = []
    with open(tqisa_fn, 'r') as tq_file:
        linenum = 0
        for line in tq_file:
            linenum += 1
            # Get instruction line
            if re.search(r"bs", line):
                # Get the timing number
                start_time = get_start_time(line)
                # Get the instr
                instr = re.split(r'bs ', line)[1][1:]
                # We now parse whether there is a | character
                if '|' in line:
                    multi_instr = re.split(r'\s\|\s', instr)
                else:
                    multi_instr = [instr]
                for instr in multi_instr:
                    instr = instr.strip()
                    op, targ = split_instr_to_op_targ(instr, reg_map)
                    result = (start_time, op, targ)
                    time_tuples.append(result)

    return time_tuples


def find_operation_idx_in_time_tuples(time_tuples, target_op: str):
    target_indices = []
    for i, tt in enumerate(time_tuples):
        t_start, cw, targets = tt
        if target_op in cw:
            target_indices.append(i)
    return (target_indices)


def split_time_tuples_on_operation(time_tuples, split_op: str):
    indices = find_operation_idx_in_time_tuples(time_tuples, split_op)

    start_indices = [0]+indices[:-1]
    stop_indices = indices

    split_tt = [time_tuples[start_indices[i]+1:stop_indices[i]+1] for
                i in range(len(start_indices))]
    return split_tt


def substract_time_offset(time_tuples, op_str: str='cw'):
    """
    """
    for tt in time_tuples:
        t_start, cw, targets = tt
        if op_str in cw:
            t_ref = t_start
            break
    corr_time_tuples = []
    for tt in time_tuples:
        t_start, cw, targets = tt
        corr_time_tuples.append((t_start-t_ref, cw, targets))
    return corr_time_tuples


def plot_time_tuples(time_tuples, ax=None, time_unit='s',
                     mw_duration=20e-9, fl_duration=240e-9,
                     ro_duration=1e-6, ypos=None):
    if ax is None:
        f, ax = plt.subplots()

    mw_patch = mpatches.Patch(color='C0', label='Microwave')
    fl_patch = mpatches.Patch(color='C1', label='Flux')
    ro_patch = mpatches.Patch(color='C4', label='Measurement')

    if time_unit == 's':
        clock_cycle = 20e-9
    elif time_unit == 'clocks':
        clock_cycle = 1
    else:
        raise ValueError()

    for i, tt in enumerate(time_tuples):
        t_start, cw, targets = tt

        if 'meas' in cw:
            c = 'C4'
            width = ro_duration
        elif isinstance((list(targets)[0]), tuple):
            # Flux pulses
            c = 'C1'
            width = fl_duration

        else:
            # Microwave pulses
            c = 'C0'
            width = mw_duration

        if 'prepz' not in cw:
            for q in targets:
                if isinstance(q, tuple):
                    for qi in q:
                        ypos = qi if ypos is None else ypos
                        ax.barh(ypos, width=width, left=t_start*clock_cycle,
                                height=0.6, align='center', color=c, alpha=.8)
                else:
                    # N.B. alpha is not 1 so that overlapping operations are easily
                    # spotted.
                    ypos = qi if ypos is None else ypos
                    ax.barh(ypos, width=width, left=t_start*clock_cycle,
                            height=0.6, align='center', color=c, alpha=.8)

    ax.legend(handles=[mw_patch, fl_patch, ro_patch], loc=(1.05, 0.5))
    set_xlabel(ax, 'Time', time_unit)
    set_ylabel(ax, 'Qubit', '#')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return ax


def plot_time_tuples_split(time_tuples, ax=None, time_unit='s',
                           mw_duration=20e-9, fl_duration=240e-9,
                           ro_duration=1e-6, split_op: str='meas',
                           align_op: str='cw'):
    ttuple_groups = split_time_tuples_on_operation(time_tuples,
                                                   split_op=split_op)
    corr_ttuple_groups = [substract_time_offset(tt, op_str=align_op) for
                          tt in ttuple_groups]

    for i, corr_tt in enumerate(corr_ttuple_groups):
        if ax is None:
            f, ax = plt.subplots()
        plot_time_tuples(corr_tt, ax=ax, time_unit=time_unit,
                         mw_duration=mw_duration, fl_duration=fl_duration,
                         ro_duration=ro_duration, ypos=i)
    ax.invert_yaxis()
    set_ylabel(ax, "Kernel idx", "#")
