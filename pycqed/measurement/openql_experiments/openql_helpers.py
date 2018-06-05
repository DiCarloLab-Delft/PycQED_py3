"""

"""
import re
import numpy as np
import json
from shutil import copyfile
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from pycqed.utilities.general import is_more_rencent

def clocks_to_s(time, clock_cycle=20e-9):
    """
    Converts a time in clocks to a time in s
    """
    return time*clock_cycle

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
        (start_time, operation, target_qubits, line_nr)
    """
    reg_map = get_register_map(qisa_fn)

    tqisa_fn = infer_tqisa_filename(qisa_fn)
    time_tuples = []
    with open(tqisa_fn, 'r') as tq_file:
        for i, line in enumerate(tq_file):
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
                    result = (start_time, op, targ, i)
                    time_tuples.append(result)

    return time_tuples


def find_operation_idx_in_time_tuples(time_tuples, target_op: str):
    target_indices = []
    for i, tt in enumerate(time_tuples):
        t_start, cw, targets, linenum = tt
        if target_op in cw:
            target_indices.append(i)
    return (target_indices)

def get_operation_tuples(time_tuples: list, target_op:str):
    """
    Returns a list of tuples that perform a specific operation

    args:
        time_tuples             : list of time tuples
        target_op               : operation to searc for
    returns
        time_tuples_op          : time_tuples containing target_op
    """
    op_indices = find_operation_idx_in_time_tuples(time_tuples,
                                               target_op=target_op)

    time_tuples_op = []
    for op_idx in op_indices:
        time_tuples_op.append(time_tuples[op_idx])
    return time_tuples_op



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
        t_start, cw, targets, linenum = tt
        if op_str in cw:
            t_ref = t_start
            break
    corr_time_tuples = []
    for tt in time_tuples:
        t_start, cw, targets, linenum = tt
        corr_time_tuples.append((t_start-t_ref, cw, targets, linenum))
    return corr_time_tuples


#############################################################################
# Plotting
#############################################################################

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
        t_start, cw, targets, linenum = tt

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
                        ypos_i = qi if ypos is None else ypos
                        ax.barh(ypos_i, width=width, left=t_start*clock_cycle,
                                height=0.6, align='center', color=c, alpha=.8)
                else:
                    # N.B. alpha is not 1 so that overlapping operations are easily
                    # spotted.
                    ypos_i = q if ypos is None else ypos
                    ax.barh(ypos_i, width=width, left=t_start*clock_cycle,
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

    return ax



#############################################################################
# File modifications
#############################################################################

def flux_pulse_replacement(qisa_fn: str):
    """
    args:
        qisa_fn : file in which to replace flux pulses

    returns:
        mod_qisa_fn : filename of the modified qisa file
        grouped_flux_tuples: : time tuples of the flux pulses grouped

    ---------------------------------------------------------------------------
    Modifies a file for use with non-codeword based flux pulses.
    Does this in the following steps

        1. create a copy of the file
        2. extract locations of pulses from source file
        3. replace content of files
        4. return filename of modified qisa file and time tuples
            grouped per kernel.

    """


    ttuple = get_timetuples(qisa_fn)
    grouped_timetuples = split_time_tuples_on_operation(ttuple, 'meas')

    grouped_fl_tuples = []
    for i, tt in enumerate(grouped_timetuples):
        fl_time_tuples = substract_time_offset(get_operation_tuples(tt, 'fl'))
        grouped_fl_tuples.append(fl_time_tuples)



    with open(qisa_fn, 'r') as source_qisa_file:
        lines = source_qisa_file.readlines()

    for k_idx, fl_time_tuples in enumerate(grouped_fl_tuples):
        for i, time_tuple in enumerate(fl_time_tuples):
            time, cw, target, line_nr = time_tuple

            l = lines[line_nr]
            if i == 0:
                new_l = l.replace(cw, 'fl_cw_{:02d}'.format(k_idx+1))
            else:
                # cw 00 is a dummy pulse that should not trigger the AWG8
                new_l = l.replace(cw, 'fl_cw_00')
            lines[line_nr] = new_l




    mod_qisa_fn = qisa_fn[:-5]+'_mod.qisa'
    with open(mod_qisa_fn, 'w') as mod_qisa_file:
        for l in lines:
            mod_qisa_file.write(l)

    return mod_qisa_fn, grouped_fl_tuples


def check_recompilation_needed(program_fn: str, platf_cfg: str,
                               recompile=True):
    """
    determines if compilation of a file is needed based on it's timestamp
    and an optional recompile option.

    The behaviour of this function depends on the recompile argument.

    recompile:
        True -> True, the program should be compiled

        'as needed' -> compares filename to timestamp of config
            and checks if the file exists, if required recompile.
        False -> compares program to timestamp of config.
            if compilation is required raises a ValueError
    """
    if recompile == True:
        return True
    elif recompile == 'as needed':
        try:
            if is_more_rencent(program_fn, platf_cfg):
                return False
            else:
                return True # compilation is required
        except FileNotFoundError:
            # File doesn't exist means compilation is required
            return True

    elif recompile == False: # if False
        if is_more_rencent(program_fn, platf_cfg):
            return False
        else:
            raise ValueError('OpenQL config has changed more recently '
                             'than program.')
    else:
        raise NotImplementedError('recompile should be True, False or "as needed"')




def load_range_of_oql_programs(programs, counter_param, CC):
    """
    This is a helper function for running an experiment that is spread over
    multiple OpenQL programs such as RB.
    """
    program = programs[counter_param()]
    counter_param((counter_param()+1) % len(programs))
    CC.eqasm_program(program.filename)

def load_range_of_oql_programs_varying_nr_shots(programs, counter_param, CC,
                                                detector):
    """
    This is a helper function for running an experiment that is spread over
    multiple OpenQL programs of varying length such as GST.

    Everytime the detector is called it will also modify the number of sweep
    points in the detector.
    """
    program = programs[counter_param()]
    counter_param((counter_param()+1) % len(programs))
    CC.eqasm_program(program.filename)

    detector.nr_shots = len(program.sweep_points)