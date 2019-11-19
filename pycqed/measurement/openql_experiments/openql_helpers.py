import re
import logging
import numpy as np
from os.path import join, dirname
from pycqed.utilities.general import suppress_stdout
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from pycqed.utilities.general import is_more_rencent
import openql.openql as ql
from openql.openql import Program, Kernel, Platform, CReg, Operation


output_dir = join(dirname(__file__), 'output')
ql.set_option('output_dir', output_dir)
ql.set_option('scheduler', 'ALAP')


def create_program(pname: str, platf_cfg: str, nregisters: int=32):
    """
    Wrapper around the constructor of openQL "Program" class.

    Args:
        pname       (str) : Name of the program
        platf_cfg   (str) : location of the platform configuration used to
            construct the OpenQL Platform used.
        nregisters  (int) : the number of classical registers required in
            the program.

    In addition to instantiating the Program, this function
        - creates a Platform based on the "platf_cfg" filename.
        - Adds the platform as an attribute  "p.platf"
        - Adds the output_dir as an attribute "p.output_dir"

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    nqubits = platf.get_qubit_number()
    p = Program(pname,
                platf,
                nqubits,
                nregisters)

    p.platf = platf
    p.output_dir = ql.get_option('output_dir')
    p.nqubits = platf.get_qubit_number()
    p.nregisters = nregisters

    # detect OpenQL backend ('eqasm_compiler') used
    p.eqasm_compiler = ''
    with open(platf_cfg) as f:
        for line in f:
            if 'eqasm_compiler' in line:
                m = re.search('"eqasm_compiler" *: *"(.*?)"', line)
                p.eqasm_compiler = m.group(1)
                break
    if p.eqasm_compiler == '':
        logging.error(f"key 'eqasm_compiler' not found in file '{platf_cfg}'")

    return p


def create_kernel(kname: str, program):
    """
    Wrapper around constructor of openQL "Kernel" class.
    """
    kname = kname.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})

    k = Kernel(kname, program.platf, program.nqubits, program.nregisters)
    return k


def compile(p, quiet: bool = True):
    """
    Wrapper around OpenQL Program.compile() method.
    """
    if quiet:
        with suppress_stdout():
            p.compile()
    else:  # show warnings
        ql.set_option('log_level', 'LOG_WARNING')
        p.compile()

    # determine extension of generated file
    if p.eqasm_compiler=='eqasm_backend_cc':
        ext = '.vq1asm' # CC
    else:
        ext = '.qisa' # CC-light, QCC
    # attribute is added to program to help finding the output files
    p.filename = join(p.output_dir, p.name + ext)
    return p


#############################################################################
# Calibration points
#############################################################################
def add_single_qubit_cal_points(p, qubit_idx,
                                f_state_cal_pts: bool=False,
                                measured_qubits=None):
    """
    Adds single qubit calibration points to an OpenQL program

    Args:
        p
        platf
        qubit_idx
        measured_qubits : selects which qubits to perform readout on
            if measured_qubits == None, it will default to measuring the
            qubit for which there are cal points.
    """
    if measured_qubits==None:
        measured_qubits = [qubit_idx]

    for i in np.arange(2):
        k = create_kernel("cal_gr_"+str(i), program=p)
        k.prepz(qubit_idx)
        k.gate('wait', measured_qubits, 0)
        for measured_qubit in measured_qubits:
            k.measure(measured_qubit)
        k.gate('wait', measured_qubits, 0)
        p.add_kernel(k)

    for i in np.arange(2):
        k = create_kernel("cal_ex_"+str(i), program=p)
        k.prepz(qubit_idx)
        k.gate('rx180', [qubit_idx])
        k.gate('wait', measured_qubits, 0)
        for measured_qubit in measured_qubits:
            k.measure(measured_qubit)
        k.gate('wait', measured_qubits, 0)
        p.add_kernel(k)
    if f_state_cal_pts:
        for i in np.arange(2):
            k = create_kernel("cal_f_"+str(i), program=p)
            k.prepz(qubit_idx)
            k.gate('rx180', [qubit_idx])
            k.gate('rx12', [qubit_idx])
            k.gate('wait', measured_qubits, 0)
            for measured_qubit in measured_qubits:
                k.measure(measured_qubit)
            k.gate('wait', measured_qubits, 0)
            p.add_kernel(k)
    return p


def add_two_q_cal_points(p, q0: int, q1: int,
                         reps_per_cal_pt: int =1,
                         f_state_cal_pts: bool=False,
                         f_state_cal_pt_cw: int = 31,
                         measured_qubits=None,
                         interleaved_measured_qubits=None,
                         interleaved_delay=None,
                         nr_of_interleaves=1):
    """
    Returns a list of kernels containing calibration points for two qubits

    Args:
        p               : OpenQL  program to add calibration points to
        q0, q1          : ints of two qubits
        reps_per_cal_pt : number of times to repeat each cal point
        f_state_cal_pts : if True, add calibration points for the 2nd exc. state
        f_state_cal_pt_cw: the cw_idx for the pulse to the ef transition.
        measured_qubits : selects which qubits to perform readout on
            if measured_qubits == None, it will default to measuring the
            qubits for which there are cal points.
    Returns:
        kernel_list     : list containing kernels for the calibration points
    """
    kernel_list = []
    combinations = (["00"]*reps_per_cal_pt +
                    ["01"]*reps_per_cal_pt +
                    ["10"]*reps_per_cal_pt +
                    ["11"]*reps_per_cal_pt)
    if f_state_cal_pts:
        extra_combs = (['02']*reps_per_cal_pt + ['20']*reps_per_cal_pt +
                       ['22']*reps_per_cal_pt)
        combinations += extra_combs

    if measured_qubits == None:
        measured_qubits = [q0, q1]


    for i, comb in enumerate(combinations):
        k = create_kernel('cal{}_{}'.format(i, comb), p)
        k.prepz(q0)
        k.prepz(q1)
        if interleaved_measured_qubits:
            for j in range(nr_of_interleaves):
                for q in interleaved_measured_qubits:
                    k.measure(q)
                k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)
                if interleaved_delay:
                    k.gate('wait', [0, 1, 2, 3, 4, 5, 6], int(interleaved_delay*1e9))

        if comb[0] =='0':
            k.gate('i', [q0])
        elif comb[0] == '1':
            k.gate('rx180', [q0])
        elif comb[0] =='2':
            k.gate('rx180', [q0])
            # FIXME: this is a workaround
            #k.gate('rx12', [q0])
            k.gate('cw_31', [q0])

        if comb[1] =='0':
            k.gate('i', [q1])
        elif comb[1] == '1':
            k.gate('rx180', [q1])
        elif comb[1] =='2':
            k.gate('rx180', [q1])
            # FIXME: this is a workaround
            #k.gate('rx12', [q1])
            k.gate('cw_31', [q1])

        # Used to ensure timing is aligned
        k.gate('wait', measured_qubits, 0)
        for q in measured_qubits:
            k.measure(q)
        k.gate('wait', measured_qubits, 0)
        kernel_list.append(k)
        p.add_kernel(k)

    return p


def add_multi_q_cal_points(p, qubits: list,
                           combinations: list):
    """
    Adds calibration points based on a list of state combinations
    """
    kernel_list = []
    for i, comb in enumerate(combinations):
        k = create_kernel('cal{}_{}'.format(i, comb), p)
        for q in qubits:
            k.prepz(q)

        for j, q in enumerate(qubits):
            if comb[j] == '1':
                k.gate('rx180', [q])
            elif comb[j] == '2':
                k.gate('rx180', [q])
                k.gate('rx12', [q])
            else:
                pass
        # Used to ensure timing is aligned
        k.gate('wait', qubits, 0)
        for q in qubits:
            k.measure(q)
        k.gate('wait', qubits, 0)
        kernel_list.append(k)
        p.add_kernel(k)
    return p


#############################################################################
# File modifications
#############################################################################


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


def get_operation_tuples(time_tuples: list, target_op: str):
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

# FIXME: platform dependent (CC-light)
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
    FIXME: program_fn is platform dependent, because it includes extension

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
                return True  # compilation is required
        except FileNotFoundError:
            # File doesn't exist means compilation is required
            return True

    elif recompile == False:  # if False
        if is_more_rencent(program_fn, platf_cfg):
            return False
        else:
            raise ValueError('OpenQL config has changed more recently '
                             'than program.')
    else:
        raise NotImplementedError(
            'recompile should be True, False or "as needed"')


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
