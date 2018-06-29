"""
This file reads in a pygsti dataset file and converts it to a valid
OpenQL sequence.
"""
import time
import numpy as np
import pygsti
from os.path import join, dirname
import openql.openql as ql
from pycqed.utilities.general import suppress_stdout
import pycqed as pq
from openql.openql import Program, Kernel, Platform
from pycqed.measurement.openql_experiments import openql_helpers as oqh


base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)

gst_exp_filepath = join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                        'template_data_files')


# used to map pygsti gates to openQL gates
# for now (Jan 2018) only contains basic pygsti gates
gatemap = {'i': 'i',
           'x': 'rx90',
           'y': 'ry90',
           'cphase': 'fl_cw_01'}


def pygsti_expList_from_dataset(dataset_filename: str):
    """
    Takes a pygsti dataset file and extracts the experiment list from it.
    The expList is a list containing pygsti GateString objects.

    Args:
        dataset_filename (str) : filename of the dataset to read
    returns:
        expList (list): a list of pygsti GateString objects.
    """
    with open(dataset_filename, 'r') as f:
        rawLines = f.readlines()
    # lines[1:]

    lines = []
    for line in rawLines:
        lines.append(line.split(' ')[0])
    expList = [
        pygsti.objects.GateString(None, stringRepresentation=line)
        for line in lines[1:]]

    return expList


def openql_program_from_pygsti_expList(expList, program_name: str,
                                       qubits: list,
                                       platf_cfg: str,
                                       start_idx: int=0,
                                       recompile=True):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name,
                nqubits=platf.get_qubit_number(),
                p=platf)
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    if oqh.check_recompilation_needed(p.filename, platf_cfg, recompile):

        for i, gatestring in enumerate(expList):
            kernel_name = 'G {} {}'.format(i, gatestring)
            k = openql_kernel_from_gatestring(
                gatestring=gatestring, qubits=qubits,
                kernel_name=kernel_name, platf=platf)
            p.add_kernel(k)
        with suppress_stdout():
            p.compile()

    p.sweep_points = np.arange(len(expList), dtype=float) + start_idx
    p.set_sweep_points(p.sweep_points, len(p.sweep_points))

    return p


def openql_kernel_from_gatestring(gatestring, qubits: list,
                                  kernel_name: str, platf):
    """
    Generates an openQL kernel for a pygsti gatestring.
    """
    k = Kernel(kernel_name, p=platf)
    for q in qubits:
        k.prepz(q)

    for gate in gatestring:
        assert gate[0] == 'G'  # for valid pyGSTi gatestrings
        if len(gate[1:]) == 1:
            # 1Q pygsti format e.g.,: Gi  or Gy
            k.gate(gatemap[gate[1]], qubits[0])

        elif len(gate[1:]) == 2:
            # 2Q pygsti format e.g.,: Gix  or Gyy
            k.gate(gatemap[gate[1]], qubits[0])
            k.gate(gatemap[gate[2]], qubits[1])
        elif gate == 'Gcphase':
            # only two qubit gate supported
            k.gate(gatemap[gate[1:]], qubits[0], qubits[1])
        else:
            raise NotImplementedError('Gate {} not supported'.format(gate))

    for q in qubits:
        k.measure(q)
    # ensures timing of readout is aligned
    k.gate('wait', qubits, 0)

    return k


def split_expList(expList, max_nr_of_instr: int=8000,
                  verbose: bool=True):
    """
    Splits a pygsti expList into sub lists to facilitate running on the CCL
    and not running into the instruction limit.

    Assumptions made:
        - there is a fixed instruction overhead per program
        - there is a fixed instruction overhead per kernel (measurement + init)
        - every gate (in the gatestring) consists of a single instruction
    """
    fixed_program_overhad = 12 + 3  # declare registers + infinite loop
    kernel_overhead = 4  # prepz wait and measure

    instr_cnt = 0
    instr_cnt += fixed_program_overhad

    # Determine where to split the expLists
    cutting_indices = [0]
    for i, gatestring in enumerate(expList):
        instr_cnt += kernel_overhead
        instr_cnt += len(gatestring)
        if instr_cnt > max_nr_of_instr:
            cutting_indices.append(i)
            instr_cnt = fixed_program_overhad

    # Create the expSubLists, a list contain expList objects for each part
    expSubLists = []
    if len(cutting_indices) == 1:
        expSubLists.append(expList)
    else:
        for exp_num, start_idx in enumerate(cutting_indices[:-1]):
            stop_idx = cutting_indices[exp_num+1]
            expSubLists.append(expList[start_idx:stop_idx])

    if verbose:
        print("Splitted expList into {} sub lists".format(len(expSubLists)))
    return expSubLists


##############################################################################
# End of helper functions
##############################################################################

def single_qubit_gst(q0: int, platf_cfg: str,
                     maxL: int=256,
                     lite_germs: bool = True,
                     recompile=True,
                     verbose: bool=True):
    """
    Generates the QISA and QASM programs for full 2Q GST.

    Args:
        q0 (int)        : target qubit
        platf_cfg (str) : string specifying config filename
        maxL (int)      : specifies the maximum germ length,
                          must be power of 2.
        lite_germs(bool): if True uses "lite" germs
        recompile:      True -> compiles the program,
                        'as needed' -> compares program to timestamp of config
                            and existence, if required recompile.
                        False -> compares program to timestamp of config.
                            if compilation is required raises a ValueError

                        If the program is more recent than the config
                        it returns an empty OpenQL program object with
                        the intended filename that can be used to upload the
                        previously compiled file.
        verbose (bool)  : if True prints extra debug info


    Returns:
        programs (list) : list of OpenQL program objects

    """
    # grab the experiment list file
    if lite_germs:
        fp = join(gst_exp_filepath, 'std1Q_XYI_lite_maxL{}.txt'.format(maxL))
    else:
        fp = join(gst_exp_filepath, 'std1Q_XYI_maxL{}.txt'.format(maxL))

    if verbose:
        print('Converting dataset to expList')
    t0 = time.time()
    expList = pygsti_expList_from_dataset(fp)
    if verbose:
        print("Converted dataset to expList in {:.2f}s".format(time.time()-t0))
    # divide into smaller experiment lists
    programs = []

    t0 = time.time()
    if verbose:
        print('Generating GST programs')

    expSubLists = split_expList(expList, verbose=verbose)

    start_idx = 0
    for exp_num, expSubList in enumerate(expSubLists):
        stop_idx = start_idx + len(expSubList)

        # turn into openql program
        p = openql_program_from_pygsti_expList(
            expSubList, 'std1Q_XYI q{} {} {} {}-{}'.format(
                q0, lite_germs, maxL, start_idx, stop_idx),
            qubits=[q0],
            start_idx=start_idx,
            platf_cfg=platf_cfg, recompile=recompile)
        # append to list of programs
        programs.append(p)
        start_idx += len(expSubList)
        if verbose:
            print('Generated {} GST programs in {:.1f}s'.format(
                  exp_num+1, time.time()-t0), end='\r')

    print('Generated {} GST programs in {:.1f}s'.format(
          exp_num+1, time.time()-t0))

    return programs


def poor_mans_2q_gst(q0: int, q1: int, platf_cfg: str,):
    """
    Generates the QISA and QASM programs for poor_mans_GST, this is 2Q GST
    without the repetitions of any gate.
    """

    fp = join(gst_exp_filepath, 'PoorMans_2Q_GST.txt')
    expList = pygsti_expList_from_dataset(fp)
    p = openql_program_from_pygsti_expList(
        expList, 'PoorMans_GST', [2, 0], platf_cfg=platf_cfg)
    return p


def full_2q_gst(q0: int, q1: int, platf_cfg: str, recompile=True):
    """
    Generates the QISA and QASM programs for full 2Q GST.
    """

    MAX_EXPLIST_SIZE = 3000  # UHFQC shot limit
    DBG = True  # debugging

    # grab the experiment list file
    fp = join(gst_exp_filepath, 'Explist_2Q_XYCphase.txt')
    # parse the file into expList object
    print('Converting dataset to expLit')
    t0 = time.time()
    expList = pygsti_expList_from_dataset(fp)
    print("converting to expList took {:.2f}".format(time.time()-t0))
    # divide into smaller experiment lists
    programs = []

    cutting_indices = [0, 3000, 5500, 7000, 9000]
    t0 = time.time()
    print('Generating {} GST programs'.format(len(cutting_indices)-1))

    for exp_num, start_idx in enumerate(cutting_indices[:-1]):
        stop_idx = cutting_indices[exp_num+1]
        el = expList[start_idx:stop_idx]

    # exp_num = int(np.ceil(len(expList)/MAX_EXPLIST_SIZE))# amount of experiments
    # for exp_i in range(exp_num-1): # loop over the experiments except the last one
        # make smaller experiment list
        # el = expList[exp_i*MAX_EXPLIST_SIZE:(exp_i+1)*MAX_EXPLIST_SIZE]
        # if DBG: print('dbg {}, {}'.format(exp_i*MAX_EXPLIST_SIZE,(exp_i+1)*MAX_EXPLIST_SIZE)) # dbg

        # turn into openql program
        p = openql_program_from_pygsti_expList(
            el, 'full 2Q GST exp {}-{}'.format(start_idx, stop_idx),
            qubits=[2, 0],
            start_idx=start_idx,
            platf_cfg=platf_cfg, recompile=recompile)
        # append to list of programs
        programs.append(p)
        print('Generated {} GST programs in {:.1f}s'.format(
              exp_num+1, time.time()-t0), end='\r')

    print('Succesfully generated {} GST programs in {:.1f}s'.format(
        len(cutting_indices)-1, time.time()-t0))

    # last experiment
    # el = expList[(exp_num-1)*MAX_EXPLIST_SIZE:] # grab experiment list
    # if DBG: print('dbg {}, end'.format((exp_num-1)*MAX_EXPLIST_SIZE)) # dbg
    # # turn into openql program
    # p = openql_program_from_pygsti_expList(
    #     el, 'full 2Q GST index {}'.format(exp_num-1),
    #     [2, 0], platf_cfg=platf_cfg,  recompile=recompile)
    # # append to list of programs
    # programs.append(p)

    return programs
