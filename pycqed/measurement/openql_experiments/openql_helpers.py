"""

"""
import re
import json
from pycqed.measurement.openql_experiments.get_qisa_tqisa_timing_tuples import (
        get_qisa_tqisa_timing_tuples
    )


def infer_tqisa_filename(qisa_fn: str):
    """
    Get's the expected tqisa filename based on the qisa filename.
    """
    return qisa_fn[:-4]+'tqisa'

def get_start_time(line:str):
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



def get_timetuples_since_event(timing_grid: list, target_labels: list,
                               start_label: str, end_label: str=None,
                               convert_clk_to_ns: bool=False) ->list:
    """
    Searches for target labels in a timing grid and returns the time between
    the the events and the timepoint corresponding to the start_label.

    N.B. timing is in clocks

    returns
        time_tuples (list) of tuples (time, label)
        end_time (int)
    """

    # return time_tuples, end_time


def get_timepoints_from_label(
        timing_grid: list, target_label: str,
        start_label: str =None, end_label: str=None,
        convert_clk_to_ns: bool =False)->dict:
    """
    Extract timepoints from a timing grid based on their label.
        target_label : the label to search for in the timing grid
        timing_grid : a list of time_points to search in

    N.B. timing grid is in clocks

    returns
        timepoints (dict) with keys
            'start_tp'    starting time_point
            'target_tps'  list of time_points found with label target_label
            'end_tp'      end time_point
    """

    # return timepoints