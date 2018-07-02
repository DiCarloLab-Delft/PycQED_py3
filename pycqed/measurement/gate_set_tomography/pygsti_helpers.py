import pygsti
from os.path import join
import pycqed as pq


gst_exp_filepath = join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                        'template_data_files')


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
        # Final slice is not by default included in the experiment list
        expSubLists.append(expList[cutting_indices[-1]:])


    if verbose:
        print("Splitted expList into {} sub lists".format(len(expSubLists)))
    return expSubLists
