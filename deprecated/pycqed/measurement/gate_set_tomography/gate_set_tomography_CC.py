import re
import pygsti
import os
from pycqed.measurement.waveform_control_CC.multi_qubit_qasm_seqs import \
    base_qasm_path
import numpy as np


def get_experiments_from_list(pygsti_list, gate_dict):
    '''
    Converts a experiment list from the pyGSTi format (list of GateStrings),
    to QASM format.

    Args:
        pygsti_list (list):
            pyGSTi experiment list (list of GateStrings) as returned by
            pygsti.construction.make_lsgst_experiment_list.
        gate_dict (dict):
            Dictionary that maps GST labels to gates in QASM syntax, including
            target qubit.
            For example {'Gx90': 'X90 QR', 'RO': 'RO QR'}.
            Note: this should always contain an entry for 'RO'.

    Returns:
        exp_list (list):
            Every element is one experiment (one line in the original template
            file). Every experiment is a list operations in QASM syntax.
    '''
    exp_list = []
    for exp in pygsti_list:
        operations = ['\ninit_all']
        operations += [gate_dict[g] for g in exp.tup]
        operations += [gate_dict['RO']]
        exp_list.append(operations)
    return exp_list


def get_experiments_from_file(filename, gate_dict, use_pygsti_parser=True):
    '''
    Parses a GST template and returns a list of experiments in QASM syntax.

    Args:
        filename (string):
            Name of the GST template file.
        gate_dict (dict):
            Dictionary that maps GST labels to gates in QASM syntax, including
            target qubit.
            For example {'Gx90': 'X90 QR', 'RO': 'RO QR'}.
            Note: this should always contain an entry for 'RO'.
        use_pygsti_parser (bool):
            True: use the functionality of the pyGSTi GateString object to
                parse the file.
            False: use custom parser implemented in this function to parse
                the file.

    Returns:
        experiments (list):
            Every element is one experiment (one line in the original template
            file). Every experiment is a list of gates in QASM syntax.
    '''
    # How this works:
    # Loops through every line.
    # For each line, repeatedly looks for the first string that matches one of
    # the GST labels (given by the keys of gate_dict), until no more match is
    # found.
    # In every iteration, first checks if the next part is a repeated sub-
    # sequence.

    # Define the regex pattern for finding the GST labels.
    labelsPattern = ''
    for label in gate_dict.keys():
        labelsPattern = '|'.join([labelsPattern, '({})'.format(label)])
    # Remove first '|' that comes from joining empty string with first key
    labelsPattern = labelsPattern[1:]

    # Read GST template file
    with open(filename) as file:
        rawLines = file.readlines()

    # The sequence is in the first column of every line.
    lines = []
    for line in rawLines:
        lines.append(line.split(' ')[0])

    # First line usually contains a comment that we remove here.
    if lines[0][0] == '#':
        lines = lines[1:]

    if use_pygsti_parser:
        gateStrList = [
            pygsti.objects.GateString(None, stringRepresentation=line)
            for line in lines]
        return get_experiments_from_list(gateStrList, gate_dict)

    # Loop to parse the file.
    experiments = []
    for line in lines:
        gates = ['\ninit_all']

        if line[0] == '#':
            # starts with # -> comment
            continue

        # First see if there is a gate in the line. res will be none if no
        # match is found. The while loop thus runs until no more gates are
        # found in the line.
        res = re.search(labelsPattern, line)
        while res != None:
            # Find out if the next part is a repeated subsequence, i.e.
            # something like '(GxGyGx)^3'.
            resSeq = re.search('^\)?\((' + labelsPattern + ')*?\)\^\d+', line)

            if resSeq != None:
                # For repeated subsequences, we find all the gates in the
                # subsequence and then append them as many times as the
                # exponent tells us.
                subStr = resSeq.group()
                exponent = int(subStr.split('^')[-1])
                base = []

                subRes = re.search(labelsPattern, subStr)  # Find first gate
                while subRes != None:
                    base.append(gate_dict[subRes.group()])
                    subStr = subStr[subRes.end():]  # Remove first gate
                    subRes = re.search(labelsPattern, subStr)  # Next gate

                gates += base * exponent
                line = line[resSeq.end():]  # Remove the part that was parsed
                res = re.search(labelsPattern, line)  # Find next gate

            else:
                # Not a repeated sequence -> directly append this gate to the
                # experiment.
                gates.append(gate_dict[res.group()])
                line = line[res.end():]  # Remove gate
                res = re.search(labelsPattern, line)  # Find next gate

        gates.append(gate_dict['RO'])
        experiments.append(gates)  # Append this experiment to the list

    return experiments


def generate_QASM(filename, exp_list, qubit_labels, max_instructions=2**15,
                  max_exp_per_file=4094):
    '''
    Generates a QASM file from an experiment list. If there are too many
    instructions in the experiment list to fit in the CBox memory, then the
    list will be split and several QASM files will be generated. A suffix '_i'
    is appended to the name of the QASM files, where i is a number (in order
    of generation).

    Args:
        filename (string):
            Name for the QASM files. A suffix '_i' is added, where i goes from
            0 to the number of QASM files generated - 1.
        exp_list (list):
            List containing the experiments, as returned by get_experiments.
        qubit_labels (list of strings):
            List of the qubit labels used in the experiment. These should be
            the same as in the gate_dict passed to get_experiments.
        max_instructions (int):
            Maximum number of instructions that will fit the CBox memory. If
            this number is exceeded in the experiment_list, more than one
            QASM file will be generated.
            Default is 2**15 (CBox memory size as of June 29 2017).
        max_exp_per_file (int):
            Maximum number of experiments allowed in one file. This is limited
            by the maximum number of shots the acquisition device can handle.

    Returns:
        qasm_files (list):
            List containing the file objects for the generated QASM files.
        exp_num_list (list):
            List containing the number of experiments run in each QASM file.
    '''
    file_list = []
    filename = os.path.join(base_qasm_path, filename)

    # The longest experiment gives us an upper bound for how many experiments
    # we can include in one QASM file before reaching the maximum number of
    # instructions the central controller can hold. 4 instructions per line of
    # QASM code is an  upper bound chosen because two simultaneous mw pulses
    # or one two-qubit gate use 4 instructions.
    exp_lens = [len(exp) for exp in exp_list]
    exp_per_file = int(max_instructions // (4 * max(exp_lens)))

    if exp_per_file >= len(exp_list):
        # All experiments fit into one file.
        exp_per_file = len(exp_list)
        nr_full_files = 1  # needs manual setting because '//' takes floor
    else:
        nr_full_files = int(len(exp_list) // exp_per_file)

    # Reshape experiment list according to how it is split into files.
    shaped_exp_list = np.reshape(exp_list[:nr_full_files*exp_per_file],
                                 (-1, exp_per_file))

    # Write the files from the nested list.
    for i, sub_list in enumerate(shaped_exp_list):
        file = open('{}_{}.qasm'.format(filename, i), 'w')
        file_list.append(file)
        for q in qubit_labels:
            file.writelines('qubit {} \n'.format(q))

        for exp in sub_list:
            for gate in exp:
                file.writelines('{}\n'.format(gate))

        file.close()

    # If there are some experiments remaining, write last file with less than
    # exp_per_file experiments
    exp_last_file = len(exp_list) - nr_full_files*exp_per_file
    if exp_last_file != 0:
        file = open('{}_{}.qasm'.format(filename, nr_full_files), 'w')
        file_list.append(file)
        for q in qubit_labels:
            file.writelines('qubit {} \n'.format(q))

        for exp in exp_list[nr_full_files*exp_per_file:]:
            for gate in exp:
                file.writelines('{}\n'.format(gate))

        file.close()

    return file_list, exp_per_file, exp_last_file
