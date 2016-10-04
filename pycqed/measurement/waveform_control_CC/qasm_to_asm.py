from pycqed.utilities.general import mopen
from os.path import join, dirname, basename, splitext
from copy import deepcopy
base_asm_path = join(dirname(__file__), 'micro_instruction_files')


preamble = ('mov r0, 20000   # r0 stores the cycle time , 100 us \n' +
            'mov r1, 0       # sets the inter pulse wait to 0\n' +
            'mov r14, 0      # r14 stores number of repetitions\n' +
            '# Experiment: repeat the rounds for infinite times\n' +
            'Exp_Start: \n')

ending = 'beq r14, r14, Exp_Start       # Infinite loop'


def qasm_to_asm(qasm_filepath, operation_dict):
    """
    Args:
        qasm_filepath: (str) location of the qasm file to convert
        operation_dict: (dict) dictionary containing info required for
            conversion.
            operation_dict (od) should have the following (nested) structure
            od = {operation: {qubit: {duration_cl: int,
                                      instruction: string or function}}

            todo: how to format multi-qubit operations in
                a) qubit key -> qubits key, combined string with dash
                b) qubit key1 -> leads to operation or to nested dict with
                                qubit_key2 that contains the 2-qubit operation
            todo: error messages
                1) undefined operation error
                2) operation not defined for qubit ""
                3) syntax errors
    returns:
        asm_file suitable for CBox Assembler, intended to be compatible
        with the central controller in the future.
    """
    filename = splitext(basename(qasm_filepath))[0]
    asm_filepath = join(base_asm_path, filename+'.asm')
    asm_file = mopen(asm_filepath, mode='w')
    asm_file.writelines(preamble)

    with open(qasm_filepath) as qasm_file:
        qubits = []  # the qubits that were defined
        for line in qasm_file:
            # Make lines interpretable
            line = line.split('#', 1)[0]  # remove comments
            line = line.strip(' \t\n\r')  # remove whitespace
            if (len(line) == 0):  # skip empty line and comment
                continue
            elts = line.split()

            # Interpret qasm elements
            commands = list(operation_dict.keys()) + ['qubit']
            if elts[0] not in commands:
                raise ValueError(
                    'Command "{}" not recognized, must be in {}'.format(
                        elts[0], commands))
            # special command: a line that defines a qubit
            elif elts[0] == 'qubit':
                qubits.append(elts[1])
            elif len(elts) == 1:
                instruction = operation_dict[elts[0]]['instruction']
                asm_file.writelines(instruction)
            # Single qubit operation
            elif len(elts) == 2:
                instruction = operation_dict[elts[0]][elts[1]]['instruction']
                asm_file.writelines(instruction)
            # two qubit operation or operation with arg
            elif len(elts) == 3:
                # single qubit op with argument
                if 'instruction' in operation_dict[elts[0]][elts[1]].keys():
                    base_ins = operation_dict[elts[0]][elts[1]]['instruction']
                    # string formatting is a constraint now but maybe we can
                    # come up with something smarter
                    if isinstance(base_ins, str):
                        instruction = base_ins.format(elts[2])
                    else:
                        instruction = base_ins(elts[2])
                else:
                    instruction = operation_dict[elts[0]][elts[1]][elts[2]]['instruction']
                asm_file.writelines(instruction)
            else: # no support yet for multi qubit ops with arguments
                raise ValueError('qasm lines has too many args {},{}'.format(
                                 elts, line))

    asm_file.writelines(ending)
    asm_file.close()
    return asm_file


def extract_required_operations(qasm_filepath):
        """
        Args:
            qasm_filepath: (str) location of the qasm file to read
        returns:
            list containing of all the used operations with args
        """
        filename = splitext(basename(qasm_filepath))[0]
        asm_filepath = join(base_asm_path, filename+'.asm')
        asm_file = mopen(asm_filepath, mode='w')
        asm_file.writelines(preamble)

        qubits = []  # the qubits that were defined
        operations = []
        with open(qasm_filepath) as qasm_file:
            for line in qasm_file:
                # Make lines interpretable
                line = line.split('#', 1)[0]  # remove comments
                line = line.strip(' \t\n\r')  # remove whitespace
                if (len(line) == 0):  # skip empty line and comment
                    continue
                elts = line.split()
                # special command: a line that defines a qubit
                if elts[0] == 'qubit':
                    qubits.append(elts[1])
                elif line not in operations:
                    operations.append(line)
        return qubits, operations


def upload_qasm_waveform(qasm_command, operation_dict):
    """
    uploads a single instruction
    """
    elts = qasm_command.split()
    upload_function = operation_dict


def create_operation_dict(required_ops, pulse_pars):
    """
    Creates the operation dictionary from a set of pulse_pars taken
    from the qubit object
    """
    operation_dict = {}
    default_control_pulses = ['X180', 'X90', 'Y180', 'Y90',
                              'mX180', 'mX90', 'mY180', 'mY90']
    for op_line in required_ops:
        elts = op_line.split()
        if not elts[0] in operation_dict.keys():
            operation_dict[elts[0]] = {}
        if elts[0] in default_control_pulses:
            d_entry = deepcopy(pulse_pars[
                'control_pulse {}'.format(elts[1])])
            if '90' in elts[0]:
                d_entry['amplitude'] = (d_entry['amp90_scale'] *
                                        d_entry['amplitude'])
            if 'X' in elts[0]:
                d_entry['phase'] = 0
            elif 'Y' in elts[0]:
                d_entry['phase'] = 90
            if 'm' in elts[0]:
                d_entry['phase'] += 180
            operation_dict[elts[0]][elts[1]] = d_entry

        elif elts[0] == 'init_all':
            operation_dict['init_all'] = {'instruction': 'WaitReg r0 \n'}
        elif elts[0] == 'RO':  # Currently only writes single qubit dict
            if op_line in pulse_pars.keys():
                operation_dict[elts[0]][elts[1]] = pulse_pars[op_line]
            else:
                raise KeyError(
                    'RO on qubits {} not defined in pulse_pars {}'.format(
                        elts[1:], pulse_pars.keys()))
        else:
            raise NotImplementedError(
                'Operation {} is not implemented'.format(op_line))
    return operation_dict



def prepare_operations(operation_dict):
    print('preparing all operations')
    print('\t setting global settings')
    print('\t uploading waveforms')



