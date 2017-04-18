from pycqed.utilities.general import mopen
from os.path import join, dirname, basename, splitext
from copy import deepcopy
from pycqed.measurement.waveform_control_CC import operation_prep as opf

base_asm_path = join(dirname(__file__), 'micro_instruction_files')


preamble = ('mov r0, 20000   # r0 stores the cycle time , 100 us \n' +
            'mov r1, 0       # sets the inter pulse wait to 0\n' +
            'mov r14, 0      # r14 stores number of repetitions\n' +
            '# Experiment: repeat the rounds for infinite times\n' +
            'Exp_Start: \n')

ending = ('beq r14, r14, Exp_Start       # Infinite loop\n')


def qasm_to_asm(qasm_filepath, operation_dict):
    """
    Args:
        qasm_filepath: (str) location of the qasm file to convert

        operation_dict: (dict) dictionary containing info required for
            conversion.
            *keys*  correspond to qasm commands
            *items* contain dicts with the reuired information.
            every item should contain the following entries:
                instruction: (str) or (fun) that defines the translation
                    from qasm to microcode/assembly
                duration: (int) length of operation expressed in clock cycles
                prepare_function: (str) str that refers to function used to
                    prepare the operation
                prepare_function_kwargs: (dict) containing arguments that get
                    passed to the prepare function

    returns:
        asm_file suitable for CBox Assembler, intended to be compatible
        with the central controller in the future.
    """
    filename = splitext(basename(qasm_filepath))[0]
    asm_filepath = join(base_asm_path, filename+'.qumis')
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
            # for single qubit gates that have an argument
            if len(elts) > 1:
                base_op = elts[0]+' '+elts[1]

            # Interpret qasm elements
            commands = list(operation_dict.keys()) + ['qubit']
            if elts[0] == 'qubit':
                qubits.append(elts[1])

            elif (line in commands):
                instruction = operation_dict[line]['instruction']
                asm_file.writelines(instruction)
            # two qubit operation or operation with arg
            elif base_op in commands:
                # single qubit op with argument
                if 'instruction' in operation_dict[base_op].keys():
                    base_ins = operation_dict[base_op]['instruction']
                    # string formatting is a constraint now but maybe we can
                    # come up with something smarter
                    if isinstance(base_ins, str):
                        instruction = base_ins.format(elts[2])
                    else:
                        instruction = base_ins(elts[2])
                else:  # no support yet for multi qubit ops with arguments
                    raise NotImplementedError(
                        'Multi qubit ops with args: "{}"'.format(line))
                asm_file.writelines(instruction)
            else:
                raise ValueError(
                    'Command "{}" not recognized, must be in {}'.format(
                        elts[0], commands))

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
        return operations


def create_operation_dict(required_ops, pulse_pars):
    """
    Creates the operation dictionary from a set of pulse_pars taken
    from the qubit object
    operation_dict: (dict) dictionary containing info required for
        compilation.
        *keys*  correspond to qasm commands
        *items* contain dicts with the reuired information.

        every item should contain the following entries:
        "instruction": (str) or (fun) that defines the translation
            from qasm to microcode/assembly
        "duration": (int) length of operation expressed in clock cycles
        "prepare_function": (str) str that refers to function used to
            prepare the operation
        "prepare_function_kwargs": (dict) containing arguments that get
            passed to the prepare function

    """
    operation_dict = {}
    default_control_pulses = ['X180', 'X90', 'Y180', 'Y90',
                              'mX180', 'mX90', 'mY180', 'mY90']
    for op_line in required_ops:
        operation_dict[op_line] = {}

        elts = op_line.split()
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

            duration = d_entry.pop('duration')
            operation_dict[op_line]['duration'] = duration
            prepare_function = d_entry.pop('prepare_function')
            operation_dict[op_line]['prepare_function'] = prepare_function
            operation_dict[op_line]['prepare_function_kwargs'] = d_entry
            # no insruction yet
            operation_dict[op_line]['instruction'] = ' \n'

        elif elts[0] == 'I':
            operation_dict[op_line] = {'instruction': 'WaitReg r0 \n',
                                          'prepare_function': None,
                                          'prepare_function_kwargs': None,
                                          'duration': None}

        elif elts[0] == 'init_all':
            operation_dict['init_all'] = {'instruction': 'WaitReg r0 \n',
                                          'prepare_function': None,
                                          'prepare_function_kwargs': None,
                                          'duration': None}
        elif elts[0] == 'RO':  # Currently only writes single qubit dict
            d_entry = deepcopy(pulse_pars[op_line])
            duration = d_entry.pop('duration')
            operation_dict[op_line]['duration'] = duration
            prepare_function = d_entry.pop('prepare_function')
            operation_dict[op_line]['prepare_function'] = prepare_function
            operation_dict[op_line]['prepare_function_kwargs'] = d_entry
            # no insruction yet
            operation_dict[op_line]['instruction'] = ' \n'
        else:
            raise NotImplementedError(
                'Operation {} is not implemented'.format(op_line))
    return operation_dict


def prepare_operations(operation_dict):
    for operation_name, operation in operation_dict.items():
        if operation['prepare_function'] is not None:
            if not hasattr(opf, operation['prepare_function']):
                raise KeyError('unknown prepare function {}'.format(
                    operation['prepare_function']))
            prepare_function = getattr(opf, operation['prepare_function'])
            pf_kwargs = operation['prepare_function_kwargs']
            if isinstance(pf_kwargs, dict):
                prepare_function(operation_name, **pf_kwargs)
            elif pf_kwargs is None:
                prepare_function(operation_name)
            else:
                raise TypeError('prepare_function_kwargs not dict "{}"'.format(
                    pf_kwargs))
            print(pf_kwargs)





