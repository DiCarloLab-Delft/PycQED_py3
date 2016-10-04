from pycqed.utilities.general import mopen
from os.path import join, dirname, basename, splitext
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
