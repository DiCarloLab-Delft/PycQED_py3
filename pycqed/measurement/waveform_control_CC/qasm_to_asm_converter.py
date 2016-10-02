from pycqed.utilities.general import mopen
from os.path import join, dirname, basename, splitext
base_asm_path = join(dirname(__file__), 'micro_instruction_files')


commands = ['qubit', 'init_all', 'X', 'I', 'RO']

op_dict = {'init_all': 'WaitReg r0 \n',
           'X': 'Trigger 1000000, 2 \n',  # Using marker 1 for the qubit pulse
           'I': 'mov r1, {} \n WaitReg r1 \n',
           'RO': 'Trigger 0100000, 4 \n'}  # using marker 2 for the RO trigger
           #  we could also use multiple instructions

preamble = ('mov r0, 20000   # r0 stores the cycle time , 100 us \n' +
            'mov r1, 0       # sets the inter pulse wait to 0\n' +
            'mov r14, 0      # r14 stores number of repetitions\n' +
            '# Experiment: repeat the rounds for infinite times\n' +
            'Exp_Start:')

ending = 'beq r14, r14, Exp_Start       # Infinite loop'


def qasm_to_asm(qasm_filepath, operation_dictionary=op_dict):
    """
    Args:
        qasm_filepath: (str) location of the qasm file to convert
        operation_dictionary: (dict) dictionary containing info required for
            conversion.
            operation_dictionary (od) should have the following (nested) structure
            od = {operation: {qubit: {duration_cl: int,
                                      instruction: string}}

            todo: how to format multi-qubit operations in
                a) qubit key -> qubits key, combined string with dash
                b) qubit key1 -> leads to operation or to nested dict with
                                qubit_key2 that contains the 2-qubit operation
            todo: dealing with operation args e.g. wait time or angle/phase
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
            line = line.split('#', 1)[0]  # remove comments
            line = line.strip(' \t\n\r')  # remove whitespace
            if (len(line) == 0):  # skip empty line and comment
                continue
            elts = line.split()
            if elts[0] not in commands:
                raise ValueError(
                    'Command "{}" not recognized, must be in {}'.format(
                        elts[0], commands))
            elif elts[0] == 'qubit':  # a line that defines a qubit
                qubits.append(elts[1])
            elif len(elts) == 1:
                asm_file.writelines(operation_dictionary[elts[0]])
            elif len(elts) == 2:
                asm_file.writelines(operation_dictionary[elts[0]])
            elif len(elts) == 3:
                asm_file.writelines(operation_dictionary[elts[0]].format(elts[2]))
            else:

                print(elts)
                print(line)
    asm_file.writelines(ending)
    asm_file.close()
    return asm_file
