import os
import sys
import inspect
import string
import numpy as np

PycQED_py3_dir = "D:\\Github\\PycQED_py3"
AssemblerDir = PycQED_py3_dir + \
                "\\instrument_drivers\\physical_instruments\\_controlbox"
currentdir = os.path.dirname(os.path.abspath(
                inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(AssemblerDir)

import Assembler
import old_assembler

qasm_ext = ".txt"

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

if len(sys.argv) != 2:
    print("Error: Asm2Mem only receives one arguments as the assembly file.")
    exit(0)

rawinput = sys.argv[1]
print("The file read from the argument is:", rawinput)

asm_name = rawinput
if not os.path.isfile(asm_name):
    print("\tError! The file does not exist")

if (asm_name[-len(qasm_ext):] != qasm_ext):
    print("\t Error! The input asm file should have the", qasm_ext,
          "extension. ")
    exit(0)


asm1 = Assembler.Assembler(asm_name)
instructions1 = asm1.convert_to_instructions()
asm2 = old_assembler.Assembler(asm_name)
instructions2 = asm2.convert_to_instructions()

print("compare Result: ", np.array_equal(instructions1, instructions2))
assert(len(instructions1) == len(instructions2))
print("instructions1", '\t', "instructions2")
for i in range(len(instructions1)):
    print(instructions1[i], '\t', instructions2[i])
