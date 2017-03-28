from unittest import TestCase
import pycqed as pq
import os
import string
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler as asm


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        print('this is setting up the test')

    def test_qasm_extract_required_ops(self):
        print('hello')
        self.assertTrue(True)

    def test_get_bin(self):
        with self.assertRaises(ValueError):
            asm.get_bin('5s', 3)

    def test_program_length(self):
        asm_filename = os.path.join(pq.__path__[0], 'tests', 'test_data', "20170328", "LabelTest.qumis")
        asm_obj = asm.Assembler(asm_filename)
        instructions = asm_obj.convert_to_instructions()
        tag_addr_dict = asm_obj.ParseLabel()


    def test_insert_nop(self):
        asm_filename = os.path.join(pq.__path__[0], 'tests', 'test_data', "20170328", "LabelTest.qumis")
        asm_obj = asm.Assembler(asm_filename)
        instructions = asm_obj.convert_to_instructions()
        text_instructions = asm_obj.getTextInstructions()


        # assert number of instructions
        self.assertEqual(len(instructions), len(text_instructions))

        # assert number of nop instructions
        number_of_text_nop = 0
        for instr in text_instructions:
            head, sep, tail = instr.partition(':')
            if (sep == ":"):
                instr = tail
            else:
                instr = head

            elements = [rawEle.strip(string.punctuation.translate(
                        {ord('-'): None})) for rawEle in instr.split()]

            if elements[0] == 'nop':
                number_of_text_nop = number_of_text_nop + 1

        self.assertEqual(instructions.count(0), number_of_text_nop)