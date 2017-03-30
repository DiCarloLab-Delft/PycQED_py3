from unittest import TestCase
import pycqed as pq
import os
import string
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler as asm


class Test_single_qubit_seqs(TestCase):
    @classmethod
    def setUpClass(self):
        self.asm_filename = os.path.join(
            pq.__path__[0], 'tests', 'test_data', "20170328",
            "LabelTest.qumis")
        self.asm_obj = asm.Assembler(self.asm_filename)
        self.instructions = self.asm_obj.convert_to_instructions()
        self.text_instructions = self.asm_obj.getTextInstructions()
        print('this is setting up the test')

    def test_is_number(self):
        self.assertFalse(asm.is_number('5s'))
        self.assertTrue(asm.is_number('128464'))


    def test_get_bin(self):
        with self.assertRaises(ValueError):
            asm.get_bin('5s', 3)

    def test_program_length(self):
        self.assertEqual(len(self.instructions), len(self.text_instructions))

    def test_insert_nop(self):
        print("hex: ", self.instructions)
        print("text:", self.text_instructions)

        # count number of nop instructions in the text version
        number_of_text_nop = 0
        for instr in self.text_instructions:
            head, sep, tail = instr.partition(':')
            if (sep == ":"):
                instr = tail
            else:
                instr = head

            elements = [rawEle.strip(string.punctuation.translate(
                        {ord('-'): None})) for rawEle in instr.split()]

            if elements[0] == 'nop':
                number_of_text_nop = number_of_text_nop + 1

        number_of_hex_nop = self.instructions.count(0)
        for i in range(1, 6):  # five nops added appending branch instructions.
            number_of_hex_nop = number_of_hex_nop + self.instructions.count(i<<8)
        print(self.instructions.count(0), number_of_text_nop)
        # compare
        self.assertEqual(number_of_hex_nop, number_of_text_nop)
