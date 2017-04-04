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

    def test_is_number(self):
        self.assertFalse(asm.is_number('5s'))
        self.assertTrue(asm.is_number('128464'))

    def test_get_bin(self):
        with self.assertRaises(ValueError):
            asm.get_bin('5s', 3)

    def test_program_length(self):
        self.assertEqual(len(self.instructions), len(self.text_instructions))

    def test_insert_nop(self):
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

        print(self.instructions.count(0), number_of_text_nop)
        # compare
        self.assertEqual(self.instructions.count(0), number_of_text_nop)

    def test_wait_zero(self):
        zero_wait_fname = os.path.join(
            pq.__path__[0], 'tests', 'test_data', "waitTest",
            "waitZeroTest.qumis")

        my_asm_obj = asm.Assembler(zero_wait_fname)
        my_instructions = my_asm_obj.convert_to_instructions()

        # my_instructions should corresponds to the following instructions:
        # wait 5
        # wait 3
        # wait  1000
        # trigger 0000001 1000
        # beq r0, r0, EndOfFileLoop
        # So the 'wait 0' line from the test file has been deleted.
        # This is represented by the array below
        self.assertEqual(my_instructions, [3229614085, 3229614083, 3229615080,
                                           2692875240, 302022653])

    def test_wait_negative(self):
        neg_wait_fname = os.path.join(
            pq.__path__[0], 'tests', 'test_data', "waitTest",
            "waitNegativeTest.qumis")

        my_asm_obj = asm.Assembler(neg_wait_fname)
        with self.assertRaises(ValueError):
            my_instructions = my_asm_obj.convert_to_instructions()
