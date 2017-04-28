from unittest import TestCase
import pycqed as pq
import os
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler as asm


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        print('this is setting up the test')

    def setAssembler(self, qumis_file_name=None):
        test_file_dir = os.path.join(
            pq.__path__[0], 'tests', 'test_data', "20170328")
        if qumis_file_name != None:
            self.qumis_file_name = os.path.join(test_file_dir,
                                                qumis_file_name)
        else:
            self.qumis_file_name = os.path.join(test_file_dir,
                                                "LabelTest.qumis")

        self.assembler = asm.Assembler(self.qumis_file_name)

    def test_is_number(self):
        self.assertFalse(asm.is_number('5s'))
        self.assertTrue(asm.is_number('128464'))

    def test_get_bin(self):
        with self.assertRaises(ValueError):
            asm.get_bin('5s', 3)

    def test_bin_to_hex(self):
        self.assertEqual(asm.bin_to_hex('11001', 2), "19")
        with self.assertRaises(ValueError):
            asm.bin_to_hex("1d1", 2)

    def test_dec_to_bin_w4(self):
        self.assertEqual(asm.dec_to_bin_w4(5), "0101")
        with self.assertRaises(ValueError):
            asm.dec_to_bin_w4("a")

    def test_dec_to_bin_w8(self):
        self.assertEqual(asm.dec_to_bin_w8(5), "00000101")
        with self.assertRaises(ValueError):
            asm.dec_to_bin_w8("a")

    def test_get_reg_num(self):
        self.setAssembler()
        self.assertEqual(self.assembler.get_reg_num("r6"), "0110")
        with self.assertRaises(ValueError):
            self.assembler.get_reg_num("ra")

        with self.assertRaises(ValueError):
            self.assembler.get_reg_num("a")

        with self.assertRaises(ValueError):
            self.assembler.get_reg_num("r17")

    def test_get_lui_pos(self):
        self.setAssembler()
        with self.assertRaises(ValueError):
            self.assembler.get_lui_pos('rb')

        with self.assertRaises(ValueError):
            self.assembler.get_lui_pos('5')

    def test_wait_negative(self):
        self.setAssembler()
        with self.assertRaises(ValueError):
            self.assembler.WaitFormat(["-11"])

    def test_program_length(self):
        self.setAssembler("LabelTest.qumis")
        self.assembler.convert_to_instructions()
        self.assertEqual(len(self.assembler.instructions),
                         len(self.assembler.getTextInstructions()))

    def test_remove_file_comment(self):
        self.setAssembler("LabelTest.qumis")
        self.assembler.get_valid_lines()
        self.assertEqual(len(self.assembler.valid_lines), 17)
        self.assertEqual(self.assembler.valid_lines[-1], 'nop')

    def test_parse_label(self):
        self.setAssembler("LabelTest.qumis")
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        for i, label_instr in enumerate(self.assembler.label_instrs):
            if (i == 4):
                self.assertEqual(label_instr[0], 'exp_start')
            else:
                self.assertEqual(label_instr[0], '')

    def test_end_of_file_loop(self):
        self.setAssembler()
        self.assembler.convert_to_instructions()
        self.assertEqual(self.assembler.instructions[-8], 3229615080)
        self.assertEqual(self.assembler.instructions[-7], 2692875240)

        if self.assembler.add_nop_after_label == True:
            self.assertEqual(self.assembler.instructions[-6], 302022652)
        else:
            self.assertEqual(self.assembler.instructions[-6], 302022653)
        self.assertEqual(self.assembler.instructions[-5], 0)
        self.assertEqual(self.assembler.instructions[-4], 0)
        self.assertEqual(self.assembler.instructions[-3], 0)
        self.assertEqual(self.assembler.instructions[-2], 0)
        self.assertEqual(self.assembler.instructions[-1], 0)

    def test_convert_line_to_ele_array(self):
        target_array = [['', 'mov', 'r0', '0'],
                        ['', 'mov', 'r1', '200'],
                        ['loop', 'waitreg', 'r1'],
                        ['', 'trigger', '1111111', '10'],
                        ['', 'pulse', '0000', '1001', '0000'],
                        ['', 'waitreg', 'r1'],
                        ['', 'beq', 'r0', 'r0', 'loop']]
        qumis_file_name = "ErrorCode.qumis"
        self.setAssembler(qumis_file_name)
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        self.assertEqual(target_array, self.assembler.label_instrs)

        addition_target_array = [['', 'mov', 'r0', '0'],
                                 ['', 'mov', 'r1', '100'],
                                 ['', 'mov', 'r2', '200'],
                                 ['', 'mov', 'r3', '400'],
                                 ['', 'mov', 'r4', '1000'],
                                 ['', 'mov', 'r5', '3000'],
                                 ['', 'mov', 'r6', '0'],
                                 ['', 'mov', 'r7', '0'],
                                 ['', 'mov', 'r8', '0'],
                                 ['', 'add', 'r6', 'r1', 'r2'],
                                 ['', 'add', 'r7', 'r6', 'r2'],
                                 ['start', 'wait', '100'],
                                 ['', 'addi', 'r8', 'r8', '1'],
                                 ['', 'beq', 'r0', 'r0', 'start'],
                                 ['', 'trigger', '1111111', '10']]
        qumis_file_name = "Addition.qumis"
        self.setAssembler(qumis_file_name)
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        self.assertEqual(addition_target_array, self.assembler.label_instrs)

    def test_insert_nops(self):
        Error_Code_target_array_no_nop = [['', 'mov', 'r0', '0'],
                                          ['', 'mov', 'r1', '200'],
                                          ['loop', 'waitreg', 'r1'],
                                          ['', 'trigger', '1111111', '10'],
                                          ['', 'pulse', '0000',
                                           '1001', '0000'],
                                          ['', 'waitreg', 'r1'],
                                          ['', 'beq', 'r0', 'r0', 'loop'],
                                          ['', 'nop'],
                                          ['', 'nop'],
                                          ['', 'nop'],
                                          ['', 'nop'],
                                          ['', 'nop']]
        Error_Code_target_array = [['', 'mov', 'r0', '0'],
                                   ['', 'mov', 'r1', '200'],
                                   ['loop', 'nop'],
                                   ['', 'waitreg', 'r1'],
                                   ['', 'trigger', '1111111', '10'],
                                   ['', 'pulse', '0000', '1001', '0000'],
                                   ['', 'waitreg', 'r1'],
                                   ['', 'beq', 'r0', 'r0', 'loop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop']]
        qumis_file_name = "ErrorCode.qumis"
        self.setAssembler(qumis_file_name)
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        self.assembler.insert_nops()
        if self.assembler.add_nop_after_label == True:
            self.assertEqual(Error_Code_target_array,
                             self.assembler.label_instrs)
        else:
            self.assertEqual(Error_Code_target_array_no_nop,
                             self.assembler.label_instrs)

        addition_target_array_no_nop = [['', 'mov', 'r0', '0'],
                                        ['', 'mov', 'r1', '100'],
                                        ['', 'mov', 'r2', '200'],
                                        ['', 'mov', 'r3', '400'],
                                        ['', 'mov', 'r4', '1000'],
                                        ['', 'mov', 'r5', '3000'],
                                        ['', 'mov', 'r6', '0'],
                                        ['', 'mov', 'r7', '0'],
                                        ['', 'mov', 'r8', '0'],
                                        ['', 'add', 'r6', 'r1', 'r2'],
                                        ['', 'add', 'r7', 'r6', 'r2'],
                                        ['start', 'wait', '100'],
                                        ['', 'addi', 'r8', 'r8', '1'],
                                        ['', 'beq', 'r0', 'r0', 'start'],
                                        ['', 'nop'],
                                        ['', 'nop'],
                                        ['', 'nop'],
                                        ['', 'nop'],
                                        ['', 'nop'],
                                        ['', 'trigger', '1111111', '10']]

        addition_target_array = [['', 'mov', 'r0', '0'],
                                 ['', 'mov', 'r1', '100'],
                                 ['', 'mov', 'r2', '200'],
                                 ['', 'mov', 'r3', '400'],
                                 ['', 'mov', 'r4', '1000'],
                                 ['', 'mov', 'r5', '3000'],
                                 ['', 'mov', 'r6', '0'],
                                 ['', 'mov', 'r7', '0'],
                                 ['', 'mov', 'r8', '0'],
                                 ['', 'add', 'r6', 'r1', 'r2'],
                                 ['', 'add', 'r7', 'r6', 'r2'],
                                 ['start', 'nop'],
                                 ['', 'wait', '100'],
                                 ['', 'addi', 'r8', 'r8', '1'],
                                 ['', 'beq', 'r0', 'r0', 'start'],
                                 ['', 'nop'],
                                 ['', 'nop'],
                                 ['', 'nop'],
                                 ['', 'nop'],
                                 ['', 'nop'],
                                 ['', 'trigger', '1111111', '10']]

        qumis_file_name = "Addition.qumis"
        self.setAssembler(qumis_file_name)
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        self.assembler.insert_nops()

        if self.assembler.add_nop_after_label == True:
            self.assertEqual(addition_target_array,
                             self.assembler.label_instrs)
        else:
            self.assertEqual(addition_target_array_no_nop,
                             self.assembler.label_instrs)

    def test_merge_wait(self):
        qumis_file_name = "ErrorCode.qumis"

        merge_wait_array = [['start', 'nop'],
                            ['', 'lui', 'r0', 0, 0],
                            ['', 'lui', 'r0', 1, 0],
                            ['', 'lui', 'r0', 2, 0],
                            ['', 'lui', 'r0', 3, 0],
                            ['round', 'nop'],
                            ['', 'wait', '25'],
                            ['', 'trigger', '0000001', '10'],
                            ['', 'add', 'r0', 'r1', 'r0'],
                            ['', 'wait', '20'],
                            ['', 'waitreg', 'r0'],
                            ['', 'trigger', '1111110', '30'],
                            ['', 'sub', 'r3', 'r2', 'r0'],
                            ['', 'wait', '10'],
                            ['test', 'nop'],
                            ['', 'wait', '5'],
                            ['', 'pulse', '0000', '0000', '0000'],
                            ['', 'wait', '31100'],
                            ['', 'wait', '20000'],
                            ['', 'waitreg', 'r3'],
                            ['end_of_file_loop', 'nop'],
                            ['', 'wait', '1000'],
                            ['', 'trigger', '0000001', '1000'],
                            ['', 'beq', 'r0', 'r0', 'end_of_file_loop'],
                            ['', 'nop'],
                            ['', 'nop'],
                            ['', 'nop'],
                            ['', 'nop'],
                            ['', 'nop']]
        merge_wait_array_no_nop = [['start', 'lui', 'r0', 0, 0],
                                   ['', 'lui', 'r0', 1, 0],
                                   ['', 'lui', 'r0', 2, 0],
                                   ['', 'lui', 'r0', 3, 0],
                                   ['round', 'wait', '25'],
                                   ['', 'trigger', '0000001', '10'],
                                   ['', 'add', 'r0', 'r1', 'r0'],
                                   ['', 'wait', '20'],
                                   ['', 'waitreg', 'r0'],
                                   ['', 'trigger', '1111110', '30'],
                                   ['', 'sub', 'r3', 'r2', 'r0'],
                                   ['', 'wait', '10'],
                                   ['test', 'wait', '5'],
                                   ['', 'pulse', '0000', '0000', '0000'],
                                   ['', 'wait', '31100'],
                                   ['', 'wait', '20000'],
                                   ['', 'waitreg', 'r3'],
                                   ['end_of_file_loop', 'wait', '1000'],
                                   ['', 'trigger', '0000001', '1000'],
                                   ['', 'beq', 'r0', 'r0', 'end_of_file_loop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop'],
                                   ['', 'nop']]

        qumis_file_name = "wait1.qumis"
        self.setAssembler(qumis_file_name)
        self.assembler.get_valid_lines()
        self.assembler.convert_line_to_ele_array()
        self.assembler.add_end_file_loop()
        self.assembler.remove_wait_zero()
        self.assembler.insert_nops()
        self.assembler.decompose()
        self.assembler.merge_consecutive_wait()

        if self.assembler.add_nop_after_label == True:
            self.assertEqual(merge_wait_array,
                             self.assembler.label_instrs)
        else:
            self.assertEqual(merge_wait_array_no_nop,
                             self.assembler.label_instrs)
