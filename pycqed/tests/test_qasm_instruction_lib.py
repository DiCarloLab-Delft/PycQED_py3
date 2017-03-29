from unittest import TestCase
import pycqed as pq
from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_qwg_cw_trigger(self):
        cw0_instr = ins_lib.qwg_cw_trigger(0)
        exp_cw0 = 'trigger 1000000, 1\n'
        exp_cw0 += 'wait 1\n'
        exp_cw0 += 'trigger 1000000, 2\n'
        exp_cw0 += 'wait 2\n'
        self.assertEqual(cw0_instr, exp_cw0)

        cw1_instr = ins_lib.qwg_cw_trigger(1)
        exp_cw1 = 'trigger 1000000, 1\n'
        exp_cw1 += 'wait 1\n'
        exp_cw1 += 'trigger 1100000, 2\n'
        exp_cw1 += 'wait 2\n'
        self.assertEqual(cw1_instr, exp_cw1)

        exp_cw2 = 'trigger 1000000, 1\n'
        exp_cw2 += 'wait 1\n'
        exp_cw2 += 'trigger 1010000, 2\n'
        exp_cw2 += 'wait 2\n'
        cw2_instr = ins_lib.qwg_cw_trigger(2)
        self.assertEqual(cw2_instr, exp_cw2)

        exp_cw3 = 'trigger 1000000, 1\n'
        exp_cw3 += 'wait 1\n'
        exp_cw3 += 'trigger 1110000, 2\n'
        exp_cw3 += 'wait 2\n'
        cw3_instr = ins_lib.qwg_cw_trigger(3)
        self.assertEqual(cw3_instr, exp_cw3)

        # Test if different channels are specified
        exp_cw3 = 'trigger 0000001, 1\n'
        exp_cw3 += 'wait 1\n'
        exp_cw3 += 'trigger 0011001, 2\n'
        exp_cw3 += 'wait 2\n'
        cw3_instr = ins_lib.qwg_cw_trigger(3, trigger_channel=7,
                                           cw_channels=[3, 4, 5])
        self.assertEqual(cw3_instr, exp_cw3)
