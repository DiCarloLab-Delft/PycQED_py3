from unittest import TestCase
import pycqed as pq
from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_qwg_cw_trigger(self):
        cw0_instr = ins_lib.qwg_cw_trigger(0)
        exp_cw0 = 'trigger 0000000, 1\n'
        exp_cw0 += 'wait 1\n'
        exp_cw0 += 'trigger 1000000, 2\n'
        exp_cw0 += 'wait 2\n'
        self.assertEqual(cw0_instr, exp_cw0)

        cw1_instr = ins_lib.qwg_cw_trigger(1)
        exp_cw1 = 'trigger 0100000, 1\n'
        exp_cw1 += 'wait 1\n'
        exp_cw1 += 'trigger 1100000, 2\n'
        exp_cw1 += 'wait 2\n'
        self.assertEqual(cw1_instr, exp_cw1)

        exp_cw2 = 'trigger 0010000, 1\n'
        exp_cw2 += 'wait 1\n'
        exp_cw2 += 'trigger 1010000, 2\n'
        exp_cw2 += 'wait 2\n'
        cw2_instr = ins_lib.qwg_cw_trigger(2)
        self.assertEqual(cw2_instr, exp_cw2)

        exp_cw3 = 'trigger 0110000, 1\n'
        exp_cw3 += 'wait 1\n'
        exp_cw3 += 'trigger 1110000, 2\n'
        exp_cw3 += 'wait 2\n'
        cw3_instr = ins_lib.qwg_cw_trigger(3)
        self.assertEqual(cw3_instr, exp_cw3)

        # Test if different channels are specified
        exp_cw3 = 'trigger 0011000, 1\n'
        exp_cw3 += 'wait 1\n'
        exp_cw3 += 'trigger 0011001, 2\n'
        exp_cw3 += 'wait 2\n'
        cw3_instr = ins_lib.qwg_cw_trigger(3, trigger_channel=7,
                                           cw_channels=[3, 4, 5])
        self.assertEqual(cw3_instr, exp_cw3)

    def test_cbox_awg_trigger(self):
        exp_cw = 'pulse 1010 0000 1010 \n'
        cw_instr = ins_lib.cbox_awg_trigger(codeword=2,
                                            awg_channels=[0, 2])
        self.assertEqual(exp_cw, cw_instr)

        exp_cw = 'pulse 0000 1001 0000 \n'
        cw_instr = ins_lib.cbox_awg_trigger(codeword=1,
                                            awg_channels=[1])
        self.assertEqual(exp_cw, cw_instr)

    def test_channel_trigger(self):
        instr = ins_lib.trigg_ch_to_instr(channel=3, duration=5)
        exp_instr = 'trigger 0010000, 5 \nwait 5\n'
        self.assertEqual(exp_instr, instr)




