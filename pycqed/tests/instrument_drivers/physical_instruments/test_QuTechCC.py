import unittest
from os.path import join
from os import makedirs

from pycqed.instrument_drivers.physical_instruments.Transport import FileTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC


class Test_QutechCC(unittest.TestCase):
    def test_all(self):
        fn = 'Test_QutechCC_test_all.scpi.txt'
        test_path = join('test_output',fn)
        makedirs('test_output', exist_ok=True)

        transport = FileTransport(test_path)
        cc = QuTechCC('cc', transport)

        cc.reset()
        cc.clear_status()
        cc.set_status_questionable_frequency_enable(0x7FFF)

        cc.dio0_out_delay(0)
        cc.dio0_out_delay(31)
        cc.dio8_out_delay(0)
        cc.dio8_out_delay(31)

        cc.vsm_channel_delay0(0)
        cc.vsm_channel_delay0(1)
        cc.vsm_channel_delay0(127)
        cc.vsm_channel_delay31(0)
        cc.vsm_channel_delay31(127)

        cc.vsm_rise_delay0(0)
        cc.vsm_rise_delay0(48)
        cc.vsm_rise_delay31(0)
        cc.vsm_rise_delay31(48)
        cc.vsm_fall_delay0(0)
        cc.vsm_fall_delay0(48)
        cc.vsm_fall_delay31(0)
        cc.vsm_fall_delay31(48)

        cc.debug_marker_in(0, cc.UHFQA_TRIG)
        cc.debug_marker_in(0, cc.UHFQA_CW[0])

        cc.debug_marker_out(0, cc.UHFQA_TRIG)
        cc.debug_marker_out(8, cc.HDAWG_TRIG)

        if 0:
            cc.sequence_program(prog)
            cc.start()

        if 0:
            cc.stop()

        transport.close()  # to allow access to file

        # check results
        test_output = open(test_path).read()
        golden = open(join('golden',fn)).read()
        self.assertEqual(test_output, golden)
