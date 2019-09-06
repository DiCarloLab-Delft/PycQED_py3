import unittest
import tempfile
import os
import numpy

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG


class Test_ZI_HDAWG8(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        print('Connecting...')
        cls.hd = HDAWG.ZI_HDAWG8(name='MOCK_HD', server='emulator',
                                 num_codewords=32, device='dev8026', interface='1GbE')

    @classmethod
    def teardown_class(cls):
        print('Disconnecting...')
        cls.hd.close()

    def test_instantiation(self):
        self.assertEqual(Test_ZI_HDAWG8.hd.devname, 'dev8026')

    def test_dynamic_waveform_upload(self):
        Test_ZI_HDAWG8.hd.system_clocks_referenceclock_source(1)
        Test_ZI_HDAWG8.hd.cfg_codeword_protocol('microwave')
        Test_ZI_HDAWG8.hd.upload_codeword_program()
        Test_ZI_HDAWG8.hd.start()

        # The program must be compiled exactly once at this point
        self.assertEqual(
            Test_ZI_HDAWG8.hd._awgModule.get_compilation_count(0), 1)

        # Modify a waveform
        Test_ZI_HDAWG8.hd.wave_ch1_cw000(
            0.5*Test_ZI_HDAWG8.hd.wave_ch1_cw000())

        # Start again
        Test_ZI_HDAWG8.hd.start()
        Test_ZI_HDAWG8.hd.stop()

        # No further compilation allowed
        self.assertEqual(
            Test_ZI_HDAWG8.hd._awgModule.get_compilation_count(0), 1)

        # Change the length of a waveform
        w0 = numpy.concatenate(
            (Test_ZI_HDAWG8.hd.wave_ch1_cw000(), Test_ZI_HDAWG8.hd.wave_ch1_cw000()))
        Test_ZI_HDAWG8.hd.wave_ch1_cw000(w0)

        # Start again
        Test_ZI_HDAWG8.hd.start()
        Test_ZI_HDAWG8.hd.stop()

        # Now the compilation must have been executed again
        self.assertEqual(
            Test_ZI_HDAWG8.hd._awgModule.get_compilation_count(0), 2)
