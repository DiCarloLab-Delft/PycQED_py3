
import unittest
import numpy as np

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF


class Test_UHFQC(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.uhf = UHF.UHFQC(name='MOCK_UHF', server='emulator',
                            device='dev2109', interface='1GbE')

        cls.uhf.reset_waveforms_zeros()

    @classmethod
    def teardown_class(cls):
        cls.uhf.close()

    def test_instantiation(self):
        self.assertEqual(Test_UHFQC.uhf.devname, 'dev2109')

    def test_DIO_program(self):
        self.uhf.awg_sequence_acquisition_and_DIO_triggered_pulse(cases=[
                                                                  0, 2, 14])
        self.uhf.start()
        uploaded_program = self.uhf._awgModule._sourcestring
        p = uploaded_program.split('\n')
        assert len(p) == 52  # program is 52 lines
        # Test that the codeword preamble is identical
        assert p[:8] == \
            ['// Start of automatically generated codeword table',
             'wave wave_ch1_cw000 = "dev2109_wave_ch1_cw000";',
             'wave wave_ch2_cw000 = "dev2109_wave_ch2_cw000";',
             'wave wave_ch1_cw002 = "dev2109_wave_ch1_cw002";',
             'wave wave_ch2_cw002 = "dev2109_wave_ch2_cw002";',
             'wave wave_ch1_cw014 = "dev2109_wave_ch1_cw014";',
             'wave wave_ch2_cw014 = "dev2109_wave_ch2_cw014";',
             '// End of automatically generated codeword table']

        assert p[39:45] == [
            '    switch (cw) {',
            '        case 0x00000000: playWave(wave_ch1_cw000, wave_ch2_cw000);',
            '        case 0x00040000: playWave(wave_ch1_cw002, wave_ch2_cw002);',
            '        case 0x001c0000: playWave(wave_ch1_cw014, wave_ch2_cw014);',
            '        default: playWave(ones(32), ones(32)); err_cnt += 1;',
            '    }']

    def test_waveform_table_generation(self):
        self.uhf.awg_sequence_acquisition_and_DIO_triggered_pulse(
            cases=[0, 2, 14])
        assert self.uhf.cases() == [0, 2, 14]
        wf_table = self.uhf._get_waveform_table(0)
        assert wf_table == [('wave_ch1_cw000', 'wave_ch2_cw000'),
                            ('wave_ch1_cw002', 'wave_ch2_cw002'),
                            ('wave_ch1_cw014', 'wave_ch2_cw014')]

    @unittest.skip('FIXME: disabled, see PR #643 and PR #635 (marked as important)')
    def test_dynamic_waveform_upload(self):
        Test_UHFQC.uhf.wave_ch1_cw000(np.ones(48))

        # resetting the compilation count to ensure test is self contained
        Test_UHFQC.uhf._awgModule._compilation_count[0] = 0
        Test_UHFQC.uhf.awg_sequence_acquisition_and_pulse()
        Test_UHFQC.uhf.start()
        Test_UHFQC.uhf.stop()

        # The program must be compiled exactly once at this point
        self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 1)

        # Modify a waveform
        Test_UHFQC.uhf.wave_ch1_cw000(0.5*Test_UHFQC.uhf.wave_ch1_cw000())

        # Start again
        Test_UHFQC.uhf.start()
        Test_UHFQC.uhf.stop()

        # No further compilation allowed
        self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 1)

        # Change the length of a waveform
        w0 = np.concatenate(
            (Test_UHFQC.uhf.wave_ch1_cw000(), Test_UHFQC.uhf.wave_ch1_cw000()))
        Test_UHFQC.uhf.wave_ch1_cw000(w0)

        # Start again
        Test_UHFQC.uhf.start()
        Test_UHFQC.uhf.stop()

        # Now the compilation must have been executed again
        self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 2)

    def test_reset_waveforms_zeros(self):
        self.uhf.wave_ch1_cw003(np.ones(80))
        assert np.allclose(self.uhf.wave_ch1_cw003(), np.ones(80))
        self.uhf.reset_waveforms_zeros()
        assert np.allclose(self.uhf.wave_ch1_cw003(), np.zeros(48))

    def test_close_open(self):
        # Close the instrument, then reopen to make sure that we can reconnect
        Test_UHFQC.uhf.close()
        self.setup_class()
        self.assertEqual(Test_UHFQC.uhf.devname, 'dev2109')

    def test_async(self):
        self.uhf.awgs_0_userregs_0(0)
        self.uhf.awgs_0_triggers_0_level(0.0)
        self.uhf.asyncBegin()
        self.uhf.awgs_0_userregs_0(100)
        self.uhf.awgs_0_triggers_0_level(1.123)
        assert self.uhf.awgs_0_userregs_0() == 0
        assert self.uhf.awgs_0_triggers_0_level() == 0
        self.uhf.asyncEnd()
        assert self.uhf.awgs_0_userregs_0() == 100
        assert self.uhf.awgs_0_triggers_0_level() == 1.123
