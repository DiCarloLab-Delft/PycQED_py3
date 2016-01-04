import unittest
import qt
from modules.utilities import general as gen


class QuTech_ControlBox_LookuptableManager_tests(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox_LookuptableManager
    Instrument.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        if self.CBox_lut_man is None:
            self.CBox_lut_man = qt.instruments.create(
                'CBox_lut_man', 'QuTech_ControlBox_LookuptableManager')
        self.CBox_lut_man.reload()
        # load some arbitrary standard settings


    def test_instrument_is_proxy(self):
        self.assertEqual(self.CBox_lut_man.__class__.__name__, 'Proxy',
                         'Instrument is not a Proxy, failure in ins creation')

    def test_generate_standard_pulses(self):
        self.CBox_lut_man.generate_standard_pulses()

    def test_uploading_pulses(self):
        self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'Y180', 'X90', 'Y90',
                                          'Block', 'X180_delayed'])
        self.CBox_lut_man.generate_standard_pulses()
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(2)

    def test_CLEAR_pulse_lenght_constraints(self):
        with self.assertRaises(ValueError):
            self.CBox_lut_man.set_M_delay(5)
            self.CBox_lut_man.set_M_length(6)
            self.CBox_lut_man.set_M_length_CLEAR_unc(5)
            self.CBox_lut_man.generate_standard_pulses()
        with self.assertRaises(ValueError):
            self.CBox_lut_man.set_M_delay(5)
            self.CBox_lut_man.set_M_length(128*5)
            self.CBox_lut_man.set_M_length_CLEAR_unc(5)
            self.CBox_lut_man.generate_standard_pulses()
        # if lenghts add up to multiple of 3*5 ns again all should be ok
        self.CBox_lut_man.set_M_delay(5)
        self.CBox_lut_man.set_M_length(5)
        self.CBox_lut_man.set_M_length_CLEAR_unc(5)
        self.CBox_lut_man.generate_standard_pulses()


    def test_tng_pulses(self):
        '''
        Tests if the parameters specified in prepare for CLEAR raise no errors
        '''
        self.CBox_lut_man.set_lut_mapping(['M0', 'M1', 'M2', 'M3_a',
                                           'M3_b', 'I', 'I'])

        self.CBox_lut_man.set_f_modulation(-0.01)
        self.CBox_lut_man.set_M_amp(300)
        self.CBox_lut_man.set_M_amp_CLEAR_1(300)
        self.CBox_lut_man.set_M_amp_CLEAR_2(300)
        self.CBox_lut_man.set_M_amp_CLEAR_a1(400)
        self.CBox_lut_man.set_M_amp_CLEAR_a2(400)
        self.CBox_lut_man.set_M_amp_CLEAR_b1(200)
        self.CBox_lut_man.set_M_amp_CLEAR_b2(200)
        self.CBox_lut_man.set_M_phase_CLEAR_1(0)
        self.CBox_lut_man.set_M_phase_CLEAR_2(0)
        self.CBox_lut_man.set_M_phase_CLEAR_a1(0)
        self.CBox_lut_man.set_M_phase_CLEAR_a2(0)
        self.CBox_lut_man.set_M_phase_CLEAR_b1(0)
        self.CBox_lut_man.set_M_phase_CLEAR_b2(0)
        self.CBox_lut_man.set_motzoi_parameter(0)
        self.CBox_lut_man.set_gauss_width(10)
        self.CBox_lut_man.set_M_delay(45)
        self.CBox_lut_man.set_M_length(1200)
        self.CBox_lut_man.set_M_length_CLEAR_unc(300)
        self.CBox_lut_man.set_M_length_CLEAR_c(300)

        self.CBox_lut_man.generate_standard_pulses()
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(2)

    def test_tng_pulses_multiple_3(self):
        self.CBox_lut_man.set_lut_mapping(['M0', 'M1', 'M2', 'M3_a',
                                           'M3_b', 'I', 'I'])
        self.CBox_lut_man.generate_standard_pulses()
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)

    @classmethod
    def tearDownClass(self):
        '''
        Restores original settings to the CBox_lut_man
        '''
        gen.load_settings_onto_instrument(self.CBox_lut_man)



if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        QuTech_ControlBox_LookuptableManager_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
