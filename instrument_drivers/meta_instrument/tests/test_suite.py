import unittest
lm = None


class LutManTests(unittest.TestCase):
    '''
    This is a test suite for the LutMan at this point it is still very primitive

    '''
    @classmethod
    def setUpClass(self):
        self.LutMan = lm
    # def test_get_all(self):
    #     CBox.get_all()
    #     return True
    def test_pars(self):
        for par in self.LutMan.parameters:
            self.LutMan.get(par)
    def test_uploading_waveforms(self):
        self.LutMan.CBox.set('acquisition_mode', 'idle')
        # self.LutMan.generate_standard_pulses()
        self.LutMan.load_pulse_onto_AWG_lookuptable('X180', 0, 0)
        self.LutMan.load_pulses_onto_AWG_lookuptable(0)
        self.LutMan.load_pulses_onto_AWG_lookuptable(1)
        self.LutMan.load_pulses_onto_AWG_lookuptable(2)
    # def test_firmware_version(self):
    #     v = CBox.get('firmware_version')
    #     self.assertTrue(int(v[1]) == 2)  # major version
    #     self.assertTrue(int(int(v[3:5])) > 13)  # minor version
