import unittest
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl


class Test_MW_LutMan(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.AWG = v8.VirtualAWG8('DummyAWG8')

        self.LutMan = mwl.AWG8_MW_LutMan('MW_LutMan')
        self.LutMan.AWG(self.AWG.name)

    def test_uploading_standard_pulses(self):
        self.LutMan.load_waveforms_onto_AWG_lookuptable()

    def test_lut_mapping(self):

        self.LutMan.I_channel(1)
        self.LutMan.Q_channel(1)
        self.LutMan.set_default_lutmap()
        expected_dict = {
            'rY90': ('wave_ch1_cw004',
                     'wave_ch1_cw004'),
            'I': ('wave_ch1_cw000',
                  'wave_ch1_cw000'),
            'rY180': ('wave_ch1_cw002',
                      'wave_ch1_cw002'),
            'rX180': ('wave_ch1_cw001',
                      'wave_ch1_cw001'),
            'rPhi90': ('wave_ch1_cw007',
                       'wave_ch1_cw007'),
            'rPhim90': ('wave_ch1_cw008',
                        'wave_ch1_cw008'),
            'rX90': ('wave_ch1_cw003',
                     'wave_ch1_cw003'),
            'rYm90': ('wave_ch1_cw006',
                      'wave_ch1_cw006'),
            'rXm90': ('wave_ch1_cw005',
                      'wave_ch1_cw005')}

        self.assertEqual(expected_dict, self.LutMan.LutMap())
