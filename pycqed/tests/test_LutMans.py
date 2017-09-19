import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf


class Test_MW_LutMan(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.AWG = v8.VirtualAWG8('DummyAWG8')

        self.LutMan = mwl.AWG8_MW_LutMan('MW_LutMan')
        self.LutMan.AWG(self.AWG.name)
        self.LutMan.I_channel(1)
        self.LutMan.Q_channel(2)
        self.LutMan.Q_modulation(100e6)
        self.LutMan.sampling_rate(2.4e9)

    def test_uploading_standard_pulses(self):
        # Tests that all waveforms are present and no error is raised.
        self.LutMan.load_waveforms_onto_AWG_lookuptable()
        expected_wf = wf.mod_gauss(
            amp=self.LutMan.Q_amp180(),
            sigma_length=self.LutMan.Q_gauss_width(),
            f_modulation=self.LutMan.Q_modulation(),
            sampling_rate=self.LutMan.sampling_rate(), phase=0,
            motzoi=self.LutMan.Q_motzoi())[0]

        uploaded_wf = self.AWG.get('wave_ch1_cw001')
        np.testing.assert_array_almost_equal(expected_wf, uploaded_wf)

    def test_lut_mapping(self):
        self.LutMan.set_default_lutmap()
        expected_dict = {
            'rY90': ('wave_ch1_cw004',
                     'wave_ch2_cw004'),
            'I': ('wave_ch1_cw000',
                  'wave_ch2_cw000'),
            'rY180': ('wave_ch1_cw002',
                      'wave_ch2_cw002'),
            'rX180': ('wave_ch1_cw001',
                      'wave_ch2_cw001'),
            'rPhi90': ('wave_ch1_cw007',
                       'wave_ch2_cw007'),
            'rPhim90': ('wave_ch1_cw008',
                        'wave_ch2_cw008'),
            'rX90': ('wave_ch1_cw003',
                     'wave_ch2_cw003'),
            'rYm90': ('wave_ch1_cw006',
                      'wave_ch2_cw006'),
            'rXm90': ('wave_ch1_cw005',
                      'wave_ch2_cw005')}

        self.assertDictEqual.__self__.maxDiff = None
        self.assertDictEqual(expected_dict, self.LutMan.LutMap())
