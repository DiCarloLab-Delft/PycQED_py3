import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8

from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.instrument_drivers.meta_instrument.LutMans.base_lutman import \
    get_redundant_codewords, get_wf_idx_from_name


class Test_MW_LutMan(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.AWG = v8.VirtualAWG8('DummyAWG8')

        self.AWG8_MW_LutMan = mwl.AWG8_MW_LutMan('MW_LutMan')
        self.AWG8_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_MW_LutMan.channel_I(1)
        self.AWG8_MW_LutMan.channel_Q(2)
        self.AWG8_MW_LutMan.mw_modulation(100e6)
        self.AWG8_MW_LutMan.sampling_rate(2.4e9)
        self.AWG8_MW_LutMan.set_default_lutmap()

        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)
        self.AWG8_VSM_MW_LutMan.set_default_lutmap()

        self.CBox_MW_LutMan = mwl.CBox_MW_LutMan('CBox_MW_LutMan')
        self.QWG_MW_LutMan = mwl.QWG_MW_LutMan('QWG_MW_LutMan')

    def test_program_hash_differs_AWG8_lutman(self):

        # set to a random value to ensure different
        self.AWG8_MW_LutMan._awgs_mw_sequencer_program_expected_hash(351340)
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertTrue(hash_differs)

        self.AWG8_MW_LutMan._update_expected_program_hash()
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertFalse(hash_differs)

    def test_program_hash_differs_AWG8_VSM_lutman(self):

        # set to a random value to ensure different
        self.AWG8_VSM_MW_LutMan._awgs_mwG_sequencer_program_expected_hash(
            351340)
        hash_differs = self.AWG8_VSM_MW_LutMan._program_hash_differs()
        self.assertTrue(hash_differs)

        self.AWG8_VSM_MW_LutMan._update_expected_program_hash()
        hash_differs = self.AWG8_VSM_MW_LutMan._program_hash_differs()
        self.assertFalse(hash_differs)

    def test_program_hash_updated_when_loading_program(self):
        self.AWG8_MW_LutMan._awgs_mw_sequencer_program_expected_hash(351340)
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertTrue(hash_differs)

        self.AWG8_MW_LutMan.load_waveforms_onto_AWG_lookuptable()
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertFalse(hash_differs)

    def test_uploading_standard_pulses(self):
        # Tests that all waveforms are present and no error is raised.
        self.AWG8_MW_LutMan.load_waveforms_onto_AWG_lookuptable()
        expected_wf = wf.mod_gauss(
            amp=self.AWG8_MW_LutMan.mw_amp180(),
            sigma_length=self.AWG8_MW_LutMan.mw_gauss_width(),
            f_modulation=self.AWG8_MW_LutMan.mw_modulation(),
            sampling_rate=self.AWG8_MW_LutMan.sampling_rate(), phase=0,
            motzoi=self.AWG8_MW_LutMan.mw_motzoi())[0]

        uploaded_wf = self.AWG.get('wave_ch1_cw001')
        np.testing.assert_array_almost_equal(expected_wf, uploaded_wf)

        expected_wf_spec = wf.block_pulse(
            length=self.AWG8_MW_LutMan.spec_length(),
            amp=self.AWG8_MW_LutMan.spec_amp(),
            sampling_rate=self.AWG8_MW_LutMan.sampling_rate(),
            delay=0, phase=0)[0]
        uploaded_wf = self.AWG.get('wave_ch1_cw008')
        np.testing.assert_array_almost_equal(expected_wf_spec, uploaded_wf)

    def test_generating_standard_pulses(self):
        """Test if standard waveforms are correctly generated."""

        self.AWG8_MW_LutMan.LutMap(mwl.default_mw_lutmap)
        print(self.AWG8_MW_LutMan.LutMap())
        self.AWG8_MW_LutMan.generate_standard_waveforms()

        # remove this line later
        self.AWG8_MW_LutMan.set_default_lutmap()

        expected_wf = wf.mod_gauss(
            amp=self.AWG8_MW_LutMan.mw_amp180(),
            sigma_length=self.AWG8_MW_LutMan.mw_gauss_width(),
            f_modulation=self.AWG8_MW_LutMan.mw_modulation(),
            sampling_rate=self.AWG8_MW_LutMan.sampling_rate(),
            phase=0,
            motzoi=self.AWG8_MW_LutMan.mw_motzoi())[0]
        # expected on cw 1 based on LutMap
        generated_wf = self.AWG8_MW_LutMan._wave_dict[1]
        np.testing.assert_array_almost_equal(expected_wf, generated_wf[0])

        generated_wf = self.AWG8_MW_LutMan._wave_dict[8]
        expected_wf_spec = wf.block_pulse(
            length=self.AWG8_MW_LutMan.spec_length(),
            amp=self.AWG8_MW_LutMan.spec_amp(),
            sampling_rate=self.AWG8_MW_LutMan.sampling_rate(),
            delay=0, phase=0)[0]
        np.testing.assert_array_almost_equal(expected_wf_spec, generated_wf[0])

    def test_codeword_idx_to_parnames(self):

        parnames = self.AWG8_MW_LutMan.codeword_idx_to_parnames(3)
        expected_parnames = ['wave_ch1_cw003', 'wave_ch2_cw003']
        self.assertEqual(parnames, expected_parnames)

        parnames = self.AWG8_VSM_MW_LutMan.codeword_idx_to_parnames(3)
        expected_parnames = ['wave_ch1_cw003', 'wave_ch2_cw003',
                             'wave_ch3_cw003', 'wave_ch4_cw003']
        self.assertEqual(parnames, expected_parnames)

    def test_lut_mapping_AWG8(self):
        self.AWG8_MW_LutMan.set_default_lutmap()
        expected_dict = {
            0: {"name": "I",        "theta": 0, "phi": 0, "type": "ge"},
            1: {"name": "rX180",    "theta": 180, "phi": 0, "type": "ge"}, }
        # Does not check the full lutmap
        dict_contained_in(expected_dict, self.AWG8_MW_LutMan.LutMap())

    def test_lut_mapping_CBox(self):
        self.CBox_MW_LutMan.set_default_lutmap()
        expected_dict = {'I': 0,
                         'rX180': 1,
                         'rY180': 2,
                         'rX90': 3,
                         'rY90': 4,
                         'rXm90': 5,
                         'rYm90': 6,
                         'rPhi90': 7}

        self.assertDictEqual.__self__.maxDiff = None
        self.assertDictEqual(expected_dict, self.CBox_MW_LutMan.LutMap())

    def test_lut_mapping_AWG8_VSM(self):
        self.AWG8_VSM_MW_LutMan.set_default_lutmap()
        expected_dict = {
            0: {"name": "I",        "theta": 0, "phi": 0, "type": "ge"},
            1: {"name": "rX180",    "theta": 180, "phi": 0, "type": "ge"}, }
        # Does not check the full lutmap
        dict_contained_in(expected_dict, self.AWG8_VSM_MW_LutMan.LutMap())

    def test_realtime_loading_square_wf_AWG8_VSM(self):
        self.AWG8_VSM_MW_LutMan.load_waveform_realtime('square', wf_nr=3)

    def test_uploading_standard_pulses_AWG8_VSM(self):
        # Tests that all waveforms are present and no error is raised.
        self.AWG8_VSM_MW_LutMan.load_waveforms_onto_AWG_lookuptable()
        expected_wfs = wf.mod_gauss_VSM(
            amp=self.AWG8_VSM_MW_LutMan.mw_amp180(),
            sigma_length=self.AWG8_VSM_MW_LutMan.mw_gauss_width(),
            f_modulation=self.AWG8_VSM_MW_LutMan.mw_modulation(),
            sampling_rate=self.AWG8_VSM_MW_LutMan.sampling_rate(), phase=0,
            motzoi=self.AWG8_VSM_MW_LutMan.mw_motzoi())

        for i in range(4):
            uploaded_wf = self.AWG.get('wave_ch{}_cw001'.format(i+1))
            np.testing.assert_array_almost_equal(expected_wfs[i], uploaded_wf)

    def test_load_ef_rabi_pulses_to_AWG_lookuptable_correct_pars(self):
        self.AWG8_VSM_MW_LutMan.load_ef_rabi_pulses_to_AWG_lookuptable()

        ef_pulse_pars = self.AWG8_VSM_MW_LutMan.LutMap()[9]
        self.assertEqual(ef_pulse_pars['type'], 'raw-drag')
        exp_amp = self.AWG8_VSM_MW_LutMan.mw_ef_amp180()
        self.assertEqual(ef_pulse_pars['drag_pars']['amp'], exp_amp)

        amps = [.1, .2, .5]
        self.AWG8_VSM_MW_LutMan.load_ef_rabi_pulses_to_AWG_lookuptable(
            amps=amps)
        for i, exp_amp in enumerate(amps):
            ef_pulse_pars = self.AWG8_VSM_MW_LutMan.LutMap()[i+9]
            self.assertEqual(ef_pulse_pars['type'], 'raw-drag')
            self.assertEqual(ef_pulse_pars['drag_pars']['amp'], exp_amp)

    def test_load_ef_rabi_pulses_to_AWG_lookuptable_correct_waveform(self):
        self.AWG8_VSM_MW_LutMan.load_ef_rabi_pulses_to_AWG_lookuptable()

        expected_wf = wf.mod_gauss(
            amp=self.AWG8_MW_LutMan.mw_ef_amp180(),
            sigma_length=self.AWG8_MW_LutMan.mw_gauss_width(),
            f_modulation=self.AWG8_MW_LutMan.mw_ef_modulation(),
            sampling_rate=self.AWG8_MW_LutMan.sampling_rate(), phase=0,
            motzoi=self.AWG8_MW_LutMan.mw_motzoi())[0]

        uploaded_wf = self.AWG.get('wave_ch1_cw009')
        np.testing.assert_array_almost_equal(expected_wf, uploaded_wf)

    def test_render_wave(self):
        self.AWG8_VSM_MW_LutMan.render_wave('rX180', show=False)

    def test_render_wave_PSD(self):
        self.AWG8_VSM_MW_LutMan.render_wave_PSD('rX180', show=False)


    @classmethod
    def tearDownClass(self):
        for inststr in list(self.AWG._all_instruments):
            try:
                inst = self.AWG.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass


class Test_LutMan_Utils(unittest.TestCase):
    def test_get_redundant_codewords(self):
        target_cw = 5
        red_cws_A = get_redundant_codewords(target_cw, 4, 0)
        for cw in red_cws_A:
            print(bin(cw))
            self.assertEqual(cw & 15, target_cw)
        self.assertEqual(len(red_cws_A), 2**4)

        red_cws_B = get_redundant_codewords(target_cw, 4, 4)
        for cw in red_cws_B:
            print(bin(cw))
            self.assertEqual((cw & (256-16)) >> 4, target_cw)
        self.assertEqual(len(red_cws_B), 2**4)

    def test_valid_mw_lutmap(self):

        valid_mw_lutmap = {
            0: {"name": "I",        "theta": 0, "phi": 0, "type": "ge"},
            1: {"name": "rX180",    "theta": 180, "phi": 0, "type": "ge"}, }
        self.assertTrue(mwl.mw_lutmap_is_valid(valid_mw_lutmap))

        invalid_mw_lutmap = {
            "0": {"name": "I",        "theta": 0, "phi": 0, "type": "ge"}, }
        with self.assertRaises(TypeError):
            mwl.mw_lutmap_is_valid(invalid_mw_lutmap)

    def test_theta_to_amp(self):
        ref_amp180 = 1.5
        thetas = [0, 180, 270, -90, -180]
        expected_amps = [0, 1.5, -.75, -.75, 1.5]
        for theta, exp_amp in zip(thetas, expected_amps):
            self.assertEqual(
                mwl.theta_to_amp(theta, ref_amp180), exp_amp)

    def test_get_wf_idx_from_name(self):
        idx = get_wf_idx_from_name('rX12', mwl.default_mw_lutmap)
        self.assertEqual(idx, 9)




def dict_contained_in(subset, superset):
    if subset.items() <= superset.items():
        return True
    else:
        raise ValueError
