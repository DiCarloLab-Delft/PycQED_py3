import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
from pycqed.instrument_drivers.meta_instrument import lfilt_kernel_object as lko

from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman as flm
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.instrument_drivers.meta_instrument.LutMans.base_lutman import \
    get_redundant_codewords


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

    def test__program_hash_differs_AWG8_lutman(self):

        # set to a random value to ensure different
        self.AWG8_MW_LutMan._awgs_mw_sequencer_program_expected_hash(351340)
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertTrue(hash_differs)

        self.AWG8_MW_LutMan._update_expected_program_hash()
        hash_differs = self.AWG8_MW_LutMan._program_hash_differs()
        self.assertFalse(hash_differs)

    def test__program_hash_differs_AWG8_VSM_lutman(self):

        # set to a random value to ensure different
        self.AWG8_VSM_MW_LutMan._awgs_mwG_sequencer_program_expected_hash(
            351340)
        hash_differs = self.AWG8_VSM_MW_LutMan._program_hash_differs()
        self.assertTrue(hash_differs)

        self.AWG8_VSM_MW_LutMan._update_expected_program_hash()
        hash_differs = self.AWG8_VSM_MW_LutMan._program_hash_differs()
        self.assertFalse(hash_differs)

    def test__program_hash_updated_when_loading_program(self):
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

    def test_lut_mapping_AWG8(self):
        self.AWG8_MW_LutMan.set_default_lutmap()
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
            'rX90': ('wave_ch1_cw003',
                     'wave_ch2_cw003'),
            'rYm90': ('wave_ch1_cw006',
                      'wave_ch2_cw006'),
            'rXm90': ('wave_ch1_cw005',
                      'wave_ch2_cw005'),
            'spec': ('wave_ch1_cw008',
                     'wave_ch2_cw008')}
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
            'rY90': ('wave_ch1_cw004',
                     'wave_ch2_cw004',
                     'wave_ch3_cw004',
                     'wave_ch4_cw004'),
            'I': ('wave_ch1_cw000',
                  'wave_ch2_cw000',
                  'wave_ch3_cw000',
                  'wave_ch4_cw000'),
            'rY180': ('wave_ch1_cw002',
                      'wave_ch2_cw002',
                      'wave_ch3_cw002',
                      'wave_ch4_cw002'),
            'rX180': ('wave_ch1_cw001',
                      'wave_ch2_cw001',
                      'wave_ch3_cw001',
                      'wave_ch4_cw001'),
            'rPhi90': ('wave_ch1_cw007',
                       'wave_ch2_cw007',
                       'wave_ch3_cw007',
                       'wave_ch4_cw007'),
            'rX90': ('wave_ch1_cw003',
                     'wave_ch2_cw003',
                     'wave_ch3_cw003',
                     'wave_ch4_cw003'),
            'rYm90': ('wave_ch1_cw006',
                      'wave_ch2_cw006',
                      'wave_ch3_cw006',
                      'wave_ch4_cw006'),
            'rXm90': ('wave_ch1_cw005',
                      'wave_ch2_cw005',
                      'wave_ch3_cw005',
                      'wave_ch4_cw005'),
            'spec': ('wave_ch1_cw008',
                     'wave_ch2_cw008',
                     'wave_ch3_cw008',
                     'wave_ch4_cw008')}
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

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.AWG._all_instruments):
            try:
                inst = self.AWG.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass


class Test_Flux_LutMan(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.AWG = v8.VirtualAWG8('DummyAWG8')

        self.fluxlutman = flm.AWG8_Flux_LutMan('fluxlutman_main')
        self.k0 = lko.LinDistortionKernel('k0')
        self.fluxlutman.instr_distortion_kernel(self.k0.name)


        self.fluxlutman.AWG(self.AWG.name)
        self.fluxlutman.sampling_rate(2.4e9)
        self.fluxlutman.cz_theta_f(80)
        self.fluxlutman.cz_freq_01_max(6.8e9)
        self.fluxlutman.cz_J2(4.1e6)
        self.fluxlutman.cfg_awg_channel(1)
        # self.fluxlutman.cz_E_c(250e6)
        self.fluxlutman.cz_freq_interaction(5.1e9)
        self.fluxlutman.cfg_max_wf_length(5e-6)

        poly_coeffs = np.array([1.95027142e+09,  -3.22560292e+08,
                                5.25834946e+07])
        self.fluxlutman.polycoeffs_freq_conv(poly_coeffs)
        self.fluxlutman.set_default_lutmap()
        self.fluxlutman.instr_partner_lutman('fluxlutman_partner')

        ################################################

        self.fluxlutman_partner = flm.AWG8_Flux_LutMan('fluxlutman_partner')
        self.fluxlutman_partner.instr_distortion_kernel(self.k0.name)
        self.fluxlutman_partner.AWG(self.AWG.name)
        self.fluxlutman_partner.sampling_rate(2.4e9)
        self.fluxlutman_partner.cz_theta_f(80)
        self.fluxlutman_partner.cz_freq_01_max(6.8e9)
        self.fluxlutman_partner.cz_J2(4.1e6)
        self.fluxlutman_partner.cfg_awg_channel(2)

        self.fluxlutman_partner.cz_freq_interaction(5.1e9)
        self.fluxlutman_partner.cfg_max_wf_length(5e-6)

        poly_coeffs = np.array([1.95027142e+09,  -3.22560292e+08,
                                5.25834946e+07])
        self.fluxlutman_partner.polycoeffs_freq_conv(poly_coeffs)
        self.fluxlutman_partner.set_default_lutmap()
        self.fluxlutman_partner.instr_partner_lutman('fluxlutman_main')


    def test__program_hash_differs_AWG8_flux_lutman(self):

        # set to a random value to ensure different
        self.fluxlutman._awgs_fl_sequencer_program_expected_hash(351340)
        hash_differs = self.fluxlutman._program_hash_differs()
        self.assertTrue(hash_differs)

        self.fluxlutman._update_expected_program_hash()
        hash_differs = self.fluxlutman._program_hash_differs()
        self.assertFalse(hash_differs)

    def test_amp_to_dac_val_conversions(self):
        self.fluxlutman.cfg_awg_channel(1)

        self.AWG.awgs_0_outputs_0_amplitude(.5)
        self.AWG.sigouts_0_range(5)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        self.assertEqual(sf, 0.5*5/2)

        self.AWG.sigouts_0_range(.8)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        self.assertEqual(sf, 0.5*0.8/2)

        self.fluxlutman.cfg_awg_channel(2)
        self.AWG.awgs_0_outputs_1_amplitude(.2)
        self.AWG.sigouts_1_range(.8)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        self.assertEqual(sf, 0.2*0.8/2)

        sc_inv = self.fluxlutman.get_amp_to_dac_val_scale_factor()
        self.assertEqual(sc_inv, 1/sf)

        self.fluxlutman.cfg_awg_channel(1)

    def test_partner_lutman_loading(self):
        self.fluxlutman.sq_amp(.3)
        self.fluxlutman_partner.sq_amp(.5)
        self.fluxlutman.load_waveform_realtime('square')

        self.assertEqual(self.AWG._realtime_w0[0], [.3])
        self.assertEqual(self.AWG._realtime_w1[0], [.5])

    # Commented out because this mode is deprecated
    # def test_operating_mode_program_loading(self):
    #     self.fluxlutman.cfg_operating_mode('Codeword_normal')
    #     self.fluxlutman.load_waveforms_onto_AWG_lookuptable()

    #     self.fluxlutman.cfg_operating_mode('CW_single_02')
    #     self.fluxlutman.load_waveforms_onto_AWG_lookuptable()

    #     self.fluxlutman.cfg_operating_mode('Codeword_normal')
    #     self.fluxlutman.load_waveforms_onto_AWG_lookuptable()


    def test_plot_flux_arc(self):
        self.fluxlutman.plot_flux_arc(show=False, plot_cz_trajectory=True)

    def test_plot_cz_trajectory(self):
        self.fluxlutman.plot_cz_trajectory(show=False)

    def test_standard_cz_waveform(self):
        self.fluxlutman.czd_double_sided(False)
        self.fluxlutman.generate_standard_waveforms()

    def test_double_sided_cz_waveform(self):
        """
        This test mostly tests if the parameters have some effect.
        They do not test the generated output.
        """
        self.fluxlutman.czd_double_sided(True)
        self.fluxlutman.generate_standard_waveforms()

        czA = self.fluxlutman._wave_dict['cz_z']
        self.fluxlutman.czd_amp_ratio(1.1)
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 czA, czB)
        self.fluxlutman.czd_amp_ratio(1.)

        czA = self.fluxlutman._wave_dict['cz_z']
        self.fluxlutman.czd_length_ratio(.6)
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 czA, czB)

        self.fluxlutman.czd_lambda_2(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_z']

        self.fluxlutman.czd_lambda_2(self.fluxlutman.cz_lambda_2())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_lambda_2(self.fluxlutman.cz_lambda_2()+.05)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 czA, czC)

        self.fluxlutman.czd_lambda_3(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_z']

        self.fluxlutman.czd_lambda_3(self.fluxlutman.cz_lambda_3())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_lambda_3(self.fluxlutman.cz_lambda_3()+0.05)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 czA, czC)

        self.fluxlutman.czd_theta_f(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_z']

        self.fluxlutman.czd_theta_f(self.fluxlutman.czd_theta_f())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_theta_f(self.fluxlutman.cz_theta_f()+15)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_z']
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 czA, czC)

    def test_freq_amp_conversions(self):

        # Test the basic inversion
        test_amps = np.linspace(0.1, .5, 11)
        freqs = self.fluxlutman.amp_to_detuning(test_amps)
        recovered_amps = self.fluxlutman.detuning_to_amp(freqs)
        np.testing.assert_array_almost_equal(test_amps, recovered_amps)

        # Test that the top of the parabola is given if asked for "impossible"
        # solutions
        recovered_amp = self.fluxlutman.detuning_to_amp(0)
        self.assertAlmostEqual(recovered_amp, 0.082696256708720065)
        recovered_amp = self.fluxlutman.detuning_to_amp(-5e9)
        self.assertAlmostEqual(recovered_amp, 0.082696256708720065)

        # Test negative branch of parabola
        test_amps = np.linspace(-0.1, -.5, 11)
        freqs = self.fluxlutman.amp_to_detuning(test_amps)
        recovered_amps = self.fluxlutman.detuning_to_amp(
            freqs, positive_branch=False)
        np.testing.assert_array_almost_equal(test_amps, recovered_amps)

    def test_custom_wf(self):
        self.fluxlutman.generate_standard_waveforms()

        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'],
            np.array([]))

        # Tests if the custom wf is part of the default lutmap
        self.fluxlutman.load_waveforms_onto_AWG_lookuptable()
        self.assertIn('custom_wf', self.fluxlutman._wave_dict_dist)

        x = np.arange(200)
        y = np.cos(x)/20
        self.fluxlutman.custom_wf(y)
        self.fluxlutman.generate_standard_waveforms()
        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'], y)

        self.fluxlutman.custom_wf_length(30e-9)
        self.fluxlutman.generate_standard_waveforms()
        y_cut = np.cos(x)/20
        cut_sample = 72  # 30ns * 2.4GSps
        y_cut[cut_sample:] = 0
        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'], y_cut)
        # base waveform is not changed
        np.testing.assert_array_almost_equal(
            self.fluxlutman.custom_wf(), y)

    def test_generate_standard_flux_waveforms(self):
        self.fluxlutman.generate_standard_waveforms()
        self.fluxlutman.render_wave('cz')

    def test_upload_and_distort(self):
        self.fluxlutman.load_waveforms_onto_AWG_lookuptable()

    def test_generate_composite(self):
        self.fluxlutman.generate_standard_waveforms()
        gen_wf = self.fluxlutman._gen_composite_wf('cz_z', time_tuples=[])
        exp_wf = np.zeros(12000)  # 5us *2.4GSps
        np.testing.assert_array_almost_equal(gen_wf, exp_wf)
        flux_tuples = [(0, 'fl_cw_01', {(2, 0)}, 323),
                       (14, 'fl_cw_01', {(2, 0)}, 326),
                       (28, 'fl_cw_01', {(2, 0)}, 329),
                       (50, 'fl_cw_01', {(2, 0)}, 340),
                       (64, 'fl_cw_01', {(2, 0)}, 343),
                       (82, 'fl_cw_01', {(2, 0)}, 350),
                       (98, 'fl_cw_01', {(2, 0)}, 355),
                       (116, 'fl_cw_01', {(2, 0)}, 362),
                       (136, 'fl_cw_01', {(2, 0)}, 371),
                       (155, 'fl_cw_01', {(2, 0)}, 379),
                       (174, 'fl_cw_01', {(2, 0)}, 387)]

        gen_wf = self.fluxlutman._gen_composite_wf(
            'cz_z', time_tuples=flux_tuples)
        # not testing for equality to some expected stuff here, prolly better

    def test_uploading_composite_waveform(self):
        self.fluxlutman.generate_standard_waveforms()
        flux_tuples = [(0, 'fl_cw_01', {(2, 0)}, 323),
                       (14, 'fl_cw_01', {(2, 0)}, 326),
                       (28, 'fl_cw_01', {(2, 0)}, 329),
                       (50, 'fl_cw_01', {(2, 0)}, 340),
                       (64, 'fl_cw_01', {(2, 0)}, 343),
                       (82, 'fl_cw_01', {(2, 0)}, 350),
                       (98, 'fl_cw_01', {(2, 0)}, 355),
                       (116, 'fl_cw_01', {(2, 0)}, 362),
                       (136, 'fl_cw_01', {(2, 0)}, 371),
                       (155, 'fl_cw_01', {(2, 0)}, 379),
                       (174, 'fl_cw_01', {(2, 0)}, 387)]

        # Testing several different base waveforms
        for prim_wf in ['cz_z', 'square', 'idle_z']:
            self.fluxlutman.load_composite_waveform_onto_AWG_lookuptable(
                prim_wf, time_tuples=flux_tuples, codeword=3)
            direct_gen_wf = self.fluxlutman._gen_composite_wf(
                prim_wf, time_tuples=flux_tuples)
            wave_dict_wf = self.fluxlutman._wave_dict[
                'comp_{}_cw003'.format(prim_wf)]
            np.testing.assert_array_almost_equal(direct_gen_wf, wave_dict_wf)

            uploaded_wf_lutman = self.fluxlutman._wave_dict_dist[
                'comp_{}_cw003'.format(prim_wf)]
            uploaded_wf_instr = self.AWG.wave_ch1_cw003()
            np.testing.assert_array_almost_equal(uploaded_wf_lutman,
                                                 uploaded_wf_instr)

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


def dict_contained_in(subset, superset):
    if subset.items() <= superset.items():
        return True
    else:
        raise ValueError
