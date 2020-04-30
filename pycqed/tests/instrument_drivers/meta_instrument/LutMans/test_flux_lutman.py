import numpy as np
import pytest

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG
from pycqed.instrument_drivers.meta_instrument import lfilt_kernel_object as lko
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman as flm
from pycqed.instrument_drivers.meta_instrument.LutMans.base_lutman import get_wf_idx_from_name
from pycqed.instrument_drivers.virtual_instruments import sim_control_CZ as scCZ
from pycqed.measurement import measurement_control as mc
from qcodes import station as st
import pycqed.analysis.analysis_toolbox as a_tools


class TestMultiQubitFluxLutMan:

    @classmethod
    def setup_class(cls):
        # gets called at initialization of test class
        cls.AWG = HDAWG.ZI_HDAWG8(name='DummyAWG8', server='emulator',
                                  num_codewords=32, device='dev8026', interface='1GbE')

        cls.fluxlutman = flm.HDAWG_Flux_LutMan('fluxlutman_main')
        cls.k0 = lko.LinDistortionKernel('k0')
        cls.fluxlutman_partner = flm.HDAWG_Flux_LutMan('fluxlutman_partner')

        # cz sim related below
        cls.fluxlutman_static = flm.HDAWG_Flux_LutMan('fluxlutman_static')
        cls.sim_control_CZ_NE = scCZ.SimControlCZ(
            cls.fluxlutman.name + '_sim_control_CZ_NE')
        cls.sim_control_CZ_NW = scCZ.SimControlCZ(
            cls.fluxlutman.name + '_sim_control_CZ_NW')
        cls.sim_control_CZ_SE = scCZ.SimControlCZ(
            cls.fluxlutman.name + '_sim_control_CZ_SE')
        cls.sim_control_CZ_SW = scCZ.SimControlCZ(
            cls.fluxlutman.name + '_sim_control_CZ_SW')

    def setup_method(self, method):
        # gets called before every test method
        self.fluxlutman.instr_distortion_kernel(self.k0.name)
        self.k0.instr_AWG(self.AWG.name)

        self.k0.filter_model_00(
            {'model': 'exponential', 'params': {'tau': 1e-8, 'amp': -0.08},
             'real-time': False})
        self.k0.filter_model_01(
            {'model': 'exponential', 'params': {'tau': 6e-9, 'amp': -0.01},
             'real-time': False})
        self.k0.filter_model_02(
            {'model': 'exponential', 'params': {'tau': 1.8e-9, 'amp': -0.1},
             'real-time': False})
        self.k0.filter_model_03(
            {'model': 'exponential', 'params': {'tau': 1.e-9, 'amp': -0.1},
             'real-time': False})

        self.fluxlutman.AWG(self.AWG.name)
        self.fluxlutman.sampling_rate(2.4e9)
        self.fluxlutman.cz_theta_f_SE(80)

        self.fluxlutman.q_freq_01(6.8e9)
        self.fluxlutman.q_freq_10_SE(5.0e9)
        self.fluxlutman.q_freq_10_NE(5.5e9)
        self.fluxlutman.q_freq_10_SW(6.0e9)
        self.fluxlutman.q_freq_10_NW(6.3e9)
        self.fluxlutman.q_J2_SE(41e6)
        self.fluxlutman.cfg_awg_channel(1)

        self.k0.cfg_awg_channel(self.fluxlutman.cfg_awg_channel())
        self.fluxlutman.cfg_max_wf_length(5e-6)

        poly_coeffs = -np.array([1.95027142e+09, -3.22560292e+08,
                                 5.25834946e+07])
        self.fluxlutman.q_polycoeffs_freq_01_det(poly_coeffs)
        self.fluxlutman.q_polycoeffs_anharm(np.array([0, 0, -300e6]))

        self.fluxlutman.set_default_lutmap()
        self.fluxlutman.instr_partner_lutman('fluxlutman_partner')

        ################################################

        self.fluxlutman_partner.instr_distortion_kernel(self.k0.name)
        self.fluxlutman_partner.AWG(self.AWG.name)
        self.fluxlutman_partner.sampling_rate(2.4e9)
        self.fluxlutman_partner.cfg_awg_channel(2)
        self.fluxlutman_partner.cfg_max_wf_length(5e-6)
        self.fluxlutman_partner.set_default_lutmap()
        self.fluxlutman_partner.instr_partner_lutman('fluxlutman_main')

        self.AWG.awgs_0_outputs_0_amplitude(1)
        self.AWG.sigouts_0_range(5)

        self.fluxlutman.czd_amp_ratio_SE(1.)
        self.fluxlutman.czd_lambda_2_SE(np.nan)
        self.fluxlutman.czd_lambda_3_SE(np.nan)
        self.fluxlutman.czd_theta_f_SE(np.nan)

    @classmethod
    def teardown_class(self):
        for inststr in list(self.AWG._all_instruments):
            try:
                inst = self.AWG.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass

    def test_amp_to_dac_val_conversions(self):
        self.fluxlutman.cfg_awg_channel(1)

        self.AWG.awgs_0_outputs_0_amplitude(.5)
        self.AWG.sigouts_0_range(5)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        np.testing.assert_allclose(sf, 0.5 * 5 / 2)

        self.AWG.sigouts_0_range(.8)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        np.testing.assert_allclose(sf, 0.5 * 0.8 / 2)

        self.fluxlutman.cfg_awg_channel(2)
        self.AWG.awgs_0_outputs_1_amplitude(.2)
        self.AWG.sigouts_1_range(.8)
        sf = self.fluxlutman.get_dac_val_to_amp_scalefactor()
        np.testing.assert_allclose(sf, 0.2 * 0.8 / 2)

        sc_inv = self.fluxlutman.get_amp_to_dac_val_scalefactor()
        np.testing.assert_allclose(sc_inv, 1 / sf)

        self.fluxlutman.cfg_awg_channel(1)

    def test_partner_lutman_loading(self):
        self.fluxlutman.sq_amp(.3)
        self.fluxlutman_partner.sq_amp(.5)
        self.k0.reset_kernels()
        self.fluxlutman.load_waveform_onto_AWG_lookuptable(
            wave_id='square', regenerate_waveforms=True)
        self.fluxlutman_partner.load_waveform_onto_AWG_lookuptable(
            wave_id='square', regenerate_waveforms=True)
        cw = self.fluxlutman._get_cw_from_wf_name('square')
        np.testing.assert_allclose(self.AWG.get(
            'wave_ch{}_cw{:03}'.format(1, cw))[0], [.3])
        np.testing.assert_allclose(self.AWG.get(
            'wave_ch{}_cw{:03}'.format(2, cw))[0], [.5])

    def test_plot_level_diagram(self):
        self.AWG.awgs_0_outputs_0_amplitude(.73)
        self.fluxlutman.plot_level_diagram(show=False, which_gate='SE')

    def test_plot_cz_trajectory(self):
        self.fluxlutman.generate_standard_waveforms()
        self.fluxlutman.plot_cz_trajectory(show=False, which_gate='SE')

    def test_standard_cz_waveform(self):
        self.fluxlutman.czd_double_sided_SE(False)
        self.fluxlutman.generate_standard_waveforms()

    def test_double_sided_cz_waveform(self):
        """
        This test mostly tests if the parameters have some effect.
        They do not test the generated output.
        """
        self.fluxlutman.czd_double_sided_SE(True)
        self.fluxlutman.generate_standard_waveforms()

        czA = self.fluxlutman._wave_dict['cz_SE']
        self.fluxlutman.czd_amp_ratio_SE(1.1)
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_SE']
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(czA, czB)
        self.fluxlutman.czd_amp_ratio_SE(1.)

        czA = self.fluxlutman._wave_dict['cz_SE']
        self.fluxlutman.czd_length_ratio_SE(.6)
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_SE']
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_lambda_2_SE(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_SE']

        self.fluxlutman.czd_lambda_2_SE(self.fluxlutman.cz_lambda_2_SE())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_SE']
        np.testing.assert_array_equal(czA, czB)

        # until here
        self.fluxlutman.czd_lambda_2_SE(self.fluxlutman.cz_lambda_2_SE()+.05)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_SE']
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(czA, czC)

        self.fluxlutman.czd_lambda_3_SE(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_SE']

        self.fluxlutman.czd_lambda_3_SE(self.fluxlutman.cz_lambda_3_SE())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_SE']
        np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_lambda_3_SE(self.fluxlutman.cz_lambda_3_SE()+0.05)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_SE']
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(czA, czC)

        self.fluxlutman.czd_theta_f_SE(np.nan)
        self.fluxlutman.generate_standard_waveforms()
        czA = self.fluxlutman._wave_dict['cz_SE']

        self.fluxlutman.czd_theta_f_SE(self.fluxlutman.czd_theta_f_SE())
        self.fluxlutman.generate_standard_waveforms()
        czB = self.fluxlutman._wave_dict['cz_SE']
        np.testing.assert_array_equal(czA, czB)

        self.fluxlutman.czd_theta_f_SE(self.fluxlutman.cz_theta_f_SE()+15)
        self.fluxlutman.generate_standard_waveforms()
        czC = self.fluxlutman._wave_dict['cz_SE']
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(czA, czC)

    def test_calc_amp_to_freq_unknown_state(self):
        with pytest.raises(ValueError):
            self.fluxlutman.get_polycoeffs_state('22')

    def test_calc_amp_to_freq_01(self):
        """
        Tests methods used to determine energy levels and their conversion
        to amplitude
        """
        freq_01 = self.fluxlutman.calc_amp_to_freq(amp=0, state='01')
        freq_01_expected = self.fluxlutman.q_freq_01() + \
            self.fluxlutman.q_polycoeffs_freq_01_det()[2]
        np.testing.assert_allclose(freq_01, freq_01_expected)

    def test_calc_amp_to_freq_02(self):
        freq_02 = self.fluxlutman.calc_amp_to_freq(amp=0, state='02')
        freq_02_expected = \
            2*(self.fluxlutman.q_freq_01() +
               self.fluxlutman.q_polycoeffs_freq_01_det()[2]) + \
            self.fluxlutman.q_polycoeffs_anharm()[2]
        np.testing.assert_allclose(freq_02, freq_02_expected)

    def test_calc_amp_to_freq_10(self):
        freq_10 = self.fluxlutman.calc_amp_to_freq(amp=0, state='10')
        freq_10_expected = self.fluxlutman.q_freq_10_NE()
        np.testing.assert_allclose(freq_10, freq_10_expected)

        freq_10 = self.fluxlutman.calc_amp_to_freq(
            amp=0, state='10', which_gate='NW')
        freq_10_expected = self.fluxlutman.q_freq_10_NW()
        np.testing.assert_allclose(freq_10, freq_10_expected)

        freq_10 = self.fluxlutman.calc_amp_to_freq(
            amp=0, state='10', which_gate='SE')
        freq_10_expected = self.fluxlutman.q_freq_10_SE()
        np.testing.assert_allclose(freq_10, freq_10_expected)

        freq_10 = self.fluxlutman.calc_amp_to_freq(
            amp=0, state='10', which_gate='SW')
        freq_10_expected = self.fluxlutman.q_freq_10_SW()
        np.testing.assert_allclose(freq_10, freq_10_expected)

    def test_calc_amp_to_freq_11(self):
        freq_11 = self.fluxlutman.calc_amp_to_freq(amp=0, state='11')
        freq_11_expected = \
            (self.fluxlutman.q_freq_01() +
             self.fluxlutman.q_polycoeffs_freq_01_det()[2]) + \
            self.fluxlutman.q_freq_10_NE()

        np.testing.assert_allclose(freq_11, freq_11_expected)

    def test_calc_transition_freq_inversion(self):
        state = '02'
        amps = np.linspace(.3, 1, 11)
        freqs_02 = self.fluxlutman.calc_amp_to_freq(amp=amps,
                                                    state=state, which_gate='SE')
        amps_inv = self.fluxlutman.calc_freq_to_amp(freqs_02,
                                                    state=state,
                                                    which_gate='SE',
                                                    positive_branch=True)
        np.testing.assert_array_almost_equal(amps, amps_inv)

        amps = np.linspace(-.3, -1, 11)
        freqs_02 = self.fluxlutman.calc_amp_to_freq(amp=amps,
                                                    state=state, which_gate='SE')
        amps_inv = self.fluxlutman.calc_freq_to_amp(freqs_02, state=state,
                                                    which_gate='SE',
                                                    positive_branch=False)
        np.testing.assert_array_almost_equal(amps, amps_inv)

    def test_calc_amp_to_eps(self):
        state_A = '02'
        state_B = '11'
        amps = np.linspace(-1, 1, 11)
        eps = self.fluxlutman.calc_amp_to_eps(amp=amps, state_A=state_A,
                                              state_B=state_B,
                                              which_gate='SE')

        freqs_02 = self.fluxlutman.calc_amp_to_freq(amp=amps, state=state_A,
                                                    which_gate='SE')
        freqs_11 = self.fluxlutman.calc_amp_to_freq(amp=amps, state=state_B,
                                                    which_gate='SE')
        expected_eps = freqs_11 - freqs_02
        np.testing.assert_array_almost_equal(eps, expected_eps)

    def test_calc_detuning_freq_inversion(self):
        state_A = '02'
        state_B = '11'

        amps = np.linspace(.3, 1, 11)
        freqs_02 = self.fluxlutman.calc_amp_to_eps(
            amp=amps, state_A=state_A, state_B=state_B,
            which_gate='SE')
        amps_inv = self.fluxlutman.calc_eps_to_amp(
            freqs_02, state_A=state_A, state_B=state_B, positive_branch=True,
            which_gate='SE')
        np.testing.assert_array_almost_equal(amps, amps_inv)

        amps = np.linspace(-.3, -1, 11)
        freqs_02 = self.fluxlutman.calc_amp_to_eps(
            amp=amps, state_A=state_A, state_B=state_B,
            which_gate='SE')
        amps_inv = self.fluxlutman.calc_eps_to_amp(
            freqs_02, state_A=state_A, state_B=state_B, positive_branch=False,
            which_gate='SE')
        np.testing.assert_array_almost_equal(amps, amps_inv)

    def test_custom_wf(self):
        self.fluxlutman.generate_standard_waveforms()

        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'],
            np.array([]))

        # Tests if the custom wf is part of the default lutmap
        self.fluxlutman.load_waveforms_onto_AWG_lookuptable()
        assert 'custom_wf' in self.fluxlutman._wave_dict_dist

        x = np.arange(200)
        y = np.cos(x) / 20
        self.fluxlutman.custom_wf(y)
        self.fluxlutman.generate_standard_waveforms()
        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'], y)

        self.fluxlutman.custom_wf_length(30e-9)
        self.fluxlutman.generate_standard_waveforms()
        y_cut = np.cos(x) / 20
        cut_sample = 72  # 30ns * 2.4GSps
        y_cut[cut_sample:] = 0
        np.testing.assert_array_almost_equal(
            self.fluxlutman._wave_dict['custom_wf'], y_cut)
        # base waveform is not changed
        np.testing.assert_array_almost_equal(
            self.fluxlutman.custom_wf(), y)

    def test_generate_standard_flux_waveforms(self):
        self.fluxlutman.generate_standard_waveforms()

    def test_load_waveforms_onto_AWG_lookuptable(self):
        self.fluxlutman.cfg_distort(True)
        self.fluxlutman.load_waveforms_onto_AWG_lookuptable()
        self.fluxlutman.cfg_distort(False)
        self.fluxlutman.load_waveforms_onto_AWG_lookuptable()

    def test_length_ratio(self):
        self.fluxlutman.czd_length_ratio_SE(.5)
        lr = self.fluxlutman.calc_net_zero_length_ratio(which_gate='SE')
        np.testing.assert_allclose(lr, 0.5)

        self.fluxlutman.czd_length_ratio_SE('auto')

        amp_J2_pos = self.fluxlutman.calc_eps_to_amp(
            0, state_A='11', state_B='02', positive_branch=True, which_gate='SE')
        amp_J2_neg = self.fluxlutman.calc_eps_to_amp(
            0, state_A='11', state_B='02', positive_branch=False, which_gate='SE')
        lr = self.fluxlutman.calc_net_zero_length_ratio(which_gate='SE')
        integral = lr * amp_J2_pos + (1 - lr) * amp_J2_neg
        np.testing.assert_almost_equal(integral, 0)

    def test_czd_signs(self):
        # Only tests get setting and validator does not test functionality.
        self.fluxlutman.czd_double_sided_SE(True)
        signs = self.fluxlutman.czd_signs_SE()
        expected_signs = ['+', '-']
        assert signs == expected_signs

        with pytest.raises(Exception):
            self.fluxlutman.czd_signs_SE(['s', 1])

        self.fluxlutman.czd_signs_SE(['+', 0])
        signs = self.fluxlutman.czd_signs_SE()
        expected_signs = ['+', 0]
        assert signs == expected_signs
        self.fluxlutman.generate_standard_waveforms()

        signs = self.fluxlutman.czd_signs_SE(['+', '-'])

    def test_render_wave(self):
        self.fluxlutman.render_wave('cz_SE', time_units='lut_index')
        self.fluxlutman.render_wave('cz_SE', time_units='s')

    # [Victor, 2020-04-28] We are testing now the VCZ gate, this old
    # simulations are useless for now, not worth fixing tests

    # def test_sim_CZ_single(self):
    #     # The simplest use case: have only one
    #     # instr_sim_control_CZ_{some_gate} in the fluxlutman and being
    #     # able to run a simulation
    #     self.fluxlutman_static.q_polycoeffs_anharm(np.array([0, 0, -318e6]))
    #     self.fluxlutman.q_freq_01(6.87e9)
    #     self.fluxlutman.sampling_rate(2400000000.0)
    #     self.fluxlutman.q_polycoeffs_anharm(np.array([0, 0, -300e6]))
    #     self.fluxlutman.q_polycoeffs_freq_01_det(np.array([-2.5e9, 0, 0]))

    #     which_gate = 'SE'
    #     self.fluxlutman.set('cz_length_{}'.format(which_gate), 48e-9)
    #     self.fluxlutman.set('cz_lambda_2_{}'.format(which_gate), 0)
    #     self.fluxlutman.set('cz_lambda_3_{}'.format(which_gate), 0)
    #     self.fluxlutman.set('cz_length_{}'.format(which_gate), 48e-9)
    #     self.fluxlutman.set('cz_theta_f_{}'.format(which_gate), 100)
    #     self.fluxlutman.set('czd_double_sided_{}'.format(which_gate), True)
    #     self.fluxlutman.set('q_J2_{}'.format(which_gate), np.sqrt(2) * 14.3e6)
    #     self.fluxlutman.set('q_freq_10_{}'.format(which_gate), 5.79e9)
    #     self.fluxlutman.set('bus_freq_{}'.format(which_gate), 8.5e9)
    #     self.fluxlutman.set('czd_length_ratio_{}'.format(which_gate), 0.5)

    #     self.sim_control_CZ_SE.which_gate('SE')
    #     self.fluxlutman.set(
    #         'instr_sim_control_CZ_SE',
    #         self.sim_control_CZ_SE.name)

    #     values, units = self.fluxlutman.sim_CZ(
    #         fluxlutman_static=self.fluxlutman_static)

    #     np.testing.assert_almost_equal(values['Cond phase'], 340.1458978296672)
    #     np.testing.assert_almost_equal(values['L1'], 10.967187671584833)
    #     np.testing.assert_almost_equal(values['L2'], 8.773750137267944)

    #     assert 'L1' in units.keys()

    # def test_sim_CZ_multiple_per_flm(self):
    #     # being able to simulate any CZ gate from the same fluxlutman
    #     self.fluxlutman_static.q_polycoeffs_anharm(np.array([0, 0, -318e6]))
    #     self.fluxlutman.q_freq_01(6.87e9)
    #     self.fluxlutman.sampling_rate(2400000000.0)
    #     self.fluxlutman.q_polycoeffs_anharm(np.array([0, 0, -300e6]))
    #     self.fluxlutman.q_polycoeffs_freq_01_det(np.array([-2.5e9, 0, 0]))

    #     for which_gate in ['NE', 'NW', 'SW', 'SE']:
    #         self.fluxlutman.set('cz_length_{}'.format(which_gate), 48e-9)
    #         self.fluxlutman.set('cz_lambda_2_{}'.format(which_gate), 0)
    #         self.fluxlutman.set('cz_lambda_3_{}'.format(which_gate), 0)
    #         self.fluxlutman.set('cz_length_{}'.format(which_gate), 48e-9)
    #         self.fluxlutman.set('cz_theta_f_{}'.format(which_gate), 100)
    #         self.fluxlutman.set('czd_double_sided_{}'.format(which_gate), True)
    #         self.fluxlutman.set('q_J2_{}'.format(
    #             which_gate), np.sqrt(2) * 14.3e6)
    #         self.fluxlutman.set('q_freq_10_{}'.format(which_gate), 5.79e9)
    #         self.fluxlutman.set('bus_freq_{}'.format(which_gate), 8.5e9)
    #         self.fluxlutman.set('czd_length_ratio_{}'.format(which_gate), 0.5)

    #     # Because simulation is slow this equivalent tests are commented out
    #     # self.sim_control_CZ_NE.which_gate('NE')
    #     # self.fluxlutman.set(
    #     #     'instr_sim_control_CZ_NE',
    #     #     self.sim_control_CZ_NE.name)
    #     # values, units = self.fluxlutman.sim_CZ(
    #     #     fluxlutman_static=self.fluxlutman_static, which_gate='NE')
    #     # np.testing.assert_almost_equal(values['Cond phase'], 340.1458978296672)
    #     # np.testing.assert_almost_equal(values['L1'], 10.967187671584833)
    #     # np.testing.assert_almost_equal(values['L2'], 8.773750137267944)
    #     # assert 'L1' in units.keys()

    #     # self.sim_control_CZ_SE.which_gate('SE')
    #     # self.fluxlutman.set(
    #     #     'instr_sim_control_CZ_SE',
    #     #     self.sim_control_CZ_SE.name)
    #     # values, units = self.fluxlutman.sim_CZ(
    #     #     fluxlutman_static=self.fluxlutman_static, which_gate='SE')
    #     # np.testing.assert_almost_equal(values['Cond phase'], 340.1458978296672)
    #     # np.testing.assert_almost_equal(values['L1'], 10.967187671584833)
    #     # np.testing.assert_almost_equal(values['L2'], 8.773750137267944)
    #     # assert 'L1' in units.keys()

    #     # self.sim_control_CZ_NW.which_gate('NW')
    #     # self.fluxlutman.set(
    #     #     'instr_sim_control_CZ_NW',
    #     #     self.sim_control_CZ_NW.name)
    #     # values, units = self.fluxlutman.sim_CZ(
    #     #     fluxlutman_static=self.fluxlutman_static, which_gate='NW')
    #     # np.testing.assert_almost_equal(values['Cond phase'], 340.1458978296672)
    #     # np.testing.assert_almost_equal(values['L1'], 10.967187671584833)
    #     # np.testing.assert_almost_equal(values['L2'], 8.773750137267944)
    #     # assert 'L1' in units.keys()

    #     # self.sim_control_CZ_SW.which_gate('SW')
    #     # self.fluxlutman.set(
    #     #     'instr_sim_control_CZ_SW',
    #     #     self.sim_control_CZ_SW.name)
    #     # values, units = self.fluxlutman.sim_CZ(
    #     #     fluxlutman_static=self.fluxlutman_static, which_gate='SW')
    #     # np.testing.assert_almost_equal(values['Cond phase'], 340.1458978296672)
    #     # np.testing.assert_almost_equal(values['L1'], 10.967187671584833)
    #     # np.testing.assert_almost_equal(values['L2'], 8.773750137267944)
    #     # assert 'L1' in units.keys()

    # def test_simulate_cz_and_select_optima(self):
    #     """
    #     Test runs a small simulation of 6 datapoints and finds the optimum from
    #     this. Tests for the optimum being what is expected.
    #     """
    #     # Set up an experiment like environment
    #     self.station = st.Station()

    #     self.MC = mc.MeasurementControl('MC', live_plot_enabled=True)
    #     self.MC.station = self.station
    #     # Ensures datadir of experiment and analysis are identical
    #     self.MC.datadir(a_tools.datadir)

    #     self.station.add_component(self.MC)
    #     self.station.add_component(self.fluxlutman)
    #     self.station.add_component(self.fluxlutman_static)

    #     self.sim_control_CZ_SE.which_gate('SE')
    #     self.fluxlutman.set(
    #         'instr_sim_control_CZ_SE',
    #         self.sim_control_CZ_SE.name)
    #     print(self.sim_control_CZ_SE.name)
    #     self.station.add_component(self.sim_control_CZ_SE)

    #     # Set all the parameters in the fluxlutman from a particular saved
    #     # configuration
    #     # SE gate
    #     fluxlutman_pars = {
    #         'instr_distortion_kernel': 'lin_dist_kern_X',
    #         'instr_partner_lutman': 'flux_lm_Z1',
    #         '_awgs_fl_sequencer_program_expected_hash': 101,
    #         'idle_pulse_length': 4e-08,
    #         'czd_double_sided_NE': False,
    #         'disable_cz_only_z_NE': False,
    #         'cz_phase_corr_length_NE': 0e-9,
    #         'cz_phase_corr_amp_NE': 0.,
    #         'cz_length_NE': 5e-08,
    #         'cz_lambda_2_NE': 0,
    #         'cz_lambda_3_NE': 0,
    #         'cz_theta_f_NE': 80,
    #         'czd_amp_ratio_NE': 1,
    #         'czd_amp_offset_NE': 0,
    #         'czd_signs_NE': ['+', '-'],
    #         'czd_length_ratio_NE': 0.5,
    #         'czd_double_sided_NW': False,
    #         'disable_cz_only_z_NW': False,
    #         'cz_phase_corr_length_NW': 0e-9,
    #         'cz_phase_corr_amp_NW': 0.,
    #         'cz_length_NW': 3.5e-08,
    #         'cz_lambda_2_NW': 0,
    #         'cz_lambda_3_NW': 0,
    #         'cz_theta_f_NW': 80,
    #         'czd_amp_ratio_NW': 1,
    #         'czd_amp_offset_NW': 0,
    #         'czd_signs_NW': ['+', '-'],
    #         'czd_length_ratio_NW': 0.5,
    #         'czd_double_sided_SW': False,
    #         'disable_cz_only_z_SW': False,
    #         'cz_phase_corr_length_SW': 0e-9,
    #         'cz_phase_corr_amp_SW': 0.,
    #         'cz_length_SW': 5e-08,
    #         'cz_lambda_2_SW': -0.424605,
    #         'cz_lambda_3_SW': -0.050327,
    #         'cz_theta_f_SW': 66.7876,
    #         'czd_amp_ratio_SW': 1,
    #         'czd_amp_offset_SW': 0,
    #         'czd_signs_SW': ['+', '-'],
    #         'czd_length_ratio_SW': 0.5,
    #         'czd_double_sided_SE': True,
    #         'disable_cz_only_z_SE': False,
    #         'cz_phase_corr_length_SE': 0e-9,
    #         'cz_phase_corr_amp_SE': 0.,
    #         'cz_length_SE': 5e-08,
    #         'cz_lambda_2_SE': -0.16,
    #         'cz_lambda_3_SE': 0,
    #         'cz_theta_f_SE': 170.0,
    #         'czd_amp_ratio_SE': 1,
    #         'czd_amp_offset_SE': 0,
    #         'czd_signs_SE': ['+', '-'],
    #         'czd_length_ratio_SE': 0.5,
    #         'sq_amp': -0.5,
    #         'sq_length': 6e-08,
    #         'park_length': 4e-08,
    #         'park_amp': 0,
    #         'custom_wf': np.array([]),
    #         'custom_wf_length': np.inf,
    #         'LutMap': {
    #             0: {'name': 'i', 'type': 'idle'},
    #             1: {'name': 'cz_NE', 'type': 'idle_z', 'which': 'NE'},
    #             2: {'name': 'cz_SE', 'type': 'cz', 'which': 'SE'},
    #             3: {'name': 'cz_SW', 'type': 'cz', 'which': 'SW'},
    #             4: {'name': 'cz_NW', 'type': 'idle_z', 'which': 'NW'},
    #             5: {'name': 'park', 'type': 'square'},
    #             6: {'name': 'square', 'type': 'square'},
    #             7: {'name': 'custom_wf', 'type': 'custom'}},
    #         'sampling_rate': 2400000000.0,
    #         'q_polycoeffs_freq_01_det': np.array([-9.06217397e+08,
    #                                               -0, -1.92463273e-07]),
    #         'q_polycoeffs_anharm': np.array([0, 0, -3.18e+08]),
    #         'q_freq_01': 5886845171.848719,
    #         'q_freq_10_NE': 6000000000.0,
    #         'q_J2_NE': 15000000.0,
    #         'q_freq_10_NW': 6000000000.0,
    #         'q_J2_NW': 15000000.0,
    #         'q_freq_10_SW': 4560202554.51,
    #         'q_J2_SW': 13901719.318127526,
    #         'q_freq_10_SE': 4560202554.51,
    #         'q_J2_SE': 13901719.318127526,
    #         'bus_freq_SE': 27e9,
    #         'bus_freq_SW': 27e9,
    #         'bus_freq_NE': 27e9,
    #         'bus_freq_NW': 27e9
    #     }

    #     for par in fluxlutman_pars.keys():
    #         self.fluxlutman.set(par, fluxlutman_pars[par])

    #     self.fluxlutman_static.q_polycoeffs_anharm(np.array([0, 0, -3.e+8]))

    #     self.sim_control_CZ_SE.gates_num(1)
    #     self.sim_control_CZ_SE.gates_interval(20e-9)
    #     self.sim_control_CZ_SE.waiting_at_sweetspot(0)
    #     self.sim_control_CZ_SE.Z_rotations_length(0)

    #     self.fluxlutman.set('cz_lambda_3_SE', 0)
    #     self.fluxlutman.set('cz_length_SE', 50e-9)

    #     # Simulation runs here
    #     guesses = self.fluxlutman.simulate_cz_and_select_optima(
    #         fluxlutman_static=self.fluxlutman_static,
    #         MC=self.MC,
    #         which_gate='SE',
    #         n_points=10,
    #         theta_f_lims=(140, 155),
    #         lambda_2_lims=(-.15, 0.))
    #     first_optimal_pars = guesses[0][0]
    #     np.testing.assert_almost_equal(first_optimal_pars['cz_theta_f_SE'],
    #                                    116.6666, decimal=1)
    #     np.testing.assert_almost_equal(first_optimal_pars['cz_lambda_2_SE'],
    #                                    -0.23333, decimal=1)
