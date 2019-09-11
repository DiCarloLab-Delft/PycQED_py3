import unittest
import numpy as np
import os
import pycqed as pq
import time
import openql
import warnings
import pycqed.analysis.analysis_toolbox as a_tools

import pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound as sh
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG

from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer


from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon import Tektronix_driven_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CC_transmon import CBox_v3_driven_transmon, QWG_driven_transmon
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL
from pycqed.instrument_drivers.physical_instruments.QuTech_VSM_Module import Dummy_QuTechVSMModule
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

Dummy_VSM_not_fixed = False


class Test_QO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.CCL_qubit = ct.CCLight_Transmon('CCL_qubit')

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')
        self.SH = sh.virtual_SignalHound_USB_SA124B('SH')
        self.UHFQC = UHF.UHFQC(name='UHFQC', server='emulator',
                               device='dev2109', interface='1GbE')

        self.CCL = dummy_CCL('CCL')
        # self.VSM = Dummy_Duplexer('VSM')
        self.VSM = Dummy_QuTechVSMModule('VSM')

        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        # Required to set it to the testing datadir
        test_datadir = os.path.join(pq.__path__[0], 'tests', 'test_output')
        self.MC.datadir(test_datadir)
        a_tools.datadir = self.MC.datadir()

        self.AWG = HDAWG.ZI_HDAWG8(name='DummyAWG8', server='emulator', num_codewords=32, device='dev8026', interface='1GbE')
        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        self.ro_lutman = UHFQC_RO_LutMan(
            'RO_lutman', num_res=5, feedline_number=0)
        self.ro_lutman.AWG(self.UHFQC.name)

        # Assign instruments
        self.CCL_qubit.instr_LutMan_MW(self.AWG8_VSM_MW_LutMan.name)
        self.CCL_qubit.instr_LO_ro(self.MW1.name)
        self.CCL_qubit.instr_LO_mw(self.MW2.name)
        self.CCL_qubit.instr_spec_source(self.MW3.name)

        self.CCL_qubit.instr_acquisition(self.UHFQC.name)
        self.CCL_qubit.instr_VSM(self.VSM.name)
        self.CCL_qubit.instr_CC(self.CCL.name)
        self.CCL_qubit.instr_LutMan_RO(self.ro_lutman.name)
        self.CCL_qubit.instr_MC(self.MC.name)

        self.CCL_qubit.instr_SH(self.SH.name)

        config_fn = os.path.join(
            pq.__path__[0], 'tests', 'openql', 'test_cfg_CCL.json')
        self.CCL_qubit.cfg_openql_platform_fn(config_fn)

        # Setting some "random" initial parameters
        self.CCL_qubit.ro_freq(5.43e9)
        self.CCL_qubit.ro_freq_mod(200e6)

        self.CCL_qubit.freq_qubit(4.56e9)
        self.CCL_qubit.freq_max(4.62e9)

        self.CCL_qubit.mw_freq_mod(-100e6)
        self.CCL_qubit.mw_awg_ch(1)
        self.CCL_qubit.cfg_qubit_nr(0)

        self.CCL_qubit.mw_vsm_delay(15)

        self.CCL_qubit.mw_mixer_offs_GI(.1)
        self.CCL_qubit.mw_mixer_offs_GQ(.2)
        self.CCL_qubit.mw_mixer_offs_DI(.3)
        self.CCL_qubit.mw_mixer_offs_DQ(.4)

    def test_instantiate_QuDevTransmon(self):
        QDT = QuDev_transmon('QuDev_transmon',
                             MC=None, heterodyne_instr=None, cw_source=None)
        QDT.close()

    def test_instantiate_TekTransmon(self):
        TT = Tektronix_driven_transmon('TT')
        TT.close()

    def test_instantiate_CBoxv3_transmon(self):
        CT = CBox_v3_driven_transmon('CT')
        CT.close()

    def test_instantiate_QWG_transmon(self):
        QT = QWG_driven_transmon('QT')
        QT.close()

    ##############################################
    # calculate methods
    ##############################################
    def test_calc_freq(self):
        self.CCL_qubit.cfg_qubit_freq_calc_method('latest')
        self.CCL_qubit.calculate_frequency()
        self.CCL_qubit.cfg_qubit_freq_calc_method('flux')
        self.CCL_qubit.calculate_frequency()

    ##############################################
    # basic prepare methods
    ##############################################

    def test_prep_for_continuous_wave(self):
        self.CCL_qubit.ro_acq_weight_type('optimal')
        with warnings.catch_warnings(record=True) as w:
            self.CCL_qubit.prepare_for_continuous_wave()
            self.assertEqual(str(w[0].message), 'Changing ro_acq_weight_type to SSB.')
        self.CCL_qubit.ro_acq_weight_type('SSB')
        self.CCL_qubit.prepare_for_continuous_wave()

    @unittest.skipIf(True, 'Test for use with an old duplexer.')
    def test_prep_cw_config_vsm(self):

        self.CCL_qubit.spec_vsm_ch_in(2)
        self.CCL_qubit.spec_vsm_ch_out(1)
        self.CCL_qubit.spec_vsm_amp(0.5)

        self.CCL_qubit.prepare_for_continuous_wave()

        self.assertEqual(self.VSM.in1_out1_switch(), 'OFF')
        self.assertEqual(self.VSM.in1_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_switch(), 'EXT')
        self.assertEqual(self.VSM.in2_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_amp(), 0.5)

    def test_prep_for_fluxing(self):
        self.CCL_qubit.prepare_for_fluxing()

    @unittest.skip('Not Implemented')
    def test_prep_flux_bias(self):
        raise NotImplementedError()

    ##############################################
    # Testing prepare for readout
    ##############################################
    def test_prep_readout(self):
        self.CCL_qubit.prepare_readout()

    def test_prep_ro_instantiate_detectors(self):
        self.MC.soft_avg(1)

        self.CCL_qubit.ro_soft_avg(4)
        detector_attributes = [
            'int_avg_det', 'int_log_det', 'int_avg_det_single',
            'input_average_detector']
        for det_attr in detector_attributes:
            if hasattr(self.CCL_qubit, det_attr):
                delattr(self.CCL_qubit, det_attr)
        # Test there are no detectors to start with
        for det_attr in detector_attributes:
            self.assertFalse(hasattr(self.CCL_qubit, det_attr))
        self.CCL_qubit.prepare_readout()
        # Test that the detectors have been instantiated
        for det_attr in detector_attributes:
            self.assertTrue(hasattr(self.CCL_qubit, det_attr))

        self.assertEqual(self.MC.soft_avg(), 4)

    def test_prep_ro_MW_sources(self):
        LO = self.CCL_qubit.instr_LO_ro.get_instr()
        LO.off()
        LO.frequency(4e9)
        LO.power(10)
        self.assertEqual(LO.status(), 'off')
        self.assertEqual(LO.frequency(), 4e9)
        self.CCL_qubit.mw_pow_td_source(20)

        self.CCL_qubit.ro_freq(5.43e9)
        self.CCL_qubit.ro_freq_mod(200e6)
        self.CCL_qubit.prepare_readout()

        self.assertEqual(LO.status(), 'on')
        self.assertEqual(LO.frequency(), 5.43e9-200e6)
        self.assertEqual(LO.power(), 20)

    def test_prep_ro_pulses(self):
        self.CCL_qubit.ro_pulse_mixer_alpha(1.1)
        self.CCL_qubit.ro_pulse_mixer_phi(4)
        self.CCL_qubit.ro_pulse_length(312e-9)
        self.CCL_qubit.ro_pulse_down_amp0(.1)
        self.CCL_qubit.ro_pulse_down_length0(23e-9)

        self.CCL_qubit.ro_pulse_mixer_offs_I(.01)
        self.CCL_qubit.ro_pulse_mixer_offs_Q(.02)

        self.CCL_qubit.prepare_readout()

        self.assertEqual(self.ro_lutman.mixer_phi(), 4)
        self.assertEqual(self.ro_lutman.mixer_alpha(), 1.1)
        self.assertEqual(self.ro_lutman.M_length_R0(), 312e-9)
        self.assertEqual(self.ro_lutman.M_down_length0_R0(), 23e-9)
        self.assertEqual(self.ro_lutman.M_down_amp0_R0(), .1)

        self.assertEqual(self.UHFQC.sigouts_0_offset(), .01)
        self.assertEqual(self.UHFQC.sigouts_1_offset(), .02)

    def test_prep_ro_integration_weigths(self):
        IF = 50e6
        self.CCL_qubit.ro_freq_mod(IF)
        self.CCL_qubit.ro_acq_weight_chI(3)
        self.CCL_qubit.ro_acq_weight_chQ(4)

        # Testing SSB
        trace_length = 4096
        self.CCL_qubit.ro_acq_weight_type('SSB')
        self.CCL_qubit.prepare_readout()
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))

        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 1 + 1j)
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), 1 - 1j)

        uploaded_wf = self.UHFQC.qas_0_integration_weights_3_real()
        np.testing.assert_array_almost_equal(cosI, uploaded_wf)
        # Testing DSB case
        self.CCL_qubit.ro_acq_weight_type('DSB')
        self.CCL_qubit.prepare_readout()
        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 2)
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), 2)

        # Testing Optimal weight uploading
        test_I = np.ones(10)
        test_Q = 0.5*test_I
        self.CCL_qubit.ro_acq_weight_func_I(test_I)
        self.CCL_qubit.ro_acq_weight_func_Q(test_Q)

        self.CCL_qubit.ro_acq_weight_type('optimal')
        self.CCL_qubit.prepare_readout()

        self.UHFQC.qas_0_rotations_4(.21 + 0.108j)
        upl_I = self.UHFQC.qas_0_integration_weights_3_real()
        upl_Q = self.UHFQC.qas_0_integration_weights_3_imag()
        np.testing.assert_array_almost_equal(test_I, upl_I)
        np.testing.assert_array_almost_equal(test_Q, upl_Q)
        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 1 - 1j)
        # These should not have been touched by optimal weights
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), .21 + .108j)

        self.CCL_qubit.ro_acq_weight_type('SSB')

    ########################################################
    #          Test prepare for timedomain                 #
    ########################################################
    def test_prep_for_timedomain(self):
        self.CCL_qubit.prepare_for_timedomain()

    def test_prep_td_sources(self):

        self.MW1.off()
        self.MW2.off()
        self.CCL_qubit.freq_qubit(4.56e9)
        self.CCL_qubit.mw_freq_mod(-100e6)
        self.CCL_qubit.mw_pow_td_source(13)

        self.CCL_qubit.prepare_for_timedomain()
        self.assertEqual(self.MW1.status(), 'on')
        self.assertEqual(self.MW2.status(), 'on')
        self.assertEqual(self.MW2.frequency(), 4.56e9 + 100e6)
        self.assertEqual(self.MW2.power(), 13)

    def test_prep_td_pulses(self):
        self.CCL_qubit.mw_awg_ch(5)
        self.CCL_qubit.mw_G_mixer_alpha(1.02)
        self.CCL_qubit.mw_D_mixer_phi(8)

        self.CCL_qubit.mw_mixer_offs_GI(.1)
        self.CCL_qubit.mw_mixer_offs_GQ(.2)
        self.CCL_qubit.mw_mixer_offs_DI(.3)
        self.CCL_qubit.mw_mixer_offs_DQ(.4)

        self.CCL_qubit.mw_ef_amp(.34)
        self.CCL_qubit.mw_freq_mod(-100e6)
        self.CCL_qubit.anharmonicity(-235e6)

        self.CCL_qubit.prepare_for_timedomain()
        self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_GI(), 5)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_GQ(), 6)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_DI(), 7)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_DQ(), 8)

        self.assertEqual(self.AWG8_VSM_MW_LutMan.G_mixer_alpha(), 1.02)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.D_mixer_phi(), 8)

        self.assertEqual(self.CCL.vsm_channel_delay0(),
                         self.CCL_qubit.mw_vsm_delay())

        self.assertEqual(self.AWG.sigouts_4_offset(), .1)
        self.assertEqual(self.AWG.sigouts_5_offset(), .2)
        self.assertEqual(self.AWG.sigouts_6_offset(), .3)
        self.assertEqual(self.AWG.sigouts_7_offset(), .4)

        self.assertEqual(self.AWG8_VSM_MW_LutMan.mw_ef_amp180(), .34)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.mw_ef_modulation(), -335e6)

    def test_prep_td_config_vsm(self):
        self.CCL_qubit.mw_vsm_G_amp(0.8)
        self.CCL_qubit.mw_vsm_D_phase(0)
        self.CCL_qubit.mw_vsm_ch_in(2)
        self.CCL_qubit.mw_vsm_mod_out(5)
        self.CCL_qubit.prepare_for_timedomain()

        self.assertEqual(self.VSM.mod5_ch2_gaussian_amp(), 0.8)
        self.assertEqual(self.VSM.mod5_ch2_derivative_phase(), 0)

    ###################################################
    #          Test basic experiments                 #
    ###################################################
    def test_cal_mixer_offsets_drive(self):
        self.CCL_qubit.calibrate_mixer_offsets_drive()

    def test_resonator_spec(self):
        self.CCL_qubit.ro_acq_weight_type('SSB')

        # set to not set to bypass validator
        self.CCL_qubit.freq_res._save_val(None)
        try:
            self.CCL_qubit.find_resonator_frequency()
        except ValueError:
            pass  # Fit can fail because testing against random data
        self.CCL_qubit.freq_res(5.4e9)
        try:
            self.CCL_qubit.find_resonator_frequency()
        except ValueError:
            pass  # Fit can fail because testing against random data
        freqs = np.linspace(6e9, 6.5e9, 31)

        self.CCL_qubit.measure_heterodyne_spectroscopy(freqs=freqs,
                                                       analyze=False)

    def test_resonator_power(self):
        self.CCL_qubit.ro_acq_weight_type('SSB')
        freqs = np.linspace(6e9, 6.5e9, 31)
        powers = np.arange(-30, -10, 5)

        # set to not set to bypass validator
        self.CCL_qubit.freq_res._save_val(None)
        self.CCL_qubit.measure_resonator_power(freqs=freqs, powers=powers)

    def test_measure_transients(self):
        self.CCL_qubit.ro_acq_input_average_length(2e-6)
        self.CCL_qubit.measure_transients()

    @unittest.skip('OpenQL bug for CCL config')
    def test_qubit_spec(self):
        freqs = np.linspace(6e9, 6.5e9, 31)
        # Data cannot be analyzed as dummy data is just random numbers
        self.CCL_qubit.measure_spectroscopy(freqs=freqs, analyze=False)

    def test_find_qubit_freq(self):
        self.CCL_qubit.cfg_qubit_freq_calc_method('latest')
        try:
            self.CCL_qubit.find_frequency()
        except TypeError:
            # Because the test runs against dummy data, the analysis
            # can fail on a failing fit which raises a type error when
            # creating the custom text string. This test now only tests
            # if the find_frequency method runs until the expected part.
            # This should be fixed by making the analysis robust.
            pass
        self.CCL_qubit.cfg_qubit_freq_calc_method('flux')
        try:
            self.CCL_qubit.find_frequency()
        except TypeError:
            pass

    def test_AllXY(self):
        self.CCL_qubit.measure_allxy()

    def test_T1(self):
        self.CCL_qubit.measure_T1(
            times=np.arange(0, 1e-6, 20e-9), update=False, analyze=False)
        self.CCL_qubit.T1(20e-6)
        self.CCL_qubit.measure_T1(update=False, analyze=False)

    def test_Ramsey(self):
        self.CCL_qubit.mw_freq_mod(100e6)
        # Cannot analyze dummy data as analysis will fail on fit
        self.CCL_qubit.measure_ramsey(times=np.arange(0, 1e-6, 20e-9),
                                      update=False, analyze=False)
        self.CCL_qubit.T2_star(20e-6)
        self.CCL_qubit.measure_ramsey(update=False, analyze=False)

    def test_echo(self):
        self.CCL_qubit.mw_freq_mod(100e6)
        # self.CCL_qubit.measure_echo(times=np.arange(0,2e-6,40e-9))
        time.sleep(1)
        self.CCL_qubit.T2_echo(40e-6)
        self.CCL_qubit.measure_echo(analyze=False, update=False)
        time.sleep(1)
        with self.assertRaises(ValueError):
            invalid_times = [0.1e-9, 0.2e-9, 0.3e-9, 0.4e-9]
            self.CCL_qubit.measure_echo(times=invalid_times)

        with self.assertRaises(ValueError):
            self.CCL_qubit.mw_freq_mod(.1e6)
            invalid_times = np.arange(0, 2e-6, 60e-9)
            self.CCL_qubit.measure_echo(times=invalid_times)
            self.CCL_qubit.mw_freq_mod(100e6)

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.CCL_qubit._all_instruments):
            try:
                inst = self.CCL_qubit.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass
