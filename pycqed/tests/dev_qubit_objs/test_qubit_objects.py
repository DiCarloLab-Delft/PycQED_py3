import unittest
import numpy as np
import os
import pycqed as pq
import time

import pycqed.analysis.analysis_toolbox as a_tools

import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
import pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound as sh
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.dummy_UHFQC import dummy_UHFQC

from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer


from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon import Tektronix_driven_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CC_transmon import CBox_v3_driven_transmon, QWG_driven_transmon
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

Dummy_VSM_not_fixed = True

try:
    import openql
    openql_import_fail = False
except:
    openql_import_fail = True


class Test_QO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.CCL_qubit = ct.CCLight_Transmon('CCL_qubit')

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')
        self.SH = sh.virtual_SignalHound_USB_SA124B('SH')
        self.UHFQC = dummy_UHFQC('UHFQC')

        self.CCL = dummy_CCL('CCL')
        self.VSM = Dummy_Duplexer('VSM')

        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        # Required to set it to the testing datadir
        test_datadir = os.path.join(pq.__path__[0], 'tests', 'test_output')
        self.MC.datadir(test_datadir)
        a_tools.datadir = self.MC.datadir()

        self.AWG = v8.VirtualAWG8('DummyAWG8')
        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        self.ro_lutman = UHFQC_RO_LutMan('RO_lutman', num_res=5)
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

        config_fn = os.path.join(pq.__path__[0], 'tests', 'test_cfg_CCL.json')
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

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_instantiate_QuDevTransmon(self):
        QDT = QuDev_transmon('QuDev_transmon',
                             MC=None, heterodyne_instr=None, cw_source=None)
        QDT.close()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_instantiate_TekTransmon(self):
        TT = Tektronix_driven_transmon('TT')
        TT.close()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_instantiate_CBoxv3_transmon(self):
        CT = CBox_v3_driven_transmon('CT')
        CT.close()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_instantiate_QWG_transmon(self):
        QT = QWG_driven_transmon('QT')
        QT.close()

    ##############################################
    # calculate methods
    ##############################################
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_calc_freq(self):
        self.CCL_qubit.cfg_qubit_freq_calc_method('latest')
        self.CCL_qubit.calculate_frequency()
        self.CCL_qubit.cfg_qubit_freq_calc_method('flux')
        self.CCL_qubit.calculate_frequency()

    ##############################################
    # basic prepare methods
    ##############################################

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_for_continuous_wave(self):
        self.CCL_qubit.ro_acq_weight_type('optimal')
        with self.assertRaises(ValueError):
            self.CCL_qubit.prepare_for_continuous_wave()
        self.CCL_qubit.ro_acq_weight_type('SSB')
        self.CCL_qubit.prepare_for_continuous_wave()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_cw_config_vsm(self):

        self.CCL_qubit.spec_vsm_ch_in(2)
        self.CCL_qubit.spec_vsm_ch_out(1)
        self.CCL_qubit.spec_vsm_att(112)

        self.CCL_qubit.prepare_for_continuous_wave()

        self.assertEqual(self.VSM.in1_out1_switch(), 'OFF')
        self.assertEqual(self.VSM.in1_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_switch(), 'EXT')
        self.assertEqual(self.VSM.in2_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_att(), 112)

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_for_fluxing(self):
        self.CCL_qubit.prepare_for_fluxing()

    @unittest.skip('Not Implemented')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_flux_bias(self):
        raise NotImplementedError()

    ##############################################
    # Testing prepare for readout
    ##############################################
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_readout(self):
        self.CCL_qubit.prepare_readout()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
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

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
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

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_ro_pulses(self):
        self.CCL_qubit.ro_pulse_res_nr(3)
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
        self.assertEqual(self.ro_lutman.M_length_R3(), 312e-9)
        self.assertEqual(self.ro_lutman.M_down_length0_R3(), 23e-9)
        self.assertEqual(self.ro_lutman.M_down_amp0_R3(), .1)

        self.assertEqual(self.UHFQC.sigouts_0_offset(), .01)
        self.assertEqual(self.UHFQC.sigouts_1_offset(), .02)

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
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

        self.assertEqual(self.UHFQC.quex_rot_3_real(), 1)
        self.assertEqual(self.UHFQC.quex_rot_3_imag(), 1)
        self.assertEqual(self.UHFQC.quex_rot_4_real(), 1)
        self.assertEqual(self.UHFQC.quex_rot_4_imag(), -1)

        uploaded_wf = self.UHFQC.quex_wint_weights_3_real()
        np.testing.assert_array_almost_equal(cosI, uploaded_wf)
        # Testing DSB case
        self.CCL_qubit.ro_acq_weight_type('DSB')
        self.CCL_qubit.prepare_readout()
        self.assertEqual(self.UHFQC.quex_rot_3_real(), 2)
        self.assertEqual(self.UHFQC.quex_rot_3_imag(), 0)
        self.assertEqual(self.UHFQC.quex_rot_4_real(), 2)
        self.assertEqual(self.UHFQC.quex_rot_4_imag(), 0)

        # Testing Optimal weight uploading
        test_I = np.ones(10)
        test_Q = 0.5*test_I
        self.CCL_qubit.ro_acq_weight_func_I(test_I)
        self.CCL_qubit.ro_acq_weight_func_Q(test_Q)

        self.CCL_qubit.ro_acq_weight_type('optimal')
        self.CCL_qubit.prepare_readout()

        self.UHFQC.quex_rot_4_real(.21)
        self.UHFQC.quex_rot_4_imag(.108)
        upl_I = self.UHFQC.quex_wint_weights_3_real()
        upl_Q = self.UHFQC.quex_wint_weights_3_imag()
        np.testing.assert_array_almost_equal(test_I, upl_I)
        np.testing.assert_array_almost_equal(test_Q, upl_Q)
        self.assertEqual(self.UHFQC.quex_rot_3_real(), 1)
        self.assertEqual(self.UHFQC.quex_rot_3_imag(), -1)
        # These should not have been touched by optimal weights
        self.assertEqual(self.UHFQC.quex_rot_4_real(), .21)
        self.assertEqual(self.UHFQC.quex_rot_4_imag(), .108)

        self.CCL_qubit.ro_acq_weight_type('SSB')

    ########################################################
    #          Test prepare for timedomain                 #
    ########################################################
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_for_timedomain(self):
        self.CCL_qubit.prepare_for_timedomain()

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
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

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_td_pulses(self):
        self.CCL_qubit.mw_awg_ch(5)
        self.CCL_qubit.mw_G_mixer_alpha(1.02)
        self.CCL_qubit.mw_D_mixer_phi(8)

        self.CCL_qubit.mw_mixer_offs_GI(.1)
        self.CCL_qubit.mw_mixer_offs_GQ(.2)
        self.CCL_qubit.mw_mixer_offs_DI(.3)
        self.CCL_qubit.mw_mixer_offs_DQ(.4)

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

    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_prep_td_config_vsm(self):
        self.CCL_qubit.mw_vsm_switch('ON')
        self.CCL_qubit.mw_vsm_G_att(10234)
        self.CCL_qubit.mw_vsm_D_phase(10206)
        self.CCL_qubit.mw_vsm_ch_Gin(3)
        self.CCL_qubit.mw_vsm_ch_Din(4)
        self.CCL_qubit.mw_vsm_ch_out(2)
        self.CCL_qubit.prepare_for_timedomain()

        self.assertEqual(self.VSM.in3_out2_switch(), 'ON')
        self.assertEqual(self.VSM.in3_out2_att(), 10234)
        self.assertEqual(self.VSM.in4_out2_phase(), 10206)

    ###################################################
    #          Test basic experiments                 #
    ###################################################
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_cal_mixer_offsets_drive(self):
        self.CCL_qubit.calibrate_mixer_offsets_drive()

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_resonator_spec(self):
        self.CCL_qubit.ro_acq_weight_type('SSB')

        # set to not set to bypass validator
        self.CCL_qubit.freq_res._save_val(None)
        with self.assertRaises(ValueError):
            self.CCL_qubit.find_resonator_frequency()
        self.CCL_qubit.freq_res(5.4e9)
        self.CCL_qubit.find_resonator_frequency()
        freqs = np.linspace(6e9, 6.5e9, 31)

        self.CCL_qubit.measure_heterodyne_spectroscopy(freqs=freqs)

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_resonator_power(self):
        self.CCL_qubit.ro_acq_weight_type('SSB')
        freqs = np.linspace(6e9, 6.5e9, 31)
        powers = np.arange(-30, -10, 5)

        # set to not set to bypass validator
        self.CCL_qubit.freq_res._save_val(None)
        self.CCL_qubit.measure_resonator_power(freqs=freqs, powers=powers)

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_measure_transients(self):
        self.CCL_qubit.ro_acq_input_average_length(2e-6)
        self.CCL_qubit.measure_transients()

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_qubit_spec(self):
        freqs = np.linspace(6e9, 6.5e9, 31)
        self.CCL_qubit.measure_spectroscopy(freqs=freqs)

    # @unittest.skip('NotImplementedError')
    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_find_qubit_freq(self):
        self.CCL_qubit.cfg_qubit_freq_calc_method('latest')
        self.CCL_qubit.find_frequency()
        self.CCL_qubit.cfg_qubit_freq_calc_method('flux')
        self.CCL_qubit.find_frequency()

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_AllXY(self):
        self.CCL_qubit.measure_allxy()

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_T1(self):
        self.CCL_qubit.measure_T1(
            times=np.arange(0, 1e-6, 20e-9), update=False)
        self.CCL_qubit.T1(20e-6)
        self.CCL_qubit.measure_T1(update=False)

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_Ramsey(self):
        self.CCL_qubit.mw_freq_mod(100e6)
        self.CCL_qubit.measure_Ramsey(times=np.arange(0, 1e-6, 20e-9),
                                      update=False)
        self.CCL_qubit.T2_star(20e-6)
        self.CCL_qubit.measure_Ramsey(update=False)

    @unittest.skipIf(openql_import_fail, 'OpenQL not present')
    @unittest.skipIf(Dummy_VSM_not_fixed, 'Dummy_VSM_not_fixed')
    def test_echo(self):
        self.CCL_qubit.mw_freq_mod(100e6)
        # self.CCL_qubit.measure_echo(times=np.arange(0,2e-6,40e-9))
        time.sleep(1)
        self.CCL_qubit.T2_echo(40e-6)
        # self.CCL_qubit.measure_echo()
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
