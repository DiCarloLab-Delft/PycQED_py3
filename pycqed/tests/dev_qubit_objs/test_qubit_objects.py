import unittest
import os
import time
import warnings
import numpy as np
import pycqed as pq

from pycqed.instrument_drivers.meta_instrument.qubit_objects.HAL_Transmon import HAL_Transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.HAL_ShimSQ import HAL_ShimSQ
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

from pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound import virtual_SignalHound_USB_SA124B
from pycqed.instrument_drivers.virtual_instruments.virtual_MW_source import VirtualMWsource

from pycqed.instrument_drivers.library.Transport import DummyTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController import UHFQC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.QuTech_VSM_Module import Dummy_QuTechVSMModule

import pycqed.analysis.analysis_toolbox as a_tools
from pycqed.measurement.measurement_control import MeasurementControl

from qcodes import station, Instrument


def _setup_hw(cls, qubit_obj):
    cls.CCL_qubit = qubit_obj

    ##############################################
    # setup (virtual) hardware
    ##############################################
    cls.CC = CC('CC', DummyTransport(), ccio_slots_driving_vsm=[5])
    cls.VSM = Dummy_QuTechVSMModule('VSM')
    cls.UHFQC = UHFQC(name='UHFQC', server='emulator', device='dev2109', interface='1GbE')
    cls.AWG = ZI_HDAWG8(name='DummyAWG8', server='emulator', num_codewords=128, device='dev8026', interface='1GbE')

    cls.MW1 = VirtualMWsource('MW1')
    cls.MW2 = VirtualMWsource('MW2')
    cls.MW3 = VirtualMWsource('MW3')
    cls.SH = virtual_SignalHound_USB_SA124B('SH')

    ##############################################
    # setup LutMans
    ##############################################
    if 0:  # FIXME: broken by _prep_mw_pulses
        cls.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        cls.AWG8_VSM_MW_LutMan.AWG(cls.AWG.name)
        cls.AWG8_VSM_MW_LutMan.channel_GI(1)
        cls.AWG8_VSM_MW_LutMan.channel_GQ(2)
        cls.AWG8_VSM_MW_LutMan.channel_DI(3)
        cls.AWG8_VSM_MW_LutMan.channel_DQ(4)
        cls.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        cls.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)
    else:
        cls.AWG8_VSM_MW_LutMan = mwl.AWG8_MW_LutMan('MW_LutMan')
        cls.AWG8_VSM_MW_LutMan.AWG(cls.AWG.name)
        cls.AWG8_VSM_MW_LutMan.channel_I(1)
        cls.AWG8_VSM_MW_LutMan.channel_Q(2)
        cls.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        cls.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        cls.CCL_qubit.cfg_with_vsm(False)
        cls.CCL_qubit.cfg_prepare_mw_awg(False)  # FIXME: load_waveform_onto_AWG_lookuptable fails
    cls.ro_lutman = UHFQC_RO_LutMan('RO_lutman', num_res=5, feedline_number=0)
    cls.ro_lutman.AWG(cls.UHFQC.name)

    ##############################################
    # Assign instruments
    ##############################################
    cls.CCL_qubit.instr_LutMan_MW(cls.AWG8_VSM_MW_LutMan.name)
    cls.CCL_qubit.instr_LO_ro(cls.MW1.name)
    cls.CCL_qubit.instr_LO_mw(cls.MW2.name)
    cls.CCL_qubit.instr_spec_source(cls.MW3.name)

    cls.CCL_qubit.instr_acquisition(cls.UHFQC.name)
    cls.CCL_qubit.instr_VSM(cls.VSM.name)
    cls.CCL_qubit.instr_CC(cls.CC.name)
    cls.CCL_qubit.instr_LutMan_RO(cls.ro_lutman.name)

    cls.CCL_qubit.instr_SH(cls.SH.name)

    ##############################################
    # setup MC. FIXME: move out of class HAL_ShimSQ
    ##############################################
    cls.station = station.Station()

    cls.MC = MeasurementControl('MC', live_plot_enabled=False, verbose=False)
    cls.MC.station = cls.station
    cls.station.add_component(cls.MC)

    # Required to set it to the testing datadir
    test_datadir = os.path.join(pq.__path__[0], 'tests', 'test_output')
    cls.MC.datadir(test_datadir)
    a_tools.datadir = cls.MC.datadir()

    cls.CCL_qubit.instr_MC(cls.MC.name)

    ##############################################
    # Setting some "random" initial parameters
    ##############################################
    cls.CCL_qubit.ro_freq(5.43e9)
    cls.CCL_qubit.ro_freq_mod(200e6)

    cls.CCL_qubit.mw_freq_mod(-100e6)
    cls.CCL_qubit.mw_awg_ch(1)
    cls.CCL_qubit.cfg_qubit_nr(0)

    cls.CCL_qubit.mw_vsm_delay(15)

    cls.CCL_qubit.mw_mixer_offs_GI(.1)
    cls.CCL_qubit.mw_mixer_offs_GQ(.2)
    cls.CCL_qubit.mw_mixer_offs_DI(.3)
    cls.CCL_qubit.mw_mixer_offs_DQ(.4)


class test_HAL_ShimSQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit_obj = HAL_ShimSQ('HAL_ShimSQ')
        _setup_hw(cls, qubit_obj)

    @classmethod
    def tearDownClass(cls):
        Instrument.close_all()

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


class Test_HAL_Transmon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        qubit_obj = HAL_Transmon('HAL_qubit')
        _setup_hw(cls, qubit_obj)

        config_fn = os.path.join(pq.__path__[0], 'tests', 'openql', 'test_cfg_cc.json')
        cls.CCL_qubit.cfg_openql_platform_fn(config_fn)

        # Setting some "random" initial parameters
        cls.CCL_qubit.freq_qubit(4.56e9)
        cls.CCL_qubit.freq_max(4.62e9)


    @classmethod
    def tearDownClass(self):
        Instrument.close_all()

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

        if 0:  # FIXME: not using VSM
            self.CCL_qubit.mw_mixer_offs_GI(.1)
            self.CCL_qubit.mw_mixer_offs_GQ(.2)
            self.CCL_qubit.mw_mixer_offs_DI(.3)
            self.CCL_qubit.mw_mixer_offs_DQ(.4)
        else:
            self.CCL_qubit.mw_mixer_offs_GI(.1)
            self.CCL_qubit.mw_mixer_offs_GQ(.2)

        self.CCL_qubit.mw_ef_amp(.34)
        self.CCL_qubit.mw_freq_mod(-100e6)
        self.CCL_qubit.anharmonicity(-235e6)

        self.CCL_qubit.prepare_for_timedomain()
        if 0:  # FIXME: not using VSM
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_GI(), 5)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_GQ(), 6)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_DI(), 7)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_DQ(), 8)
        else:
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_I(), 1)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.channel_Q(), 2)

        if 0:  # FIXME: not using VSM
            self.assertEqual(self.AWG8_VSM_MW_LutMan.G_mixer_alpha(), 1.02)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.D_mixer_phi(), 8)
        else:
            self.assertEqual(self.AWG8_VSM_MW_LutMan.mixer_alpha(), 1.02)
            self.assertEqual(self.AWG8_VSM_MW_LutMan.mixer_phi(), 0)

        self.assertEqual(self.CC.vsm_channel_delay0(),
                         self.CCL_qubit.mw_vsm_delay())

        if 0:  # FIXME: not using VSM
            self.assertEqual(self.AWG.sigouts_4_offset(), .1)
            self.assertEqual(self.AWG.sigouts_5_offset(), .2)
            self.assertEqual(self.AWG.sigouts_6_offset(), .3)
            self.assertEqual(self.AWG.sigouts_7_offset(), .4)
        else:
            self.assertEqual(self.AWG.sigouts_4_offset(), .1)
            self.assertEqual(self.AWG.sigouts_5_offset(), .2)

        self.assertEqual(self.AWG8_VSM_MW_LutMan.mw_ef_amp180(), .34)
        self.assertEqual(self.AWG8_VSM_MW_LutMan.mw_ef_modulation(), -335e6)

    @unittest.skip('VSM not setup in __init__')
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
        # [2020-07-23 Victor] commented out, it is already None by default
        # `_save_val` is not available anymore
        # self.CCL_qubit.freq_res._save_val(None)
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
        # [2020-07-23 Victor] commented out, it is already None by default
        # `_save_val` is not available anymore
        # self.CCL_qubit.freq_res._save_val(None)
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


class Test_Instantiate(unittest.TestCase):
    def test_instantiate_QuDevTransmon(self):
        QDT = QuDev_transmon('QuDev_transmon',
                             MC=None, heterodyne_instr=None, cw_source=None)
        QDT.close()
