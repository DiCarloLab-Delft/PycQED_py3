import unittest
import os
import time
import warnings
import numpy as np
import pycqed as pq

from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object import Qubit
from pycqed.instrument_drivers.meta_instrument.HAL.HAL_ShimSQ import HAL_ShimSQ
from pycqed.instrument_drivers.meta_instrument.qubit_objects.HAL_Transmon import HAL_Transmon
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
    if 0:  # FIXME: VSM configuration
        cls.MW_LutMan = mwl.MW_LutMan('MW_LutMan_VSM')
        cls.MW_LutMan.channel_GI(1)
        cls.MW_LutMan.channel_GQ(2)
        cls.MW_LutMan.channel_DI(3)
        cls.MW_LutMan.channel_DQ(4)
    else:
        cls.MW_LutMan = mwl.AWG8_MW_LutMan('MW_LutMan')
        cls.MW_LutMan.channel_I(1)
        cls.MW_LutMan.channel_Q(2)
        cls.MW_LutMan.mw_modulation(100e6)
        cls.MW_LutMan.sampling_rate(2.4e9)

        qubit_obj.cfg_with_vsm(False)
        qubit_obj.cfg_prepare_mw_awg(True)  # FIXME: load_waveform_onto_AWG_lookuptable fails
    cls.MW_LutMan.AWG(cls.AWG.name)
    cls.MW_LutMan.mw_modulation(100e6)
    cls.MW_LutMan.sampling_rate(2.4e9)

    cls.ro_lutman = UHFQC_RO_LutMan('RO_lutman', num_res=5, feedline_number=0)
    cls.ro_lutman.AWG(cls.UHFQC.name)

    ##############################################
    # Assign instruments
    ##############################################
    qubit_obj.instr_LutMan_MW(cls.MW_LutMan.name)
    qubit_obj.instr_LO_ro(cls.MW1.name)
    qubit_obj.instr_LO_mw(cls.MW2.name)
    qubit_obj.instr_spec_source(cls.MW3.name)

    qubit_obj.instr_acquisition(cls.UHFQC.name)
    qubit_obj.instr_VSM(cls.VSM.name)
    qubit_obj.instr_CC(cls.CC.name)
    qubit_obj.instr_LutMan_RO(cls.ro_lutman.name)

    qubit_obj.instr_SH(cls.SH.name)

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

    qubit_obj.instr_MC(cls.MC.name)

    ##############################################
    # Setting some "random" initial parameters
    ##############################################
    qubit_obj.ro_freq(5.43e9)
    qubit_obj.ro_freq_mod(200e6)

    qubit_obj.mw_freq_mod(-100e6)
    qubit_obj.mw_awg_ch(1)
    qubit_obj.cfg_qubit_nr(0)

    qubit_obj.mw_vsm_delay(15)

    qubit_obj.mw_mixer_offs_GI(.1)
    qubit_obj.mw_mixer_offs_GQ(.2)
    qubit_obj.mw_mixer_offs_DI(.3)
    qubit_obj.mw_mixer_offs_DQ(.4)

    # FIXME" move out of test_HAL_ShimSQ
    qubit_obj.freq_qubit(4.56e9)


class test_Qubit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transmon = Qubit('Qubit')

    @classmethod
    def tearDownClass(cls):
        Instrument.close_all()

    # FIXME: add tests


class test_HAL_ShimSQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit_obj = HAL_ShimSQ('HAL_ShimSQ')
        cls.shim = qubit_obj
        _setup_hw(cls, qubit_obj)

    @classmethod
    def tearDownClass(cls):
        Instrument.close_all()

    ##############################################
    # basic prepare methods
    ##############################################

    def test_prep_for_continuous_wave(self):
        self.shim.ro_acq_weight_type('optimal')
        with warnings.catch_warnings(record=True) as w:
            self.shim.prepare_for_continuous_wave()
            self.assertEqual(str(w[0].message), 'Changing ro_acq_weight_type to SSB.')

        self.shim.ro_acq_weight_type('SSB')
        self.shim.prepare_for_continuous_wave()

    def test_prep_readout(self):
        self.shim.prepare_readout()

    def test_prep_ro_instantiate_detectors(self):
        # delete detectors
        detector_attributes = ['int_avg_det', 'int_log_det', 'int_avg_det_single', 'input_average_detector']
        for det_attr in detector_attributes:
            if hasattr(self.shim, det_attr):
                delattr(self.shim, det_attr)

        # check there are no detectors to start with
        for det_attr in detector_attributes:
            self.assertFalse(hasattr(self.shim, det_attr))

        # run test
        self.MC.soft_avg(1)
        self.shim.ro_soft_avg(4)
        self.shim.prepare_readout()

        # Test that the detectors have been instantiated
        for det_attr in detector_attributes:
            self.assertTrue(hasattr(self.shim, det_attr))

        self.assertEqual(self.MC.soft_avg(), 4)

    @unittest.skipIf(True, 'Test for use with an old duplexer.')
    def test_prep_cw_config_vsm(self):
        self.shim.spec_vsm_ch_in(2)
        self.shim.spec_vsm_ch_out(1)
        self.shim.spec_vsm_amp(0.5)

        self.shim.prepare_for_continuous_wave()

        self.assertEqual(self.VSM.in1_out1_switch(), 'OFF')
        self.assertEqual(self.VSM.in1_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_switch(), 'EXT')
        self.assertEqual(self.VSM.in2_out2_switch(), 'OFF')
        self.assertEqual(self.VSM.in2_out1_amp(), 0.5)

    def test_prep_for_fluxing(self):
        self.shim.prepare_for_fluxing()

    @unittest.skip('Not Implemented')
    def test_prep_flux_bias(self):
        raise NotImplementedError()

    ##############################################
    # Testing prepare for readout
    ##############################################

    def test_prep_ro_MW_sources(self):
        LO = self.shim.instr_LO_ro.get_instr()
        LO.off()
        LO.frequency(4e9)
        LO.power(10)

        self.assertEqual(LO.status(), 'off')
        self.assertEqual(LO.frequency(), 4e9)

        self.shim.mw_pow_td_source(20)
        self.shim.ro_freq(5.43e9)
        self.shim.ro_freq_mod(200e6)
        self.shim.prepare_readout()

        self.assertEqual(LO.status(), 'on')
        self.assertEqual(LO.frequency(), 5.43e9-200e6)
        self.assertEqual(LO.power(), 20)

    def test_prep_ro_pulses(self):
        self.shim.ro_pulse_mixer_alpha(1.1)
        self.shim.ro_pulse_mixer_phi(4)
        self.shim.ro_pulse_length(312e-9)
        self.shim.ro_pulse_down_amp0(.1)
        self.shim.ro_pulse_down_length0(23e-9)

        self.shim.ro_pulse_mixer_offs_I(.01)
        self.shim.ro_pulse_mixer_offs_Q(.02)

        self.shim.prepare_readout()

        self.assertEqual(self.ro_lutman.mixer_phi(), 4)
        self.assertEqual(self.ro_lutman.mixer_alpha(), 1.1)
        self.assertEqual(self.ro_lutman.M_length_R0(), 312e-9)
        self.assertEqual(self.ro_lutman.M_down_length0_R0(), 23e-9)
        self.assertEqual(self.ro_lutman.M_down_amp0_R0(), .1)

        self.assertEqual(self.UHFQC.sigouts_0_offset(), .01)
        self.assertEqual(self.UHFQC.sigouts_1_offset(), .02)

    def test_prep_ro_integration_weigths(self):
        IF = 50e6
        self.shim.ro_freq_mod(IF)
        self.shim.ro_acq_weight_chI(3)
        self.shim.ro_acq_weight_chQ(4)

        # Testing SSB
        trace_length = 4096
        self.shim.ro_acq_weight_type('SSB')
        self.shim.prepare_readout()
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))

        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 1 + 1j)
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), 1 - 1j)

        uploaded_wf = self.UHFQC.qas_0_integration_weights_3_real()
        np.testing.assert_array_almost_equal(cosI, uploaded_wf)
        # Testing DSB case
        self.shim.ro_acq_weight_type('DSB')
        self.shim.prepare_readout()
        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 2)
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), 2)

        # Testing Optimal weight uploading
        test_I = np.ones(10)
        test_Q = 0.5*test_I
        self.shim.ro_acq_weight_func_I(test_I)
        self.shim.ro_acq_weight_func_Q(test_Q)

        self.shim.ro_acq_weight_type('optimal')
        self.shim.prepare_readout()

        self.UHFQC.qas_0_rotations_4(.21 + 0.108j)
        upl_I = self.UHFQC.qas_0_integration_weights_3_real()
        upl_Q = self.UHFQC.qas_0_integration_weights_3_imag()
        np.testing.assert_array_almost_equal(test_I, upl_I)
        np.testing.assert_array_almost_equal(test_Q, upl_Q)

        self.assertEqual(self.UHFQC.qas_0_rotations_3(), 1 - 1j)

        # These should not have been touched by optimal weights
        self.assertEqual(self.UHFQC.qas_0_rotations_4(), .21 + .108j)

        self.shim.ro_acq_weight_type('SSB')

    ########################################################
    #          Test prepare for timedomain                 #
    ########################################################

    def test_prep_for_timedomain(self):
        self.shim.prepare_for_timedomain()

    def test_prep_td_sources(self):
        self.MW1.off()
        self.MW2.off()
        self.shim.freq_qubit(4.56e9)
        self.shim.mw_freq_mod(-100e6)
        self.shim.mw_pow_td_source(13)

        self.shim.prepare_for_timedomain()
        self.assertEqual(self.MW1.status(), 'on')
        self.assertEqual(self.MW2.status(), 'on')
        self.assertEqual(self.MW2.frequency(), 4.56e9 + 100e6)
        self.assertEqual(self.MW2.power(), 13)

    # def test_prep_td_pulses(self):
    #     pass # FIXME: moved to HAL_Transmon
    # NB: lutman handling resides in HAL_Transmon, not HAL_ShimSQ
    def test_prep_td_pulses(self):
        self.shim.mw_awg_ch(5)

        # set mixer parameters
        self.shim.mw_G_mixer_alpha(1.02)
        self.shim.mw_D_mixer_phi(8)

        self.shim.mw_mixer_offs_GI(.1)
        self.shim.mw_mixer_offs_GQ(.2)

        # self.shim.mw_ef_amp(.34)
        self.shim.mw_freq_mod(-100e6)
        # self.shim.anharmonicity(-235e6)

        self.shim.prepare_for_timedomain()

        self.assertEqual(self.MW_LutMan.channel_I(), 1)
        self.assertEqual(self.MW_LutMan.channel_Q(), 2)

        self.assertEqual(self.MW_LutMan.mixer_alpha(), 1.02)
        self.assertEqual(self.MW_LutMan.mixer_phi(), 0)  # FIXME: why not 8 as set above

        self.assertEqual(self.CC.vsm_channel_delay0(), self.shim.mw_vsm_delay())

        self.assertEqual(self.AWG.sigouts_4_offset(), 0.1)
        self.assertEqual(self.AWG.sigouts_5_offset(), 0.2)

        # self.assertEqual(self.MW_LutMan.mw_ef_amp180(), 0.34)
        # self.assertEqual(self.MW_LutMan.mw_ef_modulation(), -335e6)

    @unittest.skip('VSM not setup in __init__')
    def test_prep_td_config_vsm(self):
        self.shim.mw_vsm_G_amp(0.8)
        self.shim.mw_vsm_D_phase(0)
        self.shim.mw_vsm_ch_in(2)
        self.shim.mw_vsm_mod_out(5)
        self.shim.prepare_for_timedomain()

        self.assertEqual(self.VSM.mod5_ch2_gaussian_amp(), 0.8)
        self.assertEqual(self.VSM.mod5_ch2_derivative_phase(), 0)


class Test_HAL_Transmon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        qubit_obj = HAL_Transmon('HAL_qubit')
        cls.transmon = qubit_obj
        _setup_hw(cls, qubit_obj)

        config_fn = os.path.join(pq.__path__[0], 'tests', 'openql', 'test_cfg_cc.json')
        cls.transmon.cfg_openql_platform_fn(config_fn)

        cls.transmon.freq_max(4.62e9)

    @classmethod
    def tearDownClass(self):
        Instrument.close_all()

    ##############################################
    # calculate methods
    ##############################################

    # FIXME: calculate_frequency is defined in class Qubit, but cfg_qubit_freq_calc_method in class HAL_Transmon
    def test_calc_freq(self):
        self.transmon.cfg_qubit_freq_calc_method('latest')
        self.transmon.calculate_frequency()

        self.transmon.cfg_qubit_freq_calc_method('flux')
        self.transmon.calculate_frequency()

    ##############################################
    # basic prepare methods
    ##############################################

    # NB: lutman handling resides in HAL_Transmon, not HAL_ShimSQ
    def test_prep_td_pulses(self):
        self.transmon.mw_awg_ch(5)

        # set mixer parameters
        # self.transmon.mw_G_mixer_alpha(1.02)
        # self.transmon.mw_D_mixer_phi(8)
        #
        # self.transmon.mw_mixer_offs_GI(.1)
        # self.transmon.mw_mixer_offs_GQ(.2)

        self.transmon.mw_ef_amp(.34)
        self.transmon.mw_freq_mod(-100e6)
        self.transmon.anharmonicity(-235e6)

        self.transmon.prepare_for_timedomain()

        # self.assertEqual(self.MW_LutMan.channel_I(), 1)
        # self.assertEqual(self.MW_LutMan.channel_Q(), 2)
        #
        # self.assertEqual(self.MW_LutMan.mixer_alpha(), 1.02)
        # self.assertEqual(self.MW_LutMan.mixer_phi(), 0)  # FIXME: why not 8 as set above
        #
        # self.assertEqual(self.CC.vsm_channel_delay0(), self.transmon.mw_vsm_delay())
        #
        # self.assertEqual(self.AWG.sigouts_4_offset(), 0.1)
        # self.assertEqual(self.AWG.sigouts_5_offset(), 0.2)

        self.assertEqual(self.MW_LutMan.mw_ef_amp180(), 0.34)
        self.assertEqual(self.MW_LutMan.mw_ef_modulation(), -335e6)


    @unittest.skip('not configured for VSM')
    def test_prep_td_pulses_vsm(self):
        # this function contains the original test_prep_td_pulses, which used the VSM.
        # To make this work again, the initialization needs to be corrected, and maybe parts of HAL_Transmon
        self.transmon.mw_awg_ch(5)

        # set mixer parameters
        self.transmon.mw_G_mixer_alpha(1.02)
        self.transmon.mw_D_mixer_phi(8)

        self.transmon.mw_mixer_offs_GI(.1)
        self.transmon.mw_mixer_offs_GQ(.2)
        self.transmon.mw_mixer_offs_DI(.3)
        self.transmon.mw_mixer_offs_DQ(.4)

        self.transmon.mw_ef_amp(.34)
        self.transmon.mw_freq_mod(-100e6)
        self.transmon.anharmonicity(-235e6)

        self.transmon.prepare_for_timedomain()

        self.assertEqual(self.MW_LutMan.channel_GI(), 5)
        self.assertEqual(self.MW_LutMan.channel_GQ(), 6)
        self.assertEqual(self.MW_LutMan.channel_DI(), 7)
        self.assertEqual(self.MW_LutMan.channel_DQ(), 8)

        self.assertEqual(self.MW_LutMan.G_mixer_alpha(), 1.02)
        self.assertEqual(self.MW_LutMan.D_mixer_phi(), 8)

        self.assertEqual(self.CC.vsm_channel_delay0(), self.transmon.mw_vsm_delay())

        self.assertEqual(self.AWG.sigouts_4_offset(), .1)
        self.assertEqual(self.AWG.sigouts_5_offset(), .2)
        self.assertEqual(self.AWG.sigouts_6_offset(), .3)
        self.assertEqual(self.AWG.sigouts_7_offset(), .4)

        self.assertEqual(self.MW_LutMan.mw_ef_amp180(), 0.34)
        self.assertEqual(self.MW_LutMan.mw_ef_modulation(), -335e6)


    ###################################################
    #          Test basic experiments                 #
    ###################################################

    def test_cal_mixer_offsets_drive(self):
        self.transmon.calibrate_mixer_offsets_drive()

    def test_resonator_spec(self):
        self.transmon.ro_acq_weight_type('SSB')

        # set to not set to bypass validator
        # [2020-07-23 Victor] commented out, it is already None by default
        # `_save_val` is not available anymore
        # self.transmon.freq_res._save_val(None)
        try:
            self.transmon.find_resonator_frequency()
        except ValueError:
            pass  # Fit can fail because testing against random data
        self.transmon.freq_res(5.4e9)
        try:
            self.transmon.find_resonator_frequency()
        except ValueError:
            pass  # Fit can fail because testing against random data
        freqs = np.linspace(6e9, 6.5e9, 31)

        self.transmon.measure_heterodyne_spectroscopy(freqs=freqs,
                                                       analyze=False)

    def test_resonator_power(self):
        self.transmon.ro_acq_weight_type('SSB')
        freqs = np.linspace(6e9, 6.5e9, 31)
        powers = np.arange(-30, -10, 5)

        # set to not set to bypass validator
        # [2020-07-23 Victor] commented out, it is already None by default
        # `_save_val` is not available anymore
        # self.transmon.freq_res._save_val(None)
        self.transmon.measure_resonator_power(freqs=freqs, powers=powers)

    def test_measure_transients(self):
        self.transmon.ro_acq_input_average_length(2e-6)
        self.transmon.measure_transients()

    def test_qubit_spec(self):
        freqs = np.linspace(6e9, 6.5e9, 31)
        # Data cannot be analyzed as dummy data is just random numbers
        self.transmon.measure_spectroscopy(freqs=freqs, analyze=False)

    def test_find_qubit_freq(self):
        self.transmon.cfg_qubit_freq_calc_method('latest')
        try:
            self.transmon.find_frequency()
        except TypeError:
            # Because the test runs against dummy data, the analysis
            # can fail on a failing fit which raises a type error when
            # creating the custom text string. This test now only tests
            # if the find_frequency method runs until the expected part.
            # This should be fixed by making the analysis robust.
            pass
        self.transmon.cfg_qubit_freq_calc_method('flux')
        try:
            self.transmon.find_frequency()
        except TypeError:
            pass

    def test_AllXY(self):
        self.transmon.measure_allxy()

    def test_T1(self):
        self.transmon.measure_T1(
            times=np.arange(0, 1e-6, 20e-9), update=False, analyze=False)
        self.transmon.T1(20e-6)
        self.transmon.measure_T1(update=False, analyze=False)

    def test_Ramsey(self):
        self.transmon.mw_freq_mod(100e6)
        # Cannot analyze dummy data as analysis will fail on fit
        self.transmon.measure_ramsey(times=np.arange(0, 1e-6, 20e-9),
                                      update=False, analyze=False)
        self.transmon.T2_star(20e-6)
        self.transmon.measure_ramsey(update=False, analyze=False)

    def test_echo(self):
        self.transmon.mw_freq_mod(100e6)
        # self.transmon.measure_echo(times=np.arange(0,2e-6,40e-9))
        time.sleep(1)
        self.transmon.T2_echo(40e-6)
        self.transmon.measure_echo(analyze=False, update=False)
        time.sleep(1)
        with self.assertRaises(ValueError):
            invalid_times = [0.1e-9, 0.2e-9, 0.3e-9, 0.4e-9]
            self.transmon.measure_echo(times=invalid_times)

        with self.assertRaises(ValueError):
            self.transmon.mw_freq_mod(.1e6)
            invalid_times = np.arange(0, 2e-6, 60e-9)
            self.transmon.measure_echo(times=invalid_times)
            self.transmon.mw_freq_mod(100e6)


class Test_Instantiate(unittest.TestCase):
    def test_instantiate_QuDevTransmon(self):
        QDT = QuDev_transmon('QuDev_transmon',
                             MC=None, heterodyne_instr=None, cw_source=None)
        QDT.close()
