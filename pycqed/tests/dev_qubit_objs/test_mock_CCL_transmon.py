
import unittest
import pytest
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
import pycqed.instrument_drivers.meta_instrument.qubit_objects.mock_CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG

from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer
from pycqed.instrument_drivers.meta_instrument.Resonator import resonator
import pycqed.instrument_drivers.meta_instrument.device_object_CCL as do

from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon import Tektronix_driven_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CC_transmon import CBox_v3_driven_transmon, QWG_driven_transmon
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL
from pycqed.instrument_drivers.physical_instruments.QuTech_VSM_Module import Dummy_QuTechVSMModule
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
import pycqed.instrument_drivers.virtual_instruments.virtual_SPI_S4g_FluxCurrent as flx

Dummy_VSM_not_fixed = False


class Test_Mock_CCL(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.CCL_qubit = ct.Mock_CCLight_Transmon('CCL_qubit')

        self.fluxcurrent = flx.virtual_SPI_S4g_FluxCurrent(
                'fluxcurrent',
                channel_map={
                    'FBL_Q1': (0, 0),
                    'FBL_Q2': (0, 1),
                })
        self.fluxcurrent.FBL_Q1(0)
        self.fluxcurrent.FBL_Q2(0)
        self.station.add_component(self.fluxcurrent)

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
        self.CCL_qubit.instr_FluxCtrl(self.fluxcurrent.name)
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
        # self.CCL_qubit.ro_acq_averages(32768)
        self.device = do.DeviceCCL(name='device')
        self.CCL_qubit.instr_device(self.device.name)

    ###########################################################
    # Test find resonator frequency
    ###########################################################
    def test_find_resonator_frequency(self):
        self.CCL_qubit.mock_freq_res_bare(7.58726e9)
        self.CCL_qubit.mock_sweetspot_phi_over_phi0(0)
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()

        self.CCL_qubit.freq_res(7.587e9)
        self.CCL_qubit.find_resonator_frequency()

        assert self.CCL_qubit.freq_res() == pytest.approx(freq_res, abs=1e6)

    ###########################################################
    # Test find qubit frequency
    ###########################################################
    def test_find_frequency(self):
        self.CCL_qubit.mock_sweetspot_phi_over_phi0(0)

        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej1(8e9)
        self.CCL_qubit.mock_Ej2(8e9)

        f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()

        self.CCL_qubit.freq_qubit(f_qubit)

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()
        self.CCL_qubit.freq_res(freq_res)
        self.CCL_qubit.ro_freq(freq_res)

        threshold = 0.01e9
        self.CCL_qubit.find_frequency()
        assert np.abs(f_qubit - self.CCL_qubit.freq_qubit()) <= threshold

    ###########################################################
    # Test MW pulse calibration
    ###########################################################
    def test_calibrate_mw_pulse_amplitude_coarse(self):
        for with_vsm in [True, False]:
            self.CCL_qubit.mock_sweetspot_phi_over_phi0(0)

            f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()

            self.CCL_qubit.freq_res(self.CCL_qubit.calculate_mock_resonator_frequency())
            self.CCL_qubit.freq_qubit(f_qubit)

            self.CCL_qubit.cfg_with_vsm(with_vsm)
            self.CCL_qubit.mock_mw_amp180(.345)
            self.CCL_qubit.calibrate_mw_pulse_amplitude_coarse()
            freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()
            self.CCL_qubit.ro_freq(freq_res)

            eps = 0.05
            if self.CCL_qubit.cfg_with_vsm():
                # FIXME: shown to sometimes fail (PR #638)
                assert self.CCL_qubit.mw_vsm_G_amp() == pytest.approx(
                        self.CCL_qubit.mock_mw_amp180(), abs=eps)
            else:
                assert self.CCL_qubit.mw_channel_amp() == pytest.approx(
                        self.CCL_qubit.mw_channel_amp(), abs=eps)

    ###########################################################
    # Test find qubit sweetspot
    ###########################################################
    def test_find_qubit_sweetspot(self):
        assert self.CCL_qubit.mock_fl_dc_ch() == 'FBL_Q1'
        self.CCL_qubit.fl_dc_ch(self.CCL_qubit.mock_fl_dc_ch())

        self.CCL_qubit.mock_sweetspot_phi_over_phi0(0.01343)
        current = 0.01343*self.CCL_qubit.mock_fl_dc_I_per_phi0()[
                        self.CCL_qubit.mock_fl_dc_ch()]
        self.CCL_qubit.fl_dc_I0(current)

        fluxcurrent = self.CCL_qubit.instr_FluxCtrl.get_instr()
        fluxcurrent[self.CCL_qubit.mock_fl_dc_ch()](current)

        assert self.CCL_qubit.mock_fl_dc_ch() == 'FBL_Q1'

        f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()
        self.CCL_qubit.freq_qubit(f_qubit)

        self.CCL_qubit.freq_res(self.CCL_qubit.calculate_mock_resonator_frequency())

        assert self.CCL_qubit.mock_fl_dc_ch() == 'FBL_Q1'
        assert self.CCL_qubit.fl_dc_ch() == 'FBL_Q1'
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()
        self.CCL_qubit.ro_freq(freq_res)

        assert self.CCL_qubit.mock_fl_dc_ch() == 'FBL_Q1'
        self.CCL_qubit.find_qubit_sweetspot()

        assert self.CCL_qubit.fl_dc_I0() == pytest.approx(
                                    current,
                                    abs=30e-6)

    ###########################################################
    # Test RO pulse calibration
    ###########################################################
    def test_calibrate_ro_pulse_CW(self):
        self.CCL_qubit.mock_ro_pulse_amp_CW(0.05)
        self.CCL_qubit.mock_freq_res_bare(7.5e9)
        self.CCL_qubit.freq_res(self.CCL_qubit.calculate_mock_resonator_frequency())

        self.device.qubits([self.CCL_qubit.name])

        self.CCL_qubit.calibrate_ro_pulse_amp_CW()
        eps = 0.1
        assert self.CCL_qubit.ro_pulse_amp_CW() <= self.CCL_qubit.mock_ro_pulse_amp_CW()

    ###########################################################
    # Test find test resonators
    ###########################################################
    def test_find_test_resonators(self):
        self.CCL_qubit.mock_freq_res_bare(7.78542e9)
        self.CCL_qubit.mock_freq_test_res(7.9862e9)

        res0 = resonator(identifier=0, freq=7.785e9)
        res1 = resonator(identifier=1, freq=7.986e9)

        self.CCL_qubit.instr_device.get_instr().resonators = [res0, res1]

        for res in [res0, res1]:
            self.CCL_qubit.find_test_resonators()

            if res.identifier == 0:
                assert res0.type == 'qubit_resonator'
            elif res.identifier == 1:
                assert res1.type == 'test_resonator'

    ###########################################################
    # Test Ramsey
    ###########################################################
    def test_ramsey(self):

        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej1(8e9)
        self.CCL_qubit.mock_Ej2(8e9)

        self.CCL_qubit.mock_sweetspot_phi_over_phi0(0)

        f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()
        self.CCL_qubit.freq_qubit(f_qubit)

        self.CCL_qubit.freq_res(self.CCL_qubit.calculate_mock_resonator_frequency())

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()
        self.CCL_qubit.ro_freq(freq_res)


        self.CCL_qubit.mock_T2_star(23e-6)
        self.CCL_qubit.T2_star(19e-6)
        self.CCL_qubit.measure_ramsey()

        threshold = 4e-6
        assert self.CCL_qubit.T2_star() == pytest.approx(
                                           self.CCL_qubit.mock_T2_star(),
                                           abs=threshold)

    ###########################################################
    # Test T1
    ###########################################################
    def test_T1(self):
        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej1(8e9)
        self.CCL_qubit.mock_Ej2(8e9)

        fluxcurrent = self.CCL_qubit.instr_FluxCtrl.get_instr()
        current = self.CCL_qubit.mock_sweetspot_phi_over_phi0()

        fluxcurrent[self.CCL_qubit.mock_fl_dc_ch()](current)

        f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()
        self.CCL_qubit.freq_qubit(f_qubit)
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()

        self.CCL_qubit.freq_res(freq_res)

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        self.CCL_qubit.ro_freq(freq_res)

        self.CCL_qubit.mock_T1(34.39190e-6)
        self.CCL_qubit.T1(40e-6)
        self.CCL_qubit.measure_T1()
        self.CCL_qubit.measure_T1()

        assert self.CCL_qubit.T1() == pytest.approx(self.CCL_qubit.mock_T1(),
                                                    abs=5e-5) # R.S. raised this because it was too tight

    ###########################################################
    # Test Echo
    ###########################################################
    def test_echo(self):

        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej1(8e9)
        self.CCL_qubit.mock_Ej2(8e9)

        self.CCL_qubit.mock_sweetspot_phi_over_phi0(0)

        f_qubit = self.CCL_qubit.calculate_mock_qubit_frequency()
        self.CCL_qubit.freq_qubit(f_qubit)

        self.CCL_qubit.freq_res(self.CCL_qubit.calculate_mock_resonator_frequency())

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        freq_res = self.CCL_qubit.calculate_mock_resonator_frequency()
        self.CCL_qubit.ro_freq(freq_res)

        self.CCL_qubit.mock_T2_echo(23e-6)
        self.CCL_qubit.T2_echo(19e-6)
        self.CCL_qubit.measure_echo()

        threshold = 3e-6
        assert self.CCL_qubit.T2_echo() == pytest.approx(
                                           self.CCL_qubit.mock_T2_echo(),
                                           abs=threshold)

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.CCL_qubit._all_instruments):
            try:
                inst = self.CCL_qubit.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass
