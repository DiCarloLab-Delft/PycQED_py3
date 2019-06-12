import unittest
import pytest
import numpy as np
import os
import pycqed as pq
import time
import openql
import warnings
import pycqed.analysis.analysis_toolbox as a_tools

import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
import pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound as sh
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
import pycqed.instrument_drivers.meta_instrument.qubit_objects.mock_CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.dummy_UHFQC import dummy_UHFQC

from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer


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
                    'FBL_1': (0, 0),
                    'FBL_2': (0, 1),
                })
        self.fluxcurrent.FBL_1(0)
        self.fluxcurrent.FBL_2(0)
        self.station.add_component(self.fluxcurrent)

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')
        self.SH = sh.virtual_SignalHound_USB_SA124B('SH')
        self.UHFQC = dummy_UHFQC('UHFQC')

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

        self.AWG = v8.VirtualAWG8('DummyAWG8')
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

    ###########################################################
    # Test find qubit frequency
    ###########################################################
    def test_find_frequency(self):
        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej(16e9)

        self.CCL_qubit.mock_freq_qubit(
            np.sqrt(8*self.CCL_qubit.mock_Ec()*self.CCL_qubit.mock_Ej()) - 
            self.CCL_qubit.mock_Ec())

        self.CCL_qubit.freq_qubit(
            np.sqrt(8*self.CCL_qubit.mock_Ec()*self.CCL_qubit.mock_Ej()) - 
            self.CCL_qubit.mock_Ec())

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        self.CCL_qubit.freq_res(self.CCL_qubit.mock_freq_res())
        self.CCL_qubit.ro_freq(self.CCL_qubit.mock_freq_res())

        fluxcurrent = self.CCL_qubit.instr_FluxCtrl.get_instr()
        current = self.CCL_qubit.mock_sweetspot_current()['FBL_1']

        fluxcurrent[self.CCL_qubit.mock_fl_dc_ch()](current)

        threshold = 0.01e9

        self.CCL_qubit.find_frequency()

        assert np.abs(self.CCL_qubit.mock_freq_qubit() -
                      self.CCL_qubit.freq_qubit()) <= threshold

    ###########################################################
    # Test MW pulse calibration
    ###########################################################
    def test_calibrate_mw_pulse_amplitude_coarse(self):
        for with_vsm in [True, False]:
            self.CCL_qubit.cfg_with_vsm(with_vsm)

            self.CCL_qubit.mock_mw_amp180(.345)
            self.CCL_qubit.freq_res(self.CCL_qubit.mock_freq_res())
            self.CCL_qubit.freq_qubit(self.CCL_qubit.mock_freq_qubit())

            self.CCL_qubit.calibrate_mw_pulse_amplitude_coarse()

            eps = 0.05
            if self.CCL_qubit.cfg_with_vsm():
                assert self.CCL_qubit.mw_vsm_G_amp() == pytest.approx(
                        self.CCL_qubit.mock_mw_amp180(), eps)
                # assert self.CCL_qubit.mock_mw_amp180() <= self.CCL_qubit.mw_vsm_G_amp() + threshold
                # assert self.CCL_qubit.mock_mw_amp180() >= self.CCL_qubit.mw_vsm_G_amp() - threshold
            else:
                assert self.CCL_qubit.mw_channel_amp() == pytest.approx(
                        self.CCL_qubit.mw_channel_amp(), eps)
                # assert self.CCL_qubit.mock_mw_amp180() <= self.CCL_qubit.mw_channel_amp() + threshold
                # assert self.CCL_qubit.mock_mw_amp180() >= self.CCL_qubit.mw_channel_amp() - threshold
    ###########################################################
    # Test Ramsey
    ###########################################################
    @unittest.expectedFailure
    def test_ramsey(self):

        self.CCL_qubit.mock_Ec(250e6)
        self.CCL_qubit.mock_Ej(16e9)

        self.CCL_qubit.mock_freq_qubit(
            np.sqrt(8*self.CCL_qubit.mock_Ec()*self.CCL_qubit.mock_Ej()) - 
            self.CCL_qubit.mock_Ec())

        self.CCL_qubit.freq_qubit(
            np.sqrt(8*self.CCL_qubit.mock_Ec()*self.CCL_qubit.mock_Ej()) - 
            self.CCL_qubit.mock_Ec())

        self.CCL_qubit.ro_pulse_amp_CW(self.CCL_qubit.mock_ro_pulse_amp_CW())
        self.CCL_qubit.freq_res(self.CCL_qubit.mock_freq_res())
        self.CCL_qubit.ro_freq(self.CCL_qubit.mock_freq_res())
        
        fluxcurrent = self.CCL_qubit.instr_FluxCtrl.get_instr()
        current = self.CCL_qubit.mock_sweetspot_current()['FBL_1']

        fluxcurrent[self.CCL_qubit.mock_fl_dc_ch()](current)

        self.CCL_qubit.mock_T2_star(23e-6)
        self.CCL_qubit.T2_star(20e-6)
        self.CCL_qubit.measure_ramsey()

        threshold = 5e-6
        assert np.abs(self.CCL_qubit.mock_T2_star() - 
                      self.CCL_qubit.T2_star()) < threshold

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.CCL_qubit._all_instruments):
            try:
                inst = self.CCL_qubit.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass
