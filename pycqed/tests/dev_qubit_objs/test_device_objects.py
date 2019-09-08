import unittest
import numpy as np
import os
import pycqed as pq
import time

from pytest import approx
import pycqed.analysis.analysis_toolbox as a_tools

import pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound as sh
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG

from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer


from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon import Tektronix_driven_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CC_transmon import CBox_v3_driven_transmon, QWG_driven_transmon
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL, CCL
from pycqed.instrument_drivers.physical_instruments.QuTech_QCC import dummy_QCC, QCC


from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

from pycqed.instrument_drivers.meta_instrument import device_object_CCL as do


try:
    import openql
    openql_import_fail = False
except:
    openql_import_fail = True


class Test_Device_obj(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        This sets up a mock setup using a CCL to control multiple qubits
        """
        self.station = station.Station()

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')
        self.SH = sh.virtual_SignalHound_USB_SA124B('SH')
        self.UHFQC_0 = UHF.UHFQC(name='UHFQC_0', server='emulator',
                                 device='dev2109', interface='1GbE')

        self.UHFQC_1 = UHF.UHFQC(name='UHFQC_1', server='emulator',
                                 device='dev2110', interface='1GbE')

        self.UHFQC_2 = UHF.UHFQC(name='UHFQC_2', server='emulator',
                                 device='dev2111', interface='1GbE')

        self.CCL = dummy_CCL('CCL')
        self.QCC = dummy_QCC('QCC')
        self.VSM = Dummy_Duplexer('VSM')

        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        # Required to set it to the testing datadir
        test_datadir = os.path.join(pq.__path__[0], 'tests', 'test_output')
        self.MC.datadir(test_datadir)
        a_tools.datadir = self.MC.datadir()

        self.AWG_mw_0 = HDAWG.ZI_HDAWG8(
            name='AWG_mw_0', server='emulator', num_codewords=32, device='dev8026', interface='1GbE')

        self.AWG_mw_1 = HDAWG.ZI_HDAWG8(
            name='AWG_mw_1', server='emulator', num_codewords=32, device='dev8027', interface='1GbE')
        self.AWG_flux_0 = HDAWG.ZI_HDAWG8(
            name='AWG_flux_0', server='emulator', num_codewords=32, device='dev8028', interface='1GbE')

        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG_mw_0.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        self.ro_lutman_0 = UHFQC_RO_LutMan(
            'ro_lutman_0', feedline_number=0, feedline_map='S17', num_res=9)
        self.ro_lutman_0.AWG(self.UHFQC_0.name)

        self.ro_lutman_1 = UHFQC_RO_LutMan(
            'ro_lutman_1', feedline_number=1, feedline_map='S17', num_res=9)
        self.ro_lutman_1.AWG(self.UHFQC_1.name)

        self.ro_lutman_2 = UHFQC_RO_LutMan(
            'ro_lutman_2', feedline_number=2, feedline_map='S17', num_res=9)
        self.ro_lutman_2.AWG(self.UHFQC_2.name)


        # Assign instruments
        qubits = []
        for q_idx in range(17):
            q = ct.CCLight_Transmon('q{}'.format(q_idx))
            qubits.append(q)

            q.instr_LutMan_MW(self.AWG8_VSM_MW_LutMan.name)
            q.instr_LO_ro(self.MW1.name)
            q.instr_LO_mw(self.MW2.name)
            q.instr_spec_source(self.MW3.name)

            if q_idx in [13, 16]:
                q.instr_acquisition(self.UHFQC_0.name)
                q.instr_LutMan_RO(self.ro_lutman_0.name)
            elif q_idx in [1, 4, 5, 7, 8, 10, 11, 14, 15]:
                q.instr_acquisition(self.UHFQC_1.name)
                q.instr_LutMan_RO(self.ro_lutman_1.name)
            elif q_idx in [0, 2, 3, 6, 9, 12]:
                q.instr_acquisition(self.UHFQC_2.name)
                q.instr_LutMan_RO(self.ro_lutman_2.name)

            q.instr_VSM(self.VSM.name)
            q.instr_CC(self.CCL.name)
            q.instr_MC(self.MC.name)

            q.instr_SH(self.SH.name)

            config_fn = os.path.join(
                pq.__path__[0], 'tests', 'test_cfg_CCL.json')
            q.cfg_openql_platform_fn(config_fn)

            # Setting some "random" initial parameters
            q.ro_freq(5.43e9+q_idx*50e6)
            q.ro_freq_mod(200e6)

            q.freq_qubit(4.56e9+q_idx*50e6)
            q.freq_max(4.62e9+q_idx*50e6)

            q.mw_freq_mod(-100e6)
            q.mw_awg_ch(1)
            q.cfg_qubit_nr(q_idx)
            # q.mw_vsm_delay(15)
            q.mw_mixer_offs_GI(.1)
            q.mw_mixer_offs_GQ(.2)
            q.mw_mixer_offs_DI(.3)
            q.mw_mixer_offs_DQ(.4)

        # Set up the device object and set required params
        self.device = do.DeviceCCL('device')
        self.device.qubits([q.name for q in qubits])
        self.device.instr_CC(self.CCL.name)
        self.device.instr_AWG_mw_0(self.AWG_mw_0.name)
        self.device.instr_AWG_mw_1(self.AWG_mw_1.name)
        self.device.instr_AWG_flux_0(self.AWG_flux_0.name)

    def test_get_dio_map(self):
        self.device.instr_CC(self.CCL.name)
        dio_map = self.device.dio_map()
        expected_dio_map = {'ro_0': 1,
                            'ro_1': 2,
                            'flux_0': 3,
                            'mw_0': 4,
                            'mw_1': 5}
        assert dio_map == expected_dio_map

        self.device.instr_CC(self.QCC.name)
        dio_map = self.device.dio_map()
        expected_dio_map = {'ro_0': 1,
                            'ro_1': 2,
                            'ro_2': 3,
                            'mw_0': 4,
                            'mw_1': 5,
                            'flux_0': 6,
                            'flux_1': 7,
                            'flux_2': 8,
                            'mw_2': 9,
                            'mw_3': 10,
                            'mw_4': 11
                            }
        assert dio_map == expected_dio_map

    def test_prepare_timing_CCL(self):
        self.device.instr_CC(self.CCL.name)
        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-40e-9)
        self.device.tim_mw_latency_0(20e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        # DIO timing map for CCL:
        # dio1: ro_latency_0
        # dio2: ro_latency_1
        # dio3: flux_latency_0
        # dio4: mw_latency_0
        # dio5: mw_latency_1

        assert(self.CCL.dio1_out_delay() == 12)
        assert(self.CCL.dio2_out_delay() == 11)
        assert(self.CCL.dio3_out_delay() == 0)
        assert(self.CCL.dio4_out_delay() == 3)
        assert(self.CCL.dio5_out_delay() == 2)

    def test_prepare_timing_QCC(self):
        self.device.instr_CC(self.QCC.name)
        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-40e-9)
        self.device.tim_flux_latency_1(100e-9)
        self.device.tim_mw_latency_0(20e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        assert(self.QCC.dio1_out_delay() == 12)
        assert(self.QCC.dio2_out_delay() == 11)
        assert(self.QCC.dio4_out_delay() == 3)
        assert(self.QCC.dio5_out_delay() == 2)
        assert(self.QCC.dio6_out_delay() == 0)
        assert(self.QCC.dio7_out_delay() == 7)

    def test_prepare_timing_QCC_fine(self):
        self.device.instr_CC(self.QCC.name)
        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-36e-9)
        self.device.tim_flux_latency_1(100e-9)
        self.device.tim_mw_latency_0(23e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        assert(self.QCC.dio1_out_delay() == 12)
        assert(self.QCC.dio2_out_delay() == 11)
        assert(self.QCC.dio4_out_delay() == 3)
        assert(self.QCC.dio5_out_delay() == 2)
        assert(self.QCC.dio6_out_delay() == 0)
        assert(self.QCC.dio7_out_delay() == 7)

        assert(self.AWG_flux_0.sigouts_0_delay() == approx(4e-9))
        assert(self.AWG_flux_0.sigouts_7_delay() == approx(4e-9))

        assert(self.AWG_mw_0.sigouts_7_delay() == approx(3e-9))
        assert(self.AWG_mw_0.sigouts_7_delay() == approx(3e-9))

        assert(self.AWG_mw_1.sigouts_7_delay() == approx(0))
        assert(self.AWG_mw_1.sigouts_7_delay() == approx(0))

    def test_prepare_readout_lo_freqs_config(self):
        # Test that the modulation frequencies of all qubits
        # are set correctly.
        qubits = self.device.qubits()

        self.device.ro_lo_freq(6e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 6e9
        for q in self.qubits():
            6e9 + q.ro_freq_mod() == q.ro_freq()

        self.device.ro_lo_freq(5.8e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 5.8
        for q in self.qubits():
            5.8 + q.ro_freq_mod() == q.ro_freq()



    @classmethod
    def tearDownClass(self):
        for inststr in list(self.device._all_instruments):
            try:
                inst = self.device.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass
