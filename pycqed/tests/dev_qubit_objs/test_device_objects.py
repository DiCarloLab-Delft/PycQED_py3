import unittest
import pytest
import numpy as np
import os

import pycqed as pq
from pytest import approx

import pycqed.analysis.analysis_toolbox as a_tools
from pycqed.measurement import measurement_control

import pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound as sh
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG
from pycqed.instrument_drivers.physical_instruments.QuTech_Duplexer import Dummy_Duplexer
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL
from pycqed.instrument_drivers.physical_instruments.QuTech_QCC import dummy_QCC
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import DummyTransport

from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
from pycqed.instrument_drivers.meta_instrument import device_object_CCL as do
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct

from qcodes import station


from pycqed.measurement.detector_functions import (
    Multi_Detector_UHF,
    UHFQC_input_average_detector,
    UHFQC_integrated_average_detector,
    UHFQC_integration_logging_det,
)

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

        self.MW1 = vmw.VirtualMWsource("MW1")
        self.MW2 = vmw.VirtualMWsource("MW2")
        self.MW3 = vmw.VirtualMWsource("MW3")
        self.SH = sh.virtual_SignalHound_USB_SA124B("SH")
        self.UHFQC_0 = UHF.UHFQC(
            name="UHFQC_0", server="emulator", device="dev2109", interface="1GbE"
        )

        self.UHFQC_1 = UHF.UHFQC(
            name="UHFQC_1", server="emulator", device="dev2110", interface="1GbE"
        )

        self.UHFQC_2 = UHF.UHFQC(
            name="UHFQC_2", server="emulator", device="dev2111", interface="1GbE"
        )

        self.CCL = dummy_CCL('CCL')
        self.QCC = dummy_QCC('QCC')
        self.CC = CC('CC', DummyTransport())
        self.VSM = Dummy_Duplexer('VSM')
        self.MC = measurement_control.MeasurementControl(
            "MC", live_plot_enabled=False, verbose=False
        )
        self.MC.station = self.station
        self.station.add_component(self.MC)

        # Required to set it to the testing datadir
        test_datadir = os.path.join(pq.__path__[0], "tests", "test_output")
        self.MC.datadir(test_datadir)
        a_tools.datadir = self.MC.datadir()

        self.AWG_mw_0 = HDAWG.ZI_HDAWG8(
            name="AWG_mw_0",
            server="emulator",
            num_codewords=32,
            device="dev8026",
            interface="1GbE",
        )

        self.AWG_mw_1 = HDAWG.ZI_HDAWG8(
            name="AWG_mw_1",
            server="emulator",
            num_codewords=32,
            device="dev8027",
            interface="1GbE",
        )
        self.AWG_flux_0 = HDAWG.ZI_HDAWG8(
            name="AWG_flux_0",
            server="emulator",
            num_codewords=32,
            device="dev8028",
            interface="1GbE",
        )

        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan("MW_LutMan_VSM")
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG_mw_0.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.mw_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        self.ro_lutman_0 = UHFQC_RO_LutMan(
            "ro_lutman_0", feedline_number=0, feedline_map="S17", num_res=9
        )
        self.ro_lutman_0.AWG(self.UHFQC_0.name)

        self.ro_lutman_1 = UHFQC_RO_LutMan(
            "ro_lutman_1", feedline_number=1, feedline_map="S17", num_res=9
        )
        self.ro_lutman_1.AWG(self.UHFQC_1.name)

        self.ro_lutman_2 = UHFQC_RO_LutMan(
            "ro_lutman_2", feedline_number=2, feedline_map="S17", num_res=9
        )
        self.ro_lutman_2.AWG(self.UHFQC_2.name)

        # Assign instruments
        qubits = []
        for q_idx in range(17):
            q = ct.CCLight_Transmon("q{}".format(q_idx))
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

            config_fn = os.path.join(pq.__path__[0], "tests", "test_cfg_CCL.json")
            q.cfg_openql_platform_fn(config_fn)

            # Setting some "random" initial parameters
            q.ro_freq(5.43e9 + q_idx * 50e6)
            q.ro_freq_mod(200e6)

            q.freq_qubit(4.56e9 + q_idx * 50e6)
            q.freq_max(4.62e9 + q_idx * 50e6)

            q.mw_freq_mod(-100e6)
            q.mw_awg_ch(1)
            q.cfg_qubit_nr(q_idx)
            # q.mw_vsm_delay(15)
            q.mw_mixer_offs_GI(0.1)
            q.mw_mixer_offs_GQ(0.2)
            q.mw_mixer_offs_DI(0.3)
            q.mw_mixer_offs_DQ(0.4)

        # Set up the device object and set required params
        self.device = do.DeviceCCL("device")
        self.device.qubits([q.name for q in qubits])
        self.device.instr_CC(self.CCL.name)

        self.device.instr_AWG_mw_0(self.AWG_mw_0.name)
        self.device.instr_AWG_mw_1(self.AWG_mw_1.name)
        self.device.instr_AWG_flux_0(self.AWG_flux_0.name)

        self.device.ro_lo_freq(6e9)

        # Fixed by design
        self.dio_map_CCL = {"ro_0": 1, "ro_1": 2, "flux_0": 3, "mw_0": 4, "mw_1": 5}
        # Fixed by design
        self.dio_map_QCC = {
            "ro_0": 1,
            "ro_1": 2,
            "ro_2": 3,
            "mw_0": 4,
            "mw_1": 5,
            "flux_0": 6,
            "flux_1": 7,
            "flux_2": 8,
            "mw_2": 9,
            "mw_3": 10,
            "mw_4": 11,
        }
        # Modular, arbitrary example here
        self.dio_map_CC = {
            "ro_0": 0,
            "ro_1": 1,
            "ro_2": 2,
            "mw_0": 3,
            "mw_1": 4,
            "flux_0": 6,
            "flux_1": 7,
            "flux_2": 8,
        }

        self.device.dio_map(self.dio_map_CCL)

    def test_get_dio_map(self):
        self.device.instr_CC(self.CCL.name)
        # 2020-03-20
        # dio_map need to be specified manually by the user for each setup
        # this is necessary due to the new modularity of CC
        expected_dio_map = self.dio_map_CCL
        self.device.dio_map(expected_dio_map)
        dio_map = self.device.dio_map()

        assert dio_map == expected_dio_map

        self.device.instr_CC(self.QCC.name)
        expected_dio_map = self.dio_map_QCC
        self.device.dio_map(expected_dio_map)
        dio_map = self.device.dio_map()

        assert dio_map == expected_dio_map

    def test_get_dio_map_CC(self):
        self.device.instr_CC(self.CC.name)
        # 2020-03-20
        # dio_map need to be specified manually by the user for each setup
        # this is necessary due to the new modularity of CC
        expected_dio_map = self.dio_map_CC
        self.device.dio_map(expected_dio_map)
        dio_map = self.device.dio_map()

        assert dio_map == expected_dio_map

    def test_prepare_timing_CCL(self):
        self.device.instr_CC(self.CCL.name)
        self.device.dio_map(self.dio_map_CCL)

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

        assert self.CCL.dio1_out_delay() == 12
        assert self.CCL.dio2_out_delay() == 11
        assert self.CCL.dio3_out_delay() == 0
        assert self.CCL.dio4_out_delay() == 3
        assert self.CCL.dio5_out_delay() == 2

    def test_prepare_timing_QCC(self):
        self.device.instr_CC(self.QCC.name)
        self.device.dio_map(self.dio_map_QCC)

        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-40e-9)
        self.device.tim_flux_latency_1(100e-9)
        self.device.tim_mw_latency_0(20e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        assert self.QCC.dio1_out_delay() == 12
        assert self.QCC.dio2_out_delay() == 11
        assert self.QCC.dio4_out_delay() == 3
        assert self.QCC.dio5_out_delay() == 2
        assert self.QCC.dio6_out_delay() == 0
        assert self.QCC.dio7_out_delay() == 7

    def test_prepare_timing_QCC_fine(self):
        self.device.instr_CC(self.QCC.name)
        self.device.dio_map(self.dio_map_QCC)

        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-36e-9)
        self.device.tim_flux_latency_1(100e-9)
        self.device.tim_mw_latency_0(23e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        assert self.QCC.dio1_out_delay() == 12
        assert self.QCC.dio2_out_delay() == 11
        assert self.QCC.dio4_out_delay() == 3
        assert self.QCC.dio5_out_delay() == 2
        assert self.QCC.dio6_out_delay() == 0
        assert self.QCC.dio7_out_delay() == 7

        assert self.AWG_flux_0.sigouts_0_delay() == approx(4e-9)
        assert self.AWG_flux_0.sigouts_7_delay() == approx(4e-9)

        assert self.AWG_mw_0.sigouts_7_delay() == approx(3e-9)
        assert self.AWG_mw_0.sigouts_7_delay() == approx(3e-9)

        assert self.AWG_mw_1.sigouts_7_delay() == approx(0)
        assert self.AWG_mw_1.sigouts_7_delay() == approx(0)

    def test_prepare_timing_CC(self):
        self.device.instr_CC(self.CC.name)
        self.device.dio_map(self.dio_map_CC)

        self.device.tim_ro_latency_0(200e-9)
        self.device.tim_ro_latency_1(180e-9)
        self.device.tim_flux_latency_0(-40e-9)
        self.device.tim_flux_latency_1(100e-9)
        self.device.tim_mw_latency_0(20e-9)
        self.device.tim_mw_latency_1(0e-9)

        self.device.prepare_timing()

        assert self.CC.dio0_out_delay() == 12
        assert self.CC.dio1_out_delay() == 11
        assert self.CC.dio3_out_delay() == 3
        assert self.CC.dio4_out_delay() == 2
        assert self.CC.dio6_out_delay() == 0
        assert self.CC.dio7_out_delay() == 7

    def test_prepare_readout_lo_freqs_config(self):
        # Test that the modulation frequencies of all qubits
        # are set correctly.
        self.device.ro_acq_weight_type("optimal")
        qubits = self.device.qubits()

        self.device.ro_lo_freq(6e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 6e9
        for qname in qubits:
            q = self.device.find_instrument(qname)
            6e9 + q.ro_freq_mod() == q.ro_freq()

        self.device.ro_lo_freq(5.8e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 5.8e9
        for qname in qubits:
            q = self.device.find_instrument(qname)
            5.8e9 + q.ro_freq_mod() == q.ro_freq()

        q = self.device.find_instrument("q5")
        q.instr_LO_ro(self.MW3.name)
        with pytest.raises(ValueError):
            self.device.prepare_readout(qubits=qubits)
        q.instr_LO_ro(self.MW1.name)

    def test_prepare_readout_assign_weights(self):
        self.device.ro_lo_freq(6e9)

        self.device.ro_acq_weight_type("optimal")
        qubits = self.device.qubits()

        q13 = self.device.find_instrument("q13")
        q13.ro_acq_weight_func_I(np.ones(128))
        q13.ro_acq_weight_func_Q(np.ones(128) * 0.5)

        self.device.prepare_readout(qubits=qubits)
        exp_ch_map = {
            "UHFQC_0": {"q13": 0, "q16": 1},
            "UHFQC_1": {
                "q1": 0,
                "q4": 1,
                "q5": 2,
                "q7": 3,
                "q8": 4,
                "q10": 5,
                "q11": 6,
                "q14": 7,
                "q15": 8,
            },
            "UHFQC_2": {"q0": 0, "q2": 1, "q3": 2, "q6": 3, "q9": 4, "q12": 5},
        }
        assert exp_ch_map == self.device._acq_ch_map

        qb = self.device.find_instrument("q12")
        assert qb.ro_acq_weight_chI() == 5
        assert qb.ro_acq_weight_chQ() == 6

    def test_prepare_readout_assign_weights_order_matters(self):
        # Test that the order of the channels is as in the order iterated over
        qubits = ["q2", "q3", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)
        exp_ch_map = {"UHFQC_2": {"q0": 2, "q2": 0, "q3": 1}}
        assert exp_ch_map == self.device._acq_ch_map
        qb = self.device.find_instrument("q3")
        assert qb.ro_acq_weight_chI() == 1
        assert qb.ro_acq_weight_chQ() == 2

    def test_prepare_readout_assign_weights_IQ_counts_double(self):
        qubits = ["q2", "q3", "q0", "q13", "q16"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)
        exp_ch_map = {
            "UHFQC_0": {"q13": 0, "q16": 2},
            "UHFQC_2": {"q0": 4, "q2": 0, "q3": 2},
        }
        assert exp_ch_map == self.device._acq_ch_map
        qb = self.device.find_instrument("q16")
        assert qb.ro_acq_weight_chI() == 2
        assert qb.ro_acq_weight_chQ() == 3

    def test_prepare_readout_assign_weights_too_many_raises(self):
        qubits = self.device.qubits()
        self.device.ro_acq_weight_type("SSB")
        with pytest.raises(ValueError):
            self.device.prepare_readout(qubits=qubits)

    def test_prepare_readout_resets_UHF(self):
        uhf = self.device.find_instrument("UHFQC_2")

        uhf.qas_0_correlations_5_enable(1)
        uhf.qas_0_correlations_5_source(3)
        uhf.qas_0_thresholds_5_correlation_enable(1)
        uhf.qas_0_thresholds_5_correlation_source(3)

        assert uhf.qas_0_correlations_5_enable() == 1
        assert uhf.qas_0_correlations_5_source() == 3
        assert uhf.qas_0_thresholds_5_correlation_enable() == 1
        assert uhf.qas_0_thresholds_5_correlation_source() == 3

        self.device.prepare_readout(qubits=["q0"])

        assert uhf.qas_0_correlations_5_enable() == 0
        assert uhf.qas_0_correlations_5_source() == 0
        assert uhf.qas_0_thresholds_5_correlation_enable() == 0
        assert uhf.qas_0_thresholds_5_correlation_source() == 0

    def test_prepare_ro_pulses_resonator_combinations(self):
        # because not all combinations are supported the default is to
        # support

        qubits = ["q2", "q3", "q0", "q13", "q16"]
        self.device.prepare_readout(qubits=qubits)

        # Combinations are based on qubit number
        res_combs0 = self.ro_lutman_0.resonator_combinations()
        exp_res_combs0 = [[13], [16], [13, 16]]
        assert res_combs0 == exp_res_combs0

        res_combs2 = self.ro_lutman_2.resonator_combinations()
        exp_res_combs2 = [[2], [3], [0], [2, 3, 0]]
        assert res_combs2 == exp_res_combs2

    def test_prepare_ro_pulses_lutman_pars_updated(self):
        q = self.device.find_instrument("q5")
        q.ro_pulse_amp(0.4)
        self.device.prepare_readout(["q5"])
        ro_amp = self.ro_lutman_1.M_amp_R5()
        assert ro_amp == 0.4

        q.ro_pulse_amp(0.2)
        self.device.prepare_readout(["q5"])
        ro_amp = self.ro_lutman_1.M_amp_R5()
        assert ro_amp == 0.2

    def test_prep_ro_input_avg_det(self):
        qubits = self.device.qubits()
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        exp_ch_map = {
            "UHFQC_0": {"q13": 0, "q16": 1},
            "UHFQC_1": {
                "q1": 0,
                "q4": 1,
                "q5": 2,
                "q7": 3,
                "q8": 4,
                "q10": 5,
                "q11": 6,
                "q14": 7,
                "q15": 8,
            },
            "UHFQC_2": {"q0": 0, "q2": 1, "q3": 2, "q6": 3, "q9": 4, "q12": 5},
        }

        inp_avg_det = self.device.input_average_detector
        assert isinstance(inp_avg_det, Multi_Detector_UHF)
        assert len(inp_avg_det.detectors) == 3
        for ch_det in inp_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_input_average_detector)
        # Note taht UHFQC_2 is first because q0 is the first in device.qubits
        assert inp_avg_det.value_names == [
            "UHFQC_2 ch0",
            "UHFQC_2 ch1",
            "UHFQC_1 ch0",
            "UHFQC_1 ch1",
            "UHFQC_0 ch0",
            "UHFQC_0 ch1",
        ]

    def test_prepare_ro_instantiate_detectors_int_avg(self):
        qubits = ["q13", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        int_avg_det = self.device.int_avg_det
        assert isinstance(int_avg_det, Multi_Detector_UHF)
        assert len(int_avg_det.detectors) == 3
        for ch_det in int_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_integrated_average_detector)
        # Note that UHFQC_2 is first because q0 is the first in device.qubits
        assert int_avg_det.value_names == [
            "UHFQC_0 w0 q13",
            "UHFQC_0 w1 q16",
            "UHFQC_1 w0 q1",
            "UHFQC_1 w1 q5",
            "UHFQC_2 w0 q0",
        ]

        qubits = ["q13", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)

        int_avg_det = self.device.int_avg_det
        assert isinstance(int_avg_det, Multi_Detector_UHF)
        assert len(int_avg_det.detectors) == 3
        for ch_det in int_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_integrated_average_detector)
        # Note that UHFQC_2 is first because q0 is the first in device.qubits
        assert int_avg_det.value_names == [
            "UHFQC_0 w0 q13 I",
            "UHFQC_0 w1 q13 Q",
            "UHFQC_0 w2 q16 I",
            "UHFQC_0 w3 q16 Q",
            "UHFQC_1 w0 q1 I",
            "UHFQC_1 w1 q1 Q",
            "UHFQC_1 w2 q5 I",
            "UHFQC_1 w3 q5 Q",
            "UHFQC_2 w0 q0 I",
            "UHFQC_2 w1 q0 Q",
        ]

        # Note that the order of channels gets ordered per feedline
        # because of the way the multi detector works

    def test_prepare_ro_instantiate_detectors_int_logging(self):
        qubits = ["q13", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        int_log_det = self.device.int_log_det
        assert isinstance(int_log_det, Multi_Detector_UHF)
        assert len(int_log_det.detectors) == 3
        for ch_det in int_log_det.detectors:
            assert isinstance(ch_det, UHFQC_integration_logging_det)
        # Note that UHFQC_2 is first because q0 is the first in device.qubits
        assert int_log_det.value_names == [
            "UHFQC_0 w0 q13",
            "UHFQC_0 w1 q16",
            "UHFQC_1 w0 q1",
            "UHFQC_1 w1 q5",
            "UHFQC_2 w0 q0",
        ]

        qubits = self.device.qubits()
        qubits = ["q13", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)

        int_log_det = self.device.int_log_det
        assert isinstance(int_log_det, Multi_Detector_UHF)
        assert len(int_log_det.detectors) == 3
        for ch_det in int_log_det.detectors:
            assert isinstance(ch_det, UHFQC_integration_logging_det)
        # Note that UHFQC_2 is first because q0 is the first in device.qubits
        assert int_log_det.value_names == [
            "UHFQC_0 w0 q13 I",
            "UHFQC_0 w1 q13 Q",
            "UHFQC_0 w2 q16 I",
            "UHFQC_0 w3 q16 Q",
            "UHFQC_1 w0 q1 I",
            "UHFQC_1 w1 q1 Q",
            "UHFQC_1 w2 q5 I",
            "UHFQC_1 w3 q5 Q",
            "UHFQC_2 w0 q0 I",
            "UHFQC_2 w1 q0 Q",
        ]

    def test_prepare_readout_mixer_settings(self):
        pass

    @classmethod
    def tearDownClass(self):
        for instr_name in list(self.device._all_instruments):
            try:
                inst = self.device.find_instrument(instr_name)
                inst.close()
            except KeyError:
                pass


def test_acq_ch_map_to_IQ_ch_map():

    ch_map = {
        "UHFQC_0": {"q13": 0, "q16": 2},
        "UHFQC_1": {"q1": 0, "q4": 4},
        "UHFQC_2": {"q0": 0, "q3": 2, "q6": 4},
    }

    IQ_ch_map = do._acq_ch_map_to_IQ_ch_map(ch_map)
    exp_IQ_ch_map = {
        "UHFQC_0": {"q13 I": 0, "q13 Q": 1, "q16 I": 2, "q16 Q": 3},
        "UHFQC_1": {"q1 I": 0, "q1 Q": 1, "q4 I": 4, "q4 Q": 5},
        "UHFQC_2": {"q0 I": 0, "q0 Q": 1, "q3 I": 2, "q3 Q": 3, "q6 I": 4, "q6 Q": 5},
    }

    assert IQ_ch_map == exp_IQ_ch_map
