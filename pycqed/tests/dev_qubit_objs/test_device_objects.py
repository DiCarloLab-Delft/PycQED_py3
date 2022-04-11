import unittest
import pytest
from pytest import approx
import numpy as np
import os
import pathlib

import pycqed as pq

import pycqed.measurement.openql_experiments.generate_CC_cfg_modular as gen
from pycqed.instrument_drivers.meta_instrument import device_object_CCL as do
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.instrument_drivers.meta_instrument.LutMans.base_lutman import Base_LutMan

import pycqed.analysis.analysis_toolbox as a_tools
from pycqed.measurement import measurement_control
from pycqed.measurement.detector_functions import (
    Multi_Detector_UHF,
    UHFQC_input_average_detector,
    UHFQC_integrated_average_detector,
    UHFQC_integration_logging_det,
)

from pycqed.instrument_drivers.virtual_instruments.virtual_SignalHound import virtual_SignalHound_USB_SA124B
from pycqed.instrument_drivers.virtual_instruments.virtual_MW_source import VirtualMWsource

from pycqed.instrument_drivers.library.Transport import DummyTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController import UHFQC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.QuTech_VSM_Module import Dummy_QuTechVSMModule

from qcodes import station, Instrument

this_path = pathlib.Path(__file__).parent
output_path = pathlib.Path(this_path) / 'test_output_cc'
platf_cfg_path = output_path / 'config_cc_s17_direct_iq_openql_0_10.json'


class Test_Device_obj(unittest.TestCase):
    # FIXME: using setUpClass is more efficient, but failing tests tend to influence each other, making debugging difficult
    #  If we stick with setUp, 'cls' should be renamed to 'self'
    # @classmethod
    # def setUpClass(cls):
    def setUp(cls):
        """
        This sets up a mock setup using a CC to control multiple qubits
        """
        # generate OpenQL configuration
        gen.generate_config_modular(platf_cfg_path)


        cls.station = station.Station()

        cls.CC = CC('CC', DummyTransport())
        cls.UHFQC_0 = UHFQC(name="UHFQC_0", server="emulator", device="dev2109", interface="1GbE")
        cls.UHFQC_1 = UHFQC(name="UHFQC_1", server="emulator", device="dev2110", interface="1GbE")
        cls.UHFQC_2 = UHFQC(name="UHFQC_2", server="emulator", device="dev2111", interface="1GbE")
        cls.AWG_mw_0 = ZI_HDAWG8(
            name="AWG_mw_0",
            server="emulator",
            num_codewords=128,
            device="dev8026",
            interface="1GbE",
        )

        cls.AWG_mw_1 = ZI_HDAWG8(
            name="AWG_mw_1",
            server="emulator",
            num_codewords=128,
            device="dev8027",
            interface="1GbE",
        )
        cls.AWG_flux_0 = ZI_HDAWG8(
            name="AWG_flux_0",
            server="emulator",
            num_codewords=128,
            device="dev8028",
            interface="1GbE",
        )
        cls.VSM = Dummy_QuTechVSMModule('VSM')

        cls.MW1 = VirtualMWsource("MW1")
        cls.MW2 = VirtualMWsource("MW2")
        cls.MW3 = VirtualMWsource("MW3")
        cls.SH = virtual_SignalHound_USB_SA124B("SH")

        cls.MC = measurement_control.MeasurementControl("MC", live_plot_enabled=False, verbose=False)
        cls.MC.station = cls.station
        cls.station.add_component(cls.MC)

        # Required to set it to the testing datadir
        test_datadir = os.path.join(pq.__path__[0], "tests", "test_output")
        cls.MC.datadir(test_datadir)
        a_tools.datadir = cls.MC.datadir()


        if 0: # FIXME: PR #658: test broken by commit bd19f56
            cls.mw_lutman = mwl.AWG8_VSM_MW_LutMan("MW_LutMan_VSM")
            cls.mw_lutman.AWG(cls.AWG_mw_0.name)
            cls.mw_lutman.channel_GI(1)
            cls.mw_lutman.channel_GQ(2)
            cls.mw_lutman.channel_DI(3)
            cls.mw_lutman.channel_DQ(4)
        else: # FIXME: workaround
            cls.mw_lutman = mwl.AWG8_MW_LutMan("MW_LutMan")
            cls.mw_lutman.AWG(cls.AWG_mw_0.name)
            cls.mw_lutman.channel_I(1)
            cls.mw_lutman.channel_Q(2)

        cls.mw_lutman.mw_modulation(100e6)
        cls.mw_lutman.sampling_rate(2.4e9)

        cls.ro_lutman_0 = UHFQC_RO_LutMan("ro_lutman_0", feedline_number=0, feedline_map="S17", num_res=9)
        cls.ro_lutman_0.AWG(cls.UHFQC_0.name)

        cls.ro_lutman_1 = UHFQC_RO_LutMan("ro_lutman_1", feedline_number=1, feedline_map="S17", num_res=9)
        cls.ro_lutman_1.AWG(cls.UHFQC_1.name)

        cls.ro_lutman_2 = UHFQC_RO_LutMan("ro_lutman_2", feedline_number=2, feedline_map="S17", num_res=9)
        cls.ro_lutman_2.AWG(cls.UHFQC_2.name)

        # Assign instruments
        qubits = []
        for q_idx in range(17):
            q = CCLight_Transmon("q{}".format(q_idx))
            qubits.append(q)

            q.instr_LutMan_MW(cls.mw_lutman.name)
            q.instr_LO_ro(cls.MW1.name)
            q.instr_LO_mw(cls.MW2.name)
            q.instr_spec_source(cls.MW3.name)

            # map qubits to UHFQC, *must* match mapping inside Base_RO_LutMan (Yuk)
            if 0:
                if q_idx in [13, 16]:
                    q.instr_acquisition(cls.UHFQC_0.name)
                    q.instr_LutMan_RO(cls.ro_lutman_0.name)
                elif q_idx in [1, 4, 5, 7, 8, 10, 11, 14, 15]:
                    q.instr_acquisition(cls.UHFQC_1.name)
                    q.instr_LutMan_RO(cls.ro_lutman_1.name)
                elif q_idx in [0, 2, 3, 6, 9, 12]:
                    q.instr_acquisition(cls.UHFQC_2.name)
                    q.instr_LutMan_RO(cls.ro_lutman_2.name)
            else:
                if q_idx in [6, 11]:
                    q.instr_acquisition(cls.UHFQC_0.name)
                    q.instr_LutMan_RO(cls.ro_lutman_0.name)
                elif q_idx in [0, 1, 2, 3, 7, 8, 12, 13, 15]:
                    q.instr_acquisition(cls.UHFQC_1.name)
                    q.instr_LutMan_RO(cls.ro_lutman_1.name)
                elif q_idx in [4, 5, 9, 10, 14, 16]:
                    q.instr_acquisition(cls.UHFQC_2.name)
                    q.instr_LutMan_RO(cls.ro_lutman_2.name)

            # q.instr_VSM(cls.VSM.name)
            q.cfg_with_vsm(False)
            q.instr_CC(cls.CC.name)
            q.instr_MC(cls.MC.name)

            q.instr_SH(cls.SH.name)

            q.cfg_openql_platform_fn(str(platf_cfg_path))

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
        cls.device = do.DeviceCCL("device")
        cls.device.qubits([q.name for q in qubits])
        cls.device.instr_CC(cls.CC.name)
        cls.device.instr_MC(cls.MC.name)
        cls.device.cfg_openql_platform_fn(str(platf_cfg_path))

        cls.device.instr_AWG_mw_0(cls.AWG_mw_0.name)
        cls.device.instr_AWG_mw_1(cls.AWG_mw_1.name)
        cls.device.instr_AWG_flux_0(cls.AWG_flux_0.name)

        if 0:
            cls.device.ro_lo_freq(6e9)
        else:  # FIXME: frequency now per LutMan
            cls.ro_lutman_0.LO_freq(6e9)
            cls.ro_lutman_1.LO_freq(6e9)
            cls.ro_lutman_2.LO_freq(6e9)


        if 0:  # FIXME: CCL/QCC deprecated
            # Fixed by design
            cls.dio_map_CCL = {"ro_0": 1, "ro_1": 2, "flux_0": 3, "mw_0": 4, "mw_1": 5}
            # Fixed by design
            cls.dio_map_QCC = {
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
        cls.dio_map_CC = {
            "ro_0": 0,
            "ro_1": 1,
            "ro_2": 2,
            "mw_0": 3,
            "mw_1": 4,
            "flux_0": 6,
            "flux_1": 7,
            "flux_2": 8,
        }

        cls.device.dio_map(cls.dio_map_CC)

    # FIXME
    # @classmethod
    # def tearDownClass(cls):
    def tearDown(self):
        try:
            Instrument.close_all()
        except Exception as e:
            print(f"Caught exception during tearDown: {str(e)}")

    ##############################################
    # HAL_Shim_MQ
    # FIXME: split into separate test class, like in test_qubit_objects.py
    ##############################################

    @unittest.skip("CCL/QCC is removed")
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

    @unittest.skip("CCL is removed")
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

    @unittest.skip("QCC is removed")
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

    @unittest.skip("QCC is removed")
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

        if 0: # # FIXME: PR #658: test broken by commit bd19f56
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
        # Test that the modulation frequencies of all qubits are set correctly.
        self.device.ro_acq_weight_type("optimal")
        qubits = self.device.qubits()

        self.ro_lutman_0.LO_freq(6e9)
        self.ro_lutman_1.LO_freq(6e9)
        self.ro_lutman_2.LO_freq(6e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 6e9
        for qname in qubits:
            q = self.device.find_instrument(qname)
            assert 6e9 + q.ro_freq_mod() == q.ro_freq()


        self.ro_lutman_0.LO_freq(5.8e9)
        self.ro_lutman_1.LO_freq(5.8e9)
        self.ro_lutman_2.LO_freq(5.8e9)
        self.device.prepare_readout(qubits=qubits)

        # MW1 is specified as the readout LO source
        assert self.MW1.frequency() == 5.8e9
        for qname in qubits:
            q = self.device.find_instrument(qname)
            assert 5.8e9 + q.ro_freq_mod() == q.ro_freq()


        # FIXME: no longer raises exception
        # q = self.device.find_instrument("q5")
        # q.instr_LO_ro(self.MW3.name)
        # with pytest.raises(ValueError):
        #     self.device.prepare_readout(qubits=qubits)
        # q.instr_LO_ro(self.MW1.name)

    def test_prepare_readout_assign_weights(self):
        self.ro_lutman_0.LO_freq(6e9)
        self.ro_lutman_1.LO_freq(6e9)
        self.ro_lutman_2.LO_freq(6e9)

        self.device.ro_acq_weight_type("optimal")
        qubits = self.device.qubits()

        q13 = self.device.find_instrument("q13")
        q13.ro_acq_weight_func_I(np.ones(128))
        q13.ro_acq_weight_func_Q(np.ones(128) * 0.5)

        self.device.prepare_readout(qubits=qubits)
        exp_ch_map = {
            'UHFQC_1': {'q0': 0, 'q1': 1, 'q2': 2, 'q3': 3, 'q7': 4, 'q8': 5, 'q12': 6, 'q13': 7, 'q15': 8},
            'UHFQC_2': {'q4': 0, 'q5': 1, 'q9': 2, 'q10': 3, 'q14': 4, 'q16': 5},
            'UHFQC_0': {'q6': 0, 'q11': 1}
        }
        assert exp_ch_map == self.device._acq_ch_map

        qb = self.device.find_instrument("q12")
        assert qb.ro_acq_weight_chI() == 6
        assert qb.ro_acq_weight_chQ() == 7

    def test_prepare_readout_assign_weights_order_matters(self):
        # Test that the order of the channels is as in the order iterated over
        qubits = ["q2", "q3", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        exp_ch_map = {"UHFQC_1": {"q0": 2, "q2": 0, "q3": 1}}
        assert exp_ch_map == self.device._acq_ch_map
        qb = self.device.find_instrument("q3")
        assert qb.ro_acq_weight_chI() == 1
        assert qb.ro_acq_weight_chQ() == 2

    def test_prepare_readout_assign_weights_IQ_counts_double(self):
        qubits = ["q2", "q3", "q0", "q13", "q16"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)
        exp_ch_map = {
            'UHFQC_1': {'q0': 4, 'q13': 6, 'q2': 0, 'q3': 2},
            'UHFQC_2': {'q16': 0}
        }

        assert exp_ch_map == self.device._acq_ch_map
        qb = self.device.find_instrument("q16")
        assert qb.ro_acq_weight_chI() == 0
        assert qb.ro_acq_weight_chQ() == 1

    def test_prepare_readout_assign_weights_too_many_raises(self):
        qubits = self.device.qubits()
        self.device.ro_acq_weight_type("SSB")
        with pytest.raises(ValueError):
            self.device.prepare_readout(qubits=qubits)

    def test_prepare_readout_resets_UHF(self):
        uhf = self.device.find_instrument("UHFQC_1")

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
        # exp_res_combs0 = [[11]]
        exp_res_combs0 = [[6]]
        assert res_combs0 == exp_res_combs0

        res_combs1 = self.ro_lutman_1.resonator_combinations()
        exp_res_combs1 = [[2, 3, 0, 13]]
        assert res_combs1 == exp_res_combs1

        res_combs2 = self.ro_lutman_2.resonator_combinations()
        exp_res_combs2 = [[16]]
        assert res_combs2 == exp_res_combs2

    def test_prepare_ro_pulses_lutman_pars_updated(self):
        q = self.device.find_instrument("q5")
        q.ro_pulse_amp(0.4)
        self.device.prepare_readout(["q5"])
        ro_amp = self.ro_lutman_2.M_amp_R5()
        assert ro_amp == 0.4

        q.ro_pulse_amp(0.2)
        self.device.prepare_readout(["q5"])
        ro_amp = self.ro_lutman_2.M_amp_R5()
        assert ro_amp == 0.2

    def test_prep_ro_input_avg_det(self):
        qubits = self.device.qubits()
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        inp_avg_det = self.device.get_input_avg_det()
        assert isinstance(inp_avg_det, Multi_Detector_UHF)
        assert len(inp_avg_det.detectors) == 3
        for ch_det in inp_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_input_average_detector)

        # Note that UHFQC_1 is first because q0 is the first in device.qubits
        assert inp_avg_det.value_names == [
            "UHFQC_1 ch0",
            "UHFQC_1 ch1",
            "UHFQC_2 ch0",
            "UHFQC_2 ch1",
            "UHFQC_0 ch0",
            "UHFQC_0 ch1",
        ]

    def test_prepare_ro_instantiate_detectors_int_avg_optimal(self):
        qubits = ["q11", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        int_avg_det = self.device.get_int_avg_det()
        assert isinstance(int_avg_det, Multi_Detector_UHF)
        assert len(int_avg_det.detectors) == 3
        for ch_det in int_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_integrated_average_detector)

        assert int_avg_det.value_names == [
            "UHFQC_0 w0 q11",
            "UHFQC_2 w0 q16",
            "UHFQC_2 w1 q5",
            "UHFQC_1 w0 q1",
            "UHFQC_1 w1 q0",
        ]

    def test_prepare_ro_instantiate_detectors_int_avg_ssb(self):
        qubits = ["q11", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)

        int_avg_det = self.device.get_int_avg_det()
        assert isinstance(int_avg_det, Multi_Detector_UHF)
        assert len(int_avg_det.detectors) == 3
        for ch_det in int_avg_det.detectors:
            assert isinstance(ch_det, UHFQC_integrated_average_detector)

        assert int_avg_det.value_names == [
            "UHFQC_0 w0 q11 I",
            "UHFQC_0 w1 q11 Q",
            "UHFQC_2 w0 q16 I",
            "UHFQC_2 w1 q16 Q",
            "UHFQC_2 w2 q5 I",
            "UHFQC_2 w3 q5 Q",
            "UHFQC_1 w0 q1 I",
            "UHFQC_1 w1 q1 Q",
            "UHFQC_1 w2 q0 I",
            "UHFQC_1 w3 q0 Q",
        ]
        # Note that the order of channels gets ordered per feedline
        # because of the way the multi detector works

    def test_prepare_ro_instantiate_detectors_int_logging_optimal(self):
        qubits = ["q11", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("optimal")
        self.device.prepare_readout(qubits=qubits)

        int_log_det = self.device.get_int_logging_detector()
        assert isinstance(int_log_det, Multi_Detector_UHF)
        assert len(int_log_det.detectors) == 3
        for ch_det in int_log_det.detectors:
            assert isinstance(ch_det, UHFQC_integration_logging_det)

        assert int_log_det.value_names == [
            "UHFQC_0 w0 q11",
            "UHFQC_2 w0 q16",
            "UHFQC_2 w1 q5",
            "UHFQC_1 w0 q1",
            "UHFQC_1 w1 q0",
        ]

        qubits = self.device.qubits()

    def test_prepare_ro_instantiate_detectors_int_logging_ssb(self):
        qubits = ["q11", "q16", "q1", "q5", "q0"]
        self.device.ro_acq_weight_type("SSB")
        self.device.prepare_readout(qubits=qubits)

        int_log_det = self.device.get_int_logging_detector()
        assert isinstance(int_log_det, Multi_Detector_UHF)
        assert len(int_log_det.detectors) == 3
        for ch_det in int_log_det.detectors:
            assert isinstance(ch_det, UHFQC_integration_logging_det)

        assert int_log_det.value_names == [
            "UHFQC_0 w0 q11 I",
            "UHFQC_0 w1 q11 Q",
            "UHFQC_2 w0 q16 I",
            "UHFQC_2 w1 q16 Q",
            "UHFQC_2 w2 q5 I",
            "UHFQC_2 w3 q5 Q",
            "UHFQC_1 w0 q1 I",
            "UHFQC_1 w1 q1 Q",
            "UHFQC_1 w2 q0 I",
            "UHFQC_1 w3 q0 Q",
        ]

    def test_prepare_readout_mixer_settings(self):
        pass

    def test_acq_ch_map_to_IQ_ch_map(self):
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

    ##############################################
    # LutMan
    # FIXME: move
    ##############################################

    def test_base_lutman_make(self):
        # make first time
        n1 = Base_LutMan.make()
        assert n1 == 4

        # make again, should now return 0
        n2 = Base_LutMan.make()
        assert n2 == 0

        # change some LutMan parameter, should rebuild
        old_val = self.mw_lutman.mw_modulation()
        self.mw_lutman.mw_modulation(old_val - 1e6)  # change modulation.
        n3 = Base_LutMan.make()
        self.mw_lutman.mw_modulation(old_val)  # restore modulation.
        assert n3 == 1

        # manually change LutMan
        # note that load_ef_rabi_pulses_to_AWG_lookuptable already updates everything, but sidesteps make, which will
        # the update again. Eventually, everything needs to go through make
        self.mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()
        n4 = Base_LutMan.make()
        assert n4 == 1

    ##############################################
    # HAL_Device
    # FIXME: split into separate test class, like in test_qubit_objects.py
    ##############################################

    ### measure_two_qubit_randomized_benchmarking

    def test_measure_two_qubit_randomized_benchmarking_sequential(self):
        self.device.measure_two_qubit_randomized_benchmarking(qubits=["q8", "q10"])

    # @unittest.skip("FIXME: WIP")
    # # FIXME: add other parallel variants once they work
    # def test_measure_two_qubit_randomized_benchmarking_parallel(self):
    #     self.device.measure_two_qubit_randomized_benchmarking(qubits=["q8", "q10"], parallel=True)

    ### measure_interleaved_randomized_benchmarking_statistics

    ### measure_two_qubit_interleaved_randomized_benchmarking

    def test_measure_two_qubit_interleaved_randomized_benchmarking(self):
        self.device.measure_two_qubit_interleaved_randomized_benchmarking(qubits=["q8", "q10"])

    ### measure_single_qubit_interleaved_randomized_benchmarking_parking

    ### measure_single_qubit_randomized_benchmarking_parking

    ### measure_two_qubit_purity_benchmarking

    ### measure_two_qubit_character_benchmarking

    ### measure_two_qubit_simultaneous_randomized_benchmarking

    ### measure_multi_qubit_simultaneous_randomized_benchmarking







    def test_measure_two_qubit_simultaneous_randomized_benchmarking(self):
        self.device.measure_two_qubit_simultaneous_randomized_benchmarking(qubits=["q8", "q10"])

    def test_measure_multi_qubit_simultaneous_randomized_benchmarking(self):
        self.device.measure_multi_qubit_simultaneous_randomized_benchmarking(qubits=["q8", "q10"])


    def test_measure_two_qubit_allxy(self):
        self.device.measure_two_qubit_allxy("q8", "q10", detector="int_avg")

    # FIXME: add more tests, above just some random routines were added