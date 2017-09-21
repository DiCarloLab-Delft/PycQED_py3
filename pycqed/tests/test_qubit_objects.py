import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station


class Test_Qubit_Object(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.CCL_qubit = ct.CCLight_Transmon('CCL_qubit')

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')

        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        self.AWG = v8.VirtualAWG8('DummyAWG8')
        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.Q_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

        # Assign instruments
        self.CCL_qubit.instr_LutMan_MW(self.AWG8_VSM_MW_LutMan.name)
        self.CCL_qubit.instr_LO(self.MW1.name)
        self.CCL_qubit.instr_cw_source(self.MW2.name)
        self.CCL_qubit.instr_td_source(self.MW3.name)

    def test_prepare_for_timedomain(self):
        self.CCL_qubit.prepare_for_timedomain()

    def test_prepare_for_continuous_wave(self):
        self.CCL_qubit.spec_pow(-20)
        self.CCL_qubit.prepare_for_continuous_wave()

    def test_prepare_for_fluxing(self):
        self.CCL_qubit.prepare_for_fluxing()

    def test_prepare_readout(self):
        self.CCL_qubit.prepare_readout()

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.CCL_qubit._all_instruments):
            try:
                inst = self.CCL_qubit.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass

