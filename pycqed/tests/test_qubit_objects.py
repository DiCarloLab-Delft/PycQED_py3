import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct


class Test_Qubit_Object(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.CCL_qubit = ct.CCLight_Transmon('CCL_qubit')

        self.AWG = v8.VirtualAWG8('DummyAWG8')
        self.AWG8_VSM_MW_LutMan = mwl.AWG8_VSM_MW_LutMan('MW_LutMan_VSM')
        self.AWG8_VSM_MW_LutMan.AWG(self.AWG.name)
        self.AWG8_VSM_MW_LutMan.channel_GI(1)
        self.AWG8_VSM_MW_LutMan.channel_GQ(2)
        self.AWG8_VSM_MW_LutMan.channel_DI(3)
        self.AWG8_VSM_MW_LutMan.channel_DQ(4)
        self.AWG8_VSM_MW_LutMan.Q_modulation(100e6)
        self.AWG8_VSM_MW_LutMan.sampling_rate(2.4e9)

    def test_prepare_for_timedomain(self):
        self.CCL_qubit.prepare_for_timedomain()

    def test_prepare_for_continuous_wave(self):
        self.CCL_qubit.prepare_for_continuous_wave()

    def test_prepare_for_fluxing(self):
        self.CCL_qubit.prepare_for_fluxing()

    def test_prepare_readout(self):
        self.CCL_qubit.prepare_readout()
