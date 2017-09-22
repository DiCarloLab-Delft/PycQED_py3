import unittest
import numpy as np
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
import pycqed.instrument_drivers.virtual_instruments.virtual_MW_source as vmw
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon as ct
from pycqed.measurement import measurement_control
from qcodes import station

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.dummy_UHFQC import dummy_UHFQC


from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon import Tektronix_driven_transmon
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CC_transmon import CBox_v3_driven_transmon, QWG_driven_transmon
from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import dummy_CCL


class Test_Qubit_Object(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.CCL_qubit = ct.CCLight_Transmon('CCL_qubit')

        self.MW1 = vmw.VirtualMWsource('MW1')
        self.MW2 = vmw.VirtualMWsource('MW2')
        self.MW3 = vmw.VirtualMWsource('MW3')

        self.UHFQC = dummy_UHFQC('UHFQC')

        self.CCL = dummy_CCL('CCL')

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
        self.CCL_qubit.instr_acquisition(self.UHFQC.name)
        self.CCL_qubit.instr_CC(self.CCL.name)

    def test_instantiate_QuDevTransmon(self):
        QDT = QuDev_transmon('QuDev_transmon',
                             MC=None, heterodyne_instr=None, cw_source=None)
        QDT.close()

    def test_instantiate_TekTransmon(self):
        TT = Tektronix_driven_transmon('TT')
        TT.close()

    def test_instantiate_CBoxv3_transmon(self):
        CT = CBox_v3_driven_transmon('CT')
        CT.close()

    def test_instantiate_QWG_transmon(self):
        QT = QWG_driven_transmon('QT')
        QT.close()

    def test_prepare_for_timedomain(self):
        self.CCL_qubit.prepare_for_timedomain()

    def test_prepare_for_continuous_wave(self):
        self.CCL_qubit.spec_pow(-20)
        self.CCL_qubit.prepare_for_continuous_wave()

    def test_prepare_for_fluxing(self):
        self.CCL_qubit.prepare_for_fluxing()

    def test_prepare_readout(self):

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

    @classmethod
    def tearDownClass(self):
        for inststr in list(self.CCL_qubit._all_instruments):
            try:
                inst = self.CCL_qubit.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass
