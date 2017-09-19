import unittest
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8
from pycqed.instrument_drivers.meta_instrument.LutMans import mw_lutman as mwl


class Test_MW_LutMan(unittest.TestCase):
    def setUp(self):
        self.AWG = v8.VirtualAWG8('DummyAWG8')

        self.LutMan = mwl.Base_MW_LutMan('MW_LutMan')
        self.LutMan.AWG(self.AWG.name)

    def test_uploading_standard_pulses(self):
        self.LutMan.load_waveforms_onto_AWG_lookuptable()
