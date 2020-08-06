import unittest

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan


class Test_ro_lutman(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.uhf = UHF.UHFQC(name='MOCK_UHF', server='emulator',
                            device='dev2109', interface='1GbE')
        cls.ro_lutman = UHFQC_RO_LutMan('RO_lutman', num_res=5)
        cls.ro_lutman.AWG(cls.uhf.name)

    @classmethod
    def teardown_class(cls):
        cls.uhf.close()

    def test_ro_lutman(self):
        pass
