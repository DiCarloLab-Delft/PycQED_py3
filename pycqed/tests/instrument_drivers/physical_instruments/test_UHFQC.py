import unittest
import tempfile
import os

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF

class Test_UHFQC(unittest.TestCase):
  def test_all(self):
    mock_uhf = UHF.UHFQC(name='MOCK_UHF', server='emulator', num_codewords=32, device='dev2109', interface='1GbE')
    self.assertEqual(mock_uhf.devname, 'dev2109')