import unittest
import tempfile
import os

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 as HDAWG

class Test_ZI_HDAWG8(unittest.TestCase):
  def test_all(self):
    mock_hd = HDAWG.ZI_HDAWG8(name='MOCK_HD', server='emulator', num_codewords=32, device='dev8026', interface='1GbE')
    self.assertEqual(mock_hd.devname, 'dev8026')