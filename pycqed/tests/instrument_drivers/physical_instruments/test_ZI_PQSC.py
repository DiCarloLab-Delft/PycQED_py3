import unittest
import tempfile
import os
import numpy

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_PQSC as PQ


class Test_PQSC(unittest.TestCase):
    #FIXME: change device to correct device identifier
    @classmethod
    def setup_class(cls):
        print('Connecting...')
        cls.pqsc = PQ.ZI_PQSC(
            name='MOCK_PQSC',
            server='emulator',
            device='dev0000',
            interface='1GbE')

    @classmethod
    def teardown_class(cls):
        print('Disconnecting...')
        cls.pqsc.close()

    def test_instantiation(self):
        self.assertEqual(Test_PQSC.pqsc.devname, 'dev0000')