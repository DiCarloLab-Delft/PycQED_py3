import pycqed as pq
import unittest
import tempfile
import os
from pathlib import Path

from pycqed.instrument_drivers.physical_instruments.Transport import FileTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG


class Test_Qutech_QWG(unittest.TestCase):
    def test_all(self):
        file_name = 'Test_Qutech_QWG_test_all.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg = QWG('qwg', transport)

        qwg.init()



        qwg.start()
        qwg.stop()

        transport.close()  # to allow access to file

        # check results
        test_output = test_path.read_text()
        golden_path = Path(pq.__path__[0]) / 'tests/instrument_drivers/physical_instruments/golden' / file_name
        golden = golden_path.read_text()
        self.assertEqual(test_output, golden)
