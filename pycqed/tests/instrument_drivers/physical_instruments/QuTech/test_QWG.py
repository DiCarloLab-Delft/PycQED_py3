import pycqed as pq
import unittest
import os
from pathlib import Path

from pycqed.instrument_drivers.library.Transport import FileTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG


class Test_QWG(unittest.TestCase):
    def test_all(self):
        file_name = 'Test_QWG_test_all.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg = QWG('qwg', transport)

        qwg.init()

        qwg.delete_waveform_all()
        qwg.new_waveform_real('test', 3)
        qwg.send_waveform_data_real('test', [-0.1, 0, 0.1])
        qwg.delete_waveform('test')
        qwg.create_waveform_real('test', [-0.1, 0, 0.1])
        qwg.sync_sideband_generators()

        qwg.start()
        qwg.stop()


        transport.close()  # to allow access to file

        # check results
        test_output = test_path.read_bytes()
        golden_path = Path(pq.__path__[0]) / 'tests/instrument_drivers/physical_instruments/QuTech/golden' / file_name
        golden = golden_path.read_bytes()
        self.assertEqual(test_output, golden)
