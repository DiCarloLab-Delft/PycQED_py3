import unittest
import os
import numpy as np
from pathlib import Path

import pycqed as pq
from pycqed.instrument_drivers.library.Transport import FileTransport
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.physical_instruments.QuTech.QWGCore import QWGCore
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG,QWGMultiDevices


class Test_QWG(unittest.TestCase):
    @unittest.skip(reason="Deprecated hardware")
    def test_qwg_core(self):
        file_name = 'Test_QWG_test_qwg_core.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwgcore = QWGCore('qwg', transport)

        qwgcore.init()

        qwgcore.delete_waveform_all()
        qwgcore.new_waveform_real('test', 3)
        if 0:  # FIXME: disabled because it produces binary data that breaks reading golden file
            qwgcore.send_waveform_data_real('test', [-0.1, 0, 0.1])
        qwgcore.delete_waveform('test')
        if 0:   # FIXME, see above
            qwgcore.create_waveform_real('test', [-0.1, 0, 0.1])
        qwgcore.sync_sideband_generators()

        qwgcore.start()
        qwgcore.stop()

        transport.close()  # to allow access to file

        # check results
        test_output = test_path.read_bytes()
        golden_path = Path(__file__).parent / 'golden' / file_name
        golden = golden_path.read_bytes()
        self.assertEqual(test_output, golden)

    def test_awg_parameters(self):
        file_name = 'Test_QWG_test_awg_parameters.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg = QWG('qwg_awg_parameters', transport)  # FIXME: names must be unique unless we properly tell QCoDes to remove
        qwg.init()

        for i in range(qwg._dev_desc.numChannels//2):
            ch_pair = i*2+1
            qwg.set(f'ch_pair{ch_pair}_sideband_frequency', 0)
            qwg.set(f'ch_pair{ch_pair}_sideband_phase', 0)
            qwg.set(f'ch_pair{ch_pair}_transform_matrix', np.array([[0,1], [1,0]]))

        for ch in range(1, qwg._dev_desc.numChannels+1):
            qwg.set(f'ch{ch}_state', 0)
            qwg.set(f'ch{ch}_amp', 0)
            qwg.set(f'ch{ch}_offset', 0)
            qwg.set(f'ch{ch}_default_waveform', '')

        qwg.run_mode('CODeword')

        # for cw in range(qwg._dev_desc.numCodewords):  # FIXME: this may give 1024 parameters per channel
        #     for j in range(qwg._dev_desc.numChannels):
        #         ch = j+1
        #         # Codeword 0 corresponds to bitcode 0
        #         qwg.set('sequence:element{:d}:waveform{:d}'.format(cw, ch), "")


        transport.close()  # to allow access to file
        qwg.close()  # release QCoDeS instrument

    def test_codeword_parameters(self):
        file_name = 'Test_QWG_test_codeword_parameters.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg = QWG('qwg_codeword_parameters', transport)  # FIXME: names must be unique unless we properly tell QCoDes to remove
        qwg.init()

        qwg.cfg_codeword_protocol('awg8-mw-direct-iq')
        for j in range(qwg._dev_desc.numChannels):
            for cw in range(qwg._dev_desc.numCodewords):
                ch = j + 1
                qwg.set('wave_ch{}_cw{:03}'.format(ch, cw), np.array([0,0.1,0.2]))

        transport.close()  # to allow access to file
        qwg.close()  # release QCoDeS instrument

    def test_dio_parameters(self):
        file_name = 'Test_QWG_test_dio_parameters.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg = QWG('qwg_dio_parameters', transport)  # FIXME: names must be unique unless we properly tell QCoDes to remove
        qwg.init()

        qwg.dio_mode('MASTER')
        # dio_is_calibrated
        qwg.dio_active_index(0)

        transport.close()  # to allow access to file
        qwg.close()  # release QCoDeS instrument

    def test_multi(self):
        file_name = 'Test_QWG_test_multi.scpi.txt'
        test_path = Path('test_output') / file_name
        os.makedirs('test_output', exist_ok=True)

        transport = FileTransport(str(test_path))
        qwg1 = QWG('qwg1', transport)
        qwg2 = QWG('qwg2', transport)

        for qwg in [qwg1, qwg2]:
            qwg.init()
            qwg.run_mode('CODeword')
            qwg.cfg_codeword_protocol('awg8-mw-direct-iq')
        qwg1.dio_mode('MASTER')
        qwg2.dio_mode('SLAVE')

        qwgs = QWGMultiDevices([qwg1, qwg2])
        if 0:  # FIXME: requires reads from instruments
            DIO.calibrate(receiver=qwgs)

        transport.close()  # to allow access to file
        qwg.close()  # release QCoDeS instrument


    # FIXME: add tests for data received from QWG