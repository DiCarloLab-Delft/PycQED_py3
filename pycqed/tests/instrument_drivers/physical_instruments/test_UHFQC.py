import unittest
import tempfile
import os
import numpy

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as UHF

class Test_UHFQC(unittest.TestCase):
  @classmethod
  def setup_class(cls):
    print('Connecting...')
    cls.uhf = UHF.UHFQC(name='MOCK_UHF', server='emulator', num_codewords=32, device='dev2109', interface='1GbE')

  @classmethod
  def teardown_class(cls):
    print('Disconnecting...')
    cls.uhf.close()

  def test_instantiation(self):
    self.assertEqual(Test_UHFQC.uhf.devname, 'dev2109')

  def test_dynamic_waveform_upload(self):
    Test_UHFQC.uhf.awg_sequence_acquisition_and_pulse()
    Test_UHFQC.uhf.start()
    Test_UHFQC.uhf.stop()
    
    # The program must be compiled exactly once at this point
    self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 1)

    # Modify a waveform
    Test_UHFQC.uhf.wave_ch1_cw000(0.5*Test_UHFQC.uhf.wave_ch1_cw000())

    # Start again
    Test_UHFQC.uhf.start()
    Test_UHFQC.uhf.stop()

    # No further compilation allowed
    self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 1)

    # Change the length of a waveform
    w0 = numpy.concatenate((Test_UHFQC.uhf.wave_ch1_cw000(), Test_UHFQC.uhf.wave_ch1_cw000()))
    Test_UHFQC.uhf.wave_ch1_cw000(w0)

    # Start again
    Test_UHFQC.uhf.start()
    Test_UHFQC.uhf.stop()

    # Now the compilation must have been executed again
    self.assertEqual(Test_UHFQC.uhf._awgModule.get_compilation_count(0), 2)