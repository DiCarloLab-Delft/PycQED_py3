import unittest
import qt
import defHeaders  # dictionar of bytestring commands
import numpy as np


class QuTech_ControlBox_tests(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the coms are working.
    '''
    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        self.CBox = qt.instruments['CBox']
        if self.CBox is None:
            self.CBox = qt.instruments.create(
                'CBox', 'QuTech_ControlBox', address='Com4', dummy_instrument=True)
        ser = self.CBox.get_serial()
        if ser != None:
            ser.flushInput()
            ser.flushOutput()

    def test_instrument_is_proxy(self):
        self.assertEqual(self.CBox.__class__.__name__, 'Proxy',
                         'Instrument is not a Proxy, failure in ins creation')

    def test_serial_port_opening(self):
        ser = self.CBox.get_serial()
        if not ser.isOpen():
            ser.open()
        self.assertTrue(ser.isOpen(), 'Connection not open')
        ser.close()
        self.assertFalse(ser.isOpen(), 'Connection not closed')
        ser.open()
        self.assertTrue(ser.isOpen(), 'Connection not reopened')

    def test_setting_acquisition_mode(self):
        stat = self.CBox.set_acquisition_mode(0)
        self.assertTrue(stat)
        self.assertEqual(0, self.CBox.get_acquisition_mode())

        stat = self.CBox.set_acquisition_mode(1)
        self.assertTrue(stat)
        self.assertEqual(1, self.CBox.get_acquisition_mode(1))

        stat = self.CBox.set_acquisition_mode(2)
        self.assertTrue(stat)
        self.assertEqual(2, self.CBox.get_acquisition_mode(2))

        stat = self.CBox.set_acquisition_mode(3)
        self.assertTrue(stat)
        self.assertEqual(3, self.CBox.get_acquisition_mode(3))

        stat = self.CBox.set_acquisition_mode(4)
        self.assertTrue(stat)
        self.assertEqual(4, self.CBox.get_acquisition_mode(4))

        stat = self.CBox.set_acquisition_mode(5)
        self.assertTrue(stat)
        self.assertEqual(5, self.CBox.get_acquisition_mode(5))

        stat = self.CBox.set_acquisition_mode(0)
        self.assertTrue(stat)
        self.assertEqual(0, self.CBox.get_acquisition_mode(0))

    def test_encoding(self):
        encoded_128 = self.CBox.encode_byte(128, 7)
        self.assertTrue(len(encoded_128) == 2)
        self.assertTrue(bin(encoded_128[0]) == '0b10000001')
        self.assertTrue(bin(encoded_128[1]) == '0b10000000')

        encoded_128 = self.CBox.encode_byte(128, 4)
        self.assertTrue(len(encoded_128) == 2)
        self.assertTrue(bin(encoded_128[0]) == '0b10001000')
        self.assertTrue(bin(encoded_128[1]) == '0b10000000')

        encoded_128 = self.CBox.encode_byte(546815, 4)
        self.assertTrue(len(encoded_128) == 5)
        self.assertTrue(bin(encoded_128[0]) == '0b10001000')
        self.assertTrue(bin(encoded_128[-1]) == '0b10001111')

        encodedTrue = self.CBox.encode_byte(True, 7)
        self.assertEqual(bin(encodedTrue[0]), '0b10000001')
        encodedFalse = self.CBox.encode_byte(False, 4)
        self.assertEqual(bin(encodedFalse[0]), '0b10000000')

        encoded_negative_4023 = self.CBox.encode_byte(4023,
                                                      data_bits_per_byte=7,
                                                      signed_integer_length=14)
        self.assertTrue(len(encoded_negative_4023) == 2)
        self.assertEqual(bin(encoded_negative_4023[0]), '0b10011111')

    def test_decoding(self):
        val = 128
        encoded_bytes = self.CBox.encode_byte(val, 7)
        self.assertEqual(self.CBox.decode_byte(encoded_bytes, 7), val)

        val = 128
        encoded_bytes = self.CBox.encode_byte(val, 4)
        self.assertEqual(self.CBox.decode_byte(encoded_bytes, 4), val)

    def test_readLog(self):
        '''
        Test for mode 2 integration logs. Only tests on length of data
        '''
        log_length = 50
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_log_length(log_length)
        self.CBox.set_signal_delay(20)
        self.CBox.set_integration_length(255)
        self.CBox.set_averaging_parameters(60, 1)
        self.CBox.set_adc_offset(1)
        self.CBox.set_lin_trans_coeffs(1, 0, 0, 1)
        weights0 = np.ones(512)
        weights1 = np.ones(512)
        self.CBox.set_integration_weights(line=0, weights=weights0)
        self.CBox.set_integration_weights(line=1, weights=weights1)
        qt.msleep(.05)
        self.CBox.set_acquisition_mode(1)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_integration_log_results()
        self.CBox.set_acquisition_mode(0)

        self.assertEqual(len(InputAvgRes0), log_length)
        self.assertEqual(len(InputAvgRes1), log_length)

    def test_integration_average_mode(self):
        self.CBox.set_acquisition_mode(0)
        NoSamples = 60
        AvgSize = 2

        self.CBox.set_signal_delay(20)
        self.CBox.set_integration_length(255)
        self.CBox.set_adc_offset(1)

        weights0 = np.ones(512) * 1
        weights1 = np.ones(512) * 0

        self.CBox.set_integration_weights(line=0, weights=weights0)
        self.CBox.set_integration_weights(line=1, weights=weights1)
        self.CBox.set_lin_trans_coeffs(1, 0, 0, 1)

        self.assertTrue(self.CBox.set_averaging_parameters(NoSamples, AvgSize))
        self.assertEqual(self.CBox.get_nr_samples(), NoSamples)
        self.assertEqual(self.CBox.get_avg_size(), AvgSize)

        self.CBox.set_acquisition_mode(4)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set_acquisition_mode(0)
        # Test signal lengths set correctly
        self.assertEqual(len(InputAvgRes0), NoSamples)
        # Test if setting weights to zero functions correctly
        self.assertTrue((InputAvgRes1 == np.zeros(NoSamples)).all())

        weights1 = np.ones(512) * 1
        self.CBox.set_integration_weights(line=1, weights=weights1)
        self.CBox.set_lin_trans_coeffs(0, 0, 0, 1)
        self.CBox.set_acquisition_mode(4)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set_acquisition_mode(0)

        # Test if setting lin trans coeff to zero functions correctly
        self.assertTrue((InputAvgRes0 == np.zeros(NoSamples)).all())
        self.assertFalse((InputAvgRes1 == np.zeros(NoSamples)).all())

    def test_streaming_mode(self):

        self.CBox.set_acquisition_mode(0)

        NoSamples = 1e3
        self.CBox.set_acquisition_mode(5)
        data = self.CBox.get_streaming_results(NoSamples)
        self.CBox.set_acquisition_mode(0)
        self.assertTrue(len(data[0]) > NoSamples)
        self.assertTrue(len(data[0]) == len(data[1]))


    def test_set_awg_lookuptable(self):
        length = np.random.randint(1, 128)
        random_lut = np.random.randint(-1000, 1000, length)
        self.assertTrue(self.CBox.set_awg_lookuptable(0, 4, 0, random_lut))

    def test_DacEnable(self):
        for awg in range(3):
            self.assertTrue(self.CBox.enable_dac(awg, 0, True))
            self.assertTrue(self.CBox.enable_dac(awg, 0, False))

            self.assertTrue(self.CBox.enable_dac(awg, 1, True))
            self.assertTrue(self.CBox.enable_dac(awg, 1, False))

    def test_DacOffset(self):
        for awg in range(3):
            self.assertTrue(self.CBox.set_dac_offset(awg, 0, True))
            self.assertTrue(self.CBox.set_dac_offset(awg, 0, False))

            self.assertTrue(self.CBox.set_dac_offset(awg, 1, True))
            self.assertTrue(self.CBox.set_dac_offset(awg, 1, False))

    def test_input_avg_mode(self):
        self.CBox.set_acquisition_mode(0)
        NoSamples = 250
        AvgSize = 2

        self.assertTrue(self.CBox.set_averaging_parameters(NoSamples, AvgSize))
        self.assertEqual(self.CBox.get_nr_samples(), NoSamples)
        self.assertEqual(self.CBox.get_avg_size(), AvgSize)

        self.CBox.set_acquisition_mode(3)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_input_avg_results()
        self.CBox.set_acquisition_mode(0)

        # Only checks on lenght of signal as test
        # No check if averaging or signal delay works
        self.assertEqual(len(InputAvgRes0), NoSamples)
        self.assertEqual(len(InputAvgRes1), NoSamples)

    def test_LinTransCoeff(self):
        with self.assertRaises(ValueError):
            self.CBox.set_lin_trans_coeffs(2, 1, 1, 1)
        with self.assertRaises(ValueError):
            self.CBox.set_lin_trans_coeffs(1.2, 2.5, 1, 1)
        coeffs = np.random.random(4)*4-2
        self.assertTrue(self.CBox.set_lin_trans_coeffs(1, .4, -.1, -1))

        self.assertTrue(self.CBox.set_lin_trans_coeffs(coeffs[0], coeffs[1],
                                                       coeffs[2], coeffs[3]))

    def test_LogLength(self):
        val = np.random.randint(0, 8000)
        stat = self.CBox.set_log_length(val)
        self.assertTrue(stat)
        self.assertEqual(val, self.CBox.get_log_length())

    def test_awg_mode(self):
        awg_nr = np.random.randint(0, 3)
        stat = self.CBox.set_awg_mode(awg_nr, True)
        self.assertTrue(stat)

    def test_firmware_version(self):
        v = self.CBox.get_firmware_version()
        self.assertEqual(v, '2.10.1')
        # Warning, only tests string length not if actual version is newer.

    def test_unrecognized_command_error(self):
        with self.assertRaises(ValueError):
            self.CBox.serial_write('\x02\x00\x7F')

    def test_SigDelay(self):
        val = np.random.randint(0, 256)
        stat = self.CBox.set_signal_delay(val)
        self.assertTrue(stat)
        self.assertEqual(val, self.CBox.get_signal_delay(val))

    def test_IntegrationLength(self):
        val = np.random.randint(0, 255)
        # Real bound is 512 but currently exception in protocol
        stat = self.CBox.set_integration_length(val)
        self.assertTrue(stat)
        self.assertEqual(val, self.CBox.get_integration_length(val))

    def test_SigThreshold(self):
        val = np.random.randint(10000)  # not the extreme value of the range
        stat = self.CBox.set_signal_threshold_line0(val)
        self.assertTrue(stat)
        self.assertEqual(val, self.CBox.get_signal_threshold_line0(val))

        val = np.random.randint(10000)
        stat = self.CBox.set_signal_threshold_line1(val)
        self.assertTrue(stat)
        self.assertEqual(val, self.CBox.get_signal_threshold_line1(val))

    def test_SigWeights(self):
        array_good = np.random.randint(-128, 127, size=512)
        array_bad = array_good + 200
        with self.assertRaises(ValueError):
            self.CBox.set_integration_weights(line=0, weights=array_bad)
        with self.assertRaises(ValueError):
                self.CBox.set_integration_weights(line=0,
                                                  weights=array_good[0:-5])
        succes = self.CBox.set_integration_weights(line=0, weights=array_good)
        self.assertTrue(succes)



if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        QuTech_ControlBox_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
