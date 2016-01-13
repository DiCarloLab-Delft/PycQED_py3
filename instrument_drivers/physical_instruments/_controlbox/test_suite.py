import unittest
import numpy as np
from . import defHeaders
CBox = None




class CBox_tests(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    @classmethod
    def setUpClass(self):
        print('CBox', CBox)
        self.CBox = CBox

    # def test_get_all(self):
    #     CBox.get_all()
    #     return True

    def test_firmware_version(self):
        v = CBox.get('firmware_version')
        self.assertTrue(int(v[1]) == 2)  # major version
        self.assertTrue(int(int(v[3:5])) > 13)  # minor version


    def test_setting_acquisition_mode(self):
        self.CBox.set('acquisition_mode', 0)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[0])

        self.CBox.set('acquisition_mode', 1)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[1])

        self.CBox.set('acquisition_mode', 2)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[2])

        self.CBox.set('acquisition_mode', 3)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[3])

        self.CBox.set('acquisition_mode', 4)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[4])

        self.CBox.set('acquisition_mode', 5)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[5])

        self.CBox.set('acquisition_mode', 0)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[0])

    def test_codec(self):
        # codec is CBox.c
        encoded_128 = self.CBox.c.encode_byte(128, 7)
        self.assertTrue(type(encoded_128) == bytes)
        self.assertTrue(len(encoded_128) == 2)
        self.assertTrue(bytes_to_binary(encoded_128)
                        == '1000000110000000')
        encoded_128 = self.CBox.c.encode_byte(128, 4)
        self.assertTrue(type(encoded_128) == bytes)
        self.assertTrue(len(encoded_128) == 2)
        self.assertTrue(bytes_to_binary(encoded_128) ==
                        '1000100010000000')

        # Encoding using 4 bits per byte
        encoded_546815 = self.CBox.c.encode_byte(546815, 4, 6)
        self.assertEqual(type(encoded_546815), bytes)
        self.assertEqual(len(encoded_546815), 6)
        sub_str = bytes([encoded_546815[0], encoded_546815[-1]])
        self.assertTrue(bin(sub_str == '1000100010001111'))
        self.assertEqual(CBox.c.decode_byte(encoded_546815, 4), 546815)

        # encoding using 7 bits per byte
        encoded_546815 = self.CBox.c.encode_byte(546815, 7, 4)
        self.assertEqual(CBox.c.decode_byte(encoded_546815, 7), 546815)

        # encoding using 7 bits per byte
        encoded_neg235 = self.CBox.c.encode_byte(-235, 7, 4)
        self.assertEqual(CBox.c.decode_byte(encoded_neg235, 7), -235)

        # Encoding and decoding array
        x = np.random.randint(0, 2565, 20)
        data_bytes = CBox.c.encode_array(x, 7, 2)
        message = CBox.c.create_message(data_bytes=data_bytes)
        x_dec = CBox.c.decode_message(message, 7, 2)
        self.assertEqual(x.all(), x_dec.all())

    def test_sig_del(self):
        s_del = self.CBox.get('signal_delay')

        self.CBox.set('signal_delay', 0)
        self.assertEqual(self.CBox.get('signal_delay'), 0)
        self.CBox.set('signal_delay', 124)
        self.assertEqual(self.CBox.get('signal_delay'), 124)

        self.CBox.set('signal_delay', s_del)

    def test_integration_length(self):
        s_del = self.CBox.get('integration_length')

        self.CBox.set('integration_length', 50)
        self.assertEqual(self.CBox.get('integration_length'), 50)
        self.CBox.set('integration_length', 124)
        self.assertEqual(self.CBox.get('integration_length'), 124)

        self.CBox.set('integration_length', s_del)

    def test_set_signal_threshold(self):
        for i in range(2):
            t = self.CBox.get('signal_threshold_line_{}'.format(i))
            self.CBox.set('signal_threshold_line_{}'.format(i), 124)
            self.assertEqual(
                self.CBox.get('signal_threshold_line_{}'.format(i)), 124)

            self.CBox.set('signal_threshold_line_{}'.format(i), t)

    def test_adc_offset(self):
        offs = self.CBox.get('adc_offset')

        self.CBox.set('adc_offset', 123)
        self.assertEqual(self.CBox.get('adc_offset'), 123)
        self.CBox.set('adc_offset', -123)
        self.assertEqual(self.CBox.get('adc_offset'), -123)

        self.CBox.set('adc_offset', offs)
    def test_log_length(self):
        initial_val = self.CBox.get('log_length')

        self.CBox.set('log_length', 2)
        self.assertEqual(self.CBox.get('log_length'), 2)
        self.CBox.set('log_length', 7500)
        self.assertEqual(self.CBox.get('log_length'), 7500)

        self.CBox.set('log_length', initial_val)

    # def test_readLog(self):
    #     '''
    #     Test for mode 2 integration logs. Only tests on length of data
    #     '''
    #     log_length = 50
    #     self.CBox.set_acquisition_mode(0)
    #     self.CBox.set_log_length(log_length)
    #     self.CBox.set_signal_delay(20)
    #     self.CBox.set_integration_length(255)
    #     self.CBox.set_averaging_parameters(60, 1)
    #     self.CBox.set_adc_offset(1)
    #     self.CBox.set_lin_trans_coeffs(1, 0, 0, 1)
    #     weights0 = np.ones(512)
    #     weights1 = np.ones(512)
    #     self.CBox.set_integration_weights(line=0, weights=weights0)
    #     self.CBox.set_integration_weights(line=1, weights=weights1)
    #     qt.msleep(.05)
    #     self.CBox.set_acquisition_mode(1)
    #     [InputAvgRes0, InputAvgRes1] = self.CBox.get_integration_log_results()
    #     self.CBox.set_acquisition_mode(0)

    #     self.assertEqual(len(InputAvgRes0), log_length)
    #     self.assertEqual(len(InputAvgRes1), log_length)

    # def test_integration_average_mode(self):
    #     self.CBox.set_acquisition_mode(0)
    #     NoSamples = 60
    #     AvgSize = 2

    #     self.CBox.set_signal_delay(20)
    #     self.CBox.set_integration_length(255)
    #     self.CBox.set_adc_offset(1)

    #     weights0 = np.ones(512) * 1
    #     weights1 = np.ones(512) * 0

    #     self.CBox.set_integration_weights(line=0, weights=weights0)
    #     self.CBox.set_integration_weights(line=1, weights=weights1)
    #     self.CBox.set_lin_trans_coeffs(1, 0, 0, 1)

    #     self.assertTrue(self.CBox.set_averaging_parameters(NoSamples, AvgSize))
    #     self.assertEqual(self.CBox.get_nr_samples(), NoSamples)
    #     self.assertEqual(self.CBox.get_avg_size(), AvgSize)

    #     self.CBox.set_acquisition_mode(4)
    #     [InputAvgRes0, InputAvgRes1] = self.CBox.get_integrated_avg_results()
    #     self.CBox.set_acquisition_mode(0)
    #     # Test signal lengths set correctly
    #     self.assertEqual(len(InputAvgRes0), NoSamples)
    #     # Test if setting weights to zero functions correctly
    #     self.assertTrue((InputAvgRes1 == np.zeros(NoSamples)).all())

    #     weights1 = np.ones(512) * 1
    #     self.CBox.set_integration_weights(line=1, weights=weights1)
    #     self.CBox.set_lin_trans_coeffs(0, 0, 0, 1)
    #     self.CBox.set_acquisition_mode(4)
    #     [InputAvgRes0, InputAvgRes1] = self.CBox.get_integrated_avg_results()
    #     self.CBox.set_acquisition_mode(0)

    #     # Test if setting lin trans coeff to zero functions correctly
    #     self.assertTrue((InputAvgRes0 == np.zeros(NoSamples)).all())
    #     self.assertFalse((InputAvgRes1 == np.zeros(NoSamples)).all())

    # def test_streaming_mode(self):

    #     self.CBox.set_acquisition_mode(0)

    #     NoSamples = 1e3
    #     self.CBox.set_acquisition_mode(5)
    #     data = self.CBox.get_streaming_results(NoSamples)
    #     self.CBox.set_acquisition_mode(0)
    #     self.assertTrue(len(data[0]) > NoSamples)
    #     self.assertTrue(len(data[0]) == len(data[1]))


    # def test_set_awg_lookuptable(self):
    #     length = np.random.randint(1, 128)
    #     random_lut = np.random.randint(-1000, 1000, length)
    #     self.assertTrue(self.CBox.set_awg_lookuptable(0, 4, 0, random_lut))

    # def test_DacEnable(self):
    #     for awg in range(3):
    #         self.assertTrue(self.CBox.enable_dac(awg, 0, True))
    #         self.assertTrue(self.CBox.enable_dac(awg, 0, False))

    #         self.assertTrue(self.CBox.enable_dac(awg, 1, True))
    #         self.assertTrue(self.CBox.enable_dac(awg, 1, False))

    # def test_DacOffset(self):
    #     for awg in range(3):
    #         self.assertTrue(self.CBox.set_dac_offset(awg, 0, True))
    #         self.assertTrue(self.CBox.set_dac_offset(awg, 0, False))

    #         self.assertTrue(self.CBox.set_dac_offset(awg, 1, True))
    #         self.assertTrue(self.CBox.set_dac_offset(awg, 1, False))

    # def test_input_avg_mode(self):
    #     self.CBox.set_acquisition_mode(0)
    #     NoSamples = 250
    #     AvgSize = 2

    #     self.assertTrue(self.CBox.set_averaging_parameters(NoSamples, AvgSize))
    #     self.assertEqual(self.CBox.get_nr_samples(), NoSamples)
    #     self.assertEqual(self.CBox.get_avg_size(), AvgSize)

    #     self.CBox.set_acquisition_mode(3)
    #     [InputAvgRes0, InputAvgRes1] = self.CBox.get_input_avg_results()
    #     self.CBox.set_acquisition_mode(0)

    #     # Only checks on lenght of signal as test
    #     # No check if averaging or signal delay works
    #     self.assertEqual(len(InputAvgRes0), NoSamples)
    #     self.assertEqual(len(InputAvgRes1), NoSamples)

    # def test_LinTransCoeff(self):
    #     with self.assertRaises(ValueError):
    #         self.CBox.set_lin_trans_coeffs(2, 1, 1, 1)
    #     with self.assertRaises(ValueError):
    #         self.CBox.set_lin_trans_coeffs(1.2, 2.5, 1, 1)
    #     coeffs = np.random.random(4)*4-2
    #     self.assertTrue(self.CBox.set_lin_trans_coeffs(1, .4, -.1, -1))

    #     self.assertTrue(self.CBox.set_lin_trans_coeffs(coeffs[0], coeffs[1],
    #                                                    coeffs[2], coeffs[3]))

    # def test_LogLength(self):
    #     val = np.random.randint(0, 8000)
    #     stat = self.CBox.set_log_length(val)
    #     self.assertTrue(stat)
    #     self.assertEqual(val, self.CBox.get_log_length())

    # def test_awg_mode(self):
    #     awg_nr = np.random.randint(0, 3)
    #     stat = self.CBox.set_awg_mode(awg_nr, True)
    #     self.assertTrue(stat)

    # def test_unrecognized_command_error(self):
    #     with self.assertRaises(ValueError):
    #         self.CBox.serial_write('\x02\x00\x7F')

    # def test_SigDelay(self):
    #     val = np.random.randint(0, 256)
    #     stat = self.CBox.set_signal_delay(val)
    #     self.assertTrue(stat)
    #     self.assertEqual(val, self.CBox.get_signal_delay(val))

    # def test_IntegrationLength(self):
    #     val = np.random.randint(0, 255)
    #     # Real bound is 512 but currently exception in protocol
    #     stat = self.CBox.set_integration_length(val)
    #     self.assertTrue(stat)
    #     self.assertEqual(val, self.CBox.get_integration_length(val))

    # def test_SigThreshold(self):
    #     val = np.random.randint(10000)  # not the extreme value of the range
    #     stat = self.CBox.set_signal_threshold_line0(val)
    #     self.assertTrue(stat)
    #     self.assertEqual(val, self.CBox.get_signal_threshold_line0(val))

    #     val = np.random.randint(10000)
    #     stat = self.CBox.set_signal_threshold_line1(val)
    #     self.assertTrue(stat)
    #     self.assertEqual(val, self.CBox.get_signal_threshold_line1(val))

    # def test_SigWeights(self):
    #     array_good = np.random.randint(-128, 127, size=512)
    #     array_bad = array_good + 200
    #     with self.assertRaises(ValueError):
    #         self.CBox.set_integration_weights(line=0, weights=array_bad)
    #     with self.assertRaises(ValueError):
    #             self.CBox.set_integration_weights(line=0,
    #                                               weights=array_good[0:-5])
    #     succes = self.CBox.set_integration_weights(line=0, weights=array_good)
    #     self.assertTrue(succes)

def bytes_to_binary(bytestring):
    '''
    used as a convenience function in codec testing
    '''
    s = ''
    for n in bytestring:
        s+= ''.join(str((n & (1 << i)) and 1) for i in reversed(range(8)))
    return s

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(CBox_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
