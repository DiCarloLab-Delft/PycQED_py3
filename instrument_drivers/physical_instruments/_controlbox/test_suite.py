import unittest
import numpy as np
from . import defHeaders
CBox = None
from modules.analysis.tools import data_manipulation as dm_tools


class CBox_tests(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    @classmethod
    def setUpClass(self):
        self.CBox = CBox

    def test_firmware_version(self):
        v = CBox.get('firmware_version')
        self.assertTrue(int(v[1]) == 2)  # major version
        self.assertTrue(int(int(v[3:5])) > 13)  # minor version

    def test_setting_mode(self):
        for i in range(6):
            self.CBox.set('acquisition_mode', i)
            self.assertEqual(self.CBox.get('acquisition_mode'),
                             defHeaders.acquisition_modes[i])
        self.CBox.set('acquisition_mode', 0)
        self.assertEqual(self.CBox.get('acquisition_mode'),
                         defHeaders.acquisition_modes[0])

        for i in range(2):
            self.CBox.set('run_mode', i)
            self.assertEqual(self.CBox.get('run_mode'),
                             defHeaders.run_modes[i])
        self.CBox.set('run_mode', 0)

        for j in range(3):
            for i in range(3):
                self.CBox.set('AWG{}_mode'.format(j), i)
                self.assertEqual(self.CBox.get('AWG{}_mode'.format(j)),
                                 defHeaders.awg_modes[i])
                self.CBox.set('AWG{}_mode'.format(j), 0)

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
            t = self.CBox.get('sig{}_threshold_line'.format(i))
            self.CBox.set('sig{}_threshold_line'.format(i), 124)
            self.assertEqual(
                self.CBox.get('sig{}_threshold_line'.format(i)), 124)

            self.CBox.set('sig{}_threshold_line'.format(i), t)

    def test_adc_offset(self):
        offs = self.CBox.get('adc_offset')
        self.CBox.set('adc_offset', 123)
        self.assertEqual(self.CBox.get('adc_offset'), 123)
        self.CBox.set('adc_offset', -123)
        self.assertEqual(self.CBox.get('adc_offset'), -123)
        self.CBox.set('adc_offset', offs)

    def test_dac_offset(self):
        for i in range(3):
            for j in range(2):
                initial_val = self.CBox.get('AWG{}_dac{}_offset'.format(i, j))
                self.CBox.set('AWG{}_dac{}_offset'.format(i, j), 200)
                self.assertEqual(
                    self.CBox.get('AWG{}_dac{}_offset'.format(i, j)), 200)
                self.CBox.set('AWG{}_dac{}_offset'.format(i, j), initial_val)

    def test_tape(self):
        for i in range(3):
            tape = [2, 4, 5, 1, 0, 3]
            initial_val = self.CBox.get('AWG{}_tape'.format(i))
            self.CBox.set('AWG{}_tape'.format(i), tape)
            self.assertEqual(self.CBox.get('AWG{}_tape'.format(i)), tape)
            self.CBox.set('AWG{}_tape'.format(i), initial_val)

    def test_log_length(self):
        initial_val = self.CBox.get('log_length')

        self.CBox.set('log_length', 2)
        self.assertEqual(self.CBox.get('log_length'), 2)
        self.CBox.set('log_length', 7500)
        self.assertEqual(self.CBox.get('log_length'), 7500)

        self.CBox.set('log_length', initial_val)

    def test_lin_trans_coeffs(self):
        initial_val = self.CBox.get('lin_trans_coeffs')

        self.CBox.set('lin_trans_coeffs', [1,.4, 0, 1.33])
        self.assertEqual(self.CBox.get('lin_trans_coeffs'), [1,.4, 0, 1.33])
        self.CBox.set('lin_trans_coeffs', [1, .4, .2, 1])
        self.assertEqual(self.CBox.get('lin_trans_coeffs'), [1, .4, .2, 1])

        self.CBox.set('lin_trans_coeffs', initial_val)

    def test_averaging_parameters(self):
        initial_val = self.CBox.get('nr_samples')

        self.CBox.set('nr_samples', 2)
        self.assertEqual(self.CBox.get('nr_samples'), 2)
        self.CBox.set('nr_samples', 1564)
        self.assertEqual(self.CBox.get('nr_samples'), 1564)
        self.CBox.set('nr_samples', initial_val)

        initial_val = self.CBox.get('nr_averages')
        self.CBox.set('nr_averages', 2)
        self.assertEqual(self.CBox.get('nr_averages'), 2)
        self.CBox.set('nr_averages', 2**15)
        self.assertEqual(self.CBox.get('nr_averages'), 2**15)
        self.CBox.set('nr_averages', initial_val)

    def test_measurement_timeout(self):
        initial_val = self.CBox.get('measurement_timeout')

        self.CBox.set('measurement_timeout', 123)
        self.assertEqual(self.CBox.get('measurement_timeout'), 123)
        self.CBox.set('measurement_timeout', -123)
        self.assertEqual(self.CBox.get('measurement_timeout'), -123)
        self.CBox.set('measurement_timeout', initial_val)

    def test_Integration_logging(self):
        '''
        Test for mode 1 integration logs. Only tests on length of data
        '''
        log_length = 8000
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('log_length', log_length)
        self.CBox.set('signal_delay', 20)
        self.CBox.set('integration_length', 255)
        self.CBox.set('nr_averages', 4)
        self.CBox.set('nr_samples', 10)
        self.CBox.set('adc_offset', 1)

        weights0 = np.ones(512)
        weights1 = np.ones(512)
        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        self.CBox.set('acquisition_mode', 1)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_integration_log_results()
        self.CBox.set('acquisition_mode', 0)

        self.assertEqual(len(InputAvgRes0), log_length)
        self.assertEqual(len(InputAvgRes1), log_length)

    def test_state_logging_and_counters(self):
        '''
        Test uses mode 1 integration logging. Checks if the results
        for the integration shots, states and state counters produce
        the same results for a given threshold.
        Does not do this using a dedicated sequence.
        '''
        log_length = 50
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('log_length', log_length)

        weights0 = np.ones(512)
        weights1 = np.ones(512)
        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        self.CBox.set('acquisition_mode', 1)
        [IntLog0, IntLog1] = self.CBox.get_integration_log_results()
        self.CBox.set('acquisition_mode', 0)

        threshold = int(np.mean(IntLog0))
        self.CBox.sig0_threshold_line.set(threshold)
        self.CBox.sig1_threshold_line.set(threshold)

        self.CBox.set('acquisition_mode', 1)
        log = self.CBox.get_integration_log_results()
        counters = self.CBox.get_qubit_state_log_counters()
        self.CBox.set('acquisition_mode', 0)

        digi_shots = dm_tools.digitize(
            log, threshold=CBox.sig0_threshold_line.get())
        software_err_fracs_0 = dm_tools.count_error_fractions(digi_shots[0])
        software_err_fracs_1 = dm_tools.count_error_fractions(digi_shots[1])

        # Test if software analysis of the counters and CBox counters are the
        # same
        self.assertTrue((software_err_fracs_0 == counters[0]).all())
        self.assertTrue((software_err_fracs_1 == counters[1]).all())




    def test_integration_average_mode(self):
        self.CBox.set('acquisition_mode', 0)
        NoSamples = 60


        weights0 = np.ones(512) * 1
        weights1 = np.ones(512) * 0

        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        self.CBox.set('nr_averages', 4)
        self.CBox.set('nr_samples', NoSamples)

        self.CBox.set('acquisition_mode', 4)
        [InputAvgRes0, IntAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 0)
        # Test signal lengths set correctly
        self.assertEqual(len(InputAvgRes0), NoSamples)
        # Test if setting weights to zero functions correctly
        self.assertTrue((IntAvgRes1 == np.zeros(NoSamples)).all())

        weights1 = np.ones(512) * 1
        self.CBox.set('sig1_integration_weights', weights1)
        self.CBox.set('lin_trans_coeffs', [0, 0, 0, 1])
        self.CBox.set('acquisition_mode', 4)
        [InputAvgRes0, IntAvgRes1] = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 0)

        # Test if setting lin trans coeff to zero functions correctly
        self.assertTrue((InputAvgRes0 == np.zeros(NoSamples)).all())
        self.assertFalse((IntAvgRes1 == np.zeros(NoSamples)).all())

    # def test_streaming_mode(self):

    #     self.CBox.set('acquisition_mode', 0)

    #     NoSamples = 1e3
    #     self.CBox.set('acquisition_mode', 5)
    #     data = self.CBox.get_streaming_results(NoSamples)
    #     self.CBox.set('acquisition_mode', 0)
    #     self.assertTrue(len(data[0]) > NoSamples)
    #     self.assertTrue(len(data[0]) == len(data[1]))


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
        self.CBox.set('acquisition_mode', 0)
        NoSamples = 250

        self.CBox.set('nr_samples', NoSamples)
        self.CBox.set('nr_averages', 2)
        self.assertEqual(self.CBox.get('nr_samples'), NoSamples)

        self.CBox.set('acquisition_mode', 3)
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_input_avg_results()
        self.CBox.set('acquisition_mode', 0)

        # Only checks on lenght of signal as test
        # No check if averaging or signal delay works
        self.assertEqual(len(InputAvgRes0), NoSamples)
        self.assertEqual(len(InputAvgRes1), NoSamples)

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
