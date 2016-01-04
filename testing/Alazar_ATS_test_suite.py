import unittest
import qt
import defHeaders  # dictionar of bytestring commands
import numpy as np


class Alazar_tests(unittest.TestCase):
    '''
    This is a test suite for testing the Alazar_ATS Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the coms are working.
    '''
    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        self.ATS = qt.instruments['ATS']
        if self.ATS is None:
            raise NameError('ATS instrument does not exist')

    def test_aborting_acquire(self):
        pass

    def test_acquire_data(self):
        '''
        This test requires the ATS to require triggers from the AWG
        '''
        old_ro_timeout = self.ATS.get_ro_timeout()
        old_records_per_buffer = self.ATS.get_records_per_buffer()
        old_points_per_trace = self.ATS.get_points_per_trace()

        # set low to ensure error is raised quickly
        self.ATS.set_ro_timeout(10)
        self.ATS.arm()
        data = self.ATS.average_data(1)
        self.assertEqual(np.shape(data),
                         (self.ATS.get_records_per_buffer(),
                          self.ATS.get_points_per_trace()))

        self.ATS.set_records_per_buffer(2)
        self.ATS.set_points_per_trace(500)
        self.ATS.arm()
        data = self.ATS.average_data(1)
        self.assertEqual(np.shape(data),
                         (self.ATS.get_records_per_buffer(),
                          self.ATS.get_points_per_trace()))

        self.ATS.allocate_memory_for_DMA_buffers()
        with self.assertRaises(NameError) as cm:
            self.ATS.average_data(1)
        self.assertEqual('ApiWaitTimeout', cm.exception.message)
        with self.assertRaises(NameError) as cm:
            self.ATS.average_data(1)
        self.assertEqual('ApiBufferNotReady', cm.exception.message)
        # set back to default value
        self.ATS.set_ro_timeout(old_ro_timeout)
        self.ATS.set_records_per_buffer(old_records_per_buffer)
        self.ATS.set_points_per_trace(old_points_per_trace)

    def test_id(self):
        ser_nr = self.ATS.get_serial_number()
        self.assertEqual('ATS9870', self.ATS.get_board_kind())
        self.assertEqual(len(str(ser_nr)), 6)

    def test_getting_setting_overload(self):
        self.ATS.set_ch1_overload(True)
        self.ATS.set_ch2_overload(False)
        self.assertEqual(True, self.ATS.get_ch1_overload())
        self.assertEqual(False, self.ATS.get_ch2_overload())

        self.ATS.set_ch1_overload(False)
        self.ATS.set_ch2_overload(True)
        self.assertEqual(False, self.ATS.get_ch1_overload())
        self.assertEqual(True, self.ATS.get_ch2_overload())

        self.ATS.set_ch1_overload(False)
        self.ATS.set_ch2_overload(False)
        self.assertEqual(False, self.ATS.get_ch1_overload())
        self.assertEqual(False, self.ATS.get_ch2_overload())



    def test_ro_timeout(self):
        old_ro_timeout = self.ATS.get_ro_timeout()

        self.ATS.set_ro_timeout(5000)
        self.assertEqual(5000, self.ATS.get_ro_timeout())
        self.ATS.set_ro_timeout(3000)
        self.assertEqual(3000, self.ATS.get_ro_timeout())
        #sets it back to old setting
        self.ATS.set_ro_timeout(old_ro_timeout)

    def test_points_per_trace(self):
        self.assertEqual(self.ATS.get_max_points_per_trace(), 1e7)
        with self.assertRaises(ValueError):
            self.ATS.set_points_per_trace(1e7+2)
        ppt = 800
        trunced_ppt = int(ppt/64)*64
        self.ATS.set_points_per_trace(ppt)
        self.assertEqual(trunced_ppt, self.ATS.get_points_per_trace())
        ppt = 640
        self.ATS.set_points_per_trace(ppt)
        self.assertEqual(ppt, self.ATS.get_points_per_trace())

    def test_set_datatype(self):
        self.ATS.set_datatype('Signed')
        self.assertEqual(self.ATS.get_datatype(), '8 bit signed integer')
        self.ATS.set_datatype('unsigned')
        self.assertEqual(self.ATS.get_datatype(), '8 bit unsigned integer')

        self.ATS.set_datatype('8 bIt signed intEger')
        self.assertEqual(self.ATS.get_datatype(), '8 bit signed integer')
        self.ATS.set_datatype('8 bIt uNsIgNed intEger')
        self.assertEqual(self.ATS.get_datatype(), '8 bit unsigned integer')

        with self.assertRaises(KeyError):
            self.ATS.set_datatype('bla')


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        Alazar_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
