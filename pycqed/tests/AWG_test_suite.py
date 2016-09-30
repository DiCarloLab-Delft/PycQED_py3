import unittest
import qt


class AWG_tests(unittest.TestCase):
    '''
    This is a test suite for testing the HeterodyneSource Instrument.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        self.AWG = qt.instruments['AWG']
        if self.AWG is None:
            self.AWG = qt.instruments.create('AWG', 'Tektronix_AWG5014',
                                             setup_folder='RabiModelSims',
                                             address='GPIB0::8')
        self.AWG.reload()

    def test_instrument_is_proxy(self):
        self.assertEqual(self.AWG.__class__.__name__, 'Proxy',
                         'Instrument is not a Proxy, failure in ins creation')

    def test_load_setup_file(self):
        '''
        Loads two existing sequences (note might want to pick shorter/faster
                                      loading sequences)
        '''
        self.AWG.set_setup_filename('Rabi_3_5014')
        self.AWG.set_setup_filename('Rabi_15_5014')


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        AWG_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
