import unittest
import qt


class HeterodyneSourceTests(unittest.TestCase):
    '''
    This is a test suite for testing the HeterodyneSource Instrument.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        self.HS = qt.instruments['HS']
        self.RF = qt.instruments['RF']
        self.LO = qt.instruments['LO']
        if self.HS is None:
            self.HS = qt.instruments.create(
                'HS', 'HeterodyneSource', RF='RF', LO='LO', IF=.01)

    def test_instrument_is_proxy(self):
        self.assertEqual(self.HS.__class__.__name__, 'Proxy',
                         'Instrument is not a Proxy, failure in ins creation')

    def test_setting_RF_power(self):
        old_RF_power = self.RF.get_power()
        test_power = old_RF_power + 3
        self.HS.set_RF_power(test_power)
        self.assertEqual(test_power, self.HS.get_RF_power())
        self.assertEqual(test_power, self.RF.get_power())
        self.RF.set_power(old_RF_power)

    def test_setting_LO_power(self):
        old_LO_power = self.LO.get_power()
        test_power = old_LO_power + 3
        self.HS.set_LO_power(test_power)
        self.assertEqual(test_power, self.HS.get_LO_power())
        self.assertEqual(test_power, self.LO.get_power())
        self.LO.set_power(old_LO_power)

    def test_setting_frequency(self):
        old_RF_frequency = self.RF.get_frequency()
        old_LO_frequency = self.LO.get_frequency()

        freq = 6.8  # GHz
        IF = self.HS.get_IF()
        self.HS.set_frequency(freq)

        self.assertEqual(self.HS.get_frequency(), freq)
        # 1e9 for Hz GHz conversion
        self.assertEqual(self.RF.get_frequency(), freq*1e9)
        self.assertEqual(self.LO.get_frequency() - IF*1e9, freq*1e9)

        self.HS.set_frequency(old_RF_frequency*1e-9)
        self.RF.set_frequency(old_RF_frequency)
        self.LO.set_frequency(old_LO_frequency)


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        HeterodyneSourceTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
