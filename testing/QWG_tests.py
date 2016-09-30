#!/usr/bin/python
import unittest
from instrument_drivers.physical_instruments.QuTech_AWG_Module \
    import QuTech_AWG_Module
from measurement.waveform_control_CC.waveform import Waveform
import time
import numpy as np
from socket import timeout
from qcodes.utils import validators as vals


class QWG_tests(unittest.TestCase):
    '''
    This is a test suite for testing the HeterodyneSource Instrument.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Try using existing instrument if it exists, otherwise create it.
        '''
        try:
            self.qwg = QWG
        except:
            self.qwg = QuTech_AWG_Module(
                'QWG', address='192.168.42.10',
                port=5025, server_name=None)

        self.qwg.reset()
        self.qwg._socket.settimeout(.3)  # set low timeout for tests

    def test_IDN(self):
        IDN = self.qwg.IDN()
        self.assertIsInstance(IDN, dict)
        self.assertEqual(IDN['vendor'], 'QuTech')
        self.assertEqual(IDN['model'], 'QWG')

    def test_Error(self):
        err_msg = self.qwg.getError()
        self.assertEqual(err_msg, '0,"No error"')

    def bool_get_set(self, par):
        old_val = par.get()
        par.set(True)
        self.assertTrue(par.get(), msg=par.name)
        par.set(False)
        self.assertFalse(par.get(), msg=par.name)
        par.set(old_val)

    def array_get_set(self, par):
        old_val = par.get()
        shape = par._vals._shape
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100
        v = (np.zeros(shape)+mn)+(mx-mn)/2
        par.set(v)
        np.testing.assert_array_equal(v, par.get(), err_msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn))
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn))
        par.set(old_val)

    def floating_get_set(self, par):
        old_val = par.get()
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100.5
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100.5
        v = (mn)+(mx-mn)/2
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn))
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn))
        par.set(old_val)

    def integer_get_set(self, par):
        old_val = par.get()
        mn = par._vals._min_value
        if mn == -float("inf"):
            mn = -100
        mx = par._vals._max_value
        if mx == float("inf"):
            mx = 100
        v = (mn)+(mx-mn)//2
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v+(mx-mn)*10)
        with self.assertRaises(ValueError, msg=par.name):
            par.set(v-(mx-mn)*10)
        par.set(old_val)

    def string_get_set(self, par):
        old_val = par.get()
        min_len = par._vals._min_length
        max_len = par._vals._max_length

        v = 'test_string'
        if len(v) > max_len:
            v = v[:max_len]
        if len(v) < min_len:
            par.set(v)  # expect failure and fix it
        par.set(v)
        self.assertEqual(v, par.get(), msg=par.name)
        par.set(old_val)

    def enum_get_set(self, par):
        old_val = par.get()
        for v in par._vals._values:
            par.set(v)
            self.assertEqual(v, par.get(), msg=par.name)
        par.set(old_val)

    def test_parameters(self):
        for parname, par in sorted(self.qwg.parameters.items()):
            # no more failing pars! still here so I can re add if needed
            failing_pars = []
            if par.name not in ['IDN'] and par.name not in failing_pars:
                old_value = par.get()
                old_value2 = par.get()
                np.testing.assert_equal(old_value2, old_value,
                                        err_msg=par.name)
                if hasattr(par, '_vals') and par.has_set:
                    validator = par._vals
                    if isinstance(validator, vals.Ints):
                        self.integer_get_set(par)
                    elif isinstance(validator, vals.Numbers):
                        self.floating_get_set(par)
                    elif isinstance(validator, vals.Arrays):
                        self.array_get_set(par)
                    elif isinstance(validator, vals.Bool):
                        self.bool_get_set(par)
                    elif isinstance(validator, vals.Strings):
                        self.string_get_set(par)
                    elif isinstance(validator, vals.Enum):
                        self.enum_get_set(par)

                    else:
                        print('{} validator "{}" not recognized'.format(
                            par.name, par._vals))
                else:
                    print(par.name, 'does not have validator, not testing')
            else:
                print('Not in pars to be tested: "{}"'.format(par.name))

    def tearDown(self):
        self.qwg._socket.settimeout(5)
        # set timeout back to default



if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(
        QWG_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
