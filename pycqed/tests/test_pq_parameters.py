import numpy as np
from unittest import TestCase
from qcodes.utils.validators import Numbers, MultiType
from pycqed.instrument_drivers import pq_parameters


class Test_Custom_Validators(TestCase):

    def test_NP_NANs(self):
        nan_val = pq_parameters.NP_NANs()

        with self.assertRaises(ValueError):
            nan_val.validate(5)

        with self.assertRaises(ValueError):
            nan_val.validate('5')

        # This should not raise a value error
        nan_val.validate(np.nan)

        print(nan_val)  # tests the __repr__

    def test_multitype_NAN(self):
        nan_val = pq_parameters.NP_NANs()
        numbers_val = Numbers()

        multi_val = MultiType(nan_val, numbers_val)

        with self.assertRaises(ValueError):
            multi_val.validate('5')

        # This should not raise a value error
        multi_val.validate(np.nan)
        multi_val.validate(5)
