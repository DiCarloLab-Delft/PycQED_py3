import unittest
import numpy as np

from pycqed.utilities import general as gen
from pycqed.analysis.tools.data_manipulation import (rotation_matrix,
                                                     rotate_complex)


class Test_misc(unittest.TestCase):

    def test_rotation_matrix(self):
        rot_mat = rotation_matrix(0)
        np.testing.assert_array_equal(np.eye(2), rot_mat)

        test_vec = np.matrix([1, 0]).T
        np.testing.assert_array_almost_equal(
            rotation_matrix(90)*test_vec, np.matrix([0, 1]).T)
        np.testing.assert_array_almost_equal(
            rotation_matrix(360)*test_vec, test_vec)
        np.testing.assert_array_almost_equal(
            rotation_matrix(180)*rotation_matrix(180)*test_vec, test_vec)

        test_vec = np.matrix([1, 0.5]).T
        np.testing.assert_array_almost_equal(
            rotation_matrix(360)*test_vec, test_vec)
        np.testing.assert_array_almost_equal(
            rotation_matrix(180)*rotation_matrix(180)*test_vec, test_vec)

    def test_rotate_complex_number(self):
        self.assertAlmostEqual(1j, rotate_complex(1, 90, deg=True))

        real_vec = np.ones(10)
        np.testing.assert_almost_equal(1j*real_vec,
                                       rotate_complex(real_vec, 90, deg=True))

    def test_span(self):
        linspan = gen.span_num(3.8, .4, 21)
        self.assertAlmostEqual(min(linspan), 3.6)
        self.assertAlmostEqual(max(linspan), 4)
        self.assertAlmostEqual(len(linspan), 21)

        rangespan = gen.span_step(3.8, .4, .05)
        self.assertAlmostEqual(min(rangespan), 3.6)
        self.assertAlmostEqual(max(rangespan), 4)
        self.assertEqual(len(rangespan), 9)

    def test_gen_sweep_pts(self):
        lin = gen.gen_sweep_pts(start=3.8, stop=4.2, num=21)
        np.testing.assert_array_equal(lin, np.linspace(3.8, 4.2, 21))

        linspan = gen.gen_sweep_pts(center=3.8, span=.2, num=21)
        linspan2 = gen.span_num(3.8, .2, 21)
        np.testing.assert_array_equal(linspan, linspan2)

        ran = gen.gen_sweep_pts(start=3.8, stop=4.2, step=.05)
        np.testing.assert_array_equal(ran, np.arange(3.8, 4.2001, .05))

        ran = gen.gen_sweep_pts(center=3.8, span=.2, step=.05)
        np.testing.assert_array_equal(ran, gen.span_step(3.8, .200, .05))

        # missing arguments or invalid combinations of arguments should
        # raise errors
        with self.assertRaises(ValueError):
            gen.gen_sweep_pts(center=3.8, stop=4, step=5)
        with self.assertRaises(ValueError):
            gen.gen_sweep_pts(start=3.8, stop=.3)
        with self.assertRaises(ValueError):
            gen.gen_sweep_pts(center=3.8, span=.3)

    def test_get_set_InDict(self):
        # Test setting in nested dict
        test_dict = {'a': {'nest_a': 5}}
        val = gen.getFromDict(test_dict, ['a', 'nest_a'])
        self.assertEqual(val, 5)

        # Test setting in nested dict
        tval = 23
        gen.setInDict(test_dict, ['a', 'nest_a'], tval)
        val = gen.getFromDict(test_dict, ['a', 'nest_a'])
        self.assertEqual(val, 23)
        self.assertEqual(test_dict, {'a': {'nest_a': 23}})


class Test_int_to_base(unittest.TestCase):

    def test_int2base(self):
        self.assertEqual(gen.int2base(11, base=3), '102')

        self.assertEqual(gen.int2base(11, base=3, fixed_length=5), '00102')
