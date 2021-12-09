import unittest
import os

import pycqed as pq
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


class Test_openql_compiler_helpers(unittest.TestCase):

    def test_create_program(self):
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        p = OqlProgram('test_program', config_fn)
        self.assertEqual(p.name, 'test_program')

    def test_create_kernel(self):
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        p = OqlProgram('test_program', config_fn)
        k = p.create_kernel('my_kernel')
        self.assertEqual(k.name, 'my_kernel')

    #@unittest.skip('FIXME: disabled, see PR #643 and PR #635 (marked as important)')
    def test_compile(self):
        """
        Only tests the compile helper by compiling an empty file.
        """
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        p = OqlProgram('test_program', config_fn)
        k = p.create_kernel('test_kernel')
        p.add_kernel(k)
        p.compile()

        # fn_split = os.path.split(p.filename)
        # self.assertEqual(fn_split[1], 'test_program.qisa')


class Test_openql_calibration_point_helpers(unittest.TestCase):
# FIXME: the tests below appear to have never been implemented, but would be useful

    @unittest.skip('Test not implemented')
    def test_add_single_qubit_cal_points(self):
        raise NotImplementedError()

    @unittest.skip('Test not implemented')
    def test_add_two_q_cal_points(self):
        raise NotImplementedError()

    @unittest.skip('Test not implemented')
    def test_add_multi_q_cal_points(self):
        raise NotImplementedError()
