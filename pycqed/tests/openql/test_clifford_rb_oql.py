import os
import unittest
from openql import openql as ql
from pycqed.measurement.openql_experiments import clifford_rb_oql as rb_oql




class Test_cliff_rb_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

        output_dir = os.path.join(curdir, 'test_output')
        ql.set_option('output_dir', output_dir)

    def test_single_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking([0], platf_cfg=self.config_fn,
                                           nr_cliffords=[1, 5], nr_seeds=1,
                                           cal_points=False)
        self.assertEqual(p.name, 'randomized_benchmarking')

    def test_two_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking([2, 0], platf_cfg=self.config_fn,
                                           nr_cliffords=[1, 5], nr_seeds=1,
                                           cal_points=False)
        self.assertEqual(p.name, 'randomized_benchmarking')


class Test_char_rb_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

        output_dir = os.path.join(curdir, 'test_output')
        ql.set_option('output_dir', output_dir)

    def test_two_qubit_character_rb(self):
        p = rb_oql.character_benchmarking(
            [2, 0], platf_cfg=self.config_fn,
            nr_cliffords=[2, 5, 11], nr_seeds=1)
        self.assertEqual(p.name, 'character_benchmarking')

    def test_two_qubit_character_rb_interleaved(self):
        p = rb_oql.character_benchmarking(
            [2, 0], platf_cfg=self.config_fn,
            interleaving_cliffords=[-4368],
            nr_cliffords=[2, 5, 11], nr_seeds=1,
            program_name='character_bench_int_CZ')
        self.assertEqual(p.name, 'character_bench_int_CZ')


"""
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            randomized benchmarking OpenQL tests for Qutech Central Controller
    Notes:              requires OpenQL with CC backend support
"""

# NB: we just hijack the parent class to run the same tests

# FIXME: This only works with Wouters custom OpenQL.
# Need a better check for this
if ql.get_version() > '0.7.0':
    class Test_cliff_rb_oql_CC(Test_cliff_rb_oql):
        def setUp(self):
            curdir = os.path.dirname(__file__)
            self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
            output_dir = os.path.join(curdir, 'test_output_cc')
            ql.set_option('output_dir', output_dir)

    class Test_char_rb_oql_CC(Test_char_rb_oql):
        def setUp(self):
            curdir = os.path.dirname(__file__)
            self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
            output_dir = os.path.join(curdir, 'test_output_cc')
            ql.set_option('output_dir', output_dir)
else:
    class Test_cliff_rb_oql_CC(unittest.TestCase):
            @unittest.skip('OpenQL version does not support CC')
            def test_fail(self):
                pass
