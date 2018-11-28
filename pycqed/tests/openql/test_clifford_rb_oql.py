import os
import unittest
from openql import openql as ql
from pycqed.measurement.openql_experiments import clifford_rb_oql as rb_oql


rootDir = os.path.dirname(os.path.realpath(__file__))
curdir = os.path.dirname(__file__)
config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

output_dir = os.path.join(curdir, 'test_output')
ql.set_option('output_dir', output_dir)


class Test_cliff_rb_oql(unittest.TestCase):
    def test_single_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking([0], platf_cfg=config_fn,
                                           nr_cliffords=[1, 5], nr_seeds=1,
                                           cal_points=False)
        self.assertEqual(p.name, 'randomized_benchmarking')

    def test_two_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking([2, 0], platf_cfg=config_fn,
                                           nr_cliffords=[1, 5], nr_seeds=1,
                                           cal_points=False)
        self.assertEqual(p.name, 'randomized_benchmarking')


class Test_char_rb_oql(unittest.TestCase):
    def test_two_qubit_character_rb(self):
        p = rb_oql.character_benchmarking(
            [2, 0], platf_cfg=config_fn,
            nr_cliffords=[2, 5, 11], nr_seeds=1)
        self.assertEqual(p.name, 'character_benchmarking')

    def test_two_qubit_character_rb_interleaved(self):
        p = rb_oql.character_benchmarking(
            [2, 0], platf_cfg=config_fn,
            interleaving_cliffords=[-4368],
            nr_cliffords=[2, 5, 11], nr_seeds=1,
            program_name='character_bench_int_CZ')
        self.assertEqual(p.name, 'character_bench_int_CZ')
