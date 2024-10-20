import os
import json
import unittest

from pycqed.measurement.openql_experiments import clifford_rb_oql as rb_oql
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


class Test_cliff_rb_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        OqlProgram.output_dir = os.path.join(curdir, 'test_output_cc')

    def test_single_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking(
            [0],
            platf_cfg=self.config_fn,
            nr_cliffords=[1, 5],
            nr_seeds=1,
            cal_points=False
        )
        self.assertEqual(p.name, 'randomized_benchmarking')
        hashes_fn = p.filename + ".hashes"
        if os.path.isfile(hashes_fn):
            # Remove the hashes file to make sure the next test runs correctly
            os.remove(hashes_fn)

    def test_rb_recompilation_needed_hash_based(self):
        """
        [2020-07-22 Victor]
        Checking for required recompilation of RB sequences was changed to a
        hash-based scheme
        """
        p = rb_oql.randomized_benchmarking(
            [0],
            platf_cfg=self.config_fn,
            nr_cliffords=[1, 5],
            nr_seeds=1,
            cal_points=False
        )
        hashes_fn = p.filename + ".hashes"
        assert os.path.isfile(hashes_fn)

        with open(hashes_fn) as json_file:
            hashes_dict = json.load(json_file)

        # Hash for the python code that generates the RB
        assert any("clifford_rb_oql.py" in key for key in hashes_dict.keys())
        # Hash for the OpenQL configuration file
        assert any("cfg" in key for key in hashes_dict.keys())

    def test_two_qubit_rb_seq(self):
        p = rb_oql.randomized_benchmarking(
            [2, 0],
            platf_cfg=self.config_fn,
            nr_cliffords=[1, 5],
            nr_seeds=1,
            cal_points=False
        )
        self.assertEqual(p.name, 'randomized_benchmarking')

    def test_two_qubit_rb_seq_interleaved(self):
        p = rb_oql.randomized_benchmarking(
            [2, 0],
            platf_cfg=self.config_fn,
            nr_cliffords=[1, 5],
            nr_seeds=1,
            cal_points=False,
            interleaving_cliffords=[104368]
        )
        self.assertEqual(p.name, 'randomized_benchmarking')

    def test_two_qubit_rb_seq_interleaved_idle(self):
        p = rb_oql.randomized_benchmarking(
            [2, 0],
            platf_cfg=self.config_fn,
            nr_cliffords=[1, 5],
            nr_seeds=1,
            cal_points=False,
            interleaving_cliffords=[100_000],
            flux_allocated_duration_ns=60,
        )
        self.assertEqual(p.name, 'randomized_benchmarking')


class Test_char_rb_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        OqlProgram.output_dir = os.path.join(curdir, 'test_output_cc')

    def test_two_qubit_character_rb(self):
        p = rb_oql.character_benchmarking(
            [2, 0],
            platf_cfg=self.config_fn,
            nr_cliffords=[2, 5, 11],
            nr_seeds=1
        )
        self.assertEqual(p.name, 'character_benchmarking')

    def test_two_qubit_character_rb_interleaved(self):
        p = rb_oql.character_benchmarking(
            [2, 0],
            platf_cfg=self.config_fn,
            interleaving_cliffords=[104368],
            nr_cliffords=[2, 5, 11],
            nr_seeds=1,
            program_name='character_bench_int_CZ'
        )
        self.assertEqual(p.name, 'character_bench_int_CZ')
