# FIXME: based on OpenQL test, cleanup

import os
import unittest
import pathlib
#from utils import file_compare

import pycqed.measurement.openql_experiments.generate_CC_cfg_modular as gen
import pycqed.measurement.openql_experiments.cqasm.special_cq as spcq
from pycqed.measurement.openql_experiments.openql_helpers import OqlCfg


curdir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(curdir, 'test_output_cc')
platf_cfg = output_dir + '/config_cc_s17_direct_iq_openql_0_10.json' # FIXME: naming, use pathlib Path


class Test_cQASM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen.generate_config_modular(platf_cfg)

    def test_nested_rus_angle_0(self):
        ancilla1_idx = 10
        ancilla2_idx = 8
        data_idx = 11
        angle = 0

        p = spcq.nested_rus(
            OqlCfg('nested_rus_angle_0', platf_cfg, output_dir),
            ancilla1_idx,
            ancilla2_idx,
            data_idx,
            angle
        )
