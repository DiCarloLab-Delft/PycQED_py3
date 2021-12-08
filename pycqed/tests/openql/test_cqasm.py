"""
Usage:
pytest -v pycqed/tests/openql/test_cqasm.py
pytest -v pycqed/tests/openql/test_cqasm.py --log-level=DEBUG --capture=no
"""

import unittest
import pathlib

#from utils import file_compare

import pycqed.measurement.openql_experiments.generate_CC_cfg_modular as gen
import pycqed.measurement.openql_experiments.cqasm.special_cq as spcq
import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


this_path = pathlib.Path(__file__).parent
output_path = pathlib.Path(this_path) / 'test_output_cc'
platf_cfg_path = output_path / 'config_cc_s17_direct_iq_openql_0_10.json'


class Test_cQASM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen.generate_config_modular(platf_cfg_path)
        OqlProgram.output_dir = str(output_path)

    if oqh.is_compatible_openql_version_cc():
        def test_nested_rus_angle_0(self):
            ancilla1_idx = 10
            ancilla2_idx = 8
            data_idx = 11
            angle = 0

            p = spcq.nested_rus(
                str(platf_cfg_path),
                ancilla1_idx,
                ancilla2_idx,
                data_idx,
                angle
            )

            # check that a file with the expected name has been generated
            assert pathlib.Path(p.filename).is_file()
    else:
        @unittest.skip('OpenQL version does not support CC')
        def test_fail(self):
            pass
