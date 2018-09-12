from pycqed.measurement.openql_experiments.generate_CCL_cfg import  \
    generate_config

from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
from pycqed.measurement.openql_experiments import multi_qubit_oql as mqo
import os
import unittest
from openql import openql as ql

rootDir = os.path.dirname(os.path.realpath(__file__))
curdir = os.path.dirname(__file__)
config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

output_dir = os.path.join(curdir, 'test_output')
ql.set_option('output_dir', output_dir)


class Test_configuration_files(unittest.TestCase):

    def test_openQL_config_valid(self):
        test_config_fn = os.path.join(curdir, 'test_gen_cfg_CCL.json')
        generate_config(filename=test_config_fn,
                        mw_pulse_duration=20, ro_duration=300,
                        init_duration=200000)

        # If this compiles we conclude that the generated config is valid
        # A single qubit sequence
        sqo.AllXY(qubit_idx=0, platf_cfg=test_config_fn)
        # A sequence containing two-qubit gates
        mqo.single_flux_pulse_seq(qubit_indices=(2, 0),
                                  platf_cfg=test_config_fn)
        # A sequence containing controlled operations
        sqo.RTE(qubit_idx=0,
                sequence_type='echo', net_gate='pi', feedback=True,
                platf_cfg=test_config_fn)
