import os
import unittest
import numpy as np

try:
    from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
    from openql import openql as ql

    rootDir = os.path.dirname(os.path.realpath(__file__))
    curdir = os.path.dirname(__file__)
    config_fn = os.path.join(curdir, 'test_data/test_cfg_CCL.json')

    output_dir = os.path.join(curdir, 'test_output')
    ql.set_output_dir(output_dir)

    class Test_single_qubit_seqs_CCL(unittest.TestCase):
        def test_allxy(self):
            # Only test if it compiles
            sqo.AllXY(qubit_idx=0, platf_cfg=config_fn)

        def test_single_elt_on(self):
            sqo.single_elt_on(qubit_idx=0, platf_cfg=config_fn)

        def test_flipping(self):
            number_of_flips = np.arange(10)
            sqo.flipping(qubit_idx=0, number_of_flips=number_of_flips,
                         platf_cfg=config_fn)

        def test_butterfly(self):
            sqo.butterfly(qubit_idx=0, initialize=True,
                          platf_cfg=config_fn)


except ImportError as e:
    class TestMissingDependency(unittest.TestCase):

        @unittest.skip('Missing dependency - ' + str(e))
        def test_fail():
            pass

if __name__ == '__main__':
    unittest.main()
