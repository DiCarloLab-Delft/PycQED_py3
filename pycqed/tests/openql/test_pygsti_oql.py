import os
import unittest
from openql import openql as ql
from pycqed.measurement.openql_experiments.pygsti_oql import \
    poor_mans_2q_gst, single_qubit_gst, two_qubit_gst


rootDir = os.path.dirname(os.path.realpath(__file__))
curdir = os.path.dirname(__file__)
config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

output_dir = os.path.join(curdir, 'test_output')
ql.set_option('output_dir', output_dir)


class Test_pygsti_oql(unittest.TestCase):

    def test_poor_mans_2q_gst(self):
        p = poor_mans_2q_gst(q0=0, q1=2, platf_cfg=config_fn)
        self.assertEqual(len(p.sweep_points), 731)

    def test_single_qubit_gst(self):
        programs = single_qubit_gst(0, config_fn,
                                    maxL=256, lite_germs=True,
                                    recompile=True)

    def test_two_qubit_gst(self):
        programs = two_qubit_gst([2, 0], config_fn,
                                 maxL=4, lite_germs=True,
                                 recompile=True)
