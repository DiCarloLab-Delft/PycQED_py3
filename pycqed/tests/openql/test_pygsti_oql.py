import os
import unittest

from pycqed.measurement.openql_experiments.pygsti_oql import \
    poor_mans_2q_gst, single_qubit_gst, two_qubit_gst
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


class Test_pygsti_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        OqlProgram.output_dir = os.path.join(curdir, 'test_output_cc')

    def test_poor_mans_2q_gst(self):
        p = poor_mans_2q_gst(q0=0, q1=2, platf_cfg=self.config_fn)
        self.assertEqual(len(p.sweep_points), 731)

    def test_single_qubit_gst(self):
        programs = single_qubit_gst(0, self.config_fn,
                                    maxL=256, lite_germs=True,
                                    recompile=True)

    def test_two_qubit_gst(self):
        programs = two_qubit_gst([2, 0], self.config_fn,
                                 maxL=4, lite_germs=True,
                                 recompile=True)
