import os
import unittest
import pytest
from openql import openql as ql
from pycqed.measurement.openql_experiments.pygsti_oql import \
    poor_mans_2q_gst, single_qubit_gst, two_qubit_gst
from pycqed.measurement.openql_experiments import openql_helpers as oqh

# pytestmark = pytest.mark.skip
class Test_pygsti_oql(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

        output_dir = os.path.join(curdir, 'test_output')
        ql.set_option('output_dir', output_dir)

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

##########################################################################
# repeat same tests for Qutech Central Controller
# NB: we just hijack the parent class to run the same tests
# NB: requires OpenQL with CC backend support
##########################################################################

if oqh.is_compatible_openql_version_cc():
    class Test_pygsti_oql_CC(Test_pygsti_oql):
        def setUp(self):
            curdir = os.path.dirname(__file__)
            self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
            output_dir = os.path.join(curdir, 'test_output_cc')
            ql.set_option('output_dir', output_dir)
else:
    class Test_pygsti_oql_CC_incompatible_openql_version(unittest.TestCase):
            @unittest.skip('OpenQL version does not support CC')
            def test_fail(self):
                pass

