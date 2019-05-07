"""
    File:               test_single_qubit_oql_CC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            single qubit OpenQL tests for Qutech Central Controller
    Notes:              requires OpenQL with CC backend support
    Usage:
    Bugs:

"""
import os
import pytest
from openql import openql as ql
from test_single_qubit_oql import Test_single_qubit_seqs_CCL


# FIXME: we just hijack the parent class to run the same tests
class Test_single_qubit_seqs_CC(Test_single_qubit_seqs_CCL):
    def setUp(self):
        rootDir = os.path.dirname(os.path.realpath(__file__))
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        output_dir = os.path.join(curdir, 'test_output_cc')
        ql.set_option('output_dir', output_dir)

    def test_RTE(self):
        pytest.skip("test_RTE() uses conditional gates, which are not implemented yet")

    def test_fast_feedback_control(self):
        pytest.skip("test_fast_feedback_control() uses conditional gates, which are not implemented yet")
