"""
    File:               test_pygsti_oql_CC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            pygsti OpenQL tests for Qutech Central Controller
    Notes:              requires OpenQL with CC backend support
    Usage:
    Bugs:

"""
import os
from openql import openql as ql
import test_pygsti_oql as parent  # rename to stop pytest from running tests directly


# NB: we just hijack the parent class to run the same tests
class Test_pygsti_oql_CC(parent.Test_pygsti_oql):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
        output_dir = os.path.join(curdir, 'test_output_cc')
        ql.set_option('output_dir', output_dir)

