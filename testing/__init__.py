'''
Upon importing testing it will run all the tests that can be run without
instantiating any instruments.
'''

import unittest
from . import clifford_tests as clt
from importlib import reload
reload(clt)

test_classes_to_run = [clt.TestLookuptable,
                       clt.TestCalculateNetClifford,
                       clt.TestRecoveryClifford,
                       clt.TestRB_sequence]

suites_list = []
for test_class in test_classes_to_run:
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    suites_list.append(suite)

combined_test_suite = unittest.TestSuite(suites_list)
runner = unittest.TextTestRunner(verbosity=2).run(combined_test_suite)
