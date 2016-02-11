'''
Upon importing testing it will run all the tests that can be run without
instantiating any instruments
'''

import unittest
from . import clifford_tests as clt
from importlib import reload
reload(clt)

suite = unittest.TestLoader().loadTestsFromTestCase(
    clt.TestLookuptable)
unittest.TextTestRunner(verbosity=2).run(suite)