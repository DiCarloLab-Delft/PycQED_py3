
import sys


def test_core(verbosity=1, failfast=False):
    """
    Run the pycqed core tests.

    Args:
        verbosity (int, optional): 0, 1, or 2, higher displays more info
            Default 1.
        failfast (bool, optional): If true, stops running on first failure
            Default False.

    Coverage testing is only available from the command line
    """
    import qcodes
    _test_core(verbosity=verbosity, failfast=failfast)


def _test_core(test_pattern='test*.py', **kwargs):
    import unittest
    import pycqed.tests as pyqtest


    suite = unittest.defaultTestLoader.discover(
        pyqtest.__path__[0], top_level_dir=pycqed.__path__[0],
        pattern=test_pattern)
    if suite.countTestCases() == 0:
        print('found no tests')
        sys.exit(1)
    print('testing %d cases' % suite.countTestCases())

    result = unittest.TextTestRunner(**kwargs).run(suite)
    return result.wasSuccessful()
