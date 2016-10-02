
import sys


def test_core(verbosity=1, failfast=False, test_pattern='test*.py'):
    """
    Run the pycqed core tests.

    Args:
        verbosity (int, optional): 0, 1, or 2, higher displays more info
            Default 1.
        failfast (bool, optional): If true, stops running on first failure
            Default False.
        test_pattern (str): the pattern used to detect test files.

    Coverage testing is only available from the command line
    """

    _test_core(verbosity=verbosity, failfast=failfast,
               test_pattern=test_pattern)


def _test_core(test_pattern='test*.py', **kwargs):
    import pycqed
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

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    try:
        import coverage
        coverage_missing = False
    except ImportError:
        coverage_missing = True

    # make sure coverage looks for .coveragerc in the right place
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description=('Core test suite for PycQED'))

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase verbosity')

    parser.add_argument('-q', '--quiet', action='store_true',
                        help='reduce verbosity (opposite of --verbose)')

    parser.add_argument('-s', '--skip-coverage', action='store_true',
                        help='skip coverage reporting')

    parser.add_argument('-t', '--test_pattern', type=str, default='test*.py',
                        help=('regexp for test name to match, '
                              'default "test*.py"'))

    parser.add_argument('-f', '--failfast', action='store_true',
                        help='halt on first error/failure')

    args = parser.parse_args()

    args.skip_coverage |= coverage_missing

    if not args.skip_coverage:
        cov = coverage.Coverage(source=['pycqed'])
        cov.start()

    success = test_core(verbosity=(1 + args.verbose - args.quiet),
                        failfast=args.failfast,
                        test_pattern=args.test_pattern)

    if not args.skip_coverage:
        cov.stop()
        cov.save()
        cov.report()

    # restore unix-y behavior
    # exit status 1 on fail
    if not success:
        sys.exit(1)
