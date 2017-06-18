import unittest
import pycqed as pq
import os
import numpy as np
from pycqed.analysis.tomography import Tomo_Multiplexed
from pycqed.analysis import analysis_toolbox as a_tools
a_tools.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')

try:
    import qutip
    from pycqed.analysis import tomography as tomo

    class Test_tomo_analysis(unittest.TestCase):

        @classmethod
        def setUpClass(self):
            pass

        def test_tomo_analysis_cardinal_state(self):

            res = Tomo_Multiplexed(label='Tomo_{}'.format(31),
                                   target_cardinal=None,
                                   MLE=False)
            res = Tomo_Multiplexed(label='Tomo_{}'.format(31),
                                   target_cardinal=31,
                                   MLE=True)

        def test_tomo_analysis_bell_state(self):
            res = Tomo_Multiplexed(label='Tomo_{}'.format(31),
                                   target_cardinal=None,
                                   target_bell=0,
                                   MLE=False)

    class Test_tomo_helpers(unittest.TestCase):

        def test_bell_fids(self):
            pass

        def test_bell_paulis(self):

            bell_0 = [1] + [0]*3 + [0]*3 + [-1, 0, 0, 0, 1, 0, 0, 0, 1]
            bell_1 = [1] + [0]*3 + [0]*3 + [1, 0, 0, 0, -1, 0, 0, 0, 1]
            bell_2 = [1] + [0]*3 + [0]*3 + [-1, 0, 0, 0, -1, 0, 0, 0, -1]
            bell_3 = [1] + [0]*3 + [0]*3 + [1, 0, 0, 0, 1, 0, 0, 0, -1]
            expected_bells = [bell_0, bell_1, bell_2, bell_3]
            # Test if the definition or order has not changed
            for bell_idx in range(4):
                bell_paulis = tomo.get_bell_pauli_exp(
                    bell_idx, theta_q0=0, theta_q1=0)
                np.testing.assert_array_equal(
                    expected_bells[bell_idx], bell_paulis)


except ImportError as e:
    if str(e).find('qutip') >= 0:
        class Test_tomo_analysis_skipped(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail():
                pass
    else:
        raise
#


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(Test_tomo_analysis)
    # suite = unittest.TestLoader().loadTestsFromTestCase(Test_tomo_helpers)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
