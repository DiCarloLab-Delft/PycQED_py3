import unittest
import os
import itertools
from pycqed.measurement.gate_set_tomography.pygsti_helpers import \
    pygsti_expList_from_dataset, gst_exp_filepath, split_expList


class Test_pygsti_helpers(unittest.TestCase):

    def test_pygsti_expList_from_dataset(self):
        dataset_filename = os.path.join(gst_exp_filepath,
                                        'std1Q_XYI_lite_maxL256.txt')
        explist = pygsti_expList_from_dataset(
            dataset_filename=dataset_filename)

        expected_firstpart = ['{}', 'Gx', 'Gy', 'GxGx', 'GxGxGx',
                              'GyGyGy', 'GxGy', 'GxGxGxGx', 'GxGyGyGy', 'GyGx']
        for i, gs in enumerate(explist[0:10]):
            self.assertEqual(gs.str, expected_firstpart[i])

        expected_finalpart = [
            'GxGxGx(GxGxGy)^85Gy', 'GxGxGx(GxGxGy)^85GxGx',
            'GxGxGx(GxGxGy)^85GxGxGx', 'GxGxGx(GxGxGy)^85GyGyGy',
            'GyGyGy(GxGxGy)^85', 'GyGyGy(GxGxGy)^85Gx', 'GyGyGy(GxGxGy)^85Gy',
            'GyGyGy(GxGxGy)^85GxGx', 'GyGyGy(GxGxGy)^85GxGxGx',
            'GyGyGy(GxGxGy)^85GyGyGy']
        for i, gs in enumerate(explist[-10:]):
            self.assertEqual(gs.str, expected_finalpart[i])

    def test_split_expList(self):
        dataset_filename = os.path.join(gst_exp_filepath,
                                        'std1Q_XYI_lite_maxL256.txt')
        explist = pygsti_expList_from_dataset(
            dataset_filename=dataset_filename)

        splitted_lists = split_expList(explist, verbose=True)
        recombined_list = list(itertools.chain.from_iterable(splitted_lists))

        self.assertListEqual(explist, recombined_list)
