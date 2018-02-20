import unittest
import os
import pycqed as pq
import pycqed.measurement.openql_experiments.openql_helpers as oqh

file_paths_root = os.path.join(pq.__path__[0], 'tests',
                               'openQL_test_files')


class Test_openql_helpers(unittest.TestCase):

    def test_get_timetuples(self):
        qisa_fn = os.path.join(file_paths_root, 'TwoQ_RB.qisa')
        exp_time_tuples = [
            (1, 'prepz', {2}),
            (4, 'prepz', {0}),
            (15001, 'cw_03', {2}),
            (15002, 'cw_04', {2}),
            (15003, 'cw_03', {2}),
            (15004, 'cw_03', {2}),
            (15004, 'cw_05', {0}),
            (15005, 'cw_04', {2}),
            (15006, 'cw_03', {0, 2}),
            (15008, 'measz', {0, 2}),
            (15159, 'prepz', {2}),
            (15160, 'prepz', {0}),
            (30159, 'cw_05', {2}),
            (30160, 'cw_01', {0}),
            (30160, 'cw_04', {2}),
            (30161, 'cw_02', {0}),
            (30162, 'fl_cw_01', {(2, 0)}),
            (30175, 'cw_00', {2}),
            (30176, 'cw_06', {2}),
            (30176, 'cw_04', {0}),
            (30177, 'cw_05', {0}),
            (30177, 'cw_03', {2}),
            (30178, 'cw_06', {0}),
            (30179, 'fl_cw_01', {(2, 0)}),
            (30192, 'cw_06', {0}),
            (30192, 'cw_04', {2}),
            (30193, 'fl_cw_01', {(2, 0)}),
            (30206, 'cw_05', {2})]

        extr_time_tuples = oqh.get_timetuples(qisa_fn)

        self.assertEqual(extr_time_tuples[0:28], exp_time_tuples)