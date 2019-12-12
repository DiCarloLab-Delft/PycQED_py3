import unittest
import os
import pycqed as pq
import pycqed.measurement.openql_experiments.openql_helpers as oqh
import openql.openql as ql

file_paths_root = os.path.join(pq.__path__[0], 'tests',
                               'openQL_test_files')


class Test_openql_helpers(unittest.TestCase):

    def test_get_timetuples(self):
        qisa_fn = os.path.join(file_paths_root, 'TwoQ_RB.qisa')
        exp_time_tuples = [
            (1, 'prepz', {2}, 15),
            (4, 'prepz', {0}, 16),
            (15001, 'cw_03', {2}, 18),
            (15002, 'cw_04', {2}, 19),
            (15003, 'cw_03', {2}, 20),
            (15004, 'cw_03', {2}, 21),
            (15004, 'cw_05', {0}, 21),
            (15005, 'cw_04', {2}, 22),
            (15006, 'cw_03', {0, 2}, 23),
            (15008, 'measz', {0, 2}, 24),
            (15159, 'prepz', {2}, 26),
            (15160, 'prepz', {0}, 27),
            (30159, 'cw_05', {2}, 29),
            (30160, 'cw_01', {0}, 30),
            (30160, 'cw_04', {2}, 30),
            (30161, 'cw_02', {0}, 31),
            (30162, 'fl_cw_01', {(2, 0)}, 32),
            (30175, 'cw_00', {2}, 34),
            (30176, 'cw_06', {2}, 35),
            (30176, 'cw_04', {0}, 35),
            (30177, 'cw_05', {0}, 36),
            (30177, 'cw_03', {2}, 36),
            (30178, 'cw_06', {0}, 37),
            (30179, 'fl_cw_01', {(2, 0)}, 38),
            (30192, 'cw_06', {0}, 40),
            (30192, 'cw_04', {2}, 40),
            (30193, 'fl_cw_01', {(2, 0)}, 41),
            (30206, 'cw_05', {2}, 43)]
        extr_time_tuples = oqh.get_timetuples(qisa_fn)

        self.assertEqual(extr_time_tuples[0:28], exp_time_tuples)

    def test_plot_tuples(self):

        qisa_fn = os.path.join(file_paths_root, 'TwoQ_RB.qisa')
        ttuple = oqh.get_timetuples(qisa_fn)

        # Test only checks if the plotting does not crash but in the process
        # does run a lot of helper functions
        oqh.plot_time_tuples_split(ttuple)

    def test_get_operation_tuples(self):
        qisa_fn = os.path.join(file_paths_root, 'TwoQ_RB.qisa')
        ttuple = oqh.get_timetuples(qisa_fn)

        grouped_timetuples = oqh.split_time_tuples_on_operation(ttuple, 'meas')
        flux_tuples = oqh.get_operation_tuples(grouped_timetuples[8], 'fl')

        exp_time_tuples = [
            (137199, 'fl_cw_01', {(2, 0)}, 400),
            (137218, 'fl_cw_01', {(2, 0)}, 408),
            (137232, 'fl_cw_01', {(2, 0)}, 411),
            (137250, 'fl_cw_01', {(2, 0)}, 418),
            (137264, 'fl_cw_01', {(2, 0)}, 421),
            (137282, 'fl_cw_01', {(2, 0)}, 428),
            (137296, 'fl_cw_01', {(2, 0)}, 431),
            (137316, 'fl_cw_01', {(2, 0)}, 440),
            (137333, 'fl_cw_01', {(2, 0)}, 446),
            (137352, 'fl_cw_01', {(2, 0)}, 454),
            (137369, 'fl_cw_01', {(2, 0)}, 460),
            (137387, 'fl_cw_01', {(2, 0)}, 467),
            (137401, 'fl_cw_01', {(2, 0)}, 470),
            (137421, 'fl_cw_01', {(2, 0)}, 479)]

        self.assertEqual(exp_time_tuples, flux_tuples)

    def test_flux_pulse_replacement(self):
        qisa_fn = os.path.join(file_paths_root, 'TwoQ_RB.qisa')

        mod_qisa_fn, grouped_fl_tuples = oqh.flux_pulse_replacement(qisa_fn)
        with open(mod_qisa_fn, 'r') as mod_file:
            lines = mod_file.readlines()
        exp_qisa_fn = os.path.join(file_paths_root,
                                   'TwoQ_RB_mod_expected.qisa')
        with open(exp_qisa_fn, 'r') as exp_file:
            exp_lines = exp_file.readlines()

        self.assertEqual(exp_lines, lines)

        expected_flux_tuples = [
            (0, 'fl_cw_01', {(2, 0)}, 601),
            (19, 'fl_cw_01', {(2, 0)}, 609),
            (33, 'fl_cw_01', {(2, 0)}, 612),
            (47, 'fl_cw_01', {(2, 0)}, 615),
            (64, 'fl_cw_01', {(2, 0)}, 621),
            (84, 'fl_cw_01', {(2, 0)}, 630),
            (100, 'fl_cw_01', {(2, 0)}, 635),
            (117, 'fl_cw_01', {(2, 0)}, 641),
            (131, 'fl_cw_01', {(2, 0)}, 644),
            (149, 'fl_cw_01', {(2, 0)}, 651),
            (168, 'fl_cw_01', {(2, 0)}, 659),
            (185, 'fl_cw_01', {(2, 0)}, 665),
            (199, 'fl_cw_01', {(2, 0)}, 668),
            (218, 'fl_cw_01', {(2, 0)}, 676),
            (232, 'fl_cw_01', {(2, 0)}, 679),
            (250, 'fl_cw_01', {(2, 0)}, 686),
            (269, 'fl_cw_01', {(2, 0)}, 694),
            (283, 'fl_cw_01', {(2, 0)}, 697),
            (297, 'fl_cw_01', {(2, 0)}, 700)]

        self.assertEqual(expected_flux_tuples, grouped_fl_tuples[10])


class Test_openql_compiler_helpers(unittest.TestCase):

    def test_create_program(self):
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_CCL.json')
        p = oqh.create_program('test_program', config_fn)
        self.assertEqual(p.name, 'test_program')
        self.assertEqual(p.output_dir, ql.get_option('output_dir'))

    def test_create_kernel(self):
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_CCL.json')
        p = oqh.create_program('test_program', config_fn)
        k = oqh.create_kernel('my_kernel', p)
        self.assertEqual(k.name, 'my_kernel')


    def test_compile(self):
        """
        Only tests the compile helper by compiling an empty file.
        """
        curdir = os.path.dirname(__file__)
        config_fn = os.path.join(curdir, 'test_cfg_CCL.json')
        p = oqh.create_program('test_program', config_fn)
        k = oqh.create_kernel('test_kernel', p)
        p.add_kernel(k)
        p = oqh.compile(p)
        fn_split = os.path.split(p.filename)

        self.assertEqual(fn_split[0], ql.get_option('output_dir'))
        self.assertEqual(fn_split[1], 'test_program.qisa')


class Test_openql_calibration_point_helpers(unittest.TestCase):

    @unittest.skip('Test not implemented')
    def test_add_single_qubit_cal_points(self):
        raise NotImplementedError()

    @unittest.skip('Test not implemented')
    def test_add_two_q_cal_points(self):
        raise NotImplementedError()

    @unittest.skip('Test not implemented')
    def test_add_multi_q_cal_points(self):
        raise NotImplementedError()
