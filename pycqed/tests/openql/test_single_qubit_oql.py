import os
import unittest
import pytest
import numpy as np

#try:  # FIXME: hides import problems
if 1:
    from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
    from pycqed.measurement.openql_experiments import openql_helpers as oqh
    from openql import openql as ql

    class Test_single_qubit_seqs_CCL(unittest.TestCase):
        def setUp(self):
            curdir = os.path.dirname(__file__)
            self.config_fn = os.path.join(curdir, 'test_cfg_CCL.json')

            output_dir = os.path.join(curdir, 'test_output')
            ql.set_option('output_dir', output_dir)

        def test_CW_tone(self):
            p = sqo.CW_tone(qubit_idx=1,
                            platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'CW_tone')

        def test_vsm_timing_cal_sequence(self):
            p = sqo.vsm_timing_cal_sequence(qubit_idx=1,
                                            platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'vsm_timing_cal_sequence')

        def test_CW_RO_seq(self):
            p = sqo.CW_RO_sequence(qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'CW_RO_sequence')

        def test_CW_RO_seq_multiple_targets(self):
            p = sqo.CW_RO_sequence(qubit_idx=[0, 1], platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'CW_RO_sequence')

        def test_pulsed_spec_seq(self):
            p = sqo.pulsed_spec_seq(qubit_idx=0, spec_pulse_length=80e-9,
                                    platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'pulsed_spec_seq')

        def test_allxy(self):
            # Only test if it compiles
            p = sqo.AllXY(qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'AllXY')

        def test_T1(self):
            # Only test if it compiles
            p = sqo.T1(times=np.arange(0, 1e-6, 20e-9),
                       qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'T1')

        def test_T1_second_excited_state(self):
            # Only test if it compiles
            p = sqo.T1_second_excited_state(times=np.arange(0, 1e-6, 20e-9),
                                            qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'T1_2nd_exc')

        def test_Ramsey(self):
            # Only test if it compiles
            p = sqo.Ramsey(times=np.arange(0, 1e-6, 20e-9),
                           qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'Ramsey')

        def test_echo(self):
            # Only test if it compiles
            p = sqo.echo(times=np.arange(0, 2e-6, 40e-9),
                         qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'echo')

        def test_single_elt_on(self):
            p = sqo.single_elt_on(qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'single_elt_on')

        def test_flipping(self):
            number_of_flips = np.arange(10)
            p = sqo.flipping(qubit_idx=0, number_of_flips=number_of_flips,
                             platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'flipping')

        def test_butterfly(self):
            p = sqo.butterfly(qubit_idx=0, initialize=True,
                              platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'butterfly')

        def test_off_on(self):
            p = sqo.off_on(0, pulse_comb='off', initialize=False,
                           platf_cfg=self.config_fn)
            p = sqo.off_on(0, pulse_comb='on', initialize=False,
                           platf_cfg=self.config_fn)
            p = sqo.off_on(0, pulse_comb='off_on', initialize=False,
                           platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'off_on')

        def test_off_on_invalid_args(self):
            with self.assertRaises(ValueError):
                p = sqo.off_on(0, pulse_comb='of', initialize=False,
                               platf_cfg=self.config_fn)

        def test_idle_error_rate_seq(self):
            p = sqo.idle_error_rate_seq(
                [5, 8], ['0', '1', '+'], gate_duration_ns=20,
                echo=False, qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'idle_error_rate')

        def test_idle_error_rate_seq_invalid_args(self):
            with self.assertRaises(ValueError):
                p = sqo.idle_error_rate_seq(
                    [5, 8], ['0', '1', '+', '-'], gate_duration_ns=20,
                    echo=False, qubit_idx=0, platf_cfg=self.config_fn)

        def test_RTE(self):
            p = sqo.RTE(qubit_idx=0,
                        sequence_type='echo', net_gate='pi', feedback=True,
                        platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'RTE')

        def test_randomized_benchmarking(self):
            nr_cliffords = 2**np.arange(10)
            p = sqo.randomized_benchmarking(0, platf_cfg=self.config_fn,
                                            program_name='test_RB',
                                            nr_cliffords=nr_cliffords, nr_seeds=3)
            self.assertEqual(p.name, 'test_RB')

        def test_motzoi_XY(self):
            p = sqo.motzoi_XY(0, platf_cfg=self.config_fn,
                              program_name='motzoi_test')
            self.assertEqual(p.name, 'motzoi_test')

        def test_FluxTimingCalibration(self):
            p = sqo.FluxTimingCalibration(0, platf_cfg=self.config_fn,
                                          times=np.arange(0, 80e-9, 20e-9))
            self.assertEqual(p.name, 'FluxTimingCalibration')

        def test_FluxTimingCalibration_2q(self):
            p = sqo.FluxTimingCalibration_2q(q0=0, q1=2, platf_cfg=self.config_fn,
                                             buffer_time1=40e-9,
                                             times=np.arange(0, 80e-9, 20e-9))
            self.assertEqual(p.name, 'FluxTimingCalibration_2q')

        def test_fast_feedback_control(self):
            p = sqo.FastFeedbackControl(latency=200e-9,
                                        qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'FastFeedbackControl')

        def test_ef_rabi_seq(self):
            p = sqo.ef_rabi_seq(0, amps=np.linspace(0, 1, 11),
                                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'ef_rabi_seq')

    ##########################################################################
    # repeat same tests for Qutech Central Controller
    # NB: we just hijack the parent class to run the same tests
    # NB: requires OpenQL with CC backend support
    ##########################################################################

    if oqh.is_compatible_openql_version_cc():
        class Test_single_qubit_seqs_CC(Test_single_qubit_seqs_CCL):
            def setUp(self):
                curdir = os.path.dirname(__file__)
                self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
                output_dir = os.path.join(curdir, 'test_output_cc')
                ql.set_option('output_dir', output_dir)

            def test_RTE(self):
                pytest.skip("test_RTE() uses conditional gates, which are not implemented yet")

            def test_fast_feedback_control(self):
                pytest.skip("test_fast_feedback_control() uses conditional gates, which are not implemented yet")
    else:
        class Test_single_qubit_seqs_CC_incompatible_openql_version(unittest.TestCase):
            @unittest.skip('OpenQL version does not support CC')
            def test_fail(self):
                pass

# FIXME: disabled
# except ImportError as e:
#     class Test_single_qubit_seqs_CCL_import_error(unittest.TestCase):
#
#         @unittest.skip('Missing dependency - ' + str(e))
#         def test_fail(self):
#             pass
