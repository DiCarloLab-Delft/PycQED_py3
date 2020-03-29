import os
import unittest
import pytest
import numpy as np

try:
    from pycqed.measurement.openql_experiments import multi_qubit_oql as mqo
    from pycqed.measurement.openql_experiments.generate_CCL_cfg import  \
        generate_config
    from openql import openql as ql

    class Test_multi_qubit_oql(unittest.TestCase):
        def setUp(self):
            curdir = os.path.dirname(__file__)
            self.config_fn = os.path.join(curdir, 'test_cfg_CCL.json')
            output_dir = os.path.join(curdir, 'test_output')
            ql.set_option('output_dir', output_dir)

        def test_single_flux_pulse_seq(self):
            # N.B. edge 0,2 is still illegal...
            p = mqo.single_flux_pulse_seq([2, 0], platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'single_flux_pulse_seq')

        def test_flux_staircase_seq(self):
            p = mqo.flux_staircase_seq(platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'flux_staircase_seq')

        def test_multi_qubit_off_on(self):
            p = mqo.multi_qubit_off_on(qubits=[0, 1, 4],
                                       initialize=True,
                                       second_excited_state=True,
                                       platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'multi_qubit_off_on')

        def test_Ramsey_msmt_induced_dephasing(self):
            p = mqo.Ramsey_msmt_induced_dephasing([3, 5], angles=[20, 40, 80],
                                                  platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'Ramsey_msmt_induced_dephasing')

        def test_echo_msmt_induced_dephasing(self):
            p = mqo.echo_msmt_induced_dephasing([3, 5], angles=[20, 40, 80],
                                                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'echo_msmt_induced_dephasing')

        def test_two_qubit_off_on(self):
            p = mqo.two_qubit_off_on(3, 5, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'two_qubit_off_on')

        def test_two_qubit_tomo_cardinal(self):
            p = mqo.two_qubit_tomo_cardinal(cardinal=3,
                                            q0=0, q1=1, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'two_qubit_tomo_cardinal')

        def test_two_qubit_AllXY(self):
            p = mqo.two_qubit_AllXY(q0=0, q1=1, platf_cfg=self.config_fn,
                                    sequence_type='sequential',
                                    replace_q1_pulses_X180=False,
                                    double_points=True)
            self.assertEqual(p.name, 'two_qubit_AllXY')
            p = mqo.two_qubit_AllXY(q0=0, q1=1, platf_cfg=self.config_fn,
                                    sequence_type='simultaneous',
                                    replace_q1_pulses_X180=False,
                                    double_points=True)
            self.assertEqual(p.name, 'two_qubit_AllXY')

        def test_residual_coupling_sequence(self):
            p = mqo.residual_coupling_sequence(
                times=np.arange(0, 100e-9, 20e-9),
                q0=0, q_spectator_idx=[1], spectator_state='1', platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'residual_coupling_sequence')

        def test_Cryoscope(self):
            p = mqo.Cryoscope(
                qubit_idx=0, platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'Cryoscope')

        def test_CryoscopeGoogle(self):
            p = mqo.CryoscopeGoogle(
                qubit_idx=0, buffer_time1=50e-9,
                times=np.arange(0, 100e-9, 20e-9),
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'CryoscopeGoogle')

        def test_Chevron_hack(self):
            p = mqo.Chevron_hack(
                qubit_idx=0, qubit_idx_spec=2,
                buffer_time=0, buffer_time2=0,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'Chevron_hack')

        def test_Chevron(self):
            for target_qubit_sequence in ['ramsey', 'excited', 'ground']:
                p = mqo.Chevron(
                    qubit_idx=0,
                    qubit_idx_spec=2,
                    qubit_idx_park=None,
                    buffer_time=0, buffer_time2=0, flux_cw=2,
                    target_qubit_sequence=target_qubit_sequence,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'Chevron')

        def test_two_qubit_ramsey(self):
            for target_qubit_sequence in ['ramsey', 'excited', 'ground']:
                p = mqo.two_qubit_ramsey(
                    qubit_idx=0,
                    times=np.arange(0, 100e-9, 20e-9),
                    qubit_idx_spec=2,
                    target_qubit_sequence=target_qubit_sequence,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'two_qubit_ramsey')


        def test_two_qubit_tomo_bell(self):
            for bell_state in [0, 1, 2, 3]:
                p = mqo.two_qubit_tomo_bell(
                    q0=0,
                    q1=3,
                    bell_state=bell_state,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'two_qubit_tomo_bell_3_0')


        def test_two_qubit_tomo_bell_by_waiting(self):
            for bell_state in [0, 1, 2, 3]:
                p = mqo.two_qubit_tomo_bell_by_waiting(
                    q0=0,
                    q1=2,
                    bell_state=bell_state,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'two_qubit_tomo_bell_by_waiting')

        def test_two_qubit_DJ(self):
            p = mqo.two_qubit_DJ(
                q0=0,
                q1=2,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'two_qubit_DJ')

        def test_two_qubit_parity_check(self):
            for initialization_msmt in [False, True]:
                p = mqo.two_qubit_parity_check(
                    qD0=0,
                    qD1=0,
                    qA=2,
                    initialization_msmt=initialization_msmt,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'two_qubit_parity_check')

        def test_conditional_oscillation_seq(self):
            # N.B. this does not check the many different variants of this
            # function
            p = mqo.conditional_oscillation_seq(
                q0=0,
                q1=3,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'conditional_oscillation_seq')


        def test_grovers_two_qubit_all_inputs(self):
            p = mqo.grovers_two_qubit_all_inputs(
                q0=0,
                q1=2,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'grovers_two_qubit_all_inputs')


        def test_grovers_tomography(self):
            for omega in range(4):
                p = mqo.grovers_tomography(
                    q0=0,
                    q1=2,
                    omega=omega,
                    platf_cfg=self.config_fn)
                self.assertEqual(p.name, 'grovers_tomography')

        def test_CZ_poisoned_purity_seq(self):
            p = mqo.CZ_poisoned_purity_seq(
                q0=0,
                q1=2,
                nr_of_repeated_gates=5,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'CZ_poisoned_purity_seq')

        def test_Chevron_first_manifold(self):
            p = mqo.Chevron_first_manifold(
                qubit_idx=0,
                qubit_idx_spec=2,
                buffer_time=20e-9, buffer_time2=40e-9, flux_cw=1,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'Chevron_first_manifold')

        def test_partial_tomography_cardinal(self):
            p = mqo.partial_tomography_cardinal(
                q0=0,
                q1=2,
                cardinal=3,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'partial_tomography_cardinal')

        def test_two_qubit_VQE(self):
            p = mqo.two_qubit_VQE(
                q0=0,
                q1=2,
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'two_qubit_VQE')

        def test_sliding_flux_pulses_seq(self):
            p = mqo.sliding_flux_pulses_seq(
                qubits=[0, 2],
                platf_cfg=self.config_fn)
            self.assertEqual(p.name, 'sliding_flux_pulses_seq')


    """
        Author:             Wouter Vlothuizen, QuTech
        Purpose:            multi qubit OpenQL tests for Qutech Central Controller
        Notes:              requires OpenQL with CC backend support
    """
    # import test_multi_qubit_oql as parent  # rename to stop pytest from running tests directly

    # NB: we just hijack the parent class to run the same tests

    # FIXME: This only works with Wouters custom OpenQL.
    # Need a better check for this
    if ql.get_version() > '0.7.0':
        class Test_multi_qubit_oql_CC(Test_multi_qubit_oql):
            def setUp(self):
                curdir = os.path.dirname(__file__)
                self.config_fn = os.path.join(curdir, 'test_cfg_cc.json')
                output_dir = os.path.join(curdir, 'test_output_cc')
                ql.set_option('output_dir', output_dir)

            def test_multi_qubit_off_on(self):
                pytest.skip("test_multi_qubit_off_on() gives signalconflict (FIXME)")
    else:
        class Test_multi_qubit_oql_CC(unittest.TestCase):
                @unittest.skip('OpenQL version does not support CC')
                def test_fail(self):
                    pass

except ImportError as e:

    class Test_multi_qubit_oql(unittest.TestCase):

        @unittest.skip('Missing dependency - ' + str(e))
        def test_fail(self):
            pass

