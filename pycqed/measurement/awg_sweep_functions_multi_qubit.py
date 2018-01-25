import numpy as np
import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as sqs2

class two_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'two_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                  q1_pulse_pars=self.q1_pulse_pars,
                                  RO_pars=self.RO_pars,
                                  return_seq=self.return_seq,
                                  verbose=self.verbose)


class three_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'three_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.three_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                    q1_pulse_pars=self.q1_pulse_pars,
                                    q2_pulse_pars=self.q2_pulse_pars,
                                    RO_pars=self.RO_pars,
                                    return_seq=self.return_seq,
                                    verbose=self.verbose)


class four_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars,  q3_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.q3_pulse_pars = q3_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'four_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.four_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                   q1_pulse_pars=self.q1_pulse_pars,
                                   q2_pulse_pars=self.q2_pulse_pars,
                                   q3_pulse_pars=self.q3_pulse_pars,
                                   RO_pars=self.RO_pars,
                                   return_seq=self.return_seq,
                                   verbose=self.verbose)


class five_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars,
                 q3_pulse_pars, q4_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.q3_pulse_pars = q3_pulse_pars
        self.q4_pulse_pars = q4_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'five_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.four_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                   q1_pulse_pars=self.q1_pulse_pars,
                                   q2_pulse_pars=self.q2_pulse_pars,
                                   q3_pulse_pars=self.q3_pulse_pars,
                                   q4_pulse_pars=self.q4_pulse_pars,
                                   RO_pars=self.RO_pars,
                                   return_seq=self.return_seq,
                                   verbose=self.verbose)

class n_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars, upload=True,
                 preselection=False, parallel_pulses=False,
                 verbose=False, RO_spacing=200e-9):
        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        samples = 2**len(pulse_pars_list)
        if preselection:
            samples *= 2
        self.sweep_points = np.arange(samples)
        self.verbose = verbose
        self.preselection = preselection
        self.parallel_pulses = parallel_pulses
        self.RO_spacing = RO_spacing
        self.name = '{}_qubit_off_on'.format(len(pulse_pars_list))

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_off_on(pulse_pars_list=self.pulse_pars_list,
                                RO_pars=self.RO_pars,
                                preselection=self.preselection,
                                parallel_pulses=self.parallel_pulses,
                                RO_spacing=self.RO_spacing,
                                verbose=self.verbose)

class n_qubit_reset(swf.Hard_Sweep):
    def __init__(self, pulse_pars_list, RO_pars, feedback_delay, nr_resets=1,
                 upload=True, verbose=False, codeword_indices=None):
        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.feedback_delay = feedback_delay
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.nr_resets = nr_resets
        samples = nr_resets*2**len(pulse_pars_list)
        self.sweep_points = np.arange(samples)
        self.verbose = verbose
        self.name = '{}_qubit_{}_reset'.format(len(pulse_pars_list), nr_resets)

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_reset(pulse_pars_list=self.pulse_pars_list,
                               RO_pars=self.RO_pars,
                               feedback_delay=self.feedback_delay,
                               nr_resets=self.nr_resets,
                               verbose=self.verbose)


class n_qubit_Simultaneous_Randomized_Benchmarking_nr_cliffords(swf.Soft_Sweep):
    def __init__(self, sweep_control='soft',
                 n_qubit_RB_sweepfunction=None):
        super().__init__()

        self.sweep_control = sweep_control
        # self.sweep_points = nr_cliffords
        self.n_qubit_RB_sweepfunction = n_qubit_RB_sweepfunction
        self.name = 'n_qubit_Simultaneous_Randomized_Benchmarking_nr_cliffords'
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'


    def set_parameter(self, val):
        self.n_qubit_RB_sweepfunction.nr_cliffords_value = val
        self.n_qubit_RB_sweepfunction.upload = True
        self.n_qubit_RB_sweepfunction.prepare()


class n_qubit_Simultaneous_Randomized_Benchmarking_one_length(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars, nr_cliffords_value, #int
                 gate_decomposition='HZ', interleaved_gate=None,
                 upload=False, return_seq=False, seq_name=None,
                 CxC_RB=True, idx_for_RB=0,
                 verbose=False):

        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.CxC_RB = CxC_RB
        self.seq_name = seq_name
        self.return_seq = return_seq
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate
        self.idx_for_RB = idx_for_RB
        self.verbose = verbose

        self.parameter_name = 'Nr of Seeds'
        self.unit = '#'
        self.name = 'n_qubit_Simultaneous_Randomized_Benchmarking_one_length'

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_simultaneous_randomized_benchmarking_seq(
                self.pulse_pars_list, self.RO_pars,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=self.sweep_points,
                CxC_RB=self.CxC_RB,
                idx_for_RB=self.idx_for_RB,
                seq_name=self.seq_name,
                return_seq=self.return_seq,
                verbose=self.verbose)


class two_qubit_AllXY(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, RO_pars, upload=True,
                 return_seq=False, verbose=False, X_mid=False, simultaneous=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(42*2)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'two_qubit_AllXY'
        self.X_mid = X_mid
        self.simultaneous = simultaneous

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_AllXY(q0_pulse_pars=self.q0_pulse_pars,
                                  q1_pulse_pars=self.q1_pulse_pars,
                                  RO_pars=self.RO_pars,
                                  double_points=True,
                                  return_seq=self.return_seq,
                                  verbose=self.verbose,
                                  X_mid=self.X_mid,
                                  simultaneous=self.simultaneous)


class tomo_Bell(swf.Hard_Sweep):

    def __init__(self, bell_state, qb_c, qb_t, RO_pars, num_flux_pulses=0, upload=True,
                 distortion_dict=None, CZ_disabled=False,  cal_points_with_flux_pulses=True,
                 verbose=False, return_seq=False):
        super().__init__()
        self.bell_state = bell_state
        self.qb_c = qb_c
        self.qb_t = qb_t
        self.RO_pars = RO_pars
        self.num_flux_pulses = num_flux_pulses
        self.upload = upload
        self.distortion_dict = distortion_dict
        self.CZ_disabled = CZ_disabled
        self.cal_points_with_flux_pulses = cal_points_with_flux_pulses
        self.parameter_name = 'sample'
        self.unit = '#'
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'tomo_Bell'

    def prepare(self, **kw):
        if self.upload:
            sqs2.BellTomo.two_qubit_tomo_bell_qudev(
                bell_state=self.bell_state,
                qb_c=self.qb_c, qb_t=self.qb_t,
                RO_pars=self.RO_pars,
                num_flux_pulses=self.num_flux_pulses,
                distortion_dict=self.distortion_dict,
                CZ_disabled=self.CZ_disabled,
                cal_points_with_flux_pulses=self.cal_points_with_flux_pulses,
                verbose=self.verbose,
                upload=self.upload)
