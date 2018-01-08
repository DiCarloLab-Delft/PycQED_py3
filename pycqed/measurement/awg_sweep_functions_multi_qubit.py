import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sqs2
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
default_gauss_width = 10  # magic number should be removed,
# note magic number only used in old mathematica seqs


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
                 upload=True, verbose=False):
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
