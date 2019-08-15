import numpy as np
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as sqs2
import time


class n_qubit_seq_sweep(swf.Hard_Sweep):
    """
    Allows an arbitrary sequence.
    """

    def __init__(self, seq_len, #upload=True,
                 verbose=False, sweep_name=""):
        super().__init__()
        self.parameter_name = 'segment'
        self.unit = '#'
        self.sweep_points = np.arange(seq_len)
        self.verbose = verbose
        self.name = sweep_name

    def prepare(self, **kw):
        pass


class n_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars_list, upload=True,
                 preselection=False, parallel_pulses=False, RO_spacing=200e-9):
        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars_list = RO_pars_list
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        samples = 2**len(pulse_pars_list)
        if preselection:
            samples *= 2
        self.preselection = preselection
        self.parallel_pulses = parallel_pulses
        self.RO_spacing = RO_spacing
        self.name = '{}_qubit_off_on'.format(len(pulse_pars_list))

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_off_on(pulse_pars_list=self.pulse_pars_list,
                                RO_pars_list=self.RO_pars_list,
                                preselection=self.preselection,
                                parallel_pulses=self.parallel_pulses,
                                RO_spacing=self.RO_spacing)


class n_qubit_reset(swf.Hard_Sweep):
    def __init__(self, qubit_names, operation_dict, reset_cycle_time,
                 nr_resets=1, upload=True, codeword_indices=None):
        super().__init__()
        self.qubit_names = qubit_names
        self.operation_dict = operation_dict
        self.reset_cycle_time = reset_cycle_time
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.nr_resets = nr_resets
        samples = nr_resets*2**len(qubit_names)
        self.sweep_points = np.arange(samples)
        self.name = '{}_reset_x{}'.format(','.join(qubit_names), nr_resets)

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_reset(qubit_names=self.qubit_names,
                               operation_dict=self.operation_dict,
                               reset_cycle_time=self.reset_cycle_time,
                               nr_resets=self.nr_resets)


class n_qubit_Simultaneous_RB_sequence_lengths(swf.Soft_Sweep):
    def __init__(self, sweep_control='soft',
                 n_qubit_RB_sweepfunction=None):
        super().__init__()

        self.sweep_control = sweep_control
        self.n_qubit_RB_sweepfunction = n_qubit_RB_sweepfunction
        self.is_first_sweeppoint = True
        self.name = 'two_qubit_Simultaneous_RB_sequence_lengths'
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'


    def set_parameter(self, val):
        self.n_qubit_RB_sweepfunction.nr_cliffords_value = val
        self.n_qubit_RB_sweepfunction.upload = True
        self.n_qubit_RB_sweepfunction.prepare(
            upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False


class n_qubit_Simultaneous_RB_fixed_length(swf.Hard_Sweep):

    def __init__(self, qubit_names_list, operation_dict,
                 nr_seeds_array, #array
                 nr_cliffords_value, #int
                 # clifford_sequence_list=None,
                 gate_decomposition='HZ', interleaved_gate=None,
                 upload=True, return_seq=False, seq_name=None,
                 verbose=False, cal_points=False):

        super().__init__()
        self.qubit_names_list = qubit_names_list
        self.operation_dict = operation_dict
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.nr_seeds_array = nr_seeds_array
        # self.clifford_sequence_list = clifford_sequence_list
        self.seq_name = seq_name
        self.return_seq = return_seq
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate
        self.verbose = verbose
        self.cal_points = cal_points

        self.parameter_name = 'Nr of Seeds'
        self.unit = '#'
        self.name = 'two_qubit_Simultaneous_RB_fixed_length'

    def prepare(self, upload_all=True,  **kw):
        if self.upload:
            sqs2.n_qubit_simultaneous_randomized_benchmarking_seq(
                self.qubit_names_list, self.operation_dict,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=self.nr_seeds_array,
                # clifford_sequence_list=self.clifford_sequence_list,
                seq_name=self.seq_name,
                return_seq=self.return_seq,
                verbose=self.verbose,
                upload=True,
                upload_all=upload_all,
                cal_points=self.cal_points)


class n_qubit_Simultaneous_RB_fixed_seeds(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars, nr_cliffords_value,
                 clifford_sequence_list=None,
                 gate_decomposition='HZ', interleaved_gate=None,
                 upload=True, return_seq=False, seq_name=None,
                 verbose=False):

        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.clifford_sequence_list = clifford_sequence_list
        self.seq_name = seq_name
        self.return_seq = return_seq
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate
        self.verbose = verbose

        self.parameter_name = 'samples'
        self.unit = '#'
        self.name = 'n_qubit_Simultaneous_RB_fixed_seed'

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_simultaneous_randomized_benchmarking_seq(
                self.pulse_pars_list, self.RO_pars,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=np.array([1]),
                clifford_sequence_list=self.clifford_sequence_list,
                seq_name=self.seq_name,
                return_seq=self.return_seq,
                verbose=self.verbose)


class parity_correction(swf.Hard_Sweep):
    def __init__(self, q0n, q1n, q2n, operation_dict, CZ_pulses, 
                 feedback_delay, prep_sequence=None, reset=True, 
                 nr_parity_measurements=1, parity_op='ZZ',
                 tomography_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                 preselection=False, ro_spacing=1e-6, dd_scheme=None, 
                 nr_dd_pulses=4, skip_n_initial_parity_checks=0, skip_elem='RO',
                 upload=True, verbose=False):
        super().__init__()
        self.q0n = q0n
        self.q1n = q1n
        self.q2n = q2n
        self.operation_dict = operation_dict
        self.CZ_pulses = CZ_pulses
        self.feedback_delay = feedback_delay
        self.prep_sequence = prep_sequence
        self.reset = reset
        self.nr_parity_measurements = nr_parity_measurements
        self.parity_op = parity_op
        self.tomography_basis = tomography_basis
        self.preselection = preselection
        self.ro_spacing = ro_spacing
        self.dd_scheme = dd_scheme
        self.nr_dd_pulses = nr_dd_pulses
        self.skip_n_initial_parity_checks = skip_n_initial_parity_checks
        self.skip_elem = skip_elem
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.verbose = verbose
        self.name = 'two_qubit_parity'

    def prepare(self, **kw):
        if self.upload:
            if self.reset == True or self.reset == False:
                sqs2.parity_correction_seq(
                    self.q0n, self.q1n, self.q2n,
                    self.operation_dict, self.CZ_pulses,
                    feedback_delay=self.feedback_delay,
                    prep_sequence=self.prep_sequence,
                    parity_op=self.parity_op,
                    reset=self.reset,
                    tomography_basis=self.tomography_basis,
                    verbose=self.verbose,
                    preselection=self.preselection,
                    ro_spacing=self.ro_spacing,
                    dd_scheme=self.dd_scheme,
                    nr_dd_pulses=self.nr_dd_pulses,
                    nr_parity_measurements=self.nr_parity_measurements,
                    skip_n_initial_parity_checks=
                        self.skip_n_initial_parity_checks,
                    skip_elem=self.skip_elem,
                    )
            elif self.reset == 'simple':
                sqs2.parity_correction_no_reset_seq(
                    self.q0n, self.q1n, self.q2n,
                    self.operation_dict,
                    CZ_pulses=self.CZ_pulses,
                    feedback_delay=self.feedback_delay,
                    prep_sequence=self.prep_sequence,
                    tomography_basis=self.tomography_basis,
                    verbose=self.verbose,
                    preselection=self.preselection,
                    ro_spacing=self.ro_spacing,
                    dd_scheme=self.dd_scheme,
                    nr_dd_pulses=self.nr_dd_pulses,
                )


class Ramsey_add_pulse_swf(swf.Hard_Sweep):

    def __init__(self, measured_qubit_name,
                 pulsed_qubit_name, operation_dict,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.measured_qubit_name = measured_qubit_name
        self.pulsed_qubit_name = pulsed_qubit_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning

        self.name = 'Ramsey Add Pulse'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.Ramsey_add_pulse_seq(
                times=self.sweep_points,
                measured_qubit_name=self.measured_qubit_name,
                pulsed_qubit_name=self.pulsed_qubit_name,
                operation_dict=self.operation_dict,
                artificial_detuning=self.artificial_detuning,
                cal_points=self.cal_points)


class Ramsey_add_pulse_sweep_phase_swf(swf.Hard_Sweep):

    def __init__(self, measured_qubit_name,
                 pulsed_qubit_name, operation_dict,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.measured_qubit_name = measured_qubit_name
        self.pulsed_qubit_name = pulsed_qubit_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.cal_points = cal_points

        self.name = 'Ramsey Add Pulse Sweep Phases'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2. Ramsey_add_pulse_sweep_phase_seq(
                phases=self.sweep_points,
                measured_qubit_name=self.measured_qubit_name,
                pulsed_qubit_name=self.pulsed_qubit_name,
                operation_dict=self.operation_dict,
                cal_points=self.cal_points)


class calibrate_n_qubits(swf.Hard_Sweep):
    def __init__(self, sweep_params, sweep_points,
                 qubit_names, operation_dict, qb_names_DD=None,
                 cal_points=True, no_cal_points=4,
                 nr_echo_pulses=0, idx_DD_start=-1, UDD_scheme=True,
                 parameter_name='sample', unit='#',
                 upload=False, return_seq=False):

        super().__init__()
        self.name = 'n_Qubits_Time_Domain'
        self.sweep_params = sweep_params
        self.sweep_pts = sweep_points
        self.qubit_names = qubit_names
        self.qb_names_DD = qb_names_DD
        self.operation_dict = operation_dict
        self.cal_points = cal_points
        self.no_cal_points = no_cal_points
        self.nr_echo_pulses = nr_echo_pulses
        self.idx_DD_start = idx_DD_start
        self.UDD_scheme = UDD_scheme
        self.upload = upload
        self.return_seq = return_seq
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        if self.upload:
            sqs2.general_multi_qubit_seq(sweep_params=self.sweep_params,
                                         sweep_points=self.sweep_pts,
                                         qb_names=self.qubit_names,
                                         qb_names_DD=self.qb_names_DD,
                                         operation_dict=self.operation_dict,
                                         cal_points=self.cal_points,
                                         no_cal_points=self.no_cal_points,
                                         nr_echo_pulses=self.nr_echo_pulses,
                                         idx_DD_start=self.idx_DD_start,
                                         UDD_scheme=self.UDD_scheme,
                                         upload=self.upload,
                                         return_seq=self.return_seq)


class GST_experiment_sublist_swf(swf.Soft_Sweep):

    def __init__(self, hard_swf, pygsti_sublistOfExperiments):

        super().__init__()
        self.hard_swf = hard_swf
        self.pygsti_sublistOfExperiments = pygsti_sublistOfExperiments
        self.is_first_sweeppoint = True

        self.name = 'pyGSTi experiment sublist sweep'
        self.parameter_name = 'Points'
        self.unit = '#'

    def set_parameter(self, val, **kw):

        self.hard_swf.pygsti_listOfExperiments = \
            self.pygsti_sublistOfExperiments[val]
        self.hard_swf.upload = True
        self.hard_swf.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False


class GST_swf(swf.Hard_Sweep):

    def __init__(self, qb_names, pygsti_listOfExperiments,
                 operation_dict,
                 preselection=True, ro_spacing=1e-6, seq_name=None,
                 upload=True, return_seq=False, verbose=False):

        super().__init__()
        self.qb_names = qb_names
        self.pygsti_listOfExperiments = pygsti_listOfExperiments
        self.operation_dict = operation_dict
        self.preselection = preselection
        self.ro_spacing = ro_spacing
        self.seq_name = seq_name
        self.upload = upload
        self.return_seq = return_seq
        self.verbose = verbose

        self.name = 'pyGSTi experiment list sweep'
        self.parameter_name = 'Segment number'
        self.unit = '#'

    def prepare(self, upload_all=True, **kw):
        if self.upload:
            sqs2.pygsti_seq(
                qb_names=self.qb_names,
                pygsti_listOfExperiments=self.pygsti_listOfExperiments,
                operation_dict=self.operation_dict,
                preselection=self.preselection,
                ro_spacing=self.ro_spacing,
                seq_name=self.seq_name,
                upload=True,
                upload_all=upload_all,
                return_seq=self.return_seq,
                verbose=self.verbose)


class RO_dynamic_phase_swf(swf.Hard_Sweep):
    def __init__(self, qbp_name, qbr_names,
                 phases, operation_dict,
                 pulse_separation, init_state,
                 verbose=False, cal_points=True,
                 upload=True, return_seq=False):

        super().__init__()
        self.qbp_name = qbp_name
        self.qbr_names = qbr_names
        self.phases = phases
        self.operation_dict = operation_dict
        self.pulse_separation = pulse_separation
        self.init_state = init_state
        self.cal_points = cal_points
        self.upload = upload
        self.return_seq = return_seq
        self.verbose = verbose

        self.name = 'RO dynamic phase sweep'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            sqs2.ro_dynamic_phase_seq(
                qbp_name=self.qbp_name, qbr_names=self.qbr_names,
                phases=self.phases, operation_dict=self.operation_dict,
                pulse_separation=self.pulse_separation,
                init_state=self.init_state,
                verbose=self.verbose, cal_points=self.cal_points,
                upload=True, return_seq=self.return_seq)


class Measurement_Induced_Dephasing_Phase_Swf(swf.Hard_Sweep):
    def __init__(self, qbn_dephased, ro_op, operation_dict, readout_separation,
                 nr_readouts, cal_points=((-4,-3), (-2,-1)), upload=True):
        super().__init__()
        self.qbn_dephased = qbn_dephased
        self.ro_op = ro_op
        self.operation_dict = operation_dict
        self.readout_separation = readout_separation
        self.nr_readouts = nr_readouts
        self.cal_points = cal_points
        self.upload = upload

        self.name = 'Measurement induced dephasing phase'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            sqs2.measurement_induced_dephasing_seq(
                phases=self.sweep_points, 
                qbn_dephased=self.qbn_dephased, 
                ro_op=self.ro_op, 
                operation_dict=self.operation_dict, 
                readout_separation=self.readout_separation, 
                nr_readouts=self.nr_readouts, 
                cal_points=self.cal_points)


class Measurement_Induced_Dephasing_Amplitude_Swf(swf.Soft_Sweep):
    class DummyQubit:
        _params = ['f_RO', 'f_RO_mod', 'RO_pulse_length', 'RO_amp',
                   'ro_pulse_shape', 'ro_pulse_filter_sigma', 
                   'ro_pulse_nr_sigma', 'ro_CLEAR_delta_amp_segment', 
                   'ro_CLEAR_segment_length']
        
        def __init__(self, qb):
            self._values = {}
            self.name = qb.name
            self.UHFQC = qb.UHFQC
            self.readout_DC_LO = qb.readout_DC_LO
            self.readout_UC_LO = qb.readout_UC_LO
            for param in self._params:
                self.make_param(param, qb.get(param))

        def make_param(self, name, val):
            self._values[name] = val
            def accessor(v=None):
                if v is None:
                    return self._values[name]
                else:
                    self._values[name] = v
            setattr(self, name, accessor)
    
    def __init__(self, qb_dephased, qb_targeted, nr_readouts, 
                 multiplexed_pulse_fn, f_LO):
        super().__init__()
        self.qb_dephased = qb_dephased
        self.qb_targeted = qb_targeted
        self.nr_readouts = nr_readouts
        self.multiplexed_pulse_fn = multiplexed_pulse_fn
        self.f_LO = f_LO

        self.name = 'Measurement induced dephasing amplitude'
        self.parameter_name = 'amp'
        self.unit = 'max'

    def prepare(self, **kw):
        pass

    def set_parameter(self, val):
        qb_targeted_dummy = self.DummyQubit(self.qb_targeted)
        qb_targeted_dummy.RO_amp(val)
        readouts = [(qb_targeted_dummy,)]*self.nr_readouts + \
                   [(self.qb_dephased,)]
        time.sleep(0.1)
        self.multiplexed_pulse_fn(readouts, self.f_LO, upload=True)
        time.sleep(0.1)
