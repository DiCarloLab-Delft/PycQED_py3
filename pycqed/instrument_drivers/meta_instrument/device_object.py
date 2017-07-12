import numpy as np
import qcodes as qc
from copy import copy
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement.waveform_control_CC import multi_qubit_module_CC as mqmc
from pycqed.measurement.waveform_control_CC import multi_qubit_qasm_seqs as mqqs
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC


class DeviceObject(Instrument):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measuremnet labels
        self.add_parameter('qasm_config',
                           docstring='used for generating qumis instructions',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())
        self.add_parameter('qubits',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Lists(elt_validator=vals.Strings()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=InstrumentParameter)
        self.add_parameter('RO_acq_averages',
                           initial_value=1024,
                           vals=vals.Ints(),
                           parameter_class=ManualParameter)
        sc_docstr = (
            'Instrument responsible for controlling the waveform sequences. '
            'This is currently either a tek5014 AWG or a CBox.')
        self.add_parameter('seq_contr', label='Sequence controller',
                           docstring=sc_docstr,
                           parameter_class=InstrumentParameter)

        self._sequencer_config = {}
        self.delegate_attr_dicts += ['_sequencer_config']

        # Add buffer parameters for every pulse type
        pulse_types = ['MW', 'Flux', 'RO']
        self.add_parameter('sequencer_config',
                           get_cmd=self._get_sequencer_config,
                           vals=vals.Anything())
        for pt_a in pulse_types:
            for pt_b in pulse_types:
                self.add_parameter('Buffer_{}_{}'.format(pt_a, pt_b),
                                   unit='s',
                                   initial_value=0,
                                   parameter_class=ManualParameter)
                self.add_sequencer_config_param(
                    self.parameters['Buffer_{}_{}'.format(pt_a, pt_b)])

        self.add_parameter(
            'RO_fixed_point', unit='s',
            initial_value=1e-6,
            docstring=('The Tektronix sequencer shifts all elements in a ' +
                       'sequence such that the first RO encountered in each ' +
                       'element is aligned with the fixpoint.\nIf the ' +
                       'sequence length is longer than the fixpoint, ' +
                       'it will shift such that the first RO aligns with a ' +
                       'multiple of the fixpoint.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e-9, 500e-6))
        self.add_sequencer_config_param(self.RO_fixed_point)
        self.add_parameter(
            'Flux_comp_dead_time', unit='s',
            initial_value=3e-6,
            docstring=('Used to determine the wait time between the end of a' +
                       'sequence and the beginning of Flux compensation' +
                       ' pulses.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e-9, 500e-6))
        self.add_sequencer_config_param(self.Flux_comp_dead_time)

    def get_idn(self):
        return self.name

    def _get_sequencer_config(self):
        seq_cfg = {}
        for par_name, config_par in self._sequencer_config.items():
            seq_cfg[par_name] = config_par()
        return seq_cfg

    def add_sequencer_config_param(self, seq_config_par):
        """
        """
        name = seq_config_par.name
        self._sequencer_config[name] = seq_config_par
        return name

    def get_operation_dict(self, operation_dict={}):
        # uses the private qubits list
        for qname in self.qubits():
            q = self.find_instrument(qname)
            q.get_operation_dict(operation_dict)
        operation_dict['sequencer_config'] = self.sequencer_config()
        return operation_dict

    def calibrate_mux_RO(self):
        raise NotImplementedError

    def prepare_for_timedomain(self):
        """
        Calls the prepare for timedomain on each of the constituent qubits.
        """
        for q in self.qubits():
            q_obj = self.find_instrument(q)
            q_obj.prepare_for_timedomain()


class TwoQubitDevice(DeviceObject):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter(
            'RO_LutManMan',
            docstring='Used for generating multiplexed RO pulses',
            parameter_class=InstrumentParameter)

        # N.B. for now the "central_controller" can be a CBox_v3
        self.add_parameter('central_controller',
                           parameter_class=InstrumentParameter)

        self.add_parameter(
            'RO_LO_freq', unit='Hz',
            docstring=('Frequency of the common LO for all RO pulses.'),
            parameter_class=ManualParameter)

    def calibrate_mux_RO(self, calibrate_optimal_weights=True,
                         verify_optimal_weights=False,
                         update: bool=True,
                         update_threshold: bool=True):

        RO_LMM = self.RO_LutManMan.get_instr()
        UHFQC = self.acquisition_instrument.get_instr()

        q0 = self.find_instrument(self.qubits()[0])
        q1 = self.find_instrument(self.qubits()[1])

        # this is to ensure that the cross talk correction matrix coefficients
        # can rey on the weight function indices being consistent
        q0.RO_acq_weight_function_I(0)
        q1.RO_acq_weight_function_I(1)

        # Set the modulation frequencies so that the LO's match
        q0.f_RO_mod(q0.f_RO() - self.RO_LO_freq())
        q1.f_RO_mod(q1.f_RO() - self.RO_LO_freq())

        # Update parameters
        q0.RO_pulse_type('IQmod_multiplexed_UHFQC')
        q1.RO_pulse_type('IQmod_multiplexed_UHFQC')

        q1.prepare_for_timedomain()
        q0.prepare_for_timedomain()
        RO_LMM.acquisition_delay(q0.RO_acq_marker_delay())

        # Generate multiplexed pulse
        multiplexed_wave = [[q0.RO_LutMan(), 'M_square'],
                            [q1.RO_LutMan(), 'M_square']]
        RO_LMM.generate_multiplexed_pulse(multiplexed_wave)
        RO_LMM.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')

        # Important that this happens before calibrating the weights
        UHFQC.quex_trans_offset_weightfunction_0(0)
        UHFQC.quex_trans_offset_weightfunction_1(0)
        UHFQC.upload_transformation_matrix([[1, 0], [0, 1]])

        if calibrate_optimal_weights:
            q0.calibrate_optimal_weights(
                analyze=True, verify=verify_optimal_weights)
            q1.calibrate_optimal_weights(
                analyze=True, verify=verify_optimal_weights)

        mqmc.measure_two_qubit_ssro(self, q0.name, q1.name, no_scaling=True,
                                    result_logging_mode='lin_trans')

        res_dict = mra.two_qubit_ssro_fidelity(
            label='{}_{}'.format(q0.name, q1.name),
            qubit_labels=[q0.name, q1.name])
        V_offset_cor = res_dict['V_offset_cor']

        # weights 0 and 1 are the correct indices because I set the numbering
        # at the start of this calibration script.
        UHFQC.quex_trans_offset_weightfunction_0(V_offset_cor[0])
        UHFQC.quex_trans_offset_weightfunction_1(V_offset_cor[1])

        # Does not work because axes are not normalized
        UHFQC.upload_transformation_matrix(res_dict['mu_matrix_inv'])

        a = self.check_mux_RO(update=update, update_threshold=update_threshold)
        return a

    def check_mux_RO(self, update: bool=True,
                         update_threshold: bool=True):
        q0_name, q1_name, = self.qubits()

        q0 = self.find_instrument(q0_name)
        q1 = self.find_instrument(q1_name)

        mqmc.measure_two_qubit_ssro(self, q0_name, q1_name, no_scaling=True,
                                    result_logging_mode='lin_trans')

        a = mra.two_qubit_ssro_fidelity(
            label='{}_{}'.format(q0.name, q1.name),
            qubit_labels=[q0.name, q1.name])

        if update:
            q0.F_ssro(a['Fa_q0'])
            q0.F_discr(a['Fd_q0'])
            q1.F_ssro(a['Fa_q1'])
            q1.F_discr(a['Fd_q1'])

        if update_threshold:
            # do not use V_th_corr as this is measured from data that already
            # includes a correction matrix
            thres = a['V_th']

            # correction for the offset (that is only applied in software)
            # happens in the qubits objects in the prep for TD where the
            # threshold is set in the UFHQC.
            q0.RO_threshold(thres[0])
            q1.RO_threshold(thres[1])

        return a

    def get_correlation_detector(self):
        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnames[1])

        w0 = q0.RO_acq_weight_function_I()
        w1 = q1.RO_acq_weight_function_I()

        d = det.UHFQC_correlation_detector(
            UHFQC=self.acquisition_instrument.get_instr(),
            thresholding=True,
            AWG=self.central_controller.get_instr(),
            channels=[w0, w1],
            correlations=[(w0, w1)],
            nr_averages=self.RO_acq_averages(),
            integration_length=q0.RO_acq_integration_length())
        return d

    def get_integration_logging_detector(self):
        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnamse[1])

        w0 = q0.RO_acq_weight_function_I()
        w1 = q1.RO_acq_weight_function_I()

        d = det.UHFQC_integration_logging_det(
            UHFQC=self.acquisition_instrument.get_instr(),
            channels=[w0, w1],
            result_logging_mode='digitized',
            integratioin_length=q0.RO_acq_integration_length())
        return d

    def measure_two_qubit_AllXY(self, sequence_type='simultaneous', MC=None):
        if MC is None:
            MC = qc.station.components['MC']

        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnames[1])
        self.prepare_for_timedomain()

        double_points = True
        AllXY = mqqs.two_qubit_AllXY(q0.name, q1.name,
                                     RO_target=q0.name, # shold be 'all'
                                     sequence_type=sequence_type,
                                     replace_q1_pulses_X180=False,
                                     double_points=double_points)

        s = swf.QASM_Sweep_v2(qasm_fn=AllXY.name,
                              config=self.qasm_config(),
                              CBox=self.central_controller.get_instr(),
                              verbosity_level=1)

        d = self.get_correlation_detector()

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(21*(1+double_points)))
        MC.set_detector_function(d)
        MC.run('AllXY_{}_{}'.format(q0.name, q1.name))
        ma.MeasurementAnalysis()

    def measure_two_qubit_GST(self, q0, q1,
                              max_lengths=[2**i for i in range(10)],
                              repetitions_per_point=500,
                              MC=None):
        '''
        Measure gate set tomography for this qubit. The gateset that is used
        is saved in
        pycqed/measurement/gate_set_tomography/Gateset_5_primitives_GST.txt,
        and corresponding germs and fiducials can be found in the same folder.

        Args:
            q0, q1 (Instr):
                Qubits on which the experiment is run.
            max_lengths (list):
                List of maximum sequence length (via germ repeats) for each
                GST iteration. The largest maximum length should be roughly
                the number of gates that can be done on a qubit before it
                completely depolarizes.
            repetitions_per_point (int):
                Number of times each experiment is repeated in total.
            MC (Instrument):
                Measurement control instrument that should be used for the
                experiment. Default (None) uses self.MC.
        '''
        # TODO: max_acq_points: hard coded?
        max_acq_points = 4094

        if MC is None:
            MC = self.MC.get_instr()

        # Load the gate set, germs, and fiducials.
        gs_target = std2Q_XYCPHASE.gs_target.copy()
        prepFids = std2Q_XYCPHASE.prepStrs
        measFids = std2Q_XYCPHASE.effectStrs
        germs = std2Q_XY.germs

        # gate_dict maps GST gate labels to QASM operations
        gate_dict = {
            'Gix': 'X180 {}'.format(q0.name),
            'Giy': 'Y180 {}'.format(q0.name),
            'Gxi': 'X180 {}'.format(q1.name),
            'Gyi': 'Y180 {}'.format(q1.name),
            'Gcphase': 'CZ {} {}'.format(q0.name, q1.name)
        }

        # Create the experiment list, translate it to QASM, and generate the
        # QASM file(s).
        try:
            max_instr = self.seq_contr.get_instr()._get_instr_mem_size()
        except AttributeError as e:
            max_instr = 2**15
            logging.warning(str(e) + '\nUsing default ({})'.format(max_instr))

        raw_exp_list = pygsti.construction.make_lsgst_experiment_list(
            gs_target, prepFids, measFids, germs, maxLengths)
        exp_list = gstCC.get_experiments_from_list(raw_exp_list, gate_dict)
        qasm_files, exp_nums = gstCC.generateQASM(
            filename='GST_2Q_{}_{}'.format(q0.name, q1.name),
            exp_list=exp_list,
            qubit_labels=[q0.name, q1.name],
            max_instructions=max_instr,
            max_exp_per_file=max_acq_points)

        # We want to measure every experiment (i.e. gate sequence) x times.
        # Also, we want low frequency noise in our system to affect each
        # experiment the same, so we don't want to do all repetitions of the
        # first experiment, then the second, etc., but rather go through all
        # experiments, then repeat. If we have more than one QASM file, this
        # would be slower, so we compromise and say we want a minimum of 5
        # (soft) repetitions of the whole sequence and adjust the hard
        # repetitions accordingly.
        # The acquisition device can acquire a maximum of m = max_acq_points
        # in one go.
        # A QASM file contains i experiments (i can be different for different
        # QASM files.
        # Take the largest i -> can measure floor(m/i) = l repetitions of this
        # QASM file. => need r = ceil(x/l) soft repetitions of that file.
        # => set max(r,5) as the number of soft repetitions for all files.
        # => set ceil(x/r) as the number of hard repetitions for each file
        # (even if for some files we could do more).
        max_exp_num = max(exp_nums)  # i
        soft_repetitions = int(np.ceil(
            repetitions_per_point / np.floor(max_acq_points / max_exp_num)))
        if soft_repetitions < min_soft_repetitions:
            soft_repetitions = 5
            hard_repetitions = int(np.ceil(repetitions_per_point /
                                       soft_repetitions))

        self.prepare_for_timedomain()

        d = self.get_integration_logging_detector()
        s = swf.Multi_QASM_Sweep(
            exp_num_list=exp_nums,
            hard_repetitions=hard_repetitions,
            soft_repetitions=soft_repetitions,
            qasm_list=[q.name for q in qasm_files],
            config=self.qasm_config(),
            detector=d,
            CBox=self.seq_contr.get_instr(),
            parameter_name='GST sequence',
            unit=None)
        total_exp_nr = np.sum(exp_nums) * hard_repetitions * soft_repetitions

        if d.result_logging_mode != 'digitized':
            logging.warning('GST is intended for use with digitized detector.'
                            ' Analysis will fail otherwise.')

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(total_exp_nr))
        MC.set_detector_function(d)
        MC.run('GST_2Q')

        # Analysis
        ma.GST_Analysis(exp_num_list=exp_nums,
                        hard_repetitions=hard_repetitions,
                        soft_repetitions=soft_repetitions,
                        exp_list=[g.str for g in raw_exp_list],
                        gs_target=gs_target,
                        prep_fiducials=prepFids,
                        meas_fiducials=measFids,
                        germs=germs,
                        max_lengths=max_lengths)
