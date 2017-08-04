import numpy as np
import qcodes as qc
import os
import logging
import pycqed.measurement.waveform_control_CC.qasm_compiler as qcx
from copy import deepcopy
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement.waveform_control_CC import multi_qubit_module_CC\
    as mqmc
from pycqed.measurement.waveform_control_CC import multi_qubit_qasm_seqs\
    as mqqs
import pycqed.measurement.waveform_control_CC.single_qubit_qasm_seqs as sqqs
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC
import pygsti
from pycqed.utilities.general import gen_sweep_pts
from pycqed_scripts.experiments.Starmon_1702.clean_scripts.functions import \
    CZ_cost_Z_amp


class DeviceObject(Instrument):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measurement labels
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
        RO_LMM = self.RO_LutManMan.get_instr()

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

    def prepare_for_fluxing(self, reset: bool=True):
        '''
        Calls the prepare for timedomain on each of the constituent qubits.
        The reset option is passed to the first qubit that is called. For the
        rest of the qubit, reset is always disabled, in order not to overwrite
        the settings of the first qubit.
        '''
        for q in self.qubits():
            q_obj = self.find_instrument(q)
            q_obj.prepare_for_fluxing(reset=reset)
            reset = False


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

        q0 = self.find_instrument(self.qubits()[0])
        q1 = self.find_instrument(self.qubits()[1])
        UHFQC = self.acquisition_instrument.get_instr()
        self.prepare_for_timedomain()

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

    def check_mux_RO(self, update: bool=True, update_threshold: bool=True):
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
            thres = a['V_th_d']

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
        q1 = self.find_instrument(qnames[1])

        w0 = q0.RO_acq_weight_function_I()
        w1 = q1.RO_acq_weight_function_I()

        d = det.UHFQC_integration_logging_det(
            UHFQC=self.acquisition_instrument.get_instr(),
            AWG=self.central_controller.get_instr(),
            channels=[w0, w1],
            result_logging_mode='lin_trans',
            integratioin_length=q0.RO_acq_integration_length())
        return d

    def get_integrated_average_detector(self, seg_per_point: int=1):
        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnames[1])

        w0 = q0.RO_acq_weight_function_I()
        w1 = q1.RO_acq_weight_function_I()

        d = det.UHFQC_integrated_average_detector(
            UHFQC=self.acquisition_instrument.get_instr(),
            AWG=self.central_controller.get_instr(),
            channels=[w0, w1],
            nr_averages=self.RO_acq_averages(),
            real_imag=True, single_int_avg=True,
            result_logging_mode='digitized',
            integration_length=q0.RO_acq_integration_length(),
            seg_per_point=seg_per_point)
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
                                     RO_target=q0.name,  # shold be 'all'
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

    def measure_two_qubit_GST(self,
                              max_germ_pow: int=10,
                              repetitions_per_point: int=500,
                              min_soft_repetitions: int=5,
                              MC=None,
                              analyze: bool=False):
        '''
        Measure gate set tomography for this qubit. The gateset that is used
        is saved in
        pycqed/measurement/gate_set_tomography/Gateset_5_primitives_GST.txt,
        and corresponding germs and fiducials can be found in the same folder.

        Args:
            max_germ_pow (int):
                Largest power of 2 used to set germ lengths.
            repetitions_per_point (int):
                Number of times each experiment is repeated in total.
            min_soft_repetitions (int):
                Minimum number of soft repetitions that should be done
                (repetitions of the whole sequene).
            MC (Instrument):
                Measurement control instrument that should be used for the
                experiment. Default (None) uses self.MC.
        '''
        # TODO: max_acq_points: hard coded?
        max_acq_points = 4094

        if MC is None:
            MC = qc.station.components['MC']

        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnames[1])
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        # Load the gate set, germs, and fiducials.
        gstPath = os.path.dirname(gstCC.__file__)
        gs_target_path = os.path.join(gstPath, 'Gateset_2Q_XYCphase.txt')
        prep_fids_path = os.path.join(gstPath,
                                      'Prep_Fiducials_2Q_XYCphase.txt')
        meas_fids_path = os.path.join(gstPath,
                                      'Meas_Fiducials_2Q_XYCphase.txt')
        germs_path = os.path.join(gstPath, 'Germs_2Q_XYCphase.txt')

        gs_target = pygsti.io.load_gateset(gs_target_path)
        prep_fids = pygsti.io.load_gatestring_list(prep_fids_path)
        meas_fids = pygsti.io.load_gatestring_list(meas_fids_path)
        germs = pygsti.io.load_gatestring_list(germs_path)

        max_lengths = [2**i for i in range(max_germ_pow + 1)]

        # gate_dict maps GST gate labels to QASM operations
        gate_dict = {
            'Gix': 'X90 {}'.format(q0.name),
            'Giy': 'Y90 {}'.format(q0.name),
            'Gxi': 'X90 {}'.format(q1.name),
            'Gyi': 'Y90 {}'.format(q1.name),
            'Gcphase': 'CZ {} {}'.format(q0.name, q1.name),
            'RO': 'RO {} | RO {}'.format(q0.name, q1.name)
        }

        # Create the experiment list, translate it to QASM, and generate the
        # QASM file(s).
        try:
            max_instr = \
                self.central_controller.get_instr().instr_mem_size()
        except AttributeError as e:
            max_instr = 2**15
            logging.warning(str(e) + '\nUsing default ({})'.format(max_instr))

        raw_exp_list = pygsti.construction.make_lsgst_experiment_list(
            gs_target.gates.keys(), prep_fids, meas_fids, germs, max_lengths)
        exp_list = gstCC.get_experiments_from_list(raw_exp_list, gate_dict)
        qasm_files, exp_per_file, exp_last_file = gstCC.generate_QASM(
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
        # A QASM file contains i experiments.
        # -> can measure floor(m/i) = l repetitions of every file
        # => need r = ceil(x/l) soft repetitions.
        # => set max(r, 5) as the number of soft repetitions.
        # => set ceil(x/r) as the number of hard repetitions.
        soft_repetitions = int(np.ceil(
            repetitions_per_point / np.floor(max_acq_points / exp_per_file)))
        if soft_repetitions < min_soft_repetitions:
            soft_repetitions = min_soft_repetitions
        hard_repetitions = int(np.ceil(repetitions_per_point /
                                       soft_repetitions))

        d = self.get_integration_logging_detector()
        s = swf.Multi_QASM_Sweep(
            exp_per_file=exp_per_file,
            hard_repetitions=hard_repetitions,
            soft_repetitions=soft_repetitions,
            qasm_list=[q.name for q in qasm_files],
            config=self.qasm_config(),
            detector=d,
            CBox=self.central_controller.get_instr(),
            parameter_name='GST sequence',
            unit='#')

        # Note: total_exp_nr can be larger than
        # len(exp_list) * repetitions_per_point, because the last segment has
        # to be filled to be the same size as the others, even if the last
        # QASM file does not contain exp_per_file experiments.
        total_exp_nr = (len(qasm_files) * exp_per_file * hard_repetitions *
                        soft_repetitions)

        if d.result_logging_mode != 'digitized':
            logging.warning('GST is intended for use with digitized detector.'
                            ' Analysis will fail otherwise.')

        metadata_dict = {
            'gs_target': gs_target_path,
            'prep_fids': prep_fids_path,
            'meas_fids': meas_fids_path,
            'germs': germs_path,
            'max_lengths': max_lengths,
            'exp_per_file': exp_per_file,
            'exp_last_file': exp_last_file,
            'nr_hard_segs': len(qasm_files),
            'hard_repetitions': hard_repetitions,
            'soft_repetitions': soft_repetitions
        }

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(total_exp_nr))
        MC.set_detector_function(d)
        MC.run('GST_2Q', exp_metadata=metadata_dict)

        # Analysis
        if analyze:
            ma.GST_Analysis()

    def measure_chevron(self, amps, lengths,
                        fluxing_qubit=None, spectator_qubit=None,
                        wait_during_flux='auto', excite_q1: bool=True,
                        MC=None):
        '''
        Measures chevron for the two qubits of the device.
        The qubit that is fluxed is q0, so the order of the qubit matters!

        Args:
            fluxing_qubit (Instr):
                    Qubit object representing the qubit which is fluxed.
            spectator_qubit (Instr):
                    Qubit object representing the qubit with which the
                    fluxing qubit interacts.
            amps (array of floats):
                    Flux pulse amplitude sweep points (in V) -> x-axis
            lengths (array of floats):
                    Flux pulse length sweep points (in s) -> y-axis
            fluxing_qubit (str):
                    name of the qubit that is fluxed/
            spectator qubit (str):
                    name of the qubit with which the fluxing qubit interacts.
            wait_during_flux (float or 'auto'):
                    Delay during the flux pulse. If this is 'auto',
                    the time is automatically picked based on the maximum
                    of the sweep points.
            excite_q1 (bool):
                    False: measure |01> - |10> chevron.
                    True: measure |11> - |02> chevron.
            MC (Instr):
                    Measurement control instrument to use for the experiment.
        '''

        if MC is None:
            MC = qc.station.components['MC']
        CBox = self.central_controller.get_instr()

        if fluxing_qubit is None:
            fluxing_qubit = self.qubits()[0]
        if spectator_qubit is None:
            spectator_qubit = self.qubits()[1]

        q0 = self.find_instrument(fluxing_qubit)
        q1 = self.find_instrument(spectator_qubit)
        self.prepare_for_timedomain()
        # only prepare q0 for fluxing, because this is the only qubit that
        # gets a flux pulse.
        q0.prepare_for_fluxing()

        # Use flux lutman from q0, since this is the qubit being fluxed.
        QWG_flux_lutman = q0.flux_LutMan.get_instr()
        QWG = QWG_flux_lutman.QWG.get_instr()

        # Set the wait time during the flux pulse to be long enough to fit the
        # longest flux pulse
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/q0.f_pulse_mod())
            wait_during_flux = (np.ceil(max(lengths) / T_pulsemod) + 2) \
                * T_pulsemod

        cfg = deepcopy(self.qasm_config())
        cfg['operation dictionary']['square']['duration'] = int(
            np.round(wait_during_flux * 1e9))

        qasm_file = mqqs.chevron_seq(fluxing_qubit=q0.name,
                                     spectator_qubit=q1.name,
                                     excite_q1=excite_q1,
                                     RO_target='all')

        CBox.trigger_source('internal')
        qasm_folder, fn = os.path.split(qasm_file.name)
        base_fn = fn.split('.')[0]
        qumis_fn = os.path.join(qasm_folder, base_fn + '.qumis')
        compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=1)
        compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)

        CBox.load_instructions(qumis_fn)
        CBox.start()

        # Set the wave dict unit to frac, because we take care of the proper
        # scaling in the sweep function
        oldWaveDictUnit = QWG_flux_lutman.wave_dict_unit()
        QWG_flux_lutman.wave_dict_unit('frac')

        s1 = swf.QWG_flux_amp(QWG=QWG, channel=QWG_flux_lutman.F_ch(),
                              frac_amp=QWG_flux_lutman.F_amp())

        s2 = swf.QWG_lutman_par(QWG_flux_lutman, QWG_flux_lutman.F_length)
        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points(amps)
        MC.set_sweep_points_2D(lengths)
        d = self.get_integrated_average_detector()

        MC.set_detector_function(d)
        MC.run('Chevron_{}_{}'.format(q0.name, q1.name), mode='2D')

        # Restore the old wave dict unit.
        QWG_flux_lutman.wave_dict_unit(oldWaveDictUnit)

    def calibrate_CZ_1Q_phase_fine(self,
                                   correction_qubit=None,
                                   spectator_qubit=None,
                                   span: float=0.04, num: int=31,
                                   min_fit_pts: int=15, MC=None) -> bool:
        '''
        Measures a the Z-amp cost function in a small range around the value
        from the last calibration, fits a parabola, extracts a new minimum,
        and sets the new Z-amp in the flux lookup table manager of the
        correction qubit.

        Args:
            correction_qubit (Instr):
                    Qubit object representing the qubit for which the single
                    qubit phase correction should be calibrated.
            spectator_qubit (Instr):
                    Qubit object representing the other qubit involved in the
                    CZ.
            span (float):
                    Full span of the range around the last known value for
                    Z-amp in which the cost function is measured.
            num (int):
                    Number of points measured in the specified range.
            min_fit_pts (int):
                    Minimum number of points that should be used for the fit.
                    The measurement is repeated with an adapted range if
                    there are less points than left after discarding points
                    that are too far away from the minimum.

        Returns:
            success (bool):
                    True if calibration succeeded, False otherwise.
        '''
        if MC is None:
            MC = qc.station.components['MC']

        if correction_qubit is None:
            correction_qubit = self.qubits()[0]
        if spectator_qubit is None:
            spectator_qubit = self.qubits()[1]

        old_z_amp = correction_qubit.flux_LutMan.get_instr().Z_amp()
        repeat_calibration = True

        while repeat_calibration:
            amp_pts = gen_sweep_pts(center=old_z_amp, span=span, num=num)
            CZ_cost_Z_amp(correction_qubit, spectator_qubit, MC,
                          Z_amps_q0=amp_pts)
            try:
                a = ma.CZ_1Q_phase_analysis()
            except RuntimeError as e:
                # Analysis returns a RuntimeError when the fit fails due to
                # bad range and it can't fix itself.
                print(e)
                return False
            except Exception as e:
                raise e
            new_z_amp = a.opt_z_amp

            if len(a.fit_data) < min_fit_pts:
                print('Bad measurement range: too large or too far from '
                      'minimum for parabolic model.\nRetrying...')
                old_z_amp = new_z_amp
                if a.del_indices[0] == 0 and a.del_indices[-1] == num-1:
                    # Values larger than one found on both sides
                    # -> reduce range
                    span *= 0.5
                continue

            if new_z_amp < amp_pts[0]:
                print('Fitted minimum below scan range. Repeating scan '
                      'around lowest point {}.'.format(amp_pts[0]))
                old_z_amp = amp_pts[0]
            elif new_z_amp > amp_pts[-1]:
                print('Fitted minimum above scan range. Repeating scan '
                      'around hightest point {}.'.format(amp_pts[-1]))
                old_z_amp = amp_pts[-1]
            else:
                repeat_calibration = False
        # This has to be set in the qubit object.
        # the "prepare_for_fluxing" in turn should ensure the right vals
        # get updated.
        correction_qubit.flux_LutMan.get_instr().Z_amp(new_z_amp)
        return True

    def calibrate_grover_1Q_phase_fine(self,
                                       correction_qubit=None,
                                       spectator_qubit=None,
                                       msmt_suffix: str=None,
                                       span: float=0.04, num: int=31,
                                       min_fit_pts: int=15, MC=None) -> bool:
        '''
        Fine-tune the single qubit phase correction for the second CZ pulse
        in Grover's algorithm based on the last known value. A range around
        the last known value is measured and a parabolic fit is used to find
        the minimum.

        Args:
            correction_qubit (obj):
                    Qubit object representing the qubit to which the phase
                    correction is applied.
            spectator_qubit (obj):
                    Qubit object representing the other qubit involved in the
                    CZ.
            span (float):
                    Full span of the range around the last known value for
                    Z_amp_grover.
            num (int):
                    Number of points measured in the specified range.
            min_fit_pts (int):
                    Minimum number of points that should be used for the fit.
                    The measurement is repeated with an adapted range if
                    there are less points than left after discarding points
                    that are too far away from the minimum.
        '''
        if MC is None:
            MC = qc.station.components['MC']

        if correction_qubit is None:
            correction_qubit = self.qubits()[0]
        if spectator_qubit is None:
            spectator_qubit = self.qubits()[1]

        old_z_amp = correction_qubit.flux_LutMan.get_instr().Z_amp_grover()
        repeat_calibration = True
        while repeat_calibration:
            amp_pts = gen_sweep_pts(center=old_z_amp, span=span, num=num)
            self.measure_grover_1Q_phase(
                amp_pts, correction_qubit, spectator_qubit,
                msmt_suffix=msmt_suffix, MC=MC)
            try:
                a = ma.CZ_1Q_phase_analysis(use_diff=False, meas_vals_idx=1)
            except RuntimeError as e:
                # Analysis returns a RuntimeError when the fit fails due to
                # bad range and it can't fix itself.
                print(e)
                return False
            except Exception as e:
                raise e
            new_z_amp = a.opt_z_amp

            if len(a.fit_data) < min_fit_pts:
                print('Bad measurement range: too large or too far from '
                      'minimum for parabolic model.\nRetrying...')
                old_z_amp = new_z_amp
                if a.del_indices[0] == 0 and a.del_indices[-1] == num-1:
                    # Values larger than one found on both sides
                    # -> reduce range
                    span *= 0.5
                continue

            if new_z_amp < amp_pts[0]:
                print('Fitted minimum below scan range. Repeating scan '
                      'around lowest point {}.'.format(amp_pts[0]))
                old_z_amp = amp_pts[0]
            elif new_z_amp > amp_pts[-1]:
                print('Fitted minimum above scan range. Repeating scan '
                      'around hightest point {}.'.format(amp_pts[-1]))
                old_z_amp = amp_pts[-1]
            else:
                repeat_calibration = False
        # This has to be set in the qubit object.
        # the "prepare_for_fluxing" in turn should ensure the right vals
        # get updated.
        correction_qubit.flux_LutMan.get_instr().Z_amp_grover(new_z_amp)
        return True


    def measure_grover_1Q_phase(self,
                                Z_amps,
                                correction_qubit=None,
                                spectator_qubit=None,
                                QWG_codeword: int=0,
                                wait_during_flux='auto',
                                # span: float=0.04, num: int=31,
                                msmt_suffix: str=None,
                                MC=None) -> bool:
        '''
        Measure the single qubit phase error of the second CZ pulse in
        Grover's algorithm. This assumes that the first CZ is calibrated
        properly. Measured values are cos(phi). The sequence is
            mX90 -- CZ -- wait -- CZ2 -- X90
        such that the minimum of the curve corresponds to zero phase error.

        Args:
            Z_amps (list of floats):
                    Sweep points; amplitudes for the single qubit phase
                    correction Z-pulse of the second CZ.
            correction_qubit (obj):
                    Qubit object representing the qubit to which the phase
                    correction is applied.
            spectator_qubit (obj):
                    Qubit object representing the other qubit involved in the
                    CZ.
            QWG_codeword (int):
                    Codeword which is used for the composite flux pulse.
            wait_during_flux (float or 'auto'):
                    Delay between the two pi-half pulses. If this is 'auto',
                    the time is automatically picked based on the duration of
                    a microwave and flux pulses (in the Grover sequence ther
                    is one microwave pulse between the flux pulses).
        '''
        if MC is None:
            MC = qc.station.components['MC']

        if msmt_suffix is None:
            msmt_suffix = self.msmt_suffix

        CBox = self.central_controller.get_instr()

        if correction_qubit is None:
            correction_qubit = self.qubits()[0]
        if spectator_qubit is None:
            spectator_qubit = self.qubits()[1]

        f_lutman_spect = spectator_qubit.flux_LutMan.get_instr()
        f_lutman_corr = correction_qubit.flux_LutMan.get_instr()

        spectator_qubit.prepare_for_timedomain()
        spectator_qubit.prepare_for_fluxing()
        correction_qubit.prepare_for_timedomain()
        correction_qubit.prepare_for_fluxing(reset=False)

        cfg = deepcopy(correction_qubit.qasm_config())

        mw_dur = cfg['operation dictionary']['y90']['duration']

        # Waveform for correction qubit
        std_CZ = deepcopy(
            f_lutman_corr.standard_waveforms()['adiabatic_Z'])

        zero_samples = 4 * int(np.round(mw_dur*1e-9 *
                                        f_lutman_corr.sampling_rate()))

        def wave_func(val):
            wf1 = std_CZ
            f_lutman_corr.Z_amp_grover(val)
            wf2 = deepcopy(
                f_lutman_corr.standard_waveforms()['adiabatic_Z_grover'])
            return np.concatenate([wf1, np.zeros(zero_samples), wf2])

        # Waveform for spectator qubit
        spect_CZ = deepcopy(
            f_lutman_spect.standard_waveforms()['adiabatic_Z'])
        wf_spect = np.concatenate([spect_CZ, np.zeros(zero_samples),
                                   spect_CZ])
        f_lutman_spect.load_custom_pulse_onto_AWG_lookuptable(
            waveform=wf_spect,
            pulse_name='square_0',
            codeword=QWG_codeword)

        # Set the delay between the pihalf pulses to be long enough to fit the
        # flux pulse
        max_len = len(wave_func(0.1))
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/correction_qubit.f_pulse_mod())
            wait_between = (np.ceil(
                max_len / f_lutman_corr.sampling_rate() /
                T_pulsemod) + 2) * T_pulsemod
            print('Autogenerated wait between pi-halfs: {} ns'
                  .format(np.round(wait_between *
                                   f_lutman_corr.sampling_rate())))
        else:
            wait_between = wait_during_flux

        cfg['luts'][1]['square_0'] = QWG_codeword
        cfg["operation dictionary"]["square_0"] = {
            "parameters": 1,
            "duration": int(np.round(wait_between *
                                     f_lutman_corr.sampling_rate())),
            "type": "flux",
            "matrix": []
        }

        # Compile sequence
        CBox.trigger_source('internal')
        qasm_file = sqqs.Ram_Z(
            qubit_name=correction_qubit.name,
            no_of_points=1,
            cal_points=False,
            rec_Y90=False)
        qasm_folder, qasm_fn = os.path.split(qasm_file.name)
        qumis_fn = os.path.join(qasm_folder,
                                qasm_fn.split('.')[0] + '.qumis')
        compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=0)
        compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)
        CBox.load_instructions(qumis_fn)

        d = self.get_integrated_average_detector()

        metadata_dict = {
            'Z_amps': Z_amps
        }

        MC.set_sweep_function(swf.QWG_lutman_custom_wave_chunks(
            LutMan=f_lutman_corr,
            wave_func=wave_func,
            sweep_points=Z_amps,
            chunk_size=1,
            codewords=[QWG_codeword],
            param_name='Z_amp',
            param_unit='V'))
        MC.set_sweep_points(Z_amps)
        MC.set_detector_function(d)

        MC.run('Grover_1Q_phase_fine{}'.format(msmt_suffix),
               exp_metadata=metadata_dict)
        ma.MeasurementAnalysis(label='Grover_1Q_phase_fine')

    # def measure_CZ_1Q_phase(
    #         q0, q1,  MC, Z_amps=gen.gen_sweep_pts(start=-.3, stop=.3, num=31),
    #         cases=('no_excitation', 'excitation')):
    #     """
    #     Measure single qubit phase of q0 after a CZ gate as function of the
    #     amplitude of the Z phase-correction pulse.
    #     Sequence
    #         q0:    X90  --      -- Z --  mX90  -- RO
    #                         CZ
    #         q1   (X180) --      -- Z -- (X180) -- RO

    #     Args:
    #         q0 (Instrument):
    #             Qubit object on which the phase is measured.
    #         q1 (Instrument):
    #             Qubit object that acts as the control qubit for the CZ.
    #         MC (Instrument):
    #             Measurement Control instrument.
    #         Z_amps (array of floats):
    #             Sweep points for the amplitude of the Z flux pulse on q0 (x-axis).
    #         Z_amps_q1 (array of floats or None):
    #             If None, the amplitude of the Z pulse on q1 is not changed, and
    #             a 1D measurement is performed.
    #             Else: Sweep points for the amplitude of the Z pulse on q1 (y-axis
    #             in the 2D sweep).
    #         cases (list or tuple):
    #             ['excitation']: Do the sequence with the X180 pulses on q1
    #                             enabled.
    #             ['no_excitation']: Do the sequence with the X180 pulses on q1
    #                                disabled.
    #             ['no_excitation', 'excitation']: Measure bothe cases.
    #     """
    #     CBox = q0.CBox.get_instr()
    #     QWG_lutman = q0.flux_LutMan.get_instr()
    #     QWG = QWG_lutman.QWG.get_instr()

    #     q1.prepare_for_timedomain()
    #     q1.prepare_for_fluxing()
    #     q0.prepare_for_timedomain()
    #     q0.prepare_for_fluxing(reset=False)

    #     qasm_file = qwfs.CZ_calibration_seq(q0=q0.name, q1=q1.name,
    #                                         RO_target=q1.name,
    #                                         vary_single_q_phase=False,
    #                                         cases=cases)
    #     qasm_fn = qasm_file.name
    #     cfg = q0.qasm_config()
    #     CBox.trigger_source('internal')
    #     qasm_folder, fn = os.path.split(qasm_fn)
    #     base_fn = fn.split('.')[0]
    #     qumis_fn = os.path.join(qasm_folder, base_fn + ".qumis")
    #     compiler = qcx.QASM_QuMIS_Compiler(
    #         verbosity_level=1)
    #     compiler.compile(qasm_fn, qumis_fn=qumis_fn,
    #                      config=cfg)

    #     CBox.load_instructions(qumis_fn)
    #     QWG.start()

    #     s = swf.QWG_lutman_par(QWG_lutman, QWG_lutman.Z_amp)

    #     MC.set_sweep_function(s)
    #     MC.set_sweep_points(Z_amps_pts)

    #     d = det.UHFQC_integrated_average_detector(
    #         UHFQC=q0._acquisition_instrument, AWG=CBox,
    #         channels=[
    #             q0.RO_acq_weight_function_I(),
    #             q1.RO_acq_weight_function_I()],
    #         nr_averages=q0.RO_acq_averages(),
    #         real_imag=True, single_int_avg=True,
    #         result_logging_mode='lin_trans',
    #         integration_length=q0.RO_acq_integration_length(),
    #         seg_per_point=len(cases))
    #     d.detector_control = 'hard'

    #     MC.set_detector_function(d)
    #     MC.run('CZ_Z_amp')
    #     ma.MeasurementAnalysis(label='CZ_Z_amp')
