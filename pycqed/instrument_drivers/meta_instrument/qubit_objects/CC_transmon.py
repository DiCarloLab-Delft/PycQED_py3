'''
File:               CC_transmon.py
Author:             Adriaan Rol
Purpose:            Multiple qubit objects for CC controlled transmons
Usage:
Notes:
Bugs:
'''
import time
import logging
import numpy as np

from .qubit_object import Transmon
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed.measurement.mc_parameter_wrapper as pw
from pycqed.analysis import measurement_analysis as ma

from pycqed.analysis.tools.data_manipulation import rotation_matrix
from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_CBox
from pycqed.measurement.calibration_toolbox import mixer_skewness_cal_CBox_adaptive

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs
import pycqed.measurement.CBox_sweep_functions as cbs
from pycqed.scripts.Experiments.intel_demo import qasm_helpers as qh
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta

from pycqed.measurement import detector_functions as det


class CBox_v3_driven_transmon(Transmon):

    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''

    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, LutMan,
                 CBox,
                 MC,
                 **kw):
        super().__init__(name, **kw)
        '''
        Adds the parameters to the qubit insrument, it provides initial values
        for some parameters but not for all. Powers have to be set by hand as
        a safety measure.
        '''
        # MW-sources
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.IVVI = IVVI
        self.LutMan = LutMan
        self.CBox = CBox
        self.MC = MC
        self.add_parameters()

    def add_parameters(self):
        self.add_parameter('acquisition_instrument',
                           set_cmd=self._set_acquisition_instr,
                           get_cmd=self._get_acquisition_instr,
                           vals=vals.Strings())

        self._RF_RO_Source_instr = None
        self.add_parameter('RF_RO_source',
                           set_cmd=self._set_RF_RO_source,
                           get_cmd=self._get_RF_RO_source,
                           vals=vals.Strings())

        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_power_cw', label='RO power cw',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('mod_amp_td', label='RO modulation ampl td',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pow', label='spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e6,
                           parameter_class=ManualParameter)

        # Time-domain parameters
        self.add_parameter('f_pulse_mod',
                           initial_value=-2e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('awg_nr', label='CBox awg nr', unit='#',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_awg_nr', label='CBox RO awg nr', unit='#',
                           initial_value=1,
                           parameter_class=ManualParameter)

        self.add_parameter('RO_acq_integration_length', initial_value=500e-9,
                           vals=vals.Numbers(min_value=0, max_value=20e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_soft_averages', initial_value=4,
                           vals=vals.Ints(min_value=1),
                           parameter_class=ManualParameter)

        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', unit='V',
                           initial_value=0.3,
                           parameter_class=ManualParameter)
        # Amp 90 is hardcoded to be half amp180
        self.add_parameter('amp90',
                           label='Pi/2-pulse amplitude', unit='V',
                           initial_value=0.3,
                           parameter_class=ManualParameter)
        self.add_parameter('gauss_width', unit='s',
                           initial_value=10e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('motzoi', label='Motzoi parameter', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_rotation_angle', unit='deg',
                           initial_value=0,
                           vals=vals.Numbers(0, 360),
                           parameter_class=ManualParameter)
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)

        # Mixer skewness correction
        self.add_parameter('mixer_drive_phi', unit='deg',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_drive_alpha', unit='',
                           parameter_class=ManualParameter, initial_value=1)
        # Mixer offsets correction, qubit drive
        self.add_parameter('mixer_offs_drive_I',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_drive_Q',
                           parameter_class=ManualParameter, initial_value=0)
        # Mixer offsets correction, RO pulse
        self.add_parameter('mixer_offs_RO_I',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_RO_Q',
                           parameter_class=ManualParameter, initial_value=0)

        self.add_parameter('RO_pulse_type', initial_value='MW_IQmod_pulse',
                           vals=vals.Enum(
                               'MW_IQmod_pulse', 'Gated_MW_RO_pulse'),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_depletion_time', initial_value=1e-6,
                           unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(min_value=0))

        self.add_parameter('RO_acq_period_cw', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 500e-6),
                           initial_value=10e-6)


        self.add_parameter('init_time',
                           label='Qubit initialization time',
                           unit='s', initial_value=200e-6,
                           parameter_class=ManualParameter,
                           # max value based on register size
                           vals=vals.Numbers(min_value=1e-6,
                                             max_value=327668e-9))

        self.add_parameter('cal_pt_zero',
                           initial_value=None,
                           vals=vals.Anything(),  # should be a tuple validator
                           label='Calibration point |0>',
                           parameter_class=ManualParameter)
        self.add_parameter('cal_pt_one',
                           initial_value=None,
                           vals=vals.Anything(),  # should be a tuple validator
                           label='Calibration point |1>',
                           parameter_class=ManualParameter)

    def prepare_for_continuous_wave(self):
        self.LO.on()
        self.td_source.off()
        # if self.RO_pulse_type() == "Gated_MW_RO_pulse":
        #     RF = self.find_instrument(self.RF_RO_source())

        self.cw_source.off()
        self.cw_source.pulsemod_state.set('off')
        self.cw_source.power.set(self.spec_pow.get())


        # REMOVE ME !!!!!!!
        # To be replaced once spec mode is there 
        # self.heterodyne_instr.acquisition_instr(self.acquisition_instrument())
        # # Heterodyne tone configuration
        # if not self.f_RO():
        #     RO_freq = self.f_res()
        # else:
        #     RO_freq = self.f_RO()

        # self.heterodyne_instr._disable_auto_seq_loading = False

        # self.heterodyne_instr.RF.on()
        # self.heterodyne_instr.LO.on()
        # if hasattr(self.heterodyne_instr, 'mod_amp'):
        #     self.heterodyne_instr.set('mod_amp', self.mod_amp_cw.get())
        # else:
        #     self.heterodyne_instr.RF_power(self.RO_power_cw())
        # self.heterodyne_instr.set('f_RO_mod', self.f_RO_mod.get())
        # self.heterodyne_instr.frequency.set(RO_freq)
        # self.heterodyne_instr.RF.power(self.RO_power_cw())
        # self.heterodyne_instr.RF_power(self.RO_power_cw())
        # self.heterodyne_instr.nr_averages(self.RO_acq_averages())


    def prepare_for_timedomain(self):
        self.MC.soft_avg(self.RO_soft_averages())
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.frequency.set(self.f_qubit.get()
                                     - self.f_pulse_mod.get())
        self.CBox.trigger_source('internal')
        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.frequency.set(f_RO - self.f_RO_mod.get())

        self.td_source.power.set(self.td_source_pow.get())

        # Mixer offsets correction
        self.CBox.set('AWG{:.0g}_dac0_offset'.format(self.awg_nr.get()),
                      self.mixer_offs_drive_I.get())
        self.CBox.set('AWG{:.0g}_dac1_offset'.format(self.awg_nr.get()),
                      self.mixer_offs_drive_Q.get())
        self.CBox.set('AWG{:.0g}_dac0_offset'.format(self.RO_awg_nr.get()),
                      self.mixer_offs_RO_I.get())
        self.CBox.set('AWG{:.0g}_dac1_offset'.format(self.RO_awg_nr.get()),
                      self.mixer_offs_RO_Q.get())

        # pulse pars
        self.LutMan.Q_amp180.set(self.amp180.get())
        self.LutMan.Q_amp90.set(self.amp90.get())
        self.LutMan.Q_gauss_width.set(self.gauss_width.get())
        self.LutMan.Q_motzoi_parameter.set(self.motzoi.get())
        self.LutMan.Q_modulation.set(self.f_pulse_mod.get())

        # RO pars
        self.LutMan.M_modulation(self.f_RO_mod())
        self.LutMan.M_amp(self.RO_amp())
        self.LutMan.M_length(self.RO_pulse_length())

        self.LutMan.lut_mapping(['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90',
                                 'mY90', 'M_square'])

        # Mixer skewness correction
        self.LutMan.mixer_IQ_phase_skewness.set(0)
        self.LutMan.mixer_QI_amp_ratio.set(1)
        self.LutMan.mixer_apply_predistortion_matrix.set(True)
        self.LutMan.mixer_alpha.set(self.mixer_drive_alpha.get())
        self.LutMan.mixer_phi.set(self.mixer_drive_phi.get())

        self.LutMan.load_pulses_onto_AWG_lookuptable(self.awg_nr.get())
        self.LutMan.load_pulses_onto_AWG_lookuptable(self.RO_awg_nr.get())

        self.CBox.set('sig{}_threshold_line'.format(
                      int(self.signal_line.get())),
                      int(self.RO_threshold.get()))

    def _get_RF_RO_source(self):
        return self._RF_RO_Source_instr.name

    def _set_RF_RO_source(self, RF_RO_source_name):
        self._RF_RO_Source_instr = self.find_instrument(RF_RO_source_name)



    def _get_acquisition_instr(self):
        # _acquisition_instrument is the class
        # acquisition_instrument (qcodes param) contains the string
        return self._acquisition_instrument.name

    def _set_acquisition_instr(self, acq_instr_name):
        self._acquisition_instrument = self.find_instrument(acq_instr_name)
        if 'CBox' in acq_instr_name:
            logging.info("setting CBox acquisition")
            self.int_avg_det = None  # FIXME: Not implemented
            self.int_avg_det_rot = None  # FIXME: Not implemented
            self.int_log_det = qh.CBox_integration_logging_det_CC(self.CBox)
            self.input_average_detector = None  # FIXME: Not implemented
        elif 'UHFQC' in acq_instr_name:
            logging.info("setting UHFQC acquisition")
            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=self._acquisition_instrument,
                AWG=self.AWG, nr_averages=self.RO_acq_averages())
            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.AWG,
                channels=[self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()], 
                nr_averages=self.RO_acq_averages(),
                real_imag=True,
                integration_length=self.RO_acq_integration_length())
            self.int_avg_det_single.detector_control = 'soft'

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.AWG,
                channels=[self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()], 
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length())
            self.int_avg_det_rot = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.AWG,
                channels=[self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()], nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length(), rotate=True)
            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=self._acquisition_instrument, AWG=self.AWG,
                channels=[
                    self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()],
                integration_length=self.RO_acq_integration_length())

        elif 'ATS' in acq_instr_name:
            logging.info("setting ATS acquisition")
            self.int_avg_det = det.ATS_integrated_average_continuous_detector(
                ATS=self._acquisition_instrument.card,
                ATS_acq=self._acquisition_instrument.controller, AWG=self.AWG,
                nr_averages=self.RO_acq_averages())
        return


    def get_resetless_rb_detector(self, nr_cliff, starting_seed=1,
                                  nr_seeds='max', pulse_p_elt='min',
                                  MC=None,
                                  upload=True):
        raise NotImplementedError()
        # if MC is None:
        #     MC = self.MC

        # if pulse_p_elt == 'min':
        #     safety_factor = 5 if nr_cliff < 8 else 3
        #     pulse_p_elt = int(safety_factor*nr_cliff)
        # if nr_seeds == 'max':
        #     nr_seeds = 29184//pulse_p_elt

        # if nr_seeds*pulse_p_elt > 29184:
        #     raise ValueError(
        #         'Too many pulses ({}), {} seeds, {} pulse_p_elt'.format(
        #             nr_seeds*pulse_p_elt, nr_seeds, pulse_p_elt))

        # resetless_interval = (
        #     np.round(pulse_p_elt*self.pulse_delay.get()*1e6)+2.5)*1e-6

        # combined_tape = []
        # for i in range(nr_seeds):
        #     if starting_seed is not None:
        #         seed = starting_seed*1000*i
        #     else:
        #         seed = None
        #     rb_seq = rb.randomized_benchmarking_sequence(nr_cliff,
        #                                                  desired_net_cl=3,
        #                                                  seed=seed)
        #     tape = rb.convert_clifford_sequence_to_tape(
        #         rb_seq, self.LutMan.lut_mapping.get())
        #     if len(tape) > pulse_p_elt:
        #         raise ValueError(
        #             'Too many pulses ({}), {} pulse_p_elt'.format(
        #                 len(tape),  pulse_p_elt))
        #     combined_tape += [0]*(pulse_p_elt-len(tape))+tape

        # # Rename IF in awg_swf_resetless tape
        # s = awg_swf.Resetless_tape(
        #     n_pulses=pulse_p_elt, tape=combined_tape,
        #     IF=self.f_RO_mod.get(),
        #     pulse_delay=self.pulse_delay.get(),
        #     resetless_interval=resetless_interval,
        #     RO_pulse_delay=self.RO_pulse_delay.get(),
        #     RO_pulse_length=self.RO_pulse_length.get(),
        #     RO_trigger_delay=self.RO_acq_marker_delay.get(),
        #     AWG=self.AWG, CBox=self.CBox, upload=upload)

        # d = cdet.CBox_trace_error_fraction_detector(
        #     'Resetless rb det',
        #     MC=MC, AWG=self.AWG, CBox=self.CBox,
        #     sequence_swf=s,
        #     threshold=self.RO_threshold.get(),
        #     save_raw_trace=False)
        # return d

    def calibrate_pulse_parameters(self, method='resetless_rb', nr_cliff=10,
                                   parameters=['amp', 'motzoi', 'frequency'],
                                   amp_guess=None, motzoi_guess=None,
                                   frequency_guess=None,
                                   a_step=30, m_step=.1, f_step=20e3,
                                   MC=None, nested_MC=None,
                                   update=False, close_fig=True,
                                   verbose=True):
        '''
        Calibrates single qubit pulse parameters currently only using
        the resetless rb method (requires reasonable (80%+?) discrimination
        fidelity)

        If it there is only one parameter to sweep it will use brent's method
        instead.

        The function returns the values it found for the optimization.
        '''
        raise NotImplementedError()
        # if method is not 'resetless_rb':
        #     raise NotImplementedError()

        # self.prepare_for_timedomain()
        # if MC is None:
        #     MC = self.MC
        # if nested_MC is None:
        #     nested_MC = self.nested_MC

        # d = self.get_resetless_rb_detector(nr_cliff=nr_cliff, MC=nested_MC)

        # name = 'RB_{}cl_numerical'.format(nr_cliff)
        # MC.set_detector_function(d)

        # if amp_guess is None:
        #     amp_guess = self.amp180.get()
        # if motzoi_guess is None:
        #     motzoi_guess = self.motzoi.get()
        # if frequency_guess is None:
        #     frequency_guess = self.f_qubit.get()
        # # Because we are sweeping the source and not the qubit frequency
        # start_freq = frequency_guess - self.f_pulse_mod.get()

        # sweep_functions = []
        # x0 = []
        # init_steps = []
        # if 'amp' in parameters:
        #     sweep_functions.append(cb_swf.LutMan_amp180_90(self.LutMan))
        #     x0.append(amp_guess)
        #     init_steps.append(a_step)
        # if 'motzoi' in parameters:
        #     sweep_functions.append(
        #         pw.wrap_par_to_swf(self.LutMan.motzoi_parameter))
        #     x0.append(motzoi_guess)
        #     init_steps.append(m_step)
        # if 'frequency' in parameters:
        #     sweep_functions.append(
        #         pw.wrap_par_to_swf(self.td_source.frequency))
        #     x0.append(start_freq)
        #     init_steps.append(f_step)
        # if len(sweep_functions) == 0:
        #     raise ValueError(
        #         'parameters "{}" not recognized'.format(parameters))

        # MC.set_sweep_functions(sweep_functions)

        # if len(sweep_functions) != 1:
        #     # noise ensures no_improv_break sets the termination condition
        #     ad_func_pars = {'adaptive_function': nelder_mead,
        #                     'x0': x0,
        #                     'initial_step': init_steps,
        #                     'no_improv_break': 10,
        #                     'minimize': False,
        #                     'maxiter': 500}
        # elif len(sweep_functions) == 1:
        #     # Powell does not work for 1D, use brent instead
        #     brack = (x0[0]-5*init_steps[0], x0[0])
        #     # Ensures relative change in parameter is relevant
        #     if parameters == ['frequency']:
        #         tol = 1e-9
        #     else:
        #         tol = 1e-3
        #     print('Tolerance:', tol, init_steps[0])
        #     print(brack)
        #     ad_func_pars = {'adaptive_function': brent,
        #                     'brack': brack,
        #                     'tol': tol,  # Relative tolerance in brent
        #                     'minimize': False}
        # MC.set_adaptive_function_parameters(ad_func_pars)
        # MC.run(name=name, mode='adaptive')
        # if len(sweep_functions) != 1:
        #     a = ma.OptimizationAnalysis(auto=True, label=name,
        #                                 close_fig=close_fig)
        #     if verbose:
        #         # Note printing can be made prettier
        #         print('Optimization converged to:')
        #         print('parameters: {}'.format(parameters))
        #         print(a.optimization_result[0])
        #     if update:
        #         for i, par in enumerate(parameters):
        #             if par == 'amp':
        #                 self.amp180.set(a.optimization_result[0][i])
        #             elif par == 'motzoi':
        #                 self.motzoi.set(a.optimization_result[0][i])
        #             elif par == 'frequency':
        #                 self.f_qubit.set(a.optimization_result[0][i] +
        #                                  self.f_pulse_mod.get())
        #     return a
        # else:
        #     a = ma.MeasurementAnalysis(label=name, close_fig=close_fig)
        #     print('Optimization for {} converged to: {}'.format(
        #         parameters[0], a.sweep_points[-1]))
        #     if update:
        #         if parameters == ['amp']:
        #             self.amp180.set(a.sweep_points[-1])
        #         elif parameters == ['motzoi']:
        #             self.motzoi.set(a.sweep_points[-1])
        #         elif parameters == ['frequency']:
        #             self.f_qubit.set(a.sweep_points[-1]+self.f_pulse_mod.get())
        #     return a.sweep_points[-1]

    def calibrate_mixer_offsets_drive(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        # ensures freq is set correctly
        self.prepare_for_timedomain()
        offset_I, offset_Q = mixer_carrier_cancellation_CBox(
            CBox=self.CBox, SH=signal_hound, source=self.td_source,
            MC=self.MC, awg_nr=self.awg_nr.get())
        if update:
            self.mixer_offs_drive_I.set(offset_I)
            self.mixer_offs_drive_Q.set(offset_Q)

    def calibrate_mixer_offsets_RO(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        self.prepare_for_timedomain()
        offset_I, offset_Q = mixer_carrier_cancellation_CBox(
            CBox=self.CBox, SH=signal_hound, source=self.LO,
            MC=self.MC, awg_nr=self.RO_awg_nr.get())
        if update:
            self.mixer_offs_RO_I.set(offset_I)
            self.mixer_offs_RO_Q.set(offset_Q)

    def calibrate_mixer_skewness(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness using mixer_skewness_cal_CBox_adaptive
        see calibration toolbox for details
        '''
        self.prepare_for_timedomain()
        phi, alpha = mixer_skewness_cal_CBox_adaptive(
            CBox=self.CBox, SH=signal_hound, source=self.td_source,
            LutMan=self.LutMan, AWG=self.AWG, MC=self.MC,
            awg_nrs=[self.awg_nr.get()], calibrate_both_sidebands=True)
        if update:
            self.mixer_drive_phi.set(phi)
            self.mixer_drive_alpha.set(alpha)

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC

        # Loading the right qumis instructions
        CW_RO_sequence = sqqs.CW_RO_sequence(self.name,
                                             self.RO_acq_period_cw())
        CW_RO_sequence_asm = qta.qasm_to_asm(CW_RO_sequence.name,
                                             self.get_operation_dict())
        qumis_file = CW_RO_sequence_asm
        self.CBox.load_instructions(qumis_file.name)
        self.CBox.run_mode('run') 

        if MC is None:
            MC = self.MC


        # Create a new sweep function that sweeps two frequencies.
        # use only if the RO pulse type is the CW-RF scan, else only sweep the LO

        # MC.set_sweep_function(self.heterodyne_instr.frequency)
        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep(
                                    RO_pulse_type=self.RO_pulse_type(),
                                    RF_source=self.find_instrument(self.RF_RO_source()),
                                    LO_source=self.LO,IF=self.f_RO_mod()))
        MC.set_sweep_points(freqs)
        # make sure we use the right acquision detector. Mind the new UHFQC spec mode
        MC.set_detector_function(self.int_avg_det_single)
        # det.Heterodyne_probe(self.heterodyne_instr)
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=True):
        raise NotImplementedError()
        # self.prepare_for_continuous_wave()
        # if MC is None:
        #     MC = self.MC
        # if pulsed:
        #     # Redirect to the pulsed spec function
        #     return self.measure_pulsed_spectroscopy(freqs,
        #                                             MC, analyze, close_fig)

        # MC.set_sweep_function(pw.wrap_par_to_swf(
        #                       self.cw_source.frequency))
        # MC.set_sweep_points(freqs)
        # MC.set_detector_function(
        #     det.Heterodyne_probe(self.heterodyne_instr))
        # MC.run(name='spectroscopy'+self.msmt_suffix)

        # if analyze:
        #     ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_pulsed_spectroscopy(self, freqs, MC=None,
                                    analyze=True, close_fig=True):
        # This is a trick so I can reuse the heterodyne instr
        # to do pulsed-spectroscopy
        raise NotImplementedError()
        # self.heterodyne_instr._disable_auto_seq_loading = True
        # if ('Pulsed_spec' not in self.AWG.setup_filename.get()):
        #     st_seqs.Pulsed_spec_seq_RF_mod(
        #         IF=self.f_RO_mod.get(),
        #         spec_pulse_length=16e-6, marker_interval=30e-6,
        #         RO_pulse_delay=self.RO_pulse_delay.get())
        # self.cw_source.pulsemod_state.set('on')
        # self.cw_source.power.set(self.spec_pow_pulsed.get())

        # self.AWG.start()
        # if hasattr(self.heterodyne_instr, 'mod_amp'):
        #     self.heterodyne_instr.set('mod_amp', self.mod_amp_cw.get())
        # else:
        #     self.heterodyne_instr.RF.power(self.RO_power_cw())
        # MC.set_sweep_function(pw.wrap_par_to_swf(self.cw_source.frequency))
        # MC.set_sweep_points(freqs)
        # MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        # MC.run(name='pulsed-spec'+self.msmt_suffix)
        # if analyze:
        #     ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_resonator_power(self, freqs, mod_amps,
                                MC=None, analyze=True, close_fig=True):
        '''
        N.B. This one does not use powers but varies the mod-amp.
        Need to find a way to keep this function agnostic to that
        '''
        raise NotImplementedError()
        # self.prepare_for_continuous_wave()
        # if MC is None:
        #     MC = self.MC
        # MC.set_sweep_functions(
        #     [pw.wrap_par_to_swf(self.heterodyne_instr.frequency),
        #      pw.wrap_par_to_swf(self.heterodyne_instr.mod_amp)])
        # MC.set_sweep_points(freqs)
        # MC.set_sweep_points_2D(mod_amps)
        # MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        # MC.run(name='Resonator_power_scan'+self.msmt_suffix, mode='2D')
        # if analyze:
        #     ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_resonator_dac(self, freqs, dac_voltages,
                              MC=None, analyze=True, close_fig=True):

        raise NotImplementedError()
        # self.prepare_for_continuous_wave()
        # if MC is None:
        #     MC = self.MC
        # MC.set_sweep_functions(
        #     [pw.wrap_par_to_swf(self.heterodyne_instr.frequency),
        #      pw.wrap_par_to_swf(
        #         self.IVVI['dac{}'.format(self.dac_channel.get())])
        #      ])
        # MC.set_sweep_points(freqs)
        # MC.set_sweep_points_2D(dac_voltages)
        # MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        # MC.run(name='Resonator_dac_scan'+self.msmt_suffix, mode='2D')
        # if analyze:
        #     ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_rabi(self, amps, n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        if n != 1:
            raise NotImplementedError('QASM/QuMis sequence for n>1')

        # Generating the qumis file
        single_pulse_elt = sqqs.single_elt_on(self.name)
        single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name,
                                           self.get_operation_dict())
        qumis_file = single_pulse_asm
        self.CBox.load_instructions(qumis_file.name)
        amp_swf = cbs.Lutman_par_with_reload_single_pulse(
            LutMan=self.LutMan, parameter=self.LutMan.Q_amp180,
            pulse_names=['X180'], awg_nrs=[self.awg_nr()])
        d = qh.CBox_single_integration_average_det_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg(),
            seg_per_point=1)
        MC.set_sweep_function(amp_swf)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)

        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            a = ma.Rabi_Analysis(auto=True, close_fig=close_fig)
            return a

    def measure_motzoi(self, motzois, MC=None, analyze=True, close_fig=True,
                       verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Generating the qumis file
        motzoi_elt = sqqs.two_elt_MotzoiXY(self.name)
        single_pulse_asm = qta.qasm_to_asm(
            motzoi_elt.name, self.get_operation_dict())
        asm_file = single_pulse_asm
        self.CBox.load_instructions(asm_file.name)

        motzoi_swf = cbs.Lutman_par_with_reload_single_pulse(
            LutMan=self.LutMan, parameter=self.LutMan.Q_motzoi_parameter,
            pulse_names=['X180', 'X90', 'Y180', 'Y90'], awg_nrs=[self.awg_nr()])
        d = qh.CBox_single_integration_average_det_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg(),
            seg_per_point=2)

        MC.set_sweep_function(motzoi_swf)
        MC.set_sweep_points(np.repeat(motzois, 2))
        MC.set_detector_function(d)

        MC.run('Motzoi_XY'+self.msmt_suffix)
        if analyze:
            a = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
            return a

    def measure_randomized_benchmarking(self, nr_cliffords,
                                        nr_seeds=100, T1=None,
                                        MC=None, analyze=True, close_fig=True,
                                        verbose=False, upload=True):
        # Adding calibration points
        nr_cliffords = np.append(
            nr_cliffords, [nr_cliffords[-1]+.5]*2 + [nr_cliffords[-1]+1.5]*2)

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        MC.soft_avg(nr_seeds)
        counter_param = ManualParameter('name_ctr', initial_value=0)
        asm_filenames = []
        for i in range(nr_seeds):
            RB_qasm = sqqs.randomized_benchmarking(self.name,
                                                   nr_cliffords=nr_cliffords, nr_seeds=1,
                                                   label='randomized_benchmarking_' +
                                                   str(i),
                                                   double_curves=False)
            asm_file = qta.qasm_to_asm(RB_qasm.name, self.get_operation_dict())
            asm_filenames.append(asm_file.name)

        prepare_function_kwargs = {
            'counter_param': counter_param,
            'asm_filenames': asm_filenames,
            'CBox': self.CBox}

        d = qh.CBox_int_avg_func_prep_det_CC(
            self.CBox, prepare_function=qh.load_range_of_asm_files,
            prepare_function_kwargs=prepare_function_kwargs,
            nr_averages=256)

        s = swf.None_Sweep()
        s.parameter_name = 'Number of Cliffords'
        s.unit = '#'
        MC.set_sweep_function(s)
        MC.set_sweep_points(nr_cliffords)

        MC.set_detector_function(d)
        MC.run('RB_{}seeds'.format(nr_seeds)+self.msmt_suffix)
        ma.RandomizedBenchmarking_Analysis(
            close_main_fig=close_fig, T1=T1,
            pulse_delay=self.gauss_width.get()*4)

    def measure_randomized_benchmarking_vs_pars(self, amps=None, motzois=None, freqs=None,
                                                nr_cliffords=80, nr_seeds=50,
                                                restless=False,
                                                log_length=8000):
        self.prepare_for_timedomain()

        if freqs!= None:
            raise NotImplementedError()
        if restless:
            net_clifford = 3
            d = qh.CBox_single_qubit_event_s_fraction_CC(self.CBox)
        else:
            d = qh.CBox_err_frac_CC(self.CBox)
            net_clifford = 0
        RB_elt = sqqs.randomized_benchmarking(self.name, nr_cliffords=[nr_cliffords], cal_points=False, nr_seeds=nr_seeds,
                                              double_curves=False, net_clifford=net_clifford, restless=restless)
        single_pulse_asm = qta.qasm_to_asm(RB_elt.name,
                                           self.get_operation_dict())
        qumis_file = single_pulse_asm
        self.CBox.load_instructions(qumis_file.name)

        ch_amp = swf.QWG_lutman_par(self.Q_LutMan, self.Q_LutMan.Q_amp180)
        motzoi = swf.QWG_lutman_par(self.Q_LutMan, self.Q_LutMan.Q_motzoi)

        self.CBox.log_length(log_length)
        self.MC.set_sweep_function(ch_amp)
        self.MC.set_sweep_function_2D(motzoi)
        self.MC.set_sweep_points(amps)
        self.MC.set_sweep_points_2D(motzois)
        self.MC.set_detector_function(d)

        self.MC.run('RB_amp_ncl{}_sds{}'.format(nr_cliffords, nr_seeds)+self.msmt_suffix, mode='2D')
        ma.TwoD_Analysis()

    def measure_T1(self, times, MC=None,
                   analyze=True, close_fig=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                                (times[-1]+times[0],
                                 times[-1]+times[1],
                                 times[-1]+times[2],
                                 times[-1]+times[3])])

        T1 = sqqs.T1(self.name, times=times)
        s = qh.QASM_Sweep(T1.name, self.CBox, self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = qh.CBox_integrated_average_detector_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg())

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)

        MC.run('T1'+self.msmt_suffix)
        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=True)
            return a.T1

    def measure_ramsey(self, times, artificial_detuning=0, f_qubit=None,
                       label='',
                       MC=None, analyze=True, close_fig=True, verbose=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                                (times[-1]+times[0],
                                 times[-1]+times[1],
                                 times[-1]+times[2],
                                 times[-1]+times[3])])

        # # This is required because I cannot change the phase in the pulses
        if not all([np.round(t*1e9) % (1/self.f_pulse_mod.get()*1e9)
                    == 0 for t in times]):
            raise ValueError('timesteps must be multiples of modulation freq')

        if f_qubit is None:
            f_qubit = self.f_qubit()
        # # this should have no effect if artificial detuning = 0
        self.td_source.set('frequency', f_qubit - self.f_pulse_mod.get() +
                           artificial_detuning)

        Ramsey = sqqs.Ramsey(
            self.name, times=times, artificial_detuning=None)
        s = qh.QASM_Sweep(Ramsey.name, self.CBox, self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = qh.CBox_integrated_average_detector_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg())
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('Ramsey'+label+self.msmt_suffix)
        if analyze:
            a = ma.Ramsey_Analysis(auto=True, close_fig=True)
            if verbose:
                fitted_freq = a.fit_res.params['frequency'].value
                print('Artificial detuning: {:.2e}'.format(
                      artificial_detuning))
                print('Fitted detuning: {:.2e}'.format(fitted_freq))
                print('Actual detuning:{:.2e}'.format(
                      fitted_freq-artificial_detuning))
            return a

    def measure_echo(self, times, artificial_detuning=0,
                     label='',
                     MC=None, analyze=True, close_fig=True, verbose=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                                (times[-1]+times[0],
                                 times[-1]+times[1],
                                 times[-1]+times[2],
                                 times[-1]+times[3])])
        # # This is required because I cannot change the phase in the pulses
        if not all([np.round(t*1e9) % (1/self.f_pulse_mod.get()*1e9)
                    == 0 for t in times]):
            raise ValueError('timesteps must be multiples of modulation freq')

        echo = sqqs.echo(self.name, times=times, artificial_detuning=None)
        s = qh.QASM_Sweep(echo.name, self.CBox, self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = qh.CBox_integrated_average_detector_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg())
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('echo'+label+self.msmt_suffix)
        if analyze:
            a = ma.Echo_analysis(auto=True, close_fig=True)
            return a

    def measure_allxy(self, MC=None, label='',
                      analyze=True, close_fig=True, verbose=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        AllXY = sqqs.AllXY(self.name, double_points=True)
        s = qh.QASM_Sweep(AllXY.name, self.CBox, self.get_operation_dict())
        d = qh.CBox_integrated_average_detector_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg())
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('AllXY'+label+self.msmt_suffix)
        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a

    def measure_flipping_sequence(self, number_of_flips, MC=None, label='',
                                  equator=False,
                                  analyze=True, close_fig=True, verbose=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        number_of_flips = np.concatenate([number_of_flips,
                                          (number_of_flips[-1]+number_of_flips[0],
                                           number_of_flips[-1] +
                                           number_of_flips[1],
                                           number_of_flips[-1] +
                                           number_of_flips[2],
                                           number_of_flips[-1]+number_of_flips[3])])
        flipping_sequence = sqqs.flipping_seq(self.name, number_of_flips,
                                              equator=equator)
        s = qh.QASM_Sweep(flipping_sequence.name, self.CBox,
                          self.get_operation_dict())
        d = qh.CBox_integrated_average_detector_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg())

        MC.set_sweep_function(s)
        MC.set_sweep_points(number_of_flips)
        MC.set_detector_function(d)
        MC.run('flipping_sequence'+label+self.msmt_suffix)
        if analyze:
            a = ma.MeasurementAnalysis(close_main_fig=close_fig)
            return a

    def measure_ssro(self, no_fits=False,
                     return_detector=False,
                     MC=None, nr_shots=1024*24,
                     analyze=True, close_fig=True, verbose=True):
        # No fancy SSRO detector here @Niels, this may be something for you

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        MC.soft_avg(1)  # don't want to average single shots
        self.CBox.log_length(1024*6)  # FIXME: remove when integrating UHFQC
        off_on = sqqs.off_on(self.name)
        s = qh.QASM_Sweep(off_on.name, self.CBox, self.get_operation_dict(),
                          parameter_name='Shots')
        d = qh.CBox_integration_logging_det_CC(self.CBox)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run('SSRO'+self.msmt_suffix)
        # turn plotting back on
        MC.live_plot_enabled(old_plot_setting)
        if analyze:
            a = ma.SSRO_Analysis(label='SSRO'+self.msmt_suffix,
                                 no_fits=no_fits, close_fig=close_fig)
            if verbose:
                print('Avg. Assignement fidelity: \t{:.4f}\n'.format(a.F_a) +
                      'Avg. Discrimination fidelity: \t{:.4f}'.format(a.F_d))
            return a

    def measure_butterfly(self, return_detector=False, MC=None,
                          analyze=True, close_fig=True,
                          verbose=True,
                          initialize=True, nr_shots=1024*24,
                          update_threshold=True):

        self.prepare_for_timedomain()
        if update_threshold:
            self.CBox.lin_trans_coeffs(
                np.reshape(rotation_matrix(0, as_array=True), (4,)))
        if MC is None:
            MC = self.MC
        MC.soft_avg(1)
        # plotting slows down single shots
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)

        # Number of shots chosen to be a multiple of 6 as req for post select
        self.CBox.log_length(1024*6)  # FIXME: remove when integrating UHFQC
        qasm_file = sqqs.butterfly(self.name, initialize=initialize)
        s = qh.QASM_Sweep(qasm_file.name, self.CBox, self.get_operation_dict(),
                          parameter_name='Shots')
        d = qh.CBox_integration_logging_det_CC(self.CBox)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)

        MC.set_detector_function(self.int_log_det)
        MC.run('Butterfly{}_initialize_{}'.format(
            self.msmt_suffix, initialize))
        # turn plotting back on
        MC.live_plot_enabled(old_plot_setting)
        # first perform SSRO analysis to extract the optimal rotation angle
        # theta
        a = ma.SSRO_discrimination_analysis(
            label='Butterfly',
            current_threshold=None,
            close_fig=close_fig,
            plot_2D_histograms=True)

        # the, run it a second time to determine the optimal threshold along the
        # rotated I axis
        b = ma.SSRO_discrimination_analysis(
            label='Butterfly',
            current_threshold=None,
            close_fig=close_fig,
            plot_2D_histograms=True, theta_in=-a.theta)

        c0 = ma.butterfly_analysis(
            close_main_fig=close_fig, initialize=initialize,
            theta_in=-a.theta,
            threshold=b.opt_I_threshold, digitize=True, case=False)
        c1 = ma.butterfly_analysis(
            close_main_fig=close_fig, initialize=initialize,
            theta_in=-a.theta % 360,
            threshold=b.opt_I_threshold, digitize=True, case=True)
        if c0.butterfly_coeffs['F_a_butterfly'] > c1.butterfly_coeffs['F_a_butterfly']:
            bf_coeffs = c0.butterfly_coeffs
        else:
            bf_coeffs = c1.butterfly_coeffs
        bf_coeffs['theta'] = -a.theta % 360
        bf_coeffs['threshold'] = b.opt_I_threshold
        if update_threshold:
            self.RO_rotation_angle(bf_coeffs['theta'])
            self.RO_threshold(bf_coeffs['threshold'])
        return bf_coeffs

    def measure_rb_vs_amp(self, amps, nr_cliff=1,
                          resetless=True,
                          MC=None, analyze=True, close_fig=True,
                          verbose=False):

        raise NotImplementedError()
        return False
        # self.prepare_for_timedomain()
        # if MC is None:
        #     MC = self.MC
        # if resetless:
        #     d = self.get_resetless_rb_detector(nr_cliff=nr_cliff)
        # else:
        #     raise NotImplementedError()
        # MC.set_detector_function(d)
        # MC.set_sweep_functions([cb_swf.LutMan_amp180_90(self.LutMan)])
        # MC.set_sweep_points(amps)
        # MC.run('RB-vs-amp_{}cliff'.format(nr_cliff) + self.msmt_suffix)
        # if analyze:
        #     ma.MeasurementAnalysis(close_fig=close_fig)

    def get_operation_dict(self, operation_dict={}):

        pulse_period_clocks = convert_to_clocks(
            self.gauss_width()*4, rounding_period=1/abs(self.f_pulse_mod()))
        RO_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())

        operation_dict['init_all'] = {'instruction':
                                      '\nWaitReg r0 \nWaitReg r0 \n'}
        operation_dict['I {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction': 'wait {} \n'}
        operation_dict['X180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1001 0000 1001  \nwait {}\n'.format(
                    pulse_period_clocks)}
        operation_dict['Y180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1010 0000 1010  \nwait {}\n'.format(
                    pulse_period_clocks)}
        operation_dict['X90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1011 0000 1011  \nwait {}\n'.format(
                    pulse_period_clocks)}
        operation_dict['Y90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1100 0000 1100  \nwait {}\n'.format(
                    pulse_period_clocks)}
        operation_dict['mX90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1101 0000 1101  \nwait {}\n'.format(
                    pulse_period_clocks)}
        operation_dict['mY90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'pulse 1110 0000 1110  \nwait {}\n'.format(
                    pulse_period_clocks)}

        if self.RO_pulse_type() == 'MW_IQmod_pulse':
            operation_dict['RO {}'.format(self.name)] = {
                'duration': RO_length_clocks,
                'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                    RO_pulse_delay_clocks)
                + '\nwait {} \nmeasure \n'.format(RO_length_clocks)}
        elif self.RO_pulse_type() == 'Gated_MW_RO_pulse':
            operation_dict['RO {}'.format(self.name)] = {
                'duration': RO_length_clocks, 'instruction':
                'wait {} \ntrigger 1000000, {} \n measure \n'.format(
                    RO_pulse_delay_clocks, RO_length_clocks)}
        if RO_depletion_clocks != 0:
            operation_dict['RO {}'.format(self.name)]['instruction'] += \
                'wait {}\n'.format(RO_depletion_clocks)

        return operation_dict


def convert_to_clocks(duration, f_sampling=200e6, rounding_period=None):
    """
    convert a duration in seconds to an integer number of clocks

        f_sampling: 200e6 is the CBox sampling frequency
    """
    if rounding_period is not None:
        duration = (duration//rounding_period+1)*rounding_period
    clock_duration = int(duration*f_sampling)
    return clock_duration


class QWG_driven_transmon(CBox_v3_driven_transmon):

    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, QWG,
                 CBox,
                 MC,
                 RO_LutMan=None, Q_LutMan=None,
                 **kw):
        super(CBox_v3_driven_transmon, self).__init__(name, **kw)
        '''
        '''
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.IVVI = IVVI
        self.QWG = QWG
        self.CBox = CBox
        self.MC = MC
        self.RO_LutMan = RO_LutMan
        self.Q_LutMan = Q_LutMan
        self.add_parameters()

    def add_parameters(self):
        super().add_parameters()
        self.add_parameter('amp90_scale',
                           label='pulse amplitude scaling factor',
                           unit='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_I', initial_value=0,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5),
                           parameter_class=ManualParameter)


    def measure_rabi(self, amps, n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        if n != 1:
            raise NotImplementedError('QASM/QuMis sequence for n>1')

        # Generating the qumis file
        single_pulse_elt = sqqs.single_elt_on(self.name)
        single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name,
                                           self.get_operation_dict())
        qumis_file = single_pulse_asm
        self.CBox.load_instructions(qumis_file.name)
        for ch in [1, 2, 3, 4]:
            self.QWG.set('ch{}_amp'.format(ch), .45)
        ch_amp = swf.QWG_lutman_par(self.Q_LutMan,
                                    self.Q_LutMan.Q_amp180)

        d = qh.CBox_single_integration_average_det_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg(),
            seg_per_point=1)
        MC.set_sweep_function(ch_amp)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)

        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            a = ma.Rabi_Analysis(auto=True, close_fig=close_fig)
            return a

    def measure_motzoi(self, motzois, MC=None, analyze=True, close_fig=True,
                       verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Generating the qumis file
        motzoi_elt = sqqs.two_elt_MotzoiXY(self.name)
        single_pulse_asm = qta.qasm_to_asm(
            motzoi_elt.name, self.get_operation_dict())
        asm_file = single_pulse_asm
        self.CBox.load_instructions(asm_file.name)

        motzoi_swf = swf.QWG_lutman_par(self.Q_LutMan,
                                        self.Q_LutMan.Q_motzoi)

        d = qh.CBox_single_integration_average_det_CC(
            self.CBox, nr_averages=self.RO_acq_averages()//MC.soft_avg(),
            seg_per_point=2)

        MC.set_sweep_function(motzoi_swf)
        MC.set_sweep_points(np.repeat(motzois, 2))
        MC.set_detector_function(d)

        MC.run('Motzoi_XY'+self.msmt_suffix)
        if analyze:
            a = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
            return a

    def prepare_for_timedomain(self):
        self.MC.soft_avg(self.RO_soft_averages())
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.frequency.set(self.f_qubit.get()
                                     - self.f_pulse_mod.get())
        # self.CBox.trigger_source('internal')
        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.frequency.set(f_RO - self.f_RO_mod.get())

        self.td_source.power.set(self.td_source_pow.get())

        # Mixer offsets correction
        # self.CBox.set('AWG{:.0g}_dac0_offset'.format(self.awg_nr.get()),
        #               self.mixer_offs_drive_I.get())
        # self.CBox.set('AWG{:.0g}_dac1_offset'.format(self.awg_nr.get()),
        #               self.mixer_offs_drive_Q.get())
        # self.CBox.set('AWG{:.0g}_dac0_offset'.format(self.RO_awg_nr.get()),
        #               self.mixer_offs_RO_I.get())
        # self.CBox.set('AWG{:.0g}_dac1_offset'.format(self.RO_awg_nr.get()),
        #               self.mixer_offs_RO_Q.get())

        # RO pars
        if self.RO_LutMan != None:
            self.RO_LutMan.M_modulation(self.f_RO_mod())
            self.RO_LutMan.M_amp(self.RO_amp())
            self.RO_LutMan.M_length(self.RO_pulse_length())
            self.RO_LutMan.load_pulses_onto_AWG_lookuptable(self.RO_awg_nr())
            self.RO_LutMan.lut_mapping(
                ['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90', 'mY90', 'M_square'])
        self.CBox.upload_standard_weights(self.f_RO_mod())
        self.CBox.integration_length(
            convert_to_clocks(self.RO_acq_integration_length()))

        self.CBox.set('sig{}_threshold_line'.format(
                      int(self.signal_line.get())),
                      int(self.RO_threshold.get()))
        self.CBox.lin_trans_coeffs(
            np.reshape(rotation_matrix(self.RO_rotation_angle(), as_array=True), (4,)))

        self.load_QWG_pulses()

    def load_QWG_pulses(self):
        # NOTE: this is currently hardcoded to use ch1 and ch2 of the QWG

        t0 = time.time()
        self.QWG.reset()

        # Sets the QWG channel amplitudes
        for ch in [1, 2, 3, 4]:
            self.QWG.set('ch{}_amp'.format(ch), self.amp180()*1.1)
        # FIXME:  Currently hardcoded to use channel 1
        self.Q_LutMan.Q_amp180(self.amp180())
        self.Q_LutMan.Q_amp90_scale(self.amp90_scale())
        self.Q_LutMan.Q_motzoi(self.motzoi())

        self.Q_LutMan.load_pulses_onto_AWG_lookuptable()


        self.QWG.stop()
        predistortion_matrix = wf.mixer_predistortion_matrix(
            alpha=self.mixer_drive_alpha(),
            phi=self.mixer_drive_phi())

        self.QWG.ch_pair1_transform_matrix(predistortion_matrix)

        for ch in [1, 2, 3, 4]:
            self.QWG.set('ch{}_state'.format(ch), True)

        self.QWG.ch1_offset(self.mixer_offs_drive_I())
        self.QWG.ch2_offset(self.mixer_offs_drive_Q())

        self.QWG.ch_pair1_sideband_frequency(self.f_pulse_mod())
        self.QWG.ch_pair3_sideband_frequency(self.f_pulse_mod())
        self.QWG.syncSidebandGenerators()

        self.QWG.stop()
        self.QWG.run_mode('CODeword')
        self.QWG.start()
        # Check for errors at the end
        for i in range(self.QWG.getSystemErrorCount()):
            logging.warning(self.QWG.getError())
        t1 = time.time()
        logging.info('Initializing QWG took {:.2f}'.format(t1-t0))

    def get_operation_dict(self, operation_dict={}):
        """
        Returns a (currently hardcoded) operation dictionary for the QWG
        codewords
        """

        pulse_period_clocks = convert_to_clocks(
            max(self.gauss_width()*4, self.pulse_delay()))
        RO_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_acq_marker_del_clocks = convert_to_clocks(
            self.RO_acq_marker_delay())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())
        init_clocks = convert_to_clocks(self.init_time()/2)

        operation_dict['init_all'] = {
            'instruction':'\wait {} \wait {} \n'.format(
                init_clocks, init_clocks)}
        operation_dict['I {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction': 'wait {} \n'}
        operation_dict['X180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0000000, 2 \nwait 2\n' +
                'trigger 1000001, 2  \nwait {}\n'.format(  # 1001001
                    pulse_period_clocks-2)}
        operation_dict['Y180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0100000, 2 \nwait 2\n' +
                'trigger 1100001, 2  \nwait {}\n'.format(
                    pulse_period_clocks-2)}
        operation_dict['X90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0010000, 2 \nwait 2\n' +
                'trigger 1010000, 2  \nwait {}\n'.format(
                    pulse_period_clocks-2)}
        operation_dict['Y90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0110000, 2 \nwait 2\n' +
                'trigger 1110000, 2  \nwait {}\n'.format(
                    pulse_period_clocks-2)}
        operation_dict['mX90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0001000, 2 \nwait 2\n' +
                'trigger 1001000, 2  \nwait {}\n'.format(
                    pulse_period_clocks-2)}
        operation_dict['mY90 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0101000, 2 \nwait 2\n' +
                'trigger 1101000, 2  \nwait {}\n'.format(
                    pulse_period_clocks-2)}

        # RO part
        measure_instruction = ''
        acq_instr = self._get_acquisition_instr()
        if 'CBox' in acq_instr:
            measure_instruction = 'measure\n'
        elif (('ATS' in acq_instr) or ('UHFQC' in acq_instr)):
            # measure_instruction = 'trigger 0000001, {}\n'.format(RO_length_clocks)
            measure_instruction = 'trigger 1111111, {}\n'.format(RO_length_clocks)
        else:
            raise NotImplementedError('Unknown acquisition device.')

        if self.RO_pulse_type() == 'MW_IQmod_pulse':
            operation_dict['RO {}'.format(self.name)] = {
                'duration': (RO_pulse_delay_clocks+RO_acq_marker_del_clocks
                             + RO_depletion_clocks),
                'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                    RO_pulse_delay_clocks)
                + '\nwait {} \n{}'.format(
                    RO_acq_marker_del_clocks, measure_instruction)}
            # + 'wait {}\n'.format(RO_depletion_clocks)}
        elif self.RO_pulse_type() == 'Gated_MW_RO_pulse':
            operation_dict['RO {}'.format(self.name)] = {
                'duration': RO_length_clocks, 'instruction':
                'wait {} \n{}'.format(RO_pulse_delay_clocks,
                                      measure_instruction)}

        if RO_depletion_clocks != 0:
            operation_dict['RO {}'.format(self.name)]['instruction'] += \
                'wait {}\n'.format(RO_depletion_clocks)

        return operation_dict

    def _get_RO_instruction(operation_dict):

        return operation_dict