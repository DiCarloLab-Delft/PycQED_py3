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
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.tools.data_manipulation import rotation_matrix
from pycqed.measurement.calibration_toolbox import (
    mixer_carrier_cancellation, mixer_skewness_calibration_CBoxV3)

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs
import pycqed.measurement.CBox_sweep_functions as cbs
from pycqed.measurement.waveform_control_CC import qasm_helpers as qh
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib

from pycqed.measurement.waveform_control_CC.instruction_lib import convert_to_clocks


from pycqed.measurement import detector_functions as det
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter


class CBox_v3_driven_transmon(Transmon):

    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        '''
        Adds the parameters to the qubit insrument, it provides initial values
        for some parameters but not for all. Powers have to be set by hand as
        a safety measure.
        '''
        # MW-sources
        self.add_parameters()
        # Adding instrument parameters
        self.add_parameter('LO', parameter_class=InstrumentParameter)
        self.add_parameter('cw_source', parameter_class=InstrumentParameter)
        self.add_parameter('td_source', parameter_class=InstrumentParameter)
        self.add_parameter('IVVI', parameter_class=InstrumentParameter)
        self.add_parameter('Q_LutMan',
                           docstring='Lookuptable manager responsible for '
                           'microwave Q control pulses',
                           parameter_class=InstrumentParameter)
        self.add_parameter('RO_LutMan',
                           docstring='Lookuptable manager responsible for '
                           'microwave RO pulses',
                           parameter_class=InstrumentParameter)
        self.add_parameter('CBox', parameter_class=InstrumentParameter)
        self.add_parameter('MC', parameter_class=InstrumentParameter)
        self.add_parameter('RF_RO_source',
                           parameter_class=InstrumentParameter)

        # Overwriting some pars from the parent class
        self.RO_acq_marker_channel._vals = vals.Ints(1, 7)
        self.RO_acq_marker_channel(7)  # initial value
        # adding marker channels
        self.add_parameter('RO_acq_pulse_marker_channel',
                           vals=vals.Ints(1, 7),
                           initial_value=6,
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_marker_channel',
                           vals=vals.Ints(1, 7),
                           initial_value=5,
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pulse_type',
                           vals=vals.Enum('gated', 'square'),
                           initial_value='gated',
                           docstring=('Use either a marker gated spec pulse or' +
                                      ' use an AWG pulse to modulate a pulse'),
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_length',
                           label='Pulsed spec pulse duration',
                           unit='s',
                           vals=vals.Numbers(5e-9, 20e-6),
                           initial_value=500e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('spec_amp',
                           unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.4)

    def add_parameters(self):
        self.add_parameter('acquisition_instrument',
                           set_cmd=self._set_acquisition_instr,
                           get_cmd=self._get_acquisition_instr,
                           vals=vals.Strings())

        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_power_cw', label='RO power cw',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_I', initial_value=0,
                           vals=vals.Ints(0, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Ints(0, 5),
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
        self.add_parameter('Q_awg_nr', label='CBox awg nr', unit='#',
                           vals=vals.Ints(),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_awg_nr', label='CBox RO awg nr', unit='#',
                           vals=vals.Ints(),
                           initial_value=1,
                           parameter_class=ManualParameter)

        self.add_parameter('RO_acq_integration_length', initial_value=500e-9,
                           vals=vals.Numbers(min_value=0, max_value=20e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_soft_averages', initial_value=1,
                           vals=vals.Ints(min_value=1),
                           parameter_class=ManualParameter)

        self.add_parameter('Q_amp180',
                           label='Pi-pulse amplitude', unit='V',
                           initial_value=0.3,
                           parameter_class=ManualParameter)
        self.add_parameter('Q_amp90_scale',
                           label='pulse amplitude scaling factor',
                           unit='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
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
                           unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_drive_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        # Mixer offsets correction, RO pulse
        self.add_parameter('mixer_offs_RO_I', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_RO_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)

        self.add_parameter('RO_pulse_type', initial_value='Gated_UHFQC',
                           vals=vals.Enum(
                               'Gated_CBox', 'Gated_UHFQC',
                               'IQmod_CBox', 'IQmod_UHFQC',
                               'IQmod_multiplexed_UHFQC'),
                           parameter_class=ManualParameter)
        RO_acq_docstr = (
            'Determines what integration weights to use: '
            '\n\t SSB: Single sideband demodulation\n\t'
            'DSB: Double sideband demodulation\n\t'
            'optimal: do not upload anything relying on preset weights')
        self.add_parameter('RO_acq_weights',
                           initial_value='DSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=RO_acq_docstr,
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

        self.add_parameter('RO_optimal_weights_I',
                           vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_optimal_weights_Q',
                           vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)


    def prepare_for_continuous_wave(self):
        self.prepare_readout()

        # LO and RF for readout are turned on in prepare_readout
        self.td_source.get_instr().off()
        self.cw_source.get_instr().off()
        self.cw_source.get_instr().pulsemod_state.set('off')
        self.cw_source.get_instr().power.set(self.spec_pow.get())

    def prepare_readout(self):
        """
        Configures the readout. Consists of the following steps
        - create the relevant detector functions
        - set the microwave frequencies and sources
        - generate the RO pulse
        - set the integration weights
        """

        #####################################
        # Creating the detector functions
        #####################################
        # Setting the acquisition instrument also instantiates the RO detectors
        self.acquisition_instrument(self.acquisition_instrument())

        #####################################
        # Setting frequencies and MW sources
        #####################################
        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()

        self.LO.get_instr().frequency.set(f_RO - self.f_RO_mod.get())
        self.LO.get_instr().on()

        if "gated" in self.RO_pulse_type().lower():
            RF = self.RF_RO_source.get_instr()
            RF.power(self.RO_power_cw())
            RF.frequency(f_RO)
            RF.on()

        self.CBox.get_instr().trigger_source('internal')

        #####################################
        # Generating the RO pulse
        #####################################
        if 'CBox' in self.acquisition_instrument():
            if 'multiplexed' not in self.RO_pulse_type().lower():
                self.RO_LutMan.get_instr().M_modulation(self.f_RO_mod())
                self.RO_LutMan.get_instr().M_amp(self.RO_amp())
                self.RO_LutMan.get_instr().M_length(self.RO_pulse_length())

                if 'awg_nr' in self.RO_LutMan.get_instr().parameters:
                    self.RO_LutMan.get_instr().awg_nr(self.RO_awg_nr())

                if 'CBox' in self.acquisition_instrument():
                    self.CBox.get_instr().set('AWG{:.0g}_dac0_offset'.format(
                                              self.RO_awg_nr.get()),
                                              self.mixer_offs_RO_I.get())
                    self.CBox.get_instr().set('AWG{:.0g}_dac1_offset'.format(
                                              self.RO_awg_nr.get()),
                                              self.mixer_offs_RO_Q.get())
                    if self.RO_LutMan() is not None:
                        self.RO_LutMan.get_instr().lut_mapping(
                            ['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90', 'mY90', 'M_square'])

                    self.CBox.get_instr().integration_length(
                        convert_to_clocks(self.RO_acq_integration_length()))

                    self.CBox.get_instr().set('sig{}_threshold_line'.format(
                        int(self.signal_line.get())),
                        int(self.RO_threshold.get()))
                    self.CBox.get_instr().lin_trans_coeffs(
                        np.reshape(rotation_matrix(self.RO_rotation_angle(),
                                                   as_array=True), (4,)))

                    self.CBox.get_instr().set('sig{}_threshold_line'.format(
                        int(self.signal_line.get())),
                        int(self.RO_threshold.get()))
                self.RO_LutMan.get_instr().load_pulses_onto_AWG_lookuptable()

        elif 'UHFQC' in self.acquisition_instrument():
            if 'gated' in self.RO_pulse_type().lower():
                UHFQC = self._acquisition_instrument
                UHFQC.awg_sequence_acquisition()
            elif 'iqmod' in self.RO_pulse_type().lower():
                RO_lm = self.RO_LutMan.get_instr()
                RO_lm.M_length(self.RO_pulse_length())
                RO_lm.M_amp(self.RO_amp())
                RO_lm.M_length(self.RO_pulse_length())
                RO_lm.M_modulation(self.f_RO_mod())
                RO_lm.acquisition_delay(self.RO_acq_marker_delay())

                if 'multiplexed' not in self.RO_pulse_type().lower():
                    RO_lm.load_pulse_onto_AWG_lookuptable('M_square')


        #####################################
        # Setting The integration weights
        #####################################
        if 'CBox' in self.acquisition_instrument():
            if self.RO_acq_weights() == 'SSB':
                raise NotImplementedError('The CBox only supports DSB demodulation')
                # self.CBox.get_instr().upload_standard_weights(self.f_RO_mod())
            elif self.RO_acq_weights() == 'DSB':
                self.CBox.get_instr().upload_standard_weights(self.f_RO_mod())
            elif self.RO_acq_weights() == 'optimal':
                # if optimal weights are used the weights will not be set
                pass

        if 'UHFQC' in self.acquisition_instrument():
            UHFQC = self._acquisition_instrument
            if self.RO_acq_weights() == 'SSB':
                UHFQC.prepare_SSB_weight_and_rotation(
                    IF=self.f_RO_mod(),
                    weight_function_I=self.RO_acq_weight_function_I(),
                    weight_function_Q=self.RO_acq_weight_function_Q())
            elif self.RO_acq_weights() == 'DSB':
                UHFQC.prepare_DSB_weight_and_rotation(
                    IF=self.f_RO_mod(),
                    weight_function_I=self.RO_acq_weight_function_I(),
                    weight_function_Q=self.RO_acq_weight_function_Q())
            elif self.RO_acq_weights() == 'optimal':
                if (self.RO_optimal_weights_I() is None or 
                    self.RO_optimal_weights_Q() is None):
                    logging.warning('Optimal weights are None, not setting integration weights')
                else:  
                    UHFQC.set('quex_wint_weights_{}_real'
                              .format(self.RO_acq_weight_function_I()),
                              self.RO_optimal_weights_I())
                    UHFQC.set('quex_wint_weights_{}_imag'
                              .format(self.RO_acq_weight_function_I()),
                              self.RO_optimal_weights_Q())
                    UHFQC.set('quex_wint_weights_{}_real'
                              .format(self.RO_acq_weight_function_Q()),
                              self.RO_optimal_weights_Q())
                    UHFQC.set('quex_wint_weights_{}_imag'
                              .format(self.RO_acq_weight_function_Q()),
                              self.RO_optimal_weights_I())

                    UHFQC.set('quex_rot_{}_real'
                              .format(self.RO_acq_weight_function_I()), 1.0)
                    UHFQC.set('quex_rot_{}_imag'
                              .format(self.RO_acq_weight_function_I()), 1.0)
                    UHFQC.set('quex_rot_{}_real'
                              .format(self.RO_acq_weight_function_Q()), 1.0)
                    UHFQC.set('quex_rot_{}_imag'
                              .format(self.RO_acq_weight_function_Q()), -1.0)


    def prepare_for_timedomain(self):

        self.prepare_readout()
        self.MC.get_instr().soft_avg(self.RO_soft_averages())
        self.LO.get_instr().on()
        self.cw_source.get_instr().off()
        self.td_source.get_instr().on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.get_instr().frequency.set(self.f_qubit.get()
                                                 - self.f_pulse_mod.get())

        self.td_source.get_instr().power.set(self.td_source_pow.get())

        # Mixer offsets correction
        self.CBox.get_instr().set('AWG{:.0g}_dac0_offset'.format(self.Q_awg_nr.get()),
                                  self.mixer_offs_drive_I.get())
        self.CBox.get_instr().set('AWG{:.0g}_dac1_offset'.format(self.Q_awg_nr.get()),
                                  self.mixer_offs_drive_Q.get())
        # pulse pars
        self.Q_LutMan.get_instr().Q_amp180(self.Q_amp180())
        self.Q_LutMan.get_instr().Q_amp90(self.Q_amp90_scale()*self.Q_amp180())
        self.Q_LutMan.get_instr().Q_gauss_width.set(self.gauss_width.get())
        self.Q_LutMan.get_instr().Q_motzoi_parameter.set(self.motzoi.get())
        self.Q_LutMan.get_instr().Q_modulation.set(self.f_pulse_mod.get())

        # pulsed spec pars
        if self.spec_pulse_type() == 'square':
            self.Q_LutMan.get_instr().Q_ampCW(self.spec_amp())
            self.Q_LutMan.get_instr().Q_block_length(self.spec_pulse_length())

        self.Q_LutMan.get_instr().lut_mapping(
            ['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90', 'mY90', 'M_square'])

        # Mixer skewness correction
        self.Q_LutMan.get_instr().mixer_IQ_phase_skewness.set(0)
        self.Q_LutMan.get_instr().mixer_QI_amp_ratio.set(1)
        self.Q_LutMan.get_instr().mixer_apply_predistortion_matrix.set(True)
        self.Q_LutMan.get_instr().mixer_alpha.set(self.mixer_drive_alpha.get())
        self.Q_LutMan.get_instr().mixer_phi.set(self.mixer_drive_phi.get())

        if 'awg_nr' in self.Q_LutMan.get_instr().parameters:
            self.Q_LutMan.get_instr().awg_nr(self.Q_awg_nr())

        # Print this should be deleted

        self.Q_LutMan.get_instr().load_pulses_onto_AWG_lookuptable()


    def _get_acquisition_instr(self):
        return self._acquisition_instrument.name

    def _set_acquisition_instr(self, acq_instr_name):
        self._acquisition_instrument = self.find_instrument(acq_instr_name)
        if 'CBox' in acq_instr_name:
            logging.info("setting CBox acquisition")
            self.int_avg_det = qh.CBox_integrated_average_detector_CC(
                self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages()//self.RO_soft_averages())
            self.int_avg_det_rot = None  # FIXME: Not implemented
            self.int_log_det = qh.CBox_integration_logging_det_CC(self.CBox)
            self.input_average_detector = None  # FIXME: Not implemented
            self.int_avg_det_single = qh.CBox_single_integration_average_det_CC(
                self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages()//self.RO_soft_averages(),
                seg_per_point=1)

        elif 'UHFQC' in acq_instr_name:
            # Setting weights and pulses in the UHFQC
            # FIXME: option to disable when using optimized weights and
            # mulltiplexed RO to be added later
            UHFQC = self._acquisition_instrument

            logging.info("setting UHFQC acquisition")
            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=self._acquisition_instrument,
                AWG=self.CBox.get_instr(), nr_averages=self.RO_acq_averages())
            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.CBox.get_instr(),
                channels=[
                    self.RO_acq_weight_function_I(),
                    self.RO_acq_weight_function_Q()],
                nr_averages=self.RO_acq_averages(),
                real_imag=True,
                integration_length=self.RO_acq_integration_length())
            self.int_avg_det_single.detector_control = 'soft'

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.CBox.get_instr(),
                channels=[
                    self.RO_acq_weight_function_I(),
                    self.RO_acq_weight_function_Q()],
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length())
            self.int_avg_det_rot = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instrument, AWG=self.CBox.get_instr(),
                channels=[self.RO_acq_weight_function_I(),
                          self.RO_acq_weight_function_Q()],
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length(),
                rotate=True)
            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=self._acquisition_instrument, AWG=self.CBox.get_instr(),
                channels=[self.RO_acq_weight_function_I(),
                          self.RO_acq_weight_function_Q()],
                integration_length=self.RO_acq_integration_length())

        elif 'ATS' in acq_instr_name:
            logging.info("setting ATS acquisition")
            self.int_avg_det = det.ATS_integrated_average_continuous_detector(
                ATS=self._acquisition_instrument.card,
                ATS_acq=self._acquisition_instrument.controller,
                AWG=self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages())
        return

    def get_resetless_rb_detector(self, nr_cliff, starting_seed=1,
                                  nr_seeds='max', pulse_p_elt='min',
                                  MC=None,
                                  upload=True):
        raise NotImplementedError()
        # if MC is None:
        #     MC = self.MC.get_instr()

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
        #         rb_seq, self.Q_LutMan.get_instr().lut_mapping.get())
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
        #     AWG=self.AWG, CBox=self.CBox.get_instr(), upload=upload)

        # d = cdet.CBox_trace_error_fraction_detector(
        #     'Resetless rb det',
        #     MC=MC, AWG=self.AWG, CBox=self.CBox.get_instr(),
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
        #     MC = self.MC.get_instr()
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
        #         pw.wrap_par_to_swf(self.Q_LutMan.get_instr().motzoi_parameter))
        #     x0.append(motzoi_guess)
        #     init_steps.append(m_step)
        # if 'frequency' in parameters:
        #     sweep_functions.append(
        #         pw.wrap_par_to_swf(self.td_source.get_instr().frequency))
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

        awg_nr = self.Q_awg_nr()
        chI_par = self.CBox.get_instr().parameters[
            'AWG{}_dac{}_offset'.format(awg_nr, 0)]
        chQ_par = self.CBox.get_instr().parameters[
            'AWG{}_dac{}_offset'.format(awg_nr, 1)]
        offset_I, offset_Q = mixer_carrier_cancellation(
            SH=signal_hound, source=self.td_source.get_instr(), MC=self.MC.get_instr(),
            chI_par=chI_par, chQ_par=chQ_par)
        if update:
            self.mixer_offs_drive_I(offset_I)
            self.mixer_offs_drive_Q(offset_Q)

    def calibrate_mixer_offsets_RO(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        self.prepare_for_timedomain()
        # Only works if CBox is generating RO pulses
        awg_nr = self.RO_awg_nr()
        chI_par = self.CBox.get_instr().parameters[
            'AWG{}_dac{}_offset'.format(awg_nr, 0)]
        chQ_par = self.CBox.get_instr().parameters[
            'AWG{}_dac{}_offset'.format(awg_nr, 1)]
        offset_I, offset_Q = mixer_carrier_cancellation(
            SH=signal_hound, source=self.LO.get_instr(), MC=self.MC.get_instr(),
            chI_par=chI_par, chQ_par=chQ_par)
        if update:
            self.mixer_offs_drive_I(offset_I)
            self.mixer_offs_drive_Q(offset_Q)

    def calibrate_mixer_skewness(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness using mixer_skewness_cal_CBox_adaptive
        see calibration toolbox for details
        '''
        self.prepare_for_timedomain()
        phi, alpha = mixer_skewness_calibration_CBoxV3(
            name='mixer_skewness_cal'+self.msmt_suffix,
            CBox=self.CBox.get_instr(), SH=signal_hound,
            source=self.td_source.get_instr(),
            LutMan=self.Q_LutMan.get_instr(),
            f_mod=self.f_pulse_mod(),
            MC=self.MC.get_instr())
        if update:
            self.mixer_drive_phi.set(phi)
            self.mixer_drive_alpha.set(alpha)

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC.get_instr()

        # Loading the right qumis instructions
        CW_RO_sequence = sqqs.CW_RO_sequence(self.name,
                                             self.RO_acq_period_cw())
        CW_RO_sequence_asm = qta.qasm_to_asm(CW_RO_sequence.name,
                                             self.get_operation_dict())
        qumis_file = CW_RO_sequence_asm
        print(qumis_file.name)
        self.CBox.get_instr().load_instructions(qumis_file.name)
        self.CBox.get_instr().run_mode('run')

        # Create a new sweep function that sweeps two frequencies.
        # use only if the RO pulse type is the CW-RF scan, else only sweep the
        # LO

        # MC.set_sweep_function(self.heterodyne_instr.frequency)
        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep(
            RO_pulse_type=self.RO_pulse_type(),
            RF_source=self.RF_RO_source.get_instr(),
            LO_source=self.LO.get_instr(), IF=self.f_RO_mod()))
        MC.set_sweep_points(freqs)
        # make sure we use the right acquision detector. Mind the new UHFQC
        # spec mode
        MC.set_detector_function(self.int_avg_det_single)
        # det.Heterodyne_probe(self.heterodyne_instr)
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()
        self.cw_source.get_instr().on()
        if MC is None:
            MC = self.MC.get_instr()
        if pulsed:
            # Redirect to the pulsed spec function
            return self.measure_pulsed_spectroscopy(freqs,
                                                    MC, analyze, close_fig)
        MC.set_sweep_function(self.cw_source.get_instr().frequency)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='spectroscopy'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_pulsed_spectroscopy(self, freqs, MC=None,
                                    analyze=True, close_fig=True):
        # This is a trick so I can reuse the heterodyne instr
        # to do pulsed-spectroscopy

        if 'gated' in self.spec_pulse_type().lower():
            self.prepare_for_continuous_wave()
            if "gated" in self.RO_pulse_type().lower():
                self.RF_RO_source.get_instr().pulsemod_state('on')
            self.cw_source.get_instr().on()
            self.cw_source.get_instr().pulsemod_state.set('on')
            self.cw_source.get_instr().power.set(self.spec_pow_pulsed.get())
            sweep_par = self.cw_source.get_instr().frequency

        else:
            self.prepare_for_timedomain()
            sweep_par = self.td_source.get_instr().frequency

        if MC is None:
            MC = self.MC.get_instr()

        # Loading the right qumis instructions
        pulsed_spec_seq = sqqs.pulsed_spec_sequence(self.name)
        pulsed_spec_seq_asm = qta.qasm_to_asm(pulsed_spec_seq.name,
                                              self.get_operation_dict())
        qumis_file = pulsed_spec_seq_asm
        self.CBox.get_instr().load_instructions(qumis_file.name)
        self.CBox.get_instr().start()

        # make sure we use the right acquision detector. Mind the new UHFQC
        # spec mode
        MC.set_sweep_function(sweep_par)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='pulsed-spec'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_resonator_power(self, freqs, mod_amps,
                                MC=None, analyze=True, close_fig=True):
        '''
        N.B. This one does not use powers but varies the mod-amp.
        Need to find a way to keep this function agnostic to that
        '''
        raise NotImplementedError()
        # self.prepare_for_continuous_wave()
        # if MC is None:
        #     MC = self.MC.get_instr()
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
        #     MC = self.MC.get_instr()
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

    def measure_rabi(self, amps=np.linspace(-.5, .5, 21), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False, real_imag=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        if n != 1:
            raise NotImplementedError('QASM/QuMis sequence for n>1')

        # Generating the qumis file
        single_pulse_elt = sqqs.single_elt_on(self.name)
        single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name,
                                           self.get_operation_dict())
        qumis_file = single_pulse_asm
        self.CBox.get_instr().load_instructions(qumis_file.name)

        self.CBox.get_instr().start()
        amp_swf = cbs.Lutman_par_with_reload_single_pulse(
            LutMan=self.Q_LutMan.get_instr(),
            parameter=self.Q_LutMan.get_instr().Q_amp180,
            pulse_names=['X180'])
        d = self.int_avg_det_single
        try:
            d._set_real_imag(real_imag)
        except AttributeError:
            # FIXME: should be added for CBox detectors as well
            pass

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
            MC = self.MC.get_instr()

        # Generating the qumis file
        motzoi_elt = sqqs.two_elt_MotzoiXY(self.name)
        single_pulse_asm = qta.qasm_to_asm(
            motzoi_elt.name, self.get_operation_dict())
        asm_file = single_pulse_asm
        self.CBox.get_instr().load_instructions(asm_file.name)

        motzoi_swf = cbs.Lutman_par_with_reload_single_pulse(
            LutMan=self.Q_LutMan.get_instr(), parameter=self.Q_LutMan.get_instr(
            ).Q_motzoi_parameter,
            pulse_names=['X180', 'X90', 'Y180', 'Y90'])
        d = self.int_avg_det_single
        d.seg_per_point = 2
        d.detector_control = 'hard'

        # d = qh.CBox_single_integration_average_det_CC(
        #     self.CBox.get_instr(), nr_averages=self.RO_acq_averages()//MC.soft_avg(),
        #     seg_per_point=2)

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
            MC = self.MC.get_instr()
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
            'CBox': self.CBox.get_instr()}

        d = qh.CBox_int_avg_func_prep_det_CC(
            self.CBox.get_instr(), prepare_function=qh.load_range_of_asm_files,
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

        if freqs != None:
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
        self.CBox.get_instr().load_instructions(qumis_file.name)

        ch_amp = swf.QWG_lutman_par(
            self.Q_LutMan, self.Q_LutMan.get_instr().Q_amp180)
        motzoi = swf.QWG_lutman_par(
            self.Q_LutMan, self.Q_LutMan.get_instr().Q_motzoi)

        self.CBox.get_instr().log_length(log_length)
        self.MC.get_instr().set_sweep_function(ch_amp)
        self.MC.get_instr().set_sweep_function_2D(motzoi)
        self.MC.get_instr().set_sweep_points(amps)
        self.MC.get_instr().set_sweep_points_2D(motzois)
        self.MC.get_instr().set_detector_function(d)

        self.MC.get_instr().run('RB_amp_ncl{}_sds{}'.format(
            nr_cliffords, nr_seeds)+self.msmt_suffix, mode='2D')
        ma.TwoD_Analysis()

    def measure_T1(self, times, MC=None,
                   analyze=True, close_fig=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                                (times[-1]+times[0],
                                 times[-1]+times[1],
                                 times[-1]+times[2],
                                 times[-1]+times[3])])

        T1 = sqqs.T1(self.name, times=times)
        s = qh.QASM_Sweep(T1.name, self.CBox.get_instr(), self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = self.int_avg_det

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
            MC = self.MC.get_instr()

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
        self.td_source.get_instr().set('frequency', f_qubit - self.f_pulse_mod.get() +
                                       artificial_detuning)

        Ramsey = sqqs.Ramsey(
            self.name, times=times, artificial_detuning=None)
        s = qh.QASM_Sweep(Ramsey.name, self.CBox.get_instr(), self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = self.int_avg_det
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
            MC = self.MC.get_instr()

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
        s = qh.QASM_Sweep(echo.name, self.CBox.get_instr(), self.get_operation_dict(),
                          parameter_name='Time', unit='s')
        d = self.int_avg_det
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
            MC = self.MC.get_instr()

        AllXY = sqqs.AllXY(self.name, double_points=True)
        s = qh.QASM_Sweep(
            AllXY.name, self.CBox.get_instr(), self.get_operation_dict())
        d = self.int_avg_det
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
            MC = self.MC.get_instr()

        number_of_flips = np.concatenate([number_of_flips,
                                          (number_of_flips[-1]+number_of_flips[0],
                                           number_of_flips[-1] +
                                           number_of_flips[1],
                                           number_of_flips[-1] +
                                           number_of_flips[2],
                                           number_of_flips[-1]+number_of_flips[3])])
        flipping_sequence = sqqs.flipping_seq(self.name, number_of_flips,
                                              equator=equator)
        s = qh.QASM_Sweep(flipping_sequence.name, self.CBox.get_instr(),
                          self.get_operation_dict())
        d = self.int_avg_det

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
            MC = self.MC.get_instr()
        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        MC.soft_avg(1)  # don't want to average single shots
        # FIXME: remove when integrating UHFQC
        self.CBox.get_instr().log_length(1024*6)
        off_on = sqqs.off_on(self.name)
        s = qh.QASM_Sweep(off_on.name, self.CBox.get_instr(),
                          self.get_operation_dict(), parameter_name='Shots')
        d = self.int_log_det
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
            return a.F_a, a.F_d

    def measure_transients(self, MC=None, analyze=True):
        '''
        Measure transients.
        Returns two numpy arrays containing the transients for qubit in state
        |0> and |1>.
        '''
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        # Loading the right qumis instructions
        transients = []
        for i, pulse_comb in enumerate(['off', 'on']):
            off_on_sequence = sqqs.off_on(self.name, pulse_comb=pulse_comb)
            MC.set_sweep_function(swf.QASM_Sweep(
                filename=off_on_sequence.name, CBox=self.CBox.get_instr(),
                op_dict=self.get_operation_dict(),
                parameter_name='Samples', unit='#'))
            MC.set_sweep_points(
                np.arange(self.input_average_detector.nr_samples))
            MC.set_detector_function(self.input_average_detector)
            data = MC.run('Measure_transients{}_{}'.format(self.msmt_suffix, i))
            transients.append(data.T[1:])

            if analyze:
                ma.MeasurementAnalysis()

        return [np.array(t, dtype=np.float64) for t in transients]


    def calibrate_optimal_weights(self, MC=None, verify=True, update=True):
        if MC is None:
            MC = self.MC.get_instr()

        # return value needs to be added in measure_transients
        transients = self.measure_transients(MC=MC, analyze=False)
        # Calculate optimal weights
        optimized_weights_I = transients[1][0] - transients[0][0]
        optimized_weights_I = optimized_weights_I - np.mean(optimized_weights_I)
        weight_scale_factor = 1./np.max(np.abs(optimized_weights_I))
        optimized_weights_I = np.array(weight_scale_factor*optimized_weights_I)

        optimized_weights_Q = transients[1][1] - transients[0][1]
        optimized_weights_Q = optimized_weights_Q - np.mean(optimized_weights_Q)
        weight_scale_factor = 1./np.max(np.abs(optimized_weights_Q))
        optimized_weights_Q = np.array(weight_scale_factor*optimized_weights_Q)

        if update:
            self.RO_optimal_weights_I(optimized_weights_I)
            self.RO_optimal_weights_Q(optimized_weights_Q)
            self.RO_acq_weights('optimal')

        if verify:
            self.measure_ssro()


    def measure_butterfly(self, return_detector=False, MC=None,
                          analyze=True, close_fig=True,
                          verbose=True,
                          initialize=True, nr_shots=1024*24,
                          update_threshold=True):

        self.prepare_for_timedomain()
        if update_threshold:
            self.CBox.get_instr().lin_trans_coeffs(
                np.reshape(rotation_matrix(0, as_array=True), (4,)))
        if MC is None:
            MC = self.MC.get_instr()
        MC.soft_avg(1)
        # plotting slows down single shots
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)

        # Number of shots chosen to be a multiple of 6 as req for post select
        # FIXME: remove when integrating UHFQC
        self.CBox.get_instr().log_length(1024*6)
        qasm_file = sqqs.butterfly(self.name, initialize=initialize)
        s = qh.QASM_Sweep(qasm_file.name, self.CBox.get_instr(), self.get_operation_dict(),
                          parameter_name='Shots')

        # d = qh.CBox_integration_logging_det_CC(self.CBox)
        d = self.int_log_det
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
        #     MC = self.MC.get_instr()
        # if resetless:
        #     d = self.get_resetless_rb_detector(nr_cliff=nr_cliff)
        # else:
        #     raise NotImplementedError()
        # MC.set_detector_function(d)
        # MC.set_sweep_functions([cb_swf.LutMan_amp180_90(self.LutMan)])
        # MC.set_sweep_points(amps)
        # MC.run('RB-vs-amp_{}cliff'.format(nr_cliff) + self.msImt_suffix)
        # if analyze:
        #     ma.MeasurementAnalysis(close_fig=close_fig)

    def get_operation_dict(self, operation_dict={}):

        pulse_period_clocks = convert_to_clocks(
            self.gauss_width()*4+self.pulse_delay(), rounding_period=1/abs(self.f_pulse_mod()))
        RO_pulse_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())
        RO_acq_marker_del_clks = convert_to_clocks(self.RO_acq_marker_delay())

        operation_dict['init_all'] = {'instruction':
                                      '\nWaitReg r0 \nWaitReg r0 \n'}

        for cw_idx, pulse_name in enumerate(
                self.Q_LutMan.get_instr().lut_mapping()[:-1]):

            operation_dict['{} {}'.format(pulse_name, self.name)] = {
                'duration': pulse_period_clocks,
                'instruction': ins_lib.cbox_awg_pulse(
                    codeword=cw_idx, awg_channels=[self.Q_awg_nr()],
                    duration=pulse_period_clocks)}

        # Identity is a special instruction
        operation_dict['I {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction': 'wait {} \n'}

        spec_length_clocks = convert_to_clocks(
            self.spec_pulse_length())
        if self.spec_pulse_type() == 'gated':
            spec_instr = ins_lib.trigg_ch_to_instr(
                self.spec_pulse_marker_channel(), spec_length_clocks)
            operation_dict['SpecPulse '+self.name] = {
                'duration': spec_length_clocks, 'instruction': spec_instr}
        elif self.spec_pulse_type() == 'square':
            operation_dict['SpecPulse {}'.format(self.name)] = {
                'duration': spec_length_clocks, 'instruction':
                    'pulse 1111 0000 1111  \nwait {}\n'.format(
                        spec_length_clocks)}
        else:
            raise NotImplementedError

        if 'CBox' in self.acquisition_instrument():
            if self.RO_pulse_type() == 'IQmod_CBox':
                # operation_dict['RO {}'.format(self.name)] = {
                #     'duration': RO_pulse_length_clocks,
                #     'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                #         RO_pulse_delay_clocks)
                #     + '\nwait {} \nmeasure \n'.format(RO_pulse_length_clocks)}
                operation_dict['RO {}'.format(self.name)] = {
                    'duration': RO_pulse_length_clocks,
                    'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                        RO_pulse_delay_clocks)
                    + '\nwait {} \nmeasure \nwait {}\n'.format(
                        RO_acq_marker_del_clks,
                        max(RO_pulse_length_clocks-RO_acq_marker_del_clks, 0))}

            if self.RO_pulse_type() == 'IQmod_UHFQC':
                raise NotImplementedError

            elif self.RO_pulse_type() == 'Gated_CBox':
                operation_dict['RO {}'.format(self.name)] = {
                    'duration': RO_pulse_length_clocks, 'instruction':
                    'wait {} \ntrigger 1000000, {} \n measure \n'.format(
                        RO_pulse_delay_clocks, RO_pulse_length_clocks)}

            elif self.RO_pulse_type() == 'Gated_UHFQC':
                operation_dict['RO {}'.format(self.name)] = {
                    'duration': RO_pulse_length_clocks, 'instruction':
                    (ins_lib.trigg_ch_to_instr(self.RO_acq_marker_channel(),
                                               RO_pulse_length_clocks) +
                     'wait {}\n'.format(RO_pulse_delay_clocks) +
                     ins_lib.trigg_ch_to_instr(self.RO_acq_marker_channel(),
                                               2))}
                # UHF triggers on the rising edge

            if RO_depletion_clocks != 0:
                operation_dict['RO {}'.format(self.name)]['instruction'] += \
                    'wait {}\n'.format(RO_depletion_clocks)
        elif (('ATS' in self.acquisition_instrument()) or
              ('UHFQC' in self.acquisition_instrument())):
            if 'Gated' in self.RO_pulse_type():
                measure_instruction = self._gated_RO_marker_instr()
            else:
                measure_instruction = self._triggered_RO_marker_instr()

            operation_dict['RO {}'.format(self.name)] = {
                'duration': RO_pulse_length_clocks,
                'instruction': measure_instruction}
        else:
            raise NotImplementedError('Unknown acquisition device.')

        return operation_dict

    def _gated_RO_marker_instr(self):

        # Convert time to clocks
        RO_pulse_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_acq_marker_del_clocks = convert_to_clocks(
            self.RO_acq_marker_delay())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())

        # Define the timings
        t_RO_p = RO_pulse_delay_clocks
        t_acq_marker = t_RO_p + RO_acq_marker_del_clocks
        RO_p_len = RO_pulse_length_clocks
        acq_marker_len = 2
        t_RO_p_end = t_RO_p+RO_p_len
        t_acq_marker_end = t_acq_marker+acq_marker_len

        cw_p = ins_lib.trigg_cw(self.RO_acq_pulse_marker_channel())
        cw_t = ins_lib.trigg_cw(self.RO_acq_marker_channel())
        cw_both = ins_lib.bin_add_cw_w7(cw_p, cw_t)

        # Only works for a specific time arangement of the pulses
        if t_acq_marker > (t_RO_p + 1) and t_RO_p_end > t_acq_marker_end:
            instr = 'wait {} \n'.format(t_RO_p)
            instr += 'trigger {}, {}\n'.format(cw_p, t_acq_marker-t_RO_p)
            instr += 'wait {} \n'.format(t_acq_marker-t_RO_p)
            instr += 'trigger {}, {}\n'.format(cw_both, acq_marker_len)
            instr += 'wait {} \n'.format(acq_marker_len)
            instr += 'trigger {}, {}\n'.format(cw_p,
                                               t_RO_p_end-t_acq_marker_end)
            instr += 'wait {} \n'.format(RO_depletion_clocks +
                                         t_RO_p_end-t_acq_marker_end)

        else:
            raise NotImplementedError
        return instr

    def _triggered_RO_marker_instr(self):

        # Convert time to clocks
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_pulse_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())

        cw_t = ins_lib.trigg_cw(self.RO_acq_marker_channel())

        instr = 'wait {} \n'.format(RO_pulse_delay_clocks)
        instr += 'trigger {}, {}\n'.format(cw_t, 2)
        instr += 'wait {} \n'.format(RO_depletion_clocks
                                     + RO_pulse_length_clocks - 2)
        return instr


class QWG_driven_transmon(CBox_v3_driven_transmon):

    def __init__(self, name, **kw):
        super(CBox_v3_driven_transmon, self).__init__(name, **kw)
        '''
        '''
        self.add_parameter('LO', parameter_class=InstrumentParameter)
        self.add_parameter('cw_source', parameter_class=InstrumentParameter)
        self.add_parameter('td_source', parameter_class=InstrumentParameter)
        self.add_parameter('IVVI', parameter_class=InstrumentParameter)

        self.add_parameter('QWG', parameter_class=InstrumentParameter)
        self.add_parameter('CBox', parameter_class=InstrumentParameter)
        self.add_parameter('MC', parameter_class=InstrumentParameter)
        self.add_parameter('RO_LutMan', parameter_class=InstrumentParameter)
        self.add_parameter('Q_LutMan', parameter_class=InstrumentParameter)

    def add_parameters(self):
        super().add_parameters()
        # FIXME: chane amp90 scale to Q_amp90 scale
        self.add_parameter('amp90_scale',
                           label='pulse amplitude scaling factor',
                           unit='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
                           parameter_class=ManualParameter)

        self.add_parameter('RO_I_channel', initial_value='ch3',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_Q_channel', initial_value='ch4',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_type',
                           vals=vals.Enum('block', 'gauss'),
                           parameter_class=ManualParameter,
                           initial_value='block')
        self.add_parameter('spec_amp',
                           unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.4)
        self.add_parameter(
            'spec_length', vals=vals.Numbers(min_value=1e-9), unit='s',
            parameter_class=ManualParameter,
            docstring=('length of the block pulse if spec_pulse_type' +
                       'is "block", gauss_width if spec_pulse_type is gauss.'),
            initial_value=100e-9)

    def measure_rabi(self, amps=np.linspace(-.5, .5, 21), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        if n != 1:
            raise NotImplementedError('QASM/QuMis sequence for n>1')

        # Generating the qumis file
        single_pulse_elt = sqqs.single_elt_on(self.name)
        single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name,
                                           self.get_operation_dict())
        qumis_file = single_pulse_asm
        self.CBox.get_instr().load_instructions(qumis_file.name)

        for ch in [1, 2, 3, 4]:
            self.QWG.set('ch{}_amp'.format(ch), .45)
        ch_amp = swf.QWG_lutman_par(self.Q_LutMan,
                                    self.Q_LutMan.get_instr().Q_amp180)

        d = self.int_avg_det
        d.detector_control = 'soft'  # FIXME THIS overwrites something!

        self.CBox.get_instr().run_mode('run')
        MC.set_sweep_function(ch_amp)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)

        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        d.detector_control = 'hard'
        if analyze:
            a = ma.Rabi_Analysis(auto=True, close_fig=close_fig)
            return a

    def measure_motzoi(self, motzois, MC=None, analyze=True, close_fig=True,
                       verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        # Generating the qumis file
        motzoi_elt = sqqs.two_elt_MotzoiXY(self.name)
        single_pulse_asm = qta.qasm_to_asm(
            motzoi_elt.name, self.get_operation_dict())
        asm_file = single_pulse_asm
        self.CBox.get_instr().load_instructions(asm_file.name)

        motzoi_swf = swf.QWG_lutman_par(self.Q_LutMan,
                                        self.Q_LutMan.get_instr().Q_motzoi)

        d = self.int_avg_det_single

        # d = qh.CBox_single_integration_average_det_CC(
        #     self.CBox.get_instr(), nr_averages=self.RO_acq_averages()//MC.soft_avg(),
        #     seg_per_point=2)

        MC.set_sweep_function(motzoi_swf)
        MC.set_sweep_points(np.repeat(motzois, 2))
        MC.set_detector_function(d)

        MC.run('Motzoi_XY'+self.msmt_suffix)
        if analyze:
            a = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
            return a

    def prepare_for_timedomain(self):
        self.acquisition_instrument(self.acquisition_instrument())
        self.MC.get_instr().soft_avg(self.RO_soft_averages())
        self.LO.get_instr().on()
        self.cw_source.get_instr().off()
        self.td_source.get_instr().on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.get_instr().frequency.set(self.f_qubit.get()
                                                 - self.f_pulse_mod.get())
        # self.CBox.get_instr().trigger_source('internal')
        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.get_instr().frequency.set(f_RO - self.f_RO_mod.get())

        self.td_source.get_instr().power.set(self.td_source_pow.get())

        # Mixer offsets correction
        # self.CBox.get_instr().set('AWG{:.0g}_dac0_offset'.format(self.awg_nr.get()),
        #               self.mixer_offs_drive_I.get())
        # self.CBox.get_instr().set('AWG{:.0g}_dac1_offset'.format(self.awg_nr.get()),
        #               self.mixer_offs_drive_Q.get())
        # self.CBox.get_instr().set('AWG{:.0g}_dac0_offset'.format(self.RO_awg_nr.get()),
        #               self.mixer_offs_RO_I.get())
        # self.CBox.get_instr().set('AWG{:.0g}_dac1_offset'.format(self.RO_awg_nr.get()),
        #               self.mixer_offs_RO_Q.get())

        # RO pars
        if 'CBox' in self.acquisition_instrument():
            if self.RO_LutMan != None:
                self.RO_LutMan.M_modulation(self.f_RO_mod())
                self.RO_LutMan.M_amp(self.RO_amp())
                self.RO_LutMan.M_length(self.RO_pulse_length())
                self.RO_LutMan.lut_mapping(
                    ['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90', 'mY90', 'Mod_M'])
                self.RO_LutMan.load_pulses_onto_AWG_lookuptable()

            self.CBox.get_instr().upload_standard_weights(self.f_RO_mod())
            self.CBox.get_instr().integration_length(
                convert_to_clocks(self.RO_acq_integration_length()))

            self.CBox.get_instr().set('sig{}_threshold_line'.format(
                int(self.signal_line.get())),
                int(self.RO_threshold.get()))
            self.CBox.get_instr().lin_trans_coeffs(
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
        self.Q_LutMan.get_instr().Q_amp180(self.amp180())
        self.Q_LutMan.get_instr().Q_amp90_scale(self.amp90_scale())
        self.Q_LutMan.get_instr().Q_motzoi(self.motzoi())
        self.Q_LutMan.get_instr().Q_gauss_width(self.gauss_width())

        self.Q_LutMan.get_instr().spec_pulse_type(self.spec_pulse_type())
        self.Q_LutMan.get_instr().spec_amp(self.spec_amp())
        self.Q_LutMan.spec_length(self.spec_length())
        self.QWG.run_mode('CODeword')
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

        if self.spec_pulse_type() == 'gauss':
            spec_pulse_clocks = convert_to_clocks(self.spec_length()*4)
        elif self.spec_pulse_type() == 'block':
            spec_pulse_clocks = convert_to_clocks(self.spec_length())

        # should be able to delete this part
        RO_pulse_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_acq_marker_del_clocks = convert_to_clocks(
            self.RO_acq_marker_delay())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())
        init_clocks = convert_to_clocks(self.init_time()/2)

        operation_dict['init_all'] = {
            'instruction': 'wait {} \nwait {} \n'.format(
                init_clocks, init_clocks)}
        operation_dict['I {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction': 'wait {} \n'}
        operation_dict['X180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0000000, 2 \nwait 2\n' +
                'trigger 1000000, 2  \nwait {}\n'.format(  # 1001001
                    pulse_period_clocks-2)}
        operation_dict['Y180 {}'.format(self.name)] = {
            'duration': pulse_period_clocks, 'instruction':
                'trigger 0100000, 2 \nwait 2\n' +
                'trigger 1100000, 2  \nwait {}\n'.format(
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

        operation_dict['SpecPulse {}'.format(self.name)] = {
            'duration': spec_pulse_clocks, 'instruction':
                'trigger 0011000, 2 \nwait 2\n' +
                'trigger 1011000, 2  \nwait {}\n'.format(
                    spec_pulse_clocks-2)}

        # RO part
        measure_instruction = ''
        acq_instr = self._get_acquisition_instr()
        if 'CBox' in acq_instr:
            measure_instruction = 'measure\n'
            if self.RO_pulse_type() == 'MW_IQmod_pulse':
                operation_dict['RO {}'.format(self.name)] = {
                    'duration': (RO_pulse_delay_clocks+RO_acq_marker_del_clocks
                                 + RO_depletion_clocks),
                    'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                        RO_pulse_delay_clocks)
                    + '\nwait {} \n{}'.format(
                        RO_acq_marker_del_clocks, measure_instruction)}

            elif (self.RO_pulse_type() == 'Gated_MW_RO_pulse' or
                    self.RO_pulse_type() == 'IQmod_UHFQC'):
                operation_dict['RO {}'.format(self.name)] = {
                    'duration': RO_pulse_length_clocks, 'instruction':
                    'wait {} \n{}'.format(RO_pulse_delay_clocks,
                                          measure_instruction)}

            if RO_depletion_clocks != 0:
                operation_dict['RO {}'.format(self.name)]['instruction'] += \
                    'wait {}\n'.format(RO_depletion_clocks)

        elif (('ATS' in acq_instr) or ('UHFQC' in acq_instr)):
            if 'gated' in self.RO_pulse_type():
                measure_instruction = self._gated_RO_marker_instr()
                operation_dict['RO {}'.format(self.name)]['instruction'] = \
                    measure_instruction
            else:
                measure_instruction = self._triggered_RO_marker_instr()
                operation_dict['RO {}'.format(self.name)]['instruction'] = \
                    measure_instruction
        else:
            raise NotImplementedError('Unknown acquisition device.')

        return operation_dict
