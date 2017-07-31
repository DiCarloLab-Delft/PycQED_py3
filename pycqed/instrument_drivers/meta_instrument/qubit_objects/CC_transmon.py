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
import copy

import os
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx
from scipy.optimize import brent
from pycqed.measurement.optimization import nelder_mead
import pygsti
import pycqed

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

from pycqed.measurement.waveform_control_CC import QWG_fluxing_seqs as qwfs
from pycqed.measurement.waveform_control_CC.instruction_lib import convert_to_clocks

from pycqed.measurement import detector_functions as det
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC


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
        self.add_parameter('flux_LutMan',
                           docstring='Lookuptable manager responsible for '
                                     'flux pulses.',
                           initial_value=None,
                           parameter_class=InstrumentParameter)
        self.add_parameter('CBox', parameter_class=InstrumentParameter)
        self.add_parameter('MC', parameter_class=InstrumentParameter)
        self.add_parameter('RF_RO_source',
                           parameter_class=InstrumentParameter)
        self.add_parameter('SH', label='SignalHound',
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
        self.add_parameter('qasm_config',
                           docstring='used for generating qumis instructions',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())

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
        self.add_parameter('RO_digitized', vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter)
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

    def upload_qasm_file(self, qasm_file):
        qasm_fn = qasm_file.name
        cfg = self.qasm_config()
        self.CBox.get_instr().trigger_source('internal')
        qasm_folder, fn = os.path.split(qasm_fn)
        base_fn = fn.split('.')[0]
        qumis_fn = os.path.join(qasm_folder, base_fn + ".qumis")
        compiler = qcx.QASM_QuMIS_Compiler(
            verbosity_level=1)
        compiler.compile(qasm_fn, qumis_fn=qumis_fn,
                         config=cfg)
        self.CBox.get_instr().load_instructions(qumis_fn)

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
                raise NotImplementedError(
                    'The CBox only supports DSB demodulation')
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
                    logging.warning('Optimal weights are None,' +
                                    ' not setting integration weights')
                else:
                    # When optimal weights are used, only the RO I weight
                    # channel is used
                    UHFQC.set('quex_wint_weights_{}_real'.format(
                        self.RO_acq_weight_function_I()),
                        self.RO_optimal_weights_I())
                    UHFQC.set('quex_wint_weights_{}_imag'.format(
                        self.RO_acq_weight_function_I()),
                        self.RO_optimal_weights_Q())
                    UHFQC.set('quex_rot_{}_real'.format(
                        self.RO_acq_weight_function_I()), 1.0)

                    UHFQC.set('quex_rot_{}_imag'.format(
                        self.RO_acq_weight_function_I()), -1.0)

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

    def prepare_for_fluxing(self, reset=True):
        '''
        Genereates flux pulses, loads the to the QWG, and sets the QWG mode.
        '''
        f_lutman = self.flux_LutMan.get_instr()
        QWG = f_lutman.QWG.get_instr()

        ch_amp = QWG.get('ch{}_amp'.format(f_lutman.F_ch()))
        if reset:
            QWG.reset()
        # QWG stop and start happens inside the above function
        QWG.run_mode('CODeword')
        QWG.set('ch{}_amp'.format(f_lutman.F_ch()), ch_amp)
        QWG.set('ch{}_state'.format(f_lutman.F_ch()), True)
        f_lutman.load_pulses_onto_AWG_lookuptable()

    def _get_acquisition_instr(self):
        return self._acquisition_instrument.name

    def _set_acquisition_instr(self, acq_instr_name):
        self._acquisition_instrument = self.find_instrument(acq_instr_name)
        if 'CBox' in acq_instr_name:
            logging.info("setting CBox acquisition")
            self.int_avg_det = qh.CBox_integrated_average_detector_CC(
                self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages()//self.RO_soft_averages())
            self.int_log_det = qh.CBox_integration_logging_det_CC(
                self.CBox.get_instr())
            self.input_average_detector = qh.CBox_input_average_detector_CC(
                CBox=self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages()//self.RO_soft_averages())
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

            if self.RO_acq_weights() == 'optimal':
                RO_channels = [self.RO_acq_weight_function_I()]
                result_logging_mode = 'lin_trans'

                if self.RO_digitized():
                    result_logging_mode = 'digitized'
                # Update the RO theshold
                acq_ch = self.RO_acq_weight_function_I()

                # The threshold that is set in the hardware  needs to be
                # corrected for the offset as this is only applied in
                # software.
                threshold = self.RO_threshold()
                offs = self._acquisition_instrument.get(
                    'quex_trans_offset_weightfunction_{}'.format(acq_ch))
                hw_threshold = threshold + offs
                self._acquisition_instrument.set(
                    'quex_thres_{}_level'.format(acq_ch), hw_threshold)

            else:
                RO_channels = [self.RO_acq_weight_function_I(),
                               self.RO_acq_weight_function_Q()]
                result_logging_mode = 'raw'

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages())

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.CBox.get_instr(),
                channels=RO_channels,
                result_logging_mode=result_logging_mode,
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length())

            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.CBox.get_instr(),
                channels=RO_channels,
                result_logging_mode=result_logging_mode,
                nr_averages=self.RO_acq_averages(),
                real_imag=True, single_int_avg=True,
                integration_length=self.RO_acq_integration_length())

            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=UHFQC, AWG=self.CBox.get_instr(),
                channels=RO_channels,
                result_logging_mode=result_logging_mode,
                integration_length=self.RO_acq_integration_length())

        elif 'ATS' in acq_instr_name:
            logging.info("setting ATS acquisition")
            self.int_avg_det = det.ATS_integrated_average_continuous_detector(
                ATS=self._acquisition_instrument.card,
                ATS_acq=self._acquisition_instrument.controller,
                AWG=self.CBox.get_instr(),
                nr_averages=self.RO_acq_averages())
        return

    def calibrate_pulse_restless(self, nr_cliff=30,
                                 parameters=['amp', 'motzoi', 'frequency'],
                                 amp_guess=None, motzoi_guess=None,
                                 frequency_guess=None,
                                 nr_seeds='max', nr_shots = 16000,
                                 a_step=.30, m_step=.1, f_step=20e3,
                                 MC=None,
                                 update=False, close_fig=True,
                                 verbose=True, upload_qasm_file=True):
        '''
        Calibrates single qubit pulse parameters currently only using
        the resetless rb method (requires reasonable (~80%+) discrimination
        fidelity)

        If it there is only one parameter to sweep it will use brent's method
        instead.

        The function returns the values it found for the optimization.
        '''
        assert(self.RO_digitized()) # RO must be digitized for this mode
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        # Estimate the number of seeds allowed
        safety_factor = 5 if nr_cliff < 8 else 3
        pulse_p_elt = int(safety_factor*nr_cliff)
        if nr_seeds == 'max':
            nr_seeds = 29184//pulse_p_elt
        if nr_seeds*pulse_p_elt > 29184:
            raise ValueError(
                'Too many pulses ({}), {} seeds, {} pulse_p_elt'.format(
                    nr_seeds*pulse_p_elt, nr_seeds, pulse_p_elt))

        name = 'RB_{}cl_numerical'.format(nr_cliff)

        qasm_file = sqqs.randomized_benchmarking(
            self.name, nr_cliffords=[nr_cliff],
            nr_seeds=nr_seeds,
            net_clifford=3, restless=True,
            cal_points=False)
        if upload_qasm_file:
            self.upload_qasm_file(qasm_file)

        if amp_guess is None:
            amp_guess = self.Q_amp180.get()
        if motzoi_guess is None:
            motzoi_guess = self.motzoi.get()
        if frequency_guess is None:
            frequency_guess = self.f_qubit.get()
        # # Because we are sweeping the source and not the qubit frequency
        start_freq = frequency_guess - self.f_pulse_mod.get()

        sweep_functions = []
        x0 = []
        init_steps = []
        if 'amp' in parameters:
            sweep_functions.append(cbs.LutMan_amp180_90(self.Q_LutMan.get_instr()))
            x0.append(amp_guess)
            init_steps.append(a_step)
        if 'motzoi' in parameters:
            sweep_functions.append(self.Q_LutMan.get_instr().Q_motzoi_parameter)
            x0.append(motzoi_guess)
            init_steps.append(m_step)
        if 'frequency' in parameters:
            sweep_functions.append(self.td_source.get_instr().frequency)
            x0.append(start_freq)
            init_steps.append(f_step)
        if len(sweep_functions) == 0:
            raise ValueError(
                'parameters "{}" not recognized'.format(parameters))

        MC.set_sweep_functions(sweep_functions)

        if len(sweep_functions) != 1:
            # noise ensures no_improv_break sets the termination condition
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'par_idx': 1,  # idx of flips in the stat log det
                            'x0': x0,
                            'initial_step': init_steps,
                            'no_improv_break': 10,
                            'minimize': False,
                            'maxiter': 500}
        elif len(sweep_functions) == 1:
            # Powell does not work for 1D, use brent instead
            brack = (x0[0]-5*init_steps[0], x0[0])
            # Ensures relative change in parameter is relevant
            if parameters == ['frequency']:
                tol = 1e-9
            else:
                tol = 1e-3
            print('Tolerance:', tol, init_steps[0])
            print(brack)
            ad_func_pars = {'adaptive_function': brent,
                            'par_idx': 1,  # idx of flips in the stat log det
                            'brack': brack,
                            'tol': tol,  # Relative tolerance in brent
                            'minimize': False}

        # a bit hacky
        UHFQC = self._acquisition_instrument
        d = det.UHFQC_single_qubit_statistics_logging_det(
                UHFQC=UHFQC, AWG=self.CBox.get_instr(),
                nr_shots=nr_shots,
                integration_length=self.RO_acq_integration_length(),
                channel=(self.RO_acq_weight_function_I()),
                statemap={'0': '1',
                          '1': '0'})
            #  hardcoded statemap does not make sense but I only use flips
            #  here anyway so it doesn't really matter...

        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.set_detector_function(d)
        MC.run(name=name, mode='adaptive')
        if len(sweep_functions) != 1:
            a = ma.OptimizationAnalysis(auto=True, label=name,
                                        close_fig=close_fig)
            if len(sweep_functions) == 2:
                ma.OptimizationAnalysis_v2(label=name)
            if verbose:
                # Note printing can be made prettier
                print('Optimization converged to:')
                print('parameters: {}'.format(parameters))
                print(a.optimization_result[0])
            if update:
                for i, par in enumerate(parameters):
                    if par == 'amp':
                        self.Q_amp180.set(a.optimization_result[0][i])
                    elif par == 'motzoi':
                        self.motzoi.set(a.optimization_result[0][i])
                    elif par == 'frequency':
                        self.f_qubit.set(a.optimization_result[0][i] +
                                         self.f_pulse_mod.get())
            return a
        else:
            a = ma.MeasurementAnalysis(label=name, close_fig=close_fig)
            print('Optimization for {} converged to: {}'.format(
                parameters[0], a.sweep_points[-1]))
            if update:
                if parameters == ['amp']:
                    self.Q_amp180.set(a.sweep_points[-1])
                elif parameters == ['motzoi']:
                    self.motzoi.set(a.sweep_points[-1])
                elif parameters == ['frequency']:
                    self.f_qubit.set(a.sweep_points[-1]+self.f_pulse_mod.get())
            return a.sweep_points[-1]

    def calibrate_mixer_offsets_drive(self, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        # ensures freq is set correctly
        self.prepare_for_timedomain()

        signal_hound = self.SH.get_instr()
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
        return True

    def calibrate_mixer_offsets_RO(self, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        self.prepare_for_timedomain()

        signal_hound = self.SH.get_instr()
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
        return True

    def calibrate_mixer_skewness(self, update=True):
        '''
        Calibrates the mixer skewness using mixer_skewness_cal_CBox_adaptive
        see calibration toolbox for details
        '''
        signal_hound = self.SH.get_instr()
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
        return True

    def calibrate_MW_RO_latency(self, MC=None, update: bool=True,
                                soft_avg: int=100)-> bool:
        # docstring is in parent class

        if 'UHFQC' not in self.acquisition_instrument():
            # N.B. made only for UHFQC
            raise NotImplementedError
        if MC is None:
            MC = self.MC.get_instr()

        self.prepare_for_timedomain()
        # when using direct drive lines a lot of averages are required
        MC.soft_avg(soft_avg)

        # set the RO LO to the drive frequency, this way the dynamics of the
        # resonator do not show up in the timing experiment but both pulses
        # MW and RO do show up in the measured trace.
        self.LO.get_instr().frequency(self.f_qubit())

        # make the RO pulse short so it is clearly visible in the trace
        self.RO_LutMan.get_instr().M_length(100e-9)
        # reupload the modified pulse to the RO lutman
        self.RO_LutMan.get_instr().load_pulse_onto_AWG_lookuptable('M_square')

        old_max_pt = MC.plotting_max_pts()
        MC.plotting_max_pts(4100)  # increase as live plot is desired here
        self.measure_transients(MC=MC, cases=['sim_on'], prepare=False,
                                analyze=True)
        MC.plotting_max_pts(old_max_pt)

        print('Manual analysis and updating of parameters is required.')
        print('Be sure to update: \n\tMW_latency in config' +
              '\n\tRO_acq_marker_delay')
        return False

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

        # FIXME: setting polar coords should be fixed properly
        self.int_avg_det_single._set_real_imag(False)
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
        self.int_avg_det_single._set_real_imag(False)
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

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep(
            RO_pulse_type=self.RO_pulse_type(),
            RF_source=self.RF_RO_source.get_instr(),
            LO_source=self.LO.get_instr(), IF=self.f_RO_mod()))
        MC.set_sweep_points(freqs)

        MC.set_sweep_function_2D(
            self.IVVI.get_instr().parameters[
                'dac{}'.format(self.dac_channel.get())])

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_voltages)
        self.int_avg_det_single._set_real_imag(False)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_dac_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_rabi(self, amps=np.linspace(-.5, .5, 21), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False, real_imag=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        # Generating the qumis file
        single_pulse_elt = sqqs.single_elt_on(self.name, n=n)
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

    def measure_motzoi(self, motzois=np.linspace(-.3, .3, 31),
                       MC=None, analyze=True, close_fig=True):
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
        if 'UHFQC' in self.acquisition_instrument.name:
            d._set_real_imag(True)

        MC.set_sweep_function(motzoi_swf)
        MC.set_sweep_points(np.repeat(motzois, 2))
        MC.set_detector_function(d)

        MC.run('Motzoi_XY'+self.msmt_suffix)
        if analyze:
            a = ma.Motzoi_XY_analysis(
                auto=True, cal_points=None, close_fig=close_fig)
            return a

    def measure_randomized_benchmarking(self, nr_cliffords=2**np.arange(12),
                                        nr_seeds=100, T1=None,
                                        MC=None, analyze=True, close_fig=True,
                                        verbose=False, upload=True,
                                        update=True):
        # Adding calibration points
        nr_cliffords = np.append(
            nr_cliffords, [nr_cliffords[-1]+.5]*2 + [nr_cliffords[-1]+1.5]*2)

        # if 'CBox' not in self.acquisition_instrument():
        #     raise NotImplementedError
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        MC.soft_avg(nr_seeds)
        counter_param = ManualParameter('name_ctr', initial_value=0)
        asm_filenames = []
        for i in range(nr_seeds):
            RB_qasm = sqqs.randomized_benchmarking(
                self.name,
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

        if 'CBox' in self.acquisition_instrument():
            d = qh.CBox_int_avg_func_prep_det_CC(
                self.CBox.get_instr(), prepare_function=qh.load_range_of_asm_files,
                prepare_function_kwargs=prepare_function_kwargs,
                nr_averages=256)
        elif 'UHFQC' in self.acquisition_instrument():
            d = qh.UHFQC_int_avg_func_prep_det_CC(
                prepare_function=qh.load_range_of_asm_files,
                prepare_function_kwargs=prepare_function_kwargs,
                UHFQC=self._acquisition_instrument, AWG=self.CBox.get_instr(),
                channels=[
                    self.RO_acq_weight_function_I(),
                    self.RO_acq_weight_function_Q()],
                integration_length=self.RO_acq_integration_length(),
                nr_averages=256)

        s = swf.None_Sweep()
        s.parameter_name = 'Number of Cliffords'
        s.unit = '#'
        MC.set_sweep_function(s)
        MC.set_sweep_points(nr_cliffords)

        MC.set_detector_function(d)
        MC.run('RB_{}seeds'.format(nr_seeds)+self.msmt_suffix)
        a = ma.RandomizedBenchmarking_Analysis(
            close_main_fig=close_fig, T1=T1,
            pulse_delay=self.gauss_width.get()*4)
        if update:
            self.F_RB(a.fit_res.params['fidelity_per_Clifford'].value)

        return a.fit_res.params['fidelity_per_Clifford'].value

    def measure_randomized_benchmarking_vs_pars(self, amps=None,
                                                motzois=None, freqs=None,
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
        RB_elt = sqqs.randomized_benchmarking(
            self.name, nr_cliffords=[nr_cliffords], cal_points=False,
            nr_seeds=nr_seeds, double_curves=False,
            net_clifford=net_clifford, restless=restless)
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

    def measure_T1(self, times=None, MC=None,
                   close_fig=True, update=True):

        if times is None:
            times = np.linspace(0, self.T1()*4, 61)
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
        s = swf.QASM_Sweep(T1.name, self.CBox.get_instr(),
                           self.get_operation_dict(),
                           parameter_name='Time', unit='s')
        d = self.int_avg_det

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)

        MC.run('T1'+self.msmt_suffix)

        a = ma.T1_Analysis(auto=True, close_fig=True)
        if update:
            self.T1(a.T1)
        return a.T1

    def measure_ramsey(self, times=None, artificial_detuning=None,
                       f_qubit=None, label='',
                       MC=None, analyze=True, close_fig=True, verbose=True,
                       update=True):
        """
        N.B. if the artificial detuning is None it will auto set it such that
        3 oscillations will show.
        """
        if times is None:
            # funny default is because CBox has no real time sideband
            # modulation
            stepsize = (self.T2_star()*4/61)//(1/abs(self.f_pulse_mod())) \
                / abs(self.f_pulse_mod())
            times = np.arange(0, self.T2_star()*4, stepsize)
        if artificial_detuning is None:
            artificial_detuning = 3/times[-1]

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
        s = swf.QASM_Sweep(Ramsey.name, self.CBox.get_instr(), self.get_operation_dict(),
                           parameter_name='Time', unit='s')
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('Ramsey'+label+self.msmt_suffix)
        if analyze:
            a = ma.Ramsey_Analysis(auto=True, close_fig=True)
            if update:
                self.T2_star(a.T2_star)
            if verbose:
                fitted_freq = a.fit_res.params['frequency'].value
                print('Artificial detuning: {:.2e}'.format(
                      artificial_detuning))
                print('Fitted detuning: {:.2e}'.format(fitted_freq))
                print('Actual detuning:{:.2e}'.format(
                      fitted_freq-artificial_detuning))
            return a

    def measure_echo(self, times=None, artificial_detuning=0,
                     label='', MC=None, analyze=True, close_fig=True,
                     update=True, verbose=True):
        if times == None:
            # funny default is because CBox has no real time sideband
            # modulation
            stepsize = (self.T2_echo()*4/61)//(1/abs(self.f_pulse_mod())) \
                / abs(self.f_pulse_mod())
            times = np.arange(0, self.T2_echo()*4, stepsize)

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
        s = swf.QASM_Sweep(echo.name, self.CBox.get_instr(), self.get_operation_dict(),
                           parameter_name='Time', unit='s')
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('echo'+label+self.msmt_suffix)
        if analyze:
            a = ma.Echo_analysis(auto=True, close_fig=True)
            if update:
                self.T2_echo(a.fit_res.params['tau'].value)
            return a

    def measure_allxy(self, MC=None, label='',
                      analyze=True, close_fig=True, verbose=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        AllXY = sqqs.AllXY(self.name, double_points=True)
        s = swf.QASM_Sweep(
            AllXY.name, self.CBox.get_instr(), self.get_operation_dict())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('AllXY'+label+self.msmt_suffix)
        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a

    def measure_flipping(self, number_of_flips=2*np.arange(60),
                         MC=None, label='', equator=True,
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
        s = swf.QASM_Sweep(flipping_sequence.name, self.CBox.get_instr(),
                           self.get_operation_dict())
        d = self.int_avg_det

        MC.set_sweep_function(s)
        MC.set_sweep_points(number_of_flips)
        MC.set_detector_function(d)
        MC.run('flipping_sequence'+label+self.msmt_suffix)
        if analyze:
            a = ma.DriveDetuning_Analysis(label='flipping_sequence')
            return a

    def measure_ssro(self, no_fits=False,
                     return_detector=False,
                     MC=None, nr_shots=1024*24,
                     analyze=True, verbose=True, update_threshold=True,
                     update=True):

        # This ensures that the detector is not digitized for the SSRO
        # experiment
        old_RO_digit = self.RO_digitized()
        self.RO_digitized(False)
        self.prepare_for_timedomain()
        self.RO_digitized(old_RO_digit)

        if MC is None:
            MC = self.MC.get_instr()
        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        MC.soft_avg(1)  # don't want to average single shots
        # FIXME: remove when integrating UHFQC
        self.CBox.get_instr().log_length(1024*6)
        off_on = sqqs.off_on(self.name)
        s = swf.QASM_Sweep_v2(qasm_fn=off_on.name,
                              config=self.qasm_config(),
                              CBox=self.CBox.get_instr(),
                              verbosity_level=1,
                              parameter_name='Shots', unit='#')

        d = self.int_log_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run('SSRO'+self.msmt_suffix)
        # turn plotting back on
        MC.live_plot_enabled(old_plot_setting)
        if analyze:
            a = ma.SSRO_Analysis(label='SSRO'+self.msmt_suffix,
                                 channels=d.value_names,
                                 no_fits=no_fits)
            if update_threshold:
                # use the threshold for the best discrimination fidelity
                self.RO_threshold(a.V_th_d)
            if update:
                self.F_ssro(a.F_a)
                self.F_discr(a.F_d)
            if verbose:
                print('Avg. Assignement fidelity: \t{:.4f}\n'.format(a.F_a) +
                      'Avg. Discrimination fidelity: \t{:.4f}'.format(a.F_d))

            return a.F_a, a.F_d

    def measure_transients(self, MC=None, analyze: bool=True,
                           cases=('off', 'on'),
                           prepare: bool=True):
        '''
        Measure transients.
        Returns two numpy arrays containing the transients for qubit in state
        |0> and |1>.
        '''
        if prepare:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        # Loading the right qumis instructions
        transients = []
        for i, pulse_comb in enumerate(cases):
            off_on_sequence = sqqs.off_on(self.name, pulse_comb=pulse_comb)
            s = swf.QASM_Sweep_v2(qasm_fn=off_on_sequence.name,
                                  config=self.qasm_config(),
                                  CBox=self.CBox.get_instr(),
                                  verbosity_level=1,
                                  parameter_name='Transient time', unit='s')
            MC.set_sweep_function(s)

            if 'UHFQC' in self.acquisition_instrument():
                sampling_rate = 1.8e9
            elif 'CBox' in self.acquisition_instrument():
                sampling_rate = 200e6
            MC.set_sweep_points(
                np.arange(self.input_average_detector.nr_samples)/sampling_rate)
            MC.set_detector_function(self.input_average_detector)
            data = MC.run(
                'Measure_transients{}_{}'.format(self.msmt_suffix, i))
            dset = data['dset']
            transients.append(dset.T[1:])
            if analyze:
                ma.MeasurementAnalysis()

        return [np.array(t, dtype=np.float64) for t in transients]

    def calibrate_optimal_weights(self, MC=None, verify=True, analyze=False,
                                  update=True):
        if MC is None:
            MC = self.MC.get_instr()

        # Ensure that enough averages are used to get accurate weights
        old_avg = self.RO_acq_averages()
        self.RO_acq_averages(2**15)
        transients = self.measure_transients(MC=MC, analyze=analyze)

        self.RO_acq_averages(old_avg)

        # Calculate optimal weights
        optimized_weights_I = transients[1][0] - transients[0][0]
        optimized_weights_Q = transients[1][1] - transients[0][1]

        # joint rescaling to +/-1 Volt
        maxI = np.max(np.abs(optimized_weights_I))
        maxQ = np.max(np.abs(optimized_weights_Q))
        weight_scale_factor = 1./np.max([maxI, maxQ])
        optimized_weights_I = np.array(
            weight_scale_factor*optimized_weights_I)
        optimized_weights_Q = np.array(
            weight_scale_factor*optimized_weights_Q)

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
                          update_threshold=False):

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
        # s = swf.QASM_Sweep(qasm_file.name, self.CBox.get_instr(),
        #                    self.get_operation_dict(),
        #                    parameter_name='Shots')
        s = swf.QASM_Sweep_v2(qasm_fn=qasm_file.name,
                              config=self.qasm_config(),
                              CBox=self.CBox.get_instr(),
                              verbosity_level=1,
                              parameter_name='Shots')
        # d = qh.CBox_integration_logging_det_CC(self.CBox)
        d = self.int_log_det
        d.nr_shots = 4092  # multiple both of 4 and 6

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)

        MC.set_detector_function(self.int_log_det)
        MC.run('Butterfly{}_initialize_{}'.format(
            self.msmt_suffix, initialize))
        # turn plotting back on
        MC.live_plot_enabled(old_plot_setting)

        if self.RO_digitized():
            c = ma.butterfly_analysis(
                close_main_fig=False, initialize=initialize,
                threshold=0.5, digitize=False, case=True)
            return c.butterfly_coeffs
        else:
            if self.RO_acq_weights() != 'optimal':
                # first perform SSRO analysis to extract the optimal rotation angle
                # theta
                a = ma.SSRO_discrimination_analysis(
                    label='Butterfly',
                    current_threshold=None,
                    close_fig=close_fig,
                    plot_2D_histograms=True)

                # the, run it a second time to determine the optimal
                # threshold along the rotated I axis
                b = ma.SSRO_discrimination_analysis(
                    label='Butterfly',
                    current_threshold=None,
                    close_fig=close_fig,
                    plot_2D_histograms=True, theta_in=-a.theta)
                threshold = b.opt_I_threshold
                theta = b.theta

            elif self.RO_acq_weights() == 'optimal':
                a = ma.SSRO_single_quadrature_discriminiation_analysis()
                threshold = a.opt_threshold
                theta = 0

            c0 = ma.butterfly_analysis(
                close_main_fig=close_fig, initialize=initialize,
                theta_in=-theta % 360,
                threshold=threshold, digitize=True, case=False)
            c1 = ma.butterfly_analysis(
                close_main_fig=close_fig, initialize=initialize,
                theta_in=-theta % 360,
                threshold=threshold, digitize=True, case=True)

            if c0.butterfly_coeffs['F_a_butterfly'] > c1.butterfly_coeffs['F_a_butterfly']:
                bf_coeffs = c0.butterfly_coeffs
            else:
                bf_coeffs = c1.butterfly_coeffs

            bf_coeffs['theta'] = theta % 360
            bf_coeffs['threshold'] = threshold
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

    def calibrate_Flux_pulse_latency(self,
                                     times=np.arange(-300e-9, 300e-9, 5e-9),
                                     MC=None, update: bool=True,
                                     wait_after_flux: float=100e-9)-> bool:
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()
        if MC is None:
            MC = self.MC.get_instr()

        cfg = self.qasm_config()
        cfg_channel = cfg['qubit_map'][self.name.lower()]
        config_par_map = ['hardware specification', 'qubit_cfgs', cfg_channel,
                          'flux', 'latency']

        # N.B. docstring in parent class
        qasm_file = qwfs.ramZ_flux_latency(self.name, int(wait_after_flux*1e9))
        MC.set_sweep_function(swf.QASM_config_sweep(
            qasm_fn=qasm_file.name, config=cfg,
            config_par_map=config_par_map, set_parser=int,
            CBox=self.CBox.get_instr(), verbosity_level=0,
            par_scale_factor=1e9,
            parameter_name='Flux latency '+self.name, unit='s'))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run('Ram_Z_latency_calibration'+self.msmt_suffix)
        ma.MeasurementAnalysis()

    def measure_flux_timing(self, taus, MC=None, wait_between=220e-9):
        '''
        Measure timing of the flux pulse relative to microwave pulses.
        The sequence of the experiment is
            trigger flux pulse -- tau -- X90 -- wait_between -- X90 -- RO
        where tau is the sweep parameter.

        Note: wait_between should be a integer multiple of the mw pulse
        modulation frequency!

        Args:
            taus (array of floats):
                    The delays between the flux pulse trigger and the first
                    microwave pulse.
            MC (instr):
                    Measurement Control instrument.
            wait_between (float):
                    The delay between the two pi-half pulses.

        At large tau we expect to measure the qubit in the |1> state (flux pulse
        before both mw pulses).
        When the flux pulse overlaps with one of the mw pulses we expect to
        measure the qubit in a equal superposition of |0> and |1> (assuming flux
        pulse amplitude is large enough s.t. mw is not resonant with qubit
        anymore).
        When the flux pulse is fully between the mw pulses we expect to measure
        a constant |1> population which depends on the area of the flux pulse.
        '''
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        if MC is None:
            MC = self.MC.get_instr()

        qasm_file = sqqs.flux_timing_seq(self.name, taus,
                                         wait_between=wait_between)

        MC.set_sweep_function(swf.QASM_Sweep(
            filename=qasm_file.name, CBox=self.CBox.get_instr(),
            op_dict=self.get_operation_dict(),
            parameter_name='tau', unit='s'))
        MC.set_sweep_points(taus)

        MC.set_detector_function(self.int_avg_det)
        MC.run('flux_timing')

        ma.MeasurementAnalysis(label='flux_timing')

    def measure_ram_z(self, lengths, amps=None, chunk_size: int=32, MC=None,
                      wait_during_flux: str='auto', cal_points: bool=False,
                      cases=('cos', 'sin'), analyze=True,
                      filter_raw=False, filter_deriv_phase=False,
                      demodulate=False, f_demod=0, flux_amp_analysis=1):
        '''
        Perform a Ram-Z experiment: Measure the accumulated phase as a function
        of flux pulse length.
        Version 2 is for new QASM compiler.

        sequence:
            mX90 -- flux_pulse -- X90 -- RO


        Args:
            lengths (array of floats):
                    Array of the flux pulse lengths (sweep points).
            amps (array of floats):
                    If not None it will do a 2D sweep of lengths vs amplitude.
            chunk_size (int):
                    Sweep points are divided into chunks of this size. The
                    total number of sweep points should be an integer multiple
                    of the chunk size.
            MC (Intsrument):
                    Measurmenet control instrument.
            wait_during_flux (float or 'auto'):
                    Delay between the two pi-half pulses. If this is 'auto',
                    the time is automatically picked based on the maximum
                    of the sweep points.
            cal_points (bool):
                    Whether calibration points should be used. Note that the
                    calibration points will be inserted in every chunk,
                    because they are part of the QASM sequence, which is not
                    regenerated.
            cases (tuple of strings):
                    Possible cases are 'cos' and 'sin'. This determines
                    if an X90 or Y90 pulse is used as second pi-half pulse.
                    Measurement is repeated for all cases given.
            analyze (bool):
                    Do the Ram-Z analysis, extracting the step response.
            filter_raw (bool):
                    Apply a Gaussian low-pass filter to the raw (or
                    demodulated if demod is True) data.
            filter_deriv_phase (bool):
                    Apply a Gaussian derivative filter on the phase, thus
                    simultaneously low-pass filtering and computing the
                    derivative.
            demodulate (bool):
                    Demodulate data befor calculating phase.
            f_demod (float):
                    Modulation frequency used if demodulate is True.
            fulx_amp_analysis (float):
                    Flux pulse amplitude by which the step response is
                    normalized in the anlaysis. Set this to 1 to see the
                    step response in units of ouput voltage.
        '''
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        if 'uhfqc' not in self._acquisition_instrument.name.lower():
            raise RuntimeError('Requires acquisition with UHFQC (detector '
                               'function only implemented for UHFQC).')

        if len(lengths) % int(chunk_size) != 0:
            raise ValueError('Total number of points ({}) should be an'
                             ' integer multiple of chunk_size ({})'.format(
                                 len(lengths), chunk_size))

        f_lutman = self.flux_LutMan.get_instr()
        CBox = self.CBox.get_instr()

        if MC is None:
            MC = self.MC.get_instr()

        # Set the delay between the pihalf pulses to be long enough to fit the
        # flux pulse
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/self.f_pulse_mod())
            wait_between = (np.ceil(max(lengths) / T_pulsemod) + 2) \
                * T_pulsemod
        else:
            wait_between = wait_during_flux

        # Set the flux pulses in the operation dictionary
        # pulse 'square_i' has codeword i
        cfg = copy.deepcopy(self.qasm_config())
        for i in range(chunk_size):
            cfg['luts'][1]['square_{}'.format(i)] = i  # assign codeword
            cfg["operation dictionary"]["square_{}".format(i)] = {
                "parameters": 1,
                "duration": int(np.round(wait_between*1e9)),  # in ns
                "type": "flux",
                "matrix": []
            }

        for case in cases:
            if case == 'cos':
                rec_Y90 = False
            elif case == 'sin':
                rec_Y90 = True
            else:
                raise ValueError('Unknown case "{}".'.format(case))

            CBox.trigger_source('internal')
            qasm_file = sqqs.Ram_Z(
                qubit_name=self.name,
                no_of_points=chunk_size,
                cal_points=cal_points,
                rec_Y90=rec_Y90)
            qasm_folder, qasm_fn = os.path.split(qasm_file.name)
            qumis_fn = os.path.join(qasm_folder,
                                    qasm_fn.split('.')[0] + '.qumis')
            compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=0)
            compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)
            CBox.load_instructions(qumis_fn)

            d = self.int_avg_det
            d.chunk_size = chunk_size

            MC.set_sweep_function(swf.QWG_lutman_par_chunks(
                LutMan=f_lutman,
                LutMan_parameter=f_lutman.F_length,
                sweep_points=lengths,
                chunk_size=chunk_size,
                codewords=range(chunk_size),
                flux_pulse_type='square'))
            MC.set_sweep_points(lengths)
            MC.set_detector_function(d)

            if amps is None:
                MC.run('Ram_Z_{}{}'.format(case, self.msmt_suffix))
                ma.MeasurementAnalysis(label='Ram_Z')
            else:
                s2 = swf.QWG_flux_amp(QWG=f_lutman.QWG.get_instr(),
                                      channel=f_lutman.F_ch(),
                                      frac_amp=f_lutman.F_amp())
                MC.set_sweep_function_2D(s2)
                MC.set_sweep_points_2D(amps)
                MC.run('Ram_Z_{}_2D{}'.format(case, self.msmt_suffix),
                       mode='2D')
                ma.TwoD_Analysis(label='Ram_Z_')

        if analyze:
            return ma.Ram_Z_Analysis(
                filter_raw=filter_raw,
                filter_deriv_phase=filter_deriv_phase,
                demodulate=demodulate,
                f_demod=f_demod,
                f01max=self.f_max(),
                E_c=self.E_c(),
                flux_amp=flux_amp_analysis,
                V_offset=self.V_offset(),
                V_per_phi0=self.V_per_phi0(),
                TwoD=(amps is not None))

    def measure_cryo_scope(self, waveform, lengths='full', chunk_size: int=32,
                           MC=None, wait_during_flux: str='auto',
                           cal_points: bool=False, filter_raw: bool=False,
                           filter_deriv_phase: bool=False,
                           demodulate: bool=False, f_demod: float=0):
        '''
        Perform a Ram-Z experiment: Measure the accumulated phase as a
        function of flux pulse length.
        Version 2 is for new QASM compiler.

        sequence:
            mX90 -- flux_pulse -- X90 -- RO


        Args:
            waveform (array of floatS):
                    Waveform that will be measured in the cryo scope.
            lengths (array of floats):
                    Array of the flux pulse lengths (sweep points).
                    If this is 'full', the lengths are automatically chosen
                    such that the full pulse is measured.
            amps (array of floats):
                    If not None it will do a 2D sweep of lengths vs amplitude.
            chunk_size (int):
                    Sweep points are divided into chunks of this size. The
                    total number of sweep points should be an integer multiple
                    of the chunk size.
            MC (Intsrument):
                    Measurmenet control instrument.
            wait_during_flux (float or 'auto'):
                    Delay between the two pi-half pulses. If this is 'auto',
                    the time is automatically picked based on the maximum
                    of the sweep points.
            cal_points (bool):
                    Whether calibration points should be used. Note that the
                    calibration points will be inserted in every chunk,
                    because they are part of the QASM sequence, which is not
                    regenerated.
            cases (tuple of strings):
                    Possible cases are 'cos' and 'sin'. This determines
                    if an X90 or Y90 pulse is used as second pi-half pulse.
                    Measurement is repeated for all cases given.
            analyze (bool):
                    Do the Ram-Z analysis, extracting the step response.
            filter_raw (bool):
                    Apply a Gaussian low-pass filter to the raw (or
                    demodulated if demod is True) data.
            filter_deriv_phase (bool):
                    Apply a Gaussian derivative filter on the phase, thus
                    simultaneously low-pass filtering and computing the
                    derivative.
            demodulate (bool):
                    Demodulate data befor calculating phase.
            f_demod (float):
                    Modulation frequency used if demodulate is True.
            fulx_amp_analysis (float):
                    Flux pulse amplitude by which the step response is
                    normalized in the anlaysis. Set this to 1 to see the
                    step response in units of ouput voltage.
        '''
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        f_lutman = self.flux_LutMan.get_instr()
        CBox = self.CBox.get_instr()

        if 'uhfqc' not in self._acquisition_instrument.name.lower():
            raise RuntimeError('Requires acquisition with UHFQC (detector '
                               'function only implemented for UHFQC).')

        if lengths == 'full':
            # Fill waveform with zeros such that it is a multiple of
            # chunk_size.
            remainder = len(waveform) % chunk_size
            padding_len = chunk_size - remainder
            while padding_len < 20:
                # We want to add at least 20 zeros at the end
                padding_len += chunk_size

            waveform = np.concatenate(waveform, np.zeros(padding_len))
            lengths = np.arange(len(waveform)) / f_lutman.sampling_rate()

        if len(lengths) % int(chunk_size) != 0:
            raise ValueError('Total number of points ({}) should be an'
                             ' integer multiple of chunk_size ({})'.format(
                                 len(lengths), chunk_size))

        if MC is None:
            MC = self.MC.get_instr()

        # Set the delay between the pihalf pulses to be long enough to fit the
        # flux pulse
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/self.f_pulse_mod())
            wait_between = (np.ceil(max(lengths) / T_pulsemod) + 2) \
                * T_pulsemod
        else:
            wait_between = wait_during_flux

        # Set the flux pulses in the operation dictionary
        # pulse 'square_i' has codeword codewords[i]
        cfg = copy.copy(self.qasm_config())
        for i in range(chunk_size):
            cfg['luts'][1]['square_{}'.format(i)] = i  # assign codeword
            cfg["operation dictionary"]["square_{}".format(i)] = {
                "parameters": 1,
                "duration": int(np.round(wait_between*1e9)),  # in ns
                "type": "flux",
                "matrix": []
            }

        for case in ('cos', 'sin'):
            if case == 'cos':
                rec_Y90 = False
            elif case == 'sin':
                rec_Y90 = True
            else:
                raise ValueError('Unknown case "{}".'.format(case))

            CBox.trigger_source('internal')
            qasm_file = sqqs.Ram_Z(
                qubit_name=self.name,
                no_of_points=chunk_size,
                cal_points=cal_points,
                rec_Y90=rec_Y90)
            qasm_folder, qasm_fn = os.path.split(qasm_file.name)
            qumis_fn = os.path.join(qasm_folder,
                                    qasm_fn.split('.')[0] + '.qumis')
            compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=0)
            compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)
            CBox.load_instructions(qumis_fn)

            d = self.int_avg_det
            d.chunk_size = chunk_size

            MC.set_sweep_function(swf.QWG_lutman_custom_wave_chunks(
                LutMan=f_lutman,
                wave_func=lambda t: waveform[:int(
                    np.round(t * f_lutman.sampling_rate()))],
                sweep_points=lengths,
                chunk_size=chunk_size,
                codewords=range(chunk_size),
                parameter_name='pulse time',
                parameter_unit='s'))
            MC.set_sweep_points(lengths)
            MC.set_detector_function(d)

            MC.run('Ram_Z_scope_{}{}'.format(case, self.msmt_suffix))
            ma.MeasurementAnalysis(label='Ram_Z_scope')

        ma.Ram_Z_Analysis(
            filter_raw=filter_raw,
            filter_deriv_phase=filter_deriv_phase,
            demodulate=demodulate,
            f_demod=f_demod,
            f01max=self.f_max(),
            E_c=self.E_c(),
            flux_amp=1,
            V_offset=self.V_offset(),
            V_per_phi0=self.V_per_phi0(),
            TwoD=False)

    def measure_ram_z_echo(self, lengths, amps=None, chunk_size: int=32,
                           MC=None, wait_during_flux='auto',
                           cal_points: bool=False, analyze=True):
        '''
        Perform a Ram-Z echo experiment.

        TODO: more description


        Args:
            lengths (array of floats):
                    Array of the flux pulse lengths (sweep points).
            amps (array of floats):
                    If not None it will do a 2D sweep of lengths vs amplitude.
            chunk_size (int):
                    Sweep points are divided into chunks of this size. The
                    total number of sweep points should be an integer multiple
                    of the chunk size.
            MC (Intsrument):
                    Measurmenet control instrument.
            wait_during_flux (float or 'auto'):
                    Delay between the two pi-half pulses. If this is 'auto',
                    the time is automatically picked based on the maximum
                    of the sweep points.
            cal_points (bool):
                    Whether calibration points should be used. Note that the
                    calibration points will be inserted in every chunk,
                    because they are part of the QASM sequence, which is not
                    regenerated.
            analyze (bool):
                    Do the Ram-Z analysis, extracting the step response.
        '''
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        if 'uhfqc' not in self._acquisition_instrument.name.lower():
            raise RuntimeError('Requires acquisition with UHFQC (detector '
                               'function only implemented for UHFQC).')

        if len(lengths) % int(chunk_size) != 0:
            raise ValueError('Total number of points ({}) should be an'
                             ' integer multiple of chunk_size ({})'.format(
                                 len(lengths), chunk_size))

        f_lutman = self.flux_LutMan.get_instr()
        CBox = self.CBox.get_instr()
        QWG = f_lutman.QWG.get_instr()

        if MC is None:
            MC = self.MC.get_instr()

        # Set the delay between the pihalf pulses to be long enough to fit the
        # flux pulse
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/self.f_pulse_mod())
            wait_between = (np.ceil(max(lengths) / T_pulsemod) + 2) \
                * T_pulsemod
        else:
            wait_between = wait_during_flux

        # Set the flux pulses in the operation dictionary
        # pulse 'square_i' has codeword codewords[i]
        cfg = copy.deepcopy(self.qasm_config())
        for i in range(chunk_size):
            cfg['luts'][1]['square_{}'.format(i)] = i  # assign codeword
            cfg["operation dictionary"]["square_{}".format(i)] = {
                "parameters": 1,
                "duration": int(np.round(wait_between*1e9)),  # in ns
                "type": "flux",
                "matrix": []
            }

        # Define a dummy pulse for the second square pulse. The whole flux
        # pulse is compiled into one waveform.
        cfg['luts'][1]['square_dummy'] = -2
        cfg['operation dictionary']['square_dummy'] = {
            'parameters': 1,
            'duration': int(np.round(wait_between*1e9)),
            'type': 'flux',
            'matrix': []
        }

        def wave_func(val):
            # Function that calculates the composite wave form from the
            # value of the sweep point.
            t_second_pulse = int(
                np.round((wait_between + 4 * self.gauss_width()) * 1e9))

            composite_pulse = np.zeros(int(np.round(wait_between * 1e9)) * 2
                                       + 100)
            pulse_len = int(np.round(val * 1e9))
            composite_pulse[:pulse_len] = (np.ones(pulse_len) *
                                           f_lutman.F_amp())
            composite_pulse[t_second_pulse:t_second_pulse+pulse_len+1] = \
                np.ones(pulse_len+1) * f_lutman.F_amp()

            return composite_pulse

        CBox.trigger_source('internal')
        qasm_file = sqqs.Ram_Z_echo(
            qubit_name=self.name,
            no_of_points=chunk_size,
            cal_points=cal_points)
        qasm_folder, qasm_fn = os.path.split(qasm_file.name)
        qumis_fn = os.path.join(qasm_folder,
                                qasm_fn.split('.')[0] + '.qumis')
        compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=0)
        compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)
        CBox.load_instructions(qumis_fn)

        d = self.int_avg_det
        d.chunk_size = chunk_size

        MC.set_sweep_function(swf.QWG_lutman_custom_wave_chunks(
            LutMan=f_lutman,
            wave_func=wave_func,
            sweep_points=lengths,
            chunk_size=chunk_size,
            codewords=range(chunk_size),
            param_name='pulse length',
            param_unit='s'))
        MC.set_sweep_points(lengths)
        MC.set_detector_function(d)

        if amps is None:
            MC.run('Ram_Z_echo{}'.format(self.msmt_suffix))
            ma.MeasurementAnalysis(label='Ram_Z')
        else:
            s2 = swf.QWG_flux_amp(QWG=f_lutman.QWG.get_instr(),
                                  channel=f_lutman.F_ch(),
                                  frac_amp=f_lutman.F_amp())
            MC.set_sweep_function_2D(s2)
            MC.set_sweep_points_2D(amps)
            MC.run('Ram_Z_2D{}'.format(self.msmt_suffix),
                   mode='2D')
            ma.TwoD_Analysis(label='Ram_Z_')

        if analyze:
            return ma.MeasurementAnalysis()

    def measure_CZ_phase_ripple(self, taus, chunk_size: int=16,
                                MC=None, wait_during_flux='auto',
                                cases=('cos', 'sin'),
                                flux_type: str='square',
                                analyze: bool=True, msmt_suffix=None):
        '''
        Measure the total phase with the sequence
            mX90 -- CZ -- tau -- flux -- rec90
        where flux can be different types of flux pulses, rec90 is X90 or Y90,
        as a function of the delay tau.
        The position of the mw pulses is fixed.

        Args:
            taus (array of floats):
                    List of the separation taus to use (sweep points).
            chunk_size (int):
                    Sweep points are divided into chunks of this size. The
                    total number of sweep points should be an integer multiple
                    of the chunk size.
            MC (Intsrument):
                    Measurmenet control instrument.
            wait_during_flux (float or 'auto'):
                    Delay between the two pi-half pulses. If this is 'auto',
                    the time is automatically picked based on the maximum
                    of the sweep points.
            cases (tuple of strings):
                    Possible cases are 'cos' and 'sin'. This determines
                    if an X90 or Y90 pulse is used as second pi-half pulse.
                    Measurement is repeated for all cases given.
            flux_type (string):
                    Type of pulse to use for the second flux pulse. This
                    string has to exactly match a key of the _wave_dict of
                    self.flux_LutMan.
            analyze (bool):
                    Do the Ram-Z analysis, extracting the step response.
        '''
        self.prepare_for_timedomain()
        self.prepare_for_fluxing()

        f_lutman = self.flux_LutMan.get_instr()
        CBox = self.CBox.get_instr()

        if msmt_suffix is None:
            msmt_suffix = self.msmt_suffix

        if 'uhfqc' not in self._acquisition_instrument.name.lower():
            raise RuntimeError('Requires acquisition with UHFQC (detector '
                               'function only implemented for UHFQC).')

        if len(taus) % int(chunk_size) != 0:
            raise ValueError('Total number of points ({}) should be an'
                             ' integer multiple of chunk_size ({})'.format(
                                 len(taus), chunk_size))

        if MC is None:
            MC = self.MC.get_instr()

        # Get the waveforms for the two flux pulses that will be used
        std_wfs = f_lutman.standard_waveforms()
        wf1 = std_wfs['adiabatic_Z']
        wf2 = std_wfs[flux_type]
        max_len = (int(np.round(max(taus) * f_lutman.sampling_rate()))
                   + len(wf1) + len(wf2))

        # Set the delay between the pihalf pulses to be long enough to fit the
        # flux pulse
        if wait_during_flux == 'auto':
            # Round to the next integer multiple of qubit pulse modulation
            # period and add two periods as buffer
            T_pulsemod = np.abs(1/self.f_pulse_mod())
            wait_between = (np.ceil(max_len / f_lutman.sampling_rate()
                                    / T_pulsemod) + 2) * T_pulsemod
            print('Autogenerated wait between pi-halfs: {} ns'
                  .format(np.round(wait_between*f_lutman.sampling_rate())))
        else:
            wait_between = wait_during_flux


        # Set the flux pulses in the operation dictionary
        # pulse 'square_i' has codeword i
        cfg = copy.deepcopy(self.qasm_config())
        for i in range(chunk_size):
            cfg['luts'][1]['square_{}'.format(i)] = i  # assign codeword
            cfg["operation dictionary"]["square_{}".format(i)] = {
                "parameters": 1,
                "duration": int(np.round(wait_between
                                         * f_lutman.sampling_rate())),
                "type": "flux",
                "matrix": []
            }

        def wave_func(val):
            zero_samples = int(np.round(val * f_lutman.sampling_rate()))
            return np.concatenate((
                wf1,
                np.zeros(zero_samples),
                wf2))

        for case in cases:
            if case == 'cos':
                rec_Y90 = False
            elif case == 'sin':
                rec_Y90 = True
            else:
                raise ValueError('Unknown case "{}".'.format(case))

            CBox.trigger_source('internal')
            qasm_file = sqqs.Ram_Z(
                qubit_name=self.name,
                no_of_points=chunk_size,
                cal_points=False,
                rec_Y90=rec_Y90)
            qasm_folder, qasm_fn = os.path.split(qasm_file.name)
            qumis_fn = os.path.join(qasm_folder,
                                    qasm_fn.split('.')[0] + '.qumis')
            compiler = qcx.QASM_QuMIS_Compiler(verbosity_level=0)
            compiler.compile(qasm_file.name, qumis_fn=qumis_fn, config=cfg)
            CBox.load_instructions(qumis_fn)

            d = self.int_avg_det
            d.chunk_size = chunk_size

            MC.set_sweep_function(swf.QWG_lutman_custom_wave_chunks(
                LutMan=f_lutman,
                wave_func=wave_func,
                sweep_points=taus,
                chunk_size=chunk_size,
                codewords=np.arange(chunk_size),
                param_name='delay',
                param_unit='s'))
            MC.set_sweep_points(taus)
            MC.set_detector_function(d)

            MC.run('CZ_phase_ripple_{}{}'.format(case, msmt_suffix))
            ma.MeasurementAnalysis(label='CZ_phase_ripple_')

        if analyze:
            return ma.Ram_Z_Analysis(
                filter_raw=False,
                filter_deriv_phase=False,
                demodulate=False,
                f01max=None,
                E_c=None,
                flux_amp=None,
                V_offset=None,
                V_per_phi0=None,
                TwoD=False)

    def get_operation_dict(self, operation_dict={}):

        pulse_period_clocks = convert_to_clocks(
            self.gauss_width()*4+self.pulse_delay(), rounding_period=1/abs(self.f_pulse_mod()))
        RO_pulse_length_clocks = convert_to_clocks(self.RO_pulse_length())
        RO_pulse_delay_clocks = convert_to_clocks(self.RO_pulse_delay())
        RO_depletion_clocks = convert_to_clocks(self.RO_depletion_time())
        RO_acq_marker_del_clks = convert_to_clocks(self.RO_acq_marker_delay())

        operation_dict['init_all'] = {'instruction':
                                      '\nWaitReg r0 \nWaitReg r0 \n'}

        # MW control pulses
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

        # Flux pulses
        if self.flux_LutMan() is not None:
            f_lutman = self.flux_LutMan.get_instr()
            for pulse_name, codeword in f_lutman.codeword_dict().items():
                operation_dict['flux {} {}'.format(pulse_name, self.name)] = {
                    'duration': 10,
                    'instruction': ins_lib.qwg_cw_trigger(
                        codeword, cw_channels=f_lutman.codeword_channels())
                }

        # Spectroscopy pulses
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

        # Readout pulse
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

    def measure_single_qubit_GST(self,
                                 max_germ_pow: int=10,
                                 repetitions_per_point: int=128,
                                 min_soft_repetitions: int=5,
                                 MC=None, analyze: bool=False):
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
            MC = self.MC.get_instr()

        # Load gate set, germs, and fiducials from file.
        gstPath = os.path.dirname(gstCC.__file__)
        gs_target_path = os.path.join(gstPath, 'Gateset_5_primitives_GST.txt')
        fiducials_path = os.path.join(gstPath,
                                      'Fiducials_5_primitives_GST.txt')
        germs_path = os.path.join(gstPath, 'Germs_5_primitives_GST.txt')

        gs_target = pygsti.io.load_gateset(gs_target_path)
        fiducials = pygsti.io.load_gatestring_list(fiducials_path)
        germs = pygsti.io.load_gatestring_list(germs_path)

        max_lengths = [2**i for i in range(max_germ_pow + 1)]

        # gate_dict maps GST gate labels to QASM operations.
        gate_dict = {
            'Gi': 'I {}'.format(self.name),
            'Gx90': 'X90 {}'.format(self.name),
            'Gy90': 'Y90 {}'.format(self.name),
            'Gx180': 'X180 {}'.format(self.name),
            'Gy180': 'Y180 {}'.format(self.name),
            'RO': 'RO {}'.format(self.name)
        }

        # Create the experiment list, translate it to QASM, and generate the
        # QASM file(s).
        raw_exp_list = pygsti.construction.make_lsgst_experiment_list(
            gs_target.gates.keys(), fiducials, fiducials, germs, max_lengths)
        exp_list = gstCC.get_experiments_from_list(raw_exp_list, gate_dict)
        qasm_files, exp_per_file, exp_last_file = gstCC.generate_QASM(
            filename='GST_{}'.format(self.name),
            exp_list=exp_list,
            qubit_labels=[self.name],
            max_instructions=self.CBox.get_instr()._get_instr_mem_size(),
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
        soft_repetitions = int(np.ceil(
            repetitions_per_point / np.floor(max_acq_points / exp_per_file)))
        if soft_repetitions < min_soft_repetitions:
            soft_repetitions = min_soft_repetitions
        hard_repetitions = int(np.ceil(repetitions_per_point /
                                       soft_repetitions))

        self.prepare_for_timedomain()
        d = self.int_log_det
        s = swf.Multi_QASM_Sweep(
            exp_per_file=exp_per_file,
            hard_repetitions=hard_repetitions,
            soft_repetitions=soft_repetitions,
            qasm_list=[q.name for q in qasm_files],
            config=self.qasm_config(),
            detector=d,
            CBox=self.CBox.get_instr(),
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
            'prep_fids': fiducials_path,
            'meas_fids': fiducials_path,
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
        MC.run('GST', exp_metadata=metadata_dict)

        # Analysis
        if analyze:
            ma.GST_Analysis()


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
        # FIXME: chanel amp90 scale to Q_amp90 scale
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
            self.QWG.set('ch{}_amp'.format(ch), self.Q_amp180()*1.1)
        # FIXME:  Currently hardcoded to use channel 1
        self.Q_LutMan.get_instr().Q_amp180(self.Q_amp180())
        self.Q_LutMan.get_instr().Q_amp90_scale(self.Q_amp90_scale())
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
                operation_dict['RO {}'.format(self.name)][
                    'instruction'] = measure_instruction
            else:
                measure_instruction = self._triggered_RO_marker_instr()
                operation_dict['RO {}'.format(self.name)][
                    'instruction'] = measure_instruction
        else:
            raise NotImplementedError('Unknown acquisition device.')

        return operation_dict
