import time
import logging
import numpy as np
import copy

import os
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx
from scipy.optimize import brent
from pycqed.measurement.optimization import nelder_mead
import pygsti

from .qubit_object import Qubit
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
import pycqed.analysis.analysis_toolbox as a_tools

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
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC


class CCLight_Transmon(Qubit):

    '''
    The CCLight_Transmon
    Setup configuration:
        Drive:                 CCLight controlling AWG8's and a VSM
        Acquisition:           UHFQC
        Readout pulse configuration: LO modulated using UHFQC AWG
    '''

    def __init__(self, name, **kw):
        t0 = time.time()
        super().__init__(name, **kw)
        self.add_parameters()
        self.connect_message(begin_time=t0)

    def add_instrument_ref_parameters(self):
        # MW sources
        self.add_parameter('instr_LO', parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_cw_source',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_td_source',
                           parameter_class=InstrumentRefParameter)

        # Control electronics
        self.add_parameter(
            'instr_CC', label='Central Controller',
            docstring=('Device responsible for controlling the experiment'
                       ' using eQASM generated using OpenQL, in the near'
                       ' future will be the CC_Light.'),
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_acquisition',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_VSM', label='Vector Switch Matrix',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('instr_MC', label='MeasurementControl',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_SH', label='SignalHound',
                           parameter_class=InstrumentRefParameter)

        # LutMan's
        self.add_parameter('instr_LutMan_MW',
                           docstring='Lookuptable manager  for '
                           'microwave control pulses.',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_LutMan_RO',
                           docstring='Lookuptable manager responsible for '
                           'microwave readout pulses.',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_LutMan_Flux',
                           docstring='Lookuptable manager responsible for '
                                     'flux pulses.',
                           initial_value=None,
                           parameter_class=InstrumentRefParameter)

    def add_ro_parameters(self):
        # adding marker channels
        self.add_parameter('RO_acq_pulse_marker_channel',
                           vals=vals.Ints(1, 7),
                           initial_value=6,
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
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e6,
                           parameter_class=ManualParameter)

        # Time-domain parameters
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

    def add_mw_parameters(self):
        self.add_parameter('mod_amp_td', label='RO modulation ampl td',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)

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

        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('f_pulse_mod',
                           initial_value=-2e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('Q_awg_nr', label='CBox awg nr', unit='#',
                           vals=vals.Ints(),
                           initial_value=0,
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

    def add_spec_parameters(self):

        self.add_parameter('spec_pow', label='spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
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

    def add_flux_parameters(self):
        pass

    def add_config_parameters(self):
        pass

    def add_generic_qubit_parameters(self):
        pass

    def prepare_for_continuous_wave(self):
        self.prepare_readout()

        # LO and RF for readout are turned on in prepare_readout
        self.instr_td_source.get_instr().off()
        self.instr_cw_source.get_instr().off()
        self.instr_cw_source.get_instr().pulsemod_state.set('off')
        self.instr_cw_source.get_instr().power.set(self.spec_pow.get())

    def prepare_readout(self):
        """
        Configures the readout. Consists of the following steps
        - instantiate the relevant detector functions
        - set the microwave frequencies and sources
        - generate the RO pulse
        - set the integration weights
        """
        pass
        # self._instantiate_detectors()
        # self._set_RO_frequencies_and_sources()
        # self._set_RO_integration_weights()
        # self._generate_RO_pulse()

    def _instantiate_detectors(self):
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

        if 'UHFQC' in self.instr_acquisition():
            UHFQC = self.instr_acquisition.get_instr()

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

        #####################################
        # Setting frequencies and MW sources
        #####################################

    def _set_RO_frequencies_and_sources(self):
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

        #####################################
        # Generating the RO pulse
        #####################################

    def _generate_RO_pulse(self):
        if 'CBox' in self.instr_acquisition():
            if 'multiplexed' not in self.RO_pulse_type().lower():
                self.RO_LutMan.get_instr().M_modulation(self.f_RO_mod())
                self.RO_LutMan.get_instr().M_amp(self.RO_amp())
                self.RO_LutMan.get_instr().M_length(self.RO_pulse_length())

                if 'awg_nr' in self.RO_LutMan.get_instr().parameters:
                    self.RO_LutMan.get_instr().awg_nr(self.RO_awg_nr())

                if 'CBox' in self.instr_acquisition():
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

        elif 'UHFQC' in self.instr_acquisition():
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

    def _set_RO_integration_weights(self):
        if 'UHFQC' in self.instr_acquisition():
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
        else:
            raise NotImplementedError('CBox, DDM or other currently not supported')

    def prepare_for_timedomain(self):
        pass

    def prepare_for_fluxing(self, reset=True):
        pass

    def _get_acquisition_instr(self):
        pass

    def _set_acquisition_instr(self, acq_instr_name):
        pass
