import time
import logging
import numpy as np
try:
    from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
except ImportError:
    logging.warning('Could not import OpenQL')
    sqo = None

from .qubit_object import Qubit
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2

from pycqed.measurement.calibration_toolbox import (
    mixer_carrier_cancellation, mixer_skewness_calibration_CBoxV3)

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det


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
        self.add_parameter('instr_LO_ro',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_LO_mw',
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
        """
        Adding the parameters relevant for readout.
        """
        ################################
        # RO stimulus/pulse parameters #
        ################################
        self.add_parameter('ro_freq',
                           label='Readout frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('ro_freq_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e6,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pow_LO', label='RO power LO',
                           unit='dBm', initial_value=14,
                           parameter_class=ManualParameter)

        # RO pulse parameters
        self.add_parameter('ro_pulse_res_nr',
                           label='Resonator number', docstring=(
                               'Resonator number used in lutman for'
                               ' uploading to the correct UHFQC codeword.'),
                           initial_value=0, vals=vals.Ints(0, 9),
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_type', initial_value='square',
                           vals=vals.Enum('gated', 'square', 'up_down_down'),
                           parameter_class=ManualParameter)

        # Mixer offsets correction, RO pulse
        self.add_parameter('ro_pulse_mixer_offs_I', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('ro_pulse_mixer_offs_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('ro_pulse_mixer_alpha', initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_mixer_phi', initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('ro_pulse_length',
                           initial_value=100e-9,
                           unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_amp', unit='V',
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_phi', unit='deg', initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('ro_pulse_down_length0', unit='s',
                           initial_value=1e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_down_amp0', unit='V', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_down_phi0', unit='deg', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_down_length1', unit='s',
                           initial_value=1e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_down_amp1', unit='V', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_pulse_down_phi1', unit='deg', initial_value=0,
                           parameter_class=ManualParameter)

        #############################
        # RO acquisition parameters #
        #############################

        ro_acq_docstr = (
            'Determines what type of integration weights to use: '
            '\n\t SSB: Single sideband demodulation\n\t'
            'DSB: Double sideband demodulation\n\t'
            'optimal: waveforms specified in "RO_acq_weight_func_I" '
            '\n\tand "RO_acq_weight_func_Q"')

        self.add_parameter('ro_acq_weight_type',
                           initial_value='DSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=ro_acq_docstr,
                           parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_weight_chI', initial_value=0, docstring=(
                'Determines the I-channel for integration. When the'
                ' ro_acq_weight_type is optimal only this channel will '
                'affect the result.'), vals=vals.Ints(0, 5),
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_weight_chQ', initial_value=1, docstring=(
                'Determines the Q-channel for integration.'),
            vals=vals.Ints(0, 5), parameter_class=ManualParameter)

        self.add_parameter('ro_acq_weight_func_I',
                           vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_weight_func_Q',
                           vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_delay',  unit='s',
            label='Readout acquisition delay',
            vals=vals.Numbers(min_value=0),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=('The time between the instruction that trigger the'
                       ' readout pulse and the instruction that triggers the '
                       'acquisition. The positive number means that the '
                       'acquisition is started after the pulse is send.'))

        self.add_parameter('ro_acq_integration_length', initial_value=500e-9,
                           vals=vals.Numbers(min_value=0, max_value=20e6),
                           parameter_class=ManualParameter)

        self.add_parameter('ro_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
                           parameter_class=ManualParameter)

        self.add_parameter('ro_soft_avg', initial_value=1,
                           docstring=('Number of soft averages to be '
                                      'performed using the MC.'),
                           vals=vals.Ints(min_value=1),
                           parameter_class=ManualParameter)

        # self.add_parameter('ro_power_cw', label='RO power cw',
        #                    unit='dBm',
        #                    parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('ro_acq_digitized', vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)

        # self.add_parameter('cal_pt_zero',
        #                    initial_value=None,
        #                    vals=vals.Anything(),  # should be a tuple validator
        #                    label='Calibration point |0>',
        #                    parameter_class=ManualParameter)
        # self.add_parameter('cal_pt_one',
        #                    initial_value=None,
        #                    vals=vals.Anything(),  # should be a tuple validator
        #                    label='Calibration point |1>',
        #                    parameter_class=ManualParameter)

    def add_mw_parameters(self):
        # Mixer skewness correction
        self.add_parameter('mw_G_mixer_phi', unit='deg',
                           label='Mixer skewness phi Gaussian quadrature',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_G_mixer_alpha', unit='',
                           label='Mixer skewness alpha Gaussian quadrature',
                           parameter_class=ManualParameter, initial_value=1)
        self.add_parameter('mw_D_mixer_phi', unit='deg',
                           label='Mixer skewness phi Derivative quadrature',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_D_mixer_alpha', unit='',
                           label='Mixer skewness alpha Derivative quadrature',
                           parameter_class=ManualParameter, initial_value=1)

        # Mixer offsets correction, qubit drive
        self.add_parameter('mw_mixer_offs_I',
                           unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_mixer_offs_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)

        self.add_parameter('mw_pow_td_source',
                           label='Time-domain power',
                           unit='dBm',
                           initial_value=14,
                           parameter_class=ManualParameter)

        self.add_parameter('mw_freq_mod',
                           initial_value=-2e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)

        self.add_parameter('mw_amp180',
                           label='Pi-pulse amplitude', unit='V',
                           initial_value=.8,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_amp90_scale',
                           label='pulse amplitude scaling factor',
                           unit='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
                           parameter_class=ManualParameter)

        self.add_parameter('mw_awg_ch', parameter_class=ManualParameter,
                           initial_value=1)
        self.add_parameter('mw_gauss_width', unit='s',
                           initial_value=10e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_motzoi', label='Motzoi parameter', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('mw_vsm_switch',
                           label='VSM switch state',
                           initial_value='EXT',
                           vals=vals.Enum('ON', 'OFF', 'EXT'),
                           parameter_class=ManualParameter)
        self.add_parameter(
            'mw_vsm_delay', label='CCL VSM trigger delay',
            vals=vals.Ints(0, 127), unit='samples',
            docstring=('This value needs to be calibrated to ensure that '
                       'the VSM mask aligns with the microwave pulses. '
                       'Calibration is done using'
                       ' self.calibrate_mw_vsm_delay.'),
            parameter_class=ManualParameter)

        self.add_parameter('mw_vsm_ch_Gin',
                           label='VSM input channel Gaussian component',
                           vals=vals.Ints(1, 4),
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_ch_Din',
                           label='VSM input channel Derivative component',
                           vals=vals.Ints(1, 4),
                           initial_value=2,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_ch_out',
                           label='VSM output channel for microwave pulses',
                           docstring=('Selects the VSM output channel for MW'
                                      ' pulses. N.B. for spec the '
                                      'spec_vsm_ch_out parameter is used.'),
                           vals=vals.Ints(1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_G_att',
                           label='VSM attenuation Gaussian component',
                           vals=vals.Numbers(0, 65536),
                           initial_value=65536/2,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_D_att',
                           label='VSM attenuation Derivative component',
                           vals=vals.Numbers(0, 65536),
                           initial_value=65536/2,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_G_phase',
                           vals=vals.Numbers(0, 65536),
                           initial_value=65536/2,
                           parameter_class=ManualParameter)
        self.add_parameter('mw_vsm_D_phase',
                           vals=vals.Numbers(0, 65536),
                           initial_value=65536/2,
                           parameter_class=ManualParameter)

    def add_spec_parameters(self):
        self.add_parameter('spec_vsm_att',
                           label='VSM attenuation for spec pulses',
                           vals=vals.Numbers(0, 65536),
                           initial_value=65536/2,
                           parameter_class=ManualParameter)
        self.add_parameter('spec_vsm_ch_out',
                           label='VSM output channel for spectroscopy pulses',
                           docstring=('Selects the VSM output channel for spec'
                                      ' pulses. N.B. for mw pulses the '
                                      'spec_mw_ch_out parameter is used.'),
                           vals=vals.Ints(1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('spec_vsm_ch_in',
                           label='VSM input channel for spec pulses',
                           docstring=('VSM input channel for spec pulses'
                                      ' generally this should be the same as '
                                      ' the mw_vsm_ch_Gin parameter.'),
                           vals=vals.Ints(1, 4),
                           initial_value=1,
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_length',
                           label='Pulsed spec pulse duration',
                           unit='s', vals=vals.Numbers(20e-9, 20e-6),
                           # FIXME validator: should be multiple of 20e-9
                           initial_value=500e-9,
                           parameter_class=ManualParameter)

        self.add_parameter(
            'spec_amp', unit='V', docstring=(
                'Amplitude of the spectroscopy pulse in the mw LutMan. '
                'The power of the spec pulse should be controlled through '
                'the vsm attenuation "spec_vsm_att"'),
            vals=vals.Numbers(0, 1), parameter_class=ManualParameter,
            initial_value=0.8)

    def add_flux_parameters(self):
        pass

    def add_config_parameters(self):
        self.add_parameter(
            'cfg_trigger_period', label='Trigger period',
            docstring=('Time between experiments, used to initialize all'
                       ' qubits in the ground state'),
            unit='s', initial_value=200e-6,
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=1e-6, max_value=327668e-9))
        self.add_parameter('cfg_openql_platform_fn',
                           label='OpenQL platform configuration filename',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())
        self.add_parameter(
            'cfg_qubit_nr', label='Qubit number', vals=vals.Ints(0, 7),
            parameter_class=ManualParameter, initial_value=0,
            docstring='The qubit number is used in the OpenQL compiler. '
            'Beware that a similar parameter (ro_pulse_res_nr) exists that is'
            ' used for uploading to the right Lookuptable. These params are '
            'oten but not always identical (e.g., multiple feedlines). ')

    def add_generic_qubit_parameters(self):
        self.add_parameter('E_c', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('E_j', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T1', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_echo', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_star', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('freq_qubit',
                           label='mwubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('freq_max',
                           label='mwubit sweet spot frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('freq_res',
                           label='Resonator frequency', unit='Hz',
                           parameter_class=ManualParameter)

        self.add_parameter('F_ssro',
                           initial_value=0,
                           label='Single shot readout assignment fidelity',
                           vals=vals.Numbers(0.0, 1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('F_discr',
                           initial_value=0,
                           label='Single shot readout discrimination fidelity',
                           vals=vals.Numbers(0.0, 1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('F_RB',
                           initial_value=0,
                           label='RB single qubit Clifford fidelity',
                           vals=vals.Numbers(0, 1.0),
                           parameter_class=ManualParameter)

    def prepare_for_continuous_wave(self):
        if self.ro_acq_weight_type() not in {'DSB', 'SSB'}:
            # this is because the CW acquisition detects using angle and phase
            # and this requires two channels to rotate the signal properly.
            raise ValueError('Readout "{}" '.format(self.ro_acq_weight_type())
                             + 'weight type must be "SSB" or "DSB"')
        self.prepare_readout()
        self._prep_cw_spec()
        # LO for readout is turned on in prepare_readout
        self.instr_LO_mw.get_instr().on()

    def _prep_cw_spec(self):
        VSM = self.instr_VSM.get_instr()
        VSM.set_all_switches_to('OFF')
        VSM.set('in{}_out{}_switch'.format(self.spec_vsm_ch_in(),
                                           self.spec_vsm_ch_out()), 'EXT')
        VSM.set('in{}_out{}_att'.format(
                self.spec_vsm_ch_in(), self.spec_vsm_ch_out()),
                self.spec_vsm_att())

    def prepare_readout(self):
        """
        Configures the readout. Consists of the following steps
        - instantiate the relevant detector functions
        - set the microwave frequencies and sources
        - generate the RO pulse
        - set the integration weights
        """
        self._prep_ro_instantiate_detectors()
        self._prep_ro_sources()
        self._prep_ro_pulse()
        self._prep_ro_integration_weights()

    def _prep_ro_instantiate_detectors(self):
        self.instr_MC.get_instr().soft_avg(self.ro_soft_avg())
        if self.ro_acq_weight_type() == 'optimal':
            ro_channels = [self.ro_acq_weight_chI()]
            result_logging_mode = 'lin_trans'

            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'
            # Update the RO theshold
            acq_ch = self.ro_acq_weight_chI()

            # The threshold that is set in the hardware  needs to be
            # corrected for the offset as this is only applied in
            # software.
            threshold = self.ro_acq_threshold()
            offs = self.instr_acquisition.get_instr().get(
                'quex_trans_offset_weightfunction_{}'.format(acq_ch))
            hw_threshold = threshold + offs
            self.instr_acquisition.get_instr().set(
                'quex_thres_{}_level'.format(acq_ch), hw_threshold)

        else:
            ro_channels = [self.ro_acq_weight_chI(),
                           self.ro_acq_weight_chQ()]
            result_logging_mode = 'raw'

        if 'UHFQC' in self.instr_acquisition():
            UHFQC = self.instr_acquisition.get_instr()

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                nr_averages=self.ro_acq_averages())

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_channels,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                integration_length=self.ro_acq_integration_length())

            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_channels,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                real_imag=True, single_int_avg=True,
                integration_length=self.ro_acq_integration_length())

            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_channels,
                result_logging_mode=result_logging_mode,
                integration_length=self.ro_acq_integration_length())
        else:
            raise NotImplementedError()

    def _prep_ro_sources(self):
        LO = self.instr_LO_ro.get_instr()
        LO.frequency.set(self.ro_freq() - self.ro_freq_mod())
        LO.on()
        LO.power(self.ro_pow_LO())

    def _prep_ro_pulse(self):
        """
        Sets the appropriate parameters in the RO LutMan and uploads the
        desired wave.
        Relevant parameters are:
            ro_pulse_type ("up_down_down", "square")
            ro_pulse_res_nr
            ro_freq_mod
            ro_acq_delay

            ro_pulse_length
            ro_pulse_amp
            ro_pulse_phi
            ro_pulse_down_length0
            ro_pulse_down_amp0
            ro_pulse_down_phi0
            ro_pulse_down_length1
            ro_pulse_down_amp1
            ro_pulse_down_phi1


            ro_pulse_mixer_alpha
            ro_pulse_mixer_phi

            ro_pulse_mixer_offs_I
            ro_pulse_mixer_offs_Q

        """
        if 'UHFQC' not in self.instr_acquisition():
            raise NotImplementedError()
        UHFQC = self.instr_acquisition.get_instr()

        if 'gated' in self.ro_pulse_type().lower():
            UHFQC.awg_sequence_acquisition()

        else:
            ro_lm = self.instr_LutMan_RO.get_instr()
            idx = self.ro_pulse_res_nr()
            # These parameters affect all resonators
            ro_lm.set('pulse_type', 'M_' + self.ro_pulse_type())
            ro_lm.set('mixer_alpha'.format(idx),
                      self.ro_pulse_mixer_alpha())
            ro_lm.set('mixer_phi'.format(idx),
                      self.ro_pulse_mixer_phi())

            ro_lm.set('M_modulation_R{}'.format(idx), self.ro_freq_mod())
            ro_lm.set('M_length_R{}'.format(idx),
                      self.ro_pulse_length())
            ro_lm.set('M_amp_R{}'.format(idx),
                      self.ro_pulse_amp())
            ro_lm.set('M_phi_R{}'.format(idx),
                      self.ro_pulse_phi())
            ro_lm.set('M_down_length0_R{}'.format(idx),
                      self.ro_pulse_down_length0())
            ro_lm.set('M_down_amp0_R{}'.format(idx),
                      self.ro_pulse_down_amp0())
            ro_lm.set('M_down_phi0_R{}'.format(idx),
                      self.ro_pulse_down_phi0())
            ro_lm.set('M_down_length1_R{}'.format(idx),
                      self.ro_pulse_down_length1())
            ro_lm.set('M_down_amp1_R{}'.format(idx),
                      self.ro_pulse_down_amp1())
            ro_lm.set('M_down_phi1_R{}'.format(idx),
                      self.ro_pulse_down_phi1())

            ro_lm.acquisition_delay(self.ro_acq_delay())
            ro_lm.load_DIO_triggered_sequence_onto_UHFQC()

            UHFQC.sigouts_0_offset(self.ro_pulse_mixer_offs_I())
            UHFQC.sigouts_1_offset(self.ro_pulse_mixer_offs_Q())

    def _prep_ro_integration_weights(self):
        """
        Sets the ro acquisition integration weights.
        The relevant parameters here are
            ro_acq_weight_type   -> 'SSB', 'DSB' or 'Optimal'
            ro_acq_weight_chI    -> Specifies which integration weight
                (channel) to use
            ro_acq_weight_chQ    -> The second channel in case of SSB/DSB
            RO_acq_weight_func_I -> A custom integration weight (array)
            RO_acq_weight_func_Q ->  ""

        """
        if 'UHFQC' in self.instr_acquisition():
            UHFQC = self.instr_acquisition.get_instr()
            if self.ro_acq_weight_type() == 'SSB':
                UHFQC.prepare_SSB_weight_and_rotation(
                    IF=self.ro_freq_mod(),
                    weight_function_I=self.ro_acq_weight_chI(),
                    weight_function_Q=self.ro_acq_weight_chQ())
            elif self.ro_acq_weight_type() == 'DSB':
                UHFQC.prepare_DSB_weight_and_rotation(
                    IF=self.ro_freq_mod(),
                    weight_function_I=self.ro_acq_weight_chI(),
                    weight_function_Q=self.ro_acq_weight_chQ())
            elif self.ro_acq_weight_type() == 'optimal':
                if (self.ro_acq_weight_func_I() is None or
                        self.ro_acq_weight_func_Q() is None):
                    logging.warning('Optimal weights are None,' +
                                    ' not setting integration weights')
                else:
                    # When optimal weights are used, only the RO I weight
                    # channel is used
                    UHFQC.set('quex_wint_weights_{}_real'.format(
                        self.ro_acq_weight_chI()),
                        self.ro_acq_weight_func_I())
                    UHFQC.set('quex_wint_weights_{}_imag'.format(
                        self.ro_acq_weight_chI()),
                        self.ro_acq_weight_func_Q())
                    UHFQC.set('quex_rot_{}_real'.format(
                        self.ro_acq_weight_chI()), 1.0)
                    UHFQC.set('quex_rot_{}_imag'.format(
                        self.ro_acq_weight_chI()), -1.0)
        else:
            raise NotImplementedError(
                'CBox, DDM or other are currently not supported')

    def prepare_for_timedomain(self):
        self.prepare_readout()
        self._prep_td_sources()
        self._prep_mw_pulses()

    def _prep_td_sources(self):
        self.instr_LO_mw.get_instr().on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.instr_LO_mw.get_instr().frequency.set(
            self.freq_qubit.get() - self.mw_freq_mod.get())

        self.instr_LO_mw.get_instr().power.set(self.mw_pow_td_source.get())

    def _prep_mw_pulses(self):
        MW_LutMan = self.instr_LutMan_MW.get_instr()

        # 4-channels are used for VSM based AWG's.
        MW_LutMan.channel_GI(0+self.mw_awg_ch())
        MW_LutMan.channel_GQ(1+self.mw_awg_ch())
        MW_LutMan.channel_DI(2+self.mw_awg_ch())
        MW_LutMan.channel_DQ(3+self.mw_awg_ch())
        # updating the lutmap is required to make sure channels are correct
        MW_LutMan.set_default_lutmap()

        # Pulse pars
        MW_LutMan.mw_amp180(self.mw_amp180())
        MW_LutMan.mw_amp90_scale(self.mw_amp90_scale())
        MW_LutMan.mw_gauss_width(self.mw_gauss_width())
        MW_LutMan.mw_motzoi(self.mw_motzoi())
        MW_LutMan.mw_modulation(self.mw_freq_mod())

        # Mixer params
        MW_LutMan.G_mixer_phi(self.mw_G_mixer_phi())
        MW_LutMan.G_mixer_alpha(self.mw_G_mixer_alpha())
        MW_LutMan.D_mixer_phi(self.mw_D_mixer_phi())
        MW_LutMan.D_mixer_alpha(self.mw_D_mixer_alpha())
        MW_LutMan.load_waveforms_onto_AWG_lookuptable()

        self._prep_td_configure_VSM()

    def _prep_td_configure_VSM(self):
        # Configure VSM
        # N.B. This configure VSM block is geared specifically to the
        # Duplexer/BlueBox VSM
        VSM = self.instr_VSM.get_instr()
        Gin = self.mw_vsm_ch_Gin()
        Din = self.mw_vsm_ch_Din()
        out = self.mw_vsm_ch_out()

        VSM.set('in{}_out{}_switch'.format(Gin, out), self.mw_vsm_switch())
        VSM.set('in{}_out{}_switch'.format(Din, out), self.mw_vsm_switch())

        VSM.set('in{}_out{}_att'.format(Gin, out), self.mw_vsm_G_att())
        VSM.set('in{}_out{}_att'.format(Din, out), self.mw_vsm_D_att())
        VSM.set('in{}_out{}_phase'.format(Gin, out), self.mw_vsm_G_phase())
        VSM.set('in{}_out{}_phase'.format(Din, out), self.mw_vsm_D_phase())

        self.instr_CC.get_instr().set('vsm_channel_delay{}'.format(self.cfg_qubit_nr()),
                          self.mw_vsm_delay())

    def prepare_for_fluxing(self, reset=True):
        pass

    ####################################################
    # CCL_transmon specifc calibrate_ methods below
    ####################################################

    def calibrate_mw_vsm_delay(self):
        """
        Uploads a sequence for calibrating the vsm
        """
        self.prepare_for_timedomain()
        CCL = self.instr_CC.get_instr()
        CCL.stop()
        p = sqo.vsm_timing_cal_sequence(
            qubit_idx=self.cfg_qubit_nr(), marker_idx=6,
            platf_cfg=self.cfg_openql_platform_fn())
        CCL.upload_instructions(p.filename)
        CCL.start()

    #####################################################
    # "measure_" methods below
    #####################################################

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()
        # Snippet here to create and upload the CCL instructions
        CCL = self.instr_CC.get_instr()
        CCL.stop()
        p = sqo.CW_RO_sequence(qubit_idx=self.cfg_qubit_nr(),
                               platf_cfg=self.cfg_openql_platform_fn())
        CCL.upload_instructions(p.filename)
        # CCL gets started in the int_avg detector

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
            MW_LO_source=self.instr_LO_ro.get_instr(),
            IF=self.ro_freq_mod()))
        MC.set_sweep_points(freqs)

        self.int_avg_det_single._set_real_imag(False)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=True, MC=None,
                             analyze=True, close_fig=True):
        if not pulsed:
            raise NotImplementedError()
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()
        # Snippet here to create and upload the CCL instructions
        CCL = self.instr_CC.get_instr()
        CCL.stop()
        p = sqo.pulsed_spec_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn())
        CCL.upload_instructions(p.filename)
        # CCL gets started in the int_avg detector

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
            MW_LO_source=self.instr_LO_ro.get_instr(),
            IF=self.ro_freq_mod()))
        MC.set_sweep_points(freqs)

        self.int_avg_det_single._set_real_imag(False)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def measure_transients(self, MC=None, analyze: bool=True,
                           cases=('off', 'on'),
                           prepare: bool=True):
        # docstring from parent class
        if MC is None:
            MC = self.instr_MC.get_instr()

        if prepare:
            self.prepare_for_timedomain()
            # off/on switching is achieved by turning the MW source on and
            # off as this is much faster than recompiling/uploading
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
                platf_cfg=self.cfg_openql_platform_fn())
            self.instr_CC.get_instr().upload_instructions(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work

        transients = []
        for i, pulse_comb in enumerate(cases):
            if 'off' in pulse_comb.lower():
                self.instr_LO_mw.get_instr().off()
            elif 'on' in pulse_comb.lower():
                self.instr_LO_mw.get_instr().on()

            s = swf.OpenQL_Sweep(openql_program=p,
                                 CCL=self.instr_CC.get_instr(),
                                 parameter_name='Transient time', unit='s',
                                 upload=prepare)
            MC.set_sweep_function(s)

            if 'UHFQC' in self.instr_acquisition():
                sampling_rate = 1.8e9
            else:
                raise NotImplementedError()
            MC.set_sweep_points(
                np.arange(self.input_average_detector.nr_samples) /
                sampling_rate)
            MC.set_detector_function(self.input_average_detector)
            data = MC.run(
                'Measure_transients{}_{}'.format(self.msmt_suffix, i))
            dset = data['dset']
            transients.append(dset.T[1:])
            if analyze:
                ma.MeasurementAnalysis()

        return [np.array(t, dtype=np.float64) for t in transients]

    def measure_allxy(self, MC=None,
                      analyze=True, close_fig=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.prepare_for_timedomain()
        p = sqo.AllXY(qubit_idx=self.cfg_qubit_nr(), double_points=True,
                      platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('AllXY'+self.msmt_suffix)
        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a
