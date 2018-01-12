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
    mixer_carrier_cancellation)

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

from pycqed.measurement.optimization import nelder_mead


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
        self.add_parameter('instr_spec_source',
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

        self.add_parameter('instr_nested_MC',
                           label='Nested MeasurementControl',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_SH', label='SignalHound',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_FluxCtrl', label='Flux control', docstring=(
                'Instrument used to control flux can either be an IVVI rack '
                'or a meta instrument such as the Flux control.'),
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
                           unit='dBm', initial_value=20,
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
        self.add_parameter(
            'ro_acq_input_average_length',  unit='s',
            label='Readout acquisition delay',
            vals=vals.Numbers(min_value=0, max_value=2.27e-6),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=('The measurement time in input averaging.'))

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
        self.add_parameter('mw_mixer_offs_GI',
                           unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_mixer_offs_GQ', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_mixer_offs_DI',
                           unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mw_mixer_offs_DQ', unit='V',
                           parameter_class=ManualParameter, initial_value=0)

        self.add_parameter('mw_pow_td_source',
                           label='Time-domain power',
                           unit='dBm',
                           initial_value=20,
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
            set_cmd=self._set_mw_vsm_delay,
            get_cmd=self._get_mw_vsm_delay)

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

    def _set_mw_vsm_delay(self, val):
        # sort of a pseudo Manual Parameter
        self.instr_CC.get_instr().set(
            'vsm_channel_delay{}'.format(self.cfg_qubit_nr()), val)
        self._mw_vsm_delay = val

    def _get_mw_vsm_delay(self):
        return self._mw_vsm_delay

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
            'spec_type', parameter_class=ManualParameter, docstring=(
                'determines what kind of spectroscopy to do, \n'
                '"CW":  opens the relevant VSM channel to always let the tone '
                'through. \n'
                '"vsm_gated":  uses the  VSM in external mode to gate the spec '
                'source. \n '
                '"IQ" uses the TD source and AWG8 to generate a spec pulse'),
            initial_value='vsm_gated',
            vals=vals.Enum('CW', 'IQ', 'vsm_gated'))

        self.add_parameter(
            'spec_amp', unit='V', docstring=(
                'Amplitude of the spectroscopy pulse in the mw LutMan. '
                'The power of the spec pulse should be controlled through '
                'the vsm attenuation "spec_vsm_att"'),
            vals=vals.Numbers(0, 1), parameter_class=ManualParameter,
            initial_value=0.8)
        self.add_parameter(
            'spec_pow', unit='dB',
            vals=vals.Numbers(-70, 20),
            parameter_class=ManualParameter,
            initial_value=-30)

    def add_flux_parameters(self):
        # fl_dc_ is the prefix for DC flux bias related params
        self.add_parameter(
            'fl_dc_V_per_phi0', label='Flux bias V/Phi0',
            docstring='Conversion factor for flux bias',
            vals=vals.Numbers(), unit='V', initial_value=1,
            parameter_class=ManualParameter)
        self.add_parameter(
            'fl_dc_V', label='Flux bias', unit='V',
            docstring='Current flux bias setting', vals=vals.Numbers(),
            initial_value=0, parameter_class=ManualParameter)
        self.add_parameter(
            'fl_dc_V0', unit='V', docstring=(
                'Flux bias offset corresponding to the sweetspot'),
            vals=vals.Numbers(), initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'fl_dc_ch',  docstring=(
                'Flux bias channel'),
            vals=vals.Ints(), initial_value=1,
            parameter_class=ManualParameter)

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

        self.add_parameter('cfg_qubit_freq_calc_method',
                           initial_value='latest',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('latest', 'flux'))
        self.add_parameter('cfg_cycle_time',
                           initial_value=20e-9,
                           unit='s',
                           parameter_class=ManualParameter,
                           # this is to effictively hardcode the cycle time
                           vals=vals.Enum(20e-9))
        # TODO: add docstring (Oct 2017)
        self.add_parameter('cfg_prepare_mw_awg', vals=vals.Bool(),
                           initial_value=True,
                           parameter_class=ManualParameter)

    def add_generic_qubit_parameters(self):
        self.add_parameter('E_c', unit='Hz',
                           initial_value=300e6,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('E_j', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T1', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 200e-6))
        self.add_parameter('T2_echo', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 200e-6))
        self.add_parameter('T2_star', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 200e-6))

        self.add_parameter('freq_qubit',
                           label='mwubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('freq_max',
                           label='mwubit sweet spot frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('freq_res',
                           label='Resonator frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('asymmetry', unit='',
                           docstring='Asymmetry parameter of the SQUID loop',
                           initial_value=0,
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
        # source is turned on in measure spec when needed
        self.instr_LO_mw.get_instr().off()
        self.instr_spec_source.get_instr().off()

    def _prep_cw_spec(self):
        VSM = self.instr_VSM.get_instr()
        VSM.set_all_switches_to('OFF')
        if self.spec_type() == 'CW':
            mode = 'ON'
        else:
            mode = 'EXT'
        VSM.set('in{}_out{}_switch'.format(self.spec_vsm_ch_in(),
                                           self.spec_vsm_ch_out()), mode)
        VSM.set('in{}_out{}_att'.format(
                self.spec_vsm_ch_in(), self.spec_vsm_ch_out()),
                self.spec_vsm_att())
        self.instr_spec_source.get_instr().power(self.spec_pow())

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
                nr_averages=self.ro_acq_averages(),
                nr_samples=int(self.ro_acq_input_average_length()*1.8e9))

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
            ro_lm.set('mixer_alpha',
                      self.ro_pulse_mixer_alpha())
            ro_lm.set('mixer_phi',
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
        if self.cfg_prepare_mw_awg():
            MW_LutMan.load_waveforms_onto_AWG_lookuptable()

        self._prep_td_configure_VSM()

        # N.B. This part is AWG8 specific
        AWG = MW_LutMan.AWG.get_instr()
        AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch()-1),
                self.mw_mixer_offs_GI())
        AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch()+0),
                self.mw_mixer_offs_GQ())
        AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch()+1),
                self.mw_mixer_offs_DI())
        AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch()+2),
                self.mw_mixer_offs_DQ())

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

        self.instr_CC.get_instr().set(
            'vsm_channel_delay{}'.format(self.cfg_qubit_nr()),
            self.mw_vsm_delay())

    def prepare_for_fluxing(self, reset=True):
        pass

    ####################################################
    # CCL_transmon specifc calibrate_ methods below
    ####################################################

    def calibrate_mw_vsm_delay(self):
        """
        Uploads a sequence for calibrating the vsm delay.
        The experiment consists of a single square pulse of 20 ns that
        triggers both the VSM channel specified and the AWG8.

        Note: there are two VSM markers, align with the first of two.

        By changing the "mw_vsm_delay" parameter the delay can be calibrated.
        N.B. Ensure that the signal is visible on a scope or in the UFHQC
        readout first!
        """
        self.prepare_for_timedomain()
        CCL = self.instr_CC.get_instr()
        CCL.stop()
        p = sqo.vsm_timing_cal_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn())
        CCL.upload_instructions(p.filename)
        CCL.start()
        print('CCL program is running. Parameter "mw_vsm_delay" can now be '
              'calibrated by hand.')

    def calibrate_mixer_offsets_drive(self, update: bool =True)-> bool:
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''

        # turn relevant channels on
        MW_LutMan = self.instr_LutMan_MW.get_instr()
        AWG = MW_LutMan.AWG.get_instr()
        # This part is AWG8 specific and wont work with a QWG
        awg_ch = self.mw_awg_ch()
        AWG.stop()
        AWG.set('sigouts_{}_on'.format(awg_ch-1), 1)
        AWG.set('sigouts_{}_on'.format(awg_ch), 1)
        chGI_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch-1)]
        chGQ_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch+0)]
        # End of AWG8 specific part

        VSM = self.instr_VSM.get_instr()
        VSM.set_all_switches_to('OFF')
        Gin = self.mw_vsm_ch_Gin()
        Din = self.mw_vsm_ch_Din()
        out = self.mw_vsm_ch_out()
        VSM.set('in{}_out{}_switch'.format(Gin, out), 'ON')
        VSM.set('in{}_out{}_switch'.format(Din, out), 'OFF')

        # Calibrate Gaussian component mixer
        offset_I, offset_Q = mixer_carrier_cancellation(
            SH=self.instr_SH.get_instr(), source=self.instr_LO_mw.get_instr(),
            MC=self.instr_MC.get_instr(),
            chI_par=chGI_par, chQ_par=chGQ_par, x0=(0.5, 0.5))
        if update:
            self.mw_mixer_offs_GI(offset_I)
            self.mw_mixer_offs_GQ(offset_Q)

        VSM.set('in{}_out{}_switch'.format(Gin, out), 'OFF')
        VSM.set('in{}_out{}_switch'.format(Din, out), 'ON')

        # This part is AWG8 specific and wont work with a QWG
        AWG.set('sigouts_{}_on'.format(awg_ch+1), 1)
        AWG.set('sigouts_{}_on'.format(awg_ch+2), 1)
        chDI_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch+1)]
        chDQ_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch+2)]
        # End of AWG8 specific part

        # Calibrate Derivative component mixer
        offset_I, offset_Q = mixer_carrier_cancellation(
            SH=self.instr_SH.get_instr(), source=self.instr_LO_mw.get_instr(),
            MC=self.instr_MC.get_instr(),
            chI_par=chDI_par, chQ_par=chDQ_par, x0=(0.5, 0.5))
        if update:
            self.mw_mixer_offs_DI(offset_I)
            self.mw_mixer_offs_DQ(offset_Q)
        return True

    def calibrate_mixer_offsets_RO(self, update: bool=True) -> bool:
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        '''
        chI_par = self.instr_acquisition.get_instr().sigouts_0_offset
        chQ_par = self.instr_acquisition.get_instr().sigouts_1_offset

        offset_I, offset_Q = mixer_carrier_cancellation(
            SH=self.instr_SH.get_instr(), source=self.instr_LO_ro.get_instr(),
            MC=self.instr_MC.get_instr(),
            chI_par=chI_par, chQ_par=chQ_par, x0=(0.05, 0.05))

        if update:
            self.ro_pulse_mixer_offs_I(offset_I)
            self.ro_pulse_mixer_offs_Q(offset_Q)
        return True

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
            logging.warning('CCL transmon can only perform '
                            'pulsed spectrocsopy')
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()
        # Snippet here to create and upload the CCL instructions
        CCL = self.instr_CC.get_instr()
        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn())
        CCL.upload_instructions(p.filename)
        # CCL gets started in the int_avg detector

        # The spec pulse is a MW pulse that contains not modulation

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        self.int_avg_det_single._set_real_imag(False)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='spectroscopy_'+self.msmt_suffix)
        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def measure_ssro(self, MC=None, analyze: bool=True, nr_shots: int=1024*8,
                     cases=('off', 'on'), update_threshold: bool=True,
                     prepare: bool=True, no_figs: bool=False,
                     update: bool=True,
                     verbose: bool=True):
        old_RO_digit = self.ro_acq_digitized()
        self.ro_acq_digitized(False)
        # docstring from parent class
        if MC is None:
            MC = self.instr_MC.get_instr()

        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        MC.soft_avg(1)  # don't want to average single shots
        if prepare:
            self.prepare_for_timedomain()
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb='off_on',
                platf_cfg=self.cfg_openql_platform_fn())
            self.instr_CC.get_instr().upload_instructions(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work

        # digitization setting is reset here but the detector still uses
        # the disabled setting that was set above
        self.ro_acq_digitized(old_RO_digit)

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=prepare)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.int_log_det
        MC.set_detector_function(d)
        MC.run(
            'Measure_SSRO_{}'.format(self.msmt_suffix))
        MC.live_plot_enabled(old_plot_setting)
        if analyze:
            if len(d.value_names) == 1:
                a = ma2.Singleshot_Readout_Analysis(
                    t_start=None, t_stop=None,
                    options_dict={'scan_label': 'SSRO'},
                    extract_only=no_figs)
                if update_threshold:
                    # UHFQC threshold is wrong, the magic number is a
                    #  dirty hack. This works. we don't know why.
                    magic_scale_factor = 0.655
                    self.ro_acq_threshold(a.proc_data_dict['threshold_raw'] *
                                          magic_scale_factor)
                if update:
                    self.F_ssro(a.proc_data_dict['F_assignment_raw'])
                    self.F_discr(a.proc_data_dict['F_discr'])
                if verbose:
                    print('Avg. Assignement fidelity: \t{:.4f}\n'.format(
                        a.proc_data_dict['F_assignment_raw']) +
                        'Avg. Discrimination fidelity: \t{:.4f}'.format(
                        a.proc_data_dict['F_discr']))
                return (a.proc_data_dict['F_assignment_raw'],
                        a.proc_data_dict['F_discr'])
            else:

                a = ma.SSRO_Analysis(label='SSRO',
                                     channels=d.value_names,
                                     no_fits=no_figs, rotate=False)
                return None, None

    def measure_transients(self, MC=None, analyze: bool=True,
                           cases=('off', 'on'),
                           prepare: bool=True, depletion_analysis: bool=True,
                           depletion_analysis_plot: bool=True,
                           depletion_optimization_window=None):
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
        if depletion_analysis:
            a = ma.Input_average_analysis(
                IF=self.ro_freq_mod(),
                optimization_window=depletion_optimization_window,
                plot=depletion_analysis_plot)
            return a
        else:
            return [np.array(t, dtype=np.float64) for t in transients]

    def calibrate_optimal_weights(self, MC=None, verify: bool=True,
                                  analyze: bool=True, update: bool=True,
                                  no_figs: bool=False)->bool:
        if MC is None:
            MC = self.instr_MC.get_instr()

        # Ensure that enough averages are used to get accurate weights
        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(2**15)
        transients = self.measure_transients(MC=MC, analyze=analyze,
                                             depletion_analysis=False)
        if analyze:
            ma.Input_average_analysis(IF=self.ro_freq_mod())

        self.ro_acq_averages(old_avg)

        # Calculate optimal weights
        optimized_weights_I = transients[1][0] - transients[0][0]
        optimized_weights_Q = transients[1][1] - transients[0][1]

        # joint rescaling to +/-1 Volt
        maxI = np.max(np.abs(optimized_weights_I))
        maxQ = np.max(np.abs(optimized_weights_Q))
        # fixme: deviding the weight functions by four to not have overflow in
        # thresholding of the UHFQC
        weight_scale_factor = 1./(4*np.max([maxI, maxQ]))
        optimized_weights_I = np.array(
            weight_scale_factor*optimized_weights_I)
        optimized_weights_Q = np.array(
            weight_scale_factor*optimized_weights_Q)
        self.ro_acq_averages(old_avg)

        if update:
            self.ro_acq_weight_func_I(optimized_weights_I)
            self.ro_acq_weight_func_Q(optimized_weights_Q)
            self.ro_acq_weight_type('optimal')

        if verify:
            self.measure_ssro(no_figs=no_figs)
        return True

    def measure_rabi_vsm(self, MC=None, atts=np.linspace(0, 65536, 31),
                         analyze=True, close_fig=True,
                         prepare_for_timedomain=True):
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        p = sqo.off_on(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
            platf_cfg=self.cfg_openql_platform_fn())

        VSM = self.instr_VSM.get_instr()
        Gin = self.mw_vsm_ch_Gin()
        # FIXME: This variable is not used, both main and derivative should
        # be swept
        Din = self.mw_vsm_ch_Din()
        out = self.mw_vsm_ch_out()

        self.instr_CC.get_instr().upload_instructions(p.filename)

        MC.set_sweep_function(VSM.__getattr__(
            'in{}_out{}_att'.format(Gin, out)))
        MC.set_sweep_points(atts)
        # real_imag is acutally not polar and as such works for opt weights
        self.int_avg_det_single._set_real_imag(True)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='rabi_'+self.msmt_suffix)

    def measure_allxy(self, MC=None,
                      analyze=True, close_fig=True,
                      prepare_for_timedomain=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
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
            return a.deviation_total

    def calibrate_single_qubit_gates_allxy(self, nested_MC=None, f_start=None,
                                           G_start=None, D_start=None,
                                           initial_steps=None):
        # FIXME: this tuneup does not update the qubit object parameters
        # FIXME2: this tuneup does not return True upon success
        if f_start is None:
            f_start = self.freq_qubit()

        if G_start is None:
            G_start = self.mw_vsm_G_att()

        if D_start is None:
            D_start = self.mw_vsm_D_att()

        if initial_steps is None:
            initial_steps = [1e6, 4e2, 2e3]

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        nested_MC.set_sweep_functions([
            self.freq_qubit,
            self.mw_vsm_G_att,
            self.mw_vsm_D_att])

        d = det.Function_Detector(self.measure_allxy,
                                  value_names=['AllXY cost'],
                                  value_units=['a.u.'],)
        nested_MC.set_detector_function(d)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': [f_start, G_start, D_start],
                        'initial_step': initial_steps,
                        'no_improv_break': 10,
                        'minimize': True,
                        'maxiter': 500}

        nested_MC.set_adaptive_function_parameters(ad_func_pars)
        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='gate_tuneup_allxy', mode='adaptive')
        ma.OptimizationAnalysis(label='gate_tuneup_allxy')

    def calibrate_deletion_pulse_transients(
            self, nested_MC=None, amp0=None,
            amp1=None, phi0=180, phi1=0, initial_steps=None, two_par=True,
            depletion_optimization_window=None, depletion_analysis_plot=False):
        """
        this function automatically tunes up a two step, four-parameter
        depletion pulse.
        It uses the averaged transients for ground and excited state for its
        cost function.
        two_par:    if readout is performed at the symmetry point and in the
                    linear regime two parameters will suffice. Othen, four
                    paramters do not converge.
                    First optimizaing the amplitudes (two paramters) and
                    then run the full 4 paramaters with the correct initial
                    amplitudes works.
        optimization_window:  optimization window determins which part of
                    the transients will be
                    nulled in the optimization. By default it uses a
                    window of 500 ns post depletiona with a 50 ns buffer.
        initial_steps:  These have to be given in the order
                       [phi0,phi1,amp0,amp1] for 4-par tuning and
                       [amp0,amp1] for 2-par tunining
        """
        # FIXME: this calibration does not update the qubit object params
        # FIXME2: this calibration does not return a boolean upon success

        # tuneup requires nested MC as the transients detector will use MC
        self.ro_pulse_type('up_down_down')
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        # setting the initial depletion amplitudes
        if amp0 is None:
            amp0 = 2*self.ro_pulse_amp()
        if amp1 is None:
            amp1 = 0.5*self.ro_pulse_amp()

        if depletion_optimization_window is None:
            depletion_optimization_window = [
                self.ro_pulse_length()+self.ro_pulse_down_length0()
                + self.ro_pulse_down_length1()+50e-9,
                self.ro_pulse_length()+self.ro_pulse_down_length0()
                + self.ro_pulse_down_length1()+550e-9]

        if two_par:
            nested_MC.set_sweep_functions([
                self.ro_pulse_down_amp0,
                self.ro_pulse_down_amp1])
        else:
            nested_MC.set_sweep_functions([self.ro_pulse_down_phi0,
                                           self.ro_pulse_down_phi1,
                                           self.ro_pulse_down_amp0,
                                           self.ro_pulse_down_amp1])
        d = det.Function_Detector(self.measure_transients,
                                  msmt_kw={'depletion_analysis': True,
                                           'depletion_analysis_plot':
                                           depletion_analysis_plot,
                                           'depletion_optimization_window':
                                           depletion_optimization_window},
                                  value_names=['depletion cost'],
                                  value_units=['au'],
                                  result_keys=['depletion_cost'])
        nested_MC.set_detector_function(d)

        if two_par:
            if initial_steps is None:
                initial_steps = [-0.1*amp0, -0.1*amp1]
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': [amp0, amp1],
                            'initial_step': initial_steps,
                            'no_improv_break': 12,
                            'minimize': True,
                            'maxiter': 500}
            self.ro_pulse_down_phi0(180)
            self.ro_pulse_down_phi1(0)

        else:
            if initial_steps is None:
                initial_steps = [15, 15, -0.1*amp0, -0.1*amp1]
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': [phi0, phi1, amp0, amp1],
                            'initial_step': initial_steps,
                            'no_improv_break': 12,
                            'minimize': True,
                            'maxiter': 500}
        nested_MC.set_adaptive_function_parameters(ad_func_pars)
        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='depletion_tuneup', mode='adaptive')
        ma.OptimizationAnalysis(label='depletion_tuneup')

    def measure_error_fraction(self, MC=None, analyze: bool=True,
                               nr_shots: int=2048*4,
                               sequence_type='echo', prepare: bool=True,
                               feedback=False,
                               depletion_time=None, net_gate='pi'):
        # docstring from parent class
        # this performs a multiround experiment, the repetition rate is defined
        # by the ro_duration which can be changed by regenerating the
        # configuration file.
        # The analysis counts single errors. The definition of an error is
        # adapted automatically by choosing feedback or the net_gate.
        # it requires high SNR single shot readout and a calibrated threshold
        self.ro_acq_digitized(True)
        if MC is None:
            MC = self.instr_MC.get_instr()

        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        MC.soft_avg(1)  # don't want to average single shots
        if prepare:
            self.prepare_for_timedomain()
            # off/on switching is achieved by turning the MW source on and
            # off as this is much faster than recompiling/uploading
            p = sqo.RTE(
                qubit_idx=self.cfg_qubit_nr(), sequence_type=sequence_type,
                platf_cfg=self.cfg_openql_platform_fn(), net_gate=net_gate,
                feedback=feedback)
            self.instr_CC.get_instr().upload_instructions(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='shot nr', unit='#',
                             upload=prepare)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.int_log_det
        MC.set_detector_function(d)

        exp_metadata = {'feedback': feedback, 'sequence_type': sequence_type,
                        'depletion_time': depletion_time, 'net_gate': net_gate}
        suffix = 'depletion_time_{}_ro_pulse_type_{}_feedback_{}_net_gate_{}'.format(
            depletion_time, self.ro_pulse_type(), feedback, net_gate)
        MC.run(
            'Measure_error_fraction_{}_{}'.format(self.msmt_suffix, suffix),
            exp_metadata=exp_metadata)
        MC.live_plot_enabled(old_plot_setting)
        if analyze:
            a = ma2.Single_Qubit_RoundsToEvent_Analysis(
                t_start=None, t_stop=None,
                options_dict={'typ_data_idx': 0,
                              'scan_label': 'error_fraction'},
                extract_only=True)
            return a.proc_data_dict['frac_single']

    def measure_T1(self, times=None, MC=None,
                   analyze=True, close_fig=True, update=True,
                   prepare_for_timedomain=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            times = np.linspace(0, self.T1()*4, 31)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                    times[-1]+3*dt,
                                    times[-1]+4*dt)])
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        p = sqo.T1(times, qubit_idx=self.cfg_qubit_nr(),
                   platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             parameter_name='Time',
                             unit='s',
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('T1'+self.msmt_suffix)
        a = ma.T1_Analysis(auto=True, close_fig=True)
        if update:
            self.T1(a.T1)
        return a.T1

    def measure_Ramsey(self, times=None, MC=None, artificial_detuning=None,
                       analyze=True, close_fig=True, update=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband
            # modulation
            stepsize = (self.T2_star()*4/61)//(abs(self.cfg_cycle_time())) \
                * abs(self.cfg_cycle_time())
            times = np.arange(0, self.T2_star()*4, stepsize)

        if artificial_detuning is None:
            artificial_detuning = 3/times[-1]

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                    times[-1]+3*dt,
                                    times[-1]+4*dt)])

        self.prepare_for_timedomain()

        # testing if the pulses are locked to the modulation frequency
        if not all([np.round(t*1e9) % (1/self.mw_freq_mod.get()*1e9)
                    == 0 for t in times]):
            raise ValueError(
                'timesteps must be multiples of modulation period')

        # adding 'artificial' detuning by detuning the qubit LO
        freq_qubit = self.freq_qubit()
        # # this should have no effect if artificial detuning = 0
        self.instr_LO_mw.get_instr().set(
            'frequency', freq_qubit -
            self.mw_freq_mod.get() + artificial_detuning)

        p = sqo.Ramsey(times, qubit_idx=self.cfg_qubit_nr(),
                       platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('Ramsey'+self.msmt_suffix)
        a = ma.Ramsey_Analysis(auto=True, close_fig=True,
                               freq_qubit=freq_qubit,
                               artificial_detuning=artificial_detuning)
        if update:
            self.T2_star(a.T2_star['T2_star']) # dict containing val and stderr
        return a.T2_star

    def measure_echo(self, times=None, MC=None,
                     analyze=True, close_fig=True, update=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband
            # modulation
            stepsize = (self.T2_echo()*2/61)//(abs(self.cfg_cycle_time())) \
                * abs(self.cfg_cycle_time())
            times = np.arange(0, self.T2_echo()*4, stepsize*2)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                    times[-1]+3*dt,
                                    times[-1]+4*dt)])

        # # Checking if pulses are on 20 ns grid
        if not all([np.round(t*1e9) % (2*self.cfg_cycle_time()*1e9) == 0 for
                    t in times]):
            raise ValueError('timesteps must be multiples of 40e-9')

        # # Checking if pulses are locked to the pulse modulation
        if not all([np.round(t/1*1e9) % (2/self.mw_freq_mod.get()*1e9)
                    == 0 for t in times]):
            raise ValueError(
                'timesteps must be multiples of 2 modulation periods')

        self.prepare_for_timedomain()
        p = sqo.echo(times, qubit_idx=self.cfg_qubit_nr(),
                     platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('echo'+self.msmt_suffix)
        a = ma.Echo_analysis(label='echo', auto=True, close_fig=True)
        if update:
            self.T2_echo(a.fit_res.params['tau'].value)
        return a
