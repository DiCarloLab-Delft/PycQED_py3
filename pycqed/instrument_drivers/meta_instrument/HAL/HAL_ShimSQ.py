"""
File:   HAL_ShimSQ.py : HAL shim Single Qubit
Note:   extracted from HAL_Transmon.py (originally CCL_Transmon.py)

This file provides class HAL_ShimSQ, which implements a shim between the HAL_Transmon and the instrument hardware for
single qubit operations. It contains hardware dependent functions extracted from CCL_Transmon.py, extended with
functions that abstract the instrument hardware that was directly accessed by the end user methods.
FIXME: the latter is Work In Progress, so old style code is still present in HAL_Transmon

QCoDeS parameters referring to instrument hardware are added here, and not in child class HAL_Transmon where they were
originally added. These parameters should only accessed here (although nothing really stops you from violating this
design). Note that we try to find a balance between compatibility with exiting code and proper design here.

The following hardware related attributes are managed here:
- physical instruments of the signal chain, and their
    - connectivity
    - settings
    - modes (if any)
    - signal chain related properties (e.g. modulation, mixer calibration)
    - FIXME: etc


A future improvement is to merge this HAL functions for Single Qubits with those for Multi Qubits.

Note:   a lot code was moved around within this file in December 2021. As a consequence, the author information provided
        by 'git blame' makes little sense. See GIT tag 'release_v0.3' for the original file.
"""


import logging
import warnings
import numpy as np
from deprecated import deprecated


from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object import Qubit
from pycqed.measurement import detector_functions as det

# Imported for type checks
from pycqed.instrument_drivers.physical_instruments.QuTech_AWG_Module import QuTech_AWG_Module
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC

from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter

log = logging.getLogger(__name__)


class HAL_ShimSQ(Qubit):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)  # FIXME: Qubit should be below us in object hierarchy, but it inherits from Instrument

        self._add_instrument_ref_parameters()
        self._add_config_parameters()
        self._add_mw_parameters()
        self._add_mw_vsm_parameters()
        self._add_spec_parameters()
        self._add_flux_parameters()
        self._add_ro_parameters()
        self._add_prep_parameters()

    ##########################################################################
    # Prepare functions
    ##########################################################################

    def prepare_for_continuous_wave(self):
        if 'optimal' in self.ro_acq_weight_type():
            warnings.warn('Changing ro_acq_weight_type to SSB.')
            self.ro_acq_weight_type('SSB')
        if self.ro_acq_weight_type() not in {'DSB', 'SSB'}:
            # this is because the CW acquisition detects using angle and phase
            # and this requires two channels to rotate the signal properly.
            raise ValueError('Readout "{}" '.format(self.ro_acq_weight_type())
                             + 'weight type must be "SSB" or "DSB"')

        if self.cfg_with_vsm():
            self._prep_cw_configure_VSM()

        self.prepare_readout(CW=True)
        self._prep_cw_spec()

        # source is turned on in measure spec when needed
        self.instr_LO_mw.get_instr().off()

        if self.instr_spec_source() != None:
            self.instr_spec_source.get_instr().off()
        if self.instr_spec_source_2() != None:
            self.instr_spec_source_2.get_instr().off()

    def prepare_readout(self, CW=False):
        """
        Configures the readout. Consists of the following steps
        - instantiate the relevant detector functions
        - set the microwave frequencies and sources
        - generate the RO pulse
        - set the integration weights
        """
        if self.cfg_prepare_ro_awg():
            self.instr_acquisition.get_instr().load_default_settings(upload_sequence=False)
            self._prep_ro_pulse(CW=CW)
            self._prep_ro_integration_weights()
            self._prep_deskewing_matrix()
        else:
            warnings.warn('"cfg_prepare_ro_awg" set to False, not preparing readout .')

        self._prep_ro_instantiate_detectors()
        self._prep_ro_sources()

    def prepare_for_timedomain(self):
        self.prepare_readout()
        self._prep_td_sources()
        self._prep_mw_pulses()
        if self.cfg_with_vsm():
            self._prep_td_configure_VSM()

    @deprecated(version='0.4', reason="unused")
    def prepare_for_fluxing(self, reset=True):
        pass

    def prepare_characterizing(self, exceptions: list = [], verbose=True):
        # USED_BY: device_dependency_graphs.py
        """
        Prepares the qubit for (automatic) characterisation. Will park all
        other qubits in the device object to their 'anti-sweetspot' (which is a
        sweetspot as well technically speaking). Afterwards, it will move
        the qubit to be characterized (self) to its sweetspot.

        Will ignore any qubit whose name (string) is in 'exceptions'
        """

        fluxcurrent = self.instr_FluxCtrl.get_instr()
        device = self.instr_device.get_instr()

        exceptions.append('fakequbit')
        Qs = device.qubits()
        for Q in Qs:
            if device.find_instrument(Q).fl_dc_I_per_phi0() == 1:
                exceptions.append(Q)
        # exceptions.append('D2')
        # First park all other qubits to anti sweetspot
        print('Moving other qubits away ...')
        for qubit_name in device.qubits():
            if (qubit_name not in exceptions) and (qubit_name != self.name):
                qubit = device.find_instrument(qubit_name)
                channel = qubit.fl_dc_ch()
                current = qubit.fl_dc_I0() + qubit.fl_dc_I_per_phi0() / 2
                fluxcurrent[channel](current)
                if verbose:
                    print('\t Moving {} to {:.3f} mA'
                          .format(qubit_name, current / 1e-3))
        # Move self to sweetspot:
        if verbose:
            print('Moving {} to {:.3f} mA'.format(
                self.name, self.fl_dc_I0() / 1e-3))
        fluxcurrent[self.fl_dc_ch()](self.fl_dc_I0())
        return True

    ##########################################################################
    # HAL functions
    # naming convention: hal_<subsystem>_<function>
    #
    # FIXME: WIP to move instrument handling out of measurement/calibration/etc
    #  routines to separate functions.
    #  These are *identical* to the original code that was replaced by function calls
    ##########################################################################

    def hal_acq_spec_mode_on(self):
        """Starting specmode of acquisition instrument if set in config"""
        if self.cfg_spec_mode():
            UHFQC = self.instr_acquisition.get_instr()
            UHFQC.spec_mode_on(
                IF=self.ro_freq_mod(),
                ro_amp=self.ro_pulse_amp_CW()
            )

    def hal_acq_spec_mode_off(self):
        """Stopping specmode of acquisition instrument"""
        if self.cfg_spec_mode():
            UHFQC = self.instr_acquisition.get_instr()
            UHFQC.spec_mode_off()
            self._prep_ro_pulse(upload=True)

    def hal_spec_source_on(self, power: float, pulsed: bool = False):
        """

        Args:
            power:
            pulsed:

        Returns:

        """
        # FIXME: pulsed operation requires:
        # - a generator that supports it
        # - a device that generates the pulses (e.g. CC with marker output)
        # - a connection between the two
        # FIXME: some functions use pulsed operation, but leave the generator in that state. Other functions don't
        #  touch pulsemod_state, and thus depend on what runs before

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        if pulsed:
            spec_source.pulsemod_state('On')
        else:
            spec_source.pulsemod_state('Off')
        spec_source.power(power)
        return spec_source  # FIXME: exposes hardware detail, but needed for sweeps

    def hal_spec_source_off(self):
        spec_source = self.instr_spec_source.get_instr()
        spec_source.off()

    def hal_flux_get_parameters(self, flux_chan):
        if 'ivvi' in self.instr_FluxCtrl().lower():  # FIXME: checks name, not type
            IVVI = self.instr_FluxCtrl.get_instr()
            if flux_chan is None:
                dac_par = IVVI.parameters['dac{}'.format(self.fl_dc_ch())]
            else:
                dac_par = IVVI.parameters[flux_chan]
        else:
            # Assume the flux is controlled using an SPI rack
            fluxcontrol = self.instr_FluxCtrl.get_instr()
            if flux_chan == None:
                dac_par = fluxcontrol.parameters[(self.fl_dc_ch())]
            else:
                dac_par = fluxcontrol.parameters[(flux_chan)]

        return dac_par

    ##########################################################################
    # Other functions
    ##########################################################################

    # FIXME: compare against self.int_avg_det_single provided by _prep_ro_instantiate_detectors, and why is this detector
    #  created on demand

    def get_int_avg_det(self, **kw):
        """
        Instantiates an integration average detector using parameters from
        the qubit object. **kw get passed on to the class when instantiating
        the detector function.
        """

        if self.ro_acq_weight_type() == 'optimal':
            ro_channels = [self.ro_acq_weight_chI()]

            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'
            else:
                result_logging_mode = 'lin_trans'
        else:
            ro_channels = [self.ro_acq_weight_chI(),
                           self.ro_acq_weight_chQ()]
            result_logging_mode = 'raw'

        int_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=self.instr_acquisition.get_instr(),
            AWG=self.instr_CC.get_instr(),
            channels=ro_channels,
            result_logging_mode=result_logging_mode,
            nr_averages=self.ro_acq_averages(),
            integration_length=self.ro_acq_integration_length(),
            **kw
        )

        return int_avg_det


    ##########################################################################
    # Private _add_*_parameters
    ##########################################################################

    def _add_instrument_ref_parameters(self):
        self.add_parameter(
            'instr_device',
            docstring='Represents sample, contains all qubits and resonators',
            parameter_class=InstrumentRefParameter)

        # MW sources
        self.add_parameter(
            'instr_LO_ro',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_LO_mw',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_spec_source',
            docstring='instrument used to apply CW excitation',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_spec_source_2',
            docstring='instrument used to apply second MW drive',
            parameter_class=InstrumentRefParameter)

        # Control electronics
        if 1:
            self.add_parameter(
                'instr_CC',
                label='Central Controller',
                docstring='Device responsible for controlling the experiment.',
                parameter_class=InstrumentRefParameter)
        else:
            # new style parameter definition, with type annotation.
            # FIXME: requires recent QCoDeS, which requires Python 3.7
            # FIXME: we should introduce base class for CC-type devices below CC
            self.instr_CC: CC = InstrumentRefParameter(
                'instr_CC',
                label='Central Controller',
            )
            """Device responsible for controlling the experiment."""

        self.add_parameter(
            'instr_acquisition',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_VSM',
            label='Vector Switch Matrix',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_FluxCtrl',
            label='Flux control',
            docstring=(
                'Instrument used to control flux can either be an IVVI rack '
                'or a meta instrument such as Flux_Control.'),
            parameter_class=InstrumentRefParameter)

        self.add_parameter(
            'instr_SH',
            label='SignalHound',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_VNA',
            docstring='Vector Network Analyzer',
            parameter_class=InstrumentRefParameter,
            initial_value=None)

        # Measurement Control
        # FIXME: move back to HAL_Transmon
        self.add_parameter(
            'instr_MC',
            label='MeasurementControl',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_nested_MC',
            label='Nested MeasurementControl',
            parameter_class=InstrumentRefParameter)

        # LutMan's
        self.add_parameter(
            'instr_LutMan_MW',
            docstring='Lookuptable manager for microwave control pulses.',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_LutMan_RO',
            docstring='Lookuptable manager responsible for microwave readout pulses.',
            parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_LutMan_Flux',
            docstring='Lookuptable manager responsible for flux pulses.',
            initial_value=None,
            parameter_class=InstrumentRefParameter)

    def _add_config_parameters(self):
        self.add_parameter(
            'cfg_qubit_nr',
            label='Qubit number',
            vals=vals.Ints(0, 20),
            parameter_class=ManualParameter,
            initial_value=0,
            docstring='The qubit number is used in the OpenQL compiler.')

        self.add_parameter(
            'cfg_prepare_ro_awg',
            vals=vals.Bool(),
            docstring=('If False, disables uploading pulses to UHFQC'),
            initial_value=True,
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_prepare_mw_awg',
            vals=vals.Bool(),
            docstring=('If False, disables uploading pulses to AWG8'),
            initial_value=True,
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_with_vsm',
            vals=vals.Bool(),
            docstring=('to avoid using the VSM if set to False bypasses all commands to vsm if set False'),
            initial_value=True,
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_spec_mode',
            vals=vals.Bool(),
            docstring=('Used to activate spec mode in measurements'),
            initial_value=False,
            parameter_class=ManualParameter)

    def _add_mw_parameters(self):
        self.add_parameter(
            'mw_awg_ch', parameter_class=ManualParameter,
            initial_value=1,
            vals=vals.Ints())

        self.add_parameter(  # NB: only used for/available on HDAWG
            'mw_channel_range',
            label='AWG channel range. WARNING: Check your hardware specific limits!',
            unit='V',
            initial_value=.8,
            vals=vals.Enum(0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5),
            parameter_class=ManualParameter)

        self.add_parameter(
            'mw_freq_mod',
            initial_value=-100e6,
            label='pulse-modulation frequency',
            unit='Hz',
            parameter_class=ManualParameter)

        self.add_parameter(
            'mw_pow_td_source',
            label='Time-domain power',
            unit='dBm',
            initial_value=20,
            parameter_class=ManualParameter)

        # parameters for *MW_LutMan: mixer skewness correction
        self.add_parameter(
            'mw_G_mixer_phi',
            unit='deg',
            label='Mixer skewness phi Gaussian quadrature',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'mw_G_mixer_alpha',
            unit='',
            label='Mixer skewness alpha Gaussian quadrature',
            parameter_class=ManualParameter,
            initial_value=1)
        self.add_parameter(
            'mw_D_mixer_phi',
            unit='deg',
            label='Mixer skewness phi Derivative quadrature',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'mw_D_mixer_alpha',
            unit='',
            label='Mixer skewness alpha Derivative quadrature',
            parameter_class=ManualParameter,
            initial_value=1)

        # Mixer offsets correction (currently applied to hardware directly)
        self.add_parameter(
            'mw_mixer_offs_GI',
            unit='V',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'mw_mixer_offs_GQ',
            unit='V',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'mw_mixer_offs_DI',
            unit='V',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'mw_mixer_offs_DQ',
            unit='V',
            parameter_class=ManualParameter, initial_value=0)

        self._mw_fine_delay = 0
        self.add_parameter(
            'mw_fine_delay',
            label='fine delay of the AWG channel',
            unit='s',
            docstring='This parameters serves for fine tuning of '
                      'the RO, MW and flux pulses. It should be kept '
                      'positive and below 20e-9. Any larger adjustments'
                      'should be done by changing CCL dio delay'
                      'through device object.',
            set_cmd=self._set_mw_fine_delay,
            get_cmd=self._get_mw_fine_delay)

    def _add_mw_vsm_parameters(self):
        self.add_parameter(
            'mw_vsm_marker_source',
            label='VSM switch state',
            initial_value='int',
            vals=vals.Enum('ext', 'int'),
            parameter_class=ManualParameter)

        self.add_parameter(
            'mw_vsm_ch_in',
            label='VSM input channel Gaussian component',
            vals=vals.Ints(1, 4),
            initial_value=1,
            parameter_class=ManualParameter)

        self.add_parameter(
            'mw_vsm_mod_out',
            label='VSM output module for microwave pulses',
            docstring=(
                'Selects the VSM output module for MW'
                ' pulses. N.B. for spec the '
                'spec_vsm_ch_out parameter is used.'),
            vals=vals.Ints(1, 8),
            initial_value=1,
            parameter_class=ManualParameter)

        self.add_parameter(
            'mw_vsm_G_amp',
            label='VSM amp Gaussian component',
            vals=vals.Numbers(0.1, 1.0),
            initial_value=1.0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_vsm_D_amp',
            label='VSM amp Derivative component',
            vals=vals.Numbers(0.1, 1.0),
            initial_value=1.0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_vsm_G_phase',
            vals=vals.Numbers(-125, 45),
            initial_value=0, unit='deg',
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_vsm_D_phase',
            vals=vals.Numbers(-125, 45),
            initial_value=0, unit='deg',
            parameter_class=ManualParameter)

        self._mw_vsm_delay = 0
        self.add_parameter(
            'mw_vsm_delay',
            label='CCL VSM trigger delay',
            vals=vals.Ints(0, 127),
            unit='samples',
            docstring=
            ('This value needs to be calibrated to ensure that '
             'the VSM mask aligns with the microwave pulses. '
             'Calibration is done using'
             ' self.calibrate_mw_vsm_delay.'),
            set_cmd=self._set_mw_vsm_delay,
            get_cmd=self._get_mw_vsm_delay)

    def _add_spec_parameters(self):
        self.add_parameter(
            'spec_pow',
            unit='dB',
            vals=vals.Numbers(-70, 20),
            parameter_class=ManualParameter,
            initial_value=-30)

        self.add_parameter(
            'spec_vsm_ch_in',
            label='VSM input channel for spec pulses',
            docstring=(
                'VSM input channel for spec pulses'
                ' generally this should be the same as '
                ' the mw_vsm_ch_Gin parameter.'),
            vals=vals.Ints(1, 4),
            initial_value=1,
            parameter_class=ManualParameter)

        self.add_parameter(
            'spec_vsm_mod_out',
            label='VSM output module for spectroscopy pulses',
            docstring=(
                'Selects the VSM output channel for spec'
                ' pulses. N.B. for mw pulses the '
                'spec_mw_ch_out parameter is used.'),
            vals=vals.Ints(1, 8),
            initial_value=1,
            parameter_class=ManualParameter)

    def _add_flux_parameters(self):
        self.add_parameter(
            'fl_dc_ch',
            label='Flux bias channel',
            docstring=('Used to determine the DAC channel used for DC '
                       'flux biasing. Should be an int when using an IVVI rack'
                       'or a str (channel name) when using an SPI rack.'),
            vals=vals.Strings(),
            initial_value=None,
            parameter_class=ManualParameter)

        self._flux_fine_delay = 0
        self.add_parameter(
            'flux_fine_delay',
            label='fine delay of the AWG channel',
            unit='s',
            docstring='This parameters serves for fine tuning of '
                      'the RO, MW and flux pulses. It should be kept '
                      'positive and below 20e-9. Any larger adjustments'
                      'should be done by changing CCL dio delay'
                      'through device object.',
            set_cmd=self._set_flux_fine_delay,
            get_cmd=self._get_flux_fine_delay)

    def _add_ro_parameters(self):
        self.add_parameter(
            'ro_acq_weight_type',
            initial_value='SSB',
            vals=vals.Enum('SSB', 'DSB', 'optimal', 'optimal IQ'),
            docstring=(
                'Determines what type of integration weights to use: '
                '\n\t SSB: Single sideband demodulation\n\t'
                'DSB: Double sideband demodulation\n\t'
                'optimal: waveforms specified in "RO_acq_weight_func_I" '
                '\n\tand "RO_acq_weight_func_Q"'),
            parameter_class=ManualParameter)

    def _add_prep_parameters(self):
        # FIXME: these are parameters referred to in functions "prepare_*".
        #  An assessment needs to be made whether they really should be in this class, and if so,
        #  they should be moved to the appropriate _add_*_parameters section

        # FIXME: move to HAL_Transmon
        self.add_parameter(
            'freq_qubit',
            label='Qubit frequency',
            unit='Hz',
            parameter_class=ManualParameter)

        ################################
        # RO stimulus/pulse parameters #
        ################################
        self.add_parameter(
            'ro_freq',
            label='Readout frequency',
            unit='Hz',
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_freq_mod',
            label='Readout-modulation frequency',
            unit='Hz',
            initial_value=-20e6,
            parameter_class=ManualParameter)

        # NB: shared between qubits on same feedline
        self.add_parameter(
            'ro_pow_LO',
            label='RO power LO',
            unit='dBm',
            initial_value=20,
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_pulse_delay', unit='s',
            label='Readout pulse delay',
            vals=vals.Numbers(0, 1e-6),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=('The delay time for the readout pulse'))

        #############################
        # RO pulse parameters
        # FIXME: move to HAL_Transmon
        #############################
        self.add_parameter(
            'ro_pulse_type',
            initial_value='simple',
            vals=vals.Enum('gated', 'simple', 'up_down_down', 'up_down_down_final'),
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_pulse_length',
            label='Readout pulse length',
            initial_value=100e-9,
            unit='s',
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_amp',
            unit='V',
            label='Readout pulse amplitude',
            initial_value=0.1,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_amp_CW',
            unit='V',
            label='Readout pulse amplitude',
            initial_value=0.1,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_phi',
            unit='deg',
            initial_value=0,
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_pulse_down_length0',
            unit='s',
            initial_value=1e-9,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_down_amp0',
            unit='V',
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_down_phi0',
            unit='deg',
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_down_length1',
            unit='s',
            initial_value=1e-9,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_down_amp1',
            unit='V',
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_down_phi1',
            unit='deg',
            initial_value=0,
            parameter_class=ManualParameter)

        # Mixer offsets correction, RO pulse
        # NB: shared between qubits on same feedline
        self.add_parameter(
            'ro_pulse_mixer_offs_I',
            unit='V',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'ro_pulse_mixer_offs_Q',
            unit='V',
            parameter_class=ManualParameter,
            initial_value=0)
        self.add_parameter(
            'ro_pulse_mixer_alpha',
            initial_value=1,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_pulse_mixer_phi',
            initial_value=0,
            parameter_class=ManualParameter)

        #############################
        # RO acquisition parameters #
        #############################

        self.add_parameter(
            'ro_soft_avg',
            initial_value=1,
            docstring=('Number of soft averages to be performed using the MC.'),
            vals=vals.Ints(min_value=1),
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_averages',
            initial_value=1024,
            vals=vals.Numbers(min_value=0, max_value=1e6),
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_input_average_length',
            unit='s',
            label='Readout input averaging time',
            vals=vals.Numbers(min_value=0, max_value=4096 / 1.8e9),
            initial_value=4096 / 1.8e9,
            parameter_class=ManualParameter,
            docstring=('The measurement time in input averaging.'))

        self.add_parameter(
            'ro_acq_integration_length',
            initial_value=500e-9,
            vals=vals.Numbers(min_value=0, max_value=4096 / 1.8e9),
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_delay',
            unit='s',
            label='Readout acquisition delay',
            vals=vals.Numbers(min_value=0),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=(
                'The time between the instruction that trigger the'
                ' readout pulse and the instruction that triggers the '
                'acquisition. The positive number means that the '
                'acquisition is started after the pulse is sent.'))

        self.add_parameter(
            'ro_acq_weight_func_I',
            vals=vals.Arrays(),
            label='Optimized weights for I channel',
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_weight_func_Q',
            vals=vals.Arrays(),
            label='Optimized weights for Q channel',
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_weight_chI',
            initial_value=0,
            docstring=(
                'Determines the I-channel for integration. When the'
                ' ro_acq_weight_type is optimal only this channel will '
                'affect the result.'),
            vals=vals.Ints(0, 9),
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_weight_chQ',
            initial_value=1,
            docstring=('Determines the Q-channel for integration.'),
            vals=vals.Ints(0, 9),
            parameter_class=ManualParameter)

        # Mixer correction parameters
        # NB: shared between qubits on same feedline
        self.add_parameter(
            'ro_acq_mixer_phi',
            unit='degree',
            label='Readout mixer phi',
            vals=vals.Numbers(),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=('acquisition mixer phi, used for mixer deskewing in real time'))
        self.add_parameter(
            'ro_acq_mixer_alpha',
            unit='',
            label='Readout mixer alpha',
            vals=vals.Numbers(min_value=0.8),
            initial_value=1,
            parameter_class=ManualParameter,
            docstring=('acquisition mixer alpha, used for mixer deskewing in real time'))

        # FIXME!: Dirty hack because of qusurf issue #63, added 2 hardcoded
        #  delay samples in the optimized weights
        self.add_parameter(
            'ro_acq_weight_func_delay_samples_hack',
            vals=vals.Ints(),
            initial_value=0,
            label='weight function delay samples',
            parameter_class=ManualParameter)

        #############################
        # Single shot readout specific parameters
        #############################
        self.add_parameter(
            'ro_acq_digitized',
            vals=vals.Bool(),
            initial_value=False,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_threshold',
            unit='dac-value',
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_rotated_SSB_when_optimal',
            vals=vals.Bool(),
            docstring=('bypasses optimal weights, and uses rotated SSB instead'),
            initial_value=False,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_rotated_SSB_rotation_angle',
            vals=vals.Numbers(min_value=-np.pi, max_value=np.pi),
            docstring=('uses this as the rotation angle for rotated SSB'),
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_acq_integration_length_weigth_function',
            vals=vals.Numbers(min_value=0, max_value=4096 / 1.8e9),
            docstring=('sets weight function elements to 0 beyond this time'),
            initial_value=4096 / 1.8e9,
            parameter_class=ManualParameter)


    ##########################################################################
    # Private parameter helpers
    ##########################################################################

    def _set_mw_vsm_delay(self, val):
        # sort of a pseudo Manual Parameter
        self.instr_CC.get_instr().set('vsm_channel_delay{}'.format(self.cfg_qubit_nr()), val)
        self._mw_vsm_delay = val

    def _get_mw_vsm_delay(self):
        return self._mw_vsm_delay

    def _set_mw_fine_delay(self, val):
        if self.cfg_with_vsm():
            logging.warning('HAL_Transmon is using VSM. Use mw_vsm_delay to adjust delay')
        else:
            lutman = self.find_instrument(self.instr_LutMan_MW())
            AWG = lutman.find_instrument(lutman.AWG())
            if self._using_QWG():
                logging.warning('HAL_Transmon is using QWG. mw_fine_delay not supported.')
            else:
                AWG.set('sigouts_{}_delay'.format(lutman.channel_I() - 1), val)
                AWG.set('sigouts_{}_delay'.format(lutman.channel_Q() - 1), val)
        self._mw_fine_delay = val

    def _get_mw_fine_delay(self):
        return self._mw_fine_delay

    def _set_flux_fine_delay(self, val):
        if self.instr_LutMan_Flux() is not None:
            lutman = self.find_instrument(self.instr_LutMan_Flux())
            AWG = lutman.find_instrument(lutman.AWG())
            if self._using_QWG():
                logging.warning('HAL_Transmon is using QWG. Not implemented.')
            else:
                AWG.set('sigouts_{}_delay'.format(lutman.cfg_awg_channel() - 1), val)
                # val = AWG.get('sigouts_{}_delay'.format(lutman.cfg_awg_channel()-1))
        else:
            logging.warning('No Flux LutMan specified, could not set flux timing fine')
        self._flux_fine_delay = val

    def _get_flux_fine_delay(self):
        return self._flux_fine_delay

    ##########################################################################
    # Private helpers
    ##########################################################################

    def _using_QWG(self):
        """
        Checks if a QWG is used for microwave control.
        """
        AWG = self.instr_LutMan_MW.get_instr().AWG.get_instr()
        return isinstance(AWG, QuTech_AWG_Module)  # FIXME: QuTech_AWG_Module will be replaced by QWG

    ##########################################################################
    # Private prepare functions: CW
    ##########################################################################

    def _prep_cw_spec(self):
        # FIXME: this code block has no effect
        # if self.cfg_with_vsm():
        #     VSM = self.instr_VSM.get_instr()
        # if self.spec_type() == 'CW':
        #     marker_source = 'int'
        # else:
        #     marker_source = 'ext'

        if self.instr_spec_source() != None:
            self.instr_spec_source.get_instr().power(self.spec_pow())

    def _prep_cw_configure_VSM(self):
        # Configure VSM
        VSM = self.instr_VSM.get_instr()
        for mod in range(1, 9):
            VSM.set('mod{}_ch{}_marker_state'.format(mod, self.mw_vsm_ch_in()), 'off')
        VSM.set('mod{}_ch{}_marker_state'.format(self.mw_vsm_mod_out(), self.spec_vsm_ch_in()), 'on')
        VSM.set('mod{}_marker_source'.format(self.mw_vsm_mod_out()), self.mw_vsm_marker_source())

    ##########################################################################
    # Private prepare functions: ro
    ##########################################################################

    # FIXME: UHFQC specific
    # FIXME: deskewing matrix is shared between all connected qubits
    def _prep_deskewing_matrix(self):
        UHFQC = self.instr_acquisition.get_instr()

        alpha = self.ro_acq_mixer_alpha()
        phi = self.ro_acq_mixer_phi()
        predistortion_matrix = np.array(
            ((1, -alpha * np.sin(phi * 2 * np.pi / 360)),
             (0, alpha * np.cos(phi * 2 * np.pi / 360)))
        )

        UHFQC.qas_0_deskew_rows_0_cols_0(predistortion_matrix[0, 0])
        UHFQC.qas_0_deskew_rows_0_cols_1(predistortion_matrix[0, 1])
        UHFQC.qas_0_deskew_rows_1_cols_0(predistortion_matrix[1, 0])
        UHFQC.qas_0_deskew_rows_1_cols_1(predistortion_matrix[1, 1])
        return predistortion_matrix

    # FIXME: UHFQC specific
    def _prep_ro_instantiate_detectors(self):
        self.instr_MC.get_instr().soft_avg(self.ro_soft_avg())

        # determine ro_channels and result_logging_mode (needed for detectors)
        if 'optimal' in self.ro_acq_weight_type():
            if self.ro_acq_weight_type() == 'optimal':
                ro_channels = [self.ro_acq_weight_chI()]
            elif self.ro_acq_weight_type() == 'optimal IQ':
                ro_channels = [self.ro_acq_weight_chI(), self.ro_acq_weight_chQ()]

            result_logging_mode = 'lin_trans'
            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'

            # Update the RO threshold
            acq_ch = self.ro_acq_weight_chI()
            # The threshold that is set in the hardware needs to be
            # corrected for the offset as this is only applied in
            # software.
            if abs(self.ro_acq_threshold()) > 32:
                threshold = 32
                warnings.warn(f'Clipping {self.name}.ro_acq_threshold {self.ro_acq_threshold()}>32')
                # working around the limitation of threshold in UHFQC which cannot be >abs(32).
            else:
                threshold = self.ro_acq_threshold()
            self.instr_acquisition.get_instr().set('qas_0_thresholds_{}_level'.format(acq_ch), threshold)

        else:
            ro_channels = [self.ro_acq_weight_chI(),
                           self.ro_acq_weight_chQ()]
            result_logging_mode = 'raw'

        # instantiate detectors
        if 'UHFQC' in self.instr_acquisition():  # FIXME: checks name, not type
            UHFQC = self.instr_acquisition.get_instr()

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                nr_averages=self.ro_acq_averages(),
                nr_samples=int(self.ro_acq_input_average_length() * 1.8e9))

            self.int_avg_det = self.get_int_avg_det()

            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                channels=ro_channels,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                real_imag=True,
                single_int_avg=True,
                integration_length=self.ro_acq_integration_length())

            self.UHFQC_spec_det = det.UHFQC_spectroscopy_detector(
                UHFQC=UHFQC,
                ro_freq_mod=self.ro_freq_mod(),
                AWG=self.instr_CC.get_instr(),  # FIXME: parameters from here now ignored by callee
                channels=ro_channels,
                nr_averages=self.ro_acq_averages(),
                integration_length=self.ro_acq_integration_length())

            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                channels=ro_channels,
                result_logging_mode=result_logging_mode,
                integration_length=self.ro_acq_integration_length())
        else:
            raise NotImplementedError()

    def _prep_ro_sources(self):
        LO = self.instr_LO_ro.get_instr()
        RO_lutman = self.instr_LutMan_RO.get_instr()
        if RO_lutman.LO_freq() is not None:
            log.info('Warning: This qubit is using a fixed RO LO frequency.')  # FIXME: log.warning?
            LO_freq = RO_lutman.LO_freq()
            LO.frequency.set(LO_freq)
            mod_freq = self.ro_freq() - LO_freq
            self.ro_freq_mod(mod_freq)
            log.info("Setting modulation freq of {} to {}".format(self.name, mod_freq))
        else:
            LO.frequency.set(self.ro_freq() - self.ro_freq_mod())

        LO.on()
        LO.power(self.ro_pow_LO())

    # FIXME: UHFQC specific
    # FIXME: align with HAL_ShimMQ::_prep_ro_pulses
    def _prep_ro_pulse(self, upload=True, CW=False):
        """
        Sets the appropriate parameters in the RO LutMan and uploads the
        desired wave.

        Relevant parameters are:
            ro_pulse_type ("up_down_down", "square")
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

        Note that the local parameters exist for each individual qubit, but qubits on the same feedline share a single
        RO_LutMan (and hardware channel, because of the multiplexed nature of the UHFQC; this situation is different
        for MW_LutMan and Flux_LutMan, ).
        To sort of cope with that, LutMan parameters starting with "M_" exist per 'channel', but clashes on other
        parameters are not explicitly managed.
        """

        if 'UHFQC' not in self.instr_acquisition():  # FIXME: checks name, not type
            raise NotImplementedError()

        UHFQC = self.instr_acquisition.get_instr()

        if 'gated' in self.ro_pulse_type().lower():
            UHFQC.awg_sequence_acquisition()
        else:
            # propagate parameters to readout lutman
            ro_lm = self.instr_LutMan_RO.get_instr()
            ro_lm.AWG(self.instr_acquisition())

            idx = self.cfg_qubit_nr()
            # These parameters affect all resonators
            ro_lm.set('resonator_combinations', [[idx]])
            ro_lm.set('pulse_type', 'M_' + self.ro_pulse_type())
            ro_lm.set('mixer_alpha', self.ro_pulse_mixer_alpha())
            ro_lm.set('mixer_phi', self.ro_pulse_mixer_phi())

            ro_lm.set('M_modulation_R{}'.format(idx), self.ro_freq_mod())
            ro_lm.set('M_length_R{}'.format(idx), self.ro_pulse_length())
            if CW:
                ro_amp = self.ro_pulse_amp_CW()
            else:
                ro_amp = self.ro_pulse_amp()
            ro_lm.set('M_amp_R{}'.format(idx), ro_amp)
            ro_lm.set('M_delay_R{}'.format(idx), self.ro_pulse_delay())
            ro_lm.set('M_phi_R{}'.format(idx), self.ro_pulse_phi())
            ro_lm.set('M_down_length0_R{}'.format(idx), self.ro_pulse_down_length0())
            ro_lm.set('M_down_amp0_R{}'.format(idx), self.ro_pulse_down_amp0())
            ro_lm.set('M_down_phi0_R{}'.format(idx), self.ro_pulse_down_phi0())
            ro_lm.set('M_down_length1_R{}'.format(idx), self.ro_pulse_down_length1())
            ro_lm.set('M_down_amp1_R{}'.format(idx), self.ro_pulse_down_amp1())
            ro_lm.set('M_down_phi1_R{}'.format(idx), self.ro_pulse_down_phi1())

            ro_lm.acquisition_delay(self.ro_acq_delay())  # FIXME: better located in _prep_ro_integration_weights?

            if upload:
                ro_lm.load_DIO_triggered_sequence_onto_UHFQC()

            # set mixer offset (NB: affects all channels)
            # FIXME: note that ro_lm.set_mixer_offsets() is not used. _prep_mw_pulses also sets mixer offset locally,
            #  and alpha/phi through LutMan. This requires cleanup
            UHFQC.sigouts_0_offset(self.ro_pulse_mixer_offs_I())
            UHFQC.sigouts_1_offset(self.ro_pulse_mixer_offs_Q())

            if [self.cfg_qubit_nr()] not in ro_lm.resonator_combinations():
                warnings.warn('Qubit number of {} is not '.format(self.name) +
                              'present in resonator_combinations of the readout lutman.')

    # FIXME: UHFQC specific
    def _prep_ro_integration_weights(self):
        """
        Sets the ro acquisition integration weights.
        The relevant parameters here are
            ro_acq_weight_type   -> 'SSB', 'DSB' or 'Optimal'
            ro_acq_weight_chI    -> Specifies which integration weight (channel) to use
            ro_acq_weight_chQ    -> The second channel in case of SSB/DSB
            ro_acq_weight_func_I -> A custom integration weight (array)
            ro_acq_weight_func_Q ->  ""

        """
        if 'UHFQC' in self.instr_acquisition():  # FIXME: checks name, not type
            UHFQC = self.instr_acquisition.get_instr()

            if self.ro_acq_weight_type() == 'SSB':
                UHFQC.prepare_SSB_weight_and_rotation(
                    IF=self.ro_freq_mod(),
                    weight_function_I=self.ro_acq_weight_chI(),  # FIXME: 'weight_function_I' is a misnomer, it specifies a channel
                    weight_function_Q=self.ro_acq_weight_chQ()
                )
            elif self.ro_acq_weight_type() == 'DSB':
                UHFQC.prepare_DSB_weight_and_rotation(
                    IF=self.ro_freq_mod(),
                    weight_function_I=self.ro_acq_weight_chI(),
                    weight_function_Q=self.ro_acq_weight_chQ()
                )
            elif 'optimal' in self.ro_acq_weight_type():
                if (self.ro_acq_weight_func_I() is None or self.ro_acq_weight_func_Q() is None):
                    logging.warning('Optimal weights are None, not setting integration weights')
                elif self.ro_acq_rotated_SSB_when_optimal():
                    # this allows bypassing the optimal weights for poor SNR qubits
                    # working around the limitation of threshold in UHFQC which cannot be >abs(32)
                    if self.ro_acq_digitized() and abs(self.ro_acq_threshold()) > 32:
                        scaling_factor = 32 / self.ro_acq_threshold()
                    else:
                        scaling_factor = 1

                    UHFQC.prepare_SSB_weight_and_rotation(
                        IF=self.ro_freq_mod(),
                        weight_function_I=self.ro_acq_weight_chI(),
                        weight_function_Q=None,
                        rotation_angle=self.ro_acq_rotated_SSB_rotation_angle(),
                        length=self.ro_acq_integration_length_weigth_function(),
                        scaling_factor=scaling_factor
                    )
                else:
                    # When optimal weights are used, only the RO I weight channel is used
                    opt_WI = self.ro_acq_weight_func_I()
                    opt_WQ = self.ro_acq_weight_func_Q()

                    if 0:  # FIXME: remove
                        # FIXME!: Dirty hack because of qusurf issue #63, adds delay samples in the optimized weights
                        #  NB: https://github.com/DiCarloLab-Delft/QuSurf-IssueTracker/issues/63 was closed in 2018
                        del_sampl = self.ro_acq_weight_func_delay_samples_hack()
                        if del_sampl > 0:
                            zeros = np.zeros(abs(del_sampl))
                            opt_WI = np.concatenate([opt_WI[abs(del_sampl):], zeros])
                            opt_WQ = np.concatenate([opt_WQ[abs(del_sampl):], zeros])
                        elif del_sampl < 0:
                            zeros = np.zeros(abs(del_sampl))
                            opt_WI = np.concatenate([zeros, opt_WI[:-abs(del_sampl)]])
                            opt_WQ = np.concatenate([zeros, opt_WQ[:-abs(del_sampl)]])
                        else:
                            pass

                    # FIXME: direct access to UHFQC nodes, consider adding function to UHFQA (like prepare_SSB_weight_and_rotation)
                    UHFQC.set(f'qas_0_integration_weights_{self.ro_acq_weight_chI()}_real', opt_WI)
                    UHFQC.set(f'qas_0_integration_weights_{self.ro_acq_weight_chI()}_imag', opt_WQ)
                    UHFQC.set(f'qas_0_rotations_{self.ro_acq_weight_chI()}', 1.0 - 1.0j)
                    if self.ro_acq_weight_type() == 'optimal IQ':
                        print('setting the optimal Q')
                        UHFQC.set(f'qas_0_integration_weights_{self.ro_acq_weight_chQ()}_real', opt_WQ)
                        UHFQC.set(f'qas_0_integration_weights_{self.ro_acq_weight_chQ()}_imag', opt_WI)
                        UHFQC.set(f'qas_0_rotations_{self.ro_acq_weight_chQ()}', 1.0 + 1.0j)

        else:
            raise NotImplementedError('CBox, DDM or other are currently not supported')

    ##########################################################################
    # Private prepare functions: MW
    ##########################################################################

    def _prep_td_sources(self):
        # turn off spec_source
        if self.instr_spec_source() is not None:
            self.instr_spec_source.get_instr().off()

        # configure LO_mw
        self.instr_LO_mw.get_instr().on()
        self.instr_LO_mw.get_instr().pulsemod_state('Off')

        MW_LutMan = self.instr_LutMan_MW.get_instr()
        if MW_LutMan.cfg_sideband_mode() == 'static':
            # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
            self.instr_LO_mw.get_instr().frequency.set(self.freq_qubit.get() - self.mw_freq_mod.get())
        elif MW_LutMan.cfg_sideband_mode() == 'real-time':
            # For historic reasons, will maintain the change qubit frequency here in
            # _prep_td_sources, even for real-time mode, where it is only changed in the HDAWG
            # FIXME: HDAWG specific, does not support QWG
            if ((MW_LutMan.channel_I() - 1) // 2 != (MW_LutMan.channel_Q() - 1) // 2):
                raise KeyError('In real-time sideband mode, channel I/Q should share same awg group.')
            self.mw_freq_mod(self.freq_qubit.get() - self.instr_LO_mw.get_instr().frequency.get())
            MW_LutMan.AWG.get_instr().set('oscs_{}_freq'.format((MW_LutMan.channel_I() - 1) // 2),
                                          self.mw_freq_mod.get())
        else:
            raise ValueError('Unexpected value for parameter cfg_sideband_mode.')

        self.instr_LO_mw.get_instr().power.set(self.mw_pow_td_source.get())

    def _prep_mw_pulses(self):
        # FIXME: hardware handling moved here from HAL_Transmon, cleanup

        # here we handle hardware specific functionality like:
        #  - mixer offsets : directly
        #  - mixer parameters other then offsets (phi, alfa) : through the MW_Lutman

        #  FIXME: This maps badly to the actual hardware capabilities, e.g. the QWG has hardware offset control
        #   A better approach may be to pass a standard set of parameters describing pulse attributes and signal chain
        #   settings to the LutMan (maybe as a class/dict instead of QCoDeS parameters), and then have the LutMan do
        #   everything necessary.
        #   Or, to have the LutMan only handle pulse attributes, and move all signal chain handling here


        MW_LutMan = self.instr_LutMan_MW.get_instr()

        # 3. Does case-dependent things:
        #                mixers offset+skewness
        #                pi-pulse amplitude
        AWG = MW_LutMan.AWG.get_instr()
        if self.cfg_with_vsm():
            # case with VSM (both QWG and AWG8) : e.g. AWG8_VSM_MW_LutMan
            MW_LutMan.mw_amp180(self.mw_amp180())

            MW_LutMan.G_mixer_phi(self.mw_G_mixer_phi())
            MW_LutMan.G_mixer_alpha(self.mw_G_mixer_alpha())
            MW_LutMan.D_mixer_phi(self.mw_D_mixer_phi())
            MW_LutMan.D_mixer_alpha(self.mw_D_mixer_alpha())

            MW_LutMan.channel_GI(0 + self.mw_awg_ch())
            MW_LutMan.channel_GQ(1 + self.mw_awg_ch())
            MW_LutMan.channel_DI(2 + self.mw_awg_ch())
            MW_LutMan.channel_DQ(3 + self.mw_awg_ch())

            if self._using_QWG():
                # N.B. This part is QWG specific
                if hasattr(MW_LutMan, 'channel_GI'):
                    # 4-channels are used for VSM based AWG's.
                    AWG.ch1_offset(self.mw_mixer_offs_GI())
                    AWG.ch2_offset(self.mw_mixer_offs_GQ())
                    AWG.ch3_offset(self.mw_mixer_offs_DI())
                    AWG.ch4_offset(self.mw_mixer_offs_DQ())
            else:  # using_AWG8
                # N.B. This part is AWG8 specific
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() - 1), self.mw_mixer_offs_GI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 0), self.mw_mixer_offs_GQ())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 1), self.mw_mixer_offs_DI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 2), self.mw_mixer_offs_DQ())
        else:  # no VSM
            if self._using_QWG():
                # case without VSM and with QWG : QWG_MW_LutMan
                if ((self.mw_G_mixer_phi() != self.mw_D_mixer_phi())
                        or (self.mw_G_mixer_alpha() != self.mw_D_mixer_alpha())):
                    logging.warning('HAL_Transmon {}; _prep_mw_pulses: '
                                    'no VSM detected, using mixer parameters'
                                    ' from gaussian channel.'.format(self.name))
                MW_LutMan.mixer_phi(self.mw_G_mixer_phi())
                MW_LutMan.mixer_alpha(self.mw_G_mixer_alpha())
                AWG.set('ch{}_offset'.format(MW_LutMan.channel_I()), self.mw_mixer_offs_GI())
                AWG.set('ch{}_offset'.format(MW_LutMan.channel_Q()), self.mw_mixer_offs_GQ())
                # FIXME: MW_LutMan.mw_amp180 untouched
            else:
                # case without VSM (and with AWG8) : AWG8_MW_LutMan
                MW_LutMan.mw_amp180(1)  # AWG8_MW_LutMan uses 'channel_amp' to allow rabi-type experiments without wave reloading.
                MW_LutMan.mixer_phi(self.mw_G_mixer_phi())
                MW_LutMan.mixer_alpha(self.mw_G_mixer_alpha())

                # N.B. This part is AWG8 specific
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() - 1), self.mw_mixer_offs_GI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 0), self.mw_mixer_offs_GQ())

        # 4. reloads the waveforms
        if self.cfg_prepare_mw_awg():
            MW_LutMan.load_waveforms_onto_AWG_lookuptable()
        else:
            warnings.warn('"cfg_prepare_mw_awg" set to False, not preparing microwave pulses.')

        # 5. upload command table for virtual-phase gates
        MW_LutMan.upload_single_qubit_phase_corrections()  # FIXME: assumes AWG8_MW_LutMan
        # 3. Does case-dependent things:
        #                mixers offset+skewness
        #                pi-pulse amplitude
        AWG = MW_LutMan.AWG.get_instr()
        if self.cfg_with_vsm():
            # case with VSM (both QWG and AWG8) : e.g. AWG8_VSM_MW_LutMan
            MW_LutMan.mw_amp180(self.mw_amp180())

            MW_LutMan.G_mixer_phi(self.mw_G_mixer_phi())
            MW_LutMan.G_mixer_alpha(self.mw_G_mixer_alpha())
            MW_LutMan.D_mixer_phi(self.mw_D_mixer_phi())
            MW_LutMan.D_mixer_alpha(self.mw_D_mixer_alpha())

            MW_LutMan.channel_GI(0 + self.mw_awg_ch())
            MW_LutMan.channel_GQ(1 + self.mw_awg_ch())
            MW_LutMan.channel_DI(2 + self.mw_awg_ch())
            MW_LutMan.channel_DQ(3 + self.mw_awg_ch())

            if self._using_QWG():
                # N.B. This part is QWG specific
                if hasattr(MW_LutMan, 'channel_GI'):
                    # 4-channels are used for VSM based AWG's.
                    AWG.ch1_offset(self.mw_mixer_offs_GI())
                    AWG.ch2_offset(self.mw_mixer_offs_GQ())
                    AWG.ch3_offset(self.mw_mixer_offs_DI())
                    AWG.ch4_offset(self.mw_mixer_offs_DQ())
            else:  # using_AWG8
                # N.B. This part is AWG8 specific
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() - 1), self.mw_mixer_offs_GI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 0), self.mw_mixer_offs_GQ())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 1), self.mw_mixer_offs_DI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 2), self.mw_mixer_offs_DQ())
        else:  # no VSM
            if self._using_QWG():
                # case without VSM and with QWG : QWG_MW_LutMan
                if ((self.mw_G_mixer_phi() != self.mw_D_mixer_phi())
                        or (self.mw_G_mixer_alpha() != self.mw_D_mixer_alpha())):
                    logging.warning('HAL_Transmon {}; _prep_mw_pulses: '
                                    'no VSM detected, using mixer parameters'
                                    ' from gaussian channel.'.format(self.name))
                MW_LutMan.mixer_phi(self.mw_G_mixer_phi())
                MW_LutMan.mixer_alpha(self.mw_G_mixer_alpha())
                AWG.set('ch{}_offset'.format(MW_LutMan.channel_I()), self.mw_mixer_offs_GI())
                AWG.set('ch{}_offset'.format(MW_LutMan.channel_Q()), self.mw_mixer_offs_GQ())
                # FIXME: MW_LutMan.mw_amp180 untouched
            else:
                # case without VSM (and with AWG8) : AWG8_MW_LutMan
                MW_LutMan.mw_amp180(
                    1)  # AWG8_MW_LutMan uses 'channel_amp' to allow rabi-type experiments without wave reloading.
                MW_LutMan.mixer_phi(self.mw_G_mixer_phi())
                MW_LutMan.mixer_alpha(self.mw_G_mixer_alpha())

                # N.B. This part is AWG8 specific
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() - 1), self.mw_mixer_offs_GI())
                AWG.set('sigouts_{}_offset'.format(self.mw_awg_ch() + 0), self.mw_mixer_offs_GQ())

        # 4. reloads the waveforms
        if self.cfg_prepare_mw_awg():
            MW_LutMan.load_waveforms_onto_AWG_lookuptable()
        else:
            warnings.warn('"cfg_prepare_mw_awg" set to False, not preparing microwave pulses.')

        # 5. upload command table for virtual-phase gates
        MW_LutMan.upload_single_qubit_phase_corrections()  # FIXME: assumes AWG8_MW_LutMan
        pass

    def _prep_td_configure_VSM(self):
        # Configure VSM
        VSM = self.instr_VSM.get_instr()
        VSM.set('ch{}_frequency'.format(self.mw_vsm_ch_in()), self.freq_qubit())
        for mod in range(1, 9):
            VSM.set('mod{}_ch{}_marker_state'.format(mod, self.spec_vsm_ch_in()), 'off')
        VSM.set('mod{}_ch{}_marker_state'.format(self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), 'on')
        VSM.set('mod{}_marker_source'.format(self.mw_vsm_mod_out()), self.mw_vsm_marker_source())
        VSM.set('mod{}_ch{}_derivative_amp'.format(self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), self.mw_vsm_D_amp())
        VSM.set('mod{}_ch{}_derivative_phase'.format(self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), self.mw_vsm_D_phase())
        VSM.set('mod{}_ch{}_gaussian_amp'.format(self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), self.mw_vsm_G_amp())
        VSM.set('mod{}_ch{}_gaussian_phase'.format(self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), self.mw_vsm_G_phase())

        self.instr_CC.get_instr().set('vsm_channel_delay{}'.format(self.cfg_qubit_nr()), self.mw_vsm_delay())
