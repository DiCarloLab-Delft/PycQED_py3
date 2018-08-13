#!/usr/bin/python3

"""
File:       QuTech_VSM_Module.py
Author:     Jeroen Bergmans, TNO/QuTech
Purpose:    Instrument driver for Qutech Vector Switch Matrix
Usage:
Notes:      # General
            The VSM consists of 32 qubit tuners:
            8 modules (numbered from left to right)
            4 channels per module (numbered from bottom to top)

            # Temperature
            Temperature of the tuners is regulated, and can be monitored via
            the _temperature_ parameters.

            # Markers
            Markers control switching the channel state on or off either by the
            external maker connector (source=external) or by software commands
            (source=internal). Marker state has no effect when source is
            external.

            # Qubits
            Qubit parameters characterise the channels. For a single channel
            row (so all modules together) one can set the frequency, led color,
            and displayed description on the VSM display.
            The frequency also determines the attenuation and phase calibration
            of the channels, see below.

            Each qubit output (a module, channel pair) can also be switched on
            or off. In the off state, the led is off, marker state is off,
            and attenuation is maximal.

            # Calibration
            Each qubit output (a module, channel pair) has two inputs:
            the _gaussian_ pulse and the _derivative_ pulse. For each pulse the
            attenuation and phase can be controlled via a 16bit DAC.
            Setting raw attenuation and phase DAC values does not
            control the channel attenuation and phase linearly, so a calibration
            table has to be used to find DAC values for given attenuation and
            phase.

            Currently the calibration table is not implemented yet on the
            VSM firmware, so only RAW calibration commands are available.

Bugs:       Probably.
"""

from .SCPI import SCPI
from qcodes.instrument.base import Instrument
from qcodes import validators
from qcodes.instrument.parameter import ManualParameter
from datetime import datetime


class QuTechVSMModule(SCPI):
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)

        self.modules = [1, 2, 3, 4, 5, 6, 7, 8]
        self.channels = [1, 2, 3, 4]

        self.add_parameters()
        self._sync_time_and_add_parameter()
        self.connect_message()


    def add_parameters(self):
        self.add_temperature_parameters()
        self.add_marker_parameters()
        self.add_qubit_parameters()
        self.add_calibration_parameters()

    def add_temperature_parameters(self):

        self.add_parameter('temperature_avg',
                           docstring='Temperature (in ℃) averaged over '
                                     'all VSM qubit tuner sensors.',
                           unit='℃',
                           get_cmd='TEMPERATURE?')

        for mod in self.modules:
            mod_name = 'mod{m}'.format(m=mod)
            mod_scpi = 'MODULE{m}'.format(m=mod)
            # Analog sensors
            for channel in self.channels:
                doc = ('Temperature (in ℃) of qubit tuner '
                       'on module {m}, channel {c}.'.format(m=mod, c=channel))
                ch_name = '{m}_ch{c}'.format(m=mod_name, c=channel)
                ch_scpi = '{m}:CHANNEL{c}'.format(m=mod_scpi, c=channel)
                self.add_parameter(ch_name + '_temperature',
                                   docstring=doc,
                                   unit='℃',
                                   get_cmd='TEMPERATURE:'+ch_scpi+'?',
                                   get_parser=float)
            # Digital sensor
            self.add_parameter(mod_name + '_temperature_digital',
                               docstring='Temperature (in ℃) of the separate '
                                         'digital temperature sensor on each '
                                         'module.',
                               unit='℃',
                               get_cmd='TEMPERATURE:' + mod_scpi + ':DIGITAL?',
                               get_parser=float)

    def add_marker_parameters(self):
        # Each (module, channel) separately
        for mod in self.modules:
            mod_name = 'mod{m}'.format(m=mod)
            mod_scpi = 'MODULE{m}'.format(m=mod)

            doc_source = 'Marker source of module {m}.'.format(m=mod)
            self.add_parameter(mod_name + '_marker_source',
                               docstring=doc_source,
                               get_cmd='MARKER:'+mod_scpi+':SOURCE?',
                               set_cmd='MARKER:'+mod_scpi+':SOURCE {}',
                               vals=validators.Enum('int', 'ext'))

            for channel in self.channels:
                mod_ch_name = 'mod{m}_ch{c}'.format(m=mod, c=channel)
                mod_ch_scpi = 'MODULE{m}:CHANNEL{c}'.format(m=mod,
                                                            c=channel)
                doc_state = 'Marker state of module {m}, ' \
                            'channel {c}.'.format(m=mod, c=channel)
                self.add_parameter(mod_ch_name + '_marker_state',
                                   docstring=doc_state,
                                   get_cmd='MARKER:'+mod_ch_scpi+':STATE?',
                                   set_cmd='MARKER:'+mod_ch_scpi+':STATE {}',
                                   vals=validators.OnOff())

        # Marker breakout board
        self.add_parameter('mbbc_state',
                           docstring='Whether the _marker breakout board_ is '
                                    'connected to the VSM.',
                           get_cmd='MBBC?',
                           vals=validators.Enum('connected', 'disconnected'))

    def add_qubit_parameters(self):
        # Qubit attributes are set per channel row (so all modules in one go)
        for channel in self.channels:
            ch_name = 'ch{c}'.format(c=channel)
            ch_scpi = ':CHANNEL{c}'.format(c=channel)

            doc_description = 'Qubit description on display ' \
                              'for row {c}'.format(c=channel)
            self.add_parameter(ch_name + '_description',
                               docstring=doc_description,
                               get_cmd='QUBIT'+ch_scpi+':DESCRIPTION?',
                               set_cmd='QUBIT'+ch_scpi+':DESCRIPTION {}',
                               vals=validators.Strings())

            doc_frequency = 'Qubit frequency in Hz for row {c}. ' \
                            'Range 4.0E9--8.0E9 Hz.'.format(c=channel)
            self.add_parameter(ch_name + '_frequency',
                               docstring=doc_frequency,
                               unit='Hz',
                               get_cmd='QUBIT'+ch_scpi+':FREQUENCY?',
                               set_cmd='QUBIT'+ch_scpi+':FREQUENCY {}',
                               vals=validators.Numbers(),
                               get_parser=float)

            colors = ', '.join(['black', 'blue', 'green', 'grey', 'orange',
                                'red', 'white', 'yellow', 'dcl_blue',
                                'dcl_green', 'dcl_red', 'dcl_violet'])
            doc_color = 'Qubit led and display color for row {c}. ' \
                        'Can either be one of the predefined colors ({lst})' \
                        'or a RGB hex string like "#rrggbb".'.format(c=channel,
                                                                     lst=colors)

            self.add_parameter(ch_name + '_led_color',
                               docstring=doc_color,
                               get_cmd='QUBIT'+ch_scpi+':LEDCOLOR?',
                               set_cmd='QUBIT'+ch_scpi+':LEDCOLOR {}',
                               vals=validators.Strings())

            # # Individual channels can be switched on or off
            # for mod in self.modules:
            #     mod_ch_name = 'mod{m}_ch{c}_switch_unknown'.format(m=mod, c=channel)
            #     mod_ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)

            #     doc_on_off = 'On/off state for channel {c} of ' \
            #                  'module {m}'.format(m=mod, c=channel)
            #     self.add_parameter(mod_ch_name,
            #                        docstring=doc_on_off,
            #                        get_cmd='QUBIT'+mod_ch_scpi+'?',
            #                        set_cmd='QUBIT'+mod_ch_scpi+' {}',
            #                        vals=validators.OnOff())

    def add_calibration_parameters(self):
        # First add a function so a user can poll if calibration data has
        # been set correctly in the VSM in order to correctly use the
        # orthogonalized parameters
        self.add_parameter(
                            'getCalibrationDataAvailable',
                            docstring='Use this to check if the calibration data has been '\
                                      'set correctly in the VSM. Outputs an integer 0 (False), 1 (True)',
                            get_cmd='CALIBRATIONDATAPATH' + '?',
                            get_parser=int,
                            vals=validators.Ints(0,1)
                        )

        # Raw attenuationa and phase
        #  Two input pulses
        for pulse in ('gaussian', 'derivative'):
            #  Two DACs
            for dac in ('att', 'phase'):
                # All channels and modules at once (no getter)
                var_name = '_{p}_{d}_raw'.format(p=pulse, d=dac)
                var_scpi = ':{p}:{d}:RAW'.format(p=pulse.upper(), d=dac.upper())
                # Individual outputs: per (module, channel) pair
                for channel in self.channels:
                    for mod in self.modules:
                        doc_dac = 'Raw {d} DAC value (0--65535) for the {p} ' \
                                  'input of channel {c} ' \
                                  'of module {m}.'.format(p=pulse, d=dac,
                                                          c=channel, m=mod)
                        ch_name = '_mod{m}_ch{c}'.format(m=mod, c=channel)
                        ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod,
                                                                 c=channel)
                        scpi_name = 'CALIBRATION' + ch_scpi + var_scpi
                        self.add_parameter(
                            ch_name + var_name,
                            label=ch_name + var_name,
                            docstring=doc_dac,
                            get_cmd=scpi_name + '?',
                            set_cmd=scpi_name + ' {}',
                            get_parser=int,
                            set_parser=int,
                            vals=validators.Numbers(min_value=0, max_value=2**16-1)
                        )

      # orthogonalized attenuation and phase
        #  Two input pulses
        for pulse in ('gaussian', 'derivative'):
            for channel in self.channels:
                for mod in self.modules:
                    ch_name = 'mod{m}_ch{c}'.format(m=mod, c=channel)
                    ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)

                    doc_var = 'Attenuation value (in dB) for the {p} ' \
                              'input of channel {c} ' \
                              'of module {m}.\nN.B. Safe range: '\
                              '0.2<=v<=2.0'.format(p=pulse, c=channel, m=mod)
                    var_name = '_'+ch_name + '_{p}_att_db'.format(p=pulse)
                    var_scpi = ch_scpi + ':{p}:ATTENUATION:DB'.format(p=pulse.upper())
                    scpi_name = 'CALIBRATION' + var_scpi
                    self.add_parameter(var_name,
                                       docstring=doc_var,
                                       get_cmd=scpi_name + '?',
                                       set_cmd=scpi_name + ' {}',
                                       unit='dB',
                                       get_parser=float,
                                       vals=validators.Numbers())
                    doc_var = 'Amplitude value (linear) for the {p} ' \
                              'input of channel {c} ' \
                              'of module {m}.'.format(p=pulse, c=channel, m=mod)
                    var_name = ch_name + '_{p}_amp'.format(p=pulse)

                    var_scpi = ch_scpi + ':{p}:ATTENUATION:LIN'.format(p=pulse.upper())
                    scpi_name = 'CALIBRATION' + var_scpi
                    self.add_parameter(var_name,
                                       docstring=doc_var,
                                       get_cmd=scpi_name + '?',
                                       set_cmd=scpi_name + ' {}',
                                       get_parser=float,
                                       vals=validators.Numbers(min_value=0.2,
                                                               max_value=3.0))

                    doc_var = 'Phase value (in rad) for the {p} ' \
                              'input of channel {c} ' \
                              'of module {m}.'.format(p=pulse, c=channel, m=mod)
                    var_name = '_' + ch_name + '_{p}_phs_rad'.format(p=pulse)
                    var_scpi = ch_scpi + ':{p}:PHASE:RAD'.format(p=pulse.upper())
                    scpi_name = 'CALIBRATION' + var_scpi
                    self.add_parameter(var_name,
                                       docstring=doc_var,
                                       get_cmd=scpi_name + '?',
                                       set_cmd=scpi_name + ' {}',
                                       unit='rad',
                                       get_parser=float,
                                       vals=validators.Numbers())
                    doc_var = 'Phase value (in deg) for the {p} ' \
                              'input of channel {c} ' \
                              'of module {m}.'.format(p=pulse, c=channel, m=mod)
                    var_name = ch_name + '_{p}_phase'.format(p=pulse)
                    var_scpi = ch_scpi + ':{p}:PHASE:DEG'.format(p=pulse.upper())
                    scpi_name = 'CALIBRATION' + var_scpi
                    self.add_parameter(var_name,
                                       docstring=doc_var,
                                       get_cmd=scpi_name + '?',
                                       set_cmd=scpi_name + ' {}',
                                       unit='deg',
                                       get_parser=float,
                                       vals=validators.Numbers(-125,45))

    def _sync_time_and_add_parameter(self):
        doc_description = 'Parameter to sync the time from user computer to VSM'
        self.add_parameter('sync_time',
                           docstring=doc_description,
                           set_cmd='SYSTEM'+':TIME {}',
                           vals=validators.Strings())
        current_time_str = datetime.now().strftime('%YT%mT%dT%HT%MT%S')
        self.sync_time(current_time_str)


class Dummy_QuTechVSMModule(QuTechVSMModule):

    def __init__(self, name, nr_input_channels=4, nr_output_channels=2,
                 **kw):
        Instrument.__init__(self, name=name, **kw)


        self._dummy_instr = True

        self.modules = [1, 2, 3, 4, 5, 6, 7, 8]
        self.channels = [1, 2, 3, 4]
        self.add_parameters()
        self._address = 'Dummy'
        self._terminator = '\n'
        current_time_str = datetime.now().strftime('%YT%mT%dT%HT%MT%S')
        self.sync_time(current_time_str)

        self.IDN({'driver': str(self.__class__), 'model': self.name,
                  'serial': 'Dummy', 'vendor': '', 'firmware': ''})
        self.connect_message()

    def add_parameter(self, name, parameter_class=ManualParameter,
                      **kwargs):
        kwargs.pop('get_cmd', 0)
        kwargs.pop('set_cmd', 0)
        kwargs.pop('get_parser', 0)
        kwargs.pop('set_parser', 0)

        super().add_parameter(name, parameter_class=parameter_class,
                              **kwargs)
