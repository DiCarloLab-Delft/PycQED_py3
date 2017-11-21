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
from qcodes import validators


class QuTechVSMModule(SCPI):
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)

        self.modules = [1, 2, 3, 4, 5, 6, 7, 8]
        self.channels = [1, 2, 3, 4]

        self.add_parameters()
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
                                   get_cmd='TEMPERATURE:'+ch_scpi+'?')
            # Digital sensor
            self.add_parameter(mod_name + '_temperature_digital',
                               docstring='Temperature (in ℃) of the separate '
                                         'digital temperature sensor on each '
                                         'module.',
                               unit='℃',
                               get_cmd='TEMPERATURE:' + mod_scpi + ':DIGITAL?')

    def add_marker_parameters(self):
        # Set all markers in one go
        self.add_parameter('marker_source',
                           docstring='Set the marker source of all channels to '
                                     'internal or external switching.',
                           set_cmd='MARKER:SOURCE {}',
                           vals=validators.Enum('int', 'ext'))
        self.add_parameter('marker_state',
                           docstring='Set the marker state of all channels on '
                                     'or off.',
                           set_cmd='MARKER:STATE {}',
                           vals=validators.OnOff())

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
                           doctring='Whether the _marker breakout board_ is '
                                    'connected to the VSM.',
                           get_cmd='MBBC?',
                           vals=validators.Enum('connected', 'disconnected'))

    def add_qubit_parameters(self):
        # Qubit attributes are set per channel row (so all modules in one go)
        for channel in self.channels:
            ch_name = '_ch{c}'.format(c=channel)
            ch_scpi = ':CHANNEL{c}'.format(c=channel)

            doc_description = 'Qubit description on display ' \
                              'for row {c}'.format(c=channel)
            self.add_parameter('qubit' + ch_name + '_description',
                               docstring=doc_description,
                               get_cmd='QUBIT'+ch_scpi+':DESCRIPTION?',
                               set_cmd='QUBIT'+ch_scpi+':DESCRIPTION {}',
                               vals=validators.Strings())

            doc_frequency = 'Qubit frequency in Hz for row {c}. ' \
                            'Range 4.0E9--8.0E9 Hz.'.format(c=channel)
            self.add_parameter('qubit' + ch_name + '_frequency',
                               docstring=doc_frequency,
                               unit='Hz',
                               get_cmd='QUBIT'+ch_scpi+':FREQUENCY?',
                               set_cmd='QUBIT'+ch_scpi+':FREQUENCY {}',
                               vals=validators.Numbers())

            colors = ', '.join(['black', 'blue', 'green', 'grey', 'orange',
                                'red', 'white', 'yellow', 'dcl_blue',
                                'dcl_green', 'dcl_red', 'dcl_violet'])
            doc_color = 'Qubit led and display color for row {c}. ' \
                        'Can either be one of the predefined colors ({lst})' \
                        'or a RGB hex string like "#rrggbb".'.format(c=channel,
                                                                     lst=colors)

            self.add_parameter('qubit' + ch_name + '_led_color',
                               docstring=doc_color,
                               get_cmd='QUBIT'+ch_scpi+':LEDCOLOR?',
                               set_cmd='QUBIT'+ch_scpi+':LEDCOLOR {}',
                               vals=validators.Strings())

            # Individual channels can be switched on or off
            for mod in self.modules:
                mod_ch_name = '_mod{m}_ch{c}'.format(m=mod, c=channel)
                mod_ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)

                doc_on_off = 'On/off state for channel {c} of ' \
                             'module {m}'.format(m=mod, c=channel)
                self.add_parameter('qubit' + mod_ch_name,
                                   docstring=doc_on_off,
                                   get_cmd='QUBIT'+mod_ch_scpi+'?',
                                   set_cmd='QUBIT'+mod_ch_scpi+' {}',
                                   vals=validators.OnOff())

    def add_calibration_parameters(self):
        # Raw calibration
        #  Two input pulses
        for pulse in ('gaussian', 'derivative'):
            #  Two DACs
            for dac in ('attenuation', 'phase'):
                # All channels and modules at once (no getter)
                doc_all_dac = 'Raw {d} DAC value (0--65535) for the {p} ' \
                              'input of all channels.'.format(p=pulse, d=dac)
                var_name = '_{p}_{d}_raw'.format(p=pulse, d=dac)
                var_scpi = ':{p}:{d}:RAW'.format(p=pulse.upper(), d=dac.upper())

                self.add_parameter('calibration' + var_name,
                                   docstring=doc_all_dac,
                                   set_cmd='CALIBRATION' + var_scpi + ' {}',
                                   vals=validators.Ints(min_value=0,
                                                        max_value=2**16-1))

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
                            'calibration' + ch_name + var_name,
                            docstring=doc_dac,
                            get_cmd=scpi_name + '?',
                            set_cmd=scpi_name + ' {}',
                            vals=validators.Ints(min_value=0, max_value=2**16-1)
                        )
