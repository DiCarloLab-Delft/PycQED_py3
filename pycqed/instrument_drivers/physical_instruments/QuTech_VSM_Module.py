#!/usr/bin/python3

"""
File:       QuTech_VSM_Module.py
Author:     Jeroen Bergmans, TNO/QuTech
Purpose:    Instrument driver for Qutech Vector Switch Matrix
Usage:
Notes:      It is possible to view the VSM SCPI log using ssh. To do this
            connect using ssh e.g., "ssh pi@192.168.0.10"
            Log file can be viewed with "tail -f /var/log/vsm_scpi.log"
Bugs:       Probably
"""

import logging
import socket
import time

from qcodes import IPInstrument
from qcodes import validators as vals


class QuTechVSMModule(IPInstrument):
    def __init__(self, name, address, port=5025, **kwargs):
        super().__init__(name, address, port,
                         write_confirmation=False,
                         **kwargs)
        self.modules = [1, 2, 3, 4, 5, 6, 7, 8]
        self.channels = [1, 2, 3, 4]

        self.add_parameters()

        # example of how the commands could look
        self.add_function('reset', call_cmd='*RST')

        # Set socket option NODELAY, this should be handled by the base class.
        if self._socket:
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.connect_message()

    def _recv(self):
        """ Overwrites base IP recv command to ensuring read till EOM
            Again, this should be handled properly in a base class.
        """
        return self._socket.makefile().readline().rstrip()

    def connect_message(self, idn_param='IDN', begin_time=None):
        """ Log a standard message on initial connection to an instrument.

            Overwritten from Instrument base class because it contained a naked
            print() statement. When using logging, this is undesirable because
            one cannot easily redirect or disable print().

            Args:
                idn_param (str): name of parameter that returns ID dict.
                    Default 'IDN'.
                begin_time (number): time.time() when init started.
                    Default is self._t0, set at start of Instrument.__init__.
        """
        # start with an empty dict, just in case an instrument doesn't
        # heed our request to return all 4 fields.
        idn = {'vendor': None, 'model': None,
               'serial': None, 'firmware': None}
        idn.update(self.get(idn_param))
        t = time.time() - (begin_time or self._t0)

        logging.info('Connected to: {vendor} {model} '
                     '(serial:{serial}, firmware:{firmware}) '
                     'in {t:.2f}s'.format(t=t, **idn))

    ###
    # Generic SCPI commands
    ###

    def get_identity(self):
        return self.ask('*IDN?')

    def get_operation_complete(self):
        return self.ask('*OPC?')

    def get_error(self):
        """ Returns:    "No error" or <error message>
        """
        return self.ask('system:error?')

    def get_system_error_count(self):
        return self.ask_int('system:error:count?')

    # Parameters

    def add_power_supply_parameters(self):
        for power_supply in ['front', 'rear', 'phantom']:
            name = 'powersupply_{p}'.format(p=power_supply)
            getter = 'POWERSUPPLY:{p}?'.format(p=power_supply.upper())
            setter = 'POWERSUPPLY:{p}'.format(p=power_supply.upper()) + ' {}'
            # Disable setting the front power supply
            if power_supply == 'front':
                setter = None
            self.add_parameter(name, get_cmd=getter, set_cmd=setter,
                               vals=vals.OnOff())

    def add_temperature_parameters(self):
        # Command class temperature
        self.add_parameter('setpoint',
                           get_cmd='TEMPERATURE:SETPOINT?',
                           set_cmd='TEMPERATURE:SETPOINT {}',
                           vals=vals.Numbers(min_value=10.0, max_value=55.0))
        self.add_parameter('temperature_avg',
                           get_cmd='TEMPERATURE?',
                           vals=vals.Numbers())

        for mod in self.modules:
            mod_name = 'mod{m}'.format(m=mod)
            mod_scpi = 'MODULE{m}'.format(m=mod)
            self.add_parameter(mod_name + '_temperature_avg',
                               get_cmd='TEMPERATURE:'+mod_scpi+'?',
                               vals=vals.Numbers())
            self.add_parameter(mod_name + '_temperature_digital',
                               get_cmd='TEMPERATURE:' + mod_scpi + ':DIGITAL?',
                               vals=vals.Numbers())
        for channel in self.channels:
            for mod in self.modules:
                mod_ch_name = 'mod{m}_ch{c}'.format(m=mod, c=channel)
                mod_ch_scpi = 'MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)
                self.add_parameter(mod_ch_name + '_temperature',
                                   get_cmd='TEMPERATURE:'+mod_ch_scpi+'?',
                                   vals=vals.Numbers())

    def add_marker_parameters(self):
        # Command class marker
        self.add_parameter('marker_source',
                           set_cmd='MARKER:SOURCE {}',
                           vals=vals.Enum('int', 'ext'))
        self.add_parameter('marker_state',
                           set_cmd='MARKER:STATE {}',
                           vals=vals.OnOff())

        for mod in self.modules:
            mod_name = 'mod{m}'.format(m=mod)
            mod_scpi = 'MODULE{m}'.format(m=mod)
            self.add_parameter(mod_name + '_marker_source',
                               get_cmd='MARKER:' + mod_scpi + ':SOURCE?',
                               set_cmd='MARKER:' + mod_scpi + ':SOURCE {}',
                               vals=vals.Enum('INT', 'EXT'))
            self.add_parameter(mod_name + '_marker_state',
                               set_cmd='MARKER:'+mod_scpi+':STATE {}',
                               vals=vals.OnOff())

        for channel in self.channels:
            for mod in self.modules:
                mod_ch_name = 'mod{m}_ch{c}'.format(m=mod, c=channel)
                mod_ch_scpi = 'MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)
                self.add_parameter(mod_ch_name + '_marker_state',
                                   get_cmd='MARKER:'+mod_ch_scpi+':STATE?',
                                   set_cmd='MARKER:'+mod_ch_scpi+':STATE {}',
                                   vals=vals.OnOff())
                self.add_parameter(mod_ch_name + '_marker_source',
                                   get_cmd='MARKER:'+mod_ch_scpi+':SOURCE?',
                                   vals=vals.Enum('int', 'ext'))

    def add_qubit_parameters(self):
        # Command class qubit
        for channel in self.channels:
            ch_name = '_ch{c}'.format(c=channel)
            ch_scpi = ':CHANNEL{c}'.format(c=channel)
            self.add_parameter('qubit' + ch_name + '_description',
                               get_cmd='QUBIT'+ch_scpi+':DESCRIPTION?',
                               set_cmd='QUBIT'+ch_scpi+':DESCRIPTION {}',
                               vals=vals.Strings())
            self.add_parameter('qubit' + ch_name + '_frequency',
                               get_cmd='QUBIT'+ch_scpi+':FREQUENCY?',
                               set_cmd='QUBIT'+ch_scpi+':FREQUENCY {}',
                               vals=vals.Numbers())
            self.add_parameter('qubit' + ch_name + '_led_color',
                               get_cmd='QUBIT'+ch_scpi+':LEDCOLOR?',
                               set_cmd='QUBIT'+ch_scpi+':LEDCOLOR {}',
                               vals=vals.Strings())
            for mod in self.modules:
                mod_ch_name = '_mod{m}_ch{c}'.format(m=mod, c=channel)
                mod_ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod, c=channel)
                self.add_parameter('qubit' + mod_ch_name,
                                   get_cmd='QUBIT'+mod_ch_scpi+'?',
                                   set_cmd='QUBIT'+mod_ch_scpi+' {}',
                                   vals=vals.OnOff())

    def add_parameters(self):
        # VSM properties

        self.add_power_supply_parameters()
        self.add_parameter('mbbc_state', get_cmd='MBBC?',
                           vals=vals.Enum('connected', 'disconnected'))
        self.add_temperature_parameters()
        self.add_marker_parameters()
        self.add_qubit_parameters()

        # Command class calibration
        for pulse in ('GAUSSIAN', 'DERIVATIVE'):
            for ap in ('ATTENUATION', 'PHASE'):
                # All channels and modules (no getter)
                var_name = '_{p}_{a}_raw'.format(p=pulse.lower(), a=ap.lower())
                var_scpi = ':{p}:{a}:RAW'.format(p=pulse, a=ap)
                self.add_parameter('calibration' + var_name,
                                   set_cmd='CALIBRATION' + var_scpi + ' {}',
                                   vals=vals.Ints())
                # Per channel, all modules (no getter)
                for channel in self.channels:
                    ch_name = '_ch{c}'.format(c=channel)
                    ch_scpi = ':CHANNEL{c}'.format(c=channel)
                    self.add_parameter('calibration' + ch_name + var_name,
                                       set_cmd='CALIBRATION' + ch_scpi +
                                               var_scpi + ' {}',
                                       vals=vals.Ints())
                # Per module, all channels (no getter)
                for mod in self.modules:
                    mod_name = '_mod{m}'.format(m=mod)
                    mod_scpi = ':MODULE{m}'.format(m=mod)
                    self.add_parameter('calibration' + mod_name + var_name,
                                       set_cmd='CALIBRATION' + mod_scpi +
                                               var_scpi + ' {}',
                                       vals=vals.Ints())
                # Individual outputs: per (module, channel) pair
                for channel in self.channels:
                    for mod in self.modules:
                        mod_ch_name = '_mod{m}_ch{c}'.format(m=mod, c=channel)
                        mod_ch_scpi = ':MODULE{m}:CHANNEL{c}'.format(m=mod,
                                                                     c=channel)
                        self.add_parameter('calibration' + mod_ch_name +
                                           var_name,
                                           get_cmd='CALIBRATION' +
                                                   mod_ch_scpi +
                                                   var_scpi + '?',
                                           set_cmd='CALIBRATION' +
                                                   mod_ch_scpi +
                                                   var_scpi + ' {}',
                                           vals=vals.Ints())


if __name__ == '__main__':
    from random import randint

    FORMAT = '[%(asctime)s.%(msecs)03d ' \
             '%(levelname)-8s:%(filename)s:%(lineno)s %(funcName)s] ' \
             '%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%H:%M:%S')

    logging.info('Start VSM driver test')

    vsm = QuTechVSMModule('VSM 1', '192.168.0.10', 5025)

#    vsm.reset()

    getattr(vsm, 'qubit_mod{m}_ch{c}'.format(m=4, c=2)).set('off')
    vsm.qubit_mod4_ch2('off')

    for attr in vsm.__dir__():
        print(attr)

    test_value = randint(0, 2**16)
    vsm.calibration_gaussian_attenuation_raw.set(test_value)
    set_value = int(vsm.calibration_mod8_ch1_gaussian_attenuation_raw.get())

    if test_value != set_value:
        logging.warning('vsm.calibration_gaussian_attenuation_raw.set({t}) '
                        'resulted in value {s}'
                        .format(t=test_value, s=set_value))
    else:
        logging.info('Success.')
