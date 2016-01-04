# Agilent_E8257D.py class, to perform the communication between the Wrapper
# and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from instrument import Instrument
import visa
import types
import logging
import numpy as np


class Agilent_E8257D(Instrument):
    '''
    This is the driver for the Agilent E8257D Signal Genarator

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_E8257D',
             address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False, step_attenuator=False):
        '''
        Initializes the Agilent_E8257D, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E8257D')
        Instrument.__init__(self, name, tags=['physical', 'source'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.instrument(self._address)

        min_power = -135 if step_attenuator else -20
        self.add_parameter('power',
                           flags=Instrument.FLAG_GETSET, units='dBm',
                           minval=min_power, maxval=16, type=float)
        self.add_parameter('phase',
                           flags=Instrument.FLAG_GETSET, units='DEG',
                           minval=0., maxval=360., type=float)
        self.add_parameter('frequency',
                           flags=Instrument.FLAG_GETSET, units='Hz',
                           minval=1e5, maxval=20e9, type=float)
        self.add_parameter('status',
                           flags=Instrument.FLAG_GETSET,
                           type=bytes)

        self.add_function('reset')
        self.add_function('get_all')

        if (reset):
            self.reset()
        else:
            self.get_all()

    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : resetting instrument')
        self._visainstrument.write('*RST')
        self.get_all()

    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : get all')
        self.get_power()
        self.get_phase()
        self.get_frequency()
        self.get_status()

    def do_get_power(self):
        '''
        Reads the power of the signal from the instrument

        Input:
            None

        Output:
            ampl (?) : power in ?
        '''
        logging.debug(__name__ + ' : get power')
        return float(self._visainstrument.ask('POW:AMPL?'))

    def do_set_power(self, amp):
        '''
        Set the power of the signal

        Input:
            amp (float) : power in ??

        Output:
            None
        '''
        logging.debug(__name__ + ' : set power to %f' % amp)
        self._visainstrument.write('POW:AMPL %s' % amp)

    def do_get_phase(self):
        '''
        Reads the phase of the signal from the instrument

        Input:
            None

        Output:
            phase (float) : Phase in degrees
        '''
        logging.debug(__name__ + ' : get phase')
        phase_radians = float(self._visainstrument.ask('PHASE?'))
        phase = phase_radians*360./(2*np.pi)
        if phase < 0:
            phase = phase + 180.
        return phase

    def do_set_phase(self, phase):
        '''
        Set the phase of the signal

        Input:
            phase (float) : Phase in degrees

            To make it consistent with R&S Sources.

        Output:
            None
        '''
        phase_radians = phase*2*np.pi/360.
        if phase_radians > np.pi:
            phase_radians = phase_radians - 2 * np.pi
        logging.debug(__name__ + ' : set phase to %f' % phase)
        self._visainstrument.write('PHASE %s' % phase_radians)

    def do_get_frequency(self):
        '''
        Reads the frequency of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency')
        return float(self._visainstrument.ask('FREQ:CW?'))

    def do_set_frequency(self, freq):
        '''
        Set the frequency of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None

        note: whenever frequency is set to the Agilent, the phase offset is set
        to 0. This is an internal feature of the Agilent.
        '''
        logging.debug(__name__ + ' : set frequency to %f' % freq)
        self._visainstrument.write('FREQ:CW %s' % freq)


    def do_get_status(self):
        '''
        Reads the output status from the instrument

        Input:
            None

        Output:
            status (string) : 'On' or 'Off'
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.ask('OUTP?')

        if (stat == '1'):
            return 'on'
        elif (stat == '0'):
            return 'off'
        else:
            raise ValueError('Output status not specified : %s' % stat)
        return

    def do_set_status(self, status):
        '''
        Set the output status of the instrument

        Input:
            status (string) : 'On' or 'Off'

        Output:
            None
        '''
        logging.debug(__name__ + ' : set status to %s' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self._visainstrument.write('OUTP %s' % status)

    # shortcuts
    def off(self):
        '''
        Set status to 'off'

        Input:
            None

        Output:
            None
        '''
        self.set_status('off')

    def on(self):
        '''
        Set status to 'on'

        Input:
            None

        Output:
            None
        '''
        self.set_status('on')