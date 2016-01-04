# RS_SGS100A.py class, to perform the communication between the Wrapper and the device
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
import socket
import select
from time import sleep, time

def has_newline(ans):
    if len(ans) > 0 and ans.find('\n') != -1:
        return True
    return False

class SocketVisa:
    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))
        self._socket.settimeout(20)

    def clear(self):
        rlist, wlist, xlist = select.select([self._socket], [], [], 0)
        if len(rlist) == 0:
            return
        ret = self.read()
        print('Unexpected data before ask(): %r' % (ret, ))

    def write(self, data):
        self.clear()
        if len(data) > 0 and data[-1] != '\r\n':
            data += '\n'
        # if len(data)<100:
        # print 'Writing %s' % (data,)
        self._socket.send(data)

    def read(self,timeouttime=20):
        start = time()
        try:
            ans = ''
            while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
                ans2 = self._socket.recv(8192)
                ans += ans2
                if len(ans2) == 0:
                    sleep(0.01)
            #print 'Read: %r (len=%s)' % (ans, len(ans))
            AWGlastdataread = ans
        except socket.timeout as e:
            print('Timed out')
            return ''

        if len(ans) > 0:
            ans = ans.rstrip('\r\n')
        return ans

    def ask(self, data):
        self.clear()
        self.write(data)
        return self.read()


class RS_SGS100A(Instrument):
    '''
    This is the python driver for the Rohde & Schwarz SMR40
    signal generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RS_SGS100A', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the RS_SGS100A, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical', 'source'])

        self._address = address
        if address[:5] == 'TCPIP':
            self._visainstrument = SocketVisa(self._address[8:], 5025)
        else:
            self._visainstrument = visa.instrument(address, timeout=60)
        self.add_parameter(
            'frequency', type=float, flags=Instrument.FLAG_GETSET,
            minval=1e9, maxval=20e9, units='Hz',  # format='%.12e',
            tags=['sweep'])
        self.add_parameter(
            'phase', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=360, units='DEG', format='%.01e', tags=['sweep'])
        self.add_parameter(
            'power',
            type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=-120, maxval=25, units='dBm',
            tags=['sweep'])
        self.add_parameter(
            'status', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'pulsemod_state', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'pulsemod_source', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_function('reset')
        self.add_function('get_all')

        if reset:
            self.reset()
        else:
            self.get_all()

    # Functions
    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Resetting instrument')
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
        logging.info(__name__ + ' : reading all settings from instrument')
        self.get_frequency()
        self.get_power()
        self.get_status()

    # communication with machine

    def do_get_frequency(self):
        '''
        Get frequency from device

        Input:
            None

        Output:
            frequency (float) : frequency in Hz
        '''
        logging.debug(__name__ + ' : reading frequency from instrument')
        return float(self._visainstrument.ask('SOUR:FREQ?'))

    def do_set_frequency(self, frequency):
        '''
        Set frequency of device

        Input:
            frequency (float) : frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting frequency to %s GHz' % frequency)
        self._visainstrument.write('SOUR:FREQ %s' % frequency)

    def do_get_power(self):
        '''
        Get output power from device

        Input:
            None

        Output:
            power (float) : output power in dBm
        '''
        logging.debug(__name__ + ' : reading power from instrument')
        return float(self._visainstrument.ask('SOUR:POW?'))

    def do_set_power(self, power):
        '''
        Set output power of device

        Input:
            power (float) : output power in dBm

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting power to %s dBm' % power)
        self._visainstrument.write('SOUR:POW %e' % power)

    def do_get_status(self):
        '''
        Get status from instrument

        Input:
            None

        Output:
            status (string) : 'on or 'off'
        '''
        logging.debug(__name__ + ' : reading status from instrument')
        stat = self._visainstrument.ask(':OUTP:STAT?')

        if stat == '1':
            return 'on'
        elif stat == '0':
            return 'off'
        else:
            raise ValueError('Output status not specified : %s' % stat)

    def do_set_status(self, status):
        '''
        Set status of instrument

        Input:
            status (string) : 'on or 'off'

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting status to "%s"' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self._visainstrument.write(':OUTP:STAT %s' % status)

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

    def _do_set_phase(self, poffset):
        self._phase_offset = poffset
        self._visainstrument.write('SOUR:PHAS %sDEG' % poffset)

    def _do_get_phase(self):
        self._phase_offset = self._visainstrument.ask('SOUR:PHAS?')
        return self._phase_offset

    def do_set_pulsemod_state(self, state):
        if (state.upper() == 'ON'):
            state_s = 'ON'
        elif (state.upper() == 'OFF'):
            state_s = 'OFF'
        else:
            logging.error(__name__ + ' : Unable to set pulsed mode to %s,\
                                         expected "ON" or "OFF"' % state)

        self._visainstrument.write(':PULM:SOUR EXT')
        self._visainstrument.write(':SOUR:PULM:STAT %s' % state_s)

    def do_get_pulsemod_state(self):
        return self._visainstrument.ask(':SOUR:PULM:STAT?') == '1'

    def do_set_pulsemod_source(self, source):
        self._visainstrument.write('SOUR:PULM:SOUR %s' % source)

    def do_get_pulsemod_source(self):
        return self._visainstrument.ask('SOUR:PULM:SOUR?')
