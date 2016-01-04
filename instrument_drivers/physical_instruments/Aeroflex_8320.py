# Keithley_2100.py driver for Keithley 2100 DMM
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Reinier Heeres <reinier@heeres.eu>, 2008 - 2010
#
# Update december 2009:
# Michiel Jol <jelle@michieljol.nl>
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
import select
import time
import socket
import qt
from time import sleep, time

def bool_to_str(val):
    '''
    Function to convert boolean to 'ON' or 'OFF'
    '''
    if val == True:
        return "ON"
    else:
        return "OFF"

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

    def read(self, timeouttime=20):
        start = time()
        try:
            ans = ''
            while len(ans) == 0 and (time() - start) < timeouttime: #or not has_newline(ans): #commented by Niels
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



class Aeroflex_8320(Instrument):
    '''
    This is the driver for the  Aeroflex_8320 variable attenuator

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Keithley_2100',
        address='<GBIP address>',
        reset=<bool>,

    '''

    def __init__(self, name, address, reset=False):
        '''


        Input:
            name (string)           : name of the instrument
            address (string)        : GPIB address
            reset (bool)            : resets to default values

        Output:
            None
        '''
        # Initialize wrapper functions
        logging.info('Initializing instrument Keithley_2100')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        if address[:5] == 'TCPIP':
            self._visainstrument = SocketVisa(self._address[8:], 10001)
        else:
            self._visainstrument = visa.instrument(address, timeout=60)
        self.add_parameter('attenuation', type=int,
                           flags=Instrument.FLAG_GETSET,
                           units='dB',
                           minval=0, maxval=60)


# --------------------------------------
#           functions
# --------------------------------------

    def send_visa_command(self, command):
        self._visainstrument.write(command)

    def ask_visa_command(self, query):
        return self._visainstrument.ask(query)

    def do_set_attenuation(self, attenuation):
        if attenuation % 2 != 0:
            attenuation = np.ceil(attenuation/2)*2
            logging.warning('Attenuation must be multiple of 2, rounded to % d'
                            % attenuation)
        self._visainstrument.write('ATTN ALL %d' % attenuation)

    def do_get_attenuation(self):
        return self._visainstrument.ask('ATTN? 1')

    def reset(self):
        '''
        Resets instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.debug('Resetting instrument')
        self._visainstrument.write('*RST')

    def reboot(self):
        '''
        Resets instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.debug('Rebooting instrument')
        self._visainstrument.write('REBOOT')
