# IVVI.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Reinier Heeres <reinier@heeres.eu>, 2008
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
import qt
from instrument import Instrument
import types
import pyvisa.vpp43 as vpp43
from time import sleep
import logging
import numpy as np
from lib import visafunc

class IVVI(Instrument):
    '''
    This is the python driver for the IVVI-rack

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'IVVI', address='<ASRL address>',
        reset=<bool>, numdacs=<multiple of 4>, polarity=<list of polarity strings>)

    TODO:
    1) fix changing all polarities instead of only the first 8
    2) explain everything /rewrite init
    '''

    def __init__(self, name, address, reset=False, numdacs=8,
        polarity=['BIP', 'BIP', 'BIP', 'BIP']):
        '''
        Initialzes the IVVI, and communicates with the wrapper

        Input:
            name (string)        : name of the instrument
            address (string)     : ASRL address
            reset (bool)         : resets to default values, default=false
            numdacs (int)        : number of dacs, multiple of 4, default=8
            polarity (string[4]) : list of polarities of each set of 4 dacs
                                   choose from 'BIP', 'POS', 'NEG',
                                   default=['BIP', 'BIP', 'BIP', 'BIP']
        Output:
            None
        '''
        logging.info('Initializing instrument IVVI')
        Instrument.__init__(self, name, tags=['physical'])

        # Set parameters
        self._address = address
        if numdacs % 4 == 0 and numdacs > 0:
            self._numdacs = int(numdacs)
        else:
            logging.error('Number of dacs needs to be multiple of 4')
        self.pol_num = list(range(self._numdacs))


        # Add functions

        # Add parameters
        self.add_parameter('pol_dacrack',
            type=bytes,
            channels=(1, self._numdacs/4),
            flags=Instrument.FLAG_SET)
        self.add_parameter('dac',
            type=float,
            flags=Instrument.FLAG_GETSET,
            channels=(1, self._numdacs),
            maxstep=10, stepdelay=50,
            units='mV', format='%.02f',
            tags=['sweep'])

        self.add_function('reset')
        self.add_function('get_all')
        self.add_function('set_dacs_zero')
        self.add_function('get_numdacs')
        self._open_serial_connection()

        # get_all calls are performed below (in reset or get_all)
        for j in range(numdacs / 4):
            self.set('pol_dacrack%d' % (j+1), polarity[j], getall=False)

        if reset:
            self.reset()
        else:
            self.get_all()

    def __del__(self):
        '''
        Closes up the IVVI driver

        Input:
            None

        Output:
            None
        '''
        logging.info('Deleting IVVI instrument')
        self._close_serial_connection()

    # open serial connection
    def _open_serial_connection(self):
        '''
        Opens the ASRL connection using vpp43
        baud=115200, databits=8, stop=one, parity=odd, no end char for reads

        Input:
            None

        Output:
            None
        '''
        logging.debug('Opening serial connection')
        self._session = vpp43.open_default_resource_manager()
        self._vi = vpp43.open(self._session, self._address)

        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_BAUD, 115200)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_DATA_BITS, 8)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_STOP_BITS,
            vpp43.VI_ASRL_STOP_ONE)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_PARITY,
            vpp43.VI_ASRL_PAR_ODD)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_END_IN,
            vpp43.VI_ASRL_END_NONE)

    # close serial connection
    def _close_serial_connection(self):
        '''
        Closes the serial connection

        Input:
            None

        Output:
            None
        '''
        logging.debug('Closing serial connection')
        vpp43.close(self._vi)

    def reset(self):
        '''
        Resets all dacs to 0 volts

        Input:
            None

        Output:
            None
        '''
        logging.info('Resetting instrument')
        self.set_dacs_zero()
        self.get_all()

    def get_all(self):
        '''
        Gets all dacvalues from the device, all polarities from memory
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info('Get all')
        for i in range(self._numdacs):
            self.get('dac%d' % (i+1))

    def set_dacs_zero(self):
        for i in range(self._numdacs):
            self.set_dac(i+1, 0)

    # Conversion of data
    def _mvoltage_to_bytes(self, mvoltage):
        '''
        Converts a mvoltage on a 0mV-4000mV scale to a 16-bit integer equivalent
        output is a list of two bytes

        Input:
            mvoltage (float) : a mvoltage in the 0mV-4000mV range

        Output:
            (dataH, dataL) (int, int) : The high and low value byte equivalent
        '''
        bytevalue = int(round(mvoltage/4000.0*65535))
        dataH = int(bytevalue/256)
        dataL = bytevalue - dataH*256
        return (dataH, dataL)

    def _numbers_to_mvoltages(self, numbers):
        '''
        Converts a list of bytes to a list containing
        the corresponding mvoltages
        '''
        values = list(range(self._numdacs))
        for i in range(self._numdacs):
            values[i] = ((numbers[2 + 2*i]*256 + numbers[3 + 2*i])/65535.0*4000.0) + self.pol_num[i]
        return values

    # Communication with device
    def get_dac(self, channel):
        '''
        Returns the value of the specified dac

        Input:
            channel (int) : 1 based index of the dac

        Output:
            voltage (float) : dacvalue in mV
        Public version of function
        '''
        exec("dac = self.get_dac%s()" % channel)
        return dac

    def set_dac(self, channel, mvoltage):
        '''
        Sets the specified dac to the specified voltage

        Input:
            mvoltage (float) : output voltage in mV
            channel (int)    : 1 based index of the dac

        Output:
            reply (string) : errormessage

        Public version of function
        '''

        exec("self.set_dac%s('%s')" % ( (channel), mvoltage))
        exec("self.get_dac%s()" % channel)


    def do_get_dac(self, channel):
        '''
        Returns the value of the specified dac

        Input:
            channel (int) : 1 based index of the dac

        Output:
            voltage (float) : dacvalue in mV
        Private version of function
        '''
        logging.debug('Reading dac%s', channel)

        mvoltages = self._get_dacs()
        return mvoltages[channel - 1]

    def do_set_dac(self,  mvoltage, channel):
        '''
        Sets the specified dac to the specified voltage

        Input:
            mvoltage (float) : output voltage in mV
            channel (int)    : 1 based index of the dac

        Output:
            reply (string) : errormessage
        Private version of function
        '''
        logging.debug('Setting dac%s to %.02f mV', channel, mvoltage)


        #mvoltage_initial=self.do_get_dac(channel)

        #setsteps=self.byte_limited_arange(mvoltage_initial,mvoltage,step=5*np.sign(mvoltage-mvoltage_initial),pol='BIP')

        #if setsteps==[]:
        #    setsteps=[0]


        #for setstep in setsteps:
        #    qt.msleep(.005)
        #    (DataH, DataL) = self._mvoltage_to_bytes(setstep - self.pol_num[channel-1])
        #    message = "%c%c%c%c%c%c%c" % (7, 0, 2, 1, channel, DataH, DataL)
        #    reply = self._send_and_read(message)

        setstep = self.do_get_dac(channel)
        (DataH, DataL) = self._mvoltage_to_bytes(mvoltage - self.pol_num[channel-1])
        message = "%c%c%c%c%c%c%c" % (7, 0, 2, 1, channel, DataH, DataL)
        reply = self._send_and_read(message)
        return reply

    def _get_dacs(self):
        '''
        Reads from device and returns all dacvoltages in a list

        Input:
            None

        Output:
            voltages (float[]) : list containing all dacvoltages (in mV)
        '''
        logging.debug('Getting dac voltages from instrument')
        message = "%c%c%c%c" % (4, 0, self._numdacs*2+2, 2)
        reply = self._send_and_read(message)
        mvoltages = self._numbers_to_mvoltages(reply)
        return mvoltages

    def _send_and_read(self, message):
        '''
        Send <message> to the device and read answer.
        Raises an error if one occurred
        Returns a list of bytes

        Input:
            message (string)    : string conform the IVVI protocol

        Output:
            data_out_numbers (int[]) : return message
        '''
        logging.debug('Sending %r', message)

        # clear input buffer
        visafunc.read_all(self._vi)
        vpp43.write(self._vi, message)

# In stead of blocking, we could also poll, but it's a bit slower
#        print visafunc.get_navail(self._vi)
#        if not visafunc.wait_data(self._vi, 2, 0.5):
#            logging.error('Failed to receive reply from IVVI rack')
#            return False

        data1 = visafunc.readn(self._vi, 2)
        data1 = [ord(s) for s in data1]

        # 0 = no error, 32 = watchdog reset
        if data1[1] not in (0, 32):
            logging.error('Error while reading: %s', data1)

        data2 = visafunc.readn(self._vi, data1[0] - 2)
        data2 = [ord(s) for s in data2]

        return data1 + data2

    def do_set_pol_dacrack(self, flag, channel, getall=True):
        '''
        Changes the polarity of the specified set of dacs

        Input:
            flag (string) : 'BIP', 'POS' or 'NEG'
            channel (int) : 0 based index of the rack
            getall (boolean): if True (default) perform a get_all

        Output:
            None
        '''
        flagmap = {'NEG': -4000, 'BIP': -2000, 'POS': 0}
        if flag.upper() not in flagmap:
            logging.error('Tried to set invalid dac polarity %s', flag)
            return

        logging.debug('Setting polarity of rack %d to %s', channel, flag)
        val = flagmap[flag.upper()]
        for i in range(4*(channel-1),4*(channel)):
            self.pol_num[i] = val
            self.set_parameter_bounds('dac%d' % (i+1), val, val + 4000.0)

        if getall:
            self.get_all()

    def get_pol_dac(self, dacnr):
        '''
        Returns the polarity of the dac channel specified

        Input:
            dacnr (int) : 1 based index of the dac

        Output:
            polarity (string) : 'BIP', 'POS' or 'NEG'
        '''
        logging.debug('Getting polarity of dac %d' % dacnr)
        val = self.pol_num[dacnr-1]

        if (val == -4000):
            return 'NEG'
        elif (val == -2000):
            return 'BIP'
        elif (val == 0):
            return 'POS'
        else:
            return 'Invalid polarity in memory'

    def byte_limited_arange(self, start, stop, step=1, pol=None, dacnr=None):
        '''
        Creates array of mvoltages, in integer steps of the dac resolution. Either
        the dac polarity, or the dacnr needs to be specified.
        '''
        if pol is not None and dacnr is not None:
            logging.error('byte_limited_arange: speficy "pol" OR "dacnr", NOT both!')
        elif pol is None and dacnr is None:
            logging.error('byte_limited_arange: need to specify "pol" or "dacnr"')
        elif dacnr is not None:
            pol = self.get_pol_dac(dacnr)

        if (pol.upper() == 'NEG'):
            polnum = -4000
        elif (pol.upper() == 'BIP'):
            polnum = -2000
        elif (pol.upper() == 'POS'):
            polnum = 0
        else:
            logging.error('Try to set invalid dacpolarity')

        start_byte = int(round((start-polnum)/4000.0*65535))
        stop_byte = int(round((stop-polnum)/4000.0*65535))
        byte_vec = np.arange(start_byte, stop_byte+1, step)
        mvolt_vec = byte_vec/65535.0 * 4000.0 + polnum
        return mvolt_vec

    def get_numdacs(self):
        '''
        Get the number of DACS.
        '''
        return self._numdacs
