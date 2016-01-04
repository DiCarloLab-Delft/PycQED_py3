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
import numpy

import qt

def bool_to_str(val):
    '''
    Function to convert boolean to 'ON' or 'OFF'
    '''
    if val == True:
        return "ON"
    else:
        return "OFF"

class Aeroflex_8310(Instrument):
    '''
    This is the driver for the  Aeroflex_8310 variable attenuator

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
        self._visainstrument = visa.instrument(self._address)

        self._mode = 'serial'

        if self._mode == 'serial':
            self.add_parameter('attenuation', type=int,
                flags=Instrument.FLAG_GETSET, maxval = 62)

        else:
            self.do_set_attenuation = self.set_ch_attenuation
            self.do_get_attenuation = self.get_ch_attenuation
            self.add_parameter('attenuation', type=int,
                flags=Instrument.FLAG_GETSET,
                channels=(1, 2), channel_prefix='ch%d_')



        # Connect to measurement flow to detect start and stop of measurement


        if reset:
            self.reset()


# --------------------------------------
#           functions
# --------------------------------------

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
        self.get_all()

    def visa_read(self):
        return self._visainstrument.read()
    def visa_write(self, cmd):
        self._visainstrument.write(cmd)
    def visa_ask(self,cmd):
        qt.mstart()
        self._visainstrument.write(cmd)
        done = False
        while not done:
            qt.msleep()
            try:
                mes = self._visainstrument.read()
                done = True
            except:
                print('busy, time out while reading response to "%s" query'%cmd)
                done = False
        qt.mend()
        return mes
    def set_ch_attenuation(self, att, channel):
            self.visa_write('CHAN %s'%channel)
            self.visa_write('ATT %s'%att)
    def get_ch_attenuation(self, channel):
        self.visa_write('CHAN %s'%channel)
        return self.visa_ask('ATT?')

    def do_set_attenuation(self, att):
            if att>60:
                self.set_ch_attenuation(60,1)
                qt.msleep(0.2)
                self.set_ch_attenuation(att-60,2)
                qt.msleep(0.2)
            else:
                self.set_ch_attenuation(0,1)
                qt.msleep(0.2)
                self.set_ch_attenuation(att,2)
                qt.msleep(0.2)
    def do_get_attenuation(self):
        att = 0
        for ch in [1,2]:
            self.visa_write('CHAN %s'%ch)
            atti =  self.visa_ask('ATT?')
            print('ch',ch,': %sdB'%atti)
            att += int(float(atti))
        return att