Switch_

# RS_Step_Attenuator.py class, to perform the communication between the Wrapper and the device
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

class Radial_Switch(Instrument):
    '''
    This is the python driver for the raidal switch

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'RS_Step_Attenuator', address='<serial address>')
    '''

    def __init__(self, name, address):
        '''
        Initializes the radial switch and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : serial address

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])


        # Add some global constants
        self._address = address
        self._visainstrument = visa.instrument(self._address)

        self.add_parameter('channel',
            flags=Instrument.FLAG_SET, minval=1, maxval=139, type=int)

        self.reset()

    def do_set_channel(self, state):
        '''
        Apply the desired attenuation

        Input:
            state and channel

        Output:
            None
        '''
        self._visainstrument.write('%s'%ch)