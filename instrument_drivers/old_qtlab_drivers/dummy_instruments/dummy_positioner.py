# dummy_positioner, dummy positioner code
# Reinier Heeres <reinier@heeres.eu>, 2009
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
import types

class dummy_positioner(Instrument):

    def __init__(self, name, channels=3):
        Instrument.__init__(self, name, tags=['positioner'])

        # Instrument parameters
        self.add_parameter('position',
            type=tuple,
            flags=Instrument.FLAG_GET,
            format='%.03f, %.03f, %.03f')
        self.add_parameter('speed',
            type=tuple,
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
            format='%.1f, %.01f, %.01f')
        self.add_parameter('channels',
            type=int,
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)

        self.set_channels(channels)

        # Instrument functions
        self.add_function('start')
        self.add_function('stop')
        self.add_function('move_abs')

    def do_get_position(self, query=True):
        return [0, 0, 0]

    def do_set_channels(self, val):
        return True

    def do_set_speed(self, val):
        print('Setting speed to %r' % (val, ))

    def start(self):
        print('Starting')

    def stop(self):
        print('Stopping')

    def step(self, chan, nsteps):
        print('Stepping channel %d by %d' % (chan, nsteps))

    def move_abs(self, pos, **kwargs):
        print('Moving to %r' % (pos, ))
