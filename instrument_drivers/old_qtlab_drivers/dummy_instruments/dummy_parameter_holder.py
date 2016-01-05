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

class dummy_parameter_holder(Instrument):

    def __init__(self, name, channels=3):
        Instrument.__init__(self, name, tags=['positioner'])

        # Instrument parameters
        self.add_parameter('x',
            type=int,
            flags=Instrument.FLAG_GET)

        self.add_parameter('y',
            type=int,
            flags=Instrument.FLAG_GET)

    def _do_get_x():
        return self.x

    def _do_set_x(self, x):
        self.x= x

    def measure_convexity(self):
        return self.x**2 + self.y**2

