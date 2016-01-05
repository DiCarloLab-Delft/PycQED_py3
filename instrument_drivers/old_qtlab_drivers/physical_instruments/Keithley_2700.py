# Keithley_2700.py driver for Keithley 2700 DMM
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Reinier Heeres <reinier@heeres.eu>, 2008
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

class Keithley_2700(Instrument):
    '''
    This is the driver for the Keithley 2700 Multimeter

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Keithley_2700',
        address='<GBIP address>',
        reset=<bool>,
        change_display=<bool>,
        change_autozero=<bool>)
    '''

    def __init__(self, name, address, reset=False,
            change_display=True, change_autozero=True):
        '''
        Initializes the Keithley_2700, and communicates with the wrapper.

        Input:
            name (string)           : name of the instrument
            address (string)        : GPIB address
            reset (bool)            : resets to default values
            change_display (bool)   : If True (default), automatically turn off
                                        display during measurements.
            change_autozero (bool)  : If True (default), automatically turn off
                                        autozero during measurements.
        Output:
            None
        '''
        # Initialize wrapper functions
        logging.info('Initializing instrument Keithley_2700')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.instrument(self._address)
        self._modes = ['VOLT:AC', 'VOLT:DC', 'CURR:AC', 'CURR:DC', 'RES',
            'FRES', 'TEMP', 'FREQ']
        self._change_display = change_display
        self._change_autozero = change_autozero
        self._averaging_types = ['MOV','REP']
        self._trigger_sent = False

        # Add parameters to wrapper
        self.add_parameter('range',
            flags=Instrument.FLAG_GETSET,
            units='', minval=0.1, maxval=1000, type=float)
        self.add_parameter('trigger_continuous',
            flags=Instrument.FLAG_GETSET,
            type=bool)
        self.add_parameter('trigger_count',
            flags=Instrument.FLAG_GETSET,
            units='#', type=int)
        self.add_parameter('trigger_delay',
            flags=Instrument.FLAG_GETSET,
            units='s', minval=0, maxval=999999.999, type=float)
        self.add_parameter('trigger_source',
            flags=Instrument.FLAG_GETSET,
            units='')
        self.add_parameter('trigger_timer',
            flags=Instrument.FLAG_GETSET,
            units='s', minval=0.001, maxval=99999.999, type=float)
        self.add_parameter('mode',
            flags=Instrument.FLAG_GETSET,
            type=bytes, units='')
        self.add_parameter('digits',
            flags=Instrument.FLAG_GETSET,
            units='#', minval=4, maxval=7, type=int)
        self.add_parameter('readval', flags=Instrument.FLAG_GET,
            units='AU',
            type=float,
            tags=['measure'])
        self.add_parameter('readlastval', flags=Instrument.FLAG_GET,
            units='AU',
            type=float,
            tags=['measure'])
        self.add_parameter('readnextval', flags=Instrument.FLAG_GET,
            units='AU',
            type=float,
            tags=['measure'])
        self.add_parameter('integrationtime',
            flags=Instrument.FLAG_GETSET,
            units='s', type=float, minval=2e-4, maxval=1)
        self.add_parameter('nplc',
            flags=Instrument.FLAG_GETSET,
            units='#', type=float, minval=0.01, maxval=50)
        self.add_parameter('display', flags=Instrument.FLAG_GETSET,
            type=bool)
        self.add_parameter('autozero', flags=Instrument.FLAG_GETSET,
            type=bool)
        self.add_parameter('averaging', flags=Instrument.FLAG_GETSET,
            type=bool)
        self.add_parameter('averaging_window',
            flags=Instrument.FLAG_GETSET,
            units='%', type=float, minval=0, maxval=10)
        self.add_parameter('averaging_count',
            flags=Instrument.FLAG_GETSET,
            units='#', type=int, minval=1, maxval=100)
        self.add_parameter('averaging_type',
            flags=Instrument.FLAG_GETSET,
            type=bytes, units='')
        self.add_parameter('autorange',
            flags=Instrument.FLAG_GETSET,
            units='',
            type=bool)

        # Add functions to wrapper
        self.add_function('set_mode_volt_ac')
        self.add_function('set_mode_volt_dc')
        self.add_function('set_mode_curr_ac')
        self.add_function('set_mode_curr_dc')
        self.add_function('set_mode_res')
        self.add_function('set_mode_fres')
        self.add_function('set_mode_temp')
        self.add_function('set_mode_freq')
        self.add_function('set_range_auto')
        self.add_function('set_trigger_cont')
        self.add_function('set_trigger_disc')
        self.add_function('reset_trigger')
        self.add_function('reset')
        self.add_function('get_all')

        self.add_function('read')
        self.add_function('readlast')

        self.add_function('send_trigger')
        self.add_function('fetch')

        # Connect to measurement flow to detect start and stop of measurement
        qt.flow.connect('measurement-start', self._measurement_start_cb)
        qt.flow.connect('measurement-end', self._measurement_end_cb)

        if reset:
            self.reset()
        else:
            self.get_all()
            self.set_defaults()

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

    def set_defaults(self):
        '''
        Set to driver defaults:
        Output=data only
        Mode=Volt:DC
        Digits=7
        Trigger=Continous
        Range=10 V
        NPLC=1
        Averaging=off
        '''

        self._visainstrument.write('SYST:PRES')
        self._visainstrument.write(':FORM:ELEM READ')
            # Sets the format to only the read out, all options are:
            # READing = DMM reading, UNITs = Units,
            # TSTamp = Timestamp, RNUMber = Reading number,
            # CHANnel = Channel number, LIMits = Limits reading

        self.set_mode_volt_dc()
        self.set_digits(7)
        self.set_trigger_continuous(True)
        self.set_range(10)
        self.set_nplc(1)
        self.set_averaging(False)

    def get_all(self):
        '''
        Reads all relevant parameters from instrument

        Input:
            None

        Output:
            None
        '''
        logging.info('Get all relevant data from device')
        self.get_mode()
        self.get_range()
        self.get_trigger_continuous()
        self.get_trigger_count()
        self.get_trigger_delay()
        self.get_trigger_source()
        self.get_trigger_timer()
        self.get_mode()
        self.get_digits()
        self.get_integrationtime()
        self.get_nplc()
        self.get_display()
        self.get_autozero()
        self.get_averaging()
        self.get_averaging_window()
        self.get_averaging_count()
        self.get_averaging_type()
        self.get_autorange()

# Link old read and readlast to new routines:
    # Parameters are for states of the machnine and functions
    # for non-states. In principle the reading of the Keithley is not
    # a state (it's just a value at a point in time) so it should be a
    # function, technically. The GUI, however, requires an parameter to
    # read it out properly, so the reading is now done as if it is a
    # parameter, and the old functions are redirected.

    def read(self):
        '''
        Old function for read-out, links to get_readval()
        '''
        logging.debug('Link to get_readval()')
        return self.get_readval()

    def readlast(self):
        '''
        Old function for read-out, links to get_readlastval()
        '''
        logging.debug('Link to get_readlastval()')
        return self.get_readlastval()

    def readnext(self):
        '''
        Links to get_readnextval
        '''
        logging.debug('Link to get_readnextval()')
        return self.get_readnextval()

    def send_trigger(self):
        '''
        Send trigger to Keithley, use when triggering is not continous.
        '''
        trigger_status = self.get_trigger_continuous(query=False)
        if (trigger_status):
            logging.warning('Trigger is set to continous, sending trigger impossible')
        elif (not trigger_status):
            logging.debug('Sending trigger')
            self._visainstrument.write('INIT')
            self._trigger_sent = True
        else:
            logging.error('Error in retrieving triggering status, no trigger sent.')

    def fetch(self):
        '''
        Get data at this instance, not recommended, use get_readlastval.
        Use send_trigger() to trigger the device.
        Note that Readval is not updated since this triggers itself.
        '''

        trigger_status = self.get_trigger_continuous(query=False)
        if self._trigger_sent and (not trigger_status):
            logging.debug('Fetching data')
            reply = self._visainstrument.ask('Fetch?')
            self._trigger_sent = False
            return float(reply[0:15])
        elif (not self._trigger_sent) and (not trigger_status):
            logging.warning('No trigger sent, use send_trigger')
        else:
            logging.error('Triggering is on continous!')

    def set_mode_volt_ac(self):
        '''
        Set mode to AC Voltage

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to AC Voltage')
        self.set_mode('VOLT:AC')

    def set_mode_volt_dc(self):
        '''
        Set mode to DC Voltage

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to DC Voltage')
        self.set_mode('VOLT:DC')

    def set_mode_curr_ac(self):
        '''
        Set mode to AC Current

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to AC Current')
        self.set_mode('CURR:AC')

    def set_mode_curr_dc(self):
        '''
        Set mode to DC Current

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to DC Current')
        self.set_mode('CURR:DC')

    def set_mode_res(self):
        '''
        Set mode to Resistance

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to Resistance')
        self.set_mode('RES')

    def set_mode_fres(self):
        '''
        Set mode to 'four wire Resistance'

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to "four wire resistance"')
        self.set_mode('FRES')

    def set_mode_temp(self):
        '''
        Set mode to Temperature

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to Temperature')
        self.set_mode('TEMP')

    def set_mode_freq(self):
        '''
        Set mode to Frequency

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set mode to Frequency')
        self.set_mode('FREQ')

    def set_range_auto(self, mode=None):
        '''
        Old function to set autorange, links to set_autorange()
        '''
        logging.debug('Redirect to set_autorange')
        self.set_autorange(True)

    def set_trigger_cont(self):
        '''
        Set trigger mode to continuous, old function, uses set_Trigger_continuous(True).

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set Trigger to continuous mode')
        self.set_trigger_continuous(True)

    def set_trigger_disc(self):
        '''
        Set trigger mode to Discrete

        Input:
            None

        Output:
            None
        '''
        logging.debug('Set Trigger to discrete mode')
        self.set_trigger_continuous(False)

    def reset_trigger(self):
        '''
        Reset trigger status

        Input:
            None

        Output:
            None
        '''
        logging.debug('Resetting trigger')
        self._visainstrument.write(':ABOR')


# --------------------------------------
#           parameters
# --------------------------------------

    def do_get_readnextval(self):
        '''
        Waits for the next value available and returns it as a float.
        Note that if the reading is triggered manually, a trigger must
        be send first to avoid a time-out.

        Input:
            None

        Output:
            value(float) : last triggerd value on input
        '''
        logging.debug('Read next value')

        trigger_status = self.get_trigger_continuous(query=False)
        if (not trigger_status) and (not self._trigger_sent):
            logging.error('No trigger has been send, return 0')
            return float(0)
        self._trigger_sent = False

        text = self._visainstrument.ask(':DATA:FRESH?')
            # Changed the query to from Data?
            # to Data:FRESH? so it will actually wait for the
            # measurement to finish.
        return float(text[0:15])

    def do_get_readlastval(self):
        '''
        Returns the last measured value available and returns it as a float.
        Note that if this command is send twice in one integration time it will
        return the same value.

        Example:
        If continually triggering at 1 PLC, don't use the command within 1 PLC
        again, but wait 20 ms. If you want the Keithley to wait for a new
        measurement, use get_readnextval.

        Input:
            None

        Output:
            value(float) : last triggerd value on input
        '''
        logging.debug('Read last value')

        text = self._visainstrument.ask(':DATA?')
        return float(text[0:15])

    def do_get_readval(self, ignore_error=False):
        '''
        Aborts current trigger and sends a new trigger
        to the device and reads float value.
        Do not use when trigger mode is 'CONT'
        Instead use readlastval

        Input:
            ignore_error (boolean): Ignore trigger errors, default is 'False'

        Output:
            value(float) : currrent value on input
        '''
        trigger_status = self.get_trigger_continuous(query=False)
        if trigger_status:
            if ignore_error:
                logging.debug('Trigger=continuous, can\'t trigger, return 0')
            else:
                logging.error('Trigger=continuous, can\'t trigger, return 0')
            text = '0'
            return float(text[0:15])
        elif not trigger_status:
            logging.debug('Read current value')
            text = self._visainstrument.ask('READ?')
            self._trigger_sent = False
            return float(text[0:15])
        else:
            logging.error('Error in retrieving triggering status, no trigger sent.')



    def do_set_range(self, val, mode=None):
        '''
        Set range to the specified value for the
        designated mode. If mode=None, the current mode is assumed

        Input:
            val (float)   : Range in specified units
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            None
        '''
        logging.debug('Set range to %s' % val)
        self._set_func_par_value(mode, 'RANG', val)

    def do_get_range(self, mode=None):
        '''
        Get range for the specified mode.
        If mode=None, the current mode is assumed.

        Input:
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            range (float) : Range in the specified units
        '''
        logging.debug('Get range')
        return float(self._get_func_par(mode, 'RANG'))

    def do_set_digits(self, val, mode=None):
        '''
        Set digits to the specified value ?? Which values are alowed?
        If mode=None the current mode is assumed

        Input:
            val (int)     : Number of digits
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            None
        '''
        logging.debug('Set digits to %s' % val)
        self._set_func_par_value(mode, 'DIG', val)

    def do_get_digits(self, mode=None):
        '''
        Get digits
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            digits (int) : Number of digits
        '''
        logging.debug('Getting digits')
        return int(self._get_func_par(mode, 'DIG'))

    def do_set_integrationtime(self, val, mode=None):
        '''
        Set integration time to the specified value in seconds.
        To set the integrationtime as a Number of PowerLine Cycles,
        use set_nplc(). Note that this will automatically update nplc as well.
        If mode=None the current mode is assumed

        Input:
            val (float)   : Integration time in seconds.
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''

        logging.debug('Set integration time to %s seconds' % val)
        self._set_func_par_value(mode, 'APER', val)
        self.get_nplc()


    def do_set_nplc(self, val, mode=None, unit='APER'):
        '''
        Set integration time to the specified value in Number of Powerline Cycles.
        To set the integrationtime in seconds, use set_integrationtime().
        Note that this will automatically update integrationtime as well.
        If mode=None the current mode is assumed

        Input:
            val (float)   : Integration time in nplc.
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''
        logging.debug('Set integration time to %s PLC' % val)
        self._set_func_par_value(mode, 'NPLC', val)
        self.get_integrationtime()

    def do_get_integrationtime(self, mode=None, unit='APER'):
        '''
        Get integration time in seconds.
        To get the integrationtime as a Number of PowerLine Cycles, use get_nplc().
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to get property of. Choose from self._modes.

        Output:
            time (float) : Integration time in seconds
        '''
        logging.debug('Read integration time in seconds')
        return float(self._get_func_par(mode, 'APER'))

    def do_get_nplc(self, mode=None, unit='APER'):
        '''
        Get integration time in Number of PowerLine Cycles.
        To get the integrationtime in seconds, use get_integrationtime().
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to get property of. Choose from self._modes.

        Output:
            time (float) : Integration time in PLCs
        '''
        logging.debug('Read integration time in PLCs')
        return float(self._get_func_par(mode, 'NPLC'))


    def do_set_trigger_continuous(self, val):
        '''
        Set trigger mode to continuous.

        Input:
            val (boolean) : Trigger on or off

        Output:
            None
        '''
        val = bool_to_str(val)
        logging.debug('Set trigger mode to %s' % val)
        self._set_func_par_value('INIT', 'CONT', val)

    def do_get_trigger_continuous(self):
        '''
        Get trigger mode from instrument

        Input:
            None

        Output:
            val (bool) : returns if triggering is continuous.
        '''
        logging.debug('Read trigger mode from instrument')
        return bool(int(self._get_func_par('INIT', 'CONT')))

    def do_set_trigger_count(self, val):
        '''
        Set trigger count
        if val>9999 count is set to INF

        Input:
            val (int) : trigger count

        Output:
            None
        '''
        logging.debug('Set trigger count to %s' % val)
        if val > 9999:
            val = 'INF'
        self._set_func_par_value('TRIG', 'COUN', val)

    def do_get_trigger_count(self):
        '''
        Get trigger count

        Input:
            None

        Output:
            count (int) : Trigger count
        '''
        logging.debug('Read trigger count from instrument')
        ans = self._get_func_par('TRIG', 'COUN')
        try:
            ret = int(ans)
        except:
            ret = 0

        return ret

    def do_set_trigger_delay(self, val):
        '''
        Set trigger delay to the specified value

        Input:
            val (float) : Trigger delay in seconds

        Output:
            None
        '''
        logging.debug('Set trigger delay to %s' % val)
        self._set_func_par_value('TRIG', 'DEL', val)

    def do_get_trigger_delay(self):
        '''
        Read trigger delay from instrument

        Input:
            None

        Output:
            delay (float) : Delay in seconds
        '''
        logging.debug('Get trigger delay')
        return float(self._get_func_par('TRIG', 'DEL'))

    def do_set_trigger_source(self, val):
        '''
        Set trigger source

        Input:
            val (string) : Trigger source

        Output:
            None
        '''
        logging.debug('Set Trigger source to %s' % val)
        self._set_func_par_value('TRIG', 'SOUR', val)

    def do_get_trigger_source(self):
        '''
        Read trigger source from instrument

        Input:
            None

        Output:
            source (string) : The trigger source
        '''
        logging.debug('Getting trigger source')
        return self._get_func_par('TRIG', 'SOUR')

    def do_set_trigger_timer(self, val):
        '''
        Set the trigger timer

        Input:
            val (float) : the value to be set

        Output:
            None
        '''
        logging.debug('Set trigger timer to %s' % val)
        self._set_func_par_value('TRIG', 'TIM', val)

    def do_get_trigger_timer(self):
        '''
        Read the value for the trigger timer from the instrument

        Input:
            None

        Output:
            timer (float) : Value of timer
        '''
        logging.debug('Get trigger timer')
        return float(self._get_func_par('TRIG', 'TIM'))

    def do_set_mode(self, mode):
        '''
        Set the mode to the specified value

        Input:
            mode (string) : mode to be set. Choose from self._modes

        Output:
            None
        '''

        logging.debug('Set mode to %s', mode)
        if mode in self._modes:
            string = ':CONF:%s' % mode
            self._visainstrument.write(string)

            if mode.startswith('VOLT'):
                self._change_units('V')
            elif mode.startswith('CURR'):
                self._change_units('A')
            elif mode.startswith('RES'):
                self._change_units('Ohm')
            elif mode.startswith('FREQ'):
                self._change_units('Hz')

        else:
            logging.error('invalid mode %s' % mode)

        self.get_all()
            # Get all values again because some paramaters depend on mode

    def do_get_mode(self):
        '''
        Read the mode from the device

        Input:
            None

        Output:
            mode (string) : Current mode
        '''
        string = ':CONF?'
        logging.debug('Getting mode')
        ans = self._visainstrument.ask(string)
        return ans.strip('"')

    def do_get_display(self):
        '''
        Read the staturs of diplay

        Input:
            None

        Output:
            True = On
            False= Off
        '''
        logging.debug('Reading display from instrument')
        reply = self._get_func_par('DISP','ENAB')
        return bool(int(reply))

    def do_set_display(self, val):
        '''
        Switch the diplay on or off.

        Input:
            val (boolean) : True for display on and False for display off

        Output

        '''
        logging.debug('Set display to %s' % val)
        val = bool_to_str(val)
        return self._set_func_par_value('DISP','ENAB',val)

    def do_get_autozero(self):
        '''
        Read the staturs of the autozero function

        Input:
            None

        Output:
            reply (boolean) : Autozero status.
        '''
        logging.debug('Reading autozero status from instrument')
        reply = self._get_func_par('SYST','AZER:STAT')
        return bool(int(reply))

    def do_set_autozero(self, val):
        '''
        Switch the diplay on or off.

        Input:
            val (boolean) : True for display on and False for display off

        Output

        '''
        logging.debug('Set autozero to %s' % val)
        val = bool_to_str(val)
        return self._set_func_par_value('SYST','AZER:STAT',val)

    def do_set_averaging(self, val, mode=None):
        '''
        Switch averaging on or off.
        If mode=None the current mode is assumed

        Input:
            val (boolean)
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''
        logging.debug('Set averaging to %s ' % val)
        val = bool_to_str(val)
        self._set_func_par_value(mode, 'AVER:STAT', val)

    def do_get_averaging(self, mode=None):
        '''
        Get status of averaging.
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            result (boolean)
        '''
        logging.debug('Get averaging')
        reply = self._get_func_par(mode, 'AVER:STAT')
        return bool(int(reply))

    def do_set_averaging_window(self, val, mode=None):
        '''
        Set window of averaging in %.
        If mode=None the current mode is assumed

        Input:
            val (float)   : Averaging window in %.
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''
        logging.debug('Set averaging_window to %s ' % val)
        self._set_func_par_value(mode, 'AVER:WIND', val)

    def do_get_averaging_window(self, mode=None):
        '''
        Get averaging window in %.
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to get property for. Choose from self._modes.

        Output:
            result (float) : Averaging window in %
        '''
        logging.debug('Get averaging window')
        reply = self._get_func_par(mode, 'AVER:WIND')
        return float(reply)

    def do_set_averaging_count(self, val, mode=None):
        '''
        Set averaging count.
        If mode=None the current mode is assumed

        Input:
            val (int)   : Averaging count.
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''
        logging.debug('Set averaging_window to %s ' % val)
        self._set_func_par_value(mode, 'AVER:COUN', val)

    def do_get_averaging_count(self, mode=None):
        '''
        Get averaging count.
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to get property for. Choose from self._modes.

        Output:
            result (int) : Averaging count
        '''
        logging.debug('Get averaging count')
        reply = self._get_func_par(mode, 'AVER:COUN')
        return int(reply)

    def do_set_autorange(self, val, mode=None):
        '''
        Switch autorange on or off.
        If mode=None the current mode is assumed

        Input:
            val (boolean)
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            None
        '''
        logging.debug('Set autorange to %s ' % val)
        val = bool_to_str(val)
        self._set_func_par_value(mode, 'RANG:AUTO', val)

    def do_get_autorange(self, mode=None):
        '''
        Get status of averaging.
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to set property for. Choose from self._modes.

        Output:
            result (boolean)
        '''
        logging.debug('Get autorange')
        reply = self._get_func_par(mode, 'RANG:AUTO')
        return bool(int(reply))

    def do_set_averaging_type(self, type, mode=None):
        '''
        Set the averaging_type to the specified value
        If mode=None the current mode is assumed

        Input:
            type (string) : averaging type to be set. Choose from self._averaging_types
                            or choose 'moving' or 'repeat'.
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            None
        '''

        logging.debug('Set averaging type to %s', type)
        if type is 'moving':
            type='MOV'
        elif type is 'repeat':
            type='REP'

        if type in self._averaging_types:
            self._set_func_par_value(mode, 'AVER:TCON', type)
        else:
            logging.error('invalid type %s' % type)

    def do_get_averaging_type(self, mode=None):
        '''
        Read the mode from the device
        If mode=None the current mode is assumed

        Input:
            mode (string) : mode to get property for. Choose from self._modes.

        Output:
            type (string) : Current avering type for specified mode.
        '''
        logging.debug('Getting mode')
        ans = self._get_func_par(mode,'AVER:TCON')
        if ans.startswith('REP'):
            ans='repeat'
        elif ans.startswith('MOV'):
            ans='moving'
        return ans
# --------------------------------------
#           Internal Routines
# --------------------------------------

    def _change_units(self, unit):
        self.set_parameter_options('readval', units=unit)
        self.set_parameter_options('readlastval', units=unit)
        self.set_parameter_options('readnextval', units=unit)

    def _determine_mode(self, mode):
        '''
        Return the mode string to use.
        If mode is None it will return the currently selected mode.
        '''
        logging.debug('Determine mode with mode=%s' % mode)
        if mode is None:
            mode = self.get_mode(query=False)
        if mode not in self._modes and mode not in ('INIT', 'TRIG', 'SYST', 'DISP'):
            logging.warning('Invalid mode %s, assuming current' % mode)
            mode = self.get_mode(query=False)
        return mode

    def _set_func_par_value(self, mode, par, val):
        '''
        For internal use only!!
        Changes the value of the parameter for the function specified

        Input:
            mode (string) : The mode to use
            par (string)  : Parameter
            val (depends) : Value

        Output:
            None
        '''
        mode = self._determine_mode(mode)
        string = ':%s:%s %s' % (mode, par, val)
        logging.debug('Set instrument to %s' % string)
        self._visainstrument.write(string)

    def _get_func_par(self, mode, par):
        '''
        For internal use only!!
        Reads the value of the parameter for the function specified
        from the instrument

        Input:
            func (string) : The mode to use
            par (string)  : Parameter

        Output:
            val (string) :
        '''
        mode = self._determine_mode(mode)
        string = ':%s:%s?' % (mode, par)
        ans = self._visainstrument.ask(string)
        logging.debug('ask instrument for %s (result %s)' % \
            (string, ans))
        return ans

    def _measurement_start_cb(self, sender):
        '''
        Things to do at starting of measurement
        '''
        if self._change_display:
            self.set_display(False)
            #Switch off display to get stable timing
        if self._change_autozero:
            self.set_autozero(False)
            #Switch off autozero to speed up measurement

    def _measurement_end_cb(self, sender):
        '''
        Things to do after the measurement
        '''
        if self._change_display:
            self.set_display(True)
        if self._change_autozero:
            self.set_autozero(True)

