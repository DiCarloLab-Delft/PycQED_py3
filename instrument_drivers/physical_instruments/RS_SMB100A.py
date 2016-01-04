# RS_SMB100.py class, to perform the communication between the Wrapper and the device
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
import numpy
from time import sleep

class RS_SMB100A(Instrument):
    '''
    This is the python driver for the Rohde & Schwarz SMB100
    signal generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RS_SMB100', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the RS_SMB100, and communicates with the wrapper.

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
        self._visainstrument = visa.instrument(self._address, timeout = 60)
        print(' SMB timeout set to: %s s'%self._visainstrument.timeout)

        self.add_parameter('frequency', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=9e3, maxval=40e9,
            units='Hz',
            tags=['sweep'])
        self.add_parameter('phase', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=360,
            units='DEG', format='%.01e',
            tags=['sweep'])
        self.add_parameter('power', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=-120, maxval=30, units='dBm',
            tags=['sweep'])
        self.add_parameter('status', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('pulsemod_state', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_function('reset')
        self.add_function('get_all')

        self._max_power     =   30

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

    def _do_set_mode(self, mode):
        '''
        Change mode to list sweep or continious
        '''
        self._visainstrument.write('SOUR:FREQ:MODE %s' % mode)

    def enable_list_mode(self):
        self._visainstrument.write('SOUR:FREQ:MODE LIST')

    def enable_list_step_mode(self):
        self._visainstrument.write('SOUR:LIST:MODE STEP')

    def set_list_ext_trigger_source(self):
        self._visainstrument.write('SOUR:LIST:TRIG:SOUR ext')

    def reset_list_mode(self):
        self._visainstrument.write('SOUR:LIST:RES')

    def learn_list(self):
        self._visainstrument.write('SOUR:LIST:LEAR')

    def do_set_pulsemod_state(self, state):
        if (state.upper() == 'ON'):
            state_s = 'ON'
        elif (state.upper() == 'OFF'):
            state_s = 'OFF'
        else:
            logging.error(__name__ + ' : Unable to set pulsed mode to %s,\
                                         expected "ON" or "OFF"' %state)
        self._visainstrument.write(':PULM:SOUR EXT')
        self._visainstrument.write(':SOUR:PULM:STAT %s'%state_s)
    def do_get_pulsemod_state(self):
        return self._visainstrument.ask(':SOUR:PULM:STAT?') == '1'

    def do_get_pulsemod_state(self):
        return self._visainstrument.ask('SOUR:PULM:STAT?') == '1'

    def set_pulsemod_source(self):
        self._visainstrument.write('SOUR:PULM:SOUR EXT')

    def send_visa(self,command):
        self._visainstrument.write(command)

    def ask_visa(self,command):
        return self._visainstrument.ask(command)

    def _do_set_phase(self, poffset):
        self._phase_offset=poffset

        self._visainstrument.write('SOUR:PHAS %sDEG'%poffset)
    def _do_get_phase(self):
        self._phase_offset=self._visainstrument.ask('SOUR:PHAS?')

        return self._phase_offset

    def perform_internal_adjustments(self,all_f = False,cal_IQ_mod=True):
        status=self.get_status()
        self.off()
        if all_f:
            s=self._visainstrument.ask('CAL:ALL:MEAS?')
        else:
            s=self._visainstrument.ask('CAL:FREQ:MEAS?')
            print('Frequency calibrated')
            s=self._visainstrument.ask('CAL:LEV:MEAS?')
            print('Level calibrated')

        self.set_status(status)
        self.set_PuM_state(False)
        sleep(0.1)
        self.set_PuM_state(True)

    def _create_list(self, start, stop, unit, number_of_steps):
        flist_l = numpy.linspace(start, stop, number_of_steps)
        flist_s = ''
        k=0
        for f_el in flist_l:
            if k is 0:
                flist_s = flist_s + '%s%s'%(int(flist_l[k]),unit)
            else:
                flist_s = flist_s + ', %s%s'%(int(flist_l[k]),unit)
            k+=1
        return flist_s
    def reset_list(self):
        self._visainstrument.write('ABOR:LIST')

    def _create_list_from_values(self, values, unit):
        flist_l = values
        flist_s = ''
        k=0
        for f_el in flist_l:
            if k is 0:
                flist_s = flist_s + '%s%s'%(int(flist_l[k]),unit)
            else:
                flist_s = flist_s + ', %s%s'%(int(flist_l[k]),unit)
            k+=1
        #print flist_s
        return flist_s

    def load_f_values_to_list(self,fvalues, power, funit='Hz'):
        plist = len(fvalues)*[power]
        self.load_fp_values_to_list(fvalues, funit , plist, 'dBm')
        #print 'List loaded'

    def load_fp_values_to_list(self, fvalues, funit , pvalues, punit):
        self._visainstrument.write('SOUR:LIST:DEL:ALL')
        self._visainstrument.write('SOUR:LIST:SEL "list_%s_%s_%s"'%(int(fvalues[0]), int(fvalues[-1]), numpy.size(fvalues)))

        flist = self._create_list_from_values(fvalues, funit)
        plist = self._create_list_from_values(pvalues, punit)
        #print flist
        #print plist

        self._visainstrument.write('SOUR:LIST:FREQ '+flist)
        self._visainstrument.write('SOUR:LIST:POW '+plist)

    def load_fplist(self, fstart, fstop, funit ,
                    pstart, pstop, punit, number_of_steps):

        fvalues = numpy.linspace(fstart, fstop, number_of_steps)
        pvalues = numpy.linspace(pstart, pstop, number_of_steps)
        self.load_fp_values_to_list(fvalues, funit, pvalues, punit)

        #self._visainstrument.write('')

    def _do_get_frequency(self):
        '''
        Get frequency from device

        Input:
            None

        Output:
            frequency (float) : frequency in Hz
        '''
        logging.debug(__name__ + ' : reading frequency from instrument')
        return float(self._visainstrument.ask('SOUR:FREQ?'))

    def _do_set_frequency(self, frequency, perform_internal_adjustments=False):
        '''
        Set frequency of device

        Input:
            frequency (float) : frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting frequency to %s GHz' % frequency)
        self._visainstrument.write('SOUR:FREQ %s' % frequency)
        if perform_internal_adjustments:
            self.perform_internal_adjustments()

    def _do_get_power(self):
        '''
        Get output power from device

        Input:
            None

        Output:
            power (float) : output power in dBm
        '''
        logging.debug(__name__ + ' : reading power from instrument')
        return float(self._visainstrument.ask('SOUR:POW?'))

    def _do_set_power(self, power):
        '''
        Set output power of device

        Input:
            power (float) : output power in dBm

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting power to %s dBm' % power)
        if power > self._max_power:
            self._visainstrument.write('SOUR:POW %e' % self._max_power)
            logging.debug(__name__ + ' : Exceeding max_power, '
                          'Setting power to %s dBm' % self._max_power)
            print(' : Exceeding max_power, Setting power to %s dBm' \
                % self._max_power)
        else:
            self._visainstrument.write('SOUR:POW %e' % power)

    def set_max_power(self, max_power):
        '''
        Set maximum output power of device

        Input:
            Maximum Power
        '''

        self._max_power =  max_power

    def get_max_power(self):
        '''
        Get output power from device

        Input:
            None

        Output:
            max power (float) : output maximum power in dBm
        '''
        logging.debug(__name__ + ' : reading max power from instrument')
        return float(self._max_power)


    def _do_get_status(self):
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

    def _do_set_status(self, status):
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
