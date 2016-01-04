# RS_SMR40.py class, to perform the communication between the Wrapper and the device
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
import qt
from time import sleep
import time
import sys

class RS_FSP(Instrument):
    '''
    This is the python driver for the Rohde & Schwarz spectrum analyzer

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RS_FSP', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the RS_FSV, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        self._visainstrument = visa.instrument(self._address, timeout=2)
        self.add_parameter('sweep_time', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, units='s', tags=['sweep'])
        self.add_parameter('bandwidth', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, units='Hz', tags=['sweep'])
        self.add_parameter('start_frequency', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=1e-8, units='GHz', tags=['sweep'])
        self.add_parameter('stop_frequency', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=1e-8, units='GHz', tags=['sweep'])
        self.add_parameter('npoints', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=101, units='points', tags=['sweep'])
        self.add_parameter('averages', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=1, units='x', tags=['sweep'])
        self.add_parameter('ext_power', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=-130, units='dBm', tags=['sweep'])
        self.add_parameter('reference_level', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=-130, maxval=30, units='dBm', tags=['sweep'])

        # self.add_parameter('frequency', type=types.FloatType,
            # flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            # minval=1e9, maxval=40e9,
            # units='Hz', format='%.04e',
            # tags=['sweep'])
        # self.add_parameter('power', type=types.FloatType,
            # flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            # minval=-30, maxval=25, units='dBm',
            # tags=['sweep'])
        # self.add_parameter('status', type=types.StringType,
            # flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

        self.add_function('reset')
        self.add_function('prepare_sweep')
        self.add_function('start_single_sweep')
        self.add_function('download_trace')
        # self.add_function('get_all')
        self.ext_gen = 0
        if reset:
            self.reset()
        # else:
            # self.get_all()

    # Functions
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
                sys.stdout.write('.')
                done = False
        qt.mend()
        return mes

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
        # self.get_all()

    # def get_all(self):
        # '''
        # Reads all implemented parameters from the instrument,
        # and updates the wrapper.

        # Input:
            # None

        # Output:
            # None
        # '''
        # logging.info(__name__ + ' : reading all settings from instrument')
        # self.get_frequency()
        # self.get_power()
        # self.get_status()

    # communication with machine
    def ask_visa_command(self, visa_str):
        return self.visa_ask(visa_str)
    def write_visa_command(self, visa_str):
        return self._visainstrument.write(visa_str)
    def set_center_marker(self):
        return self.write_visa_command('CALCULATE:MARKER:FUNCTION:CENTER')
    def set_marker_frequency(self, freq):
        return self.write_visa_command('CALCULATE:MARKER:X %f GHz' % freq)
    def set_marker_max(self):
        return self.write_visa_command('CALCULATE:MARKER:MAX')
    def get_marker_power(self):
        return float(self.visa_ask('CALCULATE:MARKER:Y?'))
    def get_marker_frequency(self):
        return float(self.visa_ask('CALCULATE:MARKER:X?'))
    def do_get_bandwidth(self):
        self.resBW = float(self.visa_ask('BAND:res?'))
        return self.resBW
    def do_set_bandwidth(self,bw):
        self.resbw = bw
        #print 'res_BW = %sHz'%BW
        self._visainstrument.write('sens:band:res %sHz'%bw)

    def do_set_reference_level(self, reference_level):
        self._visainstrument.write('DISP:TRAC:Y:RLEV %d' % reference_level)

    def do_get_reference_level(self):
        return self._visainstrument.ask('DISP:TRAC:Y:RLEV?')

    def do_set_start_frequency(self,fstart):
        self.fstart = fstart
        self._visainstrument.write('FREQ:STAR %sGHz'%fstart)
    def do_get_start_frequency(self):
        self.fstart = self.visa_ask('FREQ:STAR?')
        return self.fstart
    def do_set_stop_frequency(self,fstop):
        self.fstop=fstop
        self._visainstrument.write('FREQ:STOP %sGHz'%fstop)
    def do_get_stop_frequency(self):
        self.fstop = self.visa_ask('FREQ:STOP?')
        return self.fstop
    def do_set_npoints(self, npoints):
        self.Npoints=npoints
        self._visainstrument.write('SWE:POIN %s'%npoints)
    def do_get_npoints(self):
        self.Npoints = int(self.visa_ask('SWE:POIN?'))
        return self.Npoints
    def do_set_sweep_time(self, time):

        self._visainstrument.write('SWE:TIME %ss'%time)
    def do_get_sweep_time(self):
        return float(self.visa_ask('SWE:TIME?'))
    def do_set_averages(self,no_avg):
        self._visainstrument.write('SWE:COUN %s' %no_avg)
    def do_get_averages(self,no_avg):
        return float(self.visa_ask('SWE:COUN?'))
    def do_set_ext_power(self,power):
        self._visainstrument.write('SOUR:EXT%s:POW %sdBm'%(self.ext_gen,power))
    def do_get_ext_power(self):
        self.power = float(self.visa_ask('SOUR:EXT%s:POW?'))
        return self.power
    def get_reference_level(self):
        return int(self.visa_ask('DISP:TRAC:Y:SCAL:RLEV?'))
    def set_reference_level(self, level):
        self._visainstrument.write('DISP:TRAC:Y:RLEV %s'%level)
    def set_attenuation_level(self, level):
        self._visainstrument.write('DISP:TRAC:Y:RLEV:OFFset %s'%level)
    # utility functions
    def prepare_sweep(self,fstart,fstop,npoints,bw,power,averages, cont_sweep='OFF'):
        '''
        Prepares the FSP for doing a sweep
        Npoints = number of points in sweep
        cont_sweep = ON means sweep repeats indefinitely
        '''
        self.set_start_frequency(fstart)
        self.set_stop_frequency(fstop)
        self.set_npoints(int(npoints))
        self.set_bandwidth(bw)
        self.set_ext_power(power)
        self.set_averages(averages)
        self._visainstrument.write('INIT:CONT %s' %cont_sweep)

    def continue_meas(self):
        self._visainstrument.write('INIT2;*CONT')
    def start_sweep(self):
        self._visainstrument.write('INIT')
    def reset_sweep(self):
        '''
        stops the current sweep
        '''
        self._visainstrument.write('ABORT')
    def visa_read(self):
        return self._visainstrument.read()
    def visa_write(self, cmd):
        self._visainstrument.write(cmd)
    def enter_visa_ask(self, cmd):
        return self.visa_ask(cmd)
    def FSP_ready(self):
        try:
            self._visainstrument.ask('*OPC?')
            ready = True
        except:
            ready = False
        return ready

    def start_single_sweep(self):
        '''
        Starts a single sweep and records the result in Trace1
        input: N = number of points
        returns nothing, command is returned to the console after the sweep is finished
        '''
        qt.mstart()
        self._visainstrument.write('INIT:CONT OFF')
        done = False
        self._visainstrument.write('INIT')
        tz=time.time()
        qt.msleep(0.1)
        self.visa_ask('*OPC?')

        print('time of this sweep = %ss'%(time.time()-tz))

    def download_trace(self):
        '''
        retrieves the result of a SINGLE sweep
        returns an array of [fvals, P(fvals)]
        '''
        qt.mstart()
        trace=self.visa_ask('TRAC? TRACE1')
        freq_diff=(self.fstop-self.fstart)/self.Npoints
        #print trace
        list_yvalues=trace.rsplit(',')
        xvals=[]
        yvalues = []
        k=0

        for k in range(int(self.Npoints)):
            xvalue = self.fstart+k*freq_diff
            xvals += [xvalue]
            qt.msleep()

            yvalues+=[float(list_yvalues[k])]
            k+=1
        qt.mend()
        return [xvals,yvalues]

    def get_fstop(self):
        return self.fstop


    def acquire_S21_spectrum(self,fstart,fstop,npoints,tint,power,**kw):
        '''
        performs a single transmission (S21) measurement sweep
        fstart = start freq. GHz
        fstop = stop freq. GHz
        npoints = number of points
        tint = integration time per point (set as 1/RESBW) in ms
        power = source power dBm

        known kw
        - averages
        '''

        self._visainstrument.write('SOUR:EXT%s:STATE ON'%(self.ext_gen))


        print('ESTIMATED SWEEP TIME ~ %ss'%(npoints*tint*1e-3+26)) # this does not yet work in the sense that it is inaccurate

        self.prepare_sweep(fstart,fstop,npoints,tint,power,kw.pop('averages',1))
        self.start_single_sweep()
        s21=self.download_trace()
        return s21



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
