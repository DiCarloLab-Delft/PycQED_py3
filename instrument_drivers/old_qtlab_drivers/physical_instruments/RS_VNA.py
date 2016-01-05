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
import numpy as np
import visa
import types
import logging
import qt
from time import sleep
import time
import sys

class RS_VNA(Instrument):
    '''
    This is the python driver for the Rohde & Schwarz SMR40
    signal generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RS_SMR40', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the RS_SMR40, and communicates with the wrapper.

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

        self.add_parameter('S_parameter', type=bytes,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('sweep_time', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=0, units='s',
                tags=['sweep'])
        self.add_parameter('bandwidth', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=0, units='Hz',
                tags=['sweep'])
        self.add_parameter('start_frequency', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=1e-8, units='GHz',
                tags=['sweep'])
        self.add_parameter('stop_frequency', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=1e-8, units='GHz',
                tags=['sweep'])
        self.add_parameter('npoints', type=int,
            flags=Instrument.FLAG_GETSET,
                minval=101, units='points',
                tags=['sweep'])
        self.add_parameter('averages', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=1, units='x',
                tags=['sweep'])
        self.add_parameter('power', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=-20, units='dBm',
                tags=['sweep'])
        self.add_parameter('format', type=bytes,
            flags=Instrument.FLAG_GETSET,
                tags=['sweep'])


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

        funlist = ['reset',
                    'get_reference_level',
                    'set_reference_level',
                    'prepare_sweep',
                    'start_single_sweep',
                    'download_trace']
        for fun in funlist:
            self.add_function(fun)
        self.visa_ask('CORR:STATE?')
        # self.add_function('get_all')
        self.ext_gen = 1
        if reset:
            self.reset()
        # else:
            # self.get_all()

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
        self._visainstrument.write('*CLS')
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
    def do_get_bandwidth(self):
        self.resBW = float(self.visa_ask('BAND:res?'))
        return self.resBW
    def do_set_bandwidth(self,bw):
        self.resbw = bw
        #print 'res_BW = %sHz'%BW
        self._visainstrument.write('sens:band:res %sHz'%bw)
    def do_set_start_frequency(self,fstart):
        self.fstart = fstart
        self._visainstrument.write('FREQ:STAR %sGHz'%fstart)
    def do_get_start_frequency(self):
        self.fstart = float(self.visa_ask('FREQ:STAR?'))/1.e9
        return self.fstart
    def do_set_stop_frequency(self,fstop):
        self.fstop=fstop
        self._visainstrument.write('FREQ:STOP %sGHz'%fstop)
    def do_get_stop_frequency(self):
        self.fstop = float(self.visa_ask('FREQ:STOP?'))/1.e9
        return self.fstop
    def do_set_npoints(self, npoints):
        self.Npoints=npoints
        self._visainstrument.write('SWE:POIN %s'%npoints)
    def do_get_npoints(self):
        self.Npoints = self.visa_ask('SWE:POIN?')
        return self.Npoints
    def do_set_sweep_time(self, time):

        self._visainstrument.write('SWE:TIME %ss'%time)
    def do_get_sweep_time(self):
        return float(self.visa_ask('SWE:TIME?'))

    def do_set_power(self,power):
        self._visainstrument.write('SOUR:POW %sdBm'%(power))
    def do_get_power(self):
        self.power = float(self.visa_ask('SOUR:POW?'))
        return self.power
    def do_set_format(self,form):
        '''
        input: COMP, MAGN, PHAS, REAL, IMAG, SWR
        '''
        self.visa_write('CALC:FORM %s'%form)
    def do_get_format(self):
        '''
        '''
        return self.visa_ask('CALC:FORM?')
    def do_set_S_parameter(self,mtype):
        '''
        S11, S21, S22, S12
        '''
        self.visa_write('FUNC "XFR:POW:%s"'%mtype)
    def do_get_S_parameter(self):
        '''
        S11, S21, S22, S12
        '''
        return self.visa_ask('FUNC?')[-4:-1]


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
        self.set_averages(averages)
        #self.set_averaging_type('POIN')
        self.set_power(power)

        self._visainstrument.write('INIT:CONT %s' %cont_sweep)

    def do_set_averages(self,no_avg):
        self._visainstrument.write('SWE:TIME:AUTO')
        self._visainstrument.write('SWE:COUN 1')
        if no_avg > 1:
            self._visainstrument.write('AVER ON')
            self._visainstrument.write('AVER:COUN %s' %no_avg)
        else:
            self._visainstrument.write('AVER OFF')
            self._visainstrument.write('AVER:COUN %s' %no_avg)
    def do_get_averages(self):
        return float(self.visa_ask('AVER:COUN?'))

    def get_averaging_type(self):
        return self.visa_ask('AVER:MODE?')
    def set_averaging_type(self, avgtype):
        '''
        avgtype = "POIN" or "SWE"
        FROM ZVM manual:
        Averaging is suitable for elimination of very low-frequency interferences (< 1 Hz),
        whereas the IF filter should be used for reliable suppression of noise signals with larger offset from the
        measurement frequency.
        For both types of filter (IF filter and averaging), the measuring time of a point increases inversely
        proportional to the effective bandwidth due to the required settling time.
        '''
        self._visainstrument.write('AVER:MODE %s'%avgtype)

    def continue_meas(self):
        self._visainstrument.write('INIT2;*CONT')
    def start_sweep(self):
        self._visainstrument.write('INIT')
        sys.stdout.write('started.')
    def reset_sweep(self):
        '''
        stops the current sweep
        '''
        self._visainstrument.write('ABORT')
    def visa_read(self):
        return self._visainstrument.read()
    def visa_write(self, cmd):
        self._visainstrument.write(cmd)
    def visa_ask(self,cmd):
        syms = ['\\', '|', '/', '-']
        qt.mstart()
        self._visainstrument.write(cmd)
        done = False
        kk = 0
        sys.stdout.write(' [\]')
        while not done:
            qt.msleep()
            try:
                mes = self._visainstrument.read()
                done = True
                sys.stdout.write('\b\b\b\b')
            except:
                done = False
                sys.stdout.write('\b\b\b[{0}]'.format(syms[kk]))
                sys.stdout.flush()
                kk =  (kk+1)%len(syms)

        qt.mend()
        return mes

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
        sys.stdout.write('started.')
        qt.msleep(0.1)
        self.visa_ask('*OPC?')
        print('\tFinished.')
        print('time of this sweep = %ss'%(time.time()-tz))

    def download_trace(self):
        '''
        retrieves the result of a SINGLE sweep
        returns an array of [fvals, P(fvals)]
        '''
        qt.mstart()


        dform =  self.visa_ask('CALC:FORM?')
        stim = np.array(self.visa_ask('TRAC:STIM? CH1DATA').rsplit(','), dtype='float')/1e9
        self.set_start_frequency(stim[0])
        self.set_stop_frequency(stim[-1])
        self.set_npoints(len(stim))
        self.set_format(dform)
        yvals = np.array(self.visa_ask('TRAC? CH1DATA').rsplit(','), dtype='float')
        uncal = self.visa_ask('CORR:STATE?') == '0'
        if (dform == 'COMP'):

            yvalsRe = yvals[::2]
            yvalsIm = yvals[1::2]
            #return [stim,yvalsRe,yvalsIm]
            return [stim,yvalsRe + 1.j*yvalsIm]
        elif (dform == 'MAGN') and uncal:

            print(len(yvals))
            #return [stim,yvalsRe,yvalsIm]

            return [stim,yvals]
        else:
            return [stim,10**(yvals/20)]

    def get_fstop(self):
        return self.fstop



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
