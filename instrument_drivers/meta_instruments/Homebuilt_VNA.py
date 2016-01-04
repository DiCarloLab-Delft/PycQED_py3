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
HM = qt.instruments['HM']
RF = qt.instruments['S1'] # source used for homodyne
class Homebuilt_VNA(Instrument):
    '''
    This is the python driver for the homebuilt VNA,
    consisting of an RF source and an AWG

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Homebuilt_VNA', address='<GPIB address>',
        reset=<bool>)
    '''
    
    def __init__(self, name, reset=False):
        '''
        Initializes the RS_SMR40, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        print(' test')
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        
        
        

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
                minval=1, units='points',
                tags=['sweep'])
        self.add_parameter('averages', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=1, units='x',
                tags=['sweep'])
        self.add_parameter('power', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=20, units='dBm',
                tags=['sweep'])
        self.add_parameter('format', type=bytes,
            flags=Instrument.FLAG_GETSET,
                tags=['sweep'])
        
        self._format = 'COMP'
        #self._power = 20
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
                    'start_single_sweep',
                    'download_trace']
        for fun in funlist:
            self.add_function(fun)
        
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
        self.resBW = 1./HM.get_t_int()*1e3
        return self.resBW
    def do_set_bandwidth(self,bw):
        self.resbw = bw
        HM.set_t_int(1./bw*1e3)
        #print 'res_BW = %sHz'%BW
        
    def do_set_start_frequency(self,fstart):
        self._fstart = fstart
        
    def do_get_start_frequency(self):
        return self._fstart
    def do_set_stop_frequency(self,fstop):
        self._fstop = fstop
        
    def do_get_stop_frequency(self):
        return self._fstop
    def do_set_npoints(self, npoints):
        self._npoints = npoints
        
    def do_get_npoints(self):
        return self._npoints
    
    def do_set_averages(self,no_avg):
        self._navg = no_avg
    def do_get_averages(self,no_avg):
        return self._navg
    def do_set_power(self,power):
        self._power = RF.set_power(power)
    def do_get_power(self):
        self._power = RF.get_power()
        return self._power
    def do_set_format(self,form):
        '''
        input: COMP, MAGN, PHAS
        '''
        self._format = form
    def do_get_format(self):
        '''
        '''
        return self._format
    def set_measurement(self,mtype):
        '''
        S21, S11, S22, S12
        '''
        self._type = 'S21'
        if mtype != 'S21':
            print('Warning, your custom VNA can only measure S21!!!')
        #self.visa_write('XPOW:POW:%s'%mtype)
        #self.visa_write('XTIM:POW:%s'%mtype)
        
    
    # utility functions
    def prepare_sweep(self,fstart,fstop,npoints,bw,power,averages, cont_sweep='OFF'):
        '''
        Prepares the FSP for doing a sweep
        npoints = number of points in sweep
        cont_sweep = ON means sweep repeats indefinitely
        '''
        self.set_start_frequency(fstart)
        self.set_stop_frequency(fstop)
        self.set_npoints(int(npoints))        
        self.set_bandwidth(bw)
        self.set_power(power)
        self.set_averages(averages)
        self._farray = np.linspace(self._fstart, self._fstop, self._npoints)
        RF.set_frequency(fstart*1e9)
        RF.on()

        
    
    
    def reset_sweep(self):
        '''
        stops the current sweep
        '''
        qt.mend()
    def visa_read(self):
        return self._visainstrument.read()
    
        
    def start_single_sweep(self):
        '''
        Starts a single sweep and records the result in Trace1
        input: N = number of points
        returns nothing, command is returned to the console after the sweep is finished
        '''
        qt.mstart()
        tz = time.time()
        #measurement_loop
        if self._format == 'COMP':
            dtype = np.complex128
        else:
            dtype = np.float

        self._data = np.zeros(len(self._farray),dtype = dtype)
        
        for kk,f in enumerate(self._farray):
            RF.set_frequency(f*1e9)
            qt.msleep(0.001)
            self._data[kk] = HM.probe(self._navg, mtype = self._format)
            if kk/50==kk/50.:
                HM.plot2D(1, [self._farray[:kk], np.abs(self._data[:kk])])
       
        print('time of this sweep = %ss'%(time.time()-tz))
   
    def download_trace(self):
        '''
        retrieves the result of a SINGLE sweep
        returns an array of [fvals, P(fvals)]
        '''
        
        return [self._farray, self._data]
   
    def off(self):
        '''
        Set status to 'off'

        Input:
            None

        Output:
            None
        '''
        RF.off()

    def on(self):
        '''
        Set status to 'on'

        Input:
            None

        Output:
            None
        '''
        RF.on()
