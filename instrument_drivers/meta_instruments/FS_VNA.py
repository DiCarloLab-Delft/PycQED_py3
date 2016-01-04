
from instrument import Instrument
import numpy as np
import cmath
import visa
import types
import logging
import qt
from time import sleep
import time
import sys

RF = qt.instruments['RF'] # source used for homodyne
LO = qt.instruments['LO']
ATS = qt.instruments['ATS']
ATS_CW = qt.instruments['ATS_CW']
Plotmon = qt.instruments['Plotmon']

class FS_VNA(Instrument):
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
        self.add_parameter('IF_frequency', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=1, units='MHz',
                tags=['sweep'])
        self.add_parameter('averages', type=int,
            flags=Instrument.FLAG_GETSET,
                minval=1, units='x',
                tags=['sweep'])
        self.add_parameter('power', type=float,
            flags=Instrument.FLAG_GETSET,
                minval=-120, units='dBm',
                tags=['sweep'])
        self.add_parameter('format', type=bytes,
                flags=Instrument.FLAG_GETSET,
                tags=['sweep'])
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
        self.do_set_IF_frequency(4)
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
        self.resBW = 1./ATS_CW.get_t_int()*1e3
        return self.resBW
    def do_set_bandwidth(self,bw):
        self.resbw = bw
        t_int = 1./bw*1e3 
        print('VNA setting t_int to:', t_int)
        ATS_CW.set_t_int(t_int, silent=True)
        #print 'res_BW = %sHz'%BW
        
    def do_set_start_frequency(self,fstart):
        self._fstart = fstart
        
    def do_get_start_frequency(self):
        return self._fstart
    def do_set_IF_frequency(self,IF):
        self._IF = IF
        
    def do_get_IF_frequency(self):
        return self._IF
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

    def do_get_format(self):
        return self._format

    def do_set_format(self,form):
        '''
        input: COMP, MAGN, PHAS
        '''
        self._format = form

    def set_measurement(self,mtype):
        '''
        S21, S11, S22, S12
        '''
        self._type = 'S21'
        if mtype != 'S21':
            print('Warning, your custom VNA can only measure S21!!!')
        #self.visa_write('XPOW:POW:%s'%mtype)
        #self.visa_write('XTIM:POW:%s'%mtype)
        
    def set_format(self,type):
        self._format = type

    def get_format(self):
        return self._format

    # utility functions
    def prepare_sweep(self,fstart,fstop,npoints,bw,power,averages = 1, cont_sweep='OFF'):
        '''
        Prepares the FSP for doing a sweep
        npoints = number of points in sweep
        cont_sweep = ON means sweep repeats indefinitely
        
        TODO:
            Include cont_sweep
        '''
        self.set_start_frequency(fstart)
        self.set_stop_frequency(fstop)
        self.set_npoints(int(npoints))   
        ATS_CW.set_coupling('AC')     
        self.set_bandwidth(bw)
        self.set_power(power)
        self.set_averages(averages)
        self._farray = np.linspace(self._fstart, self._fstop, self._npoints)
        RF.set_frequency(fstart*1e9)
        RF.on()
        LO.set_power(23)
        LO.on()
        ATS.set_ch2_range(0.4)
        ATS.set_ch1_range(0.1)
        ATS.configure_board()

        
    
    
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
        npoints = ATS.get_points_per_trace()
        IF = self._IF*1e6
        sr = ATS.get_sample_rate()*1e3
        self._Idat = np.zeros(len(self._farray))
        self._Qdat = np.zeros(len(self._farray))
        self._data = np.zeros(len(self._farray), dtype = complex)
        self._refphase = np.zeros(len(self._farray))
        cosI = np.cos(2*np.pi*IF*np.arange(npoints)/sr)
        sinI = np.sin(2*np.pi*IF*np.arange(npoints)/sr)

        for kk,fval in enumerate(self._farray):
            RF.set_frequency(fval*1e9)
            LO.set_frequency(fval*1e9+IF)
            qt.msleep(0.001)
            ATS.start_acquisition()
            s21dat = ATS.average_data(1)
            phidat = ATS.average_data(2)

            phi = cmath.phase(np.average(phidat*cosI) +1.j* np.average(phidat*sinI))
            self._refphase[kk] = phi
            for aa in range(self._navg):
                Idat = np.average((np.cos(phi)*cosI - np.sin(-phi)*sinI)*s21dat)/(self._navg)
                Qdat = np.average((np.sin(-phi)*cosI + np.cos(phi)*sinI)*s21dat)/(self._navg)

                self._Idat[kk] += Idat
                self._Qdat[kk] += Qdat
                self._data[kk] += Idat + 1.j * Qdat
            self._refphase[kk] = cmath.phase(self._Idat[kk] + 1.j*self._Qdat[kk])
            
            if kk/10==kk/10.:
                Plotmon.plot2D(1,[self._Idat[:kk],self._Qdat[:kk]])#[self._farray[:kk], self._Idat[:kk]])
                Plotmon.plot2D(2,[self._farray[:kk], self._Qdat[:kk]])
                Plotmon.plot2D(3,[self._farray[:kk], np.abs(self._Idat[:kk] + 1.j*self._Qdat[:kk])])
                Plotmon.plot2D(4,[self._farray[:kk], self._refphase[:kk]])
                
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