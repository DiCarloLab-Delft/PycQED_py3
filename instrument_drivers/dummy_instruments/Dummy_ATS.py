#from _AlazarTech_ATS9870.errors import errors as _ats_errors
import qt
from ctypes import *
import _Alazar_ATS9870.AlazarCmd as ATS_cmd
from instrument import Instrument
import pickle
from time import clock as time
from time import sleep
import types
import logging
from numpy import *
import numpy as np
import sys
import numpy
import ctypes
import matplotlib.pyplot as plt

Plotmon = qt.instruments['Plotmon']


class Dummy_ATS(Instrument):
    '''
    This is the driver for the Alazar ATS9870 data acquisition card

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Alazar_ATS9870')

    TODO:
    1) add error handling AlazarError.h
    2) add aparameter trace_length, number of buffers

    '''

    def __init__(self, name, mem_size = 2048, ATS_buffer = []):
        '''
        Initializes the data acquisition card, and communicates with the
        wrapper.

        Usage:
            Use in a simple measurementloop as:
            <name>.init_default(memsize, posttrigger, amp)

            And repeat:
            <name>.start_with_trigger_and_waitready()
            <name>.readout_singlemode_float()

        Input:
            name (string) : name of the instrument

        Output:
            None
        '''
        # Initialize wrapper
        logging.info(__name__ + ' : Initializing instrument Alazar')
        Instrument.__init__(self, name, tags=['physical'])

        # Load dll and open connection
        
        
        
        # # add system parameters
        # self.add_parameter('memory_size', flags=Instrument.FLAG_GETSET, 
        #         type=types.IntType, units = 'MB', minval=10, maxval = 4096)
        # # add acquisition parameters
        self.add_parameter('records_per_buffer', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('number_of_buffers', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('trace_length', flags=Instrument.FLAG_GET, 
                units = 'ms', type=float)
        self.add_parameter('points_per_trace', flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, 
                type=int)
        # self.add_parameter('busy', flags=Instrument.FLAG_GET, 
        #         type=types.BooleanType)
        # self.add_parameter('check_overload', flags=Instrument.FLAG_GETSET, 
        #         type=types.BooleanType)
        # self.add_parameter('overload', flags=Instrument.FLAG_GETSET, 
        #         type=types.BooleanType, channels=(1,2), channel_prefix = 'ch%d_')
        # self.add_parameter('LED', flags=Instrument.FLAG_GETSET, 
        #         type=types.BooleanType)

        # # add channel parameters
        # self.add_parameter('coupling', flags=Instrument.FLAG_GETSET,
        #         type=types.StringType, channels=(1,2), channel_prefix = 'ch%d_')
        # self.add_parameter('impedance', flags=Instrument.FLAG_GETSET,
        #         units = 'Ohm',
        #         type=types.IntType, channels=(1,2), channel_prefix = 'ch%d_')
        # self.add_parameter('bandwidth_limit', flags=Instrument.FLAG_GETSET,
        #         type=types.BooleanType, channels=(1,2), 
        #         channel_prefix = 'ch%d_')
        self.add_parameter('range', flags=Instrument.FLAG_GETSET, 
                units = 'Volt',
                type=float, channels=(1,2), channel_prefix = 'ch%d_')

        # clock and sampling parameters
        # self.add_parameter('clock_source', flags=Instrument.FLAG_GETSET,
        #         type=types.StringType)
        self.add_parameter('sample_rate', flags=Instrument.FLAG_GETSET, 
                units = 'KSPS', type=int)
        # self.add_parameter('clock_edge_rising', flags=Instrument.FLAG_GETSET, 
        #         type=types.BooleanType)

        # add trigger parameters
        # self.add_parameter('trigger_oper', flags=Instrument.FLAG_GETSET, 
        #         type=types.StringType)
        # self.add_parameter('trigger_slope', flags=Instrument.FLAG_GETSET,
        #         type=types.StringType, channels=(1,2), 
        #         channel_prefix = 'eng%d_')
        # self.add_parameter('trigger_source', flags=Instrument.FLAG_GETSET,
        #         type=types.StringType, channels=(1,2), 
        #         channel_prefix = 'eng%d_')        
        # self.add_parameter('trigger_level', flags=Instrument.FLAG_GETSET,
        #         type=types.IntType, channels=(1,2), channel_prefix = 'eng%d_')
        # self.add_parameter('ext_trigger_delay', flags=Instrument.FLAG_GETSET, 
        #         units='sec', type=types.FloatType)
        # self.add_parameter('ext_trigger_timeout', flags=Instrument.FLAG_GETSET, 
        #         units='sec', type=types.FloatType)
        # self.add_parameter('ext_trigger_coupling', flags=Instrument.FLAG_GETSET, 
        #         type=types.StringType)
        
        # self.add_parameter('ext_trigger_range', flags=Instrument.FLAG_GETSET, 
        #         type=types.IntType, unit = 'Volt')
    
        
        self._mode = 'NPT' # (NoPreTrigger traditional NOT IMPLEMENTED)
        self._processing_mode = 'averaging'
        self._channel_count = 2
        self._ro_timeout = 5000 # timeout for buffer filling complete (ms)
        self.timeout = 10
        self.max_points_per_trace= 1e7
        self.set_sample_rate(100e3)
        self.set_number_of_buffers(32)
        self._range = [1,1]
        self.set_ch1_range(1.)
        self.set_ch2_range(1.)
        self.set_records_per_buffer(2)
        self.set_points_per_trace(1e4)
        self.set_monitor('mpl')
        #self.set_check_overload(True)    
        # self._pre_alloc_memory = mem_size # reserve memoryblock of N*1e6 8 byte elements
        # self._mem_offset = 0L
        # self._load_defaults()
        # self._buffer_retrieved = 0
    def mpl_monitor(self, tbase, dat1, dat2, mode = 'average', clf = True, **kw):
        
        
        fig=plt.figure('ATS_monitor', figsize = (8.5,4))
        if clf:
            plt.clf()
        ax1 = fig.add_subplot(121)
        ax1.set_ylabel('Amplitude (V)')
        
        vstime = kw.pop('vstime', False)
        if vstime:
            ax1.plot(tbase, dat1, label = 'channel1')
            ax1.set_xlabel(r'time ($\mu \mathrm{s}$)')

        else:
            ax1.plot(list(range(self._length)), dat1, label = 'channel1')
            ax1.set_xlabel('N')
        plt.legend(fontsize = 'small')
        
        ax2 = fig.add_subplot(122)
        if vstime:
            ax2.plot(tbase, dat2, label = 'channel1')
            ax2.set_xlabel(r'time ($\mu \mathrm{s}$)')
        else:
            ax2.plot(list(range(self._length)), dat2, label = 'channel1')
            ax2.set_xlabel('N')
        plt.ylabel('Amplitude (V)')
        plt.legend(fontsize = 'small')
        
        plt.tight_layout()
    def LV_monitor(self, tbase, dat1, dat2, **kw):
        Plotmon.plot2D(3, [tbase, dat1])
        Plotmon.plot2D(4, [tbase, dat2])

    def set_monitor(self, mon='lv'):
        if mon == 'LV':
            self.plot_traces = self.LV_monitor
        else:
            self.plot_traces = self.mpl_monitor


    def monitor(self, mode = 'average', **kw):
        if mode == 'average':
            dat1 = self.average_data(1)[0]
            dat2 = self.average_data(2)[0]
        else:
            dat1 = self.get_data()[0][0,0,:]
            dat2 = self.get_data()[1][0,0,:]
        tbase = np.arange(self.get_points_per_trace(), dtype= np.float)/self.get_sample_rate()*1.e3
        self.plot_traces(tbase, dat1, dat2, **kw)

    def scope(self, **kw):
        self.monitor(mode = 'not average',**kw)
    def scope_average(self, **kw):
        self.monitor(mode = 'average',**kw)

    def do_set_points_per_trace(self,N):
        '''
        length of a single record (number of points)
        '''
        self._length = self.trunc(int(N),64)
        self.get_points_per_trace()
    def do_get_trace_length(self):
        '''
        gets the trace length in ms
        '''
        return 1.*self.get_points_per_trace()/self.get_sample_rate()

    def do_get_points_per_trace(self):
        '''
        gets the trace length in number of samples
        '''
        return self._length
    def do_set_records_per_buffer(self, n):
        '''
        sets the number of records per buffer
        '''
        self._rec_per_buf = n    
    def do_get_records_per_buffer(self):
        return self._rec_per_buf

    def do_set_number_of_buffers(self, n):
        ''' 
        Sets the number of buffers
        '''
        self._n_buf = n
    def do_get_number_of_buffers(self):
        return self._n_buf
    def do_set_sample_rate(self, rate):
        '''
        Sets the sample rate
        rate = in units of KSPS
        '''
        self._sample_rate = ATS_cmd.sample_rates[rate]
    def do_get_sample_rate(self):
        '''
        retrieves sampling rate, returns int units KSPS
        '''
        rate = self._get_key_from_dict_value(
                self._sample_rate, 
                ATS_cmd.sample_rates)
        return rate

    def do_set_range(self, range, channel):
        '''
        range : range in V
        ch : channel 1,2
        '''
        
        rngl = ATS_cmd.ranges[range]
        self._range[channel-1] = rngl

    def do_get_range(self, channel):
        '''
        gets the range 9in V) of channel channel(1,2)
        '''
               
        val = 1.*self._get_key_from_dict_value(
                self._range[channel-1], 
                ATS_cmd.ranges)
        return val
    def start_acquisition(self): # keep this function for compatibility
        '''
        Use this to start the data acquisition
        '''
        #self.abort()
        #self.configure_board()
        pass
    def get_data(self):
        noise = array(10*np.random.randn(2*self._n_buf*self._rec_per_buf, self._length), dtype = np.int8)
        sig = array(10*np.cos(2*pi*np.arange(self._length)/100),dtype=np.int8)
        dat = noise
        dat.shape = (2,self._n_buf, self._rec_per_buf, self._length)
        dat +=sig
        return [dat[0], dat[1]]

    def average_data(self, channel):
        exec('ran = self.get_ch%s_range()'%channel)
        dat = (np.average(self.get_data()[channel-1],0)/128.)*ran

        return dat
        

    def trunc(self, value, truncsize):
        return int(value/truncsize)*truncsize
    def _get_key_from_dict_value(self, value, dct):
        '''
        gets the key belonging to a value in a dictionary 
        (a quantum computer and grover would be nice...)
        dct : a dictionary
        value : some value that needs to be in the dictionary
        or program will raise an error
        '''
        keys = list(dct.keys())
        k=0
        while dct[keys[k]] != value:
            k+=1
        return keys[k]