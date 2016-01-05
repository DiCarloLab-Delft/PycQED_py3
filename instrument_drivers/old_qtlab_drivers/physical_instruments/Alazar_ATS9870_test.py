
# Driver class for the AlazarTech ATS9870 Digitizer (DAQ) 
# PCI Express board, 2 channels, 8bits, 1GS/s
# Gijs de Lange, 2012
# 
# Functions exported by the DLL are in C format ("CapitalsInFunctionNames()")
# User defined functions in pythion format ("underscores_in_function_names()")
# 



#from _AlazarTech_ATS9870.errors import errors as _ats_errors
import qt
from ctypes import *
from . import _Alazar_ATS9870.AlazarCmd as ATS_cmd
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



class Alazar_ATS9870_test(Instrument):
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
        
        self._load_dll()
        self.get_channel_info()
        # add system parameters
        self.add_parameter('memory_size', flags=Instrument.FLAG_GETSET, 
                type=int, units = 'MB', minval=10, maxval = 4096)
        # add acquisition parameters
        self.add_parameter('records_per_buffer', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('number_of_buffers', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('trace_length', flags=Instrument.FLAG_GET, 
                units = 'ms', type=float)
        self.add_parameter('points_per_trace', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('busy', flags=Instrument.FLAG_GET, 
                type=bool)
        self.add_parameter('check_overload', flags=Instrument.FLAG_GETSET, 
                type=bool)
        self.add_parameter('overload', flags=Instrument.FLAG_GETSET, 
                type=bool, channels=(1,2), channel_prefix = 'ch%d_')
        self.add_parameter('LED', flags=Instrument.FLAG_GETSET, 
                type=bool)

        # add channel parameters
        self.add_parameter('coupling', flags=Instrument.FLAG_GETSET,
                type=bytes, channels=(1,2), channel_prefix = 'ch%d_')
        self.add_parameter('impedance', flags=Instrument.FLAG_GETSET,
                units = 'Ohm',
                type=int, channels=(1,2), channel_prefix = 'ch%d_')
        self.add_parameter('bandwidth_limit', flags=Instrument.FLAG_GETSET,
                type=bool, channels=(1,2), 
                channel_prefix = 'ch%d_')
        self.add_parameter('range', flags=Instrument.FLAG_GETSET, 
                units = 'Volt',
                type=float, channels=(1,2), channel_prefix = 'ch%d_')

        # clock and sampling parameters
        self.add_parameter('clock_source', flags=Instrument.FLAG_GETSET,
                type=bytes)
        self.add_parameter('sample_rate', flags=Instrument.FLAG_GETSET, 
                units = 'KSPS', type=int)
        self.add_parameter('clock_edge_rising', flags=Instrument.FLAG_GETSET, 
                type=bool)

        # add trigger parameters
        self.add_parameter('trigger_oper', flags=Instrument.FLAG_GETSET, 
                type=bytes)
        self.add_parameter('trigger_slope', flags=Instrument.FLAG_GETSET,
                type=bytes, channels=(1,2), 
                channel_prefix = 'eng%d_')
        self.add_parameter('trigger_source', flags=Instrument.FLAG_GETSET,
                type=bytes, channels=(1,2), 
                channel_prefix = 'eng%d_')        
        self.add_parameter('trigger_level', flags=Instrument.FLAG_GETSET,
                type=int, channels=(1,2), channel_prefix = 'eng%d_')
        self.add_parameter('ext_trigger_delay', flags=Instrument.FLAG_GETSET, 
                units='sec', type=float)
        self.add_parameter('ext_trigger_timeout', flags=Instrument.FLAG_GETSET, 
                units='sec', type=float)
        self.add_parameter('ext_trigger_coupling', flags=Instrument.FLAG_GETSET, 
                type=bytes)
        
        self.add_parameter('ext_trigger_range', flags=Instrument.FLAG_GETSET, 
                type=int, unit = 'Volt')
        
        self._mode = 'NPT' # (NoPreTrigger traditional NOT IMPLEMENTED)
        self._processing_mode = 'averaging'
        self._channel_count = 2
        self._ro_timeout = 5000 # timeout for buffer filling complete (ms)
        self.timeout = 10
        self.max_points_per_trace= 1e7
            
        self.set_check_overload(True)    
        self._pre_alloc_memory = mem_size # reserve memoryblock of N*1e6 8 byte elements
        self._mem_offset = 0
        self._load_defaults()
        self._buffer_retrieved = 0
        
    def get_max_ppt(self):
        '''
        maximum number of points per trace
        '''
        return self.max_points_per_trace

    def _load_defaults(self):
        '''
        Default settings are loaded here
        '''
        self._range = [None,None]
        self._coupling = [None,None]
        self._impedance = [None,None]
        self._bandwidth_limit = [None,None]
        self._trigger_slope = [None, None]
        self._trigger_source = [None, None]
        self._trigger_level = [None, None]
        
        for ch in [1,2]:
            exec('self.set_ch%d_coupling("AC")'%ch)
            exec('self.set_ch%d_range(2.)'%ch)
            exec('self.get_ch%d_range()'%ch)
            exec('self.set_ch%d_impedance(50)'%ch)
            exec('self.set_ch%d_bandwidth_limit(False)'%ch)
            self.set_trigger_oper("eng_1_only") 
            exec('self.set_eng%d_trigger_slope("pos")'%ch)
            exec('self.set_eng%d_trigger_source("external")'%ch)
            exec('self.set_eng%d_trigger_level(150)'%ch)
            #exec 'self.set_eng%d_trigger_slope("pos")'%ch
            #exec 'self.set_ch%d_

        self.set_sample_rate(1e6)
        self.set_eng2_trigger_source('chB')
        self.set_eng2_trigger_level(0)
        self.set_ext_trigger_delay(0.)
        self.set_ext_trigger_timeout(0.)
        self.set_ext_trigger_coupling('AC')
        self.set_ext_trigger_range(1)
        self.set_clock_source('internal')
        #self.set_clock_source('external_10MHz')
        self.set_sample_rate(100e3)
        self.set_clock_edge_rising(True)
        self.set_records_per_buffer(1)
        # self.set_buffers_per_acq(100) same as numer_of_buf
        self.set_number_of_buffers(10)
        self.set_points_per_trace(960) # 
        self.get_trace_length() #ms
        self.get_busy()
        self.set_LED(False)
        self.set_memory_size(self._pre_alloc_memory)
        
        
        self._fast_mode = True
    def _load_dll(self):
        '''
        Make sure the dll is located at "C:\\WINDOWS\\System32\\ATSApi"
        '''
        self._ATS9870_dll = cdll.LoadLibrary('C:\\WINDOWS\\System32\\ATSApi') 
        self._handle = self._ATS9870_dll.AlazarGetBoardBySystemID(1,1)
    def do_set_memory_size(self, msize):
        '''
        sets the size of the reserved memory size
        size : memory size (MB)
        '''
        self._memory_size = msize
        self.reserve_memory()
        
    def free_memory(self):
        self._buf_arr=numpy.array([0], dtype= numpy.int8)
    

    def do_get_memory_size(self):
        return self._memory_size
    def reserve_memory(self):
        '''
        Reserve a continuous block of memory in python. The API will store its data in this block.
        '''
        self.get_memory_size()
        self._buf_arr = numpy.zeros(self._memory_size*1024**2/4,
                dtype='int8')
        #qt.ATS_buffer = self._buf_arr
        #self._dat_ch1 = numpy.arange(self._memory_size*512*1024)
        #self._dat_ch2 = numpy.arange(self._memory_size*512*1024)
    def initialize_buffer_to_zero(self):
        self._buf_arr[:] = 0
    def do_get_busy(self):
        '''
        check if ats is still running
        '''
        state = self._ATS9870_dll.AlazarBusy(self._handle) == 1
        #self.set_LED(state)
        return state
    def do_set_check_overload(self,state):
        self._check_overload = state
    def do_get_check_overload(self):
        return self._check_overload
    def do_set_overload(self,state, channel):
        self._overload = state
    def do_get_overload(self, channel):
        return self._overload
# Acquisition

    def configure_board(self):
        '''
        writes the settings to the card
        '''
        err1 = self.update_clock_settings()
        err2 = self.update_channel_settings()
        err3 = self.update_trigger_operation()
        return err1,err2,err3

    def do_set_LED(self, state):
        '''
        switches on LED at the back of pc board
        '''
        self._led = state
        self._ATS9870_dll.AlazarSetLED(self._handle, int(state))
    def do_get_LED(self):
        return self._led

    def get_channel_info(self):
        '''
        Gets channel info such as bits per sample value 
        and max memory available
        '''
        bps = array([0],dtype = uint8)
        max_s = array([0], dtype = uint32)
        success = self._ATS9870_dll.AlazarGetChannelInfo(
                self._handle,
                max_s.ctypes.data,
                bps.ctypes.data)
        self.print_if_error(success, 'GetChannelInfo')
        self._bits_per_sample = bps[0]
        self._max_samples_per_record = max_s[0]
        self._bytes_per_sample = (bps[0]+7)/8

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

#####################  
## acquisition
#####################
    def start_acquisition(self): # keep this function for compatibility
        '''
        Use this to start the data acquisition
        '''
        #self.abort()
        #self.configure_board()
        self.arm()   

    def arm(self):
        '''
        arms the ats for a measurement
        after which it is waiting for a trigger
        '''
        self.set_data_format_to_signed()
        self.allocate_memory_for_DMA_buffers()
        self.start_capture()

    def allocate_memory_for_DMA_buffers(self):
        '''
        Prepares for the read out
        - allocates memory
        - prepares the board for an aquisition
        - passes a list of buffer pointers to the board 
        '''
        self._buffer_retrieved = 0 
        self._bytes_per_rec = self._bytes_per_sample*self._length
        self._bytes_per_buf = self._bytes_per_rec*\
                self._rec_per_buf*\
                self._channel_count
        #print self._bytes_per_buf
        self._smp_per_buf = self._length*self._rec_per_buf*self._channel_count
        
        # self._buf_arr = numpy.zeros([self._n_buf,self._smp_per_buf], 
        #         dtype = numpy.int8)
        self.set_record_size()
        dma_flag = (0x001 | 0x200 ) # ADMA_EXTERNAL_STARTCAPTURE | ADMA_NPT

        success = self._ATS9870_dll.AlazarBeforeAsyncRead(
                self._handle,
                3, # always use both channels
                0, # always use 0 pre trigger samples
                self._length,
                self._rec_per_buf,
                self._rec_per_buf*self._n_buf,
                dma_flag) 
        self.print_if_error(success, 'BeforeAsyncRead')
        #qt.msleep(0.0)
        # now send a list of pointers to buffers in the list of buffers
        t0 =time() 
    
        bpb = self._bytes_per_buf
        #print type(bpb)
        #print 'bpb: ',int(bpb)
        #print 'memaddr: ',self._buf_arr[0*2*self._rec_per_buf*self._length:].ctypes.data
        for k in range(self._n_buf):
            start_index = k*2*self._rec_per_buf*self._length
            #print 'buffer to card: ',self._buf_arr[start_index:].ctypes.data
            #print 'allocate'
            #print 'buf nr : %s'%k
            #print 'pointer : %s'%self._buf_arr[k*2*self._length:].ctypes.data
            success = self._ATS9870_dll.AlazarPostAsyncBuffer(
                    self._handle, 
                    ctypes.c_void_p(self._buf_arr[start_index:].ctypes.data), #the +3 is to set the lsbyte of the buffer, valid for sys.byteorder => little endian
                    bpb)          # of buffers 
            self.print_if_error(success, 'PostAsyncBuffer') 
            if success != 512:
                print('already armed, aborting...')
                self.abort()
                qt.msleep()
                self.allocate_memory_for_DMA_buffers()
                break
            #print time()-t0
        
    def set_data_format_to_signed(self):
        self._ATS9870_dll.AlazarSetParameter(
                self._handle,
                c_uint8(0),
                0x10000041,
                c_long(1))

    def start_capture(self):
        '''
        After this command the acquisition starts as soon 
        as the trigger condition is met
        '''
        
        success = self._ATS9870_dll.AlazarStartCapture(self._handle) 
        self.print_if_error(success, 'StartCapture')
        self.retrieve_data_while_acquiring()
        

    def retrieve_data_while_acquiring(self):
        '''
        retrieves the next buffer from the card memory.
        This allows to read out buffers while acquisition takes place.
        '''

        if self._processing_mode == 'averaging':
            #sys.stdout.write('integrating...')
            self._averaged_data = np.zeros(2*self._rec_per_buf*self._length)
        self._buffers_completed = 0
        #sys.stdout.write('acquiring')
        while self._buffers_completed < self._n_buf:
            loopcnt = 0
            start_index = self._buffers_completed*2*self._rec_per_buf*self._length
            stop_index = (self._buffers_completed+1)*2*self._rec_per_buf*self._length
            success = 0
            while success != 512:
                success = self._ATS9870_dll.AlazarWaitAsyncBufferComplete(
                        self._handle,
                        ctypes.c_void_p(self._buf_arr[start_index:].ctypes.data),
                        c_int(self._ro_timeout))
                if loopcnt >1:
                    sys.stdout.write('waiting_AWABC...')
                loopcnt+=1    
                qt.msleep()
            #print '\nbuffer filled: ',self._buf_arr[start_index:].ctypes.data
            if self._processing_mode == 'averaging':
                self._averaged_data += (self._buf_arr[start_index:stop_index]/128.)/self._n_buf
                
                if self._check_overload and self._buffers_completed==0:
                    ch1startindex = start_index
                    ch2startindex = start_index+self._rec_per_buf*self._length
                    samplech1 = self._buf_arr[ch1startindex:ch1startindex+self._length]
                    samplech2 = self._buf_arr[ch2startindex:ch2startindex+self._length]
                    if (np.max(samplech1) > 126) or (np.min(samplech1) < -126):
                            self.set_ch1_overload(True)
                    else:
                        self.set_ch1_overload(False)
                    if (np.max(samplech2) > 126) or (np.min(samplech2) < -126):
                        self.set_ch2_overload(True)
                    else:
                        self.set_ch2_overload(False)
            # success = self._ATS9870_dll.AlazarPostAsyncBuffer(
            #         self._handle, 
            #         ctypes.c_void_p(self._buf_arr[start_index:].ctypes.data), #the +3 is to set the lsbyte of the buffer, valid for sys.byteorder => little endian
            #         self._bytes_per_buf)
            # print self._buffers_completed
            # print 'buffer cleared: ',self._buf_arr[start_index:].ctypes.data
            # print ''
            self._buffers_completed += 1
        self.abort()
        return success
    
    

    def set_ro_timeout(self, to):
        '''
        Time for API to wait for a completed buffer before a time out error is given
        '''
        self._ro_timeout=to
    def get_ro_timout(self):
        return self._ro_timeout
         
    
        
        #return self.get_data()
    
    def get_buffer(self):
        return self._buf_arr
    def get_status(self):
        return self._ATS9870_dll.AlazarGetStatus(self._handle)
    def get_data(self):
        kk=1
        qt.msleep(0.005)
        while self.get_busy():  
            qt.msleep(0.005)
            if kk/50 == kk/50.:
                sys.stdout.write('.')
            kk+=1
        qt.msleep()
        self.abort()
        if self._fast_mode:
            t0 = time()
            
            #dat.reshape(ATS.get_number_of_buffers(),ATS.get_records_per_buffer(),2,ATS.get_points_per_trace())
            dat = np.array(self._buf_arr[:self._n_buf*self._rec_per_buf*self._length*2])  # 50 ms
            t1=time()
            print('', t1-t0)
            dat.shape = (self._n_buf,2,self._rec_per_buf,self._length)
            
            t2=time()
            print(' actual getting of data: ', t2-t1)
            #data = [datch1,datch2]
            t3=time()
            #print t3-t2
            #print time()-t0
            
        else:
            data = self.get_data_old()
        if self._check_overload:
            if (np.max(dat[0,0,0,:]) > 126) or (np.min(dat[0,0,0,:]) < -126):
                    self.set_ch1_overload(True)
            else:
                self.set_ch1_overload(False)
            if (np.max(dat[0,1,0,:]) > 126) or (np.min(dat[0,1,0,:]) < -126):
                self.set_ch2_overload(True)
            else:
                self.set_ch2_overload(False)
        return [dat[:,0,:,:],dat[:,1,:,:]]
    
    def average_data(self, channel):
        chrange = self._get_key_from_dict_value(
                    self._range[channel-1], 
                    ATS_cmd.ranges)
        if self._processing_mode == 'averaging':

            start_index = (channel-1)*self._rec_per_buf*self._length
            stop_index = channel*self._rec_per_buf*self._length
            print(start_index, stop_index)
            dat = self._averaged_data[start_index:stop_index]
            #print (self._rec_per_buf,self._length), len(dat)
            dat.shape = (self._rec_per_buf,self._length)
            sdat = dat*chrange
        else:
            dat = self.get_data()
            t0=time()
            
            
            sdat = (np.average(dat[channel-1],0)/128.)*chrange
        return sdat
        
   
    
        
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

    def set_record_size(self, pre = 0):
        '''
        define relative to trigger
        pre = in NPT mode this is 0 start capture 
        after pre number of clock cycles after the trigger
        length = end capture after pre + length number of 
        clockcycles after trigger
        '''
        suc6 = self._ATS9870_dll.AlazarSetRecordSize(self._handle,
                pre, pre+self._length)
        self.print_if_error(suc6, 'SetRecordSize')

# Trigger settings
    def do_set_trigger_oper(self, oper):
        '''
        sets trigger operation mode
        oper = 'eng_1_only', 'eng_2_only',
                'eng_1_OR_2', 'eng_1_AND_2',
                'eng_1_XOR_2', 'eng_1_AND_NOT_2'
        '''
        self._trigger_oper = ATS_cmd.trigger_oper[oper]
    def do_get_trigger_oper(self):
        oper = self._get_key_from_dict_value(
                self._trigger_oper,
                ATS_cmd.trigger_oper)
        return self._trigger_oper

    def do_set_trigger_source(self, src, channel):
        '''
        sets the trigger source for engine
        src : 'external', 'chA', chB, disable
        eng : 1 for engine J and 2 for engine K
        '''
        self._trigger_source[channel-1] = ATS_cmd.trigger_sources[src]
    def do_get_trigger_source(self, channel):
        src = self._get_key_from_dict_value(
                self._trigger_source[channel-1],
                ATS_cmd.trigger_sources)
        return src

    def do_set_trigger_slope(self, slp, channel):
        '''
        slope is 
        slp : slope is pos or neg
        eng : 1 for engine J and 2 for engine K
        '''
        self._trigger_slope[channel-1] = {'pos':1,'neg':2}[slp]
    def do_get_trigger_slope(self, channel):
        slp = self._get_key_from_dict_value(
                self._trigger_slope[channel-1],
                {'pos':1,'neg':2})
        return slp

    def do_set_trigger_level(self, lvl, channel):
        '''
        sets trigger threshold level
        lvl : 8 bit number 0 is neg limit, 128 = 0, 255 = pos limit 
        '''
        self._trigger_level[channel-1] = lvl
    def do_get_trigger_level(self, channel):
        return self._trigger_level[channel-1]

    def update_trigger_operation(self):
        '''
        uploads all trigger settings
        '''
        err1 = self._ATS9870_dll.AlazarSetTriggerOperation(
                self._handle,
                self._trigger_oper,
                0,
                self._trigger_source[0],
                self._trigger_slope[0],
                self._trigger_level[0],
                1,
                self._trigger_source[1],
                self._trigger_slope[1],
                self._trigger_level[1])
        err2 = self._ATS9870_dll.AlazarSetExternalTrigger(
                self._handle,
                self._ext_trigger_coupling,
                self._ext_trigger_range)
        err3 = self._ATS9870_dll.AlazarSetTriggerTimeOut(
                self._handle,
                int(self._ext_trigger_timeout*1e5))
        err4 = self._ATS9870_dll.AlazarSetTriggerDelay(
                self._handle,
                int(self._ext_trigger_delay*self.get_sample_rate()*1e3))
        return [err1, err2, err3, err4]

    # more trigger parameters
    def do_set_ext_trigger_timeout(self, to):
        '''
        starts acquisition after to (s) in case no trigger occurred
        '''
        self._ext_trigger_timeout = to
    def do_get_ext_trigger_timeout(self):
        return self._ext_trigger_timeout

    def do_set_ext_trigger_delay(self, val):
        '''
        val : start acquisition after delay (s)
        '''
        self._ext_trigger_delay = val
        
    def do_get_ext_trigger_delay(self):
        return self._ext_trigger_delay
    
    def do_set_ext_trigger_coupling(self, val):
        '''
        val = 'AC' or 'DC'
        '''
        self._ext_trigger_coupling = ATS_cmd.couplings[val]
    def do_get_ext_trigger_coupling(self):
        coupling = self._get_key_from_dict_value(
                self._ext_trigger_coupling, 
                ATS_cmd.couplings)
        return coupling

    def do_set_ext_trigger_range(self, val):
        '''
        range : 1 or 5 V
        '''
        self._ext_trigger_range = {1:1,5:0}[val]
    def do_get_ext_trigger_range(self):
        coupling = self._get_key_from_dict_value(
                self._ext_trigger_range, 
                {1:1,5:0})
        return coupling    

# clock and sampling
    
    def do_set_clock_source(self, ref):
        '''
        use 
        ref = 'internal', 'slow_external', 'external_AC' or 'external_10MHz'
        '''
        self._clock_source = ATS_cmd.clock_sources[ref]
    def do_get_clock_source(self):
        '''
        Returns the clock source ('internal', 'slow_external', 'external_AC' or 'external_10MHz')
        '''
        source = self._get_key_from_dict_value(
                self._clock_source, 
                ATS_cmd.clock_sources)
        return source
        
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

    def do_set_clock_edge_rising(self, state = True):
        '''
        determines if the clock triggers on rising or falling edge 
        of the reference

        True : 'rising' or anything else will set it to falling
        '''
        if state:
            self._clock_edge = 0
        else:
            self._clock_edge = 1
    def do_get_clock_edge_rising(self):
        return self._clock_edge == 0
        
    def set_clock_decimation(self, d = 1):
        '''
        !!! 
        User defined decim. not available in this version 
        (decimation always set to 0)
        !!! 

        d = 0 for disable, 1, 2, 4 or any multiple of 10
        sample rate = clock / d 
        '''
        self._decimation = d
    
    def update_clock_settings(self):
        '''
        Uploads all the clock settings to the card
        '''
        #print self._clock_source
        if self._clock_source == 7:
            self._decimation = int(1e9/(self.get_sample_rate()*1e3))
            err = self._ATS9870_dll.AlazarSetCaptureClock(self._handle, 
                    self._clock_source,
                    int(1e9),
                    self._clock_edge,
                    self._decimation)#self._decimation)
        else:
            
            err = self._ATS9870_dll.AlazarSetCaptureClock(self._handle, 
                    self._clock_source,
                    self._sample_rate,
                    self._clock_edge,
                    0)#self._decimation)
        return err


# Channel settings

    def do_set_coupling(self, coupling, channel):
        '''
        channel : channel 1,2
        coupling : 'AC or 'DC'
        '''
        cpl = ATS_cmd.couplings[coupling]
        self._coupling[channel-1] = cpl
    def do_get_coupling(self, channel):
        '''
        gets the coupling (AC or DC) of channel channel(1,2)
        '''
               
        cpl = self._get_key_from_dict_value(
                self._coupling[channel-1], 
                ATS_cmd.couplings)
        return cpl

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
               
        val = self._get_key_from_dict_value(
                self._range[channel-1], 
                ATS_cmd.ranges)
        return val

    def do_set_impedance(self, impedance, channel):
        '''
        sets channel impedance
        impedance : ohms
        channel : channel 'A', 'B', or 'AB'
        '''
        impl = ATS_cmd.impedances[impedance]
        self._impedance[channel-1] = impl
    def do_get_impedance(self, channel):
        '''
        gets the impedance (in ohms) of channel channel(1,2)
        '''
               
        val = self._get_key_from_dict_value(
                self._impedance[channel-1], 
                ATS_cmd.impedances)
        return val

    def do_set_bandwidth_limit(self, bw, channel):
        '''
        enables or disables bandwidth limit per channel
        bw : True or False (bool)
        ch : channel 'A', 'B', or 'AB'
        '''
        self._bandwidth_limit[channel-1] = int(bw)
    def do_get_bandwidth_limit(self, channel):
        '''
        enables or disables bandwidth limit per channel
        bw : True or False (bool)
        ch : channel 'A', 'B', or 'AB'
        '''
        return self._bandwidth_limit[channel-1] == 1
        
    def update_channel_settings(self):
        '''
        sets coupling, input_range, impedance for
        all channels
        '''
        for ch in [1, 2]:
            chl = ch-1
            err_1 = self._ATS9870_dll.AlazarInputControl(
                    self._handle,
                    ch,
                    self._coupling[chl],
                    self._range[chl],
                    self._impedance[chl])
            err_2 = self._ATS9870_dll.AlazarSetBWLimit(
                    self._handle,
                    ch,
                    self._bandwidth_limit[chl])
        return [err_1, err_2]

# Utility
    def get_time_base(self):
        '''
        Gets the time base
        returns: array
        '''
        dt = 1/self.get_sample_rate()
        time_base = arange(
                1.*self._length)/self._length*self.get_trace_length()*1e-3
        return time_base
    def ats_error_to_text(self, code):
        '''
        Takes the error code and returns a string explaining the error
        code : ATS error code
        '''
        error_to_text = self._ATS9870_dll.AlazarErrorToText
        error_to_text.restype = c_char_p # the function returns a pointer 
        return error_to_text(code)       # to a string

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

    def get_attribute(self, name):
        '''
        returns the value of the attribute self._name
        name : name of attribute (string)
        '''
        exec('att = self._' + name)
        return att

    def print_if_error(self, code, name):
        if code != 512:
            print('error executing %s, error message : %s'%(name,
                    self.ats_error_to_text(code)))
    def trunc(self, value, truncsize):
        return int(value/truncsize)*truncsize

    def abort(self):
        self._ATS9870_dll.AlazarAbortAsyncRead(
                self._handle)

    
    
    def transfer_data(self):
        '''
        Transfers data from the buffers of the card to the memory of the driver (self._buf_arr)
        buffers are retrievd one by one while acquisition is taking place
        '''
        t0 =time()
        
        self.get_busy()
        for k in range(self._n_buf):
            self._buffer_retrieved = k
            self.get_next_buffer()
            qt.msleep()

            
        self._buffer_retrieved = 0     
        self.get_busy()
        self.set_LED(False)
        #print 'end = %s'%(self._ATS9870_dll.AlazarAbortAsyncRead(
        #        self._handle) == 512)
        #print 'time'
        #print time()-t0
   
    def get_data_old(self):
        '''
        sorts the data retrieved from the buffer and normalizes the data
        according to the channel range
        '''
        #dat = self._buf_arr
        self._data = [None]
        try:
            dat_ch1 = numpy.zeros([self._n_buf,self._smp_per_buf/2], dtype=int8)
            dat_ch2 = numpy.zeros([self._n_buf,self._smp_per_buf/2], dtype=int8)
            for k in range(self._n_buf):
                (dat_ch1[k,:], dat_ch2[k,:]) = self.get_data_at_bindex(k)
            
            data = array([
                    dat_ch1.reshape(self._n_buf,self._rec_per_buf,self._length),
                    dat_ch2.reshape(self._n_buf,self._rec_per_buf,self._length)
                    ])
        except MemoryError:
            print('Data too large, reduce size')
            print('Raw data still available for processing')
            print('to get traces one by one for processing (e.g. averaging)')
            print('use self.get_data_at_bindex to get traces one by one')
        return [dat_ch1, datch2]
    def get_data_at_bindex(self, bindex):
        '''
        gets the data belonging to the "bindex"th sweep
        '''
        #t0=time()
        st_index = bindex*self._rec_per_buf*self._length*2
        ch1 = array(self._buf_arr[st_index:st_index + self._length*self._rec_per_buf]-128,dtype=int8)
        ch2 = array(self._buf_arr[st_index + self._length*self._rec_per_buf:st_index + 2*self._length*self._rec_per_buf]-128,dtype=int8)
        #print 't_bin:%s'%(time()-t0)
        return ch1, ch2