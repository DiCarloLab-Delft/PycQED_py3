
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
import _Alazar_ATS9870.AlazarCmd as ATS_cmd
from instrument import Instrument
import pickle
from time import sleep, time
import types
import logging
from numpy import *
import numpy
import qt



class Alazar_ATS9870_dummy(Instrument):
    '''
    This is the driver for the Alazar ATS9870 data acquisition card

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Alazar_ATS9870')

    TODO:
    1) add error handling AlazarError.h
    2) add aparameter trace_length, number of buffers

    '''

    def __init__(self, name, mem_size = 50, ATS_buffer = []):
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
                type=int, units = 'MB', minval=10, maxval = 400)
        # add acquisition parameters
        self.add_parameter('records_per_buffer', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('number_of_buffers', flags=Instrument.FLAG_GETSET, 
                type=int)
        self.add_parameter('trace_length', flags=Instrument.FLAG_GETSET, 
                units = 'ms', type=float)
        self.add_parameter('points_per_trace', flags=Instrument.FLAG_GET, 
                type=int)
        self.add_parameter('busy', flags=Instrument.FLAG_GET, 
                type=bool)
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
        self._channel_count = 2
        self._ro_timeout = 5000 # timeout for buffer filling complete (ms)

            
            
        self._pre_alloc_memory = mem_size # reserve memoryblock of N*1e6 8 byte elements

        self._load_defaults()
        

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
            exec('self.set_ch%d_range(0.04)'%ch)
            exec('self.set_ch%d_impedance(50)'%ch)
            exec('self.set_ch%d_bandwidth_limit(False)'%ch)
            self.set_trigger_oper("eng_1_only") 
            exec('self.set_eng%d_trigger_slope("pos")'%ch)
            exec('self.set_eng%d_trigger_source("external")'%ch)
            exec('self.set_eng%d_trigger_level(128)'%ch)
            #exec 'self.set_eng%d_trigger_slope("pos")'%ch
            #exec 'self.set_ch%d_
        self.set_eng2_trigger_source('chB')
        self.set_eng2_trigger_level(0)
        self.set_ext_trigger_delay(0.)
        self.set_ext_trigger_timeout(0.)
        self.set_ext_trigger_coupling('AC')
        self.set_ext_trigger_range(1)
        self.set_clock_source('internal')
        self.set_sample_rate(100e3)
        self.set_clock_edge_rising(True)
        self.set_records_per_buffer(1)
        # self.set_buffers_per_acq(100) same as numer_of_buf
        self.set_number_of_buffers(10)
        self.set_trace_length(0.01) # ms
        self.get_points_per_trace()
        self.get_busy()
        self.set_LED(False)
        self.set_memory_size(self._pre_alloc_memory)
        self.reserve_memory()
        self.buffer_API_controlled = False
 
    def _load_dll(self):        
        pass  
        #self._ATS9870_dll = cdll.LoadLibrary('C:\\WINDOWS\\System32\\ATSApi') 
        #self._handle = self._ATS9870_dll.AlazarGetBoardBySystemID(1,1)
    def do_set_memory_size(self, msize):
        '''
        sets the size of the reserved memory size
        size : memory size (MB)
        '''
        self._memory_size = msize
        self.reserve_memory()
    def free_memory(self):
        self._buf_arr=numpy.array([0], dtype= numpy.uint8)
    def set_buffer_API_controlled(self, state):
        '''
        specify whether the API should control the buffering
        '''
        self.buffer_API_controlled = state
    def get_buffer_API_controlled(self):
        '''
        ask whether the API should control the buffering
        '''
        return self.buffer_API_controlled

    def do_get_memory_size(self):
        return self._memory_size
    def reserve_memory(self):
        self.get_memory_size()
        self._buf_arr = numpy.zeros(self._memory_size*1024**2,
                dtype=numpy.uint8)
        #qt.ATS_buffer = self._buf_arr
        #self._dat_ch1 = numpy.arange(self._memory_size*512*1024)
        #self._dat_ch2 = numpy.arange(self._memory_size*512*1024)
    def do_get_busy(self):
        '''
        check if ats is still running
        '''
        return 0 == 1
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
        #self._ATS9870_dll.AlazarSetLED(self._handle, int(state))
    def do_get_LED(self):
        return self._led

    def get_channel_info(self):
        '''
        Gets channel info such as bits per sample value 
        and max memory available
        '''
        bps = array([0],dtype = uint8)
        max_s = array([0], dtype = uint32)
        success = 512
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

    def allocate_memory_for_DMA_buffers(self):
        '''
        Prepares for the read out
        - allocates memory
        - prepares the board for an aquisition
        - passes a list of buffer pointers to the board 
        '''
        self._bytes_per_rec = self._bytes_per_sample*self._length
        self._bytes_per_buf = self._bytes_per_rec*\
                self._rec_per_buf*\
                self._channel_count
        #print self._bytes_per_buf
        self._smp_per_buf = self._length*self._rec_per_buf*self._channel_count
        
        # self._buf_arr = numpy.zeros([self._n_buf,self._smp_per_buf], 
        #         dtype = numpy.int8)
        self.set_record_size()
        if self.buffer_API_controlled:
            dma_flag = (0x001 | 0x200 | 0x010 )
        else:
            dma_flag = (0x001 | 0x200 )

        success = 512 # ADMA_EXTERNAL_STARTCAPTURE | ADMA_NPT
        self.print_if_error(success, 'BeforeAsyncRead')
        #qt.msleep(0.0)
        # now send a list of pointers to buffers in the list of buffers
        t0 =time() 
        if self.buffer_API_controlled:
            pass

        else:
            for k in range(self._n_buf):
                bpb = self._bytes_per_buf
                #print 'allocate'
                #print 'buf nr : %s'%k
                #print 'pointer : %s'%self._buf_arr[k*2*self._length:].ctypes.data
                success =512         # of buffers 
                self.print_if_error(success, 'PostAsyncBuffer') 
            #print time()-t0

    def start_capture(self):
        '''
        After this command the acquisition starts as soon 
        as the trigger condition is met
        '''
        pass
        #success = self._ATS9870_dll.AlazarStartCapture(self._handle) 
        #self.print_if_error(success, 'StartCapture')
        
    def arm(self):
        '''
        arms the ats for a measurement
        after which it is waiting for a trigger
        '''
        self.abort_ats()
        self.allocate_memory_for_DMA_buffers()
        self.start_capture()

    def set_ro_timeout(self, to):
        self._ro_timeout=to
    def get_ro_timout(self):
        return self._ro_timeout
    def start_acquisition(self):
        '''
        Use this to start the data acquisition
        '''
        self.abort_ats()
        self.configure_board()
        
        
        
        self.allocate_memory_for_DMA_buffers()
        self.set_LED(True)
        self.start_capture()
        qt.msleep(0*self.get_trace_length()*2./1000)
        self.get_busy() # this never gives true in the GUI, considering removing it
        self.transfer_data()
        self.get_busy()
        self.set_LED(False)
        #return self.get_data()
    def transfer_data(self):
        t0 =time()
        
        for k in range(self._n_buf):
            
            success = 0
            while not (success == 512 or success == 589):
            #while success != 512:
                #print 'transfer'
                #print 'buf nr : %s'%k
                #print 'pointer : %s'%self._buf_arr[k*2*self._length:].ctypes.data
                if self.buffer_API_controlled:
                    success = 512
                           
                else:
                    #print 'retrieving %sth record'%k
                    success =512
                    if (k == self._n_buf-1) and not (success == 512 or success == 589):
                        print(self.print_if_error(success, 'WaitBufferComplete'))
                        success = 512

                self.print_if_error(success, 'WaitBufferComplete')
                qt.msleep(0.00)
                
        #print 'end = %s'%(self._ATS9870_dll.AlazarAbortAsyncRead(
        #        self._handle) == 512)
        #print 'time'
        #print time()-t0
   
    def get_data(self):
        '''
        sorts the data retrieved from the buffer and normalizes the data
        according to the channel range
        '''
        #dat = self._buf_arr
        self._data = [None]
        try:
            dat_ch1 = numpy.zeros([self._n_buf,self._smp_per_buf/2])
            dat_ch2 = numpy.zeros([self._n_buf,self._smp_per_buf/2])
            for k in range(self._n_buf):
                (dat_ch1[k,:], dat_ch2[k,:]) = self.get_scaled_data_at_bindex(k)
                
                self._data = [dat_ch1, dat_ch2]
        except MemoryError:
            print('Data too large, reduce size')
            print('Raw data still available for processing')
            print('to get traces one by one for processing (e.g. averaging)')
            print('use self.get_scaled_data_at_bindex to get traces one by one')
        return self._data
    def get_scaled_data_at_bindex(self, bindex):
        '''
        gets the data belonging to the "bindex"th sweep
        '''
        #t0=time()
        st_index = bindex*self._length*2
        ch1 = (self._buf_arr[st_index:st_index + self._length])
        ch2 = (self._buf_arr[st_index + self._length:st_index + 2*self._length])
        #print 't_bin:%s'%(time()-t0)
        return ch1, ch2

    def average_data(self):
        '''
        takes the sum of all the number_of_buffers acquired 
        '''
        
        int_dat= [numpy.zeros(self._length),numpy.zeros(self._length)]
        #t0=time()
        for bindex in arange(self._n_buf):   
            dat = self.get_scaled_data_at_bindex(bindex)
            int_dat[0]+=dat[0] 
            int_dat[1]+=dat[1]
        #print 't_avg:%s'%(time()-t0)
        int_dat[0] = (int_dat[0]*self.get_ch1_range()/256. -
                self.get_ch1_range()*0.5)/self._n_buf
        int_dat[1] = (int_dat[1]*self.get_ch2_range()/256. -
                self.get_ch2_range()*0.5)/self._n_buf
        return [int_dat[0],int_dat[1]]
            
        
    def get_time_base(self):
        dt = 1/self.get_sample_rate()
        time_base = arange(
                1.*self._length)/self._length*self.get_trace_length()*1e-3
        return time_base
    def do_set_trace_length(self,length_t):
        '''
        length : in ms
        '''
        self._length = self.trunc(int(length_t*self.get_sample_rate()),64)
        self.get_points_per_trace()
    def do_get_trace_length(self):
        return 1.*self._length/self.get_sample_rate()

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
        suc6 = 512
        self.print_if_error(suc6, 'SetRecordSize')

    def set_records_per_buffer(self, rpb):
        '''
        Sets the number of records per buffer
        '''
        self._rec_per_buf = rpb
    def do_get_records_per_buffer(self):
        return self._rec_per_buf

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
        err1 = 512
        err2 =512
        err3 = 512
        err4 = 512
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
        source = self._get_key_from_dict_value(
                self._clock_source, 
                ATS_cmd.clock_sources)
        return source
        
    def do_set_sample_rate(self, rate):
        '''
        !!! 
        User defined not available in this version 
        (decimation always set to 0)
        !!!

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
        d = 0 for disable, 1, 2, 4 or any multiple of 10
        sample rate = clock / d 
        '''
        self._decimation = d
    
    def update_clock_settings(self):
        '''
        Uploads all the clock settings to the card
        '''
        err =  512#self._decimation)
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
        enables or disables bandwidth limt per channel
        bw : True or False (bool)
        ch : channel 'A', 'B', or 'AB'
        '''
        return self._bandwidth_limit[channel-1] == 1
        
    def update_channel_settings(self):
        '''
        sets coupling, input_range, impedance for
        all channels
        '''
        pass
        for ch in [1, 2]:
            chl = ch-1
            err_1 = 512
            err_2 = 512
        return [err_1, err_2]

# Utility

    def ats_error_to_text(self, code):
        '''
        Takes the error code and returns a string explaining the error
        code : ATS error code
        '''
        
        return code       # to a string

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

    def abort_ats(self):
        pass
    def optimize_readout(self,ro_time, max_buf=500):
        '''
        optimizes n_buf and self._length (number of samples)
        ro_time = self._n_buff*self._trace_length (in ms)

        returns the optimum values for n_opt and trace_length
        '''
        n_opt=10
        tr_opt = 1000
        return n_opt, tr_opt
