
# Driver class for the AlazarTech ATS9870 Digitizer (DAQ)
# PCI Express board, 2 channels, 8bits, 1GS/s
# Gijs de Lange, 2012
# modified by Adriaan Rol, 9/2015
#
# Functions exported by the DLL are in C format ("CapitalsInFunctionNames()")
# User defined functions in pythion format ("underscores_in_function_names()")
#

import bisect
import qt
from . import _Alazar_ATS9870.AlazarCmd as ATS_cmd
import imp
imp.reload(ATS_cmd)
#  from _AlazarTech_ATS9870.errors import errors as _ats_errors
from instrument import Instrument
from time import clock as time
import types
import logging
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
import sys

Plotmon = qt.instruments['Plotmon']


class Alazar_ATS9870(Instrument):
    '''
    This is the driver for the Alazar ATS9870 data acquisition card

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Alazar_ATS9870')

    There exist some naming conventions in the data acquisition code.
    The table below explains which concepts are related.
    ---------------------------------------------------------------------------
    | ATS-docs          | ATS-driver        | ATS-TD           | ATS-CW       |
    ---------------------------------------------------------------------------
    |                   |                   | Navg             |              |
    |                   | number of buffers | NoSweeps         |              |
    | number_of_records | records_per_buffer| NoSegments       | n_buff       |
    | samples/record    | points_per_trace  | points_per_trace | self._lenght |
    ---------------------------------------------------------------------------

    '''

    def __init__(self, name, mem_size=2048, ATS_buffer=[],
                 return_signed_data=True):
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

        self._initialized = False
        # Load dll and open connection

        self._load_dll()
        self.get_channel_info()
        # add system parameters
        self.add_parameter(
            'board_kind',
            flags=Instrument.FLAG_GET,
            type=str)
        self.add_parameter(
            'serial_number',
            flags=Instrument.FLAG_GET,
            type=int)


        self.add_parameter(
            'memory_size', flags=Instrument.FLAG_GETSET,
            type=int, units='MB', minval=10, maxval=4096)
        self.add_parameter(
            'latest_cal_date', flags=Instrument.FLAG_GET,
            type=int, units='(DDMMYY)')
        # add acquisition parameters
        self.add_parameter(
            'records_per_buffer', flags=Instrument.FLAG_GETSET,
            type=int)
        self.add_parameter(
            'number_of_buffers', flags=Instrument.FLAG_GETSET,
            type=int)
        self.add_parameter(
            'trace_length', flags=Instrument.FLAG_GET,
            units='ms', type=float)
        self.add_parameter(
            'datatype',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            type=str)

        self.add_parameter(
            'points_per_trace',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            type=int)
        self.add_parameter(
            'max_points_per_trace',
            flags=Instrument.FLAG_GET,
            type=int)
        self.add_parameter(
            'busy', flags=Instrument.FLAG_GET,
            type=bool)
        self.add_parameter(
            'check_overload', flags=Instrument.FLAG_GETSET,
            type=bool)
        self.add_parameter(
            'overload', flags=Instrument.FLAG_GETSET,
            type=bool, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter(
            'LED', flags=Instrument.FLAG_GETSET,
            type=bool)

        # add channel parameters
        self.add_parameter(
            'coupling', flags=Instrument.FLAG_GETSET,
            type=bytes, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter(
            'impedance', flags=Instrument.FLAG_GETSET,
            units='Ohm',
            type=int, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter(
            'bandwidth_limit', flags=Instrument.FLAG_GETSET,
            type=bool, channels=(1, 2),
            channel_prefix='ch%d_')
        self.add_parameter(
            'range', flags=Instrument.FLAG_GETSET,
            units='Volt (Vp)',
            type=float, channels=(1, 2), channel_prefix='ch%d_')

        # clock and sampling parameters
        self.add_parameter(
            'clock_source', flags=Instrument.FLAG_GETSET,
            type=bytes)
        self.add_parameter(
            'sample_rate', flags=Instrument.FLAG_GETSET,
            units='KSPS', type=int)
        self.add_parameter(
            'clock_edge_rising', flags=Instrument.FLAG_GETSET,
            type=bool)

        # add trigger parameters
        self.add_parameter(
            'trigger_oper', flags=Instrument.FLAG_GETSET,
            type=bytes)
        self.add_parameter(
            'trigger_slope', flags=Instrument.FLAG_GETSET,
            type=bytes, channels=(1, 2),
            channel_prefix='eng%d_')
        self.add_parameter(
            'trigger_source', flags=Instrument.FLAG_GETSET,
            type=bytes, channels=(1, 2),
            channel_prefix='eng%d_')
        self.add_parameter(
            'trigger_level', flags=Instrument.FLAG_GETSET,
            type=int, channels=(1, 2), channel_prefix='eng%d_')
        self.add_parameter(
            'ext_trigger_delay', flags=Instrument.FLAG_GETSET,
            units='sec', type=float)
        self.add_parameter(
            'ext_trigger_timeout', flags=Instrument.FLAG_GETSET,
            units='sec', type=float)
        self.add_parameter(
            'ext_trigger_coupling', flags=Instrument.FLAG_GETSET,
            type=bytes)

        self.add_parameter(
            'ext_trigger_range', flags=Instrument.FLAG_GETSET,
            type=int, units='Volt')
        self.add_parameter(
            'ro_timeout', flags=Instrument.FLAG_GETSET,
            type=int, units='ms')

        self._mode = 'NPT'  # (NoPreTrigger traditional NOT IMPLEMENTED)
        self._processing_mode = 'averaging'
        self._channel_count = 2
        self.set_ro_timeout(5000)  # timeout for buffer filling complete (ms)
        self.timeout = 10
        self.max_points_per_trace = 1e7
        self.get_max_points_per_trace()
        self.set_check_overload(True)
        # reserve memoryblock of N*1e6 8 byte elements
        self._pre_alloc_memory = mem_size

        if return_signed_data:
            self.set_datatype('signed')
        else:
            self.set_datatype('unsigned')
        self._load_defaults()
        self._buffer_retrieved = 0
        self.set_monitor('mpl')

        self._overload = [False, False]
        self.get_ch1_overload()
        self.get_ch2_overload()

        self._initialized = True
        self._stuck_on_error = 0
        self.configure_board()
        self.get_serial_number()
        self.get_board_kind()
        self.get_latest_cal_date()

    def set_processing_mode(self, pmode):
        self._processing_mode = pmode

    def get_processing_mode(self):
        return self._processing_mode

    def do_get_max_points_per_trace(self):
        '''
        maximum number of points per trace.
        This is currently hardcoded in the driver as 1e7.
        Unclear where this number comes from (ATS specs?)
        '''
        return self.max_points_per_trace

    def _load_defaults(self):
        '''
        Default settings are loaded here
        '''
        self._range = [None, None]
        self._coupling = [None, None]
        self._impedance = [None, None]
        self._bandwidth_limit = [None, None]
        self._trigger_slope = [None, None]
        self._trigger_source = [None, None]
        self._trigger_level = [None, None]

        for ch in [1, 2]:
            exec('self.set_ch%d_coupling("AC")' % ch)
            exec('self.set_ch%d_range(2.)' % ch)
            exec('self.get_ch%d_range()' % ch)
            exec('self.set_ch%d_impedance(50)' % ch)
            exec('self.set_ch%d_bandwidth_limit(False)' % ch)
            self.set_trigger_oper("eng_1_only")
            exec('self.set_eng%d_trigger_slope("pos")' % ch)
            exec('self.set_eng%d_trigger_source("external")' % ch)
            exec('self.set_eng%d_trigger_level(150)' % ch)
            #  exec 'self.set_eng%d_trigger_slope("pos")'%ch
            #  exec 'self.set_ch%d_

        self.set_eng2_trigger_source('chB')
        self.set_eng2_trigger_level(0)
        self.set_ext_trigger_delay(0.)
        self.set_ext_trigger_timeout(0.)
        self.set_ext_trigger_coupling('AC')
        self.set_ext_trigger_range(1)
        self.set_clock_source('internal')
        #  self.set_clock_source('external_10MHz')
        self.set_sample_rate(100e3)
        self.set_clock_edge_rising(True)
        self.set_records_per_buffer(1)
        #  self.set_buffers_per_acq(100) same as numer_of_buf
        self.set_number_of_buffers(10)
        self.set_points_per_trace(960)
        self.get_trace_length()  # ms
        self.get_busy()
        self.set_LED(False)
        self._fast_mode = True

    def _load_dll(self):
        '''
        Make sure the dll is located at "C:\\WINDOWS\\System32\\ATSApi"
        '''
        self._ATS9870_dll = ct.cdll.LoadLibrary(
            'C:\\WINDOWS\\System32\\ATSApi')
        self._handle = self._ATS9870_dll.AlazarGetBoardBySystemID(1, 1)

    def do_set_memory_size(self, msize):
        '''
        sets the size of the reserved memory size
        size : memory size (MB)
        '''
        self._memory_size = msize
        self.reserve_memory()

    def free_memory(self):
        self._buf_arr = np.array([0], dtype=self._dtype)

    def do_get_memory_size(self):
        return self._memory_size

    def reserve_memory(self):
        '''
        Reserve a continuous block of memory in python. The API will store
        its data in this block.
        '''
        self.get_memory_size()
        self._buf_arr = np.zeros(self._memory_size*1024**2,
                                 dtype=self._dtype)

    def initialize_buffer_to_zero(self):
        self._buf_arr[:] = 0

    def do_get_busy(self):
        '''
        check if ats is still running
        '''
        state = self._ATS9870_dll.AlazarBusy(self._handle) == 1
        #  self.set_LED(state)
        return state

    def do_set_check_overload(self, state):
        self._check_overload = state

    def do_get_check_overload(self):
        return self._check_overload

    def do_set_overload(self, state, channel):
        self._overload[channel-1] = state

    def do_get_overload(self, channel):
        return self._overload[channel-1]


# Acquisition

    def configure_board(self):
        '''
        Deprecated, here to maintain backwards compatibility,
        should not be neccesarry to call for setting ATS parameters
        writes the settings to the card

        currently only used in homodyne and pulsed spectrocopy detector
        '''
        err1 = self.update_clock_settings()
        err2 = self.update_channel_settings()
        err3 = self.update_trigger_operation()
        return err1, err2, err3

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
        bps = np.array([0], dtype=np.uint8)
        max_s = np.array([0], dtype=np.uint32)
        success = self._ATS9870_dll.AlazarGetChannelInfo(
            self._handle,
            max_s.ctypes.data,
            bps.ctypes.data)
        self.print_if_error(success, 'GetChannelInfo')
        self._bits_per_sample = bps[0]
        self._max_samples_per_record = max_s[0]
        self._bytes_per_sample = (bps[0]+7)/8

    def do_set_datatype(self, datatype):
        '''
        Sets if data is returned as 8 bit signed or unsigned integers
        (uint8 or int8).
        string options are:  'signed', 'signed 8 bit integer' or
                            'unsigned', 'unsigned 8 bit integer'

        Besides setting this on the ATS it also affects the private variables
        _dtype and _data_unsigned
        '''
        if type(datatype) == str:
            if (datatype.lower() == 'signed' or
                    datatype.lower() == '8 bit signed integer' or
                    datatype.lower() == 'int8'):
                state = True
            elif (datatype.lower() == 'unsigned' or
                  datatype.lower() == '8 bit unsigned integer'or
                  datatype.lower() == 'uint8'):
                state = False
            else:
                raise KeyError('datatype "%s" not recognized' %
                               datatype.lower())
        else:
            state = datatype

        self._data_unsigned = not state

        self._ATS9870_dll.AlazarSetParameter(
            self._handle,
            ct.c_uint8(0),
            ct.c_uint32(0x10000041),
            ct.c_long(state))

        if state:
            self._dtype = np.int8
        else:
            self._dtype = np.uint8
        self.set_memory_size(self._pre_alloc_memory)
        qt.msleep()

    def do_get_datatype(self):
        '''
        returns if signed or unsigned datatype is used
        '''
        val = np.array([0], dtype=np.int32)
        ret = self._ATS9870_dll.AlazarGetParameter(
            self._handle,
            0,
            ct.c_ulong(0x10000042),
            ct.c_void_p(val.ctypes.data))
        if ret == 512:
            if val == 1:
                assert(self._data_unsigned != val)
                return '8 bit signed integer'
            elif val == 0:
                assert(self._data_unsigned != val)
                return '8 bit unsigned integer'
        else:
            msg = self.ats_error_to_text(ret)
            raise Exception(msg)

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

    def do_get_latest_cal_date(self):
        ret, val = self.get_capability(
            ATS_cmd.capabilities['LATEST_CAL_DATE'])
        if ret != 512:
            error_msg = self.ats_error_to_text(ret)
            raise Exception(error_msg)
        return val

    def do_get_serial_number(self):
        ret, val = self.get_capability(
            ATS_cmd.capabilities['SERIAL_NUMBER'])
        if ret != 512:
            error_msg = self.ats_error_to_text(ret)
            raise Exception(error_msg)
        return val

    def get_capability(self, capability):
        '''
            AlazarQueryCapability
        get a device attribute

            Return value
        The function returns ApiSuccess (512) if it was able to retrieve value
        of the specified capability. Otherwise, the function returns an error
        code that indicates the reason that it failed. See Table 1 for a list
        f error codes.
            Remarks
        capabilites and corresponding hex codes are listed in
        ATS_cmd.capabilities
        '''
        val = np.array([0], dtype=np.int32)
        ret = self._ATS9870_dll.AlazarQueryCapability(
            self._handle,
            capability,
            0,
            ct.c_void_p(val.ctypes.data))
        return ret, val[0]

    def do_get_board_kind(self):
        val = self._ATS9870_dll.AlazarGetBoardKind(self._handle)
        board_kind = self._get_key_from_dict_value(val, ATS_cmd.typedef)
        return board_kind

#####################
## acquisition
#####################

    def start_acquisition(self):  # keep this function for compatibility
        '''
        Use this to start the data acquisition

        WARNING: this is a deprecated version, use ATS.arm()
        '''
        # print 'WARNING: this is a deprecated version, use ATS.arm()'
        #  self.abort()
        #  self.configure_board()
        self.arm()

    def arm(self):
        '''
        arms the ats for a measurement
        after which it is waiting for a trigger
        '''
        self.abort()  # makes sure any uncorrectly closed proces is aborted
        self.allocate_memory_for_DMA_buffers()
        self.start_capture()
        while not self.get_busy():
            qt.msleep(0.05)

    def allocate_memory_for_DMA_buffers(self):
        '''
        Prepares for the read out
        - allocates memory
        - prepares the board for an aquisition
        - passes a list of buffer pointers to the board
        '''
        self._buffers_completed = 0
        self._bytes_per_rec = self._bytes_per_sample*self._length
        self._bytes_per_buf = self._bytes_per_rec *\
            self._rec_per_buf *\
            self._channel_count
        #  print self._bytes_per_buf
        self._smp_per_buf = self._length*self._rec_per_buf*self._channel_count

        #   self._buf_arr = np.zeros([self._n_buf,self._smp_per_buf],
        #           dtype = np.int8)
        self.set_record_size()
        dma_flag = (0x001 | 0x200)  # ADMA_EXTERNAL_STARTCAPTURE | ADMA_NPT

        return_code = self._ATS9870_dll.AlazarBeforeAsyncRead(
            self._handle,
            3,  # always use both channels
            0,  # always use 0 pre trigger samples
            self._length,
            self._rec_per_buf,
            self._rec_per_buf*self._n_buf,
            dma_flag)
        self.print_if_error(return_code, 'BeforeAsyncRead')
        if return_code == 512:
            self._stuck_on_error = 0
        else:
            self._stuck_on_error += 1
        if self._stuck_on_error > 5:
            raise NameError('Memory allocation failure')
        #  qt.msleep(0.0)
        #  now send a list of pointers to buffers in the list of buffers
        #  t0 = time()

        bpb = self._bytes_per_buf
        #  print type(bpb)
        #  print 'bpb: ',int(bpb)
        #  print 'memaddr: ',self._buf_arr[0*2*self._rec_per_buf*self._length:].ctypes.data
        for k in range(self._n_buf):
            start_index = k*2*self._rec_per_buf*self._length
            #  print 'buffer to card: ',self._buf_arr[start_index:].ctypes.data
            #  print 'allocate'
            #  print 'buf nr : %s'%k
            #  print 'pointer : ', ctypes.c_void_p(self._buf_arr[start_index:].ctypes.data)

            #  the +3 is to set the lsbyte of the buffer,
            #  valid for sys.byteorder => little endian

            return_code = self._ATS9870_dll.AlazarPostAsyncBuffer(
                self._handle,
                ct.c_void_p(self._buf_arr[start_index:].ctypes.data),
                bpb)          # of buffers

            self.print_if_error(return_code, 'PostAsyncBuffer')
            if return_code != 512:
                print('already armed, aborting...')
                self.abort()
                qt.msleep(1)
                self.allocate_memory_for_DMA_buffers()
                break
            #  print time()-t0

    def start_capture(self):
        '''
        After this command the acquisition starts as soon
        as the trigger condition is met
        '''

        success = self._ATS9870_dll.AlazarStartCapture(self._handle)
        self.print_if_error(success, 'StartCapture')

    def retrieve_data_while_acquiring(self):
        '''
        retrieves the next buffer from the card memory.
        This allows to read out buffers while acquisition takes place.

        This function works in the following way:
            Preallocate an array for average data if required
            Loop while buffers_completed < total number of buffers
                increment the start index of the extraction
                while return_code != 512 (code corresponding to succes)
                    call self._ATS9870_dll.AlazarWaitAsyncBufferComplete() to
                    write the data to the preallocated data array
                    write the data to the average array (if average)
                    check for overloads (should be a function)
                    call _ATS9870_dll.AlazarPostAsyncBuffer() and verify succes
        '''

        if self._processing_mode == 'averaging':
            self._integrated_data = np.zeros(
                2*self._rec_per_buf*self._length, dtype=np.int32)

        while self._buffers_completed < self._n_buf:
            loopcnt = 0
            start_index = self._buffers_completed*2 * \
                self._rec_per_buf*self._length
            stop_index = (self._buffers_completed+1)*2 * \
                self._rec_per_buf*self._length
            return_code = 0
            while return_code != 512:
                return_code = self._ATS9870_dll.AlazarWaitAsyncBufferComplete(
                    self._handle,
                    ct.c_void_p(self._buf_arr[start_index:].ctypes.data),
                    ct.c_int(self._ro_timeout))

                if return_code != 512:
                    if loopcnt >= 1:
                        sys.stdout.write(
                            44*'\b' +
                            'waiting for buffer completion...loopcnt = %s \n'
                            % loopcnt)
                    logging.warning(self.ats_error_to_text(return_code))
                    qt.msleep(1)
                    # sleeping 1 second only when not succesfull
                    # sleeping to ensure enough time to start triggering

                    if loopcnt > 10:
                        qt.msleep()
                        self.abort()
                        print('Acquisition failed, aborting...')
                        raise NameError(self.ats_error_to_text(return_code))

                loopcnt += 1
                qt.msleep()

            if self._processing_mode == 'averaging':
                # Averaging happens here in real time to ensure high data rate
                self._integrated_data = self._integrated_data \
                    + np.asarray(self._buf_arr[start_index:stop_index],
                                 dtype=np.int32) \
                    - 128*self._data_unsigned

                if self._check_overload and self._buffers_completed == 0:
                    ch1startindex = start_index
                    ch2startindex = start_index+self._rec_per_buf*self._length
                    samplech1 = np.array(
                        self._buf_arr[ch1startindex:ch1startindex+self._length]
                        - 128*self._data_unsigned, dtype=np.int8)
                    samplech2 = np.array(
                        self._buf_arr[ch2startindex:ch2startindex+self._length]
                        - 128*self._data_unsigned, dtype=np.int8)
                    if (np.max(samplech1) > 126) or (np.min(samplech1) < -126):
                            self.set_ch1_overload(True)
                    else:
                        self.set_ch1_overload(False)
                    if (np.max(samplech2) > 126) or (np.min(samplech2) < -126):
                        self.set_ch2_overload(True)
                    else:
                        self.set_ch2_overload(False)
            return_code = self._ATS9870_dll.AlazarPostAsyncBuffer(
                self._handle,
                ct.c_void_p(self._buf_arr[start_index:].ctypes.data),
                self._bytes_per_buf)
            self._buffers_completed += 1
        self.abort()

    def do_set_ro_timeout(self, to):
        '''
        Time for API to wait for a completed buffer before a time out error
        is given in ms
        '''
        self._ro_timeout = to

    def do_get_ro_timeout(self):
        return self._ro_timeout

    def get_buffer(self):
        return self._buf_arr

    def get_status(self):
        return self._ATS9870_dll.AlazarGetStatus(self._handle)

    def get_data(self, silent=True):
        qt.msleep(0.005)
        self.retrieve_data_while_acquiring()
        t0 = time()

        if self._data_unsigned:
            dat = np.array(
                self._buf_arr[:self._n_buf*self._rec_per_buf*self._length*2]
                - 128, dtype=np.int8)  # 50 ms
        else:
            dat = self._buf_arr[:self._n_buf*self._rec_per_buf*self._length*2]
        t1 = time()

        dat.shape = (self._n_buf, 2, self._rec_per_buf, self._length)

        t2 = time()
        if not silent:
            print(' actual getting of data from buffer: ', t2-t0, \
                ', recasting: ', t1-t0)
        # data = [datch1,datch2]

        if self._check_overload:
            test_buffer = dat[0, 0, 0, :]
            if len(np.where(abs(test_buffer) > 110)[0]) \
                > len(test_buffer)/100:
                self.set_ch1_overload(True)
                print('ATS Channel OVERLORD demands STOP')
                print('Maximum value on ATS channel 1 exceeded')
            test_buffer = dat[0, 1, 0, :]
            if len(np.where(abs(test_buffer) > 110)[0]) \
                > len(test_buffer)/100:
                self.set_ch2_overload(True)
                print('ATS Channel OVERLORD demands STOP')
                print('Maximum value on ATS channel 2 exceeded')

        return np.transpose(dat, axes=[1, 0, 2, 3])

    def get_rescaled_data(self):
        '''
        get data from ats rescaled to voltage
        '''
        dat = self.get_data()
        ch1_range = self.do_get_range(1)
        ch2_range = self.do_get_range(2)
        if ch1_range == ch2_range:  # should give a minor speedup
            dat = (dat/128.)*ch1_range
            return dat
        else:
            d1 = (dat[0]/128.)*ch1_range
            d2 = (dat[1]/128.)*ch2_range
            return d1, d2

    def get_rescaled_avg_data(self):
        '''
        get data from ats rescaled to voltage
        Averages while acquiring
        '''

        if self._buffers_completed == 0:
            self.retrieve_data_while_acquiring()
        if self._processing_mode == 'averaging':
            dat = []
            for channel in range(2):
                start_index = (channel)*self._rec_per_buf*self._length
                stop_index = (channel+1)*self._rec_per_buf*self._length
                dat.append(self._integrated_data[start_index:stop_index])
        else:
            raise Exception('get rescaled avg data only works with "averaging" mode')
        ch1_range = self.do_get_range(1)
        ch2_range = self.do_get_range(2)
        # divide by 128 bits for rescaling
        # divide by _n_buf because retrieve_data_while_acquiring only adds data
        d1 = (dat[0]/128.)*ch1_range/self._n_buf
        d2 = (dat[1]/128.)*ch2_range/self._n_buf
        return d1, d2

    def average_data(self, channel):
        if self._buffers_completed == 0:
            self.retrieve_data_while_acquiring()

        chrange = 1.*self._get_key_from_dict_value(
            self._range[channel-1],
            ATS_cmd.ranges)
        # key finding works fine 8/4/15
        if self._processing_mode == 'averaging':
            start_index = (channel-1)*self._rec_per_buf*self._length
            stop_index = channel*self._rec_per_buf*self._length
            dat = self._integrated_data[start_index:stop_index]

            dat.shape = (self._rec_per_buf, self._length)
            sdat = (dat/128.)*chrange/self._n_buf
        else:
            dat = self.get_data()
            sdat = (np.average(dat[channel-1], 0)/128.)*chrange

        return sdat

    def do_set_points_per_trace(self, N):
        '''
        length of a single record (number of points)
        must be multiple of 64
        '''
        if N < 64:
            logging.warning('Min points per trace is 64, tried to set %s, rounding up.' % N)
            N = 64
        # if N > self.max_points_per_trace:
        #     raise ValueError('Points per trace > max points per trace')
        self._length = self.trunc(int(N), 64)
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

    def set_record_size(self, pre=0):
        '''
        define relative to trigger
        pre = in NPT mode this is 0 start capture
        after pre number of clock cycles after the trigger
        length = end capture after pre + length number of
        clockcycles after trigger
        '''
        suc6 = self._ATS9870_dll.AlazarSetRecordSize(
            self._handle,
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
        self.update_trigger_operation()

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
        self.update_trigger_operation()

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
        self._trigger_slope[channel-1] = {'pos': 1, 'neg': 2}[slp]
        self.update_trigger_operation()

    def do_get_trigger_slope(self, channel):
        slp = self._get_key_from_dict_value(
            self._trigger_slope[channel-1],
            {'pos': 1, 'neg': 2})
        return slp

    def do_set_trigger_level(self, lvl, channel):
        '''
        sets trigger threshold level
        lvl : 8 bit number 0 is neg limit, 128 = 0, 255 = pos limit
        '''
        self._trigger_level[channel-1] = lvl
        self.update_trigger_operation()

    def do_get_trigger_level(self, channel):
        return self._trigger_level[channel-1]

    def update_trigger_operation(self):
        '''
        uploads all trigger settings
        '''
        if self._initialized is not True:
            return
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
        self.update_trigger_operation()

    def do_get_ext_trigger_timeout(self):
        return self._ext_trigger_timeout

    def do_set_ext_trigger_delay(self, val):
        '''
        val : start acquisition after delay (s)
        '''
        self._ext_trigger_delay = val
        self.update_trigger_operation()

    def do_get_ext_trigger_delay(self):
        return self._ext_trigger_delay

    def do_set_ext_trigger_coupling(self, val):
        '''
        val = 'AC' or 'DC'
        '''
        self._ext_trigger_coupling = ATS_cmd.couplings[val]
        self.update_trigger_operation()

    def do_get_ext_trigger_coupling(self):
        coupling = self._get_key_from_dict_value(
            self._ext_trigger_coupling,
            ATS_cmd.couplings)
        return coupling

    def do_set_ext_trigger_range(self, val):
        '''
        range : 1 or 5 V
        '''
        self._ext_trigger_range = {1: 1, 5: 0}[val]
        self.update_trigger_operation()

    def do_get_ext_trigger_range(self):
        coupling = self._get_key_from_dict_value(
            self._ext_trigger_range,
            {1: 1, 5: 0})
        return coupling

# clock and sampling

    def do_set_clock_source(self, ref):
        '''
        use
        ref = 'internal', 'slow_external', 'external_AC' or 'external_10MHz'
        '''
        self._clock_source = ATS_cmd.clock_sources[ref]
        self.update_clock_settings()

    def do_get_clock_source(self):
        '''
        Returns the clock source ('internal', 'slow_external', 'external_AC'
         or 'external_10MHz')
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
        self.update_clock_settings()
        self.update_trigger_operation()

    def do_get_sample_rate(self):
        '''
        retrieves sampling rate, returns int units KSPS
        '''
        rate = self._get_key_from_dict_value(
            self._sample_rate,
            ATS_cmd.sample_rates)
        return rate

    def do_set_clock_edge_rising(self, state=True):
        '''
        determines if the clock triggers on rising or falling edge
        of the reference

        True : 'rising' or anything else will set it to falling
        '''
        if state:
            self._clock_edge = 0
        else:
            self._clock_edge = 1
        self.update_clock_settings()

    def do_get_clock_edge_rising(self):
        return self._clock_edge == 0

    def set_clock_decimation(self, d=1):
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
        # print self._clock_source
        if self._initialized is not True:
            return
        if self._clock_source == 7:
            self._decimation = int(1e9/(self.get_sample_rate()*1e3))
            err = self._ATS9870_dll.AlazarSetCaptureClock(
                self._handle,
                self._clock_source,
                int(1e9),
                self._clock_edge,
                self._decimation)  # self._decimation)
        else:

            err = self._ATS9870_dll.AlazarSetCaptureClock(
                self._handle,
                self._clock_source,
                self._sample_rate,
                self._clock_edge,
                0)  # self._decimation)
        return err


# Channel settings

    def do_set_coupling(self, coupling, channel):
        '''
        channel : channel 1,2
        coupling : 'AC or 'DC'
        '''
        cpl = ATS_cmd.couplings[coupling]
        self._coupling[channel-1] = cpl
        self.update_channel_settings()

    def do_get_coupling(self, channel):
        '''
        gets the coupling (AC or DC) of channel channel(1,2)
        '''

        cpl = self._get_key_from_dict_value(
            self._coupling[channel-1],
            ATS_cmd.couplings)
        return cpl

    def do_set_range(self, input_range, channel):
        '''
        Set the range (in volts) of the specified ATS channel.
        range is specified as peak voltage.
        Allowed values: 0.04, 0.1, 0.2, 0.4, 1, 2, 4
        '''
        predefined_voltages = [0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0]
        index_of_bisect = bisect.bisect_left(predefined_voltages, input_range)
        rounded_range = predefined_voltages[index_of_bisect]
        rngl = ATS_cmd.ranges[rounded_range]
        if input_range != rounded_range:
            print('Rounded range to predefined voltage, set to %s' \
                % rounded_range)
        self._range[channel-1] = rngl
        self.update_channel_settings()

    def do_get_range(self, channel):
        '''
        Get the range (in volts) of the specified ATS channel.
        range is specified as peak voltage.
        '''
        val = 1.*self._get_key_from_dict_value(
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
        self.update_channel_settings()

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
        self.update_channel_settings()

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
        if self._initialized is not True:
            return
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
        time_base = np.arange(
            1.*self._length)/self._length*self.get_trace_length()*1e-3
        return time_base

    def ats_error_to_text(self, code):
        '''
        Takes the error code and returns a string explaining the error
        code : ATS error code
        '''
        error_to_text = self._ATS9870_dll.AlazarErrorToText
        error_to_text.restype = ct.c_char_p  # the function returns a pointer
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
        k = 0
        while dct[keys[k]] != value:
            k += 1
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
            print('error executing %s, error message : %s' % \
                (name, self.ats_error_to_text(code)))

    def trunc(self, value, truncsize):
        return int(value/truncsize)*truncsize

    def abort(self):
        '''
        From the ATS-SDK Programmer's guide
            AlazarAbortAsyncRead
        Aborts any in-progress DMA transfers, and cancel any pending transfers.

            Return value
        If the function succeeds, it returns ApiSuccess (512).
        If the function fails because it was unable to abort an in-progress
        DMA transfer, it returns ApiDmaInProgress (518).
        If AlazarAbortAsyncRead fails under Windows because the Windows
        CancelIo system call failed, the function returns ApiFailed (513).
        Call the Windows GetLastError API for more information.
        If the function fails for some other reason, it returns an error code
        that indicates the reason that it failed. See Table 1 for a list of
        error codes.
            Remarks
        If you have called AlazarAsyncRead or AlazarPostAsyncBuffer, and there
        are buffers pending, you must call AlazarAbortAsyncRead before your
        application exits. If you do not, when you program exits Microsoft
        Windows may stop with a blue screen error number 0x000000CB
        (DRIVER_LEFT_LOCKED_PAGES_IN_PROCESS). Linux may leak the memory used
        by the DMA buffers.
        '''
        return_code = self._ATS9870_dll.AlazarAbortAsyncRead(
            self._handle)
        self.print_if_error(return_code, 'abort in-progress DMA transfer')
        if return_code != 512:
            msg = self.ats_error_to_text(return_code)
            raise Exception(msg)
        return True

# Monitoring functions

    def mpl_monitor(self, tbase, dat1, dat2, mode='average', clf=True, **kw):
        fig = plt.figure('ATS_monitor', figsize=(8.5, 4))
        if clf:
            plt.clf()
        ax1 = fig.add_subplot(121)

        vstime = kw.pop('vstime', False)
        if vstime:
            ax1.plot(tbase, dat1, '-o', label='channel 1')
            ax1.set_xlabel(r'time ($\mu \mathrm{s}$)')

        else:
            ax1.plot(dat1, '-o', label='channel 1')
            ax1.set_xlabel('N')
        plt.legend(fontsize='small')

        ax2 = fig.add_subplot(122)
        if vstime:
            ax2.plot(tbase, dat2, '-o', label='channel 2')
            ax2.set_xlabel(r'time ($\mu \mathrm{s}$)')
        else:
            ax2.plot(dat2, '-o', label='channel 2')
            ax2.set_xlabel('N')

        plt.legend(fontsize='small')
        if mode == 'average':
            ax1.set_ylabel('Amplitude (V)')
            # ax1.set_ylim(1.3*-130, 130)
            ax1.set_ylim(-self.get_ch1_range(),  self.get_ch1_range())
            ax2.set_ylabel('Amplitude (V)')
            # ax2.set_ylim( -130, 130)
            ax2.set_ylim(-self.get_ch2_range(),  self.get_ch2_range())
        else:
            ax1.set_ylabel('Amplitude (arb. units)')
            ax2.set_ylabel('Amplitude (arb. units)')
            # ax1.set_ylim( -130, 130)
            # ax2.set_ylim( -130, 130)
        plt.tight_layout()

    def LV_monitor(self, tbase, dat1, dat2, **kw):
        Plotmon.plot2D(3, [tbase, dat1])
        Plotmon.plot2D(4, [tbase, dat2])

    def set_monitor(self, mon='LV'):
        if mon == 'LV':
            self.plot_traces = self.LV_monitor
        else:
            self.plot_traces = self.mpl_monitor

    def measure_raw_trace(self, points=500, buffers=1, plot=True, **kw):
        '''
        Acquires a raw trace of data from the ATS.
        Use for diagnostic purposes.
        If plot is True will also plot it.
        Make sure the card gets triggered.
        Use buffer parameter to get averaged traces.
        '''
        self.set_number_of_buffers(buffers)
        nr_buff = self.get_number_of_buffers()
        rec_per_buff = self.get_records_per_buffer()
        pts_per_trace = self.get_points_per_trace()
        self.set_records_per_buffer(1)
        self.set_points_per_trace(points)

        # The actual measurement loop, make sure the ATS gets triggers
        self.abort()
        self.arm()
        d1, d2 = self.get_rescaled_data()
        self.abort()
        d1_avg = np.average(d1[:, 0], axis=0)
        d2_avg = np.average(d2[:, 0], axis=0)
        tbase = np.arange(self.get_points_per_trace(),
                          dtype=np.float)/self.get_sample_rate()*1.e3
        self.set_number_of_buffers(nr_buff)
        self.set_points_per_trace(pts_per_trace)
        self.set_records_per_buffer(rec_per_buff)
        if plot:
            self.plot_traces(tbase, d1_avg, d2_avg, **kw)
        return d1, d2
