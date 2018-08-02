# Written April 2018:
# Thijs Stavenga <thijsstavenga@msn.com>
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
'''

print 'Initializing OSC'
OSC = qt.instruments.create('OSC', 'RIGOL_DS4043',
                             address='USB0::0x1AB1::0x04B1::DS4A181600316::INSTR')
OSC.set_trigger_level(0.75)
OSC.set_trigger_channel(2)
OSC.set_t_offset(0e-9)
OSC.set_resolution(1e-9)

OSC.read_channels([ch])
'''

from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum
from qcodes import VisaInstrument, validators as vals


import csv
# from instrument import Instrument
import visa
import types
import logging
# import socket
# import select
from time import sleep, time
import ctypes as ct
import numpy as np
import datetime

# def has_newline(ans):
#     if len(ans) > 0 and ans.find('\n') != -1:
#         return True
#     return False

# class SocketVisa:
#     def __init__(self, host, port):
#         self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self._socket.connect((host, port))
#         self._socket.settimeout(20)

#     def clear(self):
#         rlist, wlist, xlist = select.select([self._socket], [], [], 0)
#         if len(rlist) == 0:
#             return
#         ret = self.read()
#         print 'Unexpected data before ask(): %r' % (ret, )

#     def write(self, data):
#         self.clear()
#         if len(data) > 0 and data[-1] != '\r\n':
#             data += '\n'
#         # if len(data)<100:
#         # print 'Writing %s' % (data,)
#         self._socket.send(data)

#     def read(self,timeouttime=20):
#         start = time()
#         try:
#             ans = ''
#             while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
#                 ans2 = self._socket.recv(8192)
#                 ans += ans2
#                 if len(ans2) == 0:
#                     sleep(0.01)
#             #print 'Read: %r (len=%s)' % (ans, len(ans))
#             AWGlastdataread = ans
#         except socket.timeout, e:
#             print 'Timed out'
#             return ''

#         if len(ans) > 0:
#             ans = ans.rstrip('\r\n')
#         return ans

#     def ask(self, data):
#         self.clear()
#         self.write(data)
#         return self.read()


class RIGOL_DS4043(VisaInstrument):
    '''

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RIGOL_DS4043', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
        '''
        Initializes the RIGOL_DS4043, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        # logging.info(__name__ + ' : Initializing instrument')
        # Instrument.__init__(self, name, tags=['physical', 'source'])
        self._trig_modes = ['EDGE', 'PULS', 'SLOP','VID', 'PATT', 'RS232',
            'IIC','SPI', 'CAN', 'FLEX', 'USB']
        self._trig_sources = ['CHAN1', 'CHAN2', 'CHAN3', 'CHAN4',
            'EXT', 'EXT5', 'ACL']# also D0 - D15
        self._measure_modes = ['VAVG']
        self._measurment_par = ['SAV','SCUR','SDEV','SMAX','SMIN']
        self._channels = ['CHAN1', 'CHAN2', 'CHAN3', 'CHAN4']

        self._address = address
        self.visa_handle.write_termination = '\n'
        self.visa_handle.read_termination = '\n'
        # if address[:5] == 'TCPIP':
        #     self._visainstrument = SocketVisa(self._address[8:], 5025)
        # else:
        #     self._visainstrument = visa.instrument(address, timeout=2)
        self.add_parameter('sample_rate',
            get_cmd='ACQ:SRAT?',
            get_parser=float,
            unit='Hz')
        self.add_parameter('trigger_mode',
            get_cmd=':TRIG:MODE?',
            get_parser=str,
            set_cmd=lambda s: self._set_trigger_mode(s))
        self.add_parameter('trigger_level',
            get_cmd=lambda: self._get_trig_func_par_value(
                                self.trigger_mode(),
                                'LEV'),
            get_parser=float,
            set_cmd=lambda s: self._set_trig_func_par_value(
                                self.trigger_mode(),
                                'LEV',s),
            unit='V')
        self.add_parameter('trigger_source',
            get_cmd= lambda:_get_trig_func_par(self.trigger_mode(),
                                               'SOUR'),
            get_parser= str,
            set_cmd=self._set_trigger_source,
            unit='')
        self.add_parameter('timebase_offset',
            get_cmd='TIM:OFFS?',
            get_parser=float,
            set_cmd='TIM:OFFS {:f}',
            unit='s')
        self.add_parameter('timebase_scale',
            get_cmd='TIM:SCAL?',
            get_parser=float,
            set_cmd='TIM:SCAL {:f}',
            unit='s')
        self.add_parameter('measure_mode',
            get_cmd='TRIG:MODE?',
            get_parser=str,
            set_cmd='TIM:SCAL {:f}',
            unit='s')
        self.add_parameter('measurement_source',
            get_cmd='MEAS:SOUR?',
            get_parser=str,
            set_cmd=self._set_measurement_source,
            unit='')
        # self.add_parameter('measurement',
        #     flags=Instrument.FLAG_GET,
        #     type=types.FloatType)
        self.add_parameter('current_channel',
            get_cmd='WAV:SOUR?',
            get_parser=str,
            set_cmd=self._set_current_channel,
            unit='')
        # self.add_parameter('read_waveform',
        #     type=types.ListType,
        #     flags=Instrument.FLAG_GET, unit='AU',
        #     tags=['measure'])

        self.add_parameter('no_points',
            get_cmd='WAV:POIN?',
            get_parser=int,
            set_cmd='WAV:POIN {:d}' ,
            unit='')
        # self.add_parameter(
        #     'Navg', type=types.IntType, flags=Instrument.FLAG_GETSET,
        #     minval=1, maxval=8192, unit='s')
        # self.add_function('reset')
        # self.add_function('get_all')
        # self.add_function('get_t_int')
        # self.add_function('set_t_int')
        # if reset:
        #     self.reset()
        # else:
        #     self.get_all()
        # self._visainstrument.write(':STOP\n')
        # self._visainstrument.write(':WAV:FORM ASCII\n')
        # self._visainstrument.write(':CLE\n')
        # self._visainstrument.write(':RUN\n')
        # self._visainstrument.write(':AUT\n')






    # parameter functions
    # def do_set_timebase_offset(self, val):
    #     self._visainstrument.write('TIM:OFFS %s'%val)

    # def do_get_timebase_offset(self):
    #     ans = self._visainstrument.ask('TIM:OFFS?')
    #     return ans

    # def do_set_timebase_scale(self,val):
    #     self._visainstrument.write('TIM:SCAL %s'%val)


    # def do_get_timebase_scale(self):
    #     ans = self._visainstrument.ask('TIM:SCAL?')
    #     return ans

    def _set_trigger_mode(self, mode):

        if mode in self._trig_modes:
            self.write(':TRIG:%s' % mode)
        else:
            raise ValueError('invalid mode %s' % mode)



    # def do_set_trigger_level(self, level, mode):
    #     self._set_trig_func_par_value(mode, 'LEV', level)

    # def do_get_trigger_level(self,mode=None):
    #     return self._get_trig_func_par(mode,'LEV')

    def _set_trigger_source(self, source, mode=None):
        if source in self._trig_sources:
            string = ':TRIG:%s' % mode
            self._set_trig_func_par_value(mode, 'SOUR', source)
        else:
            raise ValueError('invalid mode %s' % mode)


    # def do_get_trigger_source(self,mode=None):
    #     return self._get_trig_func_par(mode,'SOUR')

    # def do_get_no_points(self):
    #     ans = self._visainstrument.ask('WAV:POIN?')
    #     return ans

    # def do_set_no_points(self,val):
    #     self._visainstrument.write('WAV:POIN %s'%val)

    # def do_get_current_channel(self):
    #     ans = self._visainstrument.ask('WAV:SOUR?')
    #     return ans

    def _set_current_channel(self,val):
        if val in self._channels:
            self.write('WAV:SOUR %s'%val)
        else:
            raise ValueError('invalid channel %s' % val)

    # def do_get_sample_rate(self):
    #     ans = self._visainstrument.ask('ACQ:SRAT?')
    #     return ans

    # def do_set_measure_mode(self,mode):
    #     logging.debug('Set measure mode to %s', mode)
    #     if mode in self._measure_modes:
    #         self.measure_mode = mode
    #     else:
    #         logging.error('invalid mode %s' % mode)

    def _set_measurement_source(self, source, mode=None):
        if source in self._trig_sources:
            self.write('MEAS:SOUR %s'%source)

        else:
            raise ValueError('invalid source %s' % mode)



    # def do_get_measurement_source(self,mode=None):
    #     return self._visainstrument.ask('MEAS:SOUR?')

    # def do_get_measure_mode(self):
    #     ans = self.measure_mode
    #     return ans

##################
### Important! ###
##################
    # def do_get_measurement(self,stat_fun = 'SCUR',mode=None):
    #     if stat_fun in self._measurment_par:
    #         ans = self._get_measure_func_par(mode,stat_fun)
    #         return np.double(ans)

    # def do_get_read_waveform(self):
    #     logging.debug('Read current value')
    #     text = self._visainstrument.ask('WAV:DATA?')
    #     return np.double(text.split(',')[:-1])


    # def reset(self):
    #     self._visainstrument.write(':CLE\n')
    # def get_all(self):
    #     # self.get_trigger_level()
    #     # self.get_trigger_channel()
    #     self.get_timebase_offset()
    #     self.get_timebase_scale()
    #     self.get_trigger_mode()
    #     self.get_no_points()
    #     self.get_measurement_source()

    # def _determine_mode(self, mode):
    #     '''
    #     Return the mode string to use.
    #     If mode is None it will return the currently selected mode.
    #     '''
    #     logging.debug('Determine mode with mode=%s' % mode)
    #     if mode is None:
    #         mode = self.get_trigger_mode(query=False)
    #     if mode not in self._trig_modes: # and mode not in ('INIT', 'TRIG', 'SYST', 'DISP'):
    #         logging.warning('Invalid mode %s, assuming current' % mode)
    #         mode = self.get_trigger_mode(query=False)
    #     return mode

###################
### /Important! ###
###################

    def _set_trig_func_par_value(self, mode, par, val):
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
        string = ':TRIG:%s:%s %s' % (mode, par, val)
        logging.debug('Set instrument to %s' % string)
        self._visainstrument.write(string)

    def _get_trig_func_par(self, mode, par):
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
        string = ':TRIG:%s:%s?' % (mode, par)
        ans = self._visainstrument.ask(string)
        logging.debug('ask instrument for %s (result %s)' % \
            (string, ans))
        return ans

##################
### Important! ###
##################
    # def _determine_measure_mode(self, mode):
    #     '''
    #     Return the mode string to use.
    #     If mode is None it will return the currently selected mode.
    #     '''
    #     logging.debug('Determine mode with mode=%s' % mode)
    #     if mode is None:
    #         mode = self.get_measure_mode(query=False)
    #     if mode not in self._measure_modes: # and mode not in ('INIT', 'TRIG', 'SYST', 'DISP'):
    #         logging.warning('Invalid mode %s, assuming current' % mode)
    #         mode = self.get_measure_mode(query=False)
    #     return mode
###################
### /Important! ###
###################

    # def _get_measure_func_par(self, mode, par):
    #     '''
    #     For internal use only!!
    #     Reads the value of the parameter for the function specified
    #     from the instrument

    #     Input:
    #         func (string) : The mode to use
    #         par (string)  : Parameter

    #     Output:
    #         val (string) :
    #     '''
    #     mode = self._determine_measure_mode(mode)
    #     string = ':MEAS:%s:%s?' % (mode, par)
    #     ans = self._visainstrument.ask(string)
    #     logging.debug('ask instrument for %s (result %s)' % \
    #         (string, ans))
    #     return ans





##################
### Important! ###
##################
##################
### Important! ###
##################



    #Ramiro stuff used for the rabi simulations: This can go away with their approval
    def read_channels(self,channel_list):
        self.visawrite(':ACQuire:SRATe?\n')
        srate = self.sample_rate() #np.double(self.visaread())
        data = []
        x_axis = []
        self.prepare_for_readwfm()
        for channel in channel_list:
            self.visawrite(':WAVeform:PREamble?\n')
            config_parameters = self.visaread()
            config_parameters = np.array((config_parameters).split(',')[:-1], dtype=np.double)
            # self.visawrite(':ACQuire:MDEPth?\n')
            # npoints = np.double(self.visaread())
            ch_out = [0]
            while(len(ch_out)<13):
                # print 'blah'
                ch = self.readwfm(channel)
                # print ch
                ch_out = np.double(ch[12:].split(',')[:-1])
            # print splitted_var[-1]
            # ch_out = np.array(splitted_var, dtype=np.double)
            # print ch_out.shape
            step_size = 1./srate#config_parameters[4]*config_parameters[2]/npoints
            # start_t = np.double(self.visaread())

            # Still dont undestand the scaling of 2*ste_size, we just choose it
            # to match the screen
            start_t = config_parameters[5]+step_size
            axis = start_t + np.arange(len(ch_out))*step_size
            x_axis.append(axis)
            data.append(ch_out)

        return x_axis,data

    # def prepare_for_readwfm(self):

    #     self.visawrite(':AUT')
    #     self._visainstrument.write(':TRIGger:MODE EDGe\n')
    #     self._visainstrument.write(':TRIGger:EDGe:SOURce CHANnel%s\n'%self.trigger_channel)
    #     self._visainstrument.write(':TRIGger:EDGe:LEVel %s\n'%self.trigger_level)
    #     self.visawrite(':TIMebase:HREF:MODE TPOSition\n')
    #     self.visawrite(':TIMebase:SCALe %.9f\n'%self.resolution)
    #     self.visawrite(':TIMebase:OFFSet %.9f\n'%self.t_offset)
    #     self.visawrite(':ACQuire:TYPE AVERages\n')
    #     self.visawrite(':ACQuire:AVERages %d\n'%self.Navg)
    #     self.visawrite(':STOP\n')
    #     self.visawrite(':WAV:MODE RAW\n')
    #     self.visawrite(':WAV:RESet\n')

    def readwfm(self, channel, npoints=None):
        readCnt = 0
        readSum = 0
        readTim = 0
        maxPacket = 0

        buff_size = 1024*1024*10

        buff = (ct.c_ubyte*buff_size)()
        wfmBuf = (ct. c_char * len(buff)).from_buffer(buff)

        self.visawrite(':WAV:SOURce CHAN%s\n'% channel)


        # self._visainstrument.write(':WAV:RESet\n')
        self._visainstrument.write(':WAV:BEGin\n')
        self._visainstrument.write(':WAV:STATus?\n')
        reading = True

        status = self._visainstrument.read()
        while (reading and readTim<1):
            if status is 'I':
                self._visainstrument.write(':WAV:DATA?\n')
                var = self._visainstrument.read()
                buff.value = var
                readCnt -= 12
                readSum += (readCnt)
                if readCnt > maxPacket:
                    maxPacket = readCnt

                if readCnt > 0:
                    # array, offset, max
                    pass
                # print 'I',var
                reading = False
                return readSum
            else:
                self._visainstrument.write(':WAV:DATA?\n')
                var = self._visainstrument.read()
                buff.value = var
                readCnt -= 12
                readSum += (readCnt)
                if readCnt > maxPacket:
                    maxPacket = readCnt
                readTim += 1
                if readCnt > 0:
                    # wfmStream.Write(wfmBuf, 11, readCnt);
                    pass
                # print 'Q',var

        self._visainstrument.write(':WAV:END\n')
        # self._visainstrument.write(':RUN\n')
        return var



    # def do_set_Navg(self, Navg):
    #     self.Navg = Navg

    # def do_get_Navg(self):
    #     return self.Navg

    # def readfromscreen(self, channel):
    #     '''
    #     NOT WORKING!
    #     '''
    #     self._visainstrument.write(':WAV:SOURce CHAN%s\n'%channel)
    #     self._visainstrument.write(':WAV:MODE NORM\n')
    #     self._visainstrument.write('::WAV:DATA?\n')
    #     data = self._visainstrument.query_binary_values('::WAV:DATA?\n', datatype='p', is_big_endian=True)
    #     return data

    # def visawrite(self, string):
    #     self._visainstrument.write(string)
    # def visaread(self):
    #     return self._visainstrument.read()

    # def trigger_from_channel(self, channel):
    #     self.trig_channel = channel
    #     self._visainstrument.write(':STOP\n')
    #     self._visainstrument.write(':TRIGger:MODE EDGe\n')
    #     self._visainstrument.write(':TRIGger:EDGe:SOURce CHANnel%s\n'%channel)

    # def cleanrunandscale(self):
    #     self._visainstrument.write(':CLE\n')
    #     self._visainstrument.write(':RUN\n')
    #     self._visainstrument.write(':AUT\n')

    # # Functions
    # def get_t_int(self):
    #     SR = self.get_sample_rate()
    #     no_points = self.get_no_points()
    #     return no_points/SR

    # def set_t_int(self,val):
    #     ans = self.get_sample_rate()*SR
    #     if ans > 1400:
    #         print 'max number of points is 1400'
    #     self.set_no_points(ans)

    # def measure_channels(self,channels):
    #     mode = self.get_measure_mode(query=False)
    #     avg = []
    #     temp_chan = self.get_measurement_source()
    #     for chan in channels:
    #         chan_str = 'CHAN%s'%chan
    #         self.set_measurement_source(chan_str)
    #         avg.append(self.get_measurement(mode=mode))

    #     self.set_measurement_source(temp_chan)
    #     return np.array(avg)