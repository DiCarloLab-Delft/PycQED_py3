# Tektronix_AWG5014.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <mcschaafsma@gmail.com>, 2008
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
import types
import logging
import numpy
import struct
from time import sleep, time

import socket
import select

def has_newline(ans):
    if len(ans) > 0 and ans.find('\n') != -1:
        return True
    return False

class SocketVisa:
    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))
        self._socket.settimeout(20)

    def clear(self):
        rlist, wlist, xlist = select.select([self._socket], [], [], 0)
        if len(rlist) == 0:
            return
        ret = self.read()
        print('Unexpected data before ask(): %r' % (ret, ))

    def write(self, data):
        self.clear()
        if len(data) > 0 and data[-1] != '\r\n':
            data += '\n'
        #if len(data)<100:
            #print 'Writing %s' % (data,)
        self._socket.send(data)

    def read(self,timeouttime=20):
        start = time()
        try:
            ans = ''
            while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
                ans2 = self._socket.recv(8192)
                ans += ans2
                if len(ans2) == 0:
                    sleep(0.01)
            #print 'Read: %r (len=%s)' % (ans, len(ans))
            AWGlastdataread=ans
        except socket.timeout as e:
            print('Timed out')
            return ''

        if len(ans) > 0:
            ans = ans.rstrip('\r\n')
        return ans

    def ask(self, data):
        self.clear()
        self.write(data)
        return self.read()

class Dummy_AWG5014(Instrument):
    '''
    This is the python driver for the Tektronix AWG5014
    Arbitrary Waveform Generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Tektronix_AWG5014', address='<GPIB address>',
        reset=<bool>, numpoints=<int>)

    think about:    clock, waveform length

    TODO:
    1) Get All
    2) Remove test_send??
    3) Add docstrings

    CHANGES:
    26-11-2008 by Gijs: Copied this plugin from the 520 and added support for 2 more channels, added setget marker delay functions and increased max sampling freq to 1.2 	GS/s
    28-11-2008 ''  '' : Added some functionality to manipulate and manoeuvre through the folders on the AWG
    '''

    def __init__(self, name, reset=False, clock=1e9, numpoints=1000):
        '''
        Initializes the AWG5014.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false
            numpoints (int)  : sets the number of datapoints

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Dummy'])




        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._numpoints = numpoints
        # Add parameters
        self.add_parameter('trigger_mode', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('trigger_impedance', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=49, maxval=2e3, units='Ohm')
        self.add_parameter('trigger_level', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=-5, maxval=5, units='Volts')
        self.add_parameter('clock', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=1e6, maxval=1.2e9, units='Hz')
        self.add_parameter('numpoints', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=100, maxval=1e9, units='Int')
        self.add_parameter('filename', type=bytes,
            flags=Instrument.FLAG_SET, channels=(1, 2, 3, 4),
            channel_prefix='ch%d_')
        self.add_parameter('amplitude', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3 ,4), minval=0, maxval=4.6, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('offset', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=-2.25, maxval=2.25, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_delay', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=0, maxval=1, units='ns', channel_prefix='ch%d_')
        self.add_parameter('marker2_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_delay', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4), minval=0, maxval=1, units='ns', channel_prefix='ch%d_')
        self.add_parameter('status', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2, 3, 4),channel_prefix='ch%d_')

        # Add functions
        self.add_function('reset')
        self.add_function('get_all')
        self.add_function('clear_waveforms')
        self.add_function('set_trigger_mode_on')
        self.add_function('set_trigger_mode_off')
        self.add_function('set_trigger_impedance_1e3')
        self.add_function('set_trigger_impedance_50')

        if reset:
            self.reset()
        else:
            self.get_all()
        #self._visainstrument.write('mmem:CDIrectory "\\wfs"')# NOTE! this directory has to exist on the AWG!!

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
        #self._visainstrument.write('*RST')

    def get_state(self):
        state='0'
        if state == '0':
            return 'Idle'
        elif state == '1':
            return 'Waiting for trigger'
        elif state == '2':
            return 'Running'
        else:
            logging.error(__name__  + ' : AWG in undefined state')
            return 'error'

    def start(self):

        return 'Running'

    def stop(self):
        pass
        #self._visainstrument.write('AWGC:STOP')



    def get_folder_contents(self):
        return ''#self._visainstrument.ask('mmem:cat?')

    def get_current_folder_name(self):
        return ''#self._visainstrument.ask('mmem:cdir?')

    def change_folder(self,dir):
        pass
        #self._visainstrument.write('mmem:cdir "\%s"' %dir)

    def goto_root(self):
        pass
        #self._visainstrument.write('mmem:cdir "c:\\.."')


    def create_and_goto_dir(self,dir):
        '''
        Creates (if not yet present) and sets the current directory to <dir> and displays the contents
        TODO:

        '''

        dircheck='%s,DIR' %dir
        if dircheck in self.get_folder_contents():
            self.change_folder(dir)
            logging.debug(__name__  + ' :Directory already exists')
            print('Directory already exists, changed path to %s' %dir)
            print('Contents of folder is %s' %self._visainstrument.ask('mmem:cat?'))
        elif self.get_current_folder_name()=='"\\%s"' %dir:
            print('Directory already set to %s' %dir)
        else:
            self._visainstrument.write('mmem:mdir "\%s"' %dir)
            self._visainstrument.write('mmem:cdir "\%s"' %dir)
            return self.get_folder_contents()

    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''


    def clear_waveforms(self):
        '''
        Clears the waveform on all channels.

        Input:
            None
        Output:
            None
        '''


    def get_refclock(self):
        '''
        Asks AWG whether the 10 MHz reference is set to the
        internal source or an external one.
        Input:
            None

        Output:
            'INT' or 'EXT'
        '''




    def set_refclock_int(self):
        '''
        Sets the reference clock to internal or external
        '''
        self._visainstrument.write('AWGC:CLOC:SOUR INT')

    def set_trigger_mode_on(self):
        '''
        Sets the trigger mode to 'On'

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  +' : Set trigger mode tot TRIG')
        self._visainstrument.write('AWGC:RMOD TRIG')

    def set_trigger_mode_off(self):
        '''
        Sets the trigger mode to 'Cont'

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  +' : Set trigger mode to CONT')
        self._visainstrument.write('AWGC:RMOD CONT')

    def set_trigger_impedance_1e3(self):
        '''
        Sets the trigger impedance to 1 kOhm

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Set trigger impedance to 1e3 Ohm')
        self._visainstrument.write('TRIG:IMP 1e3')

    def set_trigger_impedance_50(self):
        '''
        Sets the trigger impedance to 50 Ohm

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Set trigger impedance to 50 Ohm')
        self._visainstrument.write('TRIG:IMP 50')

    # Parameters
    def _do_get_trigger_mode(self):
        '''
        Reads the trigger mode from the instrument

        Input:
            None

        Output:
            mode (string) : 'Trig' or 'Cont' depending on the mode
        '''
        logging.debug(__name__  + ' : Get trigger mode from instrument')
        return self._visainstrument.ask('AWGC:RMOD?')

    def _do_set_trigger_mode(self, mod):
        '''
        Sets trigger mode of the instrument

        Input:
            mod (string) : Either 'Trig' or 'Cont' depending on the mode

        Output:
            None
        '''
        if (mod.upper()=='TRIG'):
            self.set_trigger_mode_on()
        elif (mod.upper()=='CONT'):
            self.set_trigger_mode_off()
        else:
            logging.error(__name__ + ' : Unable to set trigger mode to %s, expected "TRIG" or "CONT"' %mod)

    def _do_get_trigger_impedance(self):
        '''
        Reads the trigger impedance from the instrument

        Input:
            None

        Output:
            impedance (??) : 1e3 or 50 depending on the mode
        '''
        logging.debug(__name__  + ' : Get trigger impedance from instrument')
        return self._visainstrument.ask('TRIG:IMP?')

    def _do_set_trigger_impedance(self, mod):
        '''
        Sets the trigger impedance of the instrument

        Input:
            mod (int) : Either 1e3 of 50 depending on the mode

        Output:
            None
        '''
        if (mod==1e3):
            self.set_trigger_impedance_1e3()
        elif (mod==50):
            self.set_trigger_impedance_50()
        else:
            logging.error(__name__ + ' : Unable to set trigger impedance to %s, expected "1e3" or "50"' %mod)

    def _do_get_trigger_level(self):
        '''
        Reads the trigger level from the instrument

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Get trigger level from instrument')
        return float(self._visainstrument.ask('TRIG:LEV?'))

    def _do_set_trigger_level(self, level):
        '''
        Sets the trigger level of the instrument

        Input:
            level (float) : trigger level in volts
        '''
        logging.debug(__name__  + ' : Trigger level set to %.3f' %level)
        self._visainstrument.write('TRIG:LEV %.3f' %level)

    def _do_get_numpoints(self):
        '''
        Returns the number of datapoints in each wave

        Input:
            None

        Output:
            numpoints (int) : Number of datapoints in each wave
        '''
        return self._numpoints

    def _do_set_numpoints(self, numpts):
        '''
        Sets the number of datapoints in each wave.
        This acts on both channels.

        Input:
            numpts (int) : The number of datapoints in each wave

        Output:
            None
        '''
        logging.debug(__name__ + ' : Trying to set numpoints to %s' %numpts)
        if numpts != self._numpoints:
            logging.warning(__name__ + ' : changing numpoints. This will clear all waveforms!')

        response = input('type "yes" to continue')
        if response is 'yes':
            logging.debug(__name__ + ' : Setting numpoints to %s' %numpts)
            self._numpoints = numpts
            self.clear_waveforms()
        else:
            print('aborted')

    def _do_get_runmode(self):
        self._runmode = self._visainstrument.ask('AWGC:RMOD?')
        return self._runmode
    def get_runmode(self, runmode='CONT'):
        return self._do_get_runmode()

    def set_runmode(self, runmode='CONT'):
        self._do_set_runmode(runmode)

    def _do_set_runmode(self, runmode='CONT'):
        '''
        runmodes are: CONT TRIG SEQ and GAT
        '''
        self._visainstrument.write('AWGC:RMOD %s' %runmode)
##################################################################
        '''
        sequences section
        '''
    def force_trigger_event(self):
        self._visainstrument.write('TRIG:IMM')

    def force_event(self):
        self._visainstrument.write('EVEN:IMM')

    def set_sqel_event_target_index_next(self, element_no):
        self._visainstrument.write('SEQ:ELEM%s:JTARGET:TYPE NEXT' %element_no)

    def set_sqel_goto_target_index(self, element_no, goto_to_index_no):
        self._visainstrument.write('SEQ:ELEM%s:GOTO:IND  %s' %(element_no, goto_to_index_no))

    def set_sqel_goto_target_index_2(self, element_no):
        self._visainstrument.write('SEQ:ELEM%s:GOTO:IND 2' %element_no)

    def set_sqel_goto_state(self, element_no,goto_state):
        self._visainstrument.write('SEQuence:ELEMent%s:GOTO:STATe %s' %(element_no, int(goto_state)))


    def set_sqel_loopcnt_to_inf(self, element_no, state=True):
        self._visainstrument.write('seq:elem%s:loop:inf %s' %(element_no,int(state)))

    def get_sqel_loopcnt(self, element_no=1):
        return self._visainstrument.ask('SEQ:ELEM%s:LOOP:COUN?' %(element_no))

    def set_sqel_loopcnt(self, loopcount, element_no=1):
        self._visainstrument.write('SEQ:ELEM%s:LOOP:COUN %s' %(element_no,loopcount))

    def set_sqel_waveform(self, waveform_name, channel, element_no=1):
        self._visainstrument.write('SEQ:ELEM%s:WAV%s "%s"' %(element_no, channel, waveform_name))

    def get_sqel_waveform(self, channel, element_no=1):
        return self._visainstrument.ask('SEQ:ELEM%s:WAV%s?' %(element_no, channel))

    def set_sqel_trigger_wait(self, element_no, state=1):
        self._visainstrument.write('SEQ:ELEM%s:TWA %s' %(element_no, state))
        return self.get_sqel_trigger_wait(element_no)

    def get_sqel_trigger_wait(self, element_no):
        return self._visainstrument.ask('SEQ:ELEM%s:TWA?' %(element_no))

    def get_sq_length(self):
        return self._visainstrument.ask('SEQ:LENG?')

    def set_sq_length(self, seq_length):
        self._visainstrument.write('SEQ:LENG %s' %seq_length)

###################################################################






##################################################################


    def _do_get_clock(self):
        '''
        Returns the clockfrequency, which is the rate at which the datapoints are
        sent to the designated output

        Input:
            None

        Output:
            clock (int) : frequency in Hz
        '''
        return self._clock

    def _do_set_clock(self, clock):
        '''
        Sets the rate at which the datapoints are sent to the designated output channel

        Input:
            clock (int) : frequency in Hz

        Output:
            None
        '''
        '''logging.warning(__name__ + ' : Clock set to %s. This is not fully functional yet. To avoid problems, it is better not to change the clock during operation' % clock)'''''
        self._clock = clock
        self._visainstrument.write('SOUR:FREQ %f' % clock)

    def import_waveform_file(self,waveform_listname,waveform_filename,type='wfm'):
        self._import_waveform_file(waveform_listname,waveform_filename)

    def _import_waveform_file(self,waveform_listname,waveform_filename,type='wfm'):
        self._visainstrument.write('mmem:imp "%s","%s",%s'%(waveform_listname, waveform_filename, type))

    def import_and_load_waveform_file_to_channel(self, channel_no ,waveform_listname,waveform_filename,type='wfm'):
        self._import_and_load_waveform_file_to_channel(channel_no ,waveform_listname,waveform_filename)
    def _import_and_load_waveform_file_to_channel(self, channel_no ,waveform_listname,waveform_filename,type='wfm'):
        self._visainstrument.write('mmem:imp "%s","%s",%s'%(waveform_listname, waveform_filename, type))
        self._visainstrument.write('sour%s:wav "%s"' %(channel_no,waveform_listname))
        i=0
        while not self._visainstrument.ask("sour%s:wav?" %channel_no) == '"%s"' %waveform_listname:
            sleep(0.01)
            i=i+1
            print(i)
        return 1


    def _do_set_filename(self, name, channel):
        '''
        Specifies which file has to be set on which channel
        Make sure the file exists, and the numpoints and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the numpoints doesn't match
        the command is neglected

        Input:
            name (string) : filename of uploaded file
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__  + ' : Try to set %s on channel %s' %(name, channel))
        exists = False
        if name in self._values['files']:
            exists= True
            logging.debug(__name__  + ' : File exists in loacal memory')
            self._values['recent_channel_%s' %channel] = self._values['files'][name]
            self._values['recent_channel_%s' %channel]['filename'] = name
        else:
            logging.debug(__name__  + ' : File does not exist in memory, \
            reading from instrument')
            lijst = self._visainstrument.ask('MMEM:CAT? "MAIN"')
            bool = False
            bestand=""
            for i in range(len(lijst)):
                if (lijst[i]=='"'):
                    bool=True
                elif (lijst[i]==','):
                    bool=False
                    if (bestand==name): exists=True
                    bestand=""
                elif bool:
                    bestand = bestand + lijst[i]
        if exists:
            data = self._visainstrument.ask('MMEM:DATA? "%s"' %name)

            logging.debug(__name__  + ' : File exists on instrument, loading \
                    into local memory')
            self._import_waveform_file(name,name)
            # string alsvolgt opgebouwd: '#' <lenlen1> <len> 'MAGIC 1000\r\n' '#' <len waveform> 'CLOCK ' <clockvalue>
            len1=int(data[1])
            len2=int(data[2:2+len1])
            i=len1
            tekst = ""
            while (tekst!='#'):
                tekst=data[i]
                i=i+1
            len3=int(data[i])
            len4=int(data[i+1:i+1+len3])

            w=[]
            m1=[]
            m2=[]

            for q in range(i+1+len3, i+1+len3+len4,5):
                j=int(q)
                c,d = struct.unpack('<fB', data[j:5+j])
                w.append(c)
                m2.append(int(d/2))
                m1.append(d-2*int(d/2))

            clock = float(data[i+1+len3+len4+5:len(data)])

            self._values['files'][name]={}
            self._values['files'][name]['w']=w
            self._values['files'][name]['m1']=m1
            self._values['files'][name]['m2']=m2
            self._values['files'][name]['clock']=clock
            self._values['files'][name]['numpoints']=len(w)

            self._values['recent_channel_%s' %channel] = self._values['files'][name]
            self._values['recent_channel_%s' %channel]['filename'] = name
        else:
            logging.error(__name__  + ' : Invalid filename specified %s' %name)

        if (self._numpoints==self._values['files'][name]['numpoints']):
            logging.warning(__name__  + ' : Set file %s on channel %s' % (name, channel))
            self._visainstrument.write('SOUR%s:WAV "%s"' % (channel, name))
        else:
            logging.warning(__name__  + ' : Verkeerde lengte %s ipv %s'
                %(self._values['files'][name]['numpoints'], self._numpoints))

    def _do_get_amplitude(self, channel):
        '''
        Reads the amplitude of the designated channel from the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            amplitude (float) : the amplitude of the signal in Volts
        '''
        logging.debug(__name__ + ' : Get amplitude of channel %s from instrument'
            %channel)
        val=  (self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:AMPL?' % channel))
        return val

    def _do_set_amplitude(self, amp, channel):
        '''
        Sets the amplitude of the designated channel of the instrument

        Input:
            amp (float)   : amplitude in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set amplitude of channel %s to %.6f'
            %(channel, amp))
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:AMPL %.6f' % (channel, amp))

    def _do_get_offset(self, channel):
        '''
        Reads the offset of the designated channel of the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            offset (float) : offset of designated channel in Volts
        '''
        logging.debug(__name__ + ' : Get offset of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:OFFS?' % channel))

    def _do_set_offset(self, offset, channel):
        '''
        Sets the offset of the designated channel of the instrument

        Input:
            offset (float) : offset in Volts
            channel (int)  : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set offset of channel %s to %.6f' %(channel, offset))
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:OFFS %.6f' % (channel, offset))


    def _do_get_marker1_low(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:LOW?' % channel))

    def _do_set_marker1_low(self, low, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker1 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def _do_get_marker1_high(self, channel):
        '''
        Gets the high level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH?' % channel))

    def _do_set_marker1_high(self, high, channel):
        '''
        Sets the high level for marker1 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker1 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def _do_get_marker1_delay(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        logging.debug(__name__ + ' : Get delay of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:DEL?' % channel))

    def _do_set_marker1_delay(self, delay, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            delay   : in seconds
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set delay of marker1 of channel %s to %.3f'
            %(channel, delay))
        self._visainstrument.write('SOUR%s:MARK1:DEL %.3f' % (channel, delay))

    def _do_get_marker2_low(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:LOW?' % channel))

    def _do_set_marker2_low(self, low, channel):
        '''
        Sets the low level for marker2 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker2 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def _do_get_marker2_high(self, channel):
        '''
        Gets the high level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH?' % channel))

    def _do_set_marker2_high(self, high, channel):
        '''
        Sets the high level for marker2 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker2 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def _do_get_marker2_delay(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        logging.debug(__name__ + ' : Get delay of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:DEL?' % channel))

    def _do_set_marker2_delay(self, delay, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            delay   : in seconds
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set delay of marker1 of channel %s to %.3f'
            %(channel, delay))
        self._visainstrument.write('SOUR%s:MARK2:DEL %.3f' % (channel, delay))


    def delete_all_waveforms_from_list(self):
        self._visainstrument.write('WLISt:WAVeform:DELete ALL')

    def _do_get_status(self, channel):
        '''
        Gets the status of the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Get status of channel %s' %channel)
        outp = self._visainstrument.ask('OUTP%s?' %channel)
        if (outp=='0'):
            return 'off'
        elif (outp=='1'):
            return 'on'
        else:
            logging.debug(__name__ + ' : Read invalid status from instrument %s' %outp)
            return 'an error occurred while reading status from instrument'

    def set_channel_status(self, status, channel):
        self._do_set_status(status, channel)

    def _do_set_status(self, status, channel):
        '''
        Sets the status of designated channel.

        Input:
            status (string) : 'On' or 'Off'
            channel (int)   : channel number

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set status of channel %s to %s'
            %(channel, status))
        if (status.upper()=='ON'):
            self._visainstrument.write('OUTP%s ON' %channel)
        elif (status.upper()=='OFF'):
            self._visainstrument.write('OUTP%s OFF' %channel)
        else:
            logging.debug(__name__ + ' : Try to set status to invalid value %s' % status)
            print('Tried to set status to invalid value %s' %status)

    #  Ask for string with filenames
    def get_filenames(self):
        logging.debug(__name__ + ' : Read filenames from instrument')
        return self._visainstrument.ask('MMEM:CAT?')

    # Send waveform to the device
    def send_waveform(self,w,m1,m2,filename,clock=1e9):
        '''
        Sends a complete waveform. All parameters need to be specified.
        See also: resend_waveform()

        Input:
            w (float[numpoints]) : waveform
            m1 (int[numpoints])  : marker1
            m2 (int[numpoints])  : marker2
            filename (string)    : filename
            clock (int)          : frequency (Hz)

        Output:
            None
        '''
        logging.debug(__name__ + ' : Sending waveform %s to instrument' % filename)
        # Check for errors
        dim = len(w)

        if (not((len(w)==len(m1)) and ((len(m1)==len(m2))))):
            return 'error'

        self._values['files'][filename]={}
        self._values['files'][filename]['w']=w
        self._values['files'][filename]['m1']=m1
        self._values['files'][filename]['m2']=m2
        self._values['files'][filename]['clock']=clock
        self._values['files'][filename]['numpoints']=len(w)

        m = m1 + numpy.multiply(m2,2)
        ws = ''
        for i in range(0,len(w)):
            ws = ws + struct.pack('<fB',w[i],int(numpy.round(m[i],0)))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen=str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6

        self._visainstrument.write(mes)

    def send_visa_command(self, command):
        self._visainstrument.write(command)
    def query_visa(self, query):
        return self._visainstrument.ask(query)



    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=[]):
        '''
        Resends the last sent waveform for the designated channel
        Overwrites only the parameters specified

        Input: (mandatory)
            channel (int) : 1 to 4, the number of the designated channel

        Input: (optional)
            w (float[numpoints]) : waveform
            m1 (int[numpoints])  : marker1
            m2 (int[numpoints])  : marker2
            clock (int) : frequency

        Output:
            None
        '''
        filename = self._values['recent_channel_%s' %channel]['filename']
        logging.debug(__name__ + ' : Resending %s to channel %s' % (filename, channel))


        if (w==[]):
            w = self._values['recent_channel_%s' %channel]['w']
        if (m1==[]):
            m1 = self._values['recent_channel_%s' %channel]['m1']
        if (m2==[]):
            m2 = self._values['recent_channel_%s' %channel]['m2']
        if (clock==[]):
            clock = self._values['recent_channel_%s' %channel]['clock']

        if not ( (len(w) == self._numpoints) and (len(m1) == self._numpoints) and (len(m2) == self._numpoints)):
            logging.error(__name__ + ' : one (or more) lengths of waveforms do not match with numpoints')

        self.send_waveform(w,m1,m2,filename,clock)
        self.set_filename(filename, channel)

    def set_DC_out(self, DC_channel_number, Voltage):
        self._visainstrument.write('AWGControl:DC%s:VOLTage:OFFSet %sV'%(DC_channel_number, Voltage))
        self.get_DC_out(DC_channel_number)

    def get_DC_out(self, DC_channel_number):
        return self._visainstrument.ask('AWGControl:DC%s:VOLTage:OFFSet?'%(DC_channel_number))

    def send_DC_pulse(self, DC_channel_number, amplitude, length):
        '''
        sends a (slow) pulse on the DC channel specified
        Ampliude: voltage level
        length: seconds
        '''
        restore=self.get_DC_out(DC_channel_number)
        self.set_DC_out(DC_channel_number, amplitude)
       ## self.set_DC_state(True)
        sleep(length)
       ## self.set_DC_state(False)
        self.set_DC_out(DC_channel_number, restore)
       ## self.get_DC_out(DC_channel_number)

    def set_DC_state(self, state=False):
        self._visainstrument.write('AWGControl:DC:state %s' %(int(state)))
        self.get_DC_state()

    def get_DC_state(self):
        return self._visainstrument.ask('AWGControl:DC:state?')




