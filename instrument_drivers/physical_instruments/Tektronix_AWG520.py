# Tektronix_AWG520.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Vishal Ranjan, 2012
# ron schutjens, 2012
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
import time
import qt
import logging
import numpy
import struct

class Tektronix_AWG520(Instrument):
    '''
    This is the python driver for the Tektronix AWG520
    Arbitrary Waveform Generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Tektronix_AWG520', address='<GPIB address>',
        reset=<bool>, numpoints=<int>)

    think about:    clock, waveform length

    TODO:
    1) Get All
    2) Remove test_send??
    3) Add docstrings
    '''

    def __init__(self, name, address, reset=False, clock=1e9, numpoints=1000):
        '''
        Initializes the AWG520.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false
            numpoints (int)  : sets the number of datapoints

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])


        self._address = address
        self._visainstrument = visa.instrument(self._address)
        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._numpoints = numpoints
        self._fname = ''

        # Add parameters
        self.add_parameter('AWG_model' , type =str, flags=Instrument.FLAG_GET)
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
            minval=1e6, maxval=1e9, units='Hz')
        self.add_parameter('numpoints', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=100, maxval=1e9, units='Int')
        self.add_parameter('filename', type=bytes,
            flags=Instrument.FLAG_SET, channels=(1, 2),
            channel_prefix='ch%d_')

        self.add_parameter('amplitude', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2),
            minval=0,
            maxval=1.5,
            units='Volts',
            channel_prefix='ch%d_')

        self.add_parameter('offset', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2),
            minval=-0.1,
            maxval=0.1,
            units='Volts',
            channel_prefix='ch%d_')

        self.add_parameter('marker1_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('status', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2),channel_prefix='ch%d_')

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

    #get state AWG
    def get_state(self):
        state=self._visainstrument.ask('AWGC:RSTATE?')
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
        self._visainstrument.write('AWGC:RUN')
        return self.get_state()

    def stop(self):
        self._visainstrument.write('AWGC:STOP')



    def get_folder_contents(self):
        return self._visainstrument.ask('mmem:cat?')

    def get_current_folder_name(self):
        return self._visainstrument.ask('mmem:cdir?')

    def set_current_folder_name(self,file_path):
        self._visainstrument.write('mmem:cdir "%s"'%file_path)

    def change_folder(self,dir):
        self._visainstrument.write('mmem:cdir "%s"' %dir)

    def goto_root(self):
        self._visainstrument.write('mmem:cdir')

    def make_directory(self,dir,root):
        '''
        makes a directory
        if root = True, new dir in main folder
        '''
        if root == True:
            self.goto_root()
            self._visainstrument.write('MMEMory:MDIRectory "%s"' %dir)
        else:
            self._visainstrument.write('MMEMory:MDIRectory "%s"' %dir)




    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Reading all data from instrument')
        # logging.warning(__name__ + ' : get all not yet fully functional')

        self.get_trigger_mode()
        self.get_trigger_impedance()
        self.get_trigger_level()
        self.get_numpoints()
        self.get_clock()

        for i in range(1,3):
            self.get('ch%d_amplitude' % i)
            self.get('ch%d_offset' % i)
            self.get('ch%d_marker1_low' % i)
            self.get('ch%d_marker1_high' % i)
            self.get('ch%d_marker2_low' % i)
            self.get('ch%d_marker2_high' % i)
            self.get('ch%d_status' % i)

    def clear_waveforms(self):
        '''
        Clears the waveform on both channels.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear waveforms from channels')
        self._visainstrument.write('SOUR1:FUNC:USER ""')
        self._visainstrument.write('SOUR2:FUNC:USER ""')

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
    def do_get_trigger_mode(self):
        '''
        Reads the trigger mode from the instrument

        Input:
            None

        Output:
            mode (string) : 'Trig' or 'Cont' depending on the mode
        '''
        logging.debug(__name__  + ' : Get trigger mode from instrument')
        return self._visainstrument.ask('AWGC:RMOD?')

    def do_set_trigger_mode(self, mod):
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

    def do_get_trigger_impedance(self):
        '''
        Reads the trigger impedance from the instrument

        Input:
            None

        Output:
            impedance (??) : 1e3 or 50 depending on the mode
        '''
        logging.debug(__name__  + ' : Get trigger impedance from instrument')
        return self._visainstrument.ask('TRIG:IMP?')

    def do_set_trigger_impedance(self, mod):
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

    def do_get_trigger_level(self):
        '''
        Reads the trigger level from the instrument

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Get trigger level from instrument')
        return float(self._visainstrument.ask('TRIG:LEV?'))

    def do_set_trigger_level(self, level):
        '''
        Sets the trigger level of the instrument

        Input:
            level (float) : trigger level in volts
        '''
        logging.debug(__name__  + ' : Trigger level set to %.3f' %level)
        self._visainstrument.write('TRIG:LEV %.3f' %level)

    def force_trigger(self):
        '''
        forces a trigger event (used for wait_trigger option in sequences)

        Ron
        '''
        return self._visainstrument.write('TRIG:SEQ:IMM')

    def force_logicjump(self):
        '''
        forces a jumplogic event (used as a conditional event during waveform
        executions)

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self._visainstrument.write('AWGC:EVEN:SEQ:IMM')

    def set_run_mode(self,mode):
        '''
        sets the run mode of the AWG.
        mode can be: CONTinuous,TRIGgered,GATed,ENHanced

        Ron
        '''
        return self._visainstrument.write('AWGC:RMOD %s' %mode)

    def get_run_mode(self):
        '''
        sets the run mode of the AWG

        Ron
        '''
        return self._visainstrument.ask('AWGC:RMOD?')

    def set_jumpmode(self,mode):
        '''
        sets the jump mode for jump logic events, possibilities:
        LOGic,TABle,SOFTware
        give mode as string

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self._visainstrument.write('AWGC:ENH:SEQ:JMOD %s' %mode)

    def get_jumpmode(self,mode):
        '''
        get the jump mode for jump logic events

        Ron
        '''
        return self._visainstrument.ask('AWGC:ENH:SEQ:JMOD?')


    def do_get_numpoints(self):
        '''
        Returns the number of datapoints in each wave

        Input:
            None

        Output:
            numpoints (int) : Number of datapoints in each wave
        '''
        return self._numpoints

    def do_set_numpoints(self, numpts):
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

        response = 'yes'#raw_input('type "yes" to continue')
        if response is 'yes':
            logging.debug(__name__ + ' : Setting numpoints to %s' %numpts)
            self._numpoints = numpts
            self.clear_waveforms()
        else:
            print('aborted')

    def do_get_clock(self):
        '''
        Returns the clockfrequency, which is the rate at which the datapoints are
        sent to the designated output

        Input:
            None

        Output:
            clock (int) : frequency in Hz
        '''
        return self._clock

    def do_set_clock(self, clock):
        '''
        Sets the rate at which the datapoints are sent to the designated output channel

        Input:
            clock (int) : frequency in Hz

        Output:
            None
        '''
        logging.warning(__name__ + ' : Clock set to %s. This is not fully functional yet. To avoid problems, it is better not to change the clock during operation' % clock)
        self._clock = clock
        self._visainstrument.write('SOUR:FREQ %f' % clock)

    def set_setup_filename(self, fname, force_reload=False):
        if self._fname == fname and not force_reload:
            print('File %s already loaded in AWG520' %fname)
            return
        else:
            self._fname = fname
            filename = "\%s/%s.seq" % (fname, fname)
            self.set_sequence(filename=filename)
            print('Waiting for AWG to load file "%s"' % fname)
            sleeptime = 0.5
            # while state idle is not possible due to timeout error while loading
            t0 = time.time()
            while(time.time()-t0 < 360):
                try:
                    if self.get_state() == 'Idle':
                        break
                except:
                    qt.msleep(sleeptime)
                    print('.')
            self.get_state()
            print('Loading file took %.2fs' % (time.time()-t0))
            return



    def do_set_filename(self, name, channel):
        '''
        Specifies which file has to be set on which channel
        Make sure the file exists, and the numpoints and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the numpoints doesn't match
        the command is neglected

        Input:
            name (string) : filename of uploaded file
            channel (int) : 1 or 2, the number of the designated channel

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
            logging.debug(__name__  + ' : Set file %s on channel %s' % (name, channel))
            self._visainstrument.write('SOUR%s:FUNC:USER "%s","MAIN"' % (channel, name))
        else:
            self._visainstrument.write('SOUR%s:FUNC:USER "%s","MAIN"' % (channel, name))
            logging.warning(__name__  + ' : Verkeerde lengte %s ipv %s'
                %(self._values['files'][name]['numpoints'], self._numpoints))

    def do_get_amplitude(self, channel):
        '''
        Reads the amplitude of the designated channel from the instrument

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            amplitude (float) : the amplitude of the signal in Volts
        '''
        logging.debug(__name__ + ' : Get amplitude of channel %s from instrument'
            %channel)
        return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:AMPL?' % channel))

    def do_set_amplitude(self, amp, channel):
        '''
        Sets the amplitude of the designated channel of the instrument

        Input:
            amp (float)   : amplitude in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set amplitude of channel %s to %.6f'
            %(channel, amp))
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:AMPL %.6f' % (channel, amp))

    def do_get_offset(self, channel):
        '''
        Reads the offset of the designated channel of the instrument

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            offset (float) : offset of designated channel in Volts
        '''
        logging.debug(__name__ + ' : Get offset of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:OFFS?' % channel))

    def do_set_offset(self, offset, channel):
        '''
        Sets the offset of the designated channel of the instrument

        Input:
            offset (float) : offset in Volts
            channel (int)  : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set offset of channel %s to %.6f' %(channel, offset))
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:OFFS %.6f' % (channel, offset))

    def do_get_marker1_low(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:LOW?' % channel))

    def do_set_marker1_low(self, low, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker1 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def do_get_marker1_high(self, channel):
        '''
        Gets the high level for marker1 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH?' % channel))

    def do_set_marker1_high(self, high, channel):
        '''
        Sets the high level for marker1 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker1 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def do_get_marker2_low(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:LOW?' % channel))

    def do_set_marker2_low(self, low, channel):
        '''
        Sets the low level for marker2 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker2 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def do_get_marker2_high(self, channel):
        '''
        Gets the high level for marker2 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH?' % channel))

    def do_set_marker2_high(self, high, channel):
        '''
        Sets the high level for marker2 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker2 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def do_get_status(self, channel):
        '''
        Gets the status of the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

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

    def do_set_status(self, status, channel):
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
        return self._visainstrument.ask('MMEM:CAT? "MAIN"')

    def return_self(self):
        return self
    # Send waveform to the device
    def send_waveform(self,w,m1,m2,filename,clock):
        '''
        Sends a complete waveform. All parameters need to be specified.
        choose a file extension 'wfm' (must end with .pat)
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
            ws = ws + struct.pack('<fB', w[i], int(m[i]))


        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen=str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6
        self._visainstrument.write(mes)

    def send_pattern(self,w,m1,m2,filename,clock):
        '''
        Sends a pattern file.
        similar to waveform except diff file extension
        number of poitns different. diff byte conversion
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
        logging.debug(__name__ + ' : Sending pattern %s to instrument' % filename)

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
            ws = ws + struct.pack('<fB', w[i], int(m[i]))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 2000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen=str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6
        self._visainstrument.write(mes)


    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=[]):
        '''
        Resends the last sent waveform for the designated channel
        Overwrites only the parameters specifiedta

        Input: (mandatory)
            channel (int) : 1 or 2, the number of the designated channel

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
        self.do_set_filename(filename, channel)
    def delete_all_waveforms_from_list(self):
        '''
        for compatibillity with awg, is not relevant for AWG520 since it has no waveform list
        '''
        pass

    def send_sequence(self,wfs,rep,wait,goto,logic_jump,filename):
        '''
        Sends a sequence file (for the moment only for ch1)
        Inputs (mandatory):

           wfs:  list of filenames

        Output:
            None
        '''
        logging.debug(__name__ + ' : Sending sequence %s to instrument' % filename)
        N = str(len(rep))
        try:
            wfs.remove(N*[None])
        except ValueError:
            pass
        s1 = 'MMEM:DATA "%s",' % filename

        if len(shape(wfs)) ==1:
            s3 = 'MAGIC 3001\n'
            s5 = ''
            for k in range(len(rep)):
                s5 = s5+ '"%s",%s,%s,%s,%s\n'%(wfs[k],rep[k],wait[k],goto[k],logic_jump[k])

        else:
            s3 = 'MAGIC 3002\n'
            s5 = ''
            for k in range(len(rep)):
                s5 = s5+ '"%s","%s",%s,%s,%s,%s\n'%(wfs[0][k],wfs[1][k],rep[k],wait[k],goto[k],logic_jump[k])

        s4 = 'LINES %s\n'%N
        lenlen=str(len(str(len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s5) + len(s4) + len(s3))


        mes = s1 + s2 + s3 + s4 + s5
        self._visainstrument.write(mes)

    def send_sequence2(self,wfs1,wfs2,rep,wait,goto,logic_jump,filename):
        '''
        Sends a sequence file
        Inputs (mandatory):
            wfs1:  list of filenames for ch1 (all must end with .pat)
            wfs2: list of filenames for ch2 (all must end with .pat)
            rep: list
            wait: list
            goto: list
            logic_jump: list
            filename: name of output file (must end with .seq)
        Output:
            None
        '''
        logging.debug(__name__ + ' : Sending sequence %s to instrument' % filename)


        N = str(len(rep))
        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 3002\n'
        s4 = 'LINES %s\n'%N
        s5 = ''


        for k in range(len(rep)):
            s5 = s5+ '"%s","%s",%s,%s,%s,%s\n'%(wfs1[k],wfs2[k],rep[k],wait[k],goto[k],logic_jump[k])

        lenlen=str(len(str(len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s5) + len(s4) + len(s3))


        mes = s1 + s2 + s3 + s4 + s5
        self._visainstrument.write(mes)

    def set_sequence(self,filename):
        '''
        loads a sequence file on all channels.
        Waveforms/patterns to be executed on respective channel
        must be defined inside the sequence file itself
        make sure to send all waveforms before setting a seq
        '''
        self._visainstrument.write('SOUR%s:FUNC:USER "%s","MAIN"' % (1, filename))

    def load_and_set_sequence(self,wfs,rep,wait,goto,logic_jump,filename):
        '''
        Loads and sets the awg sequecne
        '''
        self.send_sequence(wfs,rep,wait,goto,logic_jump,filename)
        self.set_sequence(filename)

    def do_get_AWG_model(self):
        return 'AWG520'