# Crerates one big AWG from multiple AWG's
# UNDERS CONSTRUCTION!!!

from instrument import Instrument

import types
import logging
import numpy
from numpy import arange
import numpy as np
import struct
from time import sleep, time, localtime
import datetime

from io import StringIO
import socket
import select



class AWG_Composite(Instrument):
    '''
    This is the python driver for combining multiple AWG's together

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Composite_AWG', [AWG1,AWG2,..], reset=<bool>, clock=<int>)

    think about:    clock, waveform length

    TODO:
    1) Get All
    2) Remove test_send??
    3) Add docstrings

     
    '''

    def __init__(self, name, AWG_list, reset=False, clock=1e9):
        '''
        

        Input:
            name (string)    : name of the instrument
            AWG_list : list of AWG instruments
            reset (bool)     : resets to default values, default=false
            

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['composite'])

        
         
        
        self._clock = clock
        self._AWG_list = AWG_list
        self._number_of_AWGS = len(self._AWG_list)
        self._AWG_dict = {}
        k=0
        self.channels = 0
        self.chmap = {}
        self.awg_chspec = {'Tektronix_AWG5014':4,'Tektronix_AWG520':2}
        self._AWG_properties=[]
        for AWG in self._AWG_list:
            atype = AWG.get_type()
            channels=self.awg_chspec[atype]
            for ch in range(channels):
                self.chmap[ch+1+self.channels] = [AWG,ch+1,k]
            self.channels+=channels    
	    self._AWG_properties += [{'type':atype,'channels' : channels} ]
            k+=1
        
        tch = tuple(numpy.arange(self.channels)+1)
        # Add parameters
        
        self.add_parameter('clock', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=1e6, maxval=1.2e9, units='Hz')
        

        self.add_parameter('amplitude', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=0, maxval=4.6, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('offset', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=-2.25, maxval=2.25, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_delay', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=0, maxval=1, units='ns', channel_prefix='ch%d_')
        self.add_parameter('marker2_low', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_high', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=-2.7, maxval=2.7, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_delay', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch, minval=0, maxval=1, units='ns', channel_prefix='ch%d_')
        self.add_parameter('status', type=bytes,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=tch,channel_prefix='ch%d_')

        # Add functions
        self.add_function('reset')


        if reset:
            self.reset()
        else:
            self.get_all()


    # Functions
    def get_AWG_list(self):
        return self._AWG_list

    def start(self):
        n_awg = len(self._AWG_list)
        for n in range(n_awg):
            awg = self._AWG_list[-1-n]
            awg.start()
        

    def stop(self):
        for awg in self._AWG_list:
            awg.stop()

    def reset(self):
	'''
	not yet implemented
	'''


    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        params = self.get_parameter_names()
        for param in params:
            try:
                exec('self.get_%s()'%param)
            except AttributeError:
                pass

    def set_all_channels_on(self):
        '''
        Switches on all awg channels
        '''
        for awg in self._AWG_list:
            atype = awg.get_type()
            n_ch=self.awg_chspec[atype]
            for n in range(n_ch):
                awg.set_channel_status('On',n+1)

    def set_ch_on(self,ch):
        awg = self.chmap[ch][0]
        awg.set_channel_status('On',ch)

    def set_ch_off(self,ch):
        awg = self.chmap[ch][0]
        awg.set_channel_status('Off',ch)




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
        for awg in self._AWG_list:
            clock = awg.get_clock()
        return self._clock

    def _do_set_clock(self, clock):
        '''
        Sets the rate at which the datapoints are sent to the designated output channel

        Input:
            clock (int) : frequency in Hz

        Output:
            None
        '''
        '''logging.warning(__name__ + ' : Clock set to %s. This i
        s not fully functional yet. To avoid problems, it is better not to change the clock during operation' % clock)'''''
        self._clock = clock
        for awg in self._AWG_list:
            awg.set_clock(clock)

    def get_number_of_channels(self):
        return self.channels

    def get_awg_channel_property(self,channel, property):
        awg = self.chmap[channel][0]
        ch = self.chmap[channel][1]
        exec('val = awg.get_ch%s_%s()'%(ch,property))
        return val
    def set_awg_channel_property(self,channel, property, val):
        awg = self.chmap[channel][0]
        ch = self.chmap[channel][1]
        exec('awg.set_ch%s_%s(%s)'%(ch,property,val))
    
    def _do_get_amplitude(self, channel):
        '''
        Reads the amplitude of the designated channel from the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            amplitude (float) : the amplitude of the signal in Volts
        '''
        val = self.get_awg_channel_property(channel, 'amplitude')
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
        self.set_awg_channel_property(channel, 'amplitude', amp)

    def _do_get_offset(self, channel):
        '''
        Reads the offset of the designated channel of the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            offset (float) : offset of designated channel in Volts
        '''
        val = self.get_awg_channel_property(channel, 'offset')
        return val

    def _do_set_offset(self, offset, channel):
        '''
        Sets the offset of the designated channel of the instrument

        Input:
            offset (float) : offset in Volts
            channel (int)  : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        self.set_awg_channel_property(channel, 'offset', offset)


    def _do_get_marker1_low(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        val = self.get_awg_channel_property(channel, 'marker1_low')
        return val

    def _do_set_marker1_low(self, low, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker1_low',low)
       

    def _do_get_marker1_high(self, channel):
        '''
        Gets the high level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        val = self.get_awg_channel_property(channel, 'marker1_high')
        return val

    def _do_set_marker1_high(self, high, channel):
        '''
        Sets the high level for marker1 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker1_high',high)

    def _do_get_marker1_delay(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        val = self.get_awg_channel_property(channel, 'marker1_delay')
        return val
        

    def _do_set_marker1_delay(self, delay, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            delay   : in seconds
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker1_delay',delay)

    def _do_get_marker2_low(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        val = self.get_awg_channel_property(channel, 'marker2_low')
        return val

    def _do_set_marker2_low(self, low, channel):
        '''
        Sets the low level for marker2 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker2_low',low)
       

    def _do_get_marker2_high(self, channel):
        '''
        Gets the high level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        val = self.get_awg_channel_property(channel, 'marker2_high')
        return val

    def _do_set_marker2_high(self, high, channel):
        '''
        Sets the high level for marker2 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker2_high',high)

    def _do_get_marker2_delay(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        val = self.get_awg_channel_property(channel, 'marker2_delay')
        return val
        

    def _do_set_marker2_delay(self, delay, channel):
        '''
        Sets the low level for marker2 on the designated channel.

        Input:
            delay   : in seconds
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
         '''
        self.set_awg_channel_property(channel, 'marker2_delay',delay)
    

    def _do_get_status(self, channel):
        '''
        Gets the status of the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        
        self.get_awg_channel_property(channel, 'status')

 

    def _do_set_status(self, status, channel):
        '''
        Sets the status of designated channel.

        Input:
            status (string) : 'On' or 'Off'
            channel (int)   : channel number

        Output:
            None
        '''
        self.set_awg_channel_property(channel, 'status', status)

    #  Ask for string with filenames

    # Send waveform to the device
    def clear_waveforms(self):
        '''
        Clears waveform from the list (for AWG 5014)
        '''
        for AWG in self._AWG_list:
            AWG.delete_all_waveforms_from_list()

    def send_waveform(self, wf, wf_name, channel):
        '''
        send wf containing analog waveform and two marker (digital) waveforms to the AWG assigned to
        channel.
        
        wf = [analogwf,m1,m2]
        channel = Composite AWG channel
        name = waveform name
        
        '''
        AWG = self.chmap[channel][0]
        name = wf_name
        AWG.send_waveform(wf[0],wf[1],wf[2], name , self._clock)
        if AWG.get_type() =='Tektronix_AWG5014':
            AWG.import_waveform_file(name, name)
            
    
    def pack_waveform(self,wf,m1,m2):
        '''
        packs analog waveform in 14 bit integer, and two bits for m1 and m2 in a single 16 bit integer 
        '''
        wflen = len(wf)
        packed_wf = np.zeros(wflen,dtype=np.uint16)
        packed_wf +=np.round(wf*8191)+8191+16384*m1+32768*m2
        return packed_wf
    
    def generate_awg_file(self,packed_waveforms,wfname_l,delay_labs, nrep, trig_wait, goto_state, jump_to):
        '''
        packed_waveforms: dictionary containing packed waveforms with keys wfname_l and delay_labs
        wfname_l: list of waveform names [[segm1_ch1,segm2_ch1..],[segm1_ch2,segm2_ch2..],...]
        delay_labs = list of length len(channels) for a waveform that is repeated in between segments 
        nrep_l: list of len(segments) specifying the no of reps per segment (0,65536)
        wait_l: list of len(segments) specifying triger wait state (0,1)
        goto_l: list of len(segments) specifying goto state (0,65536, 0 means next)
        logic_jump_l: list of len(segments) specifying logic jump (0 = off)
        filestructure:
        if True:
            MAGIC
            VERSION
            SAMPLING_RATE
            RUN_MODE
            RUN_STATE
            CHANNEL_STATE_1
            CHANNEL_STATE_2
            CHANNEL_STATE_3
            CHANNEL_STATE_4

            WAVEFORM_NAME_N
            WAVEFORM_TYPE_N
            WAVEFORM_LENGTH_N
            WAVEFORM_TIMESTAMP_N
            WAVEFORM_DATA_N

            SEQUENCE_WAIT_M
            SEQUENCE_LOOP_M
            SEQUENCE_JUMP_M
            SEQUENCE_GOTO_M
            SEQUENCE_WAVEFORM_NAME_CH_1_M
            SEQUENCE_WAVEFORM_NAME_CH_2_M
            SEQUENCE_WAVEFORM_NAME_CH_3_M
            SEQUENCE_WAVEFORM_NAME_CH_4_M
        '''
        self._packed_waveforms = packed_waveforms
        self._filename_list = wfname_l
        self._delay_labs = delay_labs
        self._delay_lab = delay_labs[0][:-4]
        
        timetuple = tuple(np.array(localtime())[[0,1,8,2,3,4,5,6,7]])
        timestamp = struct.pack('8h',*timetuple[:-1])
        chstate = []
        for wfch in self._filename_list[0]:
            if wfch is None:
                chstate+=[0]
            else:
                chstate+=[0]
        head = self.pack_record('MAGIC',5000,'h')+\
                    self.pack_record('VERSION',1,'h')+\
                    self.pack_record('SAMPLING_RATE',1e9,'d')+\
                    self.pack_record('REFERENCE_SOURCE',2,'h')+\
                    self.pack_record('TRIGGER_INPUT_THRESHOLD',1.0,'d')+\
                    self.pack_record('RUN_MODE',4,'h')+\
                    self.pack_record('RUN_STATE',0,'h')+\
                    self.pack_record('CHANNEL_STATE_1',1,'h')+\
                    self.pack_record('MARKER1_METHOD_1',2,'h')+\
                    self.pack_record('MARKER2_METHOD_1',2,'h')+\
                    self.pack_record('CHANNEL_STATE_2',1,'h')+\
                    self.pack_record('MARKER1_METHOD_2',2,'h')+\
                    self.pack_record('MARKER2_METHOD_2',2,'h')+\
                    self.pack_record('CHANNEL_STATE_3',1,'h')+\
                    self.pack_record('MARKER1_METHOD_3',2,'h')+\
                    self.pack_record('MARKER2_METHOD_3',2,'h')+\
                    self.pack_record('CHANNEL_STATE_4',1,'h')+\
                    self.pack_record('MARKER1_METHOD_4',2,'h')+\
                    self.pack_record('MARKER2_METHOD_4',2,'h')
        
        ii=21
        record_str = StringIO()
        
        
        delay_wf = self._packed_waveforms.pop(self._delay_lab)
        lendelaydat = len(delay_wf)
        
        for delay_lab in self._delay_labs:
            if delay_lab is None:
                pass
            else:
                #print 'WAVEFORM_NAME_%s: '%ii, delay_lab, 'len: ',len(delay_wf)
                record_str.write(self.pack_record('WAVEFORM_NAME_%s'%ii, delay_lab+'\x00','%ss'%len(delay_lab+'\x00'))+\
                            self.pack_record('WAVEFORM_TYPE_%s'%ii, 1,'h')+\
                            self.pack_record('WAVEFORM_LENGTH_%s'%ii,lendelaydat,'l')+\
                            self.pack_record('WAVEFORM_TIMESTAMP_%s'%ii, timetuple[:-1],'8h')+\
                            self.pack_record('WAVEFORM_DATA_%s'%ii, delay_wf,'%sH'%lendelaydat))
                ii+=1        
        wlist = list(self._packed_waveforms.keys())
        wlist.sort()
        for wf in wlist:
            wfdat = self._packed_waveforms[wf]
            lenwfdat = len(wfdat)
            #print 'WAVEFORM_NAME_%s: '%ii, wf, 'len: ',len(wfdat)
            record_str.write(self.pack_record('WAVEFORM_NAME_%s'%ii, wf+'\x00','%ss'%len(wf+'\x00'))+\
                        self.pack_record('WAVEFORM_TYPE_%s'%ii, 1,'h')+\
                        self.pack_record('WAVEFORM_LENGTH_%s'%ii,lenwfdat,'l')+\
                        self.pack_record('WAVEFORM_TIMESTAMP_%s'%ii, timetuple[:-1],'8H')+\
                        self.pack_record('WAVEFORM_DATA_%s'%ii, wfdat,'%sH'%lenwfdat))
            ii+=1
        kk=1
        #nrep = self._awg_nrep
        
        seq_record_str = StringIO()
        for segment in self._filename_list.transpose():
            seq_record_str.write(
                    self.pack_record('SEQUENCE_WAIT_%s'%kk, trig_wait[kk-1],'h')+\
                            self.pack_record('SEQUENCE_LOOP_%s'%kk, int(nrep[kk-1]),'l')+\
                            self.pack_record('SEQUENCE_JUMP_%s'%kk, jump_to[kk-1],'h')+\
                            self.pack_record('SEQUENCE_GOTO_%s'%kk, goto_state[kk-1],'h')
                            )
            for wfname in segment:
                
                if wfname is not None:
                    
                    ch = wfname[-1]
                    #print wfname,'SEQUENCE_WAVEFORM_NAME_CH_'+ch+'_%s'%kk
                    seq_record_str.write(
                            self.pack_record('SEQUENCE_WAVEFORM_NAME_CH_'+ch+'_%s'%kk, wfname+'\x00','%ss'%len(wfname+'\x00'))
                            )
            kk+=1
        #self._record_str = record_str
        #self._seq_record_str = seq_record_str
        #self._head = head
        self.awg_file = head+record_str.getvalue()+seq_record_str.getvalue()
    def get_attribute(self, att_name):
        exec('retval = self.%s'%att_name)
        return retval
    def get_awg_file(self):
        return self.awg_file
    def pack_record(self,name,value,dtype):
        '''
        packs awg_file record structure: '<I(lenname)I(lendat)s[data of dtype]'
        '''
        #print name,dtype
    
        if len(dtype)==1:
            #print 'dtype:1'
            dat = struct.pack('<'+dtype,value)
            lendat=len(dat)
            #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vali: ',value
        else:
            #print 'dtype:>1'
            if dtype[-1] == 's':
                dat = struct.pack(dtype,value)
                lendat = len(dat)
                #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vals: ',len(value)
            else:
                #print tuple(value)
                dat = struct.pack('<'+dtype,*tuple(value))
                lendat = len(dat)
                #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vals: ',len(value)
        #print lendat
        return struct.pack('<II',len(name+'\x00'),lendat) + name + '\x00' + dat
    
    
    
    def set_sequence(self,packed_waveforms,wfname_l,delay_labs, nrep_l, wait_l, goto_l, logic_jump_l):
        '''
        sets the AWG in sequence mode and loads waveforms into the sequence.
        wfname_l = list of waveform names [[wf1_ch1,wf2_ch1..],[wf1_ch2,wf2_ch2..],...]
        nrep_l = list specifying the number of reps for each seq element
        wait_l = idem for wait_trigger_state
        goto_l = idem for goto_state (goto is the element where it hops to in case the element is finished)
        logic_jump_l = idem for event_jump_to (event or logic jump is the element where it hops in case of an event)

        '''
        n_ch = len(wfname_l)
        chi=0
        group=[]
        len_sq = len(nrep_l)
        
        for k in arange(len(self._AWG_list)):
            AWG = self._AWG_list[k]
            ch = self._AWG_properties[k]['channels']
            if AWG.get_type() =='Tektronix_AWG5014':
                self.generate_awg_file(packed_waveforms,wfname_l[chi:(chi+ch)],delay_labs[chi:(chi+ch)], nrep_l, wait_l, goto_l, logic_jump_l)
                AWG.set_awg_file('seq_ch_%s_to_%s.awg'%(chi,ch),self.awg_file)
            else:
                print(AWG.get_type(), ' not supported, sequence not loaded!!!')
                
            chi=ch

### Utils
    def upload_AWG_delay(self, delay_time):
        '''
        set_awg_delay seq element in ns
        '''
        n_ch=self.get_number_of_channels()
        del_n = self._clock* delay_time*1e-9
        wf=numpy.array([int(del_n)*[0.], int(del_n)*[0],int(del_n)*[0]])
        name= 'delay_%ius'%(delay_time*1e-3)
        # for awg in self._AWG_list:
            # awg.send_waveform(wf[0],wf[1],wf[2], name , self._clock)
            # if awg.get_type() =='Tektronix_AWG5014':
                # awg.import_waveform_file(name, name)
            
        self._delay_wfname = name
        return self._delay_wfname,wf[0]

                
                

                    



        


    



