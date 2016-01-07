# Tektronix_AWG5014.py class, to perform the communication between the Wrapper
# and the device
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

import numpy as np
import struct
from time import sleep, time, localtime
from io import StringIO
import os

# load the qcodes path, until we have this installed as a package
import sys
qcpath = 'D:\GitHubRepos\Qcodes'
if qcpath not in sys.path:
    sys.path.append(qcpath)

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class Tektronix_AWG5014(VisaInstrument):
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
    26-11-2008 by Gijs: Copied this plugin from the 520 and added support for
        2 more channels, added setget marker delay functions and increased max
        sampling freq to 1.2 	GS/s
    28-11-2008 ''  '' : Added some functionality to manipulate and manoeuvre
        through the folders on the AWG
    8-8-2015 by Adriaan : Merging the now diverged versions of this driver from
        the Diamond and Transmon groups @ TUD
    7-1-2016 Converted to use with QCodes
    '''
    AWG_FILE_FORMAT_HEAD = {
        'SAMPLING_RATE': 'd',    # d
        'REPETITION_RATE': 'd',    # # NAME?
        'HOLD_REPETITION_RATE': 'h',    # True | False
        'CLOCK_SOURCE': 'h',    # Internal | External
        'REFERENCE_SOURCE': 'h',    # Internal | External
        'EXTERNAL_REFERENCE_TYPE': 'h',    # Fixed | Variable
        'REFERENCE_CLOCK_FREQUENCY_SELECTION':'h',
        'REFERENCE_MULTIPLIER_RATE': 'h',    #
        'DIVIDER_RATE': 'h',   # 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256
        'TRIGGER_SOURCE': 'h',    # Internal | External
        'INTERNAL_TRIGGER_RATE': 'd',    #
        'TRIGGER_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'TRIGGER_INPUT_SLOPE': 'h',    # Positive | Negative
        'TRIGGER_INPUT_POLARITY': 'h',    # Positive | Negative
        'TRIGGER_INPUT_THRESHOLD': 'd',    #
        'EVENT_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'EVENT_INPUT_POLARITY': 'h',    # Positive | Negative
        'EVENT_INPUT_THRESHOLD': 'd',
        'JUMP_TIMING': 'h',    # Sync | Async
        'INTERLEAVE': 'h',    # On |  This setting is stronger than .
        'ZEROING': 'h',    # On | Off
        'COUPLING': 'h',    # The Off | Pair | All setting is weaker than .
        'RUN_MODE': 'h',    # Continuous | Triggered | Gated | Sequence
        'WAIT_VALUE': 'h',    # First | Last
        'RUN_STATE': 'h',    # On | Off
        'INTERLEAVE_ADJ_PHASE': 'd',
        'INTERLEAVE_ADJ_AMPLITUDE': 'd',
    }
    AWG_FILE_FORMAT_CHANNEL = {
        'OUTPUT_WAVEFORM_NAME_N': 's',  # Include NULL.(Output Waveform Name for Non-Sequence mode)
        'CHANNEL_STATE_N': 'h',  # On | Off
        'ANALOG_DIRECT_OUTPUT_N': 'h',  # On | Off
        'ANALOG_FILTER_N': 'h',  # Enum type.
        'ANALOG_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        'ANALOG_AMPLITUDE_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'ANALOG_OFFSET_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'ANALOG_HIGH_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'ANALOG_LOW_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_SKEW_N': 'd',
        'MARKER1_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        'MARKER1_AMPLITUDE_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'MARKER1_OFFSET_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'MARKER1_HIGH_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_LOW_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_SKEW_N': 'd',
        'MARKER2_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        'MARKER2_AMPLITUDE_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'MARKER2_OFFSET_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'MARKER2_HIGH_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_LOW_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        'DIGITAL_AMPLITUDE_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'DIGITAL_OFFSET_N': 'd',  # When the Input Method is High/Low, it is skipped.
        'DIGITAL_HIGH_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_LOW_N': 'd',  # When the Input Method is Amplitude/Offset, it is skipped.
        'EXTERNAL_ADD_N': 'h',  # AWG5000 only
        'PHASE_DELAY_INPUT_METHOD_N':   'h',  # Phase/DelayInme/DelayInints
        'PHASE_N': 'd',  # When the Input Method is not Phase, it is skipped.
        'DELAY_IN_TIME_N': 'd',  # When the Input Method is not DelayInTime, it is skipped.
        'DELAY_IN_POINTS_N': 'd',  # When the Input Method is not DelayInPoint, it is skipped.
        'CHANNEL_SKEW_N': 'd',
        'DC_OUTPUT_LEVEL_N': 'd',  # V
    }

    def __init__(self, name, setup_folder, address, reset=False,
                 clock=1e9, numpoints=1000):
        '''
        Initializes the AWG5014.

        Input:
            name (string)           : name of the instrument
            setup_folder (string)   : folder where externally generate seqs
                                        are stored
            address (string)        : GPIB address
            reset (bool)            : resets to default values, default=false
            numpoints (int)         : sets the number of datapoints

        Output:
            None
        '''
        super().__init__(name, address)

        self._address = address
        # if address[:5] == 'TCPIP':
        #     self.visa_handle = SocketVisa('192.168.0.3', 4000)
        # else:
        #     self.visa_handle = visa.instrument(address, timeout=10)
        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._numpoints = numpoints

        self.add_parameter('IDN', get_cmd='*IDN?')
        self.add_function('reset', call_cmd='*RST')

        self.add_parameter('state',
                           get_cmd=self.get_state)
        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Strings(
                                options=['CONT', 'TRIG', 'SEQ', 'GAT']))
        # Trigger parameters #
        # ! Warning this is the same as run mode, do not use! exists
        # Solely for legacy purposes
        self.add_parameter('trigger_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Strings(
                                options=['CONT', 'TRIG', 'SEQ', 'GAT']))
        self.add_parameter('trigger_impedance',
                           label='Trigger impedance (Ohm)',
                           get_cmd='TRIG:IMP?',
                           set_cmd='TRIG:IMP '+'{}',
                           vals=vals.Ints(),
                           parse_function=float)  # options 50 and 1000 val not implemented
        self.add_parameter('trigger_level',
                           label='Trigger level (V)',
                           get_cmd='TRIG:LEV?',
                           set_cmd='TRIG:LEV '+'{:.3f}',
                           vals=vals.Numbers(-5, 5),
                           parse_function=float)
        self.add_parameter('trigger_slope',
                           get_cmd='TRIG:SLOP?',
                           set_cmd='TRIG:SLOP '+'{}',
                           vals=vals.Strings(options=['POS', 'NEG']))
        self.add_parameter('trigger_source',
                           get_cmd='TRIG:source?',
                           set_cmd='TRIG:source '+'{}',
                           vals=vals.Strings(options=['INT', 'EXT']))
        # Event parameters #
        self.add_parameter('event_polarity',
                           get_cmd='EVEN:POL?',
                           set_cmd='EVEN:POL '+'{}',
                           vals=vals.Strings(options=['POS', 'NEG']))
        self.add_parameter('event_impedance',
                           label='Event impedance (Ohm)',
                           get_cmd='EVEN:IMP?',
                           set_cmd='EVEN:IMP '+'{}',
                           vals=vals.Ints(),
                           parse_function=float)  # options 50 and 1000 val not implemented
        self.add_parameter('event_level',
                           label='Event level (V)',
                           get_cmd='EVEN:LEV?',
                           set_cmd='EVEN:LEV '+'{:.3f}',
                           vals=vals.Numbers(-5, 5),
                           parse_function=float)

        # self.add_parameter(
        #     'clock', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     minval=1e6, maxval=1.2e9, units='Hz')

        # # Numpoints was commented out, do not know why as it is used...
        # self.add_parameter(
        #     'numpoints', type=int,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     minval=100, maxval=1e9, units='Int')

        # self.add_parameter('setup_filename', type=bytes,
        #                    flags=Instrument.FLAG_GETSET)

        # self.add_parameter(
        #     'amplitude', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4),
        #     minval=0.02,
        #     maxval=1.5,
        #     units='Volts',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'offset', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4),
        #     minval=-0.1,
        #     maxval=0.1,
        #     units='Volts',
        #     channel_prefix='ch%d_')

        # # Marker parameters
        # self.add_parameter(
        #     'marker1_low', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'marker1_high', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'marker1_delay', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=0, maxval=1, units='ns',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'marker2_low', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'marker2_high', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=-2.7, maxval=2.7, units='Volts',
        #     channel_prefix='ch%d_')
        # self.add_parameter(
        #     'marker2_delay', type=float,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #     channels=(1, 2, 3, 4), minval=0, maxval=1, units='ns',
        #     channel_prefix='ch%d_')

        # self.add_parameter('status', type=bytes,
        #                    flags=Instrument.FLAG_GETSET,
        #                    channels=(1, 2, 3, 4),
        #                    channel_prefix='ch%d_')

        # # Add functions
        # self.add_function('reset')
        # self.add_function('get_all')
        # self.add_function('clear_waveforms')
        # self.add_function('delete_all_waveforms_from_list')
        # self.add_function('set_trigger_mode_on')
        # self.add_function('set_trigger_mode_off')
        # self.add_function('set_trigger_impedance_1e3')
        # self.add_function('set_trigger_impedance_50')
        # self.add_function('import_waveform_file')
        # self.add_function('import_and_load_waveform_file_to_channel')
        # self.add_function('set_filename')
        # self.add_function('get_filenames')
        # self.add_function('send_waveform')
        # self.add_function('resend_waveform')
        # self.add_function('set_sqel_goto_state')
        # self.add_function('set_sqel_waveform')
        # self.add_function('set_sqel_loopcnt')
        # self.add_function('set_sqel_trigger_wait')
        # self.add_function('set_sqel_goto_target_index')
        # self.add_function('set_sqel_loopcnt_to_inf')
        # self.add_function('set_sqel_event_jump_type')
        # self.add_function('set_sqel_event_jump_target_index')
        # self.add_function('start')
        # self.add_function('stop')
        # self.add_function('set_sq_length')
        # self.add_function('get_state')
        # self.add_function('set_event_jump_timing')
        # self.add_function('get_event_jump_timing')
        # self.add_function('generate_awg_file')
        # self.add_function('send_awg_file')
        # self.add_function('load_awg_file')
        # self.add_function('get_error')
        # self.add_function('pack_waveform')
        # self.add_function('clear_visa')
        # self.add_function('initialize_dc_waveforms')

        # # Setup filepaths
        # self._file_path = "C:\\waveforms\\"
        # self._rem_file_path = "Z:\\waveforms\\"

        # self._setup_folder = setup_folder
        # self.set_current_folder_name(self._file_path)
        # self.get_all()
        # self.set_trigger_impedance(50)
        # if reset:
        #     self.reset()
        # else:
        #     self.get_all()

        # Commented out command from transmon version don't know if needed

        # self.visa_handle.write('mmem:CDIrectory "\\wfs"')
        # NOTE! this directory has to exist on the AWG!!
        print('Connected to \n', self.get('IDN').replace(',', '\n'))

    # Functions

    def start(self):
        self.visa_handle.write('AWGC:RUN')
        return self.get_state()

    def stop(self):
        self.visa_handle.write('AWGC:STOP')

    def get_folder_contents(self):
        return self.visa_handle.ask('mmem:cat?')

    def get_current_folder_name(self):
        return self.visa_handle.ask('mmem:cdir?')

    def set_current_folder_name(self, file_path):
        self.visa_handle.write('mmem:cdir "%s"' % file_path)

    def change_folder(self, dir):
        self.visa_handle.write('mmem:cdir "\%s"' %dir)

    def goto_root(self):
        self.visa_handle.write('mmem:cdir "c:\\.."')

    def create_and_goto_dir(self, dir):
        '''
        Creates (if not yet present) and sets the current directory to <dir>
        and displays the contents

        '''

        dircheck = '%s, DIR' % dir
        if dircheck in self.get_folder_contents():
            self.change_folder(dir)
            logging.debug(__name__ + ' :Directory already exists')
            print('Directory already exists, changed path to %s' % dir)
            print('Contents of folder is %s' % self.visa_handle.ask(
                'mmem:cat?'))
        elif self.get_current_folder_name() == '"\\%s"' % dir:
            print('Directory already set to %s' % dir)
        else:
            self.visa_handle.write('mmem:mdir "\%s"' % dir)
            self.visa_handle.write('mmem:cdir "\%s"' % dir)
            return self.get_folder_contents()

    def set_all_channels_on(self):
        for i in range(1, 5):
            exec('self.set_ch%d_status("On")' % i)

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

        self.get_trigger_mode()
        self.get_trigger_impedance()
        self.get_trigger_level()
        self.get_numpoints()
        self.get_clock()

        for i in range(1, 5):
            self.get('ch%d_amplitude' % i)
            self.get('ch%d_offset' % i)
            self.get('ch%d_marker1_low' % i)
            self.get('ch%d_marker1_high' % i)
            self.get('ch%d_marker1_delay' % i)
            self.get('ch%d_marker2_low' % i)
            self.get('ch%d_marker2_high' % i)
            self.get('ch%d_marker2_delay' % i)
            self.get('ch%d_status' % i)
        self.get('run_mode')
        self.get('setup_filename')

    def clear_waveforms(self):
        '''
        Clears the waveform on all channels.

        Input:
            None
        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear waveforms from channels')
        self.visa_handle.write('SOUR1:FUNC:USER ""')
        self.visa_handle.write('SOUR2:FUNC:USER ""')
        self.visa_handle.write('SOUR3:FUNC:USER ""')
        self.visa_handle.write('SOUR4:FUNC:USER ""')

    def get_sequence_length(self):
        return float(self.visa_handle.ask('SEQuence:LENGth?'))

    def get_refclock(self):
        '''
        Asks AWG whether the 10 MHz reference is set to the
        internal source or an external one.
        Input:
            None

        Output:
            'INT' or 'EXT'
        '''
        self.visa_handle.ask('AWGC:CLOC:SOUR?')

    def set_refclock_ext(self):
        '''
        Sets the reference clock to internal or external.
        '''
        self.visa_handle.write('AWGC:CLOC:SOUR EXT')

    def set_refclock_int(self):
        '''
        Sets the reference clock to internal or external
        '''
        self.visa_handle.write('AWGC:CLOC:SOUR INT')



    ##############
    # Parameters #
    ##############


    #
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
        logging.debug(__name__ + ' : Trying to set numpoints to %s' % numpts)

        warn_string = ' : changing numpoints. This will clear all waveforms!'
        if numpts != self._numpoints:
            logging.warning(__name__ + warn_string)
        print(__name__ + warn_string)
        # Extra print cause logging.warning does not print
        response = input('type "yes" to continue')
        if response == 'yes':
            logging.debug(__name__ + ' : Setting numpoints to %s' % numpts)
            self._numpoints = numpts
            self.clear_waveforms()
        else:
            print('aborted')

    # Sequences section
    def force_trigger_event(self):
        self.visa_handle.write('TRIG:IMM')

    def force_event(self):
        self.visa_handle.write('EVEN:IMM')

    def set_sqel_event_target_index_next(self, element_no):
        self.visa_handle.write('SEQ:ELEM%s:JTARGET:TYPE NEXT' % element_no)

    def set_sqel_event_target_index(self, element_no, index):
        self.visa_handle.write('SEQ:ELEM%s:JTARGET:INDEX %s' % (
                                   element_no, index))

    def set_sqel_goto_target_index(self, element_no, goto_to_index_no):
        self.visa_handle.write('SEQ:ELEM%s:GOTO:IND  %s' % (
                                   element_no, goto_to_index_no))

    def set_sqel_goto_state(self, element_no, goto_state):
        self.visa_handle.write('SEQuence:ELEMent%s:GOTO:STATe %s' % (
                                   element_no, int(goto_state)))

    def set_sqel_loopcnt_to_inf(self, element_no, state=True):
        self.visa_handle.write('seq:elem%s:loop:inf %s' % (
                                   element_no, int(state)))

    def get_sqel_loopcnt(self, element_no=1):
        return self.visa_handle.ask('SEQ:ELEM%s:LOOP:COUN?' % (
                                        element_no))

    def set_sqel_loopcnt(self, loopcount, element_no=1):
        self.visa_handle.write('SEQ:ELEM%s:LOOP:COUN %s' % (
                                   element_no, loopcount))

    def set_sqel_waveform(self, waveform_name, channel, element_no=1):
        self.visa_handle.write('SEQ:ELEM%s:WAV%s "%s"' % (
                                   element_no, channel, waveform_name))

    def get_sqel_waveform(self, channel, element_no=1):
        return self.visa_handle.ask('SEQ:ELEM%s:WAV%s?' % (
                                        element_no, channel))

    def set_sqel_trigger_wait(self, element_no, state=1):
        self.visa_handle.write('SEQ:ELEM%s:TWA %s' % (
                                   element_no, state))
        return self.get_sqel_trigger_wait(element_no)

    def get_sqel_trigger_wait(self, element_no):
        return self.visa_handle.ask('SEQ:ELEM%s:TWA?' % (
                                        element_no))

    def get_sq_length(self):
        return self.visa_handle.ask('SEQ:LENG?')

    def set_sq_length(self, seq_length):
        self.visa_handle.write('SEQ:LENG %s' % seq_length)

    def set_sqel_event_jump_target_index(self, element_no, jtar_index_no):
        self.visa_handle.write('SEQ:ELEM%s:JTAR:INDex %s' % (
                                   element_no, jtar_index_no))

    def set_sqel_event_jump_type(self, element_no,jtar_state):
        self.visa_handle.write('SEQuence:ELEMent%s:JTAR:TYPE %s' % (
                                   element_no, jtar_state))

    def get_sq_mode(self):
        return self.visa_handle.ask('AWGC:SEQ:TYPE?')

    def get_sq_position(self):
        return self.visa_handle.ask('AWGC:SEQ:POS?')

    def sq_forced_jump(self, jump_index_no):
        self.visa_handle.write('SEQ:JUMP:IMM %s' % jump_index_no)

    def set_event_jump_timing(self, mode):
        self.visa_handle.write('EVEN:JTIM %s' % (mode))

    def get_event_jump_timing(self):
        return self.visa_handle.ask('EVEN:JTIM?')

    #################################
    # Transmon version file loading #
    #################################

    def load_and_set_sequence(self, wfname_l, nrep_l, wait_l, goto_l,
                              logic_jump_l):
        '''
        sets the AWG in sequence mode and loads waveforms into the sequence.
        wfname_l = list of waveform names [[wf1_ch1,wf2_ch1..],[wf1_ch2,wf2_ch2..],...],
                    waveforms are assumed to be already present and imported in AWG (see send_waveform and
                    import_waveform_file)
        nrep_l = list specifying the number of reps for each seq element
        wait_l = idem for wait_trigger_state
        goto_l = idem for goto_state (goto is the element where it hops to in case the element is finished)
        logic_jump_l = idem for event_jump_to (event or logic jump is the element where it hops in case of an event)

        '''
        self._load_new_style(wfname_l, nrep_l, wait_l, goto_l, logic_jump_l)

    def _load_new_style(self, wfname_l, nrep_l, wait_l, goto_l, logic_jump_l):
        '''
        load sequence not using sequence file
        '''
        self.set_sq_length(0) # delete prev seq
        #print wfname_l
        len_sq = len(nrep_l)
        self.set_sq_length(len_sq)
        n_ch = len(wfname_l)
        for k in range(len_sq):
            #wfname_l[k]
            #print k
            for n in range(n_ch):
                #print n
                #print wfname_l[n][k]
                if wfname_l[n][k] is not None:

                    self.set_sqel_waveform(wfname_l[n][k], n+1, k+1)
            self.set_sqel_trigger_wait(k+1, int(wait_l[k]!=0))
            self.set_sqel_loopcnt_to_inf(k+1, False)
            self.set_sqel_loopcnt(nrep_l[k],k+1)
            qt.msleep()
            if  goto_l[k] == 0:
                self.set_sqel_goto_state(k+1, False)
            else:
                self.set_sqel_goto_state(k+1, True)
                self.set_sqel_goto_target_index(k+1, goto_l[k])
            if logic_jump_l[k] == -1:
                self.set_sqel_event_target_index_next(k+1)
            else:
                self.set_sqel_event_target_index(k+1, logic_jump_l[k])

    def _load_old_style(self, wfs, rep, wait, goto, logic_jump, filename):
        '''
        Sends a sequence file (for the moment only for ch1
        Inputs (mandatory):

           wfs:  list of filenames

        Output:
            None
        This needs to be written so that awg files are loaded, will be much faster!
        '''
        pass

    ##################################################################

    def _do_get_clock(self):
        '''
        Returns the clockfrequency, which is the rate at which the datapoints
        are sent to the designated output
        Corresponds to SAMPLING_RATE

        Input:
            None
        Output:
            clock (int) : frequency in Hz
        '''
        return self._clock

    def _do_set_clock(self, clock):
        '''
        Sets the rate at which the datapoints are sent to the designated output
        channel

        Input:
            clock (int) : frequency in Hz
        Output:
            None
        '''
        '''logging.warning(__name__ + ' : Clock set to %s. This is not fully functional yet. To avoid problems, it is better not to change the clock during operation' % clock)'''''
        self._clock = clock
        self.visa_handle.write('SOUR:FREQ %f' % clock)

    def import_waveform_file(self, waveform_listname, waveform_filename,
                             type='wfm'):
        self._import_waveform_file(waveform_listname, waveform_filename)

    def _import_waveform_file(self, waveform_listname, waveform_filename,
                              type='wfm'):
        self.visa_handle.write('mmem:imp "%s","%s",%s'%(waveform_listname, waveform_filename, type))

    def import_and_load_waveform_file_to_channel(self, channel_no,
                                                 waveform_listname,
                                                 waveform_filename,
                                                 type='wfm'):
        self._import_and_load_waveform_file_to_channel(channel_no,
                                                       waveform_listname,
                                                       waveform_filename)

    def _import_and_load_waveform_file_to_channel(self, channel_no,
                                                  waveform_listname,
                                                  waveform_filename, type='wfm'):
        self.visa_handle.write('mmem:imp "%s","%s",%s' % (
                                   waveform_listname, waveform_filename, type))
        self.visa_handle.write('sour%s:wav "%s"' % (
                                   channel_no, waveform_listname))
        i = 0
        while not (self.visa_handle.ask("sour%s:wav?" % channel_no)
                   == '"%s"' % waveform_listname):
            sleep(0.01)
            i = i+1
            print(i)
        return 1
    ######################
    # AWG file functions #
    ######################

    def _pack_record(self, name, value, dtype):
        '''
        packs awg_file record structure: '<I(lenname)I(lendat)s[data of dtype]'
        The file record format is as follows:

        Record Name Size
        (32-bit unsigned integer)
        Record Data Size
        (32-bit unsigned integer)
        Record Name (ASCII)
        (Include NULL.)
        Record Data

        '''
        if len(dtype) == 1:
            #print 'dtype:1'
            dat = struct.pack('<'+dtype, value)
            lendat = len(dat)
            #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vali: ',value
        else:
            #print 'dtype:>1'
            if dtype[-1] == 's':
                dat = struct.pack(dtype, value)
                lendat = len(dat)
                #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vals: ',len(value)
            else:
                #print tuple(value)
                dat = struct.pack('<'+dtype, *tuple(value))
                lendat = len(dat)
                #print 'name: ',name, 'dtype: ',dtype, 'len: ',lendat, 'vals: ',len(value)
        #print lendat
        return struct.pack('<II', len(name+'\x00'),
                           lendat) + name + '\x00' + dat

    def generate_sequence_cfg(self):
        '''
        This function is used to generate a config file, that is used when
        generating sequence files, from existing settings in the awg.
        Querying the AWG for these settings takes ~0.7 seconds
        '''
        print('Generating sequence_cfg')

        AWG_sequence_cfg = {
            'SAMPLING_RATE': self.get_clock(),
            'CLOCK_SOURCE': (1 if self.query_visa('AWGC:CLOCK:SOUR?') == 'INT'
                             else 2),  # Internal | External
            'REFERENCE_SOURCE':   2,  # Internal | External
            'EXTERNAL_REFERENCE_TYPE':   1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,
            # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE':   1 if self.get_trigger_source() == 'EXT' else 2,
            # External | Internal
            'TRIGGER_INPUT_IMPEDANCE': (1 if self.get_trigger_impedance() ==
                                        50. else 2),  # 50 ohm | 1 kohm
            'TRIGGER_INPUT_SLOPE': (1 if self.get_trigger_slope() ==
                                    'POS' else 2),  # Positive | Negative
            'TRIGGER_INPUT_POLARITY': (1 if self.query_visa('TRIG:POL?') ==
                                       'POS' else 2),  # Positive | Negative
            'TRIGGER_INPUT_THRESHOLD':  self.get_trigger_level(),  # V
            'EVENT_INPUT_IMPEDANCE':   (1 if self.get_event_impedance() ==
                                        50. else 2),  # 50 ohm | 1 kohm
            'EVENT_INPUT_POLARITY':  (1 if self.get_event_polarity() == 'POS'
                                      else 2),  # Positive | Negative
            'EVENT_INPUT_THRESHOLD':   self.get_event_level(),  # V
            'JUMP_TIMING':   (1 if self.get_event_jump_timing() == 'SYNC'
                              else 2),  # Sync | Async
            'RUN_MODE':   4,  # Continuous | Triggered | Gated | Sequence
            'RUN_STATE':  0,  # On | Off
            }
        return AWG_sequence_cfg

    def generate_awg_file(self,
                          packed_waveforms, wfname_l, nrep, trig_wait,
                          goto_state, jump_to, channel_cfg, sequence_cfg=None):
        '''
        packed_waveforms: dictionary containing packed waveforms with keys
                            wfname_l and delay_labs
        wfname_l: array of waveform names array([[segm1_ch1,segm2_ch1..],
                                                [segm1_ch2,segm2_ch2..],...])
        nrep_l: list of len(segments) specifying the no of
                    reps per segment (0,65536)
        wait_l: list of len(segments) specifying triger wait state (0,1)
        goto_l: list of len(segments) specifying goto state (0, 65536),
                    0 means next)
        logic_jump_l: list of len(segments) specifying logic jump (0 = off)
        channel_cfg: dictionary of valid channel configuration records
        sequence_cfg: dictionary of valid head configuration records
                     (see AWG_FILE_FORMAT_HEAD)
                     When an awg file is uploaded these settings will be set
                     onto the AWG, any paramter not specified will be set to
                     its default value (even overwriting current settings)

        for info on filestructure and valid record names, see AWG Help,
        File and Record Format
        '''
        wfname_l
        timetuple = tuple(np.array(localtime())[[0, 1, 8, 2, 3, 4, 5, 6, 7]])

        # general settings
        head_str = StringIO()
        head_str.write(self._pack_record('MAGIC', 5000, 'h') +
                       self._pack_record('VERSION', 1, 'h'))
        if sequence_cfg is None:
            sequence_cfg = self.generate_sequence_cfg()

        for k in list(sequence_cfg.keys()):
            if k in self.AWG_FILE_FORMAT_HEAD:
                head_str.write(self._pack_record(k, sequence_cfg[k],
                               self.AWG_FILE_FORMAT_HEAD[k]))
            else:
                logging.warning('AWG: ' + k +
                                ' not recognized as valid AWG setting')
        # channel settings
        ch_record_str = StringIO()
        for k in list(channel_cfg.keys()):
            ch_k = k[:-1] + 'N'
            if ch_k in self.AWG_FILE_FORMAT_CHANNEL:
                ch_record_str.write(self._pack_record(k, channel_cfg[k],
                                    self.AWG_FILE_FORMAT_CHANNEL[ch_k]))
            else:
                logging.warning('AWG: ' + k +
                                ' not recognized as valid AWG channel setting')
        # waveforms
        ii = 21
        wf_record_str = StringIO()
        wlist = list(packed_waveforms.keys())
        wlist.sort()
        for wf in wlist:
            wfdat = packed_waveforms[wf]
            lenwfdat = len(wfdat)
            # print 'WAVEFORM_NAME_%s: '%ii, wf, 'len: ',len(wfdat)
            wf_record_str.write(
                self._pack_record('WAVEFORM_NAME_%s' % ii, wf + '\x00',
                                  '%ss' % len(wf+'\x00')) +
                self._pack_record('WAVEFORM_TYPE_%s' % ii, 1, 'h') +
                self._pack_record('WAVEFORM_LENGTH_%s' % ii, lenwfdat, 'l') +
                self._pack_record('WAVEFORM_TIMESTAMP_%s' % ii,
                                  timetuple[:-1], '8H') +
                self._pack_record('WAVEFORM_DATA_%s' % ii, wfdat, '%sH'
                                  % lenwfdat))
            ii += 1
        # sequence
        kk = 1
        seq_record_str = StringIO()
        for segment in wfname_l.transpose():
            seq_record_str.write(
                self._pack_record('SEQUENCE_WAIT_%s' % kk, trig_wait[kk-1],
                                  'h') +
                self._pack_record('SEQUENCE_LOOP_%s' % kk, int(nrep[kk-1]),
                                  'l') +
                self._pack_record('SEQUENCE_JUMP_%s' % kk, jump_to[kk-1],
                                  'h') +
                self._pack_record('SEQUENCE_GOTO_%s' % kk, goto_state[kk-1],
                                  'h'))
            for wfname in segment:
                if wfname is not None:
                    ch = wfname[-1]
                    # print wfname,'SEQUENCE_WAVEFORM_NAME_CH_'+ch+'_%s'%kk
                    seq_record_str.write(
                        self._pack_record('SEQUENCE_WAVEFORM_NAME_CH_' + ch
                                          + '_%s' % kk, wfname + '\x00',
                                          '%ss' % len(wfname+'\x00')))
            kk += 1
        return head_str.getvalue() + ch_record_str.getvalue() + wf_record_str.getvalue() + seq_record_str.getvalue()

    def send_awg_file(self, filename, awg_file):
        # print self.visa_handle.ask('MMEMory:CDIRectory?')
        s1 = 'MMEM:DATA "%s",' % filename
        s2 = '#' + str(len(str(len(awg_file)))) + str(len(awg_file))
        mes = s1+s2+awg_file
        self.visa_handle.write(mes)

    def load_awg_file(self, filename):
        s = 'AWGCONTROL:SRESTORE "%s"' % filename
        # print s
        self.visa_handle.write(s)

    def get_error(self):
        # print self.visa_handle.ask('AWGControl:SNAMe?')
        print(self.visa_handle.ask('SYSTEM:ERROR:NEXT?'))
        # self.visa_handle.write('*CLS')

    def pack_waveform(self, wf, m1, m2):
        '''
        packs analog waveform in 14 bit integer, and two bits for m1 and m2
        in a single 16 bit integer
        '''
        wflen = len(wf)
        packed_wf = np.zeros(wflen, dtype=np.uint16)
        packed_wf += np.uint16(np.round(wf*8191)+8191+np.round(16384*m1) + \
            np.round(32768*m2))
        if len(np.where(packed_wf == -1)[0]) > 0:
            print(np.where(packed_wf == -1))
        return packed_wf

    # END AWG file functions
    ###########################
    # Waveform file functions #
    ###########################

        # Send waveform to the device
    def send_waveform(self, w, m1, m2, filename, clock=None):
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
        logging.debug(__name__ + ' : Sending waveform %s to instrument' %
                      filename)
        # Check for errors
        dim = len(w)

        if (not((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            return 'error'

        self._values['files'][filename] = {}
        self._values['files'][filename]['w'] = w
        self._values['files'][filename]['m1'] = m1
        self._values['files'][filename]['m2'] = m2
        self._values['files'][filename]['clock'] = clock
        self._values['files'][filename]['numpoints'] = len(w)

        m = m1 + np.multiply(m2, 2)
        ws = ''
        # this is probalbly verry slow and memmory consuming!
        for i in range(0, len(w)):
            ws = ws + struct.pack('<fB', w[i], int(np.round(m[i], 0)))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        if clock is not None:
            s6 = 'CLOCK %.10e\n' % clock
        else:
            s6 = ''

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6

        self.visa_handle.write(mes)

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
        filename = self._values['recent_channel_%s' % channel]['filename']
        logging.debug(__name__ + ' : Resending %s to channel %s' %
                      (filename, channel))

        if (w == []):
            w = self._values['recent_channel_%s' % channel]['w']
        if (m1 == []):
            m1 = self._values['recent_channel_%s' % channel]['m1']
        if (m2 == []):
            m2 = self._values['recent_channel_%s' % channel]['m2']
        if (clock == []):
            clock = self._values['recent_channel_%s' % channel]['clock']

        if not ((len(w) == self._numpoints) and (len(m1) == self._numpoints)
                and (len(m2) == self._numpoints)):
            logging.error(__name__ + ' : one (or more) lengths of waveforms do not match with numpoints')

        self.send_waveform(w, m1, m2, filename, clock)
        self.set_filename(filename, channel)

    def set_filename(self, name, channel):
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
            logging.debug(__name__  + ' : File exists in local memory')
            self._values['recent_channel_%s' %channel] = self._values['files'][name]
            self._values['recent_channel_%s' %channel]['filename'] = name
        else:
            logging.debug(__name__  + ' : File does not exist in memory, \
            reading from instrument')
            lijst = self.visa_handle.ask('MMEM:CAT? "MAIN"')
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
            data = self.visa_handle.ask('MMEM:DATA? "%s"' %name)

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
            self.visa_handle.write('SOUR%s:WAV "%s"' % (channel, name))
        else:
            logging.warning(__name__  + ' : Verkeerde lengte %s ipv %s'
                %(self._values['files'][name]['numpoints'], self._numpoints))

    # End of waveform file functions

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
        val=  (self.visa_handle.ask('SOUR%s:VOLT:LEV:IMM:AMPL?' % channel))
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
        self.visa_handle.write('SOUR%s:VOLT:LEV:IMM:AMPL %.6f' % (channel, amp))

    def _do_get_offset(self, channel):
        '''
        Reads the offset of the designated channel of the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            offset (float) : offset of designated channel in Volts
        '''
        logging.debug(__name__ + ' : Get offset of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:VOLT:LEV:IMM:OFFS?' % channel))

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
        self.visa_handle.write('SOUR%s:VOLT:LEV:IMM:OFFS %.6f' % (channel, offset))

    def _do_get_marker1_low(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker1 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK1:VOLT:LEV:IMM:LOW?' % channel))

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
        self.visa_handle.write('SOUR%s:MARK1:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def _do_get_marker1_high(self, channel):
        '''
        Gets the high level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker1 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH?' % channel))

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
        self.visa_handle.write('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def _do_get_marker1_delay(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        logging.debug(__name__ + ' : Get delay of marker1 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK1:DEL?' % channel))*1e9

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
        self.visa_handle.write('SOUR%s:MARK1:DEL %.3fe-9' % (channel, delay))

    def _do_get_marker2_low(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker2 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK2:VOLT:LEV:IMM:LOW?' % channel))

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
        self.visa_handle.write('SOUR%s:MARK2:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def _do_get_marker2_high(self, channel):
        '''
        Gets the high level for marker2 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker2 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH?' % channel))

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
        self.visa_handle.write('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def _do_get_marker2_delay(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
           delay(float) : delay in seconds
        '''
        logging.debug(__name__ + ' : Get delay of marker1 of channel %s' %channel)
        return float(self.visa_handle.ask('SOUR%s:MARK2:DEL?' % channel))*1e9

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
        self.visa_handle.write('SOUR%s:MARK2:DEL %.3fe-9' % (channel, delay))

    def delete_all_waveforms_from_list(self):
        self.visa_handle.write('WLISt:WAVeform:DELete ALL')

    def do_get_status(self, channel):
        '''
        Gets the status of the designated channel.
        'Tektronix_AWG5014'
        Input:
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Get status of channel %s' %channel)
        outp = self.visa_handle.ask('OUTP%s?' %channel)
        if (outp=='0'):
            return 'Off'
        elif (outp=='1'):
            return 'On'
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
            self.visa_handle.write('OUTP%s ON' %channel)
        elif (status.upper()=='OFF'):
            self.visa_handle.write('OUTP%s OFF' %channel)
        else:
            logging.debug(__name__ + ' : Try to set status to invalid value %s' % status)
            print('Tried to set status to invalid value %s' %status)

    #  Ask for string with filenames
    def get_filenames(self):
        logging.debug(__name__ + ' : Read filenames from instrument')
        return self.visa_handle.ask('MMEM:CAT?')

    def set_DC_out(self, DC_channel_number, Voltage):
        self.visa_handle.write('AWGControl:DC%s:VOLTage:OFFSet %sV' %
                                   (DC_channel_number, Voltage))

    def get_DC_out(self, DC_channel_number):
        return self.visa_handle.ask('AWGControl:DC%s:VOLTage:OFFSet?' %
                                        (DC_channel_number))

    def send_DC_pulse(self, DC_channel_number, Amplitude, length):
        '''
        sends a (slow) pulse on the DC channel specified
        Ampliude: voltage level
        length: seconds
        '''
        restore=self.get_DC_out(DC_channel_number)
        self.set_DC_out(DC_channel_number, Amplitude)
        sleep(length)
        self.set_DC_out(DC_channel_number, restore)

    def set_DC_state(self, state=False):
        self.visa_handle.write('AWGControl:DC:state %s' % (int(state)))
    # Send waveform to the device (from transmon driver)

    def upload_awg_file(self, fname, fcontents):
        t0 = time()
        self._rem_file_path
        floc = self._rem_file_path
        f = open(floc+'\\'+fname,'wb')
        f.write(fcontents)
        f.close()
        t1 = time()-t0
        print('upload time: ',t1)
        self.get_state()
        print('setting time: ',time()-t1-t0)

    def do_get_setup_filename(self):
        return self.visa_handle.ask('AWGControl:SNAMe?')

    def _set_setup_filename(self, fname):
        folder_name = 'C:/' + self._setup_folder + '/' + fname
        self.set_current_folder_name(folder_name)
        set_folder_name = self.get_current_folder_name()
        if not os.path.split(folder_name)[1] == os.path.split(set_folder_name)[1][:-1]:
            print('Warning, unsuccesfully set AWG file', folder_name)
        print('Current AWG file set to: ', self.get_current_folder_name())
        self.visa_handle.write('AWGC:SRES "%s.awg"' % fname)
        qt.msleep(.5)  # dirty hack to see if this helps

    def do_set_setup_filename(self, fname, force_load=False):
        '''
        sets the .awg file to a .awg file that already exists in the memory of
        the AWG.

        fname (string) : file to be set.
        force_load (bool): if True, sets the file even if it is already loaded

        After setting the file it resets all the instrument settings to what
        it is set in the qtlab memory.
        '''
        cfname = self.get_setup_filename()
        if cfname.find(fname) != -1 and not force_load:
            print('file: %s already loaded' % fname)
        else:
            pars = self.get_parameters()
            self._set_setup_filename(fname)
            # self.visa_handle.ask('*OPC?')
            parkeys = list(pars.keys())
            parkeys.remove('setup_filename')
            parkeys.remove('AWG_model')
            parkeys.remove('numpoints')
            # not reset because this removes all loaded waveforms
            parkeys.remove('trigger_mode')
            # Removed trigger_mode because duplicate of run mode
            comm = False
            qt.msleep()
            for key in parkeys:
                try:
                    # print 'setting: %s' % key
                    exec('self.set_%s(pars[key]["value"])' % key)
                    comm = True
                except:
                    print(key + ' not set!')
                    comm = False

                qt.msleep(0.00005)
                # Sped up by factor 10, VISA protocol should take care of wait
                if comm:
                    self.visa_handle.ask('*OPC?')

    def is_awg_ready(self):
        self.visa_handle.ask('*OPC?')
        return True

    def send_waveform(self, w, m1, m2, filename, clock=1e9):
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

        m = m1 + np.multiply(m2,2)
        ws = ''
        for i in range(0,len(w)):
            ws = ws + struct.pack('<fB',w[i],int(np.round(m[i],0)))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen=str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6
        #print 's1: ',s1
        print('record size s2: ',s2)
        print('s3: ',s3)
        print('s4: ',s4)
        print('waveform_data')
        print('s6: ',s6)
        return mes
        self.visa_handle.write(mes)

    def send_visa_command(self, command):
        self.visa_handle.write(command)
    def query_visa(self, query):
        return self.visa_handle.ask(query)


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
        self.visa_handle.write('AWGControl:DC%s:VOLTage:OFFSet %sV'%(DC_channel_number, Voltage))
        self.get_DC_out(DC_channel_number)

    def get_DC_out(self, DC_channel_number):
        return self.visa_handle.ask('AWGControl:DC%s:VOLTage:OFFSet?'%(DC_channel_number))

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
        self.visa_handle.write('AWGControl:DC:state %s' %(int(state)))
        self.get_DC_state()

    def get_DC_state(self):
        return self.visa_handle.ask('AWGControl:DC:state?')


    def initialize_dc_waveforms(self):
        self.set_runmode('CONT')
        self.visa_handle.write('SOUR1:WAV "*DC"')
        self.visa_handle.write('SOUR2:WAV "*DC"')
        self.visa_handle.write('SOUR3:WAV "*DC"')
        self.visa_handle.write('SOUR4:WAV "*DC"')
        self.set_ch1_status('on')
        self.set_ch2_status('on')
        self.set_ch3_status('on')
        self.set_ch4_status('on')



