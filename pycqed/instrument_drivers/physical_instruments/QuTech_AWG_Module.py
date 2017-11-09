'''
File:       QuTech_AWG_Module.py
Author:     Wouter Vlothuizen, TNO/QuTech,
            edited by Adriaan Rol
Purpose:    Instrument driver for Qutech QWG
Usage:
Notes:      It is possible to view the QWG log using ssh. To do this connect
            using ssh e.g., "ssh root@192.168.0.10"
            Logging can be enabled using "tail -f /tmpLog/qwg.log"
Bugs:
'''

from .SCPI import SCPI

import numpy as np
import struct
import json
from qcodes import validators as vals
import warnings


from qcodes.instrument.parameter import StandardParameter
from qcodes.instrument.parameter import Command, no_setter


# Note: the HandshakeParameter is a temporary param that should be replaced
# once qcodes issue #236 is closed
class HandshakeParameter(StandardParameter):

    """
    If a string is specified as a set command it will append '*OPC?' and use
    instrument.ask instead of instrument.write
    """
    # pass

    def _set_set(self, set_cmd, set_parser):
        exec_str = self._instrument.ask if self._instrument else None
        if isinstance(set_cmd, str):
            set_cmd += '\n *OPC?'
        self._set = Command(arg_count=1, cmd=set_cmd, exec_str=exec_str,
                            input_parser=set_parser, no_cmd_function=no_setter)

        self.has_set = set_cmd is not None


class QuTech_AWG_Module(SCPI):

    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)

        # AWG properties
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'QWG'
        self.device_descriptor.numChannels = 4
        self.device_descriptor.numDacBits = 12
        self.device_descriptor.numMarkersPerChannel = 2
        self.device_descriptor.numMarkers = 8
        self.device_descriptor.numTriggers = 8
        # Commented out until bug fixed
        self.device_descriptor.numCodewords = 128

        # valid values
        self.device_descriptor.mvals_trigger_impedance = vals.Enum(50),
        self.device_descriptor.mvals_trigger_level = vals.Numbers(0, 5.0)
        # FIXME: not in [V]

        self.add_parameters()
        self.connect_message()

    def add_parameters(self):
        #######################################################################
        # QWG specific
        #######################################################################

        for i in range(self.device_descriptor.numChannels//2):
            ch_pair = i*2+1
            sfreq_cmd = 'qutech:output{}:frequency'.format(ch_pair)
            sph_cmd = 'qutech:output{}:phase'.format(ch_pair)
            # NB: sideband frequency has a resolution of ~0.23 Hz:
            self.add_parameter('ch_pair{}_sideband_frequency'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               unit='Hz',
                               label=('Sideband frequency channel ' +
                                      'pair {} (Hz)'.format(i)),
                               get_cmd=sfreq_cmd + '?',
                               set_cmd=sfreq_cmd + ' {}',
                               vals=vals.Numbers(-300e6, 300e6),
                               get_parser=float)
            self.add_parameter('ch_pair{}_sideband_phase'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               unit='deg',
                               label=('Sideband phase channel' +
                                      ' pair {} (deg)'.format(i)),
                               get_cmd=sph_cmd + '?',
                               set_cmd=sph_cmd + ' {}',
                               vals=vals.Numbers(-180, 360),
                               get_parser=float)

            self.add_parameter('ch_pair{}_transform_matrix'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               label=('Transformation matrix channel' +
                                      'pair {}'.format(i)),
                               get_cmd=self._gen_ch_get_func(
                self._getMatrix, ch_pair),
                set_cmd=self._gen_ch_set_func(
                self._setMatrix, ch_pair),
                # NB range is not a hardware limit
                vals=vals.Arrays(-2, 2, shape=(2, 2)))

        for i in range(1, self.device_descriptor.numTriggers+1):
            triglev_cmd = 'qutech:trigger{}:level'.format(i)
            # individual trigger level per trigger input:
            self.add_parameter('tr{}_trigger_level'.format(i),
                               unit='V',
                               label='Trigger level channel {} (V)'.format(i),
                               get_cmd=triglev_cmd + '?',
                               set_cmd=triglev_cmd + ' {}',
                               vals=self.device_descriptor.mvals_trigger_level,
                               get_parser=float)

        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMO?',
                           set_cmd='AWGC:RMO ' + '{}',
                           vals=vals.Enum('NONE', 'CONt', 'SEQ', 'CODeword'))
        # NB: setting mode "CON" (valid SCPI abbreviation) reads back as "CONt"

        # Channel parameters #
        for ch in range(1, self.device_descriptor.numChannels+1):
            amp_cmd = 'SOUR{}:VOLT:LEV:IMM:AMPL'.format(ch)
            offset_cmd = 'SOUR{}:VOLT:LEV:IMM:OFFS'.format(ch)
            state_cmd = 'OUTPUT{}:STATE'.format(ch)
            waveform_cmd = 'SOUR{}:WAV'.format(ch)
            # Set channel first to ensure sensible sorting of pars
            # Compatibility: 5014, QWG
            self.add_parameter('ch{}_state'.format(ch),
                               label='Status channel {}'.format(ch),
                               get_cmd=state_cmd + '?',
                               set_cmd=state_cmd + ' {}',
                               val_mapping={True: '1', False: '0'},
                               vals=vals.Bool())

            self.add_parameter(
                'ch{}_amp'.format(ch),
                parameter_class=HandshakeParameter,
                label='Channel {} Amplitude '.format(ch),
                unit='Vpp',
                docstring='Amplitude channel {} (Vpp into 50 Ohm)'.format(ch),
                get_cmd=amp_cmd + '?',
                set_cmd=amp_cmd + ' {:.6f}',
                vals=vals.Numbers(-1.8, 1.8),
                get_parser=float)

            self.add_parameter('ch{}_offset'.format(ch),
                               # parameter_class=HandshakeParameter,
                               label='Offset channel {}'.format(ch),
                               unit='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-.25, .25),
                               get_parser=float)

            self.add_parameter('ch{}_default_waveform'.format(ch),
                               get_cmd=waveform_cmd+'?',
                               set_cmd=waveform_cmd+' "{}"',
                               vals=vals.Strings())

        for cw in range(self.device_descriptor.numCodewords):
            for j in range(self.device_descriptor.numChannels):
                ch = j+1
                # Codeword 0 corresponds to bitcode 0
                cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
                self.add_parameter('codeword_{}_ch{}_waveform'.format(cw, ch),
                                   get_cmd=cw_cmd+'?',
                                   set_cmd=cw_cmd+' "{:s}"',
                                   vals=vals.Strings())

        # Waveform parameters
        self.add_parameter('WlistSize',
                           label='Waveform list size',
                           unit='#',
                           get_cmd='wlist:size?',
                           get_parser=int)
        self.add_parameter('Wlist',
                           label='Waveform list',
                           get_cmd=self._getWlist)
        
        self.add_parameter('get_system_status',
                           unit='JSON',
                           label=('System status'),
                           get_cmd='SYSTem:STAtus?',
                           vals=vals.Strings(),
                           get_parser=self.JSON_parser,
                           docstring='Reads the current system status. E.q. channel ' \
                             +'status: on or off, overflow, underdrive.\n' \
                             +'Return:\n     JSON object with system status')

        # Trigger parameters
        doc_trgs_log_inp = 'Reads the current input values on the all the trigger ' \
                    +'inputs.\nReturn:\n    uint32 where trigger 1 (T1) ' \
                    +'is on the Least significant bit (LSB), T2 on the second  ' \
                    +'bit after LSB, etc.\n\n For example, if only T3 is ' \
                    +'connected to a high signal, the return value is: ' \
                    +'4 (0b0000100)\n\n Note: To convert the return value ' \
                    +'to a readable ' \
                    +'binary output use: `print(\"{0:#010b}\".format(qwg.' \
                    +'triggers_logic_input()))`'
        self.add_parameter('triggers_logic_input',
                           label='Read triggers input value',
                           get_cmd='QUTEch:TRIGgers:LOGIcinput?',
                           get_parser=np.uint32, # Did not convert to readable
                                                 # string because a uint32 is more
                                                 # usefull when other logic is needed
                           docstring=doc_trgs_log_inp)


        # This command is added manually
        # self.add_function('deleteWaveform'
        self.add_function('deleteWaveformAll',
                          call_cmd='wlist:waveform:delete all')

        doc_sSG = "Synchronize both sideband frequency" \
            + " generators, i.e. restart them with their defined phases."
        self.add_function('syncSidebandGenerators',
                          call_cmd='QUTEch:OUTPut:SYNCsideband',
                          docstring=doc_sSG)


    def stop(self):
        '''
        Shutsdown output on channels. When stoped will check for errors or overflow
        '''
        self.write('awgcontrol:stop:immediate')
        self.detect_overflow()
        self.getErrors()

    # command is run but using start and stop because
    # FIXME: replace custom start function when proper error message has
    # been implemented.
    # self.add_function('start',
    #                   call_cmd='awgcontrol:run:immediate')
    def start(self):
        '''
        Activates output on channels with the current settings. When started this function will check for possible warnings
        '''
        run_mode = self.run_mode()
        if run_mode == 'NONE':
            raise RuntimeError('No run mode is specified')
        self.write('awgcontrol:run:immediate')

        self.getErrors()

        status = self.get_system_status()
        warn_msg = self.detect_underdrive(status)

        if(len(warn_msg) > 0):
            warnings.warn(', '.join(warn_msg))

    def _setMatrix(self, chPair, mat):
        '''
        Args:
            chPair(int): ckannel pair for operation, 1 or 3

            matrix(np.matrix): 2x2 matrix for mixer calibration
        '''
        # function used internally for the parameters because of formatting
        self.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(
                   chPair, mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]))

    def _getMatrix(self, chPair):
        # function used internally for the parameters because of formatting
        mstring = self.ask('qutech:output{}:matrix?'.format(chPair))
        M = np.zeros(4)
        for i, x in enumerate(mstring.split(',')):
            M[i] = x
        M = M.reshape(2, 2, order='F')
        return(M)

    def detect_overflow(self):
        '''
        Will raise an error if on a channel overflow happened
        '''
        status = self.get_system_status()
        err_msg = [];
        for channel in status["channels"]:
            if(channel["overflow"] == True):
                err_msg.append("Wave overflow detected on channel: {}".format(channel["id"]))
        if(len(err_msg) > 0):
            raise RuntimeError(err_msg)

    def detect_underdrive(self, status):
        '''
        Will raise an warning if on a channel underflow is detected
        '''
        msg = [];
        for channel in status["channels"]:
            if((channel["on"] == True) and (channel["underdrive"] == True)):
                msg.append("Possible wave underdrive detected on channel: {}".format(channel["id"]))
        return msg;

    def getErrors(self):
        '''
        The SCPI protocol by default does not return errors. Therefore the user needs
        to ask for errors. This function retrieves all errors and will raise them.
        '''
        errNr = self.getSystemErrorCount()

        if errNr > 0:
            errMgs = [];
            for i in range(errNr):
                errMgs.append(self.getError())
            raise RuntimeError(', '.join(errMgs))

    def JSON_parser(self, msg):
        '''
        Converts the result of a SCPI message to a JSON.

        msg: SCPI message where the body is a JSON
        return: JSON object with the data of the SCPI message
        '''
        result = str(msg)[1:-1]
        result = result.replace('\"\"', '\"') # SCPI/visa adds additional quotes
        return json.loads(result)

    ##########################################################################
    # AWG5014 functions: SEQUENCE
    ##########################################################################
    def setSeqLength(self, length):
        '''
        Args:
            length (int): 0..max. Allocates new, or trims existing sequence
        '''
        self.write('sequence:length %d' % length)

    def setSeqElemLoopInfiniteOn(self, element):
        '''
        Args:
            element(int): 1..length
        '''
        self.write('sequence:element%d:loop:infinite on' % element)

    ##########################################################################
    # AWG5014 functions: WLIST (Waveform list)
    ##########################################################################
    # def getWlistSize(self):
    #     return self.ask_int('wlist:size?')

    def _getWlistName(self, idx):
        '''
        Args:
            idx(int): 0..size-1
        '''
        return self.ask('wlist:name? %d' % idx)

    def _getWlist(self):
        '''
        NB: takes a few seconds on 5014: our fault or Tek's?
        '''
        size = self.WlistSize()
        wlist = []                                  # empty list
        for k in range(size):                       # build list of names
            wlist.append(self._getWlistName(k+1))
        return wlist

    def deleteWaveform(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            'test'
        '''
        self.write('wlist:waveform:delete "%s"' % name)

    def getWaveformType(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            'INT' or 'REAL'
        '''
        return self.ask('wlist:waveform:type? "%s"' % name)

    def getWaveformLength(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'
        '''
        return self.ask_int('wlist:waveform:length? "%s"' % name)

    def newWaveformReal(self, name, len):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        NB: seems to do nothing (on Tek5014) if waveform already exists
        '''
        self.write('wlist:waveform:new "%s",%d,real' % (name, len))

    def getWaveformDataFloat(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            waveform  (np.array of float): waveform data

        Compatibility: QWG
        '''
        self.write('wlist:waveform:data? "%s"' % name)
        binBlock = self.binBlockRead()
        # extract waveform
        if 1:   # high performance
            waveform = np.frombuffer(binBlock, dtype=np.float32)
        else:   # more generic
            waveformLen = int(len(binBlock)/4)   # 4 bytes per record
            waveform = np.array(range(waveformLen), dtype=float)
            for k in range(waveformLen):
                val = struct.unpack_from('<f', binBlock, k*4)
                waveform[k] = val[0]
        return waveform

    def sendWaveformDataReal(self, name, waveform):
        """
        send waveform and markers directly to AWG memory, i.e. not to a file
        on the AWG disk.
        NB: uses real data normalized to the range from -1 to 1 (independent
        of number of DAC bits of AWG)

        Args:
            name (string): waveform name excluding double quotes, e.g. 'test'.
            Must already exits in AWG

            waveform (np.array of float)): vector defining the waveform,
            normalized between -1.0 and 1.0

        Compatibility:  QWG

        Based on:
            Tektronix_AWG5014.py::send_waveform, which sends data to an AWG
            _file_, not a memory waveform
            'awg_transferRealDataWithMarkers', Author = Stefano Poletto,
            Compatibility = Tektronix AWG5014, AWG7102
        """

        # generate the binblock
        if 1:   # high performance
            arr = np.asarray(waveform, dtype=np.float32)
            binBlock = arr.tobytes()
        else:   # more generic
            binBlock = b''
            for i in range(len(waveform)):
                binBlock = binBlock + struct.pack('<f', waveform[i])

        # write binblock
        hdr = 'wlist:waveform:data "{}",'.format(name)
        self.binBlockWrite(binBlock, hdr)

    def createWaveformReal(self, name, waveform):
        """
        Convenience function to create a waveform in the AWG and then send
        data to it

        Args:
            name(string): name of waveform for internal use by the AWG

            waveform (float[numpoints]): vector defining the waveform,
            normalized between -1.0 and 1.0


        Compatibility:  QWG
        """
        # wv_val = vals.Arrays(min_value=-1, max_value=1)
        # wv_val.validate(waveform)

        maxWaveLen = 2**17-4  # FIXME: this is the hardware max

        waveLen = len(waveform)
        if waveLen > maxWaveLen:
            raise ValueError('Waveform length ({}) must be < {}'.format(
                             waveLen, maxWaveLen))

        self.newWaveformReal(name, waveLen)
        self.sendWaveformDataReal(name, waveform)

    ##########################################################################
    # Generic (i.e. at least AWG520 and AWG5014) Tektronix AWG functions
    ##########################################################################

    # Tek_AWG functions: menu Setup|Waveform/Sequence
    def loadWaveformOrSequence(self, awgFileName):
        ''' awgFileName:        name referring to AWG file system
        '''
        self.write('source:def:user "%s"' % awgFileName)
        # NB: we only  support default Mass Storage Unit Specifier "Main",
        # which is the internal harddisk

    # Used for setting the channel pairs
    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func
