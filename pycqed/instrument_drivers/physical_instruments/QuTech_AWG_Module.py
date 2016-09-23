'''
File:               QuTech_AWG_Module.py
Author:             Wouter Vlothuizen, TNO/QuTech,
                    edited by Adriaan Rol
Purpose:            Instrument driver for Qutech QWG
Usage:
Notes:
Bugs:
'''

from .SCPI import SCPI

import numpy as np
import struct
from qcodes import validators as vals


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
        self.device_descriptor.numCodewords = 8

        # valid values
        self.device_descriptor.mvals_trigger_impedance = vals.Enum(50),
        self.device_descriptor.mvals_trigger_level = vals.Numbers(0, 2.5)
        self.device_descriptor.mvals_channel_amplitude = vals.Numbers(
            0, 1)  # FIXME: not in [V]
        # FIXME: not in [V]
        self.device_descriptor.mvals_channel_offset = vals.Numbers(-0.05, 0.05)

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
            mat_cmd = 'qutech:output{}:matrix'.format(ch_pair)
            # NB: sideband frequency has a resolution of ~0.23 Hz:
            self.add_parameter('ch_pair{}_sideband_frequency'.format(ch_pair),
                               units='Hz',
                               label=('Sideband frequency channel ' +
                                      'pair {} (Hz)'.format(i)),
                               get_cmd=sfreq_cmd + '?',
                               set_cmd=sfreq_cmd + ' {}',
                               vals=vals.Numbers(-300e6, 300e6),
                               get_parser=float)
            self.add_parameter('ch_pair{}_sideband_phase'.format(ch_pair),
                               units='deg',
                               label=('Sideband phase channel' +
                                      ' pair {} (deg)'.format(i)),
                               get_cmd=sph_cmd + '?',
                               set_cmd=sph_cmd + ' {}',
                               vals=vals.Numbers(-180, 360),
                               get_parser=float)

            self.add_parameter('ch_pair{}_transform_matrix'.format(ch_pair),
                               label=('Transformation matrix channel' +
                                      'pair {}'.format(i)),
                               get_cmd=mat_cmd + '?',
                               set_cmd=self._gen_ch_set_func(
                                    self._setMatrix, ch_pair),
                               # NB range is not a hardware limit
                               vals=vals.Arrays(-2, 2, shape=(2, 2)),
                               get_parser=np.array)

        for i in range(1, self.device_descriptor.numTriggers+1):
            triglev_cmd = 'qutech:trigger{}:level'.format(i)
            # individual trigger level per trigger input:
            self.add_parameter('ch{}_trigger_level'.format(i),
                               units='V',
                               label='Trigger level channel {} (V)'.format(i),
                               get_cmd=triglev_cmd + '?',
                               set_cmd=triglev_cmd + ' {}',
                               vals=self.device_descriptor.mvals_trigger_level,
                               get_parser=float)

        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Enum('CONT', 'SEQ', 'COD'))


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
                               get_parser=int,
                               vals=vals.Ints(0, 1))

            # Compatibility: 5014, QWG (FIXME: different range, not in V)
            self.add_parameter('ch{}_amp'.format(ch),
                               label='Amplitude channel {} (Vpp)'.format(ch),
                               units='Vpp',
                               get_cmd=amp_cmd + '?',
                               set_cmd=amp_cmd + ' {:.6f}',
                               vals=vals.Numbers(0.02, 4.5),
                               get_parser=float)

            # Compatibility: 5014, QWG (FIXME: different range, not in V)
            self.add_parameter('ch{}_offset'.format(ch),
                               label='Offset channel {} (V)'.format(ch),
                               units='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-.1, .1),
                               get_parser=float)

            self.add_parameter('ch{}_default_waveform'.format(ch),
                               get_cmd=waveform_cmd+'?',
                               set_cmd=waveform_cmd+' "{}"',
                               vals=vals.Strings())

        for i in range(self.device_descriptor.numCodewords):
            cw = i+1
            for j in range(self.device_descriptor.numChannels):
                ch = j+1
                cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
                self.add_parameter('codeword_{}_ch{}_waveform'.format(cw, ch),
                                   get_cmd=cw_cmd+'?',
                                   set_cmd=cw_cmd+' {:s}',
                                   vals=vals.Strings())

        doc_sSG = "Synchronize both sideband frequency" \
            + " generators, i.e. restart them with their defined phases."
        self.add_function('syncSidebandGenerators',
                          call_cmd='QUTEch:OUTPut:SYNCsideband',
                          docstring=doc_sSG)

    def _setMatrix(self, chPair, mat):
        '''
        matrix:             2x2 matrix for mixer calibration
        '''
        # function used internally for the parameters because of formatting
        print(chPair, mat)
        self.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(
                   chPair, mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]))

    ##########################################################################
    # AWG5014 functions: SEQUENCE
    ##########################################################################
    def setSeqLength(self, length):
        ''' length:     0..max. Allocates new, or trims existing sequence
        '''
        self.write('sequence:length %d' % length)

    def setSeqElemLoopInfiniteOn(self, element):
        ''' element:        1..length
        '''
        self.write('sequence:element%d:loop:infinite on' % element)

    # def setSeqElemWaveform(self, element, ch, name):
    #     """
    #     Set the waveform for a sequence element

    #     Args:
    #             element (int): index of sequence element (valid range: 1..length)

    #             ch (int): AWG channel where waveform is put

    #             waveform (string): name of waveform in AWG memory

    #     Compatibility:  5014, QWG
    #     """
    #     self.write('sequence:element%d:waveform%d "%s"' % (element, ch, name))

    ##########################################################################
    # AWG5014 functions: WLIST (Waveform list)
    ##########################################################################
    def getWlistSize(self):
        return self.askDouble('wlist:size?')

    def getWlistName(self, idx):
        ''' idx:            0..size-1
        '''
        return self.ask('wlist:name? %d' % idx)

    def getWlist(self):
        ''' NB: takes a few seconds on 5014: our fault or Tek's?
        '''
        size = self.getWlistSize()
        wlist = []                                  # empty list
        for k in range(size):                       # build list of names
            wlist.append(self.getWlistName(k))
        return wlist

    def deleteWaveform(self, name):
        ''' name:       waveform name excluding double quotes, e.g. 'test'
                Compatibility:  5014, QWG
        '''
        self.write('wlist:waveform:delete "%s"' % name)

        ''' Compatibility:  5014, QWG
        '''

    def deleteWaveformAll(self):
        self.write('wlist:waveform:delete all')

    def getWaveformType(self, name):
        ''' name:       waveform name excluding double quotes, e.g. '*Sine100'
                Returns:    'INT' or 'REAL'
        '''
        return self.ask('wlist:waveform:type? "%s"' % name)

    def getWaveformLength(self, name):
        ''' name:       waveform name excluding double quotes, e.g. '*Sine100'
        '''
        return self.askDouble('wlist:waveform:length? "%s"' % name)

    def newWaveformReal(self, name, len):
        ''' name:       waveform name excluding double quotes, e.g. 'test'
                NB: seems to do nothing if waveform already exists
        '''
        self.write('wlist:waveform:new "%s",%d,real' % (name, len))

    def getWaveformData(self, name):
        '''
                Input:
                        name:       string              waveform name excluding double quotes, e.g. '*Sine100'
                Output:
                        tuple containing lists: (waveform, marker1, marker2)

                Compatibility:  5014, QWG

                Funny old Matlab timing results:
                        tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine100');toc
                        Elapsed time is 0.265559 seconds.
                        tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine1000');toc
                        Elapsed time is 0.101930 seconds.
                        tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine3600');toc
                        Elapsed time is 0.056023 seconds.
        '''
        self.write('wlist:waveform:data? "%s"' %
                   name)                            # response starts with header, e.g. '#3500'
        binBlock = self.binBlockRead()
        # extract waveform and markers
        waveformLen = len(binBlock)/5                                           # 5 bytes per record
        waveform = []
        marker1 = []
        marker2 = []
        for k in range(waveformLen):
            (waveform[i], markers) = struct.unpack(binBlock, '<fB')
            marker1[i] = markers & 0x01
            marker2[i] = markers >> 1 & 0x01

        return (waveform, marker1, marker2)

    def sendWaveformDataReal(self, name, waveform, marker1, marker2):
        """
        send waveform and markers directly to AWG memory, i.e. not to a file on the AWG disk.
        NB: uses real data normalized to the range from -1 to 1 (independent of number of DAC bits of AWG)

                Args:
                        name (string): waveform name excluding double quotes, e.g. 'test'. Must already exits in AWG

                        waveform (float[numpoints]): vector defining the waveform, normalized between -1.0 and 1.0

                        marker1 (int[numpoints]): vector of 0 and 1 defining the first marker

                        marker2 (int[numpoints]): vector of 0 and 1 defining the second marker

                Compatibility:  5014, QWG

                Based on:
                        Tektronix_AWG5014.py::send_waveform, which sends data to an AWG _file_, not a memory waveform
                        'awg_transferRealDataWithMarkers', Author = Stefano Poletto, Compatibility = Tektronix AWG5014, AWG7102
        """

        # parameter handling
        if len(marker1) == 0 and len(marker2) == 0:                                 # no marker data
            m = np.zeros(len(waveform))
        else:
            if (not((len(waveform) == len(marker1)) and ((len(marker1) == len(marker2))))):
                raise UserWarning('length mismatch between markers/waveform')
            # prepare markers
            m = marker1 + np.multiply(marker2, 2)
            m = int(np.round(m[i], 0))

        # FIXME: check waveform amplitude and marker values (if paranoid)

        # generate the binblock
        binBlock = b''
        for i in range(len(waveform)):
            binBlock = binBlock + struct.pack('<fB', waveform[i], int(m[i]))

        # write binblock
        hdr = 'wlist:waveform:data "{}",'.format(name)
        self.binBlockWrite(binBlock, hdr)

    def createWaveformReal(self, name, waveform, marker1, marker2):
        """
        Convenience function to create a waveform in the AWG and then send data to it

        Args:
                name(string): name of waveform for internal use by the AWG

                        waveform (float[numpoints]): vector defining the waveform, normalized between -1.0 and 1.0

                        marker1 (int[numpoints]): vector of 0 and 1 defining the first marker

                        marker2 (int[numpoints]): vector of 0 and 1 defining the second marker

                Compatibility:  5014, QWG
        """
        waveLen = len(waveform)
#       if self.paranoid:
        # check waveform is there, problems might arise if it already existed
        self.newWaveformReal(name, waveLen)
        self.sendWaveformDataReal(name, waveform, marker1, marker2)

    ##########################################################################
    # AWG5014 functions: MMEM (Mass Memory)
    ##########################################################################

#   None at the moment

    ##########################################################################
    # Generic (i.e. at least AWG520 and AWG5014) Tektronix AWG functions
    ##########################################################################

    # Tek_AWG functions: menu Setup|Waveform/Sequence
    def loadWaveformOrSequence(self, awgFileName):
        ''' awgFileName:        name referring to AWG file system
        '''
        self.write('source:def:user "%s"' %
                   awgFileName)     # NB: we only support default Mass Storage Unit Specifier "Main", which is the internal harddisk

    # Tek_AWG functions: Button interface
    def run(self):
        self.write('awgcontrol:run:immediate')

    def stop(self):
        self.write('awgcontrol:stop:immediate')


    ##########################################################################
    # Generic AWG functions also implemented as Parameter
    # to be deprecated in the future
    ##########################################################################

    # NB: functions are organised by their appearance in the AWG520 user interface
    # functions: menu Setup|Vertical
    def setOffset(self, ch, offset):
        ''' ch:             AWG520: 1,2 (and 7,8 see documentation)
                offset:         AWG520: -1.000V to +1.000V in 1 mV steps
        '''
        self.write('source%d:voltage:level:immediate:offset %f' % (ch, offset))

    def getOffset(self, ch):
        ''' ch:             AWG520: 1,2 (and 7,8 see documentation)
        '''
        return self.askDouble('source%d:voltage:level:immediate:offset?' % ch)

    # functions: menu Setup|Vertical
    def setAmplitude(self, ch, amplitude):
        ''' ch:             AWG520: 1,2 (and 7,8 see documentation)
                amplitude:      AWG520: 0.020Vpp to 2.000Vpp in 1 mV steps
        '''
        self.write('source%d:voltage:level:immediate:amplitude %f' %
                   (ch, amplitude))

    def getAmplitude(self, ch):
        ''' ch:             AWG520: 1,2 (and 7,8 see documentation)
        '''
        return self.askDouble('source%d:voltage:level:immediate:amplitude?' % ch)

    # functions: Button interface
    def setOutputStateOn(self, ch):
        ''' ch:             AWG520: 1,2 (and 7 see documentation)
                NB: only works if waveform is defined or def Generator is on
        '''
        self.write('output%d:state on' % ch)

    def setOutputStateOff(self, ch):
        ''' ch:             AWG520: 1,2 (and 7 see documentation)
        '''
        self.write('output%d:state off' % ch)

    def setTriggerLevel(self, level):
        ''' level:              AWG520: -5.0 V to +5.0 V, in 0.1 V steps
        '''
        self.write('trigger:level %f' % level)

    # functions: menu Setup|Run Mode
    def setRunModeSequence(self):
        self.write('awgcontrol:rmode seq')

    def setRunModeContinuous(self):
        self.write('awgcontrol:rmode cont')


    # Used for setting the channel pairs
    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func
