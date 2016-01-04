# Driver class for the AlazarTech ATS9870 Digitizer (DAQ) for Triggered Streaming Mode
# PCI Express board, 2 channels, 8bits, 1GS/s
# Damaz de Jong and Willemijn Uilhoorn, 2015 - Triggered Streaming
# This driver is probably not forward compatible with qtlab

from instrument import Instrument
import ctypes
import time
import numpy as np
import os
import matplotlib.pyplot as pyplot
import math
import sys
import logging
import types
import datetime
import data as d
import qt


class Alazar_ATS9870_TS(Instrument):
    def __init__(self, name):
        logging.info(__name__ + ' : Initializing instrument Alazar')
        Instrument.__init__(self, name, tags=['Virtual'])
        # initialize
        self.succes = 512

        '''
        Make sure the dll is located at "C:\\WINDOWS\\System32\\ATSApi"
        '''
        print(('start dll load ' + str(time.time()) + " and my name is " + name))
        self._ATS9870_dll = ctypes.cdll.LoadLibrary('C:\\WINDOWS\\System32\\ATSApi')
        self._handle = self._ATS9870_dll.AlazarGetBoardBySystemID(1, 1)

        if not self._handle:
            raise Exception("ATS not found")

        print(('end dll load   ' + str(time.time())))
        if not (13 == self._ATS9870_dll.AlazarGetBoardKind(self._handle)):
            raise Exception("Boardkind neq ATS9870")

        self.ErrorCodeIdentifier = {513: 'ApiFailed', 514: 'ApiAccessDenied', 515: 'ApiDmaChannelUnavailable',
                                    516: 'ApiDmaChannelInvalid', 517: 'ApiDmaChannelTypeError', 518: 'ApiDmaInProgress',
                                    519: 'ApiDmaDone', 520: 'ApiDmaPaused', 521: 'ApiDmaNotPaused',
                                    522: 'ApiDmaCommandInvalid', 523: 'ApiDmaManReady', 524: 'ApiDmaManNotReady',
                                    525: 'ApiDmaInvalidChannelPriority', 526: 'ApiDmaManCorrupted',
                                    527: 'ApiDmaInvalidElementIndex', 528: 'ApiDmaNoMoreElements',
                                    529: 'ApiDmaSglInvalid',
                                    530: 'ApiDmaSglQueueFull', 531: 'ApiNullParam', 532: 'ApiInvalidBusIndex',
                                    533: 'ApiUnsupportedFunction', 534: 'ApiInvalidPciSpace', 535: 'ApiInvalidIopSpace',
                                    536: 'ApiInvalidSize', 537: 'ApiInvalidAddress', 538: 'ApiInvalidAccessType',
                                    539: 'ApiInvalidIndex', 540: 'ApiMuNotReady', 541: 'ApiMuFifoEmpty',
                                    542: 'ApiMuFifoFull',
                                    543: 'ApiInvalidRegister', 544: 'ApiDoorbellClearFailed', 545: 'ApiInvalidUserPin',
                                    546: 'ApiInvalidUserState', 547: 'ApiEepromNotPresent',
                                    548: 'ApiEepromTypeNotSupported',
                                    549: 'ApiEepromBlank', 550: 'ApiConfigAccessFailed', 551: 'ApiInvalidDeviceInfo',
                                    552: 'ApiNoActiveDriver', 553: 'ApiInsufficientResources',
                                    554: 'ApiObjectAlreadyAllocated',
                                    555: 'ApiAlreadyInitialized', 556: 'ApiNotInitialized',
                                    557: 'ApiBadConfigRegEndianMode', 558: 'ApiInvalidPowerState', 559: 'ApiPowerDown',
                                    560: 'ApiFlybyNotSupported',
                                    561: 'ApiNotSupportThisChannel', 562: 'ApiNoAction', 563: 'ApiHSNotSupported',
                                    564: 'ApiVPDNotSupported', 565: 'ApiVpdNotEnabled', 566: 'ApiNoMoreCap',
                                    567: 'ApiInvalidOffset',
                                    568: 'ApiBadPinDirection', 569: 'ApiPciTimeout', 570: 'ApiDmaChannelClosed',
                                    571: 'ApiDmaChannelError', 572: 'ApiInvalidHandle', 573: 'ApiBufferNotReady',
                                    574: 'ApiInvalidData',
                                    575: 'ApiDoNothing', 576: 'ApiDmaSglBuildFailed', 577: 'ApiPMNotSupported',
                                    578: 'ApiInvalidDriverVersion',
                                    579: 'ApiWaitTimeout: operation did not finish during timeout interval. Check your trigger.',
                                    580: 'ApiWaitCanceled', 581: 'ApiBufferTooSmall',
                                    582: 'ApiBufferOverflow:rate of acquiring data > rate of transferring data to local memory. Try reducing sample rate, reducing number of enabled channels, increasing size of each DMA buffer or increase number of DMA buffers.',
                                    583: 'ApiInvalidBuffer', 584: 'ApiInvalidRecordsPerBuffer',
                                    585: 'ApiDmaPending:Async I/O operation was succesfully started, it will be completed when sufficient trigger events are supplied to fill the buffer.',
                                    586: 'ApiLockAndProbePagesFailed:Driver or operating system was unable to prepare the specified buffer for DMA transfer. Try reducing buffer size or total number of buffers.',
                                    587: 'ApiWaitAbandoned', 588: 'ApiWaitFailed',
                                    589: 'ApiTransferComplete:This buffer is last in the current acquisition.',
                                    590: 'ApiPllNotLocked:hardware error, contact AlazarTech',
                                    591: 'ApiNotSupportedInDualChannelMode:Requested number of samples per channel is too large to fit in on-board memory. Try reducing number of samples per channel, or switch to single channel mode.'}

        # <editor-fold desc="parameter definitions">
        self.add_parameter('ClockSource', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('SampleRate', flags=Instrument.FLAG_GET, unit='samples per second',
                           type=bytes)
        self.add_parameter('ClockEdge', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('Decimation', flags=Instrument.FLAG_GET,
                           type=int)

        self.add_parameter('Coupling', flags=Instrument.FLAG_GET,
                           type=bytes, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter('VRangeId', flags=Instrument.FLAG_GET, unit='Volts',
                           type=float, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter('Impedance', flags=Instrument.FLAG_GET, unit='Ohm',
                           type=float, channels=(1, 2), channel_prefix='ch%d_')
        self.add_parameter('BW', flags=Instrument.FLAG_GET,
                           type=bytes, channels=(1, 2), channel_prefix='ch%d_')

        self.add_parameter('TriggerOperationMode', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerEngine1', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerEngine2', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerSource1', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerSource2', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerSlope1', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerSlope2', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerLevel1_V', flags=Instrument.FLAG_GET, unit='Volts',
                           type=float)
        self.add_parameter('TriggerLevel2_V', flags=Instrument.FLAG_GET, unit='Volts',
                           type=float)
        self.add_parameter('ExTriggerCoupling', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerRange', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TriggerDelay', flags=Instrument.FLAG_GET, units='sample clocks',
                           type=int)
        self.add_parameter('TimeoutTicks', flags=Instrument.FLAG_GET, units='10 ms',
                           type=int)

        self.add_parameter('samplesperrecord', flags=Instrument.FLAG_GET,
                           type=int)
        self.add_parameter('nbuffers', flags=Instrument.FLAG_GET,
                           type=int)
        self.add_parameter('BuffersPerAcquisition', flags=Instrument.FLAG_GET,
                           type=int)

        self.add_parameter('ChannelSelection', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('TransferOffset', flags=Instrument.FLAG_GET, unit='samples',
                           type=int)
        self.add_parameter('RecordsPerBuffer', flags=Instrument.FLAG_GET,
                           type=int)
        self.add_parameter('AcquireMode', flags=Instrument.FLAG_GET,
                           type=bytes)
        # </editor-fold>


    # <editor-fold desc="Definition do_get_parameter functions">
    def do_get_ClockSource(self):
        return self.ClockSource

    def do_get_SampleRate(self):
        return self.SampleRate

    def do_get_ClockEdge(self):
        return self.ClockEdge

    def do_get_Decimation(self):
        return self.Decimation

    def do_get_Coupling(self, channel):
        return self.Coupling[channel - 1]

    def do_get_VRangeId(self, channel):
        return self.VRangeId[channel - 1]

    def do_get_Impedance(self, channel):
        return self.Impedance[channel - 1]

    def do_get_BW(self, channel):
        return self.BW[channel - 1]


    def do_get_TriggerOperationMode(self):
        return self.TriggerOperationMode

    def do_get_TriggerEngine1(self):
        return self.TriggerEngine1

    def do_get_TriggerEngine2(self):
        return self.TriggerEngine2

    def do_get_TriggerSource1(self):
        return self.TriggerSource1

    def do_get_TriggerSource2(self):
        return self.TriggerSource2

    def do_get_TriggerSlope1(self):
        return self.TriggerSlope1

    def do_get_TriggerSlope2(self):
        return self.TriggerSlope2

    def do_get_TriggerLevel1_V(self):
        return self.TriggerLevel1_V

    def do_get_TriggerLevel2_V(self):
        return self.TriggerLevel2_V

    def do_get_ExTriggerCoupling(self):
        return self.ExTriggerCoupling

    def do_get_TriggerRange(self):
        return self.TriggerRange

    def do_get_TriggerDelay(self):
        return self.TriggerDelay

    def do_get_TimeoutTicks(self):
        return self.TimeoutTicks

    def do_get_samplesperrecord(self):
        return self.samplesperrecord

    def do_get_nbuffers(self):
        return self.nbuffers

    def do_get_BuffersPerAcquisition(self):
        return self.BuffersPerAcquisition

    def do_get_ChannelSelection(self):
        return self.ChannelSelection

    def do_get_TransferOffset(self):
        return self.TransferOffset

    def do_get_RecordsPerBuffer(self):
        return self.RecordsPerBuffer

    def do_get_AcquireMode(self):
        return self.AcquireMode

    # </editor-fold>

    def config(self, ClockSourceId=7, SampleRateId=1000000000, Decimation=0, RangeId=[11, 11], RangeVolt=0):
        # sourceId       capture clock source
        #SampleRateId   sample rate
        #Decimation     decimation of sample rate, see manual before using
        #RangeId        input range of CHANNEL_A and CHANNEL_B
        #RangeVolt      input range of CHANNEL_A and CHANNEL_B, takes precedence over RangeId

        #set capture clock
        ClockSourceDict = {1: 'INTERNAL_CLOCK',
                           4: 'SLOW_EXTERNAL_CLOCK',
                           5: 'EXTERNAL_CLOCK_AC',
                           7: 'EXTERNAL_CLOCK_10_MHz_REF'}
        self.ClockSourceId = ClockSourceId
        self.ClockSource = ClockSourceDict[ClockSourceId]
        self.get_ClockSource()


        #1 KSPS                     0x1
        #2 KSPS                     0x2
        #5 KSPS                     0x4
        #10 KSPS                    0x8
        #20 KSPS                    0xA
        #50 KSPS                    0xC
        #100 KSPS                   0xE
        #200 KSPS                   0x10
        #500 KSPS                   0x12
        #1 MSPS                     0x14
        #2 MSPS                     0x18
        #5 MSPS                     0x1A
        #10 MSPS                    0x1C
        #20 MSPS                    0x1E
        #50 MSPS                    0x22
        #100 MSPS                   0x24
        #250 MSPS                   0x2B
        #500 MSPS                   0X30
        #1 GSPS                     0x35
        #USER DEFINED, EXT CLOCK    0x40    Do not use. DataExtraction can not determine samplespeed in this case
        #1GHz Reference clock val   1000000000
        self.SampleRateId = SampleRateId
        SampleRateIdDict = {0x1: 1000, 0x2: 2000, 0x4: 5000,
                            0x8: 10000, 0xA: 20000, 0xC: 50000,
                            0xE: 100000, 0x10: 200000, 0x12: 500000,
                            0x14: 1000000, 0x18: 2000000, 0x1A: 5000000,
                            0x1C: 10000000, 0x1E: 20000000, 0x22: 50000000,
                            0x24: 100000000, 0x2B: 250000000, 0x30: 500000000,
                            0x35: 1000000000, 1000000000: 1000000000}
        self.SampleRate = SampleRateIdDict[SampleRateId]
        self.get_SampleRate()

        ClockEdgeDict = {0: 'CLOCK_EDGE_RISING',
                         1: 'CLOCK_EDGE_FALLING'}
        self.ClockEdgeId = 0
        self.ClockEdge = ClockEdgeDict[self.ClockEdgeId]
        self.get_ClockEdge()

        # For ATS9870 with EXTERNAL_CLOCK_10MHz_REF, decimation must be 1, 2, 4 or any multiple of 10
        # decimation=0 means disabled
        # refer to manual
        self.Decimation = Decimation
        self.get_Decimation()

        retcode = self._ATS9870_dll.AlazarSetCaptureClock(self._handle, self.ClockSourceId, self.SampleRateId,
                                                          self.ClockEdgeId, self.Decimation)
        if (retcode != self.succes):
            raise Exception("AlazarSetCaptureClock " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        #format for parameters is [CHANNEL_A,CHANNEL_B]

        #AC_COUPLING                1
        #DC_COUPLING                2
        self.CouplingId = [1, 1]
        CouplingDict = {1: 'AC_COUPLING', 2: 'DC_COUPLING'}
        self.Coupling = [CouplingDict[self.CouplingId[0]], CouplingDict[self.CouplingId[1]]]
        self.get_ch1_Coupling()
        self.get_ch2_Coupling()

        # RangeIdDict is in Volts
        RangeIdDict = {2: 0.04,
                       5: 0.1,
                       6: 0.2,
                       7: 0.4,
                       10: 1.,
                       11: 2.,
                       12: 4.}
        RangeVoltDict = {v: k for k, v in list(RangeIdDict.items())}
        if not RangeVolt == 0:
            self.RangeId = [RangeVoltDict[RangeVolt[0]], RangeVoltDict[RangeVolt[1]]]
        else:
            self.RangeId = RangeId
        self.VRangeId = [RangeIdDict[self.RangeId[0]], RangeIdDict[self.RangeId[1]]]
        self.get_ch1_VRangeId()
        self.get_ch2_VRangeId()

        # ImpedanceDict is in Ohms
        ImpedanceDict = {2: 50}
        self.ImpedanceId = [2, 2]
        self.Impedance = [ImpedanceDict[self.ImpedanceId[0]], self.ImpedanceId[1]]
        self.get_ch1_Impedance()
        self.get_ch2_Impedance()

        #BWLimit flag
        BWlimitDict = {0: 'bandwidth limit disabled',
                       1: 'bandwidth limit enabled'}
        self.BWlimit = [0, 0]
        self.BW = [BWlimitDict[self.BWlimit[0]], BWlimitDict[self.BWlimit[1]]]
        self.get_ch1_BW()
        self.get_ch2_BW()

        #for channel A and B
        for ChannelId in [0, 1]:
            #set input control
            retcode = self._ATS9870_dll.AlazarInputControl(self._handle, ChannelId + 1, self.CouplingId[ChannelId],
                                                           self.RangeId[ChannelId], self.ImpedanceId[ChannelId])
            if (retcode != self.succes):
                raise Exception("AlazarInputControl channel " + str(ChannelId) + " err: " + str(retcode) + ': ' +
                                    self.ErrorCodeIdentifier[retcode])

            #set bandwidth limit channel
            retcode = self._ATS9870_dll.AlazarSetBWLimit(self._handle, ChannelId + 1, self.BWlimit[ChannelId])
            if (retcode != self.succes):
                raise Exception("AlazarSetBWLimit channel " + str(ChannelId) + " err: " + str(retcode) + ': ' +
                                    self.ErrorCodeIdentifier[retcode])

        '''
        config the trigger of the acquisition
        '''

        #set trigger operation (config to listen to external trigger)

        TriggerOperationDict = {0: 'TRIG_ENGINE_OP_J',
                                1: 'TRIG_ENGINE_OP_K',
                                2: 'TRIG_ENGINE_OP_J_OR_K',
                                3: 'TRIG_ENGINE_OP_J_AND_K',
                                4: 'TRIG_ENGINE_OP_J_XOR_K',
                                5: 'TRIG_ENGINE_OP_J_AND_NOT_K',
                                6: 'TRIG_ENGINE_OP_NOT_J_AND_K'}
        self.TriggerOperation = 0
        self.TriggerOperationMode = TriggerOperationDict[self.TriggerOperation]
        self.get_TriggerOperationMode()

        TriggerEngineDict = {0: 'TRIG_ENGINE_J',
                             1: 'TRIG_ENGINE_K'}
        self.TriggerEngineId1 = 0
        self.TriggerEngineId2 = 1
        self.TriggerEngine1 = TriggerEngineDict[self.TriggerEngineId1]
        self.TriggerEngine2 = TriggerEngineDict[self.TriggerEngineId2]
        self.get_TriggerEngine1()
        self.get_TriggerEngine2()

        TriggerSourceDict = {0: 'TRIG_CHAN_A',
                             1: 'TRIG_CHAN_B',
                             2: 'TRIG_EXTERNAL',
                             3: 'TRIG_DISABLE'}
        self.TriggerSourceId1 = 2
        self.TriggerSourceId2 = 3
        self.TriggerSource1 = TriggerSourceDict[self.TriggerSourceId1]
        self.TriggerSource2 = TriggerSourceDict[self.TriggerSourceId2]
        self.get_TriggerSource1()
        self.get_TriggerSource2()

        TriggerSlopeDict = {1: 'TRIG_SLOPE_POSITIVE',
                            2: 'TRIG_SLOPE_NEGATIVE'}
        self.TriggerSlopeId1 = 1
        self.TriggerSlopeId2 = 1
        self.TriggerSlope1 = TriggerSlopeDict[self.TriggerSlopeId1]
        self.TriggerSlope2 = TriggerSlopeDict[self.TriggerSlopeId2]
        self.get_TriggerSlope1()
        self.get_TriggerSlope2()

        #unsigned 8-bit code, fraction of input voltage, See manual
        #TriggerLevelCode = 128 + 127 * TriggerLevelVolts / InputRangeVolts
        self.TriggerLevel1 = 128
        self.TriggerLevel2 = 128
        self.TriggerLevel1_V = (self.TriggerLevel1 - 128) * (self.VRangeId[0] / 127)
        self.TriggerLevel2_V = (self.TriggerLevel2 - 128) * (self.VRangeId[1] / 127)
        self.get_TriggerLevel1_V()
        self.get_TriggerLevel2_V()

        retcode = self._ATS9870_dll.AlazarSetTriggerOperation(self._handle, self.TriggerOperation,
                                                              self.TriggerEngineId1, self.TriggerSourceId1,
                                                              self.TriggerSlopeId1, self.TriggerLevel1,
                                                              self.TriggerEngineId2, self.TriggerSourceId2,
                                                              self.TriggerSlopeId2, self.TriggerLevel2)
        if (retcode != self.succes):
            raise Exception("AlazarSetTriggerOperation " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])


        #config external trigger
        ExTriggerCouplingDict = {1: 'AC_COUPLING',
                                 2: 'DC_COUPLING'}
        self.ExTriggerCouplingId = 1
        self.ExTriggerCoupling = ExTriggerCouplingDict[self.ExTriggerCouplingId]
        self.get_ExTriggerCoupling()

        TriggerRangeDict = {0: 'ETR_5V',
                            1: 'ETR_1V'}
        self.TriggerRangeId = 0
        self.TriggerRange = TriggerRangeDict[self.TriggerRangeId]
        self.get_TriggerRange()

        retcode = self._ATS9870_dll.AlazarSetExternalTrigger(self._handle, self.ExTriggerCouplingId,
                                                             self.TriggerRangeId)
        if (retcode != self.succes):
            raise Exception("AlazarSetExternalTrigger " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        #set trigger delay
        #trigger delay in sample clocks
        self.TriggerDelay = 0
        self.get_TriggerDelay()
        retcode = self._ATS9870_dll.AlazarSetTriggerDelay(self._handle, self.TriggerDelay)
        if (retcode != self.succes):
            raise Exception("AlazarSetTriggerDelay " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        #set trigger timeout
        #in units if 10 microseconds, or 0 to wait forever
        self.TimeoutTicks = 0
        self.get_TimeoutTicks()
        retcode = self._ATS9870_dll.AlazarSetTriggerTimeOut(self._handle, self.TimeoutTicks)
        if (retcode != self.succes):
            raise Exception("AlazarSetTriggerTimeOut " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

            #config AUXIO omitted

    def acquire(self, BuffersPerAcquisition=12, recordsperbuffer=1, samplesperrecord=96000, nbuffers=8, mode='TS', dataextractor=None):
        # type checking
        if mode == 'TS':
            # For Triggered Streaming must be 1
            if not recordsperbuffer == 1:
                raise Exception("In TS mode, the Alazar will ignore 'recordsperbuffer'. Defaulting to 1.")
            self.RecordsPerBuffer = 1
        else:
            self.RecordsPerBuffer = recordsperbuffer
        self.get_RecordsPerBuffer()

        if not (mode == 'TS' or mode == 'NPT'):
            raise Exception("Only the 'TS' and 'NPT' modes are implemented at this point")

        # acquire settings
        self.BuffersPerAcquisition = BuffersPerAcquisition
        self.get_BuffersPerAcquisition()
        self.samplesperrecord = samplesperrecord
        self.get_samplesperrecord()
        self.nbuffers = nbuffers
        self.get_nbuffers()

        SamplesPerBuffer = self.RecordsPerBuffer * self.samplesperrecord

        # abort remaining buffers
        retcode = self._ATS9870_dll.AlazarAbortAsyncRead(self._handle)
        if retcode != self.succes:
            raise Exception("AlazarAbortAsyncRead " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        # get channel info
        bps = np.array([0], dtype=np.uint8)  # bps bits per sample
        max_s = np.array([0], dtype=np.uint32)  # max_s memory size in samples

        retcode = self._ATS9870_dll.AlazarGetChannelInfo(self._handle, max_s.ctypes.data, bps.ctypes.data)
        if retcode != self.succes:
            raise Exception("AlazarGetChannelInfo " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])
        if not bps == 8:
            raise Exception("weird bit per sample :(")
        bps = bps[0]
        max_s = max_s[0]
        # print("max_s: "+str(max_s))
        if mode == 'NPT':
            pretriggersize = 0  # number of samples before trigger event, 0 for NPT
            posttriggersize = self.samplesperrecord
            retcode = self._ATS9870_dll.AlazarSetRecordSize(self._handle, pretriggersize, posttriggersize)
            if retcode != self.succes:
                raise Exception("AlazarSetRecordSize " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        # set ats to async read
        ChannelSelectDict = {1: 'CHANNEL_A',
                             2: 'CHANNEL_B',
                             3: 'CHANNEL_A | CHANNEL_B'}
        self.ChannelSelect = 3
        self.ChannelSelection = ChannelSelectDict[self.ChannelSelect]
        self.get_ChannelSelection()

        # get Number of Channels
        NumberOfChannels = 0
        if self.ChannelSelect == 3:
            NumberOfChannels = 2
        else:
            NumberOfChannels = 1

        #init local buffers
        self.buflist = list(range(self.nbuffers))
        #print("Allocating memory")
        #print(str(self.nbuffers)+'*'+str(NumberOfChannels*SamplesPerBuffer*bps/(8*1024*1024))+' Mbyte')
        for k in range(self.nbuffers):
            self.buflist[k] = Buffer(bps, SamplesPerBuffer, NumberOfChannels)

        #Specify first sample relative to trigger position
        self.TransferOffset = 0
        self.get_TransferOffset()
        # Undocumented: samples per buffer (Documentation is samples per record)
        # Already defined variable

        # Undocumented: recordsperacquisition field:
        # for TS: buffers per acquisition (Documentation is records per acquisition)
        # for NPT: records per acquisition
        # Already defined variable

        # IMPORTANT: NOT ALL SETTINGS ARE SUPPORTED YET. ONLY TRIGGERED STREAMING IS SUPPORTED NOW.
        FlagsDict = {0x0: 'ADMA_TRADITIONAL_MODE',
                     0x200: 'ADMA_NPT',
                     0x100: 'ADMA_CONTINUOUS_MODE',
                     0x400: 'ADMA_TRIGGERED_STREAMING',
                     0x1: 'AMDA_EXTRERNAL_STARTCAPTURE',
                     0x8: 'ADMA_ENABLE_RECORD_HEADERS',
                     0x20: 'ADMA_ALLOC_BUFFERS',
                     0x800: 'ADMA_FIFO_ONLY_STREAMING',
                     0x1000: 'ADMA_INTERLEAVE_SAMPLES',
                     0x2000: 'ADMA_GET_PROCESSED_DATA'}
        if mode == 'TS':
            self.AcquireFlags = 0x1 | 0x400
        elif mode == 'NPT':
            self.AcquireFlags = 0x1 | 0x200

        self.AcquireMode = ""
        for key, value in FlagsDict.items():
            if self.AcquireFlags & key:
                self.AcquireMode += '-' + value
        self.get_AcquireMode()

        if mode == 'TS':
            retcode = self._ATS9870_dll.AlazarBeforeAsyncRead(self._handle, self.ChannelSelect, self.TransferOffset,
                                                          SamplesPerBuffer, self.RecordsPerBuffer,
                                                          BuffersPerAcquisition, self.AcquireFlags)
        elif mode == 'NPT':
            RecordsPerAcquisition=BuffersPerAcquisition*recordsperbuffer
            retcode = self._ATS9870_dll.AlazarBeforeAsyncRead(self._handle, self.ChannelSelect, self.TransferOffset,
                                                          self.samplesperrecord, self.RecordsPerBuffer,
                                                          RecordsPerAcquisition, self.AcquireFlags)
        if (retcode != self.succes):
            raise Exception("AlazarBeforeAsyncRead " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        # post buffers
        # Specify buffersize in bytes
        for buf in self.buflist:
            retcode = self._ATS9870_dll.AlazarPostAsyncBuffer(self._handle, buf.addr, buf.size_bytes)
            if (retcode != self.succes):
                raise Exception("AlazarPostAsyncBuffer " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        dataextractor.pre_acquire(self)

        # start capture
        #print("Starting data acquisition")

        retcode = self._ATS9870_dll.AlazarStartCapture(self._handle)
        if (retcode != self.succes):
            raise Exception("AlazarStartCapture " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        BuffersCompleted = 0
        start = time.clock()  # Keep track of when acquisition started

        # retrieve all buffers after they are measured
        while BuffersCompleted < BuffersPerAcquisition:
            buf = self.buflist[BuffersCompleted % self.nbuffers]

            retcode = self._ATS9870_dll.AlazarWaitAsyncBufferComplete(self._handle, buf.addr, 5000)
            if (retcode != self.succes):
                raise Exception(
                    "AlazarWaitAsyncBufferComplete " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

            if not self.nbuffers == self.BuffersPerAcquisition:
                # extract useful numbers from data and write them
                dataextractor.extract(self, buf.buffer)

                retcode = self._ATS9870_dll.AlazarPostAsyncBuffer(self._handle, buf.addr, buf.size_bytes)
                if (retcode != self.succes):
                    raise Exception("AlazarPostAsyncBuffer(2) " + str(retcode) + ': ' + self.ErrorCodeIdentifier[
                        retcode] + "bufferscomplete " + str(BuffersCompleted))
            BuffersCompleted += 1

        t1 = time.clock() - start
        #print("BuffersTransferred: "+str(BuffersCompleted))
        #print("TransferSpeed:      "+str(BuffersCompleted*buf.size_bytes/((t1)*1024*1024))+" MBps")
        #print("TotalTranferred:    "+str(BuffersCompleted*buf.size_bytes/(1024*1024))+" MB")

        #abort remaining buffers
        retcode = self._ATS9870_dll.AlazarAbortAsyncRead(self._handle)
        if (retcode != self.succes):
            raise Exception("AlazarAbortAsyncRead " + str(retcode) + ': ' + self.ErrorCodeIdentifier[retcode])

        if self.nbuffers == self.BuffersPerAcquisition:
            for buf in self.buflist:
                dataextractor.extract(self, buf.buffer)

        #free memory
        for b in self.buflist:
            b.free_mem()

        return dataextractor.getresults(self)

    def get_Abstract_DataExtractor(self):
        class Abstract_DataExtractor(object):
            def __init__(self):
                """
                :return: nothing
                """
                pass

            def pre_acquire(self, alazar):
                """
                Use this method to prepare yourself for the data acquisition
                :param alazar: a reference to the alazar driver
                :return: nothing
                """
                raise NotImplementedError("This method should be implemented in the implementation")

            def extract(self, alazar, data):
                """
                :param data: np.array with the data from the alazar card
                :return: something, it is ignored in any case
                """
                raise NotImplementedError(
                    "This method should be implemented in the implementation of the DataExtrector class")

            def getresults(self, alazar):
                """

                :return: this function should return all relevant data that you want to get form the acquisition
                """
                raise NotImplementedError("This method should be implemented somewhere")

        return Abstract_DataExtractor

    def bytetovolt(self, channel, signal):
        return ((signal - 127.5) / 127.5) * (self.VRangeId[channel])

    def getsamplespeed(self):
        if self.Decimation > 0:
            return self.SampleRate / self.Decimation
        else:
            return self.SampleRate

    def clearmem(self):
        for b in self.buflist:
            b.free_mem()


class Buffer:
    def __init__(self, bits_per_sample, samples_per_buffer, NumberOfChannels):
        if not bits_per_sample == 8:
            raise Exception("Buffer: weird bit per sample :(")
        if not os.name == 'nt':
            raise Exception("Buffer: only windows supported at this moment")
        self._allocated = True

        # try to allocate memory
        self.MEM_COMMIT = 0x1000
        self.PAGE_READWRITE = 0x4

        self.size_bytes = samples_per_buffer * NumberOfChannels

        # please see https://msdn.microsoft.com/en-us/library/windows/desktop/aa366887(v=vs.85).aspx for documentation
        ctypes.windll.kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
        self.addr = ctypes.windll.kernel32.VirtualAlloc(0, ctypes.c_long(self.size_bytes), self.MEM_COMMIT,
                                                        self.PAGE_READWRITE)

        # Data format from Alazar is 0-255 where 0 is full negative and 255 full positive input voltage
        # or signed mode 0->0v -128 is full negative, 127 is full positive
        ctypes_array = ((ctypes.c_uint8) * self.size_bytes).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes_array, dtype=np.uint8)
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def free_mem(self):
        self.MEM_RELEASE = 0x8000

        # see https://msdn.microsoft.com/en-us/library/windows/desktop/aa366892(v=vs.85).aspx
        ctypes.windll.kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
        ctypes.windll.kernel32.VirtualFree.restype = ctypes.c_int
        ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), 0, self.MEM_RELEASE);
        self._allocated = False

    def __del__(self):
        if self._allocated:
            self.free_mem()
            raise Exception("Buffer: Memory leak!")


if __name__ == "__main__":
    # ATS=Alazar_ATS9870_TS()
    #ATS.config(Decimation=0)
    #start = time.clock() # Keep track of when acquisition started
    #for i in range(10):
    #    dataFile = open('data.dat', 'w')
    #    np.array(ATS.acquire(nbuffers=4000,BuffersPerAcquisition=4000,samplesperbin=96000, nbins=100)).tofile(dataFile,sep="\n",format="%s")
    #    dataFile.close()
    #t1=time.clock() - start
    #print(t1/10)
    pass

