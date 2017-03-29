'''
    File:               DDM.py
    Author:             Nikita Vodyagin, QuTech
    Purpose:            control of Qutech DDM
    Prerequisites:
    Usage:
    Bugs:
'''


from .SCPI import SCPI
import numpy as np
import struct
import math
from qcodes import validators as vals
import logging
import time

logging.warning('my warning ')
logging.info('my info ')


class DDMq(SCPI):

    # def __init__(self, logging=True, simMode=False, paranoid=False):
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'DDM'
        self.device_descriptor.numChannels = 2
        self.device_descriptor.numWeights = 5
        self.add_parameters()
        self.connect_message()

    def add_parameters(self):
        #######################################################################
        # DDM specific
        #######################################################################

        for i in range(self.device_descriptor.numChannels//2):
            ch_pair = 2*i+1
            srun_cmd = 'qutech:run{}'.format(ch_pair)
            self.add_parameter('ch_pair{}_run'.format(ch_pair),
                               label=('Run ch_pair{}'.format(ch_pair)),
                               # docstring='',
                               #get_cmd=swinten_cmd + '?',
                               set_cmd=srun_cmd
                               )
            sreset_int_cmd = 'qutech:reset{}'.format(ch_pair)
            self.add_parameter('ch_pair{}_reset'.format(ch_pair),
                               label=('Reset DDM integration modes'),
                               #get_cmd=swinten_cmd + '?',
                               set_cmd=sreset_int_cmd
                               )
            scaladc_cmd = 'qutech:adc{}:cal'.format(ch_pair)
            self.add_parameter('ch_pair{}_cal_adc'.format(ch_pair),
                               label=('Calibrate ADC{}'.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                               set_cmd=scaladc_cmd
                               # vals=vals.Numbers(1,4096)
                               )
            snavg_cmd = 'qutech:inputavg{}:naverages'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_Navg'.format(ch_pair),
                               unit='#',
                               label=('Number of averages' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=snavg_cmd + '?',
                               set_cmd=snavg_cmd + ' {}',
                               vals=vals.Numbers(1, 32768)
                               )
            snsamp_cmd = 'qutech:inputavg{}:scansize'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_scansize'.format(ch_pair),
                               unit='#',
                               label=('Number of samples' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=snsamp_cmd + '?',
                               set_cmd=snsamp_cmd + ' {}',
                               vals=vals.Numbers(1, 4096)
                               )
            senable_cmd = 'qutech:inputavg{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_enable'.format(ch_pair),
                               label=('Enable input averaging' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=senable_cmd + '?',
                               set_cmd=senable_cmd + ' {}',
                               vals=vals.Numbers(0, 1)
                               )
            sholdoff_cmd = 'qutech:input{}:holdoff'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_holdoff'.format(ch_pair),
                               label=('Set holdoff' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=sholdoff_cmd + '?',
                               set_cmd=sholdoff_cmd + ' {}',
                               vals=vals.Numbers(0, 254)
                               )
            sintlengthall_cmd = 'qutech:wint{}:intlength:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_intlength'.format(ch_pair),
                               unit='#',
                               label=('The number of sample  that' +
                                      'is used for one integration of' +
                                      'ch_pair {} all weights'.format(ch_pair)),
                               get_cmd=sintlengthall_cmd + '?',
                               set_cmd=sintlengthall_cmd + ' {}',
                               vals=vals.Numbers(1, 2048)
                               )
            '''
            TV mode
            '''

            sintavgall_cmd = 'qutech:tvmode{}:naverages:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_naverages'.format(ch_pair),
                               unit='#',
                               label=('The number of integration averages' +
                                      'of ch_pair {} all weights'.format(ch_pair)),
                               get_cmd=sintavgall_cmd + '?',
                               set_cmd=sintavgall_cmd + ' {}',
                               vals=vals.Numbers(1, 131072)
                               )
            stvsegall_cmd = 'qutech:tvmode{}:nsegments:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_nsegments'.format(ch_pair),
                               unit='#',
                               label=('The number of TV segments of' +
                                      'ch_pair {} all weights '.format(ch_pair)),
                               get_cmd=stvsegall_cmd + '?',
                               set_cmd=stvsegall_cmd + ' {}',
                               vals=vals.Numbers(1, 256)
                               )
            stvenall_cmd = 'qutech:tvmode{}:enable:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_enable'.format(ch_pair),
                               label=('Enable tv mode' +
                                      'ch_pair {} all weights '.format(ch_pair)),
                               get_cmd=stvenall_cmd + '?',
                               set_cmd=stvenall_cmd + ' {}',
                               vals=vals.Numbers(0, 1)
                               # vals=vals.Numbers(1,4096)
                               )
            '''
            Threshold
            '''
            sthlall_cmd = 'qutech:qstate{}:threshold:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_qstate_threshold'.format(ch_pair),
                               unit='#',
                               label=('Set threshold of' +
                                      'ch_pair {} all weights'.format(ch_pair)),
                               get_cmd=sthlall_cmd + '?',
                               set_cmd=sthlall_cmd + ' {}',
                               vals=vals.Numbers(-134217728, 134217727)
                               )
            '''
            Logging
            '''
            slogenall_cmd = 'qutech:logging{}:enable:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_logging_enable'.format(ch_pair),
                               label=('Enable logging mode' +
                                      'ch_pair {} all weights'.format(ch_pair)),
                               get_cmd=slogenall_cmd + '?',
                               set_cmd=slogenall_cmd + ' {}',
                               vals=vals.Numbers(0, 1)
                               # vals=vals.Numbers(1,4096)
                               )

            slogshotsall_cmd = 'qutech:logging{}:nshots:all'.format(ch_pair)
            self.add_parameter('ch_pair{}_logging_nshots'.format(ch_pair),
                               unit='#',
                               label=('The number of logging shots of' +
                                      'ch_pair {} all weights'.format(ch_pair)),
                               get_cmd=slogshotsall_cmd + '?',
                               set_cmd=slogshotsall_cmd + ' {}',
                               vals=vals.Numbers(1, 8192)
                               )
            '''
            Error fraction
            '''
            serrfarcten_cmd = 'qutech:errorfraction{}:enable:all'.format(
                ch_pair)
            self.add_parameter('ch_pair{}_err_fract_enable'.format(ch_pair),
                               label=('Enable error fraction mode' +
                                      'ch_pair {} all weights '.format(ch_pair)),
                               get_cmd=serrfarcten_cmd + '?',
                               set_cmd=serrfarcten_cmd + ' {}',
                               vals=vals.Numbers(0, 1)
                               )

            serrfractshots_cmd = 'qutech:errorfraction{}:nshots:all'.format(
                ch_pair)
            self.add_parameter('ch_pair{}_err_fract_nshots'.format(ch_pair),
                               unit='#',
                               label=('The number of error fraction shots of' +
                                      'ch_pair {} all weights '.format(ch_pair)),
                               get_cmd=serrfractshots_cmd + '?',
                               set_cmd=serrfractshots_cmd + ' {}',
                               vals=vals.Numbers(1, 2097152)
                               )

            self.add_parameter('ch_pair{}_err_fract_pattern'.format(ch_pair),
                               label=('Get error fraction pattern ' +
                                      'ch_pair {}'.format(ch_pair)),
                               set_cmd=self._gen_ch_set_func(
                                   self._sendErrFractSglQbitPatternAll, ch_pair),
                               get_cmd=self._gen_ch_get_func(
                                   self._getErrFractSglQbitPatternAll, ch_pair),

                               vals=vals.Arrays(0, 1)
                               )

            for i in range(self.device_descriptor.numWeights):
                wNr = i+1
                '''
                Weighted integral + Rotation matrix + TV mode parameters
                '''
                swinten_cmd = 'qutech:wint{}:enable{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_wint_enable'.format(ch_pair, wNr),
                                   label=('Enable wighted integral' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=swinten_cmd + '?',
                                   set_cmd=swinten_cmd + ' {}',
                                   vals=vals.Numbers(0, 1)
                                   )

                sintlength_cmd = 'qutech:wint{}:intlength{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_wint_intlength'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('The number of sample  that' +
                                          'is used for one integration of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=sintlength_cmd + '?',
                                   set_cmd=sintlength_cmd + ' {}',
                                   vals=vals.Numbers(1, 2048)
                                   )
                swintstat_cmd = 'qutech:wint{}:status{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_wint_status'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('Weighted integral status of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=swintstat_cmd + '?'
                                   # vals=vals.Numbers(1,2048)
                                   )
                srotmat00_cmd = 'qutech:rotmat{}:rotmat00{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_rotmat_rotmat00'.format(ch_pair, wNr, wNr),
                                   unit='#',
                                   label=('Rotation matrix value 00 of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=srotmat00_cmd + '?',
                                   set_cmd=srotmat00_cmd + ' {}',
                                   vals=vals.Numbers(-2, 1.99976)
                                   )
                srotmat01_cmd = 'qutech:rotmat{}:rotmat01{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_rotmat_rotmat01'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('Rotation matrix value 00 of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=srotmat01_cmd + '?',
                                   set_cmd=srotmat01_cmd + ' {}',
                                   vals=vals.Numbers(-2, 1.99976)
                                   )
                srotmat10_cmd = 'qutech:rotmat{}:rotmat10{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_rotmat_rotmat10'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('Rotation matrix value 10 of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=srotmat10_cmd + '?',
                                   set_cmd=srotmat10_cmd + ' {}'
                                   # vals=vals.Numbers(1,2048)
                                   )
                srotmat11_cmd = 'qutech:rotmat{}:rotmat11{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_rotmat_rotmat11'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('Rotation matrix value 11 of' +
                                          'ch_pair {} weight {} '.format(ch_pair, wNr)),
                                   get_cmd=srotmat11_cmd + '?',
                                   set_cmd=srotmat11_cmd + ' {}',
                                   vals=vals.Numbers(-2, 1.99976)
                                   )

                '''
                TV mode parameters
                '''
                sintavg_cmd = 'qutech:tvmode{}:naverages{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_tvmode_naverages'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('The number of integration averages' +
                                          'of ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=sintavg_cmd + '?',
                                   set_cmd=sintavg_cmd + ' {}',
                                   vals=vals.Numbers(1, 131072)
                                   )
                stvseg_cmd = 'qutech:tvmode{}:nsegments{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_tvmode_nsegments'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('The number of TV segments of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=stvseg_cmd + '?',
                                   set_cmd=stvseg_cmd + ' {}',
                                   vals=vals.Numbers(1, 256)
                                   )
                stven_cmd = 'qutech:tvmode{}:enable{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_tvmode_enable'.format(ch_pair, wNr),
                                   label=('Enable tv mode' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=stven_cmd + '?',
                                   set_cmd=stven_cmd + ' {}',
                                   vals=vals.Numbers(0, 1)
                                   # vals=vals.Numbers(1,4096)
                                   )

                self.add_parameter('ch_pair{}_weight{}_tvmode_data'.format(ch_pair, wNr),
                                   label=(
                                       'Get TV data channel {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                       self._getTVdata, ch_pair, wNr)

                                   # vals=vals.Numbers(-128,127)
                                   )

                '''
                TV mode QSTATE parameters
                '''
                sthl_cmd = 'qutech:qstate{}:threshold{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_qstate_threshold'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('Set threshold of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=sthl_cmd + '?',
                                   set_cmd=sthl_cmd + ' {}',
                                   vals=vals.Numbers(-134217728, 134217727)
                                   )
                self.add_parameter('ch_pair{}_weight{}_qstate_cnt_data'.format(ch_pair, wNr),
                                   label=('Get qstate counter' +
                                          'ch_pair {} weight {} '.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                   self._getQstateCNT, ch_pair, wNr),

                                   # vals=vals.Numbers(-128,127)
                                   )
                self.add_parameter('ch_pair{}_weight{}_qstate_avg_data'.format(ch_pair, wNr),
                                   label=('Get qstate average' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                    self._getQstateAVG, ch_pair, wNr),

                    # vals=vals.Numbers(-128,127)
                )
                '''
                Logging
                '''
                slogen_cmd = 'qutech:logging{}:enable{}'.format(ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_logging_enable'.format(ch_pair, wNr),
                                   label=('Enable logging mode' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=slogen_cmd + '?',
                                   set_cmd=slogen_cmd + ' {}',
                                   vals=vals.Numbers(0, 1)
                                   # vals=vals.Numbers(1,4096)
                                   )

                slogshots_cmd = 'qutech:logging{}:nshots{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_logging_nshots'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('The number of logging shots of' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=slogshots_cmd + '?',
                                   set_cmd=slogshots_cmd + ' {}',
                                   vals=vals.Numbers(1, 8192)
                                   )

                self.add_parameter('ch_pair{}_weight{}_logging_int'.format(ch_pair, wNr),
                                   label=('Get integration logging ' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                       self._getLoggingInt, ch_pair, wNr),

                                   )
                self.add_parameter('ch_pair{}_weight{}_logging_qstate'.format(ch_pair, wNr),
                                   label=('Get qstate logging ' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                       self._getLoggingQstate, ch_pair, wNr),

                                   # vals=vals.Numbers(-128,127)
                                   )
                '''
                Error fraction
                '''
                serrfarcten_cmd = 'qutech:errorfraction{}:enable{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_err_fract_enable'.format(ch_pair, wNr),
                                   label=('Enable error fraction mode' +
                                          'ch_pair {} weight {} '.format(ch_pair, wNr)),
                                   get_cmd=serrfarcten_cmd + '?',
                                   set_cmd=serrfarcten_cmd + ' {}',
                                   vals=vals.Numbers(0, 1)
                                   )

                serrfractshots_cmd = 'qutech:errorfraction{}:nshots{}'.format(
                    ch_pair, wNr)
                self.add_parameter('ch_pair{}_weight{}_err_fract_nshots'.format(ch_pair, wNr),
                                   unit='#',
                                   label=('The number of error fraction shots of' +
                                          'ch_pair {} weight {} '.format(ch_pair, wNr)),
                                   get_cmd=serrfractshots_cmd + '?',
                                   set_cmd=serrfractshots_cmd + ' {}',
                                   vals=vals.Numbers(1, 2097152)
                                   )
                self.add_parameter('ch_pair{}_weight{}_err_fract_cnt'.format(ch_pair, wNr),
                                   label=('Get all error fraction counters ' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                       self._getErrFractCnt, ch_pair, wNr)

                                   # vals=vals.Numbers(-128,127)
                                   )

                self.add_parameter('ch_pair{}_weight{}_err_fract_pattern'.format(ch_pair, wNr),
                                   label=('Get error fraction pattern ' +
                                          'ch_pair {} weight {}'.format(ch_pair, wNr)),
                                   set_cmd=self._gen_ch_weight_set_func(
                                       self._sendErrFractSglQbitPattern, ch_pair, wNr),
                                   get_cmd=self._gen_ch_weight_get_func(
                                       self._getErrFractSglQbitPattern, ch_pair, wNr),

                                   vals=vals.Arrays(0, 1)
                                   )
        '''
        Sorted by channel
        '''
        for i in range(self.device_descriptor.numChannels):
            ch = i+1
            self.add_parameter('ch{}_inavg_data'.format(ch),
                               label=('Get data ch {} '.format(ch)),
                               get_cmd=self._gen_ch_get_func(
                                   self._getInputAverage, ch),
                               vals=vals.Arrays(-128, 127)
                               )
            srotres_cmd = 'qutech:wintrot{}:result'.format(ch)
            self.add_parameter('ch{}_wintrot_result'.format(ch),
                               unit='#',

                               label=('Rotated integration result of' +
                                      'channel {} '.format(ch)),
                               get_cmd=srotres_cmd + '?'
                               # vals=vals.Numbers(-134217728,134217727)
                               )
            for i in range(self.device_descriptor.numWeights):
                wNr = i+1
                self.add_parameter('ch{}_weight{}_data'.format(ch, wNr),
                                   label=('Get weight data channel {}' +
                                          'weight number {}  '.format(ch, wNr)),
                                   get_cmd=self._gen_ch_weight_get_func(
                                   self._getWeightData, ch, wNr),
                                   set_cmd=self._gen_ch_weight_set_func(
                                   self._sendWeightData, ch, wNr),
                                   vals=vals.Arrays(-128, 127)
                                   )

    def _getInputAverage(self, ch):
        ch_pair = math.ceil(ch/2)
        finished = 0
        while (finished != '1'):
            finished = self._getInAvgFinished(ch_pair)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        # print(finished)
        self.write('qutech:inputavg{:d}:data? '.format(ch))
        binBlock = self.binBlockRead()
        #print('inputavgLen in bytes {:d}'.format(len(binBlock)))
        inputavg = np.frombuffer(binBlock, dtype=np.float32)
        return inputavg

    def _getTVdata(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getTVFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        self.write('qutech:tvmode{:d}:data{:d}? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('TV data in bytes {:d}'.format(len(binBlock)))
        tvmodedata = np.frombuffer(binBlock, dtype=np.float32)
        return tvmodedata

    def _sendWeightData(self, ch, wNr,  weight):
        # generate the binblock
        if 1:   # high performance
            arr = np.asarray(weight, dtype=np.int8)
            binBlock = arr.tobytes()
        else:   # more generic
            binBlock = b''
            for i in range(len(weight)):
                binBlock = binBlock + struct.pack('<f', weight[i])

        # write binblock
        hdr = 'qutech:wint:data {:d}, {:d},'.format(ch, wNr)
        self.binBlockWrite(binBlock, hdr)

    def _getWeightData(self, ch, wNr):

        self.write('qutech:wint{:d}:data{:d}? '.format(ch, wNr))
        binBlock = self.binBlockRead()
        #print('Weight data in bytes {:d}'.format(len(binBlock)))
        weightdata = np.frombuffer(binBlock, dtype=np.int8)
        return weightdata

    def _getQstateCNT(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getTVFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break

        self.write('qutech:qstate{:d}:data{:d}:counter? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('Qu state counter {:d}'.format(len(binBlock)))
        qstatecnt = np.frombuffer(binBlock, dtype=np.float32)
        return qstatecnt

    def _getQstateAVG(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getTVFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        self.write('qutech:qstate{:d}:data{:d}:average? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('Qu state average {:d}'.format(len(binBlock)))
        qstateavg = np.frombuffer(binBlock, dtype=np.float32)
        return qstateavg

    def _getLoggingInt(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getLoggingFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        self.write('qutech:logging{:d}:data{:d}:int? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('Qu state average {:d}'.format(len(binBlock)))
        intlogging = np.frombuffer(binBlock, dtype=np.float32)
        return intlogging

    def _getLoggingQstate(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getLoggingFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        self.write('qutech:logging{:d}:data{:d}:qstate? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('Qu state average {:d}'.format(len(binBlock)))
        qstatelogging = np.frombuffer(binBlock, dtype=np.float32)
        return qstatelogging
    '''
    Input averaging
    '''

    def _getInAvgStatus(self, ch_pair):
        return self.ask('qutech:inputavg{:d}:status? '.format(ch_pair))

    def _getInAvgFinished(self, ch_pair):
        finished = self.ask('qutech:inputavg{:d}:finished? '.format(ch_pair))
        fmt_finished = format(int(finished), 'x')
        return fmt_finished

    def _getInAvgBusy(self, ch_pair):
        return self.ask('qutech:inputavg{:d}:busy? '.format(ch_pair))
    '''
    TV MODe
    '''

    def _getTVStatus(self, ch_pair, wNr):
        return self.ask('qutech:tvmode{:d}:status{:d}? '.format(ch_pair, wNr))

    def _getTVFinished(self, ch_pair, wNr):
        finished = self.ask(
            'qutech:tvmode{:d}:finished{:d}? '.format(ch_pair, wNr))
        fmt_finished = format(int(finished), 'x')
        return fmt_finished

    def _getTVBusy(self, ch_pair, wNr):
        return self.ask('qutech:tvmode{:d}:busy{:d}? '.format(ch_pair, wNr))

    def _getTVpercentage(self, ch_pair, wNr):
        return self.ask('qutech:tvmode{:d}:percentage{:d}? '.format(ch_pair, wNr))
    '''
    Logging
    '''

    def _getLoggingFinished(self, ch_pair, wNr):
        finished = self.ask(
            'qutech:logging{:d}:finished{:d}? '.format(ch_pair, wNr))
        fmt_finished = format(int(finished), 'x')
        return fmt_finished

    def _getLoggingBusy(self, ch_pair, wNr):
        return self.ask('qutech:logging{:d}:busy{:d}? '.format(ch_pair, wNr))

    def _getLoggingpercentage(self, ch_pair, wNr):
        return self.ask('qutech:logging{:d}:percentage{:d}? '.format(ch_pair, wNr))

    def _getLoggingStatus(self, ch_pair, wNr):
        return self.ask('qutech:errorfraction{:d}:status{:d}? '.format(ch_pair, wNr))
    '''
    Error fraction counters
    '''

    def _sendErrFractSglQbitPatternAll(self, ch_pair, pattern):
        self.write('qutech:errorfraction{:d}:pattern:all {:d},{:d}'.format(
                   ch_pair, pattern[0], pattern[1]))

    def _getErrFractSglQbitPatternAll(self, ch_pair):
        pstring = self.ask(
            'qutech:errorfraction{:d}:pattern:all? '.format(ch_pair))
        P = np.zeros(2)
        for i, x in enumerate(pstring.split(',')):
            P[i] = x
        return (P)

    def _sendErrFractSglQbitPattern(self, ch_pair, wNr, pattern):
        self.write('qutech:errorfraction{:d}:pattern{:d} {:d},{:d}'.format(
                   ch_pair, wNr, pattern[0], pattern[1]))

    def _getErrFractSglQbitPattern(self, ch_pair, wNr):
        pstring = self.ask(
            'qutech:errorfraction{:d}:pattern{:d}? '.format(ch_pair, wNr))
        P = np.zeros(2)
        for i, x in enumerate(pstring.split(',')):
            P[i] = x
        return (P)

    def _getErrFractCnt(self, ch_pair, wNr):
        finished = 0
        complete = 0
        # while (complete != '1'):
        #    complete = self._getTVpercentage(ch_pair)
        while (finished != '1'):
            finished = self._getErrFractFinished(ch_pair, wNr)
            if (finished == 'ffffffff'):
                logging.warning('Trigger is not received: DDM timeout')
                break
        self.write('qutech:errorfraction{:d}:data{:d}? '.format(ch_pair, wNr))
        binBlock = self.binBlockRead()
        #print('Qu state average {:d}'.format(len(binBlock)))
        errfractioncnt = np.frombuffer(binBlock, dtype=np.int32)
        print('NoErrorCounterReg    = {:d}'.format(errfractioncnt[0]))
        print('SingleErrorCounterReg= {:d}'.format(errfractioncnt[1]))
        print('DoubleErrorCounterReg= {:d}'.format(errfractioncnt[2]))
        print('ZeroStateCounterReg  = {:d}'.format(errfractioncnt[3]))
        print('OneStateCounterReg   = {:d}'.format(errfractioncnt[4]))
        return errfractioncnt

    def _getErrFractFinished(self, ch_pair, wNr):
        finished = self.ask(
            'qutech:errorfraction{:d}:finished{:d}? '.format(ch_pair, wNr))
        fmt_finished = format(int(finished), 'x')
        return fmt_finished

    def _getErrFractBusy(self, ch_pair, wNr):
        return self.ask('qutech:errorfraction{:d}:busy{:d}? '.format(ch_pair, wNr))

    def _getErrFractpercentage(self, ch_pair, wNr):
        return self.ask('qutech:errorfraction{:d}:percentage{:d}? '.format(ch_pair, wNr))

    def _getErrFractStatus(self, ch_pair, wNr):
        return self.ask('qutech:errorfraction{:d}:status{:d}? '.format(ch_pair, wNr))

    def _getADCstatus(self, ch_pair):
        status = self.ask('qutech:adc{:d}:status? '.format(ch_pair))
        statusstr = format(int(status), 'b')
        reversestatusstr = statusstr[::-1]

        def _DI():
            if (reversestatusstr[0] == '1'):
                logging.warning('\nOver range on DI input. ')
            elif(reversestatusstr[1] == '1'):
                logging.warning('\nUnder range on DI input.' +
                                'Input signal is less than 25% of ADC resolution. ')
            else:
                print("\nDI input is Okay.")
            return ((reversestatusstr[0] != '1') & (reversestatusstr[1] != '1'))

        def _DQ():
            if (reversestatusstr[2] == '1'):
                logging.warning('\nOver range on DQ input. ')
            elif(reversestatusstr[3] == '1'):
                logging.warning('\nUnder range on DQ input.' +
                                ' Input signal is less than 25% of ADC resolution. ')
            else:
                print("\nDQ input is Okay.")
            return ((reversestatusstr[2] != '1') & (reversestatusstr[3] != '1'))

        def _DCLK_PLL_LOCKED():
            if (reversestatusstr[4] == '1'):
                print("\nDCLK PLL has a phase lock. There is an ADC clock.")
            else:
                print("\nThere is no ADC clock.")
            return reversestatusstr[4] == '1'

        def _CalRun():
            if (reversestatusstr[5] == '1'):
                logging.warning("\nADC calibration is in progress.")
            else:
                print("\nADC calibration is not in progress.")
            return reversestatusstr[5] == '1'
        ADCstatus = {0: _DI,
                     1: _DQ,
                     2: _DCLK_PLL_LOCKED,
                     3: _CalRun,
                     }
        for x in range(0, 4):
            print(ADCstatus[x]())
        return statusstr

    def _getTHL(self, ch_pair, weight_nr):
        ret = self.ask(
            'qutech:qstate{}:threshold{:d}?'.format(ch_pair, weight_nr))
        return ret

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_weight_get_func(self, fun, ch, wNr):
        def get_func():
            return fun(ch, wNr)
        return get_func

    def _gen_ch_weight_set_func(self, fun, ch, wNr):
        def set_func(val):
            return fun(ch, wNr, val)
        return set_func

    def get_idn(self):
        try:
            idstr = ''  # in case self.ask fails
            idstr = self.ask('*IDN?')
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            for separator in ',;:':
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 8)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 9:
                idparts += [None] * (9 - len(idparts))
            for i in range(0, 9):
                idparts[i] = idparts[i].split('=')[1]
        except:
            logging.warn('Error getting or interpreting *IDN?: ' + repr(idstr))
            idparts = [None, None, None, None, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[9:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'fwVersion', 'fwBuild',
                         'swVersion', 'swBuild', 'kmodVersion',
                         'kmodBuild'), idparts))
        print(idparts)

    def connect_message(self, idn_param='IDN', begin_time=None):
        idn = {'vendor': None, 'model': None,
               'serial': None, 'fwVersion': None,
               'swVersion': None, 'kmodVersion': None
               }
        idn.update(self.get(idn_param))
        t = time.time() - (begin_time or self._t0)

        con_msg = ('Connected to: {vendor} {model} '
                   '(serial:{serial}, fwVersion:{fwVersion} '
                   'swVersion:{swVersion}, kmodVersion:{kmodVersion}) '
                   'in {t:.2f}s'.format(t=t, **idn))
        print(con_msg)

    #initialization functions
    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=1,
                                        weight_function_Q=2):
        trace_length = 4096
        tbase = np.arange(0, trace_length/5e8, 1/5e8)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        #first pair
        eval('self.ch1_weight{}_data(np.array(cosI))'.format(weight_function_I))
        eval('self.ch2_weight{}_data(np.array(sinI))'.format(weight_function_I))
        #second pair
        eval('self.ch1_weight{}_data(np.array(sinI))'.format(weight_function_Q))
        eval('self.ch2_weight{}_data(np.array(cosI))'.format(weight_function_Q))

        #setting the rotation matrices... very danagerous
        eval('self.ch_pair1_weight{}_rotmat_rotmat00(1)'.format(weight_function_I))
        eval('self.ch_pair1_weight{}_rotmat_rotmat01(1)'.format(weight_function_I))
        eval('self.ch_pair1_weight{}_rotmat_rotmat00(1)'.format(weight_function_Q))
        eval('self.ch_pair1_weight{}_rotmat_rotmat01(-1)'.format(weight_function_Q))






# @ NIKITA LOOK HERE!
class MyCustomValidator(vals.Validator):

    """
    Validator for numerical numpy arrays
    Args:
        min_value (Optional[Union[float, int]):  Min value allowed, default inf
        max_value:  (Optional[Union[float, int]): Max  value allowed, default inf
        shape:     (Optional): None
    """

    validtypes = (int, float, np.integer, np.floating)

    def __init__(self, min_value=-float("inf"), max_value=float("inf"),
                 shape=None):

        if isinstance(min_value, self.validtypes):
            self._min_value = min_value
        else:
            raise TypeError('min_value must be a number')

        if isinstance(max_value, self.validtypes) and max_value > min_value:
            self._max_value = max_value
        else:
            raise TypeError('max_value must be a number bigger than min_value')
        self._shape = shape

    def validate(self, value, context=''):

        if not isinstance(value, np.ndarray):
            raise TypeError(
                '{} is not a numpy array; {}'.format(repr(value), context))

        if value.dtype not in self.validtypes:
            raise TypeError(
                '{} is not an int or float; {}'.format(repr(value), context))
        if self._shape is not None:
            if (np.shape(value) != self._shape):
                raise ValueError(
                    '{} does not have expected shape {}; {}'.format(
                        repr(value), self._shape, context))

        # Only check if max is not inf as it can be expensive for large arrays
        if self._max_value != (float("inf")):
            if not (np.max(value) <= self._max_value):
                raise ValueError(
                    '{} is invalid: all values must be between '
                    '{} and {} inclusive; {}'.format(
                        repr(value), self._min_value,
                        self._max_value, context))

        # Only check if min is not -inf as it can be expensive for large arrays
        if self._min_value != (-float("inf")):
            if not (self._min_value <= np.min(value)):
                raise ValueError(
                    '{} is invalid: all values must be between '
                    '{} and {} inclusive; {}'.format(
                        repr(value), self._min_value,
                        self._max_value, context))

    is_numeric = True

    def __repr__(self):
        minv = self._min_value if math.isfinite(self._min_value) else None
        maxv = self._max_value if math.isfinite(self._max_value) else None
        return '<Arrays{}, shape: {}>'.format(range_str(minv, maxv, 'v'),
                                              self._shape)


