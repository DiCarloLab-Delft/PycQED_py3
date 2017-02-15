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

logging.warning('my warning ')

class DDMq(SCPI):

    # def __init__(self, logging=True, simMode=False, paranoid=False):
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'DDM'
        self.device_descriptor.numChannels = 2
        self.add_parameters()
        self.connect_message()

    def add_parameters(self):
        #######################################################################
        # DDM specific
        #######################################################################

        for i in range(self.device_descriptor.numChannels//2):
            ch_pair = 2*i+1
            snavg_cmd = 'qutech:inputavg{}:naverages'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_Navg'.format(ch_pair),
                               units='#',
                               label=('number of averages' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=snavg_cmd + '?',
                               set_cmd=snavg_cmd + ' {}',
                               vals=vals.Numbers(1, 32768)
                               )
            snsamp_cmd = 'qutech:inputavg{}:scansize'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_scansize'.format(ch_pair),
                               units='#',
                               label=('number of samples' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=snsamp_cmd + '?',
                               set_cmd=snsamp_cmd + ' {}',
                               vals=vals.Numbers(1, 4096)
                               )
            senable_cmd = 'qutech:inputavg{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_enable'.format(ch_pair),
                               label=('enable input averaging' +
                                      'chpair {} '.format(ch_pair)),
                               get_cmd=senable_cmd + '?',
                               set_cmd=senable_cmd
                               # vals=vals.Numbers(1,4096)
                               )
            sdisable_cmd = 'qutech:inputavg{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_disable'.format(ch_pair),
                               label=('disable input averaging' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=senable_cmd + '?',
                               set_cmd=sdisable_cmd,
                               # vals=vals.Bool()
                               )
            sholdoff_cmd = 'qutech:input{}:holdoff'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_holdoff'.format(ch_pair),
                               label=('set holdoff' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=sholdoff_cmd + '?',
                               set_cmd=sholdoff_cmd + ' {}',
                               vals=vals.Numbers(0, 127)
                               )
            '''
            Weighted integral + Rotation matrix + TV mode parameters
            '''
            swinten_cmd = 'qutech:wint{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_enable'.format(ch_pair),
                               label=('enable wighted integral' +
                                      'chpair {} '.format(ch_pair)),
                               get_cmd=swinten_cmd + '?',
                               set_cmd=swinten_cmd
                               # vals=vals.Numbers(1,4096)
                               )
            swintdis_cmd = 'qutech:wint{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_disable'.format(ch_pair),
                               label=('Disable wighted integral' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=swinten_cmd + '?',
                               set_cmd=swintdis_cmd,
                               # vals=vals.Bool()
                               )
            sintlength_cmd = 'qutech:wint{}:intlength'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_intlength'.format(ch_pair),
                               units='#',
                               label=('the number of sample pairs that' +
                                      'is used for one integration of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=sintlength_cmd + '?',
                               set_cmd=sintlength_cmd + ' {}',
                               vals=vals.Numbers(1, 2048)
                               )
            swintstat_cmd = 'qutech:wint{}:status'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_status'.format(ch_pair),
                               units='#',
                               label=('weighted integral status of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=swintstat_cmd + '?'
                               # vals=vals.Numbers(1,2048)
                               )
            srotmat00_cmd = 'qutech:rotmat{}:rotmat00'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat00'.format(ch_pair),
                               units='#',
                               label=('rotation matrix value 00 of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=srotmat00_cmd + '?',
                               set_cmd=srotmat00_cmd + ' {}'
                               # vals=vals.Numbers(1,2048)
                               )
            srotmat01_cmd = 'qutech:rotmat{}:rotmat01'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat01'.format(ch_pair),
                               units='#',
                               label=('rotation matrix value 00 of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=srotmat01_cmd + '?',
                               set_cmd=srotmat01_cmd + ' {}'
                               # vals=vals.Numbers(1,2048)
                               )
            srotmat10_cmd = 'qutech:rotmat{}:rotmat10'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat10'.format(ch_pair),
                               units='#',
                               label=('rotation matrix value 10 of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=srotmat10_cmd + '?',
                               set_cmd=srotmat10_cmd + ' {}'
                               # vals=vals.Numbers(1,2048)
                               )
            srotmat11_cmd = 'qutech:rotmat{}:rotmat11'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat11'.format(ch_pair),
                               units='#',
                               label=('rotation matrix value 11 of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=srotmat11_cmd + '?',
                               set_cmd=srotmat11_cmd + ' {}'
                               # vals=vals.Numbers(1,2048)
                               )
            '''
            TV mode parameters
            '''
            sintavg_cmd = 'qutech:tvmode{}:naverages'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_naverages'.format(ch_pair),
                               units='#',
                               label=('the number of integration averages' +
                                      'of ch_pair {} '.format(ch_pair)),
                               get_cmd=sintavg_cmd + '?',
                               set_cmd=sintavg_cmd + ' {}',
                               vals=vals.Numbers(1, 131072)
                               )
            stvseg_cmd = 'qutech:tvmode{}:nsegments'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_nsegments'.format(ch_pair),
                               units='#',
                               label=('the number of TV segments of' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=stvseg_cmd + '?',
                               set_cmd=stvseg_cmd + ' {}',
                               vals=vals.Numbers(1, 256)
                               )
            stven_cmd = 'qutech:tvmode{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_enable'.format(ch_pair),
                               label=('enable tv mode' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=stven_cmd + '?',
                               set_cmd=stven_cmd
                               # vals=vals.Numbers(1,4096)
                               )
            stvdis_cmd = 'qutech:tvmode{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_disable'.format(ch_pair),
                               label=('disable tv mode' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=stven_cmd + '?',
                               set_cmd=stvdis_cmd,
                               # vals=vals.Bool()
                               )
            self.add_parameter('ch_pair{}_tvmode_data'.format(ch_pair),
                               label=('get TV data ch {} '.format(ch_pair)),
                               get_cmd=self._gen_ch_get_func(
                                   self.getTVdata, ch_pair)

                               # vals=vals.Numbers(-128,127)
                               )
            scaladc_cmd = 'qutech:caladc{}'.format(ch_pair)
            self.add_parameter('ch_pair{}_cal_adc'.format(ch_pair),
                               label=('Calibrate ADC{}'.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                               set_cmd=scaladc_cmd
                               # vals=vals.Numbers(1,4096)
                               )
            sthl_cmd = 'qutech:qstate{}:threshold'.format(ch_pair)
            self.add_parameter('ch_pair{}_qstate_threshold'.format(ch_pair),
                               units='#',
                               label=('set threshold of' +
                               'ch_pair {} '.format(ch_pair)),
                               get_cmd=sthl_cmd + '?',
                               set_cmd=sthl_cmd + ' {}'
                               # vals=vals.Numbers(-134217728,134217727)
                               )
            self.add_parameter('ch_pair{}_qstate_cnt_data'.format(ch_pair),
                               label=('get qstate counter' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=self._gen_ch_get_func(
                                   self.getQstateCNT, ch_pair),

                               # vals=vals.Numbers(-128,127)
                               )
            self.add_parameter('ch_pair{}_qstate_avg_data'.format(ch_pair),
                               label=('get qstate average' +
                                      'ch_pair {} '.format(ch_pair)),
                               get_cmd=self._gen_ch_get_func(
                                   self.getQstateAVG, ch_pair),

                               # vals=vals.Numbers(-128,127)
                               )
        for i in range(self.device_descriptor.numChannels):
            ch = i+1
            self.add_parameter('ch{}_inavg_data'.format(ch),
                               label=('get data ch {} '.format(ch)),
                               get_cmd=self._gen_ch_get_func(
                                   self.getInputAverage, ch),

                               #vals=vals.Numbers(-128, 127)
                               vals=vals.Arrays(-128, 127)
                               )
            srotres_cmd = 'qutech:wintrot{}:result'.format(ch)
            self.add_parameter('ch{}_wintrot_result'.format(ch),
                               units='#',
                               label=('rotated integration result of' +
                                      'channel {} '.format(ch)),
                               get_cmd=srotres_cmd + '?'
                               # vals=vals.Numbers(-134217728,134217727)
                               )
            self.add_parameter('ch{}_weight_data'.format(ch),
                               label=('get weigtht data ch {} '.format(ch)),
                               get_cmd=self._gen_ch_get_func(
                                   self.getWeightData, ch),
                               set_cmd=self._gen_ch_set_func(
                                   self.sendWeightData, ch),
                               vals=vals.Arrays(-128, 127)
                               )

    def getInputAverage(self, ch):
        ch_pair = math.ceil(ch/2)
        finished = 0
        while (finished != '1'):
            finished = self.getInAvgFinished(ch_pair)
        print(finished)
        self.write('qutech:inputavg{:d}:data? '.format(ch))
        binBlock = self.binBlockRead()
        print('inputavgLen in bytes {:d}'.format(len(binBlock)))
        inputavg = np.frombuffer(binBlock, dtype=np.float32)
        return inputavg

    def getTVdata(self, ch_pair):
        finished = 0
        while (finished != '1'):
            finished = self.getTVFinished(ch_pair)
            print(self.getTVpercentage(ch_pair), end=' ')
        print(finished)
        self.write('qutech:tvmode{:d}:data? '.format(ch_pair))
        binBlock = self.binBlockRead()
        print('TV data in bytes {:d}'.format(len(binBlock)))
        tvmodedata = np.frombuffer(binBlock, dtype=np.float32)
        return tvmodedata

    def sendWeightData(self, ch, weight):
        # generate the binblock
        if 1:   # high performance
            arr = np.asarray(weight, dtype=np.int8)
            binBlock = arr.tobytes()
        else:   # more generic
            binBlock = b''
            for i in range(len(weight)):
                binBlock = binBlock + struct.pack('<f', weight[i])

        # write binblock
        hdr = 'qutech:wint:data {:d},'.format(ch)
        self.binBlockWrite(binBlock, hdr)

    def getWeightData(self, ch):

        self.write('qutech:wint{:d}:data? '.format(ch))
        binBlock = self.binBlockRead()
        #print('Weight data in bytes {:d}'.format(len(binBlock)))
        weightdata = np.frombuffer(binBlock, dtype=np.int8)
        return weightdata

    def getQstateCNT(self, ch_pair):

        self.write('qutech:qstate{:d}:counter:data? '.format(ch_pair))
        binBlock = self.binBlockRead()
        #print('Qu state counter {:d}'.format(len(binBlock)))
        qstatecnt = np.frombuffer(binBlock, dtype=np.float32)
        return qstatecnt

    def getQstateAVG(self, ch_pair):

        self.write('qutech:qstate{:d}:average:data? '.format(ch_pair))
        binBlock = self.binBlockRead()
        #print('Qu state average {:d}'.format(len(binBlock)))
        qstateavg = np.frombuffer(binBlock, dtype=np.float32)
        return qstateavg
    def getInAvgStatus(self, adcnr):
        return self.ask('qutech:inputavg%d:status? ' % (adcnr))
    def getInAvgFinished(self, adcnr):
        return self.ask('qutech:inputavg%d:finished? ' % (adcnr))
    def getInAvgBusy(self, adcnr):
        return self.ask('qutech:inputavg%d:busy? ' % (adcnr))
    def getTVStatus(self, adcnr):
        return self.ask('qutech:tvmode%d:status? ' % (adcnr))
    def getTVFinished(self, adcnr):
        return self.ask('qutech:tvmode%d:finished? ' % (adcnr))
    def getTVBusy(self, adcnr):
        return self.ask('qutech:tvmode%d:busy? ' % (adcnr))
    def getTVpercentage(self, adcnr):
        return self.ask('qutech:tvmode%d:percentage? ' % (adcnr))
    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func




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
