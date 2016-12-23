'''
    File:               DDM.py
    Author:             Nikita Vodyagin, QuTech
    Purpose:            control of Qutech DDM
    Prerequisites:
    Usage:
    Bugs:
'''


from .SCPIddm import SCPIddm
import numpy as np
import struct
import array
from qcodes import validators as vals


class DDMq(SCPIddm):

   # def __init__(self, logging=True, simMode=False, paranoid=False):
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)
        # AWG properties (to be overloaded by derived class)
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'DDM'
        self.device_descriptor.numChannels = 2
        # self.numDacBits = 0
        # self.numMarkersPerChannel = 0
        # self.numMarkers = 0
        self.add_parameters()
        self.connect_message()
        # SCPIdriver.__init__(self, logging, simMode, paranoid)






    def add_parameters(self):
        #######################################################################
        # DDM specific
        #######################################################################
        
        
            #self.add_function('reset', call_cmd='*RST')
        self.add_parameter('ID_number', get_cmd='*IDN?')

        for i in range(self.device_descriptor.numChannels//2):
            ch_pair=2*i+1
            snavg_cmd= 'qutech:inputavg{}:naverages'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_Navg'.format(ch_pair),
                                units='#',
                                label=('number of averages chpair {} '.format(ch_pair)),
                                get_cmd=snavg_cmd + '?',
                                set_cmd=snavg_cmd + ' {}',
                                vals=vals.Numbers(1,32768)
                              )
            snsamp_cmd= 'qutech:inputavg{}:scansize'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_scansize'.format(ch_pair),
                                units='#',
                                label=('number of pair samples chpair {} '.format(ch_pair)),
                                get_cmd=snsamp_cmd + '?',
                                set_cmd=snsamp_cmd + ' {}',
                                vals=vals.Numbers(1,4096)
                              )
            senable_cmd= 'qutech:inputavg{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_enable'.format(ch_pair),
                                label=('enable input averaging chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=senable_cmd
                                #vals=vals.Numbers(1,4096)
                              )
            sdisable_cmd= 'qutech:inputavg{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_disable'.format(ch_pair),
                                label=('disable input averaging chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=sdisable_cmd,
                                #vals=vals.Bool()
                              )
            sholdoff_cmd= 'qutech:input{}:holdoff'.format(ch_pair)
            self.add_parameter('ch_pair{}_inavg_holdoff'.format(ch_pair),
                                label=('set holdoff chpair {} '.format(ch_pair)),
                                get_cmd=sholdoff_cmd + '?',
                                set_cmd=sholdoff_cmd + ' {}',
                                vals=vals.Numbers(0,127)
                              )
            
        for i in range(self.device_descriptor.numChannels):
            ch=i+1
            sgetdata_cmd= 'qutech:inputavg{}:data'.format(ch)
            self.add_parameter('ch{}_inavg_data'.format(ch),
                                label=('get data ch {} '.format(ch)),
                                get_cmd=self._gen_ch_get_func(self.getInputAverage,ch),
                       
                                vals=vals.Numbers(-128,127)
                              )

    def getInputAverage(self,ch):
        
        self.write('qutech:inputavg{:d}:data? '.format(ch))
        binBlock = self.binBlockRead()
        #print(binBlock)
        print ('inputavgLen in bytes {:d}'.format(len(binBlock)))
        inputavg = np.frombuffer(binBlock,dtype=np.float32)
        return inputavg

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

