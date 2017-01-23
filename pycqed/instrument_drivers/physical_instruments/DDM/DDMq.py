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
            '''
            Weighted integral + Rotation matrix + TV mode parameters
            '''
            swinten_cmd= 'qutech:wint{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_enable'.format(ch_pair),
                                label=('enable wighted integral chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=swinten_cmd
                                #vals=vals.Numbers(1,4096)
                              )
            swintdis_cmd= 'qutech:wint{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_disable'.format(ch_pair),
                                label=('disable wighted integral chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=swintdis_cmd,
                                #vals=vals.Bool()            
                              )
            sintlength_cmd= 'qutech:wint{}:intlength'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_intlength'.format(ch_pair),
                                units='#',
                                label=('the number of sample pairs that is used for one integration of chpair {} '.format(ch_pair)),
                                get_cmd=sintlength_cmd + '?',
                                set_cmd=sintlength_cmd + ' {}',
                                vals=vals.Numbers(1,2048)
                              )
            swintstat_cmd= 'qutech:wint{}:status'.format(ch_pair)
            self.add_parameter('ch_pair{}_wint_status'.format(ch_pair),
                                units='#',
                                label=('weighted integral status of chpair {} '.format(ch_pair)),
                                get_cmd=swintstat_cmd + '?'
                                #vals=vals.Numbers(1,2048)
                              )
            srotmat00_cmd= 'qutech:rotmat{}:rotmat00'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat00'.format(ch_pair),
                                units='#',
                                label=('rotation matrix value 00 of chpair {} '.format(ch_pair)),
                                get_cmd=srotmat00_cmd + '?',
                                set_cmd=srotmat00_cmd + ' {}'
                                #vals=vals.Numbers(1,2048)
                              )
            srotmat01_cmd= 'qutech:rotmat{}:rotmat01'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat01'.format(ch_pair),
                                units='#',
                                label=('rotation matrix value 00 of chpair {} '.format(ch_pair)),
                                get_cmd=srotmat01_cmd + '?',
                                set_cmd=srotmat01_cmd + ' {}'
                                #vals=vals.Numbers(1,2048)
                              )
            srotmat10_cmd= 'qutech:rotmat{}:rotmat10'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat10'.format(ch_pair),
                                units='#',
                                label=('rotation matrix value 10 of chpair {} '.format(ch_pair)),
                                get_cmd=srotmat10_cmd + '?',
                                set_cmd=srotmat10_cmd + ' {}'
                                #vals=vals.Numbers(1,2048)
                              )
            srotmat11_cmd= 'qutech:rotmat{}:rotmat11'.format(ch_pair)
            self.add_parameter('ch_pair{}_rotmat_rotmat11'.format(ch_pair),
                                units='#',
                                label=('rotation matrix value 11 of chpair {} '.format(ch_pair)),
                                get_cmd=srotmat11_cmd + '?',
                                set_cmd=srotmat11_cmd + ' {}'
                                #vals=vals.Numbers(1,2048)
                              )
            '''
            TV mode parameters
            '''
            sintavg_cmd= 'qutech:tvmode{}:naverages'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_naverages'.format(ch_pair),
                                units='#',
                                label=('the number of integration averages of chpair {} '.format(ch_pair)),
                                get_cmd=sintavg_cmd + '?',
                                set_cmd=sintavg_cmd + ' {}',
                                vals=vals.Numbers(1,1024)
                              )
            stvseg_cmd= 'qutech:tvmode{}:nsegments'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_nsegments'.format(ch_pair),
                                units='#',
                                label=('the number of TV segments of chpair {} '.format(ch_pair)),
                                get_cmd=stvseg_cmd + '?',
                                set_cmd=stvseg_cmd + ' {}',
                                vals=vals.Numbers(1,256)
                              )
            stven_cmd= 'qutech:tvmode{}:enable'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_enable'.format(ch_pair),
                                label=('enable tv mode chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=stven_cmd
                                #vals=vals.Numbers(1,4096)
                              )
            stvdis_cmd= 'qutech:tvmode{}:disable'.format(ch_pair)
            self.add_parameter('ch_pair{}_tvmode_disable'.format(ch_pair),
                                label=('disable tv mode chpair {} '.format(ch_pair)),
                               #get_cmd=snsamp_cmd + '?',
                                set_cmd=stvdis_cmd,
                                #vals=vals.Bool()            
                              )
            self.add_parameter('ch_pair{}_tvmode_data'.format(ch_pair),
                                label=('get TV data ch {} '.format(ch_pair)),
                                get_cmd=self._gen_ch_get_func(self.getTVdata,ch_pair)
                       
                                #vals=vals.Numbers(-128,127)
                              )
        for i in range(self.device_descriptor.numChannels):
            ch=i+1
            sgetdata_cmd= 'qutech:inputavg{}:data'.format(ch)
            self.add_parameter('ch{}_inavg_data'.format(ch),
                                label=('get data ch {} '.format(ch)),
                                get_cmd=self._gen_ch_get_func(self.getInputAverage,ch),
                       
                                vals=vals.Numbers(-128,127)
                              )
            srotres_cmd= 'qutech:wintrot{}:result'.format(ch)
            self.add_parameter('ch{}_wintrot_result'.format(ch),
                                units='#',
                                label=('rotated integration result of ch {} '.format(ch)),
                                get_cmd=srotres_cmd + '?'
                                #vals=vals.Numbers(1,2048)
                              )
            
    def getInputAverage(self,ch):
        
        self.write('qutech:inputavg{:d}:data? '.format(ch))
        binBlock = self.binBlockRead()
        #print(binBlock)
        print ('inputavgLen in bytes {:d}'.format(len(binBlock)))
        inputavg = np.frombuffer(binBlock,dtype=np.float32)
        return inputavg
    def getTVdata(self,ch_pair):
        
        self.write('qutech:tvmode{:d}:data? '.format(ch_pair))
        binBlock = self.binBlockRead()
        #print(binBlock)
        print ('TV data in bytes {:d}'.format(len(binBlock)))
        tvmodedata = np.frombuffer(binBlock,dtype=np.float32)
        return tvmodedata
    def sendWeightData(self, ch, weight):
        # generate the binblock
        if 1:   # high performance
            arr = np.asarray(weight, dtype=np.float32)
            binBlock = arr.tobytes()
        else:   # more generic
            binBlock = b''
            for i in range(len(weight)):
                binBlock = binBlock + struct.pack('<f', weight[i])

        # write binblock
        hdr = 'qutech:wint{}:data'.format(ch)
        self.binBlockWrite(binBlock, hdr)
    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

