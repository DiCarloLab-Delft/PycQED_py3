from instrument import Instrument
from qcodes.utils import validators as vals
import numpy as np
import telnetlib


class Weinschel_8320(Instrument):
    '''
    QCodes driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    '''

    def __init__(self, name, address, **kwargs):
        Instrument.__init__(self, name)
        self.address = address

        self.add_parameter('attenuation', units='dB',
                           set_cmd=,
                           get_cmd=,
                           vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
                           get_parser=float)

        self.connect_message()

    def telnet_set(self, val):
        cmd = 'ATTN ALL {0:0=2d}\r\n'.format(val)
        tn = telnetlib.Telnet(HOST)
        tn.write(cmd)
        tn.read_until(cmd)
        tn.close()

    def telnet_get(self):
        cmd = 'ATTN? 1\r\n'
        tn = telnetlib.Telnet(HOST)
        tn.write(cmd)
        tn.read_until(cmd)
        ret = int(tn.read_some())
        tn.close()
        return ret
