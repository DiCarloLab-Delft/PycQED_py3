from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import telnetlib


class Weinschel_8320(Instrument):
    '''
    QCodes driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    '''

    def __init__(self, name, address, timeout=.1, **kwargs):
        Instrument.__init__(self, name)
        self.address = address
        self.timeout = timeout

        self.add_parameter('attenuation', units='dB',
                           vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
                           set_cmd=self._do_set_attenuation,
                           get_cmd=self._do_get_attenuation)

        self.connect_message()

    def _do_set_attenuation(self, val):
        cmd_str = '\nATTN ALL {0:0=2d}\r\n'.format(val)
        cmd_encoded = cmd_str.encode('ascii')
        tn = telnetlib.Telnet(self.address)
        tn.write(cmd_encoded)
        tn.read_until(cmd_encoded, timeout=self.timeout)
        tn.close()

    def _do_get_attenuation(self):
        cmd_str = '\nATTN? 1\r\n'
        cmd_encoded = cmd_str.encode('ascii')
        tn = telnetlib.Telnet(self.address)
        tn.write(cmd_encoded)
        ret = tn.read_until(cmd_encoded, timeout=self.timeout)
        ret = ret.decode('ascii')
        tn.close()
        return int(ret.split('\r\n')[-3],10)
