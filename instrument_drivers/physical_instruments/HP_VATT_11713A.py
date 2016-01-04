from instrument import Instrument
import visa
import logging
import socket
import select
import types
class HP_VATT_11713A(Instrument):
    def __init__(self, name, address):
        logging.info(__name__ + ' : Initializing HP variable attenuator')

        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        self._visainstrument = visa.instrument(self._address)

        #TODO: Min, max?
        self.add_parameter('attenuation', type=int,
                flags = Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                minval = 0, maxval = 90,
                units = 'dB')

    def do_get_attenuation(self):
        return self._attenuation

    def do_set_attenuation(self, dB):
        '''
        Sets the variable attenuator to value dB
        '''
        #Round off and convert to binary form
        dBnum = dB/10
        
        dBaddr = {0:'B1234',
                1:'A1B234',
                2:'A2B134',
                3:'A3B124',
                4:'A13B24',
                5:'A23B14',
                6:'A123B4',
                7:'A134B2',
                8:'A234B1',
                9:'A1234'}
        #dBbin = '%04d' % int(bin(dBnum)[2:])
        self._attenuation = 10*dBnum
        #Seperate into buttons turned on (A), and turned off (B)

        logging.debug(__name__ + ' : Setting attenuation to %s dB'%dB)
        self._visainstrument.write(dBaddr[dBnum])