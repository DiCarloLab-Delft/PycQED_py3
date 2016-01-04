# This is a virtual instrument abstracting a homodyne
# source which controls RF and LO sources
from instrument import Instrument
import qt
import types
import logging
import numpy as np
from time import time


class Homodyne_source(Instrument):
    '''
    This is a virtual instrument for a homodyne source
    '''
    def __init__(self, name, RF=qt.instruments['RF'],
                 LO=qt.instruments['LO'], ATS_CW=qt.instruments['ATS_CW'],
                 ATS=qt.instruments['ATS'],
                 single_sideband_demod=False):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Virtual'])

        self.LO = LO
        self.RF = RF
        self.ATS = ATS  # Currently most commands flow through the ATS CW
        self.ATS_CW = ATS_CW
        self.IF = 1e6  # Hz
        self.LO_power = 13  # dBm
        self.RF_power = -60  # dBm
        self._max_tint = 2000

        self.add_parameter('frequency', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=9e3, maxval=40e9,
                           units='Hz',
                           tags=['sweep'])
        self.add_parameter('IF', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=-200e6, maxval=200e6,
                           units='Hz',
                           tags=['sweep'])
        self.add_parameter('LO_source', type=bytes,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('LO_power', type=float,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           minval=-140, maxval=30, units='dBm',
                           tags=['sweep'])
        self.add_parameter('RF_source', type=bytes,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('RF_power', type=float,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           minval=-140, maxval=30, units='dBm',
                           tags=['sweep'])
        self.add_parameter('status', type=bytes,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('t_int', type=float,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           minval=100e-6, maxval=self._max_tint, units='ms')
        self.add_parameter('trace_length', type=float,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           units='ms')
        self.add_parameter('number_of_buffers', type=int,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('channel', type=int,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('Navg', type=int,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('single_sideband_demod', type=bool,
                           flags=Instrument.FLAG_GETSET)

        self._t_int = 10.
        self.get_t_int()
        self.set_channel(1)
        self.set_RF_source(self.RF.get_name())
        self.set_RF_power(self.RF_power)
        self.set_LO_source(self.LO.get_name())
        self.set_LO_power(self.LO_power)
        self.set_IF(self.IF)
        self.set_Navg(1)

        self.set_single_sideband_demod(single_sideband_demod)
        self.init()
        self.get_status()

    def set_sources(self, status):
        self.set_RF_status(status)
        self.set_LO_status(status)

    def do_set_frequency(self, val):
        self.frequency = val
        self.RF.set_frequency(val)
        self.LO.set_frequency(val+self.IF)

    def do_get_frequency(self):
        freq = self.RF.get_frequency()
        LO_freq = self.LO.get_frequency()
        if LO_freq != freq+self.IF:
            logging.warning('IF between RF and LO is not set correctly')
        return freq

    def do_set_IF(self, val):
        self.IF = val

        if val == 0:
            self.ATS_CW.set_coupling('DC')
        else:
            self.ATS_CW.set_coupling('AC')

        self.init()

    def do_get_IF(self):
        return self.IF

    def do_set_Navg(self, val):
        self.Navg = val

    def do_get_Navg(self):
        '''
        Navg is only used in the detector function
        '''
        return self.Navg

    def do_set_LO_source(self, val):
        self.LO = qt.instruments[val]

    def do_get_LO_source(self):
        return self.LO.get_name()

    def do_set_LO_power(self, val):
        self.LO.set_power(val)
        self.LO_power = val
        # internally stored to allow setting RF from stored setting

    def do_get_LO_power(self):
        return self.LO_power

    def do_set_RF_source(self, val):
        self.RF = qt.instruments[val]

    def do_get_RF_source(self):
        return self.RF.get_name()

    def do_set_RF_power(self, val):
        self.RF.set_power(val)
        self.RF_power = val
        # internally stored to allow setting RF from stored setting

    def do_get_RF_power(self):
        return self.RF_power

    def get_RF_status(self):
        return self.RF.get_status()

    def set_RF_status(self, status):
        return self.RF.set_status(status)

    def get_LO_status(self):
        return self.LO.get_status()

    def set_LO_status(self, status):
        return self.LO.set_status(status)

    def do_set_status(self, val):
        self.state = val
        if val == 'On':
            self.LO.on()
            self.RF.on()
        else:
            self.LO.off()
            self.RF.off()

    def do_get_status(self):
        if self.LO.get_status() == 'on' and self.RF.get_status() == 'on':
            return 'on'
        elif self.LO.get_status() == 'off' and self.RF.get_status() == 'off':
            return 'off'
        else:
            return 'LO: %s, RF: %s' % (self.LO.get_status(),
                                       self.RF.get_status())

    def on(self):
        self.set_status('On')

    def off(self):
        self.set_status('Off')
        return self._max_tint

    def do_set_t_int(self, t_int):
        '''
        sets the integration time per probe shot
        '''
        self.ATS_CW.set_t_int(t_int)
        self.get_trace_length()
        self.get_number_of_buffers()

    def do_get_t_int(self):
        return self.ATS_CW.get_t_int()

    def do_set_trace_length(self, trace_length):
        self.ATS_CW.set_trace_length(trace_length)
        self.get_t_int()

    def do_get_trace_length(self):
        return self.ATS_CW.get_trace_length()

    def do_set_number_of_buffers(self, n_buff):
        self.ATS_CW.set_number_of_buffers(n_buff)
        self.get_t_int()

    def do_get_number_of_buffers(self):
        return self.ATS_CW.get_number_of_buffers()

    def init(self, optimize=False, get_t_base=True, silent=False, max_buf=1024):
        '''
        Sets parameters in the ATS_CW and turns on the sources.
        if optimize == True it will optimze the acquisition time for a fixed
        t_int.
        '''
        self.ATS_CW.init(optimize=optimize, silent=silent, max_buf=max_buf)
        self.RF.set_power(self.do_get_RF_power())
        self.LO.set_power(self.do_get_LO_power())

        if self.RF.get_type() != 'Agilent_E8257D':
            self.RF.set_pulsemod_state('Off')
        if self.LO.get_type() != 'Agilent_E8257D':
            self.LO.set_pulsemod_state('Off')

        if get_t_base is True:
            tbase = self.ATS_CW.get_t_base()
            self.cosI = np.cos(2*np.pi*self.get_IF()*tbase)
            self.sinI = np.sin(2*np.pi*self.get_IF()*tbase)
        # To ensure the gui updates
        self.get_t_int()
        self.get_number_of_buffers()
        self.get_trace_length()

    def do_set_channel(self, ch):
        self.channel = ch

    def do_get_channel(self, ch):
        return self.channel

    def prepare(self, **kw):
        self.RF.on()
        self.LO.on()

    def probe(self, mtype='MAGN', **kw):
        '''
        Starts acquisition and returns the data
        mtype: variable that returns formatting of the output. Options are:
            'MAGN' : returns magnitude of data in Volts
            'PHAS' : returns phase of data in degrees
            'POL'  : returns data in polar coordinates
            'COMP' : returns data as a complex point in the I-Q plane in Volts
        '''
        if self._t_int != self.ATS_CW.get_t_int():
            self.init()
        self.ATS_CW.start()
        data = self.ATS.get_rescaled_avg_data()  # gets data in Volts avg acq
        s21 = self.demodulate_data(data)  # returns a complex point in the IQ-plane
        if mtype.upper() == 'MAGN':
            return np.abs(s21)
        elif mtype.upper() == 'PHAS':
            return np.angle(s21, deg=True)
        elif mtype.upper() == 'POL':
            return np.abs(s21), np.angle(s21, deg=True)
        elif mtype.upper() == 'COMP':
            return s21

    def get_demod_array(self):
        return self.cosI, self.sinI

    def demodulate_data(self, dat):
        '''
        Returns a complex point in the IQ plane by integrating and demodulating
        the data. Demodulation is done based on the 'IF' and
        'single_sideband_demod' parameters of the Homodyne instrument.
        '''
        if self.IF != 0:
            # self.cosI is based on the time_base and created in self.init()
            if self.single_sideband_demod is True:
                I = np.average(self.cosI * dat[0] + self.sinI * dat[1])
                Q = np.average(self.sinI * dat[0] - self.cosI * dat[1])
            else:  # Single channel demodulation, defaults to using channel 1
                I = 2*np.average(dat[0]*self.cosI)
                Q = 2*np.average(dat[0]*self.sinI)
        else:
            I = np.average(dat[0])  # dat[0] = data of channel 1
            Q = np.average(dat[1])
        return I+1.j*Q

    def do_set_single_sideband_demod(self, val):
        '''
        Disables or enables sending data to the plotmon.
        used for example when taking single traces and plotting is undesired as
        it is done from the soft sweep in MC.
        '''
        self.single_sideband_demod = val

    def do_get_single_sideband_demod(self):
        return self.single_sideband_demod
