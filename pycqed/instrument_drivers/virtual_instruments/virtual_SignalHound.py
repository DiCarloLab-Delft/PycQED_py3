from time import sleep, time
import numpy as np
import ctypes as ct
import logging

from qcodes import Instrument, validators as vals
from qcodes.instrument.parameter import ManualParameter


class virtual_SignalHound_USB_SA124B(Instrument):
    def __init__(self, name, **kwargs):
        t0 = time()
        super().__init__(name, **kwargs)


        self.add_parameter('frequency',
                           label='Frequency ',
                           unit='Hz',
                           initial_value=5e9,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('span',
                           label='Span ',
                           unit='Hz',
                           initial_value=.25e6,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('power',
                           label='Power ',
                           unit='dBm',
                           initial_value=0,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(max_value=20))
        self.add_parameter('ref_lvl',
                           label='Reference power ',
                           unit='dBm',
                           initial_value=0,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(max_value=20))
        self.add_parameter('external_reference',
                           parameter_class=ManualParameter,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('device_type',
                           parameter_class=ManualParameter)

        self.add_parameter('device_mode',
                           initial_value='sweeping',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())
        self.add_parameter('acquisition_mode',
                           parameter_class=ManualParameter,
                           initial_value='average',
                           vals=vals.Enum('average', 'min-max'))
        self.add_parameter('scale',
                           parameter_class=ManualParameter,
                           initial_value='log-scale',
                           vals=vals.Enum('log-scale', 'lin-scale',
                                          'log-full-scale', 'lin-full-scale'))
        self.add_parameter('running',
                           parameter_class=ManualParameter,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('decimation',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Ints(1, 8))
        self.add_parameter('bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           initial_value=0,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        # rbw Resolution bandwidth in Hz. RBW can be arbitrary.
        self.add_parameter('rbw',
                           label='Resolution Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        # vbw Video bandwidth in Hz. VBW must be less than or equal to RBW.
        #  VBW can be arbitrary. For best performance use RBW as the VBW.
        self.add_parameter('vbw',
                           label='Video Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        t1 = time()
        print('Initialized SignalHound in %.2fs' % (t1-t0))

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'USB'

    def openDevice(self):
        pass

    def closeDevice(self):
        pass

    def abort(self):
        pass

    def preset(self):
        pass

    def _do_get_device_type(self):
        pass

    ########################################################################

    def initialisation(self, flag=0):
        pass

    def QuerySweep(self):
        """
        Queries the sweep for information on the parameters it uses
        """
        pass

    def configure(self, rejection=True):
        pass

    def sweep(self):
        pass

    def get_power_at_freq(self, Navg=1):
        '''
        Returns the maximum power in a window of 250kHz
        around the specified  frequency.
        The integration window is specified by the VideoBandWidth (set by vbw)
        '''
        return np.random.rand()

    def get_spectrum(self, Navg=1):
        """
        Averages over SH.sweep Navg times

        """
        pass

    def prepare_for_measurement(self):
        self.set('device_mode', 'sweeping')
        self.configure()
        self.initialisation()
        return

    def safe_reload(self):
        pass

    def check_for_error(self, err):
        pass
