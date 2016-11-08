import logging
import time


class Sweep_function(object):

    '''
    sweep_functions class for MeasurementControl(Instrument)
    '''

    def __init__(self, **kw):
        self.set_kw()

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass


class Soft_Sweep(Sweep_function):

    def __init__(self, **kw):
        self.set_kw()
        self.sweep_control = 'soft'

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass
##############################################################################


class None_Sweep(Soft_Sweep):

    def __init__(self, sweep_control='soft', sweep_points=None,
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass


class QX_Sweep(Soft_Sweep):

    """
    QX Input Test
    """

    def __init__(self, qxc, sweep_control='soft', sweep_points=None, **kw):
        super(QX_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'QX_Sweep'
        self.parameter_name = 'Error Rate'
        self.unit = 'P'
        self.sweep_points = sweep_points
        self.__qxc = qxc
        self.__qxc.create_qubits(2)
        self.__cnt = 0

    def set_parameter(self, val):
        circuit_name = ("circuit%i" % self.__cnt)
        self.__qxc.create_circuit(circuit_name,
                                  "prepz q0; h q0; x q0; z q0; y q0; y q0; z q0; x q0; h q0; measure q0;")
        self.__cnt = self.__cnt+1
        # pass


class Delayed_None_Sweep(Soft_Sweep):

    def __init__(self, sweep_control='soft', delay=0, **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.delay = delay
        self.time_last_set = 0
        if delay > 60:
            logging.warning(
                'setting a delay of {:.g}s are you sure?'.format(delay))

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        while (time.time() - self.time_last_set) < self.delay:
            pass  # wait
        self.time_last_set = time.time()


###################################

class AWG_amp(Soft_Sweep):

    def __init__(self, channel, AWG):
        super().__init__()
        self.name = 'AWG Channel Amplitude'
        self.channel = channel
        self.parameter_name = 'AWG_ch{}_amp'.format(channel)
        self.AWG = AWG
        self.unit = 'V'

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.AWG.stop()
        exec('self.AWG.ch{}_amp({})'.format(self.channel, val))
        self.AWG.start()


class AWG_multi_channel_amplitude(Soft_Sweep):

    '''
    Sweep function to sweep multiple AWG channels simultaneously
    '''

    def __init__(self, AWG, channels, delay=0, **kw):
        super().__init__()
        self.name = 'AWG channel amplitude chs %s' % channels
        self.parameter_name = 'AWG chs %s' % channels
        self.unit = 'V'
        self.AWG = AWG
        self.channels = channels
        self.delay = delay

    def set_parameter(self, val):
        for ch in self.channels:
            self.AWG.set('ch{}_amp'.format(ch), val)
        time.sleep(self.delay)

###############################################################################
####################          Hardware Sweeps      ############################
###############################################################################


class Hard_Sweep(Sweep_function):

    def __init__(self, **kw):
        super(Hard_Sweep, self).__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass

# NOTE: AWG_sweeps are located in AWG_sweep_functions


class ZNB_VNA_sweep(Hard_Sweep):

    def __init__(self, VNA,
                 start_freq=None, stop_freq=None,
                 center_freq=None, span=None,
                 npts=100, force_reset=False):
        '''
        Frequencies are in Hz.
        Defines the frequency sweep by one of the two methods:
        1) start a and stop frequency
        2) center frequency and span

        If force_reset = True the VNA is reset to default settings
        '''
        super(ZNB_VNA_sweep, self).__init__()
        self.VNA = VNA
        self.name = 'ZNB_VNA_sweep'
        self.parameter_name = 'frequency'
        self.unit = 'Hz'
        self.filename = 'VNA_sweep'

        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.center_freq = center_freq
        self.span = span
        self.npts = npts

        if force_reset == True:
            VNA.reset()

    def prepare(self):
        '''
        Prepare the VNA for measurements by defining basic settings.
        Set the frequency sweep and get the frequency points back from the insturment
        '''
        # Define basic settings for measurements
        # if self.force_reset == True:
        #     print('reset')
        #     self.VNA.reset() # reset to default settings

        self.VNA.continuous_mode_all('off')  # measure only once required
        # optimize the sweep time for the fastest measurement
        self.VNA.min_sweep_time('on')
        # start a measurement once the trigger signal arrive
        self.VNA.trigger_source('immediate')
        # trigger signal is generated with the command:
        # VNA.start_single_sweep_all()

        if self.start_freq != None and self.stop_freq != None:
            self.VNA.start_frequency(self.start_freq)
            self.VNA.stop_frequency(self.stop_freq)
        elif self.center_freq != None and self.span != None:
            self.VNA.center_frequency(self.center_freq)
            self.VNA.span_frequency(self.span)

        self.VNA.npts(self.npts)

        # get the list of frequency used in the span from the VNA
        self.sweep_points = self.VNA.get_stimulus()

