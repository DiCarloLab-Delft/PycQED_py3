from instrument import Instrument
import types

class dummy_pulse_src(Instrument):
    '''this is a dummy pulse source'''

    def __init__(self, name, reset=False):
        Instrument.__init__(self, name, tags=['Dummy'])

        self.add_parameter('start',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=0, maxval=1,
                units='sec')
        self.add_parameter('length',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=0, maxval=1,
                units='sec')
        self.add_parameter('amplitude',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=-3.8, maxval=3.8,
                units='Volts')
        self.add_parameter('status',
                type=bytes,
                option_list=('on', 'off'),
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET)

        self.add_function('reset')
        self.add_function('get_all')

        # these are dummy values for an instrument that
        # is already running
        self._dummy_start = 1e-9
        self._dummy_length = 1e-9
        self._dummy_amplitude = 1
        self._dummy_status = 'off'

        if reset:
            self.reset()
        else:
            self.get_all()

#### initialization related

    def reset(self):
        print(__name__, ': resetting instrument')
        self.set_start(2e-9)
        self.set_length(2e-9)
        self.set_amplitude(2)
        self.set_status('off')

    def get_all(self):
        print(__name__, ': reading all settings from instrument')
        self.get_start()
        self.get_length()
        self.get_amplitude()
        self.get_status()

#### communication with machine

    def do_get_start(self):
        return self._dummy_start

    def do_set_start(self, start):
        self._dummy_start = start

    def do_get_length(self):
        return self._dummy_length

    def do_set_length(self,length):
        self._dummy_length = length

    def do_get_amplitude(self):
        return self._dummy_amplitude

    def do_set_amplitude(self,amplitude):
        self._dummy_amplitude = amplitude

    def do_get_status(self):
        return self._dummy_status

    def do_set_status(self,status):
        self._dummy_status = status

