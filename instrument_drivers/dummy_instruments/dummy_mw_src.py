from instrument import Instrument
import types

class dummy_mw_src(Instrument):
    '''this is a dummy microwave source'''

    def __init__(self, name, reset=False):
        Instrument.__init__(self,name)

        self.add_parameter('frequency',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=0, maxval=20e9,
                units='Hz')
        self.add_parameter('power',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=-120, maxval=25,
                units='dBm')
        self.add_parameter('phase',
                type=float,
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET,
                minval=-180, maxval=180)
        self.add_parameter('status',
                type=bytes,
                option_list=('on', 'off'),
                flags=Instrument.FLAG_GETSET | \
                Instrument.FLAG_GET_AFTER_SET)

        # these are dummy values for an instrument that
        # is already running
        self._dummy_frequency = -99
        self._dummy_power = -99
        self._dummy_phase = -99
        self._dummy_status = 'on'

        if reset:
            self.reset()
        else:
            self.get_all()

#### initialization related

    def reset(self):
        print(__name__ + ' : resetting instrument')
        self.set_frequency(20e9)
        self.set_power(-120)
        self.set_phase(0)
        self.set_status('off')

    def get_all(self):
        print(__name__ + ' : reading all settings from instrument')
        self.get_frequency()
        self.get_power()
        self.get_phase()
        self.get_status()

#### communication with machine

    def do_get_frequency(self):
        return self._dummy_frequency

    def do_set_frequency(self, frequency):
        self._dummy_frequency = frequency

    def do_get_power(self):
        return self._dummy_power

    def do_set_power(self, power):
        self._dummy_power = power

    def do_get_phase(self):
        return self._dummy_phase

    def do_set_phase(self, phase):
        self._dummy_phase = phase

    def do_get_status(self):
        return self._dummy_status

    def do_set_status(self,status):
        self._dummy_status = status

### shorcuts
    def off(self):
        self.set_status('off')

    def on(self):
        self.set_status('on')
