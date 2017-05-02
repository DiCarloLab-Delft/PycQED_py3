from functools import partial
import time
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class ConversionBoxControl(Instrument):
    """
    This is a meta-instrument for controlling the different switches in the
    conversion box designed in the QuDev lab. This class implements the control
    over
        * 2 amplifiers, whose input port can be chosen between the fridge and a
          reference signal
        * 5 up-conversion mixers, that can be operated in modulation or mixer-
          bypass mode
        * a switchboard that can route the input signal to one of the 6 output
          ports
    The switchboard is controlled using an Advantech PCIE-1751 digital I/O card.
    This meta-instrument takes full control of the underlying DIO card which
    should not be modified by other classes.
    """

    shared_kwargs = ['dio']

    def __init__(self, name, dio, switch_time=50e-3, **kw):
        """
        :param name: name of the instrument
        :param dio: reference to an Advantech PCIE-1751 instrument
        :param switch_time: duration of the pulse to set the switch
                            configuration
        """
        super().__init__(name, **kw)
        self.dio = dio

        # configure all DIO ports for output
        for i in range(self.dio.port_count()):
            self.dio.set('port{}_dir'.format(i), 0xff)

        self._switch_state = {
            'UC1': 'modulated',
            'UC2': 'modulated',
            'UC3': 'modulated',
            'UC4': 'modulated',
            'UC5': 'modulated',
            'WA1': 'measure',
            'WA2': 'measure',
            'switch': 'block',
        }

        UCkey = ['']*6
        for i in range(1, 6):
            descr = 'switch configuration of up-conversion board {}'
            UCkey[i] = 'UC{}'.format(i)
            self.add_parameter(
                '{}_mode'.format(UCkey[i]),
                label=descr.format(i),
                vals=vals.Enum('modulated', 'bypass'),
                get_cmd=partial(self._switch_state.get, UCkey[i]),
                set_cmd=lambda x: self.set_switch({UCkey[i]: x})
            )

        WAkey = ['']*3
        for i in range(1, 3):
            descr = 'switch configuration of warm amplifier {}'
            WAkey[i] = 'WA{}'.format(i)
            self.add_parameter(
                '{}_mode'.format(WAkey[i]),
                label=descr.format(i),
                vals=vals.Enum('reference', 'measure'),
                get_cmd=partial(self._switch_state.get, WAkey[i]),
                set_cmd=lambda x: self.set_switch({WAkey[i]: x})
            )

        self.add_parameter(
            'switch_mode',
            label='switchboard configuration',
            vals=vals.Enum('block', 1, 2, 3, 4, 5, 6),
            get_cmd=partial(self._switch_state.get, 'switch'),
            set_cmd=lambda x: self.set_switch({'switch': x})
        )

        self.add_parameter('switch_time', unit='s', vals=vals.Numbers(0, 1),
                           label='Duration of the switching pulse',
                           parameter_class=ManualParameter,
                           initial_value=switch_time)

    def set_switch(self, values):
        """
        :param values: a dictionary of key: value pairs, where key is one of
                       the following: 'UC#', 'WA#' or 'switch' (# denotes the
                       board number) and value is the mode to set the switch to.
        """
        #logging.debug(values)
        for key in values:
            self.parameters['{}_mode'.format(key)].validate(values[key])
        self._switch_state.update(values)

        data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        data[0] += self._WA_bitpattern(self._switch_state['WA2']) << 0
        data[0] += self._WA_bitpattern(self._switch_state['WA1']) << 2
        data[1] += self._UC_bitpattern(self._switch_state['UC1']) << 0
        data[1] += self._UC_bitpattern(self._switch_state['UC2']) << 2
        data[1] += self._UC_bitpattern(self._switch_state['UC3']) << 4
        data[1] += self._UC_bitpattern(self._switch_state['UC4']) << 6
        data[2] += self._UC_bitpattern(self._switch_state['UC5']) << 0
        data[5] = self._switch_bitpattern(self._switch_state['switch'])

        self.dio.write_port(0, data)
        time.sleep(self.switch_time())
        self.dio.write_port(0, [0, 0, 0, 0, 0, 0])

        for key in values:
            self.parameters['{}_mode'.format(key)]._save_val(values[key])

    @classmethod
    def _WA_bitpattern(cls, mode):
        if mode == 'reference':
            return 0b01
        elif mode == 'measure':
            return 0b10
        else:
            raise ValueError('Trying to set warm amplifier board switch to '
                             'invalid mode: {}'.format(mode))

    @classmethod
    def _UC_bitpattern(cls, mode):
        if mode == 'bypass':
            return 0b01
        elif mode == 'modulated':
            return 0b10
        else:
            raise ValueError('Trying to set up-conversion board switch to '
                             'invalid mode: {}'.format(mode))

    @classmethod
    def _switch_bitpattern(cls, mode):
        if mode == 'block':
            return 0b10000000
        elif mode in range(1,7):
            return 1 << (7-mode)
        else:
            raise ValueError('Trying to set the switchboard to invalid mode: '
                             '{}'.format(mode))
