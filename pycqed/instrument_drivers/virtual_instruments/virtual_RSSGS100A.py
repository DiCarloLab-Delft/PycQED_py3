from qcodes.instrument.parameter import ManualParameter
from qcodes import VisaInstrument, validators as vals
from qcodes import Instrument


class virtualRohdeSchwarz_SGS100A(Instrument):
    """
    This is the qcodes driver for the Rohde & Schwarz SGS100A signal generator

    Status: beta-version.

    .. todo::

        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit

    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good idea to group all similar RS drivers together in one
    module.

    Tested working with

    - RS_SGS100A
    - RS_SMB100A

    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.parameters = {}
        self.functions = {}

        self.add_parameter(
            name='frequency',
            label='Frequency',
            unit='Hz',
            parameter_class=ManualParameter,
            get_parser=float,
            vals=vals.Numbers(1e6, 20e9))
        self.add_parameter(
            name='phase',
            label='Phase',
            unit='deg',
            parameter_class=ManualParameter,
            get_parser=float,
            vals=vals.Numbers(0, 360))
        self.add_parameter(
            name='power',
            label='Power',
            unit='dBm',
            parameter_class=ManualParameter,
            get_parser=float,
            vals=vals.Numbers(-120, 25))
        self.add_parameter(
            'status',
            parameter_class=ManualParameter,
            get_parser=self.parse_on_off,
            vals=vals.Strings())
        self.add_parameter(
            'pulsemod_state',
            parameter_class=ManualParameter,
            get_parser=self.parse_on_off,
            vals=vals.Strings())
        self.add_parameter(
            'pulsemod_source',
            parameter_class=ManualParameter,
            vals=vals.Strings())
        self.add_parameter(
            'ref_osc_source',
            label='Reference oscillator source',
            parameter_class=ManualParameter,
            vals=vals.Enum('INT', 'EXT'))
        # Frequency mw_source outputs when used as a reference
        self.add_parameter(
            'ref_osc_output_freq',
            label='Reference oscillator output frequency',
            parameter_class=ManualParameter,
            vals=vals.Enum('10MHz', '100MHz', '1000MHz'))
        # Frequency of the external reference mw_source uses
        self.add_parameter(
            'ref_osc_external_freq',
            label='Reference oscillator external frequency',
            parameter_class=ManualParameter,
            vals=vals.Enum('10MHz', '100MHz', '1000MHz'))

        self.add_function('reset', call_cmd='*RST')
        self.add_function('run_self_tests', call_cmd='*TST?')

        self.connect_message()

    def connect_message(self):
        return 'virtualRohdeSchwarz_SGS100A'

    def parse_on_off(self, stat):
        if stat.startswith('0'):
            stat = 'Off'
        elif stat.startswith('1'):
            stat = 'On'
        return stat

    def set_status(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.write(':OUTP:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s, ' % stat +
                             'expected "ON" or "OFF"')

    def set_pulsemod_state(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.write(':PULM:SOUR EXT')
            self.write(':SOUR:PULM:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s,' % stat +
                             'expected "ON" or "OFF"')

    def set_pulsemod_source(self, source):
        if source.upper() in ('INT', 'EXT'):
            self.write(':SOUR:PULM:SOUR %s' % source)
        else:
            raise ValueError('Unable to set source to %s,' % source +
                             'expected "INT" or "EXT"')

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')