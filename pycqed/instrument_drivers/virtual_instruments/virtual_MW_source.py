from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals


class VirtualMWsource(Instrument):

    '''
    This is a virtual MW_source based on the SGS100A driver.
    '''

    def __init__(self, name, address=None, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1e6, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 360))
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(-120, 25))
        self.add_parameter('status',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_state',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_source',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())

        self.add_parameter('ref_osc_source',
                           label='Reference oscillator source',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('INT', 'EXT'))
        # Frequency it outputs when used as a reference
        self.add_parameter('ref_osc_output_freq',
                           label='Reference oscillator output frequency',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('10MHz', '100MHz', '1000MHz'))
        # Frequency of the external reference it uses
        self.add_parameter('ref_osc_external_freq',
                           label='Reference oscillator external frequency',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('10MHz', '100MHz', '1000MHz'))

        self.connect_message()

    def reset(self):
        pass

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')
