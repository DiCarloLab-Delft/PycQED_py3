# Stroboscope for locking the measurement to the pulse tube and directing the
# triggers to the measurement setup
#
# Written April 2018:
# Thijs Stavenga <thijsstavenga@msn.com>
#


from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum
from qcodes import VisaInstrument, validators as vals
import time


class Stroboscope(VisaInstrument):
    '''
    This is the driver for the Stroboscope

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Keithley_2200',
        address='<address>',
        reset=<bool>,
        change_display=<bool>,
        change_autozero=<bool>)
    '''

    # def __init__(self, name, address, reset=False,
    #         change_display=True, change_autozero=True):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
        '''
        Initializes the Keithley_2200, and communicates with the wrapper.

        Input:
            name (string)           : name of the instrument
            address (string)        : COM port
        '''

        # Add some global constants
        self.MAX_DUTY = 698 ## milliseconds
        self.MAX_PHASE = 698 ## milliseconds
        self._address = address
        self.visa_handle.baud_rate = 115200
        self.visa_handle.write_termination = '\r'
        self.visa_handle.read_termination = '\r'

        self.add_parameter('phase',
            get_cmd='PHASE?',
            get_parser=int,
            set_cmd='PHASE {:d}',
            vals=vals.Numbers(min_value=0,
                              max_value=self.MAX_PHASE),
            unit='ms')
        self.add_parameter('duty_cycle',
            get_cmd='DUTY?',
            get_parser=int,
            set_cmd='DUTY {:d}',
            vals=vals.Numbers(min_value=0,
                              max_value=self.MAX_DUTY),
            unit='ms')
        self.add_parameter('locked',
            get_cmd='LOCK?',
            get_parser=lambda s: bool(int(s)),
            label='Trigger locked to pulse tube')

        # time.sleep(3)
        # self.get_all()

# --------------------------------------
#           functions
# --------------------------------------

    def get_all(self):
        '''
        Reads all relevant parameters from instrument

        Input:
            None

        Output:
            None
        '''

        self.phase()
        self.duty_cycle()
        self.locked()


    def get_info(self):
        '''
        Get the device descriptor.
        should be: Keithley Instruments, 2200-20-5, 1359014, 1.26-1.23
        Input:
            None
        Output:
            String: Device descriptor
        '''
        ans = self._visainstrument.ask('*IDN?')
        return ans

