from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Enum


class Agilent_33250A(VisaInstrument):
    """
    This is the driver code for the Agilent 33250A AWG.

    Status: Not tested

    Only most commonly used commands of the device integrated at this stage.
    """
    def __init__(self, name ,adress,**kwargs):

        super().__init__(name,adress,**kwargs)

        self.add_parameter('frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-6,
                                        max_value=80e6),
                           docstring=('Command for setting the pulse frequency. Min value: 1e-6Hz'
                                      'Max value: 80e6Hz'))

        self.add_parameter('pulse_shape',
                           label='Pulse_shape',
                           get_cmd='FUNC?',
                           set_cmd='FUNC {}',
                           vals=Enum('SIN','SQU'),
                           docstring=('Command for setting the desired pulse shape. '
                                      'Currently supportet: sinus, square.'))

        self.add_parameter('pulse_width',
                           label='Pulse_width',
                           get_cmd='PULS:WIDT?',
                           set_cmd='PULS:WIDT {}',
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(min_value=8e-9,max_value=2e3),
                           docstring=('Command for setting the desired pulse width. Min value: 8e-9s'
                                      'Max value: 2000s.'))

        self.add_parameter('amplitude',
                           label='Amplitude',
                           unit='V',
                           get_cmd='VOLT?',
                           set_cmd='VOLT',
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-3,max_value=10),
                           docstring=('Command for setting the desired pulse amplitude. Min value: 1e-3V'
                                      'Max value: 10V.'))
        self.add_parameter(name='output',
                           get_cmd='OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring=('Command for switching on/off the device output.'))

        self.add_parameter(name='output_sync',
                           get_cmd='OUTP:SYNC?',
                           set_cmd='OUTP:SYNC {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring='Command for switching on/off the device output synchronization.')
        self.connect_message()
