import qcodes as qcodes
import numpy as np
from time import sleep
from qcodes.selfument import visa
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

class RTO1024_scope(VisaInstrument):
    '''
    Instrument driver for the Rohde-Schwarz RTO1024 oscilloscope.
    Currently only able to measure signal on Channel 1.
    '''
    def __init__(self, name, address=None, timeout=5, terminator='',
                 **kwargs):
        super().__init__(name, address, timeout, terminator, **kwargs)

        self.add_parameter('num_averages',
                           docstring='Number of averages for measuring '
                                     'trace.',
                           vals=vals.Ints(),  # TODO: check hardware limits
                           initial_value=10000,
                           parameter_class=ManualParameter)
        self.add_parameter('trigger_interval',
                           docstring='Time between triggers',
                           vals=vals.Numbers(),
                           unit='s',
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('trigger_ch',
                           docstring='Trigger channel',
                           vals=vals.ints(1, 4),
                           initial_value=2,
                           parameter_class=ManualParameter)

    def prepare_measurement(self, t_start, t_stop, acq_rate=10000000000):
        '''
        Prepares the oscilloscope for measurement.

        Args:
            t_start    (float): starting time of the saved waveform relative
                                to trigger (in seconds)
            t_stop     (float): stop time of the saved waveform relative
                                to trigger (in seconds)
            acq_rate   (int): acquisition rate
        '''
        opc = self.visa_handle.ask('STOP;*OPC?')
        self.visa_handle.write('TRIGger1:SOURce CHAN{}'
                               .format(self.trigger_ch))
        self.visa_handle.write('ACQuire:SRATe 10000000000')
        self.visa_handle.write('EXPort:WAVeform:FASTexport ON')
        self.visa_handle.write('EXPort:WAVeform:INCXvalues ON')
        self.visa_handle.write('CHANnel1:WAVeform1:STATe 1')
        self.visa_handle.write('EXPort:WAVeform:STARt %s'%str(start))
        self.visa_handle.write('EXPort:WAVeform:STOP %s'%str(stop))
        sampling_rate = float(self.visa_handle.ask('ACQuire:POINts:ARATe?'))*1e-9

    def measure_trace():
        '''
        Measures trace num_averages times.
        Returns x_values, y_values as numpy arrays.
        '''
        # defines the acquisition no. for run single mode
        self.visa_handle.write('ACQuire:COUNt %s'%str(self.num_averages))
        self.visa_handle.write('RUNSingle')
        # Wait until measurement finishes before extracting data.
        sleep(num_averages * self.meas_interval + 1)
        self.visa_handle.write('EXPort:WAVeform:SOURce C1W1')
        self.visa_handle.write('CHANnel1:ARIThmetics AVERage')
        ret_str = self.visa_handle.ask('CHANNEL1:WAVEFORM1:DATA?')
        array = ret_str.split(',')
        array = np.double(array)
        x_values = array[::2]
        y_values = array[1::2]
        print(array)
        return x_values, y_values
