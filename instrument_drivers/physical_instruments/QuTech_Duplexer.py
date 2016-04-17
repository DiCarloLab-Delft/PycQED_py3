import logging
import numpy as np
import time
import h5py
from os import path
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class QuTech_Duplexer(VisaInstrument):
    '''
    This is the python driver for the QuTech duplexer made by TNO.
    '''

    def __init__(self, name, address='TCPIP0::192.168.0.100', reset=False,
                 nr_input_channels=4, nr_output_channels=2):
        '''
        Initializes the QuTech_Duplexer, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : TCPIP address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        t0 = time.time()
        logging.info(__name__ + ' : Initializing instrument')
        address += '::5025::SOCKET'
        print(address)
        super().__init__(name, address)
        self.SCPI_command_pause = 0.1
        self.visa_handle.read_termination = '\n'

        dirname, filename = path.split(path.abspath(__file__))
        cal_file_path = path.join(dirname,
                                  '_duplexer/Duplexer_normalized_gain.hdf5')
        cal_file = h5py.File(cal_file_path, 'r')
        self._calibration_array = list(cal_file.values())[0][1]
        self.add_parameter('mode', label='Operating mode',
                           parameter_class=ManualParameter,
                           initial_value='cal',
                           vals=vals.Enum('raw', 'cal'))

        for inp in range(nr_input_channels):
            for outp in range(nr_output_channels):
                self.add_parameter('in{}_out{}_switch'.format(inp+1, outp+1),
                                   set_cmd='ch:in{}:out{}:sw'.format(inp+1,
                                                                     outp+1)
                                   + ' {} \n',
                                   get_cmd='ch:in{}:out{}:sw?'.format(inp+1,
                                                                      outp+1),
                                   vals=vals.Enum('ON', 'OFF', 'EXT'),
                                   get_parser=self._get_parser)

                self.add_parameter('in{}_out{}_phase'.format(inp+1, outp+1),
                                   set_cmd='ch:in{}:out{}:ph:'.format(
                                   inp+1, outp+1) + '{} \n',
                                   get_cmd='ch:in{}:out{}:ph:raw?'.format(
                                        inp+1, outp+1),
                                   set_parser=self._mode_set_parser,
                                   get_parser =self._get_parser,
                                   vals=vals.Numbers(0, 65536))
                self.add_parameter('in{}_out{}_attenuation'.format(inp+1,
                                                                   outp+1),
                                   set_cmd='ch:in{}:out{}:att:'.format(
                                        inp+1, outp+1) + '{} \n',
                                   get_cmd='ch:in{}:out{}:att:raw?'.format(inp+1,
                                                                      outp+1),
                                   vals=vals.Numbers(0, 65536),
                                   set_parser=self._attenuation_set_parser,
                                   get_parser=self._attenuation_get_parser)

        # Get commands are not Implemented!!!
        # self.add_parameter('IDN', get_cmd='IDN?')
        t1 = time.time()
        print('Initialized Duplexer in {}'.format(t1-t0))

    def set_all_switches_to(self, mode):
        raise NotImplementedError()
        for i in range(4):
            for j in range(2):
                self.set_switch(i+1, j+1, mode)

    def set_all_phases_to(self, val):
        raise NotImplementedError()
        for i in range(4):
            for j in range(2):
                self.set_phase(i+1, j+1, val)

    def set_all_attenuations_to(self, val):
        raise NotImplementedError()
        for i in range(4):
            for j in range(2):
                self.set_attenuation(i+1, j+1, val)

    def _mode_set_parser(self, val):
        '''
        Parses input such that the operating mode
        '''
        if self.mode.get() == 'cal':
            # raise NotImplementedError('Not implemented in current version')
            # This awaits the implementation of the in duplxer calibration
            # memories.
            pass
        return '{} {}'.format(self.mode.get(), val)

    def _attenuation_set_parser(self, val):
        '''
        If mode is cal uses the calibration array to convert values in the
        range 0-1 to dac-values.
        '''
        if self.mode.get() == 'raw':
            return 'raw {}'.format(val)
        else:
            if val < 0 or val > 1:
                raise ValueError('value "{}" out of range (0-1)'.format(val))
            # 1- is because we use attenuation and the array contains gain
            dac_value = np.searchsorted(self._calibration_array, 1-val,
                                        side="left")
            return 'raw {}'.format(dac_value)

    def _attenuation_get_parser(self, val):
        if self.mode.get() == 'raw':
            return int(val.rsplit('=')[1])
        else:
            # 1- is because we use attenuation and the array contains gain
            dac_val = int(val.rsplit('=')[1])
            return 1-self._calibration_array[dac_val]

    def _get_parser(self, val):
        return val.rsplit('=')[1]


    def get_scaling_increment(self, scaling_factor):
        '''
        This needs a docstring!
        '''
        cal_slope = 2048  # dB/attenuation_dac_val \
        scaling_in_dB = 10 * np.log10(scaling_factor)
        scaling_increment = cal_slope * scaling_in_dB
        return scaling_increment

    def calculate_attenuation(self, current_dac_value, scaling_factor):
        '''
        This needs a docstring!
        '''
        try:
            current_val = self._calibration_array[current_dac_value]
        except:
            print('Setting wrong attenuation for Duplexer')
            if current_val < 0:
                current_val = self._calibration_array[-1]
            else:
                current_val = self.calibration_array[0]
        final_val = self._calibration_array - current_val * scaling_factor
        new_dac_val = np.argmin(np.abs(final_val))
        return new_dac_val