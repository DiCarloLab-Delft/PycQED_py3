import instrument_drivers.physical_instruments.QuTech_ControlBox as CB
import imp
imp.reload(CB)


class dummy_CBox(CB.QuTech_ControlBox):
    '''
    dummy version of qutech controlbox
    Overwrites read and write function
    '''

    def serial_write(self, command, verify_execution=True):
        if type(command) == list:
            command = ''.join(command)
        checksum = self.calculate_checksum(command)
        return True, checksum

    def _do_get_firmware_version(self):
        version = 'DUMMY'
        return version

    def open_serial_port(self, nameCOM):
        '''
        Establishes serial connection with instrument.
        Requires comport as argument e.g. 'Com3'
        rest of the settings is given the arguments required for the
        QuTech_ControlBox.
        '''
        pass

    def get_integration_log_results(self):
        return None

    def get_integrated_avg_results(self):
        return None

    def get_input_avg_results(self):
        return None

    def close_serial_port(self):
        '''
        Closes the serial port, should be used before reloading the instrument
        '''
        return True

    def get_serial(self):
        pass

    def serial_read(self, timeout=5, read_all=False):
        message = ''
        return message
