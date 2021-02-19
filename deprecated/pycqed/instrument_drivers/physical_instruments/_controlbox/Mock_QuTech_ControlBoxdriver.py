import sys
from . import defHeaders_CBox_v3 as defHeaders
from .. import QuTech_ControlBox_v3 as qcb3


'''
This mock driver is used to print commands into hex format without connection
to a real CBox.
'''


class Mock_QuTech_ControlBox(qcb3.QuTech_ControlBox_v3):
    def __init__(self, os_par=sys.stdout, *args, **kwargs):
        self.SetupStream(os_par)
        # super(Mock_QuTech_ControlBox, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

        print('Mock driver has been initialized.')

    def SetupStream(self, os_par):
        print('initialized outstream')
        self.outstream = os_par

    def set_address(self, address=None):
        return True

    def set_terminator(self, terminator=None):
        return True

    def _set_visa_timeout(self, value):
        pass

    def _get_visa_timeout(self):
        pass

    def get_master_controller_params(self):
        self._core_state = defHeaders.core_states[0]
        self._trigger_source = defHeaders.trigger_sources[0]
        message = self.create_message(defHeaders.ReadVersion)
        (stat, mesg) = self.serial_write(message)
        return (stat, mesg)

    def serial_read(self, timeout=5, read_all=False, read_N=0):
        print('serial read is invoked and a fake message is sent back'
              'in the mock driver.')
        return bytes.fromhex('7F')

    def serial_write(self, command, verify_execution=True):
        if type(command) != bytes:
            raise TypeError('command must be type bytes')

        self.outstream.write('Command: %s\n' %
                           self.format_hex_string(command.hex()))

        return (True, bytes.fromhex('7F'))

    def format_hex_string(self, hexstr):
        assert(len(hexstr) % 2 == 0)
        hexstr = hexstr.upper()
        index = 2
        while (index < len(hexstr)):
            hexstr = hexstr[:index] + " " + hexstr[index:]
            index = index + 3

        return hexstr
