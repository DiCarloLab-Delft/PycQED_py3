"""
    File:       SCPI.py
    Author:     Wouter Vlothuizen, TNO/QuTech
    Purpose:    base class for SCPI ('Standard Commands for Programmable
                Instruments') commands
    Usage:      don't use directly, use a derived class (e.g. QWG)
    Notes:      deprecation warning: to be superseded by SCPIBase
    Bugs:
    Changelog:

20190212 WJV
- addressed many warnings identified by PyCharm
- changed to Python naming conventions
- added type annotations

"""

import socket

from qcodes import IPInstrument

"""
FIXME: we would like to be able to choose the base class separately, so the
user can choose (e.g. use VISA for IEE488 bus units, and IpInstrument for
networked units). This would also make the inits cleaner
"""


class SCPI(IPInstrument):

    def __init__(self, name: str, address: str, port: int, **kwargs) -> None:
        super().__init__(name, address, port,
                         write_confirmation=False,  # required for QWG
                         **kwargs)

        # send things immediately
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # beef up buffer, to prevent socket.send() not sending all our data
        # in one go
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512*1024)

        # example of how the commands could look. FIXME
        self.add_function('reset', call_cmd='*RST')

    def _recv(self) -> str:
        """
        Overwrites base IP recv command to ensuring read till EOM
        FIXME: should be in parent class
        """
        return self._socket.makefile().readline().rstrip()

    ###
    # Helpers
    ###

    def _read_binary(self, size: int) -> bytes:
        data = self._socket.recv(size)
        act_len = len(data)
        exp_len = size
        i = 1
        while act_len != exp_len:
            data += self._socket.recv(exp_len-act_len)
            act_len = len(data)
            i = i+1
        return data

    def _write_binary(self, bin_msg: bytes) -> None:
        self._socket.send(bin_msg)       # FIXME: should be in parent class

    def ask_float(self, cmd_str: str):
        return float(self.ask(cmd_str))

    def ask_int(self, cmd_str: str):
        return int(self.ask(cmd_str))

    ###
    # Generic SCPI commands from IEEE 488.2 (IEC 625-2) standard
    ###

    def clear_status(self) -> None:
        self.write('*CLS')

    def set_event_status_enable(self, value: int) -> None:
        self.write('*ESE %d' % value)

    def get_event_status_enable(self) -> str:
        return self.ask('*ESE?')

    def get_event_status_enable_register(self) -> str:
        return self.ask('*ESR?')

    def get_identity(self) -> str:
        return self.ask('*IDN?')

    def operation_complete(self) -> None:
        self.write('*OPC')

    def get_operation_complete(self) -> str:
        return self.ask('*OPC?')

    def get_options(self) -> str:
        return self.ask('*OPT?')

    def service_request_enable(self, value: int) -> None:
        self.write('*SRE %d' % value)

    def get_service_request_enable(self) -> int:
        return self.ask_int('*SRE?')

    def get_status_byte(self) -> int:
        return self.ask_int('*STB?')

    def get_test_result(self) -> int:
        # NB: result bits are device dependent
        return self.ask_int('*TST?')

    def trigger(self) -> None:
        self.write('*TRG')

    def wait(self) -> None:
        self.write('*WAI')

    def reset(self) -> None:
        self.write('*RST')

    ###
    # Required SCPI commands (SCPI std V1999.0 4.2.1)
    ###

    def get_error(self) -> str:
        """ Returns:    '0,"No error"' or <error message>
        """
        return self.ask('system:err?')

    def get_system_error_count(self):
        return self.ask_int('system:error:count?')

    def get_system_version(self) -> str:
        return self.ask('system:version?')

    ###
    # IEEE 488.2 binblock handling
    ###

    def bin_block_write(self, bin_block: bytes, cmd_str: str) -> None:
        """
        write IEEE488.2 binblock

        Args:
            bin_block (bytearray): binary data to send
            cmd_str (str): command string to use
        """
        header = cmd_str + SCPI._build_header_string(len(bin_block))
        bin_msg = header.encode() + bin_block
        self._write_binary(bin_msg)
        self.write('')                  # add a Line Terminator

    def bin_block_read(self) -> bytes:
        """ read IEEE488.2 binblock
        """
        # get and decode header
        header_a = self._read_binary(2)                        # read '#N'
        header_a_str = header_a.decode()
        if header_a_str[0] != '#':
            s = 'SCPI header error: received {}'.format(header_a)
            raise RuntimeError(s)
        digit_cnt = int(header_a_str[1])
        header_b = self._read_binary(digit_cnt)
        byte_cnt = int(header_b.decode())
        bin_block = self._read_binary(byte_cnt)
        self._read_binary(2)                                  # consume <CR><LF>
        return bin_block

    @staticmethod
    def _build_header_string(byte_cnt: int) -> str:
        """ generate IEEE488.2 binblock header
        """
        byte_cnt_str = str(byte_cnt)
        digit_cnt_str = str(len(byte_cnt_str))
        bin_header_str = '#' + digit_cnt_str + byte_cnt_str
        return bin_header_str