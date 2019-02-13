"""
    File:       SCPIBase.py
    Author:     Wouter Vlothuizen, TNO/QuTech
    Purpose:    self contained base class for SCPI ('Standard Commands for Programmable Instruments') commands, with
                selectable transport
    Usage:      don't use directly, use a derived class (e.g. Qutech_CC)
    Notes:
    Bugs:
    Changelog:

20190213 WJV
- started, based on SCPI.py

"""

from .Transport import Transport


class SCPIBase:
    def __init__(self, name: str, transport: Transport) -> None:
        self._transport = transport

    ###
    # Helpers
    ###

    def ask_float(self, cmd_str: str) -> float:
        return float(self._transport.ask(cmd_str))

    def ask_int(self, cmd_str: str) -> int:
        return int(self._transport.ask(cmd_str))

    ###
    # Generic SCPI commands from IEEE 488.2 (IEC 625-2) standard
    ###

    def clear_status(self) -> None:
        self._transport.write('*CLS')

    def set_event_status_enable(self, value: int) -> None:
        self._transport.write('*ESE %d' % value)

    def get_event_status_enable(self) -> str:
        return self._transport.ask('*ESE?')

    def get_event_status_enable_register(self) -> str:
        return self._transport.ask('*ESR?')

    def get_identity(self) -> str:
        return self._transport.ask('*IDN?')

    def operation_complete(self) -> None:
        self._transport.write('*OPC')

    def get_operation_complete(self) -> str:
        return self._transport.ask('*OPC?')

    def get_options(self) -> str:
        return self._transport.ask('*OPT?')

    def service_request_enable(self, value: int) -> None:
        self._transport.write('*SRE %d' % value)

    def get_service_request_enable(self) -> int:
        return self.ask_int('*SRE?')

    def get_status_byte(self) -> int:
        return self.ask_int('*STB?')

    def get_test_result(self) -> int:
        # NB: result bits are device dependent
        return self.ask_int('*TST?')

    def trigger(self) -> None:
        self._transport.write('*TRG')

    def wait(self) -> None:
        self._transport.write('*WAI')

    def reset(self) -> None:
        self._transport.write('*RST')

    ###
    # Required SCPI commands (SCPI std V1999.0 4.2.1)
    ###

    def get_error(self) -> str:
        """ Returns:    '0,"No error"' or <error message>
        """
        return self._transport.ask('system:err?')

    def get_system_error_count(self):
        return self.ask_int('system:error:count?')

    def get_system_version(self) -> str:
        return self._transport.ask('system:version?')

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
        self._transport.write_binary(bin_msg)
        self._transport.write('')                  # add a Line Terminator

    def bin_block_read(self) -> bytes:
        """ read IEEE488.2 binblock
        """
        # get and decode header
        header_a = self._transport.read_binary(2)                        # read '#N'
        header_a_str = header_a.decode()
        if header_a_str[0] != '#':
            s = 'SCPI header error: received {}'.format(header_a)
            raise RuntimeError(s)
        digit_cnt = int(header_a_str[1])
        header_b = self._transport.read_binary(digit_cnt)
        byte_cnt = int(header_b.decode())
        bin_block = self._transport.read_binary(byte_cnt)
        self._transport.read_binary(2)                                  # consume <CR><LF>
        return bin_block

    @staticmethod
    def _build_header_string(byte_cnt: int) -> str:
        """ generate IEEE488.2 binblock header
        """
        byte_cnt_str = str(byte_cnt)
        digit_cnt_str = str(len(byte_cnt_str))
        bin_header_str = '#' + digit_cnt_str + byte_cnt_str
        return bin_header_str