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

    def _ask(self, cmd_str: str) -> str:
        self._transport.write(cmd_str)
        return self._transport.readline().rstrip()  # remove trailing white space, CR, LF

    def _ask_float(self, cmd_str: str) -> float:
        return float(self._ask(cmd_str))  # FIXME: can raise ValueError

    def _ask_int(self, cmd_str: str) -> int:
        return int(self._ask(cmd_str))  # FIXME: can raise ValueError

    ###
    # Generic SCPI commands from IEEE 488.2 (IEC 625-2) standard
    ###

    def clear_status(self) -> None:
        self._transport.write('*CLS')

    def set_event_status_enable(self, value: int) -> None:
        self._transport.write('*ESE %d' % value)

    def get_event_status_enable(self) -> str:
        return self._ask('*ESE?')

    def get_event_status_register(self) -> str:
        return self._ask('*ESR?')

    def get_identity(self) -> str:
        return self._ask('*IDN?')

    def operation_complete(self) -> None:
        self._transport.write('*OPC')

    def get_operation_complete(self) -> str:
        return self._ask('*OPC?')

    def get_options(self) -> str:
        return self._ask('*OPT?')

    def service_request_enable(self, value: int) -> None:
        self._transport.write('*SRE %d' % value)

    def get_service_request_enable(self) -> int:
        return self._ask_int('*SRE?')

    def get_status_byte(self) -> int:
        return self._ask_int('*STB?')

    def get_test_result(self) -> int:
        # NB: result bits are device dependent
        return self._ask_int('*TST?')

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
        return self._ask('system:err?')

    def get_system_error_count(self):
        return self._ask_int('system:error:count?')

    def get_system_version(self) -> str:
        return self._ask('system:version?')

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
        header = cmd_str + SCPIBase._build_header_string(len(bin_block))
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

    # FIXME: additions
    # enter_critical
    # exit_critical

    def get_status_questionable_condition(self):
        return self._ask_int('STATus:QUEStionable:CONDition?')

    def get_status_questionable_event(self):
        return self._ask_int('STATus:QUEStionable:EVENt?')

    def set_status_questionable_enable(self, val):
        self._transport.write('STATus:QUEStionable:ENABle {}'.format(val))

    def get_status_questionable_enable(self):
        return self._ask_int('STATus:QUEStionable:ENABle?')


    def set_status_preset(self):
        self._transport.write('STATus:PRESet')


    def get_status_operation_condition(self):
        return self._ask_int('STATus:OPERation:CONDition?')

    def get_status_operation_event(self):
        return self._ask_int('STATus:OPERation:EVENt?')

    def set_status_operation_enable(self, val):
        self._transport.write('STATus:OPERation:ENABle {}'.format(val))

    def get_status_operation_enable(self):
        return self._ask_int('STATus:OPERation:ENABle?')


    ###
    # IEEE488.2 status constants
    ###

    # bits for *ESR and *ESE
    ESR_OPERATION_COMPLETE = 1
    ESR_REQUEST_CONTROL = 2
    ESR_QUERY_ERROR = 4
    ESR_DEVICE_DEPENDENT_ERROR = 8
    ESR_EXECUTION_ERROR = 16
    ESR_COMMAND_ERROR = 32
    ESR_USER_REQUEST = 64
    ESR_POWER_ON = 128

    # bits for STATus:OPERation
    # FIXME: add the function
    STAT_OPER_CALIBRATING = 1       # The instrument is currently performing a calibration
    STAT_OPER_SETTLING = 2          # The instrument is waiting for signals it controls to stabilize enough to begin measurements
    STAT_OPER_RANGING = 4           # The instrument is currently changing its range
    STAT_OPER_SWEEPING = 8          # A sweep is in progress
    STAT_OPER_MEASURING = 16        # The instrument is actively measuring
    STAT_OPER_WAIT_TRIG = 32        # The instrument is in a “wait for trigger” state of the trigger model
    STAT_OPER_WAIT_ARM = 64         # The instrument is in a “wait for arm” state of the trigger model
    STAT_OPER_CORRECTING = 128      # The instrument is currently performing a correction
    STAT_OPER_INSTR_SUMMARY = 8192  # One of n multiple logical instruments is reporting OPERational status
    STAT_OPER_PROG_RUNNING = 16384  # A user-defined program is currently in the run state

    # bits for STATus:QUEStionable
    # FIXME: add the function
    STAT_QUES_VOLTAGE = 1
    STAT_QUES_CURRENT = 2
    STAT_QUES_TIME = 4
    STAT_QUES_POWER = 8
    STAT_QUES_TEMPERATURE = 16
    STAT_QUES_FREQUENCY = 32
    STAT_QUES_PHASE = 64
    STAT_QUES_MODULATION = 128
    STAT_QUES_CALIBRATION = 256
    STAT_QUES_INSTR_SUMMARY = 8192
    STAT_QUES_COMMAND_WARNING = 16384

