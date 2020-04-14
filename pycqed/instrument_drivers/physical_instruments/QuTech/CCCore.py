"""
    File:       CCCore.py
    Author:     Wouter Vlothuizen, QuTech
    Purpose:    Core Instrument driver for QuTech Central Controller, independent of QCoDeS.
                All instrument protocol handling is provided here
    Usage:      Can be used directly, or with CC.py, which adds access via QCoDeS parameters
    Notes:      Here, we follow the SCPI convention of NOT checking parameter values but leaving that to
                the device
                The name CCCore refers to the fact that this is a 'core' driver (just as QWGCore and ZI_HDAWG_core),
                not to the CCCORE board within the CC
    Usage:
    Bugs:

"""

import logging
import sys

from pycqed.instrument_drivers.library.SCPIBase import SCPIBase
from pycqed.instrument_drivers.library.Transport import Transport

log = logging.getLogger(__name__)


class CCCore(SCPIBase):

    MAX_PROG_STR_LEN = 40*1024*1024-1024  # size of CC input buffer, minus some room for command. FIXME: get from instrument

    # trace units
    TRACE_CCIO_DEV_IN = 0
    TRACE_CCIO_DEV_OUT = 1
    TRACE_CCIO_BP_IN = 2
    TRACE_CCIO_BP_OUT = 3

    ##########################################################################
    # 'public' functions for the end user
    ##########################################################################

    def __init__(self,
                 name: str,
                 transport: Transport):
        super().__init__(name, transport)

    ##########################################################################
    # convenience functions
    ##########################################################################

    def assemble(self, program_string: str) -> None:
        self.sequence_program_assemble(program_string)  # NB: takes ~1.1 s for RB with 2048 Cliffords (1 measurement only)
        if self.get_assembler_success() != 1:
            sys.stderr.write('assembly error log:\n{}\n'.format(self.get_assembler_log()))  # FIXME: result is messy
            raise RuntimeError('assembly failed')

    def assemble_and_start(self, program_string: str) -> None:
        self.assemble(program_string)
        log.debug('starting CC')
        self.start()
        log.debug('checking for SCPI errors on CC')
        self.check_errors()
        log.debug('done checking for SCPI errors on CC')

    ##########################################################################
    # CC SCPI protocol wrapper functions
    ##########################################################################

    def sequence_program_assemble(self, program_string: str) -> None:
        """
        upload sequence program string
        """
        # check size, because overrunning gives irrecoverable errors. FIXME: move to Transport
        if len(program_string) > self.MAX_PROG_STR_LEN:
            raise RuntimeError('source program size {len(program_string)} exceeds maximum of {self.MAX_PROG_STR_LEN}')

        hdr = 'QUTech:SEQuence:PROGram:ASSEMble ' # NB: include space as separator for binblock parameter
        bin_block = program_string.encode('ascii')
        self.bin_block_write(bin_block, hdr)

    def get_assembler_success(self) -> int:
        return self._ask_int('QUTech:SEQuence:PROGram:ASSEMble:SUCCESS?')

    def get_assembler_log(self) -> str:
        return self._ask_bin('QUTech:SEQuence:PROGram:ASSEMble:LOG?').decode('utf-8', 'ignore')

    def set_q1_reg(self, ccio: int, reg: int, val: int) -> None:
        # only possible if CC is stopped
        self._transport.write(f'QUTech:CCIO{ccio}:Q1REG{reg} {val}')

    def get_q1_reg(self, ccio: int, reg: int) -> int:
        # only possible if CC is stopped
        return self._ask_int(f'QUTech:CCIO{ccio}:Q1REG{reg}')

    def calibrate_dio(self, ccio: int, expected_bits: int) -> None:
        self._transport.write(f'QUTech:CCIO{ccio}:DIOIN:CAL {expected_bits}')

    def set_vsm_delay_rise(self, ccio: int, bit: int, cnt_in_833_ps_steps: int) -> None:
        self._transport.write(f'QUTech:CCIO{ccio}:VSMbit{bit}:RISEDELAY {cnt_in_833_ps_steps}')

    def get_vsm_delay_rise(self, ccio: int, bit: int) -> int:
        return self._ask_int(f'QUTech:CCIO{ccio}:VSMbit{bit}:RISEDELAY?')

    def set_vsm_delay_fall(self, ccio: int, bit: int, cnt_in_833_ps_steps: int) -> None:
        self._transport.write(f'QUTech:CCIO{ccio}:VSMbit{bit}:FALLDELAY {cnt_in_833_ps_steps}')

    def get_vsm_delay_fall(self, ccio: int, bit: int) -> int:
        return self._transport._ask_int(f'QUTech:CCIO{ccio}:VSMbit{bit}:FALLDELAY?')

    def debug_marker_off(self, ccio: int) -> None:
        self._transport.write(f'QUTech:DEBUG:CCIO{ccio}:MARKER:OFF')

    def debug_marker_in(self, ccio: int, bit: int) -> None:
        self._transport.write(f'QUTech:DEBUG:CCIO{ccio}:MARKER:IN {bit}')

    def debug_marker_out(self, ccio: int, bit: int) -> None:
        self._transport.write(f'QUTech:DEBUG:CCIO{ccio}:MARKER:OUT {bit}')

    def debug_get_ccio_reg(self, ccio: int, reg: int) -> int:
        return self._ask_int(f'QUTech:DEBUG:CCIO{ccio}:REG{reg}?')

    def debug_set_ccio_trace_on(self, ccio: int, tu_idx: int) -> None:
        self._transport.write(f'QUTech:DEBUG:CCIO{ccio}:TRACE{tu_idx}:ON')

    def start(self, block: bool = True) -> None:
        """
        start the CC sequencers

        :param block: call get_operation_complete to assure that the instrument has started before we return, which is a
        common assumption throughout PycQED. This behaviour can be disabled to allow asynchronous operation, e.g. to
        optimize starting a range of instruments.
        """
        self._transport.write('awgcontrol:run:immediate')
        if block:
            self.get_operation_complete()

    def stop(self, block: bool = True) -> None:
        """
        stop the CC sequencers

        :param block: call get_operation_complete to assure that the instrument has stopped before we return, which is a
        common assumption throughout PycQED. This behaviour can be disabled to allow asynchronous operation, e.g. to
        optimize stopping a range of instruments.
        """
        self._transport.write('awgcontrol:stop:immediate')
        if block:
            self.get_operation_complete()

    ### status functions ###
    def get_status_questionable_frequency_condition(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:CONDition?')

    def get_status_questionable_frequency_event(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:EVENt?')

    def set_status_questionable_frequency_enable(self, val) -> None:
        self._transport.write(f'STATus:QUEStionable:FREQ:ENABle {val}')

    def get_status_questionable_frequency_enable(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:ENABle?')

    ##########################################################################
    # constants
    ##########################################################################

    # HDAWG DIO/marker bit definitions: CC output
    HDAWG_TOGGLE_DS = 30
    HDAWG_TRIG = 31
    HDAWG_CW = range(0,29)  # NB: bits used depend on mode

    # QWG DIO/marker bit definitions: CC output
    QWG_TOGGLE_DS = 30
    QWG_TRIG = 31
    QWG1_CW = range(0,11)
    QWG2_CW = range(16,27)

    # UHFQA DIO/marker bit definitions: CC output
    UHFQA_TOGGLE_DS = 31
    UHFQA_TRIG = 16
    UHFQA_CW = range(17,26)

    # UHFQA DIO/marker bit definitions: CC input
    UHFQA_DV = 0
    UHFQA_RSLT = range(1,10)
