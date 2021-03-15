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
            sys.stderr.write('assembly error log:\n{}\n'.format(self.get_assembler_log()))
            raise RuntimeError('assembly failed')

    def assemble_and_start(self, program_string: str) -> None:
        self.assemble(program_string)
        log.debug('starting CC')
        self.start()
        log.debug('checking for SCPI errors on CC')
        self.check_errors()
        log.debug('done checking for SCPI errors on CC')

    ##########################################################################
    # overloaded status printing functions
    ##########################################################################

    def print_status_questionable_condition(self) -> None:
        sqc = self.get_status_questionable_condition()
        self._print_item("status_questionable_condition", sqc, self._stat_qeus_lookup)
        if sqc & self.STAT_QUES_FREQUENCY:
            self._print_item("status_questionable_frequency_condition", self.get_status_questionable_frequency_condition(), )

        if sqc & self.STAT_QUES_CONFIG:
            self._print_item("", self.get_stat(), )

        if sqc & self.STAT_QUES_BPLINK:
            self._print_item("", self.(), )

        if sqc & self.STAT_QUES_DIO:
            self._print_item("", self.(), )

        if sqc & self.STAT_QUES_INST_SUMMARY:
            sqis =  self.get_status_questionable_instrument_condition()
            self._print_item("", self.(), )

    def print_status_questionable_event(self) -> None:
        sqe = self.get_status_questionable_event()
        self._print_item("status_questionable_event", sqe, self._stat_qeus_lookup)
        if sqe & self.STAT_QUES_FREQUENCY:
            self._print_item("status_questionable_frequency_event", self.get_status_questionable_frequency_event())

    ##########################################################################
    # CC SCPI protocol wrapper functions
    ##########################################################################

    def sequence_program_assemble(self, program_string: str) -> None:
        """
        upload sequence program string
        """
        # check size, because overrunning gives irrecoverable errors. FIXME: move to Transport
        if len(program_string) > self._MAX_PROG_STR_LEN:
            raise RuntimeError('source program size {len(program_string)} exceeds maximum of {self._MAX_PROG_STR_LEN}')

        hdr = 'QUTech:SEQuence:PROGram:ASSEMble ' # NB: include space as separator for binblock parameter
        bin_block = program_string.encode('ascii')
        self.bin_block_write(bin_block, hdr)

    def get_sequence_program_assemble(self) -> str:
        """
        download sequence program string
        """
        return self._ask_bin('QUTech:SEQuence:PROGram:ASSEMble?').decode('utf-8', 'ignore')

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

    def set_seqbar_cnt(self, ccio: int, val: int) -> None:
        # no need to stop CC
        self._transport.write(f'QUTech:CCIO{ccio}:SEQBARcnt {val}')

    def calibrate_dio(self, ccio: int, expected_bits: int) -> None:
        self._transport.write(f'QUTech:CCIO{ccio}:DIOIN:CAL {expected_bits}')

    def get_calibrate_dio_success(self, ccio: int) -> int:
        return self._ask_int('QUTech:CCIO#:DIOIN:CALibrate:SUCCESS?')

    def get_calibrate_dio_read_index(self, ccio: int) -> int:
        return self._ask_int('QUTech:CCIO#:DIOIN:CALibrate:READINDEX?')

    def get_calibrate_dio_margin(self, ccio: int) -> int:
        return self._ask_int('QUTech:CCIO#:DIOIN:CALibrate:MARGIN?')

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

    def debug_get_ccio_trace(self, ccio: int) -> str:
        return self._ask_bin(f'QUTech:DEBUG:CCIO{ccio}:TRACE?').decode('utf-8', 'ignore')

    def debug_get_traces(self, ccio_mask: int) -> str:
        return self._ask_bin(f'QUTech:DEBUG:TRACES? {ccio_mask}').decode('utf-8', 'ignore')

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


    def get_status_questionable_instrument_condition(self) -> int:
        return self._ask_int('STATus:QUEStionable:INSTrument:CONDition?')

    def get_status_questionable_instrument_event(self) -> int:
        return self._ask_int('STATus:QUEStionable:INSTrument:EVENt?')

    def set_status_questionable_instrument_enable(self, val) -> None:
        self._transport.write(f'STATus:QUEStionable:INSTrument:ENABle {val}')

    def get_status_questionable_instrument_enable(self) -> int:
        return self._ask_int('STATus:QUEStionable:INSTrument:ENABle?')


    def get_status_questionable_instrument_isummary_condition(self, ccio: int) -> int:
        return self._ask_int(f'STATus:QUEStionable:INSTrument:ISUMmary{ccio}:CONDition?')

    def get_status_questionable_instrument_isummary_event(self, ccio: int) -> int:
        return self._ask_int(f'STATus:QUEStionable:INSTrument:ISUMmary{ccio}:EVENt?')

    def set_status_questionable_instrument_isummary_enable(self, ccio: int, val) -> None:
        self._transport.write(f'STATus:QUEStionable:INSTrument:ISUMmary{ccio}:ENABle {val}')

    def get_status_questionable_instrument_isummary_enable(self, ccio: int) -> int:
        return self._ask_int(f'STATus:QUEStionable:INSTrument:ISUMmary{ccio}:ENABle?')


    ##########################################################################
    # constants
    ##########################################################################

    _MAX_PROG_STR_LEN = 40*1024*1024-1024  # size of CC input buffer, minus some room for command. FIXME: get from instrument

    # trace units
    TRACE_CCIO_DEV_IN = 0
    TRACE_CCIO_DEV_OUT = 1
    TRACE_CCIO_BP_IN = 2
    TRACE_CCIO_BP_OUT = 3

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

    ##########################################################################
    # status constants
    ##########################################################################

    # stat_qeus extensions
    # SCPI standard: "bit 9 through 13 are available to designer"
    STAT_QUES_CONFIG            = 0x0200
    STAT_QUES_BPLINK            = 0x0400
    STAT_QUES_DIO               = 0x0800 # NB: CCIO only

    # overload _stat_ques_lookup
    _stat_ques_lookup = [
        (STAT_QUES_CONFIG, "Configuration error"),
        (STAT_QUES_BPLINK, "Backplane link error"),
        (STAT_QUES_DIO, "DIO interface error")
    ] + super()._stat_ques_lookup

    # stat_ques_freq
    SQF_CLK_SRC_INTERN          = 0x0001
    SQF_PLL_UNLOCK              = 0x0002
    SQF_CLK_MUX_SWITCH          = 0x0004

    _cc_stat_ques_freq_lookup = [
        (SQF_CLK_SRC_INTERN, "FPGA uses internal clock (not locked to external reference)"),
        (SQF_PLL_UNLOCK, "PLL unlocked (external reference missing)"),
        (SQF_CLK_MUX_SWITCH, "FPGA clock multiplexer has switched")
    ]

    # stat_ques_config
    SQC_EEPROM_CCIOCORE         = 0x0001
    SQC_EEPROM_ENCLUSTRA        = 0x0002
    SQC_INCONSISTENT_IP_ADDRESS = 0x0004

    _cc_stat_ques_config_lookup = [
        (SQC_EEPROM_CCIOCORE, "CCIO/CCCORE EEPROM contents invalid"),
        (SQC_EEPROM_ENCLUSTRA, "Enclustra FPGA module EEPROM contents invalid"),
        (SQC_INCONSISTENT_IP_ADDRESS, "IP address is inconsistent with hardware slot ID")
    ]

    # stat_ques_bplink : Backplane link status
    SQB_NO_SIGNAL               = 0x0001
    SQB_INSUF_TIMING_MARGIN     = 0x0002
    SQB_CAL_FAILED              = 0x0004
    SQB_DESYNC                  = 0x0008
    SQB_PARITY_ERROR            = 0x0010
    SQB_REPEATER_OVERFLOW       = 0x4000 # NB: CCCORE only

    _cc_stat_ques_bplink_lookup = [
        (SQB_NO_SIGNAL, "No signal detected during backplane link timing calibration"),
        (SQB_INSUF_TIMING_MARGIN, "Insufficient timing margin during backplane link timing calibration"),
        (SQB_CAL_FAILED, "Backplane link timing calibration failed"),
        (SQB_DESYNC, "Synchronization error on backplane link"),
        (SQB_PARITY_ERROR, "Parity error on backplane link"),
        (SQB_REPEATER_OVERFLOW, "Overflow on CCCORE backplane link repeater")
    ]

    # stat_ques_diocal : DIO timing calibration status (CCIO only)
    SQD_NO_SIGNAL               = 0x0001
    SQD_INSUF_TIMING_MARGIN     = 0x0002
    SQD_BITS_INACTIVE           = 0x0004
    SQD_NOT_CALIBRATED          = 0x0008
    SQD_TIMING_ERROR            = 0x0010

    _cc_stat_ques_diocal_lookup = [
        (SQD_NO_SIGNAL, "No signal detected during DIO timing calibration"),
        (SQD_INSUF_TIMING_MARGIN, "Insufficient timing margin during DIO timing calibration"),
        (SQD_BITS_INACTIVE, "Required bits were inactive during DIO timing calibration"),
        (SQD_NOT_CALIBRATED, "DIO timing calibration not yet performed (successfully)"),
        (SQD_TIMING_ERROR, "Runtime DIO timing violation found")
    ]
