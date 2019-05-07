"""
    File:               QuTechCC_core.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            Python control of Qutech Central Controller. Core driver independent of QCoDeS
    Notes:
    Usage:
    Bugs:

"""

import logging

from .SCPIBase import SCPIBase
from .Transport import Transport

log = logging.getLogger(__name__)


class QuTechCC_core(SCPIBase):

    ##########################################################################
    # 'public' functions for the end user
    ##########################################################################

    def __init__(self,
                 name: str,
                 transport: Transport):
        super().__init__(name, transport)

        # self.get_idn()
        # self._add_parameters()
        # self.connect_message()

    def sequence_program(self, program_string: str) -> None:
        """
        upload sequence program string
        """
        hdr = 'QUTech:SEQuence:PROGram ' # NB: include space as separator for binblock parameter
        bin_block = program_string.encode('ascii')
        self.bin_block_write(bin_block, hdr)

    # FIXME: add function to get assembly errors

    def set_q1_reg(self, ccio: int, reg: int, val: int) -> None:
        self._transport.write('QUTech:CCIO{}:Q1REG{} {}'.format(ccio, reg, val))

    def get_q1_reg(self, ccio: int, reg: int) -> int:
        return self._transport.ask_int('QUTech:CCIO{}:Q1REG{}'.format(ccio, reg))

    def set_vsm_delay_rise(self, ccio: int, bit: int, cnt_in_39_ps_steps: int) -> None:
        self._transport.write('QUTech:CCIO{}:VSMbit{}:RISEDELAY {}'.format(ccio, bit, cnt_in_39_ps_steps))

    def get_vsm_delay_rise(self, ccio: int, bit: int) -> int:
        return self._transport._ask_int('QUTech:CCIO#:VSMbit#:RISEDELAY?'.format(ccio, bit))

    def set_vsm_delay_fall(self, ccio: int, bit: int, cnt_in_39_ps_steps: int) -> None:
        self._transport.write('QUTech:CCIO{}:VSMbit{}:FALLDELAY {}'.format(ccio, bit, cnt_in_39_ps_steps))

    def get_vsm_delay_fall(self, ccio: int, bit: int) -> int:
        return self._transport._ask_int('QUTech:CCIO{}:VSMbit{}:FALLDELAY?'.format(ccio, bit))

    def debug_marker_off(self, ccio: int) -> None:
        self._transport.write('QUTech:DEBUG:CCIO{}:MARKER:OFF'.format(ccio))

    def debug_marker_in(self, ccio: int, bit: int) -> None:
        self._transport.write('QUTech:DEBUG:CCIO{}:MARKER:IN {}'.format(ccio, bit))

    def debug_marker_out(self, ccio: int, bit: int) -> None:
        self._transport.write('QUTech:DEBUG:CCIO{}:MARKER:OUT {}'.format(ccio, bit))

    def start(self) -> None:
        self._transport.write('awgcontrol:run:immediate')

    def stop(self) -> None:
        self._transport.write('awgcontrol:stop:immediate')

    ### status functions ###
    def get_status_questionable_frequency_condition(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:CONDition?')

    def get_status_questionable_frequency_event(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:EVENt?')

    def set_status_questionable_frequency_enable(self, val) -> None:
        self._transport.write('STATus:QUEStionable:FREQ:ENABle {}'.format(val))

    def get_status_questionable_frequency_enable(self) -> int:
        return self._ask_int('STATus:QUEStionable:FREQ:ENABle?')

    # DIO/marker bit definitions: output
    UHFQA_TOGGLE_DS = 31
    UHFQA_TRIG = 16

    HDAWG_TOGGLE_DS = 30
    HDAWG_TRIG = 31

    # DIO/marker bit definitions: input
