"""
    File:               QuTechCC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            Python control of Qutech Central Controller
    Notes:
    Usage:
    Bugs:

"""

import logging

from .SCPIBase import SCPIBase
from .Transport import Transport

log = logging.getLogger(__name__)


class QuTechCC(SCPIBase):

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

    def start(self) -> None:
        """
        """
        self._transport.write('awgcontrol:run:immediate')

    def stop(self) -> None:
        """
        """
        self._transport.write('awgcontrol:stop:immediate')


    def get_status_questionable_frequency_condition(self):
        return self._ask_int('STATus:QUEStionable:FREQ:CONDition?')

    def get_status_questionable_frequency_event(self):
        return self._ask_int('STATus:QUEStionable:FREQ:EVENt?')

    def set_status_questionable_frequency_enable(self, val):
        self._transport.write('STATus:QUEStionable:FREQ:ENABle {}'.format(val))

    def get_status_questionable_frequency_enable(self):
        return self._ask_int('STATus:QUEStionable:FREQ:ENABle?')


    ##########################################################################
    # 'public' functions for the end user, CC-light 'emulation'
    # FIXME: these are compatibility hacks to ease integration in the existing CC-light toolchain
    ##########################################################################

    def eqasm_program(self, file_name: str) -> None:
        real_file_name = file_name.replace(".qisa", ".vq1asm")  # correct assumption of openql_helpers.py::compile()
        with open(real_file_name, 'r') as f:
            prog = f.read()
        self.sequence_program(prog)
"""
TODO:
    vsm_channel_delay
    
"""
