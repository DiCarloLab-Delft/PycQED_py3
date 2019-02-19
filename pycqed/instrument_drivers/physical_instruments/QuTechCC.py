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
        """
        hdr = 'QUTech:SEQuence:PROGram'
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

