"""
    File:               QuTechCC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            Python control of Qutech Central Controller: adds application dependent stuff to QuTechCC_core
    Notes:
    Usage:
    Bugs:

"""

from .QuTechCC_core import QuTechCC_core
from .Transport import Transport

from qcodes.utils import validators as vals
from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import Parameter


class QuTechCC(QuTechCC_core, Instrument):
    def __init__(self,
                 name: str,
                 transport: Transport):
        super().__init__(name, transport) # FIXME: QuTechCC_core

        # set constants
        self._num_dio = 9  # the number of DIO connectors used
        self._ccio_slot_driving_vsm = 5  # the slot number of the CCIO driving the VSM (FIXME: supports one VSM only)
        self._num_vsm_ch = 32  # the number of VSM channels used per connector
        self._q1reg_dio_delay = 63  # the register used in OpenQL generated programs to set DIO delay

        self._add_parameters(self._num_dio, self._num_vsm_ch)
        self._add_compatibility_parameters(self._num_dio, self._num_vsm_ch)


    def _add_parameters(self, num_dio: int, num_vsm_ch: int):
        """
        add CC native parameters
        """


    def _add_compatibility_parameters(self, num_dio: int, num_vsm_ch: int):
        """
        parameters for the end user, CC-light 'emulation'
        FIXME: these are compatibility hacks to ease integration in the existing CC-light toolchain, richer functionality may be available
        """

        # support for openql_helpers.py::compile()
        self.add_parameter(
            'eqasm_program',
            label='eQASM program (compatibility function)',
            docstring='Uploads the program to the CC. Valid input is a string representing the filename.',
            set_cmd=self._eqasm_program,
            vals=vals.Strings()
        )

        # support 'dio{}_out_delay' for device_object_CCL.py::prepare_timing()
        for dio in range(1, num_dio + 1):  # NB: DIO starts from 1 on CC-light/QCC
            cmd = 'QUTech:CCIO{}:Q1REG{}'.format(dio, self._q1reg_dio_delay)
            self.add_parameter(
                'dio{}_out_delay'.format(dio),
                label='Output Delay of DIO{}'.format(dio),
                docstring='This parameter determines the extra output delay introduced for the DIO{} channel'.format(dio),
                unit='20 ns',
                vals=vals.PermissiveInts(0, 31),
                get_cmd=cmd + '?',
                set_cmd=cmd + ' {}',
                get_parser=int)

        # support for 'vsm_channel_delay{}' for CCL_Transmon.py::_set_mw_vsm_delay(), also see calibrate_mw_vsm_delay()
        # NB: CC supports 1/1200 MHz ~= 833 ps resolution
        # NB: CC supports setting trailing edge delay separately
        # FIXME: CC-light range max = 127*2.5 ns = 317.5 ns, ours is 64/1200 MHz = 53 ns, so we must also shift program
        for vsm_ch in range(0, num_vsm_ch):  # NB: VSM channel starts from 0 on CC-light/QCC
            self.add_parameter(
                'vsm_channel_delay{}'.format(vsm_ch),
                label='VSM Channel {} delay'.format(vsm_ch),
                docstring='Sets/gets the delay for VSM channel {}'.format(vsm_ch),
                unit='2.5 ns',
                vals=vals.PermissiveInts(0, 127),
                get_cmd=cmd + '?',
                set_cmd=cmd + ' {}',
                get_parser=int)

    # helper for parameter 'eqasm_program'
    def _eqasm_program(self, file_name: str) -> None:
        real_file_name = file_name.replace(".qisa", ".vq1asm")  # correct assumption of openql_helpers.py::compile()
        with open(real_file_name, 'r') as f:
            prog = f.read()
        self.sequence_program(prog)

    # helper for parameter 'vsm_channel_delay{}'
    def _set_vsm_channel_delay(self):



