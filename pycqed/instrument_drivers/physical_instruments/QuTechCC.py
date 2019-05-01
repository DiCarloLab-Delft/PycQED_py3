"""
    File:               QuTechCC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            Python control of Qutech Central Controller: adds application dependent stuff to QuTechCC_core
    Notes:              use QuTechCC_core to talk to instrument, do not add knowledge of SCPI syntax here
    Usage:
    Bugs:

"""

import logging

from .QuTechCC_core import QuTechCC_core
from .Transport import Transport

from qcodes.utils import validators as vals
from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import Parameter

log = logging.getLogger(__name__)


class QuTechCC(QuTechCC_core, Instrument):
    def __init__(self,
                 name: str,
                 transport: Transport):
        super().__init__(name, transport) # FIXME: super -> QuTechCC_core

        # set constants
        self._num_dio = 9  # the number of DIO connectors used
        self._ccio_slot_driving_vsm = 5  # the slot number of the CCIO driving the VSM (FIXME: supports one VSM only)
        self._num_vsm_ch = 32  # the number of VSM channels used per connector
        self._q1reg_dio_delay = 63  # the register used in OpenQL generated programs to set DIO delay

        self._add_parameters(self._num_dio, self._num_vsm_ch)
        self._add_compatibility_parameters(self._num_dio, self._num_vsm_ch)


    ##########################################################################
    # QCoDeS parameter definitions
    ##########################################################################

    def _add_parameters(self, num_dio: int, num_vsm_ch: int):
        """
        add CC native parameters
        """


    def _add_compatibility_parameters(self, num_dio: int, num_vsm_ch: int):
        """
        parameters for the end user, CC-light 'emulation'
        FIXME:  these are compatibility hacks to ease integration in the existing CC-light toolchain,
                richer functionality may be available via the native interface
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
            cmd = 'QUTech:CCIO{}:Q1REG{}'.format(dio, self._q1reg_dio_delay) # FIXME: use core driver, don't talk directly
            self.add_parameter(
                'dio{}_out_delay'.format(dio),
                label='Output Delay of DIO{}'.format(dio),
                docstring='This parameter determines the extra output delay introduced for the DIO{} channel'.format(dio),
                unit='20 ns',
                vals=vals.PermissiveInts(0, 31),
                get_cmd=cmd + '?',
                set_cmd=cmd + ' {}',
                get_parser=int)

        # FIXME: num_append_pts does not seem to be actually used in pycQED
        """
           "docstring": "This parameter determines the number of append points to the vsm mask signal. The unit is 2.5 ns per point.",
            "get_cmd": "QUTech:NumAppendPts?",
            "label": "Number Append Points",
            "name": "num_append_pts",
            "set_cmd": "QUTech:NumAppendPts {}",
            "unit": "2.5 ns per point",
            "vals": {
                "range": [
                    0,
                    7
                ],
                "type": "Non_Neg_Number"
            }
        """

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
                set_cmd=self._gen_set_func_1par(self._set_vsm_channel_delay, vsm_ch),
#                get_cmd=self._gen_get_func_1par(self._get_vsm_channel_delay, vsm_ch),
                )

    ##########################################################################
    # native parameter support
    ##########################################################################

    # helpers for Instrument::add_parameter.set_cmd
    def _gen_set_func_1par(self, fun, par1):
        def set_func(val):
            return fun(par1, val)
        return set_func

    def _gen_set_func_2par(self, fun, par1, par2):
        def set_func(val):
            return fun(par1, par2, val)
        return set_func

    # helpers for Instrument::add_parameter.get_cmd
    def _gen_get_func_1par(self, fun, par1):
        def get_func():
            return fun(par1)
        return get_func

    def _gen_get_func_2par(self, fun, par1, par2):
        def get_func():
            return fun(par1, par2)
        return get_func

    ##########################################################################
    # CC-light compatibility support
    ##########################################################################

    # helper for parameter 'eqasm_program'
    def _eqasm_program(self, file_name: str) -> None:
        with open(file_name, 'r') as f:
            prog = f.read()
        self.sequence_program(prog)

    # helpers for parameter 'vsm_channel_delay{}'
    def _set_vsm_channel_delay(self, bit: int, cnt_in_2ns5_steps: int):
        cnt_in_39_ps_steps = cnt_in_2ns5_steps  # FIXME
        self.set_vsm_delay_rise(self._ccio_slot_driving_vsm, bit, cnt_in_39_ps_steps)

