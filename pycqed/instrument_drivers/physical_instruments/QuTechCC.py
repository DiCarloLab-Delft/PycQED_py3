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
                 transport: Transport,
                 num_ccio: int=9,
                 ccio_slot_driving_vsm: int=5) -> None:
        super().__init__(name, transport) # calls QuTechCC_core
        super(Instrument, QuTechCC).__init__(self, name) # calls Instrument

        # user constants
        self._num_ccio = num_ccio  # the number of CCIO modules used
        self._ccio_slot_driving_vsm = ccio_slot_driving_vsm  # the slot number of the CCIO driving the VSM (FIXME: supports one VSM only)

        # fixed constants
        self._q1reg_dio_delay = 63  # the register used in OpenQL generated programs to set DIO delay
        self._num_vsm_ch = 32  # the number of VSM channels used per connector
        self._CCIO_MAX_VSM_DELAY = 48

        self._add_parameters(self._num_ccio, self._num_vsm_ch)
        self._add_compatibility_parameters(self._num_ccio, self._num_vsm_ch)


    ##########################################################################
    # QCoDeS parameter definitions
    ##########################################################################

    def _add_parameters(self, num_ccio: int, num_vsm_ch: int) -> None:
        """
        add CC native parameters
        """

        for vsm_ch in range(0, num_vsm_ch):  # NB: VSM channel starts from 0 on CC-light/QCC
            self.add_parameter(
                'vsm_rise_delay{}'.format(vsm_ch),
                label='VSM rise {} delay'.format(vsm_ch),
                docstring='Sets/gets the rise delay for VSM channel {}'.format(vsm_ch),
                unit='833 ps',
                vals=vals.PermissiveInts(0, self._CCIO_MAX_VSM_DELAY),
                set_cmd=_gen_set_func_1par(self._set_vsm_rise_delay, vsm_ch),
                get_cmd=_gen_get_func_1par(self._get_vsm_rise_delay, vsm_ch)
            )
            self.add_parameter(
                'vsm_fall_delay{}'.format(vsm_ch),
                label='VSM fall {} delay'.format(vsm_ch),
                docstring='Sets/gets the fall delay for VSM channel {}'.format(vsm_ch),
                unit='833 ps',
                vals=vals.PermissiveInts(0, self._CCIO_MAX_VSM_DELAY),
                set_cmd=_gen_set_func_1par(self._set_vsm_fall_delay, vsm_ch),
                get_cmd=_gen_get_func_1par(self._get_vsm_fall_delay, vsm_ch)
            )

    def _add_compatibility_parameters(self, num_ccio: int, num_vsm_ch: int) -> None:
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
        # NB: DIO starts from 1 on CC-light/QCC, but we use CCIO number starting from 0
        for ccio in range(0, num_ccio):
            if ccio == self._ccio_slot_driving_vsm:  # skip VSM
                continue
            self.add_parameter(
                'dio{}_out_delay'.format(ccio),
                label='Output Delay of DIO{}'.format(ccio),
                docstring='This parameter determines the extra output delay introduced for the DIO{} channel (i.e. CCIO slot number)'.format(ccio),
                unit='20 ns',
                vals=vals.PermissiveInts(0, 31), # FIXME: CC limit is 2^32-1
                set_cmd=_gen_set_func_1par(self._set_dio_delay, ccio)
#                get_cmd=cmd + '?',
                )

        # support for 'vsm_channel_delay{}' for CCL_Transmon.py::_set_mw_vsm_delay(), also see calibrate_mw_vsm_delay()
        # NB: CC supports 1/1200 MHz ~= 833 ps resolution
        # NB: CC supports setting trailing edge delay separately
        # NB: on CCL, index is qubit, not channel
        for vsm_ch in range(0, num_vsm_ch):  # NB: VSM channel starts from 0 on CC-light/QCC
            self.add_parameter(
                'vsm_channel_delay{}'.format(vsm_ch),
                label='VSM Channel {} delay'.format(vsm_ch),
                docstring='Sets/gets the delay for VSM channel {}'.format(vsm_ch),
                unit='2.5 ns',
                vals=vals.PermissiveInts(0, 127),
                set_cmd=_gen_set_func_1par(self._set_vsm_channel_delay, vsm_ch)
#                get_cmd=_gen_get_func_1par(self._get_vsm_channel_delay, vsm_ch),
                )

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

    ##########################################################################
    # parameter support
    ##########################################################################

    # helper for parameter 'vsm_rise_delay{}'
    def _set_vsm_rise_delay(self, bit: int, cnt_in_833_ps_steps: int) -> None:
        self.set_vsm_delay_rise(self._ccio_slot_driving_vsm, bit, cnt_in_833_ps_steps)

    def _get_vsm_rise_delay(self, bit: int) -> int:
        return self.set_vsm_delay_rise(self._ccio_slot_driving_vsm, bit)

    # helper for parameter 'vsm_fall_delay{}'
    def _set_vsm_fall_delay(self, bit: int, cnt_in_833_ps_steps: int) -> None:
        self.set_vsm_delay_fall(self._ccio_slot_driving_vsm, bit, cnt_in_833_ps_steps)

    def _get_vsm_fall_delay(self, bit: int) -> int:
        return self.set_vsm_delay_fall(self._ccio_slot_driving_vsm, bit)

    ##########################################################################
    # CC-light compatibility support
    ##########################################################################

    # helper for parameter 'eqasm_program'
    def _eqasm_program(self, file_name: str) -> None:
        with open(file_name, 'r') as f:
            prog = f.read()
        self.sequence_program(prog)

    # helper for parameter 'vsm_channel_delay{}'
    # NB: CC-light range max = 127*2.5 ns = 317.5 ns, our fine delay range is 48/1200 MHz = 40 ns, so we must also shift program
    def _set_vsm_channel_delay(self, bit: int, cnt_in_2ns5_steps: int) -> None:
        delay_ns = cnt_in_2ns5_steps * 2.5
        cnt_in_20ns_steps = int(delay_ns // 20)
        remain_ns = delay_ns - cnt_in_20ns_steps * 20
        cnt_in_833_ps_steps = round(remain_ns*1.2)  # NB: actual step size is 1/1200 MHz
        self.set_vsm_delay_rise(self._ccio_slot_driving_vsm, bit, cnt_in_833_ps_steps)
        self._set_dio_delay(self._ccio_slot_driving_vsm, cnt_in_20ns_steps)

    # helper for parameter 'dio{}_out_delay'
    def _set_dio_delay(self, ccio: int, cnt_in_20ns_steps: int) -> None:
        self.stop()
        self.set_q1_reg(ccio, self._q1reg_dio_delay, cnt_in_20ns_steps)
        self.start()


# helpers for Instrument::add_parameter.set_cmd
def _gen_set_func_1par(fun, par1):
    def set_func(val):
        return fun(par1, val)
    return set_func


def _gen_set_func_2par(fun, par1, par2):
    def set_func(val):
        return fun(par1, par2, val)
    return set_func


# helpers for Instrument::add_parameter.get_cmd
def _gen_get_func_1par(fun, par1):
    def get_func():
        return fun(par1)
    return get_func


def _gen_get_func_2par(fun, par1, par2):
    def get_func():
        return fun(par1, par2)
    return get_func

