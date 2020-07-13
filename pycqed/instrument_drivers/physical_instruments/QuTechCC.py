"""
    File:               QuTechCC.py
    Author:             Wouter Vlothuizen, QuTech
    Purpose:            Python control of Qutech Central Controller: adds application dependent stuff to QuTechCC_core
    Notes:              use QuTechCC_core to talk to instrument, do not add knowledge of SCPI syntax here
    Usage:
    Bugs:
    - _ccio_slots_driving_vsm not handled correctly
    - dio{}_out_delay not gettable

"""

import logging
from typing import List
from .QuTechCC_core import QuTechCC_core
from .Transport import Transport

from qcodes.utils import validators as vals
from qcodes import Instrument

log = logging.getLogger(__name__)

_cc_prog_dio_cal_microwave = """
# staircase program for HDAWG microwave mode, CW_1 31->1, CW_2 1->31
.DEF        cw_31_01        0x80003E01      # TRIG=1(0x80000000), CW_1=31(0x00003E00), CW_2=1(0x00000001)
.DEF        incr            0xFFFFFE01      # CW_1--, CW_2++
.DEF        duration        4
repeat:     move            $cw_31_01,R0
            move            31,R1           # loop counter
inner:      seq_out         R0,$duration
            add             R0,$incr,R0
            loop            R1,@inner
            jmp             @repeat
"""



class QuTechCC(QuTechCC_core, Instrument):
    def __init__(self,
                 name: str,
                 transport: Transport,
                 num_ccio: int=9,
                 ccio_slots_driving_vsm: List[int] = None  # NB: default can not be '[]' because that is a mutable default argument
                 ) -> None:
        super().__init__(name, transport) # calls QuTechCC_core
        Instrument.__init__(self, name) # calls Instrument

        # user constants
        self._num_ccio = num_ccio  # the number of CCIO modules used
        if ccio_slots_driving_vsm is None:
            self._ccio_slots_driving_vsm = []
        else:
            self._ccio_slots_driving_vsm = ccio_slots_driving_vsm  # the slot numbers of the CCIO driving the VSM

        # fixed constants
        self._Q1REG_DIO_DELAY = 63  # the register used in OpenQL generated programs to set DIO delay
        self._NUM_VSM_CH = 32  # the number of VSM channels used per CCIO connector
        self._CCIO_MAX_VSM_DELAY = 48

        self._add_parameters(self._num_ccio)
        self._add_compatibility_parameters(self._num_ccio)


    ##########################################################################
    # QCoDeS parameter definitions
    ##########################################################################

    def _add_parameters(self, num_ccio: int) -> None:
        """
        add CC native parameters
        """

        for vsm_ch in range(0, self._NUM_VSM_CH):  # NB: VSM channel starts from 0 on CC-light/QCC
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

    def _add_compatibility_parameters(self, num_ccio: int) -> None:
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
            if 1:
                # skip DIO delay setting for slots driving VSM. Note that vsm_channel_delay also sets DIO delay
                if ccio in self._ccio_slots_driving_vsm:  # skip VSM
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
        # NB: supports single VSM only, use native parameter for >1 VSM
        for vsm_ch in range(0, self._NUM_VSM_CH):  # NB: VSM channel starts from 0 on CC-light/QCC
            self.add_parameter(
                'vsm_channel_delay{}'.format(vsm_ch),
                label='VSM Channel {} delay'.format(vsm_ch),
                docstring='Sets/gets the delay for VSM channel {}'.format(vsm_ch),
                unit='2.5 ns',
                vals=vals.PermissiveInts(0, 127),
                set_cmd=_gen_set_func_1par(self._set_vsm_channel_delay, vsm_ch)
#                get_cmd=_gen_get_func_1par(self._get_vsm_channel_delay, vsm_ch),
                )

        # FIXME: num_append_pts not implemented, use vsm_fall_delay

    ##########################################################################
    # parameter support
    ##########################################################################

    # helper for parameter 'vsm_rise_delay{}'
    # FIXME: hardcoded to first VSM
    def _set_vsm_rise_delay(self, bit: int, cnt_in_833_ps_steps: int) -> None:
        self.set_vsm_delay_rise(self._ccio_slots_driving_vsm[0], bit, cnt_in_833_ps_steps)

    def _get_vsm_rise_delay(self, bit: int) -> int:
        return self.get_vsm_delay_rise(self._ccio_slots_driving_vsm[0], bit)

    # helper for parameter 'vsm_fall_delay{}'
    def _set_vsm_fall_delay(self, bit: int, cnt_in_833_ps_steps: int) -> None:
        self.set_vsm_delay_fall(self._ccio_slots_driving_vsm[0], bit, cnt_in_833_ps_steps)

    def _get_vsm_fall_delay(self, bit: int) -> int:
        return self.get_vsm_delay_fall(self._ccio_slots_driving_vsm[0], bit)

    ##########################################################################
    # CC-light compatibility support
    ##########################################################################

    # helper for parameter 'eqasm_program'
    def _eqasm_program(self, file_name: str) -> None:
        with open(file_name, 'r') as f:
            prog = f.read()
        self.sequence_program_assemble(prog)

    # helper for parameter 'vsm_channel_delay{}'
    # NB: CC-light range max = 127*2.5 ns = 317.5 ns, our fine delay range is 48/1200 MHz = 40 ns, so we must also shift program
    # NB: supports one VSM only, no intend to upgrade
    def _set_vsm_channel_delay(self, bit: int, cnt_in_2ns5_steps: int) -> None:
        delay_ns = cnt_in_2ns5_steps * 2.5
        cnt_in_20ns_steps = int(delay_ns // 20)
        remain_ns = delay_ns - cnt_in_20ns_steps * 20
        cnt_in_833_ps_steps = round(remain_ns*1.2)  # NB: actual step size is 1/1200 MHz
        self.set_vsm_delay_rise(self._ccio_slots_driving_vsm[0], bit, cnt_in_833_ps_steps)
        self._set_dio_delay(self._ccio_slots_driving_vsm[0], cnt_in_20ns_steps)

    # helper for parameter 'dio{}_out_delay'
    def _set_dio_delay(self, ccio: int, cnt_in_20ns_steps: int) -> None:
        self.stop()
        self.set_q1_reg(ccio, self._Q1REG_DIO_DELAY, cnt_in_20ns_steps)
        self.start()

    ##########################################################################
    # DIO calibration support for connected instruments
    ##########################################################################

    def output_dio_calibration_data(self, dio_mode, port=None):
        if dio_mode == "microwave":
            cc_prog = _cc_prog_dio_cal_microwave
        elif dio_mode == "new_microwave":
            # FIXME
            pass
        elif dio_mode == "new_novsm_microwave":
            # FIXME
            pass
        elif dio_mode == "flux":
            # FIXME
            pass
        else:
            raise ValueError("unsupported DIO mode")

        log.debug(f"uploading DIO calibration program for mode '{dio_mode}' to CC")
        self.sequence_program_assemble(cc_prog)
        log.debug("printing CC errors")
        err_cnt = self.get_system_error_count()
        if err_cnt > 0:
            log.warning('CC status after upload')
        for i in range(err_cnt):
            print(self.get_error())
        self.start()
        log.debug('starting CC')


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

