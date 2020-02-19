"""
    File:       QuTechCC.py
    Author:     Wouter Vlothuizen, QuTech
    Purpose:    QCoDeS instrument driver for Qutech Central Controller: adds application dependent stuff to QuTechCC_core
    Notes:      use QuTechCC_core to talk to instrument, do not add knowledge of SCPI syntax here
    Usage:
    Bugs:
    - _ccio_slots_driving_vsm not handled correctly
    - dio{}_out_delay not gettable

"""

import logging
import os
import numpy as np
from typing import Tuple,List

from .QuTechCC_core import QuTechCC_core
from .Transport import Transport
from pycqed.instrument_drivers.meta_instrument.DIOCalibration import DIOCalibration
import pycqed

from qcodes.utils import validators as vals
from qcodes import Instrument

log = logging.getLogger(__name__)


class QuTechCC(QuTechCC_core, Instrument, DIOCalibration):
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
    # overrides for DIOCalibration interface
    # FIXME: move to QuTechCC_core? or CC_DIOCAL
    ##########################################################################

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        self.calibrate_dio(port)

    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        # default return values
        expected_sequence = []
        dio_mask = 0x00000000

        if dio_mode == "microwave-FIXME":
            # based on local _cc_prog_dio_cal_microwave
            # FIXME: program does not seem to match mode/dio_mask
            cc_prog = """
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

            dio_mask = 0x7fff0000  # TRIG=0x8000000, CW=0x00FF00FF ('new' QWG compatible microwave mode)


        elif dio_mode == "microwave":
            # based on ZI_HDAWG8.py::_prepare_CC_dio_calibration_hdawg and examples/CC_examples/old_hdawg_calibration.vq1asm
            cc_prog = """
            # CC_BACKEND_VERSION 0.2.4
            # OPENQL_VERSION 0.8.0
            # Program: 'CW_RO_sequence'
            # Note:    generated by OpenQL Central Controller backend
            #
            # synchronous start and latency compensation
                            add             R63,1,R0                # R63 externally set by user, prevent 0 value which would wrap counter
                            seq_bar         20                      # synchronization
            syncLoop:       seq_out         0x00000000,1            # 20 ns delay
                            loop            R0,@syncLoop            #
            mainLoop:                                               #
            ### Kernel: 'k_main'
            ## Bundle 0: start_cycle=1, duration_in_cycles=300:
             # READOUT: measure(q0)
              # slot=1, instrument='ro_1', group=0': signal='[dummy]'
             # last bundle of kernel, will pad outputs to match durations
              # slot=1, instrument='ro_1': lastStartCycle=0, start_cycle=1, slotDurationInCycles=300
            [1]             seq_out         0x00000000,301          # cycle 0-301: padding on 'ro_2'
            [2]             seq_out         0x00000000,301          # cycle 0-301: padding on 'mw_0'
            # comment
            [3]             seq_out         0x00000000,2            # 00000000000000000000000000000000
            [3]             seq_out         0x80008000,2            # 10000000000000001000000000000000
            [3]             seq_out         0x80018001,2            # 10000000000000011000000000000001
            [3]             seq_out         0x80028002,2            # 10000000000000101000000000000010
            [3]             seq_out         0x80038003,2            # 10000000000000111000000000000011
            [3]             seq_out         0x80048004,2            # 10000000000001001000000000000100
            [3]             seq_out         0x80058005,2            # 10000000000001011000000000000101
            [3]             seq_out         0x80068006,2            # 10000000000001101000000000000110
            [3]             seq_out         0x80078007,2            # 10000000000001111000000000000111
            [3]             seq_out         0x80088008,2            # 10000000000010001000000000001000
            [3]             seq_out         0x80098009,2            # 10000000000010011000000000001001
            [3]             seq_out         0x800A800A,2            # 10000000000010101000000000001010
            [3]             seq_out         0x800B800B,2            # 10000000000010111000000000001011
            [3]             seq_out         0x800C800C,2            # 10000000000011001000000000001100
            [3]             seq_out         0x800D800D,2            # 10000000000011011000000000001101
            [3]             seq_out         0x800E800E,2            # 10000000000011101000000000001110
            [3]             seq_out         0x800F800F,2            # 10000000000011111000000000001111
            [3]             seq_out         0x80108010,2            # 10000000000100001000000000010000
            [3]             seq_out         0x80118011,2            # 10000000000100011000000000010001
            [3]             seq_out         0x80128012,2            # 10000000000100101000000000010010
            [3]             seq_out         0x80138013,2            # 10000000000100111000000000010011
            [3]             seq_out         0x80148014,2            # 10000000000101001000000000010100
            [3]             seq_out         0x80158015,2            # 10000000000101011000000000010101
            [3]             seq_out         0x80168016,2            # 10000000000101101000000000010110
            [3]             seq_out         0x80178017,2            # 10000000000101111000000000010111
            [3]             seq_out         0x80188018,2            # 10000000000110001000000000011000
            [3]             seq_out         0x80198019,2            # 10000000000110011000000000011001
            [3]             seq_out         0x801A801A,2            # 10000000000110101000000000011010
            [3]             seq_out         0x801B801B,2            # 10000000000110111000000000011011
            [3]             seq_out         0x801C801C,2            # 10000000000111001000000000011100
            [3]             seq_out         0x801D801D,2            # 10000000000111011000000000011101
            [3]             seq_out         0x801E801E,2            # 10000000000111101000000000011110
            [3]             seq_out         0x801F801F,2            # 10000000000111111000000000011111
            [3]             seq_out         0x00000000,18           # 00000000000000000000000000000000
            # digIn=2
            [5]             seq_out         0x00000000,301          # cycle 0-301: padding on 'mw_1'
            [6]             seq_out         0x00000000,301          # cycle 0-301: padding on 'flux_0'
            
                            jmp             @mainLoop               # loop indefinitely

            """
            sequence_length = 32
            staircase_sequence = range(0, sequence_length)
            expected_sequence = [(0, list(staircase_sequence)),
                                 (1, list(staircase_sequence)),
                                 (2, list(staircase_sequence)),
                                 (3, list(staircase_sequence))]
            dio_mask = 0x7fff0000  # TRIG=0x8000000, CW=0x00FF00FF ('new' QWG compatible microwave mode)


        elif dio_mode == "new_novsm_microwave":
            raise NotImplementedError  # FIXME


        elif dio_mode == "flux":
            # based on ZI_HDAWG8.py::_prepare_CC_dio_calibration_hdawg and examples/CC_examples/flux_calibration.vq1asm
            #
            cc_prog = """
            # CC_BACKEND_VERSION 0.2.4
            # OPENQL_VERSION 0.8.0
            # Program: 'CW_RO_sequence'
            # Note:    generated by OpenQL Central Controller backend
            #
            # synchronous start and latency compensation
                            add             R63,1,R0                # R63 externally set by user, prevent 0 value which would wrap counter
                            seq_bar         20                      # synchronization
            syncLoop:       seq_out         0x00000000,1            # 20 ns delay
                            loop            R0,@syncLoop            #
            mainLoop:                                               #
            ### Kernel: 'k_main'
            ## Bundle 0: start_cycle=1, duration_in_cycles=300:
             # READOUT: measure(q0)
              # slot=1, instrument='ro_1', group=0': signal='[dummy]'
             # last bundle of kernel, will pad outputs to match durations
              # slot=1, instrument='ro_1': lastStartCycle=0, start_cycle=1, slotDurationInCycles=300
            [1]             seq_out         0x00000000,301          # cycle 0-301: padding on 'ro_2'
            [2]             seq_out         0x00000000,301          # cycle 0-301: padding on 'mw_0'
            [3]             seq_out         0x00000000,301          # cycle 0-301: padding on 'flux_0'
            [4]             seq_out         0x00000000,301          # cycle 0-301: padding on 'mw_1'
            # sequence
            [6]             seq_out         0x00000000,20           # 00000000000000000000000000000000
            [6]             seq_out         0x82498249,2            # 10000010010010011000001001001001
            [6]             seq_out         0x84928492,2            # 10000100100100101000010010010010
            [6]             seq_out         0x86DB86DB,2            # 10000110110110111000011011011011
            [6]             seq_out         0x89248924,2            # 10001001001001001000100100100100
            [6]             seq_out         0x8B6D8B6D,2            # 10001011011011011000101101101101
            [6]             seq_out         0x8DB68DB6,2            # 10001101101101101000110110110110
            [6]             seq_out         0x8FFF8FFF,2            # 10001111111111111000111111111111
            
                            jmp             @mainLoop               # loop indefinitely
            """

            sequence_length = 8
            staircase_sequence = np.arange(1, sequence_length)
            # expected sequence should be ([9, 18, 27, 36, 45, 54, 63])
            expected_sequence = [(0, list(staircase_sequence + (staircase_sequence << 3))),
                                 (1, list(staircase_sequence + (staircase_sequence << 3))),
                                 (2, list(staircase_sequence + (staircase_sequence << 3))),
                                 (3, list(staircase_sequence+ (staircase_sequence << 3)))]
            dio_mask = 0x8FFF8FFF


        elif dio_mode == "uhfqa":  # FIXME: no official mode yet
            # Based on UHFQuantumController.py::_prepare_CC_dio_calibration_uhfqa and  and examples/CC_examples/uhfqc_calibration.vq1asm
            # FIXME: not generic
            #
            cc_prog = """
            # CC_BACKEND_VERSION 0.2.4
            # OPENQL_VERSION 0.8.0
            # Program: 'CW_RO_sequence'
            # Note:    generated by OpenQL Central Controller backend
            #
            # synchronous start and latency compensation
                            add             R63,1,R0                # R63 externally set by user, prevent 0 value which would wrap counter
                            seq_bar         20                      # synchronization
            syncLoop:       seq_out         0x00000000,1            # 20 ns delay
                            loop            R0,@syncLoop            #
            mainLoop:                                               #
            ### Kernel: 'k_main'
            ## Bundle 0: start_cycle=1, duration_in_cycles=300:
             # READOUT: measure(q0)
              # slot=1, instrument='ro_1', group=0': signal='[dummy]'
             # last bundle of kernel, will pad outputs to match durations
              # slot=1, instrument='ro_1': lastStartCycle=0, start_cycle=1, slotDurationInCycles=300
            [1]             seq_out         0x00000000,10
            [1]             seq_out         0x03FF0000,1
            [2]             seq_out         0x00000000,10
            [2]             seq_out         0x03FF0000,1
            [3]             seq_out         0x00000000,10
            [3]             seq_out         0x03FF0000,1
            [4]             seq_out         0x00000000,11
            [5]             seq_out         0x00000000,11
            [6]             seq_out         0x00000000,11
                            jmp             @mainLoop
            """

            dio_mask = 0x03ff0000  # DV=0x00010000, RSLT[8:0]=0x03FE0000
        else:
            raise ValueError("unsupported DIO mode")

        log.debug(f"uploading DIO calibration program for mode '{dio_mode}' to CC")
        self.assemble(cc_prog)
        log.debug("printing CC errors")
        self.check_errors()
        log.debug('starting CC')
        self.start()

        return dio_mask,expected_sequence

##########################################################################
# helpers
##########################################################################

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
