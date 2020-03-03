#!/usr/bin/python
# Based on: http://localhost:8888/notebooks/personal_folders/Miguel/qec_lut_demo.ipynb

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import logging
import sys
import inspect
import numpy as np

from pycqed.instrument_drivers.library.Transport import IPTransport
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC

# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip_cc = '192.168.0.241'
dev_uhfqa = 'dev2271'
cc_slot_uhfqa0 = 2

# FIXME: CCIO register offsets
SYS_ST_QUES_DIOCAL_COND = 18
SYS_ST_OPER_DIO_RD_INDEX = 19




log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.debug('connecting to UHFQA')
uhfqa0 = ZI_UHFQC.UHFQC('uhfqa0', device=dev_uhfqa, nr_integration_channels=9)
uhfqa0.load_default_settings(upload_sequence=False)

log.debug('connecting to CC')
cc = CC('cc', IPTransport(ip_cc))
cc.init()
log.info(cc.get_identity())


if 1:   # DIO calibration
    if 1:
        log.debug('calibration DIO: CC to UHFQA')
        DIO.calibrate(
        sender=cc,
        receiver=uhfqa0,
        receiver_port=cc_slot_uhfqa0,
        sender_dio_mode='uhfqa'
    )

    if 1:
        log.debug('calibration DIO: UHFQA to CC')
        if 0:
            DIO.calibrate(
                sender=uhfqa0,
                receiver=cc,
                receiver_port=cc_slot_uhfqa0
            )
        else: # inspired by calibrate, but with CC program to trigger UHFQA
            log.debug('sending triggered DIO test program to UHFQA')
            program = inspect.cleandoc("""
            var A = 0x00000CFF; // DV=0x0001, RSLT[8:0]=0x0CF7
            var B = 0x00000000;
        
            while (1) {
                waitDIOTrigger();
                setDIO(A);
                wait(2);
                setDIO(B);
            }
            """)
            uhfqa0.configure_awg_from_string(0, program)
            uhfqa0.seti('awgs/0/enable', 1)
            uhfqa0.start()  # FIXME?


            log.debug('sending UHFQA trigger program to CC')
            prog = inspect.cleandoc("""
            # UHFQA trigger program
            .DEF    duration    9
            .DEF    wait        100
            loop:   seq_out     0x00010000,$duration      # trigger UHFQA
                    seq_out     0x0,$wait
                    jmp         @loop
            """)
            cc.assemble_and_start(prog)
            dio_mask = 0x00000CFF
            expected_sequence = []

            log.debug('calibrating DIO protocol on CC')
            cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_DV)  # watch DV to check period/frequency
            #cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)  # watch TRIG to check period/frequency
            cc.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=cc_slot_uhfqa0)
            log.info(f'DIO calibration condition = {cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_QUES_DIOCAL_COND)} (0=OK)')
            log.info(f'DIO read index = {cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_OPER_DIO_RD_INDEX)}')
            if 0:  # allow scope measurements
                cc.stop()
                uhfqa0.stop()
                cc.get_operation_complete()  # ensure all command have finished




if 0:  # test of Distributed Shared Memory
    if 1:
        log.debug('run UHFQA codeword generator')

        # build a programs that outputs the sequence once, each entry triggered by CC
        #cw_list = [3, 2, 1, 0]
        cw_list = [7, 6, 5, 4]
        cw_array = np.array(cw_list, dtype=int).flatten()
        uhfqa0.awg_sequence_acquisition_and_DIO_RED_test(
            Iwaves=[np.ones(8), np.ones(8)],
            Qwaves=[np.ones(8), np.ones(8)],
            cases=[2, 5],
            dio_out_vect=cw_array * 2 + 1,  # shift codeword, add Data Valid
            acquisition_delay=20e-9)

        if 0:
            rolut0 = UHFQC_RO_LutMan('rolut0', num_res=5)
            rolut0.AWG(uhfqa0.name)

        if 1:  # FIXME: remove duplicates of load_default_settings
            # Prepare AWG_Seq as driver of DIO and set DIO output direction
            uhfqa0.dios_0_mode(uhfqa0.DIOS_0_MODE_AWG_SEQ)  # FIXME: change from default
    #        uhfqa0.dios_0_drive(3)

            # Determine trigger and strobe bits from DIO
    #        uhfqa0.awgs_0_dio_valid_index(16)
    #        uhfqa0.awgs_0_dio_valid_polarity(0)
    #?        uhfqa0.awgs_0_dio_strobe_index(16)
    #?       uhfqa0.awgs_0_dio_strobe_slope(1)
            uhfqa0.awgs_0_userregs_2(2)

            # Initialize UHF for consecutive triggering and enable it
            uhfqa0.awgs_0_single(0)
            uhfqa0.awgs_0_enable(1)  # ?
        uhfqa0.start()


    if 1:
        log.debug('upload CC feedback test program')

        prog = inspect.cleandoc("""
        .DEF    duration    9
        .DEF    wait        100
        .DEF    smAddr      S16
        .DEF    lut         0
        .DEF    numIter     100
        # slot 2: UHFQA
        [2]         move        $numIter,R0
        [2]loop:    seq_out     0x00010000,$duration      # UHFQA measurement
        [2]         seq_in_sm   $smAddr,$lut,0
        [2]         seq_sw_sm   $smAddr
        [2]         seq_out     0x0,$wait
        [2]         loop        R0,@loop
        [2]         stop
        # slot 3: observe
        #[1]loop:    jmp         @loop
        #[2]loop:    jmp         @loop
        [3]         stop
        #[4]loop:    jmp         @loop
        """)

        prog_new = inspect.cleandoc("""
        .DEF    duration    100 #9
        .DEF    wait        100
        .DEF    smAddr      S16
        .DEF    lut         0
        .DEF    numIter     4
                 move        $numIter,R0
        loop:    seq_out     0x00010000,$duration      # UHFQA measurement
                 seq_in_sm   $smAddr,$lut,0
                 seq_sw_sm   $smAddr
                 seq_out     0x0,$wait
                 loop        R0,@loop
                 stop
        """)

        cc.stop()   # prvent tracing previous program
        cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)
        for slot in [cc_slot_uhfqa0,3]:
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_OUT)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_OUT)
        cc.assemble_and_start(prog_new)

        # FIXME: wait for CC to finish, then ask UHFQA how many patterns it generated


if 0:
    log.debug('test: reading CCIO registers')
    ccio = 2
    for i in range(23):
        log.debug(f"ccio[{ccio}]reg[{i}] = {cc.debug_get_ccio_reg(ccio, i)}")
    cc.check_errors()

log.debug('finished')
