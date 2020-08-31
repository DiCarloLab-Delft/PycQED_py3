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
cc_slot_awg = 3

# FIXME: CCIO register offsets
SYS_ST_QUES_DIOCAL_COND = 18
SYS_ST_OPER_DIO_RD_INDEX = 19
SYS_ST_OPER_DIO_MARGIN = 20




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
        log.warning("Forcing DIO delay to fixed value (awaiting merge of optimal choice in driver)")
        uhfqa0.setd('raw/dios/0/delay', 5)

    if 1:
        log.debug('calibration DIO: UHFQA to CC')
        if 0:
            DIO.calibrate(
                sender=uhfqa0,
                receiver=cc,
                receiver_port=cc_slot_uhfqa0
            )
        else: # inspired by calibrate, but with CC program to trigger UHFQA
            log.debug('sending triggered upstream DIO calibration program to UHFQA')
            program = inspect.cleandoc("""
            // program: triggered upstream DIO calibration program
            var A = 0x000003FF; // DV=0x0001, RSLT[8:0]=0x03FE
            var B = 0x00000000;
        
            while (1) {
                waitDIOTrigger();
                setDIO(A);
                //wait(2);    // documentation: 4.44 ns periods, measured high time 14.8...15.6 ns, depending on trigger rate
                wait(3);        // ~18 ns high time
                setDIO(B);
            }
            """)
            dio_mask = 0x000003FF
            expected_sequence = []

            uhfqa0.dios_0_mode(uhfqa0.DIOS_0_MODE_AWG_SEQ) # FIXME: changes value set by load_default_settings()
            uhfqa0.configure_awg_from_string(0, program)
            uhfqa0.seti('awgs/0/enable', 1)
            uhfqa0.start()  # FIXME?


            log.debug('sending UHFQA trigger program to CC')
            prog = inspect.cleandoc("""
            # program: UHFQA trigger program
            .DEF    duration    1
            .DEF    wait        9
            
            loop:   seq_out     0x03FF0000,$duration      # NB: TRIG=0x00010000, CW[8:0]=0x03FE0000
                    seq_out     0x0,$wait
                    jmp         @loop
            """)
            cc.assemble_and_start(prog)


            log.debug('calibrating DIO protocol on CC')
            if 0:
                if 1:
                    cc.debug_marker_in(cc_slot_uhfqa0, cc.UHFQA_DV)  # watch DV to check upstream period/frequency
                else:
                    cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)  # watch TRIG to check downstream period/frequency
            cc.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=cc_slot_uhfqa0)
            log.info(f'DIO calibration condition = 0x{cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_QUES_DIOCAL_COND):x} (0=OK)')
            log.info(f'DIO read index = {cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_OPER_DIO_RD_INDEX)}')
            log.info(f'DIO margin = {cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_OPER_DIO_MARGIN)}')

            if 1:  # disable to allow scope measurements
                cc.stop()
                uhfqa0.stop()
                cc.get_operation_complete()  # ensure all commands have finished




if 1:  # test of Distributed Shared Memory
    if 1:
        log.debug('run UHFQA codeword generator')

        # build a programs that outputs the sequence once, each entry triggered by CC
        #cw_list = [3, 2, 1, 0]
        cw_list = [7, 6, 5, 4]
        cw_array = np.array(cw_list, dtype=int).flatten()
        uhfqa0.awg_sequence_acquisition_and_DIO_RED_test(
            dio_out_vect=cw_array * 2 + 1,  # shift codeword, add Data Valid
            acquisition_delay=20e-9) # FIXME: misnomer, overridden below by USER_REG_WAIT_DLY

        uhfqa0.set(f"awgs_0_userregs_{uhfqa0.USER_REG_WAIT_DLY}", 2)    # high time ~35 ns, results in spurious trigger on CC
        #uhfqa0.set(f"awgs_0_userregs_{uhfqa0.USER_REG_WAIT_DLY}", 1)    # high time ~30 ns, results in spurious trigger on CC
        #uhfqa0.set(f"awgs_0_userregs_{uhfqa0.USER_REG_WAIT_DLY}", 0)  # high time ~25 ns, gives SEQ_IN_EMPTY on CC

        if 1:  # FIXME: remove duplicates of load_default_settings
            # Prepare AWG_Seq as driver of DIO and set DIO output direction
            uhfqa0.dios_0_mode(uhfqa0.DIOS_0_MODE_AWG_SEQ)  # FIXME: change from default

    #        uhfqa0.dios_0_drive(3)

            # Determine trigger and strobe bits from DIO
    #        uhfqa0.awgs_0_dio_valid_index(16)
    #        uhfqa0.awgs_0_dio_valid_polarity(0)
    #?        uhfqa0.awgs_0_dio_strobe_index(16)
    #?       uhfqa0.awgs_0_dio_strobe_slope(1)

            # Initialize UHF for consecutive triggering and enable it
            uhfqa0.awgs_0_single(0)
            uhfqa0.awgs_0_enable(1)  # ?
        uhfqa0.start()


    if 1:
        log.debug('upload CC feedback test program')

        # shorthand slot definitions for code generation
        uhf = cc_slot_uhfqa0
        awg = cc_slot_awg
        prog = inspect.cleandoc(f"""
        # program:  CC feedback test program
        .DEF    numIter     4
        .DEF    duration    100 #9
        .DEF    wait        100
        .DEF    smAddr      S16
        .DEF    lut         0
        
                
                seq_bar     1                       # synchronize processors so markers make sense
                move        $numIter,R0
        loop:   seq_out     0x00010000,$duration    # trigger UHFQA
        [{uhf}] seq_in_sm   $smAddr,$lut,0
        [{uhf}] seq_sw_sm   $smAddr
        [{awg}] seq_out     0,2                     # balance duration with UHF
                seq_out     0x0,$wait
                loop        R0,@loop
                stop
        """)

        if 1:
            cc.debug_marker_out(cc_slot_awg, cc.UHFQA_TRIG)  # watch TRIG, so we can see TRIG to DV latency (requires seq_bar)
            cc.debug_marker_in(cc_slot_uhfqa0, cc.UHFQA_DV)  # watch DV to check upstream period/frequency
        else:
            cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)  # watch TRIG to check downstream period/frequency

        cc.stop()   # prevent tracing previous program
        for slot in [cc_slot_uhfqa0,3]:
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_OUT)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_OUT)
        cc.assemble_and_start(prog)

        # FIXME: wait for CC to finish, then ask UHFQA how many patterns it generated


if 0:
    log.debug('test: reading CCIO registers')
    ccio = 2
    for i in range(23):
        log.debug(f"ccio[{ccio}]reg[{i}] = {cc.debug_get_ccio_reg(ccio, i)}")
    cc.check_errors()

log.debug('finished')
