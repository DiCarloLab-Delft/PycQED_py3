#!/usr/bin/python
# Based on: http://localhost:8888/notebooks/personal_folders/Miguel/qec_lut_demo.ipynb

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import logging
import sys
import inspect
import time
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

# FIXME: CCIO register offsets, subject to change
SYS_ST_QUES_DIOCAL_COND = 18
SYS_ST_OPER_DIO_RD_INDEX = 19
SYS_ST_OPER_DIO_MARGIN = 20




log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.debug('connecting to UHFQA')
uhfqa0 = ZI_UHFQC.UHFQC('uhfqa0', device=dev_uhfqa, nr_integration_channels=9)
if 0:  # restart, based on zishell_NH.py
    uhfqa0. seti('/' + dev_uhfqa + '/raw/system/restart', 1)
    raise RuntimeError("restarting UHF, observe LabOne")
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
    else:
        log.warning('setting hardcoded DIO delay on OHFQA')
        uhfqa0._set_dio_calibration_delay(5)  # FIXME: improves attainable latency?
        """
            scope CC latency measurements:
            
            delay   CC DIO read index   latency
            0       5                   188 ns
            4       5                   169 ns
            5       5                   162/169 ns
            6       5                   169 ns
            10      5                   ---
            
            toggled CLK source to INT, and back:
            0       -
            
            again:
            0       12                  197 ns
            4       12                  178 ns
            5       12                  178 ns
            6       12                  178 ns
            
            again:
            0       11                  199 ns
            5       11                  179 ns
            
            again:
            5       10                  179 ns
            
            again:
            5       9                   160 ns
        """

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
            uhfqa_prog = inspect.cleandoc("""
            // program: triggered upstream DIO calibration program
            const period = 18;          // 18*4.44 ns = 80 ns, NB: 40 ns is not attainable
            const n1 = 3;               // ~20 ns high time
            const n2 = period-n1-2-1;   // penalties: 2*setDIO, 1*loop
            waitDIOTrigger();
            while (1) {
                setDIO(0x000003FF);     // DV=0x0001, RSLT[8:0]=0x03FE.
                wait(n1);        
                setDIO(0x00000000);
                wait(n2);
            }
            """)
            dio_mask = 0x000003FF
            expected_sequence = []

            uhfqa0.dios_0_mode(uhfqa0.DIOS_0_MODE_AWG_SEQ) # FIXME: changes value set by load_default_settings()
            uhfqa0.configure_awg_from_string(0, uhfqa_prog)
            uhfqa0.seti('awgs/0/enable', 1)
            uhfqa0.start()  # FIXME?


            log.debug('sending UHFQA trigger program to CC')
            # FIXME: does not match with uhfqa_prog, which requires single trigger
            cc_prog = inspect.cleandoc("""
            # program: UHFQA trigger program
            .DEF    wait        9
            
            loop:   seq_out     0x03FF0000,1            # NB: TRIG=0x00010000, CW[8:0]=0x03FE0000
                    seq_out     0x0,$wait
                    jmp         @loop
            """)
            cc.assemble_and_start(cc_prog)


            log.debug('calibrating DIO protocol on CC')
            if 0:  # marker outputs
                if 1:
                    cc.debug_marker_in(cc_slot_uhfqa0, cc.UHFQA_DV)  # watch DV to check upstream period/frequency
                else:
                    cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)  # watch TRIG to check downstream period/frequency
            cc.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=cc_slot_uhfqa0)

            dio_rd_index = cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_OPER_DIO_RD_INDEX)
            log.info(f'DIO calibration condition = 0x{cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_QUES_DIOCAL_COND):x} (0=OK)')
            log.info(f'DIO read index = {dio_rd_index}')
            log.info(f'DIO margin = {cc.debug_get_ccio_reg(cc_slot_uhfqa0, SYS_ST_OPER_DIO_MARGIN)}')
            if dio_rd_index<0:
                cc.debug_marker_in(cc_slot_uhfqa0, cc.UHFQA_DV)  # watch DV to check upstream period/frequency
                raise RuntimeError("DIO calibration failed. FIXME: try setting UHF clock to internal")

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
        uhfqa0.awg_sequence_test_pattern(dio_out_vect=cw_array * 2 + 1)  # shift codeword, add Data Valid

        if 1:  # FIXME: remove duplicates of load_default_settings
            # Prepare AWG_Seq as driver of DIO and set DIO output direction
            uhfqa0.dios_0_mode(uhfqa0.DIOS_0_MODE_AWG_SEQ)  # FIXME: change from default

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
        
        .CODE
        # constants:
        .DEF    numIter     4
        .DEF    smAddr      S16 
        #.DEF    mux         0                       # SM[3:0] := I[3:0]
        #.DEF    pl          0                       # 4 times CW=1 conditional on SM[3:0]
        .DEF    mux         1                       # SM[7:0] := I[7:0]
        .DEF    pl          1                       # O[7:0] := SM[7:0]

        # timing constants:
        .DEF    uhfLatency  10                      # 10: best latency, but SEQ_IN_EMPTY and STV, 11: stable
        .DEF    smWait      2                       # plus another 2 makes 4 total: 80 ns
        #.DEF    smWait      3                       # FIXME: extra margin
        .DEF    iterWait    11                      # wait between iterations

        # instruction set constants:
        .DEF    byte        0                       # size parameter for seq_in_sm

                seq_bar     1                       # synchronize processors so markers make sense
                move        $numIter,R0
        loop:   
        [{uhf}] seq_out     0x00010000,$uhfLatency  # trigger UHFQA
        [{awg}] seq_wait    $uhfLatency             # balance UHF duration

        [{uhf}] seq_in_sm   $smAddr,$mux,$byte
        [{uhf}] seq_sw_sm   $smAddr                 # output to ARM SW for debugging
        [{awg}] seq_inv_sm  $smAddr,1               # invalidate 1 byte at target
        [{awg}] seq_wait    1                       # balance UHF duration

                seq_wait    $smWait                 # wait for data distribution

        [{awg}] seq_out_sm  $smAddr,$pl,1
        [{uhf}] seq_wait    1

                seq_wait    $iterWait
                loop        R0,@loop
                stop
        .END                            ; .CODE


        .DATAPATH
        .MUX 0  # 4 qubits from 1 UHF-QA: SM[3:0] := I[3:0]
            SM[0] := I[0]
            SM[1] := I[1]
            SM[2] := I[2]
            SM[3] := I[3]
        
        .PL 0	# 4 times CW=1 conditional on SM[3:0]
            O[31] := 1               	; HDAWG trigger
        
            O[0] := SM[0]         		; ch 1&2
            O[7] := SM[1]         		; ch 3&4
            O[16] := SM[2]         		; ch 5&6
            O[23] := SM[3]         		; ch 7&8
            # NB: state is cleared        

        .END                            ; .DATAPATH
        """)


        # watch UHF
        cc.debug_marker_in(cc_slot_uhfqa0, cc.UHFQA_DV)
        # cc.debug_marker_out(cc_slot_uhfqa0, cc.UHFQA_TRIG)

        # watch AWG
        # FIXME: we currently use a CC-CONN-DIO (non differenential), and no connected AWG. As a result, we can only
        # watch bits [31:16], and HDAWG_TRIG is overriden by TOGGLE_DS
        #cc.debug_marker_out(cc_slot_awg, cc.UHFQA_TRIG)  #
        #cc.debug_marker_out(cc_slot_awg, cc.HDAWG_TRIG)  #
        cc.debug_marker_out(cc_slot_awg, 23) # NB: always one with our data

        cc.stop()   # prevent tracing previous program
        for slot in [cc_slot_uhfqa0, cc_slot_awg]:
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_DEV_OUT)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_IN)
            cc.debug_set_ccio_trace_on(slot, cc.TRACE_CCIO_BP_OUT)
        cc.assemble_and_start(prog)

        time.sleep(1)
        #print(cc.debug_get_ccio_trace(cc_slot_awg))
        print(cc.debug_get_traces((1<<cc_slot_uhfqa0) + (1<<cc_slot_awg)), file=open('trace.vcd', 'w'))
        # FIXME: wait for CC to finish, then ask UHFQA how many patterns it generated and stop it


if 0:
    log.debug('test: reading CCIO registers')
    ccio = 2
    for i in range(23):
        log.debug(f"ccio[{ccio}]reg[{i}] = {cc.debug_get_ccio_reg(ccio, i)}")
    cc.check_errors()

log.debug('finished')
