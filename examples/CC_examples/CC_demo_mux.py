#!/usr/bin/python
# Based on: http://localhost:8888/notebooks/personal_folders/Miguel/qec_lut_demo.ipynb

### setup logging before all imports (before any logging is done as to prevent a default root logger)

import logging
import sys
import numpy as np

from pycqed.instrument_drivers.lib.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC

# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip_cc = '192.168.0.241'
dev_uhfqa = 'dev2271'
cc_port_uhfqa0 = 2



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.debug('connecting to UHFQA')
uhfqa0 = ZI_UHFQC.UHFQC('uhfqa0', device=dev_uhfqa, nr_integration_channels=9)
uhfqa0.load_default_settings(upload_sequence=False)

log.debug('connecting to CC')
cc = QuTechCC('cc', IPTransport(ip_cc))
cc.init()
log.info(cc.get_identity())


if 1:
    log.debug('calibration DIO: CC to UHFQA')




log.debug('calibration DIO: UHFQA to CC')
if 0:
    calibrate(
        sender=uhfqa0,
        receiver=cc,
        receiver_port=cc_port_uhfqa0
    )
else: # inspired by calibrate, but with CC program to trigger UHFQA
    log.debug('sending triggered DIO test program to UHFQA')
    program = '''
    var A = 0x00000CFF; // DV=0x0001, CW=0x0CF7
    var B = 0x00000000;

    while (1) {
        waitDIOTrigger();
        setDIO(A);
        wait(2);
        setDIO(B);
    }
    '''
    uhfqa0.configure_awg_from_string(0, program)
    uhfqa0.seti('awgs/0/enable', 1)


    log.debug('sending UHFQA trigger program to CC')
    cc.debug_marker_out(cc_port_uhfqa0, cc.UHFQA_DV)  # watch DV to check period/frequency
    prog = """
    # depends on HW configuration
    .DEF    duration    9
    .DEF    wait        100
    # slot 2: UHFQA
    [0]loop:    seq_out     0x00010000,$duration      # trigger UHFQA
    [0]         seq_out     0x0,$wait
    [0]         jmp         @loop
    [0]         stop
    # slot 3: observe
    [3]loop:    jmp         @loop
    """
    cc.assemble_and_start(prog)
    dio_mask = 0x00000CFF
    expected_sequence = []

    log.debug('calibrating DIO protocol on CC')
    cc.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=cc_port_uhfqa0)
    cc.stop()
    uhfqa0.stop()





if 1:
    log.debug('run UHFQA codeword generator')
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
#?        uhfqa0.awgs_0_enable(1)
    uhfqa0.start()




if 1:
    log.debug('upload CC feedback test program')
    prog = """
    .DEF    duration    9
    .DEF    wait        100
    .DEF    smAddr      S16
    .DEF    lut         0
    .DEF    numIter     4
    # slot 0: UHFQA
    [0]         move        $numIter,R0
    [0]loop:    seq_out     0x00010000,$duration      # UHFQA measurement
    [0]         seq_in_sm   $smAddr,$lut,0
    [0]         seq_sw_sm   $smAddr
    [0]         seq_out     0x0,$wait
    [0]         loop        R0,@loop
    [0]         stop
    # slot 1-4: observe
    [1]loop:    jmp         @loop
    [2]loop:    jmp         @loop
    [3]loop:    jmp         @loop
    [4]loop:    jmp         @loop
    """
    cc.debug_marker_out(cc_port_uhfqa0, cc.UHFQA_TRIG)
    cc.assemble_and_start(prog)
    log.debug('finished')


