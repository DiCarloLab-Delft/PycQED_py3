#!/usr/bin/python
# Based on: http://localhost:8888/notebooks/personal_folders/Miguel/qec_lut_demo.ipynb

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import os
import logging
import sys
import numpy as np

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
from pycqed.instrument_drivers.meta_instrument.DIOCalibration import calibrate



# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip_cc = '192.168.0.241'
dev_uhfqa = 'dev2271'
cc_port_uhfqa = 2



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



if 1:
    log.debug('connecting to UHFQA')
#    cw_list = [3, 2, 1, 0]
    cw_list = [7, 6, 5, 4]
    cw_array = np.array(cw_list, dtype=int).flatten()

    UHFQC0 = ZI_UHFQC.UHFQC('UHFQC0', device=dev_uhfqa, nr_integration_channels=9)
    UHFQC0.load_default_settings()
    UHFQC0.awg_sequence_acquisition_and_DIO_RED_test(
        Iwaves=[np.ones(8), np.ones(8)],
        Qwaves=[np.ones(8), np.ones(8)],
        cases=[2, 5],
        dio_out_vect=cw_array * 2 + 1,  # shift codeword, add Data Valid
        acquisition_delay=20e-9)

    if 0:
        rolut0 = UHFQC_RO_LutMan('rolut0', num_res=5)
        rolut0.AWG(UHFQC0.name)

    # Prepare AWG_Seq as driver of DIO and set DIO output direction
    UHFQC0.dios_0_mode(1)
    UHFQC0.dios_0_drive(3)

    # Determine trigger and strobe bits from DIO
    UHFQC0.awgs_0_dio_valid_index(16)
    UHFQC0.awgs_0_dio_valid_polarity(0)
    UHFQC0.awgs_0_dio_strobe_index(16)
    UHFQC0.awgs_0_dio_strobe_slope(1)
    UHFQC0.awgs_0_userregs_2(2)

    # Initialize UHF for consecutive triggering and enable it
    UHFQC0.awgs_0_single(0)
    UHFQC0.awgs_0_enable(1)

    UHFQC0.start()




if 1:
    log.debug('generating program')
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

    log.debug('connecting to CC')
    cc = QuTechCC('cc', IPTransport(ip_cc))
    cc.reset()
    cc.clear_status()
    cc.status_preset()
    log.info(cc.get_identity())

    if 1:
        log.debug('calibration DIO: CC to UHFQA')


        log.debug('calibration DIO: UHFQA to CC')
        calibrate(
            sender=UHFQC0,
            receiver=cc,
            receiver_port=cc_port_uhfqa
        )

    log.debug('uploading program to CC')
    cc.sequence_program_assemble(prog)
    if cc.get_assembler_success() != 1:
        sys.stderr.write('error log = {}\n'.format(cc.get_assembler_log()))  # FIXME: result is messy
        log.warning('assembly failed')
    else:
        log.debug('checking for SCPI errors on CC')
        err_cnt = cc.get_system_error_count()
        for i in range(err_cnt):
            print(cc.get_error())
        log.debug('done checking for SCPI errors on CC')

        log.debug('starting CC')
        cc.debug_marker_out(0, cc.UHFQA_TRIG)
        cc.start()
    log.debug('finished')


