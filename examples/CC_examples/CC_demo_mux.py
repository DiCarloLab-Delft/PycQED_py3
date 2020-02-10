#!/usr/bin/python
# Based on: http://localhost:8888/notebooks/personal_folders/Miguel/qec_lut_demo.ipynb

import os
import logging
import sys
import numpy as np

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan


# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip_cc = '192.168.0.241'
dev_uhfqa = 'dev2493'



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



if 1:
    log.debug('connecting to UHFQA')
    cw_list = [0, 1, 2, 3]
    cw_array = np.array(cw_list, dtype=int).flatten()

    UHFQC0 = ZI_UHFQC.UHFQC('UHFQC0', device=dev_uhfqa, nr_integration_channels=9)
    if 1:
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




if 0:
    log.debug('connecting to CC')
    cc = QuTechCC('cc', IPTransport(ip_cc))
    cc.reset()
    cc.clear_status()
    cc.status_preset()

    if 1:
        cc.debug_marker_out(0, cc.UHFQA_TRIG) # UHF-QA trigger
        cc.debug_marker_out(8, cc.HDAWG_TRIG) # HDAWG trigger



    err_cnt = cc.get_system_error_count()
    for i in range(err_cnt):
        print(cc.get_error())

    log.debug('starting CC')
    cc.start()
