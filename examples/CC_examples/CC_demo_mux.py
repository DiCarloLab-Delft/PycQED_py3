#!/usr/bin/python

import os
import logging
import sys
import numpy as np

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC

from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo

# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip_cc = '192.168.0.241'
uhfqa = 'dev2493'



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



if 1:
    log.debug('connecting to UHFQA')


if 1:
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
