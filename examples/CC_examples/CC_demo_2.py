#!/usr/bin/python

import logging

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC

# constants
ip = '192.168.0.241'

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ccio = 8
val = 10 # 20 ns steps

# fixed constants
reg = 63

if 1:
    log.debug('connecting to CC')
    cc = QuTechCC('cc', IPTransport(ip))

    cc.stop()
    cc.set_q1_reg(ccio, reg, val)
    cc.start()
