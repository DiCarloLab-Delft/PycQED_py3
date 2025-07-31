#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
#import CC_logging

import logging
import sys

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC


# parameter handling
ip = '192.168.0.241'
if len(sys.argv)>1:
    ip = sys.argv[1]

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

log.info('connecting to CC')
cc = CC('cc', IPTransport(ip))

print('print_status: Condition')
cc.print_status(True)

# FIXME: red LOCK LED remains on after unlocking and relocking  

print('\nprint_status: Event')
cc.print_status(False)

