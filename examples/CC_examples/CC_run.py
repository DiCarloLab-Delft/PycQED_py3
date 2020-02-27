#!/usr/bin/python

import os
import logging
import sys

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC


# parameter handling
filename = ''
ip = '192.168.0.241'
if len(sys.argv)==1:
    raise("missing parameter 'filename'")
if len(sys.argv)>1:
    filename = sys.argv[1]
if len(sys.argv)>2:
    ip = sys.argv[2]

# constants
qubit_idx = 10
curdir = os.path.dirname(__file__)
cfg_openql_platform_fn = os.path.join(curdir, 'demo1_cfg.json')


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



log.debug('connecting to CC')
cc = CC('cc', IPTransport(ip))
cc.init()

if 0:
    cc.debug_marker_out(0, cc.UHFQA_TRIG) # UHF-QA trigger
    cc.debug_marker_out(8, cc.HDAWG_TRIG) # HDAWG trigger

log.debug(f'uploading {filename}')
with open(filename, 'r') as f:
    prog = f.read()
cc.sequence_program_assemble(prog)
cc.check_errors()

log.debug('starting CC')
cc.start()
