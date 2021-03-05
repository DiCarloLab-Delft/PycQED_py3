#!/usr/bin/python

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

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.info('connecting to CC')
cc = CC('cc', IPTransport(ip))
cc.init()

if 0:
    cc.debug_marker_out(0, cc.UHFQA_TRIG)
    cc.debug_marker_out(1, cc.UHFQA_TRIG)
    #cc.debug_marker_out(8, cc.HDAWG_TRIG)

log.info(f'uploading {filename} and starting CC')
with open(filename, 'r') as f:
    prog = f.read()
cc.assemble_and_start(prog)

err_cnt = cc.get_system_error_count()
for i in range(err_cnt):
    print(cc.get_error())

cc.print_event()
cc.print_status()
