#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import logging
import sys
import os
import glob

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC


# parameter handling
file_or_dir = ''
ip = '192.168.0.241'
if len(sys.argv)==1:
    raise("missing parameter 'file_or_directory'")
if len(sys.argv)>1:
    file_or_dir = sys.argv[1]
if len(sys.argv)>2:
    ip = sys.argv[2]

# get file names
filenames = []
if os.path.isdir(file_or_dir):
    filenames += glob.glob(os.path.join(file_or_dir, "*.vq1asm"))
else:
    filenames.append(file_or_dir)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

log.info('connecting to CC')
cc = CC('cc', IPTransport(ip))
#cc.init()

for filename in filenames:
    log.info(f"uploading '{filename}' and starting CC")
    with open(filename, 'r') as f:
        prog = f.read()
    cc.assemble_and_start(prog)

    err_cnt = cc.get_system_error_count()
    for i in range(err_cnt):
        print(cc.get_system_error())

#cc.print_event()
cc.print_status()
