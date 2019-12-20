#!/usr/bin/python

import logging
import sys

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC


# parameter handling
num_iter = 1000
num_run_per_iter = 1000
if len(sys.argv)>1:
    num_iter = int(sys.argv[1])
if len(sys.argv)>1:
    num_run_per_iter = int(sys.argv[2])

# constants
ip = '192.168.0.241'

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


log.debug('generating program')
prog = ''
for i in range(10000):
    prog += '    seq_out         0x00000000,301\n'


log.debug('connecting to CC')
cc = QuTechCC('cc', IPTransport(ip))
cc.reset()
cc.clear_status()
cc.status_preset()

for i in range(num_iter):
    cc.sequence_program_assemble(prog)

    for run in range(num_run_per_iter):
        print(f'starting CC iter={i}, run={run}')
        cc.start()
        cc.stop()
        err_cnt = cc.get_system_error_count()
        for j in range(err_cnt):
            print(cc.get_error())
