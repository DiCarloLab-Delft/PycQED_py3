#!/usr/bin/python

import logging
import sys

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC


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




print('connecting to CC')
cc = CC('cc', IPTransport(ip))
cc.reset()
cc.clear_status()
cc.status_preset()

for i in range(num_iter):
    if 1:
#        prog =  'loop:    seq_out         0x00000000,10\n'
        prog =  'loop:    seq_out         0x00000000,2\n'
        # 1: no ILLEGAL_INSTR_RT
        # 2: ~50%
        # 10: mostly
        prog += '         jmp            @loop\n'
    else:
        length = randint(100,10000)
        print(f'generating program of length {length}')
        prog = ''
        for line in range(length):
            prog += '    seq_out         0x00000000,301\n'
        prog += 'stop\n'

    cc.sequence_program_assemble(prog)

    for run in range(num_run_per_iter):
        print(f'starting CC iter={i}, run={run}')
        cc.start()
        cc.stop()
        err_cnt = cc.get_system_error_count()
        for j in range(err_cnt):
            print(cc.get_error())
