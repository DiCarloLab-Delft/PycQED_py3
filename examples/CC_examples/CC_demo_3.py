#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import logging
import sys

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC

# configure our logger
log = logging.getLogger('demo_3')
log.setLevel(logging.DEBUG)
log.debug('starting')

# parameter handling
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# constants
ip = '192.168.0.241'


if sel==0:
    coords = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0],
        [7, 0],
        [8, 0],
        [9, 0],
        [10, 0],
        [11, 0],
        [12, 0],
        [13, 0],
        [14, 0],
        [15, 0],
        [16, 0],
        [17, 0],
        [18, 0],
        [19, 0],
        [20, 0],
        [21, 0],
        [22, 0],
        [23, 0],
        [24, 0],
        [25, 0],
        [26, 0],
        [27, 0],
        [28, 0],
        [29, 0],
        [30, 0],
        [31, 0],

        [31, 1],
        [31, 2],
        [31, 3],
        [31, 4],
        [31, 5],
        [31, 6],
        [31, 7],
        [31, 8],
        [31, 9],
        [31, 10],
        [31, 11],
        [31, 12],
        [31, 13],
        [31, 14],
        [31, 15],
        [31, 16],
        [31, 17],
        [31, 18],
        [31, 19],
        [31, 20],
        [31, 21],
        [31, 22],
        [31, 23],
        [31, 24],
        [31, 25],
        [31, 26],
        [31, 27],
        [31, 28],
        [31, 29],
        [31, 30],
        [31, 31],

        [30, 31],
        [29, 31],
        [28, 31],
        [27, 31],
        [26, 31],
        [25, 31],
        [24, 31],
        [23, 31],
        [22, 31],
        [21, 31],
        [20, 31],
        [19, 31],
        [18, 31],
        [17, 31],
        [16, 31],
        [15, 31],
        [14, 31],
        [13, 31],
        [12, 31],
        [11, 31],
        [10, 31],
        [9, 31],
        [8, 31],
        [7, 31],
        [6, 31],
        [5, 31],
        [4, 31],
        [3, 31],
        [2, 31],
        [1, 31],
        [0, 31],

        [0, 30],
        [0, 29],
        [0, 28],
        [0, 27],
        [0, 26],
        [0, 25],
        [0, 24],
        [0, 23],
        [0, 22],
        [0, 21],
        [0, 20],
        [0, 19],
        [0, 18],
        [0, 17],
        [0, 16],
        [0, 15],
        [0, 14],
        [0, 13],
        [0, 12],
        [0, 11],
        [0, 10],
        [0, 9],
        [0, 8],
        [0, 7],
        [0, 6],
        [0, 5],
        [0, 4],
        [0, 3],
        [0, 2],
        [0, 1]
    ]

# generate program
center_x = 16
center_y = 16
scale_x = 0.5
scale_y = 1.0
scales = [100] #[100, 50, 0]:   #range(-100,100,50):
prog = 'start:\n'
for scale in scales:
    prog += ' move 50000,R0\n'
    prog += 'loop{}:\n'.format(scale+100)
    scale_x = scale/100
    for coord in coords:
        x = round((coord[0]-center_x)*scale_x + center_x)
        y = round((coord[1]-center_y)*scale_y + center_y)
        val = (1<<31) + (y<<9) + (x)
        prog += '        seq_out {},1\n'.format(val)
    prog += ' loop R0,@loop{}\n'.format(scale+100)
prog += '        jmp @start\n'


log.debug('connecting to CC')
cc = QuTechCC('cc', IPTransport(ip, timeout=5.0)) # FIXME: raised timeout until assembly time reduced
cc.reset()
cc.clear_status()
cc.set_status_questionable_frequency_enable(0x7FFF)

cc.sequence_program(prog)

err_cnt = cc.get_system_error_count()
for i in range(err_cnt):
    print(cc.get_error())

log.debug('starting CC')
cc.start()
