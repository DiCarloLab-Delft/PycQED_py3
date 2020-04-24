#!/usr/bin/python

import logging
import sys
import math
import numpy as np

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC

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


# helpers
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

# program constants
center_x = 15.5
center_y = 15.5
scales = np.arange(-1.0, 1.0, 0.01)
angles = np.arange(-1.0, 1.0, 0.01)*math.pi
mat_center = np.array(
    [[1, 0, -center_x],
     [0, 1, -center_y],
     [0, 0, 1]])
mat_uncenter = np.array(
    [[1, 0, center_x],
     [0, 1, center_y],
     [0, 0, 1]])

# generate program
log.debug('generating program')
prog = ''
prog += '   seq_bar 1\n'
prog += 'start:\n'
for i,scale in enumerate(scales):
    prog += '    move 50000,R0\n'  # determines speed
    prog += 'loop{}:\n'.format(i)
    mat_scale = np.array(
        [[scale,    0,          0],
         [0,        scale,      0],
         [0,        0,          1]])
    phi = angles[i]  # FIXME: assumes angles and scales have compatible size
    mat_rot = np.array(
        [[math.cos(phi),    -math.sin(phi), 0],
         [math.sin(phi),    math.cos(phi),  0],
         [0,                0,              1]])
    if 0:
        mat_transform = mat_uncenter @ mat_scale @ mat_center
    else:
        mat_transform = mat_uncenter @ mat_rot @ mat_scale @ mat_center

    for coord in coords:
        # compute position
        vec = np.array([coord[0], coord[1], 1])  # vector for affine transformation
        pos = mat_transform @ vec  # @=matrix multiply
        x = int(round(pos[0]))
        y = int(round(pos[1]))
        x = clamp(x, 0, 31)
        y = clamp(y, 0, 31)

        # generate instruction
        val = (1<<31) + (y<<9) + (x)  # mode "microwave"
        prog += '    seq_out {},1\n'.format(val)
    prog += '    loop R0,@loop{}\n'.format(i)
prog += '    jmp @start\n'
log.debug('program generated: {} lines, {} bytes'.format(prog.count('\n'), len(prog)))


log.debug('connecting to CC')
cc = CC('cc', IPTransport(ip))
cc.reset()
cc.clear_status()
cc.status_preset()

log.debug('uploading program to CC')
cc.sequence_program_assemble(prog)
if cc.get_assembler_success() != 1:
    sys.stderr.write('error log = {}\n'.format(cc.get_assembler_log()))  # FIXME: result is messy
    log.warning('assembly failed')
else:
    log.debug('checking for SCPI errors on CC')
    err_cnt = cc.get_system_error_count()
    for i in range(err_cnt):
        print(cc.get_error())
    log.debug('done checking for SCPI errors on CC')

    log.debug('starting CC')
    cc.start()
