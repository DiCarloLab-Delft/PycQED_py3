#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
#import CC_logging

### imports
import sys
import logging

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.wouter import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.wouter import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.wouter import ZI_tools

# configure our logger
log = logging.getLogger('DIODebug')
log.setLevel(logging.DEBUG)
log.debug('starting')


def print_var(name: str, val_format: str=''):
    fmt = '{{}} = {{{}}}'.format(val_format) # e.g. '{} = {}' or '{} = {:#08X}'
    print(fmt.format(name, instr.get(name)))


# from http://localhost:8888/notebooks/Electronics_Design/AWG8_V4_DIO_Calibration.ipynb
def get_awg_dio_data(dev, awg):
    data = dev.getv('awgs/' + str(awg) + '/dio/data')
    ts = len(data) * [0]
    cw = len(data) * [0]
    for n, d in enumerate(data):
        ts[n] = d >> 10
        cw[n] = (d & ((1 << 10) - 1))
    return (ts, cw)


# parameter handling
log.debug('started')
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

# instrument info
dev = 'dev8078'

# show DIO
log.debug('connecting to instrument')
if 1:   # HDAWG
    instr = ZI_HDAWG8.ZI_HDAWG8('mw_0', device=dev)

    dio_lines = range(31, -1, -1)

    # take a snapshot of the DIO interface
    if 1:
        # get the snapshot data. Time resolution =  3.33 ns, #samples = 1024
        # FIXME: is this still true: NB: the DIO timing is applied before the snapshot is taken
        data = instr._dev.getv('raw/dios/0/data') # FIXME: no node for that
        ZI_tools.print_timing_diagram_simple(data, dio_lines, 64)

    if 0:
        # FIXME: looking at single awg
        ts, cws = get_awg_dio_data(instr._dev, 0)
        ZI_tools.print_timing_diagram_simple(cws, dio_lines, 64)

    for awg in [0, 1, 2, 3]:
        print_var('awgs_{}_dio_error_timing'.format(awg))
        print_var('awgs_{}_dio_error_width'.format(awg))
        print_var('awgs_{}_dio_value'.format(awg), ':#08X')
        print_var('awgs_{}_dio_highbits'.format(awg), ':#08X')
        print_var('awgs_{}_dio_lowbits'.format(awg), ':#08X')

        print_var('awgs_{}_dio_mask_shift'.format(awg))
        print_var('awgs_{}_dio_mask_value'.format(awg), ':#08X')
        print_var('awgs_{}_dio_state'.format(awg))
        print_var('awgs_{}_dio_strobe_index'.format(awg))
        print_var('awgs_{}_dio_strobe_slope'.format(awg))
        print_var('awgs_{}_dio_strobe_width'.format(awg))
        print_var('awgs_{}_dio_valid_index'.format(awg))
        print_var('awgs_{}_dio_valid_polarity'.format(awg))
        print_var('awgs_{}_dio_valid_width'.format(awg))

if 0:   # FIXME: UHFQC
    # take a snapshot of the DIO interface (NB: control CC manually)
    # NB: the DIO timing is applied before the snapshot is taken
    # dio_lines = [31, 21, 20, 19, 18, 17, 16]
    dio_lines = range(31, 15, -1)

    dio_data = UHFQC.awgs_0_dio_data()  # get the snapshot data. Time resolution = 1/450MHz=2.222ns, #samples = 64
    dio_data_vector = dio_data[0]['vector']
    print_timing_diagram_simple(dio_data_vector, dio_lines, 64)