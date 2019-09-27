#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
#import CC_logging

### imports
import sys
import logging

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_tools

# configure our logger
log = logging.getLogger('DIODebug')
log.setLevel(logging.DEBUG)
log.debug('starting')


def print_var(name: str, val_format: str=''):
    fmt = '{{}} = {{{}}}'.format(val_format) # e.g. '{} = {}' or '{} = {:#08X}'
    print(fmt.format(name, instr.get(name)))


log.debug('started')

# default parameters:
dev = 'dev8068'
opt_codewords = False
opt_dio = False
# parameter handling
arg = 1
while arg < len(sys.argv):
    val = sys.argv[arg]
    if val == "-c":
        opt_codewords = True
    elif val == "-d":
        opt_dio = True
    else:
        dev = val
    arg += 1

# show DIO
log.debug('connecting to instrument')
if 1:   # HDAWG
    instr = ZI_HDAWG8.ZI_HDAWG8('mw_0', device=dev)
    instr.assure_ext_clock()

    dio_lines = range(31, -1, -1)

    if 0:  # driver function
        instr.plot_dio_snapshot()

    if opt_codewords:
        for awg in [0, 1, 2, 3]:
            instr.plot_awg_codewords(awg)

    # take a snapshot of the DIO interface
    if opt_dio:
        # get the snapshot data. Time resolution =  3.33 ns, #samples = 1024
        # NB: the DIO timing is applied before the snapshot is taken
        data = instr.getv('raw/dios/0/data')  # NB: no node for that
        ZI_tools.print_timing_diagram_simple(data, dio_lines, 10*6)  # NB: print multiple of 6 samples (i.e. 20 ns)

    if 0:
        # FIXME: looking at single awg
        ts, cws = get_awg_dio_data(instr._dev, 0)
        ZI_tools.print_timing_diagram_simple(cws, dio_lines, 64)



    if 1:  # get list of nodes
        #nodes = instr.daq.listNodes('/' + dev + '/', 7)
        nodes = instr.daq.listNodes('/', 7)
        with open("nodes.txt", "w") as file:
            file.write(str(nodes))
    #log.info(f"DIO delay is set to {instr.getd('raw/dios/0/delays/0')}")
    for awg in [0, 1, 2, 3]:
        log.info(f"AWG{awg} DIO delay is set to {instr.getd(f'awgs/{awg}/dio/delay/value')}")



    for awg in [0, 1, 2, 3]:
        print_var('awgs_{}_dio_error_timing'.format(awg))
        print_var('awgs_{}_dio_error_width'.format(awg))
        #print_var('awgs_{}_dio_value'.format(awg), ':#08X')
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