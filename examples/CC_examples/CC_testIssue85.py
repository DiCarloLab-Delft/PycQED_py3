#!/usr/bin/python

# test for https://github.com/DiCarloLab-Delft/ElecPrj_CC/issues/85

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


prog = """
        move    0xFFFFFFFF,R0
        nop
        seq_bar
        stop
"""

print('connecting to CC')
cc = CC('cc', IPTransport(ip))

print('seting up CC')
cc.reset()
cc.clear_status()
cc.status_preset()

cc.debug_set_ccio_trace_on(1, cc.TRACE_CCIO_DEV_OUT)

print('starting CC')
cc.assemble_and_start(prog)
cc.stop()

print('showing trace data')
traces = cc.debug_get_traces(0xFFFF)
print(traces)
# NB: interpreting requires insight in .VCD format. CC version 0.2.6, with the patch that makes seq_bar output zero
# instead of the contents of R0, results in (timestamps may vary):
#   [...]
#   $enddefinitions $end
#   #2665683
#   b00000000000000000000000000000000 13
#   #2665687
#   b00000000000000000000000000000000 13
# Later versions with the HDL corrected should not output trace data

