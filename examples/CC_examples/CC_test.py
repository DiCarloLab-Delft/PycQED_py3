# Test program for some CC functionality

from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC

ip_cc = '192.168.0.241'
cc = CC('cc', IPTransport(ip_cc))
cc.init()

##################################################
# SCPI_Base functions
##################################################

idn = cc.get_identity()
options = cc.get_options()

##################################################
# CCCore.py functions
##################################################

cc.debug_marker_in(0, cc.UHFQA_TRIG)
cc.debug_marker_in(0, cc.UHFQA_CW[0])
cc.debug_marker_out(0, cc.UHFQA_TRIG)
cc.debug_marker_out(1, cc.HDAWG_TRIG)
cc.debug_marker_off(0)
cc.debug_marker_off(1)

cc.set_q1_reg(0, 0, 0)
cc.set_q1_reg(0, 63, 0)

cc.set_seqbar_cnt(2,5)

cc.debug_set_ccio_trace_on(0,0)
cc.debug_set_ccio_trace_on(1,0)
traces = cc.debug_get_traces(0x03)

#cc.calibrate_dio_protocol()

prog = '    stop\n'
cc.assemble_and_start(prog)

cc.start()
cc.stop()

##################################################
# CC.py functions
##################################################

cc.dio0_out_delay(0)
cc.dio0_out_delay(31)
