from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.library.DIO import calibrate
import time
import qcodes as qc
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

##########################################################################
# Configure CC
##########################################################################

station = qc.Station()
central_controller = CC("central_controller", IPTransport("192.168.0.241"))
station.add_component(central_controller, update_snapshot=False)
central_controller.reset()
central_controller.clear_status()
central_controller.status_preset()

##########################################################################
# Configure HDAWG
##########################################################################

HDAWG = ZI_HDAWG8(
    name="hdawg",
    device="dev8066",
    interface="USB",
    server="localhost",
    port=8004
)

ccio = 3
central_controller.calibrate_dio_protocol()
# central_controller.assemble_and_start("""mainLoop:
#                     seq_out         0x10000,1 #0b10000000000000000
#                     seq_out         0x00000000,1
#                     stop""")