import matplotlib.pyplot as pl

from RsInstrument import RsInstrument

from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)
from pycqed.instrument_drivers.library.DIO import calibrate

import qcodes as qc

import utils

##########################################################################
# Instantiate devices
##########################################################################

station = qc.Station()
central_controller = CC("central_controller", IPTransport("192.168.0.241"))
station.add_component(central_controller, update_snapshot=False)
central_controller.reset()
central_controller.clear_status()
central_controller.status_preset()

shfqa = SHFQA(
    name="shf",
    device="dev12103",
    interface="usb",
    server="localhost",
    nr_integration_channels=Dio.MAX_NUM_RESULTS,
    port=8004,
)
station.add_component(shfqa)

SCOPE_IP = "192.168.0.3"
scope = RsInstrument(f"TCPIP::{SCOPE_IP}::INSTR", True, False)

calibrate(central_controller, shfqa, sender_dio_mode="shfqa")
##########################################################################
# Configure experiment
##########################################################################

# SHFQA
codewords = range(Dio.MAX_NUM_RESULTS)
samples = len(codewords)
averages = 5
averaging_mode = "cyclic"
acquisition_time = 10e-9

shfqa.acquisition_initialize(samples=samples, averages=averages, mode="rl", poll=True)
Iwaves, Qwaves = utils.make_split_waves(num_waves=samples, wave_size=2048)
shfqa.awg_sequence_acquisition_and_DIO_triggered_pulse(
    Iwaves=Iwaves,
    Qwaves=Qwaves,
    cases=codewords,
)
shfqa.acquisition_time(acquisition_time)
shfqa.averaging_mode(averaging_mode)

# Scope
class ScopeConfig:
    CHANNEL = 2
    COUPLING = "DC"
    VERTICAL_POSITION = 0  # in V
    VERTICAL_SCALE = 0.04  # in V/div
    TIMEBASE_POSITION = 30e-6  # in seconds
    TIMEBASE_SCALE = 8e-6  # in seconds
    TRIGGER_INDEX = 1
    TRIGGER_SOURCE = f"CHANNEL{CHANNEL}"
    TRIGGER_LEVEL = 0.02  # in V
    TRIGGER_MODE = "NORMAL"


# Error check after each command
scope.instrument_status_checking = True
# Send synchronization query after each command
scope.opc_query_after_write = True
scope.write_str("SYSTEM:PRESET")
scope.write_str(f"CHANNEL{ScopeConfig.CHANNEL}:STATE ON")
scope.write_str(
    f"CHANNEL{ScopeConfig.CHANNEL}:POSITION {ScopeConfig.VERTICAL_POSITION}"
)
scope.write_str(f"CHANNEL{ScopeConfig.CHANNEL}:SCALE {ScopeConfig.VERTICAL_SCALE}")
scope.write_str(f"CHANNEL{ScopeConfig.CHANNEL}:COUPLING {ScopeConfig.COUPLING}")
scope.write_str(f"TIMEBASE:HORIZONTAL:POSITION {ScopeConfig.TIMEBASE_POSITION}")
scope.write_str(f"TIMEBASE:SCALE {ScopeConfig.TIMEBASE_SCALE}")
scope.write_str(
    f"TRIGGER{ScopeConfig.TRIGGER_INDEX}:SOURCE:SELECT {ScopeConfig.TRIGGER_SOURCE}"
)
scope.write_str(
    f"TRIGGER{ScopeConfig.TRIGGER_INDEX}:LEVEL1 {ScopeConfig.TRIGGER_LEVEL}"
)
scope.write_str(f"TRIGGER{ScopeConfig.TRIGGER_INDEX}:MODE {ScopeConfig.TRIGGER_MODE}")

# Central controller
cc_program = utils.make_cc_program(
    repetitions=averages, codewords=codewords, averaging_mode=averaging_mode
)

##########################################################################
# Configure experiment
##########################################################################

shfqa.acquisition_arm()
central_controller.assemble_and_start(cc_program)
shfqa.acquisition_finalize()

##########################################################################
# Collect results
##########################################################################

scope_shot = scope.query_bin_or_ascii_float_list(
    f"FORM ASC;:CHAN{ScopeConfig.CHANNEL}:DATA?"
)

time = [i * 1 / 20e9 for i in range(len(scope_shot))]
pl.plot(time, scope_shot)
pl.show()
