from RsInstrument import RsInstrument
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)
import qcodes as qc
import pycqed.tests.instrument_drivers.physical_instruments.utils as utils
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


##########################################################################
# Filtering utilities
##########################################################################


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


##########################################################################
# Configure scope
##########################################################################

SCOPE_IP = "192.168.0.3"
scope = RsInstrument(f"TCPIP::{SCOPE_IP}::INSTR", True, False)
# Error check after each command
scope.instrument_status_checking = True
# Send synchronization query after each command
scope.opc_query_after_write = True
scope.write_str("SYSTEM:PRESET")


class ScopeConfigDioMarker:
    CHANNEL = 1
    COUPLING = "DC"
    VERTICAL_POSITION = 0  # in V
    VERTICAL_SCALE = 0.4  # in V/div
    TIMEBASE_POSITION = 242e-9  # in seconds
    TIMEBASE_SCALE = 100e-9  # in seconds
    TRIGGER_INDEX = 1
    TRIGGER_SOURCE = f"CHANNEL{CHANNEL}"
    TRIGGER_LEVEL = 1.17  # in V
    TRIGGER_MODE = "NORMAL"
    VERTICAL_POSITION_2 = 0  # in V
    VERTICAL_SCALE_2 = 0.2  # in V/div


scope.write_str(f"CHANNEL{ScopeConfigDioMarker.CHANNEL}:STATE ON")
scope.write_str(
    f"CHANNEL{ScopeConfigDioMarker.CHANNEL}:POSITION {ScopeConfigDioMarker.VERTICAL_POSITION}"
)
scope.write_str(
    f"CHANNEL{ScopeConfigDioMarker.CHANNEL}:SCALE {ScopeConfigDioMarker.VERTICAL_SCALE}"
)
scope.write_str(
    f"CHANNEL{ScopeConfigDioMarker.CHANNEL}:COUPLING {ScopeConfigDioMarker.COUPLING}"
)
scope.write_str(
    f"TIMEBASE:HORIZONTAL:POSITION {ScopeConfigDioMarker.TIMEBASE_POSITION}"
)
scope.write_str(f"TIMEBASE:SCALE {ScopeConfigDioMarker.TIMEBASE_SCALE}")
scope.write_str(
    f"TRIGGER{ScopeConfigDioMarker.TRIGGER_INDEX}:SOURCE:SELECT {ScopeConfigDioMarker.TRIGGER_SOURCE}"
)
scope.write_str(
    f"TRIGGER{ScopeConfigDioMarker.TRIGGER_INDEX}:LEVEL1 {ScopeConfigDioMarker.TRIGGER_LEVEL}"
)
scope.write_str(
    f"TRIGGER{ScopeConfigDioMarker.TRIGGER_INDEX}:MODE {ScopeConfigDioMarker.TRIGGER_MODE}"
)


class ScopeConfigShfqaPulse:
    CHANNEL = 2
    COUPLING = "DC"
    VERTICAL_POSITION = 0  # in V
    VERTICAL_SCALE = 0.2  # in V/div
    TIMEBASE_POSITION = 242e-9  # in seconds
    TIMEBASE_SCALE = 100e-9  # in seconds


scope.write_str(f"CHANNEL{ScopeConfigShfqaPulse.CHANNEL}:STATE ON")
scope.write_str(
    f"CHANNEL{ScopeConfigShfqaPulse.CHANNEL}:POSITION {ScopeConfigShfqaPulse.VERTICAL_POSITION}"
)
scope.write_str(
    f"CHANNEL{ScopeConfigShfqaPulse.CHANNEL}:SCALE {ScopeConfigShfqaPulse.VERTICAL_SCALE}"
)
scope.write_str(
    f"CHANNEL{ScopeConfigShfqaPulse.CHANNEL}:COUPLING {ScopeConfigShfqaPulse.COUPLING}"
)

##########################################################################
# Configure CC
##########################################################################

station = qc.Station()
central_controller = CC("central_controller", IPTransport("192.168.0.241"))
station.add_component(central_controller, update_snapshot=False)
central_controller.reset()
central_controller.clear_status()
central_controller.status_preset()
# Use DIO marker ouput from CC to trigger the scope
central_controller.debug_marker_out(1, 16)

##########################################################################
# Configure Shfqa
##########################################################################

shfqa = SHFQA(
    name="shf",
    device="dev12103",
    interface="usb",
    server="localhost",
    nr_integration_channels=Dio.MAX_NUM_RESULTS,
    port=8004,
)
station.add_component(shfqa)

codewords = [0]
samples = len(codewords)
averages = 1
averaging_mode = "cyclic"
acquisition_time = 2e-6

shfqa.acquisition_initialize(samples=samples, averages=averages, mode="ro", poll=True)
Iwaves, Qwaves = utils.make_split_waves(num_waves=samples, wave_size=100)
shfqa.awg_sequence_acquisition_and_DIO_triggered_pulse(
    Iwaves=Iwaves,
    Qwaves=Qwaves,
    cases=codewords,
)
shfqa.acquisition_time(acquisition_time)
shfqa.averaging_mode(averaging_mode)

##########################################################################
# Run experiment
##########################################################################

order = 2
fs = 50
cutoff = 0.5
num_samples = 10

latencies = []
for i in range(num_samples):
    # Perform acquisition
    shfqa.acquisition_arm()
    central_controller.assemble_and_start(
        """mainLoop:
                seq_out         0x10000,1 #0b10000000000000000
                seq_out         0x00000000,1
                stop"""
    )

    # Read scope
    dio_marker = scope.query_bin_or_ascii_float_list(
        f"FORM ASC;:CHAN{ScopeConfigDioMarker.CHANNEL}:DATA?"
    )
    shfqa_pulse = scope.query_bin_or_ascii_float_list(
        f"FORM ASC;:CHAN{ScopeConfigShfqaPulse.CHANNEL}:DATA?"
    )
    time = [i * 1 / 20e9 for i in range(len(dio_marker))]

    # Take absolute value of signal
    dio_marker = [abs(x) for x in dio_marker]
    shfqa_pulse = [abs(x) for x in shfqa_pulse]

    # Filter signals to obtain envelope
    shfqa_pulse = butter_lowpass_filter(shfqa_pulse, cutoff, fs, order)
    dio_marker = butter_lowpass_filter(dio_marker, cutoff, fs, order)

    # Find start of shfqa pulse
    running_max = 0
    for i in range(1, len(shfqa_pulse)):
        diff = shfqa_pulse[i] - shfqa_pulse[i - 1]
        if diff > running_max:
            running_max = diff
            shfqa_start = time[i]

    # Find start of dio marker output by cc
    running_max = 0
    for i in range(1, len(dio_marker)):
        diff = dio_marker[i] - dio_marker[i - 1]
        if diff > running_max:
            running_max = diff
            cc_start = time[i]

    latency = 1e9 * (shfqa_start - cc_start)
    latencies.append(latency)

##########################################################################
# Plot data
##########################################################################

plt.plot(range(num_samples), latencies)
plt.ylabel("Delay [ns]")
plt.xlabel("Sample")
plt.show()
