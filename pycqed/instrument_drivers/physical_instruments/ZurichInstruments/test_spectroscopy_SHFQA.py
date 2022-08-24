import numpy as np
import matplotlib.pyplot as pl

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)

import utils

# Spectroscopy specification
CHANNEL = 0
ACQUISITION_TIME = 100e-6
WAIT_DLY = 200e-9
NUM_SAMPLES = 201
NUM_AVERAGES = 100
AVERAGING_MODE = "cyclic"  # or "sequential"
RF_FREQ = 6e9
IF_START = -100e6
IF_STEP = 1e6

shfqa = SHFQA(
    name="shf",
    device="dev12103",
    interface="usb",
    server="localhost",
    nr_integration_channels=Dio.MAX_NUM_RESULTS,
    port=8004,
    use_dio=False,  # Standalone device tests require a different DIO configuration
)
utils.apply_standalone_dio_config(shfqa.daq, shfqa.devname)

# Configure spectroscopy
shfqa.acquisition_initialize(
    samples=NUM_SAMPLES, averages=NUM_AVERAGES, mode="spectroscopy", poll=True
)
shfqa.set(f"qachannels_{CHANNEL}_centerfreq", RF_FREQ)
shfqa.configure_spectroscopy(
    start_frequency=IF_START,
    frequency_step=IF_STEP,
    settling_time=WAIT_DLY,
    dio_trigger=True,
    ch=CHANNEL,
)
shfqa.acquisition_time(ACQUISITION_TIME)
shfqa.averaging_mode(AVERAGING_MODE)

# Execute
shfqa.acquisition_arm()

# Collect and plot
data = shfqa.acquisition_poll(samples=NUM_SAMPLES, arm=False)

frequencies = [(IF_START + i * IF_STEP) / 1e6 for i in range(NUM_SAMPLES)]
input_impedance_ohm = 50
dbM = 10 * np.log10((np.abs(data[CHANNEL]) ** 2) * 1e3 / input_impedance_ohm)
pl.plot(frequencies, dbM)
pl.xlabel("freq [MHz]")
pl.ylabel("power [dBm]")

pl.title(f"Sweep with center frequency {RF_FREQ / 1e9}GHz")
pl.show()
