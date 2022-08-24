import numpy as np
import matplotlib.pyplot as pl
import time

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.internal._shfqa import (
    duration_to_length,
)

import utils

##########################################################################
# Plotting helpers
##########################################################################


def plot_ro_results(ro_data, color):
    assert (
        len(ro_data) == 1
    ), "Standalone device tests assume that ro results only come from a single channel."

    for scope_data in ro_data.values():
        assert len(scope_data) == 1, (
            "Standalone device tests assume that ro results always come as single samples, that is, as one scope shot per"
            " experiment run."
        )

        y = scope_data[0]
        time = [1e6 * i / driver.clock_freq() for i in range(len(y))]
        pl.plot(time, y.imag, color=color)


def plot_rl_results(rl_data, color, max_value):
    for slots in rl_data.values():
        for complex_numbers in slots.values():
            for complex_number in complex_numbers:
                if not np.isnan(complex_number):
                    real = np.real(complex_number)
                    imag = np.imag(complex_number)
                    max_value = max(max_value, real, imag)
                    pl.plot(real, imag, "o", color=color)
    return max_value


##########################################################################
# Constants
##########################################################################

SAMPLES = 1
AVERAGES = 1000
ACQUISITION_TIME = 1.5e-6
WAIT_DLY = 0

##########################################################################
# Instantiate driver
##########################################################################

driver = SHFQA(
    name="shf",
    device="dev12103",
    interface="usb",
    server="localhost",
    nr_integration_channels=Dio.MAX_NUM_RESULTS,
    port=8004,
    use_dio=False,  # Standalone device tests require a different DIO configuration
)
utils.apply_standalone_dio_config(driver.daq, driver.devname)

codewords = range(Dio.MAX_NUM_RESULTS)
# Pick a sequential colormap to ease result visualization
colors = pl.cm.Blues(np.linspace(0.3, 1, len(codewords)))

driver.acquisition_initialize(samples=SAMPLES, averages=AVERAGES, mode="ro", poll=True)
driver.acquisition_time(ACQUISITION_TIME)

# Associate codewords to constant waveforms with amplitudes proportional to the codeword value, and prepare the
# sequencer programs
pulse_duration = 25e-9
pulse_length = duration_to_length(pulse_duration)
Iwaves, Qwaves = utils.make_split_waves(
    num_waves=len(codewords), wave_size=pulse_length
)
driver.awg_sequence_acquisition_and_DIO_triggered_pulse(
    Iwaves=Iwaves,
    Qwaves=Qwaves,
    cases=codewords,
)

# Set all the defined codewords on the DIO output one after the other
for i, codeword in enumerate(codewords):
    # Increase the acquisition delay by one pulse duration at each iteration
    driver.wait_dly(i * pulse_duration)
    driver.daq.syncSetInt(
        f"/{driver.devname}/dios/0/output", codeword << utils.STANDALONE_CODEWORD_SHIFT
    )
    driver.acquisition_arm()
    data = driver.acquisition_poll(samples=SAMPLES, arm=False)
    plot_ro_results(data, colors[i])

print(
    "Expected shape is a staircase pattern with light/dark plots representing low/high codeword values, "
    "corresponding to low/high amplitude and low/high delay values, respectively."
)
pl.show()

codewords = range(Dio.MAX_NUM_RESULTS)
colors = pl.cm.Blues(np.linspace(0.1, 1, len(codewords)))
driver.cases(codewords)

driver.samples(SAMPLES)
driver.averages(AVERAGES)
driver.result_mode("rl")
driver.wait_dly(WAIT_DLY)
driver.acquisition_time(ACQUISITION_TIME)

# Associate codewords to sine waveforms with a phase proportional to the codeword value to observe a rotating pattern
pulse_duration = 1e-6
f = 10e6
rotations = np.linspace(0, 2 * np.pi, len(codewords))
waveforms = [
    utils.generate_SSB_wave(
        frequency=f, duration=pulse_duration, amplitude=0.5, rotation=r
    )
    for r in rotations
]

# Set codeword waveforms
for i, waveform in enumerate(waveforms):
    assert i < len(codewords)
    parameter = "wave_cw{:03}".format(codewords[i])
    driver.set(parameter, waveform)

# Set identical weights on all slots
for ch in range(Dio.MAX_NUM_CHANNELS):
    for slot in range(Dio.MAX_NUM_RESULTS_PER_CHANNEL):
        driver.prepare_SSB_weight_and_rotation(IF=f, rotation_angle=0, ch=ch, slot=slot)

pl.figure()
max_value = 0
for i, codeword in enumerate(codewords):
    driver.daq.syncSetInt(
        f"/{driver.devname}/dios/0/output", codeword << utils.STANDALONE_CODEWORD_SHIFT
    )
    time.sleep(0.2)
    driver.acquisition_arm()
    time.sleep(0.2)
    data = driver.acquisition_poll(samples=SAMPLES, arm=False)

    max_value = plot_rl_results(data, colors[i], max_value)

max_value *= 1.1
pl.xlim([-max_value, max_value])
pl.ylim([-max_value, max_value])

print(
    "For each channel, observe the integrated value rotating by 180 degrees over the codeword range. Light/dark dots "
    "represent low/high codeword values, corresponding to low/high phase, respectively."
)
pl.show()
