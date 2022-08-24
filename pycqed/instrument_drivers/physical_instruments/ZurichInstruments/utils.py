import numpy as np
import pytest
import math
from contextlib import contextmanager

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.internal._shfqa import (
    duration_to_length,
    DeviceConstants,
)
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)

from zhinst.utils import wait_for_state_change


def assert_complex_data_equal(original, received):
    assert len(original) == len(received)
    conversion_tolerance = 2 / ((2 ** 18) - 2)
    data_type = "complex128"
    assert (
        received.dtype == data_type
    ), f"Received {received.dtype} but expected ''{data_type}''."

    received_imag = np.imag(received)
    received_real = np.real(received)
    orig_imag = np.imag(original)
    orig_real = np.real(original)
    assert np.allclose(
        received_real, orig_real, atol=conversion_tolerance
    ), "Mismatch of received and send data in real part."
    assert np.allclose(
        received_imag, orig_imag, atol=conversion_tolerance
    ), "Mismatch of received and send data in imag part."


def assert_expected_result_format(data, samples, acquisition_time, result_mode):
    for entry in data.values():
        if result_mode == "ro":
            assert len(entry) == samples
            for scope_shot in entry:
                assert len(scope_shot) == duration_to_length(acquisition_time)
                assert isinstance(scope_shot[0], np.complex128)
        if result_mode == "rl":
            for integrator in entry.values():
                assert len(integrator) == samples
        if result_mode == "spectroscopy":
            assert len(entry) == samples
            assert isinstance(entry[0], np.complex128)


@contextmanager
def no_raise(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"Raised exception {exception}.")


def get_uploaded_sequencer_program(daq, device_id, ch):
    generator_path = f"/{device_id}/qachannels/{ch}/generator/sequencer/program"
    return daq.get(generator_path, flat=True)[generator_path][0]["vector"]


def make_const_wave(amplitude, size):
    return np.array([amplitude - 1j * amplitude] * size)


def make_monotonically_increasing_const_waves(num_waves, wave_size):
    # Ensure waveforms always remain within the complex unit circle
    scaling = math.sqrt(2) / (2 * num_waves)
    return [
        make_const_wave(amplitude=(i + 1) * scaling, size=wave_size)
        for i in range(num_waves)
    ]


def make_split_waves(num_waves, wave_size):
    Iwaves = []
    Qwaves = []
    for wave in make_monotonically_increasing_const_waves(num_waves, wave_size):
        Iwaves.append(wave.real)
        Qwaves.append(wave.imag)
    return Iwaves, Qwaves


def generate_SSB_wave(frequency, duration, amplitude, split=False, rotation=0):
    length = duration_to_length(duration)
    ticks = np.arange(int(length))

    sin = amplitude * np.sin(
        2 * np.pi * ticks * frequency / DeviceConstants.SAMPLING_FREQUENCY + rotation
    )
    cos = amplitude * np.cos(
        2 * np.pi * ticks * frequency / DeviceConstants.SAMPLING_FREQUENCY + rotation
    )

    I = (cos + sin) / np.sqrt(2)
    Q = (cos - sin) / np.sqrt(2)

    if split:
        return (I, Q)
    return np.array([(I[i] + 1j * Q[i]) for i in range(length)])


def reset(driver):
    for ch in range(Dio.MAX_NUM_CHANNELS):
        driver.daq.syncSetInt(
            f"/{driver.devname}/qachannels/{ch}/generator/clearwave", 1
        )
        driver.daq.syncSetInt(
            f"/{driver.devname}/qachannels/{ch}/generator/reset",
            1,
        )
        driver.daq.syncSetInt(
            f"/{driver.devname}/qachannels/{ch}/readout/integration/clearweight",
            1,
        )
        driver.load_default_settings()


CODEWORD_LENGTH = 1
CODEWORD_SPACING = 100


def codeword_instruction(codeword, indent):
    value = (codeword << Dio.INPUT_SHIFT) | (1 << Dio.VALID_INDEX)
    return f"{indent}seq_out         {hex(value)},{CODEWORD_LENGTH} #{bin(value)}\n{indent}seq_out         0x00000000,{CODEWORD_SPACING}\n"


def make_cc_program(repetitions, codewords, averaging_mode="sequential"):
    program = "mainLoop:\n"
    indent = "\t\t\t"
    if averaging_mode == "cyclic":
        for i in range(repetitions):
            for codeword in codewords:
                program += codeword_instruction(codeword, indent)
    elif averaging_mode == "sequential":
        for codeword in codewords:
            for i in range(repetitions):
                program += codeword_instruction(codeword, indent)
    else:
        raise ValueError(f"Unsupported averaging mode {averaging_mode}")
    program += f"{indent}stop"
    return program


STANDALONE_CODEWORD_SHIFT = 17


def apply_standalone_dio_config(daq, device_id):
    MANUAL_MODE = 0
    DRIVE_ALL_BITS = 15
    NO_POLARITY = 0

    daq.syncSetInt(f"/{device_id}/dios/0/mode", MANUAL_MODE)
    daq.syncSetInt(f"/{device_id}/dios/0/drive", DRIVE_ALL_BITS)
    for ch in range(Dio.MAX_NUM_CHANNELS):
        polarity_path = f"/{device_id}/qachannels/{ch}/generator/dio/valid/polarity"
        daq.syncSetInt(polarity_path, NO_POLARITY)
        wait_for_state_change(daq, polarity_path, NO_POLARITY)
