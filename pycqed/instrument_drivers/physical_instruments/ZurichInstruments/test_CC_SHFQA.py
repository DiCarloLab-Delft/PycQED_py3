import numpy as np
import pytest
import logging

from qcodes import Station

from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import IPTransport

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)

import utils

log = logging.getLogger(__name__)

try:
    station = Station()
    central_controller = CC("central_controller", IPTransport("192.168.0.241"))
    station.add_component(central_controller, update_snapshot=False)
    central_controller.reset()
    central_controller.clear_status()
    central_controller.status_preset()

    driver = SHFQA(
        name="shf",
        device="dev12103",
        interface="usb",
        server="localhost",
        nr_integration_channels=Dio.MAX_NUM_RESULTS,
        port=8004,
    )
    station.add_component(driver)
    driver.assure_ext_clock()
except BaseException as e:
    pytestmark = pytest.mark.skip(
        f"Could not connect to devices; skipping SHFQA tests. "
        f"Exception that gave rise to this skip: {str(e)}"
    )

ACQUISITION_TIME = 20e-9
WAIT_DLY = 100e-9


class TestCcShfqa:
    """
    This test suite is meant to be run using an SHFQA connected to a central controller through DIO.
    """

    @pytest.fixture
    def setup(self):
        utils.reset(driver)
        return driver, central_controller

    ##########################################################################
    # Test that experiments run through, and collect the expected data
    ##########################################################################

    @pytest.mark.parametrize(
        "codewords",
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )
    @pytest.mark.parametrize("averages", [1, 200])
    @pytest.mark.parametrize("poll", [False, True])
    def test_dio_sequence_ro_collects_all_data(self, setup, codewords, averages, poll):
        testee, cc = setup

        # Configure
        samples = len(codewords)
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode="ro", poll=poll
        )
        Iwaves, Qwaves = utils.make_split_waves(
            num_waves=len(codewords), wave_size=2000
        )
        testee.awg_sequence_acquisition_and_DIO_triggered_pulse(
            Iwaves=Iwaves,
            Qwaves=Qwaves,
            cases=codewords,
        )
        testee.acquisition_time(ACQUISITION_TIME)

        # Execute
        testee.acquisition_arm()
        cc_program = utils.make_cc_program(repetitions=averages, codewords=codewords)
        cc.assemble_and_start(cc_program)

        # Collect
        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(
                    samples=samples, arm=False, acquisition_time=1
                )
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        testee.acquisition_finalize()

        utils.assert_expected_result_format(
            data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode="ro",
        )

    @pytest.mark.parametrize(
        "codewords",
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )
    @pytest.mark.parametrize("averages", [1, 200])
    @pytest.mark.parametrize("averaging_mode", ["sequential", "cyclic"])
    @pytest.mark.parametrize("poll", [False, True])
    def test_dio_sequence_rl_collects_all_data(
        self, setup, codewords, averages, averaging_mode, poll
    ):
        testee, cc = setup

        # Configure
        samples = len(codewords)
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode="rl", poll=poll
        )
        Iwaves, Qwaves = utils.make_split_waves(
            num_waves=len(codewords), wave_size=2000
        )
        testee.awg_sequence_acquisition_and_DIO_triggered_pulse(
            Iwaves=Iwaves,
            Qwaves=Qwaves,
            cases=codewords,
        )
        testee.acquisition_time(ACQUISITION_TIME)
        testee.averaging_mode(averaging_mode)

        # Execute
        testee.acquisition_arm()
        cc_program = utils.make_cc_program(
            repetitions=averages, codewords=codewords, averaging_mode=averaging_mode
        )
        cc.assemble_and_start(cc_program)

        # Collect
        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(
                    samples=samples, arm=False, acquisition_time=1
                )
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        testee.acquisition_finalize()

        utils.assert_expected_result_format(
            data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode="rl",
        )

        # Check that correct integrators have been triggered
        def was_triggered(measurement_value):
            return not np.isnan(value)

        for ch, integrators in data.items():
            for slot, values in integrators.items():
                codeword = ch * Dio.MAX_NUM_RESULTS_PER_CHANNEL + slot
                for i, value in enumerate(values):
                    if i == codeword:
                        assert was_triggered(value)
                    else:
                        assert not was_triggered(value)

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 100])
    @pytest.mark.parametrize("result_mode", ["ro", "rl"])
    @pytest.mark.parametrize("poll", [True, False])
    def test_acquisition_sequence_collects_all_data(
        self, setup, samples, averages, result_mode, poll
    ):
        testee, cc = setup

        # Configure
        testee.awg_sequence_acquisition(dly=WAIT_DLY)
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode=result_mode, poll=poll
        )
        testee.acquisition_time(ACQUISITION_TIME)

        # Execute
        testee.acquisition_arm()
        cc_program = utils.make_cc_program(
            repetitions=averages, codewords=[0] * samples
        )
        cc.assemble_and_start(cc_program)

        # Collect
        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(
                    samples=samples, arm=False, acquisition_time=1
                )
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        testee.acquisition_finalize()

        assert len(data) == 1
        utils.assert_expected_result_format(
            data=data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode=result_mode,
        )

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 100])
    @pytest.mark.parametrize("result_mode", ["ro", "rl"])
    @pytest.mark.parametrize("dio_trigger", [True, False])
    @pytest.mark.parametrize("poll", [True, False])
    def test_acquisition_pulse_sequence_collects_all_data(
        self,
        setup,
        samples,
        averages,
        result_mode,
        poll,
        dio_trigger,
    ):
        testee, cc = setup

        # Configure
        wave = [0.5] * 2000
        testee.awg_sequence_acquisition_and_pulse(
            Iwave=wave,
            Qwave=wave,
            acquisition_delay=WAIT_DLY,
            dig_trigger=dio_trigger,
            ch=0,
            slot=0,
        )
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode=result_mode, poll=poll
        )
        testee.acquisition_time(ACQUISITION_TIME)

        # Execute
        testee.acquisition_arm()
        if dio_trigger:
            cc_program = utils.make_cc_program(
                repetitions=averages, codewords=[0] * samples
            )
            cc.assemble_and_start(cc_program)

        # Collect
        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(
                    samples=samples, arm=False, acquisition_time=1
                )
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        testee.acquisition_finalize()

        assert len(data) == 1
        utils.assert_expected_result_format(
            data=data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode=result_mode,
        )

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 100])
    @pytest.mark.parametrize("averaging_mode", ["sequential", "cyclic"])
    @pytest.mark.parametrize("poll", [True, False])
    @pytest.mark.parametrize("dio_trigger", [True, False])
    def test_spectroscopy_collects_all_data(
        self,
        setup,
        samples,
        averages,
        poll,
        dio_trigger,
        averaging_mode,
    ):
        testee, cc = setup

        # Configure
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode="spectroscopy", poll=poll
        )
        testee.configure_spectroscopy(
            start_frequency=0,
            frequency_step=1e6,
            settling_time=WAIT_DLY,
            dio_trigger=dio_trigger,
        )
        testee.acquisition_time(ACQUISITION_TIME)
        testee.averaging_mode(averaging_mode)

        # Execute
        testee.acquisition_arm()
        if dio_trigger:
            cc_program = utils.make_cc_program(
                repetitions=averages, codewords=[0] * samples
            )
            cc.assemble_and_start(cc_program)

        # Collect
        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(
                    samples=samples, arm=False, acquisition_time=1
                )
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        testee.acquisition_finalize()

        assert len(data) == 1
        utils.assert_expected_result_format(
            data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode="spectroscopy",
        )
