import numpy as np
import pytest
import logging
import time

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument import (
    ziValueError,
)
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa import SHFQA
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility import (
    Dio,
)

import utils

log = logging.getLogger(__name__)

try:
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
except BaseException as e:
    pytestmark = pytest.mark.skip(
        f"Could not connect to devices; skipping SHFQA tests. "
        f"Exception that gave rise to this skip: {str(e)}"
    )

ACQUISITION_TIME = 20e-9
WAIT_DLY = 100e-9


class TestStandaloneShfqa:
    """
    This test suite is meant to be run using a standalone SHFQA device.
    """

    @pytest.fixture
    def testee(self):
        utils.reset(driver)
        return driver

    ##########################################################################
    # Test that waveforms are correctly uploaded
    ##########################################################################

    @pytest.mark.parametrize(
        "input_waveform_size, expected_upload_waveform_size, expect_throw",
        [(16, 16, False), (1001, 1004, False), (4096, 4096, False), (4097, None, True)],
    )
    def test_dio_waveform_upload_via_sequence_helper(
        self, testee, input_waveform_size, expected_upload_waveform_size, expect_throw
    ):
        codewords = range(Dio.MAX_NUM_RESULTS)
        Iwaves, Qwaves = utils.make_split_waves(
            num_waves=len(codewords), wave_size=input_waveform_size
        )

        if expect_throw:
            with pytest.raises(ziValueError):
                testee.awg_sequence_acquisition_and_DIO_triggered_pulse(
                    Iwaves=Iwaves,
                    Qwaves=Qwaves,
                    cases=codewords,
                )
            return
        testee.awg_sequence_acquisition_and_DIO_triggered_pulse(
            Iwaves=Iwaves,
            Qwaves=Qwaves,
            cases=codewords,
        )

        expected_waves = []
        for codeword in range(Dio.MAX_NUM_RESULTS):
            expected = np.array(
                [
                    Iwaves[codeword][i] + 1j * Qwaves[codeword][i]
                    if i < input_waveform_size
                    else 0
                    for i in range(expected_upload_waveform_size)
                ]
            )
            expected_waves.append(expected)

            # QCoDes waveforms in sync
            parameter = "wave_cw{:03}".format(codeword)
            waveform = testee.get(parameter)
            utils.assert_complex_data_equal(expected, waveform)

        driver.push_to_device()

        for i, codeword in enumerate(codewords):
            # Actual data uploaded to generators
            ch = codeword // Dio.MAX_NUM_RESULTS_PER_CHANNEL
            slot = codeword % Dio.MAX_NUM_RESULTS_PER_CHANNEL
            generator_path = (
                f"/{testee.devname}/qachannels/{ch}/generator/waveforms/{slot}/wave"
            )
            vector = testee.daq.get(generator_path, flat=True)[generator_path][0][
                "vector"
            ]
            utils.assert_complex_data_equal(expected_waves[i], vector)

    @pytest.mark.parametrize(
        "input_waveform_size, expected_upload_waveform_size, expect_throw",
        [(16, 16, False), (1001, 1004, False), (4096, 4096, False), (4097, None, True)],
    )
    def test_dio_waveform_upload_via_qcodes(
        self, testee, input_waveform_size, expected_upload_waveform_size, expect_throw
    ):
        codewords = range(Dio.MAX_NUM_RESULTS)
        waves = utils.make_monotonically_increasing_const_waves(
            num_waves=len(codewords), wave_size=input_waveform_size
        )

        expected_waves = []
        for codeword in codewords:
            parameter = "wave_cw{:03}".format(codeword)
            if expect_throw:
                with pytest.raises(ziValueError):
                    testee.set(parameter, waves[codeword])
                return

            testee.set(parameter, waves[codeword])
            expected = np.array(
                [
                    waves[codeword][i] if i < input_waveform_size else 0
                    for i in range(expected_upload_waveform_size)
                ]
            )
            expected_waves.append(expected)
            # QCoDes waveforms in sync
            received = testee.get(parameter)
            utils.assert_complex_data_equal(expected, received)

        driver.cases(codewords)
        driver.push_to_device()

        for i, codeword in enumerate(codewords):
            # Actual data uploaded to generators
            ch = codeword // Dio.MAX_NUM_RESULTS_PER_CHANNEL
            slot = codeword % Dio.MAX_NUM_RESULTS_PER_CHANNEL
            generator_path = (
                f"/{testee.devname}/qachannels/{ch}/generator/waveforms/{slot}/wave"
            )
            vector = testee.daq.get(generator_path, flat=True)[generator_path][0][
                "vector"
            ]
            utils.assert_complex_data_equal(expected_waves[i], vector)

    @pytest.mark.parametrize("ch, slot", [(0, 0), (1, 7)])
    def test_acquisition_and_pulse_waveform_upload(self, testee, ch, slot):
        wave = utils.make_const_wave(amplitude=0.5, size=1024)
        testee.awg_sequence_acquisition_and_pulse(
            Iwave=wave.real,
            Qwave=wave.imag,
            acquisition_delay=WAIT_DLY,
            ch=ch,
            slot=slot,
        )

        generator_path = (
            f"/{testee.devname}/qachannels/{ch}/generator/waveforms/{slot}/wave"
        )
        vector = testee.daq.get(generator_path, flat=True)[generator_path][0]["vector"]

        utils.assert_complex_data_equal(wave, vector)

    ##########################################################################
    # Test that uploaded sequencer programs are the expected ones
    ##########################################################################

    def test_dio_sequencer_program(self, testee):
        codewords = range(Dio.MAX_NUM_RESULTS)
        testee.cases(codewords)
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=0)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
var num_errors = 0;
setUserReg(3, 0);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	waitDIOTrigger();
	var codeword = (getDIOTriggered() & 0b1111111111111100000000000000000);
	switch(codeword)
	{
		case 0b0:
		{
			startQA(QA_GEN_0, QA_INT_0, true,  0, 0x0);
		}
		case 0b100000000000000000:
		{
			startQA(QA_GEN_1, QA_INT_1, true,  0, 0x0);
		}
		case 0b1000000000000000000:
		{
			startQA(QA_GEN_2, QA_INT_2, true,  0, 0x0);
		}
		case 0b1100000000000000000:
		{
			startQA(QA_GEN_3, QA_INT_3, true,  0, 0x0);
		}
		case 0b10000000000000000000:
		{
			startQA(QA_GEN_4, QA_INT_4, true,  0, 0x0);
		}
		case 0b10100000000000000000:
		{
			startQA(QA_GEN_5, QA_INT_5, true,  0, 0x0);
		}
		case 0b11000000000000000000:
		{
			startQA(QA_GEN_6, QA_INT_6, true,  0, 0x0);
		}
		case 0b11100000000000000000:
		{
			startQA(QA_GEN_7, QA_INT_7, true,  0, 0x0);
		}
		case 0b100000000000000000000:
		{
			startQA(QA_GEN_8, QA_INT_8, true,  0, 0x0);
		}
		case 0b100100000000000000000:
		{
			startQA(QA_GEN_9, QA_INT_9, true,  0, 0x0);
		}
		case 0b101000000000000000000:
		{
			startQA(QA_GEN_10, QA_INT_10, true,  0, 0x0);
		}
		case 0b101100000000000000000:
		{
			startQA(QA_GEN_11, QA_INT_11, true,  0, 0x0);
		}
		case 0b110000000000000000000:
		{
			startQA(QA_GEN_12, QA_INT_12, true,  0, 0x0);
		}
		case 0b110100000000000000000:
		{
			startQA(QA_GEN_13, QA_INT_13, true,  0, 0x0);
		}
		default:
		{
			num_errors += 1;
		}
	}
}
setUserReg(3, num_errors);
"""
        )

    def test_dio_sequencer_program_2(self, testee):
        codewords = [0, 3, 9]
        testee.cases(codewords)
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=0)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
var num_errors = 0;
setUserReg(3, 0);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	waitDIOTrigger();
	var codeword = (getDIOTriggered() & 0b1111111111111100000000000000000);
	switch(codeword)
	{
		case 0b0:
		{
			startQA(QA_GEN_0, QA_INT_0, true,  0, 0x0);
		}
		case 0b1100000000000000000:
		{
			startQA(QA_GEN_1, QA_INT_1, true,  0, 0x0);
		}
		case 0b100100000000000000000:
		{
			startQA(QA_GEN_2, QA_INT_2, true,  0, 0x0);
		}
		default:
		{
			num_errors += 1;
		}
	}
}
setUserReg(3, num_errors);
"""
        )

    def test_dio_sequencer_program_3(self, testee):
        codewords = [0, 3, 9, 13, 2, 4, 5, 7, 8]
        testee.cases(codewords)
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=0)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
var num_errors = 0;
setUserReg(3, 0);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	waitDIOTrigger();
	var codeword = (getDIOTriggered() & 0b1111111111111100000000000000000);
	switch(codeword)
	{
		case 0b0:
		{
			startQA(QA_GEN_0, QA_INT_0, true,  0, 0x0);
		}
		case 0b1100000000000000000:
		{
			startQA(QA_GEN_1, QA_INT_1, true,  0, 0x0);
		}
		case 0b100100000000000000000:
		{
			startQA(QA_GEN_2, QA_INT_2, true,  0, 0x0);
		}
		case 0b110100000000000000000:
		{
			startQA(QA_GEN_3, QA_INT_3, true,  0, 0x0);
		}
		case 0b1000000000000000000:
		{
			startQA(QA_GEN_4, QA_INT_4, true,  0, 0x0);
		}
		case 0b10000000000000000000:
		{
			startQA(QA_GEN_5, QA_INT_5, true,  0, 0x0);
		}
		case 0b10100000000000000000:
		{
			startQA(QA_GEN_6, QA_INT_6, true,  0, 0x0);
		}
		case 0b11100000000000000000:
		{
			startQA(QA_GEN_7, QA_INT_7, true,  0, 0x0);
		}
		case 0b100000000000000000000:
		{
			startQA(QA_GEN_8, QA_INT_8, true,  0, 0x0);
		}
		default:
		{
			num_errors += 1;
		}
	}
}
setUserReg(3, num_errors);
"""
        )

    @pytest.mark.parametrize("ch, slot", [(0, 0), (1, 5)])
    def test_acquisition_pulse_sequencer_program(self, testee, ch, slot):
        wave = utils.make_const_wave(amplitude=0.5, size=2000)
        testee.awg_sequence_acquisition_and_pulse(
            Iwave=wave.real,
            Qwave=wave.imag,
            acquisition_delay=WAIT_DLY,
            dig_trigger=False,
            ch=ch,
            slot=slot,
        )
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=ch)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	startQA("""
            + f"QA_GEN_{slot}, QA_INT_{slot}"
            + """, true,  0, 0x0);
}
"""
        )

        testee.awg_sequence_acquisition_and_pulse(
            Iwave=wave.real,
            Qwave=wave.imag,
            acquisition_delay=WAIT_DLY,
            dig_trigger=True,
            ch=ch,
            slot=slot,
        )
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=ch)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	waitDIOTrigger();
	startQA("""
            + f"QA_GEN_{slot}, QA_INT_{slot}"
            + """, true,  0, 0x0);
}
"""
        )

    @pytest.mark.parametrize("ch, slot", [(0, 0), (1, 5)])
    def test_acquisition_sequencer_program(self, testee, ch, slot):
        testee.awg_sequence_acquisition(ch=ch, slot=slot)
        testee.push_to_device()

        assert (
            utils.get_uploaded_sequencer_program(testee.daq, testee.devname, ch=ch)
            == """var inner_loop_size = getUserReg(0);
var holdoff_delay = getUserReg(2);
repeat (inner_loop_size)
{
	playZero(holdoff_delay);
	waitDIOTrigger();
	startQA(0, """
            + f"QA_INT_{slot}"
            + """, true,  0, 0x0);
}
"""
        )

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 5])
    @pytest.mark.parametrize("result_mode", ["ro", "rl"])
    @pytest.mark.parametrize("poll", [True, False])
    def test_acquisition_sequence_collects_all_data(
        self, testee, samples, averages, result_mode, poll
    ):
        testee.awg_sequence_acquisition(dly=WAIT_DLY)
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode=result_mode, poll=poll
        )

        testee.acquisition_time(ACQUISITION_TIME)
        testee.acquisition_arm()

        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(samples=samples, arm=False)
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        assert len(data) == 1
        utils.assert_expected_result_format(
            data=data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode=result_mode,
        )

    @pytest.mark.parametrize("codewords", [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]])
    @pytest.mark.parametrize("samples", [1, 3])
    @pytest.mark.parametrize("averages", [1, 3])
    @pytest.mark.parametrize("result_mode", ["ro", "rl"])
    @pytest.mark.parametrize("poll", [True, False])
    def test_dio_sequence_collects_all_data(
        self, testee, codewords, samples, averages, result_mode, poll
    ):
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode=result_mode, poll=poll
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

        for codeword in codewords:
            log.info(f"codeword {codeword}")
            testee.daq.syncSetInt(
                f"/{testee.devname}/dios/0/output",
                codeword << utils.STANDALONE_CODEWORD_SHIFT,
            )
            time.sleep(0.1)
            testee.acquisition_arm()
            time.sleep(0.1)

            with utils.no_raise(TimeoutError):
                if poll:
                    data = testee.acquisition_poll(samples=samples, arm=False)
                else:
                    data = testee.acquisition_get(samples=samples, arm=False)

            utils.assert_expected_result_format(
                data,
                samples=samples,
                acquisition_time=testee._acquisition_time,
                result_mode=result_mode,
            )

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 5])
    @pytest.mark.parametrize("result_mode", ["ro", "rl"])
    @pytest.mark.parametrize("dio_trigger", [True, False])
    @pytest.mark.parametrize("poll", [True, False])
    def test_acquisition_pulse_sequence_collects_all_data(
        self,
        testee,
        samples,
        averages,
        result_mode,
        poll,
        dio_trigger,
    ):
        Iwaves, Qwaves = utils.make_split_waves(num_waves=1, wave_size=2000)
        testee.awg_sequence_acquisition_and_pulse(
            Iwave=Iwaves[0],
            Qwave=Qwaves[0],
            acquisition_delay=WAIT_DLY,
            dig_trigger=dio_trigger,
            ch=0,
            slot=0,
        )
        testee.acquisition_initialize(
            samples=samples, averages=averages, mode=result_mode, poll=poll
        )
        testee.acquisition_time(ACQUISITION_TIME)

        testee.acquisition_arm()

        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(samples=samples, arm=False)
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        assert len(data) == 1
        utils.assert_expected_result_format(
            data=data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode=result_mode,
        )

    @pytest.mark.parametrize("samples", [1, 10])
    @pytest.mark.parametrize("averages", [1, 5])
    @pytest.mark.parametrize("averaging_mode", ["sequential", "cyclic"])
    @pytest.mark.parametrize("poll", [True, False])
    @pytest.mark.parametrize("dio_trigger", [True, False])
    def test_spectroscopy_collects_all_data(
        self,
        testee,
        samples,
        averages,
        poll,
        dio_trigger,
        averaging_mode,
    ):
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
        testee.acquisition_arm()

        with utils.no_raise(TimeoutError):
            if poll:
                data = testee.acquisition_poll(samples=samples, arm=False)
            else:
                data = testee.acquisition_get(samples=samples, arm=False)

        assert len(data) == 1
        utils.assert_expected_result_format(
            data,
            samples=samples,
            acquisition_time=testee._acquisition_time,
            result_mode="spectroscopy",
        )
