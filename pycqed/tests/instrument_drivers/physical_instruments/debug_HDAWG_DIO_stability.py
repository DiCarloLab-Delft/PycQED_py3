import logging
import pytest
import numpy as np
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.library.DIO import calibrate
import time
import qcodes as qc
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

class TestDIODataTransfer(object):
    ##########################################################################
    # Configure CC
    ##########################################################################
    # Calibration pattern
    def init_CC(self, central_controller):
        central_controller.reset()
        central_controller.clear_status()
        central_controller.status_preset()

        central_controller.assemble_and_start("""mainLoop:
                        seq_out         0xF0FFFFFF,1 
                        seq_out         0x0,1
                        jmp             @mainLoop
                        """)

    def run_walking_ones_pattern(self, central_controller):
        central_controller.assemble_and_start("""mainLoop:
                        seq_out         0x0,1
                        seq_out         0x1,1
                        seq_out         0x2,1
                        seq_out         0x3,1
                        seq_out         0x4,1
                        seq_out         0x5,1
                        seq_out         0x8,1
                        seq_out         0x9,1
                        seq_out         0x10,1
                        seq_out         0x11,1
                        seq_out         0x20,1
                        seq_out         0x21,1
                        seq_out         0x40,1
                        seq_out         0x41,1
                        seq_out         0x80,1
                        seq_out         0x81,1
                        seq_out         0x100,1
                        seq_out         0x101,1
                        seq_out         0x200,1
                        seq_out         0x201,1
                        seq_out         0x400,1
                        seq_out         0x401,1
                        seq_out         0x800,1
                        seq_out         0x801,1
                        seq_out         0x1000,1
                        seq_out         0x1001,1
                        seq_out         0x2000,1
                        seq_out         0x2001,1
                        seq_out         0x4000,1
                        seq_out         0x4001,1
                        seq_out         0x8000,1
                        seq_out         0x8001,1
                        jmp             @mainLoop
                        """)

    def run_random_codeword_pattern(self, central_controller):
        central_controller.assemble_and_start("""mainLoop:
                        seq_out         0x2b41678,1
                        seq_out         0x7654321,1
                        seq_out         0xffffeee,1
                        seq_out         0x0100001,1
                        seq_out         0xaaa2aaa,1
                        seq_out         0x155d555,1
                        seq_out         0xb98a148,1
                        seq_out         0xfada623,1
                        jmp             @mainLoop
                        """)

    ##########################################################################
    # Configure HDAWG
    ##########################################################################
    def init_HDAWG(self, HDAWG):
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/time", 0)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/dios/0/mode", 2)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/dios/0/interface", 1)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/system/clocks/referenceclock/source", 1)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/single", 1)

        HDAWG.daq.setInt(f"/{HDAWG.devname}/dios/0/drive", 15)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/dios/0/drive", 0)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/valid/index", 0)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/valid/polarity", 2)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/value", 0x3FF)  # 0x3FF
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/shift", 1)

        hd_awg_program = """
                    while (1) {{
                    const AWG_N = 1;
                    }}
                    """

        awgModule = HDAWG.daq.awgModule()
        awgModule.set("device", HDAWG.devname)
        HDAWG.configure_awg_from_string(0, hd_awg_program)
        awgModule.set("awg/enable", 1)

    ##########################################################################
    # Calibrate HDAWG
    ##########################################################################
    def initial_calibration(self, HDAWG, central_controller, dio_mask, specific_delay):
        log.info(f"{HDAWG.devname}: Reclocking and calibrating until delay {specific_delay} is valid...")
        HDAWG._set_dio_calibration_delay(specific_delay)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
        time.sleep(3)
        timing_error = HDAWG.daq.getInt(
            f"/{HDAWG.devname}/raw/dios/0/error/timingsticky"
        ) # & dio_mask
        while(timing_error != 0):
            HDAWG.daq.setDouble(f"/{HDAWG.devname}/system/clocks/referenceclock/source", 0)
            time.sleep(3)
            HDAWG.daq.setDouble(f"/{HDAWG.devname}/system/clocks/referenceclock/source", 1)
            time.sleep(3)
            HDAWG._set_dio_calibration_delay(specific_delay)
            HDAWG.daq.setInt(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
            time.sleep(3)
            timing_error = HDAWG.daq.getInt(
                f"/{HDAWG.devname}/raw/dios/0/error/timingsticky"
            )  # & dio_mask
            print(timing_error)
        log.info(f"{HDAWG.devname}: Delay {specific_delay} returns 0 timing error")

    def checking_valid_delays_HDAWG_dio(self, HDAWG, central_controller, dio_mask, specific_delay):
        log.info(f"{HDAWG.devname}: Finding valid delays...")
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/valid/index", 0)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/valid/polarity", 2)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/value", 0x3FF)  # 0x3FF
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/shift", 1)

        valid_delays = []
        for delay in range(16):
            log.info(f"{HDAWG.devname}: Testing delay {delay}.")
            print(delay)
            HDAWG._set_dio_calibration_delay(delay)
            HDAWG.daq.setInt(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
            time.sleep(3)
            timing_error = HDAWG.daq.getInt(
                f"/{HDAWG.devname}/raw/dios/0/error/timingsticky"
            )  # & dio_mask
            if timing_error == 0:
                valid_delays.append(1)
                print("SUCCESS")
            else:
                valid_delays.append(0)

        if not valid_delays:
            raise Exception("DIO calibration failed! No valid delays found")

        log.info(f"{HDAWG.devname}: Valid delays are {valid_delays}")

        # Clear all detected errors (caused by DIO timing calibration)
        HDAWG.check_errors(errors_to_ignore=["AWGDIOTIMING"])

        # Set back specific delay and clear errors
        HDAWG._set_dio_calibration_delay(specific_delay)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)

        return valid_delays

    def check_timing_error(self, HDAWG, central_controller, dio_mask, pattern):
        """Check if the timing error is kept at 0 when we switch from
        calibration to walking 1s."""
        HDAWG.daq.set(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
        time.sleep(2)
        timing_error = HDAWG.daq.getInt(f"/{HDAWG.devname}/raw/dios/0/error/timingsticky") # & dio_mask
        assert (
                timing_error == 0
        ), f"Timing error was not kept at 0 with {pattern} pattern, changed to {timing_error}"

    def verify_walking_ones_pattern_HD(self, HDAWG, dio_mask, highest_pattern_value):
        """Check if logged out DIO values on HD respect the walking 1s pattern."""
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/mode", 1)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/starttimestamp", 0)
        path = f"/{HDAWG.devname}/awgs/0/rtlogger/data"
        start_time = time.time()
        seconds = 300
        vector_d = []
        current_time = time.time()
        elapsed_time = current_time - start_time
        while elapsed_time <= seconds:
            current_time = time.time()
            elapsed_time = current_time - start_time
            HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/enable", 1)
            time.sleep(0.01)
            vector_d = HDAWG.daq.get(path, settingsonly=False, flat=True)[path][0]["vector"]
            HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/enable", 0)
            HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/clear", 1)
            data = vector_d.tolist()
            data = data[1::2]

            for i in range(len(data) - 1):
                # print(data[i+1] & dio_mask)
                assert (
                        ((data[i + 1] & dio_mask) == (data[i] & dio_mask) * 2 - 1)
                        or ((data[i + 1] & dio_mask) == 3 and (data[i] & dio_mask) == 1)
                        or ((data[i + 1] & dio_mask) == 1 and (data[i] & dio_mask) == highest_pattern_value)
                ), f"consecutive values {data[i]} and {data[i + 1]} do not respect the walking ones pattern"

    def check_parity_error(self, HDAWG):
        errors_dio = HDAWG.daq.getInt(f"{HDAWG.devname}/raw/dios/0/parity/dio/errors")
        errors_ctrl = HDAWG.daq.getInt(f"{HDAWG.devname}/raw/dios/0/parity/controller/errors")
        errors_slv = HDAWG.daq.getInt(f"{HDAWG.devname}/raw/dios/0/parity/processing/errors")
        # assert (
        #         errors_dio == 0 and errors_ctrl == 0 and errors_slv == 0
        # ), f"Errors detected (DIO:{errors_dio} Master: {errors_ctrl} Slave: {errors_slv}) at DIO FPGA parity checker."
        return [errors_dio, errors_ctrl, errors_slv]

    def clear_all_parity_errors_HD(self, HDAWG):
        """Clear all three DIO parity error counters."""
        HDAWG.daq.setInt(f"{HDAWG.devname}/raw/dios/0/parity/dio/clear", 1)
        HDAWG.daq.setInt(f"{HDAWG.devname}/raw/dios/0/parity/controller/clear", 1)
        HDAWG.daq.setInt(f"{HDAWG.devname}/raw/dios/0/parity/processing/clear", 1)

    def update_plot_delay(self, valid_delays_matrix):
        valid_delays_plot = np.array(valid_delays_matrix)
        plt.imshow(valid_delays_plot, cmap="RdYlGn", interpolation=None)
        plt.xlabel("Calibration - Delays")
        plt.xticks(range(16), ["0", "1", "2","3", "4", "5","6", "7", "8","9", "10", "11","12", "13", "14", "15"])
        plt.savefig("valid_delays_matrix.png")

    def update_plot_parity_error(self, parity_error_matrix):
        parity_error_plot = np.array(parity_error_matrix)
        plt.imshow(parity_error_plot, cmap="YlOrRd",interpolation=None)
        plt.xlabel("Parity Errors")
        plt.xticks(range(3), ["DIO FPGA", "Controller", "Processing"])
        plt.savefig("parity_error_matrix.png")

    ##########################################################################
    # Debugging test
    ##########################################################################
    def test_stability_HDAWG_dio(self):
        # Choose the HD
        # HD_device_id = "dev8066"
        HD_device_id = "dev8622"

        # Set up CC
        station = qc.Station()
        central_controller = CC("central_controller", IPTransport("192.168.0.241"))
        station.add_component(central_controller, update_snapshot=False)
        # Set up HDAWG
        HDAWG = ZI_HDAWG8(
            name="hdawg",
            device=HD_device_id,
            interface="1GbE",
            server="localhost",
            port=8004
        )
        # LVDS mode, masking for lower 16 bits
        dio_mask = 0xFFFF
        max_codeword = 0x8001
        # Specify timing delay
        specific_delay = 6
        valid_delays_matrix = []
        parity_error_matrix = []


        self.init_HDAWG(HDAWG)
        self.init_CC(central_controller)
        self.initial_calibration(HDAWG, central_controller, dio_mask, specific_delay)

        fig = plt.figure(figsize=(25,400))
        plt.ylabel("Iterations")

        # Set up time period for running test
        start_time = time.time()
        seconds = 57600  #16 hours
        current_time = time.time()
        elapsed_time = current_time - start_time
        while elapsed_time <= seconds:
            current_time = time.time()
            elapsed_time = current_time - start_time

            self.init_CC(central_controller)

            # Store valid delays found to check how they change over time and plot
            valid_delays = self.checking_valid_delays_HDAWG_dio(HDAWG, central_controller, dio_mask, specific_delay)
            valid_delays_matrix.append(valid_delays)
            self.update_plot_delay(valid_delays_matrix)

            # Check once more for timing error
            self.check_timing_error(HDAWG, central_controller, dio_mask, pattern = "calib")

            # Switch to walking ones
            # self.run_walking_ones_pattern(central_controller)

            # Switch to random pattern
            self.run_random_codeword_pattern(central_controller)

            # Clear all parity errors after delay checking
            self.clear_all_parity_errors_HD(HDAWG)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/valid/polarity", 2)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/valid/index", 0)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/mask/shift", 0)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/mask/value", 0x0)
            self.check_timing_error(HDAWG, central_controller, dio_mask, pattern="walking 1s")

            # Verify walking ones pattern
            # self.verify_walking_ones_pattern_HD(HDAWG, dio_mask, max_codeword)

            # Wait for 5 mins
            time.sleep(300)

            # Check for parity errors
            parity_error = self.check_parity_error(HDAWG)
            parity_error_matrix.append(parity_error)
            self.update_plot_parity_error(parity_error_matrix)

        print(f"Valid delays were the following: {valid_delays_matrix}")
        print(f"Parity errors were the following: {parity_error_matrix}")

