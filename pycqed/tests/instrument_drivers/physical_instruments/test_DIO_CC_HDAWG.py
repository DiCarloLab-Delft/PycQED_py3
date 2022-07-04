import logging
import pytest
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

    def init_CC(self, central_controller):
        central_controller.reset()
        central_controller.clear_status()
        central_controller.status_preset()

        central_controller.assemble_and_start("""mainLoop:
                        seq_out         0xFFFF,1 
                        seq_out         0x0,1
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
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/value", 0x3FF) #0x3FF
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/dio/mask/shift", 1)

        hd_awg_program = """
                    while (1) {{
                    const AWG_N = 1;
                    }}
                    """

        awgModule=HDAWG.daq.awgModule()
        awgModule.set("device", HDAWG.devname)
        HDAWG.configure_awg_from_string(0, hd_awg_program)
        awgModule.set("awg/enable", 1)

    def calibrate_HDAWG_dio(self, HDAWG):
        log.info(f"{HDAWG.devname}: Finding valid delays...")

        valid_delays = []
        for delay in range(16):
            log.info(f"{HDAWG.devname}: Testing delay {delay}.")
            print(delay)
            HDAWG._set_dio_calibration_delay(delay)
            HDAWG.daq.setInt(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
            time.sleep(3)
            timing_error = HDAWG.daq.getInt(
                f"/{HDAWG.devname}/raw/dios/0/error/timingsticky"
            )
            if timing_error == 0:
                valid_delays.append(delay)
                print("SUCCESS")

        if not valid_delays:
            raise Exception("DIO calibration failed! No valid delays found")

        log.info(f"{HDAWG.devname}: Valid delays are {valid_delays}")

        subseqs = [[valid_delays[0]]]
        for delay in valid_delays:
            last_subseq = subseqs[-1]
            last_delay = last_subseq[-1]
            delay_following_sequence = not last_subseq or last_delay == delay - 1
            if delay_following_sequence:
                subseqs[-1].append(delay)
            else:
                subseqs.append([delay])

        longest_subseq = max(subseqs, key=len)
        delay = len(longest_subseq) // 2 + longest_subseq[0]

        HDAWG._set_dio_calibration_delay(delay)
        print("Final:", delay)

        # Clear all detected errors (caused by DIO timing calibration)
        HDAWG.check_errors(errors_to_ignore=["AWGDIOTIMING"])

    def check_timing_error(self, HDAWG):
        """Check if the timing error is kept at 0 when we switch from
        calibration to walking 1s."""
        HDAWG.daq.set(f"/{HDAWG.devname}/raw/dios/0/error/timingclear", 1)
        time.sleep(1)
        timing_error = HDAWG.daq.getInt(f"/{HDAWG.devname}/raw/dios/0/error/timingsticky")
        assert (
                timing_error == 0
        ), f"Timing error was not kept at 0 after calibration, changed to {timing_error}"

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

    def verify_walking_ones_pattern_HD(self, HDAWG, dio_mask, highest_pattern_value):
        """Check if logged out DIO values on HD respect the walking 1s pattern."""
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/mode", 1)
        HDAWG.daq.setInt(f"/{HDAWG.devname}/awgs/0/rtlogger/starttimestamp", 0)
        path = f"/{HDAWG.devname}/awgs/0/rtlogger/data"
        start_time = time.time()
        seconds = 1
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



    def test_with_calib_HDAWG_dio(self):
        station = qc.Station()
        central_controller = CC("central_controller", IPTransport("192.168.0.241"))
        station.add_component(central_controller, update_snapshot=False)

        HDAWG = ZI_HDAWG8(
            name="hdawg",
            device="dev8066",
            interface="1GbE",
            server="localhost",
            port=8004
        )
        # HDAWG.daq.set(f"/{HDAWG.devname}/raw/system/restart", 1)

        start_time = time.time()
        seconds = 86400 #86400  # 24 hours
        current_time = time.time()
        elapsed_time = current_time - start_time
        while elapsed_time <= seconds:
            current_time = time.time()
            elapsed_time = current_time - start_time
            self.init_HDAWG(HDAWG)
            self.init_CC(central_controller)
            self.calibrate_HDAWG_dio(HDAWG)
            self.check_timing_error(HDAWG)

            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/valid/polarity", 2)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/valid/index", 0)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/mask/shift", 0)
            HDAWG.daq.set(f"/{HDAWG.devname}/awgs/0/dio/mask/value", 0x0)

            self.check_timing_error(HDAWG)
            self.run_walking_ones_pattern(central_controller)
            self.verify_walking_ones_pattern_HD(HDAWG, 0xFFFF, 0x8001)

    def test_calib_HDAWG_dio(self):
        station = qc.Station()
        central_controller = CC("central_controller", IPTransport("192.168.0.241"))
        station.add_component(central_controller, update_snapshot=False)

        self.init_CC(central_controller)

        HDAWG = ZI_HDAWG8(
            name="hdawg",
            device="dev8066",
            interface="1GbE",
            server="localhost",
            port=8004
        )
        self.calibrate_HDAWG_dio(HDAWG)
        self.check_timing_error(HDAWG)