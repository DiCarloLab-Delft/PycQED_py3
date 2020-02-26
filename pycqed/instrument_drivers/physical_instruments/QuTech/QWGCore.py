"""
    File:       QWGCore.py
    Author:     Wouter Vlothuizen, TNO/QuTech,
    Purpose:    Core Instrument driver for QuTech QWG, independent of QCoDeS.
                All instrument protocol handling is provided here
    Usage:      Can be used directly, or with QWG.py, which adds access via QCoDeS parameters
    Notes:      Here, we follow the SCPI convention of NOT checking parameter values but leaving that to
                the device
    Notes:      It is possible to view the QWG log using ssh. To do this:
                - connect using ssh e.g., "ssh root@192.168.0.10"
                - view log using "tail -f /var/log/qwg.log"
    Bugs:
                - requires QWG software version > 1.5.0, which isn't officially released yet

"""

import logging
import re
import numpy as np
from typing import Tuple, List

import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.library.SCPIBase import SCPIBase
from pycqed.instrument_drivers.library.Transport import Transport

log = logging.getLogger(__name__)

# FIXME: replace by info from DIO.py
# Codeword protocols: Pre-defined per channel bit maps
cw_protocols_dio = {
    # FIXME:
    #  - CCLight is limited to 8 cw bits output
    #  - QWG has 14 codeword bits input at the interface (+ trigger, toggle_ds). Out of these 14, 10 bits are
    #   selectable per channel
    'MICROWAVE': [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
        [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4

    'MICROWAVE_NO_VSM': [
        [0, 1, 2, 3, 4, 5, 6],  # Ch1
        [0, 1, 2, 3, 4, 5, 6],  # Ch2
        [7, 8, 9, 10, 11, 12, 13],  # Ch3
        [7, 8, 9, 10, 11, 12, 13]],  # Ch4

    'FLUX': [
        [0, 1, 2],  # Ch1
        [3, 4, 5],  # Ch2
        [6, 7, 8],  # Ch3
        [9, 10, 11]],  # Ch4  # See limitation/fixme; will use ch 3's bitmap
}

# Marker trigger protocols
# FIXME: which input is trigger? Do modes make sense?
cw_protocols_mt = {
    # Name
    'MICROWAVE': [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
        [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4

    'FLUX': [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
        [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4
}


##########################################################################
# class
##########################################################################

class QWGCore(SCPIBase, DIO.CalInterface):
    __doc__ = f"""
    Driver for a Qutech AWG Module (QWG) instrument. Will establish a connection to a module via ethernet.
    :param name: Name of the instrument  
    :param transport: Transport to use
    """

    ##########################################################################
    # 'public' functions for the end user
    ##########################################################################

    def __init__(self,
                 name: str,
                 transport: Transport):
        super().__init__(name, transport)

        # AWG properties
        self._dev_desc = lambda:0  # create empty device descriptor
        self._dev_desc.model = 'QWG'
        self._dev_desc.numChannels = 4
#        self._dev_desc.numDacBits = 12
#        self._dev_desc.numMarkersPerChannel = 2 # FIXME
#        self._dev_desc.numMarkers = 8 # FIXME
        self._dev_desc.numTriggers = 8  # FIXME: depends on IORear type

        # Check for driver / QWG compatibility
        version_min = (1, 5, 0)  # driver supported software version: Major, minor, patch

        if 0:  # FIXME: get_idn
            idn_firmware = self.get_idn()["firmware"]  # NB: called 'version' in QWG source code
            # FIXME: above will make usage of DummyTransport more difficult
            regex = r"swVersion=(\d).(\d).(\d)"
            sw_version = re.search(regex, idn_firmware)
            version_cur = (int(sw_version.group(1)), int(sw_version.group(2)), int(sw_version.group(3)))
            driver_outdated = True

            if sw_version and version_cur >= version_min:
                self._dev_desc.numSelectCwInputs = self.get_codewords_select()
                self._dev_desc.numMaxCwBits = self.get_max_codeword_bits()
                driver_outdated = False
            else:
                # FIXME: we could be less rude and only disable the new parameters
                # FIXME: let parameters depend on SW version, and on IORear type
                log.warning(f"Incompatible driver version of QWG ({self.name}); The version ({version_cur[0]}."
                                f"{version_cur[1]}.{version_cur[2]}) "
                                f"of the QWG software is too old and not supported by this driver anymore. Some instrument "
                                f"parameters will not operate and timeout. Please update the QWG software to "
                                f"{version_min[0]}.{version_min[1]}.{version_min[2]} or later")
                self._dev_desc.numMaxCwBits = 7
                self._dev_desc.numSelectCwInputs = 7
            self._dev_desc.numCodewords = pow(2, self._dev_desc.numSelectCwInputs)
        else:  # FIXME: hack
            self._dev_desc.numMaxCwBits = 7
            self._dev_desc.numSelectCwInputs = 7
            self._dev_desc.numCodewords = pow(2, self._dev_desc.numSelectCwInputs)

        if self._dev_desc.numMaxCwBits <= 7:    # FIXME: random constant
            self.codeword_protocols = cw_protocols_mt
        else:
            self.codeword_protocols = cw_protocols_dio

    ##########################################################################
    #  AWG control functions (AWG5014 compatible)
    ##########################################################################

    def start(self):
        """
        Activates output on channels with the current settings. When started this function will check for
        possible warnings
        """
        # run_mode = self.run_mode()
        # if run_mode == 'NONE':
        #     raise RuntimeError('No run mode is specified')
        self._transport.write('awgcontrol:run:immediate')

        #self._get_errors()

        # status = self.get_system_status()
        # warn_msg = self._detect_underdrive(status)
        # if(len(warn_msg) > 0):
        #     warnings.warn(', '.join(warn_msg))

    def stop(self):
        """
        Shutdown output on channels. When stopped will check for errors or overflow (FIXME: does it)
        """
        self._transport.write('awgcontrol:stop:immediate')

        #self._get_errors()

    def _get_errors(self):
        """
        The SCPI protocol by default does not return errors. Therefore the user needs
        to ask for errors. This function retrieves all errors and will raise them.
        """
        errNr = self.get_system_error_count()

        if errNr > 0:
            errMgs = []
            for _ in range(errNr):
                errMgs.append(self.get_error())
            raise RuntimeError(f'{repr(self)}: ' + ', '.join(errMgs))
            # FIXME: is raising a potentially very long string useful?

    ##########################################################################
    #  WLIST (Waveform list) functions (AWG5014 compatible)
    ##########################################################################

    def get_wlist_size(self) -> int:
        return self._ask_int('wlist:size?')

    def get_wlist_name(self, idx) -> str:
        """
        Args:
            idx(int): 0..size-1
        """
        return self._ask(f'wlist:name? {idx:d}')

    def get_wlist(self) -> List:
        """
        NB: takes a few seconds on 5014: our fault or Tek's?
        """
        size = self.get_wlist_size()
        wlist = []                                  # empty list
        for k in range(size):                       # build list of names
            wlist.append(self.get_wlist_name(k+1))
        return wlist

    def delete_waveform(self, name: str) -> None:
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            'test'
        """
        self._transport.write(f'wlist:waveform:delete "{name}"')

    def delete_waveform_all(self) -> None:
        self._transport.write('wlist:waveform:delete all')

    def get_waveform_type(self, name: str):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            'INT' or 'REAL'
        """
        return self._ask(f'wlist:waveform:type? "{name}"')

    def get_waveform_length(self, name: str):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'
        """
        return self._ask_int(f'wlist:waveform:length? "{name}"')

    def new_waveform_real(self, name: str, length: int):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        NB: seems to do nothing (on Tek5014) if waveform already exists
        """
        self._transport.write(f'wlist:waveform:new "{name}",{length:d},real')

    def get_waveform_data_float(self, name: str):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            waveform  (np.array of float): waveform data

        Compatibility: QWG
        """
        self._transport.write(f'wlist:waveform:data? "{name}"')
        bin_block = self.bin_block_read()
        waveform = np.frombuffer(bin_block, dtype=np.float32)  # extract waveform
        return waveform

    def send_waveform_data_real(self, name: str, waveform):
        """
        send waveform and markers directly to AWG memory, i.e. not to a file
        on the AWG disk.
        NB: uses real data normalized to the range from -1 to 1 (independent
        of number of DAC bits of AWG)

        Args:
            name (string): waveform name excluding double quotes, e.g. 'test'.
            Must already exist in AWG

            waveform (np.array of float)): vector defining the waveform,
            normalized between -1.0 and 1.0

        Compatibility:  QWG

        Based on:
            Tektronix_AWG5014.py::send_waveform, which sends data to an AWG
            _file_, not a memory waveform
            'awg_transferRealDataWithMarkers', Author = Stefano Poletto,
            Compatibility = Tektronix AWG5014, AWG7102
        """

        # generate the binblock
        arr = np.asarray(waveform, dtype=np.float32)
        bin_block = arr.tobytes()

        # write binblock
        hdr = f'wlist:waveform:data "{name}",'
        self.bin_block_write(bin_block, hdr)

    def create_waveform_real(self, name: str, waveform):
        """
        Convenience function to create a waveform in the AWG and then send
        data to it

        Args:
            name(string): name of waveform for internal use by the AWG

            waveform (float[numpoints]): vector defining the waveform,
            normalized between -1.0 and 1.0


        Compatibility:  QWG
        """
        # FIXME: disabled check
        # wv_val = vals.Arrays(min_value=-1, max_value=1)
        # wv_val.validate(waveform)

        # check length, because excessive lengths can overrun QWG SCPI buffer
        max_wave_len = 2**17-4  # NB: this is the hardware max
        wave_len = len(waveform)
        if wave_len > max_wave_len:
            raise ValueError(f'Waveform length ({wave_len}) must be < {max_wave_len}')

        self.new_waveform_real(name, wave_len)
        self.send_waveform_data_real(name, waveform)

    ##########################################################################
    # QWG specific
    ##########################################################################

    def sync_sideband_generators(self) -> None:
        """
        Synchronize both sideband generators, i.e. restart them with phase=0
        """
        self._transport.write('QUTEch:OUTPut:SYNCsideband')


    ##########################################################################
    # DIO support
    ##########################################################################

    def dio_calibrate(self, target_index: int = ''):
        # FIXME: cleanup docstring
        """
        Calibrate the DIO input signals.\n

        The QWG will analyze the input signals for each DIO input (used to transfer codeword bits), secondly,
        the most preferable index (active index) is set.\n\n

        Each signal is sampled and divided into sections. These sections are analyzed to find a stable
        signal. These stable sections are addressed by there index.\n\n

        After calibration the suitable indexes list (see dio_suitable_indexes()) contains all indexes which are stable.

        Parameters:
        :param target_index: unsigned int, optional: When provided the calibration will select an active index based
        on the target index. Used to determine the new index before or after the edge. This parameter is commonly used
        to calibrate a DIO slave where the target index is the active index after calibration of the DIO master

        Notes:
        \t- Expects a DIO calibration signal on the inputs where all codewords bits show activity (e.g. high followed \
        by all codeword bits low in a continuous repetition. This results in a square wave of 25 MHz on the DIO inputs \
        of the DIO connection).
        \t- Individual DIO inputs where no signal is detected will not be calibrated (See dio_calibrated_inputs())\n
        \t- The QWG will continuously validate if the active index is still stable.\n
        \t- If no suitable indexes are found FIXME is empty and an error is pushed onto the error stack\n
        """
        self._transport.write(f'DIO:CALibrate {target_index}')

        # FIXME: define relation with mode and #codewords in use
        # FIXME: provide high level function that performs the calibration

    def dio_suitable_indexes(self):
        """
        Get DIO all suitable indexes. The array is ordered by most preferable index first
        """
        return _int_to_array(self._ask('DIO:INDexes?'))


    def dio_calibrated_inputs(self) -> int:
        """
        'Get all DIO inputs which are calibrated\n'
        """
        return self._ask_int('DIO:INPutscalibrated?')

    def dio_lvds(self) -> bool:
        """
        Get the DIO LVDS connection status. Result:
             True: Cable detected
             No cable detected
        """
        return bool(self._ask_int('DIO:LVDS?'))

    def dio_interboard(self):
        """
        Get the DIO interboard status. Result:
             True:  To master interboard connection detected
             False: No interboard connection detected
        """
        return bool(self._ask_int('DIO:IB?'))

    def dio_calibration_report(self, extended: bool=False) -> str:
        """
        Return a string containing the latest DIO calibration report (successful and failed calibrations). Includes:
        selected index, dio mode, valid indexes, calibrated DIO bits and the DIO bitDiff table.
        :param extended: Adds more information about DIO: interboard and LVDS
        :return: String of DIO calibration rapport
        """
        info = f'- Calibrated:          {self.dio_is_calibrated()}\n' \
               f'- Mode:                {self.dio_mode()}\n' \
               f'- Selected index:      {self.dio_active_index()}\n' \
               f'- Suitable indexes:    {self.dio_suitable_indexes()}\n' \
               f'- Calibrated DIO bits: {bin(self.dio_calibrated_inputs())}\n' \
               f'- DIO bit diff table:\n{self._dio_bit_diff_table()}'

        if extended:
            info += f'- LVDS detected:       {self.dio_lvds()}\n' \
                    f'- Interboard detected: {self.dio_interboard()}'

        return info

    def get_max_codeword_bits(self) -> int:
        """
        Reads the maximum number of codeword bits for all channels
        """
        return self._ask_int("SYSTem:CODEwords:BITs?")

    def get_codewords_select(self) -> int:
        return self._ask_int("SYSTem:CODEwords:SELect?")

    def get_triggers_logic_input(self, ch: int) -> int:
        """
        Reads the current input values on the all the trigger inputs for a channel, after the bitSelect.
        Return:
            uint32 where trigger 1 (T1) is on the Least significant bit (LSB),
            T2 on the second bit after LSB, etc.
            For example, if only T3 is connected to a high signal, the return value is 4 (0b0000100)

            Note: To convert the return value to a readable binary output use: `print(\"{0:#010b}\".format(qwg.'
            'triggers_logic_input()))`')
        """
        return self._ask_int(f'QUTEch:TRIGgers{ch}:LOGIcinput?')


    def set_bitmap(self, ch: int) -> None:
        """
        Codeword bit map for a channel, 14 bits available of which 10 are selectable.
        The codeword bit map specifies which bits of the codeword (coming from a
        central controller) are used for the codeword of a channel. This allows to
        split up the codeword into sections for each channel
        FIXME: rewrite
        """
        self._transport.write(f'DAC{ch}:BITmap')

    def get_bitmap(self, ch: int) -> List:
        return _int_to_array(self._ask(f'DAC{ch}:BITmap?'))


    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        raise RuntimeError("QWG cannot output calibration data")

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        self.dio_calibrate()    # FIXME: integrate

    ##########################################################################
    # DAC calibration support
    ##########################################################################

    def get_iofront_temperature(self) -> float:
        """
        Reads the temperature of the IOFront board in Centigrade.
        Temperature measurement interval is ~10 seconds
        """
        return self._ask_float('STATus:FrontIO:TEMperature?')

    def get_fpga_temperature(self) -> float:
        """
        Reads the temperature of the FPGA in Centigrade.
        Temperature measurement interval is ~10 seconds
        """
        return self._ask_float('STATus:FPGA:TEMperature?')

    def get_dac_temperature(self, ch: int) -> float:
        """
        Reads the temperature of a DAC in Centigrade.
        Temperature measurement interval is ~10 seconds

        Args:
            ch: channel number [0..3]
        """
        return self._ask_float(f'STATus:DAC{ch}:TEMperature')

    def get_channel_output_voltage(self, ch: int) -> float:
        """
        Returns the output voltage measurement of a channel in [V].
        Only valid if the channel is disabled, i.e. .chX_state(False)

        :param ch: channel number [0..3]
        :return:
        """
        return self._ask_float(f'QUTEch:OUTPut{ch}:Voltage')

    def set_channel_gain_adjust(self, ch: int, ga: int) -> None:
        """
        Set gain adjust for the DAC of a channel.
        Used for calibration of the DAC. Do not use to set the gain of a channel

        :param ch: channel number [0..3]
        :param ga: gain setting from 0 to 4095 (0 V to 3.3V)
        """
        self._transport.write(f'DAC{ch}:GAIn:DRIFt:ADJust {ga:d}')

    def get_channel_gain_adjust(self, ch: int) -> int:
        """
        Get gain adjust for the DAC of a channel.

        :param ch: channel number [0..3]
        :return gain setting from 0 to 4095 (0 V to 3.3V)
        """
        self._ask_int(f'DAC{ch}:GAIn:DRIFt:ADJust?')

    def set_dac_digital_value(self, ch: int, val: int) -> None:
        """
        FOR DEVELOPMENT ONLY: Set a digital value directly into the DAC
        Notes:
        - This command will also set the internal correction matrix (Phase and amplitude) of the
        channel pair to [0,0,0,0], disabling any influence from the wave memory
        - This will also stop the wave on the other channel of the pair
        FIXME: change implementation: stop sequencer, use offset

        :param ch: channel number [0..3]
        :param val: DAC setting from 0 to 4095 (-FS to FS)
        """
        self._transport.write(f'DAC{ch}:DIGitalvalue {val}')

    ##########################################################################
    # private static helpers
    ##########################################################################

    @staticmethod
    def _detect_underdrive(status):
        """
        Will raise an warning if on a channel underflow is detected
        """
        msg = []
        for channel in status["channels"]:
            if(channel["on"] == True) and (channel["underdrive"] == True):
                msg.append(f"Possible wave underdrive detected on channel: {channel['id']}")
        return msg

    @staticmethod
    def _int_to_array(msg):
        """
        Convert a scpi array of ints into a python int array
        :param msg: scpi result
        :return: array of ints
        """
        if msg == '""':
            return []
        return msg.split(',')

    ##########################################################################
    # private DIO functions
    ##########################################################################

    def _dio_bit_diff_table(self):
        """
        FOR DEVELOPMENT ONLY: Get the bit diff table of the last calibration
        :return: String of the bitDiff table
        """
        return self._ask("DIO:BDT").replace("\"", '').replace(",", "\n")

    def _dio_calibrate_param(self, meas_time: float, nr_itr: int, target_index: int = ""):
        """
        FOR DEVELOPMENT ONLY: Calibrate the DIO input signals with extra arguments.\n
        Parameters:
        \t meas_time: Measurement time between indexes in seconds, resolution of 1e-6 s
        \tNote that when select a measurement time longer than 25e-2 S the scpi connection
        will timeout, but the calibration is than still running. The timeout will happen on the
        first `get` parameter after this call\n
        \tnr_itr: Number of DIO signal data (bitDiffs) gathering iterations\n
        \ttarget_index: DIO index which determines on which side of the edge to select the active index from\n
        Calibration duration = meas_time * nr_itr * 20 * 1.1 (10% to compensate for log printing time)\n
        """
        if meas_time < 1e-6:
            raise ValueError(f"Cannot calibration inputs: meas time is too low; min 1e-6, actual: {meas_time}")

        if nr_itr < 1:
            raise ValueError(f"Cannot calibration inputs: nr_itr needs to be positive; actual: {nr_itr}")

        if target_index is not "":
            target_index = f",{target_index}"

        self._transport.write(f'DIO:CALibrate:PARam {meas_time},{nr_itr}{target_index}')
