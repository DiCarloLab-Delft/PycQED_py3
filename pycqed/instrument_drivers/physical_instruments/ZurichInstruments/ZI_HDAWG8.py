import time
import logging
import os
import numpy as np
from . import zishell_NH as zs
from qcodes.utils import validators as vals
from .ZI_base_instrument import ZI_base_instrument
from qcodes.instrument.parameter import ManualParameter

import ctypes
from ctypes.wintypes import MAX_PATH


class ZI_HDAWG8(ZI_base_instrument):
    """
    This is PycQED/QCoDeS driver driver for the Zurich Instruments HD AWG-8.

    Parameter files are generated from the python API of the instrument
    using the "create_parameter_file" method in the ZI_base_instrument class.
    These are used to add parameters to the instrument.
    """

    def __init__(self,
                 name,
                 device: str,
                 server: str = 'localhost',
                 port=8004,
                 num_codewords: int = 32,
                 interface='USB',
                 **kw):
        '''
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            server          (str) the ZI data server
            port            (int) the port to connect to
        '''
        t0 = time.time()
        self._num_channels = 8
        self._num_codewords = num_codewords

        if os.name == 'nt':
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                _basedir = buf.value
            else:
                logging.warning('Could not extract my documents folder')
        else:
            _basedir = os.path.expanduser('~')
        self.lab_one_webserver_path = os.path.join(
            _basedir, 'Zurich Instruments', 'LabOne', 'WebServer')

        super().__init__(name=name, **kw)
        self._devname = device
        self._dev = zs.ziShellDevice()
        self._dev.connect_server(server, port)
        print("Trying to connect to device {}".format(self._devname))
        self._dev.connect_device(self._devname, interface)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_fn = os.path.join(dir_path, 'zi_parameter_files')

        try:
            self.add_parameters_from_file(
                filename=os.path.join(base_fn, 'node_doc_HDAWG8.json'))

        except FileNotFoundError:
            logging.warning("parameter file for data parameters"
                            " {} not found".format(self._d_file_name))
        self.add_ZIshell_device_methods_to_instrument()

        self._add_codeword_parameters()
        self._add_extra_parameters()
        self.set_default_values()

        self.connect_message(begin_time=t0)

    def set_default_values(self):
        for i in range(8):
            self._dev.seti('/{}/triggers/out/{}/source'.format(self._devname, i), 4 + 2*(i%2))
        
    def _add_extra_parameters(self):
        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=10,
            parameter_class=ManualParameter)
        self.add_parameter(
            'cfg_num_codewords',
            label='Number of used codewords',
            docstring=(
                'This parameter is used to determine how many codewords to '
                'upload in "self.upload_codeword_program".'),
            initial_value=self._num_codewords,
            # N.B. I have commentd out numbers larger than self._num_codewords
            # see also issue #358
            vals=vals.Enum(2, 4, 8, 16, 32),  # , 64, 128, 256, 1024),
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_codeword_protocol',
            initial_value='identical',
            vals=vals.Enum('identical', 'microwave', 'flux'),
            docstring=(
                'Used in the configure codeword method to determine what DIO'
                ' pins are used in for which AWG numbers.'),
            parameter_class=ManualParameter)

    def snapshot_base(self, update=False, params_to_skip_update=None):
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update)
        return snap

    def add_ZIshell_device_methods_to_instrument(self):
        """
        Some methods defined in the zishell are convenient as public
        methods of the instrument. These are added here.
        """
        self.reconnect = self._dev.reconnect
        self.restart_device = self._dev.restart_device
        self.poll = self._dev.poll
        self.sync = self._dev.sync
        self.configure_awg_from_string = self._dev.configure_awg_from_string
        self.read_from_scope = self._dev.read_from_scope
        self.restart_scope_module = self._dev.restart_scope_module
        self.restart_awg_module = self._dev.restart_awg_module

    def get_idn(self):
        idn_dict = {
            'vendor':
            'ZurichInstruments',
            'model':
            self._dev.daq.getByte('/{}/features/devtype'.format(
                self._devname)),
            'serial':
            self._devname,
            'firmware':
            int(self._dev.geti('system/fwrevision')),
            'fpga_firmware':
            int(self._dev.geti('system/fpgarevision'))
        }
        return idn_dict

    def stop(self):
        """
        Stops the program on all AWG's part of this AWG8 unit
        """
        for i in range(4):
            self.set('awgs_{}_enable'.format(i), 0)

    def start(self):
        """
        Starts the program on all AWG's part of this AWG8 unit
        """
        for i in range(4):
            self.set('awgs_{}_enable'.format(i), 1)

    def _check_protocol(self, awg):
        # TODO: Add docstrings and make this more clear (Oct 2017)
        mask = self._dev.geti('awgs/' + str(awg) +
                              '/dio/mask/value') << self._dev.geti(
                                  'awgs/' + str(awg) + '/dio/mask/shift')
        strobe = 1 << self._dev.geti('awgs/' + str(awg) + '/dio/strobe/index')
        valid = 1 << self._dev.geti('awgs/' + str(awg) + '/dio/valid/index')
        #print('Codeword mask: {:08x}'.format(mask))
        #print('Strobe mask  : {:08x}'.format(strobe))
        #print('Valid mask   : {:08x}'.format(valid))

        got_bits = 0
        got_strobe_re = False
        got_strobe_fe = False
        got_valid_re = False
        got_valid_fe = False
        last_strobe = None
        last_valid = None
        strobe_low = 0
        strobe_high = 0
        count_low = False
        count_high = False
        count_ok = True
        ok = 0
        data = self._dev.getv('awgs/' + str(awg) + '/dio/data')
        for n, d in enumerate(data):
            curr_strobe = (d & strobe) != 0
            curr_valid = (d & valid) != 0

            if count_high:
                if curr_strobe:
                    strobe_high += 1
                else:
                    if (strobe_low > 0) and (strobe_low != strobe_high):
                        count_ok = False

            if count_low:
                if not curr_strobe:
                    strobe_low += 1
                else:
                    if (strobe_high > 0) and (strobe_low != strobe_high):
                        count_ok = False

            if (last_strobe != None):
                if (curr_strobe and not last_strobe):
                    got_strobe_re = True
                    strobe_high = 0
                    count_high = True
                    count_low = False
                    index_high = n
                elif (not curr_strobe and last_strobe):
                    got_strobe_fe = True
                    strobe_low = 0
                    count_low = True
                    count_high = False
                    index_low = n

            if (last_valid != None) and (curr_valid and not last_valid):
                got_valid_re = True
            if (last_valid != None) and (not curr_valid and last_valid):
                got_valid_fe = True

            got_bits |= (d & mask)
            last_strobe = curr_strobe
            last_valid = curr_valid

        if got_bits != mask:
            ok |= 1
        if not got_strobe_re or not got_strobe_fe:
            ok |= 2
        if not got_valid_re or not got_valid_fe:
            ok |= 4
        if not count_ok:
            ok |= 8

        return ok

    def _print_check_protocol_error_message(self, ok):
        if ok & 1:
            print('Did not see all codeword bits toggling')
        if ok & 2:
            print('Did not see toggle bit toggling')
        if ok & 4:
            print('Did not see valid bit toggling')
        if ok & 8:
            print('Toggle bit is not symmetrical')

    def calibrate_dio(self):
        # TODO: add docstrings to make it more clear
        ok = True
        print("Switching to internal clock")
        # self.set('system_extclk', 0)
        self._dev.seti('system/clocks/referenceclock/source', 0)
        time.sleep(1)
        print("Switching to external clock")
        # self.set('system_extclk', 1)
        self._dev.seti('system/clocks/referenceclock/source', 1)
        time.sleep(1)
        print("Calibrating internal DIO delays...")
        for i in [1, 0, 2, 3]:  # strange order is necessary
            print('  Calibrating AWG {} '.format(i), end='')
            self._dev.seti('raw/awgs/*/dio/calib/start', 2)
            time.sleep(1)
            status = self._dev.geti('raw/awgs/' + str(i) + '/dio/calib/status')
            if (status != 0):
                print('[FAILURE]')
                ok = False
            else:
                print('[SUCCESS]')
        return ok

    def calibrate_dio_protocol(self):
        # TODO: add docstrings to make it more clear
        print("Checking DIO protocol...")
        done = 4 * [False]
        timeout = 5
        while not all(done):
            ok = True
            for i in range(4):
                print('  Checking AWG {} '.format(i), end='')
                protocol = self._check_protocol(i)
                if protocol != 0:
                    ok = False
                    done[i] = False
                    print('[FAILURE]')
                    self._print_check_protocol_error_message(protocol)
                else:
                    done[i] = True
                    print('[SUCCESS]')
            if not ok:
                print(
                    "  A problem was detected with the protocol. Will try to reinitialize the clock as it sometimes helps."
                )
                self._dev.seti('awgs/*/enable', 0)
                self._dev.seti('system/clocks/referenceclock/source', 0)
                # self._dev.seti('system/extclk', 0)
                time.sleep(1)
                self._dev.seti('system/clocks/referenceclock/source', 1)
                # self._dev.seti('system/extclk', 1)
                time.sleep(1)
                self._dev.seti('awgs/*/enable', 1)
                timeout -= 1
                if timeout <= 0:
                    print("  Too many retries, aborting!")
                    return False

        ok = True
        print("Calibrating DIO protocol delays...")
        for i in range(4):
            print('  Calibrating AWG {} '.format(i), end='')
            self._dev.seti('raw/awgs/' + str(i) + '/dio/calib/start', 1)
            time.sleep(1)
            status = self._dev.geti('raw/awgs/' + str(i) + '/dio/calib/status')
            protocol = self._check_protocol(i)
            if (status != 0) or (protocol != 0):
                print('[FAILURE]')
                self._print_check_protocol_error_message(protocol)
                ok = False
            else:
                print('[SUCCESS]')
        return ok

    def _add_codeword_parameters(self):
        """
        Adds parameters parameters that are used for uploading codewords.
        It also contains initial values for each codeword to ensure
        that the "upload_codeword_program"

        """
        docst = ('Specifies a waveform to for a specific codeword. ' +
                 'The waveforms must be uploaded using ' +
                 '"upload_codeword_program". The channel number corresponds' +
                 ' to the channel as indicated on the device (1 is lowest).')
        self._params_to_skip_update = []
        for ch in range(self._num_channels):
            for cw in range(self._num_codewords):
                parname = 'wave_ch{}_cw{:03}'.format(ch + 1, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch + 1, cw),
                    vals=vals.Arrays(),  # min_value, max_value = unknown
                    set_cmd=self._gen_write_csv(parname),
                    get_cmd=self._gen_read_csv(parname),
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    def _gen_write_csv(self, wf_name):
        def write_func(waveform):
            # The lenght of AWG8 waveforms should be a multiple of 8 samples.
            if (len(waveform) % 8) != 0:
                extra_zeros = 8 - (len(waveform) % 8)
                waveform = np.concatenate([waveform, np.zeros(extra_zeros)])
            return self._write_csv_waveform(wf_name=wf_name, waveform=waveform)

        return write_func

    def _gen_read_csv(self, wf_name):
        def read_func():
            return self._read_csv_waveform(wf_name=wf_name)

        return read_func

    def _write_csv_waveform(self, wf_name: str, waveform):
        filename = os.path.join(self.lab_one_webserver_path, 'awg', 'waves',
                                self._devname + '_' + wf_name + '.csv')
        fmt = '%.18e' if waveform.dtype == np.float else '%d'
        np.savetxt(filename, waveform, delimiter=",", fmt=fmt)

    def _read_csv_waveform(self, wf_name: str):
        filename = os.path.join(self.lab_one_webserver_path, 'awg', 'waves',
                                self._devname + '_' + wf_name + '.csv')
        try:
            return np.genfromtxt(filename, delimiter=',')
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            logging.warning(e)
            print(e)
            return None

    def _delete_chache_files(self):
        cache_dir = os.path.join(self.lab_one_webserver_path, 'awg', 'waves',
                                 '.cache')
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))

    def _delete_csv_files(self):
        csv_dir = os.path.join(self.lab_one_webserver_path, 'awg', 'waves')
        for f in os.listdir(csv_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(csv_dir, f))

    # Note: This was added for debugging by NielsH.
    # If we do not need it for a few days we should remove it. (2/10/2017)
    # def stop_awg(self):
    #     test_program = """
    #     // 'Counting'  waveform
    #     const N = 80;
    #     setWaveDIO(0, ones(N), -ones(N));
    #     setWaveDIO(1,  ones(N), -ones(N));
    #     setWaveDIO(2, -ones(N),  ones(N));
    #     setWaveDIO(3,  ones(N),  ones(N));
    #     setWaveDIO(4,  -blackman(N, 1.0, 0.2),  -blackman(N, 1.0, 0.2));
    #     setWaveDIO(5,   blackman(N, 1.0, 0.2),  -blackman(N, 1.0, 0.2));
    #     setWaveDIO(6,  -blackman(N, 1.0, 0.2),  blackman(N, 1.0, 0.2));
    #     setWaveDIO(7,  blackman(N, 1.0, 0.2),  blackman(N, 1.0, 0.2));
    #     """

    #     for awg_nr in range(4):
    #         print('Configuring AWG {} with dummy program'.format(awg_nr))

    #         # disable all AWG channels
    #         self.set('awgs_{}_enable'.format(awg_nr), 0)
    #         self.configure_awg_from_string(awg_nr, test_program, self.timeout())
    #         self.set('awgs_{}_single'.format(awg_nr), 0)
    #         self.set('awgs_{}_enable'.format(awg_nr), 1)

    #     print('Waiting...')
    #     time.sleep(1)

    #     for awg_nr in range(4):
    #         # disable all AWG channels
    #         self.set('awgs_{}_enable'.format(awg_nr), 0)

    def initialze_all_codewords_to_zeros(self):
        """
        Generates all zeros waveforms for all codewords
        """
        t0 = time.time()
        wf = np.zeros(32)
        waveform_params = [
            value for key, value in self.parameters.items()
            if 'wave_ch' in key.lower()
        ]
        for par in waveform_params:
            par(wf)
        t1 = time.time()
        print('Set all zeros waveforms in {:.1f} s'.format(t1 - t0))

    def awg_update_waveform(self, awg_nr: int, index: int, data1, data2=None):
        """Immediately updates the waveform with the given index.

        The waveform in the waveform viewer in the LabOne web interface is not
        updated. If data for both AWG channels is given, then the lengths must
        match.

        Warning! This method should be used with care.
        Note that the waveform get's overwritten if a new program is uploaded.
        Any parts of a waveform longer than data1/data2 will not be overwritten.
        Loading speed depends on the size of data1 and data2 and is ~80ms for
        20us waveform.

        Args:
            awg_nr: The index of the virtual AWG to update the channels for.
            index:  Index of the waveform to update. Corresponds to the order of
                    waveforms in the AWG/Waveform tab in the web interface.
                    Starts from 0.
            data1:  Waveform data for channel 1.
            data2:  Optional waveform data for channel 2.
        """
        if data1 is None and data2 is None:
            return
        elif data1 is None:
            data = data2
        elif data2 is None:
            data = data1
        else:
            data = np.vstack((
                data1,
                data2,
            )).reshape((-1, ), order='F')
        self.set('awgs_{}_waveform_index'.format(awg_nr), index)
        self.set('awgs_{}_waveform_data'.format(awg_nr), data)
        self._dev.daq.sync()  # Is this necessary?

    def upload_codeword_program(self, awgs=np.arange(4)):
        """
        Generates a program that plays the codeword waves for each channel.

        awgs (array): the awg numbers to which to upload the codeword program.
                    By default uploads to all channels but can be specific to
                    speed up the process.
        """
        # Type conversion to ensure lists do not produce weird results
        awgs = np.array(awgs)
        # because awg_channels come in pairs and are numbered from 1-8 in API
        awg_channels = awgs * 2 + 1

        for awg_nr in awgs:
            # disable all AWG channels
            self.set('awgs_{}_enable'.format(int(awg_nr)), 0)

        codeword_mode_snippet = (
            'while (1) { \n '
            '\t// Wait for a trigger on the DIO interface\n'
            '\twaitDIOTrigger();\n'
            '\t// Play a waveform from the table based on the DIO code-word\n'
            '\tplayWaveDIO(); \n'
            '}')
        if self.cfg_codeword_protocol() != 'flux':
            for ch in awg_channels:
                waveform_table = '// Define the waveform table\n'
                for cw in range(self.cfg_num_codewords()):
                    wf0_name = '{}_wave_ch{}_cw{:03}'.format(
                        self._devname, ch, cw)
                    wf1_name = '{}_wave_ch{}_cw{:03}'.format(
                        self._devname, ch + 1, cw)
                    waveform_table += 'setWaveDIO({}, "{}", "{}");\n'.format(
                        cw, wf0_name, wf1_name)
                program = waveform_table + codeword_mode_snippet
                # N.B. awg_nr in goes from 0 to 3 in API while in LabOne
                # it is 1 to 4
                awg_nr = ch // 2  # channels are coupled in pairs of 2
                self.configure_awg_from_string(
                    awg_nr=int(awg_nr),
                    program_string=program,
                    timeout=self.timeout())
        else:  # if protocol is flux
            for ch in awg_channels:
                waveform_table = '//Flux mode\n// Define the waveform table\n'
                mask_0 = 0b000111  # AWGx_ch0 uses lower bits for CW
                mask_1 = 0b111000  # AWGx_ch1 uses higher bits for CW

                # for cw in range(2**6):
                for cw in range(8):
                    cw0 = cw & mask_0
                    cw1 = (cw & mask_1) >> 3
                    # FIXME: this is a hack because not all AWG8 channels support
                    # amp mode. It forces all AWG8's of a pair to behave identical.
                    cw1 = cw0
                    # if both wfs are triggered play both
                    if (cw0 != 0) and (cw1 != 0):
                        # if both waveforms exist, upload
                        wf0_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch, cw0)
                        wf1_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch + 1, cw1)

                    # if single wf is triggered fill the other with zeros
                    elif (cw0 == 0) and (cw1 != 0):
                        wf0_cmd = 'zeros({})'.format(
                            len(self.get('wave_ch{}_cw{:03}'.format(ch, cw1))))
                        wf1_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch + 1, cw1)

                    elif (cw0 != 0) and (cw1 == 0):
                        wf0_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch, cw0)
                        wf1_cmd = 'zeros({})'.format(
                            len(self.get('wave_ch{}_cw{:03}'.format(ch, cw0))))
                    # if no wfs are triggered play only zeros
                    else:
                        wf0_cmd = 'zeros({})'.format(42)
                        wf1_cmd = 'zeros({})'.format(42)

                    waveform_table += 'setWaveDIO({}, {}, {});\n'.format(
                        cw, wf0_cmd, wf1_cmd)
                program = waveform_table + codeword_mode_snippet

                # N.B. awg_nr in goes from 0 to 3 in API while in LabOne it
                # is 1 to 4
                awg_nr = ch // 2  # channels are coupled in pairs of 2
                self.configure_awg_from_string(
                    awg_nr=int(awg_nr),
                    program_string=program,
                    timeout=self.timeout())
        self.configure_codeword_protocol()

    def clock_freq(self, awg_nr):
        return 2.4e9 / (2**self.get('awgs_{}_time'.format(awg_nr)))

    def configure_codeword_protocol(self, default_dio_timing: bool = False):
        """
        This method configures the AWG-8 codeword protocol.
        It includes specifying what bits are used to specify codewords
        as well as setting the delays on the different bits.

        The protocol uses several parts to specify the
        These parameters are specific to each AWG-8 channel and depend on the
        the function the AWG8 has in the setup.

        The parameter "cfg_codeword_protocol" defines what protocol is used.
        There are three options:
            identical : all AWGs have the same configuration
            microwave : AWGs 0 and 1 share bits
            flux      : Each AWG pair is responsible for 2 flux channels.
                        this also affects the "codeword_program" and
                        setting "wave_chX_cwXXX" parameters.

        Protocol definition:
        protocol
            - mask/value -> some bits are masked to allow using only a few bits
                            to specify a codeword.
            - mask/shift -> all acquired bits are shifted to allow specifying
                            which bits should be used.
        The parameters below are global to all AWG channels.
            - strobe/index -> this specifies which bit is the toggle/strobe bit
            - strobe/slope -> check for codewords on rissing/falling or both
                              edges of the toggle bit.
            - valid/index  -> specifies the codeword valid bit
            - valid/slope  -> specifies the slope of the valid bit

        Delay configuration
            In this part the DIO delay indices are set. These should be
            identical for each AWG channel.
            - dio/delay/index -> selects which delay to change next
            - dio/delay/value -> specifies an individual delay

        Trun on device
            The final step enablse the signal output of each AWG and sets
            it to the right mode.

        """
        ####################################################
        # Protocol definition
        ####################################################

        # Configure the DIO interface for triggering on

        for awg_nr in range(4):
            # This is the bit index of the valid bit,
            self.set('awgs_{}_dio_valid_index'.format(awg_nr), 31)
            # Valid polarity is 'high' (hardware value 2),
            # 'low' (hardware value 1), 'no valid needed' (hardware value 0)
            self.set('awgs_{}_dio_valid_polarity'.format(awg_nr), 2)
            # This is the bit index of the strobe signal (toggling signal),
            self.set('awgs_{}_dio_strobe_index'.format(awg_nr), 30)

            # Configure the DIO interface for triggering on the both edges of
            # the strobe/toggle bit signal.
            # 1: rising edge, 2: falling edge or 3: both edges
            self.set('awgs_{}_dio_strobe_slope'.format(awg_nr), 3)

            # the mask determines how many bits will be used in the protocol
            # e.g., mask 3 will mask the bits with bin(3) = 00000011 using
            # only the 2 Least Significant Bits.
            # N.B. cfg_num_codewords must be a power of 2
            self.set('awgs_{}_dio_mask_value'.format(awg_nr),
                     self.cfg_num_codewords() - 1)

            if self.cfg_codeword_protocol() == 'identical':
                # In the identical protocol all bits are used to trigger
                # the same codewords on all AWG's

                # N.B. The shift is applied before the mask
                # The relevant bits can be selected by first shifting them
                # and then masking them.
                self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)

            # In the mw protocol bits [0:7] -> CW0 and bits [(8+1):15] -> CW1
            # N.B. DIO bit 8 (first of 2nd byte)  not connected in AWG8!
            elif self.cfg_codeword_protocol() == 'microwave':
                if awg_nr in [0, 1]:
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
                elif awg_nr in [2, 3]:
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 9)

            elif self.cfg_codeword_protocol() == 'flux':
                # bits[0:3] for awg0_ch0, bits[4:6] for awg0_ch1 etc.
                # self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**6-1)
                # self.set('awgs_{}_dio_mask_shift'.format(awg_nr), awg_nr*6)

                # FIXME: this is a protocol that does identical flux pulses
                # on each channel.
                self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**3 - 1)
                self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 3)

        ####################################################
        # Turn on device
        ####################################################
        time.sleep(.05)
        self._dev.daq.setInt('/' + self._dev.device + '/awgs/*/enable', 1)

        # Turn on all outputs
        self._dev.daq.setInt('/' + self._dev.device + '/sigouts/*/on', 1)
        # Disable all function generators
        self._dev.daq.setInt('/' + self._dev.device + '/sigouts/*/enables/*',
                             0)
        # Switch all outputs into direct mode
        if self.cfg_codeword_protocol() == 'flux':
            for ch in range(8):
                self.set('sigouts_{}_direct'.format(ch), 0)
                self.set('sigouts_{}_range'.format(ch), 5)

        # when doing flux pulses, set everything to amp mode
        else:
            for ch in range(8):
                self.set('sigouts_{}_direct'.format(ch), 1)
                self.set('sigouts_{}_range'.format(ch), .8)
