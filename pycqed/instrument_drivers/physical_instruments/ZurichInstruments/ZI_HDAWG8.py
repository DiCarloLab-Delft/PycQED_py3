import time
import logging
import os
import sys
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
    using the "create_parameter_files" method in the ZI_base_instrument class.
    These are used to add parameters to the instrument.

    Known issues (last update 25/7/2017)
        - the parameters "sigouts/*/offset" are detected as int by the
            create parameter extraction file
        - the restart device method does not work
        - the direct/amplified output mode corresponding to node
            "raw/sigouts/*/mode" is not discoverable through the find method.
            This parameter is now added by hand as a workaround.
    """

    def __init__(self, name, device: str,
                 server: str='localhost', port=8004, **kw):
        '''
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            server          (str) the ZI data server
            port            (int) the port to connect to
        '''
        t0 = time.time()
        self._num_channels = 8
        self._num_codewords = 256

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
        self._dev.connect_device(self._devname, '1GbE')

        self.add_parameter('timeout', unit='s',
                           initial_value=10,
                           parameter_class=ManualParameter)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_fn = os.path.join(dir_path, 'zi_parameter_files')

        try:
            self.add_s_node_pars(
                filename=os.path.join(base_fn, 's_node_pars_HDAWG8.json'))
        except FileNotFoundError:
            logging.warning("parameter file for settable parameters"
                            " {} not found".format(self._s_file_name))
        try:
            self.add_d_node_pars(
                filename=os.path.join(base_fn, 'd_node_pars_HDAWG8.json'))
        except FileNotFoundError:
            logging.warning("parameter file for data parameters"
                            " {} not found".format(self._d_file_name))
        self.add_ZIshell_device_methods_to_instrument()

        # Manually added parameters
        # amplified mode is not implemented for all channels
        for i in [0, 1, 6, 7]:
            self.add_parameter(
                'raw_sigouts_{}_mode'.format(i),
                set_cmd=self._gen_set_func(
                    self._dev.seti, 'raw/sigouts/1/mode'),
                get_cmd=self._gen_get_func(
                    self._dev.geti, 'raw/sigouts/1/mode'),
                vals=vals.Ints(0, 1),  # Ideally this is an Enum
                docstring='"0" is direct mode\n"1" is amplified mode')

        self._add_codeword_parameters()
        self.connect_message(begin_time=t0)

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
        self.configure_awg_from_file = self._dev.configure_awg_from_file
        self.configure_awg_from_string = self._dev.configure_awg_from_string
        self.read_from_scope = self._dev.read_from_scope
        self.restart_scope_module = self._dev.restart_scope_module
        self.restart_awg_module = self._dev.restart_awg_module

    def get_idn(self):
        idn_dict = {'vendor': 'ZurichInstruments',
                    'model': self._dev.daq.getByte(
                        '/{}/features/devtype'.format(self._devname)),
                    'serial': self._devname,
                    'firmware': self._dev.geti('system/fwrevision'),
                    'fpga_firmware': self._dev.geti('system/fpgarevision')
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
                parname = 'wave_ch{}_cw{:03}'.format(ch+1, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch+1, cw),
                    vals=vals.Arrays(),  # min_value, max_value = unknown
                    set_cmd=self._gen_write_csv(parname),
                    get_cmd=self._gen_read_csv(parname),
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    def _gen_write_csv(self, wf_name):
        def write_func(waveform):
            # The lenght of AWG8 waveforms should be a multiple of 8 samples.
            extra_zeros = 8-(len(waveform) % 8)
            waveform = np.concatenate([waveform, np.zeros(extra_zeros)])
            return self._write_csv_waveform(
                wf_name=wf_name, waveform=waveform)
        return write_func

    def _gen_read_csv(self, wf_name):
        def read_func():
            return self._read_csv_waveform(
                wf_name=wf_name)
        return read_func

    def _write_csv_waveform(self, wf_name: str, waveform):
        filename = os.path.join(
            self.lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        with open(filename, 'w') as f:
            np.savetxt(filename, waveform, delimiter=",")

    def _read_csv_waveform(self, wf_name: str):
        filename = os.path.join(
            self.lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        try:
            return np.genfromtxt(filename, delimiter=',')
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            logging.warning(e)
            print(e)
            return None

    def initialze_all_codewords_to_zeros(self):
        """
        Generates all zeros waveforms for all codewords
        """
        t0 = time.time()
        wf = np.zeros(32)
        waveform_params = [value for key, value in self.parameters.items()
                           if 'wave_ch' in key.lower()]
        for par in waveform_params:
            par(wf)
        t1 = time.time()
        print('Set all zeros waveforms in {:.1f} s'.format(t1-t0))

    def upload_codeword_program(self):
        """
        Generates a program that plays the codeword waves for each channel.
        """

        for awg_nr in range(4):
            # disable all AWG channels
            self.set('awgs_{}_enable'.format(awg_nr), 0)

        codeword_mode_snippet = (
            'while (1) { \n '
            '\t// Wait for a trigger on the DIO interface\n'
            '\twaitDIOTrigger();\n'
            '\t// Play a waveform from the table based on the DIO code-word\n'
            '\tplayWaveDIO(); \n'
            '}')

        for ch in [1, 3, 5, 7]:
            waveform_table = '// Define the waveform table\n'
            for cw in range(self._num_codewords):
                wf0_name = '{}_wave_ch{}_cw{:03}'.format(
                    self._devname, ch, cw)
                wf1_name = '{}_wave_ch{}_cw{:03}'.format(
                    self._devname, ch+1, cw)
                waveform_table += 'setWaveDIO({}, "{}", "{}");\n'.format(
                    cw, wf0_name, wf1_name)
            program = waveform_table + codeword_mode_snippet
            # N.B. awg_nr in goes from 0 to 3 in API while in LabOne it is 1 to
            # 4
            awg_nr = ch//2  # channels are coupled in pairs of 2
            self.configure_awg_from_string(awg_nr=awg_nr,
                                           program_string=program,
                                           timeout=self.timeout())
            self._configure_codeword_protocol()

    def _configure_codeword_protocol(self):
        """
        This method configures the AWG-8 codeword protocol.
        It includes specifying what bits are used to specify codewords
        as well as setting the delays on the different bits.

        The protocol uses several parts to specify the
        These parameters are specific to each AWG-8 channel and depend on the

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
        # TODO: The snippet below uses the direct zi API rather than
        # parameters (which it should use). This is awaiting fixing
        # of issue #315.

        ####################################################
        # Protocol definition
        ####################################################

        # Configure the DIO interface for triggering on

        # This is the bit index of the valid bit,
        # we use bit 16 of the DIO in this example
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/valid/index', 31)
        # Valid polarity is 'high' (hardware value 2),
        # 'low' (hardware value 1), 'no valid needed' (hardware value 0)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/valid/polarity', 2)
        # This is the bit index of the strobe signal (toggling signal),
        # we use bit 24
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/strobe/index', 30)
        # Configure the DIO interface for triggering on the both edges of
        # the strobe/toggle bit signal.
        # 1 for rising edge, 2 for falling edge triggering or 3 for both edges
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/strobe/slope', 3)

        # Define the mask value, we use bit 0 on the DIO to index the table,
        # The acquired bits on the DIO will be masked by the mask.
        # as en example mask 3 will mask the bits with 00000011 using only the
        # 2 Least Significant Bits.
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/mask/value', 31)
        # Define the shift to apply to the DIO input before the mask is
        # applied, as we're using bit 0
        # we don't need to shift the value, so we set it to 0
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/mask/shift', 0)

        ####################################################
        # Delay configuration
        ####################################################

        codeword_delay = 2
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/index', 0)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/value', codeword_delay)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/index', 1)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/value', codeword_delay)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/index', 2)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/value', codeword_delay)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/index', 3)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/value', codeword_delay)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/index', 31)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/dio/delay/value', codeword_delay+1)

        ####################################################
        # Turn on device
        ####################################################

        time.sleep(1)
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/enable', 1)

        # Turn on all outputs
        self._dev.daq.setInt('/' + self._dev.device + '/sigouts/*/on', 1)
        # Disable all function generators
        self._dev.daq.setInt('/' + self._dev.device +
                             '/sigouts/*/enables/*', 0)
        # Switch all outputs in to DAC mode
        self._dev.daq.setInt('/' + self._dev.device + '/raw/sigouts/*/mode', 0)
