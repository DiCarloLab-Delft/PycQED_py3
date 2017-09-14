import time
import logging
import os
from . import zishell_NH as zs
from qcodes.utils import validators as vals
from .ZI_base_instrument import ZI_base_instrument
from qcodes.instrument.parameter import ManualParameter


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

        self.connect_message(begin_time=t0)

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
