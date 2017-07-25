import time
import numpy as np
from . import zishell_NH as zs
from .ZI_base_instrument import ZI_base_instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch
from qcodes.instrument.parameter import ManualParameter
import json


class ZI_HDAWG8(ZI_base_instrument):

    """
    """

    def __init__(self, name, device: str,
                 server: str='localhost', port=8004, **kw):
        '''
        Input arguments:
            name:           (str) name of the instrument
            server_name:    (str) qcodes instrument server
            address:        (int) the address of the data server e.g. 8006
        '''
        t0 = time.time()

        super().__init__(name=name, **kw)
        self._devname = device
        self._dev = zs.ziShellDevice()
        self._dev.connect_server(server, port)
        print("Trying to connect to device {}".format(self._devname))
        self._dev.connect_device(self._devname, '1GbE')
        self.connect_message(begin_time=t0)

    def get_idn(self):
        idn_dict = {'vendor': 'ZurichInstruments',
                    'model': self._dev.daq.getByte(
                        '/{}/features/devtype'.format(self._devname)),
                    'serial': self._devname,
                    'firmware': self._dev.geti('system/fwrevision'),
                    'fpga_firmware': self._dev.geti('system/fpgarevision')
                    }
        return idn_dict

    def create_parameter_files(self):
        '''
        this functions retrieves all possible settable and gettable parameters
        from the device.
        Additionally, it gets all minimum and maximum values for the
        parameters by trial and error to determine the validator bounds.

        Files are written to the zi_parameter_files folder using the names
        "{par_type}_node_pars_{dev_type}.json"

        '''
        # set the file names to write to
        dev_type = self._dev.daq.getByte(
            '/{}/features/devtype'.format(self._devname))
        # Watch out this writes the file to the directory from which you
        # call this command
        s_file_name = 's_node_pars_{}.json'.format(dev_type)
        d_file_name = 'd_node_pars_{}.json'.format(dev_type)

        s_node_pars = []
        d_node_pars = []
        patterns = [
            "awgs", "sigouts", "quex", "dios", "system"]

        t0 = time.time()
        for pattern in patterns:
            print("Extracting parameters of type: {}".format(pattern))
            all_nodes = set(self._dev.find('*{}*'.format(pattern)))
            s_nodes = set(self._dev.finds('*{}*'.format(pattern)))
            d_nodes = all_nodes.difference(s_nodes)
            print("Found {} nodes".format(len(all_nodes)))

            ###################################################
            # Extracting settable nodes
            ###################################################
            s_nodes = list(s_nodes)
            default_values = {sn: self._dev.getd(sn) for sn in s_nodes}
            for s_node in s_nodes:
                self._dev.setd(s_node,  1e12)
            max_values = {sn: self._dev.getd(sn) for sn in s_nodes}
            for s_node in s_nodes:
                self._dev.setd(s_node, -1e12)
            min_values = {sn: self._dev.getd(sn) for sn in s_nodes}
            float_values = dict.fromkeys(s_nodes)

            for s_node in s_nodes:
                if np.pi > max_values[s_node]:
                    float_values[s_node] = max_values[s_node]/np.pi
                else:
                    float_values[s_node] = np.pi
                self._dev.setd(s_node, float_values[s_node])
            actual_float_values = {sn: self._dev.getd(sn) for sn in s_nodes}

            node_types = dict.fromkeys(s_nodes)
            for s_node in sorted(s_nodes):
                # self._dev.setd(node,default_values[s_node])
                fraction, integer = np.modf(actual_float_values[s_node])
                if fraction != 0:
                    node_types[s_node] = 'float'
                    if min_values[s_node] == max_values[s_node]:
                        node_types[s_node] = 'float_small'
                        min_values[s_node] = 0
                    elif abs(min_values[s_node]) < 0.01:
                        min_values[s_node] = 0
                else:
                    node_types[s_node] = 'int'
                    min_values[s_node] = 0
                    if max_values[s_node] == 3567587328:
                        node_types[s_node] = 'int_64'
                        max_values[s_node] = 4294967295
                    elif max_values[s_node] == 1:
                        node_types[s_node] = 'bool'
                    elif max_values[s_node] == 0:
                        max_values[s_node] = 255
                        node_types[s_node] = 'int_8bit'
                    elif max_values[s_node] > 4294967295:
                        node_types[s_node] = 'float'

                print(s_node)
                line = [s_node.replace(
                    '/' + self._devname + '/', ''), node_types[s_node],
                    min_values[s_node], max_values[s_node]]
                s_node_pars.append(line)

            print('Completed settable nodes for pattern {}'.format(pattern))
            print('elapsed time {:.2f}'.format(time.time() - t0 ))

            ###################################################
            # Extracting data nodes
            ###################################################

            # extracting info from the data nodes
            d_nodes = list(d_nodes)

            node_types = ['']*len(d_nodes)

            for i, d_node in enumerate(d_nodes):
                try:
                    answer = self._dev.getv(d_node)
                    if isinstance(answer, dict):
                        node_types[i] = 'float'
                    elif isinstance(answer, list):
                        try:
                            self._dev.setv(d_node, np.array([0, 0, 0]))
                            node_types[i] = 'vector_gs'
                        except:
                            node_types[i] = 'vector_g'
                    else:
                        print("unknown type")
                except:
                    node_types[i] = 'vector_s'

                line = [d_node.replace('/' + self._devname + '/', ''),
                        node_types[i]]
                d_node_pars.append(line)

            print('Completed data nodes for pattern {}'.format(pattern))
            print('elapsed time {:.2f}'.format(time.time() - t0))

        with open(s_file_name, 'w') as s_file:
            json.dump(s_node_pars, s_file, default=int, indent=2)

        with open(d_file_name, 'w') as d_file:
            json.dump(d_node_pars, d_file, default=int, indent=2)
