import time
import json
import os
import sys
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from zhinst.ziPython import ziListEnum as ziListEnum


class ZI_base_instrument(Instrument):

    """
    This is an abstract base class for Zurich Instruments instrument drivers.
    """


    def add_s_node_pars(self, filename: str):
        f = open(filename).read()
        s_node_pars = json.loads(f)
        for par in s_node_pars:
            parname = par[0].replace("/", "_")
            parfunc = "/"+self._devname+"/"+par[0]
            if par[2] > par[3]:
                raise ValueError(
                    'Min value ({}) must be smaller than max value ({}). '
                    'par: {} '.format(par[2], par[3], par[0]))
            if par[1] == 'float':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.setd, parfunc),
                    get_cmd=self._gen_get_func(self._dev.getd, parfunc),
                    vals=vals.Numbers(par[2], par[3]))
            elif par[1] == 'float_small':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.setd, parfunc),
                    get_cmd=self._gen_get_func(self._dev.getd, parfunc),
                    vals=vals.Numbers(par[2], par[3]))
            elif par[1] == 'int_8bit':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.seti, parfunc),
                    get_cmd=self._gen_get_func(self._dev.geti, parfunc),
                    vals=vals.Ints(int(par[2]), int(par[3])))
            elif par[1] == 'int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.seti, parfunc),
                    get_cmd=self._gen_get_func(self._dev.geti, parfunc),
                    vals=vals.Ints(int(par[2]), int(par[3])))
            elif par[1] == 'int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.seti, parfunc),
                    get_cmd=self._gen_get_func(self._dev.geti, parfunc),
                    vals=vals.Ints(int(par[2]), int(par[3])))
            elif par[1] == 'bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.seti, parfunc),
                    get_cmd=self._gen_get_func(self._dev.geti, parfunc),
                    vals=vals.Ints(int(par[2]), int(par[3])))
            else:
                print("parameter {} type {} from from"
                      " s_node_pars not recognized".format(parname, par[1]))

    def add_d_node_pars(self, filename: str):
        f = open(filename).read()
        d_node_pars = json.loads(f)
        for parameter in d_node_pars:
            parname = parameter[0].replace("/", "_")
            parfunc = "/"+self._devname+"/"+parameter[0]
            if parameter[1] == 'float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self._dev.getd, parfunc))
            elif parameter[1] == 'vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self._dev.getv, parfunc))
            elif parameter[1] == 'vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.setv, parfunc),
                    vals=vals.Anything())
            elif parameter[1] == 'vector_gs':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self._dev.setv, parfunc),
                    get_cmd=self._gen_get_func(self._dev.getv, parfunc),
                    vals=vals.Anything())
            else:
                print("parameter {} type {} from d_node_pars"
                      " not recognized".format(parname, parameter[1]))

    @classmethod
    def _gen_set_func(self, dev_set_type, node_path: str):
        """
        Generates a set function based on the dev_set_type method (e.g., seti)
        and the node_path (e.g., '/dev8003/sigouts/1/mode'
        """
        def set_func(val):
            dev_set_type(node_path, val)
            return dev_set_type(node_path, value=val)
        return set_func

    @classmethod
    def _gen_get_func(self, dev_get_type, node_path: str):
        """
        Generates a get function based on the dev_set_type method (e.g., geti)
        and the node_path (e.g., '/dev8003/sigouts/1/mode'
        """
        def get_func():
            return dev_get_type(node_path)
        return get_func

    def create_parameter_files_new(self):
        """
        This generates a json file Containing the node_docs as extracted
        from the ZI instrument API.

        In the future this file (instead of the s_node_pars and d_node_pars)
        should be used to generate the drivers.
        """
        # set the file names to write to
        dev_type = self._dev.daq.getByte(
            '/{}/features/devtype'.format(self._devname))

        # Watch out this overwrites the existing json parameter files
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        par_fn = os.path.join(dir_path, 'zi_parameter_files',
                              'node_doc_{}.json'.format(dev_type))

        flags = ziListEnum.absolute | ziListEnum.recursive | ziListEnum.leafsonly
        # Get node documentation for the device's entire node tree
        node_doc = json.loads(
            self._dev.daq.listNodesJSON('/{}/'.format(self._devname), flags))

        with open(par_fn, 'w') as file:
            json.dump(node_doc, file, indent=4)

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

        # Watch out this overwrites the existing json parameter files
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        s_file_name = os.path.join(dir_path, 'zi_parameter_files',
                                   's_node_pars_{}.json'.format(dev_type))
        d_file_name = os.path.join(dir_path, 'zi_parameter_files',
                                   'd_node_pars_{}.json'.format(dev_type))

        s_node_pars = []
        d_node_pars = []
        patterns = ["awgs", "sigouts", "quex", "dios", "system"]

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
            print('elapsed time {:.2f}'.format(time.time() - t0))

            ###################################################
            # Extracting data nodes
            ###################################################

            # extracting info from the data nodes
            d_nodes = list(d_nodes)

            node_types = ['']*len(d_nodes)

            # type determination is now inferred using try except statements.
            # this is not a very good way to do it. Better would be to
            # actively check for types.
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
