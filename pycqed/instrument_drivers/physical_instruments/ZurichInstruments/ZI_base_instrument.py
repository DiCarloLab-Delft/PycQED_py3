import json
import os
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from zhinst.ziPython import ziListEnum as ziListEnum
import logging
log = logging.getLogger(__name__)


class ZI_base_instrument(Instrument):

    """
    This is an abstract base class for Zurich Instruments instrument drivers.
    """

    def add_parameters_from_file(self, filename: str):
        """
        Takes in a node_doc JSON file auto generates parameters based on
        the contents of this file.
        """
        with open(filename) as f:
            node_pars = json.loads(f.read())
        for par in node_pars.values():
            node = par['Node'].split('/')
            # The parfile is valid for all devices of a certain type
            # so the device name has to be split out.
            parname = '_'.join(node[2:]).lower()
            parfunc = "/" + self._devname + "/" + "/".join(node[2:])

            # This block provides the mapping between the ZI node and QCoDes
            # parameter.
            par_kw = {}
            par_kw['name'] = parname
            if par['Unit'] != 'None':
                par_kw['unit'] = par['Unit']
            else:
                par_kw['unit'] = 'arb. unit'

            par_kw['docstring'] = par['Description']
            if "Options" in par.keys():
                # options can be done better, this is not sorted
                par_kw['docstring'] += '\nOptions:\n' + str(par['Options'])

            # Creates type dependent get/set methods
            if par['Type'] == 'Integer (64 bit)' or par['Type'] == 'Integer (enumerated)':
                par_kw['set_cmd'] = self._gen_set_func(self._dev.seti, parfunc)
                par_kw['get_cmd'] = self._gen_get_func(self._dev.geti, parfunc)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = vals.Ints()

            elif par['Type'] == 'Double':
                par_kw['set_cmd'] = self._gen_set_func(self._dev.setd, parfunc)
                par_kw['get_cmd'] = self._gen_get_func(self._dev.getd, parfunc)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = vals.Numbers()

            elif par['Type'] == 'ZIVectorData':
                par_kw['set_cmd'] = self._gen_set_func(self._dev.setv, parfunc)
                par_kw['get_cmd'] = self._gen_get_func(self._dev.getv, parfunc)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = vals.Arrays()

            elif par['Type'] == 'String':
                par_kw['get_cmd'] = self._gen_get_func(self._dev.daq.getString, parfunc)
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = vals.Strings()

            else:
                log.error("Unimplemented parameter type '{}' for '{}'".format(par['Type'], parfunc))
                continue

            # If not readable/writable the methods are removed after the type
            # dependent loop to keep this more readable.
            if 'Read' not in par['Properties']:
                par_kw['get_cmd'] = None
            if 'Write' not in par['Properties']:
                par_kw['set_cmd'] = None
            self.add_parameter(**par_kw)

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

    def create_parameter_file(self):
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
            json.dump(node_doc, file, indent=4, sort_keys=True)
