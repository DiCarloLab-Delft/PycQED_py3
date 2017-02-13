import logging
import numpy as np
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

# from pycqed.analysis.analysis_toolbox import calculate_transmon_transitions
# from pycqed.analysis import analysis_toolbox as a_tools
# from pycqed.measurement import detector_functions as det
# from pycqed.measurement import composite_detector_functions as cdet
# from pycqed.measurement import mc_parameter_wrapper as pw


class Flux_Control(Instrument):

    '''

    Flux Control object

    '''

    def __init__(self, name, num_channels, IVVI=None, **kw):
        super().__init__(name, **kw)

        self.IVVI = IVVI

        # Add parameters
        self.add_parameter('transfer_matrix',
                           label='Transfer Matrix',
                           set_cmd=self.do_set_transfer_matrix,
                           get_cmd=self.do_get_transfer_matrix,
                           vals=vals.Anything())
        self.add_parameter('inv_transfer_matrix',
                           label='Transfer Matrix',
                           set_cmd=self.do_set_inv_transfer_matrix,
                           get_cmd=self.do_get_inv_transfer_matrix,
                           vals=vals.Anything())
        self.add_parameter('flux_offsets', units='Phi_0',
                           label='Flux offsets',
                           set_cmd=self.do_set_flux_offsets,
                           get_cmd=self.do_get_flux_offsets,
                           vals=vals.Anything())
        self.add_parameter('flux_vector', units='Phi_0',
                           label='Linear transformation coefficients',
                           set_cmd=self.do_set_flux_vector,
                           get_cmd=self.do_get_flux_vector,
                           vals=vals.Anything())
        self.add_parameter('dac_mapping',
                           label='Linear transformation coefficients',
                           set_cmd=self.do_set_dac_mapping,
                           get_cmd=self.do_get_dac_mapping,
                           vals=vals.Anything())

        for i in range(0, num_channels):
            self.add_parameter(
                'flux{}'.format(i),
                label='Flux {}'.format(i),
                units=r'$\Phi_0$',
                get_cmd=self._gen_ch_get_func(self._get_flux, i),
                set_cmd=self._gen_ch_set_func(self._set_flux, i),
                vals=vals.Numbers(-10, 10.))

    def _set_flux(self, id_flux, val):
        current_flux = self.flux_vector()
        new_flux = current_flux
        new_flux[id_flux] = val
        self.flux_vector(new_flux)

    def _get_flux(self, id_flux):
        val = self.flux_vector()
        return val[id_flux]

    def do_set_transfer_matrix(self, matrix):
        self._transfer_matrix = matrix

    def do_get_transfer_matrix(self):
        return self._transfer_matrix

    def do_set_inv_transfer_matrix(self, matrix):
        self._inv_transfer_matrix = matrix

    def do_get_inv_transfer_matrix(self):
        return self._inv_transfer_matrix

    def do_set_flux_offsets(self, vector):
        self._flux_offsets = vector

    def do_get_flux_offsets(self):
        return self._flux_offsets

    def do_set_flux_vector(self, vector):
        currents = np.dot(self._inv_transfer_matrix,
                          (vector-self._flux_offsets))
        for i in range(len(self._dac_mapping)):
            self.IVVI._set_dac(self._dac_mapping[i], currents[i])
        return currents

    def do_get_flux_vector(self):
        currents = np.zeros(len(self._dac_mapping))
        for i in range(len(self._dac_mapping)):
            currents[i] = self.IVVI._get_dac(self._dac_mapping[i])
        flux_vector = np.dot(
            self._transfer_matrix, currents) + self._flux_offsets
        return flux_vector

    def do_set_dac_mapping(self, vector):
        self._dac_mapping = vector

    def do_get_dac_mapping(self):
        return self._dac_mapping


    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func