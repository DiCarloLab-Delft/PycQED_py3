import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

# from analysis.analysis_toolbox import calculate_transmon_transitions
# from analysis import analysis_toolbox as a_tools
# from measurement import detector_functions as det
# from measurement import composite_detector_functions as cdet
# from measurement import mc_parameter_wrapper as pw


class Flux_Control(Instrument):
    '''

    Flux Control object

    '''
    def __init__(self, name, IVVI=None, **kw):
        super().__init__(name, **kw)

        self.IVVI = IVVI

        # Add parameters
        self.add_parameter('transfer_matrix',
                           label='Transfer Matrix',
                           set_cmd=self.do_set_transfer_matrix,
                           get_cmd=self.do_get_transfer_matrix,
                           vals=vals.Anything())
        self.add_parameter('flux_offsets', units='mV',
                           label='Flux offsets',
                           set_cmd=self.do_set_flux_offsets,
                           get_cmd=self.do_get_flux_offsets,
                           vals=vals.Anything())
        self.add_parameter('flux_vector', units='Phi0',
                           label='Linear transformation coefficients',
                           set_cmd=self.do_set_flux_vector,
                           get_cmd=self.do_get_flux_vector,
                           vals=vals.Anything())
        self.add_parameter('dac_mapping',
                           label='Linear transformation coefficients',
                           set_cmd=self.do_set_dac_mapping,
                           get_cmd=self.do_get_dac_mapping,
                           vals=vals.Anything())

    def do_set_transfer_matrix(self, matrix):
        self.transfer_matrix = matrix
    def do_get_transfer_matrix(self):
        return self.transfer_matrix

    def do_set_flux_offsets(self, vector):
        self.flux_offsets = vector
    def do_get_flux_offsets(self):
        return self.flux_offsets

    def do_set_flux_vector(self, vector):
        currents = np.dot(np.linalg.inv(self.transfer_matrix),
                          (vector-self.flux_offsets))
        for i in range(len(self.dac_mapping)):
            self.IVVI._set_dac(self.dac_mapping[i], currents[i])
    def do_get_flux_vector(self):
        currents = np.zeros(len(self.dac_mapping))
        for i in range(len(self.dac_mapping)):
            currents[i] = self.IVVI._get_dac(self.dac_mapping[i])
        flux_vector = np.dot(self.transfer_matrix, currents) + self.flux_offsets
        return flux_vector

    def do_set_dac_mapping(self, vector):
        self.dac_mapping = vector
    def do_get_dac_mapping(self):
        return self.dac_mapping
