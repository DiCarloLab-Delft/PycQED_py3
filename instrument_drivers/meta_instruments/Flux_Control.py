import qt
from instrument import Instrument
import logging
import numpy as np


class Flux_Control(Instrument):
    '''
    Instrument used for translating fluxes into dac settings.
    Makes use of a transfer matrix
    vec{phi}= M vec{V_dac - V_offset}
    vec{V} = M^-1 vec{phi} + V_offset
    '''

    def __init__(self, name, reset=False, IVVI_name='IVVI', **kw):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Meta-Instrument'])

        # Set parameters
        self.IVVI = qt.instruments[IVVI_name]

        # Add parameters
        self.add_parameter('transfer_matrix',
                           type=np.ndarray,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_offset_vector', units='mV',
                           type=list,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('flux_vector',
                           type=list,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_mapping',
                           type=list,
                           flags=Instrument.FLAG_GETSET)

        self.set_print_suppression(False)

    # Converting fluxes and voltages
    def set_print_suppression(self, suppress):
        self.suppress_print_statements = suppress

    def get_print_suppression(self):
        return self.suppress_print_statements

    def set_fluxes_zero(self):
        self.set_flux_vector(np.zeros(len(self.get_dac_mapping())))

    def convert_flux_vec_to_dac_voltages(self, flux_vector):
        '''
        vec{V} = M^-1 vec{phi} + V_offset
        '''
        inv_transfer_matrix = np.linalg.pinv(self.transfer_matrix)
        dac_voltages = (np.dot(inv_transfer_matrix, flux_vector) +
                        self.dac_offset_vector)
        return dac_voltages

    def convert_dac_voltages_to_flux_vec(self, dac_voltages):
        '''
        vec{phi}= M vec{V_dac -V_offset}
        '''
        flux_vector = np.dot(self.transfer_matrix, (dac_voltages -
                                                    self.dac_offset_vector))
        return flux_vector

    def set_flux(self, flux_index, value):
        '''
        Set the flux value of a SQUID loop indicated by flux_index.
        Index counting starts at 1 (non-pythonic)
        '''
        flux_vector = self.get_flux_vector()
        flux_vector[flux_index-1] = value
        self.set_flux_vector(flux_vector)

    def do_set_transfer_matrix(self, transfer_matrix):
        self.transfer_matrix = transfer_matrix

    def do_get_transfer_matrix(self):
        return self.transfer_matrix

    def do_set_flux_vector(self, flux_vector):
        self.flux_vector = flux_vector
        dac_vals = self.convert_flux_vec_to_dac_voltages(flux_vector)
        for i, dac_val in enumerate(dac_vals):
            if not self.suppress_print_statements:
                print('setting dac "%s" to: "%s" mV' % (self.dac_mapping[i],
                                                        dac_val))
            self.IVVI.set_dac(self.dac_mapping[i], dac_val)
            qt.msleep(0.01)
        self.IVVI.get_all()

    def do_get_flux_vector(self):
        dac_vector = np.zeros(len(self.dac_mapping))
        for i, dac_ch in enumerate(self.dac_mapping):
            dac_vector[i] = self.IVVI.get_dac(dac_ch)

        self.flux_vector = self.convert_dac_voltages_to_flux_vec(dac_vector)
        return self.flux_vector

    def get_flux(self, flux_channel):
        flux_vec = self.get_flux_vector()
        return flux_vec[flux_channel-1]

    def do_set_dac_offset_vector(self, dac_offset_vector):
        self.dac_offset_vector = dac_offset_vector

    def do_get_dac_offset_vector(self):
        return self.dac_offset_vector

    def do_set_dac_mapping(self, mapping):
        self.dac_mapping = mapping

    def do_get_dac_mapping(self):
        return self.dac_mapping
