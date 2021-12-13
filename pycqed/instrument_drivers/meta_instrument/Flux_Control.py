import numpy as np

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter


class Flux_Control(Instrument):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        trnsf_mat_docst = ('Converts dac voltages to virtual flux.'
                           'This matrix is defined as'
                           'flux = T dac_voltages'
                           'T is then d flux/ d dac')
        # Ramiro will expand this to include the proper equations
        self.add_parameter('transfer_matrix',
                           label='Transfer Matrix',
                           docstring=trnsf_mat_docst,
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())
        self._inv_transfer_matrix = None
        self.add_parameter('inv_transfer_matrix',
                           label='Inverse transfer Matrix',
                           docstring=('Returns the inverse of the transfer'
                                      ' matrix, unless explictly specified'),
                           set_cmd=self._do_set_inv_transfer_matrix,
                           get_cmd=self._do_get_inv_transfer_matrix,
                           vals=vals.Arrays())

        self.add_parameter('dac_mapping',
                           label='mapping of flux channels to current/voltage channels',
                           parameter_class=ManualParameter,
                           vals=vals.Lists())

    def _set_flux(self, id_flux, val):
        current_flux = self._flux_vector
        new_flux = current_flux
        new_flux[id_flux] = val
        self._do_set_flux_vector(new_flux)

    def _get_flux(self, id_flux):
        val = self._do_get_flux_vector()
        return val[id_flux]

    def _do_set_inv_transfer_matrix(self, matrix):
        self._inv_transfer_matrix = matrix

    def _do_get_inv_transfer_matrix(self):
        if self._inv_transfer_matrix is None:
            return np.linalg.inv(self.transfer_matrix())
        else:
            return self._inv_transfer_matrix

    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func


class Flux_Control_IVVI(Flux_Control):
    def __init__(self, name, num_channels, IVVI=None, **kw):
        super().__init__(name, **kw)
        self.add_parameter('IVVI',
                           initial_value=IVVI,
                           parameter_class=InstrumentParameter)
        self.add_parameter('dac_offsets', unit='mV',
                           label='Dac offsets',
                           docstring=('Offsets in mV corresponding to setting'
                                      ' all qubits to the'
                                      ' sweetspot. N.B. the order here is the'
                                      ' same as the flux vector.'),
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())
        self.add_parameter('V_per_Phi0s', unit='V',
                           label='V_per_Phi0s',
                           docstring=('V_per_phi0s. This requires the matrix to have the'
                                      'diagonal elements to be set to 1.'),
                           parameter_class=ManualParameter,
                           initial_value=np.ones([num_channels]),
                           vals=vals.Arrays())
        self._flux_vector = np.zeros(num_channels)

        for i in range(0, num_channels):
            self.add_parameter(
                'flux{}'.format(i),
                label='Flux {}'.format(i),
                unit=r'$\Phi_0$',
                get_cmd=self._gen_ch_get_func(self._get_flux, i),
                set_cmd=self._gen_ch_set_func(self._set_flux, i),
                vals=vals.Numbers())

    def _do_set_flux_vector(self, vector):
        IVVI = self.IVVI.get_instr()
        currents = np.dot(self.inv_transfer_matrix(),
                          vector) + self.dac_offsets()
        for i in range(len(self.dac_mapping())):
            IVVI._set_dac(self.dac_mapping()[i], currents[i])
        return currents

    def _do_get_flux_vector(self):
        IVVI = self.IVVI.get_instr()
        currents = np.zeros(len(self.dac_mapping()))
        for i in range(len(self.dac_mapping())):
            currents[i] = IVVI._get_dac(self.dac_mapping()[i])
        self.flux_vector = np.dot(
            self.transfer_matrix, currents - self.dac_offsets())

        return self._flux_vector


class Flux_Control_SPI(Flux_Control):
    def __init__(self, name, current_channels, fluxcurrent=None, **kw):
        super().__init__(name, **kw)
        self.add_parameter('fluxcurrent',
                           initial_value=fluxcurrent,
                           parameter_class=InstrumentParameter)
        self.num_channels = len(current_channels)
        self.add_parameter('dac_offsets', unit='A',
                           label='Dac offsets',
                           docstring=('Offsets in A corresponding to setting'
                                      ' all qubits to the'
                                      ' sweetspot. N.B. the order here is the'
                                      ' same as the flux vector.'),
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())
        self.add_parameter('A_per_Phi0s', unit='A',
                           label='A_per_Phi0s',
                           docstring=('A_per_phi0s. This requires the matrix to have the'
                                      'diagonal elements to be set to 1.'),
                           initial_value=np.ones(self.num_channels),
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())

        self._flux_vector = np.zeros(self.num_channels)
        for i, current_channel in enumerate(current_channels):
            self.add_parameter(
                'flux_'+current_channel,
                label='flux_'+current_channel,
                unit=r'$\Phi_0$',
                get_cmd=self._gen_ch_get_func(self._get_flux, i),
                set_cmd=self._gen_ch_set_func(self._set_flux, i),
                vals=vals.Numbers())
        self.dac_mapping(current_channels)

    def _do_set_flux_vector(self, vector):
        fluxcurrent = self.fluxcurrent.get_instr()
        currents = np.dot(self.inv_transfer_matrix()*self.A_per_Phi0s(),
                          vector) + self.dac_offsets()
        for i in range(self.num_channels):
            fluxcurrent.set(self.dac_mapping()[i], currents[i])
        return currents

    def _do_get_flux_vector(self):
        fluxcurrent = self.fluxcurrent.get_instr()
        currents = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            currents[i] = fluxcurrent.get(self.dac_mapping()[i])
        self._flux_vector = np.dot(
            self._transfer_matrix, currents - self.dac_offsets())
        return self._flux_vector
