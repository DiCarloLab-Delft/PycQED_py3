from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

import numpy as np


eV_to_Hz = 1/4.1357e-15


class NoiseParametersCZ(Instrument):
    '''
    Fully classical model of flipping due to Restless RB
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        self.add_parameter('T1_q0', unit='s',
                           label='T1 fluxing qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=34e-6)
        self.add_parameter('T1_q1', unit='s',
                           label='T1 static qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=42e-6)
        self.add_parameter('T2_q0_sweetspot', unit='s',
                           label='T2 fluxing qubit at sweetspot (maximal T2). T2 (or better Tphi) at intermediate points is interpolated based on the derivative of the cosine',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=23e-6)
        self.add_parameter('T2_q1', unit='s',
                           label='T2 static qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=23e-6)
        self.add_parameter('T2_q0_interaction_point', unit='s',
                           label='T2 fluxing qubit at the interaction point (minimal T2). T2 (or better Tphi) at intermediate points is interpolated based on the derivative of the cosine',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=10e-6)
        self.add_parameter('distortions',           # it's not the best point to save it but it seemed the best one
                           initial_value=True,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('voltage_scaling_factor', unit='a.u.',
                           label='scaling factor for the voltage for a CZ pulse',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=1.0)
        self.add_parameter('T2_q0_amplitude_dependent', unit='Hz, a.u., s',
                           label='fitcoefficients giving T2echo_q0 as a function of frequency_q0: gc, amp, tau. Function is gc+gc*amp*np.exp(-x/tau)',
                           parameter_class=ManualParameter,
                           vals=vals.Arrays(), initial_value=np.array([-1,-1,-1]))
        self.add_parameter('w_bus', unit='Hz',
                           label='omega of the bus resonator',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=8.08e9*2*np.pi)     

        # for qdots simulations
        self.add_parameter('detuning', unit='meV',
                           label='detuning in meV between the two valleys of the dots',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=0)
        self.add_parameter('cz_time_offset', unit='ns',
                           label='how much we want the length of a square pulse to differ from the length needed to do a CZ without noise',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=0)
        self.add_parameter('t_hopping', unit='MHz',
                           label='t_hopping parameter in the hamiltonian',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=210)
        self.add_parameter('w_q1', unit='Hz',
                           label='NB: qdots',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=18.4e9 * 2*np.pi)
        self.add_parameter('w_q2', unit='Hz',
                           label='NB: qdots',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=19.7e9 * 2*np.pi)
        self.add_parameter('U_q1', unit='eV',
                           label='NB: qdots',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=3.5e-3 * eV_to_Hz * 2*np.pi)
        self.add_parameter('U_q2', unit='eV',
                           label='NB: qdots',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=3.5e-3 * eV_to_Hz * 2*np.pi)





















