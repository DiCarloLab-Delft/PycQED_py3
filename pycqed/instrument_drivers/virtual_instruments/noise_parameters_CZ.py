from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

import numpy as np





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
        self.add_parameter('T2_q0_amplitude_dependent', unit='a.u.',
                           label='fitcoefficients giving T2_q0 as a function of detuning',
                           parameter_class=ManualParameter,
                           vals=vals.Arrays(), initial_value=np.array([-1]))
        self.add_parameter('w_bus', unit='Hz',
                           label='omega of the bus resonator',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=8.08e9*2*np.pi)





















