import time

from qcodes.instrument.mock import MockInstrument, MockModel
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints
from qcodes.instrument.parameter import ManualParameter
import numpy as np


class FlippingModel(Instrument):
    '''
    Fully classical model of flipping due to Restless RB
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        self.add_parameter('F_g', units=' ',
                           label='Gate Fidelity',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('F_discr', units='',
                           label='Discrimination Fidelity',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('N_cl', units='',
                           label='Number of Cliffords',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('tau_d', units='s',
                           label='Dead time (s)',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=0)
        self.add_parameter('T1', units='s',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('P_RB', units='',
                           label='Randomized Benchmarking',
                           get_cmd=self._get_P_RB,
                           vals=Numbers())

        self.add_parameter('measured_state', units=' ',
                           label='Measured state',
                           get_cmd=self._measure,
                           vals=Numbers())
        self.add_parameter('measure_shots', units=' ',
                           label='Measured shots',
                           get_cmd=self._measure_nshots)
        self.add_parameter('N_shots', units= '',
                           parameter_class=ManualParameter,
                           vals=Ints(), initial_value=32)

        # The state is fully classical in this model
        self.add_parameter('state', units='',
                           label='Physical state',
                           parameter_class=ManualParameter,
                           vals=Enum(1, -1), initial_value=1)


        # TODO: add full butterfly

    def _get_P_RB(self):
        return .5*(2*self.F_g() - 1)**self.N_cl() + .5

    def _measure(self):
        if self.state() == -1:
            p_relax = (1-np.exp(-self.tau_d()/self.T1()))
            if np.random.rand() < p_relax:
                self.state(self.state()*-1)
        if np.random.rand() < self.P_RB():
            self.state(self.state()*-1)

        if np.random.rand() > self.F_discr():
            self.state(self.state()*-1)
            return self.state()
        else:
            return self.state()

    def _measure_nshots(self):
        """
        Measures n-shots keeping all the probabilities fixed
        """
        n = self.N_shots()
        p_relax = (1-np.exp(-self.tau_d()/self.T1()))
        state = self.state()
        P_RB = self.P_RB()
        F_discr = self.F_discr()
        measured_shots = np.empty(n)

        for i in range(n):
            # Relaxation
            if state == -1:
                if np.random.rand() < p_relax:
                    state *= -1
            if np.random.rand() < P_RB:
                state *= -1
            # Readout
            if np.random.rand() > F_discr:
                measured_shots[i] = -1*state
            else:
                measured_shots[i] = state
        return measured_shots
