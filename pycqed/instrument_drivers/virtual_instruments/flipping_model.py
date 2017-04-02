import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints
from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis.tools.data_manipulation import count_error_fractions


class FlippingModel(Instrument):
    '''
    Fully classical model of flipping due to Restless RB
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        self.add_parameter('F_g', unit=' ',
                           label='Gate Fidelity',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('F_discr', unit='',
                           label='Discrimination Fidelity',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('N_cl', unit='',
                           label='Number of Cliffords',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('tau_d', unit='s',
                           label='Dead time (s)',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=0)
        self.add_parameter('T1', unit='s',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=1)
        self.add_parameter('T1_sigma', unit='s',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=0)

        self.add_parameter('P_RB', unit='',
                           label='Randomized Benchmarking',
                           get_cmd=self._get_P_RB,
                           vals=Numbers())

        self.add_parameter('measured_state', unit=' ',
                           label='Measured state',
                           get_cmd=self._measure,
                           vals=Numbers())
        self.add_parameter('measure_shots', unit=' ',
                           label='Measured shots',
                           get_cmd=self._measure_nshots)
        self.add_parameter('N_shots', unit='',
                           parameter_class=ManualParameter,
                           vals=Ints(), initial_value=32)
        self.add_parameter('err_frac', label='Error fraction',
                           unit='',
                           get_cmd=self._measure_err_frac)

        # The state is fully classical in this model
        self.add_parameter('state', unit='',
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
        if self.T1_sigma() != 0:
            T1 = np.random.normal(self.T1(), self.T1_sigma(), 1)[0]
        else:
            T1 = self.T1()
        p_relax = (1-np.exp(-self.tau_d()/T1))
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

    def _measure_err_frac(self):
        vals = self.measure_shots()
        fracs = np.array(count_error_fractions(vals))/self.N_shots()
        return fracs[1]
