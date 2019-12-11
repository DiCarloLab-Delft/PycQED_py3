from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import numpy as np


class SimControlCZ(Instrument):
    """
    Noise and other parameters for cz_superoperator_simulation_new
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Noise parameters
        self.add_parameter(
            "T1_q0",
            unit="s",
            label="T1 fluxing qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T1_q1",
            unit="s",
            label="T1 static qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T2_q1",
            unit="s",
            label="T2 static qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T2_q0_amplitude_dependent",
            label="fitcoefficients giving T2_q0 or Tphi_q0 as a function of inverse sensitivity (in units of w_q0/Phi_0): a, b. Function is ax+b",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([-1, -1]),
        )
        # for flux noise simulations
        self.add_parameter(
            "sigma_q0",
            unit="flux quanta",
            label="standard deviation of the Gaussian from which we sample the flux bias, q0",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "sigma_q1",
            unit="flux quanta",
            label="standard deviation of the Gaussian from which we sample the flux bias, q1",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter(
            "w_q1_sweetspot",
            label="NB: different from the operating point in general",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "w_q0_sweetspot",
            label="NB: different from the operating point in general",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "Z_rotations_length",
            unit="s",
            label="duration of the single qubit Z rotations at the end of the pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "total_idle_time",
            unit="s",
            label="duration of the idle time",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        # Control parameters for the simulations
        self.add_parameter(
            "dressed_compsub",
            label="true if we use the definition of the comp subspace that uses the dressed 00,01,10,11 states",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=True,
        )
        self.add_parameter(
            "distortions",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "voltage_scaling_factor",
            unit="a.u.",
            label="scaling factor for the voltage for a CZ pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )
        self.add_parameter(
            "n_sampling_gaussian_vec",
            label="array. each element is a number of samples from the gaussian distribution. Std to guarantee convergence is [11]. More are used only to verify convergence",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([11]),
        )
        self.add_parameter(
            "cluster",
            label="true if we want to use the cluster",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "look_for_minimum",
            label="changes cost function to optimize either research of minimum of avgatefid_pc or to get the heat map in general",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )

        self.add_parameter(
            "T2_scaling",
            unit="a.u.",
            label="scaling factor for T2_q0_amplitude_dependent",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )

        self.add_parameter(
            "waiting_at_sweetspot",
            unit="s",
            label="time spent at sweetspot during the two halves of a netzero pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0),
            initial_value=0,
        )

        self.add_parameter(
            "which_gate",
            label="Direction of the CZ gate. E.g. 'NE'. Used to extract parameters from the fluxlutman ",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="NE",
        )

        self.add_parameter(
            "simstep_div",
            label="Division of the simulation time step. 4 is a good one, corresponding to a time step of 0.1 ns. For smaller values landscapes can deviate significantly from experiment.",
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=1),
            initial_value=4,
        )

        self.add_parameter(
            "gates_num",
            label="Chain the same gate gates_num times.",
            parameter_class=ManualParameter,
            vals=vals.Ints(min_value=1),
            initial_value=1,
        )

        self.add_parameter(
            "gates_interval",
            label="Time interval that separates the the gates if gates_num > 1.",
            parameter_class=ManualParameter,
            unit='s',
            vals=vals.Numbers(min_value=0),
            initial_value=0,
        )

        self.add_parameter(
            "cost_func",
            label="Used to calculate the cost function based on the quantities of interest (qoi). Signature: cost_func(qoi). NB: qoi's that represent percentages will be in [0, 1] range. Inspect 'pycqed.simulations.cz_superoperator_simulation_new_functions.simulate_quantities_of_interest_superoperator_new??' in notebook for available qoi's.",
            parameter_class=ManualParameter,
            unit='a.u.',
            vals=vals.Callable(),
            initial_value=None,
        )

        self.add_parameter(
            "cost_func_str",
            label="Not loaded automatically. Convenience parameter to store the cost function string and use `exec('sim_control_CZ.cost_func(' + sim_control_CZ.cost_func_str() + ')')` to load it.",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="lambda qoi: np.log10((1 - qoi['avgatefid_compsubspace_pc']) * (1 - 0.5) + qoi['L1'] * 0.5)",
        )

        # for ramsey/Rabi simulations

        self.add_parameter(
            "detuning",
            unit="Hz",
            label="detuning of w_q0 from its sweet spot value",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "initial_state",
            label="determines initial state for ramsey_simulations_new",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="changeme",
        )

        # for spectral tomo

        self.add_parameter(
            "repetitions",
            label="Repetitions of CZ gate, used for spectral tomo",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )
        self.add_parameter(
            "time_series",
            label="",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "overrotation_sims",
            label="instead of constant shift in flux, we use constant rotations around some axis",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "axis_overrotation",
            label="",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([1, 0, 0]),
        )
