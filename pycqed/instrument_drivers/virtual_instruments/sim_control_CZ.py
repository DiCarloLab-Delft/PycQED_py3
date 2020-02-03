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
            docstring="T1 fluxing qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T1_q1",
            unit="s",
            label="T1 static qubit",
            docstring="T1 static qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T2_q1",
            unit="s",
            label="T2 static qubit",
            docstring="T2 static qubit",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "T2_q0_amplitude_dependent",
            docstring="fitcoefficients giving T2_q0 or Tphi_q0 as a function of inverse sensitivity (in units of w_q0/Phi_0): a, b. Function is ax+b",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([-1, -1]),
        )
        # for flux noise simulations
        self.add_parameter(
            "sigma_q0",
            unit="flux quanta",
            docstring="standard deviation of the Gaussian from which we sample the flux bias, q0",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "sigma_q1",
            unit="flux quanta",
            docstring="standard deviation of the Gaussian from which we sample the flux bias, q1",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter(
            "w_q1_sweetspot",
            docstring="NB: different from the operating point in general",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "w_q0_sweetspot",
            docstring="NB: different from the operating point in general",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "Z_rotations_length",
            unit="s",
            docstring="duration of the single qubit Z rotations at the end of the pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "total_idle_time",
            unit="s",
            docstring="duration of the idle time",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        # Control parameters for the simulations
        self.add_parameter(
            "dressed_compsub",
            docstring="true if we use the definition of the comp subspace that uses the dressed 00,01,10,11 states",
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
            docstring="scaling factor for the voltage for a CZ pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )
        self.add_parameter(
            "n_sampling_gaussian_vec",
            docstring="array. each element is a number of samples from the gaussian distribution. Std to guarantee convergence is [11]. More are used only to verify convergence",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([11]),
        )
        self.add_parameter(
            "cluster",
            docstring="true if we want to use the cluster",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "look_for_minimum",
            docstring="changes cost function to optimize either research of minimum of avgatefid_pc or to get the heat map in general",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )

        self.add_parameter(
            "T2_scaling",
            unit="a.u.",
            docstring="scaling factor for T2_q0_amplitude_dependent",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )

        self.add_parameter(
            "waiting_at_sweetspot",
            unit="s",
            docstring="time spent at sweetspot during the two halves of a netzero pulse",
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0),
            initial_value=0,
        )

        self.add_parameter(
            "which_gate",
            docstring="Direction of the CZ gate. E.g. 'NE'. Used to extract parameters from the fluxlutman ",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="NE",
        )

        self.add_parameter(
            "simstep_div",
            docstring="Division of the simulation time step. 4 is a good one, corresponding to a time step of 0.1 ns. For smaller values landscapes can deviate significantly from experiment.",
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=1),
            initial_value=4,
        )

        self.add_parameter(
            "gates_num",
            docstring="Chain the same gate gates_num times.",
            parameter_class=ManualParameter,
            # It should be an integer but the measurement control cast to float when setting sweep points
            vals=vals.Numbers(min_value=1),
            initial_value=1,
        )

        self.add_parameter(
            "gates_interval",
            docstring="Time interval that separates the gates if gates_num > 1.",
            parameter_class=ManualParameter,
            unit='s',
            vals=vals.Numbers(min_value=0),
            initial_value=0,
        )

        self.add_parameter(
            "cost_func",
            docstring="Used to calculate the cost function based on the quantities of interest (qoi). Signature: cost_func(qoi). NB: qoi's that represent percentages will be in [0, 1] range. Inspect 'pycqed.simulations.cz_superoperator_simulation_new_functions.simulate_quantities_of_interest_superoperator_new??' in notebook for available qoi's.",
            parameter_class=ManualParameter,
            unit='a.u.',
            vals=vals.Callable(),
            initial_value=None,
        )

        self.add_parameter(
            "cost_func_str",
            docstring="Not loaded automatically. Convenience parameter to store the cost function string and use `exec('sim_control_CZ.cost_func(' + sim_control_CZ.cost_func_str() + ')')` to load it.",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="lambda qoi: np.log10((1 - qoi['avgatefid_compsubspace_pc']) * (1 - 0.5) + qoi['L1'] * 0.5)",
        )

        self.add_parameter(
            "double_cz_pi_pulses",
            docstring="If set to 'no_pi_pulses' or 'with_pi_pulses' will simulate two sequential CZs with or without Pi pulses simulated as an ideal superoperator multiplication.",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="",  # Use empty string to evaluate to false
        )

        # for ramsey/Rabi simulations

        self.add_parameter(
            "detuning",
            unit="Hz",
            docstring="detuning of w_q0 from its sweet spot value",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )
        self.add_parameter(
            "initial_state",
            docstring="determines initial state for ramsey_simulations_new",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value="changeme",
        )

        # for spectral tomo

        self.add_parameter(
            "repetitions",
            docstring="Repetitions of CZ gate, used for spectral tomo",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=1,
        )
        self.add_parameter(
            "time_series",
            docstring="",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "overrotation_sims",
            docstring="instead of constant shift in flux, we use constant rotations around some axis",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=False,
        )
        self.add_parameter(
            "axis_overrotation",
            docstring="",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
            initial_value=np.array([1, 0, 0]),
        )

    def set_cost_func(self, cost_func_str=None):
        """
        Sets the self.cost_func from the self.cost_func_str string
        or from the provided string
        """
        if cost_func_str is None:
            cost_func_str = self.cost_func_str()
        else:
            self.cost_func_str(cost_func_str)
        exec("self.cost_func(" + self.cost_func_str() + ")")


def LJP(r, R_min, depth=1., p12=12, p6=6):
    """
    Lennard-Jones potential function
    Added here to be used with adaptive sampling of a cost function that
    diverges at zero and might get the adaptive learner stucked from
    samping the rest of the landscape
    """
    return depth * ((R_min / r)**p12 - 2 * (R_min / r)**p6)


def LJP_mod(r, R_min, depth=100., p12=12, p6=6):
    """
    Modiefied Lennard-Jones potential function
    Modification: moved minum at zero and made positive
    Added here to be used with adaptive sampling of a cost function that
    diverges at zero and might get the adaptive learner stucked from
    samping the rest of the landscape
    It is a nice wrapping of a cost function because it bounds the
    [0, +inf] output of any other cost function always between
    [0, depth] so that there is always an intuition of how good an
    optimization is doing
    The derivative at zero is zero and that should help not getting the
    adaptive sampling stuck
    arctan could be used for a similar purpose but is more useful in
    experiment to have high slope at zero
    """
    return LJP(r + R_min, R_min, depth=depth, p12=p12, p6=p6) + depth
