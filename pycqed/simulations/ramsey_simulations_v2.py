from importlib import reload
from pycqed.measurement import measurement_control as mc

import adaptive
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman_vcz as flm

from pycqed.instrument_drivers.virtual_instruments import sim_control_CZ_v2 as scCZ_v2
from pycqed.simulations import cz_superoperator_simulation_functions_v2 as czf_v2
from pycqed.measurement.waveform_control_CC import waveforms_vcz as wfl_dev

import numpy as np
from pycqed.measurement import detector_functions as det
import matplotlib.pyplot as plt
from qcodes import Instrument

from scipy.interpolate import interp1d
import qutip as qtp
import cma

import logging

reload(scCZ_v2)
reload(czf_v2)
reload(wfl_dev)

np.set_printoptions(threshold=np.inf)
log = logging.getLogger(__name__)


def f_to_parallelize_v2(arglist):
    # cluster wants a list as an argument.
    # Below the various list items are assigned to their own variable

    fitted_stepresponse_ty = arglist["fitted_stepresponse_ty"]
    fluxlutman_args = arglist[
        "fluxlutman_args"
    ]  # see function return_instrument_args in czf_v2
    fluxlutman_static_args = arglist[
        "fluxlutman_static_args"
    ]  # see function return_instrument_args in czf_v2
    sim_control_CZ_args = arglist[
        "sim_control_CZ_args"
    ]  # see function return_instrument_args in czf_v2
    number = arglist["number"]
    additional_pars = arglist["additional_pars"]
    live_plot_enabled = arglist["live_plot_enabled"]
    exp_metadata = arglist["exp_metadata"]
    #which_gate = arglist["which_gate"]

    try:
        MC = Instrument.find_instrument("MC" + "{}".format(number))
    except KeyError:
        MC = mc.MeasurementControl(
            "MC" + "{}".format(number), live_plot_enabled=live_plot_enabled
        )
    from qcodes import station

    station = station.Station()
    station.add_component(MC)
    MC.station = station

    fluxlutman = flm.HDAWG_Flux_LutMan("fluxlutman" + "{}".format(number))
    station.add_component(fluxlutman)
    fluxlutman_static = flm.HDAWG_Flux_LutMan("fluxlutman_static" + "{}".format(number))
    station.add_component(fluxlutman_static)
    sim_control_CZ = scCZ_v2.SimControlCZ_v2("sim_control_CZ" + "{}".format(number))
    station.add_component(sim_control_CZ)

    fluxlutman = czf_v2.return_instrument_from_arglist_v2(fluxlutman, fluxlutman_args)
    fluxlutman_static = czf_v2.return_instrument_from_arglist_v2(fluxlutman_static, fluxlutman_static_args)
    sim_control_CZ = czf_v2.return_instrument_from_arglist_v2(sim_control_CZ, sim_control_CZ_args)

    sim_control_CZ.set_cost_func()
    which_gate = sim_control_CZ.which_gate()

    d = Ramsey_experiment(
        fluxlutman=fluxlutman,
        fluxlutman_static=fluxlutman_static,
        sim_control_CZ=sim_control_CZ,
        fitted_stepresponse_ty=fitted_stepresponse_ty,
        qois="all",
    )
    MC.set_detector_function(d)

    if additional_pars["mode"] == "1D_ramsey":
        MC.set_sweep_functions([sim_control_CZ.scanning_time])
        MC.set_sweep_points(
            np.arange(0, additional_pars['max_time'], additional_pars['time_step'])
        )
        if sim_control_CZ.cluster():
            dat = MC.run(
                "1D ramsey_v2_cluster double sided {} - sigma_q0 {:.0f} - detuning {:.0f}".format(
                    sim_control_CZ.get("czd_double_sided"),
                    sim_control_CZ.sigma_q0() * 1e6,
                    sim_control_CZ.detuning() / 1e6
                ),
                mode="1D",
                exp_metadata=exp_metadata,
            )

        else:
            if additional_pars["long_name"]:
                dat = MC.run(
                    "1D ramsey_v2 double sided {} - sigma_q0 {:.0f} - detuning {:.0f}".format(
                    sim_control_CZ.get("czd_double_sided"),
                    sim_control_CZ.sigma_q0() * 1e6,
                    sim_control_CZ.detuning() / 1e6
                ),
                mode="1D",
                exp_metadata=exp_metadata,
                )
            else:
                dat = MC.run(
                    "1D ramsey_v2", exp_metadata=exp_metadata, mode="1D"
                )

    fluxlutman.close()
    fluxlutman_static.close()
    sim_control_CZ.close()
    MC.close()


def compute_propagator(arglist):
    # I was parallelizing this function in the cluster, then I changed but the list as an argument remains.
    # Below each list item is assigned to its own variable

    fluxbias_q0 = arglist["fluxbias_q0"]
    fluxbias_q1 = arglist["fluxbias_q1"]
    fitted_stepresponse_ty = arglist["fitted_stepresponse_ty"]
    fluxlutman = arglist["fluxlutman"]
    fluxlutman_static = arglist["fluxlutman_static"]
    sim_control_CZ = arglist["sim_control_CZ"]

    which_gate = sim_control_CZ.which_gate()

    sim_step = sim_control_CZ.get("scanning_time")
    subdivisions_of_simstep = 1
    sim_step_new = (
        sim_step / subdivisions_of_simstep
    )  # waveform is generated according to sampling rate of AWG

    tlist = [0]
    tlist_new = tlist


    freq = sim_control_CZ.w_q0_sweetspot() + sim_control_CZ.detuning()
    amp = [fluxlutman.calc_freq_to_amp(freq)]


    t_final = tlist_new[-1]+sim_step_new


    # Apply voltage scaling
    # [2020-05-30] probably not needed anymore
    amp = amp * sim_control_CZ.voltage_scaling_factor()
    amp_final = amp

    ### the fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    #   Since we assume the hamiltonian to be constant on each side of the pulse, we just need two time steps
    if sim_control_CZ.get("czd_double_sided"):
        amp_final=[amp_final[0],fluxlutman.calc_freq_to_amp(freq,positive_branch=False)]    # Echo-Z
    else:
        amp_final=[amp_final[0],amp_final[0]]     # Ram-Z
    sim_step_new=sim_step_new/2
    amp_final = czf_v2.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,
        amp_final=amp_final,fluxbias_q0=fluxbias_q0,sim_control_CZ=sim_control_CZ,
        which_gate=which_gate)

    # Obtain jump operators for Lindblad equation
    c_ops = czf_v2.return_jump_operators(
        sim_control_CZ=sim_control_CZ,
        amp_final=amp_final,
        fluxlutman=fluxlutman,
        which_gate=which_gate,
    )

    # Compute propagator
    U_final = czf_v2.time_evolution_new(
        c_ops=c_ops,
        sim_control_CZ=sim_control_CZ,
        fluxlutman_static=fluxlutman_static,
        fluxlutman=fluxlutman,
        fluxbias_q1=fluxbias_q1,
        amp=amp_final,
        sim_step=sim_step_new,
        which_gate=which_gate,
    )

    # important to use amp and NOT amp_final here because the fluxbias is random and unknown to us.
    U_final = czf_v2.rotating_frame_transformation_propagator_new(U=U_final, t=t_final,
        H=czf_v2.calc_hamiltonian(amp[0],fluxlutman,fluxlutman_static,which_gate))

    return [U_final, t_final]


class Ramsey_experiment(det.Soft_Detector):
    def __init__(
        self,
        fluxlutman,
        sim_control_CZ,
        fluxlutman_static,
        fitted_stepresponse_ty=None,
        qois="all",
    ):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                                required to generate the waveform for the trajectory, and the hamiltonian as well.
            sim_control_CZ: instrument that contains the noise parameters, plus some more
            fitted_stepresponse_ty: list of two elements, corresponding to the time t
                                    and the step response in volts along the y axis
            qois: list
                list of quantities of interest, this can be used to return
                only a select set of values. The list should contain
                entries of "value_names". if qois=='all', all quantities are returned.
        Structure: compute input parameters necessary to compute time evolution (propagator), then compute quantities of interest
        Returns: quantities of interest
        """

        super().__init__()

        self.value_names = ['population_higher_state','population_lower_state']
        self.value_units = ['%', '%']

        self.qois = qois
        if self.qois != "all":
            self.qoi_mask = [self.value_names.index(q) for q in qois]
            self.value_names = list(np.array(self.value_names)[self.qoi_mask])
            self.value_units = list(np.array(self.value_units)[self.qoi_mask])

        self.fluxlutman = fluxlutman
        self.fluxlutman_static = fluxlutman_static
        self.sim_control_CZ = sim_control_CZ

        if fitted_stepresponse_ty is None:
            self.fitted_stepresponse_ty = [np.array(1), np.array(1)]
        else:
            # list of 2 elements: stepresponse (=y) as a function of time (=t)
            self.fitted_stepresponse_ty = fitted_stepresponse_ty

    def acquire_data_point(self, **kw):

        # Discretize average (integral) over a Gaussian distribution
        mean = 0
        sigma_q0 = self.sim_control_CZ.sigma_q0()
        sigma_q1 = (
            self.sim_control_CZ.sigma_q1()
        )  # one for each qubit, in units of Phi_0

        qoi_plot = (
            []
        )  # used to verify convergence properties. If len(n_sampling_gaussian_vec)==1, it is useless

        # 11 guarantees excellent convergence.
        # We choose it odd so that the central point of the Gaussian is included.
        # Always choose it odd
        n_sampling_gaussian_vec = self.sim_control_CZ.n_sampling_gaussian_vec()

        for n_sampling_gaussian in n_sampling_gaussian_vec:
            # If sigma=0 there's no need for sampling
            if sigma_q0 != 0:
                samplingpoints_gaussian_q0 = np.linspace(
                    -5 * sigma_q0, 5 * sigma_q0, n_sampling_gaussian
                )  # after 5 sigmas we cut the integral
                delta_x_q0 = (
                    samplingpoints_gaussian_q0[1] - samplingpoints_gaussian_q0[0]
                )
                values_gaussian_q0 = czf_v2.gaussian(
                    samplingpoints_gaussian_q0, mean, sigma_q0
                )
            else:
                samplingpoints_gaussian_q0 = np.array([0])
                delta_x_q0 = 1
                values_gaussian_q0 = np.array([1])
            if sigma_q1 != 0:
                samplingpoints_gaussian_q1 = np.linspace(
                    -5 * sigma_q1, 5 * sigma_q1, n_sampling_gaussian
                )  # after 5 sigmas we cut the integral
                delta_x_q1 = (
                    samplingpoints_gaussian_q1[1] - samplingpoints_gaussian_q1[0]
                )
                values_gaussian_q1 = czf_v2.gaussian(
                    samplingpoints_gaussian_q1, mean, sigma_q1
                )
            else:
                samplingpoints_gaussian_q1 = np.array([0])
                delta_x_q1 = 1
                values_gaussian_q1 = np.array([1])

            # This is actually the input that was parallelized in an old version.
            # Currently it just creates a list that is provided sequentially to compute_propagator
            input_to_parallelize = []

            weights = []
            number = (
                -1
            )  # used to number instruments that are created in the parallelization, to avoid conflicts in the cluster

            for j_q0 in range(len(samplingpoints_gaussian_q0)):
                fluxbias_q0 = samplingpoints_gaussian_q0[j_q0]  # q0 fluxing qubit
                for j_q1 in range(len(samplingpoints_gaussian_q1)):
                    fluxbias_q1 = samplingpoints_gaussian_q1[j_q1]  # q1 spectator qubit

                    input_point = {
                        "fluxbias_q0": fluxbias_q0,
                        "fluxbias_q1": fluxbias_q1,
                        "fluxlutman": self.fluxlutman,
                        "fluxlutman_static": self.fluxlutman_static,
                        "sim_control_CZ": self.sim_control_CZ,
                        "fitted_stepresponse_ty": self.fitted_stepresponse_ty,
                    }

                    weight = (
                        values_gaussian_q0[j_q0]
                        * delta_x_q0
                        * values_gaussian_q1[j_q1]
                        * delta_x_q1
                    )
                    weights.append(weight)

                    input_to_parallelize.append(input_point)

            U_final_vec = []
            t_final_vec = []
            for input_arglist in input_to_parallelize:
                result_list = compute_propagator(input_arglist)
                U_final = result_list[0]
                t_final = result_list[1]
                U_final_vec.append(U_final)
                t_final_vec.append(t_final)

            t_final = t_final_vec[
                0
            ]  # equal for all entries, we need it to compute phases in the rotating frame
            # needed to compute phases in the rotating frame, not used anymore
            # w_q0, w_q1, alpha_q0, alpha_q1 = czf_v2.dressed_frequencies(self.fluxlutman, self.fluxlutman_static, self.sim_control_CZ, which_gate=self.sim_control_CZ.which_gate())

            # Reproducing Leo's plots of cond_phase and leakage vs. flux offset (I order vs II order)
            # czf_v2.sensitivity_to_fluxoffsets(U_final_vec,input_to_parallelize,t_final,self.fluxlutman,self.fluxlutman_static, which_gate=self.sim_control_CZ.which_gate())

            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == "oper":
                    U_final_vec[i] = qtp.to_super(
                        U_final_vec[i]
                    )  # weighted averaging needs to be done for superoperators
                U_final_vec[i] = U_final_vec[i] * weights[i]
            U_superop_average = sum(
                U_final_vec
            )  # computing resulting average propagator
            # print(czf_v2.verify_CPTP(U_superop_average))

            qoi = czf_v2.quantities_of_interest_ramsey(U=U_superop_average,
                initial_state=self.sim_control_CZ.initial_state(),
                fluxlutman=self.fluxlutman,
                fluxlutman_static=self.fluxlutman_static,
                sim_control_CZ=self.sim_control_CZ,
                which_gate=self.sim_control_CZ.which_gate())

            quantities_of_interest = [qoi['population_higher_state'], qoi['population_lower_state']]
            qoi_vec=np.array(quantities_of_interest)
            qoi_plot.append(qoi_vec)

            # To study the effect of the coherence of leakage on repeated CZs (simpler than simulating a full RB experiment):
            # czf_v2.repeated_CZs_decay_curves(U_superop_average,t_final,self.fluxlutman,self.fluxlutman_static, which_gate=self.sim_control_CZ.which_gate())

            # czf_v2.plot_spectrum(self.fluxlutman,self.fluxlutman_static, which_gate=self.sim_control_CZ.which_gate())

        qoi_plot = np.array(qoi_plot)

        # Uncomment to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czf_v2.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])

        return_values = [
            qoi_plot[0, 0],
            qoi_plot[0, 1]
        ]
        if self.qois != "all":
            return np.array(return_values)[self.qoi_mask]

        else:
            return return_values
