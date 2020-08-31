from importlib import reload
from pycqed.measurement import measurement_control as mc

import adaptive
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman_vcz as flm

from pycqed.instrument_drivers.virtual_instruments import sim_control_CZ_v2 as scCZ_v2
from pycqed.simulations import cz_superoperator_simulation_functions_v2 as czf_v2
from pycqed.measurement.waveform_control_CC import waveforms_vcz as wf_vcz

from pycqed.analysis_v2 import measurement_analysis as ma2

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
reload(wf_vcz)

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
    adaptive_pars = arglist["adaptive_pars"]
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

    d = CZ_trajectory_superoperator(
        fluxlutman=fluxlutman,
        fluxlutman_static=fluxlutman_static,
        sim_control_CZ=sim_control_CZ,
        fitted_stepresponse_ty=fitted_stepresponse_ty,
        qois=additional_pars['qois'],
    )
    MC.set_detector_function(d)

    if exp_metadata["mode"] == "adaptive":
        MC.set_sweep_functions(
            [
                getattr(fluxlutman, "vcz_amp_sq_{}".format(which_gate)),
                getattr(fluxlutman, "vcz_amp_fine_{}".format(which_gate)),
            ]
        )

        MC.set_adaptive_function_parameters(adaptive_pars)

        if sim_control_CZ.cluster():
            dat = MC.run(
                additional_pars["label"]+"_cluster",
                mode="adaptive",
                exp_metadata=exp_metadata,
            )

        else:
            if additional_pars["long_name"]:
                dat = MC.run(
                    additional_pars["label"],
                    mode="adaptive",
                    exp_metadata=exp_metadata,
                )
            else:
                dat = MC.run(
                "2D_simulations_v2",
                mode="adaptive",
                exp_metadata=exp_metadata,
            )


    elif exp_metadata["mode"] == "contour_scan":
        
        from pycqed.analysis_v2.tools import contours2d as c2d
        from pycqed.measurement import sweep_functions as swf

        timestamp = sim_control_CZ.timestamp_for_contour()
        coha_for_contour = ma2.Conditional_Oscillation_Heatmap_Analysis(
                    t_start=timestamp,
                    t_stop=timestamp,
                    close_figs=True,
                    extract_only=False,
                    plt_orig_pnts=True,
                    plt_contour_L1=False,
                    plt_contour_phase=True,
                    plt_optimal_values=True,
                    plt_optimal_values_max=1,
                    find_local_optimals=True,
                    plt_clusters=False,
                    cluster_from_interp=False,
                    clims={
                        "Cost func": [0., 100],
                        "missing fraction": [0, 30],
                        "offset difference": [0, 30]
                    },
                    target_cond_phase=180,
                    phase_thr=15,
                    L1_thr=5,
                    clustering_thr=0.15,
                    gen_optima_hulls=True,
                    hull_L1_thr=10,
                    hull_phase_thr=20,
                    plt_optimal_hulls=True,
                    save_cond_phase_contours=[180],
                )

        c_180 = coha_for_contour.proc_data_dict["quantities_of_interest"]["cond_phase_contours"]["180"]["0"]
        hull = coha_for_contour.proc_data_dict["quantities_of_interest"]["hull_vertices"]["0"]

        c_180_in_hull = c2d.pnts_in_hull(pnts=c_180, hull=hull)
        if c_180_in_hull[0][0] > c_180_in_hull[-1][0]:
            c_180_in_hull = np.flip(c_180_in_hull, axis=0)

        swf_2d_contour = swf.SweepAlong2DContour(getattr(fluxlutman, "vcz_amp_sq_{}".format(which_gate)), 
                                                 getattr(fluxlutman, "vcz_amp_fine_{}".format(which_gate)), 
                                                 c_180_in_hull)
        MC.set_sweep_function(swf_2d_contour)
        MC.set_sweep_points(np.linspace(0, 1, 40))

        if sim_control_CZ.cluster():
            dat = MC.run(
                additional_pars["label"]+"_cluster",
                mode="1D",
                exp_metadata=exp_metadata,
            )

        else:
            if additional_pars["long_name"]:
                dat = MC.run(
                    additional_pars["label"],
                    mode="1D",
                    exp_metadata=exp_metadata,
                )
            else:
                dat = MC.run(
                "contour_scan",
                mode="1D",
                exp_metadata=exp_metadata,
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
    gates_num = int(sim_control_CZ.gates_num())  # repeat the same gate this number of times
    gates_interval = sim_control_CZ.gates_interval()  # idle time between repeated gates

    sim_step = 1 / fluxlutman.sampling_rate()
    subdivisions_of_simstep = (
        sim_control_CZ.simstep_div()
    )  # 4 is a good one, corresponding to a time step of 0.1 ns
    sim_step_new = (
        sim_step / subdivisions_of_simstep
    )  # waveform is generated according to sampling rate of AWG

    wf_generator = getattr(
        wf_vcz,
        fluxlutman.get("cz_wf_generator_{}".format(which_gate))
    )

    wfd = wf_generator(
        fluxlutman=fluxlutman,
        sim_ctrl_cz=sim_control_CZ,
    )
    intervals_list = wfd["intervals_list"]
    amp = wfd["amp"]

    tlist_new = wfd["time"]

    # Apply voltage scaling
    # [2020-05-30] probably not needed anymore
    amp = amp * sim_control_CZ.voltage_scaling_factor()

    # For fine tuning of the waiting in the middle for matching sim-exp or studying interference fringes
    if sim_control_CZ.artificial_waiting_at_sweetspot() != 0 and not sim_control_CZ.get("optimize_const_amp"):
        index_middle = np.where(amp[1:] == 0)[0][0] + 1
        amp = np.insert(amp, index_middle, np.zeros(sim_control_CZ.artificial_waiting_at_sweetspot()))
        intervals_list = np.insert(intervals_list, index_middle, np.zeros(sim_control_CZ.artificial_waiting_at_sweetspot()) + sim_step_new)
        tlist_new = np.concatenate((tlist_new, np.arange(1, sim_control_CZ.artificial_waiting_at_sweetspot()+1)*sim_step_new + tlist_new[-1]))

    # Apply distortions
    if sim_control_CZ.distortions():
        amp_final = czf_v2.distort_amplitude(
            fitted_stepresponse_ty=fitted_stepresponse_ty,
            amp=amp,
            tlist_new=tlist_new,
            sim_step_new=sim_step_new,
        )
    else:
        amp_final = amp
    # czf_v2.plot(x_plot_vec=[np.array(tlist_new)*1e9, np.array(tlist_new)*1e9],y_plot_vec=[fluxlutman.calc_amp_to_freq(amp, '01', which_gate=which_gate) / 1e9,
    #                                                                                    fluxlutman.calc_amp_to_freq(amp_final, '01', which_gate=which_gate) / 1e9],
    #                          title='Pulse with and without distortions',
    #                          xlabel='Time (ns)',ylabel='Frequency (GHz)',
    #                          legend_labels = ['without', 'with'])

    # The fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    # [2020-05-30] the waveform generator includes corrections if desired
    # WARNING: shift_due_to_fluxbias is not ready for waveforms that include the distortions
    if sim_control_CZ.sigma_q0() != 0:
        amp_final = czf_v2.shift_due_to_fluxbias_q0(
            fluxlutman=fluxlutman,
            amp_final=amp_final,
            fluxbias_q0=fluxbias_q0,
            sim_control_CZ=sim_control_CZ,
            which_gate=which_gate,
        )

    if gates_num > 1:
        if gates_interval > 0:
            # This is intended to make the simulation faster by skipping
            # all the amp = 0 steps, verified to encrease sim speed
            # 4.7s/data point -> 4.0s/data point
            # Errors in simulation outcomes are < 1e-10
            actual_gates_interval = (
                np.arange(0, gates_interval, sim_step_new)[-1] + sim_step_new
            )

            # We add an extra small step to ensure the amp signal goes to
            # zero first
            interval_append = np.concatenate(
                ([sim_step_new, actual_gates_interval - sim_step_new], intervals_list)
            )
            amp_append = np.concatenate(([0, 0], amp_final))
        else:
            interval_append = intervals_list
            amp_append = amp_final

        # Append arbitrary number of same gate
        for gate in range(gates_num - 1):
            amp_final = np.append(amp_final, amp_append)
            intervals_list = np.append(intervals_list, interval_append)

    t_final = np.sum(intervals_list)  # actual overall gate length

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
        intervals_list=intervals_list,
        which_gate=which_gate,
    )

    return [U_final, t_final]


class CZ_trajectory_superoperator(det.Soft_Detector):
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

        self.value_names = [
            "Cost func",
            "Cond phase",
            "L1",
            "L2",
            "avgatefid_pc",
            "avgatefid_compsubspace_pc",
            "phase_q0",
            "phase_q1",
            "avgatefid_compsubspace",
            "avgatefid_compsubspace_pc_onlystaticqubit",
            "population_02_state",
            "cond_phase02",
            "coherent_leakage11",
            "offset_difference",
            "missing_fraction",
            "12_21_population_transfer",
            "12_03_population_transfer",
            "phase_diff_12_02",
            "phase_diff_21_20",
            "cond_phase12",
            "cond_phase21",
            "cond_phase03",
            "cond_phase20",
            "vcz_amp_sq",
            "vcz_amp_fine",
            "population_transfer_01_10"
        ]
        self.value_units = [
            "a.u.",
            "deg",
            "%",
            "%",
            "%",
            "%",
            "deg",
            "deg",
            "%",
            "%",
            "%",
            "deg",
            "%",
            "%",
            "%",
            "%",
            "%",
            "deg",
            "deg",
            "deg",
            "deg",
            "deg",
            "deg",
            "a.u.",
            "a.u.",
            "a.u."
        ]

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
                if self.sim_control_CZ.double_cz_pi_pulses() != "":
                    # Experimenting with single qubit ideal pi pulses
                    if self.sim_control_CZ.double_cz_pi_pulses() == "with_pi_pulses":
                        pi_single_qubit = qtp.Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                        # pi_pulse = qtp.tensor(pi_single_qubit, qtp.qeye(n_levels_q0))
                        pi_op = qtp.tensor(pi_single_qubit, pi_single_qubit)
                        # pi_super_op = qtp.to_super(pi_op)
                        U_final = result_list[0]
                        U_final = pi_op * U_final * pi_op * U_final
                    elif self.sim_control_CZ.double_cz_pi_pulses() == "no_pi_pulses":
                        U_final = result_list[0]
                        U_final = U_final * U_final
                    t_final = 2 * result_list[1]
                else:
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

            qoi = czf_v2.simulate_quantities_of_interest_superoperator_new(
                U=U_superop_average,
                t_final=t_final,
                fluxlutman=self.fluxlutman,
                fluxlutman_static=self.fluxlutman_static,
                sim_control_CZ=self.sim_control_CZ,
                which_gate=self.sim_control_CZ.which_gate(),
            )

            # if we look only for the minimum avgatefid_pc in the heat maps,
            # then we optimize the search via higher-order cost function
            if self.sim_control_CZ.cost_func() is not None:
                cost_func_val = self.sim_control_CZ.cost_func()(qoi)
            elif self.sim_control_CZ.look_for_minimum():
                cost_func_val = (
                    np.log10(1 - qoi["avgatefid_compsubspace_pc"])
                ) ** 4  # sign removed for even powers
            else:
                cost_func_val = -np.log10(1 - qoi["avgatefid_compsubspace_pc"])

            quantities_of_interest = [
                cost_func_val,
                qoi["phi_cond"],
                qoi["L1"] * 100,
                qoi["L2"] * 100,
                qoi["avgatefid_pc"] * 100,
                qoi["avgatefid_compsubspace_pc"] * 100,
                qoi["phase_q0"],
                qoi["phase_q1"],
                qoi["avgatefid_compsubspace"] * 100,
                qoi["avgatefid_compsubspace_pc_onlystaticqubit"] * 100,
                qoi["population_02_state"] * 100,
                qoi["cond_phase02"],
                qoi["coherent_leakage11"] * 100,
                qoi["offset_difference"] * 100,
                qoi["missing_fraction"] * 100,
                qoi["population_transfer_12_21"] * 100,
                qoi["population_transfer_12_03"] * 100,
                qoi["phase_diff_12_02"],
                qoi["phase_diff_21_20"],
                qoi["cond_phase12"],
                qoi["cond_phase21"],
                qoi["cond_phase03"],
                qoi["cond_phase20"],
                self.fluxlutman.get("vcz_amp_sq_{}".format(self.sim_control_CZ.which_gate())),
                self.fluxlutman.get("vcz_amp_fine_{}".format(self.sim_control_CZ.which_gate())),
                qoi["population_transfer_01_10"]
            ]
            qoi_vec = np.array(quantities_of_interest)
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
            qoi_plot[0, 1],
            qoi_plot[0, 2],
            qoi_plot[0, 3],
            qoi_plot[0, 4],
            qoi_plot[0, 5],
            qoi_plot[0, 6],
            qoi_plot[0, 7],
            qoi_plot[0, 8],
            qoi_plot[0, 9],
            qoi_plot[0, 10],
            qoi_plot[0, 11],
            qoi_plot[0, 12],
            qoi_plot[0, 13],
            qoi_plot[0, 14],
            qoi_plot[0, 15],
            qoi_plot[0, 16],
            qoi_plot[0, 17],
            qoi_plot[0, 18],
            qoi_plot[0, 19],
            qoi_plot[0, 20],
            qoi_plot[0, 21],
            qoi_plot[0, 22],
            qoi_plot[0, 23],
            qoi_plot[0, 24],
            qoi_plot[0, 25]
        ]
        if self.qois != "all":
            return np.array(return_values)[self.qoi_mask]

        else:
            return return_values
