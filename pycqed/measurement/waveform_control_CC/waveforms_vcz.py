"""
    Author: Victor Negîrneac
    Purpose: generate flux waveforms for VCZ gates and
        phase corrections; toolbox for vcz waveforms
"""

import numpy as np
import math
import logging
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


def add_vcz_parameters(this_flux_lm, which_gate: str = None):
    """
    Adds to `this_flux_lm` the necessary parameters used for the VCZ
    flux waveform including corrections
    """
    this_flux_lm.add_parameter(
        "vcz_amp_dac_at_11_02_%s" % which_gate,
        docstring="DAC amplitude (in the case of HDAWG) at the 11-02 "
        "interaction point. NB: the units might be different for some "
        "other AWG that is distinct from the HDAWG.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0.0, 10.0),
        initial_value=0.5,
        unit="a.u.",
        label="DAC amp. at the interaction point",
    )
    this_flux_lm.add_parameter(
        "vcz_amp_sq_%s" % which_gate,
        docstring="Amplitude of the square parts of the NZ pulse. "
        "1.0 means qubit detuned to the 11-02 interaction point.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0.0, 10.0),
        initial_value=1.0,
        unit="a.u.",
        label="Square relative amp.",
    )
    this_flux_lm.add_parameter(
        "vcz_amp_fine_%s" % which_gate,
        docstring="Amplitude of the single sample point inserted at "
        "the end of the first half of the NZ pulse and at the "
        "beginning of the second half. "
        "1.0 means same amplitude as `sq_amp_XX`.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0.0, 1.0),
        initial_value=.5,
        unit="a.u.",
        label="Fine tuning amp.",
    )
    this_flux_lm.add_parameter(
        "vcz_use_amp_fine_%s" % which_gate,
        docstring="",
        parameter_class=ManualParameter,
        vals=vals.Bool(),
        initial_value=True,
        label="Add extra point with amplitude `vcz_amp_fine_XX`?",
    )
    this_flux_lm.add_parameter(
        "vcz_amp_q_ph_corr_%s" % which_gate,
        docstring="Amplitude at the squares of the NZ pulse for single "
        "qubit phase correction.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0.0, 1.0),
        initial_value=0.,
        unit="a.u.",
        label="Amp. phase correction",
    )
    this_flux_lm.add_parameter(
        "vcz_time_q_ph_corr_%s" % which_gate,
        docstring="Total time of the single qubit phase correction NZ pulse.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0.0, 500e-9),
        initial_value=0.,
        unit="s",
        label="Time phase correction",
    )
    this_flux_lm.add_parameter(
        "vcz_correct_q_phase_%s" % which_gate,
        docstring="",
        parameter_class=ManualParameter,
        vals=vals.Bool(),
        initial_value=False,
        label="Correct single Q phase?",
    )
    this_flux_lm.add_parameter(
        "vcz_time_single_sq_%s" % which_gate,
        docstring="Duration of each square. "
        "You should set it close to half speed limit (minimum "
        "time required to perform a full swap, i.e. 11 -> 02 -> 11)",
        parameter_class=ManualParameter,
        vals=vals.Numbers(1.0 / 2.4e9, 500e-9),
        initial_value=15.5555555e-9,
        unit="s",
        label="Duration single square",
    )
    this_flux_lm.add_parameter(
        "vcz_time_middle_%s" % which_gate,
        docstring="Time between the two square parts.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0., 500e-9),
        initial_value=0.,
        unit="s",
        label="Time between squares",
    )
    this_flux_lm.add_parameter(
        "vcz_time_before_q_ph_corr_%s" % which_gate,
        docstring="Time after main pulse before single qubit phase "
        "correction.",
        parameter_class=ManualParameter,
        vals=vals.Numbers(0., 500e-9),
        initial_value=0.,
        unit="s",
        label="Time before correction",
    )

    for specificity in ["coarse", "fine"]:
        this_flux_lm.add_parameter(
            "vcz_{}_optimal_hull_{}".format(specificity, which_gate),
            initial_value=np.array([]),
            label="{} hull".format(specificity),
            docstring=(
                "Stores the boundary points of a optimal region 2D region "
                "generated from a landscape. Intended for data points "
                "(x, y) = (`vcz_amp_sq_XX`, `vcz_time_middle_XX`)"
            ),
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
        )
        this_flux_lm.add_parameter(
            "vcz_{}_cond_phase_contour_{}".format(specificity, which_gate),
            initial_value=np.array([]),
            label="{} contour".format(specificity),
            docstring=(
                "Stores the points for an optimal conditional phase "
                "contour generated from a landscape. Intended for data points "
                "(x, y) = (`vcz_amp_sq_XX`, `vcz_time_middle_XX`) "
                "typically for the 180 deg cond. phase."
            ),
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
        )


def align_vcz_q_phase_corr_with(
    this_flux_lm,
    this_which_gate: str,
    that_flux_lm,
    that_which_gate: str,
    allow_any_comb: bool = False,
    plot_waveforms: bool = True,
    **plt_kw
):
    """
    Copies all the relevant parameters from the other flux_lm such
    that the beginning of the corrections match on both. By coping all the
    parameters of the waveform we ensure that the waveform will be generated in
    the exact way regarding timing at the individual sample points level.
    """

    opt_1 = (this_which_gate == "NE") and (that_which_gate == "SW")
    opt_2 = (this_which_gate == "NW") and (that_which_gate == "SE")
    if not (opt_1 or opt_2):
        # To avoid stupid mistakes
        msg = "Are you sure you wanted to match `{} {}` with `{} {}`?".format(
            this_flux_lm.name, this_which_gate, that_flux_lm.name, that_which_gate
        )
        log.error(msg)
        if not allow_any_comb:
            raise Exception("Aborting copying parameters!")

    that_gen_par_name = "cz_wf_generator_{}".format(that_which_gate)
    this_gen_par_name = "cz_wf_generator_{}".format(this_which_gate)

    that_wf_generator_name = that_flux_lm.get(that_gen_par_name)
    this_wf_generator_name = this_flux_lm.get(this_gen_par_name)
    if this_wf_generator_name != that_wf_generator_name:
        raise Exception("Both waveform generators must be the same!")

    this_f_is_for = "vcz_waveform"
    if that_wf_generator_name != this_f_is_for:
        raise Exception("This alignment work only with `" + this_f_is_for + "` waveform generator! "
            "Check `{}` and see also `align_vcz_q_phase_corr_with`".format(that_gen_par_name))

    par_names = {
        "vcz_time_middle_{}",
        "vcz_time_single_sq_{}",
        "vcz_time_before_q_ph_corr_{}",
        "vcz_use_amp_fine_{}"
    }
    # Copy all relevant parameters
    for par_name in par_names:
        par_val = that_flux_lm.get(par_name.format(that_which_gate))
        this_flux_lm.set(par_name.format(this_which_gate), par_val)

    # It is assumed `this_flux_lm` is the low freq. qubit
    this_flux_lm.set("vcz_amp_sq_{}".format(this_which_gate), 0)

    for flux_lm in [this_flux_lm, that_flux_lm]:
        flux_lm.generate_standard_waveforms()

    if plot_waveforms:
        this_flux_lm.plot_cz_waveforms(
            [this_flux_lm.name.split("_")[-1], that_flux_lm.name.split("_")[-1]],
            [this_which_gate, that_which_gate],
            **plt_kw
        )

# [2020-06-23] Commented out, needs fixing

# def get_vcz_min_time(flux_lm, which_gate):
#     time_ramp_middle = flux_lm.get("czv_time_ramp_middle_{}".format(which_gate))
#     time_ramp_outside = flux_lm.get("czv_time_ramp_outside_{}".format(which_gate))
#     speed_limit = flux_lm.get("czv_speed_limit_{}".format(which_gate))
#     min_time = (time_ramp_middle +
#         2 * time_ramp_outside + speed_limit)

#     return min_time


def vcz_waveform(
    fluxlutman,
    which_gate: str = None,
    sim_ctrl_cz=None,
    return_dict=False
):

    amp_at_sweetspot = 0.0
    if which_gate is None and sim_ctrl_cz is not None:
        which_gate = sim_ctrl_cz.which_gate()

    amp_at_int_11_02 = fluxlutman.get("vcz_amp_dac_at_11_02_{}".format(which_gate))

    sampling_rate = fluxlutman.sampling_rate()

    time_sqr = fluxlutman.get("vcz_time_single_sq_{}".format(which_gate))
    time_sqr = time_sqr * sampling_rate  # avoid numerical issues
    time_before_q_ph_corr = fluxlutman.get("vcz_time_before_q_ph_corr_{}".format(which_gate))
    time_before_q_ph_corr = time_before_q_ph_corr * sampling_rate  # avoid numerical issues
    time_middle = fluxlutman.get("vcz_time_middle_{}".format(which_gate))
    time_middle = time_middle * sampling_rate  # avoid numerical issues
    time_q_ph_corr = fluxlutman.get("vcz_time_q_ph_corr_{}".format(which_gate))
    time_q_ph_corr = time_q_ph_corr * sampling_rate  # avoid numerical issues

    dt = 1

    # Normalized to the amplitude at the CZ interaction point
    norm_amp_sq = fluxlutman.get("vcz_amp_sq_{}".format(which_gate))
    norm_amp_fine = fluxlutman.get("vcz_amp_fine_{}".format(which_gate))
    amp_q_ph_corr = fluxlutman.get("vcz_amp_q_ph_corr_{}".format(which_gate))

    correct_q_phase = fluxlutman.get("vcz_correct_q_phase_{}".format(which_gate))
    # In case we might want to play only with the pulse length and/or the
    # time in the middle
    use_amp_fine = fluxlutman.get("vcz_use_amp_fine_{}".format(which_gate))

    # This is to avoid numerical issues when the user would run sweeps with
    # e.g. `time_at_swtspt = np.arange(0/2.4e9, 10/ 2.4e9, 2/2.4e9)`
    # instead of `time_at_swtspt = np.arange(0, 42, 2) / 2.4e9` and get
    # bad results for specific combinations of parameters
    time_middle = np.round(time_middle / dt) * dt
    time_sqr = np.round(time_sqr / dt) * dt
    half_time_q_ph_corr = np.round(time_q_ph_corr / 2 / dt) * dt

    if use_amp_fine:
        # such that this amp is in the range [0, 1]
        slope_amp = np.array([norm_amp_fine * norm_amp_sq])
    else:
        slope_amp = np.array([])

    sq_amps = np.full(int(time_sqr / dt), norm_amp_sq)
    amps_middle = np.full(int(time_middle / dt), amp_at_sweetspot)
    buffer_before_corr = np.full(int(time_before_q_ph_corr / dt), amp_at_sweetspot)
    pos_q_ph_corr = np.full(int(half_time_q_ph_corr / dt), amp_q_ph_corr)

    half_NZ_amps = np.concatenate((sq_amps, slope_amp))

    amp = np.concatenate((
        [amp_at_sweetspot],
        half_NZ_amps,
        amps_middle,
        -half_NZ_amps[::-1],
        [amp_at_sweetspot])
    )

    if correct_q_phase:
        amps_corr = np.concatenate((
            buffer_before_corr,
            pos_q_ph_corr,
            -pos_q_ph_corr))
        if len(amps_corr):
            amp = np.concatenate((
                amp,
                amps_corr,
                [amp_at_sweetspot]))

    amp = amp_at_int_11_02 * amp

    tlist = np.cumsum(np.full(len(amp) - 1, dt))
    tlist = np.concatenate(([0.0], tlist))  # Set first point to have t=0

    # Extra processing in case we are generating waveform for simulations
    if sim_ctrl_cz is not None:
        dt_num = np.size(tlist) - 1
        dt_num_interp = dt_num * sim_ctrl_cz.simstep_div() + 1

        time_interp = np.linspace(tlist[0], tlist[-1], dt_num_interp)
        amp_interp = np.interp(time_interp, tlist, amp)

        if sim_ctrl_cz.optimize_const_amp():
            # For simulations we skip simulating every single point if they have
            # same amplitude (eigen space does not change)
            keep = (amp_interp[:-2] == amp_interp[1:-1]) * (
                amp_interp[2:] == amp_interp[1:-1]
            )
            keep = np.concatenate(([False], keep, [False]))
            keep = np.logical_not(keep)
            amp_interp = amp_interp[keep]
            time_interp = time_interp[keep]

        intervals = time_interp[1:] - time_interp[:-1]
        intervals_list = np.concatenate((intervals, [np.min(intervals)]))

        return {
            "time": time_interp / sampling_rate,
            "amp": amp_interp,
            "intervals_list": intervals_list / sampling_rate
        }

    if return_dict:
        return {"time": tlist / sampling_rate, "amp": amp}

    return amp

# ######################################################################
# Auxiliary tools
# ######################################################################


def to_int_if_close(value, abs_tol=1e-12, **kw):
    is_close = math.isclose(int(value), value, abs_tol=abs_tol, **kw)
    return int(value) if is_close else value
