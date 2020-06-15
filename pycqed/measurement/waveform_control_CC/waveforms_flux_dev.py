"""
    File:               waveforms_flux_dev.py
    Author:             Victor Negîrneac
    Purpose:            generate flux VCZ gate waveforms
    Prerequisites:
    Usage:
    Bugs:
"""

import numpy as np
import math
import logging

log = logging.getLogger(__name__)


def vcz_dev_waveform(
    fluxlutman,
    which_gate: str,
    sim_ctrl_cz=None,
    return_dict=False,
    ensure_start_at_zero=True,
    ensure_end_at_zero=True,
    output_q_phase_corr=True
):
    # NB: the ramps are extra time, they are NOT subtracted from time_sum_sqrs!

    amp_at_sweetspot = 0.0

    if which_gate is None and sim_ctrl_cz is not None:
        which_gate = sim_ctrl_cz.which_gate()

    amp_at_int_11_02 = fluxlutman.get("czv_amp_dac_at_11_02_{}".format(which_gate))

    sampling_rate = fluxlutman.sampling_rate()

    time_ramp_middle = fluxlutman.get("czv_time_ramp_middle_{}".format(which_gate))
    time_ramp_middle = time_ramp_middle * sampling_rate  # avoid numerical issues
    time_ramp_outside = fluxlutman.get("czv_time_ramp_outside_{}".format(which_gate))
    time_ramp_outside = time_ramp_outside * sampling_rate  # avoid numerical issues
    time_sum_sqrs = fluxlutman.get("czv_time_sum_sqrs_{}".format(which_gate))
    time_sum_sqrs = time_sum_sqrs * sampling_rate  # avoid numerical issues
    time_before_q_ph_corr = fluxlutman.get("czv_time_before_q_ph_corr_{}".format(which_gate))
    time_before_q_ph_corr = time_before_q_ph_corr * sampling_rate  # avoid numerical issues
    time_at_sweetspot = fluxlutman.get("czv_time_at_sweetspot_{}".format(which_gate))
    time_at_sweetspot = time_at_sweetspot * sampling_rate  # avoid numerical issues
    time_q_ph_corr = fluxlutman.get("czv_time_q_ph_corr_{}".format(which_gate))
    time_q_ph_corr = time_q_ph_corr * sampling_rate  # avoid numerical issues
    time_step = fluxlutman.get("czv_time_step_{}".format(which_gate))
    time_step = time_step * sampling_rate  # avoid numerical issues

    dt = 1

    invert_polarity = fluxlutman.get("czv_invert_polarity_{}".format(which_gate))
    # Normalized to the amplitude at the CZ interaction point
    norm_amp_sq = fluxlutman.get("czv_amp_sq_{}".format(which_gate))
    amp_q_ph_corr = fluxlutman.get("czv_amp_q_ph_corr_{}".format(which_gate))
    amp_step = fluxlutman.get("czv_amp_step_{}".format(which_gate))
    q_ph_corr_only = fluxlutman.get("czv_q_ph_corr_only_{}".format(which_gate))
    mirror_sqrs = fluxlutman.get("czv_mirror_sqrs_{}".format(which_gate))
    flip_wf = fluxlutman.get("czv_flip_wf_{}".format(which_gate))

    if norm_amp_sq == 0.0:
        # This is somewhat special case and should not actually be used
        # but leads to some undesired behavior
        # The way to achieve the zero amplitude for the main pulse is to
        # to set `czv_q_ph_corr_only_{}` to True
        norm_amp_sq = 1.0  # arbitrary amplitude
        q_ph_corr_only = True

    # This is to avoid numerical issues when the user would run sweeps with
    # e.g. `time_at_swtspt = np.arange(0/2.4e9, 10/ 2.4e9, 2/2.4e9)`
    # instead of `time_at_swtspt = np.arange(0, 42, 2) / 2.4e9` and get
    # bad results for specific combinations of parameters
    time_at_sweetspot = to_int_if_close(time_at_sweetspot)
    half_time_ramp_middle = to_int_if_close(time_ramp_middle / 2)
    half_time_sq = to_int_if_close(time_sum_sqrs / 2)
    half_time_q_ph_corr = to_int_if_close(time_q_ph_corr / 2)
    half_time_at_swtspt = to_int_if_close(time_at_sweetspot / 2)

    if mirror_sqrs:
        # if we mirror than the time at sweet-spot in between the two squares
        # is shared between both halves of the pulse.
        # This is equivalent to playing the second pulse in reverse order
        # in time.
        time_at_sweetspot_sym = half_time_at_swtspt
    else:
        # if not mirroring, we want to play the same pulse twice
        # The time at sweet-spot in between the two squares is going to be
        # part only of the second pulse
        # When concatenating the two halves later in this function we remove
        # the zeros at the beginning from the first pulse
        time_at_sweetspot_sym = time_at_sweetspot

    sampling_duration = (
        time_at_sweetspot_sym + half_time_ramp_middle + half_time_sq + time_ramp_outside
    )

    time = np.arange(0, sampling_duration, dt)

    t1 = time_at_sweetspot_sym
    t2 = t1 + half_time_ramp_middle
    t3 = t2 + half_time_sq
    t4 = t3 + time_ramp_outside

    # The `=` in `>=` everywhere is intended to avoid edge cases of integer
    # multiples of the sampling rate
    conditions = [
        time <= t1,  # time at sweet-spot
        time >= t1,  # ramp middle
        time >= t2,  # square part
        time >= t3,  # ramp outside
        time >= t4]  # back to sweet spot
    # NB numerical issues may rise for specific combinations of time parameters
    # Avoid setting the time of the ramps below the sampling resolution
    funcs = [
        lambda x: amp_at_sweetspot,
        lambda x: (x - time_at_sweetspot_sym) * norm_amp_sq / half_time_ramp_middle,
        lambda x: norm_amp_sq,
        lambda x: -(x - t3) * norm_amp_sq / time_ramp_outside + norm_amp_sq,
        lambda x: amp_at_sweetspot
    ]

    half_NZ_amps = np.piecewise(time, conditions, funcs)

    correct_q_phase = fluxlutman.get("czv_correct_q_phase_{}".format(which_gate))
    incl_q_phase_in_cz = fluxlutman.get("czv_incl_q_phase_in_cz_{}".format(which_gate))
    if correct_q_phase:
        if incl_q_phase_in_cz:
            if not mirror_sqrs:
                raise NotImplementedError("Pulse must be symmetrical. "
                    "Double check the pulse parameters!")
            # Insert extra square part to correct single qubit phase
            if amp_q_ph_corr < norm_amp_sq:
                # When the correction amplitude is smaller then the main
                # square amplitude we insert the correction at the right
                # sample point such that the pulse amplitude does not decrease
                # ever before the main square part
                insert_idx = np.where(half_NZ_amps >= amp_q_ph_corr)[0][-1] + 1
            else:
                # If the amplitude is higher than the main square amplitude
                # we insert the correction before the main square pulse after
                # the ramp up
                # The goal of using this feature would be to require less
                # sample points to correct for single qubit phase (maybe)
                insert_idx = np.where(half_NZ_amps == norm_amp_sq)[0][-1] + 1
            amps_q_phase_correction = np.full(
                int(half_time_q_ph_corr / dt), amp_q_ph_corr
            )
            half_NZ_amps = np.insert(half_NZ_amps, insert_idx, amps_q_phase_correction)
        else:
            amps_q_phase_correction = np.full(
                int(half_time_q_ph_corr / dt), amp_q_ph_corr
            )
            amps_q_phase_correction = np.concatenate(
                (amps_q_phase_correction, -amps_q_phase_correction))

    # This is intended to replicate a new parameter explored in the CZ of
    # Quantum Inspire
    if time_step > 0.0:
        if not mirror_sqrs:
            log.error("Extra step in between the squares implemented only for "
            "symmetrical pulses. Double check the pulse parameters!")
        else:
            # Insert at the first point that is higher than the step amplitude
            # or just before the main square
            where = (half_NZ_amps > amp_step) | (half_NZ_amps == norm_amp_sq)
            insert_idx = np.where(where)[0][0]
            step_amps = np.full(int(time_step / dt), amp_step)
            half_NZ_amps = np.insert(half_NZ_amps, insert_idx, step_amps)

    if mirror_sqrs:
        # `[1:]` makes sure the two pulses share the point in the middle
        amp = np.concatenate((np.flip(half_NZ_amps, 0), - half_NZ_amps[1:]))
    else:
        # Remove zeros at the beginning
        first_sqr_amps = np.array(half_NZ_amps)
        keep = (first_sqr_amps == amp_at_sweetspot) * (time <= time_at_sweetspot_sym) ^ 1
        first_sqr_amps = first_sqr_amps[np.where(keep)[0]]
        # Remove last point if it is zero as this point will be shared with
        # the first zero of the second square
        first_sqr_amps = first_sqr_amps[:-1 if first_sqr_amps[-1] == 0 else None]

        amp = np.concatenate((first_sqr_amps, - half_NZ_amps))

    len_base_wf = len(amp)

    if flip_wf:
        # when flipping we preserve polarity as well
        amp = - np.flip(amp)

    if correct_q_phase and not incl_q_phase_in_cz:
        # For this case we always add an extra point at zero so that the NZ
        # effect is preserved before starting the correction pulse
        # This is also necessary to reliably determine the main pulse length
        # when calling with `output_q_phase_corr=False`
        if amp[-1] != amp_at_sweetspot:
            amp = np.concatenate((amp, [amp_at_sweetspot]))

        buffer_before_q_ph_corr = np.full(int(time_before_q_ph_corr / dt), amp_at_sweetspot)
        if len(buffer_before_q_ph_corr) > 0:
            amp = np.concatenate((amp, buffer_before_q_ph_corr))

        # Concatenate the actual correction
        if output_q_phase_corr:
            amp = np.concatenate((amp, amps_q_phase_correction))

    cz_start_idx = 0
    # Extra points for starting and finishing at the sweet-spot
    if ensure_start_at_zero and amp[0] != amp_at_sweetspot:
        cz_start_idx = 1
        amp = np.concatenate(([amp_at_sweetspot], amp))
    if ensure_end_at_zero and amp[-1] != amp_at_sweetspot:
        amp = np.concatenate((amp, [amp_at_sweetspot]))

    if invert_polarity:
        amp = -amp

    if q_ph_corr_only:
        # We set to zero the main pulse when the qubit is only single
        # qubit phase corrected
        # `cz_start_idx` is needed so that we don't mess the time
        # alignment between the corrections on both qubits
        amp[cz_start_idx:len_base_wf + cz_start_idx] = amp_at_sweetspot

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

        # import matplotlib.pyplot as plt
        # plt.plot(time_interp, amp_interp, ".-")
        # plt.show()

        return {
            "time": time_interp / sampling_rate,
            "amp": amp_interp,
            "intervals_list": intervals_list / sampling_rate
        }

    if return_dict:
        return {"time": tlist / sampling_rate, "amp": amp}

    return amp


# ######################################################################
# VCZ simplified
# ######################################################################

def vcz_simplified_waveform(
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
