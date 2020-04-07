"""
    File:               waveforms_flux_dev.py
    Author:             Victor Negîrneac
    Purpose:            generate flux CZ gate waveforms
    Prerequisites:
    Usage:
    Bugs:
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


def victor_waveform(
    fluxlutman,
    which_gate: str,
    sim_ctrl_cz=None,
    return_dict=False,
    force_start_end_swtspt=True,
):
    # NB: the ramps are extra time, they are NOT substracted from sq_length!

    amp_at_sweetspot = 0.0
    amp_at_int_11_02 = fluxlutman.calc_eps_to_amp(
        0, state_A="11", state_B="02", which_gate=which_gate
    ) / ( fluxlutman.cfg_awg_channel_range() / 2 * fluxlutman.cfg_awg_channel_amplitude() )

    if fluxlutman.get("czv_fixed_amp_{}".format(which_gate)):
        amp_at_int_11_02 = 0.5

    sampling_rate = fluxlutman.sampling_rate()

    # New parameters specific to this parameterization
    time_ramp_middle = fluxlutman.get("czv_time_ramp_middle_{}".format(which_gate))
    time_ramp_outside = fluxlutman.get("czv_time_ramp_outside_{}".format(which_gate))
    speed_limit = fluxlutman.get("czv_speed_limit_{}".format(which_gate))
    total_time = fluxlutman.get("czv_total_time_{}".format(which_gate))
    invert_polarity = fluxlutman.get("czv_invert_polarity_{}".format(which_gate))
    norm_sq_amp_par = fluxlutman.get("czv_sq_amp_{}".format(which_gate))
    time_q_ph_corr = fluxlutman.get("czv_time_q_ph_corr_{}".format(which_gate))
    amp_q_ph_corr = fluxlutman.get("czv_amp_q_ph_corr_{}".format(which_gate))

    dt = 1 / sampling_rate

    half_time_ramp_middle = time_ramp_middle / 2.0
    half_time_sq = speed_limit / 2.0
    half_time_q_ph_corr = time_q_ph_corr / 2.0
    half_time_at_swtspt = (
        total_time - time_ramp_middle - 2 * time_ramp_outside - speed_limit
    ) / 2.0

    if half_time_at_swtspt < 0:
        raise ValueError(
            "Total time is not enough to accomodate for speed "
            "limit and pulse ramps!"
        )

    half_total_time = (
        half_time_at_swtspt + half_time_ramp_middle + half_time_sq + time_ramp_outside
    )

    time = np.arange(0.0, half_total_time, dt)

    t1 = half_time_at_swtspt
    t2 = t1 + half_time_ramp_middle
    t3 = t2 + half_time_sq

    conditions = [time <= t1, time > t1, time >= t2, time > t3]
    funcs = [
        lambda x: amp_at_sweetspot,
        lambda x: (x - half_time_at_swtspt) * norm_sq_amp_par / half_time_ramp_middle,
        lambda x: norm_sq_amp_par,
        lambda x: -(x - t3) * norm_sq_amp_par / time_ramp_outside + norm_sq_amp_par,
    ]

    half_NZ_amps = np.piecewise(time, conditions, funcs)

    if fluxlutman.get("czv_correct_q_phase_{}".format(which_gate)):
        # Insert extra square part to correct single qubit phase
        insert_idx = np.where(half_NZ_amps >= amp_q_ph_corr)[0][-1] + 1
        amps_q_phase_correction = np.full(int(half_time_q_ph_corr / dt), amp_q_ph_corr)
        half_NZ_amps = np.insert(half_NZ_amps, insert_idx, amps_q_phase_correction)

    amp = np.concatenate((np.flip(half_NZ_amps, 0), -half_NZ_amps[1:]))
    # Extra points for starting and finishing at the sweetspot
    if force_start_end_swtspt and amp[0] != 0.0:
        amp = np.concatenate(([amp_at_sweetspot], amp, [amp_at_sweetspot]))

    if invert_polarity:
        amp = -amp

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
            # For simulations we skip simulating every single pnt if they have
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

        return_dict = {
            "time": time_interp,
            "amp": amp_interp,
            "intervals_list": intervals_list,
        }

    if return_dict:
        return {"time": tlist, "amp": amp}

    return amp
