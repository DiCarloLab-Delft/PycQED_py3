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
    ensure_start_at_zero=True,
    ensure_end_at_zero=True,
    output_q_phase_corr=True
):
    # NB: the ramps are extra time, they are NOT substracted from sq_length!

    amp_at_sweetspot = 0.0

    amp_at_int_11_02 = fluxlutman.get("czv_dac_amp_at_11_02_{}".format(which_gate))

    sampling_rate = fluxlutman.sampling_rate()

    # New parameters specific to this parameterization
    time_ramp_middle = fluxlutman.get("czv_time_ramp_middle_{}".format(which_gate))
    time_ramp_outside = fluxlutman.get("czv_time_ramp_outside_{}".format(which_gate))
    time_sum_sqrs = fluxlutman.get("czv_time_sum_sqrs_{}".format(which_gate))
    # total_time = fluxlutman.get("czv_total_time_{}".format(which_gate))
    time_at_sweetspot = fluxlutman.get("czv_time_at_sweetspot_{}".format(which_gate))
    invert_polarity = fluxlutman.get("czv_invert_polarity_{}".format(which_gate))
    # Normalized to the amplitude at the CZ interaction point
    norm_sq_amp = fluxlutman.get("czv_sq_amp_{}".format(which_gate))
    time_q_ph_corr = fluxlutman.get("czv_time_q_ph_corr_{}".format(which_gate))
    amp_q_ph_corr = fluxlutman.get("czv_amp_q_ph_corr_{}".format(which_gate))
    q_ph_corr_only = fluxlutman.get("czv_q_ph_corr_only_{}".format(which_gate))

    dt = 1 / sampling_rate

    half_time_ramp_middle = time_ramp_middle / 2.0
    half_time_sq = time_sum_sqrs / 2.0
    half_time_q_ph_corr = time_q_ph_corr / 2.0
    # half_time_at_swtspt = (
    #     total_time - time_ramp_middle - 2 * time_ramp_outside - time_sum_sqrs
    # ) / 2.0
    half_time_at_swtspt = time_at_sweetspot / 2.0

    # if half_time_at_swtspt < 0:
    #     raise ValueError(
    #         "Total time is not enough to accomodate for speed "
    #         "limit and pulse ramps!"
    #     )

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
        lambda x: (x - half_time_at_swtspt) * norm_sq_amp / half_time_ramp_middle,
        lambda x: norm_sq_amp,
        lambda x: -(x - t3) * norm_sq_amp / time_ramp_outside + norm_sq_amp,
    ]

    half_NZ_amps = np.piecewise(time, conditions, funcs)

    correct_q_phase = fluxlutman.get("czv_correct_q_phase_{}".format(which_gate))
    incl_q_phase_in_cz = fluxlutman.get("czv_incl_q_phase_in_cz_{}".format(which_gate))
    if correct_q_phase:
        if incl_q_phase_in_cz:
            # Insert extra square part to correct single qubit phase
            if amp_q_ph_corr < norm_sq_amp:
                # When the correction amplitude is smaller then the main
                # square amplitude we insert the correction at the right
                # saple point such that the pulse amplitude does not decrease
                # ever before the main square part
                insert_idx = np.where(half_NZ_amps >= amp_q_ph_corr)[0][-1] + 1
            else:
                # If the amplitude is higher than the main square amplitude
                # we insert the correction before the main square pulse after
                # the ramp up
                # The goal of using this feature would be to require less
                # sample points to correct for single qubit phase
                insert_idx = np.where(half_NZ_amps == norm_sq_amp)[0][-1] + 1
            amps_q_phase_correction = np.full(
                int(half_time_q_ph_corr / dt), amp_q_ph_corr
            )
            half_NZ_amps = np.insert(half_NZ_amps, insert_idx, amps_q_phase_correction)
        else:
            amps_q_phase_correction = np.full(
                int(half_time_q_ph_corr / dt), amp_q_ph_corr
            )
            # Ensure we end at zero
            amps_q_phase_correction = np.concatenate(
                (amps_q_phase_correction, -amps_q_phase_correction))

    amp = np.concatenate((np.flip(half_NZ_amps, 0), -half_NZ_amps[1:]))

    len_base_wf = len(amp)

    if correct_q_phase and not incl_q_phase_in_cz and amp[-1] != amp_at_sweetspot:
        # For this case we always add an extra pnt at zero so that the NZ
        # effect preserved before starting the correction pulse
        # This is also ncessary to relibly determine the main pulse length
        # when calling with `output_q_phase_corr=False`
        amp = np.concatenate((amp, [amp_at_sweetspot]))

    if correct_q_phase and output_q_phase_corr:
        amp = np.concatenate((amp, amps_q_phase_correction))

    cz_start_idx = 0
    # Extra points for starting and finishing at the sweetspot
    if ensure_start_at_zero and amp[0] != 0.0:
        cz_start_idx = 1
        amp = np.concatenate(([amp_at_sweetspot], amp))
    if ensure_end_at_zero and amp[-1] != 0.0:
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

        return {
            "time": time_interp,
            "amp": amp_interp,
            "intervals_list": intervals_list,
        }

    if return_dict:
        return {"time": tlist, "amp": amp}

    return amp
