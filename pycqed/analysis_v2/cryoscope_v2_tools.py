"""
Created: 2020-07-15
Author: Victor Negirneac
"""

from collections.abc import Iterable
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.hdf5_data as hd5
import lmfit
import numpy as np
import logging

# Filter and optimization tools
import pycqed.measurement.kernel_functions_ZI as kzi
from scipy import signal
import cma

log = logging.getLogger(__name__)

# ######################################################################
# Analysis utilities
# ######################################################################


def rough_freq_to_amp(
    amp_pars, time_ns, freqs, plateau_time_start_ns=-25, plateau_time_end_ns=-5,
):
    time_ = time_ns[: len(freqs)]
    time_ref_start = time_[-1] if plateau_time_start_ns < 0 else 0
    time_ref_stop = time_[-1] if plateau_time_end_ns < 0 else 0
    where = np.where(
        (time_ > time_ref_start + plateau_time_start_ns)
        & (time_ < time_ref_stop + plateau_time_end_ns)
    )
    avg_f = np.average(freqs[where])

    dac_amp = amp_pars["dac_amp"]
    Vpp = amp_pars["vpp"]
    cfg_amp = amp_pars["cfg_amp"]

    amp = cfg_amp * Vpp / 2 * dac_amp
    # coarse approximation of the arc assuming centered arc and quadratic fit
    a = avg_f / (amp ** 2)
    amps = np.sqrt(freqs / a)
    amp_plateau = np.average(amps[where])
    amps_norm = amps / amp_plateau

    res = {
        "amps": amps,
        "step_response": amps_norm,
        "amp_plateau": amp_plateau,
        "frequency_plateau": avg_f,
    }

    return res


def moving_cos_fitting_window(
    x_data_ns,
    y_data,
    fit_window_pnts_nr: int = 6,
    init_guess: dict = {"phase": 0.0},
    fixed_params: dict = {},
    max_params: dict = {},
    min_params: dict = {},
):
    """
    NB: Intended to be used with input data in ns, this assumption is
    used to generate educated guesses for the fitting
    """
    model = lmfit.Model(fit_mods.CosFunc)

    if "offset" not in init_guess.keys():
        offset_guess = np.average(y_data)
        init_guess["offset"] = offset_guess

    if "amplitude" not in init_guess.keys():
        amplitude_guess = np.max(y_data[len(y_data) // 2 :]) - init_guess["offset"]
        init_guess["amplitude"] = amplitude_guess

    if "frequency" not in init_guess.keys():
        min_t = x_data_ns[0]
        max_t = x_data_ns[-1]
        total_t = min_t - max_t
        y_data_for_fft = y_data[
            np.where(
                (x_data_ns > min_t + 0.1 * total_t)
                & (x_data_ns < max_t - 0.1 * total_t)
            )[0]
        ]
        w = np.fft.fft(y_data_for_fft)[
            : len(y_data_for_fft) // 2
        ]  # ignore negative values
        f = np.fft.fftfreq(len(y_data_for_fft), x_data_ns[1] - x_data_ns[0])[: len(w)]
        w[0] = 0  # ignore DC component
        frequency_guess = f[np.argmax(np.abs(w))]
        init_guess["frequency"] = frequency_guess
        print("Frequency guess from FFT: {:.3g} GHz".format(frequency_guess))
        warn_thr = 0.7  # GHz
        if frequency_guess > warn_thr:
            log.warning(
                "\nHigh detuning above {} GHz detected. Cosine fitting may fail! "
                "Consider using lower detuning!".format(warn_thr)
            )

    if "phase" not in init_guess.keys():
        init_guess["phase"] = 0.0

    params = model.make_params(**init_guess)

    def fix_pars(params, i):
        # The large range is just to allow the phase to move continuously
        # between the adjacent fits even if it is not inside [-pi, pi]
        params["phase"].min = -100.0 * np.pi
        params["phase"].max = 100.0 * np.pi
        params["amplitude"].min = 0.1 * init_guess["amplitude"]
        params["amplitude"].max = 2.0 * init_guess["amplitude"]

        # Not expected to be used for > 0.8 GHz
        params["frequency"].min = 0.1
        params["frequency"].max = 0.8

        for par, val in fixed_params.items():
            # iterable case is for the amplitude
            params[par].value = val[i] if isinstance(val, Iterable) else val
            params[par].vary = False

        for par, val in max_params.items():
            params[par].max = val

        for par, val in min_params.items():
            params[par].min = val

    pnts_per_fit = fit_window_pnts_nr
    pnts_per_fit_idx = pnts_per_fit + 1

    max_num_fits = len(x_data_ns) - pnts_per_fit + 1
    middle_fits_num = max_num_fits // 2
    results = [None for i in range(max_num_fits)]
    # print(results)
    results_stderr = [None for i in range(max_num_fits)]

    # We iterate from the middle of the data to avoid fitting issue
    # This was verified to help!
    # There is an iteration from the middle to the end and another one
    # from the middle to the beginning
    for fit_ref, iterator in zip(
        [-1, +1],
        [range(middle_fits_num, max_num_fits), reversed(range(middle_fits_num))],
    ):
        for i in iterator:
            if i != middle_fits_num:
                # Take the adjacent fit as the initial guess for the next fit
                params = model.make_params(
                    amplitude=results[i + fit_ref][0],
                    frequency=results[i + fit_ref][1],
                    phase=results[i + fit_ref][2],
                    offset=results[i + fit_ref][3],
                )
            fix_pars(params, i)

            t_fit_data = x_data_ns[i : i + pnts_per_fit_idx]
            fit_data = y_data[i : i + pnts_per_fit_idx]
            res = model.fit(fit_data, t=t_fit_data, params=params)

            res_pars = res.params.valuesdict()
            results[i] = np.fromiter(res_pars.values(), dtype=np.float64)

            results_stderr[i] = np.fromiter(
                (param.stderr for par_name, param in res.params.items()),
                dtype=np.float64,
            )

    results = np.array(results).T
    results_stderr = np.array(results_stderr).T

    results = {key: values for key, values in zip(res_pars.keys(), results)}
    results_stderr = {
        key: values for key, values in zip(res_pars.keys(), results_stderr)
    }

    return {
        "results": results,
        "results_stderr": results_stderr,
    }


def cryoscope_v2_processing(
    time_ns: np.array,
    osc_data: np.array,
    pnts_per_fit_first_pass: int = 4,
    pnts_per_fit_second_pass: int = 3,
    init_guess_first_pass: dict = {},
    fixed_params_first_pass: dict = {},
    init_guess_second_pass: dict = {},
    max_params: dict = {},
    min_params: dict = {},
    vln: str = "",
    insert_ideal_projection: bool = True,
    osc_amp_envelop_poly_deg: int = 1,
):
    """
    TBW

    Provide time in ns to avoid numerical issues, data processing here is elaborated

    `pnts_per_fit_second_pass` shouldn't be smaller than 3, this is the limit
    to fit the cosine (technically 2 is the limit but but probably will not
    work very well)
    """

    assert time_ns[0] != 0.0, "Cryoscope time should not start at zero!"

    def add_ideal_projection_at_zero(time_ns, y_data, vln, offset, osc_amp):
        """
        Inserts and ideal point at t = 0 based on the type of projection
        """
        if vln:
            if "mcos" in vln:
                time_ns = np.insert(time_ns, 0, 0)
                y_data = np.insert(y_data, 0, offset - osc_amp)
            elif "cos" in vln:
                time_ns = np.insert(time_ns, 0, 0)
                y_data = np.insert(y_data, 0, offset + osc_amp)
            elif "sin" in vln or "msin" in vln:
                time_ns = np.insert(time_ns, 0, 0)
                y_data = np.insert(y_data, 0, offset)
            else:
                log.warning(
                    "Projection type not supported. Unexpected results may arise."
                )
            return time_ns, y_data
        else:
            log.warning("\nSkipping ideal projection!")
            return time_ns, y_data

    res_dict = moving_cos_fitting_window(
        x_data_ns=time_ns,
        y_data=osc_data,
        fit_window_pnts_nr=pnts_per_fit_first_pass,
        init_guess=init_guess_first_pass,
        fixed_params=fixed_params_first_pass,
        max_params=max_params,
        min_params=min_params,
    )

    results = res_dict["results"]

    amps_from_fit = results["amplitude"]
    x_for_fit = time_ns[: len(amps_from_fit)]
    # Here we are intentionally using poly of deg 1 to avoid the amplitude to be lower
    # in the beginning which should not be physical
    line_fit = np.polyfit(x_for_fit, amps_from_fit, osc_amp_envelop_poly_deg)
    poly1d = np.poly1d(line_fit)
    fixed_offset = np.average(results["offset"])

    if not len(init_guess_second_pass):
        init_guess_second_pass = {
            "offset": fixed_offset,
            # "frequency": np.average(results["frequency"]),
            "phase": 0.0,
        }

    if insert_ideal_projection:
        # This helps with the uncertainty of not knowing very well what is
        # the amplitude of the first point of the step response
        time_ns, osc_data = add_ideal_projection_at_zero(
            time_ns=time_ns,
            y_data=osc_data,
            vln=vln,
            osc_amp=poly1d(0.0),
            offset=fixed_offset,
        )

    res_dict = moving_cos_fitting_window(
        x_data_ns=time_ns,
        y_data=osc_data,
        fit_window_pnts_nr=pnts_per_fit_second_pass,
        init_guess=init_guess_second_pass,
        fixed_params={"offset": fixed_offset, "amplitude": poly1d(time_ns)},
        max_params=max_params,
        min_params=min_params,
    )

    res_dict["time_ns"] = time_ns
    res_dict["osc_data"] = osc_data

    return res_dict


def extract_amp_pars(
    qubit: str,
    timestamp: str,
    dac_amp_key: str = "Snapshot/instruments/flux_lm_{}/parameters/sq_amp",
    vpp_key: str = "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_range",
    cfg_amp_key: str = "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_amplitude",
):
    """
    Assumes centered flux arc and converts cryoscope oscillation frequency
    to amplitude
    """

    dac_amp_key = dac_amp_key.format(qubit)
    vpp_key = vpp_key.format(qubit)
    cfg_amp_key = cfg_amp_key.format(qubit)

    filepath = ma.a_tools.get_datafilepath_from_timestamp(timestamp)

    exctraction_spec = {
        "dac_amp": (dac_amp_key, "attr:value"),
        "vpp": (vpp_key, "attr:value"),
        "cfg_amp": (cfg_amp_key, "attr:value"),
    }

    extracted = hd5.extract_pars_from_datafile(filepath, param_spec=exctraction_spec)

    return extracted


# ######################################################################
# IIRs (exponential filters) utilities
# ######################################################################


def pred_corrected_sig(sig, taus, amps):
    """
    [2020-07-15 Victor] Not tested in a while, see old cryoscope notebooks
    """
    for i, (tau, amp) in enumerate(zip(taus, amps)):
        sig = kzi.exponential_decay_correction_hw_friendly(
            sig, tau, amp, sampling_rate=2.4e9
        )
    return sig


def predicted_waveform(time, tau0, amp0, tau1, amp1, tau2, amp2, tau3, amp3):
    """
    [2020-07-15 Victor] Not tested in a while, see old cryoscope notebooks
    """
    taus = [tau0, tau1, tau2, tau3]
    amps = [amp0, amp1, amp2, amp3]
    y_pred = pred_corrected_sig(a0, taus, amps)

    # Normalized
    y_pred /= np.mean(y_pred[-100:])
    # Smooth tail
    # y_pred[100:] = filtfilt(a=[1], b=1/20*np.ones(20), x=y_pred[100:])
    # y_pred[50:100] = filtfilt(a=[1], b=1/5*np.ones(5), x=y_pred[50:100])

    return y_pred


# ######################################################################
# FIRs utilities
# ######################################################################


def print_FIR_loading(qubit, model_num, FIR, real_time=False):
    print(
        (
            "lin_dist_kern_{}.filter_model_0{:1d}({{'params': {{'weights': np."
            + repr(FIR)
            + "}}, 'model': 'FIR', 'real-time': {} }})"
        ).format(qubit, model_num, real_time)
    )


def optimize_fir_software(
    y,
    baseline_start=100,
    baseline_stop=None,
    taps=72,
    max_taps=72,
    start_sample=0,
    stop_sample=None,
    cma_options={},
):
    step_response = np.concatenate((np.array([0]), y))
    baseline = np.mean(y[baseline_start:baseline_stop])
    x0 = [1] + (max_taps - 1) * [0]

    def objective_function_fir(x):
        y = step_response
        zeros = np.zeros(taps - max_taps)
        x = np.concatenate((x, zeros))
        yc = signal.lfilter(x, 1, y)
        return np.mean(np.abs(yc[1 + start_sample : stop_sample] - baseline)) / np.abs(
            baseline
        )

    return cma.fmin2(objective_function_fir, x0, 0.1, options=cma_options)


def optimize_fir_HDAWG(
    y,
    baseline_start=100,
    baseline_stop=None,
    start_sample=0,
    stop_sample=None,
    cma_options={},
    max_taps=40,
    hdawg_taps=40,
):
    step_response = np.concatenate((np.array([0]), y))
    baseline = np.mean(y[baseline_start:baseline_stop])
    x0 = [1] + (max_taps - 1) * [0]

    def objective_function_fir(x):
        y = step_response
        zeros = np.zeros(hdawg_taps - max_taps)
        x = np.concatenate((x, zeros))
        yc = signal.lfilter(convert_FIR_from_HDAWG(x), 1, y)
        return np.mean(np.abs(yc[1 + start_sample : stop_sample] - baseline)) / np.abs(
            baseline
        )

    return cma.fmin2(objective_function_fir, x0, 0.1, options=cma_options)


def convolve_FIRs(FIRs):
    concolved_FIR = FIRs[0]
    for FIR in FIRs[1:]:
        concolved_FIR = np.convolve(concolved_FIR, FIR)
    # We keep only the first coeff
    concolved_FIR = concolved_FIR[: len(FIRs[0])]
    return concolved_FIR


def convert_FIR_for_HDAWG(k_joint):
    """
    The HDAWG imposes that beyond the first 8 coeff.,
    the rest are paired together and have the same value.

    Here account for that and take the average of the pair
    """
    dim_k_hw = 8 + (len(k_joint) - 8) / 2
    k_joint_hw = np.zeros(int(dim_k_hw))
    k_joint_hw[:8] = k_joint[:8]
    k_joint_hw[8:] = (k_joint[8::2] + k_joint[9::2]) / 2  # average pairwise
    return k_joint_hw


def convert_FIR_from_HDAWG(hardware_fir):
    """
    does the opposite of `convert_FIR_for_HDAWG`
    mind that it wont recover the same you input to `convert_FIR_for_HDAWG`
    """
    out_fir = np.concatenate(
        (hardware_fir[:8], np.repeat(hardware_fir[8:], 2))  # duplicate entries
    )
    return out_fir
