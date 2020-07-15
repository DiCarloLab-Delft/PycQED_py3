"""
Created: 2020-07-15
Author: Victor Negirneac
This includes a dirty analysis to be called from a notebook
Should be moved into its own analysis and keep here only the tools
"""

import matplotlib.pyplot as plt
from collections.abc import Iterable
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import measurement_analysis as ma
import pycqed.analysis_v2.measurement_analysis as ma2
import pycqed.measurement.hdf5_data as hd5
import lmfit
import numpy as np
import logging

log = logging.getLogger(__name__)


def cryoscope_v2(
    qubit,
    timestamp,
    kw_processing={},
    kw_extract={
        "dac_amp_key": "Snapshot/instruments/flux_lm_{}/parameters/sq_amp",
        "vpp_key": "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_range",
        "cfg_amp_key": "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_amplitude",
    },
    kw_rough_freq_to_amp={},
):
    """
    Example:
        from pycqed.analysis_v2 import cryoscope_v2_tools as cv2
        import numpy as np

        reload(cv2)
        ts_trace = "20200628_034745"
        qubit = "D1"
        res = cv2.cryoscope_v2(
            qubit=qubit,
            timestamp=ts_trace,
            kw_processing={"plot": True}
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        time_ns = res["time_ns"]

        step_response = res["step_response"]
        ax.plot(time_ns[:len(step_response)], step_response, label="Step response")

        # If the first point in the step_response is somewhat significantly below 1.0
        # Using the right shifted step response might help for the FIRs
        # [2020-07-15 Victor] Not tested yet if the shifted response allows for better
        # FIR calibration
        step_response_right_shifted = res["step_response_right_shifted"]
        ax.plot(time_ns[:len(step_response_right_shifted)],
                step_response_right_shifted, label="Step response right-shifted")

        ax.hlines(np.array([.99, .999, 1.01, 1.001, .97, 1.03]) ,
                  xmin=np.min(time_ns), xmax=np.max(time_ns[:len(step_response)]),
                  linestyle=["-", "--", "-", "--", "-.", "-."])

        ax.set_title(ts_trace + ": Cryoscope " + qubit)
        set_xlabel(ax, "Pulse duration", "ns")
        set_ylabel(ax, "Normalized amplitude", "a.u.")
        ax.legend()

    """
    a_obj = ma2.Basic1DAnalysis(t_start=timestamp)
    rdd = a_obj.raw_data_dict

    results_list = []
    mvs = rdd["measured_values"][0]

    time_ns = rdd["xvals"][0] * 1e9

    # Confirm that first point was not measured starting from zero,
    # zero has no meaning
    start_idx = 0 if time_ns[0] != 0.0 else 1

    time_ns = time_ns[start_idx:]

    pnts_per_fit_second_pass = kw_processing.get(
        "pnts_per_fit_second_pass", 3)
    pnts_per_fit_first_pass = kw_processing.get(
        "pnts_per_fit_first_pass", 4)

    kw_processing.update({
        "pnts_per_fit_first_pass": pnts_per_fit_first_pass,
        "pnts_per_fit_second_pass": pnts_per_fit_second_pass,
    })

    for mv in mvs:
        res = cryoscope_v2_processing(
            time_ns=time_ns, osc_data=mv[start_idx:], **kw_processing
        )
        results_list.append(res)

    all_freq = np.array([res[0]["frequency"] for res in results_list]).T
    av_freq = np.average(all_freq, axis=1)

    kw_extract["qubit"] = qubit
    kw_extract["timestamp"] = timestamp
    amp_pars = extract_amp_pars(**kw_extract)
    print(amp_pars)

    conversion = rough_freq_to_amp(amp_pars, time_ns, av_freq, **kw_rough_freq_to_amp)

    res = {
        "results_list": results_list,
        "averaged_frequency": av_freq,
        "amp_pars": amp_pars,
        "time_ns": time_ns
    }

    # Here we correct for the averaging effect of the moving cosine-fitting
    # window, we attribute the obtained frequency to the middle point in the
    # fitting window, and interpolate linearly the missing points do to the
    # right shift, this step response can be more accurate in certain cases
    # TO DO: try to instead fit an exponential signal to the first few
    # data points and use it to interpolate the missing points,
    # might be more accurate for distortion corrections
    step_response = conversion["step_response"]
    extra_pnts = pnts_per_fit_second_pass // 2
    step_response = np.concatenate((
        # linear extrapolation, assuming the step response is rising up
        # linearly, we don't  return the data point at zero
        np.linspace(0, step_response[0], 2 + extra_pnts)[1:-1],
        step_response
    ))
    conversion["step_response_right_shifted"] = step_response

    res.update(conversion)

    return res


def rough_freq_to_amp(
    amp_pars,
    time_ns,
    freqs,
    plateau_time_start_ns=-25,
    plateau_time_end_ns=-5,
):
    time_ = time_ns[: len(freqs)]
    where = np.where(
        (time_ > time_[-1] + plateau_time_start_ns)
        & (time_ < time_[-1] + plateau_time_end_ns)
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
    x_data,
    y_data,
    fit_window_pnts_nr: int = 6,
    init_guess: dict = {"frequency": 0.4, "amplitude": 2.0, "phase": 0.0},
    fixed_params: dict = {},
    max_params: dict = {},
    min_params: dict = {},
):
    model = lmfit.Model(fit_mods.CosFunc)

    if "offset" not in init_guess.keys():
        offset_guess = np.average(y_data)
        init_guess["offset"] = offset_guess
        print(offset_guess)
    params = model.make_params(**init_guess)

    pnts_per_fit = fit_window_pnts_nr
    pnts_per_fit_idx = pnts_per_fit + 1
    results = []
    results_stderr = []

    def fix_pars(params, i):
        params["phase"].min = -180
        params["phase"].max = 180
        params["amplitude"].min = 0  # Ensures positive amp
        params["frequency"].min = 0
        for par, val in fixed_params.items():
            params[par].value = val[i] if isinstance(val, Iterable) else val
            params[par].vary = False
        for par, val in max_params.items():
            params[par].max = val
        for par, val in min_params.items():
            params[par].min = val

    for i in range(len(x_data) - pnts_per_fit + 1):
        if len(results):
            # Take the last fit as the initial guess for the next fit
            params = model.make_params(
                amplitude=results[i - 1][0],
                frequency=results[i - 1][1],
                phase=results[i - 1][2],
                offset=results[i - 1][3],
            )
        fix_pars(params, i)

        t_fit_data = x_data[i : i + pnts_per_fit_idx]
        fit_data = y_data[i : i + pnts_per_fit_idx]
        res = model.fit(fit_data, t=t_fit_data, params=params)

        res_pars = res.params.valuesdict()
        results.append(np.fromiter(res_pars.values(), dtype=np.float64))

        results_stderr.append(
            np.fromiter(
                (param.stderr for par_name, param in res.params.items()),
                dtype=np.float64,
            )
        )

    results = np.array(results).T
    results_stderr = np.array(results_stderr).T

    results = {key: values for key, values in zip(res_pars.keys(), results)}
    results_stderr = {
        key: values for key, values in zip(res_pars.keys(), results_stderr)
    }

    return results, results_stderr


def cryoscope_v2_processing(
    time_ns: np.array,
    osc_data: np.array,
    pnts_per_fit_first_pass: int = 4,
    pnts_per_fit_second_pass: int = 3,
    init_guess_first_pass: dict = {},
    fixed_params_first_pass: dict = {},
    init_guess_second_pass: dict = {},
    plot: bool = True,
):
    """
    TBW

    Provide time in ns to avoid numerical issues, data processing here is elaborated

    `pnts_per_fit_second_pass` shouldn't be smaller than 3, this is the limit
    to fit the cosine (technically 2 is the limit but but probably will not
    work very well)

    NB max_params and min_params might be needed to be added if there are problems with the fitting
    """

    assert time_ns[0] != 0.0, "Cryoscope time should not start at zero!"
    x_data = time_ns
    y_data = osc_data

    offset = np.average(y_data)

    if not len(init_guess_first_pass):
        init_guess_first_pass = {
            "offset": offset,
            "frequency": 0.5,
            "amplitude": 1.0,
            "phase": 0.0,
        }

    if not len(fixed_params_first_pass):
        # Fixing the offset should work well
        fixed_params_first_pass = {
            "offset": offset,
        }

    results, results_stderr = moving_cos_fitting_window(
        x_data=x_data,
        y_data=y_data,
        fit_window_pnts_nr=pnts_per_fit_first_pass,
        init_guess=init_guess_first_pass,
        fixed_params=fixed_params_first_pass,
    )

    amps_from_fit = results["amplitude"]
    x_for_fit = x_data[: len(amps_from_fit)]
    # Here we are intentionally using poly of deg 1 to avoid the amplitude to be lower
    # in the beginning which should not be physical
    line_fit = np.polyfit(x_for_fit, amps_from_fit, 1)
    fixed_amps = np.poly1d(line_fit)(x_data)

    if not len(init_guess_second_pass):
        init_guess_second_pass = {
            "offset": offset,
            "frequency": results["frequency"][0],
            "amplitude": fixed_amps[0],
            "phase": 0.0,
        }

    results, results_stderr = moving_cos_fitting_window(
        x_data=x_data,
        y_data=y_data,
        fit_window_pnts_nr=pnts_per_fit_second_pass,
        init_guess=init_guess_second_pass,
        fixed_params={"offset": offset, "amplitude": fixed_amps},
    )

    if plot:
        fig, axs = plt.subplots(len(results) + 2, 1, figsize=(20, 25))

        ax = axs[0]
        ax.plot(
            x_data[: len(amps_from_fit)], amps_from_fit, label="Osc. amp. first pass"
        )
        ax.plot(x_for_fit, np.poly1d(line_fit)(x_for_fit), label="Line fit osc. amp.")
        ax.set_xlabel("Osc. amp. (a.u.)")
        ax.legend()

        ax = axs[1]
        ax.plot(x_data, y_data, "o")

        for i in range(len(results[list(results.keys())[0]])):
            res_pars = {key: results[key][i] for key in results.keys()}
            time_sample = np.linspace(
                x_data[i], x_data[i + pnts_per_fit_second_pass - 1], 20
            )
            cos_fit_sample = fit_mods.CosFunc(t=time_sample, **res_pars)
            ax.set_xlabel("Pulse duration (ns)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.plot(time_sample, cos_fit_sample, "-")

        for ax, key in zip(axs[2:], results.keys()):
            ax.plot(time_ns[: len(results[key])], results[key], "-o")

            if key == "frequency":
                ax.set_ylabel("Frequency (GHz)")
            elif key == "amplitude":
                ax.set_ylabel("Amplitude (a.u.)")
            elif key == "offset":
                ax.set_ylabel("Offset (a.u.)")
            elif key == "phase":
                ax.set_ylabel("Phase (deg)")
            ax.set_xlabel("Pulse duration (ns)")

        return results, results_stderr, axs
    else:
        return results, results_stderr


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
