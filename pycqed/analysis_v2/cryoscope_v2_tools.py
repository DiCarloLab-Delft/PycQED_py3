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

# Filter and optimization tools
import pycqed.measurement.kernel_functions_ZI as kzi
from scipy import signal
import cma

log = logging.getLogger(__name__)


# ######################################################################
# Main analysis tool
# ######################################################################

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
    savgol_window: int = 5,
    savgol_polyorder: int = 1,
):
    """
    Args:
        savgol_window: used to generated a filtered step response that is useful
        to use when the correction to apply are already very small on the order
        of 1%, at that point the step response get very sensitive to new FIRs

    Example:

        # ##############################################################
        # Analysis tool
        # ##############################################################

        from pycqed.analysis_v2 import cryoscope_v2_tools as cv2
        import numpy as np
        from scipy import signal

        reload(cv2)
        ts_trace = "20200628_034745"
        qubit = "D1"
        res = cv2.cryoscope_v2(
            qubit=qubit,
            timestamp=ts_trace,
            kw_processing={"plot": True}
        )

        # ##############################################################
        # Plot analysed step response
        # ##############################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        time_ns = res["time_ns"]

        step_response = res["step_response"]
        ax.plot(time_ns[:len(step_response)], step_response, label="Step response")

        # If the first point in the step_response is somewhat significantly far from 1.0
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

        # ##############################################################
        # IIR optimization
        # ##############################################################

        # TODO: add here for reference next time!!!

        # ##############################################################
        # FIR optimization
        # ##############################################################

        # Limit the number of point up to which this FIR should correct
        # This helps to avoid fitting noise and make the convergence faster
        # and targeted to what we want to correct
        # maximum 72 taps are available for HDAWG real-time FIR (~ 30 ns)
        max_taps = 72

        opt_fir, _ = cv2.optimize_fir_software(
            step_response,
            baseline_start=np.where(time_ns > 10)[0].min(),
            max_taps=max_taps,
            cma_options={
                "verb_disp":10000,  # Avoid too much output
                #"ftarget": 1e-3, "tolfun": 1e-15, "tolfunhist": 1e-15, "tolx": 1e-15
            }
        )

        # ##############################################################
        # FIR optimization plotting
        # ##############################################################

        ac_soft_FIR = signal.lfilter(opt_fir, 1, step_response)

        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        ax.plot(time_ns[:len(step_response)], step_response, "-o")
        ax.plot(time_ns[:len(step_response)], ac_soft_FIR, "-o")

        ax.hlines(np.array([.99, .999, 1.01, 1.001]),
                  xmin=np.min(time_ns), xmax=np.max(time_ns[:len(step_response)]),
                  linestyle=["-", "--", "-", "--"])

        # ##############################################################
        # Generate loading code (first iteration only)
        # ##############################################################

        # Run this cell for the first FIR only
        # Then copy paste below in order to keep track of the FIRs
        # You may want to go back a previous one
        filter_model_number = 4  # CHANGE ME IF NEEDED!!!

        cv2.print_FIR_loading(
            qubit,
            filter_model_number,
            cv2.convert_FIR_for_HDAWG(opt_fir),
            real_time=True)

        # Output sample:
        # lin_dist_kern_D1.filter_model_04({'params': {'weights': np.array([ 1.13092421e+00, -6.82709369e-02, -4.64421034e-02, -2.58260195e-02,
        #        -1.04921358e-02, -9.73883537e-03, -2.42308728e-03,  5.35076244e-03,
        #         3.77617499e-03,  5.28141742e-03, -6.33810801e-03,  2.69420579e-03,
        #         9.06057712e-03,  7.32841146e-03,  1.20281705e-03, -2.35979362e-03,
        #        -4.87644425e-03,  1.49692530e-03, -9.34622902e-04, -2.26087315e-04,
        #        -1.15781407e-02,  1.11572007e-03,  4.48942912e-03, -4.85723912e-03,
        #         5.10716383e-03,  2.29466092e-03,  2.88845548e-03,  1.74550550e-03,
        #        -3.71967987e-03, -3.46337041e-03,  8.76836280e-03, -7.60823516e-03,
        #         7.90059429e-03, -1.11806920e-02,  8.48894913e-03, -6.42957441e-03,
        #         3.25895281e-03, -1.24377996e-03, -8.87517579e-04,  2.20711760e-03])}, 'model': 'FIR', 'real-time': True })

        # Use the above to run on the setup if this is the first FIR

        # ##############################################################
        # Convolve new FIR iteration with the last one
        # ##############################################################

        # Keep adding fir_{i} here and convolving with the last one

        # fir_0 should be from the first optimization or the current real-time FIR in use on the setup
        fir_0 = np.array([ 1.05614572e+00, -2.53850198e-03, -2.52682533e-03, -2.51521371e-03,
               -2.50372099e-03, -2.49226498e-03, -2.48089918e-03, -2.46960924e-03,
               -2.45266068e-03, -2.43085526e-03, -2.40884910e-03, -3.96701006e-03,
                2.07353990e-03, -2.00725135e-03,  1.69462462e-03,  4.57420262e-03,
                1.29168122e-03,  1.41930085e-03,  1.19988012e-03, -2.64650972e-03,
               -1.92008328e-03, -2.09618589e-03, -4.35853136e-03, -3.46286777e-03,
               -2.70556691e-03, -1.96788087e-03, -8.97396693e-04, -7.83636242e-04,
                1.89748899e-04,  5.96137205e-04,  4.40804891e-04,  1.22959418e-03,
                6.27207165e-05,  1.78369142e-04,  5.88531033e-04,  3.75452325e-04,
               -1.52053376e-04,  7.29338599e-04, -9.92730555e-05, -7.75952068e-04])

        # fir_1 =

        last_FIR = cv2.convert_FIR_from_HDAWG(fir_0)  # UPDATE last FIR FOR EACH ITERATION!
        c1 = cv2.convolve_FIRs([last_FIR, opt_fir])

        cv2.print_FIR_loading(
            qubit,
            filter_model_number,
            cv2.convert_FIR_for_HDAWG(c1),
            real_time=True)

        # Output sample:
        # lin_dist_kern_D1.filter_model_04({'params': {'weights': np.array([ 1.19442077e+00, -7.49749111e-02, -5.17339708e-02, -2.98301540e-02,
        #        -1.35581165e-02, -1.27245568e-02, -4.96222306e-03,  3.26517377e-03,
        #         1.60572820e-03,  3.18970858e-03, -9.07183243e-03, -1.22431441e-03,
        #         1.22491343e-02,  5.37109853e-03,  3.15302396e-03,  2.27680655e-03,
        #        -4.28813944e-03,  2.85768654e-03,  4.20547339e-05, -3.31392990e-03,
        #        -1.40901704e-02, -7.79037165e-04,  3.36395118e-04, -8.28875071e-03,
        #         2.94764301e-03,  7.55326282e-04,  2.33059861e-03,  1.02747385e-03,
        #        -3.77002742e-03, -3.18534929e-03,  9.54446304e-03, -7.03080156e-03,
        #         8.12891025e-03, -1.17499888e-02,  9.59861367e-03, -6.31999213e-03,
        #         3.35793857e-03, -4.27721721e-04, -9.07270542e-04,  1.55317845e-03])}, 'model': 'FIR', 'real-time': True })

        # The above output would be used to set fir_1
        """
    a_obj = ma2.Basic1DAnalysis(t_start=timestamp)
    rdd = a_obj.raw_data_dict

    results_list = []
    mvs = rdd["measured_values"][0]
    vlns = rdd["value_names"][0]

    time_ns = rdd["xvals"][0] * 1e9

    # Confirm that first point was not measured starting from zero,
    # zero has no meaning
    start_idx = 0 if time_ns[0] != 0.0 else 1

    time_ns = time_ns[start_idx:]

    pnts_per_fit_second_pass = kw_processing.get("pnts_per_fit_second_pass", 3)
    pnts_per_fit_first_pass = kw_processing.get("pnts_per_fit_first_pass", 4)

    kw_processing.update(
        {
            "pnts_per_fit_first_pass": pnts_per_fit_first_pass,
            "pnts_per_fit_second_pass": pnts_per_fit_second_pass,
        }
    )

    for mv in mvs:
        res = cryoscope_v2_processing(
            time_ns=time_ns, osc_data=mv[start_idx:], **kw_processing
        )
        results_list.append(res)

    all_freq = np.array([res[0]["frequency"] for res in results_list])
    all_freq_T = all_freq.T
    av_freq = np.average(all_freq_T, axis=1)

    all_freq_filtered = [
        signal.savgol_filter(sig, savgol_window, savgol_polyorder, 0)
        for sig in [*all_freq, av_freq]
    ]
    all_names_filtered = [name + "_filtered" for name in vlns]

    av_freq_filtered = signal.savgol_filter(
        av_freq,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
        deriv=0)

    kw_extract["qubit"] = qubit
    kw_extract["timestamp"] = timestamp
    amp_pars = extract_amp_pars(**kw_extract)

    res = {
        "results_list": results_list,
        "averaged_frequency": av_freq,
        "amp_pars": amp_pars,
        "time_ns": time_ns,
    }

    for frequencies, name in zip(
        # Make available in the results all combinations
        [*all_freq, av_freq, *all_freq_filtered, av_freq_filtered],
        [*vlns, "average", *all_names_filtered, "average_filtered"]
    ):
        conversion = rough_freq_to_amp(amp_pars, time_ns, frequencies, **kw_rough_freq_to_amp)

        # Here we correct for the averaging effect of the moving cosine-fitting
        # window, we attribute the obtained frequency to the middle point in the
        # fitting window, and interpolate linearly the missing points do to the
        # right shift, this step response can be more accurate in certain cases
        # TO DO: try to instead fit an exponential signal to the first few
        # data points and use it to interpolate the missing points,
        # might be more accurate for distortion corrections
        step_response = conversion["step_response"]
        extra_pnts = pnts_per_fit_second_pass // 2

        # # Fit only first 15 ns
        step_response_fit = np.array(step_response)
        time_ns_fit = time_ns[extra_pnts:][:len(step_response_fit)]
        where = np.where(time_ns_fit < 15)[0]
        step_response_fit = step_response_fit[where]
        time_ns_fit = time_ns_fit[where]

        def exp_rise(t, tau):
            return 1 - np.exp(- t / tau)

        model = lmfit.Model(exp_rise)
        params = model.make_params()
        params["tau"].value = 1
        params["tau"].min = 0
        params["tau"].max = 15

        fit_res = model.fit(step_response_fit, t=time_ns_fit, params=params)
        params = {key: fit_res.params[key] for key in fit_res.params.keys()}

        if step_response[0] < 0.97:
            # Only extrapolate if the first point is significantly below
            corrected_pnts = exp_rise(time_ns[:extra_pnts], **params)
        else:
            corrected_pnts = [step_response[0]] * extra_pnts
            # For some cases maybe works better to just assume the first
            # point is calibrated, didn't test enough...
            # corrected_pnts = [1.0] * extra_pnts

        step_response = np.concatenate(
            (
                # Extrapolate the missing points assuming exponential rise
                # Seems a fair assumption and much better than a linear
                # extrapolation
                corrected_pnts,
                step_response,
            )
        )
        conversion.update(params)
        conversion["step_response_processed_" + name] = step_response

        # Renaming to be able to return the step responses from all measured
        # channels along side with the average
        step_response = conversion.pop("step_response")
        conversion["step_response_raw_" + name] = step_response
        res.update(conversion)

    return res


# ######################################################################
# Analysis utilities
# ######################################################################

def rough_freq_to_amp(
    amp_pars, time_ns, freqs, plateau_time_start_ns=-25, plateau_time_end_ns=-5,
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
        amplitude_guess = np.max(y_data) - init_guess["offset"]
        init_guess["amplitude"] = amplitude_guess

    if "frequency" not in init_guess.keys():
        w = np.fft.fft(y_data)[:len(y_data) // 2]  # ignore negative values
        f = np.fft.fftfreq(len(y_data), x_data_ns[1] - x_data_ns[0])[:len(w)]
        w[0] = 0  # ignore DC component
        frequency_guess = f[np.argmax(np.abs(w))]
        init_guess["frequency"] = frequency_guess

    params = model.make_params(**init_guess)

    def fix_pars(params, i):
        params["phase"].min = -180
        params["phase"].max = 180
        params["amplitude"].min = 0.1 * init_guess["amplitude"]
        params["amplitude"].max = 2.0 * init_guess["amplitude"]

        params["frequency"].min = 0.0
        params["frequency"].max = 0.6  # Not expected to be used for > 0.5 GHz

        for par, val in fixed_params.items():
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
        [-1, + 1],
        [
            range(middle_fits_num, max_num_fits),
            reversed(range(middle_fits_num))
        ]
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
    x_data_ns = time_ns
    y_data = osc_data

    results, results_stderr = moving_cos_fitting_window(
        x_data_ns=x_data_ns,
        y_data=y_data,
        fit_window_pnts_nr=pnts_per_fit_first_pass,
        init_guess=init_guess_first_pass,
        fixed_params=fixed_params_first_pass,
    )

    amps_from_fit = results["amplitude"]
    x_for_fit = x_data_ns[: len(amps_from_fit)]
    # Here we are intentionally using poly of deg 1 to avoid the amplitude to be lower
    # in the beginning which should not be physical
    line_fit = np.polyfit(x_for_fit, amps_from_fit, 1)
    fixed_amps = np.poly1d(line_fit)(x_data_ns)
    fixed_offset = np.average(results["offset"])

    if not len(init_guess_second_pass):
        init_guess_second_pass = {
            "offset": fixed_offset,
            # "frequency": np.average(results["frequency"]),
            "phase": 0.0,
        }

    results, results_stderr = moving_cos_fitting_window(
        x_data_ns=x_data_ns,
        y_data=y_data,
        fit_window_pnts_nr=pnts_per_fit_second_pass,
        init_guess=init_guess_second_pass,
        fixed_params={"offset": fixed_offset, "amplitude": fixed_amps},
    )

    if plot:
        fig, axs = plt.subplots(len(results) + 2, 1, figsize=(20, 25))

        ax = axs[0]
        ax.plot(
            x_data_ns[: len(amps_from_fit)], amps_from_fit, label="Osc. amp. first pass"
        )
        ax.plot(x_for_fit, np.poly1d(line_fit)(x_for_fit), label="Line fit osc. amp.")
        ax.set_xlabel("Osc. amp. (a.u.)")
        ax.legend()

        ax = axs[1]
        ax.plot(x_data_ns, y_data, "o")

        for i in range(len(results[list(results.keys())[0]])):
            res_pars = {key: results[key][i] for key in results.keys()}
            time_sample = np.linspace(
                x_data_ns[i], x_data_ns[i + pnts_per_fit_second_pass - 1], 20
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
