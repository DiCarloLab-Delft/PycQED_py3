"""
Created: 2020-07-15
Author: Victor Negirneac
"""

import matplotlib.pyplot as plt
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.analysis_v2.cryoscope_v2_tools as cv2_tools
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.hdf5_data as hd5
from collections import OrderedDict
from uncertainties import ufloat
from scipy import signal
import os
import lmfit
import numpy as np
import logging
import json
import multiprocessing as mp
import cma

log = logging.getLogger(__name__)


class Cryoscope_v2_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        qubit,
        kw_extract={
            "dac_amp_key": "Snapshot/instruments/flux_lm_{}/parameters/sq_amp",
            "vpp_key": "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_range",
            "cfg_amp_key": "Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_amplitude",
        },
        kw_rough_freq_to_amp={
            # Negative values are w.r.t the maximum time
            "plateau_time_start_ns": -25,
            "plateau_time_end_ns": -5
        },
        kw_exp_fit={
            "tau_min": 0,
            "tau_max": 15,
            "time_ns_fit_max": 15,
            "threshold_apply": 0.97,
        },
        kw_processing={
            "pnts_per_fit_second_pass": 3,
            "pnts_per_fit_first_pass": 4,
            # Controls the polynomial fitted to the oscillation envelope
            "osc_amp_envelop_poly_deg": 1,
            # Setting sensible limits for the oscillation frequency helps
            # avoiding cosine fitting failures
            "min_params": {"frequency": 0.1},
            "max_params": {"frequency": 0.6},
        },
        # Allows to exclude certain projections from the averaging
        # Handy when the fits failed for one or more projections
        average_exclusion_val_names: list = [],  # e.g. [" cos", "msin"]
        savgol_window: int = 3,  # 3 or 5 should work best
        # Might be useful to put to 0 after some iterations,
        # In order to use savgol_polyorder=0, step response should be almost flat
        # Otherwise first points get affected
        savgol_polyorder: int = 1,
        insert_ideal_projection: bool = True,
        t_start: str = None,
        t_stop: str = None,
        label="",
        options_dict: dict = None,
        auto=True,
        close_figs=True,
        **kwargs
    ):
        """
        ================================================================
        A second version of cryoscope analysis, best suited for FIRs
        calibration.

        [2020-07-22] Not tested for IIR calibration, should still be usable
        In that case you may want to apply a more aggressive savgol filter

        IMPORTANT: how to choose the detuning (i.e. amplitude of flux pulse)?
        Answer: target a detuning on the order of ~450-700 MHz and mind that
        very high detuning might difficult the fitting involved in this
        analysis, but low amplitude has low SNR

        Does not require the flux arc, assumes quadratic dependence on
        frequency and a plateau of stable frequency to extract an average
        The plateau is controlled by `plateau_time_start_ns` and
        `plateau_time_stop_ns`.

        Uses a moving cos-fitting-window to extract instantaneous oscillation
        frequency

        Requirements:
            - Single qubit gates very well calibrated for this qubit to avoid
            systematical errors in the different projections of the Bloch
            vector
            - Well calibrated RO
            - Qubit parked ~ at sweet-spot

        Generates 4 variations of the step response, use the one that look
        more suitable to the situation (initial FIR vs last FIR iteration)
        4 variations:
            - 2 w/out filtering:
                - No processing (included extra ideal projection if
                insert_ideal_projection = True)
                - Processing replaces the first point with the value from an
                exponential fit
            - 2 w/ filtered:
                A Savitzky–Golay filter is applied controlled by `savgol_window`
                and `savgol_polyorder`. NB: savgol_polyorder=0 is more
                aggressive but at expense of the points in the beginning and
                end of the step response

        Possible improvements:

        a) Instead of doing cosine fittings for each projection, I expect better
        results doing cosine fitting with COMMON frequency free parameter
        between the cosines that are being fitted to each projection, i.e. in
        the second pass after knowing the amplitude and offset, we would fit
        a cosine function with fixed amplitude, fixed offset, free phase and
        free frequency but this frequency value would be constrained such that
        it simultaneously best fits all Bloch vector projections.
        I only thought of this after implementing the version with independent
        fitting. I expect better SNR and maybe easier fitting as it constrains
        it more, or bad results if the projection have systematic errors.
        Actually, with this the amplitude and offset could be shared as well
        and therefore a second pass maybe not necessary.

        b) Don't assume fixed frequency in the fitting window and instead use
        a linear time-dependent frequency. This should help getting more
        accurate response for the rising of the flux pulse.

        ================================================================

        Full example of working with the cryoscope tools for FIR corrections

        NB the analysis is a bit heavy, might take a few minutes to run for
        very long measurements, and especially long if the fist are failing!!!

        READ FIRST!

        # ##############################################################
        # Analysis tool
        # ##############################################################

        from pycqed.analysis_v2 import cryoscope_v2_tools as cv2
        import numpy as np
        from scipy import signal
        from pycqed.analysis_v2 import measurement_analysis as ma2

        ts = "20200718_202347"
        qubit = "D1"
        a_obj = ma2.Cryoscope_v2_Analysis(
            qubit=qubit,
            t_start=ts,
            savgol_window=3,
            savgol_polyorder=1,
            kw_exp_fit={
                'tau_min': 0,
                'tau_max': 3,
                'time_ns_fit_max': 15,
                'threshold_apply': 0.99
            },
        )
        rdd = a_obj.raw_data_dict
        pdd = a_obj.proc_data_dict
        qois = pdd["quantities_of_interest"]
        time_ns = qois["time_ns"]

        # ##############################################################
        # Plot analysed step response
        # ##############################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        time_ns = qois["time_ns"]

        key = "step_response_average"
        step_response = qois[key]
        ax.plot(time_ns[:len(step_response)], step_response, label=key)

        key = "step_response_average_filtered"
        step_response = qois[key]
        ax.plot(time_ns[:len(step_response)], step_response, label=key)

        ax.hlines(np.array([.99, .999, 1.01, 1.001, .97, 1.03]) ,
                  xmin=np.min(time_ns), xmax=np.max(time_ns[:len(step_response)]),
                  linestyle=["-", "--", "-", "--", "-.", "-."])

        ax.set_title(ts + ": Cryoscope " + qubit)
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
        max_taps = int(30 * 2.4)

        t_min_baseline_ns = np.max(time_ns) - 25
        t_max_baseline_ns = np.max(time_ns) - 5

        opt_input = step_response

        opt_fir, _ = cv2.optimize_fir_software(
        #     step_response,
            opt_input,
            baseline_start=np.where(time_ns > t_min_baseline_ns)[0].min(),
            baseline_stop=np.where(time_ns > t_max_baseline_ns)[0].min(),
            max_taps=max_taps,
            cma_options={
                "verb_disp":10000,  # Avoid too much output
                #"ftarget": 1e-3, "tolfun": 1e-15, "tolfunhist": 1e-15, "tolx": 1e-15
            }
        )

        # ##############################################################
        # FIR optimization plotting
        # ##############################################################

        ac_soft_FIR = signal.lfilter(opt_fir, 1, opt_input)

        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        ax.plot(time_ns[:len(step_response)], step_response, "-o")
        ax.plot(time_ns[:len(opt_input)], opt_input, "-o")
        ax.plot(time_ns[:len(step_response)], ac_soft_FIR, "-o")

        ax.hlines(np.array([.99, .999, 1.01, 1.001]),
                  xmin=np.min(time_ns), xmax=np.max(time_ns[:len(step_response)]),
                  linestyle=["-", "--", "-", "--"])

        ax.vlines(([t_min_baseline_ns, t_max_baseline_ns]),
                  ymin=np.min(step_response), ymax=np.max(step_response), color="red")

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

        last_hardware_fir = fir_0

        last_FIR = cv2.convert_FIR_from_HDAWG(last_hardware_fir)  # UPDATE last FIR FOR EACH ITERATION!
        c1 = cv2.convolve_FIRs([last_FIR, opt_fir])

        cv2.print_FIR_loading(
            qubit,
            filter_model_number,
            cv2.convert_FIR_for_HDAWG(c1),
            real_time=True)
        """
        if options_dict is None:
            options_dict = dict()
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=close_figs,
            **kwargs
        )

        self.kw_processing = kw_processing
        self.kw_extract = kw_extract
        self.kw_exp_fit = kw_exp_fit
        self.kw_rough_freq_to_amp = kw_rough_freq_to_amp

        self.qubit = qubit
        self.insert_ideal_projection = insert_ideal_projection
        # Necessary to know how to present data correctly
        self.idx_processed = 1 if self.insert_ideal_projection else 0
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.average_exclusion_val_names = average_exclusion_val_names

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        [2020-07-15] data extraction style copied from
        `multiplexed_readout_analysis`
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {
            "data": ("Experimental Data/Data", "dset"),
            "value_units": ("Experimental Data", "attr:value_units"),
            "value_names": ("Experimental Data", "attr:value_names"),
        }

        rdd = self.raw_data_dict = hd5.extract_pars_from_datafile(data_fp, param_spec)

        # Convert to proper python types and not bytes
        rdd["value_names"] = np.array(rdd["value_names"], dtype=str)
        rdd["value_units"] = np.array(rdd["value_units"], dtype=str)
        # Parts added to be compatible with base analysis data requirements
        rdd["folder"] = os.path.dirname(data_fp)
        rdd["timestamps"] = self.timestamps
        rdd["measurementstring"] = rdd["folder"]

    def process_data(self):
        rdd = self.raw_data_dict
        vlns = rdd["value_names"]
        self.proc_data_dict = OrderedDict()
        pdd = self.proc_data_dict

        pdd["time"] = rdd["data"][:, : -len(vlns)].flatten()
        mvs = pdd["measured_values"] = rdd["data"][:, -len(vlns) :].T

        # Keep this in here to raise any error regarding this extraction
        # right away before the heavy data processing
        self.kw_extract["qubit"] = self.qubit
        self.kw_extract["timestamp"] = self.timestamp
        amp_pars = cv2_tools.extract_amp_pars(**self.kw_extract)

        results = OrderedDict()

        # Working in ns to avoid fitting and numerical problems
        time_ns = pdd["time"] * 1e9

        # Confirm that first point was not measured starting from zero,
        # zero has no meaning
        start_idx = 0 if time_ns[0] != 0.0 else 1

        time_ns = time_ns[start_idx:]

        pnts_per_fit_second_pass = self.kw_processing.get("pnts_per_fit_second_pass", 3)
        pnts_per_fit_first_pass = self.kw_processing.get("pnts_per_fit_first_pass", 4)

        self.kw_processing.update(
            {
                "pnts_per_fit_first_pass": pnts_per_fit_first_pass,
                "pnts_per_fit_second_pass": pnts_per_fit_second_pass,
                "insert_ideal_projection": self.insert_ideal_projection,
            }
        )

        for mv, vln in zip(mvs, vlns):
            res = cv2_tools.cryoscope_v2_processing(
                time_ns=time_ns,
                osc_data=mv[start_idx:],
                vln=vln,
                # NB it True, the raw step response is effectively right-shifted
                # as consequence of how the data flows in this analysis
                **self.kw_processing
            )
            results[vln] = res

        exclude = np.array(
            [
                np.all([excl not in vn for excl in self.average_exclusion_val_names])
                for vn in vlns
            ]
        )
        vlns_for_av = vlns[exclude]

        all_freq = np.array([results[key]["results"]["frequency"] for key in vlns])
        all_freq_for_average = np.array(
            [results[vln]["results"]["frequency"] for vln in vlns_for_av]
        )
        av_freq = np.average(all_freq_for_average, axis=0)

        all_names_filtered = [name + "_filtered" for name in vlns]
        all_freq_filtered = np.array(
            [
                signal.savgol_filter(sig, self.savgol_window, self.savgol_polyorder, 0)
                for sig in [*all_freq, av_freq]
            ]
        )

        res = {
            "results": results,
            "averaged_frequency": av_freq,
            "amp_pars": amp_pars,
            "time_ns": time_ns,  # This one always starts at 1/sample_rate
        }
        plateau_time_start_ns = self.kw_rough_freq_to_amp["plateau_time_start_ns"]
        plateau_time_end_ns = self.kw_rough_freq_to_amp["plateau_time_end_ns"]
        assert plateau_time_start_ns < plateau_time_end_ns

        for frequencies, name in zip(
            # Make available in the results all combinations
            [*all_freq, av_freq, *all_freq_filtered],
            [*vlns, "average", *all_names_filtered, "average_filtered"],
        ):
            conversion = cv2_tools.rough_freq_to_amp(
                amp_pars, time_ns, frequencies, **self.kw_rough_freq_to_amp
            )

            # Here we correct for the averaging effect of the moving cosine-fitting
            # window, we attribute the obtained frequency to the middle point in the
            # fitting window, and interpolate linearly the missing points do to the
            # right shift, this step response can be more accurate in certain cases
            # We try also to fit an exponential signal to the first few
            # data points and use it to interpolate the missing points,
            # might be more accurate for distortion corrections
            step_response = conversion["step_response"]
            extra_pnts = pnts_per_fit_second_pass // 2

            # Fit only first 15 ns
            step_response_fit = np.array(step_response)[self.idx_processed :]
            time_ns_fit = time_ns[extra_pnts:][: len(step_response_fit)]
            where = np.where(time_ns_fit < self.kw_exp_fit.get("time_ns_fit_max", 15))[
                0
            ]
            step_response_fit = step_response_fit[where]
            time_ns_fit = time_ns_fit[where]

            def exp_rise(t, tau):
                return 1 - np.exp(-t / tau)

            model = lmfit.Model(exp_rise)
            params = model.make_params()
            params["tau"].value = 1
            params["tau"].min = self.kw_exp_fit.get("tau_min", 0)
            params["tau"].max = self.kw_exp_fit.get("tau_max", 15)

            try:
                fit_res = model.fit(step_response_fit, t=time_ns_fit, params=params)
                params = {key: fit_res.params[key] for key in fit_res.params.keys()}
                exp_fit = exp_rise(time_ns_fit, **params)
                if step_response[self.idx_processed] < self.kw_exp_fit.get(
                    "threshold_apply", 0.97
                ):
                    # Only extrapolate if the first point is significantly below
                    corrected_pnts = exp_fit[:extra_pnts]
                else:
                    corrected_pnts = [step_response[self.idx_processed]] * extra_pnts
                    # For some cases maybe works better to just assume the first
                    # point is calibrated, didn't test enough...
                    # corrected_pnts = [1.0] * extra_pnts

                conversion.update(
                    {
                        "tau": ufloat(
                            params["tau"].value,
                            params["tau"].stderr
                            if params["tau"].stderr is not None
                            else np.nan,
                        )
                    }
                )
                conversion.update({"exp_fit": exp_fit})
            except Exception as e:
                log.warning("Exponential fit failed!\n{}".format(e))
                corrected_pnts = [step_response[self.idx_processed]] * extra_pnts

            corrected_pnts = [step_response[self.idx_processed]] * extra_pnts

            step_response = np.concatenate(
                (
                    # Extrapolate the missing points assuming exponential rise
                    # Seems a fair assumption and much better than a linear
                    # extrapolation
                    corrected_pnts,
                    step_response[self.idx_processed :],
                )
            )
            conversion["step_response_processed"] = step_response

            for key, val in conversion.items():
                res[key + "_" + name] = conversion[key]

        pdd["quantities_of_interest"] = res

    def prepare_plots(self):

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict
        vlns = rdd["value_names"]
        vlus = rdd["value_units"]
        qois = self.proc_data_dict["quantities_of_interest"]

        fs = plt.rcParams["figure.figsize"]

        fig_id_amp = "osc_amp"
        fig_id_step_resp = "step_response"
        fig_id_step_resp_av = "step_response_av"
        # define figure and axes here to have custom layout
        # One extra plot for the average

        time = pdd["time"]
        total_t = time[-1] - time[0]
        xsize = fs[0] * (np.ceil(total_t * 1e9 / 50) * 2)
        nrows = len(vlns)
        self.figs[fig_id_amp], axs_amp = plt.subplots(
            ncols=1, nrows=nrows, figsize=(xsize, fs[1] * nrows * 1.2), sharex=True
        )
        self.figs[fig_id_step_resp], axs_step_resp = plt.subplots(
            ncols=1, nrows=nrows, figsize=(fs[0] * 4, fs[1] * nrows * 1.2), sharex=True,
        )
        self.figs[fig_id_step_resp_av], axs_step_resp_av = plt.subplots(
            ncols=1, nrows=1, figsize=(fs[0] * 4, fs[1] * 1.2 * 2)
        )

        self.figs[fig_id_amp].patch.set_alpha(0)
        self.figs[fig_id_step_resp].patch.set_alpha(0)
        self.figs[fig_id_step_resp_av].patch.set_alpha(0)

        for fig_id, axs_group in zip(
            [fig_id_amp, fig_id_step_resp], [axs_amp, axs_step_resp]
        ):
            for ax_id, ax in zip([*vlns, "average"], axs_group):
                self.axs[fig_id + "_" + ax_id] = ax

        self.axs[fig_id_step_resp_av + "_average"] = axs_step_resp_av

        dt = time[1] - time[0]
        pnts_per_fit_second_pass = self.kw_processing.get("pnts_per_fit_second_pass", 3)
        num_pnts = 20 if total_t < 100e-9 else 50
        times_0 = np.linspace(
            time[0] - dt / 2, time[pnts_per_fit_second_pass - 1] + dt / 2, num_pnts
        )
        xlabel = "Square pulse truncation time"
        fig_id = fig_id_amp
        for vln, vlu in zip(vlns, vlus):
            ax_id = fig_id + "_" + vln

            time = qois["results"][vln]["time_ns"] * 1e-9
            osc_data = qois["results"][vln]["osc_data"]

            amps = qois["results"][vln]["results"]["amplitude"]
            offset = qois["results"][vln]["results"]["offset"]
            frequency = qois["results"][vln]["results"]["frequency"]
            phase = qois["results"][vln]["results"]["phase"]

            yvals = amps + offset
            self.plot_dicts[ax_id + "_amp_pos"] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": time[: len(yvals)],
                "yvals": yvals,
                "marker": "",
                "linestyle": "--",
                "color": "darkgray",
            }
            yvals = offset - amps
            self.plot_dicts[ax_id + "_amp_neg"] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": time[: len(yvals)],
                "yvals": yvals,
                "marker": "",
                "linestyle": "--",
                "color": "darkgray",
            }
            self.plot_dicts[ax_id + "_offset"] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": time[: len(offset)],
                "yvals": offset,
                "marker": "",
                "linestyle": "--",
                "color": "lightgray",
            }
            all_times = [
                times_0 + time_offset for time_offset in np.arange(0, len(amps), 1) * dt
            ]
            all_cos = [
                fit_mods.CosFunc(
                    t=time_sample,
                    amplitude=amp,
                    offset=offset_,
                    phase=phase_,
                    frequency=freq * 1e9,
                )
                for time_sample, offset_, phase_, amp, freq in zip(
                    all_times, offset, phase, amps, frequency
                )
            ]
            self.plot_dicts[ax_id + "_cos_fits"] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": all_times,
                "yvals": all_cos,
                "marker": "",
                "linestyle": "-",
            }
            self.plot_dicts[ax_id] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": time,
                "xunit": "s",
                "yunit": vlu,
                "yvals": osc_data,
                "marker": "o",
                "linestyle": "",
                "xlabel": xlabel,
                "ylabel": "Osc. ampl. " + vln,
            }

        # fig_id = fig_id_step_resp
        time = qois["time_ns"] * 1e-9
        for fig_id, vln, vlu in zip(
            [fig_id_step_resp] * len(vlns) + [fig_id_step_resp_av],
            [*vlns, "average"],
            [*vlus, "a.u."],
        ):
            ax_id = fig_id + "_" + vln

            levels = [0.005, 0.01, 0.03]
            linestyles = [":", "--", "-"]
            labels = ["±{:1.1f}%".format(level * 100) for level in levels]
            for level, linestyle, label in zip(levels, linestyles, labels):
                self.plot_dicts[ax_id + "_level_pos_" + label] = {
                    "plotfn": self.plot_matplot_ax_method,
                    "ax_id": ax_id,
                    "func": "hlines",
                    "plot_kws": {
                        "xmin": time[0],
                        "xmax": time[-1],
                        "linestyle": linestyle,
                        "label": label,
                        "y": 1 - level,
                    },
                }

            self.plot_dicts[ax_id + "_level_neg"] = {
                "plotfn": self.plot_matplot_ax_method,
                "ax_id": ax_id,
                "func": "hlines",
                "plot_kws": {
                    "xmin": time[0],
                    "xmax": time[-1],
                    "linestyle": linestyles,
                    "y": 1 + np.array(levels),
                },
            }

            label1 = "step_response_" + vln
            label2 = "step_response_" + vln + "_filtered"
            label3 = "step_response_processed_" + vln
            label4 = "step_response_processed_" + vln + "_filtered"
            for label in [label1, label2, label3, label4]:
                yvals = qois[label]
                self.plot_dicts[ax_id + label] = {
                    "plotfn": self.plot_line,
                    "ax_id": ax_id,
                    "xvals": time[: len(yvals)],
                    "yvals": yvals,
                    "marker": ".",
                    "linestyle": "-",
                    "setlabel": label,
                    "do_legend": label == label4,
                    "legend_pos": "lower center",
                    "ylabel": "Step resp. " + vln,
                    "yunit": "a.u.",
                    "xlabel": xlabel,
                    "xunit": "s",
                }

class multi_qubit_cryoscope_analysis(ba.BaseDataAnalysis):
    """
    Simultaneous cryoscope analysis.
    """

    def __init__(self,
                 update_FIRs: bool=False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True,
                 poly_params: dict = None,
                ):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.poly_params = poly_params
        self.update_FIRs = update_FIRs
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = hd5.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
        # Extract extra required quantities
        self.Qubits = [ s.split(' ')[-2] for s in self.raw_data_dict['value_names'][::2] ]
        poly_params = { f'polycoeff_{q}': (f'Instrument settings/flux_lm_{q}', 
                                           'attr:q_polycoeffs_freq_01_det') for q in self.Qubits }
        filter_params = { f'filter_{q}': (f'Instrument settings/lin_dist_kern_{q}',
                                           'attr:filter_model_04') for q in self.Qubits }
        if self.poly_params:
            self.Poly_coeffs = self.poly_params
        else:
            self.Poly_coeffs = hd5.extract_pars_from_datafile(data_fp, poly_params)
            # Parse strings
            self.Poly_coeffs = { q : [ float(n) for n in self.Poly_coeffs[f'polycoeff_{q}'][1:-1].split(' ')\
                                       if n != '' ] for q in self.Qubits }
        if self.update_FIRs:
            self.Filters = hd5.extract_pars_from_datafile(data_fp, filter_params)
            # Parse strings
            for q, s in self.Filters.items():
                s = ''.join(s.split('\n')).replace("'", '"')
                s = s.replace('array', '')
                s = s.replace('([', '[')
                s = s.replace('])', ']')
                s = s.replace('True', '"True"')
                s = json.loads(s)
                try:
                    self.Filters[q] = np.array(s['params']['weights'])
                except:
                    print(f'No filter in {q} distortion kernel.')
                    filter_array = np.zeros(40)
                    filter_array[0] = 1
                    self.Filters[q] = filter_array

    def process_data(self):
        self.proc_data_dict = {}
        ###########################
        # Perform trace analysis
        ###########################
        a_obj = ma2.Basic1DAnalysis(t_start=self.timestamp)
        self.proc_data_dict['Traces'] = []
        for n, qubit in enumerate(self.Qubits):
            data_shape = a_obj.raw_data_dict['measured_values'][0][n*2:n*2+2]
            cryoscope_no_dist = ma2.Cryoscope_Analysis(t_start=self.timestamp,
                ch_idx_cos=0, ch_idx_sin=1, derivative_window_length=8e-9, close_figs=True,
                ch_amp_key='Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_amplitude'.format(qubit),
                waveform_amp_key='Snapshot/instruments/flux_lm_{}/parameters/sq_amp'.format(qubit),
                ch_range_key='Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_range'.format(qubit),
                polycoeffs_freq_conv=self.Poly_coeffs[qubit],raw_data=data_shape,
                qubit_label='for qubit_{}'.format(qubit),
                extract_only=True)  
            t = cryoscope_no_dist.ca.time
            a = cryoscope_no_dist.ca.get_amplitudes()
            baseline_start = 50
            baseline_stop = 100
            norm = np.mean(a[baseline_start:baseline_stop])
            a /= norm
            if np.std(a) < 1e-6: # UHF is switching channels
                    cryoscope_no_dist = ma2.Cryoscope_Analysis(t_start=self.timestamp,
                        ch_idx_cos=1, ch_idx_sin=0, derivative_window_length=8e-9, close_figs=True,
                        ch_amp_key='Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_amplitude'.format(qubit),
                        waveform_amp_key='Snapshot/instruments/flux_lm_{}/parameters/sq_amp'.format(qubit),
                        ch_range_key='Snapshot/instruments/flux_lm_{}/parameters/cfg_awg_channel_range'.format(qubit),
                        polycoeffs_freq_conv=self.Poly_coeffs[qubit],raw_data=data_shape,
                        qubit_label='for qubit_{}'.format(qubit),
                        extract_only=True)  
                    t = cryoscope_no_dist.ca.time
                    a = cryoscope_no_dist.ca.get_amplitudes()
                    baseline_start = 50
                    baseline_stop = 100
                    norm = np.mean(a[baseline_start:baseline_stop])
                    a /= norm

            self.proc_data_dict['Traces'].append(a)
        self.proc_data_dict['time'] = t
        ###################################
        # Run filter optimizer in parallel
        ###################################
        if self.update_FIRs:
            # Parallel optimization crashes terminal
            # pool_ = mp.pool.ThreadPool(mp.cpu_count()-1)
            # self.proc_data_dict['opt_filter'] = pool_.map(optimize_fir_software, [t for t in self.proc_data_dict['Traces']] )
            # pool_.close()
            self.proc_data_dict['opt_filter'] = [ optimize_fir_software(t) for t in self.proc_data_dict['Traces'] ] 
            # Convolve new filter with old filter
            self.proc_data_dict['conv_filters'] = {}
            print(self.Filters)
            for i, qubit in enumerate(self.Qubits):
                old_filter = cv2_tools.convert_FIR_from_HDAWG(self.Filters[f'filter_{qubit}'])
                new_filter = self.proc_data_dict['opt_filter'][i]
                conv_filter = cv2_tools.convolve_FIRs([old_filter, new_filter])
                conv_filter = cv2_tools.convert_FIR_for_HDAWG(conv_filter)
                self.proc_data_dict['conv_filters'][qubit] = conv_filter

    def prepare_plots(self):
        for i, qubit in enumerate(self.Qubits):
            self.plot_dicts[f'Cryscope_trace_{qubit}'] = {
                'plotfn': plot_cryoscope_trace,
                'trace': self.proc_data_dict['Traces'][i],
                'time': self.proc_data_dict['time'],
                'qubit': qubit,
                'timestamp': self.timestamp
            }

def optimize_fir_software(y, baseline_start=100,
                          baseline_stop=None, taps=72, start_sample=0, stop_sample=200, cma_target=0.5):
    step_response = np.concatenate((np.array([0]), y))
    baseline = np.mean(y[baseline_start:baseline_stop])
    x0 = [1] + (taps - 1) * [0]    
    def objective_function_fir(x):
        y = step_response
        yc = signal.lfilter(x, 1, y)
        return np.mean(np.abs(yc[1+start_sample:stop_sample] - baseline))/np.abs(baseline)
    return cma.fmin2(objective_function_fir, x0, cma_target)[0]

def plot_cryoscope_trace(trace, 
                         time, 
                         timestamp,
                         qubit,
                         ax=None, **kw):
    ax.plot(time, trace)
    set_xlabel(ax, 'Time', 's')
    set_ylabel(ax, 'Amplitude', 'a.u.')
    ax.set_ylim(0.95,1.05)
    ax.set_xlim(0, time[-1])
    ax.axhline(1, color='grey', ls='-')
    ax.axhline(1.01, color='grey', ls='--')
    ax.axhline(0.99, color='grey', ls='--')
    ax.axhline(1.001, color='grey', ls=':')
    ax.axhline(0.999, color='grey', ls=':')
    ax.set_title(timestamp+': Step responses Cryoscope '+qubit)

class Time_frequency_analysis(ba.BaseDataAnalysis):
    def __init__(self,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = hd5.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        # Sort data
        self.qubit = self.raw_data_dict['folder'].split('_')[-1]
        Time = self.raw_data_dict['data'][:,0]
        X_proj = self.raw_data_dict['data'][:,1]
        Y_proj = self.raw_data_dict['data'][:,2]
        # PSD of signal
        time_step = Time[1]-Time[0]
        _X_proj = X_proj-np.mean(X_proj) # Remove 0 freq coef
        _Y_proj = Y_proj-np.mean(Y_proj) #
        PSD = np.abs(np.fft.fft(_X_proj+1j*_Y_proj))**2*time_step/len(Time)
        Freqs = np.fft.fftfreq(_X_proj.size, time_step)
        idx = np.argsort(Freqs)
        Freqs = Freqs[idx]
        PSD = PSD[idx]
        # Make lorentzian Fit to extract detuning 
        # (Doing so, overcomes the frequency sampling accuracy)
        def lorentz(x, x0, G):
            return G**2 / ( (x-x0)**2 + G**2 )
        def fit_func(x, x0, G, A, B):
            return A*lorentz(x, x0, G) + B
        # Fit guess
        det_0 = Freqs[np.argmax(PSD)] 
        G_0 = 1/(Time[-1]-Time[0])
        A_0 = np.max(PSD)-np.mean(PSD)
        B_0 = np.mean(PSD)
        p0 = [det_0, G_0, A_0, B_0]
        from scipy.optimize import curve_fit
        try:
            popt, pcov = curve_fit(fit_func, Freqs, PSD,
                                   p0=p0, maxfev=int(1e5))
            detuning = popt[0]
        except:
            print('Fitting Failed!')
            detuning = Freqs[np.argmax(PSD)] 
        # Save processed data
        self.proc_data_dict['Time'] = Time
        self.proc_data_dict['X_proj'] = X_proj
        self.proc_data_dict['Y_proj'] = Y_proj
        self.proc_data_dict['Freqs'] = Freqs
        self.proc_data_dict['PSD'] = PSD
        self.proc_data_dict['detuning'] = detuning

    def prepare_plots(self):
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(5, 4), nrows=2, dpi=100)
        axs = axs.flatten()
        # fig.patch.set_alpha(0)
        self.axs_dict[f'FFT_of_time_trace'] = axs[0]
        self.figs[f'FFT_of_time_trace'] = fig
        self.plot_dicts['FFT_of_time_trace'] = {
            'plotfn': FFT_plotfn,
            'ax_id': 'FFT_of_time_trace',
            'Time': self.proc_data_dict['Time'],
            'X_proj': self.proc_data_dict['X_proj'],
            'Y_proj': self.proc_data_dict['Y_proj'],
            'Freqs': self.proc_data_dict['Freqs'],
            'PSD': self.proc_data_dict['PSD'],
            'detuning': self.proc_data_dict['detuning'],
            'qubit': self.qubit,
            'timestamp': self.timestamps[0]
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def FFT_plotfn(
    Time,
    X_proj,
    Y_proj,
    Freqs,
    PSD,
    detuning,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    axs[0].plot(Time*1e9, X_proj, 'C0.-')
    axs[0].plot(Time*1e9, Y_proj, 'C1.-')
    axs[0].set_xlabel('Pulse duration (ns)')
    axs[0].set_ylabel('Voltage (a.u.)')

    PSD += np.mean(PSD)
    axs[1].plot(Freqs*1e-6, PSD, '-')
    axs[1].set_xlim(Freqs[0]*1e-6, Freqs[-1]*1e-6)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('PSD ($\mathrm{Hz^{-1}}$)')
    axs[1].set_yscale('log')

    axs[0].set_title(f'{timestamp}\n{qubit} Cryoscope trace')

    fig.tight_layout()

class Flux_arc_analysis(ba.BaseDataAnalysis):
    def __init__(self,
                 channel_amp:float,
                 channel_range:float,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.ch_amp = channel_amp
        self.ch_range = channel_range
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = hd5.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        # Sort data
        self.qubit = self.raw_data_dict['folder'].split('_')[-1]
        Amps = self.raw_data_dict['data'][:,0]*self.ch_amp*self.ch_range/2
        Freqs = np.abs(self.raw_data_dict['data'][:,1])
        _Amps = np.array(list(Amps)+[0])
        _Freqs = np.array(list(Freqs)+[0])
        # RDC 26/10/2023. deg was 2, I am changing it to 4 to improve the freq conversion
        P_coefs = np.polyfit(_Amps, _Freqs, deg=4)
        # Save processed data
        self.proc_data_dict['Amps'] = Amps
        self.proc_data_dict['Freqs'] = Freqs
        self.qoi = {}
        self.qoi['P_coefs'] = P_coefs 

    def prepare_plots(self):
        self.axs_dict = {}
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict[f'Voltage_arc_trace'] = ax
        self.figs[f'Voltage_arc_trace'] = fig
        self.plot_dicts['Voltage_arc_trace'] = {
            'plotfn': Voltage_arc_plotfn,
            'ax_id': 'Voltage_arc_trace',
            'Amps': self.proc_data_dict['Amps'],
            'Freqs': self.proc_data_dict['Freqs'],
            'P_coefs': self.qoi['P_coefs'],
            'qubit': self.qubit,
            'timestamp': self.timestamps[0]
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Voltage_arc_plotfn(
    Amps,
    Freqs,
    P_coefs,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    _Amps = np.linspace(Amps[0]*1.1, Amps[-1]*1.1)
    Arc = np.poly1d(P_coefs)
    ax.plot(_Amps, Arc(_Amps)*1e-6, 'C0-')
    ax.plot(Amps, Freqs*1e-6, 'C3.')
    ax.grid(ls = '--')
    ax.set_ylabel('Detuning (MHz)')
    ax.set_xlabel('Output Voltage (V)')
    ax.set_title(f'{timestamp}\n{qubit} Voltage frequency arc')
    fig.tight_layout()