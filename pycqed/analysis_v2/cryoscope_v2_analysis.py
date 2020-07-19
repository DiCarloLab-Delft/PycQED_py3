"""
Created: 2020-07-15
Author: Victor Negirneac
"""

import matplotlib.pyplot as plt
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.analysis_v2.cryoscope_v2_tools as cv2_tools
from pycqed.analysis import fitting_models as fit_mods
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.hdf5_data as hd5
from collections import OrderedDict
from uncertainties import ufloat
from scipy import signal
import os
import lmfit
import numpy as np
import logging

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
        kw_rough_freq_to_amp={"plateau_time_start_ns": -25, "plateau_time_end_ns": -5},
        kw_exp_fit={
            "tau_min": 0,
            "tau_max": 15,
            "time_ns_fit_max": 15,
            "threshold_apply": 0.97,
        },
        kw_processing={"pnts_per_fit_second_pass": 3, "pnts_per_fit_first_pass": 4},
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
        Second version of cryoscope analysis

        Does not require the flux arc, assumes quadratic dependence on frequency

        Uses a moving cos-fitting-window to extract instantaneous oscillation
        frequency

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

        self.kw_extract["qubit"] = self.qubit
        self.kw_extract["timestamp"] = self.timestamp
        amp_pars = cv2_tools.extract_amp_pars(**self.kw_extract)

        res = {
            "results": results,
            "averaged_frequency": av_freq,
            "amp_pars": amp_pars,
            "time_ns": time_ns,  # This one always starts at 1/sample_rate
        }

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
            # TO DO: try to instead fit an exponential signal to the first few
            # data points and use it to interpolate the missing points,
            # might be more accurate for distortion corrections
            step_response = conversion["step_response"]
            extra_pnts = pnts_per_fit_second_pass // 2

            # # Fit only first 15 ns
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

            step_response = np.concatenate(
                (
                    # Extrapolate the missing points assuming exponential rise
                    # Seems a fair assumption and much better than a linear
                    # extrapolation
                    corrected_pnts,
                    step_response[self.idx_processed :],
                )
            )
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
            conversion["step_response_processed"] = step_response

            for key, val in conversion.items():
                res[key + "_" + name] = conversion[key]

        pdd["quantities_of_interest"] = res

    def prepare_plots(self):

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict
        vlns = rdd["value_names"]
        vlus = rdd["value_units"]
        mvs = pdd["measured_values"]
        qois = self.proc_data_dict["quantities_of_interest"]

        fs = plt.rcParams["figure.figsize"]

        fig_id_amp = "osc_amp"
        fig_id_step_resp = "step_response"
        fig_id_step_resp_av = "step_response_av"
        # define figure and axes here to have custom layout
        # One extra plot for the average

        nrows = len(vlns)
        self.figs[fig_id_amp], axs_amp = plt.subplots(
            ncols=1, nrows=nrows, figsize=(fs[0] * 4, fs[1] * nrows * 1.2), sharex=True
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

            dt = time[1] - time[0]
            pnts_per_fit_second_pass = self.kw_processing.get(
                "pnts_per_fit_second_pass", 3
            )
            times_0 = np.linspace(
                time[0] - dt / 2, time[pnts_per_fit_second_pass - 1] + dt / 2, 20
            )
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
                "ylabel": "Oscillation amplitude",
            }

        # fig_id = fig_id_step_resp
        for fig_id, vln, vlu in zip(
            [fig_id_step_resp] * len(vlns) + [fig_id_step_resp_av],
            [*vlns, "average"],
            [*vlus, "a.u."],
        ):
            ax_id = fig_id + "_" + vln
            time = qois["time_ns"]

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
                    "ylabel": "Step response",
                    "yunit": "a.u.",
                    "xlabel": xlabel,
                    "xunit": "s"
                }

            levels = [0.005, 0.01, 0.03]
            linestyle = [":", "--", "-"]
            self.plot_dicts[ax_id + "_percent_neg"] = {
                "plotfn": self.plot_matplot_ax_method,
                "ax_id": ax_id,
                "func": "hlines",
                "plot_kws": {
                    "xmin": time[0],
                    "xmax": time[-1],
                    "linestyle": linestyle,
                    "label": ["±{:1.1f}".format(level * 100) for level in levels],
                    "y": 1 - np.array(levels),
                },
            }
            self.plot_dicts[ax_id + "_percent_pos"] = {
                "plotfn": self.plot_matplot_ax_method,
                "ax_id": ax_id,
                "func": "hlines",
                "plot_kws": {
                    "xmin": time[0],
                    "xmax": time[-1],
                    "linestyle": linestyle,
                    "y": 1 + np.array(levels),
                },
            }
