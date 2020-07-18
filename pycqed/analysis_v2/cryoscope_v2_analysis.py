"""
Created: 2020-07-15
Author: Victor Negirneac
"""

import matplotlib.pyplot as plt
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.analysis_v2.cryoscope_v2_tools as cv2_tools
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
        filtered = vlns[exclude]

        all_freq = np.array([results[key]["results"]["frequency"] for key in vlns])
        all_freq_for_average = np.array(
            [results[vln]["results"]["frequency"] for vln in filtered]
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
            "time_ns": time_ns,
        }

        idx_processed = 1 if self.insert_ideal_projection else 0

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
            step_response_fit = np.array(step_response)[idx_processed:]
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

            if step_response[idx_processed] < self.kw_exp_fit.get(
                "threshold_apply", 0.97
            ):
                # Only extrapolate if the first point is significantly below
                corrected_pnts = exp_fit[:extra_pnts]
            else:
                corrected_pnts = [step_response[idx_processed]] * extra_pnts
                # For some cases maybe works better to just assume the first
                # point is calibrated, didn't test enough...
                # corrected_pnts = [1.0] * extra_pnts

            step_response = np.concatenate(
                (
                    # Extrapolate the missing points assuming exponential rise
                    # Seems a fair assumption and much better than a linear
                    # extrapolation
                    corrected_pnts,
                    step_response[idx_processed:],
                )
            )
            conversion.update(
                {"tau": ufloat(params["tau"].value, params["tau"].stderr)}
            )
            conversion.update({"exp_fit": exp_fit})
            conversion["step_response_processed_" + name] = step_response

            # Renaming to be able to return the step responses from all measured
            # channels along side with the average
            step_response = conversion.pop("step_response")
            conversion["step_response_" + name] = step_response
            res.update(conversion)

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
        # define figure and axes here to have custom layout
        # One extra plot for the average

        nrows = len(vlns)
        self.figs[fig_id_amp], axs_amp = plt.subplots(
            ncols=1,
            nrows=nrows,
            figsize=(fs[0] * 4, fs[1] * nrows * 1.2)
        )
        self.figs[fig_id_step_resp], axs_step_resp = plt.subplots(
            ncols=1,
            nrows=nrows,
            figsize=(fs[0] * 4 + 1, fs[1] * nrows * 1.2)
        )

        self.figs[fig_id_amp].patch.set_alpha(0)
        self.figs[fig_id_step_resp].patch.set_alpha(0)

        for fig_id, axs_group in zip([fig_id_amp, fig_id_step_resp], [axs_amp, axs_step_resp]):
            for ax_id, ax in zip([*vlns, "average"], axs_group):
                self.axs[fig_id + "_" + ax_id] = ax

        fig_id = fig_id_amp
        for vln, mv, vlu in zip(vlns, mvs, vlus):
            ax_id = fig_id + "_" + vln
            self.plot_dicts[ax_id] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": pdd["time"],
                "xunit": "s",
                "yunit": vlu,
                "yvals": mv,
                "marker": "o",
                "linestyle": "",
            }
            yvals = qois["results"]
            self.plot_dicts[ax_id] = {
                "plotfn": self.plot_line,
                "ax_id": ax_id,
                "xvals": pdd["time"],
                "xunit": "s",
                "yunit": vlu,
                "yvals": mv,
                "marker": "o",
                "linestyle": "",
            }

    # if plot:
    #     fig, axs = plt.subplots(len(results) + 2, 1, figsize=(20, 25))

    #     ax = axs[0]
    #     ax.plot(
    #         time_ns[: len(amps_from_fit)], amps_from_fit, label="Osc. amp. first pass"
    #     )
    #     ax.plot(x_for_fit, np.poly1d(line_fit)(x_for_fit), label="Line fit osc. amp.")
    #     ax.set_xlabel("Osc. amp. (a.u.)")
    #     ax.legend()

    #     ax = axs[1]
    #     ax.plot(time_ns, osc_data, "o")

    #     for i in range(len(results[list(results.keys())[0]])):
    #         res_pars = {key: results[key][i] for key in results.keys()}
    #         time_sample = np.linspace(
    #             time_ns[i], time_ns[i + pnts_per_fit_second_pass - 1], 20
    #         )
    #         cos_fit_sample = fit_mods.CosFunc(t=time_sample, **res_pars)
    #         ax.set_xlabel("Pulse duration (ns)")
    #         ax.set_ylabel("Amplitude (a.u.)")
    #         ax.plot(time_sample, cos_fit_sample, "-")

    #     for ax, key in zip(axs[2:], results.keys()):
    #         ax.plot(time_ns[: len(results[key])], results[key], "-o")

    #         if key == "frequency":
    #             ax.set_ylabel("Frequency (GHz)")
    #         elif key == "amplitude":
    #             ax.set_ylabel("Amplitude (a.u.)")
    #         elif key == "offset":
    #             ax.set_ylabel("Offset (a.u.)")
    #         elif key == "phase":
    #             ax.set_ylabel("Phase (deg)")
    #         ax.set_xlabel("Pulse duration (ns)")
