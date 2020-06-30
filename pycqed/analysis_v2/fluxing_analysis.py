import lmfit
from uncertainties import ufloat
from pycqed.analysis import measurement_analysis as ma
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from scipy.spatial import ConvexHull

from pycqed.analysis.tools.plotting import (
    set_xlabel,
    set_ylabel,
    plot_fit,
    hsluv_anglemap45,
    SI_prefix_and_scale_factor,
)

from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.analysis_toolbox import color_plot

import matplotlib.pyplot as plt
import matplotlib.colors as col
from pycqed.analysis.fitting_models import (
    CosFunc,
    Cos_guess,
    avoided_crossing_freq_shift,
    ChevronInvertedFunc,
    ChevronFunc,
    ChevronGuess,
)
import pycqed.analysis_v2.simple_analysis as sa

import scipy.cluster.hierarchy as hcluster

from copy import deepcopy
import pycqed.analysis.tools.plot_interpolation as plt_interp

from pycqed.utilities import general as gen
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman as flm
from datetime import datetime
from pycqed.measurement.optimization import multi_targets_phase_offset

from pycqed.analysis_v2.tools.plotting import (
    scatter_pnts_overlay,
    contour_overlay,
    annotate_pnts,
)
from pycqed.analysis_v2.tools import contours2d as c2d

import logging

log = logging.getLogger(__name__)


class Chevron_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label=None,
        ch_idx=0,
        coupling="g",
        min_fit_amp=0,
        auto=True,
    ):
        """
        Analyzes a Chevron and fits the avoided crossing.

        Parameters
        ----------
        ts: str
            timestamp of the datafile
        label: str
            label to find the datafile (optional)
        ch_idx: int
            channel to use when fitting the avoided crossing
        coupling: Enum("g", "J1", "J2")
            used to label the avoided crossing and calculate related quantities
        min_fit_amp:
            minimal maplitude of the fitted cosine for each line cut.
            Oscillations with a smaller amplitude will be ignored in the fit
            of the avoided crossing.
        auto: bool
            if True run all parts of the analysis.

        """
        super().__init__(do_fitting=True)
        self.ts = ts
        self.label = label
        self.coupling = coupling
        self.ch_idx = ch_idx
        self.min_fit_amp = min_fit_amp
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a = ma.MeasurementAnalysis(timestamp=self.ts, label=self.label, auto=False)
        a.get_naming_and_values_2D()
        a.finish()
        self.timestamps = [a.timestamp_string]
        self.raw_data_dict["timestamps"] = self.timestamps
        self.raw_data_dict["timestamp_string"] = a.timestamp
        for attr in [
            "sweep_points",
            "sweep_points_2D",
            "measured_values",
            "parameter_names",
            "parameter_units",
            "value_names",
            "value_units",
        ]:
            self.raw_data_dict[attr] = getattr(a, attr)
        self.raw_data_dict["folder"] = a.folder

    def process_data(self):
        self.proc_data_dict = OrderedDict()

        # select the relevant data
        x = self.raw_data_dict["sweep_points"]
        t = self.raw_data_dict["sweep_points_2D"]
        Z = self.raw_data_dict["measured_values"][self.ch_idx].T

        # fit frequencies to each individual cut (time trace)
        freqs = []
        freqs_std = []
        fit_results = []
        amps = []
        for xi, z in zip(x, Z.T):
            CosModel = lmfit.Model(CosFunc)
            CosModel.guess = Cos_guess
            pars = CosModel.guess(CosModel, z, t)
            fr = CosModel.fit(data=z, t=t, params=pars)
            amps.append(fr.params["amplitude"].value)
            freqs.append(fr.params["frequency"].value)
            freqs_std.append(fr.params["frequency"].stderr)
            fit_results.append(fr)
        # N.B. the fit results are not saved in self.fit_res as this would
        # bloat the datafiles.
        self.proc_data_dict["fit_results"] = np.array(fit_results)
        self.proc_data_dict["amp_fits"] = np.array(amps)
        self.proc_data_dict["freq_fits"] = np.array(freqs)
        self.proc_data_dict["freq_fits_std"] = np.array(freqs_std)

        # take a Fourier transform (nice for plotting)
        fft_data = abs(np.fft.fft(Z.T).T)
        fft_freqs = np.fft.fftfreq(len(t), d=t[1] - t[0])
        sort_vec = np.argsort(fft_freqs)

        fft_data_sorted = fft_data[sort_vec, :]
        fft_freqs_sorted = fft_freqs[sort_vec]
        self.proc_data_dict["fft_data_sorted"] = fft_data_sorted
        self.proc_data_dict["fft_freqs_sorted"] = fft_freqs_sorted

    def run_fitting(self):
        super().run_fitting()

        fit_mask = np.where(self.proc_data_dict["amp_fits"] > self.min_fit_amp)

        avoided_crossing_mod = lmfit.Model(avoided_crossing_freq_shift)
        # hardcoded guesses! Bad practice, needs a proper guess func
        avoided_crossing_mod.set_param_hint("a", value=3e9)
        avoided_crossing_mod.set_param_hint("b", value=-2e9)
        avoided_crossing_mod.set_param_hint("g", value=20e6, min=0)
        params = avoided_crossing_mod.make_params()

        self.fit_res["avoided_crossing"] = avoided_crossing_mod.fit(
            data=self.proc_data_dict["freq_fits"][fit_mask],
            flux=self.raw_data_dict["sweep_points"][fit_mask],
            params=params,
        )

    def analyze_fit_results(self):
        self.proc_data_dict["quantities_of_interest"] = {}
        # Extract quantities of interest from the fit
        self.proc_data_dict["quantities_of_interest"] = {}
        qoi = self.proc_data_dict["quantities_of_interest"]
        g = self.fit_res["avoided_crossing"].params["g"]
        qoi["g"] = ufloat(g.value, g.stderr)

        self.coupling_msg = ""
        if self.coupling == "J1":
            qoi["J1"] = qoi["g"]
            qoi["J2"] = qoi["g"] * np.sqrt(2)
            self.coupling_msg += (
                r"Measured $J_1$ = {} MHz".format(qoi["J1"] * 1e-6) + "\n"
            )
            self.coupling_msg += r"Expected $J_2$ = {} MHz".format(qoi["J2"] * 1e-6)
        elif self.coupling == "J2":
            qoi["J1"] = qoi["g"] / np.sqrt(2)
            qoi["J2"] = qoi["g"]
            self.coupling_msg += (
                r"Expected $J_1$ = {} MHz".format(qoi["J1"] * 1e-6) + "\n"
            )
            self.coupling_msg += r"Measured $J_2$ = {} MHz".format(qoi["J2"] * 1e-6)
        else:
            self.coupling_msg += "g = {}".format(qoi["g"])

    def prepare_plots(self):
        for i, val_name in enumerate(self.raw_data_dict["value_names"]):
            self.plot_dicts["chevron_{}".format(val_name)] = {
                "plotfn": plot_chevron,
                "x": self.raw_data_dict["sweep_points"],
                "y": self.raw_data_dict["sweep_points_2D"],
                "Z": self.raw_data_dict["measured_values"][i].T,
                "xlabel": self.raw_data_dict["parameter_names"][0],
                "ylabel": self.raw_data_dict["parameter_names"][1],
                "zlabel": self.raw_data_dict["value_names"][i],
                "xunit": self.raw_data_dict["parameter_units"][0],
                "yunit": self.raw_data_dict["parameter_units"][1],
                "zunit": self.raw_data_dict["value_units"][i],
                "title": self.raw_data_dict["timestamp_string"]
                + "\n"
                + "Chevron {}".format(val_name),
            }

        self.plot_dicts["chevron_fft"] = {
            "plotfn": plot_chevron_FFT,
            "x": self.raw_data_dict["sweep_points"],
            "xunit": self.raw_data_dict["parameter_units"][0],
            "fft_freqs": self.proc_data_dict["fft_freqs_sorted"],
            "fft_data": self.proc_data_dict["fft_data_sorted"],
            "freq_fits": self.proc_data_dict["freq_fits"],
            "freq_fits_std": self.proc_data_dict["freq_fits_std"],
            "fit_res": self.fit_res["avoided_crossing"],
            "coupling_msg": self.coupling_msg,
            "title": self.raw_data_dict["timestamp_string"]
            + "\n"
            + "Fourier transform of Chevron",
        }


def plot_chevron(x, y, Z, xlabel, xunit, ylabel, yunit, zlabel, zunit, title, ax, **kw):
    colormap = ax.pcolormesh(
        x,
        y,
        Z,
        cmap="viridis",  # norm=norm,
        linewidth=0,
        rasterized=True,
        # assumes digitized readout
        vmin=0,
        vmax=1,
    )
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(colormap, cax=cax, orientation="vertical")
    cax.set_ylabel("L1 (%)")

    set_ylabel(cax, zlabel, zunit)


def plot_chevron_FFT(
    x,
    xunit,
    fft_freqs,
    fft_data,
    freq_fits,
    freq_fits_std,
    fit_res,
    coupling_msg,
    title,
    ax,
    **kw
):

    colormap = ax.pcolormesh(
        x,
        fft_freqs,
        fft_data,
        cmap="viridis",  # norm=norm,
        linewidth=0,
        rasterized=True,
        vmin=0,
        vmax=5,
    )

    ax.errorbar(
        x=x,
        y=freq_fits,
        yerr=freq_fits_std,
        ls="--",
        c="r",
        alpha=0.5,
        label="Extracted freqs",
    )
    x_fine = np.linspace(x[0], x[-1], 200)
    plot_fit(x, fit_res, ax=ax, c="C1", label="Avoided crossing fit", ls=":")

    set_xlabel(ax, "Flux bias", xunit)
    set_ylabel(ax, "Frequency", "Hz")
    ax.legend(loc=(1.05, 0.7))
    ax.text(1.05, 0.5, coupling_msg, transform=ax.transAxes)


class Chevron_Alignment_Analysis(sa.Basic2DInterpolatedAnalysis):
    """
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        close_figs: bool = True,
        options_dict: dict = None,
        extract_only: bool = False,
        do_fitting: bool = True,
        auto: bool = True,
        save_qois: bool = True,
        fit_from: str = "",
        fit_threshold: float = None,
        sq_pulse_duration: float = None,
        peak_is_inverted: bool = True,
    ):
        self.fit_from = fit_from
        self.fit_threshold = fit_threshold
        self.sq_pulse_duration = sq_pulse_duration
        self.peak_is_inverted = peak_is_inverted

        if do_fitting and sq_pulse_duration is None:
            log.error(
                "Pulse duration is required for fitting. Fitting will be skipped!"
            )
        do_fitting = do_fitting and sq_pulse_duration is not None

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            close_figs=close_figs,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
            save_qois=save_qois,
            auto=auto,
            interp_method="linear",
        )

    def extract_data(self):
        super().extract_data()

    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict

        bias_axis = "x" if "FBL" in self.raw_data_dict["xlabel"].upper() else "y"
        pdd["bias_axis"] = bias_axis
        amps_axis = "y" if bias_axis == "x" else "x"
        pdd["amps_axis"] = amps_axis
        unique_bias_values = np.unique(self.raw_data_dict[bias_axis])
        pdd["unique_bias_values"] = unique_bias_values
        bias_1D_cuts = []
        pdd["bias_1D_cuts"] = bias_1D_cuts
        bias_strs = []
        pdd["bias_strs"] = bias_strs
        for unique_bias in unique_bias_values:
            is_this_unique = self.raw_data_dict[bias_axis] == unique_bias
            is_neg_amp = self.raw_data_dict[amps_axis] < 0
            is_pos_amp = self.raw_data_dict[amps_axis] > 0
            idxs_amps = np.where(is_this_unique)[0]
            idxs_amps_neg = np.where(is_this_unique * is_neg_amp)[0]
            idxs_amps_pos = np.where(is_this_unique * is_pos_amp)[0]
            amps_neg = self.raw_data_dict[amps_axis][idxs_amps_neg]
            amps_pos = self.raw_data_dict[amps_axis][idxs_amps_pos]
            amps = self.raw_data_dict[amps_axis][idxs_amps]
            mv = self.raw_data_dict["measured_values"][:, idxs_amps]
            mv_neg = self.raw_data_dict["measured_values"][:, idxs_amps_neg]
            mv_pos = self.raw_data_dict["measured_values"][:, idxs_amps_pos]
            bias_1D_cuts.append(
                {
                    "amps_neg": amps_neg,
                    "amps_pos": amps_pos,
                    "mv_neg": mv_neg,
                    "mv_pos": mv_pos,
                    "amps": amps,
                    "mv": mv,
                }
            )

            scale_factor, unit = SI_prefix_and_scale_factor(
                val=unique_bias, unit=self.proc_data_dict["yunit"]
            )
            bias_strs.append("{:4g} ({})".format(unique_bias * scale_factor, unit))

        # values stored in quantities of interest will be saved in the data file
        self.proc_data_dict["quantities_of_interest"] = {}

    def prepare_fitting(self):
        t = self.sq_pulse_duration

        fit_d = self.fit_dicts

        pdd = self.proc_data_dict

        if self.fit_from != "":
            fit_from_idx = self.raw_data_dict["value_names"].index(self.fit_from)
        else:
            fit_from_idx = 1
            self.fit_from = self.raw_data_dict["value_names"][fit_from_idx]

        for i, bdict in enumerate(pdd["bias_1D_cuts"]):
            # Allow fitting the populations of both qubits
            fit_func = ChevronInvertedFunc if self.peak_is_inverted else ChevronFunc
            chevron_model = lmfit.Model(fit_func)
            chevron_model.guess = ChevronGuess

            fit_key = "chevron_fit_{}".format(i)
            fit_xvals = bdict["amps"]
            fit_yvals = bdict["mv"][fit_from_idx]

            if self.fit_threshold is not None:
                # For some cases the fit might not work well due to noise
                # This is to fit above a threshold only
                selection = (
                    (fit_yvals < self.fit_threshold)
                    if self.peak_is_inverted
                    else (fit_yvals > self.fit_threshold)
                )
                sel_idx = np.where(selection)[0]
                fit_yvals = fit_yvals[sel_idx]
                fit_xvals = fit_xvals[sel_idx]

            fit_d[fit_key] = {
                "model": chevron_model,
                "guessfn_pars": {"model": chevron_model, "t": t},
                "fit_xvals": {"amp": fit_xvals},
                "fit_yvals": {"data": fit_yvals},
            }

    def analyze_fit_results(self):
        pdd = self.proc_data_dict
        ubv = pdd["unique_bias_values"]
        fit_res = self.fit_res
        qoi = pdd["quantities_of_interest"]

        centers_diffs = []

        chevron_centers_L = []
        chevron_centers_R = []
        chevron_centers_L_vals = []
        chevron_centers_R_vals = []
        for bias, fit_key in zip(ubv, fit_res.keys()):
            amp_center_1 = fit_res[fit_key].params["amp_center_1"]
            amp_center_2 = fit_res[fit_key].params["amp_center_2"]
            centers = [amp_center_1, amp_center_2]
            arg_amp_L = np.argmin([amp_center_1.value, amp_center_2.value])
            arg_amp_R = np.argmax([amp_center_1.value, amp_center_2.value])

            stderr_L = (
                centers[arg_amp_L].stderr
                if centers[arg_amp_L].stderr is not None
                else np.nan
            )
            stderr_R = (
                centers[arg_amp_R].stderr
                if centers[arg_amp_R].stderr is not None
                else np.nan
            )

            chevron_centers_L.append(ufloat(centers[arg_amp_L].value, stderr_L))
            chevron_centers_R.append(ufloat(centers[arg_amp_R].value, stderr_R))

            chevron_centers_L_vals.append(centers[arg_amp_L].value)
            chevron_centers_R_vals.append(centers[arg_amp_R].value)

            centers_diffs.append(centers[arg_amp_L].value + centers[arg_amp_R].value)

        pdd["chevron_centers_L"] = chevron_centers_L
        pdd["chevron_centers_R"] = chevron_centers_R
        pdd["centers_diffs"] = centers_diffs

        bias_calibration_coeffs = np.polyfit(centers_diffs, ubv, 1)
        pdd["bias_calibration_coeffs"] = bias_calibration_coeffs
        calib_bias = bias_calibration_coeffs[1]
        pdd["calibration_bias"] = calib_bias

        bias_calibration_coeffs_L = np.polyfit(chevron_centers_L_vals, ubv, 1)
        bias_calibration_coeffs_R = np.polyfit(chevron_centers_R_vals, ubv, 1)

        p = bias_calibration_coeffs_L
        int_pnt_L = (calib_bias - p[1]) / p[0]
        p = bias_calibration_coeffs_R
        int_pnt_R = (calib_bias - p[1]) / p[0]
        pdd["interaction_pnts"] = (int_pnt_L, int_pnt_R)

        amp_interaction_pnt = (np.abs(int_pnt_L) + np.abs(int_pnt_R)) / 2
        pdd["amp_interaction_pnt"] = amp_interaction_pnt

        qoi["calibration_bias"] = calib_bias
        qoi["amp_interaction_pnt"] = amp_interaction_pnt

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()

        bias_1D_cuts = self.proc_data_dict["bias_1D_cuts"]
        num_cuts = len(bias_1D_cuts)

        for i, val_name in enumerate(self.proc_data_dict["value_names"]):
            ax_id = "all_bias_1D_cuts_" + val_name
            self.plot_dicts[ax_id] = {
                "ax_id": ax_id,
                "plotfn": plot_chevron_bias_1D_cuts,
                "bias_1D_cuts_dicts": bias_1D_cuts,
                "xlabel": self.proc_data_dict["xlabel"],
                "xunit": self.proc_data_dict["xunit"],
                "ylabel": val_name,
                "yunit": self.proc_data_dict["value_units"][i],
                "title": "{}\n{}".format(
                    self.timestamp, self.proc_data_dict["measurementstring"]
                ),
                "title_neg": val_name + " (amp < 0)",
                "title_pos": val_name + " (amp > 0)",
                "sharex": False,
                "sharey": True,
                "plotsize": (13, 5 * num_cuts),
                "numplotsy": num_cuts,
                "numplotsx": 2,
                "mv_indx": i,
            }
        if self.do_fitting:
            self._prepare_fit_plots()

    def _prepare_fit_plots(self):
        pdd = self.proc_data_dict
        pd = self.plot_dicts
        for i, fit_key in enumerate(self.fit_res.keys()):
            bias_str = pdd["bias_strs"][i]
            pd[fit_key + "_L"] = {
                "ax_id": "all_bias_1D_cuts_" + self.fit_from,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts[fit_key]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit [flux bias = " + bias_str + "]",
                "do_legend": True,
                "ax_row": i,
                "ax_col": 0,
            }
            pd[fit_key + "_R"] = {
                "ax_id": "all_bias_1D_cuts_" + self.fit_from,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts[fit_key]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit [flux bias = " + bias_str + "]",
                "do_legend": True,
                "ax_row": i,
                "ax_col": 1,
            }

            pd["all_bias_1D_cuts_" + self.fit_from][
                "fit_threshold"
            ] = self.fit_threshold
            pd["all_bias_1D_cuts_" + self.fit_from][
                "fit_threshold"
            ] = self.fit_threshold

            center_L = pdd["chevron_centers_L"][i]
            center_R = pdd["chevron_centers_R"][i]
            pd[fit_key + "_L_center"] = {
                "ax_id": "all_bias_1D_cuts_" + self.fit_from,
                "plotfn": plot_chevron_center_on_1D_cut,
                "center_amp_ufloat": center_L,
                "label": center_L,
                "ax_row": i,
                "ax_col": 0,
            }
            pd[fit_key + "_R_center"] = {
                "ax_id": "all_bias_1D_cuts_" + self.fit_from,
                "plotfn": plot_chevron_center_on_1D_cut,
                "center_amp_ufloat": center_R,
                "label": center_R,
                "ax_row": i,
                "ax_col": 1,
            }

        calib_bias = pdd["calibration_bias"]
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=calib_bias, unit=pdd["yunit"]
        )
        calib_bias_str = "{:4g} ({})".format(calib_bias * scale_factor, unit)

        poly_calib = np.poly1d(pdd["bias_calibration_coeffs"])
        xs = np.array(pdd["centers_diffs"])[[0, -1]]

        amp_interaction_pnt = pdd["amp_interaction_pnt"]
        for i, val_name in enumerate(pdd["value_names"]):
            # Order here matters due to the legend
            self.plot_dicts["int_pnts_" + val_name] = {
                "ax_id": val_name,
                "plotfn": self.plot_line,
                "func": "scatter",
                "xvals": [pdd["interaction_pnts"][0], pdd["interaction_pnts"][1]],
                "yvals": [calib_bias, calib_bias],
                "marker": "o",
                "color": "gold",
                "line_kws": {"edgecolors": "gray", "linewidth": 0.7, "s": 100},
                "setlabel": "Amp at interaction: {:3g}".format(amp_interaction_pnt),
            }
            self.plot_dicts["bias_fit_calib_" + val_name] = {
                "ax_id": val_name,
                "plotfn": self.plot_matplot_ax_method,
                "func": "axhline",
                "plot_kws": {
                    "y": calib_bias,
                    "ls": "--",
                    "color": "red",
                    "label": "Sweet spot bias: " + calib_bias_str,
                },
            }
            self.plot_dicts["bias_fit_" + val_name] = {
                "ax_id": val_name,
                "plotfn": self.plot_line,
                "xvals": xs,
                "yvals": poly_calib(xs),
                "setlabel": "Flux bias fit",
                "do_legend": True,
                "marker": "",
                "linestyles": "r--",
                "color": "red",
            }
            self.plot_dicts["bias_fit_data_" + val_name] = {
                "ax_id": val_name,
                "plotfn": self.plot_line,
                "func": "scatter",
                "xvals": pdd["centers_diffs"],
                "yvals": pdd["unique_bias_values"],
                "marker": "o",
                "color": "orange",
                "line_kws": {"edgecolors": "gray", "linewidth": 0.5},
            }


def plot_chevron_bias_1D_cuts(bias_1D_cuts_dicts, mv_indx, fig=None, ax=None, **kw):
    if ax is None:
        num_cuts = len(bias_1D_cuts_dicts)
        fig, ax = plt.subplots(
            num_cuts, 2, sharex=False, sharey=True, figsize=(13, 5 * num_cuts)
        )
        fig.tight_layout()

    xlabel = kw.get("xlabel", "")
    ylabel = kw.get("ylabel", "")
    x_unit = kw.get("xunit", "")
    y_unit = kw.get("yunit", "")

    fit_threshold = kw.get("fit_threshold", None)

    title_neg = kw.pop("title_neg", None)
    title_pos = kw.pop("title_pos", None)

    if title_neg is not None:
        ax[0][0].set_title(title_neg)
    if title_pos is not None:
        ax[0][1].set_title(title_pos)

    edgecolors = "grey"
    linewidth = 0.2
    cmap = "plasma"
    for i, d in enumerate(bias_1D_cuts_dicts):
        ax[i][0].scatter(
            d["amps_neg"],
            d["mv_neg"][mv_indx],
            edgecolors=edgecolors,
            linewidth=linewidth,
            c=range(len(d["amps_neg"])),
            cmap=cmap,
        )
        ax[i][0].set_xlim(np.min(d["amps_neg"]), np.max(d["amps_neg"]))
        ax[i][1].scatter(
            d["amps_pos"],
            d["mv_pos"][mv_indx],
            edgecolors=edgecolors,
            linewidth=linewidth,
            c=range(len(d["amps_pos"])),
            cmap=cmap,
        )
        ax[i][1].set_xlim(np.min(d["amps_pos"]), np.max(d["amps_pos"]))

        # shide the spines between
        ax[i][0].spines["right"].set_visible(False)
        ax[i][1].spines["left"].set_visible(False)
        ax[i][0].yaxis.tick_left()
        ax[i][1].tick_params(labelleft=False)
        ax[i][1].yaxis.tick_right()

        set_ylabel(ax[i][0], ylabel, unit=y_unit)

        if fit_threshold is not None:
            label = "Fit threshold"
            ax[i][0].axhline(fit_threshold, ls="--", color="green", label=label)
            ax[i][1].axhline(fit_threshold, ls="--", color="green", label=label)

    set_xlabel(ax[-1][0], xlabel, unit=x_unit)
    set_xlabel(ax[-1][1], xlabel, unit=x_unit)

    return fig, ax


def plot_chevron_center_on_1D_cut(
    center_amp_ufloat, ax_row, ax_col, label, ax, fig=None, **kw
):
    ax[ax_row][ax_col].axvline(
        center_amp_ufloat.n, ls="--", label="Center: " + str(label)
    )
    ax[ax_row][ax_col].legend()
    ax[ax_row][ax_col].axvline(
        center_amp_ufloat.n - center_amp_ufloat.s, ls=":", color="grey"
    )
    ax[ax_row][ax_col].axvline(
        center_amp_ufloat.n + center_amp_ufloat.s, ls=":", color="grey"
    )
    return fig, ax


class Conditional_Oscillation_Heatmap_Analysis(ba.BaseDataAnalysis):
    """
    Intended for the analysis of CZ tuneup heatmaps
    The data can be from an experiment or simulation
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        close_figs: bool = True,
        options_dict: dict = None,
        extract_only: bool = False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto: bool = True,
        interp_method: str = "linear",
        plt_orig_pnts: bool = True,
        plt_contour_phase: bool = True,
        plt_contour_L1: bool = False,
        plt_optimal_values: bool = True,
        plt_optimal_values_max: int = 1,
        plt_clusters: bool = True,
        clims: dict = None,
        # e.g. clims={'L1': [0, 0.3], "Cost func": [0., 100]},
        L1_contour_levels: list = [1, 5, 10],
        phase_contour_levels: list = [90, 180, 270],
        find_local_optimals: bool = True,
        phase_thr=5,
        L1_thr=0.5,
        clustering_thr=10 / 360,
        cluster_from_interp: bool = True,
        _opt_are_interp: bool = True,
        sort_clusters_by: str = "cost",
        target_cond_phase: float = 180.0,
        single_q_phase_offset: bool = False,
        calc_L1_from_missing_frac: bool = True,
        calc_L1_from_offset_diff: bool = False,
        hull_clustering_thr=0.1,
        hull_phase_thr=5,
        hull_L1_thr=5,
        gen_optima_hulls=True,
        plt_optimal_hulls=True,
        comparison_timestamp: str = None,
        interp_grid_data: bool = False,
        save_cond_phase_contours: list = [180],
    ):

        self.plt_orig_pnts = plt_orig_pnts
        self.plt_contour_phase = plt_contour_phase
        self.plt_contour_L1 = plt_contour_L1
        self.plt_optimal_values = plt_optimal_values
        self.plt_optimal_values_max = plt_optimal_values_max
        self.plt_clusters = plt_clusters

        # Optimals are interpolated
        # Manually set to false if the default analysis flow is changed
        # e.g. in get_guesses_from_cz_sim in flux_lutman
        # In that case we re-evaluate the optimals to be able to return
        # true values and not interpolated, even though the optimal is
        # obtained from interpolation
        self._opt_are_interp = _opt_are_interp

        self.clims = clims
        self.L1_contour_levels = L1_contour_levels
        self.phase_contour_levels = phase_contour_levels

        self.find_local_optimals = find_local_optimals
        self.phase_thr = phase_thr
        self.L1_thr = L1_thr
        self.clustering_thr = clustering_thr
        self.cluster_from_interp = cluster_from_interp
        # This alows for different strategies of scoring several optima
        # NB: When interpolation we will not get any lower value than what
        # already exists on the landscape
        self.sort_clusters_by = sort_clusters_by
        assert sort_clusters_by in {"cost", "L1_av_around"}

        self.target_cond_phase = target_cond_phase
        # Used when applying Pi pulses to check if both single qubits
        # have the same phase as in the ideal case
        self.single_q_phase_offset = single_q_phase_offset
        # Handy calculation for comparing experiment and simulations
        # but using the same analysis code
        self.calc_L1_from_missing_frac = calc_L1_from_missing_frac
        self.calc_L1_from_offset_diff = calc_L1_from_offset_diff
        # Compare to any other dataset that has the same shape for
        # 'measured_values'
        self.comparison_timestamp = comparison_timestamp

        # Used to generate the vertices of hulls that can be used later
        # reoptimize only in the regions of interest
        self.hull_clustering_thr = hull_clustering_thr
        self.hull_phase_thr = hull_phase_thr
        self.hull_L1_thr = hull_L1_thr
        self.gen_optima_hulls = gen_optima_hulls
        self.plt_optimal_hulls = plt_optimal_hulls

        self.interp_method = interp_method
        # Be able to also analyze linear 2D sweeps without interpolating
        self.interp_grid_data = interp_grid_data
        self.save_cond_phase_contours = save_cond_phase_contours

        # FIXME this is overkill, using .upper() and .lower() would simplify
        cost_func_Names = {
            "Cost func",
            "Cost func.",
            "cost func",
            "cost func.",
            "cost function",
            "Cost function",
            "Cost function value",
        }
        L1_names = {"L1", "Leakage", "half missing fraction"}
        ms_names = {
            "missing fraction",
            "Missing fraction",
            "missing frac",
            "missing frac.",
            "Missing frac",
            "Missing frac.",
        }
        cond_phase_names = {
            "Cond phase",
            "Cond. phase",
            "Conditional phase",
            "cond phase",
            "cond. phase",
            "conditional phase",
        }
        offset_diff_names = {
            "offset difference",
            "offset diff",
            "offset diff.",
            "Offset difference",
            "Offset diff",
            "Offset diff.",
        }
        phase_q0_names = {"Q0 phase", "phase q0"}

        # also account for possible underscores instead of a spaces between words
        allNames = [
            cost_func_Names,
            L1_names,
            ms_names,
            cond_phase_names,
            offset_diff_names,
            phase_q0_names,
        ]
        allNames = [
            names.union({name.replace(" ", "_") for name in names})
            for names in allNames
        ]
        allNames = [
            names.union(
                {name + " 1" for name in names}.union({name + " 2" for name in names})
            )
            for names in allNames
        ]
        [
            self.cost_func_Names,
            self.L1_names,
            self.ms_names,
            self.cond_phase_names,
            self.offset_diff_names,
            self.phase_q0_names,
        ] = allNames

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            close_figs=close_figs,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
            save_qois=save_qois,
        )

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop, label=self.labels
        )
        self.raw_data_dict["timestamps"] = self.timestamps

        self.timestamp = self.timestamps[0]
        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamp, auto=False, close_file=False
        )
        a.get_naming_and_values()

        for idx, lab in enumerate(["x", "y"]):
            self.raw_data_dict[lab] = a.sweep_points[idx]
            self.raw_data_dict["{}label".format(lab)] = a.parameter_names[idx]
            self.raw_data_dict["{}unit".format(lab)] = a.parameter_units[idx]

        self.raw_data_dict["measured_values"] = a.measured_values
        self.raw_data_dict["value_names"] = a.value_names
        self.raw_data_dict["value_units"] = a.value_units
        self.raw_data_dict["measurementstring"] = a.measurementstring
        self.raw_data_dict["folder"] = a.folder
        a.finish()

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()
        anglemap = hsluv_anglemap45
        found_optimals = np.size(self.proc_data_dict["x_optimal"]) > 0
        for i, val_name in enumerate(self.proc_data_dict["value_names"]):

            zlabel = "{} ({})".format(val_name, self.proc_data_dict["value_units"][i])
            self.plot_dicts[val_name] = {
                "ax_id": val_name,
                "plotfn": color_plot,
                "x": self.proc_data_dict["x_int"],
                "y": self.proc_data_dict["y_int"],
                "z": self.proc_data_dict["interpolated_values"][i],
                "xlabel": self.proc_data_dict["xlabel"],
                "x_unit": self.proc_data_dict["xunit"],
                "ylabel": self.proc_data_dict["ylabel"],
                "y_unit": self.proc_data_dict["yunit"],
                "zlabel": zlabel,
                "title": "{}\n{}".format(
                    self.timestamp, self.proc_data_dict["measurementstring"]
                ),
            }

            if self.plt_orig_pnts:
                self.plot_dicts[val_name + "_non_interpolated"] = {
                    "ax_id": val_name,
                    "plotfn": scatter_pnts_overlay,
                    "x": self.proc_data_dict["x"],
                    "y": self.proc_data_dict["y"],
                }
            unit = self.proc_data_dict["value_units"][i]
            vmin = np.min(self.proc_data_dict["interpolated_values"][i])
            vmax = np.max(self.proc_data_dict["interpolated_values"][i])

            if unit == "deg":
                self.plot_dicts[val_name]["cbarticks"] = np.arange(0.0, 360.1, 45)
                self.plot_dicts[val_name]["cmap_chosen"] = anglemap
                self.plot_dicts[val_name]["clim"] = [0.0, 360.0]
            elif unit == "%":
                self.plot_dicts[val_name]["cmap_chosen"] = "hot"
            elif unit.startswith("∆ "):
                self.plot_dicts[val_name]["cmap_chosen"] = "terrain"
                # self.plot_dicts[val_name]['cmap_chosen'] = 'RdBu'
                vcenter = 0
                if vmin * vmax < 0:
                    divnorm = col.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    self.plot_dicts[val_name]["norm"] = divnorm
                else:
                    self.plot_dicts[val_name]["clim"] = [
                        np.max((vcenter, vmin)),
                        np.min((vcenter, vmax)),
                    ]

            if self.clims is not None and val_name in self.clims.keys():
                self.plot_dicts[val_name]["clim"] = self.clims[val_name]
                # Visual indicator when saturating the color range
                clims = self.clims[val_name]
                cbarextend = "min" if min(clims) > vmin else "neither"
                cbarextend = "max" if max(clims) < vmax else cbarextend
                cbarextend = (
                    "both" if min(clims) > vmin and max(clims) < vmax else cbarextend
                )
                self.plot_dicts[val_name]["cbarextend"] = cbarextend

            if self.plt_contour_phase:
                # Find index of Conditional Phase
                z_cond_phase = None
                for j, val_name_j in enumerate(self.proc_data_dict["value_names"]):
                    if val_name_j in self.cond_phase_names:
                        z_cond_phase = self.proc_data_dict["interpolated_values"][j]
                        break

                if z_cond_phase is not None:
                    self.plot_dicts[val_name + "_cond_phase_contour"] = {
                        "ax_id": val_name,
                        "plotfn": contour_overlay,
                        "x": self.proc_data_dict["x_int"],
                        "y": self.proc_data_dict["y_int"],
                        "z": z_cond_phase,
                        "colormap": anglemap,
                        "cyclic_data": True,
                        "contour_levels": self.phase_contour_levels,
                        "vlim": (0, 360),
                        # "linestyles": "-",
                    }
                else:
                    log.warning("No data found named {}".format(self.cond_phase_names))

            if self.plt_contour_L1:
                # Find index of Leakage or Missing Fraction
                z_L1 = None
                for j, val_name_j in enumerate(self.proc_data_dict["value_names"]):
                    if val_name_j in self.L1_names or val_name_j in self.ms_names:
                        z_L1 = self.proc_data_dict["interpolated_values"][j]
                        break

                if z_L1 is not None:
                    vlim = (
                        self.proc_data_dict["interpolated_values"][j].min(),
                        self.proc_data_dict["interpolated_values"][j].max(),
                    )

                    contour_levels = np.array(self.L1_contour_levels)
                    # Leakage is estimated as (Missing fraction/2)
                    contour_levels = (
                        contour_levels
                        if self.proc_data_dict["value_names"][j] in self.L1_names
                        else 2 * contour_levels
                    )

                    self.plot_dicts[val_name + "_L1_contour"] = {
                        "ax_id": val_name,
                        "plotfn": contour_overlay,
                        "x": self.proc_data_dict["x_int"],
                        "y": self.proc_data_dict["y_int"],
                        "z": z_L1,
                        # 'unit': self.proc_data_dict['value_units'][j],
                        "contour_levels": contour_levels,
                        "vlim": vlim,
                        "colormap": "hot",
                        "linestyles": "-",
                        # "linestyles": "dashdot",
                    }
                else:
                    log.warning("No data found named {}".format(self.L1_names))

            if self.plt_optimal_hulls and self.gen_optima_hulls:
                sorted_hull_vertices = self.proc_data_dict["hull_vertices"]
                for hull_i, hull_vertices in sorted_hull_vertices.items():
                    vertices_x, vertices_y = np.transpose(hull_vertices)

                    # Close the start and end of the line
                    x_vals = np.concatenate((vertices_x, vertices_x[:1]))
                    y_vals = np.concatenate((vertices_y, vertices_y[:1]))

                    self.plot_dicts[val_name + "_hull_{}".format(hull_i)] = {
                        "ax_id": val_name,
                        "plotfn": self.plot_line,
                        "xvals": x_vals,
                        "yvals": y_vals,
                        "marker": "",
                        "linestyles": "-",
                        "color": "blue",
                    }

            if (
                self.plt_optimal_values
                and found_optimals
                and val_name in self.cost_func_Names
            ):
                self.plot_dicts[val_name + "_optimal_pars"] = {
                    "ax_id": val_name,
                    "ypos": -0.25,
                    "xpos": 0,
                    "plotfn": self.plot_text,
                    "box_props": "fancy",
                    "line_kws": {"alpha": 0},
                    "text_string": self.get_readable_optimals(
                        optimal_end=self.plt_optimal_values_max
                    ),
                    "horizontalalignment": "left",
                    "verticalaligment": "top",
                    "fontsize": 14,
                }

            if self.plt_clusters and found_optimals:
                self.plot_dicts[val_name + "_clusters"] = {
                    "ax_id": val_name,
                    "plotfn": scatter_pnts_overlay,
                    "x": self.proc_data_dict["clusters_pnts_x"],
                    "y": self.proc_data_dict["clusters_pnts_y"],
                    "color": None,
                    "edgecolors": None if self.cluster_from_interp else "black",
                    "marker": "o",
                    # 'linewidth': 1,
                    "c": self.proc_data_dict["clusters_pnts_colors"],
                }
            if self.plt_optimal_values and found_optimals:
                self.plot_dicts[val_name + "_optimal_pnts_annotate"] = {
                    "ax_id": val_name,
                    "plotfn": annotate_pnts,
                    "txt": np.arange(np.size(self.proc_data_dict["x_optimal"])),
                    "x": self.proc_data_dict["x_optimal"],
                    "y": self.proc_data_dict["y_optimal"],
                }

        # Extra plot to easily identify the location of the optimal hulls
        # and cond. phase contours
        sorted_hull_vertices = self.proc_data_dict.get("hull_vertices", [])
        if self.gen_optima_hulls and len(sorted_hull_vertices):
            for hull_id, hull_vertices in sorted_hull_vertices.items():
                vertices_x, vertices_y = np.transpose(hull_vertices)

                # Close the start and end of the line
                x_vals = np.concatenate((vertices_x, vertices_x[:1]))
                y_vals = np.concatenate((vertices_y, vertices_y[:1]))

                self.plot_dicts["hull_" + hull_id] = {
                    "ax_id": "hull_and_contours",
                    "plotfn": self.plot_line,
                    "xvals": x_vals,
                    "xlabel": self.raw_data_dict["xlabel"],
                    "xunit": self.raw_data_dict["xunit"],
                    "yvals": y_vals,
                    "ylabel": self.raw_data_dict["ylabel"],
                    "yunit": self.raw_data_dict["yunit"],
                    "yrange": self.options_dict.get("yrange", None),
                    "xrange": self.options_dict.get("xrange", None),
                    "setlabel": "hull #" + hull_id,
                    "title": "{}\n{}".format(
                        self.timestamp, self.proc_data_dict["measurementstring"]
                    ),
                    "do_legend": True,
                    "legend_pos": "best",
                    "marker": "",  # don't use markers
                    "linestyle": "-",
                    # Fixing the assigned color so that it can be matched on
                    # other plots
                    "color": "C" + str(int(hull_id) % 10),
                }

        if len(self.save_cond_phase_contours):
            c_dict = self.proc_data_dict["cond_phase_contours"]
            i = 0
            for level, contours in c_dict.items():
                for contour_id, contour in contours.items():
                    x_vals, y_vals = np.transpose(contour)

                    self.plot_dicts["contour_" + level + "_" + contour_id] = {
                        "ax_id": "hull_and_contours",
                        "plotfn": self.plot_line,
                        "xvals": x_vals,
                        "xlabel": self.raw_data_dict["xlabel"],
                        "xunit": self.raw_data_dict["xunit"],
                        "yvals": y_vals,
                        "ylabel": self.raw_data_dict["ylabel"],
                        "yunit": self.raw_data_dict["yunit"],
                        "yrange": self.options_dict.get("yrange", None),
                        "xrange": self.options_dict.get("xrange", None),
                        "setlabel": level + " #" + contour_id,
                        "title": "{}\n{}".format(
                            self.timestamp, self.proc_data_dict["measurementstring"]
                        ),
                        "do_legend": True,
                        "legend_pos": "best",
                        "legend_ncol": 2,
                        "marker": "",  # don't use markers
                        "linestyle": "--",
                        # Continuing the color cycle
                        "color": "C" + str(len(sorted_hull_vertices) % 10 + i),
                    }
                    i += 1

        # Plotting all quantities along the raw contours of conditional phase
        mvac = self.proc_data_dict.get("measured_values_along_contours", [])
        for i, mv_levels_dict in enumerate(mvac):
            # We iterate over all measured quantities and for each create a
            # plot that has the measured quantity along all contours
            j = 0
            for level, cntrs_dict in mv_levels_dict.items():
                for cntr_id, mvs in cntrs_dict.items():
                    c_pnts = self.proc_data_dict["cond_phase_contours"][level][cntr_id]
                    x_vals = c2d.distance_along_2D_contour(c_pnts, True, True)

                    vln = self.proc_data_dict["value_names"][i]
                    vlu = self.proc_data_dict["value_units"][i]
                    plt_dict_label = "contour_" + vln + "_" + level + "_#" + cntr_id
                    self.plot_dicts[plt_dict_label] = {
                        "ax_id": "contours_" + vln,
                        "plotfn": self.plot_line,
                        "xvals": x_vals,
                        "xlabel": "Normalized distance along contour",
                        "xunit": "a.u.",
                        "yvals": mvs,
                        "ylabel": vln,
                        "yunit": vlu,
                        "setlabel": level + " #" + cntr_id,
                        "title": "{}\n{}".format(
                            self.timestamp, self.proc_data_dict["measurementstring"]
                        ),
                        "do_legend": True,
                        "legend_pos": "best",
                        "legend_ncol": 2,
                        "marker": "",  # don't use markers
                        "linestyle": "-",
                        "color": "C" + str(len(sorted_hull_vertices) % 10 + j),
                    }
                    j += 1

        # Plotting all quantities along the raw contours of conditional phase
        # only inside hulls
        mvac = self.proc_data_dict.get("measured_values_along_contours_in_hulls", [])
        for i, hulls_dict in enumerate(mvac):
            # We iterate over all measured quantities and for each create a
            # plot that has the measured quantity along all contours
            for hull_id, mv_levels_dict in hulls_dict.items():
                j = 0
                for level, cntrs_dict in mv_levels_dict.items():
                    for cntr_id, c_dict in cntrs_dict.items():
                        c_pnts = c_dict["pnts"]
                        mvs = c_dict["vals"]
                        if len(c_pnts):
                            # Only do stuff if there are any point in the hull
                            x_vals = c2d.distance_along_2D_contour(c_pnts, True, True)

                            vln = self.proc_data_dict["value_names"][i]
                            vlu = self.proc_data_dict["value_units"][i]
                            plt_dict_label = (
                                "contour_"
                                + vln
                                + "_hull_#"
                                + hull_id
                                + level
                                + "_#"
                                + cntr_id
                            )
                            self.plot_dicts[plt_dict_label] = {
                                "ax_id": "contours_" + vln + "_in_hull",
                                "plotfn": self.plot_line,
                                "xvals": x_vals,
                                "xlabel": "Normalized distance along contour",
                                "xunit": "a.u.",
                                "yvals": mvs,
                                "ylabel": vln,
                                "yunit": vlu,
                                "setlabel": level + " #" + cntr_id,
                                "title": "{}\n{}".format(
                                    self.timestamp,
                                    self.proc_data_dict["measurementstring"],
                                ),
                                "do_legend": True,
                                "legend_pos": "best",
                                "legend_ncol": 2,
                                "marker": "",  # don't use markers
                                "linestyle": "-",
                                "color": "C" + str(len(sorted_hull_vertices) % 10 + j),
                            }
                            plt_dict_label = (
                                "contour_"
                                + vln
                                + "_hull_#"
                                + hull_id
                                + level
                                + "_#"
                                + cntr_id
                                + "_hull_color"
                            )
                            # We plot with the contour color so that things
                            # can be matched with the contours on the 2D plot
                            extra_pnts_idx = len(x_vals) // 3
                            self.plot_dicts[plt_dict_label] = {
                                "ax_id": "contours_" + vln + "_in_hull",
                                "plotfn": self.plot_line,
                                "xvals": x_vals[[0, extra_pnts_idx, -extra_pnts_idx, -1]],
                                "xlabel": "Normalized distance along contour",
                                "xunit": "a.u.",
                                "yvals": mvs[[0, extra_pnts_idx, -extra_pnts_idx, -1]],
                                "ylabel": vln,
                                "yunit": vlu,
                                "setlabel": "hull #" + hull_id,
                                "title": "{}\n{}".format(
                                    self.timestamp,
                                    self.proc_data_dict["measurementstring"],
                                ),
                                "do_legend": True,
                                "legend_pos": "best",
                                "legend_ncol": 2,
                                "marker": "o",  # don't use markers
                                "linestyle": "",
                                "color": "C" + str(int(hull_id) % 10),
                            }
                        j += 1

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        phase_q0_name = "phase_q0"
        phase_q1_name = "phase_q1"
        if self.single_q_phase_offset and {phase_q0_name, phase_q1_name} <= set(
            self.proc_data_dict["value_names"]
        ):
            # This was used for some debugging
            self.proc_data_dict["value_names"].append("phase_q1 - phase_q0")
            self.proc_data_dict["value_units"].append("deg")
            phase_q0 = self.proc_data_dict["measured_values"][
                self.proc_data_dict["value_names"].index(phase_q0_name)
            ]
            phase_q1 = self.proc_data_dict["measured_values"][
                self.proc_data_dict["value_names"].index(phase_q1_name)
            ]
            self.proc_data_dict["measured_values"] = np.vstack(
                (self.proc_data_dict["measured_values"], (phase_q1 - phase_q0) % 360)
            )

        # Calculate L1 from missing fraction and/or offset difference if available
        vln_set = set(self.proc_data_dict["value_names"])
        for names, do_calc in [
            (self.ms_names, self.calc_L1_from_missing_frac),
            (self.offset_diff_names, self.calc_L1_from_offset_diff),
        ]:
            found_name = len(vln_set.intersection(names)) > 0
            if do_calc and found_name:
                name = vln_set.intersection(names).pop()
                self.proc_data_dict["value_names"].append("half " + name)
                self.proc_data_dict["value_units"].append("%")
                L1_equiv = (
                    self.proc_data_dict["measured_values"][
                        self.proc_data_dict["value_names"].index(name)
                    ]
                    / 2
                )
                self.proc_data_dict["measured_values"] = np.vstack(
                    (self.proc_data_dict["measured_values"], L1_equiv)
                )

        vln = self.proc_data_dict["value_names"]
        measured_vals = self.proc_data_dict["measured_values"]
        vlu = self.proc_data_dict["value_units"]

        # Calculate comparison heatmaps
        if self.comparison_timestamp is not None:
            coha_comp = Conditional_Oscillation_Heatmap_Analysis(
                t_start=self.comparison_timestamp, extract_only=True
            )
            # Because there is no standart what measured quantities are named
            # have to do some magic name matching here
            for names in [
                self.cost_func_Names,
                self.L1_names,
                self.ms_names,
                self.cond_phase_names,
                self.offset_diff_names,
                self.phase_q0_names,
            ]:
                inters_this = names.intersection(self.proc_data_dict["value_names"])
                inters_comp = names.intersection(
                    coha_comp.proc_data_dict["value_names"]
                )
                if len(inters_this) > 0 and len(inters_comp) > 0:
                    this_name = inters_this.pop()
                    comp_name = inters_comp.pop()
                    indx_this_name = self.proc_data_dict["value_names"].index(this_name)
                    self.proc_data_dict["value_names"].append(
                        "[{}]\n{} - {}".format(
                            self.comparison_timestamp, comp_name, this_name
                        )
                    )
                    self.proc_data_dict["value_units"].append(
                        "∆ " + self.proc_data_dict["value_units"][indx_this_name]
                    )
                    this_mv = self.proc_data_dict["measured_values"][indx_this_name]
                    ref_mv = coha_comp.proc_data_dict["measured_values"][
                        coha_comp.proc_data_dict["value_names"].index(comp_name)
                    ]
                    delta_mv = ref_mv - this_mv
                    self.proc_data_dict["measured_values"] = np.vstack(
                        (self.proc_data_dict["measured_values"], delta_mv)
                    )

        self.proc_data_dict["interpolated_values"] = []
        self.proc_data_dict["interpolators"] = []
        interps = self.proc_data_dict["interpolators"]
        for i in range(len(self.proc_data_dict["value_names"])):
            if self.proc_data_dict["value_units"][i] == "deg":
                interp_method = "deg"
            else:
                interp_method = self.interp_method

            ip = plt_interp.HeatmapInterpolator(
                self.proc_data_dict["x"],
                self.proc_data_dict["y"],
                self.proc_data_dict["measured_values"][i],
                interp_method=interp_method,
                rescale=True,
            )
            interps.append(ip)

            x_int, y_int, z_int = plt_interp.interpolate_heatmap(
                x=self.proc_data_dict["x"],
                y=self.proc_data_dict["y"],
                ip=ip,
                n=300,  # avoid calculation of areas
                interp_grid_data=self.interp_grid_data,
            )
            self.proc_data_dict["interpolated_values"].append(z_int)

        interp_vals = self.proc_data_dict["interpolated_values"]
        self.proc_data_dict["x_int"] = x_int
        self.proc_data_dict["y_int"] = y_int

        # Processing for optimal points
        if not self.cluster_from_interp:
            where = [(name in self.cost_func_Names) for name in vln]
            cost_func_indxs = np.where(where)[0][0]
            cost_func = measured_vals[cost_func_indxs]

            try:
                where = [(name in self.cond_phase_names) for name in vln]
                cond_phase_indx = np.where(where)[0][0]
                cond_phase_arr = measured_vals[cond_phase_indx]
            except Exception:
                # Ignore if was not measured
                log.error("\n" + gen.get_formatted_exception())

            try:
                where = [(name in self.L1_names) for name in vln]
                L1_indx = np.where(where)[0][0]
                L1_arr = measured_vals[L1_indx]
            except Exception:
                # Ignore if was not measured
                log.error("\n" + gen.get_formatted_exception())

            theta_f_arr = self.proc_data_dict["x"]
            lambda_2_arr = self.proc_data_dict["y"]

            extract_optimals_from = "measured_values"
        else:
            where = [(name in self.cost_func_Names) for name in vln]
            cost_func_indxs = np.where(where)[0][0]
            cost_func = interp_vals[cost_func_indxs]
            cost_func = interp_to_1D_arr(z_int=cost_func)

            where = [(name in self.cond_phase_names) for name in vln]
            cond_phase_indx = np.where(where)[0][0]
            cond_phase_arr = interp_vals[cond_phase_indx]
            cond_phase_arr = interp_to_1D_arr(z_int=cond_phase_arr)

            where = [(name in self.L1_names) for name in vln]
            L1_indx = np.where(where)[0][0]
            L1_arr = interp_vals[L1_indx]
            L1_arr = interp_to_1D_arr(z_int=L1_arr)

            theta_f_arr = self.proc_data_dict["x_int"]
            lambda_2_arr = self.proc_data_dict["y_int"]

            theta_f_arr, lambda_2_arr = interp_to_1D_arr(
                x_int=theta_f_arr, y_int=lambda_2_arr
            )

            extract_optimals_from = "interpolated_values"

        if self.find_local_optimals:
            optimal_idxs, clusters_by_indx = get_optimal_pnts_indxs(
                theta_f_arr=theta_f_arr,
                lambda_2_arr=lambda_2_arr,
                cond_phase_arr=cond_phase_arr,
                L1_arr=L1_arr,
                cost_arr=cost_func,
                target_phase=self.target_cond_phase,
                phase_thr=self.phase_thr,
                L1_thr=self.L1_thr,
                clustering_thr=self.clustering_thr,
                sort_by_mode=self.sort_clusters_by,
            )
        else:
            optimal_idxs = np.array([cost_func.argmin()])
            clusters_by_indx = np.array([optimal_idxs])

        if self.cluster_from_interp:
            x_arr = theta_f_arr
            y_arr = lambda_2_arr
        else:
            x_arr = self.proc_data_dict["x"]
            y_arr = self.proc_data_dict["y"]

        clusters_pnts_x = np.array([])
        clusters_pnts_y = np.array([])
        clusters_pnts_colors = np.array([])

        for l, cluster_by_indx in enumerate(clusters_by_indx):
            clusters_pnts_x = np.concatenate((clusters_pnts_x, x_arr[cluster_by_indx]))
            clusters_pnts_y = np.concatenate((clusters_pnts_y, y_arr[cluster_by_indx]))
            clusters_pnts_colors = np.concatenate(
                (clusters_pnts_colors, np.full(np.shape(cluster_by_indx)[0], l))
            )

        self.proc_data_dict["optimal_idxs"] = optimal_idxs

        self.proc_data_dict["clusters_pnts_x"] = clusters_pnts_x
        self.proc_data_dict["clusters_pnts_y"] = clusters_pnts_y
        self.proc_data_dict["clusters_pnts_colors"] = clusters_pnts_colors

        self.proc_data_dict["x_optimal"] = x_arr[optimal_idxs]
        self.proc_data_dict["y_optimal"] = y_arr[optimal_idxs]

        optimal_pars_values = []
        for x, y in zip(
            self.proc_data_dict["x_optimal"], self.proc_data_dict["y_optimal"]
        ):
            optimal_pars_values.append(
                {self.proc_data_dict["xlabel"]: x, self.proc_data_dict["ylabel"]: y}
            )
        self.proc_data_dict["optimal_pars_values"] = optimal_pars_values

        self.proc_data_dict["optimal_pars_units"] = {
            self.proc_data_dict["xlabel"]: self.proc_data_dict["xunit"],
            self.proc_data_dict["ylabel"]: self.proc_data_dict["yunit"],
        }

        optimal_measured_values = []
        optimal_measured_units = []
        mvs = self.proc_data_dict[extract_optimals_from]
        for optimal_idx in optimal_idxs:
            optimal_measured_values.append(
                {name: np.ravel(mvs[ii])[optimal_idx] for ii, name in enumerate(vln)}
            )
        optimal_measured_units = {name: vlu[ii] for ii, name in enumerate(vln)}
        self.proc_data_dict["optimal_measured_values"] = optimal_measured_values
        self.proc_data_dict["optimal_measured_units"] = optimal_measured_units

        if self.gen_optima_hulls:
            self._proc_hulls()

        if len(self.save_cond_phase_contours):
            self._proc_cond_phase_contours(angle_thr=0.3)
            self._proc_mv_along_contours()
            if self.gen_optima_hulls:
                self._proc_mv_along_contours_in_hulls()

        # Save quantities of interest
        save_these = {
            "optimal_pars_values",
            "optimal_pars_units",
            "optimal_measured_values",
            "optimal_measured_units",
            "clusters_pnts_y",
            "clusters_pnts_x",
            "clusters_pnts_colors",
            "hull_vertices",
            "cond_phase_contours",
            "cond_phase_contours_simplified",
        }
        pdd = self.proc_data_dict
        quantities_of_interest = dict()
        for save_this in save_these:
            if save_this in pdd.keys():
                if pdd[save_this] is not None:
                    quantities_of_interest[save_this] = pdd[save_this]
        if bool(quantities_of_interest):
            self.proc_data_dict["quantities_of_interest"] = quantities_of_interest

    def _proc_hulls(self):
        # Must be at the end of the main process_data

        vln = self.proc_data_dict["value_names"]

        interp_vals = self.proc_data_dict["interpolated_values"]

        # where = [(name in self.cost_func_Names) for name in vln]
        # cost_func_indxs = np.where(where)[0][0]
        # cost_func = interp_vals[cost_func_indxs]
        # cost_func = interp_to_1D_arr(z_int=cost_func)

        where = [(name in self.cond_phase_names) for name in vln]
        cond_phase_indx = np.where(where)[0][0]
        cond_phase_arr = interp_vals[cond_phase_indx]
        cond_phase_arr = interp_to_1D_arr(z_int=cond_phase_arr)

        # Avoid runtime errors
        cond_phase_arr[np.isnan(cond_phase_arr)] = 359.0

        where = [(name in self.L1_names) for name in vln]
        L1_indx = np.where(where)[0][0]
        L1_arr = interp_vals[L1_indx]
        L1_arr = interp_to_1D_arr(z_int=L1_arr)

        # Avoid runtime errors
        L1_arr[np.isnan(L1_arr)] = 100

        x_int = self.proc_data_dict["x_int"]
        y_int = self.proc_data_dict["y_int"]

        x_int_reshaped, y_int_reshaped = interp_to_1D_arr(x_int=x_int, y_int=y_int)

        sorted_hull_vertices = generate_optima_hull_vertices(
            x_arr=x_int_reshaped,
            y_arr=y_int_reshaped,
            L1_arr=L1_arr,
            cond_phase_arr=cond_phase_arr,
            target_phase=self.target_cond_phase,
            clustering_thr=self.hull_clustering_thr,
            phase_thr=self.hull_phase_thr,
            L1_thr=self.hull_L1_thr,
        )

        # We save this as a dictionary so that we don't have hdf5 issues
        self.proc_data_dict["hull_vertices"] = {
            str(h_i): hull_vertices
            for h_i, hull_vertices in enumerate(sorted_hull_vertices)
        }
        log.debug("Hulls are sorted by increasing y value.")

    def _proc_cond_phase_contours(self, angle_thr: float = 0.5):
        """
        Increasing `angle_thr` will make the contours' paths more coarse
        but more simple
        """
        # get the interpolated cond. phase data (if any)
        vln = self.proc_data_dict["value_names"]
        interp_vals = self.proc_data_dict["interpolated_values"]
        x_int = self.proc_data_dict["x_int"]
        y_int = self.proc_data_dict["y_int"]

        where = [(name in self.cond_phase_names) for name in vln]
        cond_phase_indx = np.where(where)[0][0]
        cond_phase_int = interp_vals[cond_phase_indx]

        c_dict = OrderedDict()
        c_dict_orig = OrderedDict()

        if len(cond_phase_int):
            # use the contours function to generate them
            levels_list = self.save_cond_phase_contours
            contours = contour_overlay(
                x_int,
                y_int,
                cond_phase_int,
                contour_levels=levels_list,
                cyclic_data=True,
                vlim=(0, 360),
                return_contours_only=True
            )
            for i, level in enumerate(levels_list):
                # Just saving in more friendly format
                # Each entry in the `c_dict` is a dict of 2D arrays for
                # disjoint contours for the same `level`
                same_level_dict = OrderedDict()
                same_level_dict_orig = OrderedDict()
                for j, c in enumerate(contours[i]):
                    # To save in hdf5 several unpredictably shaped np.arrays
                    # we need a dictionary format here

                    # By convention we will make the contours start left to
                    # right on the x axis
                    if c[0][0] > c[-1][0]:
                        c = np.flip(c, axis=0)
                    same_level_dict_orig[str(j)] = c
                    same_level_dict[str(j)] = c2d.simplify_2D_path(c, angle_thr)

                c_dict[str(level)] = same_level_dict
                c_dict_orig[str(level)] = same_level_dict_orig

        else:
            log.debug("Conditional phase data for contours not found.")

        self.proc_data_dict["cond_phase_contours_simplified"] = c_dict
        self.proc_data_dict["cond_phase_contours"] = c_dict_orig

    def _proc_mv_along_contours(self):
        interpolators = self.proc_data_dict["interpolators"]
        self.proc_data_dict["measured_values_along_contours"] = []
        mvac = self.proc_data_dict["measured_values_along_contours"]
        cpc = self.proc_data_dict["cond_phase_contours"]

        for interp in interpolators:
            mv_levels_dict = OrderedDict()
            for level, cntrs_dict in cpc.items():
                mv_cntrs_dict = OrderedDict()
                for cntr_id, pnts in cntrs_dict.items():
                    scaled_pnts = interp.scale(pnts)
                    mv_cntrs_dict[cntr_id] = interp(*scaled_pnts.T)

                mv_levels_dict[level] = mv_cntrs_dict

            mvac.append(mv_levels_dict)

    def _proc_mv_along_contours_in_hulls(self):
        self.proc_data_dict["measured_values_along_contours_in_hulls"] = []

        hvs = self.proc_data_dict["hull_vertices"]
        mvach = self.proc_data_dict["measured_values_along_contours_in_hulls"]
        cpc = self.proc_data_dict["cond_phase_contours"]

        for i, mvac in enumerate(self.proc_data_dict["measured_values_along_contours"]):
            hulls_dict = OrderedDict()
            for hull_id, hv in hvs.items():
                mv_levels_dict = OrderedDict()
                for level, cntrs_dict in cpc.items():
                    mv_cntrs_dict = OrderedDict()
                    for cntr_id, pnts in cntrs_dict.items():
                        where = np.where(c2d.in_hull(pnts, hv))
                        # The empty entries are kept in here so that the color
                        # matching between plots can be achieved
                        mv_cntrs_dict[cntr_id] = {
                            "pnts": pnts[where],
                            "vals": mvac[level][cntr_id][where],
                        }
                    mv_levels_dict[level] = mv_cntrs_dict

                hulls_dict[hull_id] = mv_levels_dict

            mvach.append(hulls_dict)

    def plot_text(self, pdict, axs):
        """
        Helper function that adds text to a plot
        Overriding here in order to make the text bigger
        and put it below the the cost function figure
        """
        pfunc = getattr(axs, pdict.get("func", "text"))
        plot_text_string = pdict["text_string"]
        plot_xpos = pdict.get("xpos", 0.98)
        plot_ypos = pdict.get("ypos", 0.98)
        fontsize = pdict.get("fontsize", 10)
        verticalalignment = pdict.get("verticalalignment", "top")
        horizontalalignment = pdict.get("horizontalalignment", "left")
        fontdict = {
            "horizontalalignment": horizontalalignment,
            "verticalalignment": verticalalignment,
        }

        if fontsize is not None:
            fontdict["fontsize"] = fontsize

        # fancy box props is based on the matplotlib legend
        box_props = pdict.get("box_props", "fancy")
        if box_props == "fancy":
            box_props = self.fancy_box_props

        # pfunc is expected to be ax.text
        pfunc(
            x=plot_xpos,
            y=plot_ypos,
            s=plot_text_string,
            transform=axs.transAxes,
            bbox=box_props,
            fontdict=fontdict,
        )

    def get_readable_optimals(
        self,
        optimal_pars_values=None,
        optimal_measured_values=None,
        optimal_start: int = 0,
        optimal_end: int = np.inf,
        sig_digits: int = 4,
        opt_are_interp=None,
    ):
        if not optimal_pars_values:
            optimal_pars_values = self.proc_data_dict["optimal_pars_values"]
        if not optimal_measured_values:
            optimal_measured_values = self.proc_data_dict["optimal_measured_values"]
        if opt_are_interp is None:
            opt_are_interp = self._opt_are_interp

        optimals_max = len(optimal_pars_values)

        string = ""
        for opt_idx in range(optimal_start, int(min(optimal_end + 1, optimals_max))):
            string += "========================\n"
            string += "Optimal #{}\n".format(opt_idx)
            string += "========================\n"
            for pv_name, pv_value in optimal_pars_values[opt_idx].items():
                string += "{} = {:.{sig_digits}g} {}\n".format(
                    pv_name,
                    pv_value,
                    self.proc_data_dict["optimal_pars_units"][pv_name],
                    sig_digits=sig_digits,
                )
            string += "------------\n"
            if (
                self.cluster_from_interp
                and opt_are_interp
                and optimal_pars_values is self.proc_data_dict["optimal_pars_values"]
            ):
                string += "[!!! Interpolated values !!!]\n"
            for mv_name, mv_value in optimal_measured_values[opt_idx].items():
                string += "{} = {:.{sig_digits}g} {}\n".format(
                    mv_name,
                    mv_value,
                    self.proc_data_dict["optimal_measured_units"][mv_name],
                    sig_digits=sig_digits,
                )
        return string


def get_optimal_pnts_indxs(
    theta_f_arr,
    lambda_2_arr,
    cond_phase_arr,
    L1_arr,
    cost_arr,
    target_phase=180,
    phase_thr=5,
    L1_thr=0.3,
    clustering_thr=10 / 360,
    tolerances=[1, 2, 3],
    sort_by_mode="cost",
):
    """
    target_phase and low L1 need to match roughtly cost function's minima

    Args:
    target_phase: unit = deg
    phase_thr: unit = deg, only points with cond phase below this threshold
        will be used for clustering

    L1_thr: unit = %, only points with leakage below this threshold
        will be used for clustering

    clustering_thr: unit = deg, represents distance between points on the
        landscape (lambda_2 gets normalized to [0, 360])

    tolerances: phase_thr and L1_thr will be multiplied by the values in
    this list successively if no points are found for the first element
    in the list
    """
    x = np.array(theta_f_arr)
    y = np.array(lambda_2_arr)

    # Normalize distance
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    # Select points based on low leakage and on how close to the
    # target_phase they are
    for tol in tolerances:
        phase_thr *= tol
        L1_thr *= tol
        cond_phase_dev_f = multi_targets_phase_offset(target_phase, 2 * target_phase)
        # np.abs(cond_phase_arr - target_phase)
        cond_phase_abs_diff = cond_phase_dev_f(cond_phase_arr)
        sel = cond_phase_abs_diff <= phase_thr
        sel = sel * (L1_arr <= L1_thr)
        # sel = sel * (x_norm > y_norm)

        # Exclude point on the boundaries of the entire landscape
        # This is because of some interpolation problems
        sel = (
            sel * (x < np.max(x)) * (x > np.min(x)) * (y < np.max(y)) * (y > np.min(y))
        )
        selected_points_indxs = np.where(sel)[0]
        if np.size(selected_points_indxs) == 0:
            log.warning(
                "No optimal points found with  |target_phase - cond phase| < {} and L1 < {}.".format(
                    phase_thr, L1_thr
                )
            )
            if tol == tolerances[-1]:
                log.warning("No optima found giving up.")
                return np.array([], dtype=int), np.array([], dtype=int)
            log.warning(
                "Increasing tolerance for phase_thr and L1 to x{}.".format(tol + 1)
            )
        elif np.size(selected_points_indxs) == 1:
            return np.array(selected_points_indxs), np.array([selected_points_indxs])
        else:
            x_filt = x_norm[selected_points_indxs]
            y_filt = y_norm[selected_points_indxs]
            break

    # Cluster points based on distance
    x_y_filt = np.transpose([x_filt, y_filt])
    clusters = hcluster.fclusterdata(x_y_filt, clustering_thr, criterion="distance")

    # Sorting the clusters
    cluster_id_min = np.min(clusters)
    cluster_id_max = np.max(clusters)
    clusters_by_indx = []
    optimal_idxs = []
    av_L1 = []
    # av_cp_diff = []
    # neighbors_num = []
    if sort_by_mode == "cost":
        # Iterate over all clusters and calculate the metrics we want
        for cluster_id in range(cluster_id_min, cluster_id_max + 1):

            cluster_indxs = np.where(clusters == cluster_id)
            indxs_in_orig_array = selected_points_indxs[cluster_indxs]

            min_cost_idx = np.argmin(cost_arr[indxs_in_orig_array])
            optimal_idx = indxs_in_orig_array[min_cost_idx]

            optimal_idxs.append(optimal_idx)
            clusters_by_indx.append(indxs_in_orig_array)

        # Low cost function is considered the most interesting optimum
        sort_by = cost_arr[optimal_idxs]

        if np.any(np.array(sort_by) != np.sort(sort_by)):
            log.debug(" Optimal points rescored based on cost function.")

    elif sort_by_mode == "L1_av_around":
        # Iterate over all clusters and calculate the metrics we want
        for cluster_id in range(cluster_id_min, cluster_id_max + 1):

            cluster_indxs = np.where(clusters == cluster_id)
            indxs_in_orig_array = selected_points_indxs[cluster_indxs]
            L1_av_around = [
                av_around(x_norm, y_norm, L1_arr, idx, clustering_thr * 1.5)[0]
                for idx in indxs_in_orig_array
            ]
            min_idx = np.argmin(L1_av_around)

            optimal_idx = indxs_in_orig_array[min_idx]
            optimal_idxs.append(optimal_idx)

            clusters_by_indx.append(indxs_in_orig_array)

            # sq_dist = (x_norm - x_norm[optimal_idx])**2 + (y_norm - y_norm[optimal_idx])**2
            # neighbors_indx = np.where(sq_dist <= (clustering_thr * 1.5)**2)
            # neighbors_num.append(np.size(neighbors_indx))
            # av_cp_diff.append(np.average(cond_phase_abs_diff[neighbors_indx]))
            # av_L1.append(np.average(L1_arr[neighbors_indx]))

            av_L1.append(L1_av_around[min_idx])

        # Here I tried different strategies for scoring the local optima
        # For landscapes that didn't look very regular

        # low leakage is best
        w1 = (
            np.array(av_L1)
            / np.max(av_L1)
            /  # normalize to maximum leakage
            # and consider bigger clusters more interesting
            np.array([it for it in map(np.size, clusters_by_indx)])
        )

        # value more the points with more neighbors as a confirmation of
        # low leakage area and also scores less points near the boundaries
        # of the sampling area
        # w2 = (1 - np.flip(np.array(neighbors_num) / np.max(neighbors_num)))

        # Very few points will actually be precisely on the target phase contour
        # Therefore not used
        # low phase diff is best
        # w3 = np.array(av_cp_diff) / np.max(av_cp_diff)

        sort_by = w1  # + w2 + w3

        if np.any(np.array(sort_by) != np.sort(sort_by)):
            log.debug(" Optimal points rescored based on low leakage areas.")

    optimal_idxs = np.array(optimal_idxs)[np.argsort(sort_by)]
    clusters_by_indx = np.array(clusters_by_indx)[np.argsort(sort_by)]

    return optimal_idxs, clusters_by_indx


def generate_optima_hull_vertices(
    x_arr,
    y_arr,
    cond_phase_arr,
    L1_arr,
    target_phase=180,
    phase_thr=5,
    L1_thr=np.inf,
    clustering_thr=0.1,
    tolerances=[1, 2, 3],
):
    """
    WARNING: docstring

    Args:
    target_phase: unit = deg
    phase_thr: unit = deg, only points with cond phase below this threshold
        will be used for clustering

    L1_thr: unit = %, only points with leakage below this threshold
        will be used for clustering

    clustering_thr: unit = deg, represents distance between points on the
        landscape (lambda_2 gets normalized to [0, 360])

    tolerances: phase_thr and L1_thr will be multiplied by the values in
    this list successively if no points are found for the first element
    in the list
    """
    x = np.array(x_arr)
    y = np.array(y_arr)

    # Normalize distance
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    # Select points based on low leakage and on how close to the
    # target_phase they are
    for tol in tolerances:
        phase_thr *= tol
        L1_thr *= tol
        cond_phase_dev_f = multi_targets_phase_offset(target_phase, 2 * target_phase)

        cond_phase_abs_diff = cond_phase_dev_f(cond_phase_arr)
        sel = cond_phase_abs_diff <= phase_thr
        sel = sel * (L1_arr <= L1_thr)

        selected_points_indxs = np.where(sel)[0]
        if np.size(selected_points_indxs) == 0:
            log.warning(
                "No optimal points found with  |target_phase - cond phase| < {} and L1 < {}.".format(
                    phase_thr, L1_thr
                )
            )
            if tol == tolerances[-1]:
                log.warning("No optima found giving up.")
                return []
            log.warning(
                "Increasing tolerance for phase_thr and L1 to x{}.".format(tol + 1)
            )
        else:
            x_filt = x_norm[selected_points_indxs]
            y_filt = y_norm[selected_points_indxs]
            break

    # Cluster points based on distance
    x_y_filt = np.transpose([x_filt, y_filt])
    clusters = hcluster.fclusterdata(x_y_filt, clustering_thr, criterion="distance")

    # Sorting the clusters
    cluster_id_min = np.min(clusters)
    cluster_id_max = np.max(clusters)
    clusters_by_indx = []
    sort_by_idx = []

    # Iterate over all clusters and calculate the metrics we want
    for cluster_id in range(cluster_id_min, cluster_id_max + 1):

        cluster_indxs = np.where(clusters == cluster_id)
        indxs_in_orig_array = selected_points_indxs[cluster_indxs]
        clusters_by_indx.append(indxs_in_orig_array)

        min_sort_idx = np.argmin(y[indxs_in_orig_array])
        optimal_idx = indxs_in_orig_array[min_sort_idx]

        sort_by_idx.append(optimal_idx)

    # Low cost function is considered the most interesting optimum
    sort_by = y[sort_by_idx]

    if np.any(np.array(sort_by) != np.sort(sort_by)):
        log.debug(" Optimal points rescored.")

    # optimal_idxs = np.array(optimal_idxs)[np.argsort(sort_by)]
    clusters_by_indx = np.array(clusters_by_indx)[np.argsort(sort_by)]

    x_y = np.transpose([x, y])

    sorted_hull_vertices = []
    # Generate the list of vertices for each optimal hull
    for cluster_by_indx in clusters_by_indx:
        pnts_for_hull = x_y[cluster_by_indx]
        try:
            hull = ConvexHull(pnts_for_hull)
            vertices = hull.points[hull.vertices]
            angle_thr = 5.0
            # Remove redundant points that deviate little from a straight line
            simplified_hull = c2d.simplify_2D_path(vertices, angle_thr)
            sorted_hull_vertices.append(simplified_hull)
        except Exception as e:
            # There might not be enough points for a hull
            log.debug(e)

    return sorted_hull_vertices


def av_around(x, y, z, idx, radius):
    sq_dist = (x - x[idx]) ** 2 + (y - y[idx]) ** 2
    neighbors_indx = np.where(sq_dist <= radius ** 2)
    return np.average(z[neighbors_indx]), neighbors_indx


def interp_to_1D_arr(x_int=None, y_int=None, z_int=None, slice_above_len=None):
    """
    Turns interpolated heatmaps into linear 1D array
    Intended for data reshaping for get_optimal_pnts_indxs
    """
    if slice_above_len is not None:
        if x_int is not None:
            size = np.size(x_int)
            slice_step = np.int(np.ceil(size / slice_above_len))
            x_int = np.array(x_int)[::slice_step]
        if y_int is not None:
            size = np.size(y_int)
            slice_step = np.int(np.ceil(size / slice_above_len))
            y_int = np.array(y_int)[::slice_step]
        if z_int is not None:
            size_0 = np.shape(z_int)[0]
            size_1 = np.shape(z_int)[1]
            slice_step_0 = np.int(np.ceil(size_0 / slice_above_len))
            slice_step_1 = np.int(np.ceil(size_1 / slice_above_len))
            z_int = np.array(z_int)[::slice_step_0, ::slice_step_1]

    if x_int is not None and y_int is not None and z_int is not None:
        x_int_1D = np.ravel(np.repeat([x_int], np.size(y_int), axis=0))
        y_int_1D = np.ravel(np.repeat([y_int], np.size(x_int), axis=1))
        z_int_1D = np.ravel(z_int)
        return x_int_1D, y_int_1D, z_int
    elif z_int is not None:
        z_int_1D = np.ravel(z_int)
        return z_int_1D
    elif x_int is not None and y_int is not None:
        x_int_1D = np.ravel(np.repeat([x_int], np.size(y_int), axis=0))
        y_int_1D = np.ravel(np.repeat([y_int], np.size(x_int), axis=1))
        return x_int_1D, y_int_1D
    else:
        return None
