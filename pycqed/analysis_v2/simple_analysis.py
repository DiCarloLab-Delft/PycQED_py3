"""
This file contains examples for the most basic/simplest of analyses.
They only do plotting of the data and can be used as sort of a template
when making more complex analyses.

We distinguish 3 cases (for the most trivial analyses)
- 1D (single or multi-file)
- 2D_single_file
- 2D_multi_file (which inherits from 1D single file)

"""
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.tools.plot_interpolation import interpolate_heatmap
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.analysis_toolbox import color_plot
from pycqed.analysis_v2.tools.plotting import scatter_pnts_overlay
from scipy.stats import sem


class Basic1DAnalysis(ba.BaseDataAnalysis):
    """
    Basic 1D analysis.

    Creates a line plot for every parameter measured in a set of datafiles.
    Creates a single plot for each parameter measured.

    Supported options_dict keys
        x2 (str)  : name of a parameter that is varied if multiple datasets
                    are combined.
        average_sets (bool)  : if True averages all datasets together.
            requires shapes of the different datasets to be the same.
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        do_fitting: bool = True,
        close_figs: bool = True,
        auto: bool = True,
        hide_lines: bool = False,
        hide_pnts: bool = False,
        plt_sorted_x: bool = True,
        legend_labels: list = None
    ):
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
            close_figs=close_figs,
        )
        # self.single_timestamp = False
        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "xvals": "sweep_points",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }

        # x2 is whatever parameter is varied between sweeps
        self.numeric_params = []
        x2 = self.options_dict.get("x2", None)
        if x2 is not None:
            self.params_dict["x2"] = x2
            self.numeric_params = ["x2"]

        # Adaptive measurements need sorting to avoid messy line plotting
        self.plt_sorted_x = plt_sorted_x

        # In case you only want one of them
        self.hide_pnts = hide_pnts
        self.hide_lines = hide_lines

        # Set specific legend label when specifying  `t_start` and `t_stop`
        self.legend_labels = legend_labels

        if auto:
            self.run_analysis()

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        labels = self.legend_labels if self.legend_labels is not None else self.timestamps
        setlabel = self.raw_data_dict.get("x2", labels)
        if "x2" in self.options_dict.keys():
            legend_title = self.options_dict.get("x2_label", self.options_dict["x2"])
        else:
            legend_title = "timestamp" if self.legend_labels is None else ""

        for i, val_name in enumerate(self.raw_data_dict["value_names"][0]):

            yvals = self.raw_data_dict["measured_values_ord_dict"][val_name]

            if self.options_dict.get("average_sets", False):
                xvals = self.raw_data_dict["xvals"][0]
                yvals = np.mean(yvals, axis=0)
                setlabel = ["Averaged data"]
            else:
                xvals = self.raw_data_dict["xvals"]

            if (len(np.shape(yvals)) == 1) or (np.shape(yvals)[0] == 1):
                do_legend = False
            else:
                do_legend = True

            if (len(np.shape(yvals)) == 1):
                # Keep the data shaping to avoid non-geral constructions
                # in the plotting below
                xvals = [xvals]
                yvals = [yvals]

            # Sort points, necessary for adaptive sampling
            arg_sort = np.argsort(xvals)

            if not self.hide_lines:
                self.plot_dicts[val_name + "_line"] = {
                    "ax_id": val_name,
                    "plotfn": self.plot_line,
                    "xvals": [xval_i[argsort_i] for xval_i, argsort_i in zip(xvals, arg_sort)],
                    "xlabel": self.raw_data_dict["xlabel"][0],
                    "xunit": self.raw_data_dict["xunit"][0][0],
                    "yvals": [yval_i[argsort_i] for yval_i, argsort_i in zip(yvals, arg_sort)],
                    "ylabel": val_name,
                    "yrange": self.options_dict.get("yrange", None),
                    "xrange": self.options_dict.get("xrange", None),
                    "yunit": self.raw_data_dict["value_units"][0][i],
                    "setlabel": setlabel,
                    "legend_title": legend_title,
                    "title": (
                        self.raw_data_dict["timestamps"][0]
                        + " - "
                        + self.raw_data_dict["timestamps"][-1]
                        + "\n"
                        + self.raw_data_dict["measurementstring"][0]
                    ),
                    "do_legend": do_legend,
                    "legend_pos": "best",
                    "marker": "",  # don't use markers
                    "linestyle": "-"
                }

            if not self.hide_pnts:
                self.plot_dicts[val_name + "_scatter"] = {
                    "ax_id": val_name,
                    "plotfn": scatter_pnts_overlay,
                    "x": xvals,
                    "y": yvals,
                    "color": None,
                    "edgecolors": "black",
                    "marker": "o",
                }
                if self.plt_sorted_x:
                    # For adaptive sampling it is useful to know the sampling
                    # order
                    self.plot_dicts[val_name + "_scatter"]["c"] = (
                        [range(len(xval)) for xval in xvals]
                    )
                    self.plot_dicts[val_name + "_scatter"]["cmap"] = (
                        "plasma"
                    )


class Basic1DBinnedAnalysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = True,
        auto=True,
    ):
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            options_dict=options_dict,
            extract_only=extract_only,
            close_figs=False,
            do_fitting=do_fitting,
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

        self.raw_data_dict["xvals"] = a.sweep_points
        self.raw_data_dict["xlabel"] = a.parameter_names[0]
        self.raw_data_dict["xunit"] = a.parameter_units[0]

        self.raw_data_dict["bins"] = a.data_file["Experimental Data"][
            "Experimental Metadata"
        ]["bins"].value
        self.raw_data_dict["measured_values"] = a.measured_values
        self.raw_data_dict["value_names"] = a.value_names
        self.raw_data_dict["value_units"] = a.value_units
        self.raw_data_dict["measurementstring"] = a.measurementstring
        self.raw_data_dict["folder"] = a.folder
        a.finish()

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        bins = self.proc_data_dict["bins"]

        self.proc_data_dict["binned_values"] = []
        self.proc_data_dict["binned_values_stderr"] = []
        for i, y in enumerate(self.proc_data_dict["measured_values"]):
            if len(y) % len(bins) != 0:
                missing_vals = missing_vals = int(len(bins) - len(y) % len(bins))
                y_ext = np.concatenate([y, np.ones(missing_vals) * np.nan])
            else:
                y_ext = y

            y_binned = np.nanmean(y_ext.reshape((len(bins), -1), order="F"), axis=1)
            y_binned_stderr = sem(
                y_ext.reshape((len(bins), -1), order="F"), axis=1, nan_policy="omit"
            )
            self.proc_data_dict["binned_values"].append(y_binned)
            self.proc_data_dict["binned_values_stderr"].append(y_binned_stderr)

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        # pass
        for i, val_name in enumerate(self.raw_data_dict["value_names"]):

            self.plot_dicts["binned_{}".format(val_name)] = {
                "plotfn": "plot_errorbar",
                "xlabel": self.proc_data_dict["xlabel"],
                "xunit": self.proc_data_dict["xunit"],
                "ylabel": self.proc_data_dict["value_names"][i],
                "yunit": self.proc_data_dict["value_units"][i],
                "x": self.proc_data_dict["bins"],
                "y": self.proc_data_dict["binned_values"][i],
                "yerr": self.proc_data_dict["binned_values_stderr"][i],
                "marker": "o",
                "title": "{}\nBinned {}".format(self.timestamp, val_name),
            }


class Basic2DAnalysis(Basic1DAnalysis):
    """
    Extracts a 2D dataset from a set of 1D scans and plots the data.

    Special options dict kwargs
        "x2"  specifies the name of the parameter varied between the different
              linescans.
    """

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()
        for i, val_name in enumerate(self.raw_data_dict["value_names"][0]):

            if not self.hide_lines:
                # Use same color scale for 1D curves
                self.plot_dicts[val_name + "_line"]["cmap"] = "viridis"

            if "x2" in self.raw_data_dict.keys():
                xvals = self.raw_data_dict["x2"]
                x2 = self.options_dict["x2"]
                xlabel = self.options_dict.get("x2_label", x2)
                xunit = self.options_dict.get("x2_unit", "")
            else:
                xvals = np.arange(len(self.raw_data_dict["xvals"]))
                xlabel = "Experiment idx"
                xunit = ""

            self.plot_dicts[val_name + "_heatmap"] = {
                "plotfn": self.plot_colorx,
                "xvals": xvals,
                "xlabel": xlabel,
                "xunit": xunit,
                "yvals": self.raw_data_dict["xvals"],
                "ylabel": self.raw_data_dict["xlabel"][0],
                "yunit": self.raw_data_dict["xunit"][0][0],
                "zvals": self.raw_data_dict["measured_values_ord_dict"][val_name],
                "zlabel": val_name,
                "zunit": self.raw_data_dict["value_units"][0][i],
                "cmap": "viridis",
                "title": (
                    self.raw_data_dict["timestamps"][0]
                    + " - "
                    + self.raw_data_dict["timestamps"][-1]
                    + "\n"
                    + self.raw_data_dict["measurementstring"][0]
                ),
                "do_legend": True,
                "legend_pos": "upper right",
            }


class Basic2DInterpolatedAnalysis(ba.BaseDataAnalysis):
    """
    Basic 2D analysis, produces interpolated heatmaps for all measured
    quantities.
    This is intended to be a fully featured class to create fancy figures.
    If you want special options, implement a dedicated class.
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
        auto: bool = True,
        interp_method="linear",
        save_qois: bool = True,
        plt_orig_pnts: bool = True
    ):
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

        self.plt_orig_pnts = plt_orig_pnts
        self.interp_method = interp_method

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

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        self.proc_data_dict["interpolated_values"] = []
        for i in range(len(self.proc_data_dict["value_names"])):
            x_int, y_int, z_int = interpolate_heatmap(
                self.proc_data_dict["x"],
                self.proc_data_dict["y"],
                self.proc_data_dict["measured_values"][i],
                interp_method=self.interp_method,
            )
            self.proc_data_dict["interpolated_values"].append(z_int)
        self.proc_data_dict["x_int"] = x_int
        self.proc_data_dict["y_int"] = y_int

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()
        for i, val_name in enumerate(self.proc_data_dict["value_names"]):

            zlabel = "{} ({})".format(val_name, self.proc_data_dict["value_units"][i])
            self.plot_dicts[val_name] = {
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
                self.plot_dicts[val_name + "_measured"] = {
                    "ax_id": val_name,
                    "plotfn": scatter_pnts_overlay,
                    "x": self.proc_data_dict["x"],
                    "y": self.proc_data_dict["y"],
                    "setlabel": "Raw data"
                }
