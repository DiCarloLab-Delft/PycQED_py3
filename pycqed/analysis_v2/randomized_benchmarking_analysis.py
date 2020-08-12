import lmfit
from uncertainties import ufloat
import pandas as pd
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
import logging
from scipy.stats import sem
from pycqed.analysis.tools.data_manipulation import populations_using_rate_equations
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, plot_fit
from pycqed.utilities.general import format_value_string
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm
from sklearn import linear_model
from matplotlib import colors as c
from pycqed.analysis_v2.tools import geometry_utils as geo

log = logging.getLogger(__name__)


class RandomizedBenchmarking_SingleQubit_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label="",
        options_dict: dict = None,
        auto=True,
        close_figs=True,
        classification_method="rates",
        rates_I_quad_ch_idx: int = 0,
        rates_Q_quad_ch_idx: int = None,
        rates_ch_idx=None,  # Deprecated
        cal_pnts_in_dset: list = np.repeat(["0", "1", "2"], 2),
        ignore_f_cal_pts: bool = False,
        do_fitting: bool = True,
        **kwargs
    ):
        """
        Analysis for single qubit randomized benchmarking.

        For basic options see docstring of BaseDataAnalysis

        Args:
            classification_method ["rates", ]   sets method to determine
                populations of g,e and f states. Currently only supports "rates"
                    rates: uses calibration points and rate equation from
                        Asaad et al. to determine populations
            rates_I_quad_ch_idx (int) : sets the I quadrature channel from which
                to use the data for the rate equations,
                `rates_I_quad_ch_idx + 1` is assumed to be the Q quadrature,
                both quadratures are used in the rate equation,
                this analysis expects the RO mode to be "optimal IQ"
            ignore_f_cal_pts (bool) : if True, ignores the f-state calibration
                points and instead makes the approximation that the f-state
                looks the same as the e-state in readout. This is useful when
                the ef-pulse is not calibrated.
        """
        if options_dict is None:
            options_dict = dict()
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=close_figs,
            do_fitting=do_fitting,
            **kwargs
        )
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        # [2020-07-09 Victor] RB has been used with the "optimal IQ" RO mode
        # for a while in the lab, both quadratures are necessary for plotting
        # and correct calculation using the rates equation
        if rates_ch_idx is not None:
            log.warning(
                "`rates_ch_idx` is deprecated `rates_I_quad_ch_idx` "
                + "and `rates_I_quad_ch_idx + 1` are used for population "
                + "rates calculation! Please apply changes to `pycqed`."
            )
        self.rates_I_quad_ch_idx = rates_I_quad_ch_idx
        self.rates_Q_quad_ch_idx = rates_Q_quad_ch_idx
        if self.rates_Q_quad_ch_idx is None:
            self.rates_Q_quad_ch_idx = rates_I_quad_ch_idx + 1
        self.d1 = 2
        self.cal_pnts_in_dset = np.array(cal_pnts_in_dset)
        self.ignore_f_cal_pts = ignore_f_cal_pts

        # Allows to run this analysis for different qubits in same dataset
        self.overwrite_qois = False
        if auto:
            self.run_analysis()

        # NB all the fit_res, plot_dicts, qois are appended the `value_name`
        # corresponding to `rates_I_quad_ch_idx` so that this analysis can be
        # run several times targeting a different measured qubit

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop, label=self.labels
        )

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False
        )
        a.get_naming_and_values()

        if "bins" in a.data_file["Experimental Data"]["Experimental Metadata"].keys():
            bins = a.data_file["Experimental Data"]["Experimental Metadata"]["bins"][()]

            num_cal_pnts = len(self.cal_pnts_in_dset)

            self.raw_data_dict["ncl"] = bins[:-num_cal_pnts:2]
            self.raw_data_dict["bins"] = bins

            self.raw_data_dict["value_names"] = a.value_names
            self.raw_data_dict["value_units"] = a.value_units
            self.raw_data_dict["measurementstring"] = a.measurementstring
            self.raw_data_dict["timestamp_string"] = a.timestamp_string

            self.raw_data_dict["binned_vals"] = OrderedDict()
            self.raw_data_dict["cal_pts_zero"] = OrderedDict()
            self.raw_data_dict["cal_pts_one"] = OrderedDict()
            self.raw_data_dict["cal_pts_two"] = OrderedDict()
            self.raw_data_dict["measured_values_I"] = OrderedDict()
            self.raw_data_dict["measured_values_X"] = OrderedDict()

            # [2020-07-08 Victor] don't know why is this here, seems like
            # a nasty hack... will keep it to avoid braking some more stuff...
            selection = a.measured_values[0] == 0
            for i in range(1, len(a.measured_values)):
                selection &= a.measured_values[i] == 0
            invalid_idxs = np.where(selection)[0]

            if len(invalid_idxs):
                log.warning(
                    "Found zero values at {} indices!".format(len(invalid_idxs))
                )
                log.warning(invalid_idxs[:10])
                a.measured_values[:, invalid_idxs] = np.array(
                    [[np.nan] * len(invalid_idxs)] * len(a.value_names)
                )

            zero_idxs = np.where(self.cal_pnts_in_dset == "0")[0] - num_cal_pnts
            one_idxs = np.where(self.cal_pnts_in_dset == "1")[0] - num_cal_pnts
            two_idxs = np.where(self.cal_pnts_in_dset == "2")[0] - num_cal_pnts

            for i, val_name in enumerate(a.value_names):

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order="F"
                )

                self.raw_data_dict["binned_vals"][val_name] = binned_yvals

                vlns = a.value_names
                if val_name in (
                    vlns[self.rates_I_quad_ch_idx],
                    vlns[self.rates_Q_quad_ch_idx],
                ):
                    self.raw_data_dict["cal_pts_zero"][val_name] = binned_yvals[
                        zero_idxs, :
                    ].flatten()
                    self.raw_data_dict["cal_pts_one"][val_name] = binned_yvals[
                        one_idxs, :
                    ].flatten()

                    if self.ignore_f_cal_pts:
                        self.raw_data_dict["cal_pts_two"][
                            val_name
                        ] = self.raw_data_dict["cal_pts_one"][val_name]
                    else:
                        self.raw_data_dict["cal_pts_two"][val_name] = binned_yvals[
                            two_idxs, :
                        ].flatten()

                    self.raw_data_dict["measured_values_I"][val_name] = binned_yvals[
                        :-num_cal_pnts:2, :
                    ]
                    self.raw_data_dict["measured_values_X"][val_name] = binned_yvals[
                        1:-num_cal_pnts:2, :
                    ]
        else:
            bins = None

        self.raw_data_dict["folder"] = a.folder
        self.raw_data_dict["timestamps"] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        rdd = self.raw_data_dict
        self.proc_data_dict = deepcopy(rdd)
        pdd = self.proc_data_dict
        for key in [
            "V0",
            "V1",
            "V2",
            "SI",
            "SI_corr",
            "SX",
            "SX_corr",
            "P0",
            "P1",
            "P2",
            "M_inv",
            "M0",
            "X1",
        ]:
            # Nesting dictionaries allows to generate all this quantities
            # for different qubits by just running the analysis several times
            # with different rates_I_quad_ch_idx and cal points
            pdd[key] = OrderedDict()

        val_name_I = rdd["value_names"][self.rates_I_quad_ch_idx]
        val_name_Q = rdd["value_names"][self.rates_Q_quad_ch_idx]

        V0_I = np.nanmean(rdd["cal_pts_zero"][val_name_I])
        V1_I = np.nanmean(rdd["cal_pts_one"][val_name_I])
        V2_I = np.nanmean(rdd["cal_pts_two"][val_name_I])

        V0_Q = np.nanmean(rdd["cal_pts_zero"][val_name_Q])
        V1_Q = np.nanmean(rdd["cal_pts_one"][val_name_Q])
        V2_Q = np.nanmean(rdd["cal_pts_two"][val_name_Q])

        pdd["V0"][val_name_I] = V0_I
        pdd["V1"][val_name_I] = V1_I
        pdd["V2"][val_name_I] = V2_I

        pdd["V0"][val_name_Q] = V0_Q
        pdd["V1"][val_name_Q] = V1_Q
        pdd["V2"][val_name_Q] = V2_Q

        SI_I = np.nanmean(rdd["measured_values_I"][val_name_I], axis=1)
        SX_I = np.nanmean(rdd["measured_values_X"][val_name_I], axis=1)
        SI_Q = np.nanmean(rdd["measured_values_I"][val_name_Q], axis=1)
        SX_Q = np.nanmean(rdd["measured_values_X"][val_name_Q], axis=1)

        pdd["SI"][val_name_I] = SI_I
        pdd["SX"][val_name_I] = SX_I
        pdd["SI"][val_name_Q] = SI_Q
        pdd["SX"][val_name_Q] = SX_Q

        cal_triangle = np.array([[V0_I, V0_Q], [V1_I, V1_Q], [V2_I, V2_Q]])
        pdd["cal_triangle"] = cal_triangle
        # [2020-07-11 Victor]
        # Here we correct for the cases when the measured points fall outside
        # the triangle of the calibration points, such a case breaks the
        # assumptions that S = V0 * P0 + V1 * P1 + V2 * P2

        SI_I_corr, SI_Q_corr = geo.constrain_to_triangle(cal_triangle, SI_I, SI_Q)
        SX_I_corr, SX_Q_corr = geo.constrain_to_triangle(cal_triangle, SX_I, SX_Q)

        pdd["SI_corr"][val_name_I] = SI_I_corr
        pdd["SX_corr"][val_name_I] = SX_I_corr
        pdd["SI_corr"][val_name_Q] = SI_Q_corr
        pdd["SX_corr"][val_name_Q] = SX_Q_corr

        P0, P1, P2, M_inv = populations_using_rate_equations(
            SI_I_corr + 1j * SI_Q_corr,
            SX_I_corr + 1j * SX_Q_corr,
            V0_I + 1j * V0_Q,
            V1_I + 1j * V1_Q,
            V2_I + 1j * V2_Q,
        )

        # There might be other qubits being measured at some point so we keep
        # the results with the I quadrature label
        pdd["P0"][val_name_I] = P0
        pdd["P1"][val_name_I] = P1
        pdd["P2"][val_name_I] = P2
        pdd["M_inv"][val_name_I] = M_inv

        # [2020-07-09 Victor] This is not being used for anything...
        # classifier = logisticreg_classifier_machinelearning(
        #     pdd["cal_pts_zero"],
        #     pdd["cal_pts_one"],
        #     pdd["cal_pts_two"],
        # )
        # pdd["classifier"] = classifier

        if self.classification_method == "rates":
            pdd["M0"][val_name_I] = P0
            pdd["X1"][val_name_I] = 1 - P2
        else:
            raise NotImplementedError()

    def run_fitting(self, fit_input_tag: str = None):
        """
        Args:
            fit_input_tag (str): allows to fit specific M0 and X1
                intended for use in 2Q RBs
        """
        super().run_fitting()
        rdd = self.raw_data_dict
        pdd = self.proc_data_dict

        if fit_input_tag is None:
            # Default value for single qubit RB analysis
            fit_input_tag = rdd["value_names"][self.rates_I_quad_ch_idx]

        leak_mod = lmfit.Model(leak_decay, independent_vars="m")
        leak_mod.set_param_hint("A", value=0.95, min=0, vary=True)
        leak_mod.set_param_hint("B", value=0.1, min=0, vary=True)

        leak_mod.set_param_hint("lambda_1", value=0.99, vary=True)
        leak_mod.set_param_hint("L1", expr="(1-A)*(1-lambda_1)")
        leak_mod.set_param_hint("L2", expr="A*(1-lambda_1)")

        leak_mod.set_param_hint("L1_cz", expr="1-(1-(1-A)*(1-lambda_1))**(1/1.5)")
        leak_mod.set_param_hint("L2_cz", expr="1-(1-(A*(1-lambda_1)))**(1/1.5)")

        params = leak_mod.make_params()
        try:
            fit_res_leak = leak_mod.fit(
                data=pdd["X1"][fit_input_tag], m=pdd["ncl"], params=params,
            )
            self.fit_res["leakage_decay_" + fit_input_tag] = fit_res_leak
            lambda_1 = fit_res_leak.best_values["lambda_1"]
            L1 = fit_res_leak.params["L1"].value
        except Exception as e:
            log.warning("Fitting {} failed!".format("leakage_decay"))
            log.warning(e)
            lambda_1 = 1
            L1 = 0
            self.fit_res["leakage_decay_" + fit_input_tag] = {}

        fit_res_rb = self.fit_rb_decay(
            fit_input_tag, lambda_1=lambda_1, L1=L1, simple=False
        )
        self.fit_res["rb_decay_" + fit_input_tag] = fit_res_rb
        fit_res_rb_simple = self.fit_rb_decay(
            fit_input_tag, lambda_1=1, L1=0, simple=True
        )
        self.fit_res["rb_decay_simple_" + fit_input_tag] = fit_res_rb_simple

        def safe_get_par_from_fit_result(fit_res, par_name):
            """
            Ensures an `lmfit.Parameter` is always returned even when the fit
            failed and an empty dict is provided
            """
            if fit_res:  # Check for empty dict
                params = fit_res.params
                par = params[par_name]
            else:
                par = lmfit.Parameter(par_name)
                par.value = np.NaN
                par.stderr = np.NaN

            return par

        fr_rb_dict = self.fit_res["rb_decay_" + fit_input_tag]
        eps = safe_get_par_from_fit_result(fr_rb_dict, "eps")

        fr_rb_simple_dict = self.fit_res["rb_decay_simple_" + fit_input_tag]
        eps_simple = safe_get_par_from_fit_result(fr_rb_simple_dict, "eps")

        fr_dec = self.fit_res["leakage_decay_" + fit_input_tag]
        L1 = safe_get_par_from_fit_result(fr_dec, "L1")
        L2 = safe_get_par_from_fit_result(fr_dec, "L2")

        text_msg = "Summary: \n"
        text_msg += format_value_string(
            r"$\epsilon_{{\mathrm{{simple}}}}$", eps_simple, "\n"
        )
        text_msg += format_value_string(r"$\epsilon_{{\chi_1}}$", eps, "\n")
        text_msg += format_value_string(r"$L_1$", L1, "\n")
        text_msg += format_value_string(r"$L_2$", L2, "\n")
        pdd["rb_msg_" + fit_input_tag] = text_msg

        pdd["quantities_of_interest"] = {}
        qoi = pdd["quantities_of_interest"]
        qoi["eps_simple_" + fit_input_tag] = ufloat(
            eps_simple.value, eps_simple.stderr or np.NaN
        )
        qoi["eps_X1_" + fit_input_tag] = ufloat(eps.value, eps.stderr or np.NaN)
        qoi["L1_" + fit_input_tag] = ufloat(L1.value, L1.stderr or np.NaN)
        qoi["L2_" + fit_input_tag] = ufloat(L2.value, L2.stderr or np.NaN)

    def fit_rb_decay(
        self, val_name: str, lambda_1: float, L1: float, simple: bool = False
    ):
        """
        Fits the data
        """
        pdd = self.proc_data_dict

        fit_mod_rb = lmfit.Model(full_rb_decay, independent_vars="m")
        fit_mod_rb.set_param_hint("A", value=0.5, min=0, vary=True)
        if simple:
            fit_mod_rb.set_param_hint("B", value=0, vary=False)
        else:
            fit_mod_rb.set_param_hint("B", value=0.1, min=0, vary=True)
        fit_mod_rb.set_param_hint("C", value=0.4, min=0, max=1, vary=True)

        fit_mod_rb.set_param_hint("lambda_1", value=lambda_1, vary=False)
        fit_mod_rb.set_param_hint("lambda_2", value=0.95, vary=True)

        # d1 = dimensionality of computational subspace
        fit_mod_rb.set_param_hint("d1", value=self.d1, vary=False)
        fit_mod_rb.set_param_hint("L1", value=L1, vary=False)

        # Note that all derived quantities are expressed directly in
        fit_mod_rb.set_param_hint("F", expr="1/d1*((d1-1)*lambda_2+1-L1)", vary=True)
        fit_mod_rb.set_param_hint("eps", expr="1-(1/d1*((d1-1)*lambda_2+1-L1))")
        # Only valid for single qubit RB assumption equal error rates
        fit_mod_rb.set_param_hint(
            "F_g", expr="(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)"
        )
        fit_mod_rb.set_param_hint(
            "eps_g", expr="1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)"
        )
        # Only valid for two qubit RB assumption all error in CZ
        fit_mod_rb.set_param_hint("F_cz", expr="(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)")
        fit_mod_rb.set_param_hint(
            "eps_cz", expr="1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)"
        )

        params = fit_mod_rb.make_params()

        try:
            fit_res_rb = fit_mod_rb.fit(
                data=pdd["M0"][val_name], m=pdd["ncl"], params=params
            )
        except Exception as e:
            log.warning("Fitting failed!")
            log.warning(e)
            fit_res_rb = {}

        return fit_res_rb

    def prepare_plots(self, fit_input_tag: str = None):
        """
        Args:
            fit_input_tag (str): allows to fit specific M0 and X1
                intended for use in 2Q RBs
        """

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict

        if fit_input_tag is None:
            val_name_I = rdd["value_names"][self.rates_I_quad_ch_idx]
            fit_input_tag = val_name_I

            val_names = rdd["value_names"]
            for i, val_name in enumerate(val_names):
                self.plot_dicts["binned_data_{}".format(val_name)] = {
                    "plotfn": self.plot_line,
                    "xvals": rdd["bins"],
                    "yvals": np.nanmean(rdd["binned_vals"][val_name], axis=1),
                    "yerr": sem(rdd["binned_vals"][val_name], axis=1),
                    "xlabel": "Number of Cliffords",
                    "xunit": "#",
                    "ylabel": val_name,
                    "yunit": rdd["value_units"][i],
                    "title": rdd["timestamp_string"] + "\n" + rdd["measurementstring"],
                }

            fs = plt.rcParams["figure.figsize"]

            fig_id_hex = "cal_points_hexbin_{}".format(val_name_I)
            self.plot_dicts[fig_id_hex] = {
                "plotfn": plot_cal_points_hexbin,
                "shots_0": (
                    rdd["cal_pts_zero"][val_names[self.rates_I_quad_ch_idx]],
                    rdd["cal_pts_zero"][val_names[self.rates_Q_quad_ch_idx]],
                ),
                "shots_1": (
                    rdd["cal_pts_one"][val_names[self.rates_I_quad_ch_idx]],
                    rdd["cal_pts_one"][val_names[self.rates_Q_quad_ch_idx]],
                ),
                "shots_2": (
                    rdd["cal_pts_two"][val_names[self.rates_I_quad_ch_idx]],
                    rdd["cal_pts_two"][val_names[self.rates_Q_quad_ch_idx]],
                ),
                "xlabel": val_names[self.rates_I_quad_ch_idx],
                "xunit": rdd["value_units"][0],
                "ylabel": val_names[self.rates_Q_quad_ch_idx],
                "yunit": rdd["value_units"][1],
                "title": rdd["timestamp_string"]
                + "\n"
                + rdd["measurementstring"]
                + " hexbin plot",
                "plotsize": (fs[0] * 1.5, fs[1]),
            }

            num_cal_pnts = len(pdd["cal_triangle"])
            fig_id_RB_on_IQ = "rb_on_iq_{}".format(val_name_I)
            for ax_id in [fig_id_hex, fig_id_RB_on_IQ]:
                self.plot_dicts[ax_id + "_cal_pnts"] = {
                    "plotfn": self.plot_line,
                    "ax_id": ax_id,
                    "xvals": pdd["cal_triangle"].T[0].reshape(num_cal_pnts, 1),
                    "yvals": pdd["cal_triangle"].T[1].reshape(num_cal_pnts, 1),
                    "setlabel": [
                        r"V$_{\left |" + str(i) + r"\right >}$"
                        for i in range(num_cal_pnts)
                    ],
                    "marker": "d",
                    "line_kws": {"markersize": 14, "markeredgecolor": "white"},
                    "do_legend": True,
                    # "legend_title": "Calibration points",
                    "legend_ncol": 3,
                    "linestyle": "",
                }

            # define figure and axes here to have custom layout
            self.figs[fig_id_RB_on_IQ], axs = plt.subplots(
                ncols=2, figsize=(fs[0] * 2.0, fs[1])
            )
            self.figs[fig_id_RB_on_IQ].patch.set_alpha(0)
            self.axs[fig_id_RB_on_IQ] = axs[0]
            fig_id_RB_on_IQ_det = fig_id_RB_on_IQ + "_detailed"
            self.axs[fig_id_RB_on_IQ_det] = axs[1]
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()

            close_triangle = list(range(num_cal_pnts)) + [0]
            self.plot_dicts[fig_id_RB_on_IQ] = {
                "ax_id": fig_id_RB_on_IQ,
                "plotfn": self.plot_line,
                "xvals": pdd["cal_triangle"].T[0][close_triangle],
                "yvals": pdd["cal_triangle"].T[1][close_triangle],
                "xlabel": val_names[self.rates_I_quad_ch_idx],
                "xunit": rdd["value_units"][0],
                "ylabel": val_names[self.rates_Q_quad_ch_idx],
                "yunit": rdd["value_units"][1],
                "title": rdd["timestamp_string"]
                + "\n"
                + rdd["measurementstring"]
                + " hexbin plot",
                "marker": "",
                "color": "black",
                "line_kws": {"linewidth": 1},
                "setlabel": "NONE",
            }

            self.plot_dicts[fig_id_RB_on_IQ_det] = {
                "ax_id": fig_id_RB_on_IQ_det,
                "plotfn": self.plot_line,
                "xvals": pdd["cal_triangle"].T[0][:2],
                "yvals": pdd["cal_triangle"].T[1][:2],
                "xlabel": val_names[self.rates_I_quad_ch_idx],
                "xunit": rdd["value_units"][0],
                "ylabel": val_names[self.rates_Q_quad_ch_idx],
                "yunit": rdd["value_units"][1],
                "title": r"Detailed view",
                "marker": "",
                "color": "black",
                "line_kws": {"linewidth": 1},
                "setlabel": "NONE",
            }

            val_name_Q = rdd["value_names"][self.rates_Q_quad_ch_idx]
            rb_SI = (pdd["SI"][val_name_I], pdd["SI"][val_name_Q])
            rb_SX = (pdd["SX"][val_name_I], pdd["SX"][val_name_Q])
            rb_SI_corr = (pdd["SI_corr"][val_name_I], pdd["SI_corr"][val_name_Q])
            rb_SX_corr = (pdd["SX_corr"][val_name_I], pdd["SX_corr"][val_name_Q])

            sigs = (rb_SI, rb_SI_corr, rb_SX, rb_SX_corr)
            ids = ("SI", "SI_corr", "SX", "SX_corr")
            labels = ("SI", "SI corrected", "SX", "SX corrected")

            cols = ["royalblue", "dodgerblue", "red", "salmon"]
            mks = [8, 4, 8, 4]
            for ax_id, do_legend in zip(
                [fig_id_RB_on_IQ, fig_id_RB_on_IQ_det], [True, False]
            ):
                for S, col, mk_size, ID, label in zip(sigs, cols, mks, ids, labels):
                    self.plot_dicts[ax_id + "_{}".format(ID)] = {
                        "plotfn": self.plot_line,
                        "ax_id": ax_id,
                        "xvals": S[0],
                        "yvals": S[1],
                        "setlabel": label,
                        "marker": "o",
                        "line_kws": {"markersize": mk_size},
                        "color": col,
                        "do_legend": do_legend,
                        "legend_ncol": 3,
                        "linestyle": "",
                    }

            for idx in [self.rates_I_quad_ch_idx, self.rates_Q_quad_ch_idx]:
                val_name = rdd["value_names"][idx]
                self.plot_dicts["raw_RB_curve_data_{}".format(val_name)] = {
                    "plotfn": plot_raw_RB_curve,
                    "ncl": pdd["ncl"],
                    "SI": pdd["SI"][val_name],
                    "SX": pdd["SX"][val_name],
                    "V0": pdd["V0"][val_name],
                    "V1": pdd["V1"][val_name],
                    "V2": pdd["V2"][val_name],
                    "xlabel": "Number of Cliffords",
                    "xunit": "#",
                    "ylabel": val_name,
                    "yunit": pdd["value_units"][idx],
                    "title": pdd["timestamp_string"] + "\n" + pdd["measurementstring"],
                }

            self.plot_dicts["rb_rate_eq_pops_{}".format(val_name_I)] = {
                "plotfn": plot_populations_RB_curve,
                "ncl": pdd["ncl"],
                "P0": pdd["P0"][val_name_I],
                "P1": pdd["P1"][val_name_I],
                "P2": pdd["P2"][val_name_I],
                "title": pdd["timestamp_string"]
                + "\n"
                + "Population using rate equations ch{}".format(val_name_I),
            }

            # [2020-07-09 Victor] This is not being used for anything...
            # self.plot_dicts["logres_decision_bound"] = {
            #     "plotfn": plot_classifier_decission_boundary,
            #     "classifier": pdd["classifier"],
            #     "shots_0": (
            #         pdd["cal_pts_zero"][val_names[ch_idx_0]],
            #         pdd["cal_pts_zero"][val_names[ch_idx_1]],
            #     ),
            #     "shots_1": (
            #         pdd["cal_pts_one"][val_names[ch_idx_0]],
            #         pdd["cal_pts_one"][val_names[ch_idx_1]],
            #     ),
            #     "shots_2": (
            #         pdd["cal_pts_two"][val_names[ch_idx_0]],
            #         pdd["cal_pts_two"][val_names[ch_idx_1]],
            #     ),
            #     "xlabel": val_names[ch_idx_0],
            #     "xunit": pdd["value_units"][0],
            #     "ylabel": val_names[ch_idx_1],
            #     "yunit": pdd["value_units"][1],
            #     "title": pdd["timestamp_string"]
            #     + "\n"
            #     + pdd["measurementstring"]
            #     + " Decision boundary",
            #     "plotsize": (fs[0] * 1.5, fs[1]),
            # }

        # #####################################################################
        # End of plots for single qubit only
        # #####################################################################

        if self.do_fitting:
            # define figure and axes here to have custom layout
            rb_fig_id = "main_rb_decay_{}".format(fit_input_tag)
            leak_fig_id = "leak_decay_{}".format(fit_input_tag)
            self.figs[rb_fig_id], axs = plt.subplots(
                nrows=2, sharex=True, gridspec_kw={"height_ratios": (2, 1)}
            )
            self.figs[rb_fig_id].patch.set_alpha(0)
            self.axs[rb_fig_id] = axs[0]
            self.axs[leak_fig_id] = axs[1]
            self.plot_dicts[rb_fig_id] = {
                "plotfn": plot_rb_decay_woods_gambetta,
                "ncl": pdd["ncl"],
                "M0": pdd["M0"][fit_input_tag],
                "X1": pdd["X1"][fit_input_tag],
                "ax1": axs[1],
                "title": pdd["timestamp_string"] + "\n" + pdd["measurementstring"],
            }

            self.plot_dicts["fit_leak"] = {
                "plotfn": self.plot_fit,
                "ax_id": leak_fig_id,
                "fit_res": self.fit_res["leakage_decay_" + fit_input_tag],
                "setlabel": "Leakage fit",
                "do_legend": True,
                "color": "C2",
            }
            self.plot_dicts["fit_rb_simple"] = {
                "plotfn": self.plot_fit,
                "ax_id": rb_fig_id,
                "fit_res": self.fit_res["rb_decay_simple_" + fit_input_tag],
                "setlabel": "Simple RB fit",
                "do_legend": True,
            }
            self.plot_dicts["fit_rb"] = {
                "plotfn": self.plot_fit,
                "ax_id": rb_fig_id,
                "fit_res": self.fit_res["rb_decay_" + fit_input_tag],
                "setlabel": "Full RB fit",
                "do_legend": True,
                "color": "C2",
            }

            self.plot_dicts["rb_text"] = {
                "plotfn": self.plot_text,
                "text_string": pdd["rb_msg_" + fit_input_tag],
                "xpos": 1.05,
                "ypos": 0.6,
                "ax_id": rb_fig_id,
                "horizontalalignment": "left",
            }


class RandomizedBenchmarking_TwoQubit_Analysis(
    RandomizedBenchmarking_SingleQubit_Analysis
):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label="",
        options_dict: dict = None,
        auto=True,
        close_figs=True,
        classification_method="rates",
        rates_I_quad_ch_idxs: list = [0, 2],
        ignore_f_cal_pts: bool = False,
        extract_only: bool = False,
    ):
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=close_figs,
            do_fitting=True,
            extract_only=extract_only,
        )
        self.d1 = 4
        self.rates_I_quad_ch_idxs = rates_I_quad_ch_idxs
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method

        # The interleaved analysis does a bit of nasty things and this becomes
        # necessary
        self.overwrite_qois = True

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        # We run the single qubit analysis twice for each qubit
        # It will generate all the quantities we want for each qubit

        cal_2Q = ["00", "01", "10", "11", "02", "20", "22"]
        rates_I_quad_ch_idx = self.rates_I_quad_ch_idxs[0]
        cal_1Q = [state[rates_I_quad_ch_idx // 2] for state in cal_2Q]

        a_q0 = RandomizedBenchmarking_SingleQubit_Analysis(
            t_start=self.t_start,
            rates_I_quad_ch_idx=rates_I_quad_ch_idx,
            cal_pnts_in_dset=cal_1Q,
            do_fitting=False,
            extract_only=self.extract_only,
        )

        rates_I_quad_ch_idx = self.rates_I_quad_ch_idxs[1]
        cal_1Q = [state[rates_I_quad_ch_idx // 2] for state in cal_2Q]
        a_q1 = RandomizedBenchmarking_SingleQubit_Analysis(
            t_start=self.t_start,
            rates_I_quad_ch_idx=rates_I_quad_ch_idx,
            cal_pnts_in_dset=cal_1Q,
            do_fitting=False,
            extract_only=self.extract_only,
        )

        # Upwards and downwards hierarchical compatibilities
        rdd = self.raw_data_dict
        self.timestamps = a_q0.timestamps
        rdd["analyses"] = {"q0": a_q0, "q1": a_q1}

        rdd["folder"] = a_q0.raw_data_dict["folder"]
        rdd["timestamps"] = a_q0.raw_data_dict["timestamps"]
        rdd["timestamp_string"] = a_q0.raw_data_dict["timestamp_string"]
        rdd["measurementstring"] = a_q1.raw_data_dict["measurementstring"]

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        pdd = self.proc_data_dict
        for key in ["M0", "X1"]:
            # Keeping it compatible with 1Q on purpose
            pdd[key] = OrderedDict()

        rdd = self.raw_data_dict

        pdd["folder"] = rdd["folder"]
        pdd["timestamps"] = rdd["timestamps"]
        pdd["timestamp_string"] = rdd["timestamp_string"]
        pdd["measurementstring"] = rdd["measurementstring"]

        val_names = rdd["analyses"]["q0"].raw_data_dict["value_names"]

        if self.classification_method == "rates":
            val_name_q0 = val_names[self.rates_I_quad_ch_idxs[0]]
            val_name_q1 = val_names[self.rates_I_quad_ch_idxs[1]]

            fit_input_tag = "2Q"
            self.proc_data_dict["M0"][fit_input_tag] = (
                rdd["analyses"]["q0"].proc_data_dict["P0"][val_name_q0]
                * rdd["analyses"]["q1"].proc_data_dict["P0"][val_name_q1]
            )

            self.proc_data_dict["X1"][fit_input_tag] = (
                1
                - rdd["analyses"]["q0"].proc_data_dict["P2"][val_name_q0]
                - rdd["analyses"]["q1"].proc_data_dict["P2"][val_name_q1]
            )
        else:
            raise NotImplementedError()

        # Required for the plotting in super()
        pdd["ncl"] = rdd["analyses"]["q0"].raw_data_dict["ncl"]

    def run_fitting(self):
        # Call the prepare plots of the class above
        fit_input_tag = "2Q"
        super().run_fitting(fit_input_tag=fit_input_tag)

    def prepare_plots(self):
        # Call the prepare plots of the class above
        fit_input_tag = "2Q"
        super().prepare_plots(fit_input_tag=fit_input_tag)


class UnitarityBenchmarking_TwoQubit_Analysis(
    RandomizedBenchmarking_SingleQubit_Analysis
):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label="",
        options_dict: dict = None,
        auto=True,
        close_figs=True,
        classification_method="rates",
        rates_ch_idxs: list = [0, 2],
        ignore_f_cal_pts: bool = False,
        nseeds: int = None,
        **kwargs
    ):
        """Analysis for unitarity benchmarking.

        This analysis is based on
        """
        log.error(
            "[2020-07-12 Victor] This analysis requires to be "
            "upgraded to the new version of the 1Q-RB analysis."
        )
        if nseeds is None:
            raise TypeError("You must specify number of seeds!")
        self.nseeds = nseeds
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=close_figs,
            do_fitting=True,
            **kwargs
        )
        self.d1 = 4
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idxs = rates_ch_idxs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """Custom data extraction for Unitarity benchmarking.

        To determine the unitarity data is acquired in different bases.
        This method extracts that data and puts it in specific bins.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop, label=self.labels
        )

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False
        )
        a.get_naming_and_values()

        if "bins" in a.data_file["Experimental Data"]["Experimental Metadata"].keys():
            bins = a.data_file["Experimental Data"]["Experimental Metadata"]["bins"][()]
            self.raw_data_dict["ncl"] = bins[:-7:10]  # 7 calibration points
            self.raw_data_dict["bins"] = bins

            self.raw_data_dict["value_names"] = a.value_names
            self.raw_data_dict["value_units"] = a.value_units
            self.raw_data_dict["measurementstring"] = a.measurementstring
            self.raw_data_dict["timestamp_string"] = a.timestamp_string

            self.raw_data_dict["binned_vals"] = OrderedDict()
            self.raw_data_dict["cal_pts_x0"] = OrderedDict()
            self.raw_data_dict["cal_pts_x1"] = OrderedDict()
            self.raw_data_dict["cal_pts_x2"] = OrderedDict()
            self.raw_data_dict["cal_pts_0x"] = OrderedDict()
            self.raw_data_dict["cal_pts_1x"] = OrderedDict()
            self.raw_data_dict["cal_pts_2x"] = OrderedDict()

            self.raw_data_dict["measured_values_ZZ"] = OrderedDict()
            self.raw_data_dict["measured_values_XZ"] = OrderedDict()
            self.raw_data_dict["measured_values_YZ"] = OrderedDict()
            self.raw_data_dict["measured_values_ZX"] = OrderedDict()
            self.raw_data_dict["measured_values_XX"] = OrderedDict()
            self.raw_data_dict["measured_values_YX"] = OrderedDict()
            self.raw_data_dict["measured_values_ZY"] = OrderedDict()
            self.raw_data_dict["measured_values_XY"] = OrderedDict()
            self.raw_data_dict["measured_values_YY"] = OrderedDict()
            self.raw_data_dict["measured_values_mZmZ"] = OrderedDict()

            for i, val_name in enumerate(a.value_names):
                invalid_idxs = np.where(
                    (a.measured_values[0] == 0)
                    & (a.measured_values[1] == 0)
                    & (a.measured_values[2] == 0)
                    & (a.measured_values[3] == 0)
                )[0]
                a.measured_values[:, invalid_idxs] = np.array(
                    [[np.nan] * len(invalid_idxs)] * 4
                )

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order="F"
                )
                self.raw_data_dict["binned_vals"][val_name] = binned_yvals

                # 7 cal points:  [00, 01, 10, 11, 02, 20, 22]
                #      col_idx:  [-7, -6, -5, -4, -3, -2, -1]
                self.raw_data_dict["cal_pts_x0"][val_name] = binned_yvals[
                    (-7, -5), :
                ].flatten()
                self.raw_data_dict["cal_pts_x1"][val_name] = binned_yvals[
                    (-6, -4), :
                ].flatten()
                self.raw_data_dict["cal_pts_x2"][val_name] = binned_yvals[
                    (-3, -1), :
                ].flatten()

                self.raw_data_dict["cal_pts_0x"][val_name] = binned_yvals[
                    (-7, -6), :
                ].flatten()
                self.raw_data_dict["cal_pts_1x"][val_name] = binned_yvals[
                    (-5, -4), :
                ].flatten()
                self.raw_data_dict["cal_pts_2x"][val_name] = binned_yvals[
                    (-2, -1), :
                ].flatten()

                self.raw_data_dict["measured_values_ZZ"][val_name] = binned_yvals[
                    0:-7:10, :
                ]
                self.raw_data_dict["measured_values_XZ"][val_name] = binned_yvals[
                    1:-7:10, :
                ]
                self.raw_data_dict["measured_values_YZ"][val_name] = binned_yvals[
                    2:-7:10, :
                ]
                self.raw_data_dict["measured_values_ZX"][val_name] = binned_yvals[
                    3:-7:10, :
                ]
                self.raw_data_dict["measured_values_XX"][val_name] = binned_yvals[
                    4:-7:10, :
                ]
                self.raw_data_dict["measured_values_YX"][val_name] = binned_yvals[
                    5:-7:10, :
                ]
                self.raw_data_dict["measured_values_ZY"][val_name] = binned_yvals[
                    6:-7:10, :
                ]
                self.raw_data_dict["measured_values_XY"][val_name] = binned_yvals[
                    7:-7:10, :
                ]
                self.raw_data_dict["measured_values_YY"][val_name] = binned_yvals[
                    8:-7:10, :
                ]
                self.raw_data_dict["measured_values_mZmZ"][val_name] = binned_yvals[
                    9:-7:10, :
                ]

        else:
            bins = None

        self.raw_data_dict["folder"] = a.folder
        self.raw_data_dict["timestamps"] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        """Averages shot data and calculates unitarity from raw_data_dict.

        Note: this doe not correct the outcomes for leakage.



        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        keys = [
            "Vx0",
            "V0x",
            "Vx1",
            "V1x",
            "Vx2",
            "V2x",
            "SI",
            "SX",
            "Px0",
            "P0x",
            "Px1",
            "P1x",
            "Px2",
            "P2x",
            "M_inv_q0",
            "M_inv_q1",
        ]
        keys += [
            "XX",
            "XY",
            "XZ",
            "YX",
            "YY",
            "YZ",
            "ZX",
            "ZY",
            "ZZ",
            "XX_sq",
            "XY_sq",
            "XZ_sq",
            "YX_sq",
            "YY_sq",
            "YZ_sq",
            "ZX_sq",
            "ZY_sq",
            "ZZ_sq",
            "unitarity_shots",
            "unitarity",
        ]
        keys += [
            "XX_q0",
            "XY_q0",
            "XZ_q0",
            "YX_q0",
            "YY_q0",
            "YZ_q0",
            "ZX_q0",
            "ZY_q0",
            "ZZ_q0",
        ]
        keys += [
            "XX_q1",
            "XY_q1",
            "XZ_q1",
            "YX_q1",
            "YY_q1",
            "YZ_q1",
            "ZX_q1",
            "ZY_q1",
            "ZZ_q1",
        ]
        for key in keys:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict["value_names"]:
            for idx in ["x0", "x1", "x2", "0x", "1x", "2x"]:
                self.proc_data_dict["V{}".format(idx)][val_name] = np.nanmean(
                    self.raw_data_dict["cal_pts_{}".format(idx)][val_name]
                )
            SI = np.nanmean(self.raw_data_dict["measured_values_ZZ"][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict["measured_values_mZmZ"][val_name], axis=1
            )
            self.proc_data_dict["SI"][val_name] = SI
            self.proc_data_dict["SX"][val_name] = SX

            Px0, Px1, Px2, M_inv_q0 = populations_using_rate_equations(
                SI,
                SX,
                self.proc_data_dict["Vx0"][val_name],
                self.proc_data_dict["Vx1"][val_name],
                self.proc_data_dict["Vx2"][val_name],
            )
            P0x, P1x, P2x, M_inv_q1 = populations_using_rate_equations(
                SI,
                SX,
                self.proc_data_dict["V0x"][val_name],
                self.proc_data_dict["V1x"][val_name],
                self.proc_data_dict["V2x"][val_name],
            )

            for key, val in [
                ("Px0", Px0),
                ("Px1", Px1),
                ("Px2", Px2),
                ("P0x", P0x),
                ("P1x", P1x),
                ("P2x", P2x),
                ("M_inv_q0", M_inv_q0),
                ("M_inv_q1", M_inv_q1),
            ]:
                self.proc_data_dict[key][val_name] = val

            for key in ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]:
                Vmeas = self.raw_data_dict["measured_values_" + key][val_name]
                Px2 = self.proc_data_dict["Px2"][val_name]
                V0 = self.proc_data_dict["Vx0"][val_name]
                V1 = self.proc_data_dict["Vx1"][val_name]
                V2 = self.proc_data_dict["Vx2"][val_name]
                val = Vmeas + 0  # - (Px2*V2 - (1-Px2)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key + "_q0"][val_name] = val * 2 - 1

                P2x = self.proc_data_dict["P2x"][val_name]
                V0 = self.proc_data_dict["V0x"][val_name]
                V1 = self.proc_data_dict["V1x"][val_name]

                # Leakage is ignored in this analysis.
                # V2 = self.proc_data_dict['V2x'][val_name]
                val = Vmeas + 0  # - (P2x*V2 - (1-P2x)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key + "_q1"][val_name] = val * 2 - 1

        if self.classification_method == "rates":
            val_name_q0 = self.raw_data_dict["value_names"][self.rates_ch_idxs[0]]
            val_name_q1 = self.raw_data_dict["value_names"][self.rates_ch_idxs[1]]

            self.proc_data_dict["M0"] = (
                self.proc_data_dict["Px0"][val_name_q0]
                * self.proc_data_dict["P0x"][val_name_q1]
            )

            self.proc_data_dict["X1"] = (
                1
                - self.proc_data_dict["Px2"][val_name_q0]
                - self.proc_data_dict["P2x"][val_name_q1]
            )

            # The unitarity is calculated here.
            self.proc_data_dict["unitarity_shots"] = (
                self.proc_data_dict["ZZ_q0"][val_name_q0] * 0
            )

            # Unitarity according to Eq. (10) Wallman et al. New J. Phys. 2015
            # Pj = d/(d-1)*|n(rho_j)|^2
            # Note that the dimensionality prefix is ignored here as it
            # should drop out in the fits.
            for key in ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]:
                self.proc_data_dict[key] = (
                    self.proc_data_dict[key + "_q0"][val_name_q0]
                    * self.proc_data_dict[key + "_q1"][val_name_q1]
                )
                self.proc_data_dict[key + "_sq"] = self.proc_data_dict[key] ** 2

                self.proc_data_dict["unitarity_shots"] += self.proc_data_dict[
                    key + "_sq"
                ]

            self.proc_data_dict["unitarity"] = np.mean(
                self.proc_data_dict["unitarity_shots"], axis=1
            )
        else:
            raise NotImplementedError()

    def run_fitting(self):
        super().run_fitting()
        self.fit_res["unitarity_decay"] = self.fit_unitarity_decay()

        unitarity_dec = self.fit_res["unitarity_decay"].params

        text_msg = "Summary: \n"
        text_msg += format_value_string(
            "Unitarity\n" + r"$u$", unitarity_dec["u"], "\n"
        )
        text_msg += format_value_string(
            "Error due to\nincoherent mechanisms\n" + r"$\epsilon$",
            unitarity_dec["eps"],
        )

        self.proc_data_dict["unitarity_msg"] = text_msg

    def fit_unitarity_decay(self):
        """Fits the data using the unitarity model."""
        fit_mod_unitarity = lmfit.Model(unitarity_decay, independent_vars="m")
        fit_mod_unitarity.set_param_hint("A", value=0.1, min=0, max=1, vary=True)
        fit_mod_unitarity.set_param_hint("B", value=0.8, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint("u", value=0.9, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint("d1", value=self.d1, vary=False)
        # Error due to incoherent sources
        # Feng Phys. Rev. Lett. 117, 260501 (2016) eq. (4)
        fit_mod_unitarity.set_param_hint("eps", expr="((d1-1)/d1)*(1-u**0.5)")

        params = fit_mod_unitarity.make_params()
        fit_mod_unitarity = fit_mod_unitarity.fit(
            data=self.proc_data_dict["unitarity"],
            m=self.proc_data_dict["ncl"],
            params=params,
        )

        return fit_mod_unitarity

    def prepare_plots(self):
        val_names = self.proc_data_dict["value_names"]

        for i, val_name in enumerate(val_names):
            self.plot_dicts["binned_data_{}".format(val_name)] = {
                "plotfn": self.plot_line,
                "xvals": self.proc_data_dict["bins"],
                "yvals": np.nanmean(
                    self.proc_data_dict["binned_vals"][val_name], axis=1
                ),
                "yerr": sem(self.proc_data_dict["binned_vals"][val_name], axis=1),
                "xlabel": "Number of Cliffords",
                "xunit": "#",
                "ylabel": val_name,
                "yunit": self.proc_data_dict["value_units"][i],
                "title": self.proc_data_dict["timestamp_string"]
                + "\n"
                + self.proc_data_dict["measurementstring"],
            }
        fs = plt.rcParams["figure.figsize"]

        # define figure and axes here to have custom layout
        self.figs["rb_populations_decay"], axs = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(fs[0] * 1.5, fs[1])
        )
        self.figs["rb_populations_decay"].suptitle(
            self.proc_data_dict["timestamp_string"]
            + "\n"
            + "Population using rate equations",
            y=1.05,
        )
        self.figs["rb_populations_decay"].patch.set_alpha(0)
        self.axs["rb_pops_q0"] = axs[0]
        self.axs["rb_pops_q1"] = axs[1]

        val_name_q0 = val_names[self.rates_ch_idxs[0]]
        val_name_q1 = val_names[self.rates_ch_idxs[1]]
        self.plot_dicts["rb_rate_eq_pops_{}".format(val_name_q0)] = {
            "plotfn": plot_populations_RB_curve,
            "ncl": self.proc_data_dict["ncl"],
            "P0": self.proc_data_dict["Px0"][val_name_q0],
            "P1": self.proc_data_dict["Px1"][val_name_q0],
            "P2": self.proc_data_dict["Px2"][val_name_q0],
            "title": " {}".format(val_name_q0),
            "ax_id": "rb_pops_q0",
        }

        self.plot_dicts["rb_rate_eq_pops_{}".format(val_name_q1)] = {
            "plotfn": plot_populations_RB_curve,
            "ncl": self.proc_data_dict["ncl"],
            "P0": self.proc_data_dict["P0x"][val_name_q1],
            "P1": self.proc_data_dict["P1x"][val_name_q1],
            "P2": self.proc_data_dict["P2x"][val_name_q1],
            "title": " {}".format(val_name_q1),
            "ax_id": "rb_pops_q1",
        }

        self.plot_dicts["cal_points_hexbin_q0"] = {
            "plotfn": plot_cal_points_hexbin,
            "shots_0": (
                self.proc_data_dict["cal_pts_x0"][val_names[0]],
                self.proc_data_dict["cal_pts_x0"][val_names[1]],
            ),
            "shots_1": (
                self.proc_data_dict["cal_pts_x1"][val_names[0]],
                self.proc_data_dict["cal_pts_x1"][val_names[1]],
            ),
            "shots_2": (
                self.proc_data_dict["cal_pts_x2"][val_names[0]],
                self.proc_data_dict["cal_pts_x2"][val_names[1]],
            ),
            "xlabel": val_names[0],
            "xunit": self.proc_data_dict["value_units"][0],
            "ylabel": val_names[1],
            "yunit": self.proc_data_dict["value_units"][1],
            "common_clims": False,
            "title": self.proc_data_dict["timestamp_string"]
            + "\n"
            + self.proc_data_dict["measurementstring"]
            + " hexbin plot q0",
            "plotsize": (fs[0] * 1.5, fs[1]),
        }
        self.plot_dicts["cal_points_hexbin_q1"] = {
            "plotfn": plot_cal_points_hexbin,
            "shots_0": (
                self.proc_data_dict["cal_pts_0x"][val_names[2]],
                self.proc_data_dict["cal_pts_0x"][val_names[3]],
            ),
            "shots_1": (
                self.proc_data_dict["cal_pts_1x"][val_names[2]],
                self.proc_data_dict["cal_pts_1x"][val_names[3]],
            ),
            "shots_2": (
                self.proc_data_dict["cal_pts_2x"][val_names[2]],
                self.proc_data_dict["cal_pts_2x"][val_names[3]],
            ),
            "xlabel": val_names[2],
            "xunit": self.proc_data_dict["value_units"][2],
            "ylabel": val_names[3],
            "yunit": self.proc_data_dict["value_units"][3],
            "common_clims": False,
            "title": self.proc_data_dict["timestamp_string"]
            + "\n"
            + self.proc_data_dict["measurementstring"]
            + " hexbin plot q1",
            "plotsize": (fs[0] * 1.5, fs[1]),
        }

        # define figure and axes here to have custom layout
        self.figs["main_rb_decay"], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": (2, 1)}
        )
        self.figs["main_rb_decay"].patch.set_alpha(0)
        self.axs["main_rb_decay"] = axs[0]
        self.axs["leak_decay"] = axs[1]
        self.plot_dicts["main_rb_decay"] = {
            "plotfn": plot_rb_decay_woods_gambetta,
            "ncl": self.proc_data_dict["ncl"],
            "M0": self.proc_data_dict["M0"],
            "X1": self.proc_data_dict["X1"],
            "ax1": axs[1],
            "title": self.proc_data_dict["timestamp_string"]
            + "\n"
            + self.proc_data_dict["measurementstring"],
        }

        self.plot_dicts["fit_leak"] = {
            "plotfn": self.plot_fit,
            "ax_id": "leak_decay",
            "fit_res": self.fit_res["leakage_decay"],
            "setlabel": "Leakage fit",
            "do_legend": True,
            "color": "C2",
        }
        self.plot_dicts["fit_rb_simple"] = {
            "plotfn": self.plot_fit,
            "ax_id": "main_rb_decay",
            "fit_res": self.fit_res["rb_decay_simple"],
            "setlabel": "Simple RB fit",
            "do_legend": True,
        }
        self.plot_dicts["fit_rb"] = {
            "plotfn": self.plot_fit,
            "ax_id": "main_rb_decay",
            "fit_res": self.fit_res["rb_decay"],
            "setlabel": "Full RB fit",
            "do_legend": True,
            "color": "C2",
        }

        self.plot_dicts["rb_text"] = {
            "plotfn": self.plot_text,
            "text_string": self.proc_data_dict["rb_msg"],
            "xpos": 1.05,
            "ypos": 0.6,
            "ax_id": "main_rb_decay",
            "horizontalalignment": "left",
        }

        self.plot_dicts["correlated_readouts"] = {
            "plotfn": plot_unitarity_shots,
            "ncl": self.proc_data_dict["ncl"],
            "unitarity_shots": self.proc_data_dict["unitarity_shots"],
            "xlabel": "Number of Cliffords",
            "xunit": "#",
            "ylabel": "Unitarity",
            "yunit": "",
            "title": self.proc_data_dict["timestamp_string"]
            + "\n"
            + self.proc_data_dict["measurementstring"],
        }

        self.figs["unitarity"] = plt.subplots(nrows=1)
        self.plot_dicts["unitarity"] = {
            "plotfn": plot_unitarity,
            "ax_id": "unitarity",
            "ncl": self.proc_data_dict["ncl"],
            "P": self.proc_data_dict["unitarity"],
            "xlabel": "Number of Cliffords",
            "xunit": "#",
            "ylabel": "Unitarity",
            "yunit": "frac",
            "title": self.proc_data_dict["timestamp_string"]
            + "\n"
            + self.proc_data_dict["measurementstring"],
        }
        self.plot_dicts["fit_unitarity"] = {
            "plotfn": self.plot_fit,
            "ax_id": "unitarity",
            "fit_res": self.fit_res["unitarity_decay"],
            "setlabel": "Simple unitarity fit",
            "do_legend": True,
        }
        self.plot_dicts["unitarity_text"] = {
            "plotfn": self.plot_text,
            "text_string": self.proc_data_dict["unitarity_msg"],
            "xpos": 0.6,
            "ypos": 0.8,
            "ax_id": "unitarity",
            "horizontalalignment": "left",
        }


class InterleavedRandomizedBenchmarkingAnalysis(ba.BaseDataAnalysis):
    """
    Analysis for two qubit interleaved randomized benchmarking of a CZ gate.
    [2020-07-12 Victor] upgraded to allow for analysis of iRB for the
    parked qubit during CZ on the other qubits

    This is a meta-analysis. It runs
    "RandomizedBenchmarking_TwoQubit_Analysis" for each of the individual
    datasets in the "extract_data" method and uses the quantities of interest
    to create the combined figure.

    The figure as well as the quantities of interest are stored in
    the interleaved data file.
    """

    def __init__(
        self,
        ts_base: str = None,
        ts_int: str = None,
        ts_int_idle: str = None,
        label_base: str = "",
        label_int: str = "",
        label_int_idle: str = "",
        options_dict: dict = {},
        auto=True,
        close_figs=True,
        rates_I_quad_ch_idxs: list = [0, 2],
        ignore_f_cal_pts: bool = False,
        plot_label="",
        extract_only=False,
    ):
        super().__init__(
            do_fitting=True,
            close_figs=close_figs,
            options_dict=options_dict,
            extract_only=extract_only,
        )
        self.ts_base = ts_base
        self.ts_int = ts_int
        self.ts_int_idle = ts_int_idle
        self.label_base = label_base
        self.label_int = label_int
        self.label_int_idle = label_int_idle
        self.include_idle = self.ts_int_idle or self.label_int_idle

        assert ts_base or label_base
        assert ts_int or label_int

        self.rates_I_quad_ch_idxs = rates_I_quad_ch_idxs
        self.options_dict = options_dict
        self.close_figs = close_figs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        self.plot_label = plot_label

        # For other classes derived from this one this will change
        self.fit_tag = "2Q"
        self.int_name = "CZ"

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a_base = RandomizedBenchmarking_TwoQubit_Analysis(
            t_start=self.ts_base,
            label=self.label_base,
            options_dict=self.options_dict,
            auto=True,
            close_figs=self.close_figs,
            rates_I_quad_ch_idxs=self.rates_I_quad_ch_idxs,
            extract_only=True,
            ignore_f_cal_pts=self.ignore_f_cal_pts,
        )
        a_int = RandomizedBenchmarking_TwoQubit_Analysis(
            t_start=self.ts_int,
            label=self.label_int,
            options_dict=self.options_dict,
            auto=True,
            close_figs=self.close_figs,
            rates_I_quad_ch_idxs=self.rates_I_quad_ch_idxs,
            extract_only=True,
            ignore_f_cal_pts=self.ignore_f_cal_pts,
        )
        if self.include_idle:
            a_int_idle = RandomizedBenchmarking_TwoQubit_Analysis(
                t_start=self.ts_int_idle,
                label=self.label_int_idle,
                options_dict=self.options_dict,
                auto=True,
                close_figs=self.close_figs,
                rates_I_quad_ch_idxs=self.rates_I_quad_ch_idxs,
                extract_only=True,
                ignore_f_cal_pts=self.ignore_f_cal_pts,
            )

        # order is such that any information (figures, quantities of interest)
        # are saved in the interleaved file.
        self.timestamps = [a_int.timestamps[0], a_base.timestamps[0]]

        self.raw_data_dict["timestamps"] = self.timestamps
        self.raw_data_dict["timestamp_string"] = a_int.proc_data_dict[
            "timestamp_string"
        ]
        self.raw_data_dict["folder"] = a_int.proc_data_dict["folder"]
        a_dict = {"base": a_base, "int": a_int}
        if self.include_idle:
            a_dict["int_idle"] = a_int_idle
        self.raw_data_dict["analyses"] = a_dict

        if not self.plot_label:
            self.plot_label = a_int.proc_data_dict["measurementstring"]

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        self.proc_data_dict["quantities_of_interest"] = {}
        qoi = self.proc_data_dict["quantities_of_interest"]

        qoi_base = self.raw_data_dict["analyses"]["base"].proc_data_dict[
            "quantities_of_interest"
        ]
        qoi_int = self.raw_data_dict["analyses"]["int"].proc_data_dict[
            "quantities_of_interest"
        ]

        self.overwrite_qois = True
        qoi.update({k + "_ref": v for k, v in qoi_base.items()})
        qoi.update({k + "_int": v for k, v in qoi_int.items()})

        # The functionality of this analysis was extended to make it usable for
        # interleaved parking idle flux pulse
        fit_tag = self.fit_tag
        int_name = self.int_name

        qoi["eps_%s_X1" % int_name] = interleaved_error(
            eps_int=qoi_int["eps_X1_%s" % fit_tag],
            eps_base=qoi_base["eps_X1_%s" % fit_tag],
        )
        qoi["eps_%s_simple" % int_name] = interleaved_error(
            eps_int=qoi_int["eps_simple_%s" % fit_tag],
            eps_base=qoi_base["eps_simple_%s" % fit_tag],
        )
        qoi["L1_%s" % int_name] = interleaved_error(
            eps_int=qoi_int["L1_%s" % fit_tag], eps_base=qoi_base["L1_%s" % fit_tag]
        )
        if self.include_idle:
            qoi_int_idle = self.raw_data_dict["analyses"]["int_idle"].proc_data_dict[
                "quantities_of_interest"
            ]
            qoi.update({k + "_int_idle": v for k, v in qoi_int_idle.items()})
            qoi["eps_idle_X1"] = interleaved_error(
                eps_int=qoi_int_idle["eps_X1_%s" % fit_tag],
                eps_base=qoi_base["eps_X1_%s" % fit_tag],
            )
            qoi["eps_idle_simple"] = interleaved_error(
                eps_int=qoi_int_idle["eps_simple_%s" % fit_tag],
                eps_base=qoi_base["eps_simple_%s" % fit_tag],
            )
            qoi["L1_idle"] = interleaved_error(
                eps_int=qoi_int_idle["L1_%s" % fit_tag],
                eps_base=qoi_base["L1_%s" % fit_tag],
            )

        if int_name == "CZ":
            # This is the naive estimate, when all observed error is assigned
            # to the CZ gate
            try:
                qoi["L1_%s_naive" % int_name] = 1 - (
                    1 - qoi_base["L1_%s" % fit_tag]
                ) ** (1 / 1.5)
                qoi["eps_%s_simple_naive" % int_name] = 1 - (
                    1 - qoi_base["eps_simple_%s" % fit_tag]
                ) ** (1 / 1.5)
                qoi["eps_%s_X1_naive" % int_name] = 1 - (
                    1 - qoi_base["eps_X1_%s" % fit_tag]
                ) ** (1 / 1.5)
            except ValueError:
                # prevents the analysis from crashing if the fits are bad.
                qoi["L1_%s_naive" % int_name] = ufloat(np.NaN, np.NaN)
                qoi["eps_%s_simple_naive" % int_name] = ufloat(np.NaN, np.NaN)
                qoi["eps_%s_X1_naive" % int_name] = ufloat(np.NaN, np.NaN)

    def prepare_plots(self):
        # Might seem that are not used but there is an `eval` below
        dd_ref = self.raw_data_dict["analyses"]["base"].proc_data_dict
        dd_int = self.raw_data_dict["analyses"]["int"].proc_data_dict
        fr_ref = self.raw_data_dict["analyses"]["base"].fit_res
        fr_int = self.raw_data_dict["analyses"]["int"].fit_res
        dds = {
            "int": dd_int,
            "ref": dd_ref,
        }
        frs = {
            "int": fr_int,
            "ref": fr_ref,
        }
        if self.include_idle:
            fr_int_idle = self.raw_data_dict["analyses"]["int_idle"].fit_res
            dd_int_idle = self.raw_data_dict["analyses"]["int_idle"].proc_data_dict
            dds["int_idle"] = dd_int_idle
            frs["int_idle"] = fr_int_idle

        fs = plt.rcParams["figure.figsize"]
        self.figs["main_irb_decay"], axs = plt.subplots(
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": (2, 1)},
            figsize=(fs[0] * 1.3, fs[1] * 1.3),
        )

        self.figs["main_irb_decay"].patch.set_alpha(0)
        self.axs["main_irb_decay"] = axs[0]
        self.axs["leak_decay"] = axs[1]
        self.plot_dicts["main_irb_decay"] = {
            "plotfn": plot_irb_decay_woods_gambetta,
            "ncl": dd_ref["ncl"],
            "include_idle": self.include_idle,
            "fit_tag": self.fit_tag,
            "int_name": self.int_name,
            "qoi": self.proc_data_dict["quantities_of_interest"],
            "ax1": axs[1],
            "title": "{} - {}\n{}".format(
                self.timestamps[0], self.timestamps[1], self.plot_label
            ),
        }

        def add_to_plot_dict(
            plot_dict: dict,
            tag: str,
            dd_quantities: list,
            fit_quantities: list,
            dds: dict,
            frs: dict,
        ):
            for dd_q in dd_quantities:
                plot_dict[dd_q + "_" + tag] = dds[tag][dd_q][self.fit_tag]
            for fit_q in fit_quantities:
                trans = {
                    "rb_decay": "fr_M0",
                    "rb_decay_simple": "fr_M0_simple",
                    "leakage_decay": "fr_X1",
                }
                plot_dict[trans[fit_q] + "_" + tag] = frs[tag][
                    fit_q + "_{}".format(self.fit_tag)
                ]

        tags = ["ref", "int"]
        if self.include_idle:
            tags.append("int_idle")
        for tag in tags:
            add_to_plot_dict(
                self.plot_dicts["main_irb_decay"],
                tag=tag,
                dd_quantities=["M0", "X1"],
                fit_quantities=["rb_decay", "rb_decay_simple", "leakage_decay"],
                dds=dds,
                frs=frs,
            )


class InterleavedRandomizedBenchmarkingParkingAnalysis(
    InterleavedRandomizedBenchmarkingAnalysis, ba.BaseDataAnalysis
):
    """
    Analysis for single qubit interleaved randomized benchmarking where the
    interleaved gate is a parking identity (with the corresponding CZ being
    applied on the other two qubits)

    This is a meta-analysis. It runs
    "RandomizedBenchmarking_SingleQubit_Analysis" for each of the individual
    datasets in the "extract_data" method and uses the quantities of interest
    to create the combined figure.

    The figure as well as the quantities of interest are stored in
    the interleaved data file.
    """

    def __init__(
        self,
        ts_base: str = None,
        ts_int: str = None,
        label_base: str = "",
        label_int: str = "",
        options_dict: dict = {},
        auto=True,
        close_figs=True,
        rates_I_quad_ch_idx: int = -2,
        rates_Q_quad_ch_idx: int = None,
        ignore_f_cal_pts: bool = False,
        plot_label="",
    ):
        # Here we don't want to run the __init__ of the Interleaved analysis,
        # only the __init__ of the base class
        ba.BaseDataAnalysis.__init__(
            self, do_fitting=True, close_figs=close_figs, options_dict=options_dict
        )
        self.ts_base = ts_base
        self.ts_int = ts_int
        self.label_base = label_base
        self.label_int = label_int

        assert ts_base or label_base
        assert ts_int or label_int

        self.rates_I_quad_ch_idx = rates_I_quad_ch_idx
        self.rates_Q_quad_ch_idx = rates_Q_quad_ch_idx
        if self.rates_Q_quad_ch_idx is None:
            self.rates_Q_quad_ch_idx = rates_I_quad_ch_idx + 1

        self.options_dict = options_dict
        self.close_figs = close_figs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        self.plot_label = plot_label

        # For other classes derived from this one this will change
        self.fit_tag = None  # to be set in the extract data
        self.int_name = "Idle flux"
        self.include_idle = False

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a_base = RandomizedBenchmarking_SingleQubit_Analysis(
            t_start=self.ts_base,
            label=self.label_base,
            options_dict=self.options_dict,
            auto=True,
            close_figs=self.close_figs,
            rates_I_quad_ch_idx=self.rates_I_quad_ch_idx,
            extract_only=True,
            ignore_f_cal_pts=self.ignore_f_cal_pts,
        )
        a_int = RandomizedBenchmarking_SingleQubit_Analysis(
            t_start=self.ts_int,
            label=self.label_int,
            options_dict=self.options_dict,
            auto=True,
            close_figs=self.close_figs,
            rates_I_quad_ch_idx=self.rates_I_quad_ch_idx,
            extract_only=True,
            ignore_f_cal_pts=self.ignore_f_cal_pts,
        )

        self.fit_tag = a_base.raw_data_dict["value_names"][self.rates_I_quad_ch_idx]

        # order is such that any information (figures, quantities of interest)
        # are saved in the interleaved file.
        self.timestamps = [a_int.timestamps[0], a_base.timestamps[0]]

        self.raw_data_dict["timestamps"] = self.timestamps
        self.raw_data_dict["timestamp_string"] = a_int.proc_data_dict[
            "timestamp_string"
        ]
        self.raw_data_dict["folder"] = a_int.proc_data_dict["folder"]
        self.raw_data_dict["analyses"] = {"base": a_base, "int": a_int}

        if not self.plot_label:
            self.plot_label = a_int.proc_data_dict["measurementstring"]


class CharacterBenchmarking_TwoQubit_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for character benchmarking.
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label="",
        options_dict: dict = None,
        auto=True,
        close_figs=True,
        ch_idxs: list = [0, 2],
    ):
        if options_dict is None:
            options_dict = dict()
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=close_figs,
            do_fitting=True,
        )

        self.d1 = 4
        self.ch_idxs = ch_idxs
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop, label=self.labels
        )

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False
        )
        a.get_naming_and_values()
        bins = a.data_file["Experimental Data"]["Experimental Metadata"]["bins"][()]
        a.finish()

        self.raw_data_dict["measurementstring"] = a.measurementstring
        self.raw_data_dict["timestamp_string"] = a.timestamp_string
        self.raw_data_dict["folder"] = a.folder
        self.raw_data_dict["timestamps"] = self.timestamps

        df = pd.DataFrame(
            columns={"ncl", "pauli", "I_q0", "Q_q0", "I_q1", "Q_q1", "interleaving_cl"}
        )
        df["ncl"] = bins

        # Assumptions on the structure of the datafile are made here.
        # For every Clifford, 4 random pauli's are sampled from the different
        # sub sets:
        paulis = [
            "II",  # 'IZ', 'ZI', 'ZZ',  # P00
            "IX",  # 'IY', 'ZX', 'ZY',  # P01
            "XI",  # 'XZ', 'YI', 'YZ',  # P10
            "XX",
        ]  # 'XY', 'YX', 'YY']  # P11

        paulis_df = np.tile(paulis, 34)[: len(bins)]
        # The calibration points do not correspond to a Pauli
        paulis_df[-7:] = np.nan
        df["pauli"] = paulis_df

        # The four different random Pauli's are performed both with
        # and without the interleaving CZ gate.
        df["interleaving_cl"] = np.tile([""] * 4 + ["CZ"] * 4, len(bins) // 8 + 1)[
            : len(bins)
        ]

        # Data is grouped and single shots are averaged.
        for i, ch in enumerate(["I_q0", "Q_q0", "I_q1", "Q_q1"]):
            binned_yvals = np.reshape(a.measured_values[i], (len(bins), -1), order="F")
            yvals = np.mean(binned_yvals, axis=1)
            df[ch] = yvals

        self.raw_data_dict["df"] = df

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        df = self.raw_data_dict["df"]
        cal_points = [
            # calibration point indices are when ignoring the f-state cal pts
            [[-7, -5], [-6, -4], [-3, -1]],  # q0
            [[-7, -5], [-6, -4], [-3, -1]],  # q0
            [[-7, -6], [-5, -4], [-2, -1]],  # q1
            [[-7, -6], [-5, -4], [-2, -1]],  # q1
        ]

        for ch, cal_pt in zip(["I_q0", "Q_q0", "I_q1", "Q_q1"], cal_points):
            df[ch + "_normed"] = a_tools.normalize_data_v3(
                df[ch].values, cal_zero_points=cal_pt[0], cal_one_points=cal_pt[1]
            )

        df["P_|00>"] = (1 - df["I_q0_normed"]) * (1 - df["Q_q1_normed"])

        P00 = (
            df.loc[df["pauli"].isin(["II", "IZ", "ZI", "ZZ"])]
            .loc[df["interleaving_cl"] == ""]
            .groupby("ncl")
            .mean()
        )
        P01 = (
            df.loc[df["pauli"].isin(["IX", "IY", "ZX", "ZY"])]
            .loc[df["interleaving_cl"] == ""]
            .groupby("ncl")
            .mean()
        )
        P10 = (
            df.loc[df["pauli"].isin(["XI", "XZ", "YI", "YZ"])]
            .loc[df["interleaving_cl"] == ""]
            .groupby("ncl")
            .mean()
        )
        P11 = (
            df.loc[df["pauli"].isin(["XX", "XY", "YX", "YY"])]
            .loc[df["interleaving_cl"] == ""]
            .groupby("ncl")
            .mean()
        )

        P00_CZ = (
            df.loc[df["pauli"].isin(["II", "IZ", "ZI", "ZZ"])]
            .loc[df["interleaving_cl"] == "CZ"]
            .groupby("ncl")
            .mean()
        )
        P01_CZ = (
            df.loc[df["pauli"].isin(["IX", "IY", "ZX", "ZY"])]
            .loc[df["interleaving_cl"] == "CZ"]
            .groupby("ncl")
            .mean()
        )
        P10_CZ = (
            df.loc[df["pauli"].isin(["XI", "XZ", "YI", "YZ"])]
            .loc[df["interleaving_cl"] == "CZ"]
            .groupby("ncl")
            .mean()
        )
        P11_CZ = (
            df.loc[df["pauli"].isin(["XX", "XY", "YX", "YY"])]
            .loc[df["interleaving_cl"] == "CZ"]
            .groupby("ncl")
            .mean()
        )

        # Calculate the character function
        # Eq. 7 of Xue et al. ArXiv 1811.04002v1
        C1 = P00["P_|00>"] - P01["P_|00>"] + P10["P_|00>"] - P11["P_|00>"]
        C2 = P00["P_|00>"] + P01["P_|00>"] - P10["P_|00>"] - P11["P_|00>"]
        C12 = P00["P_|00>"] - P01["P_|00>"] - P10["P_|00>"] + P11["P_|00>"]
        C1_CZ = (
            P00_CZ["P_|00>"] - P01_CZ["P_|00>"] + P10_CZ["P_|00>"] - P11_CZ["P_|00>"]
        )
        C2_CZ = (
            P00_CZ["P_|00>"] + P01_CZ["P_|00>"] - P10_CZ["P_|00>"] - P11_CZ["P_|00>"]
        )
        C12_CZ = (
            P00_CZ["P_|00>"] - P01_CZ["P_|00>"] - P10_CZ["P_|00>"] + P11_CZ["P_|00>"]
        )

        char_df = pd.DataFrame(
            {
                "P00": P00["P_|00>"],
                "P01": P01["P_|00>"],
                "P10": P10["P_|00>"],
                "P11": P11["P_|00>"],
                "P00_CZ": P00_CZ["P_|00>"],
                "P01_CZ": P01_CZ["P_|00>"],
                "P10_CZ": P10_CZ["P_|00>"],
                "P11_CZ": P11_CZ["P_|00>"],
                "C1": C1,
                "C2": C2,
                "C12": C12,
                "C1_CZ": C1_CZ,
                "C2_CZ": C2_CZ,
                "C12_CZ": C12_CZ,
            }
        )
        self.proc_data_dict["char_df"] = char_df

    def run_fitting(self):
        super().run_fitting()

        char_df = self.proc_data_dict["char_df"]
        # Eq. 8 of Xue et al. ArXiv 1811.04002v1
        for char_key in ["C1", "C2", "C12", "C1_CZ", "C2_CZ", "C12_CZ"]:
            char_mod = lmfit.Model(char_decay, independent_vars="m")
            char_mod.set_param_hint("A", value=1, vary=True)
            char_mod.set_param_hint("alpha", value=0.95)
            params = char_mod.make_params()
            self.fit_res[char_key] = char_mod.fit(
                data=char_df[char_key].values, m=char_df.index, params=params
            )

    def analyze_fit_results(self):
        fr = self.fit_res
        self.proc_data_dict["quantities_of_interest"] = {}
        qoi = self.proc_data_dict["quantities_of_interest"]
        qoi["alpha1"] = ufloat(
            fr["C1"].params["alpha"].value, fr["C1"].params["alpha"].stderr
        )
        qoi["alpha2"] = ufloat(
            fr["C2"].params["alpha"].value, fr["C2"].params["alpha"].stderr
        )
        qoi["alpha12"] = ufloat(
            fr["C12"].params["alpha"].value, fr["C12"].params["alpha"].stderr
        )
        # eq. 9 from Xue et al. ArXiv 1811.04002v1
        qoi["alpha_char"] = (
            3 / 15 * qoi["alpha1"] + 3 / 15 * qoi["alpha2"] + 9 / 15 * qoi["alpha12"]
        )

        qoi["alpha1_CZ_int"] = ufloat(
            fr["C1_CZ"].params["alpha"].value, fr["C1_CZ"].params["alpha"].stderr
        )
        qoi["alpha2_CZ_int"] = ufloat(
            fr["C2_CZ"].params["alpha"].value, fr["C2_CZ"].params["alpha"].stderr
        )
        qoi["alpha12_CZ_int"] = ufloat(
            fr["C12_CZ"].params["alpha"].value, fr["C12_CZ"].params["alpha"].stderr
        )

        qoi["alpha_char_CZ_int"] = (
            3 / 15 * qoi["alpha1_CZ_int"]
            + 3 / 15 * qoi["alpha2_CZ_int"]
            + 9 / 15 * qoi["alpha12_CZ_int"]
        )

        qoi["eps_ref"] = depolarizing_par_to_eps(qoi["alpha_char"], d=4)
        qoi["eps_int"] = depolarizing_par_to_eps(qoi["alpha_char_CZ_int"], d=4)
        # Interleaved error calculation  Magesan et al. PRL 2012
        qoi["eps_CZ"] = 1 - (1 - qoi["eps_int"]) / (1 - qoi["eps_ref"])

    def prepare_plots(self):
        char_df = self.proc_data_dict["char_df"]

        # self.figs['puali_decays']
        self.plot_dicts["pauli_decays"] = {
            "plotfn": plot_char_RB_pauli_decays,
            "ncl": char_df.index.values,
            "P00": char_df["P00"].values,
            "P01": char_df["P01"].values,
            "P10": char_df["P10"].values,
            "P11": char_df["P11"].values,
            "P00_CZ": char_df["P00_CZ"].values,
            "P01_CZ": char_df["P01_CZ"].values,
            "P10_CZ": char_df["P10_CZ"].values,
            "P11_CZ": char_df["P11_CZ"].values,
            "title": self.raw_data_dict["measurementstring"]
            + "\n"
            + self.raw_data_dict["timestamp_string"]
            + "\nPauli decays",
        }
        self.plot_dicts["char_decay"] = {
            "plotfn": plot_char_RB_decay,
            "ncl": char_df.index.values,
            "C1": char_df["C1"].values,
            "C2": char_df["C2"].values,
            "C12": char_df["C12"].values,
            "C1_CZ": char_df["C1_CZ"].values,
            "C2_CZ": char_df["C2_CZ"].values,
            "C12_CZ": char_df["C12_CZ"].values,
            "fr_C1": self.fit_res["C1"],
            "fr_C2": self.fit_res["C2"],
            "fr_C12": self.fit_res["C12"],
            "fr_C1_CZ": self.fit_res["C1_CZ"],
            "fr_C2_CZ": self.fit_res["C2_CZ"],
            "fr_C12_CZ": self.fit_res["C12_CZ"],
            "title": self.raw_data_dict["measurementstring"]
            + "\n"
            + self.raw_data_dict["timestamp_string"]
            + "\nCharacter decay",
        }
        self.plot_dicts["quantities_msg"] = {
            "plotfn": plot_char_rb_quantities,
            "ax_id": "char_decay",
            "qoi": self.proc_data_dict["quantities_of_interest"],
        }


def plot_cal_points_hexbin(
    shots_0,
    shots_1,
    shots_2,
    xlabel: str,
    xunit: str,
    ylabel: str,
    yunit: str,
    title: str,
    ax,
    common_clims: bool = True,
    **kw
):
    # Choose colormap
    cmaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

    alpha_cmaps = []
    for cmap in cmaps:
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        alpha_cmaps.append(my_cmap)

    f = plt.gcf()
    mincnt = 1

    hbs = []
    shots_list = [shots_0, shots_1, shots_2]
    for i, shots in enumerate(shots_list):
        hb = ax.hexbin(
            x=shots[0],
            y=shots[1],
            cmap=alpha_cmaps[i],
            mincnt=mincnt,
            norm=PowerNorm(gamma=0.25),
        )
        cb = f.colorbar(hb, ax=ax)
        cb.set_label(r"Counts $|{}\rangle$".format(i))
        hbs.append(hb)

    if common_clims:
        clims = [hb.get_clim() for hb in hbs]
        clim = np.min(clims), np.max(clims)
        for hb in hbs:
            hb.set_clim(clim)

    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)


def plot_raw_RB_curve(
    ncl, SI, SX, V0, V1, V2, title, ax, xlabel, xunit, ylabel, yunit, **kw
):
    ax.plot(ncl, SI, label="SI", marker="o")
    ax.plot(ncl, SX, label="SX", marker="o")
    ax.plot(ncl[-1] + 0.5, V0, label="V0", marker="d", c="C0")
    ax.plot(ncl[-1] + 1.5, V1, label="V1", marker="d", c="C1")
    ax.plot(ncl[-1] + 2.5, V2, label="V2", marker="d", c="C2")
    ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.legend()


def plot_populations_RB_curve(ncl, P0, P1, P2, title, ax, **kw):
    ax.axhline(0.5, c="k", lw=0.5, ls="--")
    ax.plot(ncl, P0, c="C0", label=r"P($|g\rangle$)", marker="v")
    ax.plot(ncl, P1, c="C3", label=r"P($|e\rangle$)", marker="^")
    ax.plot(ncl, P2, c="C2", label=r"P($|f\rangle$)", marker="d")

    ax.set_xlabel("Number of Cliffords (#)")
    ax.set_ylabel("Population")
    ax.grid(axis="y")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)


def plot_unitarity_shots(ncl, unitarity_shots, title, ax=None, **kw):
    ax.axhline(0.5, c="k", lw=0.5, ls="--")

    ax.plot(ncl, unitarity_shots, ".")

    ax.set_xlabel("Number of Cliffords (#)")
    ax.set_ylabel("unitarity")
    ax.grid(axis="y")
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)


def plot_unitarity(ncl, P, title, ax=None, **kw):
    ax.plot(ncl, P, "o")

    ax.set_xlabel("Number of Cliffords (#)")
    ax.set_ylabel("unitarity")
    ax.grid(axis="y")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)


def plot_char_RB_pauli_decays(
    ncl, P00, P01, P10, P11, P00_CZ, P01_CZ, P10_CZ, P11_CZ, title, ax, **kw
):
    """
    Plots the raw recovery probabilities for a character RB experiment.
    """
    ax.plot(ncl, P00, c="C0", label=r"$P_{00}$", marker="o", ls="--")
    ax.plot(ncl, P01, c="C1", label=r"$P_{01}$", marker="o", ls="--")
    ax.plot(ncl, P10, c="C2", label=r"$P_{10}$", marker="o", ls="--")
    ax.plot(ncl, P11, c="C3", label=r"$P_{11}$", marker="o", ls="--")

    ax.plot(
        ncl, P00_CZ, c="C0", label=r"$P_{00}$-int. CZ", marker="d", alpha=0.5, ls=":"
    )
    ax.plot(
        ncl, P01_CZ, c="C1", label=r"$P_{01}$-int. CZ", marker="d", alpha=0.5, ls=":"
    )
    ax.plot(
        ncl, P10_CZ, c="C2", label=r"$P_{10}$-int. CZ", marker="d", alpha=0.5, ls=":"
    )
    ax.plot(
        ncl, P11_CZ, c="C3", label=r"$P_{11}$-int. CZ", marker="d", alpha=0.5, ls=":"
    )

    ax.set_xlabel("Number of Cliffords (#)")
    ax.set_ylabel(r"$P |00\rangle$")
    ax.legend(loc=(1.05, 0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)


def plot_char_RB_decay(
    ncl,
    C1,
    C2,
    C12,
    C1_CZ,
    C2_CZ,
    C12_CZ,
    fr_C1,
    fr_C2,
    fr_C12,
    fr_C1_CZ,
    fr_C2_CZ,
    fr_C12_CZ,
    title,
    ax,
    **kw
):

    ncl_fine = np.linspace(np.min(ncl), np.max(ncl), 101)

    plot_fit(ncl_fine, fr_C1, ax, ls="-", c="C0")
    ax.plot(
        ncl, C1, c="C0", label=r"$C_1$: $A_1\cdot {\alpha_{1|2}}^m$", marker="o", ls=""
    )
    plot_fit(ncl_fine, fr_C2, ax, ls="-", c="C1")
    ax.plot(
        ncl, C2, c="C1", label=r"$C_2$: $A_1\cdot {\alpha_{2|1}}^m$", marker="o", ls=""
    )
    plot_fit(ncl_fine, fr_C12, ax, ls="-", c="C2")
    ax.plot(
        ncl,
        C12,
        c="C2",
        label=r"$C_{12}$: $A_1\cdot {\alpha_{12}}^m$",
        marker="o",
        ls="",
    )

    plot_fit(ncl_fine, fr_C1_CZ, ax, ls="--", c="C0", alpha=0.5)
    ax.plot(
        ncl,
        C1_CZ,
        c="C0",
        label=r"$C_1^{int.}$: $A_1' \cdot {\alpha_{1|2}'}^m$",
        marker="d",
        ls="",
        alpha=0.5,
    )
    plot_fit(ncl_fine, fr_C2_CZ, ax, ls="--", c="C1", alpha=0.5)
    ax.plot(
        ncl,
        C2_CZ,
        c="C1",
        label=r"$C_2^{int.}$: $A_2' \cdot {\alpha_{2|1}'}^m$",
        marker="d",
        ls="",
        alpha=0.5,
    )
    plot_fit(ncl_fine, fr_C12_CZ, ax, ls="--", c="C2", alpha=0.5)
    ax.plot(
        ncl,
        C12_CZ,
        c="C2",
        label=r"$C_{12}^{int.}$: $A_{12}' \cdot {\alpha_{12}'}^m$",
        marker="d",
        ls="",
        alpha=0.5,
    )

    ax.set_xlabel("Number of Cliffords (#)")
    ax.set_ylabel("Population")
    ax.legend(title="Character decay", ncol=2, loc=(1.05, 0.6))

    ax.set_title(title)


def plot_char_rb_quantities(ax, qoi, **kw):
    """
    Plots a text message of the main quantities extracted from char rb
    """

    def gen_val_str(alpha, alpha_p):

        val_str = "   {:.3f}$\pm${:.3f}    {:.3f}$\pm${:.3f}"
        return val_str.format(
            alpha.nominal_value, alpha.std_dev, alpha_p.nominal_value, alpha_p.std_dev
        )

    alpha_msg = "            Reference         Interleaved"
    alpha_msg += "\n" r"$\alpha_{1|2}$" + "\t"
    alpha_msg += gen_val_str(qoi["alpha1"], qoi["alpha1_CZ_int"])
    alpha_msg += "\n" r"$\alpha_{2|1}$" + "\t"
    alpha_msg += gen_val_str(qoi["alpha2"], qoi["alpha2_CZ_int"])
    alpha_msg += "\n" r"$\alpha_{12}$" + "\t"
    alpha_msg += gen_val_str(qoi["alpha12"], qoi["alpha12_CZ_int"])
    alpha_msg += "\n" + "_" * 40 + "\n"

    alpha_msg += "\n" r"$\epsilon_{Ref.}$" + "\t"
    alpha_msg += "{:.3f}$\pm${:.3f}%".format(
        qoi["eps_ref"].nominal_value * 100, qoi["eps_ref"].std_dev * 100
    )
    alpha_msg += "\n" r"$\epsilon_{Int.}$" + "\t"
    alpha_msg += "{:.3f}$\pm${:.3f}%".format(
        qoi["eps_int"].nominal_value * 100, qoi["eps_int"].std_dev * 100
    )
    alpha_msg += "\n" r"$\epsilon_{CZ.}$" + "\t"
    alpha_msg += "{:.3f}$\pm${:.3f}%".format(
        qoi["eps_CZ"].nominal_value * 100, qoi["eps_CZ"].std_dev * 100
    )

    ax.text(1.05, 0.0, alpha_msg, transform=ax.transAxes)


def logisticreg_classifier_machinelearning(shots_0, shots_1, shots_2):
    """
    """
    # reshaping of the entries in proc_data_dict
    shots_0 = np.array(list(zip(list(shots_0.values())[0], list(shots_0.values())[1])))

    shots_1 = np.array(list(zip(list(shots_1.values())[0], list(shots_1.values())[1])))
    shots_2 = np.array(list(zip(list(shots_2.values())[0], list(shots_2.values())[1])))

    shots_0 = shots_0[~np.isnan(shots_0[:, 0])]
    shots_1 = shots_1[~np.isnan(shots_1[:, 0])]
    shots_2 = shots_2[~np.isnan(shots_2[:, 0])]

    X = np.concatenate([shots_0, shots_1, shots_2])
    Y = np.concatenate(
        [
            0 * np.ones(shots_0.shape[0]),
            1 * np.ones(shots_1.shape[0]),
            2 * np.ones(shots_2.shape[0]),
        ]
    )

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    return logreg


def plot_classifier_decission_boundary(
    shots_0,
    shots_1,
    shots_2,
    classifier,
    xlabel: str,
    xunit: str,
    ylabel: str,
    yunit: str,
    title: str,
    ax,
    **kw
):
    """
    Plot decision boundary on top of the hexbin plot of the training dataset.
    """
    grid_points = 200

    x_min = np.nanmin([shots_0[0], shots_1[0], shots_2[0]])
    x_max = np.nanmax([shots_0[0], shots_1[0], shots_2[0]])
    y_min = np.nanmin([shots_0[1], shots_1[1], shots_2[1]])
    y_max = np.nanmax([shots_0[1], shots_1[1], shots_2[1]])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points), np.linspace(y_min, y_max, grid_points)
    )
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plot_cal_points_hexbin(
        shots_0=shots_0,
        shots_1=shots_1,
        shots_2=shots_2,
        xlabel=xlabel,
        xunit=xunit,
        ylabel=ylabel,
        yunit=yunit,
        title=title,
        ax=ax,
    )
    ax.pcolormesh(xx, yy, Z, cmap=c.ListedColormap(["C0", "C3", "C2"]), alpha=0.2)


def plot_rb_decay_woods_gambetta(ncl, M0, X1, ax, ax1, title="", **kw):
    ax.plot(ncl, M0, marker="o", linestyle="")
    ax1.plot(ncl, X1, marker="d", linestyle="")
    ax.grid(axis="y")
    ax1.grid(axis="y")
    ax.set_ylim(-0.05, 1.05)
    ax1.set_ylim(min(min(0.97 * X1), 0.92), 1.01)
    ax.set_ylabel(r"$M_0$ probability")
    ax1.set_ylabel(r"$\chi_1$ population")
    ax1.set_xlabel("Number of Cliffords")
    ax.set_title(title)


def plot_irb_decay_woods_gambetta(
    ncl,
    M0_ref,
    M0_int,
    X1_ref,
    X1_int,
    fr_M0_ref,
    fr_M0_int,
    fr_M0_simple_ref,
    fr_M0_simple_int,
    fr_X1_ref,
    fr_X1_int,
    qoi,
    ax,
    ax1,
    fit_tag,
    int_name,
    title="",
    include_idle=False,
    M0_int_idle=None,
    X1_int_idle=None,
    fr_M0_int_idle=None,
    fr_M0_simple_int_idle=None,
    fr_X1_int_idle=None,
    **kw
):
    ncl_fine = np.linspace(ncl[0], ncl[-1], 1001)

    ax.plot(ncl, M0_ref, marker="o", linestyle="", c="C0", label="Reference")
    plot_fit(ncl_fine, fr_M0_ref, ax=ax, c="C0")

    ax.plot(
        ncl,
        M0_int,
        marker="d",
        linestyle="",
        c="C1",
        label="Interleaved {}".format(int_name),
    )
    plot_fit(ncl_fine, fr_M0_int, ax=ax, c="C1")
    if include_idle:
        ax.plot(
            ncl, M0_int_idle, marker="^", linestyle="", c="C2", label="Interleaved Idle"
        )
        plot_fit(ncl_fine, fr_M0_int_idle, ax=ax, c="C2")

    ax.grid(axis="y")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$M_0$ probability")

    ax1.plot(ncl, X1_ref, marker="o", linestyle="", c="C0")
    ax1.plot(ncl, X1_int, marker="d", linestyle="", c="C1")
    plot_fit(ncl_fine, fr_X1_ref, ax=ax1, c="C0")
    plot_fit(ncl_fine, fr_X1_int, ax=ax1, c="C1")

    if include_idle:
        ax1.plot(ncl, X1_int_idle, marker="^", linestyle="", c="C2")
        plot_fit(ncl_fine, fr_X1_int_idle, ax=ax1, c="C2")

    ax1.grid(axis="y")

    ax1.set_ylim(min(min(0.97 * X1_int), 0.92), 1.01)
    ax1.set_ylabel(r"$\chi_1$ population")
    ax1.set_xlabel("Number of Cliffords")
    ax.set_title(title)
    ax.legend(loc="best")

    collabels = [r"$\epsilon_{\chi1}~(\%)$", r"$\epsilon~(\%)$", r"$L_1~(\%)$"]

    idle_r_labels0 = ["Interl. Idle curve"] if include_idle else []
    idle_r_labels1 = ["Idle-interleaved"] if include_idle else []

    rowlabels = (
        ["Ref. curve"]
        + idle_r_labels0
        + ["Interl. {} curve".format(int_name)]
        + idle_r_labels1
        + ["{}-interleaved".format(int_name)]
    )

    if int_name == "CZ":
        rowlabels += ["{}-naive".format(int_name)]

    idle_r_extracted = (
        [[qoi["eps_idle_X1"] * 100, qoi["eps_idle_simple"] * 100, qoi["L1_idle"] * 100]]
        if include_idle
        else []
    )

    idle_r_fit = (
        [
            [
                qoi["eps_X1_{}_int_idle".format(fit_tag)] * 100,
                qoi["eps_simple_{}_int_idle".format(fit_tag)] * 100,
                qoi["L1_{}_int_idle".format(fit_tag)] * 100,
            ]
        ]
        if include_idle
        else []
    )

    table_data = (
        [
            [
                qoi["eps_X1_{}_ref".format(fit_tag)] * 100,
                qoi["eps_simple_{}_ref".format(fit_tag)] * 100,
                qoi["L1_{}_ref".format(fit_tag)] * 100,
            ]
        ]
        + idle_r_fit
        + [
            [
                qoi["eps_X1_{}_int".format(fit_tag)] * 100,
                qoi["eps_simple_{}_int".format(fit_tag)] * 100,
                qoi["L1_{}_int".format(fit_tag)] * 100,
            ]
        ]
        + idle_r_extracted
        + [
            [
                qoi["eps_{}_X1".format(int_name)] * 100,
                qoi["eps_{}_simple".format(int_name)] * 100,
                qoi["L1_{}".format(int_name)] * 100,
            ]
        ]
    )

    if int_name == "CZ":
        table_data += [
            [
                qoi["eps_{}_X1_naive".format(int_name)] * 100,
                qoi["eps_{}_simple_naive".format(int_name)] * 100,
                qoi["L1_{}_naive".format(int_name)] * 100,
            ]
        ]

    # Avoid too many digits when the uncertainty is np.nan
    for i, row in enumerate(table_data):
        for j, u_val in enumerate(row):
            if np.isnan(u_val.n) and np.isnan(u_val.s):
                table_data[i][j] = "nan+/-nan"
            elif np.isnan(u_val.s):
                # Keep 3 significant digits only
                table_data[i][j] = "{:.3g}+/-nan".format(u_val.n)

    ax1.table(
        cellText=table_data,
        colLabels=collabels,
        rowLabels=rowlabels,
        transform=ax1.transAxes,
        cellLoc="center",
        rowLoc="center",
        bbox=(0.1, -2.5, 1, 2),
    )


def interleaved_error(eps_int, eps_base):
    # Interleaved error calculation  Magesan et al. PRL 2012
    eps = 1 - (1 - eps_int) / (1 - eps_base)
    return eps


def leak_decay(A, B, lambda_1, m):
    """
    Eq. (9) of Wood Gambetta 2018.

        A ~= L2/ (L1+L2)
        B ~= L1/ (L1+L2) + eps_m
        lambda_1 = 1 - L1 - L2

    """
    return A + B * lambda_1 ** m


def full_rb_decay(A, B, C, lambda_1, lambda_2, m):
    """Eq. (15) of Wood Gambetta 2018."""
    return A + B * lambda_1 ** m + C * lambda_2 ** m


def unitarity_decay(A, B, u, m):
    """Eq. (8) of Wallman et al. New J. Phys. 2015."""
    return A + B * u ** m


def char_decay(A, alpha, m):
    """
    From Helsen et al. A new class of efficient RB protocols.

    Theory in Helsen et al. arXiv:1806.02048
    Eq. 8 of Xue et al. ArXiv 1811.04002v1 (experimental implementation)

    Parameters
    ----------
    A (float):
        Scaling factor of the decay
    alpha (float):
        depolarizing parameter to be estimated
    m (array)
        number of cliffords

    returns:
       A * α**m
    """
    return A * alpha ** m


def depolarizing_par_to_eps(alpha, d):
    """
    Convert depolarizing parameter to infidelity.

    Dugas et al.  arXiv:1610.05296v2 contains a nice overview table of
    common RB paramater conversions.

    Parameters
    ----------
    alpha (float):
        depolarizing parameter, also commonly referred to as lambda or p.
    d (int):
        dimension of the system, 2 for a single qubit, 4 for two-qubits.

    Returns
    -------
        eps = (1-alpha)*(d-1)/d

    """
    return (1 - alpha) * (d - 1) / d
