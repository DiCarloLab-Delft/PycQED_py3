from importlib import reload
import lmfit
import numpy as np
from uncertainties import ufloat
from scipy.stats import sem
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
reload(fit_mods)
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from pycqed.utilities.general import format_value_string
from copy import deepcopy
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor

from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor


class Single_Qubit_TimeDomainAnalysis(ba.BaseDataAnalysis):
    def process_data(self):
        """
        This takes care of rotating and normalizing the data if required.
        this should work for several input types.
            - I/Q values (2 quadratures + cal points)
            - weight functions (1 quadrature + cal points)
            - counts (no cal points)

        There are several options possible to specify the normalization
        using the options dict.
            cal_points (tuple) of indices of the calibration points

            zero_coord, one_coord
        """

        cal_points = self.options_dict.get("cal_points", None)
        zero_coord = self.options_dict.get("zero_coord", None)
        one_coord = self.options_dict.get("one_coord", None)

        # FIXME THIS IS A HACK related to recent issue
        self.data_dict = self.raw_data_dict
        if cal_points is None:
            # default for all standard Timedomain experiments
            cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if len(self.raw_data_dict["measured_values"]) == 1:
            # if only one weight function is used rotation is not required
            self.proc_data_dict["corr_data"] = a_tools.normalize_data_v3(
                self.raw_data_dict["measured_values"][0],
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1],
            )
        else:
            (
                self.proc_data_dict["corr_data"],
                zero_coord,
                one_coord,
            ) = a_tools.rotate_and_normalize_data(
                data=self.raw_data_dict["measured_values"][0:2],
                zero_coord=zero_coord,
                one_coord=one_coord,
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1],
            )

        # This should be added to the hdf5 datafile but cannot because of the
        # way that the "new" analysis works. Thijs: "? I dont get thiscomment, just
        # do it like this:"
        self.proc_data_dict['quantities_of_interest'] = {'Corrected data': self.proc_data_dict['corr_data']}
        # self.add_dataset_to_analysisgroup('Corrected data',
        #                                   self.proc_data_dict['corr_data'])

    def prepare_plots(self):
        self.plot_dicts["raw_data"] = {
            "plotfn": self.plot_line,
            "xvals": self.raw_data_dict["sweep_points"],
            "xlabel": self.raw_data_dict["xlabel"],
            "xunit": self.raw_data_dict["xunit"],  # does not do anything yet
            "yvals": self.proc_data_dict["corr_data"],
            "ylabel": "Excited state population",
            "yunit": "",
            "setlabel": "data",
            "title": (
                self.raw_data_dict["timestamp"]
                + " "
                + self.raw_data_dict["measurementstring"]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }
        for i, name in enumerate(pdict_names):
            combined_name = 'combined_' + name
            self.axs[combined_name] = axs[i]
            self.plot_dicts[combined_name] = self.plot_dicts[name].copy()
            self.plot_dicts[combined_name]['ax_id'] = combined_name

            # shorter label as the axes are now shared
            self.plot_dicts[combined_name]['ylabel'] = name
            self.plot_dicts[combined_name]['xlabel'] = None if i in [
                0, 1, 2, 3] else self.plot_dicts[combined_name]['xlabel']
            self.plot_dicts[combined_name]['title'] = None if i in [
                0, 1, 2, 3] else self.plot_dicts[combined_name]['title']
            self.plot_dicts[combined_name]['touching'] = True



class Idling_Error_Rate_Analyisis(ba.BaseDataAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
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
            do_fitting=do_fitting,
        )

        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "xvals": "sweep_points",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        post_sel_th = self.options_dict.get("post_sel_th", 0.5)
        raw_shots = self.raw_data_dict["measured_values"][0][0]
        post_sel_shots = raw_shots[::2]
        data_shots = raw_shots[1::2]
        data_shots[np.where(post_sel_shots > post_sel_th)] = np.nan

        states = ["0", "1", "+"]
        self.proc_data_dict["xvals"] = np.unique(self.raw_data_dict["xvals"])
        for i, state in enumerate(states):
            self.proc_data_dict["shots_{}".format(state)] = data_shots[i::3]

            self.proc_data_dict["yvals_{}".format(state)] = np.nanmean(
                np.reshape(
                    self.proc_data_dict["shots_{}".format(state)],
                    (len(self.proc_data_dict["xvals"]), -1),
                    order="F",
                ),
                axis=1,
            )

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        states = ["0", "1", "+"]
        for i, state in enumerate(states):
            yvals = self.proc_data_dict["yvals_{}".format(state)]
            xvals = self.proc_data_dict["xvals"]

            self.plot_dicts["Prepare in {}".format(state)] = {
                "ax_id": "main",
                "plotfn": self.plot_line,
                "xvals": xvals,
                "xlabel": self.raw_data_dict["xlabel"][0],
                "xunit": self.raw_data_dict["xunit"][0][0],
                "yvals": yvals,
                "ylabel": "Counts",
                "yrange": [0, 1],
                "xrange": self.options_dict.get("xrange", None),
                "yunit": "frac",
                "setlabel": "Prepare in {}".format(state),
                "do_legend": True,
                "title": (
                    self.raw_data_dict["timestamps"][0]
                    + " - "
                    + self.raw_data_dict["timestamps"][-1]
                    + "\n"
                    + self.raw_data_dict["measurementstring"][0]
                ),
                "legend_pos": "upper right",
            }
        if self.do_fitting:
            for state in ["0", "1", "+"]:
                self.plot_dicts["fit_{}".format(state)] = {
                    "ax_id": "main",
                    "plotfn": self.plot_fit,
                    "fit_res": self.fit_dicts["fit {}".format(state)]["fit_res"],
                    "plot_init": self.options_dict["plot_init"],
                    "setlabel": "fit |{}>".format(state),
                    "do_legend": True,
                    "legend_pos": "upper right",
                }

                self.plot_dicts["fit_text"] = {
                    "ax_id": "main",
                    "box_props": "fancy",
                    "xpos": 1.05,
                    "horizontalalignment": "left",
                    "plotfn": self.plot_text,
                    "text_string": self.proc_data_dict["fit_msg"],
                }

    def analyze_fit_results(self):
        fit_msg = ""
        states = ["0", "1", "+"]
        for state in states:
            fr = self.fit_res["fit {}".format(state)]

            fit_msg += "Prep |{}> :\n\t"
            fit_msg += format_value_string("$N_1$", fr.params["N1"], end_char="\n\t")
            fit_msg += format_value_string("$N_2$", fr.params["N2"], end_char="\n")

        self.proc_data_dict["fit_msg"] = fit_msg

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        states = ["0", "1", "+"]
        for i, state in enumerate(states):
            yvals = self.proc_data_dict["yvals_{}".format(state)]
            xvals = self.proc_data_dict["xvals"]

            mod = lmfit.Model(fit_mods.idle_error_rate_exp_decay)
            mod.guess = fit_mods.idle_err_rate_guess.__get__(mod, mod.__class__)

            # Done here explicitly so that I can overwrite a specific guess
            guess_pars = mod.guess(N=xvals, data=yvals)
            vary_N2 = self.options_dict.get("vary_N2", True)

            if not vary_N2:
                guess_pars["N2"].value = 1e21
                guess_pars["N2"].vary = False
            # print(guess_pars)
            self.fit_dicts["fit {}".format(states[i])] = {
                "model": mod,
                "fit_xvals": {"N": xvals},
                "fit_yvals": {"data": yvals},
                "guess_pars": guess_pars,
            }
            # Allows fixing the double exponential coefficient


class Grovers_TwoQubitAllStates_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs: bool = True,
        auto=True,
    ):
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            options_dict=options_dict,
            close_figs=close_figs,
            extract_only=extract_only,
            do_fitting=True,
        )

        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "xvals": "sweep_points",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        normalize_to_cal_points = self.options_dict.get("normalize_to_cal_points", True)
        cal_points = [
            [[-4, -3], [-2, -1]],
            [[-4, -2], [-3, -1]],
        ]
        for idx in [0, 1]:
            yvals = list(self.raw_data_dict["measured_values_ord_dict"].values())[idx][
                0
            ]

            self.proc_data_dict["ylabel_{}".format(idx)] = self.raw_data_dict[
                "value_names"
            ][0][idx]
            self.proc_data_dict["yunit"] = self.raw_data_dict["value_units"][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.normalize_data_v3(
                    yvals,
                    cal_zero_points=cal_points[idx][0],
                    cal_one_points=cal_points[idx][1],
                )
            self.proc_data_dict["yvals_{}".format(idx)] = yvals

        y0 = self.proc_data_dict["yvals_0"]
        y1 = self.proc_data_dict["yvals_1"]
        p_success = (
            (y0[0] * y1[0])
            + (1 - y0[1]) * y1[1]
            + (y0[2]) * (1 - y1[2])
            + (1 - y0[3]) * (1 - y1[3])
        ) / 4
        print(y0[0] * y1[0])
        print((1 - y0[1]) * y1[1])
        print((y0[2]) * (1 - y1[2]))
        print((1 - y0[3]) * (1 - y1[3]))
        self.proc_data_dict["p_success"] = p_success

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i in [0, 1]:
            yvals = self.proc_data_dict["yvals_{}".format(i)]
            xvals = self.raw_data_dict["xvals"][0]
            ylabel = self.proc_data_dict["ylabel_{}".format(i)]
            self.plot_dicts["main_{}".format(ylabel)] = {
                "plotfn": self.plot_line,
                "xvals": self.raw_data_dict["xvals"][0],
                "xlabel": self.raw_data_dict["xlabel"][0],
                "xunit": self.raw_data_dict["xunit"][0][0],
                "yvals": self.proc_data_dict["yvals_{}".format(i)],
                "ylabel": ylabel,
                "yunit": self.proc_data_dict["yunit"],
                "title": (
                    self.raw_data_dict["timestamps"][0]
                    + " \n"
                    + self.raw_data_dict["measurementstring"][0]
                ),
                "do_legend": False,
                "legend_pos": "upper right",
            }

        self.plot_dicts["limit_text"] = {
            "ax_id": "main_{}".format(ylabel),
            "box_props": "fancy",
            "xpos": 1.05,
            "horizontalalignment": "left",
            "plotfn": self.plot_text,
            "text_string": "P succes = {:.3f}".format(self.proc_data_dict["p_success"]),
        }


class FlippingAnalysis(Single_Qubit_TimeDomainAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        do_fitting: bool = True,
        auto=True,
    ):
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            data_file_path=data_file_path,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
        )
        self.single_timestamp = True

        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "measurementstring": "measurementstring",
            "sweep_points": "sweep_points",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict["cal_points"] = [list(range(-4, -2)), list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # Even though we expect an exponentially damped oscillation we use
        # a simple cosine as this gives more reliable fitting and we are only
        # interested in extracting the frequency of the oscillation
        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(
            model=cos_mod,
            t=self.raw_data_dict["sweep_points"][:-4],
            data=self.proc_data_dict["corr_data"][:-4],
        )

        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars["amplitude"].value = 0.5
        guess_pars["amplitude"].vary = True
        guess_pars["offset"].value = 0.5
        guess_pars["offset"].vary = True

        self.fit_dicts["cos_fit"] = {
            "fit_fn": fit_mods.CosFunc,
            "fit_xvals": {"t": self.raw_data_dict["sweep_points"][:-4]},
            "fit_yvals": {"data": self.proc_data_dict["corr_data"][:-4]},
            "guess_pars": guess_pars,
        }

        # In the case there are very few periods we fall back on a small
        # angle approximation to extract the drive detuning
        poly_mod = lmfit.models.PolynomialModel(degree=1)
        # the detuning can be estimated using on a small angle approximation
        # c1 = d/dN (cos(2*pi*f N) ) evaluated at N = 0 -> c1 = -2*pi*f
        poly_mod.set_param_hint("frequency", expr="-c1/(2*pi)")
        guess_pars = poly_mod.guess(
            x=self.raw_data_dict["sweep_points"][:-4],
            data=self.proc_data_dict["corr_data"][:-4],
        )
        # Constraining the line ensures that it will only give a good fit
        # if the small angle approximation holds
        guess_pars["c0"].vary = True
        guess_pars["c0"].value = 0.5

        self.fit_dicts["line_fit"] = {
            "model": poly_mod,
            "fit_xvals": {"x": self.raw_data_dict["sweep_points"][:-4]},
            "fit_yvals": {"data": self.proc_data_dict["corr_data"][:-4]},
            "guess_pars": guess_pars,
        }

    def analyze_fit_results(self):
        sf_line = self._get_scale_factor_line()
        sf_cos = self._get_scale_factor_cos()
        self.proc_data_dict["scale_factor"] = self.get_scale_factor()

        msg = "Scale fact. based on "
        if self.proc_data_dict["scale_factor"] == sf_cos:
            msg += "cos fit\n"
        else:
            msg += "line fit\n"
        msg += "cos fit: {:.4f}\n".format(sf_cos)
        msg += "line fit: {:.4f}".format(sf_line)

        self.raw_data_dict["scale_factor_msg"] = msg
        # TODO: save scale factor to file

    def get_scale_factor(self):
        """
        Returns the scale factor that should correct for the error in the
        pulse amplitude.
        """
        # Model selection based on the Bayesian Information Criterion (BIC)
        # as  calculated by lmfit
        if (
            self.fit_dicts["line_fit"]["fit_res"].bic
            < self.fit_dicts["cos_fit"]["fit_res"].bic
        ):
            scale_factor = self._get_scale_factor_line()
        else:
            scale_factor = self._get_scale_factor_cos()
        return scale_factor

    def _get_scale_factor_cos(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts["cos_fit"]["fit_res"].params["frequency"]

        # the square is needed to account for the difference between
        # power and amplitude
        scale_factor = (1 + frequency) ** 2

        phase = np.rad2deg(self.fit_dicts["cos_fit"]["fit_res"].params["phase"]) % 360
        # phase ~90 indicates an under rotation so the scale factor
        # has to be larger than 1. A phase ~270 indicates an over
        # rotation so then the scale factor has to be smaller than one.
        if phase > 180:
            scale_factor = 1 / scale_factor

        return scale_factor

    def _get_scale_factor_line(self):
        # 2/period (ref is 180 deg) of the oscillation corresponds
        # to the (fractional) over/under rotation error per gate
        frequency = self.fit_dicts["line_fit"]["fit_res"].params["frequency"]
        scale_factor = (1 + 2 * frequency) ** 2
        # no phase sign check is needed here as this is contained in the
        # sign of the coefficient

        return scale_factor

    def prepare_plots(self):
        self.plot_dicts["main"] = {
            "plotfn": self.plot_line,
            "xvals": self.raw_data_dict["sweep_points"],
            "xlabel": self.raw_data_dict["xlabel"],
            "xunit": self.raw_data_dict["xunit"],  # does not do anything yet
            "yvals": self.proc_data_dict["corr_data"],
            "ylabel": "Excited state population",
            "yunit": "",
            "setlabel": "data",
            "title": (
                self.raw_data_dict["timestamp"]
                + " "
                + self.raw_data_dict["measurementstring"]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }

        if self.do_fitting:
            self.plot_dicts["line_fit"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["line_fit"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "line fit",
                "do_legend": True,
                "legend_pos": "upper right",
            }

            self.plot_dicts["cos_fit"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["cos_fit"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "cos fit",
                "do_legend": True,
                "legend_pos": "upper right",
            }

            self.plot_dicts["text_msg"] = {
                "ax_id": "main",
                "ypos": 0.15,
                "plotfn": self.plot_text,
                "box_props": "fancy",
                "text_string": self.raw_data_dict["scale_factor_msg"],
            }



class EFRabiAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 close_figs: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs,
                         extract_only=extract_only, do_fitting=True)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # Even though we expect an exponentially damped oscillation we use
        # a simple cosine as this gives more reliable fitting and we are only
        # interested in extracting the frequency of the oscillation
        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.raw_data_dict['sweep_points'][:-4],
            data=self.proc_data_dict['corr_data'][:-4])

        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars['amplitude'].value = 0.5
        guess_pars['amplitude'].vary = True
        guess_pars['amplitude'].min = -10
        guess_pars['amplitude'].max = 10
        guess_pars['offset'].value = 0.5
        guess_pars['offset'].vary = True
        guess_pars['phase'].value = 0
        guess_pars['phase'].vary = False

        self.fit_dicts['cos_fit'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}


    def analyze_fit_results(self):
        sf_cos = self._get_ef_pi_amp()
        self.proc_data_dict['ef_pi_amp'] = sf_cos

        msg = r'$\pi$-ef amp '
        msg += ': {:.4f}\n'.format(sf_cos)


        self.raw_data_dict['scale_factor_msg'] = msg
        # TODO: save scale factor to file
        return sf_cos


    def _get_ef_pi_amp(self):

        frequency = self.fit_dicts['cos_fit']['fit_res'].params['frequency']
        # calculate the pi pulse amplitude using 2* pi *f* amp = pi/2
        ef_pi_amp = 1/(2*frequency)

        return ef_pi_amp



    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['corr_data'],
            'ylabel': 'Normalized data',
            'yunit': '',
            'setlabel': 'data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'do_legend': True,
            'legend_pos': 'best'}

        if self.do_fitting:

            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'cos fit',
                'do_legend': True,
                'legend_pos': 'best'}

            self.plot_dicts['pi_amp'] = {
                'plotfn': self.plot_line,
                'ax_id': 'main',
                'xvals': [self.proc_data_dict['ef_pi_amp']],
                # 'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'],  # does not do anything yet
                'yvals': [fit_mods.CosFunc(self.proc_data_dict['ef_pi_amp'],
                    **self.fit_dicts['cos_fit']['fit_res'].best_values)],
                'marker':'o',
                'line_kws':{'markersize':10}}
            self.plot_dicts['text_msg'] = {
                'ax_id': 'main',
                'ypos': 0.15,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'text_string': self.raw_data_dict['scale_factor_msg']}



class DecoherenceAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 close_figs: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs,
                         extract_only=extract_only, do_fitting=True)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # The fitting function is a cosine that decays exponentionally

        # Make model from function
        mod_Fit_Func = lmfit.Model(fit_mods.ExpGaussDecayCos)

        # Recover parameters to fit over
        guess_pars = fit_mods.ExpGaussDecayCos_guess(
        model=mod_Fit_Func, t=self.raw_data_dict['sweep_points'][:-4],
        data=self.proc_data_dict['corr_data'][:-4])


        # Result of the fit
        fit_res = mod_Fit_Func.fit(self.proc_data_dict['corr_data'][:-4], guess_pars, t = self.raw_data_dict['sweep_points'][:-4])


        self.fit_dicts['ExpGaussDecayCos'] = {
            'fit_fn': fit_mods.ExpGaussDecayCos,
            'fit_xvals': {'t': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'fit_res': {'res': fit_res},
            'guess_pars': guess_pars}

        # Chisqr is saved in quantities of interest to see how well the fit works
        self.proc_data_dict['quantities_of_interest'] = {
            'Chisqr': {'chisqr' :fit_res.chisqr}}




    def analyze_fit_results(self):
        sf_cos = self._get_ef_pi_amp()
        self.proc_data_dict['ef_pi_amp'] = sf_cos

        msg = r'$\pi$-ef amp '
        msg += ': {:.4f}\n'.format(sf_cos)


        self.raw_data_dict['scale_factor_msg'] = msg
        # TODO: save scale factor to file
        return sf_cos


    def _get_ef_pi_amp(self):

        frequency = self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['frequency']
        # calculate the pi pulse amplitude using 2* pi *f* amp = pi/2
        ef_pi_amp = 1/(2*frequency)

        return ef_pi_amp



    def prepare_plots(self):
        self.plot_dicts = OrderedDict()


        # Plot the normalized measured data
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'][:-4],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['corr_data'][:-4],
            'ylabel': 'Normalized Data',
            # 'yunit': '',
            'marker': 'o',
            'linestyle': '-',
            'setlabel': 'Measured Data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            # 'plotsize': (20,10),
            'do_legend': True,
            'legend_pos': (1.04,.786),
            'legend_title': 'hoi'}


        # Plot calibration points seperately so it is clear that it isn't just some crappy data
        self.plot_dicts['calibration'] = {
            'ax_id': 'main',
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'][-4:],
            'yvals': self.proc_data_dict['corr_data'][-4:],
            'marker': 'o',
            'linestyle': '',
            'setlabel': 'Calibration Points',
            'do_legend': True,
            'legend_pos': (1.04,.786)}



        if self.do_fitting:
            # Initialize parameters for pure decay curves (so freq = 0)
            # Get values of fit and get rid of frequency

            # First make deepcopy of results
            pars_decay = deepcopy(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params)
            # Then get rid of the frequency
            pars_decay.pop('frequency', None)

            # Wanted to initialize x_vals for decay curve to plot the function ExpGaussDecay_only
            # But then the legend will freak out
            # decay_x_vals = np.linspace(self.raw_data_dict['sweep_points'][0],self.raw_data_dict['sweep_points'][-5],1000)



            # Plot the fit function using the resulted fitted parameters
            self.plot_dicts['Fit_Func'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['ExpGaussDecayCos']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'linestyle': '-',
                'setlabel': 'Fit Function',
                'do_legend': True,
                'legend_pos': (1.04,.786),}



            # Pure decay curves
            # Top curve
            self.plot_dicts['Decay1'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': np.linspace(self.raw_data_dict['sweep_points'][0],self.raw_data_dict['sweep_points'][-5],1000),
                'yvals': fit_mods.ExpGaussDecay_only(np.linspace(self.raw_data_dict['sweep_points'][0],self.raw_data_dict['sweep_points'][-5],1000), *np.array(list(pars_decay.valuesdict().values()))),
                'marker': '',
                'color': 'r',
                'setlabel': '',
                'linestyle': '-'}

            # Bottom curve
            pars_decay['amplitude'].value = -pars_decay['amplitude'].value

            self.plot_dicts['Decay2'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': np.linspace(self.raw_data_dict['sweep_points'][0],self.raw_data_dict['sweep_points'][-5],1000),
                'yvals': fit_mods.ExpGaussDecay_only(np.linspace(self.raw_data_dict['sweep_points'][0],self.raw_data_dict['sweep_points'][-5],1000), *np.array(list(pars_decay.valuesdict().values()))),
                'marker': '',
                'color': 'r',
                'linestyle': '-',
                'setlabel': 'Decay Curve',
                'do_legend': True,
                'legend_pos': (1.04,.75)}

            del pars_decay

            # Show the fit function that was used
            self.plot_dicts['Fit_Def'] = {
                'ax_id': 'main',
                'ypos': .7,
                'xpos': 1.04,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'horizontalalignment': 'left',
                'text_string': 'The fit function is defined as' + '\n' +
                 '$A e^{-t\Gamma_{exp} - (t \Gamma_{\phi})^2}\cos(2 \pi f t) + Off$'}





            # Cool box showing all fit results and chi_sqr
            self.plot_dicts['Parameters'] = {
                'ax_id': 'main',
                'ypos': .5,
                'xpos': 1.04,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'horizontalalignment': 'left',
                # 'text_string': 'Chi = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr),
                'text_string': 'Fit results' + '\n'
                                + '$\chi^2$ = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr) + '\n'
                                + '$\Gamma_{exp}$ = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['Gexp'].value) + ' $s^{-1}$' + '\n'
                                + '$\Gamma_{\phi}$ = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['Gphi'].value) + ' $s^{-1}$' + '\n'
                                + 'A = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['amplitude'].value) + '\n'
                                + 'Off = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['offset'].value) + '\n'
                                + 'f = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].params['frequency'].value) + 'Hz'}






class ComplexRamseyAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 close_figs: bool=True, auto=True,do_fitting=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_A" and "ch_idx_B"
        specified in the options dict. If ch_idx_A and ch_idx_B are the same
        it will unzip the data.
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        indices_I = np.hstack([np.arange(0,len(self.proc_data_dict['sweep_points'])-4,2),
                     np.arange(len(self.proc_data_dict['sweep_points'])-4,
                               len(self.proc_data_dict['sweep_points']))])
        self.proc_data_dict['data_I_I'] = self.proc_data_dict['measured_values'][0][indices_I]
        self.proc_data_dict['data_Q_I'] = self.proc_data_dict['measured_values'][1][indices_I]
        self.proc_data_dict['plot_times_I'] = self.proc_data_dict['sweep_points'][indices_I]
        self.proc_data_dict['data0_I_I'] = np.mean(self.proc_data_dict['data_I_I'][-4:-2])
        self.proc_data_dict['data0_Q_I'] = np.mean(self.proc_data_dict['data_Q_I'][-4:-2])

        indices_Q = np.hstack([np.arange(1,len(self.proc_data_dict['sweep_points'])-4,2),
                     np.arange(len(self.proc_data_dict['sweep_points'])-4,
                               len(self.proc_data_dict['sweep_points']))])
        self.proc_data_dict['data_I_Q'] = self.proc_data_dict['measured_values'][0][indices_Q]
        self.proc_data_dict['data_Q_Q'] = self.proc_data_dict['measured_values'][1][indices_Q]
        self.proc_data_dict['plot_times_Q'] = self.proc_data_dict['sweep_points'][indices_Q]
        self.proc_data_dict['data0_I_Q'] = np.mean(self.proc_data_dict['data_I_Q'][-4:-2])
        self.proc_data_dict['data0_Q_Q'] = np.mean(self.proc_data_dict['data_Q_Q'][-4:-2])

        self.proc_data_dict['data_A_I'] = np.sqrt((self.proc_data_dict['data_I_I']-self.proc_data_dict['data0_I_I'])**2 +
                                              (self.proc_data_dict['data_Q_I']-self.proc_data_dict['data0_Q_I'])**2)
        self.proc_data_dict['data_A_Q'] = np.sqrt((self.proc_data_dict['data_I_Q']-self.proc_data_dict['data0_I_Q'])**2 +
                                              (self.proc_data_dict['data_Q_Q']-self.proc_data_dict['data0_Q_Q'])**2)
        self.proc_data_dict['data0_A_I'] = np.mean(self.proc_data_dict['data_A_I'][-4:-2])
        self.proc_data_dict['data1_A_I'] = np.mean(self.proc_data_dict['data_A_I'][-2:])
        self.proc_data_dict['dataA_I_avg'] = np.mean([self.proc_data_dict['data0_A_I'],
                                                        self.proc_data_dict['data1_A_I']])
        self.proc_data_dict['dataA_I_amp'] = self.proc_data_dict['data1_A_I'] - self.proc_data_dict['data0_A_I']

        self.proc_data_dict['data0_A_Q'] = np.mean(self.proc_data_dict['data_A_Q'][-4:-2])
        self.proc_data_dict['data1_A_Q'] = np.mean(self.proc_data_dict['data_A_Q'][-2:])
        self.proc_data_dict['dataA_Q_avg'] = np.mean([self.proc_data_dict['data0_A_Q'],
                                                        self.proc_data_dict['data1_A_Q']])
        self.proc_data_dict['dataA_Q_amp'] = self.proc_data_dict['data1_A_Q'] - self.proc_data_dict['data0_A_Q']

        self.proc_data_dict['plot_data_A_I'] = (self.proc_data_dict['data_A_I'] - self.proc_data_dict['dataA_I_avg'])/\
                                            self.proc_data_dict['dataA_I_amp']*2
        self.proc_data_dict['plot_data_A_Q'] = (self.proc_data_dict['data_A_Q'] - self.proc_data_dict['dataA_Q_avg'])/\
                                            self.proc_data_dict['dataA_Q_amp']*2


        self.proc_data_dict['phase'] = np.unwrap(np.arctan2(self.proc_data_dict['plot_data_A_Q'][:-4],self.proc_data_dict['plot_data_A_I'][:-4]))
        self.proc_data_dict['amp'] = np.hstack([np.sqrt(self.proc_data_dict['plot_data_A_Q'][:-4]**2+self.proc_data_dict['plot_data_A_I'][:-4]**2),
                 np.abs(self.proc_data_dict['plot_data_A_Q'][-4:])])


    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()


        phase_guess_fit = np.polyfit(self.proc_data_dict['plot_times_I'][:-4],
                                     self.proc_data_dict['phase'],1,
                                     w=self.proc_data_dict['amp'][:-4])
        freq_guess, phase_guess = phase_guess_fit
        offset_Q_guess = 0.0
        offset_I_guess = 0.0
        # if max(self.proc_data_dict['amp'][:-4]) > 1.5:
        freq_guess1 = np.fft.fft(1j*self.proc_data_dict['plot_data_A_Q'][:-4] + self.proc_data_dict['plot_data_A_I'][:-4])
        freqaxis = np.fft.fftfreq(len(freq_guess1),self.proc_data_dict['plot_times_I'][1] - self.proc_data_dict['plot_times_I'][0])
        freqaxis1 = freqaxis[1:]
        freq_guess = freqaxis1[np.argmax(np.abs(freq_guess1[1:]))]*2*np.pi
        # import matplotlib.pyplot as plt
        # plt.plot(freqaxis[1:],np.abs(freq_guess1[1:]))
        # plt.plot(freq_guess,np.max(np.abs(freq_guess1[1:])),'x',markersize=8)

        phase_guess = self.proc_data_dict['phase'][0]-freq_guess*self.proc_data_dict['plot_times_I'][0]
        offset_Q_guess = 0.5
        offset_I_guess = 0.5
        t_index = np.argmin(abs(self.proc_data_dict['amp'][:-4]-np.exp(-1)))
        tau_guess = self.proc_data_dict['plot_times_I'][:-4][t_index]
        complex_guess = {}
        complex_guess['amplitude'] = {'value':max(self.proc_data_dict['amp'][:-4]),
                    'min':0,
                    # 'max':10,
                    'vary':True}
        complex_guess['offset_I'] = {'value':offset_I_guess,
                    'min':-10,
                    'max':10,
                    'vary':True}
        complex_guess['offset_Q'] = {'value':offset_Q_guess,
                    'min':-10,
                    'max':10,
                    'vary':True}
        complex_guess['phase'] = {'value':np.remainder(phase_guess,4*np.pi),
                    'min':-4*np.pi,
                    'max':4*np.pi,
                    'vary':True}
        complex_guess['frequency'] = {'value':freq_guess/2/np.pi,
                    'min':-50e6,
                    'max':50e6,
                    'vary':True}
        complex_guess['tau'] = {'value': 1e-6,
                    'min':1e-7,
                    'vary':True}
        # print(complex_guess)
        complex_data = np.add(self.proc_data_dict['plot_data_A_I'][:-4],
                            1.j*self.proc_data_dict['plot_data_A_Q'][:-4])
        self.fit_dicts['exp_fit'] = {'fit_fn': fit_mods.ExpDampOscFuncComplex,
                                      'guess_dict':complex_guess,
                                      'fit_yvals': {'data': complex_data},
                                      'fit_xvals': {'t': self.proc_data_dict['plot_times_I'][:-4]},
                                      'fitting_type':'minimize'}






    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['plot_times_I'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['plot_data_A_I'],
            'ylabel': 'Normalized data',
            'yunit': '',
            'setlabel': '<x> Data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'dpi': 200,
            'do_legend': True,
            'legend_pos': 'best'}
        self.plot_dicts['mainQ'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['plot_times_Q'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['plot_data_A_Q'],
            'ylabel': 'Normalized data',
            'yunit': '',
            'setlabel': '<y> Data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'dpi': 200,
            'do_legend': True,
            'legend_pos': 'best'}
        self.plot_dicts['Phase'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['plot_times_I'][:-4],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['phase'],
            'ylabel': 'Phase',
            'yunit': 'rad',
            'setlabel': 'Phase Data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'dpi': 200,
            'do_legend': True,
            'legend_pos': 'best'}

        self.plot_dicts['Amp'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['plot_times_I'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['amp'],
            'ylabel': 'Coherence',
            'yunit': '',
            'setlabel': 'Coherence',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'dpi': 200,
            'do_legend': True,
            'legend_pos': 'best'}

        self.plot_dicts['Parametric'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['plot_data_A_I'][:-4],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0],  # does not do anything yet
            'yvals': self.proc_data_dict['plot_data_A_Q'][:-4],
            'ylabel': self.raw_data_dict['xlabel'][0],
            'yunit': '',
            'setlabel': 'Data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'dpi': 200,
            'do_legend': True,
            'legend_pos': 'best'}

        if self.do_fitting:

            self.plot_dicts['exp_fit_real'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'output_mod_fn':np.real,
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'exp fit real part',
                'do_legend': True,
                'legend_pos': 'best'}
            self.plot_dicts['exp_fit_imag'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'output_mod_fn':np.imag,
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'exp fit imaginary part',
                'do_legend': True,
                'legend_pos': 'best'}

            self.plot_dicts['exp_fit_amp'] = {
                'ax_id': 'Amp',
                'plotfn': self.plot_fit,
                'output_mod_fn':np.abs,
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit amplitude',
                'do_legend': True,
                'legend_pos': 'best'}

            self.plot_dicts['exp_fit_phase'] = {
                'ax_id': 'Phase',
                'plotfn': self.plot_fit,
                'output_mod_fn':lambda a: np.unwrap(np.angle(a)),
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit phase',
                'do_legend': True,
                'legend_pos': 'best'}

            self.plot_dicts['exp_fit_parametric'] = {
                'ax_id': 'Parametric',
                'plotfn': self.plot_fit,
                'output_mod_fn':np.imag,
                'output_mod_fn_x':np.real,
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'exp fit parametric',
                'do_legend': True,
                'legend_pos': 'best'}

            fit_res_params = self.fit_dicts['exp_fit']['fit_res'].params
            scale_frequency, unit_frequency = SI_prefix_and_scale_factor(fit_res_params['frequency'].value,'Hz')
            plot_frequency = fit_res_params['frequency'].value*scale_frequency
            scale_amplitude, unit_amplitude = SI_prefix_and_scale_factor(fit_res_params['amplitude'].value)
            plot_amplitude = fit_res_params['amplitude'].value*scale_amplitude
            scale_tau, unit_tau = SI_prefix_and_scale_factor(fit_res_params['tau'].value,'s')
            plot_tau = fit_res_params['tau'].value*scale_tau
            scale_offset_I, unit_offset_I = SI_prefix_and_scale_factor(fit_res_params['offset_I'].value)
            plot_offset_I = fit_res_params['offset_I'].value*scale_offset_I
            scale_offset_Q, unit_offset_Q = SI_prefix_and_scale_factor(fit_res_params['offset_Q'].value)
            plot_offset_Q = fit_res_params['offset_Q'].value*scale_offset_Q
            # scale_phase, label_phase = SI_prefix_and_scale_factor(fit_res_params['phase'].value, 'rad')
            # print(SI_prefix_and_scale_factor(fit_res_params['frequency'].value,'Hz'))
            self.plot_dicts['Parameters'] = {
                'ax_id': 'main',
                'ypos': .5,
                'xpos': 1.04,
                'plotfn': self.plot_text,
                'dpi': 200,
                'box_props': 'fancy',
                'horizontalalignment': 'left',
                # 'text_string': 'Chi = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr),
                'text_string': 'Fit results' + '\n'
                                + '$\mathrm{\chi}^2$ = %.3f'%(self.fit_dicts['exp_fit']['fit_res'].chisqr)  + '\n'
                                + 'Detuning = %.2f '%(plot_frequency) + unit_frequency + '\n'
                                + '$\mathrm{T}_2$ = %.2f '%(plot_tau) + unit_tau + '\n'
                                + 'A = %.2f '%(plot_amplitude) + unit_amplitude + '\n'
                                + 'Offset I = %.2f ' %(plot_offset_I) + unit_offset_I + '\n'
                                + 'Offset Q = %.2f ' %(plot_offset_Q) + unit_offset_Q + '\n'}




            self.plot_dicts['exp_fit_parametric'] = {
                'ax_id': 'Parametric',
                'plotfn': self.plot_fit,
                'output_mod_fn':np.imag,
                'output_mod_fn_x':np.real,
                'fit_res': self.fit_dicts['exp_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'exp fit parametric',
                'do_legend': True,
                'legend_pos': 'best'}

            fit_res_params = self.fit_dicts['exp_fit']['fit_res'].params
            scale_frequency, unit_frequency = SI_prefix_and_scale_factor(fit_res_params['frequency'].value,'Hz')
            plot_frequency = fit_res_params['frequency'].value*scale_frequency
            scale_amplitude, unit_amplitude = SI_prefix_and_scale_factor(fit_res_params['amplitude'].value)
            plot_amplitude = fit_res_params['amplitude'].value*scale_amplitude
            scale_tau, unit_tau = SI_prefix_and_scale_factor(fit_res_params['tau'].value,'s')
            plot_tau = fit_res_params['tau'].value*scale_tau
            scale_offset_I, unit_offset_I = SI_prefix_and_scale_factor(fit_res_params['offset_I'].value)
            plot_offset_I = fit_res_params['offset_I'].value*scale_offset_I
            scale_offset_Q, unit_offset_Q = SI_prefix_and_scale_factor(fit_res_params['offset_Q'].value)
            plot_offset_Q = fit_res_params['offset_Q'].value*scale_offset_Q
            # scale_phase, label_phase = SI_prefix_and_scale_factor(fit_res_params['phase'].value, 'rad')
            # print(SI_prefix_and_scale_factor(fit_res_params['frequency'].value,'Hz'))
            self.plot_dicts['Parameters'] = {
                'ax_id': 'main',
                'ypos': .5,
                'xpos': 1.04,
                'plotfn': self.plot_text,
                'dpi': 200,
                'box_props': 'fancy',
                'horizontalalignment': 'left',
                # 'text_string': 'Chi = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr),
                'text_string': 'Fit results' + '\n'
                                + '$\mathrm{\chi}^2$ = %.3f'%(self.fit_dicts['exp_fit']['fit_res'].chisqr)  + '\n'
                                + 'Detuning = %.2f '%(plot_frequency) + unit_frequency + '\n'
                                + '$\mathrm{T}_2$ = %.2f '%(plot_tau) + unit_tau + '\n'
                                + 'A = %.2f '%(plot_amplitude) + unit_amplitude + '\n'
                                + 'Offset I = %.2f ' %(plot_offset_I) + unit_offset_I + '\n'
                                + 'Offset Q = %.2f ' %(plot_offset_Q) + unit_offset_Q + '\n'}



class Intersect_Analysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analysis to extract the intercept of two parameters.

    relevant options_dict parameters
        ch_idx_A (int) specifies first channel for intercept
        ch_idx_B (int) specifies second channel for intercept if same as first
            it will assume data was taken interleaved.
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        do_fitting: bool = True,
        auto=True,
        normalized_probability=False,
    ):

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            data_file_path=data_file_path,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
        )
        self.single_timestamp = False

        self.normalized_probability = normalized_probability

        self.params_dict = {
            "xlabel": "sweep_name",
            "xvals": "sweep_points",
            "xunit": "sweep_unit",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_A" and "ch_idx_B"
        specified in the options dict. If ch_idx_A and ch_idx_B are the same
        it will unzip the data.
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        ch_idx_A = self.options_dict.get("ch_idx_A", 0)
        ch_idx_B = self.options_dict.get("ch_idx_B", 0)

        self.proc_data_dict["ylabel"] = self.raw_data_dict["value_names"][0][ch_idx_A]
        self.proc_data_dict["yunit"] = self.raw_data_dict["value_units"][0][ch_idx_A]

        if ch_idx_A == ch_idx_B:
            yvals = list(self.raw_data_dict["measured_values_ord_dict"].values())[
                ch_idx_A
            ][0]
            self.proc_data_dict["xvals_A"] = self.raw_data_dict["xvals"][0][::2]
            self.proc_data_dict["xvals_B"] = self.raw_data_dict["xvals"][0][1::2]
            self.proc_data_dict["yvals_A"] = yvals[::2]
            self.proc_data_dict["yvals_B"] = yvals[1::2]
        else:
            self.proc_data_dict["xvals_A"] = self.raw_data_dict["xvals"][0]
            self.proc_data_dict["xvals_B"] = self.raw_data_dict["xvals"][0]

            self.proc_data_dict["yvals_A"] = list(
                self.raw_data_dict["measured_values_ord_dict"].values()
            )[ch_idx_A][0]
            self.proc_data_dict["yvals_B"] = list(
                self.raw_data_dict["measured_values_ord_dict"].values()
            )[ch_idx_B][0]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts["line_fit_A"] = {
            "model": lmfit.models.PolynomialModel(degree=2),
            "fit_xvals": {"x": self.proc_data_dict["xvals_A"]},
            "fit_yvals": {"data": self.proc_data_dict["yvals_A"]},
        }

        self.fit_dicts["line_fit_B"] = {
            "model": lmfit.models.PolynomialModel(degree=2),
            "fit_xvals": {"x": self.proc_data_dict["xvals_B"]},
            "fit_yvals": {"data": self.proc_data_dict["yvals_B"]},
        }

    def analyze_fit_results(self):
        fr_0 = self.fit_res["line_fit_A"].best_values
        fr_1 = self.fit_res["line_fit_B"].best_values

        c0 = fr_0["c0"] - fr_1["c0"]
        c1 = fr_0["c1"] - fr_1["c1"]
        c2 = fr_0["c2"] - fr_1["c2"]
        poly_coeff = [c0, c1, c2]
        poly = np.polynomial.polynomial.Polynomial([fr_0["c0"], fr_0["c1"], fr_0["c2"]])
        ic = np.polynomial.polynomial.polyroots(poly_coeff)

        self.proc_data_dict["intersect_L"] = ic[0], poly(ic[0])
        self.proc_data_dict["intersect_R"] = ic[1], poly(ic[1])

        if ((np.min(self.proc_data_dict["xvals"])) < ic[0]) and (
            ic[0] < (np.max(self.proc_data_dict["xvals"]))
        ):
            self.proc_data_dict["intersect"] = self.proc_data_dict["intersect_L"]
        else:
            self.proc_data_dict["intersect"] = self.proc_data_dict["intersect_R"]

    def prepare_plots(self):
        self.plot_dicts["main"] = {
            "plotfn": self.plot_line,
            "xvals": self.proc_data_dict["xvals_A"],
            "xlabel": self.proc_data_dict["xlabel"][0],
            "xunit": self.proc_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_A"],
            "ylabel": self.proc_data_dict["ylabel"],
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "A",
            "title": (
                self.proc_data_dict["timestamps"][0]
                + " \n"
                + self.proc_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }

        # if self.normalized_probability:
        #     self.plot_dicts["main"]["yrange"] = (0, 1)

        self.plot_dicts["on"] = {
            "plotfn": self.plot_line,
            "ax_id": "main",
            "xvals": self.proc_data_dict["xvals_B"],
            "xlabel": self.proc_data_dict["xlabel"][0],
            "xunit": self.proc_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_B"],
            "ylabel": self.proc_data_dict["ylabel"],
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "B",
            "do_legend": True,
            "legend_pos": "upper right",
        }

        if self.do_fitting:
            self.plot_dicts["line_fit_A"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["line_fit_A"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit A",
                "do_legend": True,
            }
            self.plot_dicts["line_fit_B"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["line_fit_B"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit B",
                "do_legend": True,
            }

            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict["intersect"][0],
                self.proc_data_dict["xunit"][0][0],
                return_type=float,
            )
            self.plot_dicts["intercept_message"] = {
                "ax_id": "main",
                "plotfn": self.plot_line,
                "xvals": [self.proc_data_dict["intersect"][0]],
                "yvals": [self.proc_data_dict["intersect"][1]],
                "line_kws": {"alpha": 0.5, "color": "gray", "markersize": 15},
                "marker": "o",
                "setlabel": "Intercept: {:.3f} {}".format(ic, ic_unit),
                "do_legend": True,
            }

    def get_intersect(self):

        return self.proc_data_dict["intersect"]


class Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Very basic analysis to determine the phase of a single oscillation
    that has an assumed period of 360 degrees.
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        data_file_path: str = None,
        label: str = "",
        ch_idx: int = 0,
        options_dict: dict = None,
        extract_only: bool = False,
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
            do_fitting=do_fitting,
        )
        self.single_timestamp = False
        self.ch_idx = ch_idx
        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "xvals": "sweep_points",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        idx = self.ch_idx

        normalize_to_cal_points = self.options_dict.get(
            "normalize_to_cal_points", False
        )
        cal_points = [
            [[-4, -3], [-2, -1]],
            [[-4, -2], [-3, -1]],
        ]

        yvals = list(self.raw_data_dict["measured_values_ord_dict"].values())[idx][0]
        if normalize_to_cal_points:
            yvals = a_tools.normalize_data_v3(
                yvals,
                cal_zero_points=cal_points[idx][0],
                cal_one_points=cal_points[idx][1],
            )
        self.proc_data_dict["yvals"] = yvals

        self.proc_data_dict["ylabel"] = self.raw_data_dict["value_names"][0][idx]
        self.proc_data_dict["yunit"] = self.raw_data_dict["value_units"][0][idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)

        if not (self.options_dict.get("normalize_to_cal_points", False)):
            t = self.raw_data_dict["xvals"][0]
            data = self.proc_data_dict["yvals"]
        else:
            t = self.raw_data_dict["xvals"][0][:-4]
            data = self.proc_data_dict["yvals"][:-4]

        self.fit_dicts["cos_fit"] = {
            "model": cos_mod,
            "guess_dict": {"frequency": {"value": 1 / 360, "vary": False}},
            "fit_xvals": {"t": t},
            "fit_yvals": {"data": data},
        }

    def analyze_fit_results(self):
        fr = self.fit_res["cos_fit"].best_values
        self.proc_data_dict["phi"] = np.rad2deg(fr["phase"])

    def prepare_plots(self):
        self.plot_dicts["main"] = {
            "plotfn": self.plot_line,
            "xvals": self.raw_data_dict["xvals"][0],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals"],
            "ylabel": self.proc_data_dict["ylabel"],
            "yunit": self.proc_data_dict["yunit"],
            "title": (
                self.raw_data_dict["timestamps"][0]
                + " \n"
                + self.raw_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            # 'yrange': (0,1),
            "legend_pos": "upper right",
        }

        if self.do_fitting:
            self.plot_dicts["cos_fit"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["cos_fit"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit",
                "do_legend": True,
            }


class Conditional_Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract quantities from a conditional oscillation.
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        data_file_path: str = None,
        label: str = "",
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs: bool = True,
        auto=True,
    ):
        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            options_dict=options_dict,
            close_figs=close_figs,
            extract_only=extract_only,
            do_fitting=True,
        )
        self.single_timestamp = False

        self.params_dict = {
            "xlabel": "sweep_name",
            "xunit": "sweep_unit",
            "xvals": "sweep_points",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_osc" and
        "ch_idx_spec" in the options dict and then splits the data for the
        off and on cases
        """
        self.proc_data_dict = OrderedDict()
        # values stored in quantities of interest will be saved in the data file
        self.proc_data_dict["quantities_of_interest"] = {}
        qoi = self.proc_data_dict["quantities_of_interest"]
        # The channel containing the data must be specified in the options dict
        ch_idx_spec = self.options_dict.get("ch_idx_spec", 0)
        ch_idx_osc = self.options_dict.get("ch_idx_osc", 1)

        # Necessary for when reading parked qubit
        self.include_park = "ch_idx_park" in self.options_dict.keys()
        ch_idx_park = self.options_dict.get("ch_idx_park", 2)

        qoi["ch_idx_osc"] = ch_idx_osc
        qoi["ch_idx_spec"] = ch_idx_spec
        qoi["ch_idx_park"] = ch_idx_park

        x_vals = self.raw_data_dict["xvals"][0]

        nr_osc_pnts = np.sum(x_vals <= 360)

        normalize_to_cal_points = self.options_dict.get("normalize_to_cal_points", True)

        cal_points_idxs = [
            [
                # Ramsey qubit
                [nr_osc_pnts + 0, nr_osc_pnts + 1],
                [nr_osc_pnts + 2, nr_osc_pnts + 3]
            ],
            [
                # Spectators qubit
                [nr_osc_pnts + 0, nr_osc_pnts + 2],
                [nr_osc_pnts + 1, nr_osc_pnts + 3]
            ],
        ]

        ch_idx_list = [ch_idx_osc, ch_idx_spec]
        type_list = ["osc", "spec"]

        cal_labels = ["00", "01", "10", "11"]

        cs_idx = [0,1]

        if self.include_park:
            # add calibration points same as first qubit
            cal_points_idxs += [cal_points_idxs[0]]
            ch_idx_list.append(ch_idx_park)
            type_list.append("park")
            cs_idx.append(2)
            cal_labels = ["000", "010", "101", "111"]

        osc_idxs = np.where(x_vals <= 360)[0]
        cal_idx = np.where(x_vals > 360)[0]

        self.proc_data_dict["xvals"] = x_vals[osc_idxs][::2]
        self.proc_data_dict["xvals_cal"] = np.arange(365, 365 + len(cal_idx) * 25, 25)
        self.proc_data_dict["cal_labels"] = cal_labels

        for ch_idx, c_idx, type_str in zip(ch_idx_list, cs_idx, type_list):
            yvals = list(self.raw_data_dict["measured_values_ord_dict"].values())[
                ch_idx
            ][0]

            self.proc_data_dict["ylabel_{}".format(type_str)] = self.raw_data_dict[
                "value_names"
            ][0][ch_idx]
            self.proc_data_dict["yunit"] = self.raw_data_dict["value_units"][0][ch_idx]

            if normalize_to_cal_points:
                yvals = a_tools.normalize_data_v3(
                    yvals,
                    cal_zero_points=cal_points_idxs[c_idx][0],
                    cal_one_points=cal_points_idxs[c_idx][1],
                )

            yvals_osc = yvals[osc_idxs]
            yvals_cal = yvals[cal_idx]

            self.proc_data_dict["yvals_{}_cal".format(type_str)] = yvals_cal
            self.proc_data_dict["yvals_{}_off".format(type_str)] = yvals_osc[::2]
            self.proc_data_dict["yvals_{}_on".format(type_str)] = yvals_osc[1::2]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod0 = lmfit.Model(fit_mods.CosFunc)
        cos_mod0.guess = fit_mods.Cos_guess.__get__(cos_mod0, cos_mod0.__class__)
        self.fit_dicts["cos_fit_off"] = {
            "model": cos_mod0,
            "guess_dict": {"frequency": {"value": 1 / 360, "vary": False}},
            "fit_xvals": {"t": self.proc_data_dict["xvals"]},
            "fit_yvals": {"data": self.proc_data_dict["yvals_osc_off"]},
        }

        cos_mod1 = lmfit.Model(fit_mods.CosFunc)
        cos_mod1.guess = fit_mods.Cos_guess.__get__(cos_mod1, cos_mod1.__class__)
        self.fit_dicts["cos_fit_on"] = {
            "model": cos_mod1,
            "guess_dict": {"frequency": {"value": 1 / 360, "vary": False}},
            "fit_xvals": {"t": self.proc_data_dict["xvals"]},
            "fit_yvals": {"data": self.proc_data_dict["yvals_osc_on"]},
        }

        if self.include_park:
            cos_mod_park_0 = lmfit.Model(fit_mods.CosFunc)
            cos_mod_park_0.guess = fit_mods.Cos_guess.__get__(
                cos_mod_park_0, cos_mod_park_0.__class__
            )
            self.fit_dicts["park_fit_off"] = {
                "model": cos_mod_park_0,
                "guess_dict": {"frequency": {"value": 1 / 360, "vary": False}},
                "fit_xvals": {"t": self.proc_data_dict["xvals"]},
                "fit_yvals": {"data": self.proc_data_dict["yvals_park_off"]},
            }

            cos_mod_park_1 = lmfit.Model(fit_mods.CosFunc)
            cos_mod_park_1.guess = fit_mods.Cos_guess.__get__(
                cos_mod_park_1, cos_mod_park_1.__class__
            )
            self.fit_dicts["park_fit_on"] = {
                "model": cos_mod_park_1,
                "guess_dict": {"frequency": {"value": 1 / 360, "vary": False}},
                "fit_xvals": {"t": self.proc_data_dict["xvals"]},
                "fit_yvals": {"data": self.proc_data_dict["yvals_park_on"]},
            }

    def analyze_fit_results(self):
        qoi = self.proc_data_dict["quantities_of_interest"]
        fr_0 = self.fit_res["cos_fit_off"]
        fr_1 = self.fit_res["cos_fit_on"]

        phi0 = ufloat(
            np.rad2deg(fr_0.params["phase"].value),
            np.rad2deg(
                fr_0.params["phase"].stderr
                if fr_0.params["phase"].stderr is not None
                else np.nan
            ),
        )

        phi1 = ufloat(
            np.rad2deg(fr_1.params["phase"].value),
            np.rad2deg(
                fr_1.params["phase"].stderr
                if fr_1.params["phase"].stderr is not None
                else np.nan
            ),
        )
        qoi["phi_0"] = phi0
        qoi["phi_1"] = phi1
        qoi["phi_cond"] = (phi0 - phi1) % 360

        qoi["osc_amp_0"] = ufloat(
            fr_0.params["amplitude"].value,
            fr_0.params["amplitude"].stderr
            if fr_0.params["amplitude"].stderr is not None
            else np.nan,
        )
        qoi["osc_amp_1"] = ufloat(
            fr_1.params["amplitude"].value,
            fr_1.params["amplitude"].stderr
            if fr_1.params["amplitude"].stderr is not None
            else np.nan,
        )

        qoi["osc_offs_0"] = ufloat(
            fr_0.params["offset"].value,
            fr_0.params["offset"].stderr
            if fr_0.params["offset"].stderr is not None
            else np.nan,
        )

        qoi["osc_offs_1"] = ufloat(
            fr_1.params["offset"].value,
            fr_1.params["offset"].stderr
            if fr_1.params["offset"].stderr is not None
            else np.nan,
        )

        qoi["offs_diff"] = qoi["osc_offs_1"] - qoi["osc_offs_0"]

        spec_on = ufloat(
            np.mean(self.proc_data_dict["yvals_spec_on"]),
            sem(self.proc_data_dict["yvals_spec_on"]),
        )
        spec_off = ufloat(
            np.mean(self.proc_data_dict["yvals_spec_off"]),
            sem(self.proc_data_dict["yvals_spec_off"]),
        )
        qoi["missing_fraction"] = spec_on - spec_off

        if self.include_park:
            fp_0 = self.fit_res["park_fit_off"]
            fp_1 = self.fit_res["park_fit_on"]
            park_phase_off = ufloat(
                np.rad2deg(fp_0.params["phase"].value),
                np.rad2deg(
                    fp_0.params["phase"].stderr
                    if fp_0.params["phase"].stderr is not None
                    else np.nan
                ),
            )
            park_phase_on = ufloat(
                np.rad2deg(fp_1.params["phase"].value),
                np.rad2deg(
                    fp_1.params["phase"].stderr
                    if fp_1.params["phase"].stderr is not None
                    else np.nan
                ),
            )
        else:
            park_phase_off = ufloat(0, 0)
            park_phase_on = ufloat(0, 0)

        qoi["park_phase_off"] = park_phase_off
        qoi["park_phase_on"] = park_phase_on

        if self.include_park:
            fp_0 = self.fit_res['park_fit_off']
            fp_1 = self.fit_res['park_fit_on']
            park_phase_off=ufloat(np.rad2deg(fp_0.params['phase'].value),
                                  np.rad2deg(fp_0.params['phase'].stderr if
                                             fp_0.params['phase'].stderr is not None
                                             else np.nan))
            park_phase_on=ufloat(np.rad2deg(fp_1.params['phase'].value),
                                 np.rad2deg(fp_1.params['phase'].stderr if
                                            fp_1.params['phase'].stderr is not None
                                            else np.nan))
        else:
            park_phase_off = ufloat(0,0)
            park_phase_on = ufloat(0,0)

        qoi['park_phase_off'] = park_phase_off
        qoi['park_phase_on'] = park_phase_on

    def prepare_plots(self):
        self._prepare_main_oscillation_figure()
        self._prepare_spectator_qubit_figure()
        if self.include_park:
            self._prepare_park_oscillation_figure()

    def _prepare_main_oscillation_figure(self):

        y_label = self.proc_data_dict["ylabel_osc"]
        ax_id = "main_" + y_label

        self.plot_dicts[ax_id] = {
            "plotfn": self.plot_line,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_osc_off"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ off",
            "title": (
                self.raw_data_dict["timestamps"][0]
                + " \n"
                + self.raw_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            # 'yrange': (0,1),
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_on"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_osc_on"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ on",
            "do_legend": True,
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_cal_pnts"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals_cal"],
            "yvals": self.proc_data_dict["yvals_osc_cal"],
            "setlabel": "Calib.",
            "do_legend": True,
            "marker": "d",
        }

        if self.do_fitting:
            self.plot_dicts[ax_id + "_cos_fit_off"] = {
                "ax_id": ax_id,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["cos_fit_off"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit CZ off",
                "do_legend": True,
            }
            self.plot_dicts[ax_id + "_cos_fit_on"] = {
                "ax_id": ax_id,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["cos_fit_on"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit CZ on",
                "do_legend": True,
            }

            # offset as a guide for the eye
            y = self.fit_res["cos_fit_off"].params["offset"].value
            self.plot_dicts[ax_id + "_cos_off_offset"] = {
                "plotfn": self.plot_matplot_ax_method,
                "ax_id": ax_id,
                "func": "axhline",
                "plot_kws": {"y": y, "color": "C0", "linestyle": "dotted"},
            }

            # offset as a guide for the eye
            y = self.fit_res["cos_fit_on"].params["offset"].value
            self.plot_dicts[ax_id + "_cos_on_offset"] = {
                "plotfn": self.plot_matplot_ax_method,
                "ax_id": ax_id,
                "func": "axhline",
                "plot_kws": {"y": y, "color": "C1", "linestyle": "dotted"},
            }

            qoi = self.proc_data_dict["quantities_of_interest"]
            phase_message = (
                "Phase diff.: {}  deg\n"
                "Phase off: {} deg\n"
                "Phase on: {} deg\n\n"
                "Offs. diff.: {} %\n"
                "Osc. offs. off: {} \n"
                "Osc. offs. on: {}\n\n"
                "Osc. amp. off: {} \n"
                "Osc. amp. on: {} ".format(
                    qoi["phi_cond"],
                    qoi["phi_0"],
                    qoi["phi_1"],
                    qoi["offs_diff"] * 100,
                    qoi["osc_offs_0"],
                    qoi["osc_offs_1"],
                    qoi["osc_amp_0"],
                    qoi["osc_amp_1"],
                )
            )

            self.plot_dicts[ax_id + "_phase_message"] = {
                "ax_id": ax_id,
                "ypos": 0.9,
                "xpos": 1.45,
                "plotfn": self.plot_text,
                "box_props": "fancy",
                "line_kws": {"alpha": 0},
                "horizontalalignment": "right",
                "text_string": phase_message,
            }

        self.plot_dicts[ax_id + "_xlabels"] = {
            "ax_id": ax_id,
            "plotfn": self._plot_cal_pnts,
            "x_vals": self.proc_data_dict["xvals_cal"],
            "x_labels": self.proc_data_dict["cal_labels"],
        }

    def _prepare_spectator_qubit_figure(self):
        y_label = self.proc_data_dict["ylabel_spec"]
        ax_id = "spectator_qubit_" + y_label

        self.plot_dicts[ax_id] = {
            "plotfn": self.plot_line,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_spec_off"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ off",
            "title": (
                self.raw_data_dict["timestamps"][0]
                + " \n"
                + self.raw_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_spec_on"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_spec_on"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ on",
            "do_legend": True,
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_cal_pnts"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals_cal"],
            "yvals": self.proc_data_dict["yvals_spec_cal"],
            "setlabel": "Calib.",
            "do_legend": True,
            "marker": "d",
        }

        if self.do_fitting:
            leak_msg = "Missing frac.: {} % ".format(
                self.proc_data_dict["quantities_of_interest"]["missing_fraction"] * 100
            )
            self.plot_dicts[ax_id + "_leak_msg"] = {
                "ax_id": ax_id,
                "ypos": 0.9,
                "xpos": 1.45,
                "plotfn": self.plot_text,
                "box_props": "fancy",
                "line_kws": {"alpha": 0},
                "horizontalalignment": "right",
                "text_string": leak_msg,
            }

        self.plot_dicts[ax_id + "_xlabels"] = {
            "ax_id": ax_id,
            "plotfn": self._plot_cal_pnts,
            "x_vals": self.proc_data_dict["xvals_cal"],
            "x_labels": self.proc_data_dict["cal_labels"],
        }

    def _prepare_park_oscillation_figure(self):
        y_label = self.proc_data_dict["ylabel_park"]
        ax_id = "park_" + y_label

        self.plot_dicts[ax_id] = {
            "plotfn": self.plot_line,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_park_off"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ off",
            "title": (
                self.raw_data_dict["timestamps"][0]
                + " \n"
                + self.raw_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_on"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.raw_data_dict["xlabel"][0],
            "xunit": self.raw_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals_park_on"],
            "ylabel": y_label,
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "CZ on",
            "do_legend": True,
            "legend_pos": "upper right",
        }

        self.plot_dicts[ax_id + "_cal_pnts"] = {
            "plotfn": self.plot_line,
            "ax_id": ax_id,
            "xvals": self.proc_data_dict["xvals_cal"],
            "yvals": self.proc_data_dict["yvals_park_cal"],
            "setlabel": "Calib.",
            "do_legend": True,
            "marker": "d",
        }

        if self.do_fitting:
            self.plot_dicts[ax_id + "_park_fit_off"] = {
                "ax_id": ax_id,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["park_fit_off"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit CZ off",
                "do_legend": True,
            }
            self.plot_dicts[ax_id + "_park_fit_on"] = {
                "ax_id": ax_id,
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["park_fit_on"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit CZ on",
                "do_legend": True,
            }

            qoi = self.proc_data_dict["quantities_of_interest"]
            # calate average of angles accounting for wrapping
            angles = [qoi["park_phase_off"].n, qoi["park_phase_on"].n]
            stderrs = [qoi["park_phase_off"].s, qoi["park_phase_on"].s]
            av_sin = np.average(np.sin(np.deg2rad(angles)))
            av_cos = np.average(np.cos(np.deg2rad(angles)))
            phase_av = np.rad2deg(np.arctan2(av_sin, av_cos))

            phase_message = "Phase off: {} deg\n" "Phase on: {} deg\n" "Phase av.: {} deg".format(
                qoi["park_phase_off"], qoi["park_phase_on"],
                ufloat(phase_av, np.max(stderrs))
            )
            self.plot_dicts[ax_id + "_phase_message"] = {
                "ax_id": ax_id,
                "ypos": 0.9,
                "xpos": 1.45,
                "plotfn": self.plot_text,
                "box_props": "fancy",
                "line_kws": {"alpha": 0},
                "horizontalalignment": "right",
                "text_string": phase_message,
            }

        self.plot_dicts[ax_id + "_xlabels"] = {
            "ax_id": ax_id,
            "plotfn": self._plot_cal_pnts,
            "x_vals": self.proc_data_dict["xvals_cal"],
            "x_labels": self.proc_data_dict["cal_labels"],
        }

    def _plot_cal_pnts(self, ax, x_vals, x_labels, **kw):

        phi = np.arange(0, 360, 60)
        ax.set_xticks(np.concatenate((phi, x_vals)))
        deg_sign = u"\N{DEGREE SIGN}"
        ax.set_xticklabels(["{:3.0f}".format(ang) + deg_sign for ang in phi] + x_labels)
        ax.tick_params(axis="x", labelrotation=45)


class Crossing_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract the intercept of a parameter with the `target_crossing`
    The interception measured quantity is defined by `ch_idx`
    """

    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "",
        target_crossing: float = 0,
        ch_idx: int = -1,
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
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
            do_fitting=do_fitting,
        )
        self.single_timestamp = False

        self.params_dict = {
            "xlabel": "sweep_name",
            "xvals": "sweep_points",
            "xunit": "sweep_unit",
            "measurementstring": "measurementstring",
            "value_names": "value_names",
            "value_units": "value_units",
            "measured_values": "measured_values",
        }

        self.target_crossing = target_crossing
        self.ch_idx = ch_idx
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        if str(self.ch_idx).isdigit():
            ch_idx = ch_idx
        else:
            ch_idx = list(
               self.raw_data_dict["measured_values_ord_dict"].keys()).index(str(self.ch_idx))

        # print('Fitting Crossing to {}'.format(list(
        #     self.raw_data_dict["measured_values_ord_dict"].keys()
        # )[ch_idx]))

        self.proc_data_dict["xvals"] = self.raw_data_dict["xvals"][0]
        self.proc_data_dict["ylabel"] = self.raw_data_dict["value_names"][0][ch_idx]
        self.proc_data_dict["yunit"] = self.raw_data_dict["value_units"][0][ch_idx]
        self.proc_data_dict["yvals"] = list(
            self.raw_data_dict["measured_values_ord_dict"].values()
        )[ch_idx][0]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        self.fit_dicts["line_fit"] = {
            "model": lmfit.models.PolynomialModel(degree=2),
            "fit_xvals": {"x": self.proc_data_dict["xvals"]},
            "fit_yvals": {"data": np.unwrap(self.proc_data_dict["yvals"], 360)},
        }

    def analyze_fit_results(self):
        # pack function
        target_crossing = self.target_crossing
        fit_res = self.fit_dicts["line_fit"]["fit_res"]
        c0 = fit_res.best_values["c0"] - target_crossing  # constant term
        c1 = fit_res.best_values["c1"]  # linear term
        c2 = fit_res.best_values["c2"]  # quadratic term

        ######################################
        # WARNING:
        # NUMPY HANDLES A DIFFERENT CONVENTION FOR THE FUNCTIONS
        # np.polynomial.polynomial.polyroots = [coeff 0, coeff x, coeff x**2,..]
        # np.polyval; np.polyfit = [coeff x**N, x**N-1, ..., x, 0]
        ######################################
        poly_coeff = [c0, c1, c2]
        roots = np.real_if_close(np.polynomial.polynomial.polyroots(poly_coeff))
        # only keep roots within range
        min_xrange = np.min(self.proc_data_dict["xvals"])
        max_xrange = np.max(self.proc_data_dict["xvals"])
        is_root_in_range = np.where(
            np.logical_and(roots >= min_xrange, roots <= max_xrange), True, False
        )

        # check whether there is roots within range
        # print(roots,is_root_in_range)
        # print('Fitlered',roots[is_root_in_range])
        roots_available_within_range = roots[is_root_in_range][0]
        if roots_available_within_range > 0:  # sums Trues as 1, Falses as 0
            self.proc_data_dict["root"] = roots[is_root_in_range][
                0
            ]  # selects first root available
        elif roots_available_within_range < 0:
            self.proc_data_dict["root"] = roots[0]  # selects first root available
        else:
            self.proc_data_dict["root"] = np.nan

        self.proc_data_dict["intersect"] = [
            self.proc_data_dict["root"],
            np.polyval(poly_coeff[::-1], self.proc_data_dict["root"]),
        ]
        print("Intersect found at: \n", [
            self.proc_data_dict["intersect"][0], self.proc_data_dict["intersect"][1] + self.target_crossing])

    def prepare_plots(self):
        pass
        self.plot_dicts["main"] = {
            "plotfn": self.plot_line,
            "xvals": self.proc_data_dict["xvals"],
            "xlabel": self.proc_data_dict["xlabel"][0],
            "xunit": self.proc_data_dict["xunit"][0][0],
            "yvals": self.proc_data_dict["yvals"],
            "ylabel": self.proc_data_dict["ylabel"],
            "yunit": self.proc_data_dict["yunit"],
            "setlabel": "A",
            "title": (
                self.proc_data_dict["timestamps"][0]
                + " \n"
                + self.proc_data_dict["measurementstring"][0]
            ),
            "do_legend": True,
            "legend_pos": "upper right",
        }

        if self.do_fitting:
            self.plot_dicts["line_fit"] = {
                "ax_id": "main",
                "plotfn": self.plot_fit,
                "fit_res": self.fit_dicts["line_fit"]["fit_res"],
                "plot_init": self.options_dict["plot_init"],
                "setlabel": "Fit",
                "do_legend": True,
            }
            self.plot_dicts["intercept_message"] = {
                "ax_id": "main",
                "plotfn": self.plot_line,
                "xvals": [self.proc_data_dict["intersect"][0]],
                "yvals": [self.proc_data_dict["intersect"][1] + self.target_crossing],
                "line_kws": {"alpha": 0.5, "color": "gray", "markersize": 15},
                "marker": "o",
                "setlabel": "Intercept: {:.3f}".format(self.proc_data_dict["root"]),
                "do_legend": True,
            }

    def get_intersect(self):
        return self.proc_data_dict["root"]
