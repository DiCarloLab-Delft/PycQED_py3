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
from pycqed.analysis.tools.data_manipulation import \
    populations_using_rate_equations
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, plot_fit
from pycqed.utilities.general import SafeFormatter, format_value_string
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from matplotlib import colors as c


class RandomizedBenchmarking_SingleQubit_Analysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idx: int =1,
                 ignore_f_cal_pts: bool=False, **kwargs
                 ):
        """
        Analysis for single qubit randomized benchmarking.

        For basic options see docstring of BaseDataAnalysis

        Args:
            classification_method ["rates", ]   sets method to determine
                populations of g,e and f states. Currently only supports "rates"
                    rates: uses calibration points and rate equation from
                        Asaad et al. to determine populations
            rates_ch_idx (int) : sets the channel from which to use the data
                for the rate equations
            ignore_f_cal_pts (bool) : if True, ignores the f-state calibration
                points and instead makes the approximation that the f-state
                looks the same as the e-state in readout. This is useful when
                the ef-pulse is not calibrated.
        """
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs,
                         do_fitting=True, **kwargs)
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idx = rates_ch_idx
        self.d1 = 2
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-6:2]
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_zero'] = OrderedDict()
            self.raw_data_dict['cal_pts_one'] = OrderedDict()
            self.raw_data_dict['cal_pts_two'] = OrderedDict()
            self.raw_data_dict['measured_values_I'] = OrderedDict()
            self.raw_data_dict['measured_values_X'] = OrderedDict()
            for i, val_name in enumerate(a.value_names):

                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*2)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')

                self.raw_data_dict['binned_vals'][val_name] = binned_yvals
                self.raw_data_dict['cal_pts_zero'][val_name] =\
                    binned_yvals[-6:-4, :].flatten()
                self.raw_data_dict['cal_pts_one'][val_name] =\
                    binned_yvals[-4:-2, :].flatten()

                if self.ignore_f_cal_pts:
                    self.raw_data_dict['cal_pts_two'][val_name] =\
                        self.raw_data_dict['cal_pts_one'][val_name]
                else:
                    self.raw_data_dict['cal_pts_two'][val_name] =\
                        binned_yvals[-2:, :].flatten()

                self.raw_data_dict['measured_values_I'][val_name] =\
                    binned_yvals[:-6:2, :]
                self.raw_data_dict['measured_values_X'][val_name] =\
                    binned_yvals[1:-6:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        for key in ['V0', 'V1', 'V2', 'SI', 'SX', 'P0', 'P1', 'P2', 'M_inv']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            V0 = np.nanmean(
                self.raw_data_dict['cal_pts_zero'][val_name])
            V1 = np.nanmean(
                self.raw_data_dict['cal_pts_one'][val_name])
            V2 = np.nanmean(
                self.raw_data_dict['cal_pts_two'][val_name])

            self.proc_data_dict['V0'][val_name] = V0
            self.proc_data_dict['V1'][val_name] = V1
            self.proc_data_dict['V2'][val_name] = V2

            SI = np.nanmean(
                self.raw_data_dict['measured_values_I'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_X'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            P0, P1, P2, M_inv = populations_using_rate_equations(
                SI, SX, V0, V1, V2)
            self.proc_data_dict['P0'][val_name] = P0
            self.proc_data_dict['P1'][val_name] = P1
            self.proc_data_dict['P2'][val_name] = P2
            self.proc_data_dict['M_inv'][val_name] = M_inv

        classifier = logisticreg_classifier_machinelearning(
            self.proc_data_dict['cal_pts_zero'],
            self.proc_data_dict['cal_pts_one'],
            self.proc_data_dict['cal_pts_two'])
        self.proc_data_dict['classifier'] = classifier

        if self.classification_method == 'rates':
            val_name = self.raw_data_dict['value_names'][self.rates_ch_idx]
            self.proc_data_dict['M0'] = self.proc_data_dict['P0'][val_name]
            self.proc_data_dict['X1'] = 1-self.proc_data_dict['P2'][val_name]
        else:
            raise NotImplementedError()

    def run_fitting(self):
        super().run_fitting()

        leak_mod = lmfit.Model(leak_decay, independent_vars='m')
        leak_mod.set_param_hint('A', value=.95, min=0, vary=True)
        leak_mod.set_param_hint('B', value=.1, min=0, vary=True)

        leak_mod.set_param_hint('lambda_1', value=.99, vary=True)
        leak_mod.set_param_hint('L1', expr='(1-A)*(1-lambda_1)')
        leak_mod.set_param_hint('L2', expr='A*(1-lambda_1)')
        leak_mod.set_param_hint(
            'L1_cz', expr='1-(1-(1-A)*(1-lambda_1))**(1/1.5)')
        leak_mod.set_param_hint(
            'L2_cz', expr='1-(1-(A*(1-lambda_1)))**(1/1.5)')

        params = leak_mod.make_params()
        try:
            fit_res_leak = leak_mod.fit(data=self.proc_data_dict['X1'],
                                        m=self.proc_data_dict['ncl'],
                                        params=params)
            self.fit_res['leakage_decay'] = fit_res_leak
            lambda_1 = fit_res_leak.best_values['lambda_1']
            L1 = fit_res_leak.params['L1'].value
        except Exception as e:
            logging.warning("Fitting failed")
            logging.warning(e)
            lambda_1 = 1
            L1 = 0
            self.fit_res['leakage_decay'] = {}

        fit_res_rb = self.fit_rb_decay(lambda_1=lambda_1, L1=L1, simple=False)
        self.fit_res['rb_decay'] = fit_res_rb
        fit_res_rb_simple = self.fit_rb_decay(lambda_1=1, L1=0, simple=True)
        self.fit_res['rb_decay_simple'] = fit_res_rb_simple

        fr_rb = self.fit_res['rb_decay'].params
        fr_rb_simple = self.fit_res['rb_decay_simple'].params
        fr_dec = self.fit_res['leakage_decay'].params

        text_msg = 'Summary: \n'
        text_msg += format_value_string(r'$\epsilon_{{\mathrm{{simple}}}}$',
                                        fr_rb_simple['eps'], '\n')
        text_msg += format_value_string(r'$\epsilon_{{X_1}}$',
                                        fr_rb['eps'], '\n')
        text_msg += format_value_string(r'$L_1$', fr_dec['L1'], '\n')
        text_msg += format_value_string(r'$L_2$', fr_dec['L2'], '\n')
        self.proc_data_dict['rb_msg'] = text_msg

        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']
        qoi['eps_simple'] = ufloat(fr_rb_simple['eps'].value,
                                   fr_rb_simple['eps'].stderr or np.NaN)
        qoi['eps_X1'] = ufloat(fr_rb['eps'].value,
                               fr_rb['eps'].stderr or np.NaN)
        qoi['L1'] = ufloat(fr_dec['L1'].value,
                           fr_dec['L1'].stderr or np.NaN)
        qoi['L2'] = ufloat(fr_dec['L2'].value,
                           fr_dec['L2'].stderr or np.NaN)

    def fit_rb_decay(self, lambda_1: float, L1: float, simple: bool=False):
        """
        Fits the data
        """
        fit_mod_rb = lmfit.Model(full_rb_decay, independent_vars='m')
        fit_mod_rb.set_param_hint('A', value=.5, min=0, vary=True)
        if simple:
            fit_mod_rb.set_param_hint('B', value=0, vary=False)
        else:
            fit_mod_rb.set_param_hint('B', value=.1, min=0, vary=True)
        fit_mod_rb.set_param_hint('C', value=.4, min=0, max=1, vary=True)

        fit_mod_rb.set_param_hint('lambda_1', value=lambda_1, vary=False)
        fit_mod_rb.set_param_hint('lambda_2', value=.95, vary=True)

        # d1 = dimensionality of computational subspace
        fit_mod_rb.set_param_hint('d1', value=self.d1, vary=False)
        fit_mod_rb.set_param_hint('L1', value=L1, vary=False)

        # Note that all derived quantities are expressed directly in
        fit_mod_rb.set_param_hint(
            'F', expr='1/d1*((d1-1)*lambda_2+1-L1)', vary=True)
        fit_mod_rb.set_param_hint('eps',
                                  expr='1-(1/d1*((d1-1)*lambda_2+1-L1))')
        # Only valid for single qubit RB assumption equal error rates
        fit_mod_rb.set_param_hint(
            'F_g', expr='(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)')
        fit_mod_rb.set_param_hint(
            'eps_g', expr='1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)')
        # Only valid for two qubit RB assumption all error in CZ
        fit_mod_rb.set_param_hint(
            'F_cz', expr='(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)')
        fit_mod_rb.set_param_hint(
            'eps_cz', expr='1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)')

        params = fit_mod_rb.make_params()
        fit_res_rb = fit_mod_rb.fit(data=self.proc_data_dict['M0'],
                                    m=self.proc_data_dict['ncl'],
                                    params=params)

        return fit_res_rb

    def prepare_plots(self):
        val_names = self.raw_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['bins'],
                'yvals': np.nanmean(self.raw_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.raw_data_dict['binned_vals'][val_name], axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.raw_data_dict['value_units'][i],
                'title': self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'],
            }

        fs = plt.rcParams['figure.figsize']
        self.plot_dicts['cal_points_hexbin'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.raw_data_dict['cal_pts_zero'][val_names[0]],
                        self.raw_data_dict['cal_pts_zero'][val_names[1]]),
            'shots_1': (self.raw_data_dict['cal_pts_one'][val_names[0]],
                        self.raw_data_dict['cal_pts_one'][val_names[1]]),
            'shots_2': (self.raw_data_dict['cal_pts_two'][val_names[0]],
                        self.raw_data_dict['cal_pts_two'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.raw_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.raw_data_dict['value_units'][1],
            'title': self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'] + ' hexbin plot',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        for i, val_name in enumerate(val_names):
            self.plot_dicts['raw_RB_curve_data_{}'.format(val_name)] = {
                'plotfn': plot_raw_RB_curve,
                'ncl': self.proc_data_dict['ncl'],
                'SI': self.proc_data_dict['SI'][val_name],
                'SX': self.proc_data_dict['SX'][val_name],
                'V0': self.proc_data_dict['V0'][val_name],
                'V1': self.proc_data_dict['V1'][val_name],
                'V2': self.proc_data_dict['V2'][val_name],

                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string']+'\n'+self.proc_data_dict['measurementstring'],
            }

            self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name)] = {
                'plotfn': plot_populations_RB_curve,
                'ncl': self.proc_data_dict['ncl'],
                'P0': self.proc_data_dict['P0'][val_name],
                'P1': self.proc_data_dict['P1'][val_name],
                'P2': self.proc_data_dict['P2'][val_name],
                'title': self.proc_data_dict['timestamp_string']+'\n' +
                'Population using rate equations ch{}'.format(val_name)
            }
        self.plot_dicts['logres_decision_bound'] = {
            'plotfn': plot_classifier_decission_boundary,
            'classifier': self.proc_data_dict['classifier'],
            'shots_0': (self.proc_data_dict['cal_pts_zero'][val_names[0]],
                        self.proc_data_dict['cal_pts_zero'][val_names[1]]),
            'shots_1': (self.proc_data_dict['cal_pts_one'][val_names[0]],
                        self.proc_data_dict['cal_pts_one'][val_names[1]]),
            'shots_2': (self.proc_data_dict['cal_pts_two'][val_names[0]],
                        self.proc_data_dict['cal_pts_two'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.proc_data_dict['value_units'][1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring'] +
            ' Decision boundary',
            'plotsize': (fs[0]*1.5, fs[1])}

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}


class RandomizedBenchmarking_TwoQubit_Analysis(
        RandomizedBenchmarking_SingleQubit_Analysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idxs: list =[2, 0],
                 ignore_f_cal_pts: bool=False, extract_only: bool = False,
                 ):
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs,
            do_fitting=True, extract_only=extract_only)
        self.d1 = 4
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idxs = rates_ch_idxs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-7:2]  # 7 calibration points
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_x0'] = OrderedDict()
            self.raw_data_dict['cal_pts_x1'] = OrderedDict()
            self.raw_data_dict['cal_pts_x2'] = OrderedDict()
            self.raw_data_dict['cal_pts_0x'] = OrderedDict()
            self.raw_data_dict['cal_pts_1x'] = OrderedDict()
            self.raw_data_dict['cal_pts_2x'] = OrderedDict()

            self.raw_data_dict['measured_values_I'] = OrderedDict()
            self.raw_data_dict['measured_values_X'] = OrderedDict()

            for i, val_name in enumerate(a.value_names):
                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0) &
                                        (a.measured_values[2] == 0) &
                                        (a.measured_values[3] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*4)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')
                self.raw_data_dict['binned_vals'][val_name] = binned_yvals

                # 7 cal points:  [00, 01, 10, 11, 02, 20, 22]
                #      col_idx:  [-7, -6, -5, -4, -3, -2, -1]
                self.raw_data_dict['cal_pts_x0'][val_name] =\
                    binned_yvals[(-7, -5), :].flatten()
                self.raw_data_dict['cal_pts_x1'][val_name] =\
                    binned_yvals[(-6, -4), :].flatten()
                self.raw_data_dict['cal_pts_x2'][val_name] =\
                    binned_yvals[(-3, -1), :].flatten()

                self.raw_data_dict['cal_pts_0x'][val_name] =\
                    binned_yvals[(-7, -6), :].flatten()
                self.raw_data_dict['cal_pts_1x'][val_name] =\
                    binned_yvals[(-5, -4), :].flatten()
                self.raw_data_dict['cal_pts_2x'][val_name] =\
                    binned_yvals[(-2, -1), :].flatten()

                self.raw_data_dict['measured_values_I'][val_name] =\
                    binned_yvals[:-7:2, :]
                self.raw_data_dict['measured_values_X'][val_name] =\
                    binned_yvals[1:-7:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        for key in ['Vx0', 'V0x', 'Vx1', 'V1x', 'Vx2', 'V2x',
                    'SI', 'SX',
                    'Px0', 'P0x', 'Px1', 'P1x', 'Px2', 'P2x',
                    'M_inv_q0', 'M_inv_q1']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            for idx in ['x0', 'x1', 'x2', '0x', '1x', '2x']:
                self.proc_data_dict['V{}'.format(idx)][val_name] = \
                    np.nanmean(self.raw_data_dict['cal_pts_{}'.format(idx)]
                               [val_name])
            SI = np.nanmean(
                self.raw_data_dict['measured_values_I'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_X'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            Px0, Px1, Px2, M_inv_q0 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['Vx0'][val_name],
                self.proc_data_dict['Vx1'][val_name],
                self.proc_data_dict['Vx2'][val_name])
            P0x, P1x, P2x, M_inv_q1 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['V0x'][val_name],
                self.proc_data_dict['V1x'][val_name],
                self.proc_data_dict['V2x'][val_name])

            for key, val in [('Px0', Px0), ('Px1', Px1), ('Px2', Px2),
                             ('P0x', P0x), ('P1x', P1x), ('P2x', P2x),
                             ('M_inv_q0', M_inv_q0), ('M_inv_q1', M_inv_q1)]:
                self.proc_data_dict[key][val_name] = val

        if self.classification_method == 'rates':
            val_name_q0 = self.raw_data_dict['value_names'][self.rates_ch_idxs[0]]
            val_name_q1 = self.raw_data_dict['value_names'][self.rates_ch_idxs[1]]

            self.proc_data_dict['M0'] = (
                self.proc_data_dict['Px0'][val_name_q0] *
                self.proc_data_dict['P0x'][val_name_q1])

            self.proc_data_dict['X1'] = (
                1-self.proc_data_dict['Px2'][val_name_q0]
                - self.proc_data_dict['P2x'][val_name_q1])
        else:
            raise NotImplementedError()

    def prepare_plots(self):
        val_names = self.proc_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bins'],
                'yvals': np.nanmean(self.proc_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.proc_data_dict['binned_vals'][val_name], axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string'] +
                '\n'+self.proc_data_dict['measurementstring'],
            }
        fs = plt.rcParams['figure.figsize']

        # define figure and axes here to have custom layout
        self.figs['rb_populations_decay'], axs = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(fs[0]*1.5, fs[1]))
        self.figs['rb_populations_decay'].suptitle(
            self.proc_data_dict['timestamp_string']+'\n' +
            'Population using rate equations', y=1.05)
        self.figs['rb_populations_decay'].patch.set_alpha(0)
        self.axs['rb_pops_q0'] = axs[0]
        self.axs['rb_pops_q1'] = axs[1]

        val_name_q0 = val_names[self.rates_ch_idxs[0]]
        val_name_q1 = val_names[self.rates_ch_idxs[1]]
        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q0)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['Px0'][val_name_q0],
            'P1': self.proc_data_dict['Px1'][val_name_q0],
            'P2': self.proc_data_dict['Px2'][val_name_q0],
            'title': ' {}'.format(val_name_q0),
            'ax_id': 'rb_pops_q0'}

        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q1)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['P0x'][val_name_q1],
            'P1': self.proc_data_dict['P1x'][val_name_q1],
            'P2': self.proc_data_dict['P2x'][val_name_q1],
            'title': ' {}'.format(val_name_q1),
            'ax_id': 'rb_pops_q1'}

        # This exists for when the order of RO of qubits is
        # different than expected.
        if self.rates_ch_idxs[1] > 0:
            v0_q0 = val_names[0]
            v1_q0 = val_names[1]
            v0_q1 = val_names[2]
            v1_q1 = val_names[3]
        else:
            v0_q0 = val_names[2]
            v1_q0 = val_names[3]
            v0_q1 = val_names[0]
            v1_q1 = val_names[1]

        self.plot_dicts['cal_points_hexbin_q0'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_x0'][v0_q0],
                        self.proc_data_dict['cal_pts_x0'][v1_q0]),
            'shots_1': (self.proc_data_dict['cal_pts_x1'][v0_q0],
                        self.proc_data_dict['cal_pts_x1'][v1_q0]),
            'shots_2': (self.proc_data_dict['cal_pts_x2'][v0_q0],
                        self.proc_data_dict['cal_pts_x2'][v1_q0]),
            'xlabel': v0_q0,
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': v1_q0,
            'yunit': self.proc_data_dict['value_units'][1],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q0',
            'plotsize': (fs[0]*1.5, fs[1])
        }
        self.plot_dicts['cal_points_hexbin_q1'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_0x'][v0_q1],
                        self.proc_data_dict['cal_pts_0x'][v1_q1]),
            'shots_1': (self.proc_data_dict['cal_pts_1x'][v0_q1],
                        self.proc_data_dict['cal_pts_1x'][v1_q1]),
            'shots_2': (self.proc_data_dict['cal_pts_2x'][v0_q1],
                        self.proc_data_dict['cal_pts_2x'][v1_q1]),
            'xlabel': v0_q1,
            'xunit': self.proc_data_dict['value_units'][2],
            'ylabel': v1_q1,
            'yunit': self.proc_data_dict['value_units'][3],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q1',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}


class UnitarityBenchmarking_TwoQubit_Analysis(
        RandomizedBenchmarking_SingleQubit_Analysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idxs: list =[0, 2],
                 ignore_f_cal_pts: bool=False, nseeds: int=None, **kwargs
                 ):
        """Analysis for unitarity benchmarking.

        This analysis is based on
        """
        if nseeds is None:
            raise TypeError('You must specify number of seeds!')
        self.nseeds = nseeds
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs,
            do_fitting=True, **kwargs)
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
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-7:10]  # 7 calibration points
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_x0'] = OrderedDict()
            self.raw_data_dict['cal_pts_x1'] = OrderedDict()
            self.raw_data_dict['cal_pts_x2'] = OrderedDict()
            self.raw_data_dict['cal_pts_0x'] = OrderedDict()
            self.raw_data_dict['cal_pts_1x'] = OrderedDict()
            self.raw_data_dict['cal_pts_2x'] = OrderedDict()

            self.raw_data_dict['measured_values_ZZ'] = OrderedDict()
            self.raw_data_dict['measured_values_XZ'] = OrderedDict()
            self.raw_data_dict['measured_values_YZ'] = OrderedDict()
            self.raw_data_dict['measured_values_ZX'] = OrderedDict()
            self.raw_data_dict['measured_values_XX'] = OrderedDict()
            self.raw_data_dict['measured_values_YX'] = OrderedDict()
            self.raw_data_dict['measured_values_ZY'] = OrderedDict()
            self.raw_data_dict['measured_values_XY'] = OrderedDict()
            self.raw_data_dict['measured_values_YY'] = OrderedDict()
            self.raw_data_dict['measured_values_mZmZ'] = OrderedDict()

            for i, val_name in enumerate(a.value_names):
                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0) &
                                        (a.measured_values[2] == 0) &
                                        (a.measured_values[3] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*4)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')
                self.raw_data_dict['binned_vals'][val_name] = binned_yvals

                # 7 cal points:  [00, 01, 10, 11, 02, 20, 22]
                #      col_idx:  [-7, -6, -5, -4, -3, -2, -1]
                self.raw_data_dict['cal_pts_x0'][val_name] =\
                    binned_yvals[(-7, -5), :].flatten()
                self.raw_data_dict['cal_pts_x1'][val_name] =\
                    binned_yvals[(-6, -4), :].flatten()
                self.raw_data_dict['cal_pts_x2'][val_name] =\
                    binned_yvals[(-3, -1), :].flatten()

                self.raw_data_dict['cal_pts_0x'][val_name] =\
                    binned_yvals[(-7, -6), :].flatten()
                self.raw_data_dict['cal_pts_1x'][val_name] =\
                    binned_yvals[(-5, -4), :].flatten()
                self.raw_data_dict['cal_pts_2x'][val_name] =\
                    binned_yvals[(-2, -1), :].flatten()

                self.raw_data_dict['measured_values_ZZ'][val_name] =\
                    binned_yvals[0:-7:10, :]
                self.raw_data_dict['measured_values_XZ'][val_name] =\
                    binned_yvals[1:-7:10, :]
                self.raw_data_dict['measured_values_YZ'][val_name] =\
                    binned_yvals[2:-7:10, :]
                self.raw_data_dict['measured_values_ZX'][val_name] =\
                    binned_yvals[3:-7:10, :]
                self.raw_data_dict['measured_values_XX'][val_name] =\
                    binned_yvals[4:-7:10, :]
                self.raw_data_dict['measured_values_YX'][val_name] =\
                    binned_yvals[5:-7:10, :]
                self.raw_data_dict['measured_values_ZY'][val_name] =\
                    binned_yvals[6:-7:10, :]
                self.raw_data_dict['measured_values_XY'][val_name] =\
                    binned_yvals[7:-7:10, :]
                self.raw_data_dict['measured_values_YY'][val_name] =\
                    binned_yvals[8:-7:10, :]
                self.raw_data_dict['measured_values_mZmZ'][val_name] =\
                    binned_yvals[9:-7:10, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        """Averages shot data and calculates unitarity from raw_data_dict.

        Note: this doe not correct the outcomes for leakage.



        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        keys = ['Vx0', 'V0x', 'Vx1', 'V1x', 'Vx2', 'V2x',
                'SI', 'SX',
                'Px0', 'P0x', 'Px1', 'P1x', 'Px2', 'P2x',
                'M_inv_q0', 'M_inv_q1']
        keys += ['XX', 'XY', 'XZ',
                 'YX', 'YY', 'YZ',
                 'ZX', 'ZY', 'ZZ',
                 'XX_sq', 'XY_sq', 'XZ_sq',
                 'YX_sq', 'YY_sq', 'YZ_sq',
                 'ZX_sq', 'ZY_sq', 'ZZ_sq',
                 'unitarity_shots', 'unitarity']
        keys += ['XX_q0', 'XY_q0', 'XZ_q0',
                 'YX_q0', 'YY_q0', 'YZ_q0',
                 'ZX_q0', 'ZY_q0', 'ZZ_q0']
        keys += ['XX_q1', 'XY_q1', 'XZ_q1',
                 'YX_q1', 'YY_q1', 'YZ_q1',
                 'ZX_q1', 'ZY_q1', 'ZZ_q1']
        for key in keys:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            for idx in ['x0', 'x1', 'x2', '0x', '1x', '2x']:
                self.proc_data_dict['V{}'.format(idx)][val_name] = \
                    np.nanmean(self.raw_data_dict['cal_pts_{}'.format(idx)]
                               [val_name])
            SI = np.nanmean(
                self.raw_data_dict['measured_values_ZZ'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_mZmZ'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            Px0, Px1, Px2, M_inv_q0 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['Vx0'][val_name],
                self.proc_data_dict['Vx1'][val_name],
                self.proc_data_dict['Vx2'][val_name])
            P0x, P1x, P2x, M_inv_q1 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['V0x'][val_name],
                self.proc_data_dict['V1x'][val_name],
                self.proc_data_dict['V2x'][val_name])

            for key, val in [('Px0', Px0), ('Px1', Px1), ('Px2', Px2),
                             ('P0x', P0x), ('P1x', P1x), ('P2x', P2x),
                             ('M_inv_q0', M_inv_q0), ('M_inv_q1', M_inv_q1)]:
                self.proc_data_dict[key][val_name] = val

            for key in ['XX', 'XY', 'XZ',
                        'YX', 'YY', 'YZ',
                        'ZX', 'ZY', 'ZZ']:
                Vmeas = self.raw_data_dict['measured_values_'+key][val_name]
                Px2 = self.proc_data_dict['Px2'][val_name]
                V0 = self.proc_data_dict['Vx0'][val_name]
                V1 = self.proc_data_dict['Vx1'][val_name]
                V2 = self.proc_data_dict['Vx2'][val_name]
                val = Vmeas+0  # - (Px2*V2 - (1-Px2)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(
                    val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key+'_q0'][val_name] = val*2-1

                P2x = self.proc_data_dict['P2x'][val_name]
                V0 = self.proc_data_dict['V0x'][val_name]
                V1 = self.proc_data_dict['V1x'][val_name]

                # Leakage is ignored in this analysis.
                # V2 = self.proc_data_dict['V2x'][val_name]
                val = Vmeas + 0  # - (P2x*V2 - (1-P2x)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(
                    val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key+'_q1'][val_name] = val*2-1

        if self.classification_method == 'rates':
            val_name_q0 = self.raw_data_dict['value_names'][self.rates_ch_idxs[0]]
            val_name_q1 = self.raw_data_dict['value_names'][self.rates_ch_idxs[1]]

            self.proc_data_dict['M0'] = (
                self.proc_data_dict['Px0'][val_name_q0] *
                self.proc_data_dict['P0x'][val_name_q1])

            self.proc_data_dict['X1'] = (
                1-self.proc_data_dict['Px2'][val_name_q0]
                - self.proc_data_dict['P2x'][val_name_q1])

            # The unitarity is calculated here.
            self.proc_data_dict['unitarity_shots'] = \
                self.proc_data_dict['ZZ_q0'][val_name_q0]*0

            # Unitarity according to Eq. (10) Wallman et al. New J. Phys. 2015
            # Pj = d/(d-1)*|n(rho_j)|^2
            # Note that the dimensionality prefix is ignored here as it
            # should drop out in the fits.
            for key in ['XX', 'XY', 'XZ',
                        'YX', 'YY', 'YZ',
                        'ZX', 'ZY', 'ZZ']:
                self.proc_data_dict[key] = (
                    self.proc_data_dict[key+'_q0'][val_name_q0]
                    * self.proc_data_dict[key+'_q1'][val_name_q1])
                self.proc_data_dict[key+'_sq'] = self.proc_data_dict[key]**2

                self.proc_data_dict['unitarity_shots'] += \
                    self.proc_data_dict[key+'_sq']

            self.proc_data_dict['unitarity'] = np.mean(
                self.proc_data_dict['unitarity_shots'], axis=1)
        else:
            raise NotImplementedError()

    def run_fitting(self):
        super().run_fitting()
        self.fit_res['unitarity_decay'] = self.fit_unitarity_decay()

        unitarity_dec = self.fit_res['unitarity_decay'].params

        text_msg = 'Summary: \n'
        text_msg += format_value_string('Unitarity\n' +
                                        r'$u$', unitarity_dec['u'], '\n')
        text_msg += format_value_string(
            'Error due to\nincoherent mechanisms\n'+r'$\epsilon$',
            unitarity_dec['eps'])

        self.proc_data_dict['unitarity_msg'] = text_msg

    def fit_unitarity_decay(self):
        """Fits the data using the unitarity model."""
        fit_mod_unitarity = lmfit.Model(unitarity_decay, independent_vars='m')
        fit_mod_unitarity.set_param_hint(
            'A', value=.1, min=0, max=1, vary=True)
        fit_mod_unitarity.set_param_hint(
            'B', value=.8, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint(
            'u', value=.9, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint('d1', value=self.d1, vary=False)
        # Error due to incoherent sources
        # Feng Phys. Rev. Lett. 117, 260501 (2016) eq. (4)
        fit_mod_unitarity.set_param_hint('eps', expr='((d1-1)/d1)*(1-u**0.5)')

        params = fit_mod_unitarity.make_params()
        fit_mod_unitarity = fit_mod_unitarity.fit(
            data=self.proc_data_dict['unitarity'],
            m=self.proc_data_dict['ncl'], params=params)

        return fit_mod_unitarity

    def prepare_plots(self):
        val_names = self.proc_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bins'],
                'yvals': np.nanmean(
                    self.proc_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.proc_data_dict['binned_vals'][val_name],
                             axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string'] +
                '\n'+self.proc_data_dict['measurementstring'],
            }
        fs = plt.rcParams['figure.figsize']

        # define figure and axes here to have custom layout
        self.figs['rb_populations_decay'], axs = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(fs[0]*1.5, fs[1]))
        self.figs['rb_populations_decay'].suptitle(
            self.proc_data_dict['timestamp_string']+'\n' +
            'Population using rate equations', y=1.05)
        self.figs['rb_populations_decay'].patch.set_alpha(0)
        self.axs['rb_pops_q0'] = axs[0]
        self.axs['rb_pops_q1'] = axs[1]

        val_name_q0 = val_names[self.rates_ch_idxs[0]]
        val_name_q1 = val_names[self.rates_ch_idxs[1]]
        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q0)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['Px0'][val_name_q0],
            'P1': self.proc_data_dict['Px1'][val_name_q0],
            'P2': self.proc_data_dict['Px2'][val_name_q0],
            'title': ' {}'.format(val_name_q0),
            'ax_id': 'rb_pops_q0'}

        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q1)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['P0x'][val_name_q1],
            'P1': self.proc_data_dict['P1x'][val_name_q1],
            'P2': self.proc_data_dict['P2x'][val_name_q1],
            'title': ' {}'.format(val_name_q1),
            'ax_id': 'rb_pops_q1'}

        self.plot_dicts['cal_points_hexbin_q0'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_x0'][val_names[0]],
                        self.proc_data_dict['cal_pts_x0'][val_names[1]]),
            'shots_1': (self.proc_data_dict['cal_pts_x1'][val_names[0]],
                        self.proc_data_dict['cal_pts_x1'][val_names[1]]),
            'shots_2': (self.proc_data_dict['cal_pts_x2'][val_names[0]],
                        self.proc_data_dict['cal_pts_x2'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.proc_data_dict['value_units'][1],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q0',
            'plotsize': (fs[0]*1.5, fs[1])
        }
        self.plot_dicts['cal_points_hexbin_q1'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_0x'][val_names[2]],
                        self.proc_data_dict['cal_pts_0x'][val_names[3]]),
            'shots_1': (self.proc_data_dict['cal_pts_1x'][val_names[2]],
                        self.proc_data_dict['cal_pts_1x'][val_names[3]]),
            'shots_2': (self.proc_data_dict['cal_pts_2x'][val_names[2]],
                        self.proc_data_dict['cal_pts_2x'][val_names[3]]),
            'xlabel': val_names[2],
            'xunit': self.proc_data_dict['value_units'][2],
            'ylabel': val_names[3],
            'yunit': self.proc_data_dict['value_units'][3],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q1',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}

        self.plot_dicts['correlated_readouts'] = {
            'plotfn': plot_unitarity_shots,
            'ncl': self.proc_data_dict['ncl'],
            'unitarity_shots': self.proc_data_dict['unitarity_shots'],
            'xlabel': 'Number of Cliffords',
            'xunit': '#',
            'ylabel': 'Unitarity',
            'yunit': '',
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'],
        }

        self.figs['unitarity'] = plt.subplots(nrows=1)
        self.plot_dicts['unitarity'] = {
            'plotfn': plot_unitarity,
            'ax_id': 'unitarity',
            'ncl': self.proc_data_dict['ncl'],
            'P': self.proc_data_dict['unitarity'],
            'xlabel': 'Number of Cliffords',
            'xunit': '#',
            'ylabel': 'Unitarity',
            'yunit': 'frac',
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'],
        }
        self.plot_dicts['fit_unitarity'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'unitarity',
            'fit_res': self.fit_res['unitarity_decay'],
            'setlabel': 'Simple unitarity fit',
            'do_legend': True,
        }
        self.plot_dicts['unitarity_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['unitarity_msg'],
            'xpos': 0.6, 'ypos': .8, 'ax_id': 'unitarity',
            'horizontalalignment': 'left'}


class InterleavedRandomizedBenchmarkingAnalysis(ba.BaseDataAnalysis):
    """
    Analysis for two qubit interleaved randomized benchmarking of a CZ gate.

    This is a meta-analysis. It runs
    "RandomizedBenchmarking_TwoQubit_Analysis" for each of the individual
    datasets in the "extract_data" method and uses the quantities of interest
    to create the combined figure.

    The figure as well as the quantities of interest are stored in
    the interleaved data file.
    """

    def __init__(self, ts_base: str, ts_int: str,
                 label_base: str='', label_int: str='',
                 options_dict: dict={}, auto=True, close_figs=True,
                 ch_idxs: list =[2, 0],
                 ignore_f_cal_pts: bool=False, plot_label=''):
        super().__init__(do_fitting=True, close_figs=close_figs,
                         options_dict=options_dict)
        self.ts_base = ts_base
        self.ts_int = ts_int
        self.label_base = label_base
        self.label_int = label_int
        self.ch_idxs = ch_idxs
        self.options_dict = options_dict
        self.close_figs = close_figs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        self.plot_label=plot_label
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a_int = RandomizedBenchmarking_TwoQubit_Analysis(
            t_start=self.ts_int, label=self.label_int,
            options_dict=self.options_dict, auto=True,
            close_figs=self.close_figs, rates_ch_idxs=self.ch_idxs,
            extract_only=True, ignore_f_cal_pts=self.ignore_f_cal_pts)

        a_base = RandomizedBenchmarking_TwoQubit_Analysis(
            t_start=self.ts_base, label=self.label_base,
            options_dict=self.options_dict, auto=True,
            close_figs=self.close_figs, rates_ch_idxs=self.ch_idxs,
            extract_only=True, ignore_f_cal_pts=self.ignore_f_cal_pts)

        # order is such that any information (figures, quantities of interest)
        # are saved in the interleaved file.
        self.timestamps = [a_int.timestamps[0], a_base.timestamps[0]]

        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['timestamp_string'] = \
            a_int.raw_data_dict['timestamp_string']
        self.raw_data_dict['folder'] = a_int.raw_data_dict['folder']
        self.raw_data_dict['analyses'] = {'base': a_base, 'int': a_int}

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']

        qoi_base = self.raw_data_dict['analyses']['base'].\
            proc_data_dict['quantities_of_interest']
        qoi_int = self.raw_data_dict['analyses']['int'].\
            proc_data_dict['quantities_of_interest']

        qoi.update({k+'_ref': v for k, v in qoi_base.items()})
        qoi.update({k+'_int': v for k, v in qoi_int.items()})

        qoi['eps_CZ_X1'] = interleaved_error(eps_int=qoi_int['eps_X1'],
                                             eps_base=qoi_base['eps_X1'])
        qoi['eps_CZ_simple'] = interleaved_error(
            eps_int=qoi_int['eps_simple'], eps_base=qoi_base['eps_simple'])
        qoi['L1_CZ'] = interleaved_error(eps_int=qoi_int['L1'],
                                         eps_base=qoi_base['L1'])

        # This is the naive estimate, when all observed error is assigned
        # to the CZ gate
        try:
            qoi['L1_CZ_naive'] = 1-(1-qoi_base['L1'])**(1/1.5)
            qoi['eps_CZ_simple_naive'] = 1-(1-qoi_base['eps_X1'])**(1/1.5)
            qoi['eps_CZ_X1_naive'] = 1-(1-qoi_base['eps_simple'])**(1/1.5)
        except ValueError:
            # prevents the analysis from crashing if the fits are bad.
            qoi['L1_CZ_naive'] = ufloat(np.NaN, np.NaN)
            qoi['eps_CZ_simple_naive'] = ufloat(np.NaN, np.NaN)
            qoi['eps_CZ_X1_naive'] = ufloat(np.NaN, np.NaN)

    def prepare_plots(self):
        dd_base = self.raw_data_dict['analyses']['base'].proc_data_dict
        dd_int = self.raw_data_dict['analyses']['int'].proc_data_dict

        fr_base = self.raw_data_dict['analyses']['base'].fit_res
        fr_int = self.raw_data_dict['analyses']['int'].fit_res

        self.figs['main_irb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_irb_decay'].patch.set_alpha(0)
        self.axs['main_irb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_irb_decay'] = {
            'plotfn': plot_irb_decay_woods_gambetta,

            'ncl':     dd_base['ncl'],
            'M0_ref':    dd_base['M0'],
            'M0_int':    dd_int['M0'],
            'X1_ref':    dd_base['X1'],
            'X1_int':   dd_int['X1'],
            'fr_M0_ref':  fr_base['rb_decay'],
            'fr_M0_int':  fr_int['rb_decay'],
            'fr_M0_simple_ref':  fr_base['rb_decay_simple'],
            'fr_M0_simple_int':  fr_int['rb_decay_simple'],
            'fr_X1_ref':  fr_base['leakage_decay'],
            'fr_X1_int':  fr_int['leakage_decay'],
            'qoi': self.proc_data_dict['quantities_of_interest'],
            'ax1': axs[1],
            'title': '{}\n{} - {}'.format(
                self.plot_label,
                self.timestamps[0], self.timestamps[1])}


class CharacterBenchmarking_TwoQubit_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for character benchmarking.
    """

    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 ch_idxs: list =[0, 2]):
        if options_dict is None:
            options_dict = dict()
        super().__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs,
            do_fitting=True)

        self.d1 = 4
        self.ch_idxs = ch_idxs
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()
        bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
        a.finish()

        self.raw_data_dict['measurementstring'] = a.measurementstring
        self.raw_data_dict['timestamp_string'] = a.timestamp_string
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps

        df = pd.DataFrame(
            columns={'ncl', 'pauli', 'I_q0', 'Q_q0', 'I_q1', 'Q_q1',
                     'interleaving_cl'})
        df['ncl'] = bins

        # Assumptions on the structure of the datafile are made here.
        # For every Clifford, 4 random pauli's are sampled from the different
        # sub sets:
        paulis = ['II',  # 'IZ', 'ZI', 'ZZ',  # P00
                  'IX',  # 'IY', 'ZX', 'ZY',  # P01
                  'XI',  # 'XZ', 'YI', 'YZ',  # P10
                  'XX']  # 'XY', 'YX', 'YY']  # P11

        paulis_df = np.tile(paulis, 34)[:len(bins)]
        # The calibration points do not correspond to a Pauli
        paulis_df[-7:] = np.nan
        df['pauli'] = paulis_df

        # The four different random Pauli's are performed both with
        # and without the interleaving CZ gate.
        df['interleaving_cl'] = np.tile(
            ['']*4 + ['CZ']*4, len(bins)//8+1)[:len(bins)]

        # Data is grouped and single shots are averaged.
        for i, ch in enumerate(['I_q0', 'Q_q0', 'I_q1', 'Q_q1']):
            binned_yvals = np.reshape(
                a.measured_values[i], (len(bins), -1), order='F')
            yvals = np.mean(binned_yvals, axis=1)
            df[ch] = yvals

        self.raw_data_dict['df'] = df

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        df = self.raw_data_dict['df']
        cal_points = [
            # calibration point indices are when ignoring the f-state cal pts
            [[-7, -5], [-6, -4], [-3, -1]],  # q0
            [[-7, -5], [-6, -4], [-3, -1]],  # q0
            [[-7, -6], [-5, -4], [-2, -1]],  # q1
            [[-7, -6], [-5, -4], [-2, -1]],  # q1
        ]

        for ch, cal_pt in zip(['I_q0', 'Q_q0', 'I_q1', 'Q_q1'], cal_points):
            df[ch+'_normed'] = a_tools.normalize_data_v3(
                df[ch].values,
                cal_zero_points=cal_pt[0],
                cal_one_points=cal_pt[1])

        df['P_|00>'] = (1-df['I_q0_normed'])*(1-df['Q_q1_normed'])

        P00 = df.loc[df['pauli'].isin(['II', 'IZ', 'ZI', 'ZZ'])]\
            .loc[df['interleaving_cl'] == ''].groupby('ncl').mean()
        P01 = df.loc[df['pauli'].isin(['IX', 'IY', 'ZX', 'ZY'])]\
            .loc[df['interleaving_cl'] == ''].groupby('ncl').mean()
        P10 = df.loc[df['pauli'].isin(['XI', 'XZ', 'YI', 'YZ'])]\
            .loc[df['interleaving_cl'] == ''].groupby('ncl').mean()
        P11 = df.loc[df['pauli'].isin(['XX', 'XY', 'YX', 'YY'])]\
            .loc[df['interleaving_cl'] == ''].groupby('ncl').mean()

        P00_CZ = df.loc[df['pauli'].isin(['II', 'IZ', 'ZI', 'ZZ'])]\
            .loc[df['interleaving_cl'] == 'CZ'].groupby('ncl').mean()
        P01_CZ = df.loc[df['pauli'].isin(['IX', 'IY', 'ZX', 'ZY'])]\
            .loc[df['interleaving_cl'] == 'CZ'].groupby('ncl').mean()
        P10_CZ = df.loc[df['pauli'].isin(['XI', 'XZ', 'YI', 'YZ'])]\
            .loc[df['interleaving_cl'] == 'CZ'].groupby('ncl').mean()
        P11_CZ = df.loc[df['pauli'].isin(['XX', 'XY', 'YX', 'YY'])]\
            .loc[df['interleaving_cl'] == 'CZ'].groupby('ncl').mean()

        # Calculate the character function
        # Eq. 7 of Xue et al. ArXiv 1811.04002v1
        C1 = P00['P_|00>']-P01['P_|00>']+P10['P_|00>']-P11['P_|00>']
        C2 = P00['P_|00>']+P01['P_|00>']-P10['P_|00>']-P11['P_|00>']
        C12 = P00['P_|00>']-P01['P_|00>']-P10['P_|00>']+P11['P_|00>']
        C1_CZ = P00_CZ['P_|00>']-P01_CZ['P_|00>'] + \
            P10_CZ['P_|00>']-P11_CZ['P_|00>']
        C2_CZ = P00_CZ['P_|00>']+P01_CZ['P_|00>'] - \
            P10_CZ['P_|00>']-P11_CZ['P_|00>']
        C12_CZ = P00_CZ['P_|00>']-P01_CZ['P_|00>'] - \
            P10_CZ['P_|00>']+P11_CZ['P_|00>']

        char_df = pd.DataFrame(
            {'P00': P00['P_|00>'], 'P01': P01['P_|00>'],
             'P10': P10['P_|00>'], 'P11': P11['P_|00>'],
             'P00_CZ': P00_CZ['P_|00>'], 'P01_CZ': P01_CZ['P_|00>'],
             'P10_CZ': P10_CZ['P_|00>'], 'P11_CZ': P11_CZ['P_|00>'],
             'C1': C1, 'C2': C2, 'C12': C12,
             'C1_CZ': C1_CZ, 'C2_CZ': C2_CZ, 'C12_CZ': C12_CZ})
        self.proc_data_dict['char_df'] = char_df

    def run_fitting(self):
        super().run_fitting()

        char_df = self.proc_data_dict['char_df']
        # Eq. 8 of Xue et al. ArXiv 1811.04002v1
        for char_key in ['C1', 'C2', 'C12', 'C1_CZ', 'C2_CZ', 'C12_CZ']:
            char_mod = lmfit.Model(char_decay, independent_vars='m')
            char_mod.set_param_hint('A', value=1, vary=True)
            char_mod.set_param_hint('alpha', value=.95)
            params = char_mod.make_params()
            self.fit_res[char_key] = char_mod.fit(
                data=char_df[char_key].values,
                m=char_df.index, params=params)

    def analyze_fit_results(self):
        fr = self.fit_res
        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']
        qoi['alpha1'] = ufloat(fr['C1'].params['alpha'].value,
                               fr['C1'].params['alpha'].stderr)
        qoi['alpha2'] = ufloat(fr['C2'].params['alpha'].value,
                               fr['C2'].params['alpha'].stderr)
        qoi['alpha12'] = ufloat(fr['C12'].params['alpha'].value,
                                fr['C12'].params['alpha'].stderr)
        # eq. 9 from Xue et al. ArXiv 1811.04002v1
        qoi['alpha_char'] = 3/15*qoi['alpha1']+3/15*qoi['alpha2']\
            + 9/15*qoi['alpha12']

        qoi['alpha1_CZ_int'] = ufloat(fr['C1_CZ'].params['alpha'].value,
                                      fr['C1_CZ'].params['alpha'].stderr)
        qoi['alpha2_CZ_int'] = ufloat(fr['C2_CZ'].params['alpha'].value,
                                      fr['C2_CZ'].params['alpha'].stderr)
        qoi['alpha12_CZ_int'] = ufloat(fr['C12_CZ'].params['alpha'].value,
                                       fr['C12_CZ'].params['alpha'].stderr)

        qoi['alpha_char_CZ_int'] = 3/15*qoi['alpha1_CZ_int'] \
            + 3/15*qoi['alpha2_CZ_int'] + 9/15*qoi['alpha12_CZ_int']

        qoi['eps_ref'] = depolarizing_par_to_eps(qoi['alpha_char'], d=4)
        qoi['eps_int'] = depolarizing_par_to_eps(qoi['alpha_char_CZ_int'], d=4)
        # Interleaved error calculation  Magesan et al. PRL 2012
        qoi['eps_CZ'] = 1-(1-qoi['eps_int'])/(1-qoi['eps_ref'])

    def prepare_plots(self):
        char_df = self.proc_data_dict['char_df']

        fs = plt.rcParams['figure.figsize']

        # self.figs['puali_decays']
        self.plot_dicts['pauli_decays'] = {
            'plotfn': plot_char_RB_pauli_decays,
            'ncl': char_df.index.values,
            'P00': char_df['P00'].values,
            'P01': char_df['P01'].values,
            'P10': char_df['P10'].values,
            'P11': char_df['P11'].values,
            'P00_CZ': char_df['P00_CZ'].values,
            'P01_CZ': char_df['P01_CZ'].values,
            'P10_CZ': char_df['P10_CZ'].values,
            'P11_CZ': char_df['P11_CZ'].values,
            'title': self.raw_data_dict['measurementstring']
            + '\n'+self.raw_data_dict['timestamp_string']
            + '\nPauli decays',
        }
        self.plot_dicts['char_decay'] = {
            'plotfn': plot_char_RB_decay,
            'ncl': char_df.index.values,
            'C1': char_df['C1'].values,
            'C2': char_df['C2'].values,
            'C12': char_df['C12'].values,
            'C1_CZ': char_df['C1_CZ'].values,
            'C2_CZ': char_df['C2_CZ'].values,
            'C12_CZ': char_df['C12_CZ'].values,
            'fr_C1': self.fit_res['C1'],
            'fr_C2': self.fit_res['C2'],
            'fr_C12': self.fit_res['C12'],
            'fr_C1_CZ': self.fit_res['C1_CZ'],
            'fr_C2_CZ': self.fit_res['C2_CZ'],
            'fr_C12_CZ': self.fit_res['C12_CZ'],
            'title': self.raw_data_dict['measurementstring']
            + '\n'+self.raw_data_dict['timestamp_string']
            + '\nCharacter decay',
        }
        self.plot_dicts['quantities_msg'] = {
            'plotfn': plot_char_rb_quantities,
            'ax_id': 'char_decay',
            'qoi': self.proc_data_dict['quantities_of_interest']}


def plot_cal_points_hexbin(shots_0,
                           shots_1,
                           shots_2,
                           xlabel: str, xunit: str,
                           ylabel: str, yunit: str,
                           title: str,
                           ax,
                           common_clims: bool=True,
                           **kw):
    # Choose colormap
    alpha_cmaps = []
    for cmap in [pl.cm.Blues, pl.cm.Reds, pl.cm.Greens]:
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        alpha_cmaps.append(my_cmap)

    f = plt.gcf()
    hb2 = ax.hexbin(x=shots_2[0], y=shots_2[1], cmap=alpha_cmaps[2])
    cb = f.colorbar(hb2, ax=ax)
    cb.set_label(r'Counts $|2\rangle$')

    hb1 = ax.hexbin(x=shots_1[0], y=shots_1[1], cmap=alpha_cmaps[1])
    cb = f.colorbar(hb1, ax=ax)
    cb.set_label(r'Counts $|1\rangle$')

    hb0 = ax.hexbin(x=shots_0[0], y=shots_0[1], cmap=alpha_cmaps[0])
    cb = f.colorbar(hb0, ax=ax)
    cb.set_label(r'Counts $|0\rangle$')

    if common_clims:
        clims = hb0.get_clim(), hb1.get_clim(), hb2.get_clim()
        clim = np.min(clims), np.max(clims)
        for hb in hb0, hb1, hb2:
            hb.set_clim(clim)

    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)


def plot_raw_RB_curve(ncl, SI, SX, V0, V1, V2, title, ax,
                      xlabel, xunit, ylabel, yunit, **kw):
    ax.plot(ncl, SI, label='SI', marker='o')
    ax.plot(ncl, SX, label='SX', marker='o')
    ax.plot(ncl[-1]+.5, V0, label='V0', marker='d', c='C0')
    ax.plot(ncl[-1]+1.5, V1, label='V1', marker='d', c='C1')
    ax.plot(ncl[-1]+2.5, V2, label='V2', marker='d', c='C2')
    ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.legend()


def plot_populations_RB_curve(ncl, P0, P1, P2, title, ax, **kw):
    ax.axhline(.5, c='k', lw=.5, ls='--')
    ax.plot(ncl, P0, c='C0', label=r'P($|g\rangle$)', marker='v')
    ax.plot(ncl, P1, c='C3', label=r'P($|e\rangle$)', marker='^')
    ax.plot(ncl, P2, c='C2', label=r'P($|f\rangle$)', marker='d')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('Population')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def plot_unitarity_shots(ncl, unitarity_shots, title, ax=None, **kw):
    ax.axhline(.5, c='k', lw=.5, ls='--')

    ax.plot(ncl, unitarity_shots, '.')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('unitarity')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)


def plot_unitarity(ncl, P, title, ax=None, **kw):
    ax.plot(ncl, P, 'o')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('unitarity')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def plot_char_RB_pauli_decays(ncl, P00, P01, P10, P11,
                              P00_CZ, P01_CZ, P10_CZ, P11_CZ,
                              title, ax, **kw):
    """
    Plots the raw recovery probabilities for a character RB experiment.
    """
    ax.plot(ncl, P00, c='C0', label=r'$P_{00}$', marker='o', ls='--')
    ax.plot(ncl, P01, c='C1', label=r'$P_{01}$', marker='o', ls='--')
    ax.plot(ncl, P10, c='C2', label=r'$P_{10}$', marker='o', ls='--')
    ax.plot(ncl, P11, c='C3', label=r'$P_{11}$', marker='o', ls='--')

    ax.plot(ncl, P00_CZ, c='C0', label=r'$P_{00}$-int. CZ',
            marker='d', alpha=.5, ls=':')
    ax.plot(ncl, P01_CZ, c='C1', label=r'$P_{01}$-int. CZ',
            marker='d', alpha=.5, ls=':')
    ax.plot(ncl, P10_CZ, c='C2', label=r'$P_{10}$-int. CZ',
            marker='d', alpha=.5, ls=':')
    ax.plot(ncl, P11_CZ, c='C3', label=r'$P_{11}$-int. CZ',
            marker='d', alpha=.5, ls=':')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel(r'$P |00\rangle$')
    ax.legend(loc=(1.05, 0))
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def plot_char_RB_decay(ncl, C1, C2, C12,
                       C1_CZ, C2_CZ, C12_CZ,
                       fr_C1, fr_C2, fr_C12,
                       fr_C1_CZ, fr_C2_CZ, fr_C12_CZ,
                       title, ax, **kw):

    ncl_fine = np.linspace(np.min(ncl), np.max(ncl), 101)

    plot_fit(ncl_fine, fr_C1, ax, ls='-', c='C0')
    ax.plot(ncl, C1, c='C0', label=r'$C_1$: $A_1\cdot {\alpha_{1|2}}^m$',
            marker='o', ls='')
    plot_fit(ncl_fine, fr_C2, ax, ls='-', c='C1')
    ax.plot(ncl, C2, c='C1', label=r'$C_2$: $A_1\cdot {\alpha_{2|1}}^m$',
            marker='o', ls='')
    plot_fit(ncl_fine, fr_C12, ax, ls='-', c='C2')
    ax.plot(ncl, C12, c='C2', label=r'$C_{12}$: $A_1\cdot {\alpha_{12}}^m$',
            marker='o', ls='')

    plot_fit(ncl_fine, fr_C1_CZ, ax, ls='--', c='C0', alpha=.5)
    ax.plot(ncl, C1_CZ, c='C0',
            label=r"$C_1^{int.}$: $A_1' \cdot {\alpha_{1|2}'}^m$",
            marker='d', ls='', alpha=.5)
    plot_fit(ncl_fine, fr_C2_CZ, ax, ls='--', c='C1', alpha=.5)
    ax.plot(ncl, C2_CZ, c='C1',
            label=r"$C_2^{int.}$: $A_2' \cdot {\alpha_{2|1}'}^m$",
            marker='d', ls='', alpha=.5)
    plot_fit(ncl_fine, fr_C12_CZ, ax, ls='--', c='C2', alpha=.5)
    ax.plot(ncl, C12_CZ, c='C2',
            label=r"$C_{12}^{int.}$: $A_{12}' \cdot {\alpha_{12}'}^m$",
            marker='d', ls='', alpha=.5)

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('Population')
    ax.legend(title='Character decay',
              ncol=2, loc=(1.05, 0.6))

    ax.set_title(title)


def plot_char_rb_quantities(ax, qoi, **kw):
    """
    Plots a text message of the main quantities extracted from char rb
    """
    def gen_val_str(alpha, alpha_p):

        val_str = '   {:.3f}$\pm${:.3f}    {:.3f}$\pm${:.3f}'
        return val_str.format(alpha.nominal_value, alpha.std_dev,
                              alpha_p.nominal_value, alpha_p.std_dev)

    alpha_msg = '            Reference         Interleaved'
    alpha_msg += '\n'r'$\alpha_{1|2}$'+'\t'
    alpha_msg += gen_val_str(qoi['alpha1'], qoi['alpha1_CZ_int'])
    alpha_msg += '\n'r'$\alpha_{2|1}$'+'\t'
    alpha_msg += gen_val_str(qoi['alpha2'], qoi['alpha2_CZ_int'])
    alpha_msg += '\n'r'$\alpha_{12}$'+'\t'
    alpha_msg += gen_val_str(qoi['alpha12'], qoi['alpha12_CZ_int'])
    alpha_msg += '\n' + '_'*40+'\n'

    alpha_msg += '\n'r'$\epsilon_{Ref.}$'+'\t'
    alpha_msg += '{:.3f}$\pm${:.3f}%'.format(
        qoi['eps_ref'].nominal_value*100, qoi['eps_ref'].std_dev*100)
    alpha_msg += '\n'r'$\epsilon_{Int.}$'+'\t'
    alpha_msg += '{:.3f}$\pm${:.3f}%'.format(
        qoi['eps_int'].nominal_value*100, qoi['eps_int'].std_dev*100)
    alpha_msg += '\n'r'$\epsilon_{CZ.}$'+'\t'
    alpha_msg += '{:.3f}$\pm${:.3f}%'.format(
        qoi['eps_CZ'].nominal_value*100, qoi['eps_CZ'].std_dev*100)

    ax.text(1.05, 0.0, alpha_msg, transform=ax.transAxes)


def logisticreg_classifier_machinelearning(shots_0, shots_1, shots_2):
    """
    """
    # reshaping of the entries in proc_data_dict
    shots_0 = np.array(list(
        zip(list(shots_0.values())[0],
            list(shots_0.values())[1])))

    shots_1 = np.array(list(
        zip(list(shots_1.values())[0],
            list(shots_1.values())[1])))
    shots_2 = np.array(list(
        zip(list(shots_2.values())[0],
            list(shots_2.values())[1])))

    shots_0 = shots_0[~np.isnan(shots_0[:, 0])]
    shots_1 = shots_1[~np.isnan(shots_1[:, 0])]
    shots_2 = shots_2[~np.isnan(shots_2[:, 0])]

    X = np.concatenate([shots_0, shots_1, shots_2])
    Y = np.concatenate([0*np.ones(shots_0.shape[0]),
                        1*np.ones(shots_1.shape[0]),
                        2*np.ones(shots_2.shape[0])])

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    return logreg


def plot_classifier_decission_boundary(shots_0, shots_1, shots_2,
                                       classifier,
                                       xlabel: str, xunit: str,
                                       ylabel: str, yunit: str,
                                       title: str, ax, **kw):
    """
    Plot decision boundary on top of the hexbin plot of the training dataset.
    """
    grid_points = 200

    x_min = np.nanmin([shots_0[0], shots_1[0], shots_2[0]])
    x_max = np.nanmax([shots_0[0], shots_1[0], shots_2[0]])
    y_min = np.nanmin([shots_0[1], shots_1[1], shots_2[1]])
    y_max = np.nanmax([shots_0[1], shots_1[1], shots_2[1]])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                         np.linspace(y_min, y_max, grid_points))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plot_cal_points_hexbin(shots_0=shots_0,
                           shots_1=shots_1,
                           shots_2=shots_2,
                           xlabel=xlabel, xunit=xunit,
                           ylabel=ylabel, yunit=yunit,
                           title=title, ax=ax)
    ax.pcolormesh(xx, yy, Z,
                  cmap=c.ListedColormap(['C0', 'C3', 'C2']),
                  alpha=.2)


def plot_rb_decay_woods_gambetta(ncl, M0, X1, ax, ax1, title='', **kw):
    ax.plot(ncl, M0, marker='o', linestyle='')
    ax1.plot(ncl, X1, marker='d', linestyle='')
    ax.grid(axis='y')
    ax1.grid(axis='y')
    ax.set_ylim(-.05, 1.05)
    ax1.set_ylim(min(min(.97*X1), .92), 1.01)
    ax.set_ylabel(r'$M_0$ probability')
    ax1.set_ylabel(r'$X_1$ population')
    ax1.set_xlabel('Number of Cliffords')
    ax.set_title(title)


def plot_irb_decay_woods_gambetta(
        ncl, M0_ref, M0_int,
        X1_ref, X1_int,
        fr_M0_ref, fr_M0_int,
        fr_M0_simple_ref, fr_M0_simple_int,
        fr_X1_ref, fr_X1_int,
        qoi,
        ax, ax1, title='', **kw):

    ncl_fine = np.linspace(ncl[0], ncl[-1], 1001)

    ax.plot(ncl, M0_ref, marker='o', linestyle='', c='C0', label='Reference')
    plot_fit(ncl_fine, fr_M0_ref, ax=ax, c='C0')

    ax.plot(ncl, M0_int, marker='d', linestyle='', c='C1', label='Interleaved')
    plot_fit(ncl_fine, fr_M0_int, ax=ax, c='C1')

    ax.grid(axis='y')
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel(r'$M_0$ probability')

    ax1.plot(ncl, X1_ref, marker='o', linestyle='',
             label='Reference', c='C0')
    ax1.plot(ncl, X1_int, marker='d', linestyle='', c='C1')

    plot_fit(ncl_fine, fr_X1_ref, ax=ax1, c='C0')
    plot_fit(ncl_fine, fr_X1_int, ax=ax1, c='C1')

    ax1.grid(axis='y')

    ax1.set_ylim(min(min(.97*X1_int), .92), 1.01)
    ax1.set_ylabel(r'$X_1$ population')
    ax1.set_xlabel('Number of Cliffords')
    ax.set_title(title)
    ax.legend(loc=(1.05, .6))

    collabels = ['$\epsilon_{X1}$ (%)', '$\epsilon$ (%)', 'L1 (%)']
    rowlabels = ['Ref. curve', 'Int. curve', 'CZ-int.', 'CZ-naive']
    table_data = [
        [qoi['eps_X1_ref']*100, qoi['eps_simple_ref']*100,
            qoi['L1_ref']*100],
        [qoi['eps_X1_int']*100, qoi['eps_simple_int']*100, qoi['L1_int']*100],
        [qoi['eps_CZ_X1']*100, qoi['eps_CZ_simple']*100, qoi['L1_CZ']*100],
        [qoi['eps_CZ_X1_naive']*100, qoi['eps_CZ_simple_naive']*100,
            qoi['L1_CZ_naive']*100], ]
    ax.table(cellText=table_data,
             colLabels=collabels,
             rowLabels=rowlabels,
             transform=ax1.transAxes,
             bbox=(1.25, 0.05, .7, 1.4))


def interleaved_error(eps_int, eps_base):
    # Interleaved error calculation  Magesan et al. PRL 2012
    eps = 1-(1-eps_int)/(1-eps_base)
    return eps


def leak_decay(A, B, lambda_1, m):
    """
    Eq. (9) of Wood Gambetta 2018.

        A ~= L2/ (L1+L2)
        B ~= L1/ (L1+L2) + eps_m
        lambda_1 = 1 - L1 - L2

    """
    return A + B*lambda_1**m


def full_rb_decay(A, B, C, lambda_1, lambda_2, m):
    """Eq. (15) of Wood Gambetta 2018."""
    return A + B*lambda_1**m+C*lambda_2**m


def unitarity_decay(A, B, u, m):
    """Eq. (8) of Wallman et al. New J. Phys. 2015."""
    return A + B*u**m


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
    return A * alpha**m


def format_value_string(par_name: str, lmfit_par, end_char=''):
    """Format an lmfit par to a  string of value with uncertainty."""
    val_string = par_name
    val_string += ': {:.4f}'.format(lmfit_par.value)
    if lmfit_par.stderr is not None:
        val_string += r'$\pm$' + '{:.4f}'.format(lmfit_par.stderr)
    else:
        val_string += r'$\pm$' + 'NaN'
    val_string += end_char
    return val_string


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
    return (1-alpha)*(d-1)/d
