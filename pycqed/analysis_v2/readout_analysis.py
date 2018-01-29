"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis
"""

import lmfit
from collections import OrderedDict
import numpy as np
import pycqed.analysis.fitting_models as fit_mods
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from scipy.optimize import minimize
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
import pycqed.analysis.tools.data_manipulation as dm_tools


class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Responsible for creating the histograms based on the raw data
        """
        # Determine the shape of the data to extract wheter to rotate or not
        post_select = self.options_dict.get('post_select', False)
        post_select_threshold = self.options_dict.get('post_select_threshold', 0)
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)
        nr_bins = self.options_dict.get('nr_bins', 100)

        nr_expts = self.raw_data_dict['nr_experiments']
        self.proc_data_dict['shots_0'] = [''] * nr_expts
        self.proc_data_dict['shots_1'] = [''] * nr_expts
        self.proc_data_dict['nr_shots'] = [0] * nr_expts

        ######################################################
        #  Separating data into shots for 0 and shots for 1  #
        ######################################################

        for i, meas_val in enumerate(self.raw_data_dict['measured_values']):
            dat = meas_val[0]  # hardcoded for 1 channel for now
            sh_0, sh_1 = get_shots_zero_one(
                dat, post_select=post_select, nr_samples=nr_samples,
                post_select_threshold=post_select_threshold,
                sample_0=sample_0, sample_1=sample_1)
            min_sh = np.min(sh_0)
            max_sh = np.max(sh_1)
            self.proc_data_dict['shots_0'][i] = sh_0
            self.proc_data_dict['shots_1'][i] = sh_1
            self.proc_data_dict['nr_shots'][i] = len(sh_0)

        ##################################
        #  Binning data into histograms  #
        ##################################
        self.proc_data_dict['hist_0'] = np.histogram(
            self.proc_data_dict['shots_0'][0], bins=nr_bins,
            range=(min_sh, max_sh))
        # range given to ensure both use histograms use same bins
        self.proc_data_dict['hist_1'] = np.histogram(
            self.proc_data_dict['shots_1'][0], bins=nr_bins,
            range=(min_sh, max_sh))

        self.proc_data_dict['bin_centers'] = (
            self.proc_data_dict['hist_0'][1][:-1] +
            self.proc_data_dict['hist_0'][1][1:]) / 2
        self.proc_data_dict['binsize'] = (self.proc_data_dict['hist_0'][1][1] -
                                          self.proc_data_dict['hist_0'][1][0])
        ##########################
        #  Cumulative histograms #
        ##########################

        # the cumulative histograms are normalized to ensure the right
        # fidelities can be calculated
        self.proc_data_dict['cumhist_0'] = np.cumsum(
            self.proc_data_dict['hist_0'][0])/(
                self.proc_data_dict['nr_shots'][0])
        self.proc_data_dict['cumhist_1'] = np.cumsum(
            self.proc_data_dict['hist_1'][0])/(
                self.proc_data_dict['nr_shots'][0])

        self.proc_data_dict['shots_xlabel'] = \
            self.raw_data_dict['value_names'][0][0]
        self.proc_data_dict['shots_xunit'] = \
            self.raw_data_dict['value_units'][0][0]

        ###########################################################
        #  Threshold and fidelity based on cumulative histograms  #
        ###########################################################
        # Average assignment fidelity: F_ass = (P01 - P10 )/2
        # where Pxy equals probability to measure x when starting in y
        F_vs_th = (1-(1-abs(self.proc_data_dict['cumhist_1'] -
                            self.proc_data_dict['cumhist_0']))/2)
        opt_idx = np.argmax(F_vs_th)
        self.proc_data_dict['F_assignment_raw'] = F_vs_th[opt_idx]
        self.proc_data_dict['threshold_raw'] = \
            self.proc_data_dict['bin_centers'][opt_idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        # An initial guess is done on the single guassians to constrain
        # the fit params and avoid fitting noise if
        # e.g., mmt. ind. rel. is very low
        gmod0 = lmfit.models.GaussianModel()
        guess0 = gmod0.guess(data=self.proc_data_dict['hist_0'][0],
                             x=self.proc_data_dict['bin_centers'])
        gmod1 = lmfit.models.GaussianModel()
        guess1 = gmod1.guess(data=self.proc_data_dict['hist_1'][0],
                             x=self.proc_data_dict['bin_centers'])

        DoubleGaussMod_0 = (lmfit.models.GaussianModel(prefix='A_') +
                            lmfit.models.GaussianModel(prefix='B_'))
        DoubleGaussMod_0.set_param_hint(
            'A_center',
            value=guess0['center'],
            min=guess0['center']-2*guess0['sigma'],
            max=guess0['center']+2*guess0['sigma'])
        DoubleGaussMod_0.set_param_hint(
            'B_center',
            value=guess1['center'],
            min=guess1['center']-2*guess1['sigma'],
            max=guess1['center']+2*guess1['sigma'])
        # This way of assigning makes the guess function a method of the
        # model (ensures model/self is passed as first argument.
        DoubleGaussMod_0.guess = fit_mods.double_gauss_guess.__get__(
            DoubleGaussMod_0, DoubleGaussMod_0.__class__)

        # Two instances of the model are required to avoid
        # memory interference between the two fits
        DoubleGaussMod_1 = (lmfit.models.GaussianModel(prefix='A_') +
                            lmfit.models.GaussianModel(prefix='B_'))
        DoubleGaussMod_1.set_param_hint(
            'A_center',
            value=guess0['center'],
            min=guess0['center']-2*guess0['sigma'],
            max=guess0['center']+2*guess0['sigma'])
        DoubleGaussMod_1.set_param_hint(
            'B_center',
            value=guess1['center'],
            min=guess1['center']-2*guess1['sigma'],
            max=guess1['center']+2*guess1['sigma'])
        DoubleGaussMod_1.guess = fit_mods.double_gauss_guess.__get__(
            DoubleGaussMod_1, DoubleGaussMod_1.__class__)

        self.fit_dicts['shots_0'] = {
            'model': DoubleGaussMod_0,
            # 'fit_guess_fn': DoubleGaussMod_0.guess,
            'fit_xvals': {'x': self.proc_data_dict['bin_centers']},
            'fit_yvals': {'data': self.proc_data_dict['hist_0'][0]}}

        self.fit_dicts['shots_1'] = {
            'model': DoubleGaussMod_1,
            # 'fit_guess_fn': DoubleGaussMod_1.guess,
            'fit_xvals': {'x': self.proc_data_dict['bin_centers']},
            'fit_yvals': {'data': self.proc_data_dict['hist_1'][0]}}

    def analyze_fit_results(self):
        # Create a CDF based on the fit functions of both fits.
        fr_0 = self.fit_res['shots_0']
        fr_1 = self.fit_res['shots_1']
        bv0 = fr_0.best_values
        bv1 = fr_1.best_values
        norm_factor = 1/(self.proc_data_dict['binsize'] *
                         self.proc_data_dict['nr_shots'][0])

        def CDF_0(x):
            return fit_mods.double_gaussianCDF(
                x, A_amplitude=bv0['A_amplitude']*norm_factor,
                A_sigma=bv0['A_sigma'], A_mu=bv0['A_center'],
                B_amplitude=bv0['B_amplitude']*norm_factor,
                B_sigma=bv0['B_sigma'], B_mu=bv0['B_center'])

        def CDF_1(x):
            return fit_mods.double_gaussianCDF(
                x, A_amplitude=bv1['A_amplitude']*norm_factor,
                A_sigma=bv1['A_sigma'], A_mu=bv1['A_center'],
                B_amplitude=bv1['B_amplitude']*norm_factor,
                B_sigma=bv1['B_sigma'], B_mu=bv1['B_center'])

        def infid_vs_th(x):
            return (1-abs(CDF_0(x) - CDF_1(x)))/2

        self._CDF_0 = CDF_0
        self._CDF_1 = CDF_1
        self._infid_vs_th = infid_vs_th

        opt_fid = minimize(infid_vs_th, (bv0['A_center']+bv0['B_center'])/2)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid['fun'], float):
            self.proc_data_dict['F_assignment_fit'] = (1-opt_fid['fun'])
        else:
            self.proc_data_dict['F_assignment_fit'] = (1-opt_fid['fun'])[0]

        self.proc_data_dict['threshold_fit'] = opt_fid['x'][0]

        # Calculate the fidelity of both

        ###########################################
        #  Extracting the discrimination fidelity #
        ###########################################
        if bv0['A_amplitude'] > bv0['B_amplitude']:
            mu_0 = bv0['A_center']
            sigma_0 = bv0['A_sigma']
            # under the assumption of perfect gates and no mmt induced exc.
            self.proc_data_dict['residual_excitation'] = \
                norm_factor * bv0['B_amplitude']
        else:
            mu_0 = bv0['B_center']
            sigma_0 = bv0['B_sigma']
            # under the assumption of perfect gates and no mmt induced exc.
            self.proc_data_dict['residual_excitation'] = \
                norm_factor * bv0['A_amplitude']

        if bv1['A_amplitude'] > bv1['B_amplitude']:
            mu_1 = bv1['A_center']
            sigma_1 = bv1['A_sigma']
            # under the assumption of perfect gates and no mmt induced exc.
            self.proc_data_dict['measurement_induced_relaxation'] = \
                norm_factor * bv1['B_amplitude']

        else:
            mu_1 = bv1['B_center']
            sigma_1 = bv1['B_sigma']
            # under the assumption of perfect gates and no mmt induced exc.
            self.proc_data_dict['measurement_induced_relaxation'] = \
                norm_factor * bv1['A_amplitude']

        def CDF_0_discr(x):
            return fit_mods.gaussianCDF(x, 1, mu=mu_0, sigma=sigma_0)

        def CDF_1_discr(x):
            return fit_mods.gaussianCDF(x, 1, mu=mu_1, sigma=sigma_1)

        def disc_infid_vs_th(x):
            return (1-abs(CDF_0_discr(x) - CDF_1_discr(x)))/2

        self._CDF_0_discr = CDF_0_discr
        self._CDF_1_discr = CDF_1_discr
        self._disc_infid_vs_th = disc_infid_vs_th

        opt_fid = minimize(disc_infid_vs_th, (mu_0 + mu_1)/2)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid['fun'], float):
            self.proc_data_dict['F_discr'] = (1-opt_fid['fun'])
        else:
            self.proc_data_dict['F_discr'] = (1-opt_fid['fun'])[0]

        self.proc_data_dict['threshold_discr'] = opt_fid['x'][0]

    def prepare_plots(self):
        # N.B. If the log option is used we should manually set the
        # yscale to go from .5 to the current max as otherwise the fits
        # mess up the log plots.
        log_hist = self.options_dict.get('log_hist', False)

        # The histograms
        self.plot_dicts['1D_histogram'] = {
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_0'][1],
            'yvals': self.proc_data_dict['hist_0'][0],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C0',
                        'edgecolor': 'C0'},
            'setlabel': 'Shots 0',
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts',
            'title': (self.proc_data_dict['timestamps'][0] + ' \n' +
                      self.proc_data_dict['measurementstring'][0])}
        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_1'][1],
            'yvals': self.proc_data_dict['hist_1'][0],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': 'Shots 1', 'do_legend': True,
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts'}

        # The cumulative histograms
        self.plot_dicts['cum_histogram'] = {
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_0'][1],
            'yvals': self.proc_data_dict['cumhist_0'],
            'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                        'edgecolor': 'C0'},
            'setlabel': 'Shots 0',
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts',
            'title': 'Cumulative Histograms' + '\n'+self.timestamps[0]}
        self.plot_dicts['cumhist_1'] = {
            'ax_id': 'cum_histogram',
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_1'][1],
            'yvals': self.proc_data_dict['cumhist_1'],
            'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': 'Shots 1', 'do_legend': True,
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts'}

        #####################################
        # Adding the fits to the figures    #
        #####################################
        if self.do_fitting:
            self.plot_dicts['fit_shots_0'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['shots_0']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit shots 0',
                'line_kws': {'color': 'C0'},
                'do_legend': True}
            self.plot_dicts['fit_shots_1'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['shots_1']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit shots 1',
                'line_kws': {'color': 'C3'},
                'do_legend': True}

            x = np.linspace(self.proc_data_dict['bin_centers'][0],
                            self.proc_data_dict['bin_centers'][-1], 100)
            self.plot_dicts['cdf_fit_shots_0'] = {
                'ax_id': 'cum_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_0(x),
                'setlabel': 'Fit shots 0',
                'line_kws': {'color': 'C0'},
                'marker': '',
                'do_legend': True}
            self.plot_dicts['cdf_fit_shots_1'] = {
                'ax_id': 'cum_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_1(x),
                'marker': '',
                'setlabel': 'Fit shots 1',
                'line_kws': {'color': 'C3'},
                'do_legend': True}

        ###########################################
        # Thresholds and fidelity information     #
        ###########################################

        if not self.presentation_mode:
            max_cnts = np.max([np.max(self.proc_data_dict['hist_0'][0]),
                               np.max(self.proc_data_dict['hist_1'][0])])

            thr, th_unit = SI_val_to_msg_str(
                self.proc_data_dict['threshold_raw'],
                self.proc_data_dict['shots_xunit'], return_type=float)

            raw_th_msg = (
                'Raw threshold: {:.2f} {}\n'.format(
                    thr, th_unit) +
                r'$F_{A}$-raw: ' +
                r'{:.3f}'.format(
                    self.proc_data_dict['F_assignment_raw']))
            self.plot_dicts['cumhist_threshold'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_vlines,
                'x': self.proc_data_dict['threshold_raw'],
                'ymin': 0,
                'ymax': max_cnts*1.05,
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8},
                'setlabel': raw_th_msg,
                'do_legend': True}
            if self.do_fitting:
                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_fit'],
                    self.proc_data_dict['shots_xunit'], return_type=float)
                fit_th_msg = (
                    'Fit threshold: {:.2f} {}\n'.format(
                        thr, th_unit) +
                     r'$F_{A}$-fit: '+
                     r'{:.3f}'.format(self.proc_data_dict['F_assignment_fit']))


                self.plot_dicts['fit_threshold'] = {
                    'ax_id': '1D_histogram',
                    'plotfn': self.plot_vlines,
                    'x': self.proc_data_dict['threshold_fit'],
                    'ymin': 0,
                    'ymax': max_cnts*1.05,
                    'colors': '.4',
                    'linestyles': 'dotted',
                    'line_kws': {'linewidth': .8},
                    'setlabel': fit_th_msg,
                    'do_legend': True}

                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_discr'],
                    self.proc_data_dict['shots_xunit'], return_type=float)
                fit_th_msg = (
                    'Discr. threshold: {:.2f} {}\n'.format(
                        thr, th_unit) +
                    r'$F_{D}$: ' +
                    ' {:.3f}'.format(self.proc_data_dict['F_discr']))
                self.plot_dicts['discr_threshold'] = {
                    'ax_id': '1D_histogram',
                    'plotfn': self.plot_vlines,
                    'x': self.proc_data_dict['threshold_discr'],
                    'ymin': 0,
                    'ymax': max_cnts*1.05,
                    'colors': '.3',
                    'linestyles': '-.',
                    'line_kws': {'linewidth': .8},
                    'setlabel': fit_th_msg,
                    'do_legend': True}

                # To add text only to the legend I create some "fake" data
                rel_exc_str = ('Mmt. Ind. Rel.: {:.1f}%\n'.format(
                    self.proc_data_dict['measurement_induced_relaxation']*100) +
                    'Residual Exc.: {:.1f}%'.format(
                        self.proc_data_dict['residual_excitation']*100))
                self.plot_dicts['rel_exc_msg'] = {
                    'ax_id': '1D_histogram',
                    'plotfn': self.plot_line,
                    'xvals': [self.proc_data_dict['threshold_discr']],
                    'yvals': [max_cnts/2],
                    'line_kws': {'alpha': 0},
                    'setlabel': rel_exc_str,
                    'do_legend': True}


class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
    pass


def get_shots_zero_one(data, post_select: bool=False,
                       nr_samples: int=2, sample_0: int=0, sample_1: int=1,
                       post_select_threshold: float = None):
    if not post_select:
        shots_0, shots_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)
    else:
        presel_0, presel_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)

        shots_0, shots_1 = a_tools.zigzag(
            data, sample_0+1, sample_1+1, nr_samples)


    if post_select:
        post_select_shots_0 = data[0::nr_samples]
        shots_0 = data[1::nr_samples]

        post_select_shots_1 = data[nr_samples//2::nr_samples]
        shots_1 = data[nr_samples//2+1::nr_samples]

        # Determine shots to remove
        post_select_indices_0 = dm_tools.get_post_select_indices(
            thresholds=[post_select_threshold],
            init_measurements=[post_select_shots_0])

        post_select_indices_1 = dm_tools.get_post_select_indices(
            thresholds=[post_select_threshold],
            init_measurements=[post_select_shots_1])

        shots_0[post_select_indices_0] = np.nan
        shots_0 = shots_0[~np.isnan(shots_0)]

        shots_1[post_select_indices_1] = np.nan
        shots_1 = shots_1[~np.isnan(shots_1)]

    return shots_0, shots_1