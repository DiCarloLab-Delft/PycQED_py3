"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis
"""

import lmfit
import logging
import itertools
from collections import OrderedDict
import numpy as np
import pycqed.analysis.fitting_models as fit_mods
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from scipy.optimize import minimize
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
import pycqed.analysis.tools.data_manipulation as dm_tools

from matplotlib.colors import LinearSegmentedColormap as lscmap


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
        post_select_threshold = \
            self.options_dict.get('post_select_threshold', 0)
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
            'title': (self.timestamps[0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0])}
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


class MultiQubit_SingleShot_Analysis(ba.BaseDataAnalysis):
    """
    Extracts table of counts from multiplexed single shot readout experiment.
    Intended to be the bases class for more complex multi qubit experiment
    analysis.


    Required options in the options_dict:
        n_readouts: Assumed to be the period in the list of shots between
            experiments with the same prepared state. If shots_of_qubits
            includes preselection readout results or if there was several
            readouts for a single readout then n_readouts has to include them.
        channel_map: dictionary with qubit names as keys and channel channel
            names as values.
        thresholds: dictionary with qubit names as keys and threshold values as
            values.
    Optional options in the options_dict:
        observables: Dictionary with observable names as a key and observable
            as a value. Observable is a dictionary with name of the qubit as
            key and boolean value indicating if it is selecting exited states.
            If the qubit is missing from the list of states it is averaged out.
        readout_names: used as y-axis labels for the default figure
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.n_readouts = options_dict['n_readouts']
        self.thresholds = options_dict['thresholds']
        self.channel_map = options_dict['channel_map']
        qubits = list(self.channel_map.keys())

        self.readout_names = options_dict.get('readout_names', None)
        if self.readout_names is None:
            # TODO Default values should come from the MC parameters
            None

        self.observables = options_dict.get('observables', None)
        if self.observables is None:
            combination_list = list(itertools.product([False, True],
                                                      repeat=len(qubits)))
            self.observables = {}
            for i, states in enumerate(combination_list):
                name = ''.join(['e' if s else 'g' for s in states])
                obs_name = '$\| ' + name + '\\rangle$'
                self.observables[obs_name] = dict(zip(qubits, states))
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
        shots_thresh = {}
        logging.info("Loading from file")

        for qubit, channel in self.channel_map.items():
            shots_cont = np.array(
                self.raw_data_dict['measured_values_ord_dict'][channel])
            shots_thresh[qubit] = (shots_cont > self.thresholds[qubit])[0]
        self.proc_data_dict['shots_thresholded'] = shots_thresh

        logging.info("Calculating observables")
        self.proc_data_dict['probability_table'] = self.probability_table(
                shots_thresh,
                list(self.observables.values()),
                self.n_readouts
        )

    @staticmethod
    def probability_table(shots_of_qubits, observables, n_readouts):
        """
        Creates a general table of counts averaging out all but specified set of
        correlations.

        Args:
            shots_of_qubits: Dictionary of np.arrays of thresholded shots for
                each qubit.
            observables: List of observables. Observable is a dictionary with
                name of the qubit as key and boolean value indicating if it is
                selecting exited states. If the qubit is missing from the list
                of states it is averaged out. Instead of just the qubit name, a
                tuple of qubit name and a shift value can be passed, where the
                shift value specifies the relative readout index for which the
                state is checked.
            n_readouts: Assumed to be the period in the list of shots between
                experiments with the same prepared state. If shots_of_qubits
                includes preselection readout results or if there was several
                readouts for a single readout then n_readouts has to include
                them.
        Returns:
            np.array: counts with
                dimensions (n_readouts, len(states_to_be_counted))
        """

        res_e = {}
        res_g = {}

        n_shots = next(iter(shots_of_qubits.values())).shape[0]

        table = np.zeros((n_readouts, len(observables)))

        for qubit, results in shots_of_qubits.items():
            res_e[qubit] = np.array(results).reshape((n_readouts, -1),
                                                     order='F')
            # This makes copy, but allows faster AND later
            res_g[qubit] = np.logical_not(
                np.array(results)).reshape((n_readouts, -1), order='F')

        for readout_n in range(n_readouts):
            # first result all ground
            for state_n, states_of_qubits in enumerate(observables):
                mask = np.ones((n_shots//n_readouts), dtype=np.bool)
                # slow qubit is the first in channel_map list
                for qubit, state in states_of_qubits.items():
                    if isinstance(qubit, tuple):
                        seg = (readout_n+qubit[1]) % n_readouts
                        qubit = qubit[0]
                    else:
                        seg = readout_n
                    if state:
                        mask = np.logical_and(mask, res_e[qubit][seg])
                    else:
                        mask = np.logical_and(mask, res_g[qubit][seg])
                table[readout_n, state_n] = np.count_nonzero(mask)

        return table*n_readouts/n_shots

    @staticmethod
    def observable_product(*observables):
        """
        Finds the product-observable of the input observables.
        If the observable conditions are contradicting, returns None. For the
        format of the observables, see the docstring of `probability_table`.
        """
        res_obs = {}
        for obs in observables:
            for k in obs:
                if k in res_obs:
                    if obs[k] != res_obs[k]:
                        return None
                else:
                    res_obs[k] = obs[k]
        return res_obs



    def prepare_plots(self):
        self.prepare_plot_prob_table()

    def prepare_plot_prob_table(self, only_odd=False):
        # colormap which has a lot of contrast for small and large values
        v = [0, 0.1, 0.2, 0.8, 1]
        c = [(1, 1, 1),
             (191/255, 38/255, 11/255),
             (155/255, 10/255, 106/255),
             (55/255, 129/255, 214/255),
             (0, 0, 0)]
        cdict = {'red':   [(v[i], c[i][0], c[i][0]) for i in range(len(v))],
                 'green': [(v[i], c[i][1], c[i][1]) for i in range(len(v))],
                 'blue':  [(v[i], c[i][2], c[i][2]) for i in range(len(v))]}
        cm = lscmap('customcmap', cdict)

        if only_odd:
            ylist = list(range(int(self.n_readouts/2)))
            plt_data = self.proc_data_dict['probability_table'][1::2].T
        else:
            ylist = list(range(self.n_readouts))
            plt_data = self.proc_data_dict['probability_table'].T

        plot_dict = {
            'axid': "ptable",
            'plotfn': self.plot_colorx,
            'xvals': np.arange(len(self.observables)),
            'yvals': np.array(len(self.observables)*[ylist]),
            'zvals': plt_data,
            'xlabel': "Channels",
            'ylabel': "Segments",
            'zlabel': "Counts",
            'zrange': [0,1],
            'title': (self.timestamps[0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'xunit': None,
            'yunit': None,
            'xtick_loc': np.arange(len(self.observables)),
            'xtick_labels': list(self.observables.keys()),
            'origin': 'upper',
            'cmap': cm,
            'aspect': 'equal',
            'plotsize': (8, 8)
        }

        # todo to not rely on readout names
        if self.readout_names is not None:
            if only_odd:
                plot_dict['ytick_loc'] = \
                    np.arange(len(self.readout_names[1::2]))
                plot_dict['ytick_labels'] = self.readout_names[1::2]
            else:
                plot_dict['ytick_loc'] = np.arange(len(self.readout_names))
                plot_dict['ytick_labels'] = self.readout_names

        self.plot_dicts['counts_table'] = plot_dict

    def measurement_operators_and_results(self):
        """
        Calculates and returns:
            A tuple of
                count tables for each data segment for the observables;
                the measurement operators corresponding to each observable;
                and the expected covariation matrix between the operators.

        If the calibration segments are passed, there must be a calibration
        segments for each of the computational basis states of the Hilber space.
        If there are no calibration segments, perfect readout is assumed.
        """
        qubits = list(self.channel_map.keys())
        d = 2**len(qubits)
        data = self.proc_data_dict['probability_table']
        data = data / data[0].sum()
        data = data.T
        if not 'cal_points' in self.options_dict:
            Fsingle = {None: np.array([[1, 0], [0, 1]]),
                       True: np.array([[0, 0], [0, 1]]),
                       False: np.array([[1, 0], [0, 0]])}
            Fs = []
            Omega = []
            for obs in self.observables.values():
                F = np.array([[1]])
                nr_meas = 0
                for qb in qubits:
                    Fqb = Fsingle[obs.get(qb, None)]
                    # Kronecker product convention - assumed the same as QuTiP
                    F = np.kron(F, Fqb)
                    if qb in obs:
                        nr_meas += 1
                Fs.append(F)
                # The variation is proportional to the number of qubits we have
                # a condition on, assuming that all readout errors are small
                # and equal.
                Omega.append(nr_meas)
            Omega = np.array(Omega)
            return data, Fs, Omega
        else:
            means, covars = \
                self.calibration_point_means_and_channel_covariations()
            cal_points_list = self.proc_data_dict['cal_points_list']
            data_idx = np.arange(data.shape[1])
            data_idx = np.setdiff1d(data_idx,
                                    np.array(cal_points_list).flatten())
            data = data[:, data_idx]
            Fs = [np.diag(ms) for ms in means.T]
            return data, Fs, covars

    def calibration_point_means_and_channel_covariations(self):
        observables = list(self.observables.values())

        # calculate the mean for each reference state and each observable
        try:
            cal_points_list = convert_channel_names_to_index(
                self.options_dict.get('cal_points'), self.n_readouts,
                self.raw_data_dict['value_names'][0]
            )
        except KeyError:
            cal_points_list = convert_channel_names_to_index(
                self.options_dict.get('cal_points'), self.n_readouts,
                list(self.channel_map.keys())
            )
        self.proc_data_dict['cal_points_list'] = cal_points_list
        means = np.zeros((len(cal_points_list), len(observables)))
        cal_readouts = set()
        for i, cal_point in enumerate(cal_points_list):
            for j, cal_point_chs in enumerate(cal_point):
                if j == 0:
                    readout_list = cal_point_chs
                else:
                    if readout_list != cal_point_chs:
                        raise Exception('Different readout indices for a '
                                        'single reference state: {} and {}'
                                        .format(readout_list, cal_point_chs))
            cal_readouts.update(cal_point[0])
            val_list = [self.proc_data_dict['probability_table'][idx_ro]
                        for idx_ro in cal_point[0]]
            means[i] = np.mean(val_list, axis=0)

        # find the means for all the products of the operators and the average
        # covariation of the operators
        prod_obss = []
        prod_obs_idxs = {}
        obs_products = np.zeros([self.n_readouts] + [len(observables)]*2)
        for i, obsi in enumerate(observables):
            for j, obsj in enumerate(observables):
                if i > j:
                    continue
                obsp = self.observable_product(obsi, obsj)
                if obsp is None:
                    obs_products[:, i, j] = 0
                    obs_products[:, j, i] = 0
                else:
                    prod_obs_idxs[(i, j)] = len(prod_obss)
                    prod_obs_idxs[(j, i)] = len(prod_obss)
                    prod_obss.append(obsp)
        prod_prob_table = self.probability_table(
            self.proc_data_dict['shots_thresholded'],
            prod_obss, self.n_readouts)
        for (i, j), k in prod_obs_idxs.items():
            obs_products[:, i, j] = prod_prob_table[:, k]
        covars = -np.array([np.outer(ro, ro) for ro in self.proc_data_dict[
            'probability_table']]) + obs_products
        covars = np.mean(covars[list(cal_readouts)], 0)

        return means, covars


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


class Multiplexed_Readout_Analysis(MultiQubit_SingleShot_Analysis):
    """
    Analysis results of an experiment meant for characterization of multiplexed
    readout.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        self.n_readouts = options_dict['n_readouts']
        self.channel_map = options_dict['channel_map']
        qubits = list(self.channel_map.keys())

        def_seg_names_prep = ["".join(l) for l in list(
            itertools.product(["$0$", "$\pi$"],
                              repeat=len(self.channel_map)))]
        preselection = False
        if self.n_readouts == len(def_seg_names_prep):
            def_seg_names = def_seg_names_prep
        elif self.n_readouts == 2*len(def_seg_names_prep):
            preselection = True
            def_seg_names = [x for t in zip(*[
                ["sel"]*len(def_seg_names_prep),
                def_seg_names_prep]) for x in t]
        else:
            def_seg_names = list(range(len(def_seg_names_prep)))

        # User can override the automatic value determined from the
        #   number of readouts
        self.preselection = options_dict.get('preselection', preselection)

        self.observables = options_dict.get('observables', None)
        if self.observables is None:
            combination_list = list(itertools.product([False, True],
                                                      repeat=len(qubits)))
            preselection_condition = dict(zip(
                [(qb, 1) for qb in qubits],  # keys contain shift
                combination_list[0]  # first comb has all ground
            ))

            self.observables = OrderedDict([])
            # add preselection condition also as an observable
            if self.preselection:
                self.observables["pre"] = preselection_condition
            # add all combinations
            for i, states in enumerate(combination_list):
                obs_name = '$\| ' + \
                           ''.join(['e' if s else 'g' for s in states]) + \
                           '\\rangle$'
                self.observables[obs_name] = dict(zip(qubits, states))
                # add preselection condition
                if self.preselection:
                    self.observables[obs_name].update(preselection_condition)

        options_dict['observables'] = self.observables
        options_dict['readout_names'] = options_dict.get('readout_names',
                                                         def_seg_names)

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting,
                         auto=False)

        # here we can do more stuff before analysis runs

        if auto:
            self.run_analysis()

    def prepare_plots(self):
        super().prepare_plot_prob_table(only_odd=self.preselection)

def convert_channel_names_to_index(cal_points, nr_segments, value_names):
    """
    Converts the calibration points list from the format
    cal_points = [{'ch1': [-4, -3], 'ch2': [-4, -3]},
                  {0: [-2, -1], 1: [-2, -1]}]
    to the format (for a 100-segment dataset)
    cal_points_list = [[[96, 97], [96, 97]],
                       [[98, 99], [98, 99]]]

    Args:
        cal_points: the list of calibration points to convert
        nr_segments: number of segments in the dataset to convert negative
                     indices to positive indices.
        value_names: a list of channel names that is used to determine the
                     index of the channels
    Returns:
        cal_points_list in the converted format
    """
    cal_points_list = []
    for observable in cal_points:
        if isinstance(observable, (list, np.ndarray)):
            observable_list = [[]] * len(value_names)
            for i, idxs in enumerate(observable):
                observable_list[i] = \
                    [idx % nr_segments for idx in idxs]
            cal_points_list.append(observable_list)
        else:
            observable_list = [[]] * len(value_names)
            for channel, idxs in observable.items():
                if isinstance(channel, int):
                    observable_list[channel] = \
                        [idx % nr_segments for idx in idxs]
                else:  # assume str
                    ch_idx = value_names.index(channel)
                    observable_list[ch_idx] = \
                        [idx % nr_segments for idx in idxs]
            cal_points_list.append(observable_list)
    return cal_points_list
