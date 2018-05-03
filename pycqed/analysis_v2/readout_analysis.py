"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis
"""
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import lmfit
from collections import OrderedDict
import numpy as np
import pycqed.analysis.fitting_models as fit_mods
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from scipy.optimize import minimize
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
import pycqed.analysis.tools.data_manipulation as dm_tools
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.utilities.general import int2base


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
        post_select = self.options_dict.get('post_select', False)
        post_select_threshold = self.options_dict.get(
            'post_select_threshold', 0)
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)
        nr_bins = self.options_dict.get('nr_bins', 100)

        nr_expts = self.raw_data_dict['nr_experiments']
        self.proc_data_dict['all_channel_int_voltages'] = [None] * nr_expts
        self.proc_data_dict['nr_shots'] = [0] * nr_expts
        self.proc_data_dict['eff_int_voltages'] = [None] * nr_expts
        self.proc_data_dict['min_sh'] = [0] * nr_expts
        self.proc_data_dict['max_sh'] = [0] * nr_expts
        self.proc_data_dict['cumsum_x'] = [None] * nr_expts
        self.proc_data_dict['cumsum_y'] = [None] * nr_expts
        self.proc_data_dict['cumsum_x_ds'] = [None] * nr_expts
        self.proc_data_dict['cumsum_y_ds'] = [None] * nr_expts
        self.proc_data_dict['hist'] = [None] * nr_expts
        self.proc_data_dict['bin_centers'] = [None] * nr_expts
        self.proc_data_dict['bin_edges'] = [None] * nr_expts
        self.proc_data_dict['binsize'] = [0] * nr_expts
        self.proc_data_dict['hist'] = [None] * nr_expts
        self.proc_data_dict['shots_xlabel'] = [None] * nr_expts
        self.proc_data_dict['shots_xunit'] = [0] * nr_expts
        self.proc_data_dict['F_assignment_raw'] = [0] * nr_expts
        self.proc_data_dict['threshold_raw'] = [0] * nr_expts

        ######################################################
        #  Separating data into shots for 0 and shots for 1  #
        ######################################################

        for i, meas_val in enumerate(self.raw_data_dict['measured_values']):
            # loop through channels
            shots = np.zeros((2, len(meas_val),), dtype=np.ndarray)
            for j, dat in enumerate(meas_val):
                sh_0, sh_1 = get_shots_zero_one(
                    dat, post_select=post_select, nr_samples=nr_samples,
                    post_select_threshold=post_select_threshold,
                    sample_0=sample_0, sample_1=sample_1)
                shots[0, j] = sh_0
                shots[1, j] = sh_1
            #shots = np.array(shots, dtype=float)

            print(shots.shape)
            # Do we have two quadratures?
            if len(meas_val) == 2:
                # If we do, find and apply the effective/rotated integrated voltage
                self.proc_data_dict['all_channel_int_voltages'][i] = shots
                angle = self.options_dict.get('rotation_angle', 0)
                auto_angle = self.options_dict.get('auto_rotation_angle', True)
                if auto_angle:
                    ##########################################
                    #  Determining the rotation of the data  #
                    ##########################################
                    # todo!
                    raise NotImplementedError('Rotation detection not implemented yet!')
                # create matrix
                rot_mat = [[+np.cos(angle), -np.sin(angle)],
                           [+np.sin(angle), +np.cos(angle)]]
                # rotate data accordingly
                eff_sh = np.dot(rot_mat[0], shots)
            else:
                # If we have only one quadrature, use that (doh!)
                print(shots[:, 0])
                eff_sh = shots[:, 0]

            self.proc_data_dict['shots_xlabel'][i] = self.raw_data_dict['value_names'][i][0]
            self.proc_data_dict['shots_xunit'][i] = self.raw_data_dict['value_units'][i][0]
            print(eff_sh.shape)
            self.proc_data_dict['eff_int_voltages'][i] = eff_sh
            self.proc_data_dict['nr_shots'][i] = len(eff_sh[0])
            sh_min = min(np.min(eff_sh[0]), np.min(eff_sh[1]))
            sh_max = max(np.max(eff_sh[0]), np.max(eff_sh[1]))
            data_range = (sh_min, sh_max)

            eff_sh_sort = np.sort(list(eff_sh), axis=1)
            x0, n0 = np.unique(eff_sh_sort[0], return_counts=True)
            cumsum0 = np.cumsum(n0)
            x1, n1 = np.unique(eff_sh_sort[1], return_counts=True)
            cumsum1 = np.cumsum(n1)

            self.proc_data_dict['cumsum_x'][i] = [x0, x1]
            self.proc_data_dict['cumsum_y'][i] = [cumsum0, cumsum1]

            print(x0, x1)
            all_x = np.unique(np.sort(np.concatenate((x0, x1))))
            md = self.options_dict.get('max_datapoints', 1000)
            if len(all_x) > md:
                all_x = np.linspace(*data_range, md)
            ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
            necumsum0 = ecumsum0/np.max(ecumsum0)
            ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
            necumsum1 = ecumsum1/np.max(ecumsum1)

            self.proc_data_dict['cumsum_x_ds'][i] = all_x
            self.proc_data_dict['cumsum_y_ds'][i] = [ecumsum0, ecumsum1]

            ##################################
            #  Binning data into histograms  #
            ##################################
            h0, bin_edges = np.histogram(eff_sh[0], bins=nr_bins,
                                         range=data_range)
            h1, bin_edges = np.histogram(eff_sh[1], bins=nr_bins,
                                         range=data_range)
            self.proc_data_dict['hist'][i] = [h0, h1]
            binsize = (bin_edges[1] - bin_edges[0])
            self.proc_data_dict['bin_edges'][i] = bin_edges
            self.proc_data_dict['bin_centers'][i] = bin_edges[:-1]+binsize
            self.proc_data_dict['binsize'][i] = binsize

            #######################################################
            #  Threshold and fidelity based on culmulative counts #
            #######################################################
            # Average assignment fidelity: F_ass = (P01 - P10 )/2
            # where Pxy equals probability to measure x when starting in y
            F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
            opt_idx = np.argmax(F_vs_th)
            self.proc_data_dict['F_assignment_raw'][i] = F_vs_th[opt_idx]
            self.proc_data_dict['threshold_raw'][i] = all_x[opt_idx]
            print('raw', all_x[opt_idx])

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        nr_expts = self.raw_data_dict['nr_experiments']
        self.proc_data_dict['F_discr'] = [None]*nr_expts
        self.proc_data_dict['threshold_discr'] = [None]*nr_expts
        self.proc_data_dict['F_assignment_fit'] = [None]*nr_expts
        self.proc_data_dict['threshold_fit'] = [None]*nr_expts

        for i in range(nr_expts):
            bin_x = self.proc_data_dict['bin_centers'][i]
            bin_ys = self.proc_data_dict['hist'][i]
            m = lmfit.model.Model(ro_gauss)
            m.guess = fit_mods.double_gauss_guess_2.__get__(m, m.__class__)
            #params = m.guess()
            #res = m.fit(x, ro_g, **params)
            #m_cul = lmfit.model.Model(ro_gauss_cul)


            self.fit_dicts['shots_all_%d'%i] = {
                'model': m,
                'fit_xvals': {'x': [bin_x, bin_x]},
                'fit_yvals': {'data': bin_ys},
                'guessfn_pars': {'fixed_p01': self.options_dict.get('fixed_p01', False),
                                 'fixed_p10': self.options_dict.get('fixed_p10', False)},
            }
            cdf_xs = self.proc_data_dict['cumsum_x'][i]
            cdf_ys = self.proc_data_dict['cumsum_y'][i]

    def analyze_fit_results(self):
        #nr_expts = self.raw_data_dict['nr_experiments']
        #for i in range(nr_expts):
        i = 0

        # Create a CDF based on the fit functions of both fits.
        fr = self.fit_res['shots_all_%d'%i]
        bv = fr.best_values

        bvn = deepcopy(bv)
        bvn['A_amplitude'] = 1
        bvn['B_amplitude'] = 1
        def CDF(x):
            return ro_CDF(x=x, **bvn)

        def CDF_0(x):
            return CDF(x=[x, x])[0]

        def CDF_1(x):
            return CDF(x=[x, x])[1]

        def infid_vs_th(x):
            cdf = ro_CDF(x=[x, x], **bvn)
            return (1-np.abs(cdf[0] - cdf[1]))/2

        self._CDF_0 = CDF_0
        self._CDF_1 = CDF_1
        self._infid_vs_th = infid_vs_th

        thr_guess = (bv['B_center'] - bv['A_center'])/2
        opt_fid = minimize(infid_vs_th, thr_guess)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid['fun'], float):
            self.proc_data_dict['F_assignment_fit'][i] = (1-opt_fid['fun'])
        else:
            self.proc_data_dict['F_assignment_fit'][i] = (1-opt_fid['fun'])[0]

        self.proc_data_dict['threshold_fit'][i] = opt_fid['x'][0]

        # Calculate the fidelity of both

        ###########################################
        #  Extracting the discrimination fidelity #
        ###########################################

        def CDF_0_discr(x):
            return fit_mods.gaussianCDF(x, amplitude=1,
                                        mu=bv['A_center'], sigma=bv['A_sigma'])

        def CDF_1_discr(x):
            return fit_mods.gaussianCDF(x, amplitude=1,
                                        mu=bv['B_center'], sigma=bv['B_sigma'])

        def disc_infid_vs_th(x):
            cdf0 = fit_mods.gaussianCDF(x, amplitude=1, mu=bv['A_center'],
                                        sigma=bv['A_sigma'])
            cdf1 = fit_mods.gaussianCDF(x, amplitude=1, mu=bv['B_center'],
                                        sigma=bv['B_sigma'])
            return (1-np.abs(cdf0 - cdf1))/2

        self._CDF_0_discr = CDF_0_discr
        self._CDF_1_discr = CDF_1_discr
        self._disc_infid_vs_th = disc_infid_vs_th

        opt_fid_discr = minimize(disc_infid_vs_th, thr_guess)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid_discr['fun'], float):
            self.proc_data_dict['F_discr'][i] = (1-opt_fid_discr['fun'])
        else:
            self.proc_data_dict['F_discr'][i] = (1-opt_fid_discr['fun'])[0]

        self.proc_data_dict['threshold_discr'][i] = opt_fid_discr['x'][0]

    def prepare_plots(self):
        # N.B. If the log option is used we should manually set the
        # yscale to go from .5 to the current max as otherwise the fits
        # mess up the log plots.
        i = 0
        x_label = self.proc_data_dict['shots_xlabel'][i]
        x_unit = self.proc_data_dict['shots_xunit'][i]

        ### The histograms
        log_hist = self.options_dict.get('log_hist', False)
        bin_x = self.proc_data_dict['bin_edges'][i]
        bin_y = self.proc_data_dict['hist'][i]
        self.plot_dicts['1D_histogram'] = {
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[0],
            'xwidth' : self.proc_data_dict['binsize'][i],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C0',
                        'edgecolor': 'C0'},
            'setlabel': 'Shots 0',
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': 'Counts',
            'title': (self.timestamps[0] + ' \n' +
                      self.raw_data_dict['measurementstring'][i])}

        th_raw = self.proc_data_dict['threshold_raw'][i]
        threshs = [th_raw,]
        if self.do_fitting:
            threshs.append(self.proc_data_dict['threshold_fit'])
            threshs.append(self.proc_data_dict['threshold_discr'])
        self.plot_dicts['v_lines_hist'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_vlines_auto,
            'xdata': threshs,
            'linestyles': ['--', '-.', ':'],
            'labels': ['th_raw', 'th_fit', 'th_d'],
            'colors': ['0.3', '0.5', '0.2'],
            'do_legend': True,
        }
        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[1],
            'xwidth' : self.proc_data_dict['binsize'][i],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': 'Shots 1', 'do_legend': True,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': 'Counts'}
        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[1],
            'xwidth' : self.proc_data_dict['binsize'][i],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': 'Shots 1', 'do_legend': True,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': 'Counts'}


        ######## PLOT CDF
        self.plot_dicts['v_lines_cdf'] = deepcopy(self.plot_dicts['v_lines_hist'])
        self.plot_dicts['v_lines_cdf']['ax_id'] = 'cdf'
        cdf_xs = self.proc_data_dict['cumsum_x'][i]
        cdf_ys = self.proc_data_dict['cumsum_y'][i]
        cdf_ys[0] = cdf_ys[0]/np.max(cdf_ys[0])
        cdf_ys[1] = cdf_ys[1]/np.max(cdf_ys[1])

        self.plot_dicts['cdf_shots_0'] = {
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[0],
            'yvals': cdf_ys[0],
            'setlabel': 'CDF shots 0',
            'line_kws': {'color': 'C0', 'alpha': 0.3},
            'marker': '',
            'do_legend': True}
        self.plot_dicts['cdf_shots_1'] = {
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[1],
            'yvals': cdf_ys[1],
            'setlabel': 'CDF shots 0',
            'line_kws': {'color': 'C3', 'alpha': 0.3},
            'marker': '',
            'do_legend': True}

        # self.plot_dicts['2D_shots_0'] = {
        #     'ax_id': '2D_shots',
        #     'plotfn': self.plot_xy,
        #     'xvals': self.proc_data_dict['all_channel_int_voltages'][i][0][0],
        #     'yvals': self.proc_data_dict['all_channel_int_voltages'][i][0][1],
        # }
        # self.plot_dicts['2D_histogram_0'] = {
        #     'ax_id': '2D_histogram',
        #     'plotfn': self.plot_xy,
        #     'xvals': self.proc_data_dict['all_channel_int_voltages'][i][0][0],
        #     'yvals': self.proc_data_dict['all_channel_int_voltages'][i][0][1],
        #     'yvals': None,
        # }


        # The cumulative histograms
        #####################################
        # Adding the fits to the figures    #
        #####################################
        if self.do_fitting:
            x = np.linspace(bin_x[0], bin_x[-1], 150)
            ro_g = ro_gauss(x=[x, x], **self.fit_res['shots_all_%d'%i].best_values)
            self.plot_dicts['new_fit_shots_0'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[0],
                'setlabel': 'Fit shots 0',
                'line_kws': {'color': 'C0'},
                'marker': '',
                'do_legend': True}
            self.plot_dicts['new_fit_shots_1'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[1],
                'marker': '',
                'setlabel': 'Fit shots 1',
                'line_kws': {'color': 'C3'},
                'do_legend': True}

            self.plot_dicts['cdf_fit_shots_0'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_0(x),
                'setlabel': 'Fit shots 0',
                'line_kws': {'color': 'C0', 'alpha': 1},
                'linestyle': '--',
                'marker': '',
                'do_legend': True}
            self.plot_dicts['cdf_fit_shots_1'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_1(x),
                'marker': '',
                'linestyle': '--',
                'setlabel': 'Fit shots 1',
                'line_kws': {'color': 'C3', 'alpha': 1},
                'do_legend': True}


        ###########################################
        # Thresholds and fidelity information     #
        ###########################################

        if not self.presentation_mode:
            thr, th_unit = SI_val_to_msg_str(
                self.proc_data_dict['threshold_raw'][i],
                x_unit, return_type=float)

            raw_th_msg = (
                'Raw threshold: {:.2f} {}\n'.format(
                    thr, th_unit) +
                r'$F_{A}$-raw: ' +
                r'{:.3f}'.format(
                    self.proc_data_dict['F_assignment_raw'][i]))
            if self.do_fitting:
                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_fit'],
                    self.proc_data_dict['shots_xunit'], return_type=float)
                fit_th_msg = (
                    'Fit threshold: {:.2f} {}\n'.format(
                        thr, th_unit) +
                    r'$F_{A}$-fit: ' +
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
    """
    For two qubits, to make an n-qubit mux readout experiment.
    we should vectorize this analysis
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 nr_of_qubits: int = 2,
                 qubit_names: list=None,
                 do_fitting: bool=True, auto=True):
        """
        Inherits from BaseDataAnalysis.
        Extra arguments of interest
            qubit_names (list) : used to label the experiments, names of the
                qubits. LSQ is last name in the list. If not specified will
                set qubit_names to [qN, ..., q1, q0]


        """
        self.nr_of_qubits = nr_of_qubits
        if qubit_names is None:
            self.qubit_names = list(reversed(['q{}'.format(i)
                                              for i in range(nr_of_qubits)]))
        else:
            self.qubit_names = qubit_names

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
        nr_bins = self.options_dict.get('nr_bins', 100)

        # self.proc_data_dict['shots_0'] = [''] * nr_expts
        # self.proc_data_dict['shots_1'] = [''] * nr_expts

        #################################################################
        #  Separating data into shots for the different prepared states #
        #################################################################
        self.proc_data_dict['nr_of_qubits'] = self.nr_of_qubits
        self.proc_data_dict['qubit_names'] = self.qubit_names

        self.proc_data_dict['ch_names'] = self.raw_data_dict['value_names'][0]

        for ch_name, shots in self.raw_data_dict['measured_values_ord_dict'].items():
            # print(ch_name)
            self.proc_data_dict[ch_name] = shots[0]  # only 1 dataset
            self.proc_data_dict[ch_name +
                                ' all'] = self.proc_data_dict[ch_name]
            min_sh = np.min(self.proc_data_dict[ch_name])
            max_sh = np.max(self.proc_data_dict[ch_name])
            self.proc_data_dict['nr_shots'] = len(self.proc_data_dict[ch_name])

            base = 2
            number_of_experiments = base ** self.nr_of_qubits

            combinations = [int2base(
                i, base=base, fixed_length=self.nr_of_qubits) for i in
                range(number_of_experiments)]
            self.proc_data_dict['combinations'] = combinations

            for i, comb in enumerate(combinations):
                # No post selection implemented yet
                self.proc_data_dict['{} {}'.format(ch_name, comb)] = \
                    self.proc_data_dict[ch_name][i::number_of_experiments]
                ##################################
                #  Binning data into histograms  #
                ##################################
                hist_name = 'hist {} {}'.format(
                    ch_name, comb)
                self.proc_data_dict[hist_name] = np.histogram(
                    self.proc_data_dict['{} {}'.format(
                        ch_name, comb)],
                    bins=nr_bins, range=(min_sh, max_sh))
                #  Cumulative histograms #
                chist_name = 'c'+hist_name
                # the cumulative histograms are normalized to ensure the right
                # fidelities can be calculated
                self.proc_data_dict[chist_name] = np.cumsum(
                    self.proc_data_dict[hist_name][0])/(
                    np.sum(self.proc_data_dict[hist_name][0]))

            self.proc_data_dict['bin_centers {}'.format(ch_name)] = (
                self.proc_data_dict[hist_name][1][:-1] +
                self.proc_data_dict[hist_name][1][1:]) / 2

            self.proc_data_dict['binsize {}'.format(ch_name)] = (
                self.proc_data_dict[hist_name][1][1] -
                self.proc_data_dict[hist_name][1][0])

        #####################################################################
        # Combining histograms of all different combinations and calc Fid.
        ######################################################################
        for ch_idx, ch_name in enumerate(self.proc_data_dict['ch_names']):
            # Create labels for the specific combinations
            comb_str_0, comb_str_1, comb_str_2 = get_arb_comb_xx_label(
                self.proc_data_dict['nr_of_qubits'], qubit_idx=ch_idx)

            # Initialize the arrays
            self.proc_data_dict['hist {} {}'.format(ch_name, comb_str_0)] = \
                [np.zeros(nr_bins), np.zeros(nr_bins+1)]
            self.proc_data_dict['hist {} {}'.format(ch_name, comb_str_1)] = \
                [np.zeros(nr_bins), np.zeros(nr_bins+1)]
            zero_hist = self.proc_data_dict['hist {} {}'.format(
                ch_name, comb_str_0)]
            one_hist = self.proc_data_dict['hist {} {}'.format(
                ch_name, comb_str_1)]

            # Fill them with data from the relevant combinations
            for i, comb in enumerate(self.proc_data_dict['combinations']):
                if comb[-(ch_idx+1)] == '0':
                    zero_hist[0] += self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][0]
                    zero_hist[1] = self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][1]
                elif comb[-(ch_idx+1)] == '1':
                    one_hist[0] += self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][0]
                    one_hist[1] = self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][1]
                elif comb[-(ch_idx+1)] == '2':
                    # Fixme add two state binning
                    raise NotImplementedError()

            chist_0 = np.cumsum(zero_hist[0])/(np.sum(zero_hist[0]))
            chist_1 = np.cumsum(one_hist[0])/(np.sum(one_hist[0]))

            self.proc_data_dict['chist {} {}'.format(ch_name, comb_str_0)] \
                = chist_0
            self.proc_data_dict['chist {} {}'.format(ch_name, comb_str_1)] \
                = chist_1
            ###########################################################
            #  Threshold and fidelity based on cumulative histograms  #

            qubit_name = self.proc_data_dict['qubit_names'][-(ch_idx+1)]
            centers = self.proc_data_dict['bin_centers {}'.format(ch_name)]
            fid, th = get_assignement_fid_from_cumhist(chist_0, chist_1,
                                                       centers)
            self.proc_data_dict['F_ass_raw {}'.format(qubit_name)] = fid
            self.proc_data_dict['threshold_raw {}'.format(qubit_name)] = th

    def prepare_plots(self):
        # N.B. If the log option is used we should manually set the
        # yscale to go from .5 to the current max as otherwise the fits
        # mess up the log plots.
        # log_hist = self.options_dict.get('log_hist', False)

        for ch_idx, ch_name in enumerate(self.proc_data_dict['ch_names']):
            q_name = self.proc_data_dict['qubit_names'][-(ch_idx+1)]
            th_raw = self.proc_data_dict['threshold_raw {}'.format(q_name)]
            F_raw = self.proc_data_dict['F_ass_raw {}'.format(q_name)]

            self.plot_dicts['histogram_{}'.format(ch_name)] = {
                'plotfn': make_mux_ssro_histogram,
                'data_dict': self.proc_data_dict,
                'ch_name': ch_name,
                'title': (self.timestamps[0] + ' \n' +
                          'SSRO histograms {}'.format(ch_name))}

            thresholds = [th_raw]
            threshold_labels = ['thresh. raw']

            self.plot_dicts['comb_histogram_{}'.format(q_name)] = {
                'plotfn': make_mux_ssro_histogram_combined,
                'data_dict': self.proc_data_dict,
                'ch_name': ch_name,
                'thresholds': thresholds,
                'threshold_labels': threshold_labels,
                'qubit_idx': ch_idx,
                'title': (self.timestamps[0] + ' \n' +
                          'Combined SSRO histograms {}'.format(q_name))}

            fid_threshold_msg = 'Summary {}\n'.format(q_name)
            fid_threshold_msg += r'$F_{A}$-raw: ' + '{:.3f} \n'.format(F_raw)
            fid_threshold_msg += r'thresh. raw: ' + '{:.3f} \n'.format(th_raw)

            self.plot_dicts['fid_threshold_msg_{}'.format(q_name)] = {
                'plotfn': self.plot_text,
                'xpos': 1.05,
                'ypos': .9,
                'horizontalalignment': 'left',
                'text_string': fid_threshold_msg,
                'ax_id': 'comb_histogram_{}'.format(q_name)}


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


def get_arb_comb_xx_label(nr_of_qubits, qubit_idx: int):
    """
    Returns labels of the form "xx0xxx", "xx1xxx", "xx2xxx"
    Length of the label is equal to the number of qubits
    """
    comb_str_0 = list('x'*nr_of_qubits)
    comb_str_0[-(qubit_idx+1)] = '0'
    comb_str_0 = "".join(comb_str_0)

    comb_str_1 = list('x'*nr_of_qubits)
    comb_str_1[-(qubit_idx+1)] = '1'
    comb_str_1 = "".join(comb_str_1)

    comb_str_2 = list('x'*nr_of_qubits)
    comb_str_2[-(qubit_idx+1)] = '2'
    comb_str_2 = "".join(comb_str_2)

    return comb_str_0, comb_str_1, comb_str_2


def get_assignement_fid_from_cumhist(chist_0, chist_1, bin_centers=None):
    """
    Returns the average assignment fidelity and threshold
        F_assignment_raw = (P01 - P10 )/2
            where Pxy equals probability to measure x when starting in y
    """
    F_vs_th = (1-(1-abs(chist_1 - chist_0))/2)
    opt_idx = np.argmax(F_vs_th)
    F_assignment_raw = F_vs_th[opt_idx]

    if bin_centers is None:
        bin_centers = np.arange(len(chist_0))
    threshold = bin_centers[opt_idx]

    return F_assignment_raw, threshold


def make_mux_ssro_histogram_combined(data_dict, ch_name, qubit_idx,
                                     thresholds=None, threshold_labels=None,
                                     title=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    markers = itertools.cycle(('v', '^', 'd'))

    comb_str_0, comb_str_1, comb_str_2 = get_arb_comb_xx_label(
        data_dict['nr_of_qubits'], qubit_idx=qubit_idx)

    ax.plot(data_dict['bin_centers {}'.format(ch_name)],
            data_dict['hist {} {}'.format(ch_name, comb_str_0)][0],
            linestyle='',
            marker=next(markers), alpha=.7, label=comb_str_0)
    ax.plot(data_dict['bin_centers {}'.format(ch_name)],
            data_dict['hist {} {}'.format(ch_name, comb_str_1)][0],
            linestyle='',
            marker=next(markers), alpha=.7, label=comb_str_1)

    if thresholds is not None:
        # this is to support multiple threshold types such as raw, fitted etc.
        th_styles = itertools.cycle(('--', '-.', '..'))
        for threshold, label in zip(thresholds, threshold_labels):
            ax.axvline(threshold, linestyle=next(th_styles), color='grey',
                       label=label)

    legend_title = "Prep. state [%s]" % ', '.join(data_dict['qubit_names'])
    ax.legend(title=legend_title, loc=1)  # top right corner
    ax.set_ylabel('Counts')
    # arbitrary units as we use optimal weights
    set_xlabel(ax, ch_name, 'a.u.')

    if title is not None:
        ax.set_title(title)


def make_mux_ssro_histogram(data_dict, ch_name, title=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    nr_of_qubits = data_dict['nr_of_qubits']
    markers = itertools.cycle(('v', '<', '>', '^', 'd', 'o', 's', '*'))
    for i in range(2**nr_of_qubits):
        format_str = '{'+'0:0{}b'.format(nr_of_qubits) + '}'
        binning_string = format_str.format(i)
        ax.plot(data_dict['bin_centers {}'.format(ch_name)],
                data_dict['hist {} {}'.format(ch_name, binning_string)][0],
                linestyle='',
                marker=next(markers), alpha=.7, label=binning_string)

    legend_title = "Prep. state \n[%s]" % ', '.join(data_dict['qubit_names'])
    ax.legend(title=legend_title, loc=1)
    ax.set_ylabel('Counts')
    # arbitrary units as we use optimal weights
    set_xlabel(ax, ch_name, 'a.u.')

    if title is not None:
        ax.set_title(title)


def ro_gauss(x, A_center, B_center, A_sigma, B_sigma, A_amplitude, B_amplitude, A_spurious, B_spurious):
    gauss = lmfit.lineshapes.gaussian
    A_gauss = gauss(x=x[0], center=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = gauss(x=x[1], center=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]


def ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude, B_amplitude, A_spurious, B_spurious):
    cdf = fit_mods.gaussianCDF
    A_gauss = cdf(x=x[0], mu=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = cdf(x=x[1], mu=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]

def ro_CDF_discr(x, A_center, B_center, A_sigma, B_sigma, A_amplitude, B_amplitude, A_spurious, B_spurious):
    #A_amplitude /= 1-A_spurious
    #B_amplitude /= 1-B_spurious
    return ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude, B_amplitude, A_spurious=0, B_spurious=0)

def sum_int(x,y):
    return np.cumsum(y[:-1]*(x[1:]-x[:-1]))


def optimize_IQ_angle(self, shots_I_1, shots_Q_1, shots_I_0,
                      shots_Q_0, min_len, plot_2D_histograms=True,
                      **kw):
    n_bins = 120  # the bins we want to have around our data
    I_min = min(min(shots_I_0), min(shots_I_1))
    I_max = max(max(shots_I_0), max(shots_I_1))
    Q_min = min(min(shots_Q_0), min(shots_Q_1))
    Q_max = max(max(shots_Q_0), max(shots_Q_1))
    edge = max(abs(I_min), abs(I_max), abs(Q_min), abs(Q_max))
    H0, xedges0, yedges0 = np.histogram2d(shots_I_0, shots_Q_0,
                                          bins=n_bins,
                                          range=[[I_min, I_max],
                                                 [Q_min, Q_max]],
                                          normed=True)
    H1, xedges1, yedges1 = np.histogram2d(shots_I_1, shots_Q_1,
                                          bins=n_bins,
                                          range=[[I_min, I_max, ],
                                                 [Q_min, Q_max, ]],
                                          normed=True)

    # this part performs 2D gaussian fits and calculates coordinates of the
    # maxima
    def gaussian(height, center_x, center_y, width_x, width_y):
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + (
                (center_y - y) / width_y) ** 2) / 2)

    def fitgaussian(data):
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(
            data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def moments(data):
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        eps = 1e-8  # To prevent division by zero
        width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / (
                col.sum() + eps))
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / (
                row.sum() + eps))
        height = data.max()
        return height, x, y, width_x, width_y

    data0 = H0
    params0 = fitgaussian(data0)
    fit0 = gaussian(*params0)
    data1 = H1
    params1 = fitgaussian(data1)
    fit1 = gaussian(*params1)
    # interpolating to find the gauss top x and y coordinates
    x_lin = np.linspace(0, n_bins, n_bins + 1)
    y_lin = np.linspace(0, n_bins, n_bins + 1)
    f_x_1 = interp1d(x_lin, xedges1, fill_value='extrapolate')
    x_1_max = f_x_1(params1[1])
    f_y_1 = interp1d(y_lin, yedges1, fill_value='extrapolate')
    y_1_max = f_y_1(params1[2])

    f_x_0 = interp1d(x_lin, xedges0, fill_value='extrapolate')
    x_0_max = f_x_0(params0[1])
    f_y_0 = interp1d(y_lin, yedges0, fill_value='extrapolate')
    y_0_max = f_y_0(params0[2])

    # following part will calculate the angle to rotate the IQ plane
    # All information is to be rotated to the I channel
    y_diff = y_1_max - y_0_max
    x_diff = x_1_max - x_0_max
    theta = -np.arctan2(y_diff, x_diff)

    shots_I_1_rot = np.cos(theta) * shots_I_1 - np.sin(theta) * shots_Q_1
    shots_Q_1_rot = np.sin(theta) * shots_I_1 + np.cos(theta) * shots_Q_1

    shots_I_0_rot = np.cos(theta) * shots_I_0 - np.sin(theta) * shots_Q_0
    shots_Q_0_rot = np.sin(theta) * shots_I_0 + np.cos(theta) * shots_Q_0

    return (theta, shots_I_1_rot, shots_I_0_rot)