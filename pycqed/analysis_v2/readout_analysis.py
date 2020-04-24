"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis (to be updated!)

Originally written by Adriaan, updated/rewritten by Rene May 2018
"""
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import lmfit
from collections import OrderedDict
import numpy as np
import pycqed.analysis.fitting_models as fit_mods
from pycqed.analysis.fitting_models import ro_gauss, ro_CDF, ro_CDF_discr, gaussian_2D, gauss_2D_guess, gaussianCDF, ro_double_gauss_guess
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.simple_analysis as sa
from scipy.optimize import minimize
from pycqed.analysis.tools.plotting import SI_val_to_msg_str, \
    set_xlabel, set_ylabel, set_cbarlabel, flex_colormesh_plot_vs_xy
from pycqed.analysis_v2.tools.plotting import scatter_pnts_overlay
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pycqed.analysis.tools.data_manipulation as dm_tools
from pycqed.utilities.general import int2base
from pycqed.utilities.general import format_value_string


class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 options_dict: dict=None, auto=True,
                 **kw):
        '''
        options dict options:
            'fixed_p10'   fixes p(e|g)  res_exc (do not vary in fit)
            'fixed_p01' : fixes p(g|pi) mmt_rel (do not vary in fit)
            'auto_rotation_angle' : (bool) automatically find the I/Q mixing angle
            'rotation_angle' : manually define the I/Q mixing angle (ignored if auto_rotation_angle is set to True)
            'nr_bins' : number of bins to use for the histograms
            'post_select' : (bool) sets on or off the post_selection based on an initialization measurement (needs to be in agreement with nr_samples)
            'post_select_threshold' : (float) threshold used for post-selection (only activated by above parameter)
            'nr_samples' : amount of different samples (e.g. ground and excited = 2 and with post-selection = 4)
            'sample_0' : index of first sample (ground-state)
            'sample_1' : index of second sample (first excited-state)
            'max_datapoints' : maximum amount of datapoints for culumative fit
            'log_hist' : use log scale for the y-axis of the 1D histograms
            'verbose' : see BaseDataAnalysis
            'presentation_mode' : see BaseDataAnalysis
            see BaseDataAnalysis for more.
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)
        self.single_timestamp = True
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        self.numeric_params = []

        # Determine the default for auto_rotation_angle
        man_angle = self.options_dict.get('rotation_angle', False) is False
        self.options_dict['auto_rotation_angle'] = self.options_dict.get(
            'auto_rotation_angle', man_angle)

        self.predict_qubit_temp = 'predict_qubit_temp' in self.options_dict
        if self.predict_qubit_temp:
            self.qubit_freq = self.options_dict['qubit_freq']

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
        nr_bins = int(self.options_dict.get('nr_bins', 100))

        ######################################################
        #  Separating data into shots for 0 and shots for 1  #
        ######################################################
        meas_val = self.raw_data_dict['measured_values']
        unit = self.raw_data_dict['value_units'][0]
        # loop through channels
        shots = np.zeros((2, len(meas_val),), dtype=np.ndarray)
        for j, dat in enumerate(meas_val):
            assert unit == self.raw_data_dict['value_units'][
                j], 'The channels have been measured using different units. This is not supported yet.'
            sh_0, sh_1 = get_shots_zero_one(
                dat, post_select=post_select, nr_samples=nr_samples,
                post_select_threshold=post_select_threshold,
                sample_0=sample_0, sample_1=sample_1)
            shots[0, j] = sh_0
            shots[1, j] = sh_1
        #shots = np.array(shots, dtype=float)

        # Do we have two quadratures?
        if len(meas_val) == 2:
            ########################################################
            # Bin the data in 2D, to calculate the opt. angle
            ########################################################
            data_range_x = (np.min([np.min(b) for b in shots[:, 0]]),
                            np.max([np.max(b) for b in shots[:, 0]]))
            data_range_y = (np.min([np.min(b) for b in shots[:, 1]]),
                            np.max([np.max(b) for b in shots[:, 1]]))
            data_range_xy = (data_range_x, data_range_y)
            nr_bins_2D = int(self.options_dict.get(
                'nr_bins_2D', 6*np.sqrt(nr_bins)))
            H0, xedges, yedges = np.histogram2d(x=shots[0, 0],
                                                y=shots[0, 1],
                                                bins=nr_bins_2D,
                                                range=data_range_xy)
            H1, xedges, yedges = np.histogram2d(x=shots[1, 0],
                                                y=shots[1, 1],
                                                bins=nr_bins_2D,
                                                range=data_range_xy)
            binsize_x = xedges[1] - xedges[0]
            binsize_y = yedges[1] - yedges[0]
            bin_centers_x = xedges[:-1] + binsize_x
            bin_centers_y = yedges[:-1] + binsize_y
            self.proc_data_dict['2D_histogram_x'] = bin_centers_x
            self.proc_data_dict['2D_histogram_y'] = bin_centers_y
            self.proc_data_dict['2D_histogram_z'] = [H0, H1]

            # Find and apply the effective/rotated integrated voltage
            angle = self.options_dict.get('rotation_angle', 0)
            auto_angle = self.options_dict.get('auto_rotation_angle', True)
            if auto_angle:
                ##########################################
                #  Determining the rotation of the data  #
                ##########################################
                gauss2D_model_0 = lmfit.Model(gaussian_2D,
                                              independent_vars=['x', 'y'])
                gauss2D_model_1 = lmfit.Model(gaussian_2D,
                                              independent_vars=['x', 'y'])
                guess0 = gauss_2D_guess(model=gauss2D_model_0, data=H0.transpose(),
                                        x=bin_centers_x, y=bin_centers_y)
                guess1 = gauss_2D_guess(model=gauss2D_model_1, data=H1.transpose(),
                                        x=bin_centers_x, y=bin_centers_y)

                x2d = np.array([bin_centers_x]*len(bin_centers_y))
                y2d = np.array([bin_centers_y]*len(bin_centers_x)).transpose()
                fitres0 = gauss2D_model_0.fit(data=H0.transpose(), x=x2d, y=y2d,
                                              **guess0)
                fitres1 = gauss2D_model_1.fit(data=H1.transpose(),  x=x2d, y=y2d,
                                              **guess1)

                fr0 = fitres0.best_values
                fr1 = fitres1.best_values
                x0 = fr0['center_x']
                x1 = fr1['center_x']
                y0 = fr0['center_y']
                y1 = fr1['center_y']

                self.proc_data_dict['IQ_pos'] = [[x0, x1], [y0, y1]]
                dx = x1 - x0
                dy = y1 - y0
                mid = [x0 + dx/2, y0 + dy/2]
                angle = np.arctan2(dy, dx)
            else:
                mid = [0, 0]

            if self.verbose:
                ang_deg = (angle*180/np.pi)
                print('Mixing I/Q channels with %.3f degrees ' % ang_deg +
                      #'around point (%.2f, %.2f)%s'%(mid[0], mid[1], unit) +
                      ' to obtain effective voltage.')

            self.proc_data_dict['raw_offset'] = [*mid, angle]
            # create matrix
            rot_mat = [[+np.cos(-angle), -np.sin(-angle)],
                       [+np.sin(-angle), +np.cos(-angle)]]
            # rotate data accordingly
            eff_sh = np.zeros(len(shots[0]), dtype=np.ndarray)
            eff_sh[0] = np.dot(rot_mat[0], shots[0])  # - mid
            eff_sh[1] = np.dot(rot_mat[0], shots[1])  # - mid
        else:
            # If we have only one quadrature, use that (doh!)
            eff_sh = shots[:, 0]

        self.proc_data_dict['all_channel_int_voltages'] = shots
        # self.raw_data_dict['value_names'][0]
        self.proc_data_dict['shots_xlabel'] = 'Effective integrated Voltage'
        self.proc_data_dict['shots_xunit'] = unit
        self.proc_data_dict['eff_int_voltages'] = eff_sh
        self.proc_data_dict['nr_shots'] = [len(eff_sh[0]), len(eff_sh[1])]
        sh_min = min(np.min(eff_sh[0]), np.min(eff_sh[1]))
        sh_max = max(np.max(eff_sh[0]), np.max(eff_sh[1]))
        data_range = (sh_min, sh_max)
        eff_sh_sort = [np.sort(eff_sh[0]), np.sort(eff_sh[1])]
        x0, n0 = np.unique(eff_sh_sort[0], return_counts=True)
        cumsum0 = np.cumsum(n0)
        x1, n1 = np.unique(eff_sh_sort[1], return_counts=True)
        cumsum1 = np.cumsum(n1)

        self.proc_data_dict['cumsum_x'] = [x0, x1]
        self.proc_data_dict['cumsum_y'] = [cumsum0, cumsum1]

        all_x = np.unique(np.sort(np.concatenate((x0, x1))))
        md = self.options_dict.get('max_datapoints', 1000)
        if len(all_x) > md:
            all_x = np.linspace(*data_range, md)
        ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
        necumsum0 = ecumsum0/np.max(ecumsum0)
        ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
        necumsum1 = ecumsum1/np.max(ecumsum1)

        self.proc_data_dict['cumsum_x_ds'] = all_x
        self.proc_data_dict['cumsum_y_ds'] = [ecumsum0, ecumsum1]
        self.proc_data_dict['cumsum_y_ds_n'] = [necumsum0, necumsum1]

        ##################################
        #  Binning data into histograms  #
        ##################################
        h0, bin_edges = np.histogram(eff_sh[0], bins=nr_bins,
                                     range=data_range)
        h1, bin_edges = np.histogram(eff_sh[1], bins=nr_bins,
                                     range=data_range)
        self.proc_data_dict['hist'] = [h0, h1]
        binsize = (bin_edges[1] - bin_edges[0])
        self.proc_data_dict['bin_edges'] = bin_edges
        self.proc_data_dict['bin_centers'] = bin_edges[:-1]+binsize
        self.proc_data_dict['binsize'] = binsize

        #######################################################
        #  Threshold and fidelity based on culmulative counts #
        #######################################################
        # Average assignment fidelity: F_ass = (P01 - P10 )/2
        # where Pxy equals probability to measure x when starting in y
        F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
        opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
        opt_idx = int(round(np.average(opt_idxs)))
        self.proc_data_dict['F_assignment_raw'] = F_vs_th[opt_idx]
        self.proc_data_dict['threshold_raw'] = all_x[opt_idx]

    def prepare_fitting(self):
        ###################################
        #  First fit the histograms (PDF) #
        ###################################
        self.fit_dicts = OrderedDict()

        bin_x = self.proc_data_dict['bin_centers']
        bin_xs = [bin_x, bin_x]
        bin_ys = self.proc_data_dict['hist']
        m = lmfit.model.Model(ro_gauss)
        m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
        params = m.guess(x=bin_xs, data=bin_ys,
                         fixed_p01=self.options_dict.get('fixed_p01', False),
                         fixed_p10=self.options_dict.get('fixed_p10', False))
        res = m.fit(x=bin_xs, data=bin_ys, params=params)

        self.fit_dicts['shots_all_hist'] = {
            'model': m,
            'fit_xvals': {'x': bin_xs},
            'fit_yvals': {'data': bin_ys},
            'guessfn_pars': {'fixed_p01': self.options_dict.get('fixed_p01', False),
                             'fixed_p10': self.options_dict.get('fixed_p10', False)},
        }

        ###################################
        #  Fit the CDF                    #
        ###################################
        m_cul = lmfit.model.Model(ro_CDF)
        cdf_xs = self.proc_data_dict['cumsum_x_ds']
        cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
        cdf_ys = self.proc_data_dict['cumsum_y_ds']
        cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]
        #cul_res = m_cul.fit(x=cdf_xs, data=cdf_ys, params=res.params)
        cum_params = res.params
        cum_params['A_amplitude'].value = np.max(cdf_ys[0])
        cum_params['A_amplitude'].vary = False
        cum_params['B_amplitude'].value = np.max(cdf_ys[1])
        cum_params['A_amplitude'].vary = False # FIXME: check if correct
        self.fit_dicts['shots_all'] = {
            'model': m_cul,
            'fit_xvals': {'x': cdf_xs},
            'fit_yvals': {'data': cdf_ys},
            'guess_pars': cum_params,
        }

    def analyze_fit_results(self):
        # Create a CDF based on the fit functions of both fits.
        fr = self.fit_res['shots_all']
        bv = fr.best_values

        # best values new
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

        thr_guess = (3*bv['B_center'] - bv['A_center'])/2
        opt_fid = minimize(infid_vs_th, thr_guess)

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

        def CDF_0_discr(x):
            return gaussianCDF(x, amplitude=1,
                               mu=bv['A_center'], sigma=bv['A_sigma'])

        def CDF_1_discr(x):
            return gaussianCDF(x, amplitude=1,
                               mu=bv['B_center'], sigma=bv['B_sigma'])

        def disc_infid_vs_th(x):
            cdf0 = gaussianCDF(x, amplitude=1, mu=bv['A_center'],
                               sigma=bv['A_sigma'])
            cdf1 = gaussianCDF(x, amplitude=1, mu=bv['B_center'],
                               sigma=bv['B_sigma'])
            return (1-np.abs(cdf0 - cdf1))/2

        self._CDF_0_discr = CDF_0_discr
        self._CDF_1_discr = CDF_1_discr
        self._disc_infid_vs_th = disc_infid_vs_th

        opt_fid_discr = minimize(disc_infid_vs_th, thr_guess)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid_discr['fun'], float):
            self.proc_data_dict['F_discr'] = (1-opt_fid_discr['fun'])
        else:
            self.proc_data_dict['F_discr'] = (1-opt_fid_discr['fun'])[0]

        self.proc_data_dict['threshold_discr'] = opt_fid_discr['x'][0]

        fr = self.fit_res['shots_all']
        bv = fr.params
        self.proc_data_dict['residual_excitation'] = bv['A_spurious'].value
        self.proc_data_dict['relaxation_events'] = bv['B_spurious'].value

        ###################################
        #  Save quantities of interest.   #
        ###################################
        self.proc_data_dict['quantities_of_interest'] = {
            'SNR': self.fit_res['shots_all'].params['SNR'].value,
            'F_d': self.proc_data_dict['F_discr'],
            'F_a': self.proc_data_dict['F_assignment_raw'],
            'residual_excitation': self.proc_data_dict['residual_excitation'],
            'relaxation_events':
                self.proc_data_dict['relaxation_events']
        }
        self.qoi = self.proc_data_dict['quantities_of_interest']

    def prepare_plots(self):
        # Did we load two voltage components (shall we do 2D plots?)
        two_dim_data = len(
            self.proc_data_dict['all_channel_int_voltages'][0]) == 2

        eff_voltage_label = self.proc_data_dict['shots_xlabel']
        eff_voltage_unit = self.proc_data_dict['shots_xunit']
        x_volt_label = self.raw_data_dict['value_names'][0]
        x_volt_unit = self.raw_data_dict['value_units'][0]
        if two_dim_data:
            y_volt_label = self.raw_data_dict['value_names'][1]
            y_volt_unit = self.raw_data_dict['value_units'][1]
        z_hist_label = 'Counts'
        labels = self.options_dict.get(
            'preparation_labels', ['|g> prep.', '|e> prep.'])
        label_0 = labels[0]
        label_1 = labels[1]
        title = ('\n' + self.timestamps[0] + ' - "' +
                 self.raw_data_dict['measurementstring'] + '"')

        # 1D histograms (PDF)
        log_hist = self.options_dict.get('log_hist', False)
        bin_x = self.proc_data_dict['bin_edges']
        bin_y = self.proc_data_dict['hist']
        self.plot_dicts['hist_0'] = {
            'title': 'Binned Shot Counts' + title,
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[0],
            'xwidth': self.proc_data_dict['binsize'],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C0',
                        'edgecolor': 'C0'},
            'setlabel': label_0,
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': z_hist_label,
        }

        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[1],
            'xwidth': self.proc_data_dict['binsize'],
            'bar_kws': {'log': log_hist, 'alpha': .3, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': label_1,
            'do_legend': True,
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': z_hist_label,
        }
        if log_hist:
            self.plot_dicts['hist_0']['yrange'] = (0.5, 1.5*np.max(bin_y[0]))
            self.plot_dicts['hist_1']['yrange'] = (0.5, 1.5*np.max(bin_y[1]))

        # CDF
        cdf_xs = self.proc_data_dict['cumsum_x']
        cdf_ys = self.proc_data_dict['cumsum_y']
        cdf_ys[0] = cdf_ys[0]/np.max(cdf_ys[0])
        cdf_ys[1] = cdf_ys[1]/np.max(cdf_ys[1])
        xra = (bin_x[0], bin_x[-1])

        self.plot_dicts['cdf_shots_0'] = {
            'title': 'Culmulative Shot Counts (no binning)' + title,
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[0],
            'yvals': cdf_ys[0],
            'setlabel': label_0,
            'xrange': xra,
            'line_kws': {'color': 'C0', 'alpha': 0.3},
            'marker': '',
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': 'Culmulative Counts',
            'yunit': 'norm.',
            'do_legend': True,
        }
        self.plot_dicts['cdf_shots_1'] = {
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[1],
            'yvals': cdf_ys[1],
            'setlabel': label_1,
            'line_kws': {'color': 'C3', 'alpha': 0.3},
            'marker': '',
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': 'Culmulative Counts',
            'yunit': 'norm.',
            'do_legend': True,
        }

        # Vlines for thresholds
        th_raw = self.proc_data_dict['threshold_raw']
        threshs = [th_raw, ]
        if self.do_fitting:
            threshs.append(self.proc_data_dict['threshold_fit'])
            threshs.append(self.proc_data_dict['threshold_discr'])

        for ax in ['1D_histogram', 'cdf']:
            self.plot_dicts[ax+'_vlines_thresh'] = {
                'ax_id': ax,
                'plotfn': self.plot_vlines_auto,
                'xdata': threshs,
                'linestyles': ['--', '-.', ':'],
                'labels': ['$th_{raw}$', '$th_{fit}$', '$th_{d}$'],
                'colors': ['0.3', '0.5', '0.2'],
                'do_legend': True,
            }

        # 2D Histograms
        if two_dim_data:
            iq_centers = None
            if 'IQ_pos' in self.proc_data_dict and self.proc_data_dict['IQ_pos'] is not None:
                iq_centers = self.proc_data_dict['IQ_pos']
                peak_marker_2D = {
                    'plotfn': self.plot_line,
                    'xvals': iq_centers[0],
                    'yvals': iq_centers[1],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': 'x',
                    'aspect': 'equal',
                    'linestyle': '',
                    'color': 'black',
                    #'line_kws': {'markersize': 1, 'color': 'black', 'alpha': 1},
                    'setlabel': 'Peaks',
                    'do_legend': True,
                }
                peak_marker_2D_rot = deepcopy(peak_marker_2D)
                peak_marker_2D_rot['xvals'] = iq_centers[0]
                peak_marker_2D_rot['yvals'] = iq_centers[1]

            self.plot_dicts['2D_histogram_0'] = {
                'title': 'Raw '+label_0+' Binned Shot Counts' + title,
                'ax_id': '2D_histogram_0',
                # 'plotfn': self.plot_colorxy,
                'plotfn': plot_2D_ssro_histogram,
                'xvals': self.proc_data_dict['2D_histogram_x'],
                'yvals': self.proc_data_dict['2D_histogram_y'],
                'zvals': self.proc_data_dict['2D_histogram_z'][0].T,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'zunit': '-',
                'cmap': 'Blues',
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_histogram_0'
                self.plot_dicts['2D_histogram_0_marker'] = dp

            self.plot_dicts['2D_histogram_1'] = {
                'title': 'Raw '+label_1+' Binned Shot Counts' + title,
                'ax_id': '2D_histogram_1',
                # 'plotfn': self.plot_colorxy,
                'plotfn': plot_2D_ssro_histogram,
                'xvals': self.proc_data_dict['2D_histogram_x'],
                'yvals': self.proc_data_dict['2D_histogram_y'],
                'zvals': self.proc_data_dict['2D_histogram_z'][1].T,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'zunit': '-',
                'cmap': 'Reds',
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_histogram_1'
                self.plot_dicts['2D_histogram_1_marker'] = dp

            # Scatter Shots
            volts = self.proc_data_dict['all_channel_int_voltages']

            v_flat = np.concatenate(np.concatenate(volts))
            plot_range = (np.min(v_flat), np.max(v_flat))

            vxr = plot_range
            vyr = plot_range
            self.plot_dicts['2D_shots_0'] = {
                'title': 'Raw Shots' + title,
                'ax_id': '2D_shots',
                'aspect': 'equal',
                'plotfn': self.plot_line,
                'xvals': volts[0][0],
                'yvals': volts[0][1],
                'range': [vxr, vyr],
                'xrange': vxr,
                'yrange': vyr,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'marker': 'o',
                'linestyle': '',
                'color': 'C0',
                'line_kws': {'markersize': 0.25, 'color': 'C0', 'alpha': 0.5},
                'setlabel': label_0,
                'do_legend': True,
            }
            self.plot_dicts['2D_shots_1'] = {
                'ax_id': '2D_shots',
                'plotfn': self.plot_line,
                'xvals': volts[1][0],
                'yvals': volts[1][1],
                'aspect': 'equal',
                'range': [vxr, vyr],
                'xrange': vxr,
                'yrange': vyr,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'marker': 'o',
                'linestyle': '',
                'color': 'C3',
                'line_kws': {'markersize': 0.25, 'color': 'C3', 'alpha': 0.5},
                'setlabel': label_1,
                'do_legend': True,
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_shots'
                self.plot_dicts['2D_shots_marker'] = dp

        # The cumulative histograms
        #####################################
        # Adding the fits to the figures    #
        #####################################
        if self.do_fitting:
            # todo: add seperate fits for residual and main gaussians
            x = np.linspace(bin_x[0], bin_x[-1], 150)
            para_hist_tmp = self.fit_res['shots_all_hist'].best_values
            para_cdf = self.fit_res['shots_all'].best_values
            para_hist = para_cdf
            para_hist['A_amplitude'] = para_hist_tmp['A_amplitude']
            para_hist['B_amplitude'] = para_hist_tmp['B_amplitude']

            ro_g = ro_gauss(x=[x, x], **para_hist)
            self.plot_dicts['new_fit_shots_0'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[0],
                'setlabel': 'Fit '+label_0,
                'line_kws': {'color': 'C0'},
                'marker': '',
                'do_legend': True,
            }
            self.plot_dicts['new_fit_shots_1'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[1],
                'marker': '',
                'setlabel': 'Fit '+label_1,
                'line_kws': {'color': 'C3'},
                'do_legend': True,
            }

            self.plot_dicts['cdf_fit_shots_0'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_0(x),
                'setlabel': 'Fit '+label_0,
                'line_kws': {'color': 'C0', 'alpha': 0.8},
                'linestyle': ':',
                'marker': '',
                'do_legend': True,
            }
            self.plot_dicts['cdf_fit_shots_1'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_1(x),
                'marker': '',
                'linestyle': ':',
                'setlabel': 'Fit '+label_1,
                'line_kws': {'color': 'C3', 'alpha': 0.8},
                'do_legend': True,
            }

        ##########################################
        # Add textbox (eg.g Thresholds, fidelity #
        # information, number of shots etc)      #
        ##########################################
        if not self.presentation_mode:
            fit_text = 'Thresholds:'
            fit_text += '\nName | Level | Fidelity'
            thr, th_unit = SI_val_to_msg_str(
                self.proc_data_dict['threshold_raw'],
                eff_voltage_unit, return_type=float)
            raw_th_msg = (
                '\n>raw   | ' +
                '{:.2f} {} | '.format(thr, th_unit) +
                '{:.1f}%'.format(
                    self.proc_data_dict['F_assignment_raw']*100))

            fit_text += raw_th_msg

            if self.do_fitting:
                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_fit'],
                    eff_voltage_unit, return_type=float)
                fit_th_msg = (
                    '\n>fit     | ' +
                    '{:.2f} {} | '.format(thr, th_unit) +
                    '{:.1f}%'.format(
                        self.proc_data_dict['F_assignment_fit']*100))
                fit_text += fit_th_msg

                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_discr'],
                    eff_voltage_unit, return_type=float)
                fit_th_msg = (
                    '\n>dis    | ' +
                    '{:.2f} {} | '.format(thr, th_unit) +
                    '{:.1f}%'.format(
                        self.proc_data_dict['F_discr']*100))
                fit_text += fit_th_msg
                snr = self.fit_res['shots_all'].params['SNR']

                fit_text += format_value_string('\nSNR (fit)', lmfit_par=snr)

                fr = self.fit_res['shots_all']
                bv = fr.params
                a_sp = bv['A_spurious']
                fit_text += '\n\nSpurious Excitations:'

                fit_text += format_value_string('\n$p(e|0)$', lmfit_par=a_sp)

                b_sp = bv['B_spurious']
                fit_text += format_value_string('\n$p(g|\\pi)$',
                                                lmfit_par=b_sp)

            if two_dim_data:
                offs = self.proc_data_dict['raw_offset']

                fit_text += '\n\nRotated by ${:.1f}^\\circ$'.format(
                    (offs[2]*180/np.pi) % 180)
                auto_rot = self.options_dict.get('auto_rotation_angle', True)
                fit_text += '(auto)' if auto_rot else '(man.)'
            else:
                fit_text += '\n\n(Single quadrature data)'

            fit_text += '\n\nTotal shots: %d+%d' % (*self.proc_data_dict['nr_shots'],)
            if self.predict_qubit_temp:
                h = 6.62607004e-34
                kb = 1.38064852e-23
                res_exc = a_sp.value
                effective_temp = h*6.42e9/(kb*np.log((1-res_exc)/res_exc))
                fit_text += '\n\nQubit '+'$T_{eff}$'+\
                    ' = {:.2f} mK\n@{:.0f}'.format(effective_temp*1e3,
                                                  self.qubit_freq)

            for ax in ['cdf', '1D_histogram']:
                self.plot_dicts['text_msg_' + ax] = {
                    'ax_id': ax,
                    'xpos': 1.05,
                    'horizontalalignment': 'left',
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': fit_text,
                }


class Dispersive_shift_Analysis(ba.BaseDataAnalysis):
    '''
    Analisys for dispersive shift.
    Designed to be used with <CCL_Transmon>.measure-dispersive_shift_pulsed
    '''
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 options_dict: dict=None, auto=True,
                 **kw):
        '''
        Extract ground and excited state timestamps
        '''
        if (t_start is None) and (t_stop is None):
            ground_ts = a_tools.return_last_n_timestamps(1, contains='Resonator_scan_off')
            excited_ts= a_tools.return_last_n_timestamps(1, contains='Resonator_scan_on')
        elif (t_start is None) ^ (t_stop is None):
            raise ValueError('Must provide either none or both timestamps.')
        else:
            ground_ts = t_start # t_start is assigned to ground state
            excited_ts= t_stop  # t_stop is assigned to excited state

        super().__init__(t_start=ground_ts, t_stop=excited_ts,
                         label='Resonator_scan', do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'
                            }
        self.numeric_params = []
        #self.proc_data_dict = OrderedDict()
        if auto:
            self.run_analysis()

    def process_data(self):
        '''
        Processing data
        '''
        # Frequencu sweep range in the ground/excited state
        self.proc_data_dict['data_freqs_ground'] = \
            self.raw_data_dict['sweep_points'][0]
        self.proc_data_dict['data_freqs_excited'] = \
            self.raw_data_dict['sweep_points'][1]

        # S21 mag (transmission) in the ground/excited state
        self.proc_data_dict['data_S21_ground'] = \
            self.raw_data_dict['measured_values'][0][0]
        self.proc_data_dict['data_S21_excited'] = \
            self.raw_data_dict['measured_values'][1][0]

        #self.proc_data_dict['f0_ground'] = self.raw_data_dict['f0'][0]

        #############################
        # Find resonator dips
        #############################
        pk_rep_ground = a_tools.peak_finder( \
                            self.proc_data_dict['data_freqs_ground'],
                            self.proc_data_dict['data_S21_ground'],
                            window_len=5)
        pk_rep_excited= a_tools.peak_finder( \
                            self.proc_data_dict['data_freqs_excited'],
                            self.proc_data_dict['data_S21_excited'],
                            window_len=5)

        min_idx_ground = np.argmin(pk_rep_ground['dip_values'])
        min_idx_excited= np.argmin(pk_rep_excited['dip_values'])

        min_freq_ground = pk_rep_ground['dips'][min_idx_ground]
        min_freq_excited= pk_rep_excited['dips'][min_idx_excited]

        min_S21_ground = pk_rep_ground['dip_values'][min_idx_ground]
        min_S21_excited= pk_rep_excited['dip_values'][min_idx_excited]

        dispersive_shift = min_freq_excited-min_freq_ground

        self.proc_data_dict['Res_freq_ground'] = min_freq_ground
        self.proc_data_dict['Res_freq_excited']= min_freq_excited
        self.proc_data_dict['Res_S21_ground'] = min_S21_ground
        self.proc_data_dict['Res_S21_excited']= min_S21_excited
        self.proc_data_dict['quantities_of_interest'] = \
            {'dispersive_shift': dispersive_shift}

        self.qoi = self.proc_data_dict['quantities_of_interest']

    def prepare_plots(self):


        x_range = [min(self.proc_data_dict['data_freqs_ground'][0],
                       self.proc_data_dict['data_freqs_excited'][0]) ,
                   max(self.proc_data_dict['data_freqs_ground'][-1],
                       self.proc_data_dict['data_freqs_excited'][-1])]

        y_range = [0, max(max(self.proc_data_dict['data_S21_ground']),
                          max(self.proc_data_dict['data_S21_excited']))]

        x_label = self.raw_data_dict['xlabel'][0]
        y_label = self.raw_data_dict['value_names'][0][0]

        x_unit = self.raw_data_dict['xunit'][0][0]
        y_unit = self.raw_data_dict['value_units'][0][0]

        title = 'Transmission in the ground and excited state'

        self.plot_dicts['S21_ground'] = {
            'title': title,
            'ax_id': 'Transmission_axis',
            'xvals': self.proc_data_dict['data_freqs_ground'],
            'yvals': self.proc_data_dict['data_S21_ground'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': y_label,
            'yunit': y_unit,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['S21_excited'] = {
            'title': title,
            'ax_id': 'Transmission_axis',
            'xvals': self.proc_data_dict['data_freqs_excited'],
            'yvals': self.proc_data_dict['data_S21_excited'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': y_label,
            'yunit': y_unit,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C1', 'alpha': 1},
            'marker': ''
            }

        ####################################
        # Plot arrow
        ####################################
        min_freq_ground = self.proc_data_dict['Res_freq_ground']
        min_freq_excited= self.proc_data_dict['Res_freq_excited']
        yval = y_range[1]/2
        dispersive_shift = int((min_freq_excited-min_freq_ground)*1e-4)*1e-2
        txt_str = r'$2_\chi/2\pi=$' + str(dispersive_shift) + ' MHz'

        self.plot_dicts['Dispersive_shift_line'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_ground , min_freq_excited] ,
            'yvals': [yval, yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Dispersive_shift_vline'] = {
            'ax_id': 'Transmission_axis',
            'ymin': y_range[0],
            'ymax': y_range[1],
            'x': [min_freq_ground, min_freq_excited],
            'xrange': x_range,
            'yrange': y_range,
            'plotfn': self.plot_vlines,
            'line_kws': {'color': 'black', 'alpha': 0.5}
            }

        self.plot_dicts['Dispersive_shift_rmarker'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_ground] ,
            'yvals': [yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': 5
            }
        self.plot_dicts['Dispersive_shift_lmarker'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_excited] ,
            'yvals': [yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': 4
            }

        self.plot_dicts['Dispersive_shift_text'] = {
            'ax_id': 'Transmission_axis',
            'plotfn': self.plot_text,
            'xpos': .5,
            'ypos': .5,
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'text_string': txt_str,
            'box_props': dict(boxstyle='round', pad=.4,
                              facecolor='white', alpha=0.)
            }


class RO_acquisition_delayAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 qubit_name = '',
                 options_dict: dict=None, auto=True,
                 **kw):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)

        self.single_timestamp = True
        self.qubit_name = qubit_name
        self.params_dict = {'ro_pulse_length': '{}.ro_pulse_length'.format(self.qubit_name),
                            'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'
                            }
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Processing data
        """
        self.Times = self.raw_data_dict['sweep_points']
        self.I_data_UHF = self.raw_data_dict['measured_values'][0]
        self.Q_data_UHF = self.raw_data_dict['measured_values'][1]
        self.pulse_length = float(self.raw_data_dict['ro_pulse_length'.format(self.qubit_name)])

        #######################################
        # Determine the start of the pusle
        #######################################
        def get_pulse_start(x, y, tolerance=2):
            '''
            The start of the pulse is estimated in three steps:
                1. Evaluate signal standard deviation in a certain interval as
                   function of time: f(t).
                2. Calculate the derivative of the aforementioned data: f'(t).
                3. Evaluate when the derivative exceeds a threshold. This
                   threshold is defined as max(f'(t))/5.
            This approach is more tolerant to noisy signals.
            '''
            pulse_baseline = np.mean(y) # get pulse baseline
            pulse_std      = np.std(y)  # get pulse standard deviation

            nr_points_interval = 200        # number of points in the interval
            aux = int(nr_points_interval/2)

            iteration_idx = np.arange(-aux, len(y)+aux)     # mask for circular array
            aux_list = [ y[i%len(y)] for i in iteration_idx] # circular array

            # Calculate standard deviation for each interval
            y_std = []
            for i in range(len(y)):
                interval = aux_list[i : i+nr_points_interval]
                y_std.append( np.std(interval) )

            y_std_derivative = np.gradient(y_std[:-aux])# calculate derivative
            threshold = max(y_std_derivative)/5        # define threshold
            start_index = np.where( y_std_derivative > threshold )[0][0] + aux

            return start_index-tolerance

        #######################################
        # Determine the end of depletion
        #######################################
        def get_pulse_length(x, y):
            '''
            Similarly to get_pulse_start, the end of depletion is
            set when the signal goes below 5% of its standard dev.
            '''
            pulse_baseline = np.mean(y)
            threshold      = 0.05*np.std(y)
            pulse_std = threshold+1
            i = 0
            while pulse_std > threshold:
                pulse_std = np.std(y[i:]-pulse_baseline)
                i += 1
            end_index = i-1
            return end_index

        Amplitude_I = max(abs(self.I_data_UHF))
        baseline_I = np.mean(self.I_data_UHF)
        start_index_I = get_pulse_start(self.Times, self.I_data_UHF)
        end_index_I = get_pulse_length(self.Times, self.I_data_UHF)

        Amplitude_Q = max(abs(self.Q_data_UHF))
        baseline_Q = np.mean(self.Q_data_UHF)
        start_index_Q = get_pulse_start(self.Times, self.Q_data_UHF)
        end_index_Q = get_pulse_length(self.Times, self.Q_data_UHF)

        self.proc_data_dict['I_Amplitude'] = Amplitude_I
        self.proc_data_dict['I_baseline'] = baseline_I
        self.proc_data_dict['I_pulse_start_index'] = start_index_I
        self.proc_data_dict['I_pulse_end_index'] = end_index_I
        self.proc_data_dict['I_pulse_start'] = self.Times[start_index_I]
        self.proc_data_dict['I_pulse_end'] = self.Times[end_index_I]

        self.proc_data_dict['Q_Amplitude'] = Amplitude_Q
        self.proc_data_dict['Q_baseline'] = baseline_Q
        self.proc_data_dict['Q_pulse_start_index'] = start_index_Q
        self.proc_data_dict['Q_pulse_end_index'] = end_index_Q
        self.proc_data_dict['Q_pulse_start'] = self.Times[start_index_Q]
        self.proc_data_dict['Q_pulse_end'] = self.Times[end_index_Q]

    def prepare_plots(self):

        I_start_line_x = [self.proc_data_dict['I_pulse_start'],
                          self.proc_data_dict['I_pulse_start']]
        I_pulse_line_x = [self.proc_data_dict['I_pulse_start']+self.pulse_length,
                          self.proc_data_dict['I_pulse_start']+self.pulse_length]
        I_end_line_x = [self.proc_data_dict['I_pulse_end'],
                        self.proc_data_dict['I_pulse_end']]

        Q_start_line_x = [self.proc_data_dict['Q_pulse_start'],
                          self.proc_data_dict['Q_pulse_start']]
        Q_pulse_line_x = [self.proc_data_dict['Q_pulse_start']+self.pulse_length,
                           self.proc_data_dict['Q_pulse_start']+self.pulse_length]
        Q_end_line_x = [self.proc_data_dict['Q_pulse_end'],
                        self.proc_data_dict['Q_pulse_end']]

        Amplitude = max(self.proc_data_dict['I_Amplitude'],
                        self.proc_data_dict['Q_Amplitude'])
        vline_y = np.array([1.1*Amplitude, -1.1*Amplitude])

        x_range= [self.Times[0], self.Times[-1]]
        y_range= [vline_y[1], vline_y[0]]

        I_title = str(self.qubit_name)+' Measured transients $I_{quadrature}$'
        Q_title = str(self.qubit_name)+' Measured transients $Q_{quadrature}$'

        ##########################
        # Transients
        ##########################
        self.plot_dicts['I_transients'] = {
            'title': I_title,
            'ax_id': 'I_axis',
            'xvals': self.Times,
            'yvals': self.I_data_UHF,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_transients'] = {
            'title': Q_title,
            'ax_id': 'Q_axis',
            'xvals': self.Times,
            'yvals': self.Q_data_UHF,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        ##########################
        # Vertical lines
        ##########################
        # I quadrature
        self.plot_dicts['I_pulse_start'] = {
            'ax_id': 'I_axis',
            'xvals': I_start_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['I_pulse_end'] = {
            'ax_id': 'I_axis',
            'xvals': I_pulse_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['I_depletion_end'] = {
            'ax_id': 'I_axis',
            'xvals': I_end_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        # Q quadrature
        self.plot_dicts['Q_pulse_start'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_start_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_pulse_end'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_pulse_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_depletion_end'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_end_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        ########################
        # Plot pulse windows
        ########################

        I_pulse_bin = np.array([self.proc_data_dict['I_pulse_start'],
                    self.proc_data_dict['I_pulse_start']+self.pulse_length])
        I_depletion_bin = np.array([self.proc_data_dict['I_pulse_start']
                        +self.pulse_length, self.proc_data_dict['I_pulse_end']])

        Q_pulse_bin = np.array([self.proc_data_dict['Q_pulse_start'],
                    self.proc_data_dict['Q_pulse_start']+self.pulse_length])
        Q_depletion_bin = np.array([self.proc_data_dict['Q_pulse_start']
                        +self.pulse_length, self.proc_data_dict['Q_pulse_end']])

        self.plot_dicts['I_pulse_length'] = {
            'ax_id': 'I_axis',
            'xvals': I_pulse_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['I_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C0'}
            }

        self.plot_dicts['I_pulse_depletion'] = {
            'ax_id': 'I_axis',
            'xvals': I_depletion_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['I_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C1'}
            }

        self.plot_dicts['Q_pulse_length'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_pulse_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['Q_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C0'}
            }

        self.plot_dicts['Q_pulse_depletion'] = {
            'ax_id': 'Q_axis',
            'grid': True,
            'grid_kws': {'alpha': .25, 'linestyle': '--'},
            'xvals': Q_depletion_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['Q_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C1'}
            }


class Readout_landspace_Analysis(sa.Basic2DInterpolatedAnalysis):
    '''
    Analysis for Readout landscapes using adaptive sampling.
    Stores maximum fidelity parameters in quantities of interest dict as:
        - <analysis_object>.qoi['Optimal_parameter_X']
        - <analysis_object>.qoi['Optimal_parameter_Y']
    '''
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 interp_method: str = 'linear',
                 options_dict: dict=None, auto=True,
                 **kw):

        super().__init__(t_start = t_start, t_stop = t_stop,
                         label = label,
                         data_file_path = data_file_path,
                         options_dict = options_dict,
                         auto = auto,
                         interp_method=interp_method,
                         **kw)
        if auto:
            self.run_analysis()

    def process_data(self):
        super().process_data()

        # Extract maximum interpolated fidelity
        idx = [i for i, s in enumerate(self.proc_data_dict['value_names']) \
                if 'F_a' in s][0]
        X = self.proc_data_dict['x_int']
        Y = self.proc_data_dict['y_int']
        Z = self.proc_data_dict['interpolated_values'][idx]

        max_idx = np.unravel_index(np.argmax(Z), (len(X),len(Y)) )
        self.proc_data_dict['Max_F_a_idx'] = max_idx
        self.proc_data_dict['Max_F_a'] = Z[max_idx[1],max_idx[0]]

        self.proc_data_dict['quantities_of_interest'] = {
            'Optimal_parameter_X': X[max_idx[1]],
            'Optimal_parameter_Y': Y[max_idx[0]]
            }

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i, val_name in enumerate(self.proc_data_dict['value_names']):

            zlabel = '{} ({})'.format(val_name,
                                      self.proc_data_dict['value_units'][i])
            # Plot interpolated landscape
            self.plot_dicts[val_name] = {
                'ax_id': val_name,
                'plotfn': a_tools.color_plot,
                'x': self.proc_data_dict['x_int'],
                'y': self.proc_data_dict['y_int'],
                'z': self.proc_data_dict['interpolated_values'][i],
                'xlabel': self.proc_data_dict['xlabel'],
                'x_unit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'y_unit': self.proc_data_dict['yunit'],
                'zlabel': zlabel,
                'title': '{}\n{}'.format(
                    self.timestamp, self.proc_data_dict['measurementstring'])
                }
            # Plot sampled values
            self.plot_dicts[val_name+str('_sampled_values')] = {
                'ax_id': val_name,
                'plotfn': scatter_pnts_overlay,
                'x': self.proc_data_dict['x'],
                'y': self.proc_data_dict['y'],
                'xlabel': self.proc_data_dict['xlabel'],
                'x_unit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'y_unit': self.proc_data_dict['yunit'],
                'alpha': .75,
                'setlabel': 'Sampled points',
                'do_legend': True
                }
            # Plot maximum fidelity point
            self.plot_dicts[val_name+str('_max_fidelity')] = {
                'ax_id': val_name,
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['x_int']\
                            [self.proc_data_dict['Max_F_a_idx'][1]]],
                'yvals': [self.proc_data_dict['y_int']\
                            [self.proc_data_dict['Max_F_a_idx'][0]]],
                'xlabel': self.proc_data_dict['xlabel'],
                'xunit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'yunit': self.proc_data_dict['yunit'],
                'marker': 'x',
                'linestyle': '',
                'color': 'red',
                'setlabel': 'Max fidelity',
                'do_legend': True,
                'legend_pos': 'upper right'
                }


class Multiplexed_Readout_Analysis_deprecated(ba.BaseDataAnalysis):
    """
    For two qubits, to make an n-qubit mux readout experiment.
    we should vectorize this analysis

    TODO: This needs to be rewritten/debugged!
    Suggestion:
        Use N*(N-1)/2 instances of Singleshot_Readout_Analysis,
          run them without saving the plots and then merge together the
          plot_dicts as in the cross_dephasing_analysis.
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
        nr_bins = int(self.options_dict.get('nr_bins', 100))

        # self.proc_data_dict['shots_0'] = [''] * nr_expts
        # self.proc_data_dict['shots_1'] = [''] * nr_expts

        #################################################################
        #  Separating data into shots for the different prepared states #
        #################################################################
        self.proc_data_dict['nr_of_qubits'] = self.nr_of_qubits
        self.proc_data_dict['qubit_names'] = self.qubit_names

        self.proc_data_dict['ch_names'] = self.raw_data_dict['value_names'][0]

        for ch_name, shots in self.raw_data_dict['measured_values_ord_dict'].items():
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
                #####################################
                #  Binning data into 1D histograms  #
                #####################################
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


def plot_2D_ssro_histogram(xvals, yvals, zvals, xlabel, xunit, ylabel, yunit, zlabel, zunit,
                           xlim=None, ylim=None,
                           title='',
                           cmap='viridis',
                           cbarwidth='10%',
                           cbarpad='5%',
                           no_label=False,
                           ax=None, cax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    if not no_label:
        ax.set_title(title)

    # Plotting the "heatmap"
    out = flex_colormesh_plot_vs_xy(xvals, yvals, zvals, ax=ax,
                                    plot_cbar=True, cmap=cmap)
    # Adding the colorbar
    if cax is None:
        ax.ax_divider = make_axes_locatable(ax)
        ax.cax = ax.ax_divider.append_axes(
            'right', size=cbarwidth, pad=cbarpad)
    else:
        ax.cax = cax
    ax.cbar = plt.colorbar(out['cmap'], cax=ax.cax)

    # Setting axis limits aspect ratios and labels
    ax.set_aspect(1)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    set_cbarlabel(ax.cbar, zlabel, zunit)
    if xlim is None:
        xlim = np.min([xvals, yvals]), np.max([xvals, yvals])
    ax.set_xlim(xlim)
    if ylim is None:
        ylim = np.min([xvals, yvals]), np.max([xvals, yvals])
    ax.set_ylim(ylim)
