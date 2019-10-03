"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis (to be updated!)

Originally written by Adriaan, updated/rewritten by Rene May 2018
"""
import itertools
import logging
log = logging.getLogger(__name__)
from collections import OrderedDict
from copy import deepcopy

import lmfit
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap as lscmap
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as GM
from sklearn.tree import DecisionTreeClassifier as DTC

import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis.tools.data_manipulation as dm_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.fitting_models import ro_gauss, ro_CDF, gaussian_2D, \
gauss_2D_guess, \
    gaussianCDF, ro_double_gauss_guess
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from pycqed.analysis.tools.plotting import set_xlabel

odict = OrderedDict

class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 options_dict: dict=None, auto=True, **kw):
        '''
        options dict options:
            'fixed_p10' fixes p(e|g) (do not vary in fit)
            'fixed_p01' : fixes p(g|pi) (do not vary in fit)
            'auto_rotation_angle' : (bool) automatically find the I/Q mixing angle
            'rotation_angle' : manually define the I/Q mixing angle (ignored if auto_rotation_angle is set to True)
            'nr_bins' : number of bins to use for the histograms
            'post_select' :
            'post_select_threshold' :
            'nr_samples' : amount of different samples (e.g. ground and excited = 2)
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
        self.options_dict['auto_rotation_angle'] = self.options_dict.get('auto_rotation_angle', man_angle)

        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Responsible for creating the histograms based on the raw data
        """
        post_select = self.options_dict.get('post_select', False)
        post_select_threshold = \
            self.options_dict.get('post_select_threshold', 0)
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)
        nr_bins = self.options_dict.get('nr_bins', 100)

        ######################################################
        #  Separating data into shots for 0 and shots for 1  #
        ######################################################
        meas_val = self.raw_data_dict['measured_values']
        unit = self.raw_data_dict['value_units'][0]
        # loop through channels
        shots = np.zeros((2, len(meas_val),), dtype=np.ndarray)
        for j, dat in enumerate(meas_val):
            assert unit == self.raw_data_dict['value_units'][j], 'The channels have been measured using different units. This is not supported yet.'
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
            #
            ########################################################
            data_range_x = (np.min([np.min(b) for b in shots[:, 0]]),
                            np.max([np.max(b) for b in shots[:, 0]]))
            data_range_y = (np.min([np.min(b) for b in shots[:, 1]]),
                            np.max([np.max(b) for b in shots[:, 1]]))
            data_range_xy = (data_range_x, data_range_y)
            nr_bins_2D = self.options_dict.get('nr_bins_2D', 6*np.sqrt(nr_bins))
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
                print('Mixing I/Q channels with %.3f degrees '%ang_deg +
                      #'around point (%.2f, %.2f)%s'%(mid[0], mid[1], unit) +
                      ' to obtain effective voltage.')

            self.proc_data_dict['raw_offset'] = [*mid, angle]
            # create matrix
            rot_mat = [[+np.cos(-angle), -np.sin(-angle)],
                       [+np.sin(-angle), +np.cos(-angle)]]
            # rotate data accordingly
            eff_sh = np.zeros(len(shots[0]), dtype=np.ndarray)
            eff_sh[0] = np.dot(rot_mat[0], shots[0])# - mid
            eff_sh[1] = np.dot(rot_mat[0], shots[1])# - mid
        else:
            # If we have only one quadrature, use that (doh!)
            eff_sh = shots[:, 0]

        self.proc_data_dict['all_channel_int_voltages'] = shots
        self.proc_data_dict['shots_xlabel'] = 'Effective integrated Voltage'#self.raw_data_dict['value_names'][0]
        self.proc_data_dict['shots_xunit'] = unit
        self.proc_data_dict['eff_int_voltages'] = eff_sh
        self.proc_data_dict['nr_shots'] = [len(eff_sh[0]), len(eff_sh[1])]
        sh_min = min(np.min(eff_sh[0]), np.min(eff_sh[1]))
        sh_max = max(np.max(eff_sh[0]), np.max(eff_sh[1]))
        data_range = (sh_min, sh_max)

        eff_sh_sort = np.sort(list(eff_sh), axis=1)
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
        cum_params['A_amplitude'].vary = False
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
        self.proc_data_dict['residual_excitation'] = bv['B_spurious'].value
        self.proc_data_dict['measurement_induced_relaxation'] = bv['A_spurious'].value

    def prepare_plots(self):
        # Did we load two voltage components (shall we do 2D plots?)
        two_dim_data = len(self.proc_data_dict['all_channel_int_voltages'][0]) == 2

        eff_voltage_label = self.proc_data_dict['shots_xlabel']
        eff_voltage_unit = self.proc_data_dict['shots_xunit']
        x_volt_label = self.raw_data_dict['value_names'][0]
        x_volt_unit = self.raw_data_dict['value_units'][0]
        if two_dim_data:
            y_volt_label = self.raw_data_dict['value_names'][1]
            y_volt_unit = self.raw_data_dict['value_units'][1]
        z_hist_label = 'Counts'
        label_0 = '|g> prep.'
        label_1 = '|e> prep.'
        title = ('\n' + self.timestamps[0] + ' - "' +
                 self.raw_data_dict['measurementstring'] + '"')


        #### 1D histograms
        log_hist = self.options_dict.get('log_hist', False)
        bin_x = self.proc_data_dict['bin_edges']
        bin_y = self.proc_data_dict['hist']
        self.plot_dicts['hist_0'] = {
            'title': 'Binned Shot Counts' + title,
            'ax_id' : '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[0],
            'xwidth' : self.proc_data_dict['binsize'],
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
            'xwidth' : self.proc_data_dict['binsize'],
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

        #### CDF
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

        ### Vlines for thresholds
        th_raw = self.proc_data_dict['threshold_raw']
        threshs = [th_raw,]
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

        #### 2D Histograms
        if two_dim_data:
            iq_centers = None
            if 'IQ_pos' in self.proc_data_dict and self.proc_data_dict['IQ_pos'] is not None:
                iq_centers = self.proc_data_dict['IQ_pos']
                peak_marker_2D = {
                    'plotfn': self.plot_line,
                    'xvals': iq_centers[1],
                    'yvals': iq_centers[0],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': 'x',
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
                'plotfn': self.plot_colorxy,
                'xvals': self.proc_data_dict['2D_histogram_y'],
                'yvals': self.proc_data_dict['2D_histogram_x'],
                'zvals': self.proc_data_dict['2D_histogram_z'][0],
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
                'plotfn': self.plot_colorxy,
                'xvals': self.proc_data_dict['2D_histogram_y'],
                'yvals': self.proc_data_dict['2D_histogram_x'],
                'zvals': self.proc_data_dict['2D_histogram_z'][1],
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

            #### Scatter Shots
            volts = self.proc_data_dict['all_channel_int_voltages']
            vxr = [np.min([np.min(a) for a in volts[:][1]]),
                   np.max([np.max(a) for a in volts[:][1]])]
            vyr = [np.min([np.min(a) for a in volts[:][0]]),
                   np.max([np.max(a) for a in volts[:][0]])]
            self.plot_dicts['2D_shots_0'] = {
                'title': 'Raw Shots' + title,
                'ax_id': '2D_shots',
                'plotfn': self.plot_line,
                'xvals': volts[0][1],
                'yvals': volts[0][0],
                #'range': [vxr, vyr],
                #'xrange': vxr,
                #'yrange': vyr,
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
                'xvals': volts[1][1],
                'yvals': volts[1][0],
                #'range': [vxr, vyr],
                #'xrange': vxr,
                #'yrange': vyr,
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
            #todo: add seperate fits for residual and main gaussians
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
                fit_text += '\nSNR (fit) = ${:.3f}\\pm{:.3f}$'.format(snr.value, snr.stderr)

                fr = self.fit_res['shots_all']
                bv = fr.params
                a_sp = bv['A_spurious']
                fit_text += '\n\nSpurious Excitations:'
                fit_text += '\n$p(e|0) = {:.3f}$'.format(a_sp.value)
                if self.options_dict.get('fixed_p01', True) == True:
                    fit_text += '$\\pm{:.3f}$'.format(a_sp.stderr)
                else:
                    fit_text += ' (fixed)'

                b_sp = bv['B_spurious']
                fit_text += ' \n$p(g|\\pi) = {:.3f}$'.format(b_sp.value)
                if self.options_dict.get('fixed_p10', True) == True:
                    fit_text += '$\\pm{:.3f}$'.format(b_sp.stderr)
                else:
                    fit_text += ' (fixed)'

            if two_dim_data:
                offs = self.proc_data_dict['raw_offset']
                #fit_text += '\nOffset from raw:\n'
                #fit_text += '({:.3f},{:.3f}) {},\n'.format(offs[0], offs[1], eff_voltage_unit)
                fit_text += '\n\nRotated by ${:.1f}^\\circ$'.format((offs[2]*180/np.pi)%180)
                auto_rot = self.options_dict.get('auto_rotation_angle', True)
                fit_text += '(auto)' if auto_rot else '(man.)'
            else:
                fit_text += '\n\n(Single quadrature data)'

            fit_text += '\n\nTotal shots: %d+%d'%(*self.proc_data_dict['nr_shots'],)

            for ax in ['cdf', '1D_histogram']:
                self.plot_dicts['text_msg_' + ax] = {
                        'ax_id': ax,
                        # 'ypos': 0.15,
                        'xpos' : 1.05,
                        'horizontalalignment' : 'left',
                        'plotfn': self.plot_text,
                        'box_props': 'fancy',
                        'text_string': fit_text,
                    }

class Singleshot_Readout_Analysis_Qutrit(ba.BaseDataAnalysis):
    def __init__(self, t_start: str or list = None, t_stop: str = None,
                 label: str or list = '', do_fitting: bool = True,
                 data_file_path: str = None, levels = ('g', 'e', 'f'),
                 options_dict: dict = None, auto=True, **kw):
        '''
        options dict options:
            'nr_bins' : number of bins to use for the histograms
            'post_select' :
            'post_select_threshold' :
            'nr_samples' : amount of different samples (e.g. ground and excited = 2)
            'sample_0' : index of first sample (ground-state)
            'sample_1' : index of second sample (first excited-state)
            'max_datapoints' : maximum amount of datapoints for culumative fit
            'log_hist' : use log scale for the y-axis of the 1D histograms
            'verbose' : see BaseDataAnalysis
            'presentation_mode' : see BaseDataAnalysis
            'classif_method': how to classify the data.
                'ncc' : default. Nearest Cluster Center
                'gmm': gaussian mixture model.
                'threshold': finds optimal vertical and horizontal thresholds.
            'classif_kw': kw to pass to the classifier
            see BaseDataAnalysis for more.
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_data': 'measured_data',
            'value_names': 'value_names',
            'value_units': 'value_units'}
        self.numeric_params = []
        self.DEFAULT_CLASSIF = "gmm"
        self.DEFAULT_PRE_SEL = False

        self.levels = levels
        # empty dict for analysis results
        self.proc_data_dict = OrderedDict()
        self.pre_selection = self.options_dict.get('pre_selection',
                                                   self.DEFAULT_PRE_SEL)
        self.classif_method = self.options_dict.get("classif_method",
                                                    self.DEFAULT_CLASSIF)
        if auto:
            self.run_analysis()


    def process_data(self):
        """
        Create the histograms based on the raw data
        """
        ######################################################
        #  Separating data into shots for each level         #
        ######################################################
        # measured values is a list of arrays with measured
        # values for each level in self.levels
        meas_val = {l: np.array([self.raw_data_dict[i]['measured_data'][c]
                        for c in self.raw_data_dict[i]['measured_data'].keys()])
                    for i, l in enumerate(self.levels)}
        # print([ c for c in meas_val['e'].keys()])
        intermediate_ro = dict()    # store intermediate ro (preselection)
        data = dict()               # store final data
        mu = dict()                 # store mean of measurements
        # loop through levels
        for l, l_data in meas_val.items():
            if self.pre_selection:
                intermediate_ro[l], data[l] = self._filter(l_data)
            else:
                data[l] = l_data
            mu[l] = np.mean(data[l], axis=-1)
            # make 2D array in case only one channel (1D array)
            if len(data[l].shape) == 1:
                data[l] = np.array([data[l]])

        X = np.vstack([data[l].transpose() for l in self.levels])
        prep_states = np.hstack(
            [np.ones_like(data[l][0]) * i for i, l in enumerate(self.levels)])

        self.proc_data_dict['analysis_params'] = OrderedDict()
        self.proc_data_dict['analysis_params']['mu'] = deepcopy(mu)
        self.proc_data_dict['data'] = dict(X=deepcopy(X), prep_states=prep_states)
        self.proc_data_dict['keyed_data'] = deepcopy(data)

        assert np.ndim(X) == 2, "Data must be a two D array. " \
                                "Received shape {}, ndim {}"\
                                .format(X.shape, np.ndim(X))
        pred_states, clf_params = \
            self._classify(X, prep_states,
                           method=self.classif_method,
                           **self.options_dict.get("classif_kw", dict()))
        fm = self.fidelity_matrix(prep_states, pred_states)

        self.proc_data_dict['analysis_params']['state_prob_mtx'] = fm
        self.proc_data_dict['analysis_params']['n_shots'] = X.shape[0]
        self.proc_data_dict['analysis_params'] \
                           ['classifier_params'] = clf_params

        if self.pre_selection:
            prep_states = []
            X = []
            #re do with classification first of preselection and masking
            pred_presel = dict()
            for i, l in enumerate(self.levels):
                data[l] = data[l].transpose()
                pred_presel[l] = self.clf_.predict(intermediate_ro[l]
                                                   .transpose())
                data_masked = data[l][pred_presel[l] == 0.]
                X.append(data_masked)
                prep_states.append(np.ones((data_masked.shape[0]))*i)

            X = np.vstack(X)
            pred_states = self.clf_.predict(X)
            prep_states = np.hstack(prep_states)

            fm = self.fidelity_matrix(prep_states, pred_states)
            self.proc_data_dict['data_masked'] = dict(X=deepcopy(X),
                                                      prep_states=prep_states)
            self.proc_data_dict['analysis_params']\
                               ['state_prob_mtx_masked'] = fm
            self.proc_data_dict['analysis_params']['n_shots_masked'] = \
                X.shape[0]

    def _filter(self, data):
        """
        Filters data of level and returns intermediate ro and data separately
        """
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)
        intermediate_ro, data = data.transpose()[sample_0::nr_samples], \
                                data.transpose()[sample_1::nr_samples]
        return intermediate_ro.transpose(), data.transpose()

    def _classify(self, X, prep_state, method, **kw):
        """

        Args:
            X: measured data to classify
            prep_state: prepared states (true values)
            type: classification method

        Returns:

        """
        if np.ndim(X) == 1:
            X = X.reshape((-1,1))

        params = dict()

        if method == 'ncc':
            class NCC:
                def __init__(self, cluster_centers):
                    """
                    cluster_centers is a dict of cluster centers
                    (name as key, n dimensional array as value)

                    """
                    self.cluster_centers = cluster_centers
                def predict(self, X):
                    pred_states = []
                    for pt in X:
                        dist = []
                        for _, cluster_center in self.cluster_centers.items():
                            dist.append(np.linalg.norm(pt - cluster_center))
                        dist = np.asarray(dist)
                        pred_states.append(np.argmin(dist))
                    pred_states = np.array(pred_states)
                    return pred_states
                def predict_proba(self, X):
                    raise NotImplementedError("Not implemented for NCC")
            ncc = NCC(self.proc_data_dict['analysis_params']['mu'])
            pred_states = ncc.predict(X)
            self.clf_ = ncc
            return pred_states, dict()

        elif method == 'gmm':
            cov_type = kw.pop("covariance_type", "tied")
            # full allows full covariance matrix for each level. Other options
            # see GM documentation
            gm = GM(n_components=len(self.levels), covariance_type=cov_type,
                    random_state=0,
                    means_init=[mu for _, mu in
                                self.proc_data_dict['analysis_params']
                                    ['mu'].items()])
            gm.fit(X)
            pred_states = np.argmax(gm.predict_proba(X), axis=1)

            if cov_type == "tied":
                # in case all components share the same cov mtx return a list
                # of identical cov matrices
                covs = [gm.covariances_ for _ in range(gm.n_components)]
            elif cov_type == "full":
                # already of the right shape (n_comp, n_features, n_features)
                covs = gm.covariances_
            elif cov_type == "spherical":
                # return list of sigma_i^2 * I instead of list of sigma_i^2
                covs = [np.diag([gm.covariances_[i]
                                 for _ in range(X.shape[1])])
                        for i in range(gm.n_components)]
            elif cov_type == "diag":
                # make covariance matrices from diagonals
                covs = [np.diag(gm.covariances_[i])
                            for i in range(gm.n_components)]
            else:
                raise ValueError("covariance type: {} is not supported"
                                 .format(cov_type))
            params['means_'] = gm.means_
            params['covariances_'] = gm.covariances_ #covs
            params['covariance_type'] = gm.covariance_type
            params['weights_'] = gm.weights_
            params['precisions_cholesky_'] = gm.precisions_cholesky_
            self.clf_ = gm
            return pred_states, params

        elif method == "threshold":
            tree = DTC(max_depth=kw.pop("max_depth", X.ndim),
                       random_state=0, **kw)
            tree.fit(X, prep_state)
            pred_states = tree.predict(X)
            params["thresholds"], params["mapping"] = \
                self._extract_tree_info(tree, self.levels)
            self.clf_ = tree
            if len(params["thresholds"]) == 1:
                msg = "Best 2 thresholds to separate this data lie on axis {}" \
                    ", most probably because the data is not well separated." \
                    "The classifier attribute clf_ can still be used for " \
                    "classification (which was done to obtain the state " \
                    "assignment probability matrix), but only the threshold" \
                    " yielding highest gini impurity decrease was returned." \
                    "\nTo circumvent this problem, you can either choose" \
                    " a second threshold manually (fidelity will likely be " \
                    "worse), make the data more separable, or use another " \
                    "classification method."
                logging.warning(msg.format(list(params['thresholds'].keys())[0]))
            return pred_states, params
        elif method == "threshold_brute":
            raise NotImplementedError()
        else:
            raise NotImplementedError("Classification method: {} is not "
                                      "implemented. Available methods: {}"
                                      .format(method, ['ncc', 'gmm',
                                                       'threshold']))


    @staticmethod
    def fidelity_matrix(prep_states, pred_states, levels=('g', 'e', 'f'),
                        plot=False, normalize=True):
        fm = confusion_matrix(prep_states, pred_states)
        if plot:
            Singleshot_Readout_Analysis_Qutrit.plot_fidelity_matrix(fm,
                                                                    levels)
        if normalize:
            fm = fm.astype('float') / fm.sum(axis=1)[:, np.newaxis]
        return fm

    @staticmethod
    def plot_fidelity_matrix(fm, target_names,
                             title="State Assignment Probability Matrix",
                             auto_shot_info=True,
                             cmap=None, normalize=True, show=False):
        fidelity_avg = np.trace(fm) / float(np.sum(fm))
        if auto_shot_info:
            title += '\nTotal # shots:{}'.format(np.sum(fm))
        if cmap is None:
            cmap = plt.get_cmap('Reds')

        fig, ax = plt.subplots(1, figsize=(8, 6))

        if normalize:
            fm = fm.astype('float') / fm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(fm, interpolation='nearest', cmap=cmap,
                       norm=mc.LogNorm(), vmin=5e-3, vmax=1.)
        ax.set_title(title)
        fig.colorbar(im)

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels( target_names, rotation=45)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(target_names)

        thresh = fm.max() / 1.5 if normalize else fm.max() / 2
        for i, j in itertools.product(range(fm.shape[0]), range(fm.shape[1])):
            if normalize:
                ax.text(j, i, "{:0.4f}".format(fm[i, j]),
                         horizontalalignment="center",
                         color="white" if fm[i, j] > thresh else "black")
            else:
                ax.text(j, i, "{:,}".format(fm[i, j]),
                         horizontalalignment="center",
                         color="white" if fm[i, j] > thresh else "black")
        plt.tight_layout()
        ax.set_ylabel('Prepared State')
        ax.set_xlabel('Assigned State\n$\mathcal{{F}}_{{avg}}$={:0.2f} %'
                      .format(fidelity_avg * 100))
        if show:
            plt.show()
        return fig


    @staticmethod
    def _extract_tree_info(tree_clf, class_names=None):
        tree_ = tree_clf.tree_
        feature_name = [np.arange(tree_.n_features)[i]
                        for i in tree_.feature]
        if class_names is None:
            class_names = np.arange(len(tree_.value[0]))
        thresholds, mapping = OrderedDict(), dict()

        def recurse(node, thresholds_final, loc, mapping, feature_depth_path):
            name = feature_name[node]
            feature_depth_path = feature_depth_path.copy() + \
                                 [tree_.feature[node]]

            if tree_.feature[node] != -2:
                threshold = tree_.threshold[node]
                if not name in thresholds_final.keys():
                    thresholds_final[name] = threshold
                recurse(tree_.children_left[node], thresholds_final,
                        loc + [0], mapping, feature_depth_path)
                recurse(tree_.children_right[node], thresholds_final,
                        loc + [1], mapping, feature_depth_path)
            else:
                log.debug(loc, tree_.value[node], feature_depth_path)
                if len(loc) < tree_.n_features:
                    log.warning(
                        "Location < n_features, threshold mapping might not be "
                        "correct")
                    for l in itertools.combinations_with_replacement(
                            [0, 1], tree_.n_features - len(loc)):
                        loc_full = loc + list(l)

                        mapping[tuple(
                            [loc_full[i]
                             for i in np.argsort(feature_depth_path[:-1])])] = \
                                class_names[np.argmax(tree_.value[node])]

                else:
                    # swap if first threshold is on axis 1
                    # FIXME: will work only when 2 integration units are used
                    if list(thresholds_final.keys())[0] == 1:
                        loc = list(reversed(loc))
                    mapping[tuple(loc)] = class_names[np.argmax(tree_.value[node])]
                    log.debug(mapping)

        recurse(0, thresholds, [], mapping, feature_depth_path=[])

        # translate keys to codeword index format
        mapping = {Singleshot_Readout_Analysis_Qutrit._to_codeword_idx(k): v
                   for k, v in mapping.items()}
        if len(mapping) < 2 ** tree_clf.max_depth:
            log.warning(f"threshold mapping is of length {len(mapping)} "
                        f"instead of expected  length "
                        f"{2 ** tree_clf.max_depth}. Mapping may be incorrect.")
        return thresholds, mapping

    @staticmethod
    def _to_codeword_idx(tuple):
        """
        Maps a binary tuple (in ascending axis order) to codeword index.
        eg. for 4 tuples:
        (0, 1) | (1, 1)          2 | 3
        ---------------    -->   -----
        (0, 0) | (1, 0)          0 | 1
        :param tuple:
        :return:
        """
        return np.sum([i * 2**n for n, i in enumerate(tuple)])

    @staticmethod
    def plot_scatter_and_marginal_hist(data, y_true=None, plot_fitting=False,
                                       **kwargs):
        """
        Plot data with classifier boundary functions,
        side histograms and possibly thresholds
        Args:
            data: array of size (n_samples, 2)
            y_true: array of size (n_samples,) with classification label
                for each class
            plot_fitting: Not implemented (not useful?)
            **kwargs: plotting keywords

        Returns:

        """
        if kwargs.get("fig", None) is None:
            fig, axes = plt.subplots(figsize=(10, 8))
            kwargs['fig'] = fig

        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],
                               figure=kwargs['fig'])

        # Create scatter plot
        ax = plt.subplot(gs[1, 0])
        for yval in np.unique(y_true):
            ax.scatter(data[:, 0][y_true == yval], data[:, 1][y_true == yval],
                       alpha=kwargs.get('alpha', 0.6), marker='.',
                       label=kwargs.get("legend_labels",
                                        [yval] * len(np.unique(y_true)))[int(yval)])
        #h, labels = sc.legend_elements()
        #legend = ax.legend(h, kwargs.get("legend_labels", labels))
        #ax.add_artist(legend)
        # Create Y-marginal (right)
        axr = plt.subplot(gs[1, 1], sharey=ax, frameon=False)
        for yval in np.unique(y_true):
            axr.hist(data[:, 1][y_true == yval], bins=50,
                     orientation='horizontal', density=False,
                     alpha=kwargs.get('alpha', 0.6))
        axr.set_xscale(kwargs.get("scale", "log"))
        plt.setp(axr.get_yticklabels(), visible=False)

        # Create X-marginal (top)
        axt = plt.subplot(gs[0, 0], sharex=ax, frameon=False)
        plt.setp(axt.get_xticklabels(), visible=False)

        for yval in np.unique(y_true):
            axt.hist(data[:, 0][y_true == yval], bins=50,
                     orientation='vertical', density=False,
                     alpha=kwargs.get('alpha', 0.6))
        axt.set_yscale(kwargs.get("scale", "log"))

        axt.set_title(kwargs.get('title', None))
        ax.set_xlabel(kwargs.get('xlabel', None))
        ax.set_ylabel(kwargs.get('ylabel', None))

        # Bring the marginals closer to the scatter plot
        kwargs['fig'].tight_layout(pad=1)

        if plot_fitting:
            ymin, ymax = data[:, 1].min(), data[:, 1].max()
            xmin, xmax = data[:, 0].min(), data[:, 0].max()
            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            raise NotImplementedError()
        return kwargs['fig']

    @staticmethod
    def plot_clf_boundaries(X, clf, ax=None, cmap=None):
        def make_meshgrid(x, y, h=None, margin=None):
            if margin is None:
                deltax = x.max() - x.min()
                deltay = y.max() - y.min()
                margin_x = deltax * 0.10
                margin_y = deltay * 0.10
            else:
                margin_x, margin_y = margin, margin
            x_min, x_max = x.min() - margin_x, x.max() + margin_x
            y_min, y_max = y.min() - margin_y, y.max() + margin_y
            if h is None:
                h = 0.01*(x_max - x_min)
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 10))

        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(ax, clf, xx, yy, cmap=cmap, alpha=0.3)

    @staticmethod
    def plot_std(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms

        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = 1#np.sqrt(1 + pearson)
        ell_radius_y = 1# np.sqrt(1 - pearson)

        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(mean[0])

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(mean[1])

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def prepare_plots(self):
        cmap = plt.get_cmap('tab10')
        tab_x = a_tools.truncate_colormap(cmap, 0, len(self.levels)/10)

        show = self.options_dict.get("show", False)

        kwargs = dict(legend_labels=self.levels,
                      xlabel="Integration Unit 0",
                      ylabel="Integration Unit 1",
                      scale=self.options_dict.get("hist_scale", "linear"),
                      cmap=tab_x)
        data_keys = [k for k in list(self.proc_data_dict.keys()) if
                        k.startswith("data")]
        for dk in data_keys:
            data = self.proc_data_dict[dk]
            title =  self.raw_data_dict[0]['timestamp'] + " " + dk + \
                "\n{} classifier".format(self.classif_method)
            kwargs.update(dict(title=title))
            # plot data and histograms
            fig = self.plot_scatter_and_marginal_hist(data['X'],
                                                      data["prep_states"],
                                                      **kwargs)
            # plot means
            main_ax =  fig.get_axes()[0]
            for _ , mu in self.proc_data_dict['analysis_params']['mu'].items():
               main_ax.scatter(mu[0], mu[1], color='r', s=80)

            # plot clf_boundaries
            self.plot_clf_boundaries(data['X'], self.clf_, ax=main_ax,
                                     cmap=tab_x)

            # plot thresholds
            plt_fn = {0: main_ax.axvline, 1: main_ax.axhline}
            thresholds = self.proc_data_dict['analysis_params'][
                'classifier_params'].get("thresholds", dict())
            for k, thres in thresholds.items():
                plt_fn[k](thres, linewidth=2,
                          label="threshold i.u. {}: {:.5f}".format(k, thres),
                          color='k', linestyle="--")
                main_ax.legend(loc=[0.2,-0.62])

            self.figs['{}_classifier_{}'.format(self.classif_method, dk)] = fig
        if show:
            plt.show()

        title = self.raw_data_dict[0]['timestamp'] + "\n{} State Assignment" \
            " Probability Matrix\nTotal # shots:{}"\
            .format(self.classif_method,
                    self.proc_data_dict['analysis_params']['n_shots'])
        fig = self.plot_fidelity_matrix(
            self.proc_data_dict['analysis_params']['state_prob_mtx'],
            self.levels, title=title, show=show, auto_shot_info=False)
        self.figs['state_prob_matrix_{}'.format(self.classif_method)] = fig

        if self.pre_selection:
            title = self.raw_data_dict[0]['timestamp'] + \
                "\n{} State Assignment Probability Matrix Masked"\
                "\nTotal # shots:{}".format(
                    self.classif_method,
                    self.proc_data_dict['analysis_params']['n_shots_masked'])

            fig = self.plot_fidelity_matrix(
                self.proc_data_dict['analysis_params'] \
                                   ['state_prob_mtx_masked'],
                self.levels, title=title, show=show, auto_shot_info=False)
            fig_key = 'state_prob_matrix_masked_{}'.format(self.classif_method)
            self.figs[fig_key] = fig


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
        self.kept_shots = 0
        self.thresholds = options_dict['thresholds']
        self.channel_map = options_dict['channel_map']
        self.use_preselection = options_dict.get('use_preselection', False)
        qubits = list(self.channel_map.keys())

        self.readout_names = options_dict.get('readout_names', None)
        if self.readout_names is None:
            # TODO Default values should come from the MC parameters
            None

        self.observables = options_dict.get('observables', None)

        if self.observables is None:
            combination_list = list(itertools.product([False, True],
                                                      repeat=len(qubits)))
            preselection_condition = dict(zip(
                [(qb, self.options_dict.get('preselection_shift', -1))
                 for qb in qubits],  # keys contain shift
                combination_list[0]  # first comb has all ground
            ))
            self.observables = odict()

            # add preselection condition also as an observable
            if self.use_preselection:
                self.observables["pre"] = preselection_condition
            # add all combinations
            for i, states in enumerate(combination_list):
                name = ''.join(['e' if s else 'g' for s in states])
                obs_name = '$\| ' + name + '\\rangle$'
                self.observables[obs_name] = dict(zip(qubits, states))
                # add preselection condition
                if self.use_preselection:
                    self.observables[obs_name].update(preselection_condition)

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
                self.raw_data_dict['measured_data'][channel])

            shots_thresh[qubit] = (shots_cont > self.thresholds[qubit])
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

        This function has been check with a profiler and 85% of the time is
        spent on comparison with the mask. Thus there is no trivial optimization
        possible.

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
        self.prepare_plot_prob_table(self.use_preselection)

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
        cm = mc.LinearSegmentedColormap('customcmap', cdict)

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

    def measurement_operators_and_results(self, tomography_qubits=None):
        """
        Calculates and returns:
            A tuple of
                count tables for each data segment for the observables;
                the measurement operators corresponding to each observable;
                and the expected covariation matrix between the operators.

        If the calibration segments are passed, there must be a calibration
        segments for each of the computational basis states of the Hilber space.
        If there are no calibration segments, perfect readout is assumed.

        The calling class must filter out the relevant data segments by itself!
        """
        try:
            preselection_obs_idx = list(self.observables.keys()).index('pre')
        except ValueError:
            preselection_obs_idx = None
        observabele_idxs = [i for i in range(len(self.observables))
                            if i != preselection_obs_idx]

        qubits = list(self.channel_map.keys())
        if tomography_qubits is None:
            tomography_qubits = qubits
        d = 2**len(tomography_qubits)
        data = self.proc_data_dict['probability_table']
        data = data.T[observabele_idxs]
        if not 'cal_points' in self.options_dict:
            Fsingle = {None: np.array([[1, 0], [0, 1]]),
                       True: np.array([[0, 0], [0, 1]]),
                       False: np.array([[1, 0], [0, 0]])}
            Fs = []
            Omega = []
            for obs in self.observables.values():
                F = np.array([[1]])
                nr_meas = 0
                for qb in tomography_qubits:
                    # TODO: does not handle conditions on previous readouts
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
            Fs = [np.diag(ms) for ms in means.T]
            return data, Fs, covars

    def calibration_point_means_and_channel_covariations(self):
        observables = [v for k, v in self.observables.items() if k != 'pre']
        try:
            preselection_obs_idx = list(self.observables.keys()).index('pre')
        except ValueError:
            preselection_obs_idx = None
        observabele_idxs = [i for i in range(len(self.observables))
                            if i != preselection_obs_idx]

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
                        [observabele_idxs] for idx_ro in cal_point[0]]
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
            'probability_table'][:,observabele_idxs]]) + obs_products
        covars = np.mean(covars[list(cal_readouts)], 0)

        return means, covars


def get_shots_zero_one(data, post_select: bool=False,
                       nr_samples: int=2, sample_0: int=0, sample_1: int=1,
                       post_select_threshold: float = None):
    if not post_select:
        shots_0, shots_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)
    else:
        # FIXME nathan 2019.05.17: This is useless?
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
        self.preselection_available = False
        if self.n_readouts == len(def_seg_names_prep):
            def_seg_names = def_seg_names_prep
        elif self.n_readouts == 2*len(def_seg_names_prep):
            self.preselection_available = True
            def_seg_names = [x for t in zip(*[
                ["sel"]*len(def_seg_names_prep),
                def_seg_names_prep]) for x in t]
        else:
            def_seg_names = list(range(len(def_seg_names_prep)))

        # User can override the automatic value determined from the
        #   number of readouts
        self.use_preselection = options_dict.get('use_preselection',
                                             self.preselection_available)

        self.observables = options_dict.get('observables', None)

        if self.observables is None:
            combination_list = list(itertools.product([False, True],
                                                      repeat=len(qubits)))
            preselection_condition = dict(zip(
                [(qb, -1) for qb in qubits],  # keys contain shift
                combination_list[0]  # first comb has all ground
            ))

            self.observables = OrderedDict([])
            # add preselection condition also as an observable
            if self.use_preselection:
                self.observables["pre"] = preselection_condition
            # add all combinations
            for i, states in enumerate(combination_list):
                obs_name = '$\| ' + \
                           ''.join(['e' if s else 'g' for s in states]) + \
                           '\\rangle$'
                self.observables[obs_name] = dict(zip(qubits, states))
                # add preselection condition
                if self.use_preselection:
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
        super().prepare_plot_prob_table(only_odd=self.use_preselection)

    def process_data(self):
        super().process_data()

        table_norm = self.proc_data_dict['probability_table']
        if self.use_preselection:
            table_norm = table_norm[1::2, 1:] / (table_norm[1::2, 0][:, None])
        self.proc_data_dict['probability_table_data_only'] = table_norm

        if self.options_dict.get('do_cross_fidelity', True):
            self.proc_data_dict['cross_fidelity_matrix'] = \
                self.cross_fidelity_matrix(table_norm, len(self.channel_map))
            self.save_processed_data('cross_fidelity_matrix')
        # self.proc_data_dict['cross_correlations_matrix'] = \
        #     self.cross_correlations_matrix()

        self.save_processed_data('probability_table')
        self.save_processed_data('probability_table_data_only')

    @staticmethod
    def cross_fidelity_matrix(table_norm, n_qubits):
        masks = []
        for i in reversed(range(n_qubits)):
            masks.append(np.arange(2**n_qubits)//(2**i)%2 != 0)
        cf = np.zeros((n_qubits, n_qubits))
        for qb_prep in range(n_qubits):
            for qb_assign in range(n_qubits):
                err = np.mean(table_norm[
                    masks[qb_prep]][:,
                    np.logical_not(masks[qb_assign])])
                err += np.mean(table_norm[
                    np.logical_not(masks[qb_prep])][:,
                    masks[qb_assign]])
                err *= 2**(n_qubits-1)
                cf[qb_prep, qb_assign] = 1 - err
        return cf

    @staticmethod
    def cross_correlations_matrix(table_norm, n_qubits):
        gres = np.array([np.power(-1, np.arange(2**n_qubits)//(2**(n_qubits-i-1))) >= 0 for i in range(n_qubits)])
        pee = np.zeros((n_qubits, n_qubits))
        peg = np.zeros((n_qubits, n_qubits))
        pge = np.zeros((n_qubits, n_qubits))
        pgg = np.zeros((n_qubits, n_qubits))
        pg = np.zeros(n_qubits)
        pe = np.zeros(n_qubits)
        for k in np.arange(2**n_qubits): # prepare state
            for l in np.arange(2**n_qubits): # result state
                for i in np.arange(n_qubits): # qubit 1
                    if gres[i][l]:
                        pg[i] += table_norm[k, l]/2**n_qubits
                    else:
                        pe[i] += table_norm[k, l]/2**n_qubits
                    for j in np.arange(n_qubits): # qubit 2
                        if gres[i][l] and gres[j][l]:
                            pgg[i, j] += table_norm[k, l]/2**n_qubits
                        elif gres[i][l] and not gres[j][l]:
                            pge[i, j] += table_norm[k, l]/2**n_qubits
                        elif not gres[i][l] and gres[j][l]:
                            peg[i, j] += table_norm[k, l]/2**n_qubits
                        else:
                            pee[i, j] += table_norm[k, l]/2**n_qubits
        C = pgg + pee - pge - peg - np.outer(pg, pg) - np.outer(pe, pe) +\
            np.outer(pg, pe) + np.outer(pe, pg)
        C = np.diag(np.diagonal(C)**-0.5).dot(C).dot(np.diag(np.diagonal(C)**-0.5))
        return C


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


class SingleQubitResetAnalysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        # only 1 datafile should be processed
        self.single_timestamp = True

        # these parameters are converted to floats
        self.numeric_params = []

        # these parameters are extracted from the hdf5 file
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        if auto:
            self.run_analysis()

    def process_data(self):
        nr_reset = self.options_dict.get('nr_reset')
        nr_readout = nr_reset + 1
        qubit_idx = self.options_dict.get('qubit_idx')
        nr_qubits = len(self.raw_data_dict['value_names'])
        nr_bins = self.options_dict.get('nr_bins', 100)
        
        ######################################
        # extract shots to individual arrays #
        ######################################
        self.shots_max = float('-inf')
        self.shots_min = float('inf')
        self.proc_data_dict['shots_0'] = ['']*nr_readout
        self.proc_data_dict['shots_1'] = ['']*nr_readout
        self.proc_data_dict['shots_0_dig'] = ['']*nr_readout
        self.proc_data_dict['shots_1_dig'] = ['']*nr_readout
        self.proc_data_dict['channel_idx'] = self.raw_data_dict['value_names'] \
            .index(self.options_dict['channel_name'])
        channel_idx = self.proc_data_dict['channel_idx']
        
        readout_idxs = np.arange(len(self.raw_data_dict['measured_values'][0]))
        for i in range(nr_readout):
            mask0 = (readout_idxs % nr_readout == i)
            mask1 = mask0*((readout_idxs//(nr_readout*2**qubit_idx))%2 == 1)
            mask0 = mask0*((readout_idxs//(nr_readout*2**qubit_idx))%2 == 0)
            shots0 = self.raw_data_dict['measured_values'] \
                [channel_idx][mask0]
            shots1 = self.raw_data_dict['measured_values'] \
                [channel_idx][mask1]
            self.shots_max = max(self.shots_max, max(shots0.max(), shots1.max()))
            self.shots_min = min(self.shots_min, min(shots0.min(), shots1.min()))
            self.proc_data_dict['shots_0'][i] = shots0
            self.proc_data_dict['shots_1'][i] = shots1
            self.proc_data_dict['shots_0_dig'][i] = shots0 >= self.options_dict['threshold']
            self.proc_data_dict['shots_1_dig'][i] = shots1 >= self.options_dict['threshold']

        ###########################################
        # generate 1D histograms for each readout #
        ###########################################
        self.proc_data_dict['hist_0'] = ['']*nr_readout
        self.proc_data_dict['hist_1'] = ['']*nr_readout
        for i in range(nr_readout):
            hist0, bins = np.histogram(self.proc_data_dict['shots_0'][i],
                                       bins=nr_bins, range=(self.shots_min, self.shots_max))
            hist1, bins = np.histogram(self.proc_data_dict['shots_1'][i],
                                       bins=nr_bins, range=(self.shots_min, self.shots_max))
            self.proc_data_dict['hist_0'][i] = hist0
            self.proc_data_dict['hist_1'][i] = hist1
        self.proc_data_dict['bin_edges'] = bins
        self.proc_data_dict['bin_centers'] = (bins[1:] + bins[:-1])/2

        #########################################
        # generate 2D histograms for each reset #
        #########################################
        self.proc_data_dict['hist2_0'] = ['']*nr_reset
        self.proc_data_dict['hist2_1'] = ['']*nr_reset
        for i in range(nr_reset):
            hist0, _, _ = np.histogram2d(self.proc_data_dict['shots_0'][i], self.proc_data_dict['shots_0'][i+1],
                                         bins=nr_bins, range=((self.shots_min, self.shots_max), (self.shots_min, self.shots_max)))
            hist1, _, _ = np.histogram2d(self.proc_data_dict['shots_1'][i], self.proc_data_dict['shots_1'][i+1],
                                         bins=nr_bins, range=((self.shots_min, self.shots_max), (self.shots_min, self.shots_max)))
            self.proc_data_dict['hist2_0'][i] = hist0
            self.proc_data_dict['hist2_1'][i] = hist1

        ###############################
        # Extract state probabilities #
        ###############################
        self.proc_data_dict['pg0'] = ['']*nr_readout
        self.proc_data_dict['pe0'] = ['']*nr_readout
        self.proc_data_dict['pg1'] = ['']*nr_readout
        self.proc_data_dict['pe1'] = ['']*nr_readout
        self.proc_data_dict['pgg0'] = ['']*nr_reset
        self.proc_data_dict['pge0'] = ['']*nr_reset
        self.proc_data_dict['peg0'] = ['']*nr_reset
        self.proc_data_dict['pee0'] = ['']*nr_reset
        self.proc_data_dict['pgg1'] = ['']*nr_reset
        self.proc_data_dict['pge1'] = ['']*nr_reset
        self.proc_data_dict['peg1'] = ['']*nr_reset
        self.proc_data_dict['pee1'] = ['']*nr_reset
        for i in range(nr_readout):
            ce0 = np.count_nonzero(self.proc_data_dict['shots_0_dig'][i])
            ce1 = np.count_nonzero(self.proc_data_dict['shots_1_dig'][i])
            cg0 = np.count_nonzero(np.logical_not(self.proc_data_dict['shots_0_dig'][i]))
            cg1 = np.count_nonzero(np.logical_not(self.proc_data_dict['shots_1_dig'][i]))
            self.proc_data_dict['pe0'][i] = ce0/(ce0 + cg0)
            self.proc_data_dict['pe1'][i] = ce1/(ce1 + cg1)
            self.proc_data_dict['pg0'][i] = cg0/(ce0 + cg0)
            self.proc_data_dict['pg1'][i] = cg1/(ce1 + cg1)
            if i < nr_readout - 1:
                cgg0 = np.count_nonzero(~self.proc_data_dict['shots_0_dig'][i] *
                                        ~self.proc_data_dict['shots_0_dig'][i+1])
                cge0 = np.count_nonzero(~self.proc_data_dict['shots_0_dig'][i] *
                                        self.proc_data_dict['shots_0_dig'][i+1])
                ceg0 = np.count_nonzero(self.proc_data_dict['shots_0_dig'][i] *
                                        ~self.proc_data_dict['shots_0_dig'][i+1])
                cee0 = np.count_nonzero(self.proc_data_dict['shots_0_dig'][i] *
                                        self.proc_data_dict['shots_0_dig'][i+1])
                cgg1 = np.count_nonzero(~self.proc_data_dict['shots_1_dig'][i] *
                                        ~self.proc_data_dict['shots_1_dig'][i+1])
                cge1 = np.count_nonzero(~self.proc_data_dict['shots_1_dig'][i] *
                                        self.proc_data_dict['shots_1_dig'][i+1])
                ceg1 = np.count_nonzero(self.proc_data_dict['shots_1_dig'][i] *
                                        ~self.proc_data_dict['shots_1_dig'][i+1])
                cee1 = np.count_nonzero(self.proc_data_dict['shots_1_dig'][i] *
                                        self.proc_data_dict['shots_1_dig'][i+1])
                self.proc_data_dict['pgg0'][i] = cgg0/(cgg0 + cge0 + ceg0 + cee0)
                self.proc_data_dict['pge0'][i] = cge0/(cgg0 + cge0 + ceg0 + cee0)
                self.proc_data_dict['peg0'][i] = ceg0/(cgg0 + cge0 + ceg0 + cee0)
                self.proc_data_dict['pee0'][i] = cee0/(cgg0 + cge0 + ceg0 + cee0)
                self.proc_data_dict['pgg1'][i] = cgg1/(cgg1 + cge1 + ceg1 + cee1)
                self.proc_data_dict['pge1'][i] = cge1/(cgg1 + cge1 + ceg1 + cee1)
                self.proc_data_dict['peg1'][i] = ceg1/(cgg1 + cge1 + ceg1 + cee1)
                self.proc_data_dict['pee1'][i] = cee1/(cgg1 + cge1 + ceg1 + cee1)

        ###############################################
        # Extract readout result vector probabilities #
        ###############################################
        nr_readout_analysis = self.options_dict.get('nr_analysis_readouts', False)
        if nr_readout_analysis:
            self.proc_data_dict['result_vector_count_0'] = np.zeros(2**nr_readout_analysis)
            self.proc_data_dict['result_vector_count_1'] = np.zeros(2**nr_readout_analysis)
            for i in range(len(self.proc_data_dict['shots_0'][0])):
                result_index_0 = 0
                result_index_1 = 0
                for j in range(nr_readout_analysis):
                    if self.proc_data_dict['shots_0'][j][i] >= self.options_dict['threshold']:
                        result_index_0 += 1 << j
                    if self.proc_data_dict['shots_1'][j][i] >= self.options_dict['threshold']:
                        result_index_1 += 1 << j
                self.proc_data_dict['result_vector_count_0'][result_index_0] += 1
                self.proc_data_dict['result_vector_count_1'][result_index_1] += 1

        ##################################################
        # Extract transfer matrices for each reset event #
        ##################################################
        self.proc_data_dict['p_g->g'] = [['']*nr_reset, ['']*nr_reset]
        self.proc_data_dict['p_g->e'] = [['']*nr_reset, ['']*nr_reset]
        self.proc_data_dict['p_e->g'] = [['']*nr_reset, ['']*nr_reset]
        self.proc_data_dict['p_e->e'] = [['']*nr_reset, ['']*nr_reset]
        for i in range(nr_reset):
            for init in range(2):
                cgg = self.proc_data_dict['pgg{}'.format(init)][i]
                cge = self.proc_data_dict['pge{}'.format(init)][i]
                ceg = self.proc_data_dict['peg{}'.format(init)][i]
                cee = self.proc_data_dict['pee{}'.format(init)][i]
                self.proc_data_dict['p_g->g'][init][i] = cgg/(cgg + cge)
                self.proc_data_dict['p_g->e'][init][i] = cge/(cgg + cge)
                self.proc_data_dict['p_e->g'][init][i] = ceg/(ceg + cee)
                self.proc_data_dict['p_e->e'][init][i] = cee/(ceg + cee)

    def prepare_plots(self):
        nr_reset = self.options_dict.get('nr_reset')
        nr_readout = nr_reset + 1
        qubit_idx = self.options_dict.get('qubit_idx')
        
        # readout histograms
        for i in range(nr_readout):
            self.plot_dicts['ro_{}_hist_qb_idx{}'.format(i+1, qubit_idx)] = {
                'title': 'Readout {} histogram'.format(i+1),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bin_centers'],
                'yvals': self.proc_data_dict['hist_0'][i],
                'xlabel': 'Readout signal',
                'xunit': self.raw_data_dict['value_units'][0],
                'ylabel': 'Counts',
                'setlabel': r'Prepared 0',
                'linestyle': '',
                'line_kws': {'color': 'C0'},
                'marker': 'o',
                'do_legend': True}
            self.plot_dicts['hist_1_{}'.format(i)] = {
                'ax_id': 'ro_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bin_centers'],
                'yvals': self.proc_data_dict['hist_1'][i],
                'setlabel': r'Prepared $\pi$',
                'linestyle': '',
                'line_kws': {'color': 'C3'},
                'marker': 'o'}
            max_cnts = max(self.proc_data_dict['hist_1'][i].max(),
                           self.proc_data_dict['hist_0'][i].max())
            self.plot_dicts['ro_threshold_{}'.format(i)] = {
                'ax_id': 'ro_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'x': self.options_dict['threshold'],
                'ymin': 0,
                'ymax': max_cnts*1.05,
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8},
                'setlabel': 'Threshold',
                'do_legend': True}
            self.plot_dicts['prob_0_{}'.format(i)] = {
                'ax_id': 'ro_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_line,
                'xvals': [self.options_dict['threshold']],
                'yvals': [max_cnts/2],
                'line_kws': {'alpha': 0},
                'setlabel': r'$p(e|0) = {:.1f}$'.format(100*self.proc_data_dict['pe0'][i]),
                'do_legend': True}
            self.plot_dicts['prob_1_{}'.format(i)] = {
                'ax_id': 'ro_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_line,
                'xvals': [self.options_dict['threshold']],
                'yvals': [max_cnts/2],
                'line_kws': {'alpha': 0},
                'setlabel': r'$p(e|\pi) = {:.1f}$'.format(100*self.proc_data_dict['pe1'][i]),
                'do_legend': True}

        # reset histograms
        for i in range(nr_reset):
            hist2D = self.proc_data_dict['hist2_0'][i] + self.proc_data_dict['hist2_1'][i]
            self.plot_dicts['reset_{}_hist_qb_idx{}'.format(i+1, qubit_idx)] = {
                'title': 'Reset {} histogram'.format(i+1),
                'plotfn': self.plot_colorxy,
                'xvals': self.proc_data_dict['bin_centers'],
                'yvals': self.proc_data_dict['bin_centers'],
                'zvals': hist2D.T,
                'xlabel': 'Readout signal {}'.format(i+1),
                'xunit': self.raw_data_dict['value_units'][0],
                #'xrange': (-2.5e-3, -2.0e-3),
                'ylabel': 'Readout signal {}'.format(i+2),
                'yunit': self.raw_data_dict['value_units'][0],
                'zrange': (0, np.log10(hist2D.max())),
                'logzscale': True,
                'clabel': 'log10(counts)'}
            self.plot_dicts['reset_{}_hist_vline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}
            self.plot_dicts['reset_{}_hist_hline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'func': 'hlines',
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}
            self.plot_dicts['reset_{}_hist_g_qb_idx{}'.format(i+1, qubit_idx)] = {
                'title': 'Reset {} histogram, ground state preparation'.format(i+1),
                'plotfn': self.plot_colorxy,
                'xvals': self.proc_data_dict['bin_centers'],
                'yvals': self.proc_data_dict['bin_centers'],
                'zvals': self.proc_data_dict['hist2_0'][i].T,
                'xlabel': 'Readout signal {}'.format(i+1),
                'xunit': self.raw_data_dict['value_units'][0],
                'ylabel': 'Readout signal {}'.format(i+2),
                'yunit': self.raw_data_dict['value_units'][0],
                'zrange': (0, np.log10(hist2D.max())),
                'logzscale': True,
                'zlabel': 'log10(counts)'}
            self.plot_dicts['reset_{}_hist_g_vline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_g_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}
            self.plot_dicts['reset_{}_hist_g_hline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_g_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'func': 'hlines',
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}
            self.plot_dicts['reset_{}_hist_e_qb_idx{}'.format(i+1, qubit_idx)] = {
                'title': 'Reset {} histogram, excited state preparation'.format(i+1),
                'plotfn': self.plot_colorxy,
                'xvals': self.proc_data_dict['bin_centers'],
                'yvals': self.proc_data_dict['bin_centers'],
                'zvals': self.proc_data_dict['hist2_1'][i].T,
                'xlabel': 'Readout signal {}'.format(i+1),
                'xunit': self.raw_data_dict['value_units'][0],
                'ylabel': 'Readout signal {}'.format(i+2),
                'yunit': self.raw_data_dict['value_units'][0],
                'zrange': (0, np.log10(hist2D.max())),
                'logzscale': True,
                'zlabel': 'log10(counts)'}
            self.plot_dicts['reset_{}_hist_e_vline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_e_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}
            self.plot_dicts['reset_{}_hist_e_hline'.format(i+1)] = {
                'ax_id': 'reset_{}_hist_e_qb_idx{}'.format(i+1, qubit_idx),
                'plotfn': self.plot_vlines,
                'func': 'hlines',
                'x': self.options_dict['threshold'],
                'ymin': self.proc_data_dict['bin_edges'].min(),
                'ymax': self.proc_data_dict['bin_edges'].max(),
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8}}