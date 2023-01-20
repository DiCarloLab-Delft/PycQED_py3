"""
File containing analyses for single qubit readout.
"""
import itertools
from copy import deepcopy
import os

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
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
import matplotlib.patches as patches

import pathlib
from copy import copy, deepcopy
from typing import List
from itertools import repeat
from warnings import warn

import xarray as xr
from scipy import optimize as opt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pycqed.utilities.general import int2base
from pycqed.utilities.general import format_value_string

# This analysis is deprecated
class Singleshot_Readout_Analysis_old(ba.BaseDataAnalysis):

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
                self.plot_dicts['2D_shots_marker_line_0']={
                    'plotfn': self.plot_line,
                    'ax_id': '2D_shots',
                    'xvals': [0, iq_centers[0][0]],
                    'yvals': [0, iq_centers[1][0]],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': '',
                    'aspect': 'equal',
                    'linestyle': '--',
                    'color': 'black'
                }
                self.plot_dicts['2D_shots_marker_line_1']={
                    'plotfn': self.plot_line,
                    'ax_id': '2D_shots',
                    'xvals': [0, iq_centers[0][1]],
                    'yvals': [0, iq_centers[1][1]],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': '',
                    'aspect': 'equal',
                    'linestyle': '--',
                    'color': 'black'
                }

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


def _decision_boundary_points(coefs, intercepts):
    '''
    Find points along the decision boundaries of 
    LinearDiscriminantAnalysis (LDA).
    This is performed by finding the interception
    of the bounds of LDA. For LDA, these bounds are
    encoded in the coef_ and intercept_ parameters
    of the classifier.
    Each bound <i> is given by the equation:
    y + coef_i[0]/coef_i[1]*x + intercept_i = 0
    Note this only works for LinearDiscriminantAnalysis.
    Other classifiers might have diferent bound models.
    '''
    points = {}
    # Cycle through model coeficients
    # and intercepts.
    n = len(intercepts)
    if n == 3:
        _bounds = [[0,1], [1,2], [0,2]]
    if n == 4:
        _bounds = [[0,1], [1,2], [2,3], [3,0]]
    for i, j in _bounds:
        c_i = coefs[i]
        int_i = intercepts[i]
        c_j = coefs[j]
        int_j = intercepts[j]
        x =  (- int_j/c_j[1] + int_i/c_i[1])/(-c_i[0]/c_i[1] + c_j[0]/c_j[1])
        y = -c_i[0]/c_i[1]*x - int_i/c_i[1]
        points[f'{i}{j}'] = (x, y)
    # Find mean point
    points['mean'] = np.mean([ [x, y] for (x, y) in points.values()], axis=0)
    return points

class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for single-shot readout experiment
    updated in September 2022 (Jorge).
    This new analysis now supports post-selection
    with two quadratures and 3 state readout.
    """
    def __init__(self,
                 qubit: str,
                 qubit_freq: float,
                 heralded_init: bool,
                 f_state: bool = False,
                 h_state: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.heralded_init = heralded_init
        self.qubit_freq = qubit_freq
        self.f_state = f_state
        self.h_state = h_state

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        # Perform measurement post-selection
        _cycle = 2
        if self.f_state:
            _cycle += 1
        if self.h_state:
            _cycle += 1
        if self.heralded_init:
            _cycle *= 2
        ############################################
        # Rotate shots when data in two quadratures
        ############################################
        if self.raw_data_dict['data'].shape[1] == 3:
            # Sort shots
            _raw_shots = self.raw_data_dict['data'][:,1:]
            if self.heralded_init:
                _shots_0 = _raw_shots[1::_cycle]
                _shots_1 = _raw_shots[3::_cycle]
                if self.f_state:
                    _shots_2 = _raw_shots[5::_cycle]
                    self.proc_data_dict['shots_2_IQ'] = _shots_2
                    if self.h_state:
                        _shots_3 = _raw_shots[7::_cycle]
                        self.proc_data_dict['shots_3_IQ'] = _shots_3
            else:
                _shots_0 = _raw_shots[0::_cycle]
                _shots_1 = _raw_shots[1::_cycle]
                if self.f_state:
                    _shots_2 = _raw_shots[2::_cycle]
                    self.proc_data_dict['shots_2_IQ'] = _shots_2
                    if self.h_state:
                        _shots_3 = _raw_shots[3::_cycle]
                        self.proc_data_dict['shots_3_IQ'] = _shots_3
            # Save raw shots
            self.proc_data_dict['shots_0_IQ'] = _shots_0
            self.proc_data_dict['shots_1_IQ'] = _shots_1
            # Rotate data along 01
            center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
            center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
            def rotate_and_center_data(I, Q, vec0, vec1, phi=0):
                vector = vec1-vec0
                angle = np.arctan(vector[1]/vector[0])
                rot_matrix = np.array([[ np.cos(-angle+phi),-np.sin(-angle+phi)],
                                       [ np.sin(-angle+phi), np.cos(-angle+phi)]])
                proc = np.array((I, Q))
                proc = np.dot(rot_matrix, proc)
                return proc.transpose()
            raw_shots = rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        else:
            # Remove shot number
            raw_shots = self.raw_data_dict['data'][:,1:]
        #####################################################
        # From this point onward raw shots has shape 
        # (nr_shots, nr_quadratures).
        # Post select based on heralding measurement result.
        #####################################################
        if self.heralded_init:
            # estimate post-selection threshold
            shots_0 = raw_shots[1::_cycle, 0]
            shots_1 = raw_shots[3::_cycle, 0]
            ps_th = (np.mean(shots_0)+np.mean(shots_1))/2
            # Sort heralding shots from experiment shots
            ps_shots = raw_shots[0::2,0] # only I quadrature needed for postselection
            exp_shots = raw_shots[1::2] # Here we want to keep both quadratures
            # create post-selection mask
            _mask = [ 1 if s<ps_th else np.nan for s in ps_shots ]
            for i, s in enumerate(_mask):
                exp_shots[i] *= s
            # Remove marked shots
            Shots_0 = exp_shots[0::int(_cycle/2)]
            Shots_1 = exp_shots[1::int(_cycle/2)]
            Shots_0 = Shots_0[~np.isnan(Shots_0[:,0])]
            Shots_1 = Shots_1[~np.isnan(Shots_1[:,0])]
            if self.f_state:
                Shots_2 = exp_shots[2::int(_cycle/2)]
                Shots_2 = Shots_2[~np.isnan(Shots_2[:,0])]
                if self.h_state:
                    Shots_3 = exp_shots[3::int(_cycle/2)]
                    Shots_3 = Shots_3[~np.isnan(Shots_3[:,0])]
        else:
            # Sort 0 and 1 shots
            Shots_0 = raw_shots[0::_cycle]
            Shots_1 = raw_shots[1::_cycle]
            if self.f_state:
                Shots_2 = raw_shots[2::_cycle]
                if self.h_state:
                    Shots_3 = raw_shots[3::_cycle]
        self.proc_data_dict['Shots_0'] = Shots_0
        self.proc_data_dict['Shots_1'] = Shots_1
        if self.f_state:
            self.proc_data_dict['Shots_2'] = Shots_2
            if self.h_state:
                self.proc_data_dict['Shots_3'] = Shots_3
        ##############################################################
        # From this point onward Shots_<i> contains post-selected
        # shots of state <i> and has shape (nr_ps_shots, nr_quadtrs).
        # Next we will analyze shots projected along axis and 
        # therefore use a single quadrature. shots_<i> will be used
        # to denote that array of shots.
        ##############################################################
        # Analyse data in quadrature of interest
        # (01 projection axis)
        ##############################################################
        shots_0 = Shots_0[:,0]
        shots_1 = Shots_1[:,0]
        # total number of shots (after postselection)
        n_shots_0 = len(shots_0)
        n_shots_1 = len(shots_1)
        # find range
        _all_shots = np.concatenate((shots_0, shots_1))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x0, n0 = np.unique(shots_0, return_counts=True)
        x1, n1 = np.unique(shots_1, return_counts=True)
        # Calculate fidelity and optimal threshold
        def _calculate_fid_and_threshold(x0, n0, x1, n1):
            """
            Calculate fidelity and threshold from histogram data:
            x0, n0 is the histogram data of shots 0 (value and occurences),
            x1, n1 is the histogram data of shots 1 (value and occurences).
            """
            # Build cumulative histograms of shots 0 
            # and 1 in common bins by interpolation.
            all_x = np.unique(np.sort(np.concatenate((x0, x1))))
            cumsum0, cumsum1 = np.cumsum(n0), np.cumsum(n1)
            ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
            necumsum0 = ecumsum0/np.max(ecumsum0)
            ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
            necumsum1 = ecumsum1/np.max(ecumsum1)
            # Calculate optimal threshold and fidelity
            F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
            opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
            opt_idx = int(round(np.average(opt_idxs)))
            F_assignment_raw = F_vs_th[opt_idx]
            threshold_raw = all_x[opt_idx]
            return F_assignment_raw, threshold_raw
        Fid_raw, threshold_raw = _calculate_fid_and_threshold(x0, n0, x1, n1)
        ######################
        # Fit data
        ######################
        def _fit_double_gauss(x_vals, hist_0, hist_1):
            '''
            Fit two histograms to a double gaussian with
            common parameters. From fitted parameters,
            calculate SNR, Pe0, Pg1, Teff, Ffit and Fdiscr.
            '''
            from scipy.optimize import curve_fit
            # Double gaussian model for fitting
            def _gauss_pdf(x, x0, sigma):
                return np.exp(-((x-x0)/sigma)**2/2)
            global double_gauss
            def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
                _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
                return _dist0
            # helper function to simultaneously fit both histograms with common parameters
            def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
                _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
                _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
                return np.concatenate((_dist0, _dist1))
            # Guess for fit
            pdf_0 = hist_0/np.sum(hist_0) # Get prob. distribution
            pdf_1 = hist_1/np.sum(hist_1) # 
            _x0_guess = np.sum(x_vals*pdf_0) # calculate mean
            _x1_guess = np.sum(x_vals*pdf_1) #
            _sigma0_guess = np.sqrt(np.sum((x_vals-_x0_guess)**2*pdf_0)) # calculate std
            _sigma1_guess = np.sqrt(np.sum((x_vals-_x1_guess)**2*pdf_1)) #
            _r0_guess = 0.01
            _r1_guess = 0.05
            _A0_guess = np.max(hist_0)
            _A1_guess = np.max(hist_1)
            p0 = [_x0_guess, _x1_guess, _sigma0_guess, _sigma1_guess, _A0_guess, _A1_guess, _r0_guess, _r1_guess]
            # Bounding parameters
            _x0_bound = (-np.inf,np.inf)
            _x1_bound = (-np.inf,np.inf)
            _sigma0_bound = (0,np.inf)
            _sigma1_bound = (0,np.inf)
            _r0_bound = (0,1)
            _r1_bound = (0,1)
            _A0_bound = (0,np.inf)
            _A1_bound = (0,np.inf)
            bounds = np.array([_x0_bound, _x1_bound, _sigma0_bound, _sigma1_bound, _A0_bound, _A1_bound, _r0_bound, _r1_bound])
            # Fit parameters within bounds
            popt, pcov = curve_fit(
                _double_gauss_joint, bin_centers,
                np.concatenate((hist_0, hist_1)),
                p0=p0, bounds=bounds.transpose())
            popt0 = popt[[0,1,2,3,4,6]]
            popt1 = popt[[1,0,3,2,5,7]]
            # Calculate quantities of interest
            SNR = abs(popt0[0] - popt1[0])/((abs(popt0[2])+abs(popt1[2]))/2)
            P_e0 = popt0[5]*popt0[2]/(popt0[2]*popt0[5] + popt0[3]*(1-popt0[5]))
            P_g1 = popt1[5]*popt1[2]/(popt1[2]*popt1[5] + popt1[3]*(1-popt1[5]))
            # Effective qubit temperature
            h = 6.62607004e-34
            kb = 1.38064852e-23
            T_eff = h*self.qubit_freq/(kb*np.log((1-P_e0)/P_e0))
            # Fidelity from fit
            _x_data = np.linspace(*_range, 10001)
            _h0 = double_gauss(_x_data, *popt0)# compute distrubition from
            _h1 = double_gauss(_x_data, *popt1)# fitted parameters.
            Fid_fit, threshold_fit = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
            # Discrimination fidelity
            _h0 = double_gauss(_x_data, *popt0[:-1], 0)# compute distrubition without residual
            _h1 = double_gauss(_x_data, *popt1[:-1], 0)# excitation of relaxation.
            Fid_discr, threshold_discr = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
            # return results
            qoi = { 'SNR': SNR,
                    'P_e0': P_e0, 'P_g1': P_g1, 
                    'T_eff': T_eff, 
                    'Fid_fit': Fid_fit, 'Fid_discr': Fid_discr }
            return popt0, popt1, qoi
        # Histogram of shots for 0 and 1
        h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
        # Save data in processed data dictionary
        self.proc_data_dict['n_shots_0'] = n_shots_0
        self.proc_data_dict['n_shots_1'] = n_shots_1
        self.proc_data_dict['bin_centers'] = bin_centers
        self.proc_data_dict['h0'] = h0
        self.proc_data_dict['h1'] = h1
        self.proc_data_dict['popt0'] = popt0
        self.proc_data_dict['popt1'] = popt1
        self.proc_data_dict['threshold_raw'] = threshold_raw
        self.proc_data_dict['F_assignment_raw'] = Fid_raw
        self.proc_data_dict['F_fit'] = params_01['Fid_fit']
        self.proc_data_dict['F_discr'] = params_01['Fid_discr']
        self.proc_data_dict['residual_excitation'] = params_01['P_e0']
        self.proc_data_dict['relaxation_events'] = params_01['P_g1']
        self.proc_data_dict['effective_temperature'] = params_01['T_eff']
        # Save quantities of interest
        self.qoi = {}
        self.qoi['SNR'] = params_01['SNR']
        self.qoi['F_a'] = Fid_raw
        self.qoi['F_d'] = params_01['Fid_discr']
        self.proc_data_dict['quantities_of_interest'] = self.qoi
        ############################################
        # If second state data is use classifier
        # to assign states in the IQ plane and 
        # calculate qutrit fidelity.
        ############################################
        if self.f_state:
            # Parse data for classifier
            data = np.concatenate((Shots_0, Shots_1, Shots_2))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            clf.fit(data, labels)
            dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2'],
                                    [Shots_0, Shots_1, Shots_2]):
                _res = clf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((3,3))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
                for j, state in enumerate(['0', '1', '2']):
                    _res = clf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            self.proc_data_dict['classifier'] = clf
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.proc_data_dict['Fid_dict'] = Fid_dict
            self.qoi['Fid_dict'] = Fid_dict
            self.qoi['Assignment_matrix'] = M
            #########################################
            # Project data along axis perpendicular
            # to the decision boundaries.
            #########################################
            ############################
            # Projection along 10 axis.
            ############################
            # Rotate shots over 01 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
            shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_1 = shots_1[:,0]
            n_shots_1 = len(shots_1)
            # find range
            _all_shots = np.concatenate((shots_0, shots_1))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x1, n1 = np.unique(shots_1, return_counts=True)
            Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
            # Save processed data
            self.proc_data_dict['projection_01'] = {}
            self.proc_data_dict['projection_01']['h0'] = h0
            self.proc_data_dict['projection_01']['h1'] = h1
            self.proc_data_dict['projection_01']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_01']['popt0'] = popt0
            self.proc_data_dict['projection_01']['popt1'] = popt1
            self.proc_data_dict['projection_01']['SNR'] = params_01['SNR']
            self.proc_data_dict['projection_01']['Fid'] = Fid_01
            self.proc_data_dict['projection_01']['threshold'] = threshold_01
            ############################
            # Projection along 12 axis.
            ############################
            # Rotate shots over 12 decision boundary axis
            shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            # Take relavant quadrature
            shots_1 = shots_1[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_1, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x1, n1 = np.unique(shots_1, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
            # Histogram of shots for 1 and 2
            h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
            # Save processed data
            self.proc_data_dict['projection_12'] = {}
            self.proc_data_dict['projection_12']['h1'] = h1
            self.proc_data_dict['projection_12']['h2'] = h2
            self.proc_data_dict['projection_12']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_12']['popt1'] = popt1
            self.proc_data_dict['projection_12']['popt2'] = popt2
            self.proc_data_dict['projection_12']['SNR'] = params_12['SNR']
            self.proc_data_dict['projection_12']['Fid'] = Fid_12
            self.proc_data_dict['projection_12']['threshold'] = threshold_12
            ############################
            # Projection along 02 axis.
            ############################
            # Rotate shots over 02 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_0, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
            # Save processed data
            self.proc_data_dict['projection_02'] = {}
            self.proc_data_dict['projection_02']['h0'] = h0
            self.proc_data_dict['projection_02']['h2'] = h2
            self.proc_data_dict['projection_02']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_02']['popt0'] = popt0
            self.proc_data_dict['projection_02']['popt2'] = popt2
            self.proc_data_dict['projection_02']['SNR'] = params_02['SNR']
            self.proc_data_dict['projection_02']['Fid'] = Fid_02
            self.proc_data_dict['projection_02']['threshold'] = threshold_02

            if self.h_state: 
                # Parse data for classifier
                data = np.concatenate((Shots_0, Shots_1, Shots_2, Shots_3))
                labels = [0 for s in Shots_0]+[1 for s in Shots_1]+\
                         [2 for s in Shots_2]+[3 for s in Shots_3]
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf = LinearDiscriminantAnalysis()
                clf.fit(data, labels)
                dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
                # dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
                Fid_dict = {}
                for state, shots in zip([    '0',     '1',     '2',     '3'],
                                        [Shots_0, Shots_1, Shots_2, Shots_3]):
                    _res = clf.predict(shots)
                    _fid = np.mean(_res == int(state))
                    Fid_dict[state] = _fid
                Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
                # Get assignment fidelity matrix
                M = np.zeros((4,4))
                for i, shots in enumerate([Shots_0, Shots_1, Shots_2, Shots_3]):
                    for j, state in enumerate(['0', '1', '2', '3']):
                        _res = clf.predict(shots)
                        M[i][j] = np.mean(_res == int(state))
                self.proc_data_dict['h_classifier'] = clf
                self.proc_data_dict['h_dec_bounds'] = dec_bounds
                self.proc_data_dict['h_Fid_dict'] = Fid_dict
                self.qoi['h_Fid_dict'] = Fid_dict
                self.qoi['h_Assignment_matrix'] = M

    def prepare_plots(self):
        self.axs_dict = {}
        fig, ax = plt.subplots(figsize=(5,4), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['main'] = ax
        self.figs['main'] = fig
        self.plot_dicts['main'] = {
            'plotfn': ssro_hist_plotfn,
            'ax_id': 'main',
            'bin_centers': self.proc_data_dict['bin_centers'],
            'h0': self.proc_data_dict['h0'],
            'h1': self.proc_data_dict['h1'],
            'popt0': self.proc_data_dict['popt0'], 
            'popt1': self.proc_data_dict['popt1'],
            'threshold': self.proc_data_dict['threshold_raw'],
            'Fid_raw': self.qoi['F_a'],
            'Fid_fit': self.proc_data_dict['F_fit'],
            'Fid_disc': self.qoi['F_d'],
            'SNR': self.qoi['SNR'],
            'P_e0': self.proc_data_dict['residual_excitation'], 
            'P_g1': self.proc_data_dict['relaxation_events'],
            'n_shots_0': self.proc_data_dict['n_shots_0'],
            'n_shots_1': self.proc_data_dict['n_shots_1'],
            'T_eff': self.proc_data_dict['effective_temperature'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        if self.raw_data_dict['data'].shape[1] == 3:
            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict['main2'] = ax
            self.figs['main2'] = fig
            self.plot_dicts['main2'] = {
                'plotfn': ssro_IQ_plotfn,
                'ax_id': 'main2',
                'shots_0': self.proc_data_dict['shots_0_IQ'],
                'shots_1': self.proc_data_dict['shots_1_IQ'],
                'shots_2': self.proc_data_dict['shots_2_IQ'] if self.f_state else None,
                'shots_3': self.proc_data_dict['shots_3_IQ'] if self.h_state else None,
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }
            if self.f_state:
                fig = plt.figure(figsize=(8,4), dpi=100)
                axs = [fig.add_subplot(121),
                       fig.add_subplot(322),
                       fig.add_subplot(324),
                       fig.add_subplot(326)]
                # fig.patch.set_alpha(0)
                self.axs_dict['main3'] = axs[0]
                self.figs['main3'] = fig
                self.plot_dicts['main3'] = {
                    'plotfn': ssro_IQ_projection_plotfn,
                    'ax_id': 'main3',
                    'shots_0': self.proc_data_dict['Shots_0'],
                    'shots_1': self.proc_data_dict['Shots_1'],
                    'shots_2': self.proc_data_dict['Shots_2'],
                    'projection_01': self.proc_data_dict['projection_01'],
                    'projection_12': self.proc_data_dict['projection_12'],
                    'projection_02': self.proc_data_dict['projection_02'],
                    'classifier': self.proc_data_dict['classifier'],
                    'dec_bounds': self.proc_data_dict['dec_bounds'],
                    'Fid_dict': self.proc_data_dict['Fid_dict'],
                    'qubit': self.qubit,
                    'timestamp': self.timestamp
                }
                fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                # fig.patch.set_alpha(0)
                self.axs_dict['Assignment_matrix'] = ax
                self.figs['Assignment_matrix'] = fig
                self.plot_dicts['Assignment_matrix'] = {
                    'plotfn': assignment_matrix_plotfn,
                    'ax_id': 'Assignment_matrix',
                    'M': self.qoi['Assignment_matrix'],
                    'qubit': self.qubit,
                    'timestamp': self.timestamp
                }
                if self.h_state:
                    fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                    # fig.patch.set_alpha(0)
                    self.axs_dict['Assignment_matrix_h'] = ax
                    self.figs['Assignment_matrix_h'] = fig
                    self.plot_dicts['Assignment_matrix_h'] = {
                        'plotfn': assignment_matrix_plotfn,
                        'ax_id': 'Assignment_matrix_h',
                        'M': self.qoi['h_Assignment_matrix'],
                        'qubit': self.qubit,
                        'timestamp': self.timestamp
                    }
                    fig, ax = plt.subplots(figsize=(4,4), dpi=100)
                    # fig.patch.set_alpha(0)
                    self.axs_dict['main4'] = ax
                    self.figs['main4'] = fig
                    self.plot_dicts['main4'] = {
                        'plotfn': ssro_IQ_plotfn,
                        'ax_id': 'main4',
                        'shots_0': self.proc_data_dict['Shots_0'],
                        'shots_1': self.proc_data_dict['Shots_1'],
                        'shots_2': self.proc_data_dict['Shots_2'],
                        'shots_3': self.proc_data_dict['Shots_3'],
                        'qubit': self.qubit,
                        'timestamp': self.timestamp,
                        'dec_bounds': self.proc_data_dict['h_dec_bounds'],
                        'Fid_dict': self.proc_data_dict['h_Fid_dict'],
                    }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def ssro_hist_plotfn(
    bin_centers,
    h0, h1,
    popt0, popt1,
    threshold,
    qubit,
    timestamp,
    Fid_raw,
    Fid_fit,
    Fid_disc,
    SNR,
    P_e0, P_g1,
    n_shots_0,
    n_shots_1,
    ax,
    T_eff=None,
    **kw):
    fig = ax.get_figure()
    bin_width = bin_centers[1]-bin_centers[0]
    ax.bar(bin_centers, h0, bin_width, fc='C0', alpha=0.4)
    ax.bar(bin_centers, h1, bin_width, fc='C3', alpha=0.4)
    ax.plot(bin_centers, double_gauss(bin_centers, *popt0), '-C0', label='ground state')
    ax.plot(bin_centers, double_gauss(bin_centers, *popt1), '-C3', label='excited state')
    ax.axvline(threshold, ls='--', color='k', label='threshold')
    # Write results
    text = '\n'.join(('Fidelity and fit results:',
                      rf'$\mathrm{"{F_{assign}}"}:\:\:\:{Fid_raw*100:.2f}$%',
                      rf'$\mathrm{"{F_{fit}}"}:\:\:\:\:\:\:\:\:\:\:{Fid_fit*100:.2f}$%',
                      rf'$\mathrm{"{F_{discr}}"}:\:\:\:\:\:\:{Fid_disc*100:.2f}$%',
                      rf'$\mathrm{"{SNR}"}:\:\:\:\:\:\:\:{SNR:.2f}$',
                      '',
                      'Spurious events:',
                      rf'$P(e|0)={P_e0*100:.2f}$%',
                      rf'$P(g|\pi)={P_g1*100:.2f}$%',
                      '',
                      'Number of shots:',
                      f'$0$: {n_shots_0}\t$\pi$: {n_shots_1}',
                      ''))
    if T_eff:
        text += '\nEffective temperature:\n'+\
               f'$T_{"{qubit}"}$ : {T_eff*1e3:.0f} mK'
    props = dict(boxstyle='round', facecolor='gray', alpha=0.15)
    ax.text(1.05, 0.8, text, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1.05))
    # ax.set_yscale('log')
    ax.set_xlim(bin_centers[[0,-1]])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Integrated voltage (a.u.)')
    ax.set_ylabel('Number of shots')
    ax.set_title(f'{timestamp}\nHistogram of shots qubit {qubit}')

def ssro_IQ_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    shots_3,
    timestamp,
    qubit,
    ax, 
    dec_bounds=None,
    Fid_dict=None,
    **kw):
    fig = ax.get_figure()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    # Plot stuff
    ax.plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    ax.plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    ax.plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    ax.plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    ax.plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    ax.plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    ax.plot(popt_0[1], popt_0[2], 'x', color='white')
    ax.plot(popt_1[1], popt_1[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_1)
    _all_shots = np.concatenate((shots_0, shots_1))
    if type(shots_2) != type(None):
        popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
        ax.plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
        ax.plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
        ax.plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
        ax.plot(popt_2[1], popt_2[2], 'x', color='white')
        # Draw 4sigma ellipse around mean
        circle_2 = Ellipse((popt_2[1], popt_2[2]),
                          width=4*popt_2[3], height=4*popt_2[4],
                          angle=-popt_2[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        ax.add_patch(circle_2)
        _all_shots = np.concatenate((_all_shots, shots_2))
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
        ax.plot(shots_3[:,0], shots_3[:,1], '.', color='gold', alpha=0.05)
        ax.plot([0, popt_3[1]], [0, popt_3[2]], '--', color='k', lw=.5)
        ax.plot(popt_3[1], popt_3[2], '.', color='gold', label='$3^\mathrm{rd}$ excited')
        ax.plot(popt_3[1], popt_3[2], 'x', color='white')
        # Draw 4sigma ellipse around mean
        circle_3 = Ellipse((popt_3[1], popt_3[2]),
                          width=4*popt_3[3], height=4*popt_3[4],
                          angle=-popt_3[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        ax.add_patch(circle_3)
        _all_shots = np.concatenate((_all_shots, shots_3))

    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.legend(frameon=False)
    ax.set_xlabel('Integrated voltage I')
    ax.set_ylabel('Integrated voltage Q')
    ax.set_title(f'{timestamp}\nIQ plot qubit {qubit}')
    if dec_bounds:
        # Plot decision boundary
        _bounds = list(dec_bounds.keys())
        _bounds.remove('mean')
        Lim_points = {}
        for bound in _bounds:
            dec_bounds['mean']
            _x0, _y0 = dec_bounds['mean']
            _x1, _y1 = dec_bounds[bound]
            a = (_y1-_y0)/(_x1-_x0)
            b = _y0 - a*_x0
            _xlim = 1e2*np.sign(_x1-_x0)
            _ylim = a*_xlim + b
            Lim_points[bound] = _xlim, _ylim
        # Plot classifier zones
        from matplotlib.patches import Polygon
        # Plot 0 area
        _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['30']]
        _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        # Plot 1 area
        _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
        _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        # Plot 2 area
        _points = [dec_bounds['mean'], Lim_points['23'], Lim_points['12']]
        _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        if type(shots_3) != type(None):
            # Plot 3 area
            _points = [dec_bounds['mean'], Lim_points['23'], Lim_points['30']]
            _patch = Polygon(_points, color='gold', alpha=0.2, lw=0)
            ax.add_patch(_patch)
        for bound in _bounds:
            _x0, _y0 = dec_bounds['mean']
            _x1, _y1 = Lim_points[bound]
            ax.plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
        if Fid_dict:
            # Write fidelity textbox
            text = '\n'.join(('Assignment fidelity:',
                              f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                              f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                              f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                              f'$F_h$ : {Fid_dict["3"]*100:.1f}%',
                              f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.text(1.05, 1, text, transform=ax.transAxes,
                    verticalalignment='top', bbox=props)

def ssro_IQ_projection_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    projection_01,
    projection_12,
    projection_02,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'IQ plot qubit {qubit}')
    fig.suptitle(f'{timestamp}\n')
    ##########################
    # Plot projections
    ##########################
    # 01 projection
    _bin_c = projection_01['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[1].bar(_bin_c, projection_01['h0'], bin_width, fc='C0', alpha=0.4)
    axs[1].bar(_bin_c, projection_01['h1'], bin_width, fc='C3', alpha=0.4)
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt0']), '-C0')
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt1']), '-C3')
    axs[1].axvline(projection_01['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_01["Fid"]*100:.1f}%',
                      f'SNR : {projection_01["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[1].text(.775, .9, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[1].text(projection_01['popt0'][0], projection_01['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[1].text(projection_01['popt1'][0], projection_01['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[1].set_xticklabels([])
    axs[1].set_xlim(_bin_c[0], _bin_c[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Projection of data')
    # 12 projection
    _bin_c = projection_12['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[2].bar(_bin_c, projection_12['h1'], bin_width, fc='C3', alpha=0.4)
    axs[2].bar(_bin_c, projection_12['h2'], bin_width, fc='C2', alpha=0.4)
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt1']), '-C3')
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt2']), '-C2')
    axs[2].axvline(projection_12['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_12["Fid"]*100:.1f}%',
                      f'SNR : {projection_12["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[2].text(.775, .9, text, transform=axs[2].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[2].text(projection_12['popt1'][0], projection_12['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[2].text(projection_12['popt2'][0], projection_12['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[2].set_xticklabels([])
    axs[2].set_xlim(_bin_c[0], _bin_c[-1])
    axs[2].set_ylim(bottom=0)
    # 02 projection
    _bin_c = projection_02['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[3].bar(_bin_c, projection_02['h0'], bin_width, fc='C0', alpha=0.4)
    axs[3].bar(_bin_c, projection_02['h2'], bin_width, fc='C2', alpha=0.4)
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt0']), '-C0')
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt2']), '-C2')
    axs[3].axvline(projection_02['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_02["Fid"]*100:.1f}%',
                      f'SNR : {projection_02["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[3].text(projection_02['popt0'][0], projection_02['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[3].text(projection_02['popt2'][0], projection_02['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[3].set_xticklabels([])
    axs[3].set_xlim(_bin_c[0], _bin_c[-1])
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlabel('Integrated voltage')
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def assignment_matrix_plotfn(
    M,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    im = ax.imshow(M, cmap=plt.cm.Reds, vmin=0, vmax=1)
    n = len(M)
    for i in range(n):
        for j in range(n):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                             color = 'white')
            else:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f'$|{i}\\rangle$' for i in range(n)])
    ax.set_xlabel('Assigned state')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f'$|{i}\\rangle$' for i in range(n)])
    ax.set_ylabel('Prepared state')
    name = qubit
    if n==3:
        name = 'Qutrit'
    elif n==4:
        name = 'Ququat'
    ax.set_title(f'{timestamp}\n{name} assignment matrix qubit {qubit}')
    cbar_ax = fig.add_axes([.95, .15, .03, .7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('assignment probability')


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


class Optimal_integration_weights_analysis(ba.BaseDataAnalysis):
    """
    Mux transient analysis.
    """
    def __init__(self,
                 IF: float,
                 input_waveform: tuple = None,
                 t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.IF = IF
        self.input_waveform = input_waveform
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.raw_data_dict = {}
        for ts in self.timestamps:
            data_fp = get_datafilepath_from_timestamp(ts)
            param_spec = {'data': ('Experimental Data/Data', 'dset'),
                          'value_names': ('Experimental Data', 'attr:value_names')}
            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(
                data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        ts_off = self.timestamps[0]
        ts_on = self.timestamps[1]
        Time = self.raw_data_dict[ts_off]['data'][:,0]
        Trace_I_0 = self.raw_data_dict[ts_off]['data'][:,1]
        Trace_Q_0 = self.raw_data_dict[ts_off]['data'][:,2]
        Trace_I_1 = self.raw_data_dict[ts_on]['data'][:,1]
        Trace_Q_1 = self.raw_data_dict[ts_on]['data'][:,2]
        # Subtract offset
        _trace_I_0 = Trace_I_0 - np.mean(Trace_I_0)
        _trace_Q_0 = Trace_Q_0 - np.mean(Trace_Q_0)
        _trace_I_1 = Trace_I_1 - np.mean(Trace_I_1)
        _trace_Q_1 = Trace_Q_1 - np.mean(Trace_Q_1)
        # Demodulate traces
        def _demodulate(Time, I, Q, IF):
            Complex_vec = I + 1j*Q
            I_demod = np.real(np.exp(1j*2*np.pi*IF*Time)*Complex_vec)
            Q_demod = np.imag(np.exp(1j*2*np.pi*IF*Time)*Complex_vec)
            return I_demod, Q_demod
        Trace_I_0_demod, Trace_Q_0_demod = _demodulate(Time, _trace_I_0, _trace_Q_0, self.IF)
        Trace_I_1_demod, Trace_Q_1_demod = _demodulate(Time, _trace_I_1, _trace_Q_1, self.IF)

        # Calculate optimal weights
        Weights_I = _trace_I_1 - _trace_I_0
        Weights_Q = _trace_Q_1 - _trace_Q_0
        # joint rescaling to +/-1 Volt
        maxI = np.max(np.abs(Weights_I))
        maxQ = np.max(np.abs(Weights_Q))
        # Dividing the weight functions by four to not have overflow in
        # thresholding of the UHFQC
        weight_scale_factor = 1./(4*np.max([maxI, maxQ]))
        Weights_I = np.array(weight_scale_factor*Weights_I)
        Weights_Q = np.array(weight_scale_factor*Weights_Q)

        # Demodulate weights
        Weights_I_demod, Weights_Q_demod = _demodulate(Time, Weights_I, Weights_Q, self.IF)
        # Smooth weights
        from scipy.signal import medfilt
        Weights_I_demod_s = medfilt(Weights_I_demod, 31)
        Weights_Q_demod_s = medfilt(Weights_Q_demod, 31)
        Weights_I_s, Weights_Q_s = _demodulate(Time, Weights_I_demod_s, Weights_Q_demod_s, -self.IF)

        # PSD of output signal
        time_step = Time[1]
        ps_0 = np.abs(np.fft.fft(1j*_trace_I_0+_trace_Q_0))**2*time_step/len(Time)
        ps_1 = np.abs(np.fft.fft(1j*_trace_I_1+_trace_Q_1))**2*time_step/len(Time)
        Freqs = np.fft.fftfreq(_trace_I_0.size, time_step)
        idx = np.argsort(Freqs)
        Freqs = Freqs[idx]
        ps_0 = ps_0[idx]
        ps_1 = ps_1[idx]
        # PSD of input signal
        if type(self.input_waveform) != type(None):
            _n_tt = len(Time)
            _n_wf = len(self.input_waveform[0])
            in_wf_I = np.concatenate((self.input_waveform[0],
                                      np.zeros(_n_tt-_n_wf)))
            in_wf_Q = np.concatenate((self.input_waveform[1],
                                      np.zeros(_n_tt-_n_wf)))
            in_wf = 1j*in_wf_I + in_wf_Q
            ps_wf = np.abs(np.fft.fft(in_wf))**2*time_step/len(in_wf)
            Freqs_wf = np.fft.fftfreq(in_wf.size, time_step)
            idx_wf = np.argsort(Freqs_wf)
            Freqs_wf = Freqs_wf[idx_wf]
            ps_wf = ps_wf[idx_wf]
            # normalize (for plotting purposes)
            ps_wf = ps_wf/np.max(ps_wf)*max([np.max(ps_0),np.max(ps_1)])*1.1
            self.proc_data_dict['Freqs_wf'] = Freqs_wf
            self.proc_data_dict['ps_wf'] = ps_wf

        self.proc_data_dict['Time'] = Time
        self.proc_data_dict['Trace_I_0'] = Trace_I_0
        self.proc_data_dict['Trace_Q_0'] = Trace_Q_0
        self.proc_data_dict['Trace_I_1'] = Trace_I_1
        self.proc_data_dict['Trace_Q_1'] = Trace_Q_1
        self.proc_data_dict['Trace_I_0_demod'] = Trace_I_0_demod
        self.proc_data_dict['Trace_Q_0_demod'] = Trace_Q_0_demod
        self.proc_data_dict['Trace_I_1_demod'] = Trace_I_1_demod
        self.proc_data_dict['Trace_Q_1_demod'] = Trace_Q_1_demod
        self.proc_data_dict['Weights_I_demod'] = Weights_I_demod
        self.proc_data_dict['Weights_Q_demod'] = Weights_Q_demod
        self.proc_data_dict['Weights_I_demod_s'] = Weights_I_demod_s
        self.proc_data_dict['Weights_Q_demod_s'] = Weights_Q_demod_s
        self.proc_data_dict['Weights_I_s'] = Weights_I_s
        self.proc_data_dict['Weights_Q_s'] = Weights_Q_s
        self.proc_data_dict['Freqs'] = Freqs
        self.proc_data_dict['ps_0'] = ps_0
        self.proc_data_dict['ps_1'] = ps_1

        self.qoi = {}
        self.qoi['Weights_I_s'] = Weights_I_s
        self.qoi['Weights_Q_s'] = Weights_Q_s

        # If second state
        if len(self.timestamps) == 3:
            self.f_state = True
        else:
            self.f_state = False
        if self.f_state:
            ts_two = self.timestamps[2]
            Trace_I_2 = self.raw_data_dict[ts_two]['data'][:,1]
            Trace_Q_2 = self.raw_data_dict[ts_two]['data'][:,2]
            # Subtract offset
            _trace_I_2 = Trace_I_2 - np.mean(Trace_I_2)
            _trace_Q_2 = Trace_Q_2 - np.mean(Trace_Q_2)
            # Demodulate traces
            Trace_I_2_demod, Trace_Q_2_demod = _demodulate(Time, _trace_I_2,
                                                           _trace_Q_2, self.IF)
            # Calculate optimal weights
            Weights_I_ef = _trace_I_2 - _trace_I_1
            Weights_Q_ef = _trace_Q_2 - _trace_Q_1
            # joint rescaling to +/-1 Volt
            maxI = np.max(np.abs(Weights_I_ef))
            maxQ = np.max(np.abs(Weights_Q_ef))
            # Dividing the weight functions by four to not have overflow in
            # thresholding of the UHFQC
            weight_scale_factor = 1./(4*np.max([maxI, maxQ]))
            Weights_I_ef = np.array(weight_scale_factor*Weights_I_ef)
            Weights_Q_ef = np.array(weight_scale_factor*Weights_Q_ef)
            # Demodulate weights
            Weights_I_ef_demod, Weights_Q_ef_demod = _demodulate(Time, Weights_I_ef,
                                                                 Weights_Q_ef, self.IF)
            # Smooth weights
            from scipy.signal import medfilt
            Weights_I_ef_demod_s = medfilt(Weights_I_ef_demod, 31)
            Weights_Q_ef_demod_s = medfilt(Weights_Q_ef_demod, 31)
            Weights_I_ef_s, Weights_Q_ef_s = _demodulate(Time, Weights_I_ef_demod_s,
                                                         Weights_Q_ef_demod_s, -self.IF)
            # Save quantities
            self.proc_data_dict['Trace_I_2'] = Trace_I_2
            self.proc_data_dict['Trace_Q_2'] = Trace_Q_2
            self.proc_data_dict['Trace_I_2_demod'] = Trace_I_2_demod
            self.proc_data_dict['Trace_Q_2_demod'] = Trace_Q_2_demod
            self.proc_data_dict['Weights_I_ef_demod'] = Weights_I_ef_demod
            self.proc_data_dict['Weights_Q_ef_demod'] = Weights_Q_ef_demod
            self.proc_data_dict['Weights_I_ef_demod_s'] = Weights_I_ef_demod_s
            self.proc_data_dict['Weights_Q_ef_demod_s'] = Weights_Q_ef_demod_s
            self.proc_data_dict['Weights_I_ef_s'] = Weights_I_ef_s
            self.proc_data_dict['Weights_Q_ef_s'] = Weights_Q_ef_s
            self.qoi['Weights_I_ef_s'] = Weights_I_ef_s
            self.qoi['Weights_Q_ef_s'] = Weights_Q_ef_s

    def prepare_plots(self):

        self.axs_dict = {}
        n = len(self.timestamps)
        fig, axs = plt.subplots(figsize=(9.75/2*n, 5.2), nrows=2, ncols=n, sharex=True, sharey='row', dpi=100)
        axs = axs.flatten()
        # fig.patch.set_alpha(0)
        self.axs_dict['Transients_plot'] = axs[0]
        self.figs['Transients_plot'] = fig
        self.plot_dicts['Transients_plot'] = {
            'plotfn': Transients_plotfn,
            'ax_id': 'Transients_plot',
            'Time': self.proc_data_dict['Time'],
            'Trace_I_0': self.proc_data_dict['Trace_I_0'],
            'Trace_Q_0': self.proc_data_dict['Trace_Q_0'],
            'Trace_I_1': self.proc_data_dict['Trace_I_1'],
            'Trace_Q_1': self.proc_data_dict['Trace_Q_1'],
            'Trace_I_2': self.proc_data_dict['Trace_I_2'] if self.f_state else None,
            'Trace_Q_2': self.proc_data_dict['Trace_Q_2'] if self.f_state else None,
            'Trace_I_0_demod': self.proc_data_dict['Trace_I_0_demod'],
            'Trace_Q_0_demod': self.proc_data_dict['Trace_Q_0_demod'],
            'Trace_I_1_demod': self.proc_data_dict['Trace_I_1_demod'],
            'Trace_Q_1_demod': self.proc_data_dict['Trace_Q_1_demod'],
            'Trace_I_2_demod': self.proc_data_dict['Trace_I_2_demod'] if self.f_state else None,
            'Trace_Q_2_demod': self.proc_data_dict['Trace_Q_2_demod'] if self.f_state else None,
            'timestamp': self.timestamps[1]
        }

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['IQ_trajectory_plot'] = ax
        self.figs['IQ_trajectory_plot'] = fig
        self.plot_dicts['IQ_trajectory_plot'] = {
            'plotfn': IQ_plotfn,
            'ax_id': 'IQ_trajectory_plot',
            'Trace_I_0_demod': self.proc_data_dict['Trace_I_0_demod'],
            'Trace_Q_0_demod': self.proc_data_dict['Trace_Q_0_demod'],
            'Trace_I_1_demod': self.proc_data_dict['Trace_I_1_demod'],
            'Trace_Q_1_demod': self.proc_data_dict['Trace_Q_1_demod'],
            'Trace_I_2_demod': self.proc_data_dict['Trace_I_2_demod'] if self.f_state else None,
            'Trace_Q_2_demod': self.proc_data_dict['Trace_Q_2_demod'] if self.f_state else None,
            'timestamp': self.timestamps[1]
        }
        
        fig, axs = plt.subplots(figsize=(9*1.4, 3*1.4), ncols=2,
                gridspec_kw={'width_ratios': [5*1.4, 3*1.4]}, dpi=100)
        axs = axs.flatten()
        # fig.patch.set_alpha(0)
        self.axs_dict['Optimal_weights_plot'] = axs[0]
        self.figs['Optimal_weights_plot'] = fig
        self.plot_dicts['Optimal_weights_plot'] = {
            'plotfn': Weights_plotfn,
            'ax_id': 'Optimal_weights_plot',
            'Time': self.proc_data_dict['Time'],
            'Weights_I_demod': self.proc_data_dict['Weights_I_demod'],
            'Weights_Q_demod': self.proc_data_dict['Weights_Q_demod'],
            'Weights_I_demod_s': self.proc_data_dict['Weights_I_demod_s'],
            'Weights_Q_demod_s': self.proc_data_dict['Weights_Q_demod_s'],
            'Weights_I_ef_demod': self.proc_data_dict['Weights_I_ef_demod'] if self.f_state else None,
            'Weights_Q_ef_demod': self.proc_data_dict['Weights_Q_ef_demod'] if self.f_state else None,
            'Weights_I_ef_demod_s': self.proc_data_dict['Weights_I_ef_demod_s'] if self.f_state else None,
            'Weights_Q_ef_demod_s': self.proc_data_dict['Weights_Q_ef_demod_s'] if self.f_state else None,
            'timestamp': self.timestamps[1]
        }

        fig, axs = plt.subplots(figsize=(8,3), ncols=2, dpi=100,
                                sharey=True)
        axs = axs.flatten()
        # fig.patch.set_alpha(0)
        self.axs_dict['FFT_plot'] = axs[0]
        self.figs['FFT_plot'] = fig
        self.plot_dicts['FFT_plot'] = {
            'plotfn': FFT_plotfn,
            'ax_id': 'FFT_plot',
            'Freqs': self.proc_data_dict['Freqs'],
            'ps_0': self.proc_data_dict['ps_0'],
            'ps_1': self.proc_data_dict['ps_1'],
            'Freqs_wf': self.proc_data_dict['Freqs_wf'] if type(self.input_waveform)!=type(None) else None,
            'ps_wf': self.proc_data_dict['ps_wf'] if type(self.input_waveform)!=type(None) else None,
            'IF': self.IF,
            'timestamp': self.timestamps[1]
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Transients_plotfn(
    Time,
    Trace_I_0, Trace_Q_0,
    Trace_I_1, Trace_Q_1,
    Trace_I_2, Trace_Q_2,
    Trace_I_0_demod, Trace_Q_0_demod,
    Trace_I_1_demod, Trace_Q_1_demod,
    Trace_I_2_demod, Trace_Q_2_demod,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    if type(Trace_I_2) != type(None):
        n = 3
    else:
        n = 2
    axs[0].plot(Time*1e6, Trace_I_0, color='#82B1FF', ls='-', lw=1, label='In phase component')
    axs[n].plot(Time*1e6, Trace_Q_0, color='#82B1FF', ls='-', lw=1, label='Quadrature component')
    axs[1].plot(Time*1e6, Trace_I_1, color='#E57373', ls='-', lw=1, label='In phase component')
    axs[n+1].plot(Time*1e6, Trace_Q_1, color='#E57373', ls='-', lw=1, label='Quadrature component')
    axs[0].plot(Time*1e6, Trace_I_0_demod, color='#0D47A1', ls='-', lw=1)
    axs[n].plot(Time*1e6, Trace_Q_0_demod, color='#0D47A1', ls='-', lw=1)
    axs[1].plot(Time*1e6, Trace_I_1_demod, color='#C62828', ls='-', lw=1)
    axs[n+1].plot(Time*1e6, Trace_Q_1_demod, color='#C62828', ls='-', lw=1)
    if n == 3:
        axs[2].plot(Time*1e6, Trace_I_2, color='#A5D6A7', ls='-', lw=1, label='In phase component')
        axs[n+2].plot(Time*1e6, Trace_Q_2, color='#A5D6A7', ls='-', lw=1, label='Quadrature component')
        axs[2].plot(Time*1e6, Trace_I_2_demod, color='#2E7D32', ls='-', lw=1)
        axs[n+2].plot(Time*1e6, Trace_Q_2_demod, color='#2E7D32', ls='-', lw=1)
        axs[n+2].set_xlabel('Time ($\mathrm{\mu s}$)')
        axs[2].set_title(r'$2^\mathrm{nd}$ excited state')
        axs[2].legend(frameon=False, fontsize=9)
        axs[n+2].legend(frameon=False, fontsize=9)

    axs[n].set_xlabel('Time ($\mathrm{\mu s}$)')
    axs[n+1].set_xlabel('Time ($\mathrm{\mu s}$)')
    axs[0].set_ylabel('Voltage (V)')
    axs[n].set_ylabel('Voltage (V)')
    axs[0].set_title('Ground state')
    axs[1].set_title('Excited state')
    axs[0].legend(frameon=False, fontsize=9)
    axs[1].legend(frameon=False, fontsize=9)
    axs[n].legend(frameon=False, fontsize=9)
    axs[n+1].legend(frameon=False, fontsize=9)
    fig.suptitle(f'{timestamp}\nReadout transients', y=.95)
    fig.tight_layout()

def IQ_plotfn(
    Trace_I_0_demod, Trace_Q_0_demod,
    Trace_I_1_demod, Trace_Q_1_demod,
    Trace_I_2_demod, Trace_Q_2_demod,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    ax.plot(Trace_I_0_demod*1e3, Trace_Q_0_demod*1e3, color='#0D47A1', ls='-', lw=.5, label='ground')
    ax.plot(Trace_I_1_demod*1e3, Trace_Q_1_demod*1e3, color='#C62828', ls='-', lw=.5, label='excited')
    _lim = np.max(np.concatenate((np.abs(Trace_I_0_demod*1e3), np.abs(Trace_Q_0_demod*1e3),
                                  np.abs(Trace_I_1_demod*1e3), np.abs(Trace_Q_1_demod*1e3))))
    if type(Trace_I_2_demod) != type(None):
        ax.plot(Trace_I_2_demod*1e3, Trace_Q_2_demod*1e3, color='C2', ls='-', lw=.5, label='$2^{nd}$ excited')    
        _lim = np.max(np.concatenate((np.abs(Trace_I_0_demod*1e3), np.abs(Trace_Q_0_demod*1e3),
                                      np.abs(Trace_I_1_demod*1e3), np.abs(Trace_Q_1_demod*1e3),
                                      np.abs(Trace_I_2_demod*1e3), np.abs(Trace_Q_2_demod*1e3))))
    ax.set_xlim(-_lim*1.2, _lim*1.2)
    ax.set_ylim(-_lim*1.2, _lim*1.2)
    ax.set_xlabel('I Voltage (mV)')
    ax.set_ylabel('Q Voltage (mV)')
    ax.set_title(f'{timestamp}\nIQ trajectory')
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1))

def Weights_plotfn(
    Time, 
    Weights_I_demod, Weights_Q_demod, 
    Weights_I_demod_s, Weights_Q_demod_s,
    Weights_I_ef_demod, Weights_Q_ef_demod, 
    Weights_I_ef_demod_s, Weights_Q_ef_demod_s,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    
    axs[0].plot(Time*1e6, Weights_I_demod, color='C0', ls='-', lw=1, alpha=.25)
    axs[0].plot(Time*1e6, Weights_Q_demod, color='#6A1B9A', ls='-', lw=1, alpha=.25)
    axs[0].plot(Time*1e6, Weights_I_demod_s, color='C0', ls='-', lw=2, alpha=1, label='Weight function I')
    axs[0].plot(Time*1e6, Weights_Q_demod_s, color='#6A1B9A', ls='-', lw=2, alpha=1, label='Weight function Q')
    axs[1].plot(Weights_I_demod, Weights_Q_demod, color='C0', ls='-', lw=.5, alpha=.5)
    axs[1].plot(Weights_I_demod_s, Weights_Q_demod_s, color='C0', ls='-', lw=2, alpha=1, label='$ge$ weights')
    _lim = np.max(np.concatenate((np.abs(Weights_I_demod), np.abs(Weights_Q_demod))))
    if type(Weights_I_ef_demod) != type(None):
        axs[0].plot(Time*1e6, Weights_I_ef_demod, color='#008b00', ls='-', lw=1, alpha=.25)
        axs[0].plot(Time*1e6, Weights_Q_ef_demod, color='#B71C1C', ls='-', lw=1, alpha=.25)
        axs[0].plot(Time*1e6, Weights_I_ef_demod_s, color='#008b00', ls='-', lw=2, alpha=1, label='Weight function I ef')
        axs[0].plot(Time*1e6, Weights_Q_ef_demod_s, color='#B71C1C', ls='-', lw=2, alpha=1, label='Weight function Q ef')
        axs[1].plot(Weights_I_ef_demod, Weights_Q_ef_demod, color='C2', ls='-', lw=.5, alpha=.5)
        axs[1].plot(Weights_I_ef_demod_s, Weights_Q_ef_demod_s, color='C2', ls='-', lw=2, alpha=1, label='$ef$ weights')
        _lim = np.max(np.concatenate((np.abs(Weights_I_demod), np.abs(Weights_Q_demod),
                                      np.abs(Weights_I_ef_demod), np.abs(Weights_Q_ef_demod))))
    axs[0].set_xlabel('Time ($\mathrm{\mu s}$)')
    axs[0].set_ylabel('Amplitude (a.u.)')
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].set_xlim(-_lim*1.1, _lim*1.1)
    axs[1].set_ylim(-_lim*1.1, _lim*1.1)
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].set_xlabel('I component (a.u.)')
    axs[1].set_ylabel('Q component (a.u.)')
    axs[0].set_title('Optimal integration weights')
    axs[1].set_title('IQ trajectory')
    axs[1].legend(frameon=False)
    fig.suptitle(f'{timestamp}')

def FFT_plotfn(
    Freqs, 
    IF,
    ps_0,
    ps_1,
    timestamp,
    ax,
    Freqs_wf = None,
    ps_wf = None,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Remove first and last points of
    # array to remove nyquist frequency
    Freqs = Freqs[1:-1]
    ps_0 = ps_0[1:-1]
    ps_1 = ps_1[1:-1]
    axs[0].plot(Freqs*1e-6, ps_0, 'C0')
    axs[0].plot(Freqs*1e-6, ps_1, 'C3')
    if type(Freqs_wf) != None:
        Freqs_wf = Freqs_wf[1:-1]
        ps_wf = ps_wf[1:-1]
        axs[0].plot(Freqs_wf*1e-6, ps_wf, '--', color='#607D8B', alpha=.5)
    axs[0].axvline(IF*1e-6, color='k', ls='--', lw=1, label=f'IF : {IF*1e-6:.1f} MHz')
    axs[0].set_xlim(left=np.min(Freqs*1e-6), right=np.max(Freqs*1e-6))
    axs[0].set_xlabel('Frequency (MHz)')
    axs[0].set_ylabel('PSD ($\mathrm{V^2/Hz}$)')

    axs[1].plot(Freqs*1e-6, ps_0, 'C0', label='ground')
    axs[1].plot(Freqs*1e-6, ps_1, 'C3', label='excited')
    if type(Freqs_wf) != None:
        axs[1].plot(Freqs_wf*1e-6, ps_wf, 'C2--',
                    label='input pulse', alpha=.5)
    axs[1].axvline(IF*1e-6, color='k', ls='--', lw=1)
    axs[1].set_xlim(left=IF*1e-6-50, right=IF*1e-6+50)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1,1))
    fig.suptitle(f'{timestamp}\nTransients PSD', y=1.025)


class measurement_QND_analysis(ba.BaseDataAnalysis):
    """
    This analysis extracts measurement QND metrics 
    For details on the procedure see:
    arXiv:2110.04285
    """
    def __init__(self,
                 qubit:str,
                 f_state: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.f_state = f_state

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        if self.f_state:
            _cycle = 6
        else:
            _cycle = 5
        # Calibration shots
        I0, Q0 = self.raw_data_dict['data'][:,1][3::_cycle], self.raw_data_dict['data'][:,2][3::_cycle]
        I1, Q1 = self.raw_data_dict['data'][:,1][4::_cycle], self.raw_data_dict['data'][:,2][4::_cycle]
        if self.f_state:
            I2, Q2 = self.raw_data_dict['data'][:,1][5::_cycle], self.raw_data_dict['data'][:,2][5::_cycle]
            center_2 = np.array([np.mean(I2), np.mean(Q2)])
        # Measurement
        IM1, QM1 = self.raw_data_dict['data'][0::_cycle,1], self.raw_data_dict['data'][0::_cycle,2]
        IM2, QM2 = self.raw_data_dict['data'][1::_cycle,1], self.raw_data_dict['data'][1::_cycle,2]
        IM3, QM3 = self.raw_data_dict['data'][2::_cycle,1], self.raw_data_dict['data'][2::_cycle,2]
        # Rotate data
        center_0 = np.array([np.mean(I0), np.mean(Q0)])
        center_1 = np.array([np.mean(I1), np.mean(Q1)])
        def rotate_and_center_data(I, Q, vec0, vec1):
            vector = vec1-vec0
            angle = np.arctan(vector[1]/vector[0])
            rot_matrix = np.array([[ np.cos(-angle),-np.sin(-angle)],
                                   [ np.sin(-angle), np.cos(-angle)]])
            # Subtract mean
            proc = np.array((I-(vec0+vec1)[0]/2, Q-(vec0+vec1)[1]/2))
            # Rotate theta
            proc = np.dot(rot_matrix, proc)
            return proc
        I0_proc, Q0_proc = rotate_and_center_data(I0, Q0, center_0, center_1)
        I1_proc, Q1_proc = rotate_and_center_data(I1, Q1, center_0, center_1)
        IM1_proc, QM1_proc = rotate_and_center_data(IM1, QM1, center_0, center_1)
        IM2_proc, QM2_proc = rotate_and_center_data(IM2, QM2, center_0, center_1)
        IM3_proc, QM3_proc = rotate_and_center_data(IM3, QM3, center_0, center_1)
        if np.mean(I0_proc) > np.mean(I1_proc):
            I0_proc *= -1
            I1_proc *= -1
            IM1_proc *= -1
            IM2_proc *= -1
            IM3_proc *= -1
        # Calculate optimal threshold
        ubins_A_0, ucounts_A_0 = np.unique(I0_proc, return_counts=True)
        ubins_A_1, ucounts_A_1 = np.unique(I1_proc, return_counts=True)
        ucumsum_A_0 = np.cumsum(ucounts_A_0)
        ucumsum_A_1 = np.cumsum(ucounts_A_1)
        # merge |0> and |1> shot bins
        all_bins_A = np.unique(np.sort(np.concatenate((ubins_A_0, ubins_A_1))))
        # interpolate cumsum for all bins
        int_cumsum_A_0 = np.interp(x=all_bins_A, xp=ubins_A_0, fp=ucumsum_A_0, left=0)
        int_cumsum_A_1 = np.interp(x=all_bins_A, xp=ubins_A_1, fp=ucumsum_A_1, left=0)
        norm_cumsum_A_0 = int_cumsum_A_0/np.max(int_cumsum_A_0)
        norm_cumsum_A_1 = int_cumsum_A_1/np.max(int_cumsum_A_1)
        # Calculating threshold
        F_vs_th = (1-(1-abs(norm_cumsum_A_0-norm_cumsum_A_1))/2)
        opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
        opt_idx = int(round(np.average(opt_idxs)))
        threshold = all_bins_A[opt_idx]
        # digitize data
        P0_dig = np.array([ 0 if s<threshold else 1 for s in I0_proc ])
        P1_dig = np.array([ 0 if s<threshold else 1 for s in I1_proc ])
        M1_dig = np.array([ 0 if s<threshold else 1 for s in IM1_proc ])
        M2_dig = np.array([ 0 if s<threshold else 1 for s in IM2_proc ])
        M3_dig = np.array([ 0 if s<threshold else 1 for s in IM3_proc ])
        # Calculate qoi
        Fidelity = (np.mean(1-P0_dig) + np.mean(P1_dig))/2
        p0 = 1-np.mean(M1_dig)
        p1 = np.mean(M1_dig)
        p00 = np.mean(1-np.logical_or(M1_dig, M2_dig))/p0
        p11 = np.mean(np.logical_and(M1_dig, M2_dig))/p1
        P_QND = np.mean([p00, p11])
        p0p = 1-np.mean(M2_dig)
        p1p = np.mean(M2_dig)
        p01p = np.mean(1-np.logical_or(np.logical_not(M3_dig), M2_dig))/p0p
        p10p = np.mean(1-np.logical_or(np.logical_not(M2_dig), M3_dig))/p1p
        P_QNDp = np.mean([p01p, p10p])

        self.proc_data_dict['I0'], self.proc_data_dict['Q0'] = I0, Q0
        self.proc_data_dict['I1'], self.proc_data_dict['Q1'] = I1, Q1
        if self.f_state:
            self.proc_data_dict['I2'], self.proc_data_dict['Q2'] = I2, Q2
            self.proc_data_dict['center_2'] = center_2
        self.proc_data_dict['I0_proc'], self.proc_data_dict['Q0_proc'] = I0_proc, Q0_proc
        self.proc_data_dict['I1_proc'], self.proc_data_dict['Q1_proc'] = I1_proc, Q1_proc
        self.proc_data_dict['center_0'] = center_0
        self.proc_data_dict['center_1'] = center_1
        self.proc_data_dict['threshold'] = threshold
        self.qoi = {}
        self.qoi['p00'] = p00
        self.qoi['p11'] = p11
        self.qoi['p01p'] = p01p
        self.qoi['p10p'] = p10p
        self.qoi['Fidelity'] = Fidelity
        self.qoi['P_QND'] = P_QND
        self.qoi['P_QNDp'] = P_QNDp

    def prepare_plots(self):

        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(4,2), ncols=2, dpi=200)
        # fig.patch.set_alpha(0)
        self.axs_dict['main'] = axs[0]
        self.figs['main'] = fig
        self.plot_dicts['main'] = {
            'plotfn': plot_QND_metrics,
            'ax_id': 'main',
            'I0': self.proc_data_dict['I0'],
            'Q0': self.proc_data_dict['Q0'],
            'I1': self.proc_data_dict['I1'],
            'Q1': self.proc_data_dict['Q1'],
            'I2': self.proc_data_dict['I2'] if self.f_state else None,
            'Q2': self.proc_data_dict['Q2'] if self.f_state else None,
            'center_0': self.proc_data_dict['center_0'],
            'center_1': self.proc_data_dict['center_1'],
            'center_2': self.proc_data_dict['center_2'] if self.f_state else None,
            'I0_proc': self.proc_data_dict['I0_proc'],
            'I1_proc': self.proc_data_dict['I1_proc'],
            'threshold': self.proc_data_dict['threshold'],
            'p00': self.qoi['p00'],
            'p11': self.qoi['p11'],
            'p01p': self.qoi['p01p'],
            'p10p': self.qoi['p10p'],
            'P_QND': self.qoi['P_QND'],
            'P_QNDp': self.qoi['P_QNDp'],
            'Fidelity': self.qoi['Fidelity'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def plot_QND_metrics(I0, Q0,
                     I1, Q1,
                     I2, Q2,
                     center_0,
                     center_1,
                     center_2,
                     I0_proc,
                     I1_proc,
                     threshold,
                     p00, p11,
                     p01p, p10p,
                     P_QND, P_QNDp,
                     Fidelity,
                     timestamp,
                     qubit,
                     ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # plot raw shots on IQ plane
    axs[0].plot(I0, Q0, 'C0.', alpha=.05, markersize=1)
    axs[0].plot(I1, Q1, 'C3.', alpha=.05, markersize=1)
    if type(I2) != type(None):
        axs[0].plot(I2, Q2, 'C2.', alpha=.05, markersize=1)
    axs[0].plot([0, center_0[0]], [0, center_0[1]], ls='--', lw=.75, color='k', alpha=1)
    axs[0].plot([0, center_1[0]], [0, center_1[1]], ls='--', lw=.75, color='k', alpha=1)
    axs[0].plot(center_0[0], center_0[1], marker='x', color='k', markersize=3)
    axs[0].plot(center_1[0], center_1[1], marker='x', color='k', markersize=3)
    if type(center_2) != type(None):
        axs[0].plot([0, center_2[0]], [0, center_2[1]], ls='--', lw=.75, color='k', alpha=1)
        axs[0].plot(center_2[0], center_2[1], marker='x', color='k', markersize=3)
    # plot threshold
    x = np.arange(-10, 10)
    vector = center_1-center_0
    angle = np.arctan(vector[1]/vector[0])
    axs[0].plot(x+(center_0+center_1)[0]/2, np.tan(angle+np.pi/2)*x+(center_0+center_1)[1]/2, 
                ls='--', lw=.5, color='k')
    # plot histogram of rotated shots
    rang = np.max(list(np.abs(I0_proc))+list(np.abs(I1_proc)))
    axs[1].hist(I0_proc, range=[-rang, rang], bins=100, color='C0', alpha=.75, label='ground')
    axs[1].hist(I1_proc, range=[-rang, rang], bins=100, color='C3', alpha=.75, label='excited')
    axs[1].axvline(threshold, ls='--', lw=.5, color='k', label='threshold')
    axs[1].legend(loc='upper right', fontsize=3, frameon=False)
    
    rang = np.max(list(np.abs(I0))+list(np.abs(I1))+
                  list(np.abs(Q0))+list(np.abs(Q1)))
    axs[0].set_xlim(-1.15*rang,1.15*rang)
    axs[0].set_ylim(-1.15*rang,1.15*rang)
    axs[0].set_title('Raw calibration shots', fontsize=9)
    axs[0].set_ylabel('Q quadrature (mV)', size=8)
    axs[0].set_xlabel('I quadrature (mV)', size=8)
    axs[1].set_yticks([])
    axs[1].set_title('Rotated data', fontsize=9)
    axs[1].set_xlabel('Integrated voltage (mV)', size=8)
    # Write results
    text = '\n'.join((f'P$(0_2|0_1)$  = {p00*100:.2f} %',
                      f'P$(1_2|1_1)$  = {p11*100:.2f} %',
                      f'P$(1_3|0_2)$  = {p01p*100:.2f} %',
                      f'P$(0_3|1_2)$  = {p10p*100:.2f} %',
                      '',
                      f'Fidelity$= {Fidelity*100:.2f}$ %',
                      '$P_{QND}$ = '+f'{P_QND*100:.2f} %',
                      '$P_{QND,X_\pi}$ = '+f'{P_QNDp*100:.2f} %'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.15)
    axs[1].text(1.05, 1, 'Experiment', transform=axs[1].transAxes, fontsize=6,
            verticalalignment='top')
    axs[1].text(1.05, .975-.225, 'Results', transform=axs[1].transAxes, fontsize=6,
            verticalalignment='top')
    axs[1].text(1.05, 0.9-.225, text, transform=axs[1].transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    # Plot experiment
    ax1 = fig.add_subplot(212)
    ax1.set_position((.9, .7 , .225, .15))
    ax1.set_xlim(0,  1*1.12)
    ax1.set_ylim(0, .4*1.12)
    ax1.axis('off')
    ax1.plot([.1, 2.01], [.2, .2], 'k', lw=.5)
    rect = patches.Rectangle((.05, .125), .15, .15, linewidth=.25, edgecolor='k', facecolor='white', zorder=3)
    ax1.add_patch(rect)
    ax1.text(.125, .185, '$X_{\pi/2}$', va='center', ha='center', size=4)
    rect = patches.Rectangle((.22, .125), .22, .15, linewidth=.25, edgecolor='k', facecolor='white', zorder=3)
    ax1.add_patch(rect)
    ax1.text(.33, .185, '$m_1$', va='center', ha='center', size=4)
    rect = patches.Rectangle((.47, .125), .22, .15, linewidth=.25, edgecolor='k', facecolor='white', zorder=3)
    ax1.add_patch(rect)
    ax1.text(.58, .185, '$m_2$', va='center', ha='center', size=4)
    rect = patches.Rectangle((.72, .125), .15, .15, linewidth=.25, edgecolor='k', facecolor='white', zorder=3)
    ax1.add_patch(rect)
    ax1.text(.8, .185, '$X_{\pi}$', va='center', ha='center', size=4)
    rect = patches.Rectangle((.89, .125), .22, .15, linewidth=.25, edgecolor='k', facecolor='white', zorder=3)
    ax1.add_patch(rect)
    ax1.text(1, .185, '$m_3$', va='center', ha='center', size=4)
    fig.suptitle(f'Qubit {qubit}\n{timestamp}', y=1.1, size=9)


def logisticreg_classifier_machinelearning(shots_0, shots_1, shots_2):
    """ """
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

def create_xr_data(proc_data_dict,qubit,timestamp):

    NUM_STATES = 3

    calibration_data = np.array([[proc_data_dict[f"{comp}{state}"] for comp in ("I", "Q")] for state in range(NUM_STATES)])

    arr_data = []
    for state_ind in range(NUM_STATES):
        state_data = []
        for meas_ind in range(1, NUM_STATES + 1):
            ind = NUM_STATES*state_ind + meas_ind
            meas_data = [proc_data_dict[f"{comp}M{ind}"] for comp in ("I", "Q")]
            state_data.append(meas_data)
        arr_data.append(state_data)

    butterfly_data = np.array(arr_data)

    NUM_STATES, NUM_MEAS_INDS, NUM_COMPS, NUM_SHOTS = butterfly_data.shape

    assert NUM_COMPS == 2

    STATES = list(range(0, NUM_STATES))
    MEAS_INDS = list(range(1, NUM_MEAS_INDS + 1))
    SHOTS = list(range(1, NUM_SHOTS + 1))

    exp_dataset = xr.Dataset(
        data_vars = dict(
            calibration = (["state", "comp", "shot"], calibration_data),
            characterization = (["state", "meas_ind", "comp", "shot"], butterfly_data),
        ),
        coords = dict(
            state = STATES,
            meas_ind = MEAS_INDS,
            comp = ["in-phase", "quadrature"],
            shot = SHOTS,
            qubit = qubit,
        ),
        attrs = dict(
            description = "Qutrit measurement butterfly data.",
            timestamp = timestamp,
        )
    )

    return exp_dataset

def QND_qutrit_anaylsis(NUM_STATES,
                        NUM_OUTCOMES,
                        STATES,
                        OUTCOMES,
                        char_data,
                        cal_data,
                        fid,
                        accuracy,
                        timestamp,
                        classifier):

    data = char_data.stack(stacked_dim = ("state", "meas_ind", "shot"))
    data = data.transpose("stacked_dim", "comp")

    predictions = classifier.predict(data)
    outcome_vec = xr.DataArray(
        data = predictions,
        dims = ["stacked_dim"],
        coords = dict(stacked_dim = data.stacked_dim)
    )

    digital_char_data = outcome_vec.unstack()
    matrix = np.zeros((NUM_STATES, NUM_OUTCOMES, NUM_OUTCOMES), dtype=float)

    for state in STATES:
        meas_arr = digital_char_data.sel(state=state)
        postsel_arr = meas_arr.where(meas_arr[0] == 0, drop=True)

        for first_out in OUTCOMES:
            first_cond = xr.where(postsel_arr[1] == first_out, 1, 0)

            for second_out in OUTCOMES:
                second_cond = xr.where(postsel_arr[2] == second_out, 1, 0)

                sel_shots = first_cond & second_cond
                joint_prob = np.mean(sel_shots)

                matrix[state, first_out, second_out] = joint_prob

    joint_probs = xr.DataArray(
        matrix,
        dims = ["state", "meas_1", "meas_2"],
        coords = dict(
            state = STATES,
            meas_1 = OUTCOMES,
            meas_2 = OUTCOMES,
        )
    )

    num_constraints = NUM_STATES
    num_vars = NUM_OUTCOMES * (NUM_STATES ** 2)

    def opt_func(variables, obs_probs, num_states: int) -> float:
        meas_probs = variables.reshape(num_states, num_states, num_states)
        probs = np.einsum("ijk, klm -> ijl", meas_probs, meas_probs)
        return np.linalg.norm(np.ravel(probs - obs_probs))

    cons_mat = np.zeros((num_constraints, num_vars), dtype=int)
    num_cons_vars = int(num_vars / num_constraints)
    for init_state in range(NUM_STATES):
        var_ind = init_state * num_cons_vars
        cons_mat[init_state, var_ind : var_ind + num_cons_vars] = 1

    constraints = {"type": "eq", "fun": lambda variables: cons_mat @ variables - 1}
    bounds = opt.Bounds(0, 1)

    ideal_probs = np.zeros((NUM_STATES, NUM_OUTCOMES, NUM_OUTCOMES), dtype=float)
    for state in range(NUM_STATES):
        ideal_probs[state, state, state] = 1
    init_vec = np.ravel(ideal_probs)

    result = opt.basinhopping(
        opt_func,
        init_vec,
        niter=500,
        minimizer_kwargs=dict(
            args=(joint_probs.data, NUM_STATES),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            tol=1e-12,
            options=dict(
                maxiter=10000,
            )
        )
    )
    # if not result.success:
    #     raise ValueError("Unsuccessful optimization, please check parameters and tolerance.")
    res_data = result.x.reshape((NUM_STATES, NUM_OUTCOMES, NUM_STATES))

    meas_probs = xr.DataArray(
        res_data,
        dims = ["input_state", "outcome", "output_state"],
        coords = dict(
            input_state = STATES,
            outcome = OUTCOMES,
            output_state = STATES,
        )
    )

    pred_joint_probs = np.einsum("ijk, klm -> ijl", meas_probs, meas_probs)

    true_vals = np.ravel(joint_probs)
    pred_vals = np.ravel(pred_joint_probs)

    ms_error = mean_squared_error(true_vals, pred_vals)
    rms_error = np.sqrt(ms_error)
    ma_error = mean_absolute_error(true_vals, pred_vals)

    print(f"RMS error of the optimised solution: {rms_error}")
    print(f"MA error of the optimised solution: {ma_error}")

    num_vars = 3 * (NUM_STATES ** 2)
    num_constraints = 3 * NUM_STATES

    def opt_func(
        variables,
        obs_probs,
        num_states: int,
    ) -> float:
        pre_mat, ro_mat, post_mat = variables.reshape(3, num_states, num_states)
        probs = np.einsum("ih, hm, ho -> imo", pre_mat, ro_mat, post_mat)
        return np.linalg.norm(probs - obs_probs)

    cons_mat = np.zeros((num_constraints, num_vars), dtype=int)
    for op_ind in range(3):
        for init_state in range(NUM_STATES):
            cons_ind = op_ind*NUM_STATES + init_state
            var_ind = (op_ind*NUM_STATES + init_state)*NUM_STATES
            cons_mat[cons_ind, var_ind : var_ind + NUM_STATES] = 1

    ideal_probs = np.tile(np.eye(NUM_STATES), (3, 1))
    init_vec = np.ravel(ideal_probs)

    constraints = {"type": "eq", "fun": lambda variables: cons_mat @ variables - 1}
    bounds = opt.Bounds(0, 1, keep_feasible=True)

    result = opt.basinhopping(
        opt_func,
        init_vec,
        minimizer_kwargs = dict(
            args = (meas_probs.data, NUM_STATES),
            bounds = bounds,
            constraints = constraints,
            method = "SLSQP",
            tol = 1e-12,
            options = dict(
                maxiter = 10000,
            )
        ),
        niter=500
    )


    # if not result.success:
    #     raise ValueError("Unsuccessful optimization, please check parameters and tolerance.")

    pre_trans, ass_errors, post_trans = result.x.reshape((3, NUM_STATES, NUM_STATES))

    pred_meas_probs = np.einsum("ih, hm, ho -> imo", pre_trans, ass_errors, post_trans)

    true_vals = np.ravel(meas_probs)
    pred_vals = np.ravel(pred_meas_probs)

    ms_error = mean_squared_error(true_vals, pred_vals)
    rms_error = np.sqrt(ms_error)
    ma_error = mean_absolute_error(true_vals, pred_vals)

    print(f"RMS error of the optimised solution: {rms_error}")
    print(f"MA error of the optimised solution: {ma_error}")

    QND_state = {}
    for state in STATES:
        state_qnd = np.sum(meas_probs.data[state,:, state])
        QND_state[f'{state}'] = state_qnd

    meas_qnd = np.mean(np.diag(meas_probs.sum(axis=1)))
    meas_qnd

    fit_res = {}
    fit_res['butter_prob'] = pred_meas_probs
    fit_res['mean_QND'] = meas_qnd
    fit_res['state_qnd'] = QND_state
    fit_res['ass_errors'] = ass_errors
    fit_res['qutrit_fidelity'] = accuracy*100
    fit_res['fidelity'] = fid
    fit_res['timestamp'] = timestamp

    # Meas leak rate
    L1 = 100*np.sum(fit_res['butter_prob'][:2,:,2])/2

    # Meas seepage rate
    s = 100*np.sum(fit_res['butter_prob'][2,:,:2])

    fit_res['L1'] = L1
    fit_res['seepage'] = s

    return fit_res

class measurement_butterfly_analysis(ba.BaseDataAnalysis):
    """
    This analysis extracts measurement butter fly
    """
    def __init__(self,
                 qubit:str,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 f_state: bool = False,
                 cycle : int = 6,
                 options_dict: dict = None,
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.f_state = f_state

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):

        if self.f_state:
            _cycle = 12
            I0, Q0 = self.raw_data_dict['data'][:,1][9::_cycle], self.raw_data_dict['data'][:,2][9::_cycle]
            I1, Q1 = self.raw_data_dict['data'][:,1][10::_cycle], self.raw_data_dict['data'][:,2][10::_cycle]
            I2, Q2 = self.raw_data_dict['data'][:,1][11::_cycle], self.raw_data_dict['data'][:,2][11::_cycle]
        else:
            _cycle = 8
            I0, Q0 = self.raw_data_dict['data'][:,1][6::_cycle], self.raw_data_dict['data'][:,2][6::_cycle]
            I1, Q1 = self.raw_data_dict['data'][:,1][7::_cycle], self.raw_data_dict['data'][:,2][7::_cycle]
        # Measurement
        IM1, QM1 = self.raw_data_dict['data'][0::_cycle,1], self.raw_data_dict['data'][0::_cycle,2]
        IM2, QM2 = self.raw_data_dict['data'][1::_cycle,1], self.raw_data_dict['data'][1::_cycle,2]
        IM3, QM3 = self.raw_data_dict['data'][2::_cycle,1], self.raw_data_dict['data'][2::_cycle,2]
        IM4, QM4 = self.raw_data_dict['data'][3::_cycle,1], self.raw_data_dict['data'][3::_cycle,2]
        IM5, QM5 = self.raw_data_dict['data'][4::_cycle,1], self.raw_data_dict['data'][4::_cycle,2]
        IM6, QM6 = self.raw_data_dict['data'][5::_cycle,1], self.raw_data_dict['data'][5::_cycle,2]
        # Rotate data
        center_0 = np.array([np.mean(I0), np.mean(Q0)])
        center_1 = np.array([np.mean(I1), np.mean(Q1)])
        if self.f_state:
            IM7, QM7 = self.raw_data_dict['data'][6::_cycle,1], self.raw_data_dict['data'][6::_cycle,2]
            IM8, QM8 = self.raw_data_dict['data'][7::_cycle,1], self.raw_data_dict['data'][7::_cycle,2]
            IM9, QM9 = self.raw_data_dict['data'][8::_cycle,1], self.raw_data_dict['data'][8::_cycle,2]
            center_2 = np.array([np.mean(I2), np.mean(Q2)])
        def rotate_and_center_data(I, Q, vec0, vec1):
            vector = vec1-vec0
            angle = np.arctan(vector[1]/vector[0])
            rot_matrix = np.array([[ np.cos(-angle),-np.sin(-angle)],
                                   [ np.sin(-angle), np.cos(-angle)]])
            # Subtract mean
            proc = np.array((I-(vec0+vec1)[0]/2, Q-(vec0+vec1)[1]/2))
            # Rotate theta
            proc = np.dot(rot_matrix, proc)
            return proc
        # proc cal points
        I0_proc, Q0_proc = rotate_and_center_data(I0, Q0, center_0, center_1)
        I1_proc, Q1_proc = rotate_and_center_data(I1, Q1, center_0, center_1)
        # proc M
        IM1_proc, QM1_proc = rotate_and_center_data(IM1, QM1, center_0, center_1)
        IM2_proc, QM2_proc = rotate_and_center_data(IM2, QM2, center_0, center_1)
        IM3_proc, QM3_proc = rotate_and_center_data(IM3, QM3, center_0, center_1)
        IM4_proc, QM4_proc = rotate_and_center_data(IM4, QM4, center_0, center_1)
        IM5_proc, QM5_proc = rotate_and_center_data(IM5, QM5, center_0, center_1)
        IM6_proc, QM6_proc = rotate_and_center_data(IM6, QM6, center_0, center_1)
        if np.mean(I0_proc) > np.mean(I1_proc):
            I0_proc *= -1
            I1_proc *= -1
            IM1_proc *= -1
            IM2_proc *= -1
            IM3_proc *= -1
            IM4_proc *= -1
            IM5_proc *= -1
            IM6_proc *= -1
        # Calculate optimal threshold
        ubins_A_0, ucounts_A_0 = np.unique(I0_proc, return_counts=True)
        ubins_A_1, ucounts_A_1 = np.unique(I1_proc, return_counts=True)
        ucumsum_A_0 = np.cumsum(ucounts_A_0)
        ucumsum_A_1 = np.cumsum(ucounts_A_1)
        # merge |0> and |1> shot bins
        all_bins_A = np.unique(np.sort(np.concatenate((ubins_A_0, ubins_A_1))))
        # interpolate cumsum for all bins
        int_cumsum_A_0 = np.interp(x=all_bins_A, xp=ubins_A_0, fp=ucumsum_A_0, left=0)
        int_cumsum_A_1 = np.interp(x=all_bins_A, xp=ubins_A_1, fp=ucumsum_A_1, left=0)
        norm_cumsum_A_0 = int_cumsum_A_0/np.max(int_cumsum_A_0)
        norm_cumsum_A_1 = int_cumsum_A_1/np.max(int_cumsum_A_1)
        # Calculating threshold
        F_vs_th = (1-(1-abs(norm_cumsum_A_0-norm_cumsum_A_1))/2)
        opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
        opt_idx = int(round(np.average(opt_idxs)))
        threshold = all_bins_A[opt_idx]
        # fidlity calculation from cal point
        P0_dig = np.array([ 0 if s<threshold else 1 for s in I0_proc ])
        P1_dig = np.array([ 0 if s<threshold else 1 for s in I1_proc ])
        M1_dig = np.array([ 0 if s<threshold else 1 for s in IM1_proc ])
        M2_dig = np.array([ 0 if s<threshold else 1 for s in IM2_proc ])
        M3_dig = np.array([ 0 if s<threshold else 1 for s in IM3_proc ])
        M4_dig = np.array([ 0 if s<threshold else 1 for s in IM4_proc ])
        M5_dig = np.array([ 0 if s<threshold else 1 for s in IM5_proc ])
        M6_dig = np.array([ 0 if s<threshold else 1 for s in IM6_proc ])
        Fidelity = (np.mean(1-P0_dig) + np.mean(P1_dig))/2
        # postselected init for zero
        I_mask = np.ones(len(IM1_proc))
        Q_mask = np.ones(len(QM1_proc))
        IM1_init =   np.array([ +1 if shot < threshold else np.nan for shot in IM1_proc ], dtype=float)
        QM1_init =   np.array([ +1 if shot < threshold else np.nan for shot in QM1_proc ], dtype=float)
        IM2_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM2_proc ], dtype=float)
        QM2_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM2_proc ], dtype=float)
        IM3_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM3_proc ], dtype=float)
        QM3_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM3_proc ], dtype=float)
        I_mask *= IM1_init
        Q_mask *= QM1_init
        IM2_dig_ps *= I_mask
        QM2_dig_ps *= Q_mask
        IM3_dig_ps *= I_mask
        QM3_dig_ps *= Q_mask
        fraction_discarded_I = np.sum(np.isnan(I_mask))/len(I_mask)
        fraction_discarded_Q = np.sum(np.isnan(Q_mask))/len(Q_mask)
        # # postselectted init for 1
        I1_mask = np.ones(len(IM4_proc))
        Q1_mask = np.ones(len(QM4_proc))
        IM4_init =  np.array([ +1 if shot < threshold else np.nan for shot in IM4_proc ], dtype=float)
        QM4_init =  np.array([ +1 if shot < threshold else np.nan for shot in QM4_proc ], dtype=float)
        IM5_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM5_proc ], dtype=float)
        QM5_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM5_proc ], dtype=float)
        IM6_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM6_proc ], dtype=float)
        QM6_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM6_proc ], dtype=float)
        I1_mask *= IM4_init
        Q1_mask *= QM4_init
        IM5_dig_ps *= I1_mask
        QM5_dig_ps *= Q1_mask
        IM6_dig_ps *= I1_mask
        QM6_dig_ps *= Q1_mask
        fraction_discarded_I1 = np.sum(np.isnan(I1_mask))/len(I1_mask)
        fraction_discarded_Q1 = np.sum(np.isnan(Q1_mask))/len(Q1_mask)
        # digitize data
        M1_dig_ps = (1-IM1_init)/2 # turn into binary
        M2_dig_ps = (1-IM2_dig_ps)/2 # turn into binary
        M3_dig_ps = (1-IM3_dig_ps)/2 # turn into binary
        M4_dig_ps = (1-IM4_init)/2 # turn into binary
        M5_dig_ps = (1-IM5_dig_ps)/2 # turn into binary
        M6_dig_ps = (1-IM6_dig_ps)/2 # turn into binary
        #############
        ## p for prep 0
        #############
        p0_init_prep0 = 1-np.nanmean(M1_dig_ps)
        p1_init_prep0 = np.nanmean(M1_dig_ps)
        p0_M1_prep0 = 1-np.nanmean(M2_dig_ps)
        p1_M1_prep0 = np.nanmean(M2_dig_ps)
        #############
        ## p for prep 1
        #############
        p0_init_prep1 = 1-np.nanmean(M4_dig_ps)
        p1_init_prep1 = np.nanmean(M4_dig_ps)
        p0_M1_prep1 = 1-np.nanmean(M5_dig_ps)
        p1_M1_prep1 = np.nanmean(M5_dig_ps)
        #############
        ## pij calculation
        #############
        p00_prep0 = np.nanmean(1-np.logical_or(M2_dig_ps, M3_dig_ps))
        p00_prep1 = np.nanmean(1-np.logical_or(M5_dig_ps, M6_dig_ps))
        p01_prep0 = np.nanmean(1-np.logical_or(np.logical_not(M3_dig), M2_dig))
        p01_prep1 = np.nanmean(1-np.logical_or(np.logical_not(M6_dig), M5_dig))
        p10_prep0 = np.nanmean(1-np.logical_or(np.logical_not(M2_dig), M3_dig))
        p10_prep1 = np.nanmean(1-np.logical_or(np.logical_not(M5_dig), M6_dig))
        p11_prep0 = np.nanmean(np.logical_and(M2_dig, M3_dig))
        p11_prep1 = np.nanmean(np.logical_and(M5_dig, M6_dig))

        #############
        ## eps calculations
        #############
        ## calculation e +1 when prep 0
        t1 = ((p1_M1_prep1*p00_prep0)/(p0_M1_prep1))-p01_prep0
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        ep1_0_0 =  t1/t2
        ep1_1_0 = p0_M1_prep0 - ep1_0_0
        ## calculation e -1 when prep 0
        t1 = ((p1_M1_prep1*p10_prep0)/(p0_M1_prep1))-p11_prep0
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        en1_0_0 =  t1/t2
        en1_1_0 = p1_M1_prep0 - en1_0_0
        ## calculation e +1 when prep 1
        t1 = ((p1_M1_prep1*p00_prep1)/(p0_M1_prep1))-p01_prep1
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        ep1_0_1 =  t1/t2
        ep1_1_1 = p0_M1_prep1 - ep1_0_1
        ## calculation e -1 when prep 1
        t1 = ((p1_M1_prep1*p10_prep1)/(p0_M1_prep1))-p11_prep1
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        en1_0_1 =  t1/t2
        en1_1_1 = p1_M1_prep1 - en1_0_1
        # save proc dict
        # cal points
        self.proc_data_dict["I0"], self.proc_data_dict["Q0"] = I0, Q0
        self.proc_data_dict["I1"], self.proc_data_dict["Q1"] = I1, Q1
        if self.f_state:
            self.proc_data_dict["I2"], self.proc_data_dict["Q2"] = I2, Q2
            self.proc_data_dict["center_2"] = center_2
        # shots
        self.proc_data_dict["IM1"], self.proc_data_dict["QM1"] = (IM1,QM1,)  # used for postselection for 0 state
        self.proc_data_dict["IM2"], self.proc_data_dict["QM2"] = IM2, QM2
        self.proc_data_dict["IM3"], self.proc_data_dict["QM3"] = IM3, QM3
        self.proc_data_dict["IM4"], self.proc_data_dict["QM4"] = (IM4,QM4,)  # used for postselection for 1 state
        self.proc_data_dict["IM5"], self.proc_data_dict["QM5"] = IM5, QM5
        self.proc_data_dict["IM6"], self.proc_data_dict["QM6"] = IM6, QM6
        if self.f_state:
            self.proc_data_dict["IM7"], self.proc_data_dict["QM7"] = (IM7,QM7,)  # used for postselection for 2 state
            self.proc_data_dict["IM8"], self.proc_data_dict["QM8"] = IM8, QM8
            self.proc_data_dict["IM9"], self.proc_data_dict["QM9"] = IM9, QM9
        # center of the prepared states
        self.proc_data_dict["center_0"] = center_0
        self.proc_data_dict["center_1"] = center_1
        if self.f_state:
            self.proc_data_dict["center_2"] = center_2
        self.proc_data_dict['I0_proc'], self.proc_data_dict['Q0_proc'] = I0_proc, Q0_proc
        self.proc_data_dict['I1_proc'], self.proc_data_dict['Q1_proc'] = I1_proc, Q1_proc
        self.proc_data_dict['threshold'] = threshold
        self.qoi = {}
        self.qoi['p00_0'] = p00_prep0
        self.qoi['p00_1'] = p00_prep1
        self.qoi['p11_0'] = p11_prep0
        self.qoi['p11_1'] = p11_prep1
        self.qoi['p01_0'] = p01_prep0
        self.qoi['p01_1'] = p01_prep1
        self.qoi['p10_0'] = p10_prep0
        self.qoi['p10_1'] = p10_prep1
        self.qoi['Fidelity'] = Fidelity
        self.qoi['ps_fraction_0'] = fraction_discarded_I
        self.qoi['ps_fraction_1'] = fraction_discarded_I1
        self.qoi['ep1_0_0'] = ep1_0_0
        self.qoi['ep1_1_0'] = ep1_1_0
        self.qoi['en1_0_0'] = en1_0_0
        self.qoi['en1_1_0'] = en1_1_0
        self.qoi['ep1_0_1'] = ep1_0_1
        self.qoi['ep1_1_1'] = ep1_1_1
        self.qoi['en1_0_1'] = en1_0_1
        self.qoi['en1_1_1'] = en1_1_1
        if self.f_state:
            ## QND for qutrit RO
            dataset = create_xr_data(self.proc_data_dict,self.qubit,self.timestamp)
            NUM_STATES = NUM_OUTCOMES = 3
            STATES = np.arange(NUM_STATES)
            OUTCOMES = np.arange(NUM_OUTCOMES)
            data_subset = dataset.sel(state=STATES)
            char_data = data_subset.characterization.copy(deep=True)
            cal_data = data_subset.calibration.copy(deep=True)
            classifier = LinearDiscriminantAnalysis(
                solver="svd",
                shrinkage=None,
                tol=1e-4)
            train_data = cal_data.stack(stacked_dim = ("state", "shot"))
            train_data = train_data.transpose("stacked_dim", "comp")
            classifier.fit(train_data, train_data.state)
            accuracy = classifier.score(train_data, train_data.state)
            print(f"Total classifier accuracy on the calibration dataset: {accuracy*100:.3f} %")
            fid = {}
            for state in STATES:
                data_subset = train_data.sel(state = state)
                subset_labels = xr.full_like(data_subset.shot, state)
                state_accuracy = classifier.score(data_subset, subset_labels)
                fid[f'{state}'] = state_accuracy*100
                print(f"State {state} accuracy on the calibration dataset: {state_accuracy*100:.3f} %")
            fit_res = QND_qutrit_anaylsis(NUM_STATES,
                                          NUM_OUTCOMES,
                                          STATES,
                                          OUTCOMES,
                                          char_data,
                                          cal_data,
                                          fid,
                                          accuracy,
                                          self.timestamp,
                                          classifier)
            dec_bounds = _decision_boundary_points(classifier.coef_, classifier.intercept_)
            self.proc_data_dict['classifier'] = classifier
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.qoi['fit_res'] = fit_res

    def prepare_plots(self):

        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(4,2), dpi=200)
        # fig.patch.set_alpha(0)
        self.axs_dict['msmt_butterfly'] = axs
        self.figs['msmt_butterfly'] = fig
        self.plot_dicts['msmt_butterfly'] = {
            'plotfn': plot_msmt_butterfly,
            'ax_id': 'msmt_butterfly',
            'I0_proc': self.proc_data_dict['I0_proc'],
            'I1_proc': self.proc_data_dict['I1_proc'],
            'threshold': self.proc_data_dict['threshold'],
            'Fidelity': self.qoi['Fidelity'],
            'fit_res':self.qoi['fit_res'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        if self.f_state:
            _shots_0 = np.hstack((np.array([self.proc_data_dict['I0']]).T,
                                  np.array([self.proc_data_dict['Q0']]).T))
            _shots_1 = np.hstack((np.array([self.proc_data_dict['I1']]).T,
                                  np.array([self.proc_data_dict['Q1']]).T))
            _shots_2 = np.hstack((np.array([self.proc_data_dict['I2']]).T,
                                  np.array([self.proc_data_dict['Q2']]).T))
            fig = plt.figure(figsize=(8,4), dpi=100)
            axs = [fig.add_subplot(121)]
            # fig.patch.set_alpha(0)
            self.axs_dict[f'IQ_readout_histogram_{self.qubit}'] = axs[0]
            self.figs[f'IQ_readout_histogram_{self.qubit}'] = fig
            self.plot_dicts[f'IQ_readout_histogram_{self.qubit}'] = {
                'plotfn': ssro_IQ_projection_plotfn2,
                'ax_id': f'IQ_readout_histogram_{self.qubit}',
                'shots_0': _shots_0,
                'shots_1': _shots_1,
                'shots_2': _shots_2,
                'classifier': self.proc_data_dict['classifier'],
                'dec_bounds': self.proc_data_dict['dec_bounds'],
                'Fid_dict': self.qoi['fit_res']['fidelity'],
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def plot_msmt_butterfly(I0_proc, I1_proc,
                        threshold,Fidelity,
                        timestamp,
                        qubit,
                        fit_res,
                        ax, **kw
                        ):
    # plot histogram of rotated shots
    fig = ax.get_figure()
    rang = np.max(list(np.abs(I0_proc))+list(np.abs(I1_proc)))
    ax.hist(I0_proc, range=[-rang, rang], bins=100, color='C0', alpha=.75, label='ground')
    ax.hist(I1_proc, range=[-rang, rang], bins=100, color='C3', alpha=.75, label='excited')
    ax.axvline(threshold, ls='--', lw=.5, color='k', label='threshold')
    ax.legend(loc='upper right', fontsize=3, frameon=False)
    ax.set_yticks([])
    # ax.set_title('Rotated data', fontsize=9)
    ax.set_xlabel('Integrated voltage (mV)', size=8)
    # Write results
    text = '\n'.join(('Assignment Fid:',
                      rf'$\mathrm{"{F_{g}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["0"]:.2f}$%',
                      rf'$\mathrm{"{F_{e}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["1"]:.2f}$%',
                      rf'$\mathrm{"{F_{f}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["2"]:.2f}$%',
                      rf'$\mathrm{"{F_{avg}}"}:\:\:\:{fit_res["qutrit_fidelity"]:.2f}$%',
                      '',
                      'QNDness:',
                      rf'$\mathrm{"{QND_{g}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["0"]*100:.2f}$%',
                      rf'$\mathrm{"{QND_{e}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["1"]*100:.2f}$%',
                      rf'$\mathrm{"{QND_{f}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["2"]*100:.2f}$%',
                      rf'$\mathrm{"{QND_{avg}}"}:\:\:\:{fit_res["mean_QND"]*100:.2f}$%',
                      '',
                      'L1 & seepage',
                      rf'$\mathrm{"{L_{1}}"}:\:\:\:{fit_res["L1"]:.2f}$%',
                      rf'$\mathrm{"{L_{2}}"}:\:\:\:{fit_res["seepage"]:.2f}$%',
                      ))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.15)
    ax.text(1.05, 0.99, text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    fig.suptitle(f'Qubit {qubit}\n{timestamp}', y=1.1, size=9)

def ssro_IQ_projection_plotfn2(
    shots_0, 
    shots_1,
    shots_2,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:10000,0], shots_0[:10000,1], '.', color='C0', alpha=0.025)
    axs[0].plot(shots_1[:10000,0], shots_1[:10000,1], '.', color='C3', alpha=0.025)
    axs[0].plot(shots_2[:10000,0], shots_2[:10000,1], '.', color='C2', alpha=0.025)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False, loc=1)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'{timestamp}\nIQ plot qubit {qubit}')
    # Write fidelity textbox
    _f_avg = np.mean((Fid_dict["0"], Fid_dict["1"], Fid_dict["2"]))
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {_f_avg:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[0].text(1.05, 1, text, transform=axs[0].transAxes,
                verticalalignment='top', bbox=props)


class Depletion_AllXY_analysis(ba.BaseDataAnalysis):
    """
    """
    def __init__(self,
                 qubit,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.qubit = qubit
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        _cycle = 6
        data_0 = self.raw_data_dict['data'][:,1][0::_cycle]
        data_1 = self.raw_data_dict['data'][:,1][2::_cycle]
        data_2 = self.raw_data_dict['data'][:,1][3::_cycle]
        data_3 = self.raw_data_dict['data'][:,1][5::_cycle]
        zero_lvl = np.mean(data_0[:2])
        one_lvl = np.mean(data_0[-2:])
        data_0 = (data_0 - zero_lvl)/(one_lvl-zero_lvl)
        data_1 = (data_1 - zero_lvl)/(one_lvl-zero_lvl)
        data_2 = (data_2 - zero_lvl)/(one_lvl-zero_lvl)
        data_3 = (data_3 - zero_lvl)/(one_lvl-zero_lvl)
        self.proc_data_dict['data_0'] = data_0
        self.proc_data_dict['data_1'] = data_1
        self.proc_data_dict['data_2'] = data_2
        self.proc_data_dict['data_3'] = data_3
        
    def prepare_plots(self):
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(12,4), ncols=2)
        axs = axs.flatten()
        self.figs['main'] = fig
        self.axs_dict['main'] = axs[0]
        self.plot_dicts['main'] = {
            'plotfn': plot_depletion_allxy,
            'ax_id': 'main',
            'data_0': self.proc_data_dict['data_0'],
            'data_1': self.proc_data_dict['data_1'],
            'data_2': self.proc_data_dict['data_2'],
            'data_3': self.proc_data_dict['data_3'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def plot_depletion_allxy(qubit, timestamp,
                         data_0, data_1,
                         data_2, data_3,
                         ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    allXY = ['II', 'XX', 'YY', 'XY', 'YX', 'xI', 'yI',
             'xy', 'yx', 'xY', 'yX', 'Xy', 'Yx', 'xX',
             'Xx', 'yY', 'Yy', 'XI', 'YI', 'xx', 'yy']

    ideal = [0 for i in range(10)] + [.5 for i in range(24)] + [1 for i in range(8)]

    axs[0].set_xticks(np.arange(0, 42, 2)+.5)
    axs[0].set_xticklabels(allXY)
    axs[0].set_ylabel(r'P($|1\rangle$)')
    axs[0].plot(ideal, 'k--', lw=1, label='ideal')
    axs[0].plot(data_0, 'C0o-', alpha=1, label='Standard sequence')
    axs[0].plot(data_1, 'C1.-', alpha=.75, label='post-measurement')
    axs[0].legend(loc=0)
    axs[0].set_title(r'Qubit initialized in $|0\rangle$')

    axs[1].set_xticks(np.arange(0, 42, 2)+.5)
    axs[1].set_xticklabels(allXY)
    axs[1].set_ylabel(r'P($|1\rangle$)')
    axs[1].plot(1-np.array(ideal), 'k--', lw=1, label='ideal')
    axs[1].plot(data_2, 'C0o-', alpha=1, label='Standard sequence')
    axs[1].plot(data_3, 'C1.-', alpha=.75, label='post-measurement')
    axs[1].legend(loc=0)
    axs[1].set_title(r'Qubit initialized in $|1\rangle$')

    fig.suptitle(timestamp+'\nDepletion_ALLXY_'+qubit, y=1.0)


def calc_assignment_prob_matrix(combinations, digitized_data):

    assignment_prob_matrix = np.zeros((len(combinations), len(combinations)))

    for i, input_state in enumerate(combinations):
        for j, outcome in enumerate(combinations):
            first_key = next(iter(digitized_data))
            Check = np.ones(len(digitized_data[first_key][input_state]))
            for k, ch in enumerate(digitized_data.keys()):
                check = digitized_data[ch][input_state] == int(outcome[k])
                Check *= check

            assignment_prob_matrix[i][j] = sum(Check)/len(Check)

    return assignment_prob_matrix

class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for single-shot Multiplexed readout experiment.
    This new analysis now supports post-selection
    with two quadratures and 3 state readout.
    """
    def __init__(self,
                 qubits: list,
                 heralded_init: bool,
                 f_state: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubits = qubits
        self.heralded_init = heralded_init
        self.f_state = f_state
        
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        n_qubits = len(self.qubits)
        _cycle = 2**n_qubits
        states = ['0', '1']
        if self.f_state:
            _cycle = 3**n_qubits
            states = ['0', '1', '2']
        if self.heralded_init:
            _cycle *= 2
        combinations = [''.join(s) for s in itertools.product(states, repeat=n_qubits)]
        self.combinations = combinations
        # Sort acquisition channels
        _channels = [ name.decode() for name in self.raw_data_dict['value_names'] ]
        acq_channel_dict = { q : (None, None) for q in self.qubits }
        for q in self.qubits:
            _channel_I = [i for i, s in enumerate(_channels) if f'{q} I' in s]
            _channel_Q = [i for i, s in enumerate(_channels) if f'{q} Q' in s]
            assert len(_channel_I) == 1
            assert len(_channel_Q) == 1
            acq_channel_dict[q] = (_channel_I[0]+1, _channel_Q[0]+1)
        # Sort qubit shots per state
        raw_shots = {q:None for q in self.qubits}
        self.qoi = {}
        for q_idx, q in enumerate(self.qubits):
            raw_shots[q] = self.raw_data_dict['data'][:,acq_channel_dict[q]]
            self.proc_data_dict[q] = {}
            self.proc_data_dict[q]['shots_0_IQ'] = []
            self.proc_data_dict[q]['shots_1_IQ'] = []
            if self.f_state:
                self.proc_data_dict[q]['shots_2_IQ'] = []
            for i, comb in enumerate(combinations):
                if comb[q_idx] == '0':
                    self.proc_data_dict[q]['shots_0_IQ'] += list(raw_shots[q][i::_cycle])
                elif comb[q_idx] == '1':
                    self.proc_data_dict[q]['shots_1_IQ'] += list(raw_shots[q][i::_cycle])
                elif (comb[q_idx] == '2') and self.f_state:
                    self.proc_data_dict[q]['shots_2_IQ'] += list(raw_shots[q][i::_cycle])
            # Convert list into array
            self.proc_data_dict[q]['shots_0_IQ'] = np.array(self.proc_data_dict[q]['shots_0_IQ'])
            self.proc_data_dict[q]['shots_1_IQ'] = np.array(self.proc_data_dict[q]['shots_1_IQ'])
            if self.f_state:
                self.proc_data_dict[q]['shots_2_IQ'] = np.array(self.proc_data_dict[q]['shots_2_IQ'])
            # Rotate data along 01
            center_0 = np.mean(self.proc_data_dict[q]['shots_0_IQ'], axis=0)
            center_1 = np.mean(self.proc_data_dict[q]['shots_1_IQ'], axis=0)
            def rotate_and_center_data(I, Q, vec0, vec1, phi=0):
                vector = vec1-vec0
                angle = np.arctan(vector[1]/vector[0])
                rot_matrix = np.array([[ np.cos(-angle+phi),-np.sin(-angle+phi)],
                                       [ np.sin(-angle+phi), np.cos(-angle+phi)]])
                proc = np.array((I, Q))
                proc = np.dot(rot_matrix, proc)
                return proc.transpose()
            raw_shots[q] = rotate_and_center_data(
                    raw_shots[q][:,0], raw_shots[q][:,1], center_0, center_1)

            self.proc_data_dict[q]['Shots_0'] = []
            self.proc_data_dict[q]['Shots_1'] = []
            if self.f_state:
                self.proc_data_dict[q]['Shots_2'] = []
            for i, comb in enumerate(combinations):
                self.proc_data_dict[q][f'shots_{comb}'] = raw_shots[q][i::_cycle]
                if comb[q_idx] == '0':
                    self.proc_data_dict[q]['Shots_0'] += list(raw_shots[q][i::_cycle])
                elif comb[q_idx] == '1':
                    self.proc_data_dict[q]['Shots_1'] += list(raw_shots[q][i::_cycle])
                elif (comb[q_idx] == '2') and self.f_state:
                    self.proc_data_dict[q]['Shots_2'] += list(raw_shots[q][i::_cycle])
            # Convert list into array
            self.proc_data_dict[q]['Shots_0'] = np.array(self.proc_data_dict[q]['Shots_0'])
            self.proc_data_dict[q]['Shots_1'] = np.array(self.proc_data_dict[q]['Shots_1'])
            if self.f_state:
                self.proc_data_dict[q]['Shots_2'] = np.array(self.proc_data_dict[q]['Shots_2'])
            #####################################################
            # From this point onward raw shots has shape 
            # (nr_shots, nr_quadratures).
            # Post select based on heralding measurement result.
            #####################################################
            if self.heralded_init:
                pass # Not implemented yet        
            ##############################################################
            # From this point onward Shots_<i> contains post-selected
            # shots of state <i> and has shape (nr_ps_shots, nr_quadtrs).
            # Next we will analyze shots projected along axis and 
            # therefore use a single quadrature. shots_<i> will be used
            # to denote that array of shots.
            ##############################################################
            # Analyse data in quadrature of interest
            # (01 projection axis)
            ##############################################################
            shots_0 = self.proc_data_dict[q]['Shots_0'][:,0]
            shots_1 = self.proc_data_dict[q]['Shots_1'][:,0]
            # total number of shots (after postselection)
            n_shots_0 = len(shots_0)
            n_shots_1 = len(shots_1)
            # find range
            _all_shots = np.concatenate((shots_0, shots_1))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x1, n1 = np.unique(shots_1, return_counts=True)
            # Calculate fidelity and optimal threshold
            def _calculate_fid_and_threshold(x0, n0, x1, n1):
                """
                Calculate fidelity and threshold from histogram data:
                x0, n0 is the histogram data of shots 0 (value and occurences),
                x1, n1 is the histogram data of shots 1 (value and occurences).
                """
                # Build cumulative histograms of shots 0 
                # and 1 in common bins by interpolation.
                all_x = np.unique(np.sort(np.concatenate((x0, x1))))
                cumsum0, cumsum1 = np.cumsum(n0), np.cumsum(n1)
                ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
                necumsum0 = ecumsum0/np.max(ecumsum0)
                ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
                necumsum1 = ecumsum1/np.max(ecumsum1)
                # Calculate optimal threshold and fidelity
                F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
                opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
                opt_idx = int(round(np.average(opt_idxs)))
                F_assignment_raw = F_vs_th[opt_idx]
                threshold_raw = all_x[opt_idx]
                return F_assignment_raw, threshold_raw
            Fid_raw, threshold_raw = _calculate_fid_and_threshold(x0, n0, x1, n1)
            ######################
            # Fit data
            ######################
            def _fit_double_gauss(x_vals, hist_0, hist_1):
                '''
                Fit two histograms to a double gaussian with
                common parameters. From fitted parameters,
                calculate SNR, Pe0, Pg1, Teff, Ffit and Fdiscr.
                '''
                from scipy.optimize import curve_fit
                # Double gaussian model for fitting
                def _gauss_pdf(x, x0, sigma):
                    return np.exp(-((x-x0)/sigma)**2/2)
                global double_gauss
                def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
                    _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
                    return _dist0
                # helper function to simultaneously fit both histograms with common parameters
                def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
                    _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
                    _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
                    return np.concatenate((_dist0, _dist1))
                # Guess for fit
                pdf_0 = hist_0/np.sum(hist_0) # Get prob. distribution
                pdf_1 = hist_1/np.sum(hist_1) # 
                _x0_guess = np.sum(x_vals*pdf_0) # calculate mean
                _x1_guess = np.sum(x_vals*pdf_1) #
                _sigma0_guess = np.sqrt(np.sum((x_vals-_x0_guess)**2*pdf_0)) # calculate std
                _sigma1_guess = np.sqrt(np.sum((x_vals-_x1_guess)**2*pdf_1)) #
                _r0_guess = 0.01
                _r1_guess = 0.05
                _A0_guess = np.max(hist_0)
                _A1_guess = np.max(hist_1)
                p0 = [_x0_guess, _x1_guess, _sigma0_guess, _sigma1_guess, _A0_guess, _A1_guess, _r0_guess, _r1_guess]
                # Bounding parameters
                _x0_bound = (-np.inf,np.inf)
                _x1_bound = (-np.inf,np.inf)
                _sigma0_bound = (0,np.inf)
                _sigma1_bound = (0,np.inf)
                _r0_bound = (0,1)
                _r1_bound = (0,1)
                _A0_bound = (0,np.inf)
                _A1_bound = (0,np.inf)
                bounds = np.array([_x0_bound, _x1_bound, _sigma0_bound, _sigma1_bound, _A0_bound, _A1_bound, _r0_bound, _r1_bound])
                # Fit parameters within bounds
                popt, pcov = curve_fit(
                    _double_gauss_joint, bin_centers,
                    np.concatenate((hist_0, hist_1)),
                    p0=p0, bounds=bounds.transpose())
                popt0 = popt[[0,1,2,3,4,6]]
                popt1 = popt[[1,0,3,2,5,7]]
                # Calculate quantities of interest
                SNR = abs(popt0[0] - popt1[0])/((abs(popt0[2])+abs(popt1[2]))/2)
                P_e0 = popt0[5]*popt0[2]/(popt0[2]*popt0[5] + popt0[3]*(1-popt0[5]))
                P_g1 = popt1[5]*popt1[2]/(popt1[2]*popt1[5] + popt1[3]*(1-popt1[5]))
                # # Effective qubit temperature
                # h = 6.62607004e-34
                # kb = 1.38064852e-23
                # T_eff = h*self.qubit_freq/(kb*np.log((1-P_e0)/P_e0))
                # Fidelity from fit
                _x_data = np.linspace(*_range, 10001)
                _h0 = double_gauss(_x_data, *popt0)# compute distrubition from
                _h1 = double_gauss(_x_data, *popt1)# fitted parameters.
                Fid_fit, threshold_fit = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
                # Discrimination fidelity
                _h0 = double_gauss(_x_data, *popt0[:-1], 0)# compute distrubition without residual
                _h1 = double_gauss(_x_data, *popt1[:-1], 0)# excitation of relaxation.
                Fid_discr, threshold_discr = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
                # return results
                qoi = { 'SNR': SNR,
                        'P_e0': P_e0, 'P_g1': P_g1, 
                        # 'T_eff': T_eff, 
                        'Fid_fit': Fid_fit, 'Fid_discr': Fid_discr }
                return popt0, popt1, qoi
            # Histogram of shots for 0 and 1
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
            # Save data in processed data dictionary
            self.proc_data_dict[q]['n_shots_0'] = n_shots_0
            self.proc_data_dict[q]['n_shots_1'] = n_shots_1
            self.proc_data_dict[q]['bin_centers'] = bin_centers
            self.proc_data_dict[q]['h0'] = h0
            self.proc_data_dict[q]['h1'] = h1
            self.proc_data_dict[q]['popt0'] = popt0
            self.proc_data_dict[q]['popt1'] = popt1
            self.proc_data_dict[q]['threshold_raw'] = threshold_raw
            self.proc_data_dict[q]['F_assignment_raw'] = Fid_raw
            self.proc_data_dict[q]['F_fit'] = params_01['Fid_fit']
            self.proc_data_dict[q]['F_discr'] = params_01['Fid_discr']
            self.proc_data_dict[q]['residual_excitation'] = params_01['P_e0']
            self.proc_data_dict[q]['relaxation_events'] = params_01['P_g1']
            # self.proc_data_dict[q]['effective_temperature'] = params_01['T_eff']
            # Save quantities of interest
            self.qoi[q] = {}
            self.qoi[q]['SNR'] = params_01['SNR']
            self.qoi[q]['F_a'] = Fid_raw
            self.qoi[q]['F_d'] = params_01['Fid_discr']
            ############################################
            # If second state data is use classifier
            # to assign states in the IQ plane and 
            # calculate qutrit fidelity.
            ############################################
            if self.f_state:
                # Parse data for classifier
                Shots_0 = self.proc_data_dict[q]['Shots_0']
                Shots_1 = self.proc_data_dict[q]['Shots_1']
                Shots_2 = self.proc_data_dict[q]['Shots_2']
                data = np.concatenate((Shots_0, Shots_1, Shots_2))
                labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf = LinearDiscriminantAnalysis()
                clf.fit(data, labels)
                dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
                Fid_dict = {}
                for state, shots in zip([    '0',     '1',     '2'],
                                        [Shots_0, Shots_1, Shots_2]):
                    _res = clf.predict(shots)
                    _fid = np.mean(_res == int(state))
                    Fid_dict[state] = _fid
                Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
                # Get assignment fidelity matrix
                M = np.zeros((3,3))
                for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
                    for j, state in enumerate(['0', '1', '2']):
                        _res = clf.predict(shots)
                        M[i][j] = np.mean(_res == int(state))
                self.proc_data_dict[q]['classifier'] = clf
                self.proc_data_dict[q]['dec_bounds'] = dec_bounds
                self.proc_data_dict[q]['Fid_dict'] = Fid_dict
                self.qoi[q]['Fid_dict'] = Fid_dict
                self.qoi[q]['Assignment_matrix'] = M
                #########################################
                # Project data along axis perpendicular
                # to the decision boundaries.
                #########################################
                ############################
                # Projection along 10 axis.
                ############################
                # Rotate shots over 01 decision boundary axis
                shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
                shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
                # Take relavant quadrature
                shots_0 = shots_0[:,0]
                shots_1 = shots_1[:,0]
                n_shots_1 = len(shots_1)
                # find range
                _all_shots = np.concatenate((shots_0, shots_1))
                _range = (np.min(_all_shots), np.max(_all_shots))
                # Sort shots in unique values
                x0, n0 = np.unique(shots_0, return_counts=True)
                x1, n1 = np.unique(shots_1, return_counts=True)
                Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
                # Histogram of shots for 1 and 2
                h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
                h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
                # Save processed data
                self.proc_data_dict[q]['projection_01'] = {}
                self.proc_data_dict[q]['projection_01']['h0'] = h0
                self.proc_data_dict[q]['projection_01']['h1'] = h1
                self.proc_data_dict[q]['projection_01']['bin_centers'] = bin_centers
                self.proc_data_dict[q]['projection_01']['popt0'] = popt0
                self.proc_data_dict[q]['projection_01']['popt1'] = popt1
                self.proc_data_dict[q]['projection_01']['SNR'] = params_01['SNR']
                self.proc_data_dict[q]['projection_01']['Fid'] = Fid_01
                self.proc_data_dict[q]['projection_01']['threshold'] = threshold_01
                ############################
                # Projection along 12 axis.
                ############################
                # Rotate shots over 12 decision boundary axis
                shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
                shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
                # Take relavant quadrature
                shots_1 = shots_1[:,0]
                shots_2 = shots_2[:,0]
                n_shots_2 = len(shots_2)
                # find range
                _all_shots = np.concatenate((shots_1, shots_2))
                _range = (np.min(_all_shots), np.max(_all_shots))
                # Sort shots in unique values
                x1, n1 = np.unique(shots_1, return_counts=True)
                x2, n2 = np.unique(shots_2, return_counts=True)
                Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
                # Histogram of shots for 1 and 2
                h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
                h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
                # Save processed data
                self.proc_data_dict[q]['projection_12'] = {}
                self.proc_data_dict[q]['projection_12']['h1'] = h1
                self.proc_data_dict[q]['projection_12']['h2'] = h2
                self.proc_data_dict[q]['projection_12']['bin_centers'] = bin_centers
                self.proc_data_dict[q]['projection_12']['popt1'] = popt1
                self.proc_data_dict[q]['projection_12']['popt2'] = popt2
                self.proc_data_dict[q]['projection_12']['SNR'] = params_12['SNR']
                self.proc_data_dict[q]['projection_12']['Fid'] = Fid_12
                self.proc_data_dict[q]['projection_12']['threshold'] = threshold_12
                ############################
                # Projection along 02 axis.
                ############################
                # Rotate shots over 02 decision boundary axis
                shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
                shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
                # Take relavant quadrature
                shots_0 = shots_0[:,0]
                shots_2 = shots_2[:,0]
                n_shots_2 = len(shots_2)
                # find range
                _all_shots = np.concatenate((shots_0, shots_2))
                _range = (np.min(_all_shots), np.max(_all_shots))
                # Sort shots in unique values
                x0, n0 = np.unique(shots_0, return_counts=True)
                x2, n2 = np.unique(shots_2, return_counts=True)
                Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
                # Histogram of shots for 1 and 2
                h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
                h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
                # Save processed data
                self.proc_data_dict[q]['projection_02'] = {}
                self.proc_data_dict[q]['projection_02']['h0'] = h0
                self.proc_data_dict[q]['projection_02']['h2'] = h2
                self.proc_data_dict[q]['projection_02']['bin_centers'] = bin_centers
                self.proc_data_dict[q]['projection_02']['popt0'] = popt0
                self.proc_data_dict[q]['projection_02']['popt2'] = popt2
                self.proc_data_dict[q]['projection_02']['SNR'] = params_02['SNR']
                self.proc_data_dict[q]['projection_02']['Fid'] = Fid_02
                self.proc_data_dict[q]['projection_02']['threshold'] = threshold_02
        ############################################
        # Calculate Mux assignment fidelity matrix #
        ############################################
        # Get assignment fidelity matrix
        # M = np.zeros((len(self.combinations),len(self.combinations)))
        # # Calculate population vector for each input state
        # for i, comb in enumerate(self.combinations):
        #     _res = []
        #     # Assign shots for each qubit
        #     for q in self.qubits:
        #         _clf = self.proc_data_dict[q]['classifier']
        #         _res.append(_clf.predict(self.proc_data_dict[q][f'shots_{comb}']).astype(str))
        #     # <res> holds the outcome of shots for each qubit
        #     res = np.array(_res).T
        #     for j, comb in enumerate(self.combinations):
        #         M[i][j] = np.mean(np.logical_and(*(res == list(comb)).T))
        # Calculate population vector for each input state
        if self.f_state:
            _res = { q : {} for q in self.qubits}
            for i, comb_i in enumerate(self.combinations):
                # Assign shots for each qubit
                for q in self.qubits:
                    _clf = self.proc_data_dict[q]['classifier']
                    _res[q][comb_i] = np.array(_clf.predict(self.proc_data_dict[q][f'shots_{comb_i}']).astype(int))
                # <_res> holds the outcome of shots for each qubit
            M = calc_assignment_prob_matrix(self.combinations,_res)
            self.proc_data_dict['Mux_assignment_matrix'] = M

    def prepare_plots(self):
        self.axs_dict = {}
        for q in self.qubits:
            fig, ax = plt.subplots(figsize=(5,4), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict[f'main_{q}'] = ax
            self.figs[f'main_{q}'] = fig
            self.plot_dicts[f'main_{q}'] = {
                'plotfn': ssro_hist_plotfn,
                'ax_id': f'main_{q}',
                'bin_centers': self.proc_data_dict[q]['bin_centers'],
                'h0': self.proc_data_dict[q]['h0'],
                'h1': self.proc_data_dict[q]['h1'],
                'popt0': self.proc_data_dict[q]['popt0'], 
                'popt1': self.proc_data_dict[q]['popt1'],
                'threshold': self.proc_data_dict[q]['threshold_raw'],
                'Fid_raw': self.qoi[q]['F_a'],
                'Fid_fit': self.proc_data_dict[q]['F_fit'],
                'Fid_disc': self.qoi[q]['F_d'],
                'SNR': self.qoi[q]['SNR'],
                'P_e0': self.proc_data_dict[q]['residual_excitation'], 
                'P_g1': self.proc_data_dict[q]['relaxation_events'],
                'n_shots_0': self.proc_data_dict[q]['n_shots_0'],
                'n_shots_1': self.proc_data_dict[q]['n_shots_1'],
                'T_eff': None,
                'qubit': q,
                'timestamp': self.timestamp
            }

            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict[f'main2_{q}'] = ax
            self.figs[f'main2_{q}'] = fig
            self.plot_dicts[f'main2_{q}'] = {
                'plotfn': ssro_IQ_plotfn,
                'ax_id': f'main2_{q}',
                'shots_0': self.proc_data_dict[q]['shots_0_IQ'],
                'shots_1': self.proc_data_dict[q]['shots_1_IQ'],
                'shots_2': self.proc_data_dict[q]['shots_2_IQ'] if self.f_state else None,
                'shots_3': None,
                'qubit': q,
                'timestamp': self.timestamp
            }
            if self.f_state:
                fig = plt.figure(figsize=(8,4), dpi=100)
                axs = [fig.add_subplot(121),
                       fig.add_subplot(322),
                       fig.add_subplot(324),
                       fig.add_subplot(326)]
                # fig.patch.set_alpha(0)
                self.axs_dict[f'main3_{q}'] = axs[0]
                self.figs[f'main3_{q}'] = fig
                self.plot_dicts[f'main3_{q}'] = {
                    'plotfn': ssro_IQ_projection_plotfn,
                    'ax_id': f'main3_{q}',
                    'shots_0': self.proc_data_dict[q]['Shots_0'],
                    'shots_1': self.proc_data_dict[q]['Shots_1'],
                    'shots_2': self.proc_data_dict[q]['Shots_2'],
                    'projection_01': self.proc_data_dict[q]['projection_01'],
                    'projection_12': self.proc_data_dict[q]['projection_12'],
                    'projection_02': self.proc_data_dict[q]['projection_02'],
                    'classifier': self.proc_data_dict[q]['classifier'],
                    'dec_bounds': self.proc_data_dict[q]['dec_bounds'],
                    'Fid_dict': self.proc_data_dict[q]['Fid_dict'],
                    'qubit': q,
                    'timestamp': self.timestamp
                }
                fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                # fig.patch.set_alpha(0)
                self.axs_dict[f'Assignment_matrix_{q}'] = ax
                self.figs[f'Assignment_matrix_{q}'] = fig
                self.plot_dicts[f'Assignment_matrix_{q}'] = {
                    'plotfn': assignment_matrix_plotfn,
                    'ax_id': f'Assignment_matrix_{q}',
                    'M': self.qoi[q]['Assignment_matrix'],
                    'qubit': q,
                    'timestamp': self.timestamp
                }

        if self.f_state:

            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict[f'Mux_assignment_matrix'] = ax
            self.figs[f'Mux_assignment_matrix'] = fig
            self.plot_dicts[f'Mux_assignment_matrix'] = {
                'plotfn': mux_assignment_matrix_plotfn,
                'ax_id': 'Mux_assignment_matrix',
                'M': self.proc_data_dict['Mux_assignment_matrix'],
                'Qubits': self.qubits,
                'combinations': self.combinations,
                'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def mux_assignment_matrix_plotfn(
    M,
    Qubits,
    timestamp,
    combinations,
    ax, **kw):

    fig = ax.get_figure()

    im = ax.imshow(M*100, cmap='Reds', vmin=0, vmax=100)
    n = len(combinations)
    for i in range(n):
        for j in range(n):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.0f}'.format(c*100), va='center', ha='center',
                        color = 'white', size=8)
            elif abs(c)>.01:
                ax.text(i, j, '{:.0f}'.format(c*100), va='center', ha='center',
                        size=8)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    _labels = [''.join([f'{comb[i]}_\mathrm{{{Qubits[i]}}}' for i in range(3)]) for comb in combinations]
    ax.set_xticklabels([f'${label}$' for label in _labels], size=8, rotation=90)
    ax.set_yticklabels([f'${label}$' for label in _labels], size=8)
    ax.set_xlabel(f'Assigned state')
    ax.set_ylabel('Input state')
    cb = fig.colorbar(im, orientation='vertical', aspect=35)
    pos = ax.get_position()
    pos = [ pos.x0+.65, pos.y0, pos.width, pos.height ]
    fig.axes[-1].set_position(pos)
    cb.set_label('Assignment probability (%)', rotation=-90, labelpad=15)
    ax.set_title(f'{timestamp}\nMultiplexed qutrit assignment matrix {" ".join(Qubits)}')
