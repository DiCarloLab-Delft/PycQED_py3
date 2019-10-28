import lmfit
from uncertainties import ufloat
from pycqed.analysis import measurement_analysis as ma
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis.tools.data_manipulation import \
    populations_using_rate_equations
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, plot_fit, \
    make_anglemap, make_segmented_cmap
import matplotlib.pyplot as plt
from pycqed.analysis.fitting_models import CosFunc, Cos_guess, \
    avoided_crossing_freq_shift
from pycqed.analysis_v2.simple_analysis import Basic2DInterpolatedAnalysis

from pycqed.analysis.analysis_toolbox import color_plot
import scipy.cluster.hierarchy as hcluster

from matplotlib import colors
from copy import deepcopy
from pycqed.analysis.tools.plot_interpolation import interpolate_heatmap

import logging

log = logging.getLogger(__name__)


class Chevron_Analysis(ba.BaseDataAnalysis):
    def __init__(self, ts: str = None, label=None,
                 ch_idx=0,
                 coupling='g', min_fit_amp=0, auto=True):
        """
        Analyzes a Chevron and fits the avoided crossing.

        Parameters
        ----------
        ts: str
            timestamp of the datafile
        label: str
            label to find the datafile (optional)
        ch_idx: int
            channel to use when fitting the avoided crossing
        coupling: Enum("g", "J1", "J2")
            used to label the avoided crossing and calculate related quantities
        min_fit_amp:
            minimal maplitude of the fitted cosine for each line cut.
            Oscillations with a smaller amplitude will be ignored in the fit
            of the avoided crossing.
        auto: bool
            if True run all parts of the analysis.

        """
        super().__init__(do_fitting=True)
        self.ts = ts
        self.label = label
        self.coupling = coupling
        self.ch_idx = ch_idx
        self.min_fit_amp = min_fit_amp
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a = ma.MeasurementAnalysis(
            timestamp=self.ts, label=self.label, auto=False)
        a.get_naming_and_values_2D()
        a.finish()
        self.timestamps = [a.timestamp_string]
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['timestamp_string'] = a.timestamp
        for attr in ['sweep_points', 'sweep_points_2D', 'measured_values',
                     'parameter_names', 'parameter_units', 'value_names',
                     'value_units']:
            self.raw_data_dict[attr] = getattr(a, attr)
        self.raw_data_dict['folder'] = a.folder

    def process_data(self):
        self.proc_data_dict = OrderedDict()

        # select the relevant data
        x = self.raw_data_dict['sweep_points']
        t = self.raw_data_dict['sweep_points_2D']
        Z = self.raw_data_dict['measured_values'][self.ch_idx].T

        # fit frequencies to each individual cut (time trace)
        freqs = []
        freqs_std = []
        fit_results = []
        amps = []
        for xi, z in zip(x, Z.T):
            CosModel = lmfit.Model(CosFunc)
            CosModel.guess = Cos_guess
            pars = CosModel.guess(CosModel, z, t)
            fr = CosModel.fit(data=z, t=t, params=pars)
            amps.append(fr.params['amplitude'].value)
            freqs.append(fr.params['frequency'].value)
            freqs_std.append(fr.params['frequency'].stderr)
            fit_results.append(fr)
        # N.B. the fit results are not saved in self.fit_res as this would
        # bloat the datafiles.
        self.proc_data_dict['fit_results'] = np.array(fit_results)
        self.proc_data_dict['amp_fits'] = np.array(amps)
        self.proc_data_dict['freq_fits'] = np.array(freqs)
        self.proc_data_dict['freq_fits_std'] = np.array(freqs_std)

        # take a Fourier transform (nice for plotting)
        fft_data = abs(np.fft.fft(Z.T).T)
        fft_freqs = np.fft.fftfreq(len(t), d=t[1]-t[0])
        sort_vec = np.argsort(fft_freqs)

        fft_data_sorted = fft_data[sort_vec, :]
        fft_freqs_sorted = fft_freqs[sort_vec]
        self.proc_data_dict['fft_data_sorted'] = fft_data_sorted
        self.proc_data_dict['fft_freqs_sorted'] = fft_freqs_sorted

    def run_fitting(self):
        super().run_fitting()

        fit_mask = np.where(self.proc_data_dict['amp_fits'] > self.min_fit_amp)

        avoided_crossing_mod = lmfit.Model(avoided_crossing_freq_shift)
        # hardcoded guesses! Bad practice, needs a proper guess func
        avoided_crossing_mod.set_param_hint('a', value=3e9)
        avoided_crossing_mod.set_param_hint('b', value=-2e9)
        avoided_crossing_mod.set_param_hint('g', value=20e6, min=0)
        params = avoided_crossing_mod.make_params()

        self.fit_res['avoided_crossing'] = avoided_crossing_mod.fit(
            data=self.proc_data_dict['freq_fits'][fit_mask],
            flux=self.raw_data_dict['sweep_points'][fit_mask],
            params=params)

    def analyze_fit_results(self):
        self.proc_data_dict['quantities_of_interest'] = {}
        # Extract quantities of interest from the fit
        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']
        g = self.fit_res['avoided_crossing'].params['g']
        qoi['g'] = ufloat(g.value, g.stderr)

        self.coupling_msg = ''
        if self.coupling == 'J1':
            qoi['J1'] = qoi['g']
            qoi['J2'] = qoi['g']*np.sqrt(2)
            self.coupling_msg += r'Measured $J_1$ = {} MHz'.format(
                qoi['J1']*1e-6)+'\n'
            self.coupling_msg += r'Expected $J_2$ = {} MHz'.format(
                qoi['J2']*1e-6)
        elif self.coupling == 'J2':
            qoi['J1'] = qoi['g']/np.sqrt(2)
            qoi['J2'] = qoi['g']
            self.coupling_msg += r'Expected $J_1$ = {} MHz'.format(
                qoi['J1']*1e-6)+'\n'
            self.coupling_msg += r'Measured $J_2$ = {} MHz'.format(
                qoi['J2']*1e-6)
        else:
            self.coupling_msg += 'g = {}'.format(qoi['g'])

    def prepare_plots(self):
        for i, val_name in enumerate(self.raw_data_dict['value_names']):
            self.plot_dicts['chevron_{}'.format(val_name)] = {
                'plotfn': plot_chevron,
                'x': self.raw_data_dict['sweep_points'],
                'y': self.raw_data_dict['sweep_points_2D'],
                'Z': self.raw_data_dict['measured_values'][i].T,
                'xlabel': self.raw_data_dict['parameter_names'][0],
                'ylabel': self.raw_data_dict['parameter_names'][1],
                'zlabel': self.raw_data_dict['value_names'][i],
                'xunit': self.raw_data_dict['parameter_units'][0],
                'yunit': self.raw_data_dict['parameter_units'][1],
                'zunit': self.raw_data_dict['value_units'][i],
                'title': self.raw_data_dict['timestamp_string']+'\n' +
                'Chevron {}'.format(val_name)
            }

        self.plot_dicts['chevron_fft'] = {
            'plotfn': plot_chevron_FFT,
            'x': self.raw_data_dict['sweep_points'],
            'xunit': self.raw_data_dict['parameter_units'][0],
            'fft_freqs': self.proc_data_dict['fft_freqs_sorted'],
            'fft_data': self.proc_data_dict['fft_data_sorted'],
            'freq_fits': self.proc_data_dict['freq_fits'],
            'freq_fits_std': self.proc_data_dict['freq_fits_std'],
            'fit_res': self.fit_res['avoided_crossing'],
            'coupling_msg': self.coupling_msg,
            'title': self.raw_data_dict['timestamp_string']+'\n' +
            'Fourier transform of Chevron'}


def plot_chevron(x, y, Z, xlabel, xunit, ylabel, yunit,
                 zlabel, zunit,
                 title, ax, **kw):
    colormap = ax.pcolormesh(x, y, Z, cmap='viridis',  # norm=norm,
                             linewidth=0, rasterized=True,
                             # assumes digitized readout
                             vmin=0, vmax=1)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='5%', pad='2%')
    cbar = plt.colorbar(colormap, cax=cax, orientation='vertical')
    cax.set_ylabel('L1 (%)')

    set_ylabel(cax, zlabel, zunit)


def plot_chevron_FFT(x, xunit,  fft_freqs, fft_data, freq_fits, freq_fits_std,
                     fit_res, coupling_msg, title, ax, **kw):

    colormap = ax.pcolormesh(x,
                             fft_freqs, fft_data, cmap='viridis',  # norm=norm,
                             linewidth=0, rasterized=True, vmin=0, vmax=5)

    ax.errorbar(x=x, y=freq_fits, yerr=freq_fits_std, ls='--', c='r', alpha=.5,
                label='Extracted freqs')
    x_fine = np.linspace(x[0], x[-1], 200)
    plot_fit(x, fit_res, ax=ax, c='C1', label='Avoided crossing fit', ls=':')

    set_xlabel(ax, 'Flux bias', xunit)
    set_ylabel(ax, 'Frequency', 'Hz')
    ax.legend(loc=(1.05, .7))
    ax.text(1.05, 0.5, coupling_msg, transform=ax.transAxes)


class Conditional_Oscillation_Heatmap_Analysis(Basic2DInterpolatedAnalysis):
    """
    Intended for the analysis of CZ tuneup (theta_f, lambda_2) heatmaps
    The data can be from an experiment or simulation
    """
    def __init__(self,
                t_start: str = None,
                t_stop: str = None,
                label: str = '',
                data_file_path: str = None,
                close_figs: bool = True,
                options_dict: dict = None,
                extract_only: bool = False,
                do_fitting: bool = False,
                auto: bool = True,
                interp_method: str = 'linear',
                plt_orig_pnts: bool = True,
                plt_contour_phase: bool = True,
                plt_contour_L1: bool = False,
                plt_optimal_values: bool = True,
                plt_optimal_values_max: int = None,
                clims: dict = None,
                find_local_optimals: bool = True):

        self.plt_orig_pnts = plt_orig_pnts
        self.plt_contour_phase = plt_contour_phase
        self.plt_contour_L1 = plt_contour_L1
        self.plt_optimal_values = plt_optimal_values
        self.plt_optimal_values_max = plt_optimal_values_max
        self.clims = clims
        self.find_local_optimals = find_local_optimals

        cost_func_Names = {'Cost func', 'Cost func.', 'cost func',
        'cost func.', 'cost function', 'Cost function', 'Cost function value'}
        L1_names = {'L1', 'Leakage'}
        ms_names = {'missing fraction', 'Missing fraction', 'missing frac',
            'missing frac.', 'Missing frac', 'Missing frac.'}
        cond_phase_names = {'Cond phase', 'Cond. phase', 'Conditional phase',
            'cond phase', 'cond. phase', 'conditional phase'}
        offset_diff_names = {'offset difference', 'offset diff',
            'offset diff.', 'Offset difference', 'Offset diff',
            'Offset diff.'}

        # also account for possible underscores instead of a spaces between words
        allNames = [cost_func_Names, L1_names, ms_names, cond_phase_names,
            offset_diff_names]
        [self.cost_func_Names, self.L1_names, self.ms_names, self.cond_phase_names,
            self.offset_diff_names] = \
            [names.union({name.replace(' ', '_') for name in names})
                for names in allNames]

        cost_func_Names = {'Cost func', 'Cost func.', 'cost func',
        'cost func.', 'cost function', 'Cost function', 'Cost function value'}
        L1_names = {'L1', 'Leakage'}
        ms_names = {'missing fraction', 'Missing fraction', 'missing frac',
            'missing frac.', 'Missing frac', 'Missing frac.'}
        cond_phase_names = {'Cond phase', 'Cond. phase', 'Conditional phase',
            'cond phase', 'cond. phase', 'conditional phase'}
        offset_diff_names = {'offset difference', 'offset diff',
            'offset diff.', 'Offset difference', 'Offset diff',
            'Offset diff.'}

        # also account for possible underscores instead of a spaces between words
        allNames = [cost_func_Names, L1_names, ms_names, cond_phase_names,
            offset_diff_names]
        [self.cost_func_Names, self.L1_names, self.ms_names, self.cond_phase_names,
            self.offset_diff_names] = \
            [names.union({name.replace(' ', '_') for name in names})
                for names in allNames]

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            close_figs=close_figs,
            options_dict=options_dict,
            extract_only=extract_only,
            do_fitting=do_fitting,
            auto=auto,
            interp_method=interp_method
        )

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()
        anglemap = make_anglemap()

        for i, val_name in enumerate(self.proc_data_dict['value_names']):

            zlabel = '{} ({})'.format(val_name,
                                      self.proc_data_dict['value_units'][i])
            self.plot_dicts[val_name] = {
                'ax_id': val_name,
                'plotfn': color_plot,
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

            if self.clims is not None and val_name in self.clims.keys():
                self.plot_dicts[val_name]['clim'] = self.clims[val_name]

            if self.plt_orig_pnts:
                self.plot_dicts[val_name + '_non_interpolated'] = {
                    'ax_id': val_name,
                    'plotfn': scatter_pnts_overlay,
                    'x': self.proc_data_dict['x'],
                    'y': self.proc_data_dict['y']
                }

            if self.proc_data_dict['value_units'][i] == 'deg':
                self.plot_dicts[val_name]['cmap_chosen'] = anglemap

            if self.plt_contour_phase:
                # Find index of Conditional Phase
                z_cond_phase = None
                for j, val_name_j in enumerate(self.proc_data_dict['value_names']):
                    pass
                    if val_name_j in self.cond_phase_names:
                        z_cond_phase = self.proc_data_dict['interpolated_values'][j]
                        break

                if z_cond_phase is not None:
                    self.plot_dicts[val_name + '_cond_phase_contour'] = {
                        'ax_id': val_name,
                        'plotfn': contour_overlay,
                        'x': self.proc_data_dict['x_int'],
                        'y': self.proc_data_dict['y_int'],
                        'z': z_cond_phase,
                        'colormap': anglemap,
                        'cyclic_data': True,
                        'contour_levels': [90, 180, 270],
                        'vlim': (0, 360)
                    }
                else:
                    log.warning('No data found named {}'.format(self.cond_phase_names))

            if self.plt_contour_L1:
                # Find index of Leakage or Missing Fraction
                z_L1 = None
                for j, val_name_j in enumerate(self.proc_data_dict['value_names']):
                    if val_name_j in self.L1_names or val_name_j in self.ms_names:
                        z_L1 = self.proc_data_dict['interpolated_values'][j]
                        break

                if z_L1 is not None:
                    vlim = (self.proc_data_dict['interpolated_values'][j].min(),
                        self.proc_data_dict['interpolated_values'][j].max())

                    contour_levels = np.array([1, 5, 10])
                    # Leakage is estimated as (Missing fraction/2)
                    contour_levels = contour_levels if \
                        self.proc_data_dict['value_names'][j] in self.L1_names \
                        else 2 * contour_levels

                    self.plot_dicts[val_name + '_L1_contour'] = {
                        'ax_id': val_name,
                        'plotfn': contour_overlay,
                        'x': self.proc_data_dict['x_int'],
                        'y': self.proc_data_dict['y_int'],
                        'z': z_L1,
                        # 'unit': self.proc_data_dict['value_units'][j],
                        'contour_levels': contour_levels,
                        'vlim': vlim,
                        'colormap': 'hot',
                        'linestyles': 'dashdot'
                    }
                else:
                    log.warning('No data found named {}'.format(self.L1_names))

            if val_name in set().union(self.L1_names).union(self.ms_names)\
                    .union(self.offset_diff_names):
                self.plot_dicts[val_name]['cmap_chosen'] = 'hot'

            if self.plt_optimal_values and val_name in self.cost_func_Names:
                optimal_pnts = self.proc_data_dict['optimal_pnts']
                optimal_pars = 'Optimal Parameters:'
                for m, optimal_pnt in enumerate(optimal_pnts):
                    # Handy to limit the number of optimal pnts
                    # being printed when a lot of optimal values are found
                    if self.plt_optimal_values_max is not None and m > self.plt_optimal_values_max:
                        break
                    optimal_pars += '\nPoint #{}'.format(m)
                    for key, val in optimal_pnt.items():
                        optimal_pars += '\n{}: {:4.4f} {}'.format(key, val['value'], val['unit'])
                self.plot_dicts[val_name + '_optimal_pars'] = {
                    'ax_id': val_name,
                    'ypos': -0.25,
                    'xpos': 0,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'line_kws': {'alpha': 0},
                    'text_string': optimal_pars,
                    'horizontalalignment': 'left',
                    'verticalaligment': 'top',
                    'fontsize': 14
                }

            if self.find_local_optimals:
                # if np.size(self.proc_data_dict['optimal_idxs']) != 0:
                clusters_pnts_x = np.array([])
                clusters_pnts_y = np.array([])
                clusters_pnts_colors = np.array([])
                clusters_by_indx = self.proc_data_dict['clusters_by_indx']
                x = self.proc_data_dict['x']
                y = self.proc_data_dict['y']
                for l, cluster_by_indx in enumerate(clusters_by_indx):
                    clusters_pnts_x = np.concatenate((clusters_pnts_x, x[cluster_by_indx]))
                    clusters_pnts_y = np.concatenate((clusters_pnts_y, y[cluster_by_indx]))
                    clusters_pnts_colors = np.concatenate((clusters_pnts_colors,
                        np.full(np.shape(cluster_by_indx)[0], l)))
                self.plot_dicts[val_name + '_clusters'] = {
                    'ax_id': val_name,
                    'plotfn': scatter_pnts_overlay,
                    'x': clusters_pnts_x,
                    'y': clusters_pnts_y,
                    'color': None,
                    'edgecolors': 'black',
                    'marker': 'o',
                    'linewidth': 1,
                    'c': clusters_pnts_colors
                }

                x_optimal = x[self.proc_data_dict['optimal_idxs']]
                y_optimal = y[self.proc_data_dict['optimal_idxs']]

                self.plot_dicts[val_name + '_optimal_pnts_annotate'] = {
                    'ax_id': val_name,
                    'plotfn': annotate_pnts,
                    'txt': np.arange(np.shape(x_optimal)[0]),
                    'x': x_optimal,
                    'y': y_optimal
                }

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        self.proc_data_dict['interpolated_values'] = []
        for i in range(len(self.proc_data_dict['value_names'])):
            if self.proc_data_dict['value_units'][i] == 'deg':
                interp_method = 'deg'
            else:
                interp_method = self.interp_method

            x_int, y_int, z_int = interpolate_heatmap(
                self.proc_data_dict['x'],
                self.proc_data_dict['y'],
                self.proc_data_dict['measured_values'][i],
                interp_method=interp_method)
            self.proc_data_dict['interpolated_values'].append(z_int)

            if self.proc_data_dict['value_names'][i] in self.cost_func_Names:
                # Find the optimal point(s)
                x = self.proc_data_dict['x']
                y = self.proc_data_dict['y']
                z = self.proc_data_dict['measured_values'][i]

                if not self.find_local_optimals:
                    optimal_idxs = np.array([z.argmin()])
                    self.proc_data_dict['clusters_by_indx'] = None
                else:
                    where = [(name in self.cond_phase_names) for name in self.proc_data_dict['value_names']]
                    cond_phase_indx = np.where(where)[0][0]
                    cond_phase_arr = self.proc_data_dict['measured_values'][cond_phase_indx]

                    where = [(name in self.L1_names) for name in self.proc_data_dict['value_names']]
                    L1_indx = np.where(where)[0][0]
                    L1_arr = self.proc_data_dict['measured_values'][L1_indx]
                    optimal_idxs, clusters_by_indx = get_optimal_pnts_indxs(
                        theta_f_arr=x,
                        lambda_2_arr=y,
                        cost_func_arr=z,
                        cond_phase_arr=cond_phase_arr,
                        L1_arr=L1_arr)
                    self.proc_data_dict['clusters_by_indx'] = clusters_by_indx

                self.proc_data_dict['optimal_idxs'] = optimal_idxs

                self.proc_data_dict['optimal_pnts'] = []
                for optimal_idx in optimal_idxs:
                    optimal_pnt = {
                        self.proc_data_dict['xlabel']: {'value': x[optimal_idx], 'unit': self.proc_data_dict['xunit']},
                        self.proc_data_dict['ylabel']: {'value': y[optimal_idx], 'unit': self.proc_data_dict['yunit']}
                    }
                    for k, measured_value in enumerate(self.proc_data_dict['measured_values']):
                        optimal_pnt[self.proc_data_dict['value_names'][k]] = {'value': measured_value[optimal_idx], 'unit': self.proc_data_dict['value_units'][k]}
                    self.proc_data_dict['optimal_pnts'].append(optimal_pnt)

        self.proc_data_dict['x_int'] = x_int
        self.proc_data_dict['y_int'] = y_int

    def plot_text(self, pdict, axs):
        """
        Helper function that adds text to a plot
        Overriding here in order to make the text bigger
        and put it below the the cost function figure
        """
        pfunc = getattr(axs, pdict.get('func', 'text'))
        plot_text_string = pdict['text_string']
        plot_xpos = pdict.get('xpos', .98)
        plot_ypos = pdict.get('ypos', .98)
        fontsize = pdict.get('fontsize', 10)
        verticalalignment = pdict.get('verticalalignment', 'top')
        horizontalalignment = pdict.get('horizontalalignment', 'left')
        fontdict = {
            'horizontalalignment': horizontalalignment,
            'verticalalignment': verticalalignment
        }

        if fontsize is not None:
            fontdict['fontsize'] = fontsize

        # fancy box props is based on the matplotlib legend
        box_props = pdict.get('box_props', 'fancy')
        if box_props == 'fancy':
            box_props = self.fancy_box_props

        # pfunc is expected to be ax.text
        pfunc(x=plot_xpos, y=plot_ypos, s=plot_text_string,
              transform=axs.transAxes,
              bbox=box_props, fontdict=fontdict)


def scatter_pnts_overlay(
        x,
        y,
        fig=None,
        ax=None,
        transpose=False,
        color='w',
        edgecolors='gray',
        linewidth=0.5,
        marker='.',
        s=None,
        c=None,
        **kw):
    """
    Adds a scattered overlay of the provided data points
    x, and y are lists.
    Args:
        x (array [shape: n*1]):     x data
        y (array [shape: m*1]):     y data
        fig (Object):
            figure object
    """
    if ax is None:
        fig, ax = plt.subplots()

    if transpose:
        log.debug('Inverting x and y axis for non-interpolated points')
        ax.scatter(y, x, marker=marker,
            color=color, edgecolors=edgecolors, linewidth=linewidth, s=s, c=c)
    else:
        ax.scatter(x, y, marker=marker,
            color=color, edgecolors=edgecolors, linewidth=linewidth, s=s, c=c)

    return fig, ax


def contour_overlay(x, y, z, colormap, transpose=False,
        contour_levels=[90, 180, 270], vlim=(0, 360), fig=None,
        linestyles='dashed',
        cyclic_data=False,
        ax=None, **kw):
    """
    x, and y are lists, z is a matrix with shape (len(x), len(y))
    N.B. The contour overaly suffers from artifacts sometimes
    Args:
        x (array [shape: n*1]):     x data
        y (array [shape: m*1]):     y data
        z (array [shape: n*m]):     z data for the contour
        colormap (matplotlib.colors.Colormap or str): colormap to be used
        unit (str): 'deg' is a special case
        vlim (tuple(vmin, vmax)): required for the colormap nomalization
        fig (Object):
            figure object
    """
    if ax is None:
        fig, ax = plt.subplots()

    vmin = vlim[0]
    vmax = vlim[-1]

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    linewidth = 2
    fontsize = 'smaller'

    if transpose:
        y_tmp = np.copy(y)
        y = np.copy(x)
        x = y_tmp
        z = np.transpose(z)

    if cyclic_data:
        # Avoid contour plot artifact for cyclic data by removing the
        # data half way to the cyclic boundary
        minz = (vmin + np.min(contour_levels)) / 2
        maxz = (vmax + np.max(contour_levels)) / 2
        z = np.copy(z)  # don't change the original data
        z[(z < minz) | (z > maxz)] = np.nan

    c = ax.contour(x, y, z,
        levels=contour_levels, linewidths=linewidth, cmap=colormap,
        norm=norm, linestyles=linestyles)
    ax.clabel(c, fmt='%.1f', inline='True', fontsize=fontsize)

    return fig, ax


def annotate_pnts(txt, x, y,
        textcoords='offset points',
        ha='center',
        va='center',
        xytext=(0, 0),
        bbox=dict(boxstyle='circle, pad=0.2', fc='white', alpha=0.7),
        arrowprops=None,
        transpose=False,
        fig=None,
        ax=None,
        **kw):
    """
    A handy for loop for the ax.annotate
    """
    if ax is None:
        fig, ax = plt.subplots()

    if transpose:
        y_tmp = np.copy(y)
        y = np.copy(x)
        x = y_tmp

    for i, text in enumerate(txt):
            ax.annotate(text,
                xy=(x[i], y[i]),
                textcoords=textcoords,
                ha=ha,
                va=va,
                xytext=xytext,
                bbox=bbox)
    return fig, ax


def get_optimal_pnts_indxs(
        theta_f_arr,
        lambda_2_arr,
        cost_func_arr,
        cond_phase_arr,
        L1_arr,
        target_phase=180,
        phase_thr=5,
        L1_thr=0.5,
        clustering_thr=10):
    """
    target_phase and low L1 need to match roughtly cost function's minimums

    Args:
    cost_func_arr: bestter = lower values

    target_phase: unit = deg

    L1_thr: unit = %

    clustering_thr: unit = deg, represents distance between points on the
        landscape (lambda_2 gets normalized to [0, 360])
    """
    x = np.array(theta_f_arr)
    y = np.array(lambda_2_arr)

    # Normalize distance
    x_norm = x / 360.
    y_norm = y / (2 * np.pi)

    # Select points based low leakage and on how close to the
    # target_phase they are
    tolerances = [1, 2, 3, 4]
    for tol in tolerances:
        target_phase_min = target_phase - phase_thr * tol
        target_phase_max = target_phase + phase_thr * tol
        L1_thr *= tol
        sel = (cond_phase_arr > (target_phase_min)) & (cond_phase_arr < (target_phase_max))
        sel = sel * (L1_arr < L1_thr)
        selected_point_indx = np.where(sel)[0]
        if np.size(selected_point_indx) == 0:
            log.warning('No optimal points found with {} < target_phase < {} and L1 < {}.'.format(
                target_phase_min, target_phase_max, L1_thr))
            if tol == tolerances[-1]:
                return np.array([], dtype=int), np.array([], dtype=int)
            log.warning('Increasing tolerance for phase_thr and L1 to x{}.'.format(tol + 1))
        elif np.size(selected_point_indx) == 1:
            return np.array(selected_point_indx), np.array([selected_point_indx])
        else:
            x_filt = x_norm[selected_point_indx]
            y_filt = y_norm[selected_point_indx]
            break

    # Cluster points based on distance
    x_y_filt = np.transpose([x_filt, y_filt])
    thresh = clustering_thr / 360.
    clusters = hcluster.fclusterdata(x_y_filt, thresh, criterion="distance")

    cluster_id_min = np.min(clusters)
    cluster_id_max = np.max(clusters)
    clusters_by_indx = []
    optimal_idxs = []
    optimal_cost_func_values = []
    av_costfn_vals = []
    av_L1 = []
    for cluster_id in range(cluster_id_min, cluster_id_max + 1):
        cluster_indxs = np.where(clusters == cluster_id)
        indxs_in_orig_array = selected_point_indx[cluster_indxs]

        min_indx = np.argmin(cost_func_arr[indxs_in_orig_array])
        clusters_by_indx.append(indxs_in_orig_array)

        optimal_idxs.append(indxs_in_orig_array[min_indx])
        optimal_cost_func_values.append(cost_func_arr[indxs_in_orig_array[min_indx]])

        sq_dist = (x_norm - x_norm[min_indx])**2 + (y_norm - y_norm[min_indx])**2
        neighbors_indx = np.where(sq_dist < (thresh * 2.5)**2)
        av_L1.append(np.average(L1_arr[neighbors_indx]))
        av_costfn_vals.append(np.average(cost_func_arr[neighbors_indx]))

    w1 = 0.5 * np.array(av_costfn_vals) / np.min(av_costfn_vals)
    w2 = 0.5 * np.array(av_L1) / np.max(av_L1)

    sort_by = w1 + w2
    optimal_idxs = np.array(optimal_idxs)[np.argsort(sort_by)]
    clusters_by_indx = np.array(clusters_by_indx)[np.argsort(sort_by)]

    return optimal_idxs, clusters_by_indx
