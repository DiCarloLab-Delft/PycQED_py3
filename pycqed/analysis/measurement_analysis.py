import os
import logging
import numpy as np
import pickle
from collections import OrderedDict
import h5py
import matplotlib.lines as mlines
import matplotlib
from matplotlib import pyplot as plt
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods
import pycqed.measurement.hdf5_data as h5d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as optimize
from scipy import stats
import lmfit
from collections import Counter  # used in counting string fractions
import textwrap
from scipy.interpolate import interp1d
import pylab
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.utilities.general import SafeFormatter, format_value_string
from scipy.ndimage.filters import gaussian_filter
from importlib import reload
import math

# try:
#     import pygsti
# except ImportError as e:
#     if str(e).find('pygsti') >= 0:
#         logging.warning('Could not import pygsti')
#     else:
#         raise

from math import erfc
from scipy.signal import argrelmax, argrelmin
from scipy.constants import *
from copy import deepcopy
from pycqed.analysis.fit_toolbox import functions as func
from pprint import pprint

import pycqed.analysis.tools.plotting as pl_tools
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel,
                                            data_to_table_png,
                                            SI_prefix_and_scale_factor)

try:
    from nathan_plotting_tools import *
except:
    pass
from pycqed.analysis import composite_analysis as ca

# try:
#     import qutip as qtp

# except ImportError as e:
#     if str(e).find('qutip') >= 0:
#         logging.warning('Could not import qutip')
#     else:
#         raise

reload(dm_tools)

sfmt = SafeFormatter()


class MeasurementAnalysis(object):

    def __init__(self, TwoD=False, folder=None, auto=True,
                 cmap_chosen='viridis', no_of_columns=1, qb_name=None, **kw):
        if folder is None:
            self.folder = a_tools.get_folder(**kw)
        else:
            self.folder = folder
        self.load_hdf5data(**kw)
        self.fit_results = []
        self.cmap_chosen = cmap_chosen
        self.no_of_columns = no_of_columns

        # for retrieving correct values of qubit parameters from data file
        self.qb_name = qb_name

        # set line widths, marker sizes, tick sizes etc. from the kw
        self.set_plot_parameter_values(**kw)

        if auto is True:
            self.run_default_analysis(TwoD=TwoD, **kw)

    def set_plot_parameter_values(self, **kw):
        # dpi for plots
        self.dpi = kw.pop('dpi', 300)
        # font sizes
        self.font_size = kw.pop('font_size', 11)
        # line widths connecting data points
        self.line_width = kw.pop('line_width', 2)
        # lw of axes and text boxes
        self.axes_line_width = kw.pop('axes_line_width', 0.5)
        # tick lengths
        self.tick_length = kw.pop('tick_length', 4)
        # tick line widths
        self.tick_width = kw.pop('tick_width', 0.5)
        # marker size for data points
        self.marker_size = kw.pop('marker_size', None)
        # marker size for special points like
        self.marker_size_special = kw.pop('marker_size_special', 8)
        # peak freq., Rabi pi and pi/2 amplitudes
        self.box_props = kw.pop('box_props',
                                dict(boxstyle='Square', facecolor='white',
                                     alpha=0.8, lw=self.axes_line_width))

        self.tick_color = kw.get('tick_color', 'k')
        # tick label color get's updated in savefig
        self.tick_labelcolor = kw.get('tick_labelcolor', 'k')
        self.axes_labelcolor = kw.get('axes_labelcolor', 'k')

        params = {"ytick.color": self.tick_color,
                  "xtick.color": self.tick_color,
                  "axes.labelcolor": self.axes_labelcolor, }
        plt.rcParams.update(params)

    def load_hdf5data(self, folder=None, file_only=False, **kw):
        if folder is None:
            folder = self.folder
        self.h5filepath = a_tools.measurement_filename(folder)
        h5mode = kw.pop('h5mode', 'r+')
        self.data_file = h5py.File(self.h5filepath, h5mode)
        if not file_only:
            for k in list(self.data_file.keys()):
                if type(self.data_file[k]) == h5py.Group:
                    self.name = k
            self.g = self.data_file['Experimental Data']
            self.measurementstring = os.path.split(folder)[1]
            self.timestamp = os.path.split(os.path.split(folder)[0])[1] \
                + '/' + self.measurementstring[:6]
            self.timestamp_string = os.path.split(os.path.split(folder)[0])[1] \
                + '_' + self.measurementstring[:6]
            self.measurementstring = self.measurementstring[7:]
            self.default_plot_title = self.measurementstring
        return self.data_file

    def finish(self, close_file=True, **kw):
        if close_file:
            self.data_file.close()

    def analysis_h5data(self, name='analysis'):
        if not os.path.exists(os.path.join(self.folder, name + '.hdf5')):
            mode = 'w'
        else:
            mode = 'r+'
        return h5py.File(os.path.join(self.folder, name + '.hdf5'), mode)

    def default_fig(self, **kw):
        figsize = kw.pop('figsize', None)

        if figsize is None:
            # these are the standard figure sizes for PRL
            if self.no_of_columns == 1:
                figsize = (7, 4)
            elif self.no_of_columns == 2:
                figsize = (3.375, 2.25)
        else:
            pass

        return plt.figure(figsize=figsize, dpi=self.dpi, **kw)

    def default_ax(self, fig=None, *arg, **kw):
        if fig is None:
            fig = self.default_fig(*arg, **kw)
        ax = fig.add_subplot(111)

        ax.ticklabel_format(useOffset=False)
        return fig, ax

    def save_fig(self, fig, figname=None, xlabel='x',
                 ylabel='measured_values',
                 fig_tight=True, **kw):
        # N.B. this save_fig method is the one from the base
        # MeasurementAnalysis class
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)

        if type(plot_formats) == str:
            plot_formats = [plot_formats]

        for plot_format in plot_formats:
            if figname is None:
                if xlabel == 'x':
                    xlabel = self.sweep_name

                figname = (self.measurementstring + '_' + ylabel +
                           '_vs_' + xlabel + '.' + plot_format)
            else:
                figname = (figname + '.' + plot_format)
            self.savename = os.path.abspath(os.path.join(
                self.folder, figname))
            if fig_tight:
                try:
                    fig.tight_layout()
                except ValueError:
                    print('WARNING: Could not set tight layout')
            try:
                # Before saving some plotting properties are updated
                for ax in fig.axes:
                    plt.setp(ax.get_xticklabels(), color=self.tick_labelcolor)
                    plt.setp(ax.get_yticklabels(), color=self.tick_labelcolor)

                # This makes the background around the axes transparent
                fig.patch.set_alpha(0)
                # FIXME: the axes labels and unit rescaling could also be
                # repeated here as the last step before saving

                fig.savefig(
                    self.savename, dpi=self.dpi, format=plot_format,
                    bbox_inches='tight')
            except Exception as e:
                print(e)
                fail_counter = True
        if fail_counter:
            logging.warning('Figure "%s" has not been saved.' % self.savename)
        if close_fig:
            plt.close(fig)
        return

    def get_folder(self, timestamp=None, older_than=None, label='', **kw):
        suppress_printing = kw.pop('suppress_printing', False)
        if timestamp is not None:
            folder = a_tools.data_from_time(timestamp)
            if not suppress_printing:
                print('loaded "%s"' % (folder))
        elif older_than is not None:
            folder = a_tools.latest_data(older_than=older_than)
            if not suppress_printing:
                print('loaded "%s"' % (folder))
        else:
            folder = a_tools.latest_data(label)
            if not suppress_printing:
                print('loaded "%s"' % (folder))
        return folder

    def setup_figures_and_axes(self, main_figs=1):

        # The main figure
        for main_fig in range(main_figs):
            self.f = [self.default_fig() for fig in range(main_figs)]
            self.ax = [self.f[k].add_subplot(111) for k in range(main_figs)]
        val_len = len(self.value_names)
        if val_len == 4:
            if self.no_of_columns == 2:
                self.figarray, self.axarray = plt.subplots(
                    val_len, 1, figsize=(3.375, 2.25 ** len(self.value_names)),
                    dpi=self.dpi)
            else:
                self.figarray, self.axarray = plt.subplots(
                    val_len, 1, figsize=(7, 4 * len(self.value_names)),
                    dpi=self.dpi)
                # val_len, 1, figsize=(min(8*len(self.value_names), 11),
                #                      4*len(self.value_names)))
        else:
            if self.no_of_columns == 2:
                self.figarray, self.axarray = plt.subplots(
                    max(len(self.value_names), 1), 1,
                    figsize=(3.375, 2.25 * len(self.value_names)), dpi=self.dpi)
                # max(len(self.value_names), 1), 1,
                # figsize=(8, 4*len(self.value_names)))
            else:
                self.figarray, self.axarray = plt.subplots(
                    max(len(self.value_names), 1), 1,
                    figsize=(7, 4 * len(self.value_names)), dpi=self.dpi)
                # max(len(self.value_names), 1), 1,
                # figsize=(8, 4*len(self.value_names)))

        return tuple(self.f + [self.figarray] + self.ax + [self.axarray])

    def get_values(self, key):
        if key in self.get_key('sweep_parameter_names'):
            names = self.get_key('sweep_parameter_names')

            ind = names.index(key)
            values = self.g['Data'][()][:, ind]
        elif key in self.get_key('value_names'):
            names = self.get_key('value_names')
            ind = (names.index(key) +
                   len(self.get_key('sweep_parameter_names')))
            values = self.g['Data'][()][:, ind]
        else:
            values = self.g[key][()]  # changed deprecated self.g[key].value => self.g[key][()]
        # Makes sure all data is np float64
        return np.asarray(values, dtype=np.float64)

    def get_key(self, key):
        '''
        Returns an attribute "key" of the group "Experimental Data"
        in the hdf5 datafile.
        '''
        s = self.g.attrs[key]
        # converts byte type to string because of h5py datasaving
        if type(s) == bytes:
            s = s.decode('utf-8')
        # If it is an array of value decodes individual entries
        if type(s) == np.ndarray:
            s = [s.decode('utf-8') for s in s]
        return s

    def group_values(self, group_name):
        '''
        Returns values for group with the name "group_name" from the
        hdf5 data file.
        '''
        group_values = self.g[group_name].value
        return np.asarray(group_values, dtype=np.float64)

    def add_analysis_datagroup_to_file(self, group_name='Analysis'):
        if group_name in self.data_file:
            self.analysis_group = self.data_file[group_name]
        else:
            self.analysis_group = self.data_file.create_group(group_name)

    def add_dataset_to_analysisgroup(self, datasetname, data):
        try:
            self.analysis_group.create_dataset(
                name=datasetname, data=data)
        except:
            self.add_analysis_datagroup_to_file()
            try:
                self.analysis_group.create_dataset(
                    name=datasetname, data=data)
            except:
                del self.analysis_group[datasetname]
                self.analysis_group.create_dataset(
                    name=datasetname, data=data)

    def save_dict_to_analysis_group(self, save_dict: dict, group_name: str):
        """
        Saves a dictionary to the analysis_group in the hdf5 datafile
        corresponding to the experiment.
        Convenient for storing parameters extracted in the analysis.
        """
        if group_name not in self.analysis_group:
            dict_grp = self.analysis_group.create_group(group_name)
        else:
            dict_grp = self.analysis_group[group_name]

        for key, value in save_dict.items():
            dict_grp.attrs[key] = str(value)

    def save_fitted_parameters(self, fit_res, var_name, save_peaks=False,
                               weights=None):
        fit_name = 'Fitted Params ' + var_name
        if fit_name not in self.analysis_group:
            fit_grp = self.analysis_group.create_group(fit_name)
        else:
            fit_grp = self.analysis_group[fit_name]

        # fit_grp.attrs['Fit Report'] = \
        #     '\n'+'*'*80+'\n' + \
        #     fit_res.fit_report() + \
        #     '\n'+'*'*80 + '\n\n'
        fit_grp.attrs['Fit Report'] = \
            '\n' + '*' * 80 + '\n' + \
            lmfit.fit_report(fit_res) + \
            '\n' + '*' * 80 + '\n\n'

        fit_grp.attrs.create(name='chisqr', data=fit_res.chisqr)
        fit_grp.attrs.create(name='redchi', data=fit_res.redchi)
        fit_grp.attrs.create(name='var_name', data=var_name.encode('utf-8'))
        if fit_res.covar is not None:
            if 'covar' in list(fit_grp.keys()):
                del fit_grp['covar']
            fit_grp.create_dataset(name='covar', data=fit_res.covar)
        for parname, par in fit_res.params.items():
            try:
                par_group = fit_grp.create_group(parname)
            except:  # if it already exists overwrite existing
                par_group = fit_grp[parname]
            par_dict = vars(par)
            for val_name, val in par_dict.items():
                if val_name == '_val':
                    val_name = 'value'
                if val_name == 'correl' and val is not None:
                    try:
                        correl_group = par_group.create_group(val_name)
                    except:
                        correl_group = par_group[val_name]
                    for cor_name, cor_val in val.items():
                        correl_group.attrs.create(name=cor_name, data=cor_val)
                else:
                    try:
                        par_group.attrs.create(name=val_name, data=val)
                    except:
                        pass

        if save_peaks and hasattr(self, 'peaks'):
            if 'Peaks' not in fit_grp:
                peaks_grp = fit_grp.create_group('Peaks')
            else:
                peaks_grp = fit_grp['Peaks']
            for key, value in list(self.peaks.items()):
                if value is not None:
                    peaks_grp.attrs.create(name=key, data=value)
        if weights is not None:
            mean = np.mean(fit_res.data)
            std = np.std(fit_res.data)
            weight = ((fit_res.data - mean) / std) ** weights
            weighted_chisqr = np.sum(
                weight * (fit_res.data - fit_res.best_fit) ** 2)
            fit_grp.attrs.create(name='weighted_chisqr', data=weighted_chisqr)

    def save_computed_parameters(self, computed_params, var_name):
        """ Allows to save additional parameters computed from fit results,
        such as the pi pulse and pi/2 pulse amplitudes. Each will be
        saved as a new attribute in the FittedParams+var_name group created
        in "save_fitted_parameters."

        Input parameters:
            computed_params:       DICTIONARY of parameters to be saved; first
                                   value of main variable, then
                                   its statistical data such as stddev.
                                   Ex: {'piPulse':piPulse_val,
                                   'piPulse_std':piPulse_std_val}.
            var_name:              same var_name used in
                                  'save_fitted_parameters'
        """

        fit_name = 'Fitted Params ' + var_name
        if fit_name not in self.analysis_group:
            fit_grp = self.analysis_group.create_group(fit_name)
        else:
            fit_grp = self.analysis_group[fit_name]

        if len(computed_params) == 0:
            logging.warning('Nothing to save. Parameters dictionary is empty.')
        else:
            # for i in computed_params.items():
            #     fit_grp.attrs.create(name=i[0], data=i[1])
            for par_name, par_val in computed_params.items():
                if ('std' or 'stddev' or 'stderr') not in par_name:
                    try:
                        par_group = fit_grp.create_group(par_name)
                    except:  # if it already exists overwrite existing
                        par_group = fit_grp[par_name]
                    par_group.attrs.create(name=par_name, data=par_val)
                    # par_dict = vars(par_val)
                else:
                    fit_grp.attrs.create(name=par_name, data=par_val)

    def run_default_analysis(self, TwoD=False, close_file=True,
                             show=False, log=False, transpose=False, **kw):

        if TwoD is False:

            self.get_naming_and_values()
            self.sweep_points = kw.pop('sweep_points', self.sweep_points)
            # Preallocate the array of axes in the figure
            # Creates either a 2x2 grid or a vertical list

            if len(self.value_names) == 4:
                if self.no_of_columns == 2:
                    fig, axs = plt.subplots(
                        nrows=int(len(self.value_names) / 2), ncols=2,
                        figsize=(3.375, 2.25 * len(self.value_names)),
                        dpi=self.dpi)
                else:
                    fig, axs = plt.subplots(
                        nrows=len(self.value_names), ncols=1,
                        figsize=(7, 4 * len(self.value_names)), dpi=self.dpi)

            else:

                if self.no_of_columns == 2:
                    fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                            figsize=(3.375,
                                                     2.25 * len(self.value_names)),
                                            dpi=self.dpi)
                else:
                    fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                            figsize=(
                                                7, 4 * len(self.value_names)),
                                            dpi=self.dpi)
                # Add all the sweeps to the plot 1 by 1
                # indices are determined by it's shape/number of sweeps
            for i in range(len(self.value_names)):
                if len(self.value_names) == 1:
                    ax = axs
                elif self.no_of_columns == 1:
                    ax = axs[i]
                elif self.no_of_columns == 2:
                    ax = axs[i // 2, i % 2]
                else:
                    ax = axs[i]  # If not 2 or 4 just gives a list of plots
                if i != 0:
                    plot_title = ' '
                else:
                    plot_title = kw.pop('plot_title', self.measurementstring +
                                        '\n' + self.timestamp_string)
                try:
                    ax.ticklabel_format(useOffset=False)
                except AttributeError:
                    # only mpl scalar formatters support this feature
                    pass

                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=fig, ax=ax, log=log,
                                                xlabel=self.sweep_name,
                                                x_unit=self.sweep_unit[0],
                                                ylabel=self.ylabels[i],
                                                save=False)
                # fig.suptitle(self.plot_title)
            fig.subplots_adjust(hspace=0.5)
            if show:
                plt.show()

        elif TwoD is True:
            self.get_naming_and_values_2D()
            self.sweep_points = kw.pop('sweep_points', self.sweep_points)
            self.sweep_points_2D = kw.pop(
                'sweep_points_2D', self.sweep_points_2D)

            if len(self.value_names) == 4:
                if self.no_of_columns == 2:
                    fig, axs = plt.subplots(int(len(self.value_names) / 2), 2,
                                            figsize=(3.375,
                                                     2.25 * len(self.value_names)),
                                            dpi=self.dpi)
                else:
                    fig, axs = plt.subplots(max(len(self.value_names)), 1,
                                            figsize=(7,
                                                     4 * len(self.value_names)),
                                            dpi=self.dpi)
            else:
                if self.no_of_columns == 2:
                    fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                            figsize=(3.375,
                                                     2.25 * len(self.value_names)),
                                            dpi=self.dpi)
                else:
                    fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                            figsize=(7,
                                                     4 * len(self.value_names)),
                                            dpi=self.dpi)

            for i in range(len(self.value_names)):
                if len(self.value_names) == 1:
                    ax = axs
                elif len(self.value_names) == 2:
                    ax = axs[i % 2]
                elif len(self.value_names) == 4:
                    ax = axs[i // 2, i % 2]
                else:
                    ax = axs[i]  # If not 2 or 4 just gives a list of plots

                [fig, ax, colormap, cbar] = a_tools.color_plot(
                    x=self.sweep_points,
                    y=self.sweep_points_2D,
                    z=self.measured_values[i].transpose(),
                    plot_title=self.zlabels[i],
                    fig=fig, ax=ax,
                    xlabel=self.sweep_name,
                    x_unit=self.sweep_unit,
                    ylabel=self.sweep_name_2D,
                    y_unit=self.sweep_unit_2D,
                    zlabel=self.zlabels[i],
                    save=False,
                    transpose=transpose,
                    cmap_chosen=self.cmap_chosen,
                    **kw)

                ax.set_title(self.zlabels[i], y=1.05, size=self.font_size)
                ax.xaxis.label.set_size(self.font_size)
                ax.yaxis.label.set_size(self.font_size)
                ax.tick_params(labelsize=self.font_size,
                               length=self.tick_length, width=self.tick_width)
                cbar.set_label(self.zlabels[i], size=self.font_size)
                cbar.ax.tick_params(labelsize=self.font_size,
                                    length=self.tick_length,
                                    width=self.tick_width)

            fig.subplots_adjust(hspace=0.5)

            plot_title = sfmt.format('{measurement}\n{timestamp}',
                                     timestamp=self.timestamp_string,
                                     measurement=self.measurementstring)
            fig.text(0.5, 1, plot_title, fontsize=self.font_size,
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=ax.transAxes)
            if show:
                plt.show()

        self.save_fig(fig, fig_tight=True, **kw)

        if close_file:
            self.data_file.close()

    def get_naming_and_values(self):
        '''
        Works both for the 'old' 1D sweeps and the new datasaving format.
        The new datasaving format also works for nD sweeps but the loading is
        done in such a way that all the old analysis should keep working if
        the data is saved in this new format.
        '''

        if 'datasaving_format' in list(self.g.attrs.keys()):
            datasaving_format = self.get_key('datasaving_format')
        else:
            print('Using legacy data loading, assuming old formatting')
            datasaving_format = 'Version 1'

        if datasaving_format == 'Version 1':
            # Get naming
            self.sweep_name = self.get_key('sweep_parameter_name')
            self.sweep_unit = self.get_key('sweep_parameter_unit')

            self.value_names = self.get_key('value_names')
            value_units = self.get_key('value_units')

            # get values
            self.sweep_points = self.get_values(self.sweep_name)
            self.measured_values = []
            self.ylabels = []
            for i in range(len(self.value_names)):
                self.measured_values.append(
                    self.get_values(self.value_names[i]))
                self.ylabels.append(str(
                    self.value_names[i] + '(' + value_units[i] + ')'))
            self.xlabel = str(self.sweep_name + '(' + self.sweep_unit + ')')

        elif datasaving_format == 'Version 2':

            self.parameter_names = self.get_key('sweep_parameter_names')
            self.sweep_name = self.parameter_names[0]
            self.parameter_units = self.get_key('sweep_parameter_units')
            self.sweep_unit = self.parameter_units  # for legacy reasons
            self.value_names = self.get_key('value_names')
            self.value_units = self.get_key('value_units')

            # data is transposed first to allow the individual parameter or value
            # types to be read out using a single array index (no colons
            # required)
            self.data = self.get_values('Data').transpose()
            if len(self.parameter_names) == 1:
                self.sweep_points = self.data[0, :]
            else:
                self.sweep_points = self.data[0:len(self.parameter_names), :]
            self.measured_values = self.data[-len(self.value_names):, :]

            self.xlabel = self.parameter_names[0] + ' (' + \
                self.parameter_units[0] + ')'
            self.parameter_labels = [a + ' (' + b + ')' for a, b in zip(
                self.parameter_names,
                self.parameter_units)]

            self.ylabels = [a + ' (' + b + ')' for a, b in zip(self.value_names,
                                                               self.value_units)]

            if 'optimization_result' in self.g:
                self.optimization_result = OrderedDict({
                    'generation': self.g['optimization_result'][:, 0],
                    'evals': self.g['optimization_result'][:, 1],
                    'xfavorite': self.g['optimization_result'][:, 2:2 + len(self.parameter_names)],
                    'stds': self.g['optimization_result'][:,
                                                          2 + len(self.parameter_names):2 + 2 * len(self.parameter_names)],
                    'fbest': self.g['optimization_result'][:, -len(self.parameter_names) - 1],
                    'xbest': self.g['optimization_result'][:, -len(self.parameter_names):]})
        else:
            raise ValueError('datasaving_format "%s " not recognized'
                             % datasaving_format)

    def plot_results_vs_sweepparam(self, x, y, fig, ax, show=False, marker='-o',
                                   log=False, ticks_around=True, label=None,
                                   **kw):

        save = kw.get('save', False)
        font_size = kw.pop('font_size', None)
        if font_size is not None:
            self.font_size = font_size

        self.plot_title = kw.get('plot_title',
                                 self.measurementstring + '\n' +
                                 self.timestamp_string)

        # ax.set_title(self.plot_title)
        fig.text(0.5, 1, self.plot_title, fontsize=self.font_size,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 transform=ax.transAxes)

        # Plot:
        ax.plot(x, y, marker, markersize=self.marker_size,
                linewidth=self.line_width, label=label)
        if log:
            ax.set_yscale('log')

        # Adjust ticks
        # set axes labels format to scientific when outside interval [0.01,99]
        from matplotlib.ticker import ScalarFormatter
        fmt = ScalarFormatter()
        fmt.set_powerlimits((-4, 4))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        # Set the line width of the scientific notation exponent
        ax.xaxis.offsetText.set_fontsize(self.font_size)
        ax.yaxis.offsetText.set_fontsize(self.font_size)
        if ticks_around:
            ax.xaxis.set_tick_params(labeltop=False, top=True, direction='in')
            ax.yaxis.set_tick_params(labeltop=False, top=True, direction='in')
        ax.tick_params(axis='both', labelsize=self.font_size,
                       length=self.tick_length, width=self.tick_width)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(self.axes_line_width)

        # Set axis labels
        xlabel = kw.get('xlabel', None)
        ylabel = kw.get('ylabel', None)
        x_unit = kw.get('x_unit', None)
        y_unit = kw.get('y_unit', None)

        if xlabel is not None:
            set_xlabel(ax, xlabel, unit=x_unit)
            ax.xaxis.label.set_fontsize(self.font_size)
        if ylabel is not None:
            set_ylabel(ax, ylabel, unit=y_unit)
            ax.yaxis.label.set_fontsize(self.font_size)

        fig.tight_layout()

        if show:
            plt.show()
        if save:
            if log:
                # litle hack to only change savename if logarithmic
                self.save_fig(fig, xlabel=xlabel,
                              ylabel=(ylabel + '_log'), **kw)
            else:
                self.save_fig(fig, xlabel=xlabel, ylabel=ylabel, **kw)
        return

    def plot_complex_results(self, cmp_data, fig, ax, show=False, marker='.', **kw):
        '''
        Plot real and imaginary values measured vs a sweeped parameter
        Example: complex S21 of a resonator

        Author: Stefano Poletto
        Date: November 15, 2016
        '''
        save = kw.pop('save', False)
        self.plot_title = kw.pop('plot_title',
                                 textwrap.fill(self.timestamp_string + '_' +
                                               self.measurementstring, 40))

        xlabel = 'Real'
        ylabel = 'Imag'
        ax.set_title(self.plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(np.real(cmp_data), np.imag(cmp_data), marker)
        if show:
            plt.show()
        if save:
            self.save_fig(fig, xlabel=xlabel, ylabel=ylabel, **kw)

        return

    def plot_dB_from_linear(self, x, lin_amp, fig, ax, show=False, marker='.', **kw):
        '''
        Plot linear data in dB.
        This is usefull for measurements performed with VNA and Homodyne

        Author: Stefano Poletto
        Date: May 5, 2017
        '''
        save = kw.pop('save', False)
        self.plot_title = kw.pop('plot_title',
                                 textwrap.fill(self.timestamp_string + '_' +
                                               self.measurementstring, 40))

        xlabel = 'Freq'
        ylabel = 'Transmission (dB)'
        ax.set_title(self.plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        y_dB = 20 * np.log10(lin_amp)
        ax.plot(x, y_dB, marker)
        if show:
            plt.show()
        if save:
            self.save_fig(fig, xlabel=xlabel, ylabel=ylabel, **kw)

        return

    def get_naming_and_values_2D(self):
        '''
        This should also be adjusted for 2D.
        Data should directly be turned into a convenient
        Matrix.
        '''
        if 'datasaving_format' in list(self.g.attrs.keys()):
            datasaving_format = self.get_key('datasaving_format')
        else:
            print('Using legacy data loading, assuming old formatting')
            datasaving_format = 'Version 1'

        if datasaving_format == 'Version 1':
            # Get naming
            self.sweep_name = self.get_key('sweep_parameter_name')
            self.sweep_unit = self.get_key('sweep_parameter_unit')
            self.sweep_name_2D = self.get_key('sweep_parameter_2D_name')
            self.sweep_unit_2D = self.get_key('sweep_parameter_2D_unit')
            self.value_names = self.get_key('value_names')

            value_units = self.get_key('value_units')

            # get values
            self.sweep_points = self.get_values(self.sweep_name)
            self.sweep_points_2D = self.get_values(self.sweep_name_2D)
            self.measured_values = []
            self.zlabels = []
            for i in range(len(self.value_names)):
                self.measured_values.append(
                    self.get_values(self.value_names[i]))
                self.zlabels.append(str(
                    self.value_names[i] + '(' + value_units[i] + ')'))
            self.xlabel = str(self.sweep_name + '(' + self.sweep_unit + ')')
            self.ylabel = str(self.sweep_name_2D +
                              '(' + self.sweep_unit_2D + ')')

        elif datasaving_format == 'Version 2':

            self.parameter_names = self.get_key('sweep_parameter_names')
            self.parameter_units = self.get_key('sweep_parameter_units')
            self.sweep_name = self.parameter_names[0]
            self.sweep_name_2D = self.parameter_names[1]
            self.sweep_unit = self.parameter_units[0]
            self.sweep_unit_2D = self.parameter_units[1]

            self.value_names = self.get_key('value_names')
            self.value_units = self.get_key('value_units')

            self.data = self.get_values('Data').transpose()
            x = self.data[0]
            y = self.data[1]
            cols = np.unique(x).shape[0]

            # Adding np.nan for prematurely interupted experiments
            nr_missing_values = 0
            if len(x) % cols != 0:
                nr_missing_values = cols - len(x) % cols
            x = np.append(x, np.zeros(nr_missing_values) + np.nan)
            y = np.append(y, np.zeros(nr_missing_values) + np.nan)

            # X,Y,Z can be put in colormap directly
            self.X = x.reshape(-1, cols)
            self.Y = y.reshape(-1, cols)
            self.sweep_points = self.X[0]
            self.sweep_points_2D = self.Y.T[0]

            if len(self.value_names) == 1:
                z = self.data[2]
                z = np.append(z, np.zeros(nr_missing_values) + np.nan)
                self.Z = z.reshape(-1, cols)
                self.measured_values = [self.Z.T]
            else:
                self.Z = []
                self.measured_values = []
                for i in range(len(self.value_names)):
                    z = self.data[2 + i]
                    z = np.append(z, np.zeros(nr_missing_values) + np.nan)
                    Z = z.reshape(-1, cols)
                    self.Z.append(Z)
                    self.measured_values.append(Z.T)

            self.xlabel = self.parameter_names[0] + ' (' + \
                self.parameter_units[0] + ')'
            self.ylabel = self.parameter_names[1] + ' (' + \
                self.parameter_units[1] + ')'

            self.parameter_labels = [a + ' (' + b + ')' for a, b in zip(
                self.parameter_names,
                self.parameter_units)]

            self.zlabels = [a + ' (' + b + ')' for a, b in zip(self.value_names,
                                                               self.value_units)]

        else:
            raise ValueError('datasaving_format "%s " not recognized'
                             % datasaving_format)

    def get_best_fit_results(self, peak=False, weighted=False):
        if len(self.data_file['Analysis']) is 1:
            return list(self.data_file['Analysis'].values())[0]
        else:
            normalized_chisquares = {}
            haspeak_lst = []
            for key, item in self.data_file['Analysis'].items():
                if weighted is False:
                    chisqr = item.attrs['chisqr']
                else:
                    chisqr = item.attrs['weighted_chisqr']
                var = item.attrs['var_name']
                i = np.where(self.value_names == var)[0]  # relies
                # on looping order
                # of get_naming and variables, not the most robust way
                norm_chisq = chisqr / np.std(self.measured_values[i])
                normalized_chisquares[key] = norm_chisq

                if peak:
                    try:
                        if ('dip' in item['Peaks'].attrs) or \
                                ('peak' in item['Peaks'].attrs):
                            haspeak_lst += [key]
                    except:
                        pass
            if haspeak_lst != []:
                chisquares = {k: v for (k, v) in list(normalized_chisquares.items())
                              if k in haspeak_lst}
                best_key = min(chisquares, key=normalized_chisquares.get)
            else:
                best_key = min(normalized_chisquares,
                               key=normalized_chisquares.get)
            print('Best key: ', best_key)
            best_fit_results = self.data_file['Analysis'][best_key]
            return best_fit_results


class OptimizationAnalysis_v2(MeasurementAnalysis):

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        self.make_figures(**kw)
        if close_file:
            self.data_file.close()
        return

    def make_figures(self, **kw):
        for i in range(len(self.value_names)):
            base_figname = 'optimization of ' + self.value_names[i]
            if np.shape(self.sweep_points)[0] == 2:
                f, ax = plt.subplots()
                a_tools.color_plot_interpolated(
                    x=self.sweep_points[0], y=self.sweep_points[1],
                    z=self.measured_values[i], ax=ax,
                    zlabel=self.value_names[i])
                ax.plot(self.sweep_points[0],
                        self.sweep_points[1], linewidth=.5, marker='.',
                        alpha=.3, c='grey')
                ax.plot(self.sweep_points[0][-1], self.sweep_points[1][-1],
                        'x', markersize=5, c='w')
                plot_title = kw.pop('plot_title', textwrap.fill(
                    self.timestamp_string + '_' +
                    self.measurementstring, 40))
                ax.set_title(plot_title)
                set_xlabel(
                    ax, self.parameter_names[0], self.parameter_units[0])
                set_ylabel(
                    ax, self.parameter_names[1], self.parameter_units[1])
                self.save_fig(f, figname=base_figname, **kw)


class OptimizationAnalysis(MeasurementAnalysis):

    def run_default_analysis(self, close_file=True, show=False, plot_all=False, **kw):
        self.get_naming_and_values()
        try:
            optimization_method = self.data_file['Instrument settings']['MC'].attrs['optimization_method']
        except:
            optimization_method = 'Numerical'

        for i, meas_vals in enumerate(self.measured_values):
            if (not plot_all) & (i >= 1):
                break

            base_figname = optimization_method + ' optimization of ' + \
                self.value_names[i]
            # Optimizable value vs n figure
            fig1_type = '%s vs n' % self.value_names[i]
            figname1 = base_figname + '\n' + fig1_type
            savename1 = self.timestamp_string + '_' + base_figname + '_' + \
                fig1_type
            fig1, ax = self.default_ax()
            ax.plot(self.measured_values[i], marker='o')
            # assumes only one value exists because it is an optimization
            ax.set_xlabel('iteration (n)')
            ax.set_ylabel(self.ylabels[i])
            ax.set_title(self.timestamp_string + ' ' + figname1)

            textstr = 'Optimization converged to: \n   %s: %.3g %s' % (
                self.value_names[i], self.measured_values[0][-1],
                self.value_units[i])
            for j in range(len(self.parameter_names)):
                textstr += '\n   %s: %.4g %s' % (self.parameter_names[j],
                                                 self.sweep_points[j][-1],
                                                 self.parameter_units[j])

            # y coord 0.4 ensures there is no overlap for both maximizing and
            # minim
            if i == 0:
                ax.text(0.95, 0.4, textstr,
                        transform=ax.transAxes,
                        fontsize=11, verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=self.box_props)

            self.save_fig(fig1, figname=savename1, **kw)

        # Parameters vs n figure
        fig2, axarray = plt.subplots(len(self.parameter_names), 1,
                                     figsize=(8,
                                              4 * len(self.parameter_names)))
        fig2_type = 'parameters vs n'
        figname2 = base_figname + '\n' + fig2_type
        savename2 = self.timestamp_string + '_' + base_figname + '_' + \
            fig2_type

        if len(self.parameter_names) != 1:
            axarray[0].set_title(self.timestamp_string + ' ' + figname2)
            for i in range(len(self.parameter_names)):
                axarray[i].plot(self.sweep_points[i], marker='o')
                # assumes only one value exists because it is an optimization
                axarray[i].set_xlabel('iteration (n)')
                axarray[i].set_ylabel(self.parameter_labels[i])
        else:
            axarray.plot(self.sweep_points, marker='o')
            # assumes only one value exists because it is an optimization
            axarray.set_xlabel('iteration (n)')
            axarray.set_ylabel(self.parameter_labels[0])
            axarray.set_title(self.timestamp_string + ' ' + figname2)

        # Optimizable value vs paramter
        fig3, axarray = plt.subplots(len(self.parameter_names), 1,
                                     figsize=(8,
                                              4 * len(self.parameter_names)))
        fig3_type = '%s vs parameters' % self.value_names[0]
        figname3 = base_figname + '\n' + fig3_type
        savename3 = self.timestamp_string + '_' + base_figname + '_' + \
            fig3_type

        cm = plt.cm.get_cmap('RdYlBu')
        if len(self.parameter_names) != 1:
            axarray[0].set_title(self.timestamp_string + ' ' + figname3)
            for i in range(len(self.parameter_names)):
                # axarray[i].plot(self.sweep_points[i], self.measured_values[0],
                #                 linestyle='--', c='k')
                # assumes only one value exists because it is an optimization
                sc = axarray[i].scatter(self.sweep_points[i],
                                        self.measured_values[0],
                                        c=np.arange(len(self.sweep_points[i])),
                                        cmap=cm, marker='o', lw=0.1)
                axarray[i].set_xlabel(self.parameter_labels[i])
                axarray[i].set_ylabel(self.ylabels[0])
            fig3.subplots_adjust(right=0.8)
            # WARNING: Command does not work in ipython notebook
            cbar_ax = fig3.add_axes([.85, 0.15, 0.05, 0.7])
            cbar = fig3.colorbar(sc, cax=cbar_ax)
            cbar.set_label('iteration (n)')
        else:
            # axarray.plot(self.sweep_points, self.measured_values[0],
            #              linestyle='--', c='k')
            sc = axarray.scatter(self.sweep_points, self.measured_values[0],
                                 c=np.arange(len(self.sweep_points)),
                                 cmap=cm, marker='o', lw=0.1)
            # assumes only one value exists because it is an optimization
            axarray.set_xlabel(self.parameter_labels[0])
            axarray.set_ylabel(self.ylabels[0])
            axarray.set_title(self.timestamp_string + ' ' + figname3)
            cbar = fig3.colorbar(sc)
            cbar.set_label('iteration (n)')

        self.save_fig(fig2, figname=savename2, **kw)
        self.save_fig(fig3, figname=savename3, fig_tight=False, **kw)

        self.add_analysis_datagroup_to_file()
        if 'optimization_result' not in self.analysis_group:
            fid_grp = self.analysis_group.create_group('optimization_result')
        else:
            fid_grp = self.analysis_group['optimization_result']
        fid_grp.attrs.create(name=self.value_names[0],
                             data=self.measured_values[0, -1])

        for i in range(len(self.parameter_names)):
            fid_grp.attrs.create(name=self.parameter_names[i],
                                 data=self.sweep_points[i][-1])

        print('Optimization converged to:')
        prt_str = '    %s: %.4f %s' % (self.value_names[0],
                                       self.measured_values[0][-1],
                                       self.value_units[0])
        print(prt_str)

        for i in range(len(self.parameter_names)):
            prt_str = '    %s: %.4f %s' % (self.parameter_names[i],
                                           self.sweep_points[i][-1],
                                           self.parameter_units[i])
            print(prt_str)

        if show:
            plt.show()

        self.optimization_result = (self.sweep_points[:, -1],
                                    self.measured_values[:, -1])
        if close_file:
            self.data_file.close()


class TD_Analysis(MeasurementAnalysis):
    '''
    Parent class for Time Domain (TD) analysis. Contains functions for
    rotating and normalizing data based on calibration coordinates.
    '''

    def __init__(self, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 rotate_and_normalize=True, plot_cal_points=True,
                 for_ef=False, qb_name=None, **kw):
        self.NoCalPoints = NoCalPoints
        self.normalized_values = []
        self.normalized_cal_vals = []
        self.normalized_data_points = []
        self.cal_points = cal_points
        self.make_fig = make_fig
        self.rotate_and_normalize = rotate_and_normalize
        self.zero_coord = zero_coord
        self.one_coord = one_coord
        self.center_point = center_point
        self.plot_cal_points = plot_cal_points
        self.for_ef = for_ef

        super(TD_Analysis, self).__init__(qb_name=qb_name, **kw)

    # def run_default_analysis(self, close_file=True, **kw):
    #     self.get_naming_and_values()
    #     self.fit_data(**kw)
    #     self.make_figures(**kw)
    #     if close_file:
    #         self.data_file.close()
    #     return self.fit_res

    def rotate_and_normalize_data(self):
        if len(self.measured_values) == 1:
            # if only one weight function is used rotation is not required
            self.norm_data_to_cal_points()
            return

        if self.cal_points is None:
            # 42 is nr. of points in AllXY
            if len(self.measured_values[0]) == 42:
                self.corr_data, self.zero_coord, self.one_coord = \
                    a_tools.rotate_and_normalize_data(
                        data=self.measured_values[0:2],
                        zero_coord=self.zero_coord,
                        one_coord=self.one_coord,
                        cal_zero_points=list(range(2)),
                        cal_one_points=list(range(-8, -4)))
            elif len(self.measured_values[0]) == 21:
                self.corr_data, self.zero_coord, self.one_coord = \
                    a_tools.rotate_and_normalize_data(
                        data=self.measured_values[0:2],
                        zero_coord=self.zero_coord,
                        one_coord=self.one_coord,
                        cal_zero_points=list(range(1)),
                        cal_one_points=list(range(-4, -2)))
            else:
                self.corr_data, self.zero_coord, self.one_coord = \
                    a_tools.rotate_and_normalize_data(
                        data=self.measured_values[0:2],
                        zero_coord=self.zero_coord,
                        one_coord=self.one_coord,
                        cal_zero_points=list(range(1)),
                        cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    zero_coord=self.zero_coord,
                    one_coord=self.one_coord,
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])

    def norm_data_to_cal_points(self):
        # Used if data is based on only one weight
        if self.cal_points is None:
            # implicit in double point AllXY
            if len(self.measured_values[0]) == 42:
                cal_zero_points = list(range(2))
                cal_one_points = list(range(-8, -4))
            # implicit in single point AllXY
            elif len(self.measured_values[0]) == 21:
                cal_zero_points = list(range(1))
                cal_one_points = list(range(-4, -2))
            else:
                cal_zero_points = list(range(1))
                cal_one_points = list(range(-2, 0))
        else:
            cal_zero_points = self.cal_points[0]
            cal_one_points = self.cal_points[1]
        self.corr_data = a_tools.normalize_data_v3(
            self.measured_values[0],
            cal_zero_points=cal_zero_points,
            cal_one_points=cal_one_points)

    def run_default_analysis(self,
                             close_main_fig=True,
                             show=False, **kw):

        save_fig = kw.pop('save_fig', True)
        close_file = kw.pop('close_file', True)

        super().run_default_analysis(show=show,
                                     close_file=close_file, **kw)

        self.add_analysis_datagroup_to_file()

        norm = self.normalize_data_to_calibration_points(
            self.measured_values[0], calsteps=self.NoCalPoints, **kw)
        self.normalized_values = norm[0]
        self.normalized_data_points = norm[1]
        self.normalized_cal_vals = norm[2]

        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points'.encode('utf-8'))

        # Plotting
        if self.make_fig:
            self.fig, self.ax = self.default_ax()

            if self.for_ef:
                ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
            else:
                # ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'
                ylabel = r'$F$ $|1 \rangle$'

            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.normalized_values,
                                            fig=self.fig, ax=self.ax,
                                            xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=ylabel,
                                            marker='o-',
                                            save=False)
            if save_fig:
                if not close_main_fig:
                    # Hacked in here, good idea to only show the main fig but
                    # can be optimized somehow
                    self.save_fig(self.fig, figname=self.measurementstring,
                                  close_fig=False, **kw)
                else:
                    self.save_fig(self.fig, figname=self.measurementstring,
                                  **kw)

        if close_file:
            self.data_file.close()
        return

    def normalize_data_to_calibration_points(self, values, calsteps,
                                             save_norm_to_data_file=True,
                                             verbose=False, **kw):
        '''
        Rotates and normalizes the data based on the calibration points.

        values: array of measured values, uses only the length of this
        calsteps: number of points that corresponds to calibration points
        '''

        last_ge_pulse = kw.pop('last_ge_pulse', False)

        # Extract the indices of the cal points
        NoPts = len(values)
        if calsteps == 2:
            # both are I pulses
            if verbose:
                print('Only I calibration point')
            cal_zero_points = list(range(NoPts - int(calsteps), NoPts))
            cal_one_points = None
        elif calsteps == 4:
            if verbose:
                print('I and X180 calibration points')
            # first two cal points are I pulses, last two are X180 pulses
            cal_zero_points = list(range(NoPts - int(calsteps),
                                         int(NoPts - int(calsteps) / 2)))
            cal_one_points = list(range(int(NoPts - int(calsteps) / 2), NoPts))
        elif (calsteps == 6) and last_ge_pulse:
            # oscillations between |g>-|f>
            # use the I cal points (data[-6] and data[-5]) and
            # the X180_ef cal points (data[-2] and data[-1])
            if verbose:
                print('Oscillations between |g> - |f>')
                print('I and X180_ef calibration points')
            cal_zero_points = list(range(NoPts - int(calsteps),
                                         NoPts - int(2 * calsteps / 3)))
            cal_one_points = list(range(NoPts - int(calsteps / 3), NoPts))
        elif (calsteps == 6) and (not last_ge_pulse):
            # oscillations between |e>-|f>
            # use the X180 cal points (data[-4] and data[-3])
            # and the X180_ef cal points (data[-2] and data[-1])
            if verbose:
                print('Oscillations between |e> - |f>')
                print('X180 and X180_ef calibration points')
            cal_zero_points = list(range(NoPts - int(2 * calsteps / 3),
                                         NoPts - int(calsteps / 3)))
            cal_one_points = list(range(NoPts - int(calsteps / 3), NoPts))

        else:
            # assume no cal points were used
            if verbose:
                print('No calibration points')
            cal_zero_points = None
            cal_one_points = None

        # Rotate and normalize data
        if len(self.measured_values) == 1:
            # Only one quadrature was measured
            if cal_zero_points is None and cal_one_points is None:
                # a_tools.normalize_data_v3 does not work with 0 cal_points. Use
                # 4 cal_points.
                logging.warning('a_tools.normalize_data_v3 does not have support'
                                ' for 0 cal_points. Setting NoCalPoints to 4.')
                self.NoCalPoints = 4
                calsteps = 4
                cal_zero_points = list(range(NoPts - int(self.NoCalPoints),
                                             int(NoPts - int(self.NoCalPoints) / 2)))
                cal_one_points = list(
                    range(int(NoPts - int(self.NoCalPoints) / 2), NoPts))
            self.corr_data = a_tools.normalize_data_v3(
                self.measured_values[0], cal_zero_points, cal_one_points)
        else:
            if (calsteps == 6) and (not last_ge_pulse):
                # For this case we pass in the calibration data, not the indices
                # of the cal points.
                # zero_coord takes the cal_one_points and one_coord takes the
                # cal_zero_points because in a_tools.rotate_and_normalize_data
                # we must have for this case "calculate_rotation_matrix(
                # -(I_one-I_zero), -(Q_one-Q_zero))" in order to get
                # correct rotation
                zero_coord = [np.mean(self.measured_values[0][cal_one_points]),
                              np.mean(self.measured_values[1][cal_one_points])]
                one_coord = [np.mean(self.measured_values[0][cal_zero_points]),
                             np.mean(self.measured_values[1][cal_zero_points])]
                self.corr_data = a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    zero_coord=zero_coord, one_coord=one_coord)[0]
            else:
                self.corr_data = a_tools.rotate_and_normalize_data(
                    self.measured_values[0:2], cal_zero_points,
                    cal_one_points)[0]

        if save_norm_to_data_file:
            self.add_dataset_to_analysisgroup('Corrected data',
                                              self.corr_data)
            self.analysis_group.attrs.create(
                'corrected data based on',
                'calibration points'.encode('utf-8'))

        normalized_values = self.corr_data
        if calsteps == 0:
            normalized_data_points = normalized_values
            normalized_cal_vals = np.empty(0)
        else:
            normalized_data_points = normalized_values[:-int(calsteps)]
            normalized_cal_vals = normalized_values[-int(calsteps):]

        # Optionally, normalize to range [0,1]:
        # If we are calibrating only to a pulse with no amplitude
        # (i.e. do nothing), then manually
        # normalize the y axis. (Needed for Rabi for example)
        # if calsteps <= 2:
        #     max_min_distance = max(normalized_values) - \
        #                        min(normalized_values)
        #     normalized_values = (normalized_values -
        #                          min(normalized_values))/max_min_distance
        #     normalized_data_points = normalized_values[:-int(calsteps)]
        #     normalized_cal_vals = normalized_values[-int(calsteps):]

        return [normalized_values, normalized_data_points, normalized_cal_vals]

    def fit_data(*kw):
        '''
        Exists to be able to include it in the TD_Analysis run default
        '''
        pass


class chevron_optimization_v1(TD_Analysis):

    def __init__(self, cost_function=0, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 plot_cal_points=True, **kw):
        self.cost_function = cost_function
        super(chevron_optimization_v1, self).__init__(**kw)

    def run_default_analysis(self,
                             close_main_fig=True, **kw):
        super(chevron_optimization_v1, self).run_default_analysis(**kw)
        sweep_points_wocal = self.sweep_points[:-4]
        measured_values_wocal = self.measured_values[0][:-4]

        output_fft = np.real_if_close(np.fft.rfft(measured_values_wocal))
        ax_fft = np.fft.rfftfreq(len(measured_values_wocal),
                                 d=sweep_points_wocal[1] - sweep_points_wocal[0])
        order_mask = np.argsort(ax_fft)
        y = output_fft[order_mask]
        y = y / np.sum(np.abs(y))

        u = np.where(np.arange(len(y)) == 0, 0, y)
        array_peaks = a_tools.peak_finder(np.arange(len(np.abs(y))),
                                          np.abs(u),
                                          window_len=0)
        if array_peaks['peak_idx'] is None:
            self.period = 0.
            self.cost_value = 100.
        else:
            self.period = 1. / ax_fft[order_mask][array_peaks['peak_idx']]
            if self.period == np.inf:
                self.period = 0.
            if self.cost_function == 0:
                self.cost_value = -np.abs(y[array_peaks['peak_idx']])
            else:
                self.cost_value = self.get_cost_value(sweep_points_wocal,
                                                      measured_values_wocal)

    def get_cost_value(self, x, y):
        num_periods = np.floor(x[-1] / self.period)
        if num_periods == np.inf:
            num_periods = 0
        # sum of mins
        sum_min = 0.
        for i in range(int(num_periods)):
            sum_min += np.interp((i + 0.5) * self.period, x, y)
            # print(sum_min)

        # sum of maxs
        sum_max = 0.
        for i in range(int(num_periods)):
            sum_max += 1. - np.interp(i * self.period, x, y)
            # print(sum_max)

        return sum_max + sum_min


class chevron_optimization_v2(TD_Analysis):

    def __init__(self, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 plot_cal_points=True, **kw):
        super(chevron_optimization_v2, self).__init__(**kw)

    def run_default_analysis(self,
                             close_main_fig=True, **kw):
        super(chevron_optimization_v2, self).run_default_analysis(**kw)
        measured_values = a_tools.normalize_data_v3(self.measured_values[0])
        self.cost_value_1, self.period = self.sum_cost(self.sweep_points * 1e9,
                                                       measured_values)
        self.cost_value_2 = self.swap_cost(self.sweep_points * 1e9,
                                           measured_values)
        self.cost_value = [self.cost_value_1, self.cost_value_2]

        fig, ax = plt.subplots(1, figsize=(8, 6))

        min_idx, max_idx = self.return_max_min(self.sweep_points * 1e9,
                                               measured_values, 1)
        ax.plot(self.sweep_points * 1e9, measured_values, 'b-')
        ax.plot(self.sweep_points[min_idx] * 1e9,
                measured_values[min_idx], 'r*')
        ax.plot(self.sweep_points[max_idx] * 1e9,
                measured_values[max_idx], 'g*')
        ax.plot(self.period * 0.5, self.cost_value_2, 'b*', label='SWAP cost')
        ax.set_ylim(-0.05, 1.05)
        ax.text(35, 0.05, r'%.3f' % (self.cost_value_1), color='red')
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'$F |1\rangle$')

        ax.set_title('%s: Chevorn slice: Cost functions' %
                     self.timestamp_string)

        self.save_fig(fig, fig_tight=False, **kw)

    def analysis_on_fig(self, ax):
        measured_values = a_tools.normalize_data_v3(self.measured_values[0])
        self.cost_value_1, self.period = self.sum_cost(self.sweep_points * 1e9,
                                                       measured_values)
        self.cost_value_2 = self.swap_cost(self.sweep_points * 1e9,
                                           measured_values)
        self.cost_value = [self.cost_value_1, self.cost_value_2]

        min_idx, max_idx = self.return_max_min(self.sweep_points * 1e9,
                                               measured_values, 1)
        ax.plot(self.sweep_points * 1e9, measured_values, 'b-')
        ax.plot(self.sweep_points[min_idx] * 1e9,
                measured_values[min_idx], 'r*')
        ax.plot(self.sweep_points[max_idx] * 1e9,
                measured_values[max_idx], 'g*')
        ax.plot(self.period * 0.5, self.cost_value_2, 'b*', label='SWAP cost')
        ax.set_ylim(-0.05, 1.05)
        ax.text(35, 0.05, r'%.3f' % (self.cost_value_1), color='red')
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'$F |1\rangle$')

        ax.set_title('%s: Chevorn slice: Cost functions' %
                     self.timestamp_string)

    def return_max_min(self, data_x, data_y, window):
        x_points = data_x[:-4]
        y_points = a_tools.smooth(data_y[:-4], window_len=window)
        return argrelmin(y_points), argrelmax(y_points)

    def get_period(self, min_array, max_array):
        all_toghether = np.concatenate((min_array, max_array))
        sorted_vec = np.sort(all_toghether)
        diff = sorted_vec[1:] - sorted_vec[:-1]
        avg = np.mean(diff)
        std = np.std(diff)
        diff_filtered = np.where(np.abs(diff - avg) < std, diff, np.nan)
        diff_filtered = diff_filtered[~np.isnan(diff_filtered)]
        #     diff_filtered = diff
        return 2. * np.mean(diff_filtered), np.std(diff_filtered)

    def spec_power(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 1)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        f = 1. / period

        output_fft = np.real_if_close(np.fft.rfft(y_points))
        ax_fft = np.fft.rfftfreq(len(y_points),
                                 d=x_points[1] - x_points[0])
        order_mask = np.argsort(ax_fft)
        y = output_fft[order_mask]
        y = y / np.sum(np.abs(y))
        return -np.interp(f, ax_fft, np.abs(y))

    def sum_cost(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 4)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        num_periods = np.floor(x_points[-1] / period)

        sum_min = 0.
        for i in range(int(num_periods)):
            sum_min += np.interp((i + 0.5) * period, x_points, y_points)
        sum_max = 0.
        for i in range(int(num_periods)):
            sum_max += 1. - np.interp(i * period, x_points, y_points)

        return sum_max + sum_min, period

    def swap_cost(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 4)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        return np.interp(period * 0.5, x_points, y_points)


class Rabi_Analysis(TD_Analysis):
    """
    Analysis script for a Rabi measurement:
        if not separate_fits:
        1. The I and Q data are rotated and normalized based on the calibration
            points. In most analysis routines, the latter are typically 4:
            2 X180 measurements, and 2 identity measurements, which get averaged
            resulting in one X180 point and one identity point. However, the
            default for Rabi is 2 (2 identity measurements) because we typically
            do Rabi in order to find the correct amplitude for an X180 pulse.
            However, if a previous such value exists, this routine also accepts
            4 cal pts. If X180_ef pulse was also previously calibrated, this
            routine also accepts 6 cal pts.
        2. The normalized data is fitted to a cosine function.
        3. The pi-pulse and pi/2-pulse amplitudes are calculated from the fit.
        4. The normalized data, the best fit results, and the pi and pi/2
            pulses are plotted.

        else:
        Runs the original Rabi routine, which fits the two quadratures
        separately.

    Possible input parameters:
        auto              (default=True)
            automatically perform the entire analysis upon call
        label='Rabi'      (default=none?)
            Label of the analysis routine
        folder            (default=working folder)
            Working folder
        separate_fits     (default=False)
            if True, runs the original Rabi analysis routine which fits the
            I and Q points separately
        NoCalPoints       (default=0)
            Number of calibration points
        for_ef            (default=False)
            analyze for EF transition
        make_fig          (default=True)
            plot the fitted data
        print_fit_results (default=True)
            print the fit report
        show              (default=False)
            show the plots
        show_guess        (default=False)
            plot with initial guess values
        print_parameters   (default=True)
            print the pi&piHalf pulses amplitudes
        plot_amplitudes   (default=True)
            plot the pi&piHalf pulses amplitudes
        plot_errorbars    (default=True)
            plot standard error for each sample point
        close_file        (default=True)
            close the hdf5 file
        no_of_columns     (default=1)
            number of columns in your paper; figure sizes will be adjusted
            accordingly (1 col: figsize = ( 7in , 4in ) 2 cols: figsize =
            ( 3.375in , 2.25in ), PRL guidelines)
        verbose           (default=False)

    """

    def __init__(self, label='Rabi', qb_name=None, NoCalPoints=0, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'

        super().__init__(qb_name=qb_name,
                         NoCalPoints=NoCalPoints, **kw)

    def fit_data(self, print_fit_results=True, verbose=False, separate_fits=False):

        self.fit_res = [''] * self.nr_quadratures  # for legacy reasons

        if not separate_fits:

            cos_mod = fit_mods.CosModel
            # Find guess values
            # Frequency guess
            fft_of_data = np.fft.fft(self.normalized_data_points, norm='ortho')
            power_spectrum = np.abs(fft_of_data) ** 2
            index_of_fourier_maximum = np.argmax(
                power_spectrum[1:len(fft_of_data) // 2]) + 1

            top_x_val = np.take(self.sweep_points,
                                np.argmax(self.normalized_data_points))
            bottom_x_val = np.take(self.sweep_points,
                                   np.argmin(self.normalized_data_points))

            if index_of_fourier_maximum == 1:
                if verbose:
                    print('Initial guesses obtained by assuming the data trace '
                          'is between one half and one period of the cosine.')
                freq_guess = 1.0 / (2.0 * np.abs(bottom_x_val - top_x_val))
            else:
                if verbose:
                    print('Initial guesses obtained from fft of data.')
                fft_scale = 1.0 / (self.sweep_points[-1] -
                                   self.sweep_points[0])
                freq_guess = fft_scale * index_of_fourier_maximum

            # Amplitude guess
            diff = 0.5 * (max(self.normalized_data_points) -
                          min(self.normalized_data_points))
            amp_guess = -diff

            # phase guess --> NOT NEEDED because in cal pts calibration we make sure
            #                 Rabi trace starts at (closest to) zero
            # phase_guess = np.angle(fft_of_data[index_of_fourier_maximum])
            # if phase_guess<0:
            #     phase_guess=-phase_guess

            # Offset guess
            if np.abs(np.abs(min(self.normalized_data_points)) -
                      np.abs(max(self.normalized_data_points))) < 3:
                offset_guess = (min(self.normalized_data_points) +
                                max(self.normalized_data_points)) / 2
            elif np.abs(min(self.normalized_data_points)) > \
                    np.abs(max(self.normalized_data_points)):
                offset_guess = (min(self.normalized_data_points) -
                                max(self.normalized_data_points)) / 2
            else:
                offset_guess = (max(self.normalized_data_points) -
                                min(self.normalized_data_points)) / 2

            # Set up fit parameters and perform fit
            cos_mod.set_param_hint('amplitude',
                                   value=amp_guess,
                                   vary=True)
            cos_mod.set_param_hint('phase',
                                   value=0,
                                   vary=False)
            cos_mod.set_param_hint('frequency',
                                   value=freq_guess,
                                   vary=True,
                                   min=(
                                       1 / (100 * self.sweep_pts_wo_cal_pts[-1])),
                                   max=(20 / self.sweep_pts_wo_cal_pts[-1]))
            cos_mod.set_param_hint('offset',
                                   value=offset_guess,
                                   vary=True)
            cos_mod.set_param_hint('period',
                                   expr='1/frequency',
                                   vary=False)
            self.params = cos_mod.make_params()
            self.fit_result = cos_mod.fit(data=self.normalized_data_points,
                                          t=self.sweep_pts_wo_cal_pts,
                                          params=self.params)

            init_data_diff = np.abs(self.fit_result.init_fit[0] -
                                    self.normalized_data_points[0])
            if False:#(self.fit_result.chisqr > .35) or (init_data_diff > offset_guess):
                logging.warning('Fit did not converge, varying phase.')

                fit_res_lst = []

                for phase_estimate in np.linspace(0, 2 * np.pi, 8):
                    cos_mod.set_param_hint('phase',
                                           value=phase_estimate,
                                           vary=True)
                    self.params = cos_mod.make_params()
                    fit_res_lst += [cos_mod.fit(
                        data=self.normalized_data_points,
                        t=self.sweep_pts_wo_cal_pts,
                        params=self.params)]

                chisqr_lst = [fit_res.chisqr for fit_res in fit_res_lst]
                self.fit_result = fit_res_lst[np.argmin(chisqr_lst)]

            for i in range(self.nr_quadratures):  # for legacy reasons
                self.fit_res[i] = self.fit_result

            try:
                self.add_analysis_datagroup_to_file()
                self.save_fitted_parameters(self.fit_result,
                                            var_name=self.value_names[0])
            except Exception as e:
                logging.warning(e)

            if print_fit_results:
                print(self.fit_result.fit_report())

        else:
            model = fit_mods.lmfit.Model(fit_mods.CosFunc)
            if self.nr_quadratures != 1:
                # It would be best to do 1 fit to both datasets but since it is
                # easier to do just one fit we stick to that.
                # We make an initial guess of the Rabi period using both
                # quadratures
                data = np.sqrt(self.measured_values[0] ** 2 +
                               self.measured_values[1] ** 2)

                params = fit_mods.Cos_guess(
                    model, data=data, t=self.sweep_points)
                fitRes = model.fit(
                    data=data,
                    t=self.sweep_points,
                    params=params)
                freq_guess = fitRes.values['frequency']
            for i in range(self.nr_quadratures):
                model = fit_mods.lmfit.Model(fit_mods.CosFunc)
                params = fit_mods.Cos_guess(model, data=self.measured_values[i],
                                            t=self.sweep_points)
                if self.nr_quadratures != 1:
                    params['frequency'].value = freq_guess
                self.fit_res[i] = model.fit(
                    data=self.measured_values[i],
                    t=self.sweep_points,
                    params=params)

                try:
                    self.add_analysis_datagroup_to_file()
                    self.save_fitted_parameters(fit_res=self.fit_res[i],
                                                var_name=self.value_names[i])
                except Exception as e:
                    logging.warning(e)

            if print_fit_results:
                for fit_res in self.fit_res:
                    print(fit_res.fit_report())

    def run_default_analysis(self, show=False,
                             close_file=False, **kw):

        super().run_default_analysis(show=show,
                                     close_file=close_file,
                                     close_main_figure=True,
                                     save_fig=False, **kw)

        show_guess = kw.get('show_guess', False)
        plot_amplitudes = kw.get('plot_amplitudes', True)
        plot_errorbars = kw.get('plot_errorbars', False)
        print_fit_results = kw.get('print_fit_results', False)
        separate_fits = kw.get('separate_fits', False)

        self.nr_quadratures = len(self.ylabels)  # for legacy reasons
        # Create new sweep points without cal pts variable. Needed here because
        # we may have 0 cal pts, so writing self.sweep_points[:-self.NoCalPoints]
        # will give an error if self.NoCalPoints==0.
        self.sweep_pts_wo_cal_pts = deepcopy(self.sweep_points)
        if self.NoCalPoints is not 0:
            self.sweep_pts_wo_cal_pts = \
                self.sweep_pts_wo_cal_pts[:-self.NoCalPoints]

        # get the fit results (lmfit.ModelResult) and save them
        self.fit_data(print_fit_results, separate_fits=separate_fits)

        # if not separate_fits, get the computed pi and piHalf amplitudes
        # and save them
        if not separate_fits:
            self.get_amplitudes(**kw)
            self.save_computed_parameters(self.rabi_amplitudes,
                                          var_name=self.value_names[0])

        # Plot results
        if self.make_fig:
            self.make_figures(show=show, show_guess=show_guess,
                              plot_amplitudes=plot_amplitudes,
                              plot_errorbars=plot_errorbars,
                              separate_fits=separate_fits)

        if close_file:
            self.data_file.close()

        return self.fit_result

    def make_figures(self, show=False, show_guess=False, plot_amplitudes=True,
                     plot_errorbars=True, separate_fits=False, **kw):

        if not separate_fits:
            pi_pulse = self.rabi_amplitudes['piPulse']
            pi_half_pulse = self.rabi_amplitudes['piHalfPulse']

            # Get previously measured values from the data file
            instr_set = self.data_file['Instrument settings']
            try:
                if self.for_ef:
                    pi_pulse_old = float(
                        instr_set[self.qb_name].attrs['amp180_ef'])
                    pi_half_pulse_old = \
                        pi_pulse_old * \
                        float(instr_set[self.qb_name].attrs['amp90_scale_ef'])
                else:
                    pi_pulse_old = float(
                        instr_set[self.qb_name].attrs['amp180'])
                    pi_half_pulse_old = \
                        pi_pulse_old * \
                        float(instr_set[self.qb_name].attrs['amp90_scale'])
                old_vals = '\n  $\pi-Amp_{old}$ = %.3g ' % (pi_pulse_old) + \
                           self.parameter_units[0] + \
                           '\n$\pi/2-Amp_{old}$ = %.3g ' % (pi_half_pulse_old) + \
                           self.parameter_units[0]
            except(TypeError, KeyError, ValueError):
                logging.warning('qb_name is None. Default value qb_name="qb" is '
                                'used. Old parameter values will not be retrieved.')
                old_vals = ''

            textstr = ('  $\pi-Amp$ = %.3g ' % (pi_pulse) + self.parameter_units[0] +
                       ' $\pm$ (%.3g) ' % (self.rabi_amplitudes['piPulse_std']) +
                       self.parameter_units[0] +
                       '\n$\pi/2-Amp$ = %.3g ' % (pi_half_pulse) +
                       self.parameter_units[0] +
                       ' $\pm$ (%.3g) ' % (self.rabi_amplitudes['piHalfPulse_std']) +
                       self.parameter_units[0] + old_vals)

            self.fig.text(0.5, 0, textstr,
                          transform=self.ax.transAxes, fontsize=self.font_size,
                          verticalalignment='top',
                          horizontalalignment='center', bbox=self.box_props)

            # Used for plotting the fit (line 1776)
            best_vals = self.fit_result.best_values

            def cos_fit_func(a): return fit_mods.CosFunc(
                a,
                amplitude=best_vals['amplitude'],
                frequency=best_vals['frequency'],
                phase=best_vals['phase'],
                offset=best_vals['offset'])

            # Plot error bars
            if plot_errorbars:
                a_tools.plot_errorbars(self.sweep_pts_wo_cal_pts,
                                       self.normalized_data_points,
                                       ax=self.ax, only_bars=True,
                                       linewidth=self.axes_line_width,
                                       marker='none',
                                       markersize=self.marker_size)

            # Plot with initial guess
            if show_guess:
                self.ax.plot(self.sweep_pts_wo_cal_pts,
                             self.fit_result.init_fit, 'k--', linewidth=self.line_width)

            # Plot the calculated pi and pi/2 amplitudes
            if plot_amplitudes:
                piPulse_fit = cos_fit_func(pi_pulse)
                piHalfPulse_fit = cos_fit_func(pi_half_pulse)

                # Plot 2 horizontal lines for piAmpl and piHalfAmpl
                self.ax.plot([min(self.sweep_points),
                              max(self.sweep_points)],
                             [piPulse_fit, piPulse_fit], 'k--',
                             linewidth=self.axes_line_width)
                self.ax.plot([min(self.sweep_points),
                              max(self.sweep_points)],
                             [piHalfPulse_fit, piHalfPulse_fit], 'k--',
                             linewidth=self.axes_line_width)

                # Plot two points for the pi and piHalf pulses
                self.ax.plot(pi_pulse, piPulse_fit, 'ro',
                             markersize=self.marker_size_special)
                self.ax.plot(pi_half_pulse, piHalfPulse_fit, 'ro',
                             markersize=self.marker_size_special)

            # Plot with best fit results
            x = np.linspace(self.sweep_points[0],
                            self.sweep_pts_wo_cal_pts[-1],
                            len(self.sweep_points) * 100)
            y = cos_fit_func(x)
            self.ax.plot(x, y, 'r-', linewidth=self.line_width)

            # display figure
            if show:
                plt.show()
            self.ax.set_ylabel('V_homodyne (a.u)')
            # save figure
            self.save_fig(self.fig, figname=self.measurementstring + '_Rabi_fit',
                          **kw)

        else:
            if self.nr_quadratures == 2:
                self.figure, self.axs = plt.subplots(self.nr_quadratures, 1,
                                                     figsize=(5, 6))
            else:
                self.figure, ax = plt.subplots(self.nr_quadratures, 1,
                                               figsize=(5, 6))
                self.axs = [ax]
                # to ensure it is a list of axes, as figure making relies on this
            x_fine = np.linspace(min(self.sweep_points), max(self.sweep_points),
                                 1000)
            for i in range(self.nr_quadratures):
                if i == 0:
                    plot_title = kw.pop('plot_title', textwrap.fill(
                        self.timestamp_string + '_' +
                        self.measurementstring, 40))
                else:
                    plot_title = ''
                self.axs[i].ticklabel_format(useOffset=False)
                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=self.figure, ax=self.axs[i],
                                                xlabel=self.xlabel,
                                                ylabel=self.ylabels[i],
                                                save=False,
                                                plot_title=plot_title)

                fine_fit = self.fit_res[i].model.func(
                    x_fine, **self.fit_res[i].best_values)
                # adding the fitted amp180
                if 'period' in self.fit_res[i].params.keys():
                    label = 'amp180 = {:.3e}'.format(
                        abs(self.fit_res[i].params['period'].value) / 2)
                else:
                    label = 'amp180 = {:.3e}'.format(
                        abs(self.fit_res[i].params['x0'].value))
                self.axs[i].plot(x_fine, fine_fit, label=label)
                ymin = min(self.measured_values[i])
                ymax = max(self.measured_values[i])
                yspan = ymax - ymin
                self.axs[i].set_ylim(ymin - 0.23 * yspan, 0.05 * yspan + ymax)
                self.axs[i].legend(frameon=False, loc='lower left')

                if show_guess:
                    fine_fit = self.fit_res[i].model.func(
                        x_fine, **self.fit_res[i].init_values)
                    self.axs[i].plot(x_fine, fine_fit, label='guess')
                    self.axs[i].legend(loc='best')

            # display figure
            if show:
                plt.show()

            self.save_fig(self.figure, fig_tight=False, **kw)

    def get_amplitudes(self, **kw):

        # Extract the best fitted frequency and phase.
        freq_fit = self.fit_result.best_values['frequency']
        phase_fit = self.fit_result.best_values['phase']

        freq_std = self.fit_result.params['frequency'].stderr
        phase_std = self.fit_result.params['phase'].stderr

        if freq_fit != 0:

            # If fitted_phase<0, shift fitted_phase by 4. This corresponds to a
            # shift of 2pi in the argument of cos.
            if np.abs(phase_fit) < 0.1:
                phase_fit = 0

            # If phase_fit<1, the piHalf amplitude<0.
            if phase_fit < 1:
                logging.info('The data could not be fitted correctly. '
                             'The fitted phase "%s" <1, which gives '
                             'negative piHalf '
                             'amplitude.' % phase_fit)

            stepsize = self.sweep_points[1] - self.sweep_points[0]
            # Nyquist: wavelength>2*stepsize
            if (freq_fit) > 2 * stepsize:
                logging.info('The data could not be fitted correctly. The '
                             'frequency "%s" is too high.' % freq_fit)

            # Extract pi and pi/2 amplitudes from best fit values
            if phase_fit == 0:
                piPulse = 1 / (2 * freq_fit)
                piHalfPulse = 1 / (4 * freq_fit)
                piPulse_std = freq_std / freq_fit
                piHalfPulse_std = freq_std / freq_fit

            else:
                n = np.arange(-2, 3, 0.5)

                piPulse_vals = (2 * n * np.pi + np.pi -
                                phase_fit) / (2 * np.pi * freq_fit)
                piHalfPulse_vals = (2 * n * np.pi + np.pi /
                                    2 - phase_fit) / (2 * np.pi * freq_fit)

                try:
                    piHalfPulse = np.min(np.take(piHalfPulse_vals,
                                                 np.where(piHalfPulse_vals >= 0)))
                except ValueError:
                    piHalfPulse = np.asarray([])

                try:
                    if piHalfPulse.size != 0:
                        piPulse = np.min(np.take(
                            piPulse_vals, np.where(piPulse_vals >= piHalfPulse)))
                    else:
                        piPulse = np.min(np.take(piPulse_vals,
                                                 np.where(piPulse_vals >= 0.001)))
                except ValueError:
                    piPulse = np.asarray([])

                if piPulse.size == 0 or piPulse > max(self.sweep_points):
                    i = 0
                    while (piPulse_vals[i] < min(self.sweep_points) and
                           i < piPulse_vals.size):
                        i += 1
                    piPulse = piPulse_vals[i]

                if piHalfPulse.size == 0 or piHalfPulse > max(self.sweep_points):
                    i = 0
                    while (piHalfPulse_vals[i] < min(self.sweep_points) and
                           i < piHalfPulse_vals.size):
                        i += 1
                    piHalfPulse = piHalfPulse_vals[i]
                # piPulse = 1/(2*freq_fit) - phase_fit/(2*np.pi*freq_fit)
                # piHalfPulse = 1/(4*freq_fit) - phase_fit/(2*np.pi*freq_fit)

                # Calculate std. deviation for pi and pi/2 amplitudes based on error
                # propagation theory
                # (http://ugastro.berkeley.edu/infrared09/PDF-2009/statistics1.pdf)
                # Errors were assumed to be uncorrelated.

                # extract cov(phase,freq)
                freq_idx = self.fit_result.var_names.index('frequency')
                phase_idx = self.fit_result.var_names.index('phase')
                if self.fit_result.covar is not None:
                    cov_freq_phase = self.fit_result.covar[freq_idx, phase_idx]
                else:
                    cov_freq_phase = 0

                piPulse_std = piPulse * np.sqrt((2 * np.pi * freq_std / freq_fit) ** 2 +
                                                (phase_std / phase_fit) ** 2
                                                - cov_freq_phase /
                                                (np.pi * freq_fit * phase_fit))
                piHalfPulse_std = np.sqrt((piPulse_std) ** 2 +
                                          (freq_std / freq_fit) ** 2)

            if kw.get('print_parameters', False):
                print('\u03C0' + '-Pulse Amplitude = {:.6} '.format(piPulse) +
                      '(' + self.parameter_units[-1] + ')' + '\t' +
                      '\u03C0' + '-Pulse Stddev = {:.6} '.format(piPulse_std) +
                      '(' + self.parameter_units[-1] + ')' + '\n' +
                      '\u03C0' + '/2-Pulse Amplitude = {:.6} '.format(piHalfPulse) +
                      '(' + self.parameter_units[-1] + ')' + '\t' +
                      '\u03C0' + '/2-Pulse Stddev = {:.6} '.format(piHalfPulse_std) +
                      '(' + self.parameter_units[-1] + ')')

            # return as dict for ease of use with "save_computed_parameters"
            self.rabi_amplitudes = {'piPulse': piPulse,
                                    'piPulse_std': piPulse_std,
                                    'piHalfPulse': piHalfPulse,
                                    'piHalfPulse_std': piHalfPulse_std}
        else:
            logging.warning("Fitted frequency is zero. The pi-pulse and "
                            "pi/2-pulse will not be computed.")
            return

    def get_measured_amp180(self):
        # Retrieve amp180 value from data file
        # The "Analysis" group might contain the "Corrected data" dataset
        # fit_grps = list(self.data_file['Analysis'].keys())
        # fitted_pars_0 = self.data_file['Analysis'][fit_grps[0]]
        a = self.data_file['Analysis']
        fit_grps = [i for i in a.values() if isinstance(i, h5py.Group)]
        fitted_pars_0 = fit_grps[0]
        amp180 = fitted_pars_0['period'].attrs['value'] / 2
        # If there are two quadratures, return the amplitude with the smallest
        # errorbar
        if len(fit_grps) == 2:
            fitted_pars_1 = fit_grps[1]
            if (fitted_pars_1['period'].attrs['stderr'] <
                    fitted_pars_0['period'].attrs['stderr']):
                amp180 = fitted_pars_1['period'].attrs['value'] / 2
        return amp180


class Flipping_Analysis(TD_Analysis):

    def __init__(self, label='Flipping', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)


    def run_default_analysis(self, close_file=True, show_guess=False, **kw):
        # Returns the drive scaling factor.
        show = kw.pop('show', False)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        fig1, fig2, ax, axarray = self.setup_figures_and_axes()

        norm = self.normalize_data_to_calibration_points(
            self.measured_values[0], self.NoCalPoints)
        self.normalized_values = norm[0]
        self.normalized_data_points = norm[1]
        self.normalized_cal_vals = norm[2]
        self.fit_Flipping()

        # self.save_fitted_parameters(self.fit_res, var_name=self.value_names[0])
        self.plot_results(fig1, ax, show_guess=show_guess,
                          ylabel=r'$F$ $|1 \rangle$')

        for i, name in enumerate(self.value_names):
            if len(self.value_names) == 4:
                if i < 2:
                    ax2 = axarray[0, i]
                else:
                    ax2 = axarray[1, i-2]
            else:
                ax2 = axarray[i]

            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=fig2, ax=ax2,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            save=False)

        if show:
            plt.show()
        self.save_fig(fig1, figname=self.measurementstring+'_Flipping_fit', **kw)
        self.save_fig(fig2, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return self.drive_scaling_factor

    def plot_results(self, fig, ax, ylabel,show_guess=False):
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.normalized_values,
                                        fig=fig, ax=ax,
                                        xlabel=r'No. Pulses',
                                        ylabel=ylabel,
                                        save=False)
        if show_guess:
            y_init = fit_mods.ExpDampOscFunc(self.sweep_points,
                **self.fit_res.init_values)
            ax.plot(self.sweep_points, y_init, 'k--')

        ax.plot(self.fit_plot_points_x, self.fit_plot_points_y, 'r-')

    def fit_Flipping(self, **kw):
        # Split it up in two cases, one where we have oscillations,
        # one where we are already quite close
        data = self.normalized_data_points
        sweep_points = self.sweep_points[:-self.NoCalPoints]
        w = np.abs(np.fft.fft(data)[1:len(data)//2])
        if np.argmax(w)==0:
            # This is the case where we don't have oscilaltions.
            # We fit a parabola to it.
            def quadratic_fit_data():
                M = np.array([sweep_points**2, sweep_points, [1]*len(sweep_points)])
                Minv = np.linalg.pinv(M)
                [a, b, c] = np.dot(data, Minv)
                fit_data = (a*sweep_points**2 + b*sweep_points + c)
                return fit_data, (a, b, c)
            self.fit_results_quadratic = quadratic_fit_data()
            slope = self.fit_results_quadratic[1][1]
            amplitude = np.average(self.normalized_cal_vals) / 2
            drive_detuning = slope / (2 * np.pi * abs(amplitude))
            self.drive_scaling_factor = 1. / (1. + drive_detuning)
            self.fit_plot_points_x = sweep_points
            self.fit_plot_points_y = self.fit_results_quadratic[0]
        else:
            # This is the case where we have oscilaltions.
            # We fit a decaying cos to it.
            def fit_dec_cos_mod(negative_amplitude=False):
                model = fit_mods.CosModel
                params = model.guess(model, data=data,
                                     t=sweep_points)
                new_mod = fit_mods.ExpDampOscModel
                if negative_amplitude:
                    new_mod.set_param_hint('amplitude',
                                    value=-1*np.abs(params['amplitude'].value),
                                    min=-1, max=0)
                else:
                    new_mod.set_param_hint('amplitude',
                                    value=np.abs(params['amplitude'].value),
                                    min=0, max=1)
                new_mod.set_param_hint('frequency',
                                    value=params['frequency'].value,
                                    min=0.2/sweep_points[-1],
                                    max=10./sweep_points[-1])
                new_mod.set_param_hint('phase',
                                    value=-np.pi/2, vary=False)
                new_mod.set_param_hint('oscillation_offset',
                                    value=0)
                new_mod.set_param_hint('exponential_offset',
                                    value=params['offset'].value)
                new_mod.set_param_hint('n',
                                    value=1, vary=False)
                new_mod.set_param_hint('tau',
                                    value=sweep_points[-1],
                                    min=sweep_points[1],
                                    max=sweep_points[-1]*10)
                new_params = new_mod.make_params()
                return new_mod.fit(data=data,
                                    t=sweep_points, params=new_params)
            pos_amp_fit = fit_dec_cos_mod(negative_amplitude=False)
            neg_amp_fit = fit_dec_cos_mod(negative_amplitude=True)

            if pos_amp_fit.chisqr>neg_amp_fit.chisqr:
                self.fit_res = neg_amp_fit
            else:
                self.fit_res = pos_amp_fit

            self.fit_plot_points_x = np.linspace(sweep_points[0],
                                                 sweep_points[-1],200)
            self.fit_plot_points_y = fit_mods.ExpDampOscFunc(
                t=self.fit_plot_points_x,
                amplitude=self.fit_res.values['amplitude'],
                frequency=self.fit_res.values['frequency'],
                phase=self.fit_res.values['phase'],
                oscillation_offset=self.fit_res.values['oscillation_offset'],
                exponential_offset=self.fit_res.values['exponential_offset'],
                n=self.fit_res.values['n'],
                tau=self.fit_res.values['tau'])
            self.drive_detuning = self.fit_res.values['frequency']*\
                np.sign(self.fit_res.values['amplitude'])
            self.drive_scaling_factor = 1. / (1. + self.drive_detuning)


class TD_UHFQC(TD_Analysis):

    def __init__(self, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 plot_cal_points=True, **kw):
        super(TD_UHFQC, self).__init__(**kw)

    def run_default_analysis(self,
                             close_main_fig=True, **kw):
        super(TD_UHFQC, self).run_default_analysis(**kw)
        measured_values = a_tools.normalize_data_v3(self.measured_values[0])

        fig, ax = plt.subplots(1, figsize=(8, 6))

        ax.plot(self.sweep_points * 1e9, measured_values, '-o')
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'$F |1\rangle$')

        ax.set_title('%s: TD Scan' % self.timestamp_string)

        self.save_fig(fig, fig_tight=False, **kw)


class Echo_analysis(TD_Analysis):
    def __init__(self,vary_n=False,**kw):
        self.vary_n = vary_n
        super(Echo_analysis, self).__init__(**kw)

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        norm = self.normalize_data_to_calibration_points(
            self.measured_values[0], self.NoCalPoints)
        self.normalized_values = norm[0]
        self.normalized_data_points = norm[1]
        self.normalized_cal_vals = norm[2]
        self.fit_data(**kw)
        self.make_figures(**kw)
        if close_file:
            self.data_file.close()
        return self.fit_res
        pass

    def fit_data(self, print_fit_results=False, **kw):

        self.add_analysis_datagroup_to_file()

        model = lmfit.Model(fit_mods.ExpDecayFunc)
        model.guess = fit_mods.exp_dec_guess

        params = model.guess(model, data=self.corr_data[:-self.NoCalPoints],
                             t=self.sweep_points[:-self.NoCalPoints],
                             vary_n=self.vary_n)
        self.fit_res = model.fit(data=self.corr_data[:-self.NoCalPoints],
                                 t=self.sweep_points[:-self.NoCalPoints],
                                 params=params)
        self.save_fitted_parameters(fit_res=self.fit_res,
                                    var_name='corr_data')

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        x_fine = np.linspace(min(self.sweep_points), max(self.sweep_points),
                             1000)
        plot_title = kw.pop('plot_title', textwrap.fill(
            self.timestamp_string + '_' +
            self.measurementstring, 40))
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.corr_data,
                                        fig=self.fig, ax=self.ax,
                                        xlabel=self.parameter_names[0],
                                        x_unit=self.parameter_units[0],
                                        ylabel=r'F$|1\rangle$',
                                        save=False,
                                        plot_title=plot_title)

        self.ax.plot(x_fine, self.fit_res.eval(t=x_fine), label='fit')

        textstr = format_value_string(par_name='$T_2$',
                                      lmfit_par=self.fit_res.params['tau'],
                                      unit=self.parameter_units[0],
                                      end_char='\n')
        textstr += format_value_string(
            '$n$', lmfit_par=self.fit_res.params['n'])

        if show_guess:
            self.ax.plot(x_fine, self.fit_res.eval(
                t=x_fine, **self.fit_res.init_values), label='guess')
            self.ax.legend(loc='best')

        self.ax.text(0.4, 0.95, textstr, transform=self.ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=self.box_props)
        self.save_fig(self.fig, fig_tight=True, **kw)

class Echo_analysis_V15(TD_Analysis):
    """
    New echo analysis for varying phase pulses. Based on old ramsey analysis.
    Should be replaced asap by a V2-style analysis

    -Luc
    """

    def __init__(self, label='echo', phase_sweep_only=False, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        self.phase_sweep_only = phase_sweep_only
        self.artificial_detuning = kw.pop('artificial_detuning', 0)
        if self.artificial_detuning == 0:
            logging.warning('Artificial detuning is unknown. Defaults to %s MHz. '
                            'New qubit frequency might be incorrect.'
                            % self.artificial_detuning)

        # The routines for 2 art_dets does not use the self.fig and self.ax
        # created in TD_Analysis for make_fig==False for TD_Analysis but
        # still want make_fig to decide whether two_art_dets_analysis should
        # make a figure
        self.make_fig_two_dets = kw.get('make_fig', True)
        if (type(self.artificial_detuning) is list) and \
                (len(self.artificial_detuning) > 1):
            kw['make_fig'] = False

        super().__init__(**kw)

    def fit_Echo(self, x, y, **kw):
        self.add_analysis_datagroup_to_file()
        print_fit_results = kw.pop('print_fit_results',False)
        damped_osc_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
        average = np.mean(y)

        ft_of_data = np.fft.fft(y)
        index_of_fourier_maximum = np.argmax(np.abs(
            ft_of_data[1:len(ft_of_data) // 2])) + 1
        max_echo_delay = x[-1] - x[0]

        fft_axis_scaling = 1 / (max_echo_delay)
        freq_est = fft_axis_scaling * index_of_fourier_maximum
        est_number_of_periods = index_of_fourier_maximum
        if self.phase_sweep_only:
            damped_osc_mod.set_param_hint('frequency',
                                          value=1/360,
                                          vary=False)
            damped_osc_mod.set_param_hint('phase',
                                          value=0, vary=True)
            damped_osc_mod.set_param_hint('amplitude',
                                          value=0.5*(max(self.normalized_data_points)-min(self.normalized_data_points)),
                                          min=0.0, max=4.0)
            fixed_tau=1e9
            damped_osc_mod.set_param_hint('tau',
                                          value=fixed_tau,
                                          vary=False)
        else:
            if ((average > 0.7*max(y)) or
                    (est_number_of_periods < 2) or
                    est_number_of_periods > len(ft_of_data)/2.):
                print('the trace is too short to find multiple periods')

                if print_fit_results:
                    print('Setting frequency to 0 and ' +
                          'fitting with decaying exponential.')
                damped_osc_mod.set_param_hint('frequency',
                                              value=freq_est,
                                              vary=False)
                damped_osc_mod.set_param_hint('phase',
                                              value=0,
                                              vary=False)
            else:
                damped_osc_mod.set_param_hint('frequency',
                                              value=freq_est,
                                              vary=True,
                                              min=(1/(100 *x[-1])),
                                              max=(20/x[-1]))

            if (np.average(y[:4]) >
                    np.average(y[4:8])):
                phase_estimate = 0
            else:
                phase_estimate = np.pi
            damped_osc_mod.set_param_hint('phase',
                                              value=phase_estimate, vary=True)

            amplitude_guess = 1
            damped_osc_mod.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.4,
                                          max=4.0)
            damped_osc_mod.set_param_hint('tau',
                                          value=x[1]*10,
                                          min=x[1],
                                          max=x[1]*1000)
        damped_osc_mod.set_param_hint('exponential_offset',
                                      value=0.5,
                                      min=0.4,
                                      max=4.0)
        damped_osc_mod.set_param_hint('oscillation_offset',
                                      value=0,
                                      vary=False)
        damped_osc_mod.set_param_hint('n',
                                      value=1,
                                      vary=False)
        self.params = damped_osc_mod.make_params()

        fit_res = damped_osc_mod.fit(data=y,
                                     t=x,
                                     params=self.params)
        if self.phase_sweep_only:
            chi_sqr_bound = 0
        else:
            chi_sqr_bound = 0.35

        if fit_res.chisqr > chi_sqr_bound:
            logging.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2 * np.pi, 8):
                damped_osc_mod.set_param_hint('phase',
                                              value=phase_estimate)
                self.params = damped_osc_mod.make_params()
                fit_res_lst += [damped_osc_mod.fit(
                    data=y,
                    t=x,
                    params=self.params)]

            chisqr_lst = [fit_res.chisqr for fit_res in fit_res_lst]
            fit_res = fit_res_lst[np.argmin(chisqr_lst)]
        self.fit_results.append(fit_res)

        if print_fit_results:
            print(fit_res.fit_report())
        return fit_res

    def plot_results(self, fit_res, show_guess=False, art_det=0,
                     fig=None, ax=None, textbox=True):

        self.units = SI_prefix_and_scale_factor(val=max(abs(ax.get_xticks())),
                                                unit=self.sweep_unit[0])[1]  # list

        if isinstance(art_det, list):
            art_det = art_det[0]

        if textbox:
            textstr = ('$f_{qubit \_ old}$ = %.7g GHz'
                       % (self.qubit_freq_spec * 1e-9) +
                       '\n$f_{qubit \_ new}$ = %.7g $\pm$ (%.5g) GHz'
                       % (self.qubit_frequency * 1e-9,
                          fit_res.params['frequency'].stderr * 1e-9) +
                       '\n$\Delta f$ = %.5g $ \pm$ (%.5g) MHz'
                       % ((self.qubit_frequency - self.qubit_freq_spec) * 1e-6,
                          fit_res.params['frequency'].stderr * 1e-6) +
                       '\n$f_{Ramsey}$ = %.5g $ \pm$ (%.5g) MHz'
                       % (fit_res.params['frequency'].value * 1e-6,
                          fit_res.params['frequency'].stderr * 1e-6) +
                       '\n$T_2$ = %.6g '
                       % (fit_res.params['tau'].value * self.scale) +
                       self.units + ' $\pm$ (%.6g) '
                       % (fit_res.params['tau'].stderr * self.scale) +
                       self.units +
                       '\nartificial detuning = %.2g MHz'
                       % (art_det * 1e-6))

            fig.text(0.5, 0, textstr, fontsize=self.font_size,
                     transform=ax.transAxes,
                     verticalalignment='top',
                     horizontalalignment='center', bbox=self.box_props)

        x = np.linspace(self.sweep_points[0],
                        self.sweep_points[-self.NoCalPoints - 1],
                        len(self.sweep_points) * 100)

        if show_guess:
            y_init = fit_mods.ExpDampOscFunc(x, **fit_res.init_values)
            ax.plot(x, y_init, 'k--', linewidth=self.line_width)

        best_vals = fit_res.best_values
        y = fit_mods.ExpDampOscFunc(
            x, tau=best_vals['tau'],
            n=best_vals['n'],
            frequency=best_vals['frequency'],
            phase=best_vals['phase'],
            amplitude=best_vals['amplitude'],
            oscillation_offset=best_vals['oscillation_offset'],
            exponential_offset=best_vals['exponential_offset'])
        ax.plot(x, y, 'r-', linewidth=self.line_width)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, **kw):

        super().run_default_analysis(
            close_file=close_file,
            close_main_figure=True, save_fig=False, **kw)

        verbose = kw.get('verbose', False)
        # Get old values for qubit frequency
        instr_set = self.data_file['Instrument settings']
        try:
            if self.for_ef:
                self.qubit_freq_spec = \
                    float(instr_set[self.qb_name].attrs['f_ef_qubit'])
            elif 'freq_qubit' in kw.keys():
                self.qubit_freq_spec = kw['freq_qubit']
            else:
                try:
                    self.qubit_freq_spec = \
                        float(instr_set[self.qb_name].attrs['f_qubit'])
                except KeyError:
                    self.qubit_freq_spec = \
                        float(instr_set[self.qb_name].attrs['freq_qubit'])

        except (TypeError, KeyError, ValueError):
            logging.warning('qb_name is unknown. Setting previously measured '
                            'value of the qubit frequency to 0. New qubit '
                            'frequency might be incorrect.')
            self.qubit_freq_spec = 0

        self.scale = 1e6

        # artificial detuning with one value can be passed as either an int or
        # a list with one elements
        if (type(self.artificial_detuning) is list) and \
                (len(self.artificial_detuning) > 1):
            if verbose:
                print('Performing Ramsey Analysis for 2 artificial detunings.')
            self.two_art_dets_analysis(**kw)
        else:
            if type(self.artificial_detuning) is list:
                self.artificial_detuning = self.artificial_detuning[0]
            if verbose:
                print('Performing Ramsey Analysis for 1 artificial detuning.')
            self.one_art_det_analysis(**kw)

        self.save_computed_parameters(self.T2,
                                      var_name=self.value_names[0])

        # Print the T2 values on screen
        unit = self.parameter_units[0][-1]
        if kw.pop('print_parameters', False):
            print('New qubit frequency = {:.7f} (GHz)'.format(
                self.qubit_frequency * 1e-9) +
                  '\t\tqubit frequency stderr = {:.7f} (MHz)'.format(
                      self.ramsey_freq['freq_stderr'] * 1e-6) +
                  '\nT2* = {:.5f} '.format(
                      self.T2['T2'] * self.scale) + '(' + 'μ' + unit + ')' +
                  '\t\tT2* stderr = {:.5f} '.format(
                      self.T2['T2_stderr'] * self.scale) +
                  '(' + 'μ' + unit + ')')
        if close_file:
            self.data_file.close()

        return self.fit_res

    def one_art_det_analysis(self, **kw):

        # Perform fit and save fitted parameters
        self.fit_res = self.fit_Echo(x=self.sweep_points[:-self.NoCalPoints],
                                       y=self.normalized_data_points, **kw)
        self.save_fitted_parameters(self.fit_res, var_name=self.value_names[0])
        self.get_measured_freq(fit_res=self.fit_res, **kw)

        # Calculate new qubit frequency
        self.qubit_frequency = self.qubit_freq_spec + self.artificial_detuning \
                               - self.echo_freq['freq']

        # Extract T2 and save it
        self.get_measured_T2(fit_res=self.fit_res, **kw)
        # the call above defines self.T2 as a dict; units are seconds

        self.total_detuning = self.fit_res.params['frequency'].value
        self.detuning_stderr = self.fit_res.params['frequency'].stderr
        self.detuning = self.total_detuning - self.artificial_detuning

        if self.make_fig:
            # Plot results
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)
            self.plot_results(self.fit_res, show_guess=show_guess,
                              art_det=self.artificial_detuning,
                              fig=self.fig, ax=self.ax)

            # dispaly figure
            if show:
                plt.show()

            # save figure
            self.save_fig(self.fig, figname=self.measurementstring + '_Echo_fit',
                          **kw)

    def two_art_dets_analysis(self, **kw):

        # Extract the data for each echo
        len_art_det = len(self.artificial_detuning)
        sweep_pts_1 = self.sweep_points[0:-self.NoCalPoints:len_art_det]
        sweep_pts_2 = self.sweep_points[1:-self.NoCalPoints:len_art_det]
        echo_data_1 = self.normalized_values[0:-self.NoCalPoints:len_art_det]
        echo_data_2 = self.normalized_values[1:-self.NoCalPoints:len_art_det]

        # Perform fit
        fit_res_1 = self.fit_Echo(x=sweep_pts_1,
                                    y=echo_data_1, **kw)
        fit_res_2 = self.fit_Echo(x=sweep_pts_2,
                                    y=echo_data_2, **kw)

        self.save_fitted_parameters(fit_res_1, var_name=(self.value_names[0] +
                                                         ' ' + str(self.artificial_detuning[0] * 1e-6) + ' MHz'))
        self.save_fitted_parameters(fit_res_2, var_name=(self.value_names[0] +
                                                         ' ' + str(self.artificial_detuning[1] * 1e-6) + ' MHz'))

        echo_freq_dict_1 = self.get_measured_freq(fit_res=fit_res_1, **kw)
        echo_freq_1 = echo_freq_dict_1['freq']
        echo_freq_dict_2 = self.get_measured_freq(fit_res=fit_res_2, **kw)
        echo_freq_2 = echo_freq_dict_2['freq']

        # Calculate possible detunings from real qubit frequency
        self.new_qb_freqs = {
            '0': self.qubit_freq_spec + self.artificial_detuning[0] + echo_freq_1,
            '1': self.qubit_freq_spec + self.artificial_detuning[0] - echo_freq_1,
            '2': self.qubit_freq_spec + self.artificial_detuning[1] + echo_freq_2,
            '3': self.qubit_freq_spec + self.artificial_detuning[1] - echo_freq_2}

        print('The 4 possible cases for the new qubit frequency give:')
        pprint(self.new_qb_freqs)

        # Find which ones match
        self.diff = {}
        self.diff.update({'0': self.new_qb_freqs['0'] - self.new_qb_freqs['2']})
        self.diff.update({'1': self.new_qb_freqs['1'] - self.new_qb_freqs['3']})
        self.diff.update({'2': self.new_qb_freqs['1'] - self.new_qb_freqs['2']})
        self.diff.update({'3': self.new_qb_freqs['0'] - self.new_qb_freqs['3']})
        self.correct_key = np.argmin(np.abs(list(self.diff.values())))
        # Get new qubit frequency
        self.qubit_frequency = self.new_qb_freqs[str(self.correct_key)]

        if self.correct_key < 2:
            # art_det 1 was correct direction
            # print('Artificial detuning {:.1f} MHz gave the best results.'.format(
            #     self.artificial_detuning[0]*1e-6))
            self.fit_res = fit_res_1
            self.echo_data = echo_data_1
            self.sweep_pts = sweep_pts_1
            self.good_echo_freq = echo_freq_1
            qb_stderr = echo_freq_dict_1['freq_stderr']

        else:
            # art_det 2 was correct direction
            # print('Artificial detuning {:.1f} MHz gave the best results.'.format(
            #     self.artificial_detuning[1]*1e-6))
            self.fit_res = fit_res_2
            self.echo_data = echo_data_2
            self.sweep_pts = sweep_pts_2
            self.good_echo_freq = echo_freq_2
            qb_stderr = echo_freq_dict_2['freq_stderr']

        # Extract T2 and save it
        self.get_measured_T2(fit_res=self.fit_res, **kw)  # defines self.T2 as a dict;
        # units are seconds

        ################
        # Plot results #
        ################
        if self.make_fig_two_dets:
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)

            if self.for_ef:
                ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
            else:
                ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'
            if self.no_of_columns == 2:
                figsize = (3.375, 2.25 * len_art_det)
            else:
                figsize = (7, 4 * len_art_det)
            self.fig, self.axs = plt.subplots(len_art_det, 1,
                                              figsize=figsize,
                                              dpi=self.dpi)

            fit_res_array = [fit_res_1, fit_res_2]
            echo_data_dict = {'0': echo_data_1,
                                '1': echo_data_2}

            for i in range(len_art_det):
                ax = self.axs[i]
                self.plot_results_vs_sweepparam(x=self.sweep_pts,
                                                y=echo_data_dict[str(i)],
                                                fig=self.fig, ax=ax,
                                                xlabel=self.sweep_name,
                                                x_unit=self.sweep_unit[0],
                                                ylabel=ylabel,
                                                marker='o-',
                                                save=False)
                self.plot_results(fit_res_array[i], show_guess=show_guess,
                                  art_det=self.artificial_detuning[i],
                                  fig=self.fig, ax=ax, textbox=False)

                textstr = ('artificial detuning = %.2g MHz'
                           % (self.artificial_detuning[i] * 1e-6) +
                           '\n$f_{Echo}$ = %.5g $ MHz \pm$ (%.5g) MHz'
                           % (fit_res_array[i].params['frequency'].value * 1e-6,
                              fit_res_array[i].params['frequency'].stderr * 1e6) +
                           '\n$T_2$ = %.3g '
                           % (fit_res_array[i].params['tau'].value * self.scale) +
                           self.units + ' $\pm$ (%.3g) '
                           % (fit_res_array[i].params['tau'].stderr * self.scale) +
                           self.units)
                ax.annotate(textstr, xy=(0.99, 0.98), xycoords='axes fraction',
                            fontsize=self.font_size, bbox=self.box_props,
                            horizontalalignment='right', verticalalignment='top')

                if i == (len_art_det - 1):
                    textstr_main = ('$f_{qubit \_ old}$ = %.5g GHz'
                                    % (self.qubit_freq_spec * 1e-9) +
                                    '\n$f_{qubit \_ new}$ = %.5g $ GHz \pm$ (%.5g) GHz'
                                    % (self.qubit_frequency * 1e-9,
                                       qb_stderr * 1e-9) +
                                    '\n$T_2$ = %.3g '
                                    % (self.T2['T2'] * self.scale) +
                                    self.units + ' $\pm$ (%.3g) '
                                    % (self.T2['T2_stderr'] * self.scale) +
                                    self.units)

                    self.fig.text(0.5, 0, textstr_main, fontsize=self.font_size,
                                  transform=self.axs[i].transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='center', bbox=self.box_props)

            # dispaly figure
            if show:
                plt.show()

            # save figure
            self.save_fig(self.fig, figname=self.measurementstring + '_Echo_fit',
                          **kw)

    def get_measured_freq(self, fit_res, **kw):
        freq = fit_res.params['frequency'].value
        freq_stderr = fit_res.params['frequency'].stderr

        self.echo_freq = {'freq': freq, 'freq_stderr': freq_stderr}

        return self.echo_freq

    def get_measured_T2(self, fit_res, **kw):
        '''
        Returns measured T2 from the fit to the Ical data.
         return T2, T2_stderr
        '''
        T2 = fit_res.params['tau'].value
        T2_stderr = fit_res.params['tau'].stderr

        self.T2 = {'T2': T2, 'T2_stderr': T2_stderr}

        return self.T2



class Rabi_parabola_analysis(Rabi_Analysis):

    def fit_data(self, print_fit_results=False, **kw):
        self.add_analysis_datagroup_to_file()
        model = lmfit.models.ParabolicModel()
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.
        for i in [0, 1]:
            model.set_param_hint('x0', expr='-b/(2*a)')
            params = model.guess(data=self.measured_values[i],
                                 x=self.sweep_points)
            self.fit_res[i] = model.fit(
                data=self.measured_values[i],
                x=self.sweep_points,
                params=params)
            self.save_fitted_parameters(fit_res=self.fit_res[i],
                                        var_name=self.value_names[i])


class CPhase_2Q_amp_cost_analysis(Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):
        normalize_to_cal_points = kw.get('normalize_to_cal_points', True)
        self.normalize_to_cal_points = normalize_to_cal_points

        self.get_naming_and_values()
        self.oscillating_qubit = kw.get('oscillating_qubit', 0)

        if normalize_to_cal_points:
            if self.oscillating_qubit == 0:
                cal_0I = np.mean([self.measured_values[0][-4],
                                  self.measured_values[0][-3]])

                cal_1I = np.mean([self.measured_values[0][-2],
                                  self.measured_values[0][-1]])

                cal_0Q = np.mean([self.measured_values[1][-4],
                                  self.measured_values[1][-2]])

                cal_1Q = np.mean([self.measured_values[1][-3],
                                  self.measured_values[1][-1]])
            else:
                cal_0I = np.mean([self.measured_values[0][-4],
                                  self.measured_values[0][-2]])

                cal_1I = np.mean([self.measured_values[0][-3],
                                  self.measured_values[0][-1]])

                cal_0Q = np.mean([self.measured_values[1][-4],
                                  self.measured_values[1][-3]])

                cal_1Q = np.mean([self.measured_values[1][-2],
                                  self.measured_values[1][-1]])

            self.measured_values[0][:] = (
                self.measured_values[0] - cal_0I) / (cal_1I - cal_0I)
            self.measured_values[1][:] = (
                self.measured_values[1] - cal_0Q) / (cal_1Q - cal_0Q)

        self.sort_data()

        # self.calculate_cost_func(**kw)
        self.fit_data(**kw)
        self.make_figures(**kw)

        if close_file:
            self.data_file.close()

    def sort_data(self):
        self.x_exc = self.sweep_points[1::2]
        self.x_idx = self.sweep_points[::2]
        self.y_exc = ['', '']
        self.y_idx = ['', '']

        for i in range(2):
            self.y_exc[i] = self.measured_values[i][1::2]
            self.y_idx[i] = self.measured_values[i][::2]
            if self.normalize_to_cal_points:
                self.y_exc[i] = self.y_exc[i][:-2]
                self.y_idx[i] = self.y_idx[i][:-2]
        if self.normalize_to_cal_points:
            self.x_idx = self.x_idx[:-2]
            self.x_exc = self.x_exc[:-2]

    def calculate_cost_func(self, **kw):
        num_points = len(self.sweep_points) - 4

        id_dat_swp = self.measured_values[1][:num_points // 2]
        ex_dat_swp = self.measured_values[1][num_points // 2:-4]

        id_dat_cp = self.measured_values[0][:num_points // 2]
        ex_dat_cp = self.measured_values[0][num_points // 2:-4]

        maximum_difference = max((id_dat_cp - ex_dat_cp))
        # I think the labels are wrong in excited and identity but the value
        # we get is correct
        missing_swap_pop = np.mean(ex_dat_swp - id_dat_swp)
        self.cost_func_val = maximum_difference, missing_swap_pop

    def make_figures(self, **kw):
        # calculate fitted curves
        x_points_fit = np.linspace(self.x_idx[0], self.x_idx[-1], 50)
        fit_idx = self.fit_result['idx_amp'] \
            * np.cos(2 * np.pi * self.fit_result['idx_freq']
                     * x_points_fit + self.fit_result['idx_phase']) \
            + self.fit_result['idx_offset']

        fit_exc = self.fit_result['exc_amp'] \
            * np.cos(2 * np.pi * self.fit_result['exc_freq']
                     * x_points_fit + self.fit_result['exc_phase']) \
            + self.fit_result['exc_offset']

        self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 6))
        for i in [0, 1]:
            self.axs[i].plot(self.x_idx, self.y_idx[i], '-o',
                             label='no excitation')
            self.axs[i].plot(self.x_exc, self.y_exc[i], '-o',
                             label='excitation')
            if i == self.oscillating_qubit:
                plot_title = kw.pop('plot_title', textwrap.fill(
                    self.timestamp_string + '_' +
                    self.measurementstring, 40))
                self.axs[i].plot(x_points_fit, fit_idx, '-')
                self.axs[i].plot(x_points_fit, fit_exc, '-')
                self.axs[i].legend()
            else:
                plot_title = ''
            set_xlabel(self.axs[i], self.sweep_name, self.sweep_unit[0])
            set_ylabel(self.axs[i], self.value_names[i], self.value_units[i])

        self.save_fig(self.fig, fig_tight=True, **kw)

    def fit_data(self, **kw):
        # Frequency is known, because we sweep the phase of the second pihalf
        # pulse in a Ramsey-type experiment.
        model = lmfit.Model((lambda t, amplitude, phase, offset:
                             amplitude * np.cos(2 * np.pi * t / 360.0 + phase) + offset))
        self.fit_result = {}

        # Fit case with no excitation first
        guess_params = fit_mods.Cos_amp_phase_guess(
            model,
            data=self.y_idx[self.oscillating_qubit],
            f=1.0 / 360.0, t=self.x_idx)
        fit_res = model.fit(
            data=self.y_idx[self.oscillating_qubit],
            t=self.x_idx,
            params=guess_params)
        self.fit_result['idx_amp'] = fit_res.values['amplitude']
        self.fit_result['idx_freq'] = 1.0 / 360.0
        self.fit_result['idx_phase'] = fit_res.values['phase']
        self.fit_result['idx_offset'] = fit_res.values['offset']

        # Fit case with excitation
        guess_params = fit_mods.Cos_amp_phase_guess(
            model,
            data=self.y_exc[self.oscillating_qubit],
            f=1.0 / 360.0, t=self.x_exc)
        fit_res = model.fit(
            data=self.y_exc[self.oscillating_qubit],
            t=self.x_exc,
            params=guess_params)
        self.fit_result['exc_amp'] = fit_res.values['amplitude']
        self.fit_result['exc_freq'] = 1.0 / 360.0
        self.fit_result['exc_phase'] = fit_res.values['phase']
        self.fit_result['exc_offset'] = fit_res.values['offset']

        # TODO: save fit params


class Motzoi_XY_analysis(TD_Analysis):
    '''
    Analysis for the Motzoi XY sequence (Xy-Yx)
    Extracts the alternating datapoints and then fits two polynomials.
    The intersect of the fits corresponds to the optimum motzoi parameter.
    '''

    def __init__(self, label='Motzoi', cal_points=[[-4, -3], [-2, -1]], **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        self.cal_points = cal_points
        super().__init__(**kw)

    def run_default_analysis(self, close_file=True, close_main_fig=True, **kw):
        self.get_naming_and_values()
        self.add_analysis_datagroup_to_file()
        if self.cal_points is None:
            if len(self.measured_values) == 2:

                self.corr_data = (self.measured_values[0] ** 2 +
                                  self.measured_values[1] ** 2)

            else:
                self.corr_data = self.measured_values[0]
        else:
            self.rotate_and_normalize_data()
            self.add_dataset_to_analysisgroup('Corrected data',
                                              self.corr_data)
            self.analysis_group.attrs.create('corrected data based on',
                                             'calibration points'.encode('utf-8'))
        # Only the unfolding part here is unique to this analysis
        self.sweep_points_Xy = self.sweep_points[:-4:2]
        self.sweep_points_Yx = self.sweep_points[1:-4:2]
        self.corr_data_Xy = self.corr_data[:-4:2]
        self.corr_data_Yx = self.corr_data[1:-4:2]

        self.fit_data(**kw)
        self.make_figures(**kw)

        opt_motzoi = self.calculate_optimal_motzoi()

        if close_file:
            self.data_file.close()
        return opt_motzoi

    def make_figures(self, **kw):
        # Unique in that it has hardcoded names and ponits to plot
        show_guess = kw.pop('show_guess', False)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 3))
        x_fine = np.linspace(min(self.sweep_points), max(self.sweep_points),
                             1000)
        plot_title = kw.pop('plot_title', textwrap.fill(
            self.timestamp_string + '_' +
            self.measurementstring, 40))
        self.ax.set_title(plot_title)

        self.ax.ticklabel_format(useOffset=False)
        self.ax.set_xlabel(kw.pop('xlabel', self.xlabel))
        self.ax.set_ylabel(kw.pop('ylabel', r'$F|1\rangle$'))
        self.ax.plot(self.sweep_points_Xy, self.corr_data_Xy,
                     'o', c='b', label='Xy')
        self.ax.plot(self.sweep_points_Yx, self.corr_data_Yx,
                     'o', c='r', label='Yx')
        c = ['b', 'r']
        if hasattr(self, 'fit_res'):
            for i in range(len(self.fit_res)):
                fine_fit = self.fit_res[i].model.func(
                    x_fine, **self.fit_res[i].best_values)
                self.ax.plot(x_fine, fine_fit, c=c[i], label='fit')
                if show_guess:
                    fine_fit = self.fit_res[i].model.func(
                        x_fine, **self.fit_res[i].init_values)
                    self.ax.plot(x_fine, fine_fit, c=c[i], label='guess')

        self.ax.legend(loc='best')
        if self.cal_points is not None:
            self.ax.set_ylim(-.1, 1.1)
        self.save_fig(self.fig, fig_tight=True, **kw)

    def fit_data(self, **kw):
        model = lmfit.models.ParabolicModel()
        self.fit_res = ['', '']

        params = model.guess(data=self.corr_data_Xy,
                             x=self.sweep_points_Xy)
        self.fit_res[0] = model.fit(
            data=self.corr_data_Xy,
            x=self.sweep_points_Xy,
            params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[0],
                                    var_name='Xy')

        params = model.guess(data=self.corr_data_Yx,
                             x=self.sweep_points_Yx)
        self.fit_res[1] = model.fit(
            data=self.corr_data_Yx,
            x=self.sweep_points_Yx,
            params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[1],
                                    var_name='Yx')

    def calculate_optimal_motzoi(self):
        '''
        The best motzoi parameter is there where both curves intersect.
        As a parabola can have 2 intersects.
        Will default to picking the one closest to zero
        '''
        # Fit res 0 is the fit res for Xy, and fit_res 1 for Yx
        b_vals0 = self.fit_res[0].best_values
        b_vals1 = self.fit_res[1].best_values
        x1, x2 = a_tools.solve_quadratic_equation(
            b_vals1['a'] - b_vals0['a'], b_vals1['b'] - b_vals0['b'],
            b_vals1['c'] - b_vals0['c'])
        self.optimal_motzoi = min(x1, x2, key=lambda x: abs(x))
        return self.optimal_motzoi


class QScale_Analysis(TD_Analysis):
    '''
    Analysis for a DRAG pulse calibration measurement as described in
    Baur, M. PhD Thesis(2012): measurement sequence ( (xX)-(xY)-(xmY) ).
    Extracts the alternating data points and then fits two lines
    ((xY) and (xmY)) and a constant (xX).
    The intersect of the fits corresponds to the optimum motzoi parameter.

    1. The I and Q data are rotated and normalized based on the calibration
        points. In most
        analysis routines, the latter are typically 4: 2 X180 measurements,
        and 2 identity measurements,
        which get averaged resulting in one X180 point and one identity point.
    2. The data points for the same qscale value are extracted (every other 3rd
        point because the sequence
       used for this measurement applies the 3 sets of pulses
       ( (xX)-(xY)-(xmY) ) consecutively for each qscale value).
    3. The xX data is fitted to a lmfit.models.ConstantModel(), and the other 2
        to an lmfit.models.LinearModel().
    4. The data and the resulting fits are all plotted on the same graph
        (self.make_figures).
    5. The optimal qscale parameter is obtained from the point where the 2
        linear fits intersect.

    Possible input parameters:
        auto              (default=True)
            automatically perform the entire analysis upon call
        label             (default=none?)
            Label of the analysis routine
        folder            (default=working folder)
            Working folder
        NoCalPoints       (default=4)
            Number of calibration points
        cal_points        (default=[[-4, -3], [-2, -1]])
            The indices of the calibration points
        for_ef            (default=False)
            analyze for EF transition
        make_fig          (default=True)
            plot the fitted data
        show              (default=False)
            show the plot
        show_guess        (default=False)
            plot with initial guess values
        print_parameters       (default=True)
            print the found qscale value and stddev
        plot_title        (default=measurementstring)
            the title for the plot as a string
        xlabel            (default=self.xlabel)
            the label for the x axis as a string
        ylabel            (default=r'$F|1\rangle$')
            the label for the x axis as a string
        close_file        (default=True)
            close the hdf5 file

    The default analysis (auto=True) returns the fit results
    '''

    def __init__(self, label='QScale', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'

        self.make_fig_qscale = kw.get('make_fig', True)
        kw['make_fig'] = False
        super().__init__(**kw)

    def run_default_analysis(self, close_file=False,
                             show=False, **kw):

        super().run_default_analysis(show=show,
                                     close_file=close_file,
                                     close_main_figure=True,
                                     save_fig=False, **kw)

        # Only the unfolding part here is unique to this analysis
        self.sweep_points_xX = self.sweep_points[:-self.NoCalPoints:3]
        self.sweep_points_xY = self.sweep_points[1:-self.NoCalPoints:3]
        self.sweep_points_xmY = self.sweep_points[2:-self.NoCalPoints:3]
        self.corr_data_xX = self.normalized_values[:-self.NoCalPoints:3]
        self.corr_data_xY = self.normalized_values[1:-self.NoCalPoints:3]
        self.corr_data_xmY = self.normalized_values[2:-self.NoCalPoints:3]

        self.fit_data(**kw)

        self.calculate_optimal_qscale(**kw)
        self.save_computed_parameters(self.optimal_qscale,
                                      var_name=self.value_names[0])

        if self.make_fig_qscale:
            fig, ax = self.default_ax()
            self.make_figures(fig=fig, ax=ax, **kw)

            if show:
                plt.show()

            if kw.pop('save_fig', True):
                self.save_fig(fig,
                              figname=self.measurementstring + '_Qscale_fit', **kw)

        if close_file:
            self.data_file.close()

        return self.fit_res

    def make_figures(self, fig=None, ax=None, **kw):

        # Unique in that it has hardcoded names and points to plot
        show_guess = kw.pop('show_guess', False)

        x_fine = np.linspace(min(self.sweep_points[:-self.NoCalPoints]),
                             max(self.sweep_points[:-self.NoCalPoints]),
                             1000)

        # Get old values
        instr_set = self.data_file['Instrument settings']
        try:
            if self.for_ef:
                qscale_old = float(instr_set[self.qb_name].attrs['motzoi_ef'])
            else:
                qscale_old = float(instr_set[self.qb_name].attrs['motzoi'])
            old_vals = '\n$qscale_{old} = $%.5g' % (qscale_old)
        except (TypeError, KeyError, ValueError):
            logging.warning('qb_name is None. Old parameter values will '
                            'not be retrieved.')
            old_vals = ''

        textstr = ('qscale = %.5g $\pm$ %.5g'
                   % (self.optimal_qscale['qscale'],
                      self.optimal_qscale['qscale_std']) + old_vals)

        if self.for_ef:
            ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
        else:
            ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'

        fig.text(0.5, 0, textstr, fontsize=self.font_size,
                 transform=ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='center', bbox=self.box_props)

        self.plot_results_vs_sweepparam(self.sweep_points_xX, self.corr_data_xX,
                                        fig, ax,
                                        marker='ob',
                                        label=r'$X_{\frac{\pi}{2}}X_{\pi}$',
                                        ticks_around=True)
        self.plot_results_vs_sweepparam(self.sweep_points_xY, self.corr_data_xY,
                                        fig, ax,
                                        marker='og',
                                        label=r'$X_{\frac{\pi}{2}}Y_{\pi}$',
                                        ticks_around=True)
        self.plot_results_vs_sweepparam(self.sweep_points_xmY,
                                        self.corr_data_xmY, fig, ax,
                                        marker='or',
                                        label=r'$X_{\frac{\pi}{2}}Y_{-\pi}$',
                                        ticks_around=True,
                                        xlabel=r'$q_{scales}$',
                                        ylabel=ylabel)
        ax.legend(loc='best', prop={'size': self.font_size})
        # c = ['b', 'g', 'r']
        c = ['g', 'r']
        if hasattr(self, 'fit_res'):
            # for i in range(len(self.fit_res)):
            for i in range(len(c)):
                fine_fit = self.fit_res[i + 1].model.func(
                    x_fine, **self.fit_res[i + 1].best_values)
                # if i == 0:
                #     fine_fit = self.fit_res[i+1].best_values['c'] * \
                #        np.ones(x_fine.size)
                ax.plot(x_fine, fine_fit, c=c[i], linewidth=self.axes_line_width,
                        label='fit')
                if show_guess:
                    fine_fit = self.fit_res[i + 1].model.func(
                        x_fine, **self.fit_res[i + 1].init_values)
                    if i == 0:
                        fine_fit = self.fit_res[i + 1].best_values['c'] * \
                            np.ones(x_fine.size)
                    ax.plot(x_fine, fine_fit, c=c[i], linewidth=self.axes_line_width,
                            label='guess')

        # Create custom legend
        blue_line = mlines.Line2D([], [], color='blue', marker='o',
                                  markersize=self.marker_size,
                                  label=r'$X_{\frac{\pi}{2}}X_{\pi}$')
        green_line = mlines.Line2D([], [], color='green', marker='o',
                                   markersize=self.marker_size,
                                   label=r'$X_{\frac{\pi}{2}}Y_{\pi}$')
        red_line = mlines.Line2D([], [], color='red', marker='o',
                                 markersize=self.marker_size,
                                 label=r'$X_{\frac{\pi}{2}}Y_{-\pi}$')
        ax.legend(handles=[blue_line, green_line, red_line], loc='upper right',
                  prop={'size': self.font_size})

        qscale = self.optimal_qscale['qscale']
        ax.plot([qscale, qscale],
                [min(ax.get_ylim()),
                 max(ax.get_ylim())],
                'k--',
                linewidth=self.axes_line_width)
        ax.plot([min(ax.get_xlim()),
                 max(ax.get_xlim())],
                [0.5, 0.5], 'k--',
                linewidth=self.axes_line_width)

    def fit_data(self, **kw):

        model_const = lmfit.models.ConstantModel()
        model_linear = lmfit.models.LinearModel()
        self.fit_res = ['', '', '']

        # Fit xX measurement - constant
        params = model_const.guess(data=self.corr_data_xX,
                                   x=self.sweep_points_xX)
        self.fit_res[0] = model_const.fit(
            data=self.corr_data_xX,
            x=self.sweep_points_xX,
            params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[0],
                                    var_name='xX')

        # Fit xY measurement
        params = model_linear.guess(data=self.corr_data_xY,
                                    x=self.sweep_points_xY)
        self.fit_res[1] = model_linear.fit(
            data=self.corr_data_xY,
            x=self.sweep_points_xY,
            params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[1],
                                    var_name='xY')

        # Fit xmY measurement
        params = model_linear.guess(data=self.corr_data_xmY,
                                    x=self.sweep_points_xmY)
        self.fit_res[2] = model_linear.fit(
            data=self.corr_data_xmY,
            x=self.sweep_points_xmY,
            params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[2],
                                    var_name='xmY')

        if kw.get('print_fit_results', False):
            print('Fit Report - X' + '\u03C0' + '/2 X' + '\u03C0' + ':\n{}\n'.
                  format(self.fit_res[0].fit_report()) +
                  'Fit Report - X' + '\u03C0' + '/2 Y' + '\u03C0' + ':\n{}\n'.
                  format(self.fit_res[1].fit_report()) +
                  'Fit Report - X' + '\u03C0' + '/2 Y-' + '\u03C0' + ':\n{}\n'.
                  format(self.fit_res[2].fit_report()))

    def calculate_optimal_qscale(self, threshold=0.02, **kw):

        # The best qscale parameter is the point where all 3 curves intersect.

        print_parameters = kw.get('print_parameters', False)

        b_vals0 = self.fit_res[0].best_values
        b_vals1 = self.fit_res[1].best_values
        b_vals2 = self.fit_res[2].best_values
        optimal_qscale = (b_vals1['intercept'] - b_vals2['intercept']) / \
                         (b_vals2['slope'] - b_vals1['slope'])

        # Warning if Xpi/2Xpi line is not within +/-threshold of 0.5
        if (b_vals0['c'] > (0.5 + threshold)) or (b_vals0['c'] < (0.5 - threshold)):
            logging.warning('The trace from the X90-X180 pulses is NOT within '
                            '+/-%s of the expected value of 0.5.' % threshold)
        # Warning if optimal_qscale is not within +/-threshold of 0.5
        optimal_qscale_pop = optimal_qscale * \
            b_vals2['slope'] + b_vals2['intercept']
        if (optimal_qscale_pop > (0.5 + threshold)) or \
                (optimal_qscale_pop < (0.5 - threshold)):
            logging.warning('The optimal qscale found gives a population that is '
                            'NOT within +/-%s of the expected value of 0.5.'
                            % threshold)

        # Calculate standard deviation
        # (http://ugastro.berkeley.edu/infrared09/PDF-2009/statistics1.pdf)
        b1_idx = self.fit_res[1].var_names.index('intercept')
        m1_idx = self.fit_res[1].var_names.index('slope')
        b2_idx = self.fit_res[2].var_names.index('intercept')
        m2_idx = self.fit_res[2].var_names.index('slope')

        if self.fit_res[1].covar is not None:
            cov_b1_m1 = self.fit_res[1].covar[b1_idx, m1_idx]
        else:
            cov_b1_m1 = 0
        if self.fit_res[2].covar is not None:
            cov_b2_m2 = self.fit_res[2].covar[b2_idx, m2_idx]
        else:
            cov_b2_m2 = 0

        cov_qscale_squared = (- cov_b1_m1 - cov_b2_m2) ** 2

        intercept_diff_mean = self.fit_res[1].params['intercept'].value - \
            self.fit_res[2].params['intercept'].value
        slope_diff_mean = self.fit_res[2].params['slope'].value - \
            self.fit_res[1].params['slope'].value

        intercept_diff_std_squared = \
            (self.fit_res[1].params['intercept'].stderr) ** 2 + \
            (self.fit_res[2].params['intercept'].stderr) ** 2
        slope_diff_std_squared = \
            (self.fit_res[2].params['slope'].stderr) ** 2 + \
            (self.fit_res[1].params['slope'].stderr) ** 2

        sqrt_quantity = intercept_diff_std_squared / ((intercept_diff_mean) ** 2) + \
            slope_diff_std_squared / ((slope_diff_mean) ** 2) - \
            2 * cov_qscale_squared / (intercept_diff_mean * slope_diff_mean)
        if sqrt_quantity < 0:
            optimal_qscale_stddev = optimal_qscale * np.sqrt(
                intercept_diff_std_squared / ((intercept_diff_mean) ** 2) +
                slope_diff_std_squared / ((slope_diff_mean) ** 2))
        else:
            optimal_qscale_stddev = optimal_qscale * np.sqrt(sqrt_quantity)

        if print_parameters:
            print('Optimal QScale Parameter = {} \t QScale Stddev = {}'.format(
                optimal_qscale, optimal_qscale_stddev))

        # return as dict for use with "save_computed_parameters"
        self.optimal_qscale = {'qscale': optimal_qscale,
                               'qscale_std': optimal_qscale_stddev}

        return self.optimal_qscale


class Rabi_Analysis_old(TD_Analysis):
    '''
    This is the old Rabi analysis for the mathematica sequences of 60 points
    '''

    def __init__(self, label='Rabi', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):
        self.add_analysis_datagroup_to_file()
        show_guess = kw.pop('show_guess', False)
        show_fig = kw.pop('show_fig', False)
        close_file = kw.pop('close_file', True)
        figsize = kw.pop('figsize', (11, 10))

        self.get_naming_and_values()
        if self.sweep_unit != 'arb unit':
            # If the control is not off the tektronix the center should be 0
            self.center_point = 0

        fig, axarray = plt.subplots(2, 1, figsize=figsize)
        fit_res = [None] * len(self.value_names)

        for i, name in enumerate(self.value_names):
            offset_estimate = np.mean(self.measured_values[i])
            if (np.mean(self.measured_values[i][30:34]) <
                    np.mean(self.measured_values[i][34:38])):
                amplitude_sign = -1.
            else:
                amplitude_sign = 1.
            amplitude_estimate = amplitude_sign * abs(max(
                self.measured_values[i]) - min(self.measured_values[i])) / 2
            w = np.fft.fft(
                self.measured_values[i][:-self.NoCalPoints] - offset_estimate)
            index_of_fourier_maximum = np.argmax(np.abs(w[1:len(w) / 2])) + 1
            fourier_index_to_freq = 1 / abs(self.sweep_points[0] -
                                            self.sweep_points[-self.NoCalPoints])
            if index_of_fourier_maximum < 3:
                print(
                    'Rabi period too long for fourier analysis, using single period as default guess')
                frequency_estimate = fourier_index_to_freq
            else:
                frequency_estimate = fourier_index_to_freq * \
                    index_of_fourier_maximum
            # Guess for params

            fit_mods.CosModel.set_param_hint('amplitude',
                                             value=amplitude_estimate)
            fit_mods.CosModel.set_param_hint('frequency',
                                             value=frequency_estimate,
                                             min=0, max=1 / 8.)
            fit_mods.CosModel.set_param_hint('offset',
                                             value=offset_estimate)
            fit_mods.CosModel.set_param_hint('phase',
                                             value=0,
                                             # Should be at the center
                                             # we let sign take care of
                                             # flipping
                                             vary=False),

            self.params = fit_mods.CosModel.make_params()
            displaced_fitting_axis = self.sweep_points[:-self.NoCalPoints] - \
                self.center_point

            fit_res[i] = fit_mods.CosModel.fit(
                data=self.measured_values[i][:-self.NoCalPoints],
                t=displaced_fitting_axis,
                params=self.params)
            self.fit_results.append(fit_res[i])
            self.save_fitted_parameters(fit_res[i],
                                        var_name=name)
            best_vals = fit_res[i].best_values

            if print_fit_results:
                print(fit_res[i].fit_report())

            if not best_vals['frequency'] == 0:
                self.drive_scaling_factor = self.calculate_drive_scaling_factor(
                    best_vals['frequency'])
            else:
                logging.warning('FIXME something wrong with frequency fit')
                self.drive_scaling_factor = 1

            if show_guess:
                axarray[i].plot(self.sweep_points[:-self.NoCalPoints],
                                fit_res[i].init_fit, 'k--')
            x = np.linspace(min(displaced_fitting_axis),
                            max(displaced_fitting_axis),
                            len(displaced_fitting_axis) * 100)

            y = fit_mods.CosFunc(x,
                                 frequency=best_vals['frequency'],
                                 phase=best_vals['phase'],
                                 amplitude=best_vals['amplitude'],
                                 offset=best_vals['offset'])
            axarray[i].plot(x + self.center_point, y, 'r-')

            textstr = (
                '''    $f$ = %.3g $\pm$ (%.3g)
                           $A$ = %.3g $\pm$ (%.3g)
                           $\phi$ = %.3g $\pm$ (%.3g)
                           $a_0$ = %.3g $\pm$ (%.3g)''' % (
                    fit_res[i].params['frequency'].value,
                    fit_res[i].params['frequency'].stderr,
                    fit_res[i].params['amplitude'].value,
                    fit_res[i].params['amplitude'].stderr,
                    fit_res[i].params['phase'].value,
                    fit_res[i].params['phase'].stderr,
                    fit_res[i].params['offset'].value,
                    fit_res[i].params['offset'].stderr))

            axarray[i].text(0.65, 0.95, textstr,
                            transform=axarray[i].transAxes,
                            fontsize=11, verticalalignment='top',
                            horizontalalignment='left',
                            bbox=self.box_props)
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=fig, ax=axarray[i],
                                            xlabel=self.xlabel,
                                            ylabel=str(self.value_names[i]),
                                            save=False)

        if show_fig:
            plt.show()
        self.save_fig(fig, figname=self.sweep_name + 'Rabi_fit', **kw)
        if close_file:
            self.data_file.close()
        return fit_res

    def calculate_drive_scaling_factor(self, frequency):
        '''
        This works by the assumption that you want to have 1.5 Rabi periods
        in your signal. This means that the pi amplitude should be at .75 of
        the max amplitude.
        '''
        desired_period_in_indices = \
            (len(self.sweep_points) - self.NoCalPoints) / 1.5
        sorted_swp = np.sort(self.sweep_points)
        # Sorting needed for when data is taken in other than ascending order
        step_per_index = sorted_swp[1] - sorted_swp[0]
        desired_period = desired_period_in_indices * step_per_index
        # calibration points max should be at -20
        # and + 20 from the center -> period of 80
        desired_freq = 1 / desired_period
        rabi_scaling = desired_freq / frequency
        return rabi_scaling

    def get_drive_scaling_factor(self):
        best_fit = self.get_best_fit_results()
        frequency = best_fit['frequency'].attrs['value']

        drive_scaling_factor = self.calculate_drive_scaling_factor(frequency)

        print('Drive scaling factor: %.2f' % drive_scaling_factor)
        return drive_scaling_factor


class SSRO_Analysis(MeasurementAnalysis):
    '''
    Analysis class for Single Shot Readout.
    Scripts finds optimum rotation of IQ plane leaving all information in the
    I-quadrature.
    Then, for both On and Off datasets unbinned s-curves are fitted with the
    sum of two gaussians. From the fits two fidelity numbers are extracted:

    outputs two fidelity numbers:
        - F: the maximum separation between the two double gauss fits
        - F_corrected: the maximum separation between the largest normalized
                    gausses of both double gauss fits
                    this thereby aims to correct the data for
                    - imperfect pulses
                    - relaxation
                    - residual excitation
                    This figure of merit is unstable for low (<0.30 fidelity)
    outputs one optimum voltage
        -V_opt: the optimum threshold voltage is equal for both definitions of
                fidelity.

    Nofits option is added to skip the double gaussian fitting and extract
    the optimum threshold and fidelity from cumulative histograms.
    '''

    def __init__(self, rotate=True, close_fig=True, channels=None,
                 hist_log_scale: bool = True, **kw):
        if channels is None:
            channels = ['I', 'Q']

        logging.warning('The use of this class is deprectated!' +
                        ' Use the new v2 analysis instead.')

        kw['h5mode'] = 'r+'
        self.rotate = rotate
        self.channels = channels
        self.hist_log_scale = hist_log_scale
        self.close_fig = close_fig
        self.F_a = 0
        self.F_d = 0  # Placeholder values until analysis completes
        # this is added to prevent bugs if fits are not run
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, rotate=True,
                             nr_samples=2,
                             sample_0=0,
                             sample_1=1,
                             channels=['I', 'Q'],
                             no_fits=False,
                             print_fit_results=False,
                             pge=None, peg=None,
                             preselection=False,
                             n_bins: int = 120, **kw):

        self.add_analysis_datagroup_to_file()
        self.no_fits = no_fits
        # fixed fraction of ground state in the excited state histogram (relaxation)
        self.pge = pge
        # fixed fraction of excited state in the ground state hitogram (residual population)
        self.peg = peg
        self.get_naming_and_values()
        # plotting histograms of the raw shots on I and Q axis

        if len(self.channels) == 1:
            shots_I_data = self.get_values(key=self.channels[0])
            if not preselection:
                shots_I_data_0, shots_I_data_1 = a_tools.zigzag(
                    shots_I_data, sample_0, sample_1, nr_samples)
            else:
                shots_I_presel_0, shots_I_presel_1 = a_tools.zigzag(
                    shots_I_data, sample_0, sample_1, nr_samples)
                shots_I_data_0, shots_I_data_1 = a_tools.zigzag(
                    shots_I_data, sample_0 + 1, sample_1 + 1, nr_samples)
            shots_Q_data_0 = shots_I_data_0 * 0
            shots_Q_data_1 = shots_I_data_1 * 0
            if preselection:
                shots_Q_presel_0 = shots_I_presel_0 * 0
                shots_Q_presel_1 = shots_I_presel_1 * 0

        else:
            # Try getting data by name first and by index otherwise
            try:
                shots_I_data = self.get_values(key=self.channels[0])
                shots_Q_data = self.get_values(key=self.channels[1])
            except:
                shots_I_data = self.measured_values[0]
                shots_Q_data = self.measured_values[1]

            if not preselection:
                shots_I_data_0, shots_I_data_1 = a_tools.zigzag(
                    shots_I_data, sample_0, sample_1, nr_samples)
                shots_Q_data_0, shots_Q_data_1 = a_tools.zigzag(
                    shots_Q_data, sample_0, sample_1, nr_samples)
            else:
                shots_I_presel_0, shots_I_presel_1 = a_tools.zigzag(
                    shots_I_data, sample_0, sample_1, nr_samples)
                shots_Q_presel_0, shots_Q_presel_1 = a_tools.zigzag(
                    shots_Q_data, sample_0, sample_1, nr_samples)
                shots_I_data_0, shots_I_data_1 = a_tools.zigzag(
                    shots_I_data, sample_0 + 1, sample_1 + 1, nr_samples)
                shots_Q_data_0, shots_Q_data_1 = a_tools.zigzag(
                    shots_Q_data, sample_0 + 1, sample_1 + 1, nr_samples)

        # cutting off half data points (odd number of data points)
        min_len = np.min([np.size(shots_I_data_0), np.size(shots_I_data_1),
                          np.size(shots_Q_data_0), np.size(shots_Q_data_1)])
        shots_I_data_0 = shots_I_data_0[0:min_len]
        shots_I_data_1 = shots_I_data_1[0:min_len]
        shots_Q_data_0 = shots_Q_data_0[0:min_len]
        shots_Q_data_1 = shots_Q_data_1[0:min_len]
        if preselection:
            shots_I_presel_0 = shots_I_presel_0[0:min_len]
            shots_I_presel_1 = shots_I_presel_1[0:min_len]
            shots_Q_presel_0 = shots_Q_presel_0[0:min_len]
            shots_Q_presel_1 = shots_Q_presel_1[0:min_len]

        # rotating IQ-plane to transfer all information to the I-axis
        if self.rotate:
            theta, shots_I_data_1_rot, shots_I_data_0_rot = \
                self.optimize_IQ_angle(shots_I_data_1, shots_Q_data_1,
                                       shots_I_data_0, shots_Q_data_0, min_len,
                                       **kw)
            self.theta = theta
            if preselection:
                shots_presel_1_rot = np.cos(theta) * shots_I_presel_1 - \
                    np.sin(theta) * shots_Q_presel_1
                shots_presel_0_rot = np.cos(theta) * shots_I_presel_0 - \
                    np.sin(theta) * shots_Q_presel_0

        else:
            self.theta = 0
            shots_I_data_1_rot = shots_I_data_1
            shots_I_data_0_rot = shots_I_data_0
            if preselection:
                shots_presel_1_rot = shots_I_presel_1
                shots_presel_0_rot = shots_I_presel_0

        if kw.get('plot', True):
            self.plot_2D_histograms(shots_I_data_0, shots_Q_data_0,
                                    shots_I_data_1, shots_Q_data_1)

        self.no_fits_analysis(shots_I_data_1_rot, shots_I_data_0_rot, min_len,
                              **kw)
        if self.no_fits is False:
            # making gaussfits of s-curves
            self.s_curve_fits(shots_I_data_1_rot, shots_I_data_0_rot, min_len,
                              **kw)

        if preselection:
            try:
                V_th = self.V_th_d
            except:
                V_th = self.V_th_a
            s = np.sign(np.mean(shots_I_data_1_rot - shots_I_data_0_rot))
            shots_gmask_0 = s * (V_th - shots_presel_0_rot) > 0
            shots_gmask_1 = s * (V_th - shots_presel_1_rot) > 0

            shots_masked_0 = shots_I_data_0_rot[shots_gmask_0]
            shots_masked_1 = shots_I_data_1_rot[shots_gmask_1]

            self.total_points = np.size(shots_I_data_0_rot) + \
                np.size(shots_I_data_1_rot)
            self.removed_points = self.total_points - \
                np.size(shots_masked_0) - \
                np.size(shots_masked_1)

            min_len_masked = np.min([np.size(shots_masked_0),
                                     np.size(shots_masked_1)])
            shots_masked_0 = shots_masked_0[:min_len_masked]
            shots_masked_1 = shots_masked_1[:min_len_masked]

            self.no_fits_analysis(shots_masked_1, shots_masked_0,
                                  min_len_masked, masked=True, **kw)
            if self.no_fits is False:
                # making gaussfits of s-curves
                self.s_curve_fits(shots_masked_1, shots_masked_0,
                                  min_len_masked, masked=True, **kw)

        self.finish(**kw)

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
                                              density=True)
        H1, xedges1, yedges1 = np.histogram2d(shots_I_1, shots_Q_1,
                                              bins=n_bins,
                                              range=[[I_min, I_max, ],
                                                     [Q_min, Q_max, ]],
                                              density=True)

        # this part performs 2D gaussian fits and calculates coordinates of the
        # maxima
        def gaussian(height, center_x, center_y, width_x, width_y):
            width_x = float(width_x)
            width_y = float(width_y)
            return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + (
                (center_y - y) / width_y) ** 2) / 2)

        def fitgaussian(data):
            params = moments(data)

            def errorfunction(p): return np.ravel(gaussian(*p)(*np.indices(
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

    def no_fits_analysis(self, shots_I_1_rot, shots_I_0_rot, min_len,
                         masked=False, **kw):

        plot = kw.get('plot', True)

        min_voltage_1 = np.min(shots_I_1_rot)
        min_voltage_0 = np.min(shots_I_0_rot)
        min_voltage = np.min([min_voltage_1, min_voltage_0])

        max_voltage_1 = np.max(shots_I_1_rot)
        max_voltage_0 = np.max(shots_I_0_rot)
        max_voltage = np.max([max_voltage_1, max_voltage_0])

        hist_1, bins = np.histogram(shots_I_1_rot, bins=1000,
                                    range=(min_voltage, max_voltage),
                                    density=1)
        cumsum_1 = np.cumsum(hist_1)
        self.cumsum_1 = cumsum_1 / cumsum_1[-1]  # renormalizing

        hist_0, bins = np.histogram(shots_I_0_rot, bins=1000,
                                    range=(min_voltage, max_voltage),
                                    density=1)
        cumsum_0 = np.cumsum(hist_0)
        self.cumsum_0 = cumsum_0 / cumsum_0[-1]  # renormalizing

        cumsum_diff = (abs(self.cumsum_1 - self.cumsum_0))
        cumsum_diff_list = cumsum_diff.tolist()
        self.index_V_th_a = int(cumsum_diff_list.index(np.max(
            cumsum_diff_list)))
        V_th_a = bins[self.index_V_th_a] + (bins[1] - bins[0]) / 2
        # adding half a bin size
        F_a = 1 - (1 - cumsum_diff_list[self.index_V_th_a]) / 2

        if plot:
            fig, ax = plt.subplots()
            ax.plot(bins[0:-1], self.cumsum_1, label='cumsum_1', color='blue')
            ax.plot(bins[0:-1], self.cumsum_0, label='cumsum_0', color='red')
            ax.axvline(V_th_a, ls='--', label="V_th_a = %.3f" % V_th_a,
                       linewidth=2, color='grey')
            ax.text(.7, .6, '$Fa$ = %.4f' % F_a, transform=ax.transAxes,
                    fontsize='large')
            ax.set_title('raw cumulative histograms')
            plt.xlabel('DAQ voltage integrated (AU)', fontsize=14)
            plt.ylabel('Fraction', fontsize=14)

            # plt.hist(SS_Q_data, bins=40,label = '0 Q')
            plt.legend(loc=2)
            if masked:
                filename = 'raw-cumulative-histograms-masked'
            else:
                filename = 'raw-cumulative-histograms'
            self.save_fig(fig, figname=filename,
                          close_fig=self.close_fig, **kw)

        # saving the results
        if 'SSRO_Fidelity' not in self.analysis_group:
            fid_grp = self.analysis_group.create_group('SSRO_Fidelity')
        else:
            fid_grp = self.analysis_group['SSRO_Fidelity']
        fid_grp.attrs.create(name='V_th_a', data=V_th_a)
        fid_grp.attrs.create(name='F_a', data=F_a)

        self.F_a = F_a
        self.V_th_a = V_th_a

    def s_curve_fits(self, shots_I_1_rot, shots_I_0_rot, min_len, masked=False,
                     **kw):

        plot = kw.get('plot', True)

        # Sorting data for analytical fitting
        S_sorted_I_1 = np.sort(shots_I_1_rot)
        S_sorted_I_0 = np.sort(shots_I_0_rot)
        p_norm_I_1 = 1. * np.arange(len(S_sorted_I_1)) / \
            (len(S_sorted_I_1) - 1)
        p_norm_I_0 = 1. * np.arange(len(S_sorted_I_0)) / \
            (len(S_sorted_I_0) - 1)

        # fitting the curves with integral normal distribution
        def erfcc(x):
            """
            Complementary error function.
            """
            z = abs(x)
            out = np.zeros(np.size(x))
            t = 1. / (1. + 0.5 * z)
            r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 +
                                                                        t * (.09678418 + t * (
                                                                            -.18628806 + t * (.27886807 +
                                                                                              t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 +
                                                                                                                                        t * .17087277)))))))))
            if np.size(x) > 1:
                for k in range(np.size(x)):
                    if (x[k] >= 0.):
                        out[k] = r[k]
                    else:
                        out[k] = 2. - r[k]
            else:
                if (x > 0):
                    out = r
                else:
                    out = 2 - r
            return out

        def NormCdf(x, mu, sigma):
            t = x - mu
            y = 0.5 * erfcc(-t / (sigma * np.sqrt(2.0)))
            for k in range(np.size(x)):
                if y[k] > 1.0:
                    y[k] = 1.0
            return y

        NormCdfModel = lmfit.Model(NormCdf)

        def NormCdf2(x, mu0, mu1, sigma0, sigma1, frac1):
            t0 = x - mu0
            t1 = x - mu1
            frac0 = 1 - frac1
            y = frac1 * 0.5 * erfcc(-t1 / (sigma1 * np.sqrt(2.0))) + \
                frac0 * 0.5 * erfcc(-t0 / (sigma0 * np.sqrt(2.0)))
            for k in range(np.size(x)):
                if y[k] > 1.0:
                    y[k] = 1.0
            return y

        NormCdf2Model = lmfit.Model(NormCdf2)
        NormCdfModel.set_param_hint('mu', value=(np.average(shots_I_0_rot) +
                                                 np.average(shots_I_1_rot)) / 2)
        NormCdfModel.set_param_hint('sigma', value=(np.std(shots_I_0_rot) +
                                                    np.std(shots_I_1_rot)) / 2,
                                    min=0)

        params = NormCdfModel.make_params()

        fit_res_0 = NormCdfModel.fit(
            data=p_norm_I_0,
            x=S_sorted_I_0,
            params=params)

        fit_res_1 = NormCdfModel.fit(
            data=p_norm_I_1,
            x=S_sorted_I_1,
            params=params)
        # extracting the fitted parameters for the gaussian fits
        mu0 = fit_res_0.params['mu'].value
        sigma0 = fit_res_0.params['sigma'].value
        mu1 = fit_res_1.params['mu'].value
        sigma1 = fit_res_1.params['sigma'].value

        # setting hint parameters for double gaussfit of 'on' measurements
        NormCdf2Model.set_param_hint('mu0', value=mu0, vary=False)
        NormCdf2Model.set_param_hint('sigma0', value=sigma0, min=0, vary=False)
        NormCdf2Model.set_param_hint('mu1', value=np.average(shots_I_1_rot))
        NormCdf2Model.set_param_hint(
            'sigma1', value=np.std(shots_I_1_rot), min=0)
        if self.pge == None:
            NormCdf2Model.set_param_hint('frac1', value=0.9, min=0, max=1)
        else:
            NormCdf2Model.set_param_hint('frac1', value=1-self.pge, vary=False)

        # performing the double gaussfits of on 1 data
        params = NormCdf2Model.make_params()
        fit_res_double_1 = NormCdf2Model.fit(
            data=p_norm_I_1,
            x=S_sorted_I_1,
            params=params)

        # extracting the fitted parameters for the double gaussian fit 'on'
        sigma0_1 = fit_res_double_1.params['sigma0'].value
        sigma1_1 = fit_res_double_1.params['sigma1'].value
        mu0_1 = fit_res_double_1.params['mu0'].value
        mu1_1 = fit_res_double_1.params['mu1'].value
        frac1_1 = fit_res_double_1.params['frac1'].value

        NormCdf2Model = lmfit.Model(NormCdf2)
        # adding hint parameters for double gaussfit of 'off' measurements
        NormCdf2Model.set_param_hint('mu0', value=mu0)
        NormCdf2Model.set_param_hint('sigma0', value=sigma0, min=0)
        NormCdf2Model.set_param_hint('mu1', value=mu1_1, vary=False)
        NormCdf2Model.set_param_hint(
            'sigma1', value=sigma1_1, min=0, vary=False)
        if self.peg == None:
            NormCdf2Model.set_param_hint(
                'frac1', value=0.025, min=0, max=1, vary=True)
        else:
            NormCdf2Model.set_param_hint(
                'frac1', value=self.peg, vary=False)

        params = NormCdf2Model.make_params()
        fit_res_double_0 = NormCdf2Model.fit(
            data=p_norm_I_0,
            x=S_sorted_I_0,
            params=params)

        # extracting the fitted parameters for the double gaussian fit 'off'
        sigma0_0 = fit_res_double_0.params['sigma0'].value
        sigma1_0 = fit_res_double_0.params['sigma1'].value
        mu0_0 = fit_res_double_0.params['mu0'].value
        mu1_0 = fit_res_double_0.params['mu1'].value
        frac1_0 = fit_res_double_0.params['frac1'].value

        def NormCdf(x, mu, sigma):
            t = x - mu
            y = 0.5 * erfcc(-t / (sigma * np.sqrt(2.0)))
            return y

        def NormCdfdiff(x, mu0=mu0, mu1=mu1, sigma0=sigma0, sigma1=sigma1):
            y = -abs(NormCdf(x, mu0, sigma0) - NormCdf(x, mu1, sigma1))
            return y

        V_opt_single = optimize.brent(NormCdfdiff)
        F_single = -NormCdfdiff(x=V_opt_single)

        # print 'V_opt_single', V_opt_single
        # print 'F_single', F_single

        # redefining the function with different variables to avoid problems
        # with arguments in brent optimization
        def NormCdfdiff(x, mu0=mu0_0, mu1=mu1_1, sigma0=sigma0_0,
                        sigma1=sigma1_1):
            y0 = -abs(NormCdf(x, mu0, sigma0) - NormCdf(x, mu1, sigma1))
            return y0

        self.V_th_d = optimize.brent(NormCdfdiff)
        F_d = 1 - (1 + NormCdfdiff(x=self.V_th_d)) / 2

        # print 'F_corrected',F_corrected

        def NormCdfdiffDouble(x, mu0_0=mu0_0,
                              sigma0_0=sigma0_0, sigma1_0=sigma1_0,
                              frac1_0=frac1_0, mu1_1=mu1_1,
                              sigma0_1=sigma0_1, sigma1_1=sigma1_1,
                              frac1_1=frac1_1):
            distr0 = (1 - frac1_0) * NormCdf(x, mu0_0, sigma0_0) + \
                     (frac1_0) * NormCdf(x, mu1_1, sigma1_1)

            distr1 = (1 - frac1_1) * NormCdf(x, mu0_0, sigma0_0) + \
                     (frac1_1) * NormCdf(x, mu1_1, sigma1_1)
            y = - abs(distr1 - distr0)
            return y

        # print "refresh"
        # self.V_th_d = optimize.brent(NormCdfdiffDouble)
        # F_d = -NormCdfdiffDouble(x=self.V_th_d)

        # calculating the signal-to-noise ratio
        signal = abs(mu0_0 - mu1_1)
        noise = (sigma0_0 + sigma1_1) / 2
        SNR = signal / noise

        if plot:
            # plotting s-curves
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title('S-curves (not binned) and fits, determining fidelity '
                         'and threshold optimum, %s shots' % min_len)
            ax.set_xlabel('DAQ voltage integrated (V)')  # , fontsize=14)
            ax.set_ylabel('Fraction of counts')  # , fontsize=14)
            ax.set_ylim((-.01, 1.01))
            ax.plot(S_sorted_I_0, p_norm_I_0, label='0 I', linewidth=2,
                    color='red')
            ax.plot(S_sorted_I_1, p_norm_I_1, label='1 I', linewidth=2,
                    color='blue')

            # ax.plot(S_sorted_I_0, fit_res_0.best_fit,
            #         label='0 I single gaussian fit', ls='--', linewidth=3,
            #         color='lightblue')
            # ax.plot(S_sorted_I_1, fit_res_1.best_fit, label='1 I',
            #         linewidth=2, color='red')

            ax.plot(S_sorted_I_0, fit_res_double_0.best_fit,
                    label='0 I double gaussfit', ls='--', linewidth=3,
                    color='darkred')
            ax.plot(S_sorted_I_1, fit_res_double_1.best_fit,
                    label='1 I double gaussfit', ls='--', linewidth=3,
                    color='lightblue')
            labelstring = 'V_th_a= %.3f V' % (self.V_th_a)
            labelstring_corrected = 'V_th_d= %.3f V' % (self.V_th_d)

            ax.axvline(self.V_th_a, ls='--', label=labelstring,
                       linewidth=2, color='grey')
            ax.axvline(self.V_th_d, ls='--', label=labelstring_corrected,
                       linewidth=2, color='black')

            leg = ax.legend(loc='best')
            leg.get_frame().set_alpha(0.5)
            if masked:
                filename = 'S-curves-masked'
            else:
                filename = 'S-curves'
            self.save_fig(fig, figname=filename, **kw)

            # plotting the histograms
            fig, axes = plt.subplots(figsize=(7, 4))
            n1, bins1, patches = pylab.hist(shots_I_1_rot, bins=40,
                                            label='1 I', histtype='step',
                                            color='red', density=False)
            n0, bins0, patches = pylab.hist(shots_I_0_rot, bins=40,
                                            label='0 I', histtype='step',
                                            color='blue', density=False)
            pylab.clf()
            # n0, bins0 = np.histogram(shots_I_0_rot, bins=int(min_len/50),
            #                          normed=1)
            # n1, bins1 = np.histogram(shots_I_1_rot, bins=int(min_len/50),
            #                          normed=1)

            gdat, = pylab.plot(bins0[:-1] + 0.5 *
                               (bins0[1] - bins0[0]), n0, 'C0o')
            edat, = pylab.plot(bins1[:-1] + 0.5 *
                               (bins1[1] - bins1[0]), n1, 'C3o')

            # n, bins1, patches = np.hist(shots_I_1_rot, bins=int(min_len/50),
            #                               label = '1 I',histtype='step',
            #                               color='red',normed=1)
            # n, bins0, patches = pylab.hist(shots_I_0_rot, bins=int(min_len/50),
            #                               label = '0 I',histtype='step',
            #                               color='blue',normed=1)

            # add lines showing the fitted distribution
            # building up the histogram fits for off measurements

            norm0 = (bins0[1] - bins0[0]) * min_len
            norm1 = (bins1[1] - bins1[0]) * min_len

            y0 = norm0 * (1 - frac1_0) * stats.norm.pdf(bins0, mu0_0, sigma0_0) + \
                norm0 * frac1_0 * stats.norm.pdf(bins0, mu1_0, sigma1_0)
            y1_0 = norm0 * frac1_0 * stats.norm.pdf(bins0, mu1_0, sigma1_0)
            y0_0 = norm0 * (1 - frac1_0) * \
                stats.norm.pdf(bins0, mu0_0, sigma0_0)

            # building up the histogram fits for on measurements
            y1 = norm1 * (1 - frac1_1) * stats.norm.pdf(bins1, mu0_1, sigma0_1) + \
                norm1 * frac1_1 * stats.norm.pdf(bins1, mu1_1, sigma1_1)
            y1_1 = norm1 * frac1_1 * stats.norm.pdf(bins1, mu1_1, sigma1_1)
            y0_1 = norm1 * (1 - frac1_1) * \
                stats.norm.pdf(bins1, mu0_1, sigma0_1)

            pylab.semilogy(bins0, y0, 'C0', linewidth=1.5)
            pylab.semilogy(bins0, y1_0, 'C0--', linewidth=3.5)
            pylab.semilogy(bins0, y0_0, 'C0--', linewidth=3.5)

            pylab.semilogy(bins1, y1, 'C3', linewidth=1.5)
            pylab.semilogy(bins1, y0_1, 'C3--', linewidth=3.5)
            pylab.semilogy(bins1, y1_1, 'C3--', linewidth=3.5)
            pdf_max = (max(max(y0), max(y1)))
            (pylab.gca()).set_ylim(pdf_max / 1000, 2 * pdf_max)

            plt.title('Histograms of {} shots, {}'.format(
                min_len, self.timestamp_string))
            plt.xlabel('DAQ voltage integrated (V)')
            plt.ylabel('Number of counts')

            thaline = plt.axvline(self.V_th_a, ls='--', linewidth=1,
                                  color='grey')
            thdline = plt.axvline(self.V_th_d, ls='--', linewidth=1,
                                  color='black')
            nomarker = matplotlib.patches.Rectangle((0, 0), 0, 0, alpha=0.0)

            markers = [gdat, edat, thaline, thdline, nomarker, nomarker,
                       nomarker]
            labels = [r'$\left| g \right\rangle$ prepared',
                      r'$\left| e \right\rangle$ prepared',
                      '$F_a$ = {:.4f}'.format(self.F_a),
                      '$F_d$ = {:.4f}'.format(F_d),
                      'SNR = {:.2f}'.format(SNR),
                      '$p(e|0)$ = {:.4f}'.format(frac1_0),
                      '$p(g|\pi)$ = {:.4f}'.format(1 - frac1_1)]
            if masked:
                p_rem = self.removed_points / self.total_points
                markers += [nomarker]
                labels += ['$p_{{rem}}$ = {:.4f}'.format(p_rem)]
            lgd = plt.legend(markers, labels, bbox_to_anchor=(1.05, 1),
                             loc=2, borderaxespad=0., framealpha=0.5)

            if masked:
                filename = 'Histograms-masked'
            else:
                filename = 'Histograms'

            self.save_fig(fig, figname=filename, **kw)

        self.save_fitted_parameters(fit_res_double_0,
                                    var_name='fit_res_double_0')
        self.save_fitted_parameters(fit_res_double_1,
                                    var_name='fit_res_double_1')

        if 'SSRO_Fidelity' not in self.analysis_group:
            fid_grp = self.analysis_group.create_group('SSRO_Fidelity')
        else:
            fid_grp = self.analysis_group['SSRO_Fidelity']

        fid_grp.attrs.create(name='sigma0_0', data=sigma0_0)
        fid_grp.attrs.create(name='sigma1_1', data=sigma1_1)
        fid_grp.attrs.create(name='sigma0_1', data=sigma0_1)
        fid_grp.attrs.create(name='sigma1_0', data=sigma1_0)
        fid_grp.attrs.create(name='mu0_1', data=mu0_1)
        fid_grp.attrs.create(name='mu1_0', data=mu1_0)

        fid_grp.attrs.create(name='mu0_0', data=mu0_0)
        fid_grp.attrs.create(name='mu1_1', data=mu1_1)
        fid_grp.attrs.create(name='frac1_0', data=frac1_0)
        fid_grp.attrs.create(name='frac1_1', data=frac1_1)
        fid_grp.attrs.create(name='F_d', data=F_d)
        fid_grp.attrs.create(name='SNR', data=SNR)
        fid_grp.attrs.create(name='V_th_a', data=self.V_th_a)
        fid_grp.attrs.create(name='V_th_d', data=self.V_th_d)
        fid_grp.attrs.create(name='F_a', data=self.F_a)

        self.sigma0_0 = sigma0_0
        self.sigma1_1 = sigma1_1
        self.mu0_0 = mu0_0
        self.mu1_1 = mu1_1
        self.frac1_0 = frac1_0
        self.frac1_1 = frac1_1
        self.F_d = F_d
        self.SNR = SNR
        # Add the fit and data to the analysis object
        self.n0 = n0 # x-Data for the histograms
        self.n1 = n1 # x-Data for the histograms
        self.bins0 = bins0 # y-Data for the histograms
        self.bins1 = bins1 # y-Data for the histograms
        self.norm0 = norm0 # x-values of histogram fit
        self.y0 = y0 # y-values of histogram fit
        self.y0_0 = y0_0
        self.y0_1 = y0_1
        self.norm0 = norm0 # x-values of histogram fit
        self.y1 = y1 # y-values of histogram fit
        self.y1_0 = y1_0
        self.y1_1 = y1_1

    def plot_2D_histograms(self, shots_I_0, shots_Q_0, shots_I_1, shots_Q_1,
                           **kw):
        cmap = kw.pop('cmap', 'viridis')

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
                                              density=True)
        H1, xedges1, yedges1 = np.histogram2d(shots_I_1, shots_Q_1,
                                              bins=n_bins,
                                              range=[[I_min, I_max, ],
                                                     [Q_min, Q_max, ]],
                                              density=True)

        fig, axarray = plt.subplots(nrows=1, ncols=2)
        axarray[0].tick_params(axis='both', which='major',
                               labelsize=5, direction='out')
        axarray[1].tick_params(axis='both', which='major',
                               labelsize=5, direction='out')

        plt.subplots_adjust(hspace=20)

        axarray[0].set_title('2D histogram, pi pulse')
        im1 = axarray[0].imshow(np.transpose(H1), interpolation='nearest',
                                origin='lower', aspect='auto',
                                extent=[xedges1[0], xedges1[-1],
                                        yedges1[0], yedges1[-1]], cmap=cmap)

        set_xlabel(axarray[0], self.value_names[0], self.value_units[0])
        if len(self.channels) == 2:
            set_ylabel(axarray[0], self.value_names[
                1], self.value_units[1])
        else:
            set_ylabel(axarray[0], 'Dummy axis')
        # axarray[0].set_xlim(-edge, edge)
        # axarray[0].set_ylim(-edge, edge)

        # plotting 2D histograms of mmts with no pulse
        axarray[1].set_title('2D histogram, no pi pulse')
        im0 = axarray[1].imshow(np.transpose(H0), interpolation='nearest',
                                origin='lower', aspect='auto',
                                extent=[xedges0[0], xedges0[-1], yedges0[0],
                                        yedges0[-1]], cmap=cmap)

        set_xlabel(axarray[1], self.value_names[0], self.value_units[0])
        if len(self.channels) == 2:
            set_ylabel(axarray[1], self.value_names[
                1], self.value_units[1])
        else:
            set_ylabel(axarray[1], 'Dummy axis')
        # axarray[1].set_xlim(-edge, edge)
        # axarray[1].set_ylim(-edge, edge)
        self.save_fig(fig, figname='SSRO_Density_Plots',
                      close_fig=self.close_fig, **kw)


class SSRO_discrimination_analysis(MeasurementAnalysis):
    '''
    Analysis that takes IQ-shots and extracts discrimination fidelity from
    it by fitting 2 2D gaussians. It does not assumption on what state the
    individual shots belong to.

    This method will only work if the gaussians belonging to both distributions
    are distinguisable.

    The 2D gauss does not include squeezing and assumes symmetric (in x/y)
    distributions.
    '''

    def __init__(self, **kw):
        kw['h5mode'] = 'r+'
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, plot_2D_histograms=True,
                             current_threshold=None, theta_in=0,
                             n_bins: int = 120, **kw):
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        I_shots = self.measured_values[0]
        Q_shots = self.measured_values[1]

        if theta_in != 0:
            shots = I_shots + 1j * Q_shots
            rot_shots = dm_tools.rotate_complex(
                shots, angle=theta_in, deg=True)
            I_shots = rot_shots.real
            Q_shots = rot_shots.imag

        # Reshaping the data
        # min min and max max constructions exist so that it also works
        # if one dimension only conatins zeros
        H, xedges, yedges = np.histogram2d(I_shots, Q_shots,
                                           bins=n_bins,
                                           range=[[min(min(I_shots), -1e-6),
                                                   max(max(I_shots), 1e-6)],
                                                  [min(min(Q_shots), -1e-6),
                                                   max(max(Q_shots), 1e-6)]],
                                           density=True)
        self.H = H
        self.xedges = xedges
        self.yedges = yedges
        H_flat, x_tiled, y_rep = dm_tools.flatten_2D_histogram(
            H, xedges, yedges)

        # Performing the fits
        g2_mod = fit_mods.DoubleGauss2D_model
        params = g2_mod.guess(model=g2_mod, data=H_flat, x=x_tiled, y=y_rep)
        # assume symmetry of the gaussian blobs in x and y
        params['A_sigma_y'].set(expr='A_sigma_x')
        params['B_sigma_y'].set(expr='B_sigma_x')
        self.fit_res = g2_mod.fit(data=H_flat, x=x_tiled, y=y_rep,
                                  params=params)

        # Saving the fit results to the datafile
        self.save_fitted_parameters(self.fit_res, 'Double gauss fit')
        if plot_2D_histograms:  # takes ~350ms, speedup quite noticable
            fig, axs = plt.subplots(nrows=1, ncols=3)
            fit_mods.plot_fitres2D_heatmap(self.fit_res, x_tiled, y_rep,
                                           axs=axs, cmap='viridis')
            for ax in axs:
                ax.ticklabel_format(style='sci',
                                    scilimits=(0, 0))
                set_xlabel(ax, 'I', self.value_units[0])
                edge = max(max(abs(xedges)), max(abs(yedges)))
                ax.set_xlim(-edge, edge)
                ax.set_ylim(-edge, edge)

            set_ylabel(axs[0], 'Q', self.value_units[1])


            self.save_fig(
                fig, figname='2D-Histograms_rot_{:.1f} deg'.format(theta_in), **kw)

        #######################################################
        #         Extract quantities of interest              #
        #######################################################
        self.mu_a = (self.fit_res.params['A_center_x'].value +
                     1j * self.fit_res.params['A_center_y'].value)
        self.mu_b = (self.fit_res.params['B_center_x'].value +
                     1j * self.fit_res.params['B_center_y'].value)

        # only look at sigma x because we assume sigma_x = sigma_y
        sig_a = self.fit_res.params['A_sigma_x'].value
        sig_b = self.fit_res.params['B_sigma_x'].value
        # Picking threshold in the middle assumes same sigma for both
        # distributions, this can be improved by optimizing the F_discr
        diff_vec = self.mu_b - self.mu_a

        self.opt_I_threshold = np.mean([self.mu_a.real, self.mu_b.real])
        self.theta = np.angle(diff_vec, deg=True)
        self.mean_sigma = np.mean([sig_a, sig_b])
        # relative separation of the gaussians in units of sigma
        self.relative_separation = abs(diff_vec) / self.mean_sigma
        # relative separation of the gaussians when projected on the I-axis
        self.relative_separation_I = diff_vec.real / self.mean_sigma

        #######################################################
        # Calculating discrimanation fidelities based on erfc #
        #######################################################
        # CDF of gaussian is P(X<=x) = .5 erfc((mu-x)/(sqrt(2)sig))

        # Along the optimal direction
        CDF_a = .5 * math.erfc((abs(diff_vec / 2)) /
                               (np.sqrt(2) * sig_a))
        CDF_b = .5 * math.erfc((-abs(diff_vec / 2)) /
                               (np.sqrt(2) * sig_b))
        self.F_discr = 1 - (1 - abs(CDF_a - CDF_b)) / 2

        # Projected on the I-axis
        CDF_a = .5 * math.erfc((self.mu_a.real - self.opt_I_threshold) /
                               (np.sqrt(2) * sig_a))
        CDF_b = .5 * math.erfc((self.mu_b.real - self.opt_I_threshold) /
                               (np.sqrt(2) * sig_b))

        self.F_discr_I = abs(CDF_a - CDF_b)
        # Current threshold projected on the I-axis
        if current_threshold is not None:
            CDF_a = .5 * math.erfc((self.mu_a.real - current_threshold) /
                                   (np.sqrt(2) * sig_a))
            CDF_b = .5 * math.erfc((self.mu_b.real - current_threshold) /
                                   (np.sqrt(2) * sig_b))
            self.F_discr_curr_t = 1 - (1 - abs(CDF_a - CDF_b)) / 2

        self.finish(**kw)


class touch_n_go_SSRO_Analysis(MeasurementAnalysis):
    '''
    Script to analyze the single shots used for touch and go selection
    '''

    def __init__(self, label='touch_n_go', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):
        self.add_analysis_datagroup_to_file()

        # plotting histograms of the raw shots on I and Q axis

        shots_I_data = self.get_values(key='touch_n_go_I_shots')
        shots_Q_data = self.get_values(key='touch_n_go_Q_shots')
        instrument_settings = self.data_file['Instrument settings']
        threshold = instrument_settings['CBox'].attrs['signal_threshold_line0']
        # plotting the histograms before rotation
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.hist(shots_I_data, bins=100, label='I', histtype='step', normed=1)
        # axes.hist(shots_Q_data, bins=40, label = '0 Q',histtype='step',normed=1)
        axes.axvline(x=threshold, ls='--', label='threshold')

        axes.set_title(
            'Histogram of I-shots for touch and go measurement and threshold')
        plt.xlabel('DAQ voltage integrated (AU)', fontsize=14)
        plt.ylabel('Fraction', fontsize=14)

        # plt.hist(SS_Q_data, bins=40,label = '0 Q')
        plt.legend()
        self.save_fig(fig, figname='raw-histograms', **kw)
        plt.show()

        self.finish(**kw)


class SSRO_single_quadrature_discriminiation_analysis(MeasurementAnalysis):
    '''
    Analysis that fits two gaussians to a histogram of a dataset.
    Uses this to extract F_discr and the optimal threshold
    '''

    def __init__(self, weight_func: str = None, **kw):
        """
        Bin all acquired data into historgrams and fit two gaussians to
        determine the
        """
        # Note: weight_func is a bit of misnomer here
        # it represents the channel/weight of the data we want to bin
        kw['h5mode'] = 'r+'
        self.weight_func = weight_func
        super().__init__(**kw)

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        hist, bins, centers = self.histogram_shots(self.shots)
        self.fit_data(hist, centers)
        self.F_discr, self.opt_threshold = \
            self.calculate_discrimination_fidelity(fit_res=self.fit_res)

        self.make_figures(hist=hist, centers=centers, **kw)

        if close_file:
            self.data_file.close()
        return

    def get_naming_and_values(self):
        super().get_naming_and_values()
        if type(self.weight_func) is str:
            self.shots = self.get_values(self.weight_func)
            # Potentially bug sensitive!!
            self.units = self.value_units[0]
        elif type(self.weight_func) is int:
            self.shots = self.measured_values[self.weight_func]
            self.units = self.value_units[self.weight_func]
        elif self.weight_func is None:
            self.weight_func = self.value_names[0]
            self.shots = self.measured_values[0]
            self.units = self.value_units[0]

    def histogram_shots(self, shots):
        hist, bins = np.histogram(shots, bins=90, density=True)
        # 0.7 bin widht is a sensible default for plotting
        centers = (bins[:-1] + bins[1:]) / 2
        return hist, bins, centers

    def fit_data(self, hist, centers):
        self.add_analysis_datagroup_to_file()
        self.model = fit_mods.DoubleGaussModel
        params = self.model.guess(self.model, hist, centers)
        self.fit_res = self.model.fit(data=hist, x=centers, params=params)
        self.save_fitted_parameters(
            fit_res=self.fit_res, var_name='{}shots'.format(self.weight_func))
        return self.fit_res

    def make_figures(self, hist, centers, show_guess=False, **kw):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        width = .7 * (centers[1] - centers[0])
        plot_title = kw.pop('plot_title', textwrap.fill(
            self.timestamp_string + '_' +
            self.measurementstring, 40))

        x_fine = np.linspace(min(centers),
                             max(centers), 1000)
        # Plotting the data
        self.ax.bar(centers, hist, align='center', width=width, label='data')

        pars = self.fit_res.best_values
        Agauss = lmfit.models.gaussian(x=x_fine, sigma=pars['A_sigma'],
                                       amplitude=pars['A_amplitude'],
                                       center=pars['A_center'])

        Bgauss = lmfit.models.gaussian(x=x_fine, sigma=pars['B_sigma'],
                                       amplitude=pars['B_amplitude'],
                                       center=pars['B_center'])

        self.ax.plot(x_fine, self.fit_res.eval(x=x_fine), label='fit', c='r')
        self.ax.plot(x_fine, Agauss, label='fit_A', c='r', ls='--')
        self.ax.plot(x_fine, Bgauss, label='fit_B', c='r', ls='--')

        if show_guess:
            self.ax.plot(x_fine, self.fit_res.eval(
                x=x_fine, **self.fit_res.init_values), label='guess', c='g')
            self.ax.legend(loc='best')

        ylim = self.ax.get_ylim()
        self.ax.vlines(self.opt_threshold, ylim[0], ylim[1], linestyles='--',
                       label='opt. threshold')
        self.ax.text(.95, .95, 'F_discr {:.2f}\nOpt.thresh. {:.2f}'.format(
            self.F_discr, self.opt_threshold),
            verticalalignment='top', horizontalalignment='right',
            transform=self.ax.transAxes)
        self.ax.legend()

        # Prettifying the plot
        self.ax.ticklabel_format(useOffset=False)
        self.ax.set_title(plot_title)
        self.ax.set_xlabel('{} ({})'.format(self.weight_func, self.units))
        self.ax.set_ylabel('normalized counts')
        self.save_fig(self.fig, fig_tight=True, **kw)

    def calculate_discrimination_fidelity(self, fit_res):
        '''
        Calculate fidelity based on the overlap of the two fits.
        Does this by numerically evaluating the function.
        Analytic is possible but not done here.
        '''
        mu_a = fit_res.best_values['A_center']
        mu_b = fit_res.best_values['B_center']
        s_a = fit_res.best_values['A_sigma']
        s_b = fit_res.best_values['B_sigma']

        x_fine = np.linspace(min(mu_a - 4 * s_a, mu_b - 4 * s_b),
                             max(mu_b + 4 * s_a, mu_b + 4 * s_b), 1000)
        CDF_a = np.zeros(len(x_fine))
        CDF_b = np.zeros(len(x_fine))
        for i, x in enumerate(x_fine):
            CDF_a[i] = .5 * erfc((mu_a - x) / (np.sqrt(2) * s_a))
            CDF_b[i] = .5 * erfc((mu_b - x) / (np.sqrt(2) * s_b))
        F_discr_conservative = np.max(abs(CDF_a - CDF_b))
        F_discr = 1 - (1 - F_discr_conservative) / 2
        opt_threshold = x_fine[np.argmax(abs(CDF_a - CDF_b))]
        return F_discr, opt_threshold


class T1_Analysis(TD_Analysis):
    """
    Most kw parameters for Rabi_Analysis are also used here.
    """

    def __init__(self, label='T1', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super().__init__(**kw)

    def fit_T1(self, **kw):

        # Guess for params
        fit_mods.ExpDecayModel.set_param_hint('amplitude',
                                              value=1,
                                              min=0,
                                              max=2)
        fit_mods.ExpDecayModel.set_param_hint('tau',
                                              value=self.sweep_points[1] * 50,
                                              min=self.sweep_points[1],
                                              max=self.sweep_points[-1] * 1000)
        fit_mods.ExpDecayModel.set_param_hint('offset',
                                              value=0,
                                              vary=False)
        fit_mods.ExpDecayModel.set_param_hint('n',
                                              value=1,
                                              vary=False)
        self.params = fit_mods.ExpDecayModel.make_params()

        fit_res = fit_mods.ExpDecayModel.fit(data=self.normalized_data_points,
                                             t=self.sweep_points[:-
                                                                 self.NoCalPoints],
                                             params=self.params)

        if kw.get('print_fit_results', False):
            print(fit_res.fit_report())

        return fit_res

    def run_default_analysis(self, show=False, close_file=False, **kw):

        super().run_default_analysis(show=show,
                                     close_file=close_file,
                                     close_main_figure=True,
                                     save_fig=False, **kw)

        show_guess = kw.get('show_guess', False)
        # make_fig = kw.get('make_fig',True)

        self.add_analysis_datagroup_to_file()

        # Perform fit and save fitted parameters
        self.fit_res = self.fit_T1(**kw)
        self.save_fitted_parameters(fit_res=self.fit_res, var_name='F|1>')

        # Create self.T1 and self.T1_stderr and save them
        self.get_measured_T1()  # in seconds
        self.save_computed_parameters(
            self.T1_dict, var_name=self.value_names[0])

        T1_micro_sec = self.T1_dict['T1'] * 1e6
        T1_err_micro_sec = self.T1_dict['T1_stderr'] * 1e6
        # Print T1 and error on screen
        if kw.get('print_parameters', False):
            print('T1 = {:.5f} ('.format(T1_micro_sec) + 'μs) \t '
                                                         'T1 StdErr = {:.5f} ('.format(
                T1_err_micro_sec) + 'μs)')

        # Plot best fit and initial fit + data
        if self.make_fig:

            units = SI_prefix_and_scale_factor(val=max(abs(self.ax.get_xticks())),
                                               unit=self.sweep_unit[0])[1]
            # Get old values
            instr_set = self.data_file['Instrument settings']
            try:
                if self.for_ef:
                    T1_old = float(
                        instr_set[self.qb_name].attrs['T1_ef']) * 1e6
                else:
                    T1_old = float(instr_set[self.qb_name].attrs['T1']) * 1e6
                old_vals = '\nold $T_1$ = {:.5f} '.format(T1_old) + units
            except (TypeError, KeyError, ValueError):
                logging.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
                old_vals = ''

            textstr = ('$T_1$ = {:.5f} '.format(T1_micro_sec) +
                       units +
                       ' $\pm$ {:.5f} '.format(T1_err_micro_sec) +
                       units + old_vals)

            self.fig.text(0.5, 0, textstr, transform=self.ax.transAxes,
                          fontsize=self.font_size,
                          verticalalignment='top',
                          horizontalalignment='center',
                          bbox=self.box_props)

            if show_guess:
                self.ax.plot(self.sweep_points[:-self.NoCalPoints],
                             self.fit_res.init_fit, 'k--', linewidth=self.line_width)

            best_vals = self.fit_res.best_values
            t = np.linspace(self.sweep_points[0],
                            self.sweep_points[-self.NoCalPoints], 1000)

            y = fit_mods.ExpDecayFunc(
                t, tau=best_vals['tau'],
                n=best_vals['n'],
                amplitude=best_vals['amplitude'],
                offset=best_vals['offset'])

            self.ax.plot(t, y, 'r-', linewidth=self.line_width)

            self.ax.locator_params(axis='x', nbins=6)

            if show:
                plt.show()

            self.save_fig(
                self.fig, figname=self.measurementstring + '_Fit', **kw)

        if close_file:
            self.data_file.close()

        return self.fit_res

    def get_measured_T1(self):
        fitted_pars = self.data_file['Analysis']['Fitted Params F|1>']

        self.T1 = fitted_pars['tau'].attrs['value']
        T1_stderr = fitted_pars['tau'].attrs['stderr']
        # T1 = self.fit_res.params['tau'].value
        # T1_stderr = self.fit_res.params['tau'].stderr

        # return as dict for use with "save_computed_parameters"; units are
        # seconds
        self.T1_dict = {'T1': self.T1, 'T1_stderr': T1_stderr}

        return self.T1, T1_stderr


class Ramsey_Analysis(TD_Analysis):
    """
    Now has support for one and two artificial_detuning values. If the
    keyword parameter "artificial_detuning" is passed as an int or a list with
    one element, the old Ramsey routine, now under "one_art_det_analysis", will
    be used. If it is passed in as a list with 2 elements (there is only support
    for 2 artificial detunings), the new routine, "two_art_dets_analysis" will
    be used.

    Most kw parameters for Rabi_Analysis are also used here.
    """

    def __init__(self, label='Ramsey', phase_sweep_only=False, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        self.phase_sweep_only = phase_sweep_only
        self.artificial_detuning = kw.pop('artificial_detuning', 0)
        if self.artificial_detuning == 0:
            logging.warning('Artificial detuning is unknown. Defaults to %s MHz. '
                            'New qubit frequency might be incorrect.'
                            % self.artificial_detuning)

        # The routines for 2 art_dets does not use the self.fig and self.ax
        # created in TD_Analysis for make_fig==False for TD_Analysis but
        # still want make_fig to decide whether two_art_dets_analysis should
        # make a figure
        self.make_fig_two_dets = kw.get('make_fig', True)
        if (type(self.artificial_detuning) is list) and \
                (len(self.artificial_detuning) > 1):
            kw['make_fig'] = False

        super(Ramsey_Analysis, self).__init__(**kw)

    def fit_Ramsey(self, x, y, **kw):

        print_fit_results = kw.pop('print_fit_results', False)
        damped_osc_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
        average = np.mean(y)

        ft_of_data = np.fft.fft(y)
        index_of_fourier_maximum = np.argmax(np.abs(
            ft_of_data[1:len(ft_of_data) // 2])) + 1
        max_ramsey_delay = x[-1] - x[0]

        fft_axis_scaling = 1 / (max_ramsey_delay)
        freq_est = fft_axis_scaling * index_of_fourier_maximum
        est_number_of_periods = index_of_fourier_maximum
        if self.phase_sweep_only:
            damped_osc_mod.set_param_hint('frequency',
                                          value=1/360,
                                          vary=False)
            damped_osc_mod.set_param_hint('phase',
                                          value=0, vary=True)
            damped_osc_mod.set_param_hint('amplitude',
                                          value=0.5 *
                                          (max(self.normalized_data_points) -
                                           min(self.normalized_data_points)),
                                          min=0.0, max=4.0)
            fixed_tau = 1e9
            damped_osc_mod.set_param_hint('tau',
                                          value=fixed_tau,
                                          vary=False)
        else:
            if ((average > 0.7*max(y)) or
                    (est_number_of_periods < 2) or
                    est_number_of_periods > len(ft_of_data)/2.):
                print('the trace is too short to find multiple periods')

                if print_fit_results:
                    print('Setting frequency to 0 and ' +
                          'fitting with decaying exponential.')
                damped_osc_mod.set_param_hint('frequency',
                                              value=freq_est,
                                              vary=False)
                damped_osc_mod.set_param_hint('phase',
                                              value=0,
                                              vary=False)
            else:
                damped_osc_mod.set_param_hint('frequency',
                                              value=freq_est,
                                              vary=True,
                                              min=(1/(100 * x[-1])),
                                              max=(20/x[-1]))

            if (np.average(y[:4]) >
                    np.average(y[4:8])):
                phase_estimate = 0
            else:
                phase_estimate = np.pi
            damped_osc_mod.set_param_hint('phase',
                                          value=phase_estimate, vary=True)

            amplitude_guess = 1
            damped_osc_mod.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.4,
                                          max=4.0)
            damped_osc_mod.set_param_hint('tau',
                                          value=x[1]*10,
                                          min=x[1],
                                          max=x[1]*1000)
        damped_osc_mod.set_param_hint('exponential_offset',
                                      value=0.5,
                                      min=0.4,
                                      max=4.0)
        damped_osc_mod.set_param_hint('oscillation_offset',
                                      value=0,
                                      vary=False)
        damped_osc_mod.set_param_hint('n',
                                      value=1,
                                      vary=False)
        self.params = damped_osc_mod.make_params()

        fit_res = damped_osc_mod.fit(data=y,
                                     t=x,
                                     params=self.params)
        if self.phase_sweep_only:
            chi_sqr_bound = 0
        else:
            chi_sqr_bound = 0.35

        if fit_res.chisqr > chi_sqr_bound:
            logging.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2 * np.pi, 8):
                damped_osc_mod.set_param_hint('phase',
                                              value=phase_estimate)
                self.params = damped_osc_mod.make_params()
                fit_res_lst += [damped_osc_mod.fit(
                    data=y,
                    t=x,
                    params=self.params)]

            chisqr_lst = [fit_res.chisqr for fit_res in fit_res_lst]
            fit_res = fit_res_lst[np.argmin(chisqr_lst)]
        self.fit_results.append(fit_res)

        if print_fit_results:
            print(fit_res.fit_report())
        return fit_res

    def plot_results(self, fit_res, show_guess=False, art_det=0,
                     fig=None, ax=None, textbox=True):

        self.units = SI_prefix_and_scale_factor(val=max(abs(ax.get_xticks())),
                                                unit=self.sweep_unit[0])[1]  # list

        if isinstance(art_det, list):
            art_det = art_det[0]

        if textbox:
            #TODO: this crashes the analysis when stderr == None
            textstr = ('$f_{qubit \_ old}$ = %.7g GHz'
                       % (self.qubit_freq_spec * 1e-9) +
                       '\n$f_{qubit \_ new}$ = %.7g $\pm$ (%.5g) GHz'
                       % (self.qubit_frequency * 1e-9,
                          fit_res.params['frequency'].stderr * 1e-9) +
                       '\n$\Delta f$ = %.5g $ \pm$ (%.5g) MHz'
                       % ((self.qubit_frequency - self.qubit_freq_spec) * 1e-6,
                          fit_res.params['frequency'].stderr * 1e-6) +
                       '\n$f_{Ramsey}$ = %.5g $ \pm$ (%.5g) MHz'
                       % (fit_res.params['frequency'].value * 1e-6,
                          fit_res.params['frequency'].stderr * 1e-6) +
                       '\n$T_2^\star$ = %.6g '
                       % (fit_res.params['tau'].value * self.scale) +
                       self.units + ' $\pm$ (%.6g) '
                       % (fit_res.params['tau'].stderr * self.scale) +
                       self.units +
                       '\nartificial detuning = %.2g MHz'
                       % (art_det * 1e-6))

            fig.text(0.5, 0, textstr, fontsize=self.font_size,
                     transform=ax.transAxes,
                     verticalalignment='top',
                     horizontalalignment='center', bbox=self.box_props)

        x = np.linspace(self.sweep_points[0],
                        self.sweep_points[-self.NoCalPoints - 1],
                        len(self.sweep_points) * 100)

        if show_guess:
            y_init = fit_mods.ExpDampOscFunc(x, **fit_res.init_values)
            ax.plot(x, y_init, 'k--', linewidth=self.line_width)

        best_vals = fit_res.best_values
        y = fit_mods.ExpDampOscFunc(
            x, tau=best_vals['tau'],
            n=best_vals['n'],
            frequency=best_vals['frequency'],
            phase=best_vals['phase'],
            amplitude=best_vals['amplitude'],
            oscillation_offset=best_vals['oscillation_offset'],
            exponential_offset=best_vals['exponential_offset'])
        ax.plot(x, y, 'r-', linewidth=self.line_width)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, **kw):

        super().run_default_analysis(
            close_file=close_file,
            close_main_figure=True, save_fig=False, **kw)

        verbose = kw.get('verbose', False)
        # Get old values for qubit frequency
        instr_set = self.data_file['Instrument settings']
        try:
            if self.for_ef:
                self.qubit_freq_spec = \
                    float(instr_set[self.qb_name].attrs['f_ef_qubit'])
            elif 'freq_qubit' in kw.keys():
                self.qubit_freq_spec = kw['freq_qubit']
            else:
                try:
                    self.qubit_freq_spec = \
                        float(instr_set[self.qb_name].attrs['f_qubit'])
                except KeyError:
                    self.qubit_freq_spec = \
                        float(instr_set[self.qb_name].attrs['freq_qubit'])

        except (TypeError, KeyError, ValueError):
            logging.warning('qb_name is unknown. Setting previously measured '
                            'value of the qubit frequency to 0. New qubit '
                            'frequency might be incorrect.')
            self.qubit_freq_spec = 0

        self.scale = 1e6

        # artificial detuning with one value can be passed as either an int or
        # a list with one elements
        if (type(self.artificial_detuning) is list) and \
                (len(self.artificial_detuning) > 1):
            if verbose:
                print('Performing Ramsey Analysis for 2 artificial detunings.')
            self.two_art_dets_analysis(**kw)
        else:
            if type(self.artificial_detuning) is list:
                self.artificial_detuning = self.artificial_detuning[0]
            if verbose:
                print('Performing Ramsey Analysis for 1 artificial detuning.')
            self.one_art_det_analysis(**kw)

        self.save_computed_parameters(self.T2_star,
                                      var_name=self.value_names[0])

        # Print the T2_star values on screen
        unit = self.parameter_units[0][-1]
        if kw.pop('print_parameters', False):
            print('New qubit frequency = {:.7f} (GHz)'.format(
                self.qubit_frequency * 1e-9) +
                '\t\tqubit frequency stderr = {:.7f} (MHz)'.format(
                self.ramsey_freq['freq_stderr'] * 1e-6) +
                '\nT2* = {:.5f} '.format(
                self.T2_star['T2_star'] * self.scale) + '(' + 'μ' + unit + ')' +
                '\t\tT2* stderr = {:.5f} '.format(
                self.T2_star['T2_star_stderr'] * self.scale) +
                '(' + 'μ' + unit + ')')

        if close_file:
            self.data_file.close()

        return self.fit_res

    def one_art_det_analysis(self, **kw):

        # Perform fit and save fitted parameters
        self.fit_res = self.fit_Ramsey(x=self.sweep_points[:-self.NoCalPoints],
                                       y=self.normalized_data_points, **kw)
        self.save_fitted_parameters(self.fit_res, var_name=self.value_names[0])
        self.get_measured_freq(fit_res=self.fit_res, **kw)

        # Calculate new qubit frequency
        self.qubit_frequency = self.qubit_freq_spec + self.artificial_detuning \
            - self.ramsey_freq['freq']

        # Extract T2 star and save it
        self.get_measured_T2_star(fit_res=self.fit_res, **kw)
        # the call above defines self.T2_star as a dict; units are seconds

        self.total_detuning = self.fit_res.params['frequency'].value
        self.detuning_stderr = self.fit_res.params['frequency'].stderr
        self.detuning = self.total_detuning - self.artificial_detuning

        if self.make_fig:
            # Plot results
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)
            self.plot_results(self.fit_res, show_guess=show_guess,
                              art_det=self.artificial_detuning,
                              fig=self.fig, ax=self.ax)

            # dispaly figure
            if show:
                plt.show()

            # save figure
            self.save_fig(self.fig, figname=self.measurementstring + '_Ramsey_fit',
                          **kw)

    def two_art_dets_analysis(self, **kw):

        # Extract the data for each ramsey
        len_art_det = len(self.artificial_detuning)
        sweep_pts_1 = self.sweep_points[0:-self.NoCalPoints:len_art_det]
        sweep_pts_2 = self.sweep_points[1:-self.NoCalPoints:len_art_det]
        ramsey_data_1 = self.normalized_values[0:-self.NoCalPoints:len_art_det]
        ramsey_data_2 = self.normalized_values[1:-self.NoCalPoints:len_art_det]

        # Perform fit
        fit_res_1 = self.fit_Ramsey(x=sweep_pts_1,
                                    y=ramsey_data_1, **kw)
        fit_res_2 = self.fit_Ramsey(x=sweep_pts_2,
                                    y=ramsey_data_2, **kw)

        self.save_fitted_parameters(fit_res_1, var_name=(self.value_names[0] +
                                                         ' ' + str(self.artificial_detuning[0] * 1e-6) + ' MHz'))
        self.save_fitted_parameters(fit_res_2, var_name=(self.value_names[0] +
                                                         ' ' + str(self.artificial_detuning[1] * 1e-6) + ' MHz'))

        ramsey_freq_dict_1 = self.get_measured_freq(fit_res=fit_res_1, **kw)
        ramsey_freq_1 = ramsey_freq_dict_1['freq']
        ramsey_freq_dict_2 = self.get_measured_freq(fit_res=fit_res_2, **kw)
        ramsey_freq_2 = ramsey_freq_dict_2['freq']

        # Calculate possible detunings from real qubit frequency
        self.new_qb_freqs = {
            '0': self.qubit_freq_spec + self.artificial_detuning[0] + ramsey_freq_1,
            '1': self.qubit_freq_spec + self.artificial_detuning[0] - ramsey_freq_1,
            '2': self.qubit_freq_spec + self.artificial_detuning[1] + ramsey_freq_2,
            '3': self.qubit_freq_spec + self.artificial_detuning[1] - ramsey_freq_2}

        print('The 4 possible cases for the new qubit frequency give:')
        pprint(self.new_qb_freqs)

        # Find which ones match
        self.diff = {}
        self.diff.update(
            {'0': self.new_qb_freqs['0'] - self.new_qb_freqs['2']})
        self.diff.update(
            {'1': self.new_qb_freqs['1'] - self.new_qb_freqs['3']})
        self.diff.update(
            {'2': self.new_qb_freqs['1'] - self.new_qb_freqs['2']})
        self.diff.update(
            {'3': self.new_qb_freqs['0'] - self.new_qb_freqs['3']})
        self.correct_key = np.argmin(np.abs(list(self.diff.values())))
        # Get new qubit frequency
        self.qubit_frequency = self.new_qb_freqs[str(self.correct_key)]

        if self.correct_key < 2:
            # art_det 1 was correct direction
            # print('Artificial detuning {:.1f} MHz gave the best results.'.format(
            #     self.artificial_detuning[0]*1e-6))
            self.fit_res = fit_res_1
            self.ramsey_data = ramsey_data_1
            self.sweep_pts = sweep_pts_1
            self.good_ramsey_freq = ramsey_freq_1
            qb_stderr = ramsey_freq_dict_1['freq_stderr']

        else:
            # art_det 2 was correct direction
            # print('Artificial detuning {:.1f} MHz gave the best results.'.format(
            #     self.artificial_detuning[1]*1e-6))
            self.fit_res = fit_res_2
            self.ramsey_data = ramsey_data_2
            self.sweep_pts = sweep_pts_2
            self.good_ramsey_freq = ramsey_freq_2
            qb_stderr = ramsey_freq_dict_2['freq_stderr']

        # Extract T2 star and save it
        # defines self.T2_star as a dict;
        self.get_measured_T2_star(fit_res=self.fit_res, **kw)
        # units are seconds

        ################
        # Plot results #
        ################
        if self.make_fig_two_dets:
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)

            if self.for_ef:
                ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
            else:
                ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'
            if self.no_of_columns == 2:
                figsize = (3.375, 2.25 * len_art_det)
            else:
                figsize = (7, 4 * len_art_det)
            self.fig, self.axs = plt.subplots(len_art_det, 1,
                                              figsize=figsize,
                                              dpi=self.dpi)

            fit_res_array = [fit_res_1, fit_res_2]
            ramsey_data_dict = {'0': ramsey_data_1,
                                '1': ramsey_data_2}

            for i in range(len_art_det):
                ax = self.axs[i]
                self.plot_results_vs_sweepparam(x=self.sweep_pts,
                                                y=ramsey_data_dict[str(i)],
                                                fig=self.fig, ax=ax,
                                                xlabel=self.sweep_name,
                                                x_unit=self.sweep_unit[0],
                                                ylabel=ylabel,
                                                marker='o-',
                                                save=False)
                self.plot_results(fit_res_array[i], show_guess=show_guess,
                                  art_det=self.artificial_detuning[i],
                                  fig=self.fig, ax=ax, textbox=False)

                textstr = ('artificial detuning = %.2g MHz'
                           % (self.artificial_detuning[i] * 1e-6) +
                           '\n$f_{Ramsey}$ = %.5g $ MHz \pm$ (%.5g) MHz'
                           % (fit_res_array[i].params['frequency'].value * 1e-6,
                              fit_res_array[i].params['frequency'].stderr * 1e6) +
                           '\n$T_2^\star$ = %.3g '
                           % (fit_res_array[i].params['tau'].value * self.scale) +
                           self.units + ' $\pm$ (%.3g) '
                           % (fit_res_array[i].params['tau'].stderr * self.scale) +
                           self.units)
                ax.annotate(textstr, xy=(0.99, 0.98), xycoords='axes fraction',
                            fontsize=self.font_size, bbox=self.box_props,
                            horizontalalignment='right', verticalalignment='top')

                if i == (len_art_det - 1):
                    textstr_main = ('$f_{qubit \_ old}$ = %.5g GHz'
                                    % (self.qubit_freq_spec * 1e-9) +
                                    '\n$f_{qubit \_ new}$ = %.5g $ GHz \pm$ (%.5g) GHz'
                                    % (self.qubit_frequency * 1e-9,
                                       qb_stderr * 1e-9) +
                                    '\n$T_2^\star$ = %.3g '
                                    % (self.T2_star['T2_star'] * self.scale) +
                                    self.units + ' $\pm$ (%.3g) '
                                    % (self.T2_star['T2_star_stderr'] * self.scale) +
                                    self.units)

                    self.fig.text(0.5, 0, textstr_main, fontsize=self.font_size,
                                  transform=self.axs[i].transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='center', bbox=self.box_props)

            # dispaly figure
            if show:
                plt.show()

            # save figure
            self.save_fig(self.fig, figname=self.measurementstring + '_Ramsey_fit',
                          **kw)

    def get_measured_freq(self, fit_res, **kw):
        freq = fit_res.params['frequency'].value
        freq_stderr = fit_res.params['frequency'].stderr

        self.ramsey_freq = {'freq': freq, 'freq_stderr': freq_stderr}

        return self.ramsey_freq

    def get_measured_T2_star(self, fit_res, **kw):
        '''
        Returns measured T2 star from the fit to the Ical data.
         return T2, T2_stderr
        '''
        T2 = fit_res.params['tau'].value
        T2_stderr = fit_res.params['tau'].stderr

        self.T2_star = {'T2_star': T2, 'T2_star_stderr': T2_stderr}

        return self.T2_star


class DragDetuning_Analysis(TD_Analysis):

    def __init__(self, label='DragDetuning', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):
        close_file = kw.pop('close_file', True)
        figsize = kw.pop('figsize', (11, 10))
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        fig, axarray = plt.subplots(2, 2, figsize=figsize)

        XpY90_data = self.measured_values[0][0::2] + \
            1.j * self.measured_values[1][0::2]
        YpX90_data = self.measured_values[0][1::2] + \
            1.j * self.measured_values[1][1::2]

        self.XpY90 = np.mean(XpY90_data)
        self.YpX90 = np.mean(YpX90_data)
        self.detuning = np.abs(self.XpY90 - self.YpX90)

        for i, name in enumerate(self.value_names):
            ax = axarray[i / 2, i % 2]
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=fig,
                                            ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            **kw)

        self.save_fig(fig, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return (self.detuning, self.XpY90, self.YpX90)


class TransientAnalysis(TD_Analysis):

    def run_default_analysis(self, print_fit_results=False, **kw):
        close_file = kw.pop('close_file', True)
        demodulate = kw.pop('demodulate', False)
        figsize = kw.pop('figsize', (11, 4))
        self.IF = kw.pop('IF', 10)
        self.load_hdf5data()
        keys = list(self.g.keys())
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        self.valuenames = ["transient_0", "transient_1"]
        if 'touch_n_go_transient_0' in keys:
            mode = 'CBox'
            transient_0 = self.get_values(key='touch_n_go_transient_0')
            transient_1 = self.get_values(key='touch_n_go_transient_1')
            sampling_rate = 0.2  # Gsample/s
            kw.pop('plot_title', "CBox transient")
            samples = len(transient_0)

        elif 'average_transients_I' in keys:
            mode = 'ATS'
            transients_0 = self.get_values(key='average_transients_I')
            transients_1 = self.get_values(key='average_transients_Q')
            samples = len(transients_0[:, 0])
            sampling_rate = 1  # Gsample/s

        self.time = np.linspace(0, samples / sampling_rate, samples)
        if mode == 'CBox':
            self.plot_results_vs_sweepparam(x=self.time,
                                            y=transient_0,
                                            fig=fig,
                                            ax=ax,
                                            marker='-o',
                                            xlabel="time (ns)",
                                            ylabel="amplitude (au)",
                                            **kw)
        else:

            ax.plot(self.time, transients_0[:, 0], marker='.',
                    label='Average transient ch A')
            ax.plot(self.time, transients_1[:, 0], marker='.',
                    label='Average transient ch B')
            ax.legend()

            ax.set_xlabel('time (ns)')
            ax.set_ylabel('dac voltage (V)')

        if demodulate:
            print('demodulating using IF = %.2f GHz' % self.IF)
            dem_cos = np.cos(2 * np.pi * self.IF * self.time)
            dem_sin = np.sin(2 * np.pi * self.IF * self.time)

            self.demod_transient_I = dem_cos * transients_0[:, 0] + \
                dem_sin * transients_1[:, 0]
            self.demod_transient_Q = -dem_sin * transients_0[:, 0] + \
                dem_cos * transients_1[:, 0]

            fig2, axs2 = plt.subplots(1, 1, figsize=figsize, sharex=True)
            axs2.plot(self.time, self.demod_transient_I, marker='.',
                      label='I demodulated')
            axs2.plot(self.time, self.demod_transient_Q, marker='.',
                      label='Q demodulated')
            axs2.legend()
            self.save_fig(fig2, figname=self.measurementstring + 'demod', **kw)
            axs2.set_xlabel('time (ns)')
            axs2.set_ylabel('dac voltage (V)')

            self.power = self.demod_transient_I ** 2 + self.demod_transient_Q ** 2
            fig3, ax3 = plt.subplots(1, 1, figsize=figsize, sharex=True)
            ax3.plot(self.time, self.power, marker='.')
            ax3.set_ylabel('Power (a.u.)')
            self.save_fig(fig3, figname=self.measurementstring + 'Power', **kw)
            ax3.set_xlabel('time (ns)')

        self.save_fig(fig, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return


class DriveDetuning_Analysis(TD_Analysis):

    def __init__(self, label='DriveDetuning', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super().__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):

        def sine_fit_data():
            self.fit_type = 'sine'
            model = fit_mods.lmfit.Model(fit_mods.CosFunc)

            params = fit_mods.Cos_guess(model, data=data,
                                        t=sweep_points)
            # This ensures that phase is *always* ~90 deg if it is different
            # this shows up in the amplitude and prevents the correct detuning
            # is shown.
            params['phase'].min = np.deg2rad(80)
            params['phase'].max = np.deg2rad(100)

            fit_results = model.fit(data=data, t=sweep_points,
                                    params=params)
            return fit_results

        def quadratic_fit_data():
            M = np.array(
                [sweep_points ** 2, sweep_points, [1] * len(sweep_points)])
            Minv = np.linalg.pinv(M)
            [a, b, c] = np.dot(data, Minv)
            fit_data = (a * sweep_points ** 2 + b * sweep_points + c)
            return fit_data, (a, b, c)

        close_file = kw.pop('close_file', True)
        figsize = kw.pop('figsize', (11, 5))
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()

        self.NoCalPoints = 4

        self.normalize_data_to_calibration_points(
            self.measured_values[0], self.NoCalPoints)
        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points'.encode('utf-8'))

        data = self.corr_data[:-self.NoCalPoints]
        cal_data = np.split(self.corr_data[-self.NoCalPoints:], 2)
        cal_data_mean = np.mean(cal_data, axis=1)
        cal_peak_to_peak = abs(cal_data_mean[1] - cal_data_mean[0])

        sweep_points = self.sweep_points[:-self.NoCalPoints]
        data_peak_to_peak = max(data) - min(data)

        self.fit_results_sine = sine_fit_data()
        self.fit_results_quadratic = quadratic_fit_data()

        chisqr_sine = self.fit_results_sine.chisqr
        chisqr_quadratic = np.sum((self.fit_results_quadratic[0] - data) ** 2)

        if (chisqr_quadratic < chisqr_sine) or \
                (data_peak_to_peak / cal_peak_to_peak < .5):
            self.fit_type = 'quadratic'
            self.slope = self.fit_results_quadratic[1][1]
            amplitude = cal_peak_to_peak / 2

        else:
            self.fit_type = 'sine'
            amplitude = self.fit_results_sine.params['amplitude']
            frequency = self.fit_results_sine.params['frequency']
            self.slope = 2 * np.pi * amplitude * frequency

        self.drive_detuning = -1 * self.slope / (2 * np.pi * abs(amplitude))
        self.drive_scaling_factor = 1. / (1. + self.drive_detuning)

        # Plotting
        fig, axarray = plt.subplots(2, figsize=figsize)
        for k, name in enumerate(self.value_names):
            ax = axarray[k]
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[k],
                                            fig=fig,
                                            ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[k],
                                            **kw)
        self.save_fig(fig, figname=self.measurementstring, **kw)
        fig, ax = self.default_ax()
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.corr_data,
                                        fig=fig,
                                        ax=ax,
                                        xlabel=self.xlabel,
                                        ylabel=r'$F$  $|1\rangle$',
                                        **kw)
        if self.fit_type is 'sine':
            ax.plot(sweep_points, self.fit_results_sine.best_fit)
        else:
            ax.plot(sweep_points, self.fit_results_quadratic[0])
        # plt.show()
        self.save_fig(fig, figname=self.measurementstring + '_fit', **kw)
        if close_file:
            self.data_file.close()
        return self.drive_scaling_factor


class OnOff_Analysis(TD_Analysis):

    def __init__(self, label='OnOff', idx=None, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        self.idx = idx
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        figsize = kw.pop('figsize', (11, 2 * len(self.value_names)))
        if self.idx is not None:
            idx_val = np.where(self.value_names == 'I_cal_%d' % self.idx)[0][0]
        else:
            try:
                idx_val = np.where(self.value_names == 'I_cal')[0][0]
            except:  # Kind of arbitrarily choose axis 0
                idx_val = 0

        fig, axarray = plt.subplots(len(self.value_names) / 2, 2,
                                    figsize=figsize)

        I_cal = self.measured_values[idx_val]
        zero_mean = np.mean(I_cal[0::2])
        zero_std = np.std(I_cal[0::2])

        one_mean = np.mean(I_cal[1::2])
        one_std = np.std(I_cal[1::2])

        self.distance = np.power(zero_mean - one_mean, 2)
        distance_error = np.sqrt(
            np.power(2. * (zero_mean - one_mean) * zero_std, 2)
            + np.power(2. * (one_mean - zero_mean) * one_std, 2))
        self.contrast = self.distance / distance_error

        for i, name in enumerate(self.value_names):
            if len(self.value_names) == 4:
                ax = axarray[i / 2, i % 2]
            elif len(self.value_names) == 2:
                ax = axarray[i]

            self.plot_results_vs_sweepparam(x=self.sweep_points[::2],
                                            y=self.measured_values[i][::2],
                                            fig=fig,
                                            ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            label='On',
                                            marker='o:',
                                            **kw)
            self.plot_results_vs_sweepparam(x=self.sweep_points[1::2],
                                            y=self.measured_values[i][1::2],
                                            fig=fig,
                                            ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            label='Off',
                                            marker='o:',
                                            **kw)
            ax.legend()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        self.plot_results_vs_sweepparam(x=self.sweep_points[::2],
                                        y=I_cal[::2],
                                        fig=fig2,
                                        ax=ax2,
                                        xlabel=self.xlabel,
                                        ylabel=self.ylabels[idx_val],
                                        label='Off',
                                        marker='o:',
                                        **kw)
        self.plot_results_vs_sweepparam(x=self.sweep_points[1::2],
                                        y=I_cal[1::2],
                                        fig=fig2,
                                        ax=ax2,
                                        xlabel=self.xlabel,
                                        ylabel=self.ylabels[idx_val],
                                        label='Off',
                                        marker='o:',
                                        **kw)
        ax2.hlines((zero_mean), 0, len(self.sweep_points),
                   linestyle='solid', color='blue')
        ax2.hlines((one_mean), 0, len(self.sweep_points),
                   linestyle='solid', color='green')
        ax2.text(2, zero_mean, "Zero mean", bbox=self.box_props, color='blue')
        ax2.text(2, one_mean, "One mean", bbox=self.box_props, color='green')
        ax2.hlines((zero_mean + zero_std, zero_mean - zero_std),
                   0, len(self.sweep_points), linestyle='dashed', color='blue')
        ax2.hlines((one_mean + one_std, one_mean - one_std),
                   0, len(self.sweep_points), linestyle='dashed', color='green')
        ax2.text(2, max(I_cal) + (max(I_cal) - min(I_cal)) * .04,
                 "Contrast: %.2f" % self.contrast,
                 bbox=self.box_props)
        self.save_fig(fig, figname=self.measurementstring, **kw)
        self.save_fig(fig2, figname=self.measurementstring +
                      '_calibrated', **kw)
        if close_file:
            self.data_file.close()
        print('Average contrast: %.2f' % self.contrast)
        return self.contrast


class AllXY_Analysis(TD_Analysis):
    '''
    Performs a rotation and normalization on the data and calculates a
    deviation from the expected ideal data.

    Automatically works for the standard AllXY sequences of 42 and 21 points.
    Optional keyword arguments can be used to specify
    'ideal_data': np.array equal in lenght to the data
    '''

    def __init__(self, label='AllXY', zero_coord=None, one_coord=None,
                 make_fig=True, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        self.zero_coord = zero_coord
        self.one_coord = one_coord
        self.make_fig = make_fig

        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_main_fig=True, flip_axis=False, **kw):
        close_file = kw.pop('close_file', True)
        self.flip_axis = flip_axis
        self.cal_points = kw.pop('cal_points', None)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()

        if len(self.measured_values[0]) == 42:
            ideal_data = np.concatenate((0 * np.ones(10), 0.5 * np.ones(24),
                                         np.ones(8)))
        else:
            ideal_data = np.concatenate((0 * np.ones(5), 0.5 * np.ones(12),
                                         np.ones(4)))
        self.rotate_and_normalize_data()
        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points'.encode('utf-8'))
        data_error = self.corr_data - ideal_data
        self.deviation_total = np.mean(abs(data_error))
        # Plotting
        if self.make_fig:
            self.make_figures(ideal_data=ideal_data,
                              close_main_fig=close_main_fig, **kw)
        if close_file:
            self.data_file.close()
        return self.deviation_total

    def make_figures(self, ideal_data, close_main_fig, **kw):
        fig1, fig2, ax1, axarray = self.setup_figures_and_axes()
        for i in range(len(self.value_names)):
            if len(self.value_names) == 2:
                ax = axarray[i]
            else:
                ax = axarray
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=fig2, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=str(
                                                self.value_names[i]),
                                            save=False, label="Measurement")
        ax1.set_ylim(min(self.corr_data) - .1, max(self.corr_data) + .1)
        if self.flip_axis:
            ylabel = r'$F$ $|0 \rangle$'
        else:
            ylabel = r'$F$ $|1 \rangle$'
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.corr_data,
                                        fig=fig1, ax=ax1,
                                        xlabel='',
                                        ylabel=ylabel,
                                        save=False, label="Measurement")
        ax1.plot(self.sweep_points, ideal_data, label="Ideal")
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        if len(self.measured_values[0]) == 42:
            locs = np.arange(1, 42, 2)
        else:
            locs = np.arange(0, 21, 1)
        labels = ['II', 'XX', 'YY', 'XY', 'YX',
                  'xI', 'yI', 'xy', 'yx', 'xY', 'yX',
                  'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy',
                  'XI', 'YI', 'xx', 'yy']

        ax1.xaxis.set_ticks(locs)
        ax1.set_xticklabels(labels, rotation=60)

        if kw.pop("plot_deviation", True):
            deviation_text = r'Deviation: %.5f' % self.deviation_total
            ax1.text(1, 1.05, deviation_text, fontsize=11,
                     bbox=self.box_props)
        legend_loc = "lower right"
        if len(self.value_names) > 1:
            [ax.legend(loc=legend_loc) for ax in axarray]
        else:
            axarray.legend(loc=legend_loc)

        ax1.legend(loc=legend_loc)

        if not close_main_fig:
            # Hacked in here, good idea to only show the main fig but can
            # be optimized somehow
            self.save_fig(fig1, ylabel='Amplitude (normalized)',
                          close_fig=False, **kw)
        else:
            self.save_fig(fig1, ylabel='Amplitude (normalized)', **kw)
        self.save_fig(fig2, ylabel='Amplitude', **kw)


class FFC_Analysis(TD_Analysis):
    '''
    Performs a rotation and normalization on the data and calculates a
    deviation from the expected ideal data.

    Automatically works for the standard AllXY sequences of 42 and 21 points.
    Optional keyword arguments can be used to specify
    'ideal_data': np.array equal in lenght to the data
    '''

    def __init__(self, label='FFC', make_fig=True, zero_coord=None, one_coord=None, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        self.zero_coord = zero_coord
        self.one_coord = one_coord
        self.make_fig = make_fig

        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_main_fig=True, flip_axis=False, **kw):
        close_file = kw.pop('close_file', True)
        self.flip_axis = flip_axis
        self.cal_points = kw.pop('cal_points', None)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()

        ideal_data = np.concatenate((0.5 * np.ones(1), 1 * np.ones(1)))
        self.rotate_and_normalize_data()
        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points'.encode('utf-8'))
        data_error = self.corr_data - ideal_data
        self.deviation_total = np.mean(abs(data_error))
        # Plotting
        if self.make_fig:
            self.make_figures(ideal_data=ideal_data,
                              close_main_fig=close_main_fig, **kw)
        if close_file:
            self.data_file.close()
        return self.deviation_total

    def make_figures(self, ideal_data, close_main_fig, **kw):
        fig1, fig2, ax1, axarray = self.setup_figures_and_axes()
        for i in range(len(self.value_names)):
            if len(self.value_names) == 2:
                ax = axarray[i]
            else:
                ax = axarray
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=fig2, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=str(
                                                self.value_names[i]),
                                            save=False)
        ax1.set_ylim(min(self.corr_data) - .1, max(self.corr_data) + .1)
        if self.flip_axis:
            ylabel = r'$F$ $|0 \rangle$'
        else:
            ylabel = r'$F$ $|1 \rangle$'
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.corr_data,
                                        fig=fig1, ax=ax1,
                                        xlabel='',
                                        ylabel=ylabel,
                                        save=False)
        ax1.plot(self.sweep_points, ideal_data)
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        locs = np.arange(0, 2)
        labels = ['NoFB', 'FB']

        ax1.xaxis.set_ticks(locs)
        ax1.set_xticklabels(labels, rotation=60)

        deviation_text = r'Deviation: %.5f' % self.deviation_total
        ax1.text(1, 1.05, deviation_text, fontsize=11,
                 bbox=self.box_props)
        if not close_main_fig:
            # Hacked in here, good idea to only show the main fig but can
            # be optimized somehow
            self.save_fig(fig1, ylabel='Amplitude (normalized)',
                          close_fig=False, **kw)
        else:
            self.save_fig(fig1, ylabel='Amplitude (normalized)', **kw)
        self.save_fig(fig2, ylabel='Amplitude', **kw)


class RandomizedBenchmarking_Analysis(TD_Analysis):
    '''
    Rotates and normalizes the data before doing a fit with a decaying
    exponential to extract the Clifford fidelity.
    By optionally specifying T1 and the pulse separation (time between start
    of pulses) the T1 limited fidelity will be given and plotted in the
    same figure.
    '''

    def __init__(self, label='RB', T1=None, pulse_delay=None, **kw):
        self.T1 = T1
        self.pulse_delay = pulse_delay

        super().__init__(**kw)

    def run_default_analysis(self, **kw):
        close_main_fig = kw.pop('close_main_fig', True)
        close_file = kw.pop('close_file', True)
        if self.cal_points is None:
            self.cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        super().run_default_analysis(close_file=False, make_fig=False,
                                     **kw)

        data = self.corr_data[:-1 * (len(self.cal_points[0] * 2))]
        n_cl = self.sweep_points[:-1 * (len(self.cal_points[0] * 2))]

        self.fit_res = self.fit_data(data, n_cl)
        self.fit_results = [self.fit_res]
        self.save_fitted_parameters(fit_res=self.fit_res, var_name='F|1>')
        if self.make_fig:
            self.make_figures(close_main_fig=close_main_fig, **kw)

        if close_file:
            self.data_file.close()
        return

    def calc_T1_limited_fidelity(self, T1, pulse_delay):
        '''
        Formula from Asaad et al.
        pulse separation is time between start of pulses
        '''
        Np = 1.875  # Number of gates per Clifford
        F_cl = (1 / 6 * (3 + 2 * np.exp(-1 * pulse_delay / (2 * T1)) +
                         np.exp(-pulse_delay / T1))) ** Np
        p = 2 * F_cl - 1

        return F_cl, p

    def add_textbox(self, ax, F_T1=None):

        textstr = ('\t$F_{Cl}$' + ' \t= {:.4g} $\pm$ ({:.4g})%'.format(
            self.fit_res.params['fidelity_per_Clifford'].value * 100,
            self.fit_res.params['fidelity_per_Clifford'].stderr * 100) +
            '\n  $1-F_{Cl}$' + '  = {:.4g} $\pm$ ({:.4g})%'.format(
            (1 - self.fit_res.params['fidelity_per_Clifford'].value) * 100,
            (self.fit_res.params['fidelity_per_Clifford'].stderr) * 100) +
            '\n\tOffset\t= {:.4g} $\pm$ ({:.4g})'.format(
            (self.fit_res.params['offset'].value),
            (self.fit_res.params['offset'].stderr)))
        if F_T1 is not None:
            textstr += ('\n\t  $F_{Cl}^{T_1}$  = ' +
                        '{:.6g}%'.format(F_T1 * 100))

        self.ax.text(0.1, 0.95, textstr, transform=self.ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=self.box_props)

    def make_figures(self, close_main_fig, **kw):

        ylabel = r'$F$ $\left(|1 \rangle \right)$'
        self.fig, self.ax = self.default_ax()
        if self.plot_cal_points:
            x = self.sweep_points
            y = self.corr_data
        else:
            logging.warning('not tested for all types of calpoints')
            if type(self.cal_points[0]) is int:
                x = self.sweep_points[:-2]
                y = self.corr_data[:-2]
            else:
                x = self.sweep_points[:-1 * (len(self.cal_points[0]) * 2)]
                y = self.corr_data[:-1 * (len(self.cal_points[0]) * 2)]

        self.plot_results_vs_sweepparam(x=x,
                                        y=y,
                                        fig=self.fig, ax=self.ax,
                                        xlabel=self.xlabel,
                                        ylabel=ylabel,
                                        save=False)

        x_fine = np.linspace(0, self.sweep_points[-1], 1000)
        for fit_res in self.fit_results:
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **fit_res.best_values)
            self.ax.plot(x_fine, best_fit, label='Fit')
        self.ax.set_ylim(min(min(self.corr_data) - .1, -.1),
                         max(max(self.corr_data) + .1, 1.1))

        # Here we add the line corresponding to T1 limited fidelity
        F_T1 = None
        if self.T1 is not None and self.pulse_delay is not None:
            F_T1, p_T1 = self.calc_T1_limited_fidelity(
                self.T1, self.pulse_delay)
            T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, -0.5, p_T1, 0.5)
            self.ax.plot(x_fine, T1_limited_curve, label='T1-limit')

            self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Add a textbox
        self.add_textbox(self.ax, F_T1)

        if not close_main_fig:
            # Hacked in here, good idea to only show the main fig but can
            # be optimized somehow
            self.save_fig(self.fig, ylabel='Amplitude (normalized)',
                          close_fig=False, **kw)
        else:
            self.save_fig(self.fig, ylabel='Amplitude (normalized)', **kw)

    def fit_data(self, data, numCliff,
                 print_fit_results=False,
                 show_guess=False,
                 plot_results=False):

        RBModel = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
        # RBModel = fit_mods.RBModel
        RBModel.set_param_hint('Amplitude', value=-0.5)
        RBModel.set_param_hint('p', value=.99)
        RBModel.set_param_hint('offset', value=.5)
        RBModel.set_param_hint('fidelity_per_Clifford',  # vary=False,
                               expr='(p + (1-p)/2)')
        RBModel.set_param_hint('error_per_Clifford',  # vary=False,
                               expr='1-fidelity_per_Clifford')
        RBModel.set_param_hint('fidelity_per_gate',  # vary=False,
                               expr='fidelity_per_Clifford**(1./1.875)')
        RBModel.set_param_hint('error_per_gate',  # vary=False,
                               expr='1-fidelity_per_gate')

        params = RBModel.make_params()
        fit_res = RBModel.fit(data, numCliff=numCliff,
                              params=params)
        if print_fit_results:
            print(fit_res.fit_report())
        if plot_results:
            plt.plot(fit_res.data, 'o-', label='data')
            plt.plot(fit_res.best_fit, label='best fit')
            if show_guess:
                plt.plot(fit_res.init_fit, '--', label='init fit')

        return fit_res


class RB_double_curve_Analysis(RandomizedBenchmarking_Analysis):

    def run_default_analysis(self, **kw):
        close_main_fig = kw.pop('close_main_fig', True)
        close_file = kw.pop('close_file', True)
        if self.cal_points is None:
            self.cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        super(RandomizedBenchmarking_Analysis, self).run_default_analysis(
            close_file=False, make_fig=False, **kw)

        data = self.corr_data[:-1 * (len(self.cal_points[0] * 2))]
        # 1- minus all populations because we measure fidelity to 1
        data_0 = 1 - data[::2]
        data_1 = 1 - data[1::2]
        # 2-state population is just whatever is missing in 0 and 1 state
        # assumes that 2 looks like 1 state
        data_2 = 1 - (data_1) - (data_0)
        n_cl = self.sweep_points[:-1 * (len(self.cal_points[0] * 2)):2]

        self.fit_results = self.fit_data(data_0, data_1, n_cl)

        self.save_fitted_parameters(fit_res=self.fit_results,
                                    var_name='Double_curve_RB')

        if self.make_fig:
            self.make_figures(n_cl, data_0, data_1, data_2,
                              close_main_fig=close_main_fig, **kw)
        if close_file:
            self.data_file.close()
        return

    def fit_data(self, data0, data1, numCliff,
                 print_fit_results=False,
                 show_guess=False,
                 plot_results=False):
        data = np.concatenate([data0, data1])
        numCliff = 2 * list(numCliff)
        invert = np.concatenate([np.ones(len(data0)),
                                 np.zeros(len(data1))])

        RBModel = lmfit.Model(fit_mods.double_RandomizedBenchmarkingDecay,
                              independent_vars=['numCliff', 'invert'])
        RBModel.set_param_hint('p', value=.99)
        RBModel.set_param_hint('offset', value=.5)
        RBModel.set_param_hint('fidelity_per_Clifford',  # vary=False,
                               expr='(p + (1-p)/2)')
        RBModel.set_param_hint('error_per_Clifford',  # vary=False,
                               expr='1-fidelity_per_Clifford')
        RBModel.set_param_hint('fidelity_per_gate',  # vary=False,
                               expr='fidelity_per_Clifford**(1./1.875)')
        RBModel.set_param_hint('error_per_gate',  # vary=False,
                               expr='1-fidelity_per_gate')

        params = RBModel.make_params()
        fit_res = RBModel.fit(data, numCliff=numCliff, invert=invert,
                              params=params)
        if print_fit_results:
            print(fit_res.fit_report())
        return fit_res

    def add_textbox(self, f, ax, F_T1=None):
        fr0 = self.fit_results.params
        textstr = (
            '$F_{\mathrm{Cl}}$' + '= {:.5g} \n\t$\pm$ ({:.2g})%'.format(
                fr0['fidelity_per_Clifford'].value * 100,
                fr0['fidelity_per_Clifford'].stderr * 100) +
            '\nOffset ' + '= {:.4g} \n\t$\pm$ ({:.2g})%'.format(
                fr0['offset'].value * 100, fr0['offset'].stderr * 100))
        if F_T1 is not None:
            textstr += ('\n\t  $F_{Cl}^{T_1}$  = ' +
                        '{:.5g}%'.format(F_T1 * 100))
        ax.text(0.95, 0.1, textstr, transform=f.transFigure,
                fontsize=11, verticalalignment='bottom',
                horizontalalignment='right')

    def make_figures(self, n_cl, data_0, data_1, data_2,
                     close_main_fig, **kw):
        f, ax = plt.subplots()
        ax.plot(n_cl, data_0, 'o', color='b', label=r'$|0\rangle$')
        ax.plot(n_cl, data_1, '^', color='r', label=r'$|1\rangle$')
        ax.plot(n_cl, data_2, 'p', color='g', label=r'$|2\rangle$')
        ax.hlines(0, n_cl[0], n_cl[-1] * 1.05, linestyle='--')
        ax.hlines(1, n_cl[0], n_cl[-1] * 1.05, linestyle='--')
        ax.plot([n_cl[-1]] * 4, self.corr_data[-4:], 'o', color='None')
        ax.set_xlabel('Number of Cliffords')
        ax.set_ylabel('State populations')
        plot_title = kw.pop('plot_title', textwrap.fill(
            self.timestamp_string + '_' +
            self.measurementstring, 40))
        ax.set_title(plot_title)
        ax.set_xlim(n_cl[0], n_cl[-1] * 1.02)
        ax.set_ylim(-.1, 1.1)
        x_fine = np.linspace(0, self.sweep_points[-1] * 1.05, 1000)
        fit_0 = fit_mods.double_RandomizedBenchmarkingDecay(
            x_fine, invert=1, **self.fit_results.best_values)
        fit_1 = fit_mods.double_RandomizedBenchmarkingDecay(
            x_fine, invert=0, **self.fit_results.best_values)
        fit_2 = 1 - fit_1 - fit_0

        ax.plot(x_fine, fit_0, color='darkgray', label='fit')
        ax.plot(x_fine, fit_1, color='darkgray')
        ax.plot(x_fine, fit_2, color='darkgray')

        F_T1 = None
        if self.T1 is not None and self.pulse_delay is not None:
            F_T1, p_T1 = self.calc_T1_limited_fidelity(
                self.T1, self.pulse_delay)
            T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, -0.5, p_T1, 0.5)
            ax.plot(x_fine, T1_limited_curve,
                    linestyle='--', color='lightgray', label='T1-limit')
            T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, 0.5, p_T1, 0.5)
            ax.plot(x_fine, T1_limited_curve,
                    linestyle='--', color='lightgray')
        self.add_textbox(f, ax, F_T1)
        ax.legend(frameon=False, numpoints=1,
                  # bbox_transform=ax.transAxes,#
                  bbox_transform=f.transFigure,
                  loc='upper right',
                  bbox_to_anchor=(.95, .95))
        ax.set_xscale("log", nonposx='clip')
        plt.subplots_adjust(left=.1, bottom=None, right=.7, top=None)
        self.save_fig(f, figname='Two_curve_RB', close_fig=close_main_fig,
                      fig_tight=False, **kw)


class RandomizedBench_2D_flat_Analysis(RandomizedBenchmarking_Analysis):
    '''
    Analysis for the specific RB sequenes used in the CBox that require
    doing a 2D scan in order to get enough seeds in (due to the limit of the
    max number of pulses).
    '''

    def get_naming_and_values(self):
        '''
        Extracts the data as if it is 2D then takes the mean and stores it as
        if it it is just a simple line scan.
        '''
        self.get_naming_and_values_2D()
        self.measured_values = np.array([np.mean(self.Z[0][:], axis=0),
                                         np.mean(self.Z[1][:], axis=0)])


#######################################################
# End of time domain analyses
#######################################################


class Homodyne_Analysis(MeasurementAnalysis):

    def __init__(self, label='HM', custom_power_message: dict = None, **kw):
        # Custome power message is used to create a message in resonator measurements
        # dict must be custom_power_message={'Power': -15, 'Atten': 86, 'res_len':3e-6}
        # Power in dBm, Atten in dB and resonator length in m
        kw['label'] = label
        kw['h5mode'] = 'r+'
        kw['custom_power_message'] = custom_power_message
        super().__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, fitting_model='hanger',
                             show_guess=False, show=False,
                             fit_window=None, **kw):
        '''
        Available fitting_models:
            - 'hanger' = amplitude fit with slope
            - 'complex' = complex transmission fit WITHOUT slope
            - 'lorentzian' = fit to a Lorentzian lineshape

        'fit_window': allows to select the windows of data to fit.
                      Example: fit_window=[100,-100]
        '''
        super(self.__class__, self).run_default_analysis(
            close_file=False, show=show, **kw)
        self.add_analysis_datagroup_to_file()

        window_len_filter = kw.get('window_len_filter', 11)

        ########## Fit data ##########

        # Fit Power to a Lorentzian
        self.measured_powers = self.measured_values[0] ** 2

        min_index = np.argmin(self.measured_powers)
        max_index = np.argmax(self.measured_powers)

        self.min_frequency = self.sweep_points[min_index]
        self.max_frequency = self.sweep_points[max_index]

        measured_powers_smooth = a_tools.smooth(self.measured_powers,
                                                window_len=window_len_filter)
        self.peaks = a_tools.peak_finder((self.sweep_points),
                                         measured_powers_smooth,
                                         window_len=0)

        # Search for peak
        if self.peaks['dip'] is not None:  # look for dips first
            f0 = self.peaks['dip']
            amplitude_factor = -1.
        elif self.peaks['peak'] is not None:  # then look for peaks
            f0 = self.peaks['peak']
            amplitude_factor = 1.
        else:  # Otherwise take center of range
            f0 = np.median(self.sweep_points)
            amplitude_factor = -1.
            logging.warning('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object
            # N.B. This not updating is not implemented as of 9/2017

        # Fit data according to the model required
        if 'hanger' in fitting_model:
            if fitting_model == 'hanger':
                # f is expected in Hz but f0 in GHz!
                Model = fit_mods.SlopedHangerAmplitudeModel
            # this in not working at the moment (need to be fixed)
            elif fitting_model == 'simple_hanger':
                Model = fit_mods.HangerAmplitudeModel
            else:
                raise ValueError(
                    'The fitting model specified is not available')
            # added reject outliers to be robust agains CBox data acq bug.
            # this should have no effect on regular data acquisition and is
            # only used in the guess.
            amplitude_guess = max(
                dm_tools.reject_outliers(self.measured_values[0]))

            # Creating parameters and estimations
            S21min = (min(dm_tools.reject_outliers(self.measured_values[0])) /
                      max(dm_tools.reject_outliers(self.measured_values[0])))

            Q = kw.pop('Q', f0 / abs(self.min_frequency - self.max_frequency))
            Qe = abs(Q / abs(1 - S21min))

            # Note: input to the fit function is in GHz for convenience
            Model.set_param_hint('f0', value=f0 * 1e-9,
                                 min=min(self.sweep_points) * 1e-9,
                                 max=max(self.sweep_points) * 1e-9)
            Model.set_param_hint('A', value=amplitude_guess)
            Model.set_param_hint('Q', value=Q, min=1, max=50e6)
            Model.set_param_hint('Qe', value=Qe, min=1, max=50e6)
            # NB! Expressions are broken in lmfit for python 3.5 this has
            # been fixed in the lmfit repository but is not yet released
            # the newest upgrade to lmfit should fix this (MAR 18-2-2016)
            Model.set_param_hint('Qi', expr='abs(1./(1./Q-1./Qe*cos(theta)))',
                                 vary=False)
            Model.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
            Model.set_param_hint('theta', value=0, min=-np.pi / 2,
                                 max=np.pi / 2)
            Model.set_param_hint('slope', value=0, vary=True)

            self.params = Model.make_params()

            if fit_window == None:
                data_x = self.sweep_points
                self.data_y = self.measured_values[0]
            else:
                data_x = self.sweep_points[fit_window[0]:fit_window[1]]
                data_y_temp = self.measured_values[0]
                self.data_y = data_y_temp[fit_window[0]:fit_window[1]]

            # # make sure that frequencies are in Hz
            # if np.floor(data_x[0]/1e8) == 0:  # frequency is defined in GHz
            #     data_x = data_x*1e9

            fit_res = Model.fit(data=self.data_y,
                                f=data_x, verbose=False)

        elif fitting_model == 'complex':
            # Implement slope fitting with Complex!! Xavi February 2018
            # this is the fit with a complex transmission curve WITHOUT slope
            data_amp = self.measured_values[0]
            data_angle = self.measured_values[1]
            data_complex = data_amp * \
                np.cos(data_angle) + 1j * data_amp * np.sin(data_angle)
            # np.add(self.measured_values[2], 1j*self.measured_values[3])

            # Initial guesses
            guess_A = max(data_amp)
            # this has to been improved
            guess_Q = f0 / abs(self.min_frequency - self.max_frequency)
            guess_Qe = guess_Q / (1 - (max(data_amp) - min(data_amp)))
            # phi_v
            # number of 2*pi phase jumps
            nbr_phase_jumps = (np.diff(data_angle) > 4).sum()
            guess_phi_v = (2 * np.pi * nbr_phase_jumps + (data_angle[0] - data_angle[-1])) / (
                self.sweep_points[0] - self.sweep_points[-1])
            # phi_0
            angle_resonance = data_angle[int(len(self.sweep_points) / 2)]
            phase_evolution_resonance = np.exp(1j * guess_phi_v * f0)
            angle_phase_evolution = np.arctan2(
                np.imag(phase_evolution_resonance), np.real(phase_evolution_resonance))
            guess_phi_0 = angle_resonance - angle_phase_evolution

            # prepare the parameter dictionary
            P = lmfit.Parameters()
            #           (Name,         Value, Vary,      Min,     Max,  Expr)
            P.add_many(('f0', f0 / 1e9, True, None, None, None),
                       ('Q', guess_Q, True, 1, 50e6, None),
                       ('Qe', guess_Qe, True, 1, 50e6, None),
                       ('A', guess_A, True, 0, None, None),
                       ('theta', 0, True, -np.pi / 2, np.pi / 2, None),
                       ('phi_v', guess_phi_v, True, None, None, None),
                       ('phi_0', guess_phi_0, True, -np.pi, np.pi, None))
            P.add('Qi', expr='1./(1./Q-1./Qe*cos(theta))', vary=False)
            P.add('Qc', expr='Qe/cos(theta)', vary=False)

            # Fit
            fit_res = lmfit.minimize(fit_mods.residual_complex_fcn, P,
                                     args=(fit_mods.HangerFuncComplex, self.sweep_points, data_complex))

        elif fitting_model == 'lorentzian':
            Model = fit_mods.LorentzianModel

            kappa_guess = 2.5e6

            amplitude_guess = amplitude_factor * np.pi * kappa_guess * abs(
                max(self.measured_powers) - min(self.measured_powers))

            Model.set_param_hint('f0', value=f0,
                                 min=min(self.sweep_points),
                                 max=max(self.sweep_points))
            Model.set_param_hint('A', value=amplitude_guess)

            # Fitting
            Model.set_param_hint('offset',
                                 value=np.mean(self.measured_powers),
                                 vary=True)
            Model.set_param_hint('kappa',
                                 value=kappa_guess,
                                 min=0,
                                 vary=True)
            Model.set_param_hint('Q',
                                 expr='0.5*f0/kappa',
                                 vary=False)
            self.params = Model.make_params()

            fit_res = Model.fit(data=self.measured_powers,
                                f=self.sweep_points,
                                params=self.params)

        else:
            raise ValueError('fitting model "{}" not recognized'.format(
                fitting_model))

        self.fit_results = fit_res
        self.save_fitted_parameters(fit_res, var_name='HM')

        if print_fit_results is True:
            # print(fit_res.fit_report())
            print(lmfit.fit_report(fit_res))

        ########## Plot results ##########
        xlabel = kw.get('xlabel', self.xlabel)
        ylabel = kw.get('ylabel', 'S21_Mag')

        fig, ax = self.default_ax()


        if 'hanger' in fitting_model:
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[0],
                                            fig=fig, ax=ax,
                                            xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=ylabel,
                                            y_unit=self.value_units[0],
                                            save=False,
                                            **kw)
            # ensures that amplitude plot starts at zero
            ax.set_ylim(ymin=0.000)

        elif 'complex' in fitting_model:
            self.plot_complex_results(
                data_complex, fig=fig, ax=ax, show=False, save=False)
            # second figure with amplitude
            fig2, ax2 = self.default_ax()
            self.plot_results_vs_sweepparam(x=self.sweep_points, y=data_amp,
                                            fig=fig2, ax=ax2,
                                            show=False, xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=str('S21_mag'),
                                            y_unit=self.value_units[0])

        elif fitting_model == 'lorentzian':
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_powers,
                                            fig=fig, ax=ax,
                                            xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=str('Power (arb. units)'),
                                            save=False)

        scale = SI_prefix_and_scale_factor(val=max(abs(ax.get_xticks())),
                                           unit=self.sweep_unit[0])[0]

        instr_set = self.data_file['Instrument settings']
        try:
            old_RO_freq = float(instr_set[self.qb_name].attrs['f_RO'])
            old_vals = '\n$f_{\mathrm{old}}$ = %.5f GHz' % (
                old_RO_freq * scale)
        except (TypeError, KeyError, ValueError):
            logging.warning('qb_name is None. Old parameter values will '
                            'not be retrieved.')
            old_vals = ''

        if ('hanger' in fitting_model) or ('complex' in fitting_model):
            if kw['custom_power_message'] is None:
                textstr = format_value_string(
                    '$f_{{\mathrm{{center}}}}$', fit_res.params['f0'],
                    end_char='\n', unit=self.sweep_unit[0])
                textstr += format_value_string(
                    '$Qc$', fit_res.params['Qc'], end_char='\n')
                textstr += format_value_string(
                    '$Qi$', fit_res.params['Qi'], end_char='\n')
                textstr += old_vals
            else:
                ###############################################################################
                # Custom must be a dictionary                                                #
                # custom_power = {'Power':-15, 'Atten':30, 'res_len':3.6e-6}                   #
                # Power is power at source in dBm                                             #
                # Atten is attenuation at sample, including sources attenuation in dB         #
                # res_len is the lenght of the resonator in m                                 #
                # All of this is needed to calculate mean photon number and phase velocity    #
                ###############################################################################

                custom_power = kw['custom_power_message']
                power_in_w = 10 ** ((custom_power['Power'] -
                                     custom_power['Atten']) / 10) * 1e-3
                mean_ph = (2 * (fit_res.params['Q'].value ** 2) / (fit_res.params['Qc'].value * hbar * (
                    2 * pi * fit_res.params['f0'].value * 1e9) ** 2)) * power_in_w
                phase_vel = 4 * custom_power['res_len'] * \
                    fit_res.params['f0'].value * 1e9

                textstr = format_value_string(
                    '$f_{{\mathrm{{center}}}}$', fit_res.params['f0'],
                    end_char='\n', unit=self.sweep_unit[0])
                textstr += format_value_string(
                    '$Qc$', fit_res.params['Qc'], end_char='\n')
                textstr += format_value_string(
                    '$Qi$', fit_res.params['Qi'], end_char='\n')
                textstr += old_vals+'\n'\
                    '$< n_{\mathrm{ph} }>$ = %.1f' % (mean_ph) + '\n' \
                    '$v_{\mathrm{phase}}$ = %.3e m/s' % (phase_vel)

        elif fitting_model == 'lorentzian':
            textstr = format_value_string(
                    '$f_{{\mathrm{{center}}}}$', fit_res.params['f0'],
                    end_char='\n', unit=self.sweep_unit[0])
            textstr += format_value_string(
                    '$Q$', fit_res.params['Q'], end_char='\n')
            textstr+=old_vals


        fig.text(0.5, 0, textstr, transform=ax.transAxes,
                 fontsize=self.font_size,
                 verticalalignment='top',
                 horizontalalignment='center', bbox=self.box_props)

        if 'complex' in fitting_model:
            fig2.text(0.5, 0, textstr, transform=ax.transAxes,
                      fontsize=self.font_size,
                      verticalalignment='top', horizontalalignment='center',
                      bbox=self.box_props)

        if fit_window == None:
            data_x = self.sweep_points
        else:
            data_x = self.sweep_points[fit_window[0]:fit_window[1]]

        if show_guess:
            ax.plot(self.sweep_points, fit_res.init_fit, 'k--',
                    linewidth=self.line_width)

        # this part is necessary to separate fit perfomed with lmfit.minimize
        if 'complex' in fitting_model:
            fit_values = fit_mods.HangerFuncComplex(
                self.sweep_points, fit_res.params)
            ax.plot(np.real(fit_values), np.imag(fit_values), 'r-')

            ax2.plot(self.sweep_points, np.abs(fit_values), 'r-')

            # save both figures
            self.save_fig(fig, figname='complex', **kw)
            self.save_fig(fig2, xlabel='Mag', **kw)
        else:
            ax.plot(self.sweep_points, fit_res.best_fit, 'r-',
                    linewidth=self.line_width)

            f0 = fit_res.params['f0'].value
            if 'hanger' in fitting_model:
                # f is expected in Hz but f0 in GHz!
                ax.plot(f0 * 1e9, Model.func(f=f0 * 1e9, **fit_res.best_values), 'o',
                        ms=self.marker_size_special)
            else:
                ax.plot(f0, Model.func(f=f0, **fit_res.best_values), 'o',
                        ms=self.marker_size_special)

            if show:
                plt.show()

            # save figure
            self.save_fig(fig, xlabel=self.xlabel, ylabel=ylabel, **kw)

        # self.save_fig(fig, xlabel=self.xlabel, ylabel='Mag', **kw)
        if close_file:
            self.data_file.close()
        return fit_res


class Homodyne_Analysis_Mutipeak(MeasurementAnalysis):

    def __init__(self, label='', dip=False, **kw):
        # Custome power message is used to create a message in resonator measurements
        # dict must be custom_power_message={'Power': -15, 'Atten': 86, 'res_len':3e-6}
        # Power in dBm, Atten in dB and resonator length in m
        kw['label'] = label
        kw['h5mode'] = 'r+'
        self.dip = dip
        super().__init__(**kw)

    def run_default_analysis(self,
                             close_file=False,
                             show=False, **kw):
        '''

        '''
        super(self.__class__, self).run_default_analysis(
            close_file=False, show=show, **kw)
        self.add_analysis_datagroup_to_file()

        window_len_filter = kw.get('window_len_filter', 11)

        data_x = self.sweep_points
        data_y = self.measured_values[0]
        if self.dip:
            data_y_find = -data_y
        else:
            data_y_find = data_y

        self.peaks = a_tools.peak_finder_v2(data_x, data_y_find,
                                            window_len=window_len_filter, perc=92)
        self.peak_indices = np.array([list(data_x).index(p) for p in self.peaks])
        self.peak_vals = data_y[self.peak_indices]
        self.peak_amps = np.abs(self.peak_vals-np.mean(data_y))

        fig, ax = self.default_ax()
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.measured_values[0],
                                        fig=fig, ax=ax,
                                        xlabel=self.sweep_name,
                                        x_unit=self.sweep_unit[0],
                                        ylabel=str('S21_mag'),
                                        y_unit=self.value_units[0],
                                        save=False)
        # ensures that amplitude plot starts at zero
        ax.set_ylim(ymin=0.000)

        ax.plot(self.peaks, self.peak_vals, 'o',
                        ms=self.marker_size_special)

        textstr = 'Peak positions and heights'
        for pos, amp in zip(self.peaks, self.peak_amps):
            textstr += '\n $f=${:.4g} GHz; $A=${:.2g}'.format(pos/1e9,amp)

        fig.text(1, 0.5, textstr, transform=ax.transAxes,
                 fontsize=self.font_size,
                 verticalalignment='center',
                 horizontalalignment='left', bbox=self.box_props)

        self.save_fig(fig, xlabel=self.xlabel, ylabel='peaks', **kw)

        return self.peaks


################
# VNA analysis #
################
class VNA_Analysis(MeasurementAnalysis):
    '''
    Nice to use with all measurements performed with the VNA.
    '''

    def __init__(self, label='VNA', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self, **kw):
        super(self.__class__, self).run_default_analysis(
            close_file=False, **kw)

        # prepare figure in log scale
        data_amp = self.measured_values[0]
        print(data_amp)

        fig, ax = self.default_ax()
        self.plot_dB_from_linear(x=self.sweep_points,
                                 lin_amp=data_amp,
                                 fig=fig, ax=ax,
                                 save=False)

        self.save_fig(fig, figname='dB_plot', **kw)


class Acquisition_Delay_Analysis(MeasurementAnalysis):

    def __init__(self, label='AD', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, window_len=11,
                             print_results=False, close_file=False, **kw):
        super(self.__class__, self).run_default_analysis(
            close_file=False, **kw)
        self.add_analysis_datagroup_to_file()

        # smooth the results
        self.y_smoothed = a_tools.smooth(self.measured_values[0],
                                         window_len=window_len)
        max_index = np.argmax(self.y_smoothed)
        self.max_delay = self.sweep_points[max_index]

        grp_name = "Maximum Analysis: Acquisition Delay"
        if grp_name not in self.analysis_group:
            grp = self.analysis_group.create_group(grp_name)
        else:
            grp = self.analysis_group[grp_name]
        grp.attrs.create(name='max_delay', data=self.max_delay)
        grp.attrs.create(name='window_length', data=window_len)

        textstr = "optimal delay = {:.0f} ns".format(self.max_delay * 1e9)

        if print_results:
            print(textstr)

        fig, ax = self.default_ax()
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=self.box_props)

        self.plot_results_vs_sweepparam(x=self.sweep_points * 1e9,
                                        y=self.measured_values[0],
                                        fig=fig, ax=ax,
                                        xlabel='Acquisition delay (ns)',
                                        ylabel='Signal amplitude (arb. units)',
                                        save=False)

        ax.plot(self.sweep_points * 1e9, self.y_smoothed, 'r-')
        ax.plot((self.max_delay * 1e9, self.max_delay * 1e9), ax.get_ylim(), 'g-')
        self.save_fig(fig, xlabel='delay', ylabel='amplitude', **kw)

        if close_file:
            self.data_file.close()

        return self.max_delay


class Hanger_Analysis_CosBackground(MeasurementAnalysis):

    def __init__(self, label='HM', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, fitting_model='hanger',
                             show_guess=False, show=False, **kw):
        super(self.__class__, self).run_default_analysis(
            close_file=False, **kw)
        self.add_analysis_datagroup_to_file()

        # Fit Power to a Lorentzian
        self.measured_powers = self.measured_values[0] ** 2

        min_index = np.argmin(self.measured_powers)
        max_index = np.argmax(self.measured_powers)

        self.min_frequency = self.sweep_points[min_index]
        self.max_frequency = self.sweep_points[max_index]

        self.peaks = a_tools.peak_finder((self.sweep_points),
                                         self.measured_values[0])

        if self.peaks['dip'] is not None:  # look for dips first
            f0 = self.peaks['dip']
            amplitude_factor = -1.
        elif self.peaks['peak'] is not None:  # then look for peaks
            f0 = self.peaks['peak']
            amplitude_factor = 1.
        else:  # Otherwise take center of range
            f0 = np.median(self.sweep_points)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        def poly(x, c0, c1, c2):
            "line"
            return c2 * x ** 2 + c1 * x + c0

        def cosine(x, amplitude, frequency, phase, offset):
            # Naming convention, frequency should be Hz
            # omega is in radial freq
            return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

        def hanger_function_amplitude(x, f0, Q, Qe, A, theta):
            '''
            This is the function for a hanger  which does not take into account
            a possible slope.
            This function may be preferred over SlopedHangerFunc if the area around
            the hanger is small.
            In this case it may misjudge the slope
            Theta is the asymmetry parameter
            '''
            return abs(A * (1. - Q / Qe * np.exp(1.j * theta) / (1. + 2.j * Q * (x - f0) / f0)))

        HangerModel = lmfit.Model(hanger_function_amplitude) \
            + lmfit.Model(cosine) \
            + lmfit.Model(poly)

        # amplitude_guess = np.pi*sigma_guess * abs(
        #     max(self.measured_powers)-min(self.measured_powers))
        amplitude_guess = max(self.measured_powers) - min(self.measured_powers)

        S21min = min(self.measured_values[0])
        # Creating parameters and estimations
        Q = f0 / abs(self.min_frequency - self.max_frequency)
        Qe = abs(Q / abs(1 - S21min))

        HangerModel.set_param_hint('f0', value=f0,
                                   min=min(self.sweep_points),
                                   max=max(self.sweep_points))
        HangerModel.set_param_hint('A', value=1)
        HangerModel.set_param_hint('Q', value=Q)
        HangerModel.set_param_hint('Qe', value=Qe)
        HangerModel.set_param_hint('Qi', expr='1./(1./Q-1./Qe*cos(theta))',
                                   vary=False)
        HangerModel.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
        HangerModel.set_param_hint('theta', value=0, min=-np.pi / 2,
                                   max=np.pi / 2)
        HangerModel.set_param_hint('slope', value=0, vary=True)

        HangerModel.set_param_hint('c0', value=0, vary=False)
        HangerModel.set_param_hint('c1', value=0, vary=True)
        HangerModel.set_param_hint('c2', value=0, vary=True)

        HangerModel.set_param_hint('amplitude', value=0.05, min=0, vary=False)
        HangerModel.set_param_hint(
            'frequency', value=50, min=0, max=300, vary=True)
        HangerModel.set_param_hint(
            'phase', value=0, min=0, max=2 * np.pi, vary=True)
        HangerModel.set_param_hint('offset', value=0, vary=True)

        self.params = HangerModel.make_params()

        fit_res = HangerModel.fit(data=self.measured_powers,
                                  x=self.sweep_points,
                                  params=self.params)

        self.fit_results = fit_res
        self.save_fitted_parameters(fit_res, var_name='HM')

        if print_fit_results is True:
            print(fit_res.fit_report())

        fig, ax = self.default_ax()
        # textstr = '$f_{\mathrm{center}}$ = %.4f $\pm$ (%.3g) GHz' % (
        #     fit_res.params['f0'].value, fit_res.params['f0'].stderr)
        # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        # verticalalignment='top', bbox=self.box_props)
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.measured_powers,
                                        fig=fig, ax=ax,
                                        xlabel=self.xlabel,
                                        ylabel=str('Power (arb. units)'),
                                        save=False)
        if show_guess:
            ax.plot(self.sweep_points, fit_res.init_fit, 'k--')
        ax.plot(self.sweep_points, fit_res.best_fit, 'r-')
        f0 = self.fit_results.values['f0']
        plt.plot(f0, fit_res.eval(x=f0), 'o', ms=8)
        if show:
            plt.show()
        self.save_fig(fig, xlabel=self.xlabel, ylabel='Power', **kw)
        if close_file:
            self.data_file.close()
        return fit_res


class Qubit_Spectroscopy_Analysis(MeasurementAnalysis):
    """
    Analysis script for a regular (ge peak/dip only) or a high power
    (ge and gf/2 peaks/dips) Qubit Spectroscopy:
        1. The I and Q data are combined using
            a_tools.calculate_distance_from_ground_state.
        2. The peaks/dips of the data are found using a_tools.peak_finder.
        3. If analyze_ef == False: the data is then fitted to a Lorentzian;
            else: to a double Lorentzian.
        3. The data, the best fit, and peak points are then plotted.

    Note:
    analyze_ef==True tells this routine to look for the gf/2 peak/dip.
    Even though the parameters for this second peak/dip use the termination
    "_ef," they refer to the parameters for the gf/2 transition NOT for the ef
    transition. It was easier to write x_ef than x_gf_over_2 or x_gf_half.

    Possible kw parameters:

        frequency_guess         (default="max")
            manually set the initial guess for qubit frequency
            options are
                None -> uses the peak finder to use a peak or dip
                max -> uses the maximally measured value to guess the freq
                float -> specify a value as the guess

        analyze_ef              (default=False)
            whether to look for a second peak/dip, which would be the at f_gf/2

        percentile              (default=20)
            percentile of the  data that is   considered background noise

        num_sigma_threshold     (default=5)
            used to define the threshold above(below) which to look for
            peaks(dips); threshold = background_mean + num_sigma_threshold *
            background_std

        window_len              (default=3)
            filtering window length; uses a_tools.smooth

        analysis_window         (default=10)
            how many data points (calibration points) to remove before sending
            data to peak_finder; uses a_tools.cut_edges,
            data = data[(analysis_window//2):-(analysis_window//2)]

        amp_only                (default=False)
            whether only I data exists
        save_name               (default='Source Frequency')
            figure name with which it will be saved
        auto                    (default=True)
            automatically perform the entire analysis upon call
        label                   (default=none?)
            label of the analysis routine
        folder                  (default=working folder)
            working folder
        NoCalPoints             (default=4)
            number of calibration points
        print_fit_results       (default=True)
            print the fit report
        print_frequency         (default=False)
            whether to print the f_ge and f_gf/2
        show                    (default=True)
            show the plots
        show_guess              (default=False)
            plot with initial guess values
        close_file              (default=True)
            close the hdf5 file
    """

    def __init__(self, label='Source', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def fit_data(self, analyze_ef=False, **kw):

        frequency_guess = kw.get('frequency_guess', 'max')
        percentile = kw.get('percentile', 20)
        num_sigma_threshold = kw.get('num_sigma_threshold', 5)
        window_len_filter = kw.get('window_len_filter', 3)
        optimize = kw.pop('optimize', True)
        verbose = kw.get('verbose', False)

        try:
            data_amp = self.measured_values[0]
            data_phase = self.measured_values[1]
            data_real = data_amp * np.cos(np.pi * data_phase / 180)
            data_imag = data_amp * np.sin(np.pi * data_phase / 180)
            self.data_dist = a_tools.calculate_distance_ground_state(
                data_real=data_real,
                data_imag=data_imag,
                normalize=False,
                percentile=60)
        except:
            # Quick fix to make it work with pulsed spec which does not
            # return both I,Q and, amp and phase
            # only using the amplitude!!
            self.data_dist = self.measured_values[0] - np.min(self.measured_values[0])

        # Smooth the data by "filtering"
        data_dist_smooth = a_tools.smooth(self.data_dist,
                                          window_len=window_len_filter)

        # Find peaks
        self.peaks = a_tools.peak_finder(self.sweep_points,
                                         data_dist_smooth,
                                         percentile=percentile,
                                         num_sigma_threshold=num_sigma_threshold,
                                         optimize=optimize,
                                         window_len=0)

        # Determine the guess
        # extract highest peak -> ge transition
        if frequency_guess is not None:
            if isinstance(frequency_guess, float):
                f0 = frequency_guess
            elif frequency_guess == 'max':
                f0 = self.sweep_points[np.argmax(data_dist_smooth)]
            kappa_guess = (max(self.sweep_points)-min(self.sweep_points))/20
            key = 'peak'
        elif self.peaks['dip'] is None:
            f0 = self.peaks['peak']
            kappa_guess = self.peaks['peak_width'] / 4
            key = 'peak'
        elif self.peaks['peak'] is None:
            f0 = self.peaks['dip']
            kappa_guess = self.peaks['dip_width'] / 4
            key = 'dip'
        # elif self.peaks['dip'] < self.peaks['peak']:
        elif np.abs(data_dist_smooth[self.peaks['dip_idx']]) < \
                np.abs(data_dist_smooth[self.peaks['peak_idx']]):
            f0 = self.peaks['peak']
            kappa_guess = self.peaks['peak_width'] / 4
            key = 'peak'
        # elif self.peaks['peak'] < self.peaks['dip']:
        elif np.abs(data_dist_smooth[self.peaks['dip_idx']]) > \
                np.abs(data_dist_smooth[self.peaks['peak_idx']]):
            f0 = self.peaks['dip']
            kappa_guess = self.peaks['dip_width'] / 4
            key = 'dip'
        else:  # Otherwise take center of range and raise warning
            f0 = np.median(self.sweep_points)
            kappa_guess = 0.005 * 1e9
            logging.warning('No peaks or dips have been found. Initial '
                            'frequency guess taken '
                            'as median of sweep points (f_guess={}), '
                            'initial linewidth '
                            'guess was taken as kappa_guess={}'.format(
                                f0, kappa_guess))
            key = 'peak'

        tallest_peak = f0  # the ge freq
        if verbose:
            print('Largest ' + key + ' is at ', tallest_peak)
        if f0 == self.peaks[key]:
            tallest_peak_idx = self.peaks[key + '_idx']
            if verbose:
                print('Largest ' + key + ' idx is ', tallest_peak_idx)

        amplitude_guess = np.pi * kappa_guess * \
            abs(max(self.data_dist) - min(self.data_dist))
        if key == 'dip':
            amplitude_guess = -amplitude_guess

        if analyze_ef is False:  # fit to a regular Lorentzian

            LorentzianModel = fit_mods.LorentzianModel

            LorentzianModel.set_param_hint('f0',
                                           min=min(self.sweep_points),
                                           max=max(self.sweep_points),
                                           value=f0)
            LorentzianModel.set_param_hint('A',
                                           value=amplitude_guess)

            LorentzianModel.set_param_hint('offset',
                                           value=np.mean(self.data_dist),
                                           vary=True)
            LorentzianModel.set_param_hint('kappa',
                                           value=kappa_guess,
                                           min=1,
                                           vary=True)
            LorentzianModel.set_param_hint('Q',
                                           expr='f0/kappa',
                                           vary=False)
            self.params = LorentzianModel.make_params()

            self.fit_res = LorentzianModel.fit(data=self.data_dist,
                                               f=self.sweep_points,
                                               params=self.params)

        else:  # fit a double Lorentzian and extract the 2nd peak as well
            # extract second highest peak -> ef transition

            f0, f0_gf_over_2, \
                kappa_guess, kappa_guess_ef = a_tools.find_second_peak(
                    sweep_pts=self.sweep_points,
                    data_dist_smooth=data_dist_smooth,
                    key=key,
                    peaks=self.peaks,
                    percentile=percentile,
                    verbose=verbose)

            if f0 == 0:
                f0 = tallest_peak
            if f0_gf_over_2 == 0:
                f0_gf_over_2 = tallest_peak
            if kappa_guess == 0:
                kappa_guess = 5e6
            if kappa_guess_ef == 0:
                kappa_guess_ef = 2.5e6

            amplitude_guess = np.pi * kappa_guess * \
                abs(max(self.data_dist) - min(self.data_dist))

            amplitude_guess_ef = 0.5 * np.pi * kappa_guess_ef * \
                abs(max(self.data_dist) -
                    min(self.data_dist))

            if key == 'dip':
                amplitude_guess = -amplitude_guess
                amplitude_guess_ef = -amplitude_guess_ef

            DoubleLorentzianModel = fit_mods.TwinLorentzModel

            DoubleLorentzianModel.set_param_hint('f0',
                                                 min=min(self.sweep_points),
                                                 max=max(self.sweep_points),
                                                 value=f0)
            DoubleLorentzianModel.set_param_hint('f0_gf_over_2',
                                                 min=min(self.sweep_points),
                                                 max=max(self.sweep_points),
                                                 value=f0_gf_over_2)
            DoubleLorentzianModel.set_param_hint('A',
                                                 value=amplitude_guess)  # ,
            # min=4*np.var(self.data_dist))
            DoubleLorentzianModel.set_param_hint('A_gf_over_2',
                                                 value=amplitude_guess_ef)  # ,
            # min=4*np.var(self.data_dist))
            DoubleLorentzianModel.set_param_hint('kappa',
                                                 value=kappa_guess,
                                                 min=0,
                                                 vary=True)
            DoubleLorentzianModel.set_param_hint('kappa_gf_over_2',
                                                 value=kappa_guess_ef,
                                                 min=0,
                                                 vary=True)
            DoubleLorentzianModel.set_param_hint('Q',
                                                 expr='f0/kappa',
                                                 vary=False)
            DoubleLorentzianModel.set_param_hint('Q_ef',
                                                 expr='f0_gf_over_2/kappa'
                                                      '_gf_over_2',
                                                 vary=False)
            self.params = DoubleLorentzianModel.make_params()

            self.fit_res = DoubleLorentzianModel.fit(data=self.data_dist,
                                                     f=self.sweep_points,
                                                     params=self.params)

            self.fit_results.append(self.fit_res)

    def run_default_analysis(self, print_fit_results=False, analyze_ef=False,
                             show=False, fit_results_peak=True, **kw):

        super(self.__class__, self).run_default_analysis(
            close_file=False, show=show, **kw)

        # Expects to have self.font_size, self.line_width,
        # self.marker_size_special, self.qb_name which should be defined in the
        # MeasurementAnalysis init.
        if not hasattr(self, 'font_size') and not hasattr(self, 'line_width') \
                and not hasattr(self, 'marker_size_special') \
                and not hasattr(self, 'qb_name'):
            try:
                q_idx = self.folder[-10::].index('q')
                self.qb_name = self.folder[-10::][q_idx::]
            except ValueError:
                self.qb_name = 'qb'
            self.font_size = 11
            self.line_width = 2
            self.marker_size_special = 8

        self.add_analysis_datagroup_to_file()
        self.savename = kw.get('save_name', 'Source Frequency')
        show_guess = kw.get('show_guess', False)
        close_file = kw.get('close_file', True)

        use_max = kw.get('use_max', False)

        self.fit_data(analyze_ef=analyze_ef, **kw)
        # get fitted frequency; gets called in QuDev_transmon.find_frequency
        self.fitted_freq = self.fit_res.params['f0'].value

        self.save_fitted_parameters(self.fit_res,
                                    var_name='distance', save_peaks=True)

        # Plotting distance from |0>
        fig_dist, ax_dist = self.default_ax()
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.data_dist,
                                        fig=fig_dist, ax=ax_dist,
                                        xlabel=self.sweep_name,
                                        x_unit=self.sweep_unit[0],
                                        ylabel='S21 distance (arb.units)',
                                        label=False,
                                        save=False)

        # plot Lorentzian with the fit results
        ax_dist.plot(self.sweep_points, self.fit_res.best_fit,
                     'r-', linewidth=self.line_width)

        # Plot a point for each plot at the chosen best fit f0 frequency (f_ge)
        f0 = self.fit_res.params['f0'].value
        f0_idx = a_tools.nearest_idx(self.sweep_points, f0)
        ax_dist.plot(f0, self.fit_res.best_fit[f0_idx], 'o',
                     ms=self.marker_size_special)

        if analyze_ef:
            # plot the ef/2 point as well
            f0_gf_over_2 = self.fit_res.params['f0_gf_over_2'].value
            self.fitted_freq_gf_over_2 = f0_gf_over_2
            f0_gf_over_2_idx = a_tools.nearest_idx(self.sweep_points,
                                                   f0_gf_over_2)
            ax_dist.plot(f0_gf_over_2,
                         self.fit_res.best_fit[f0_gf_over_2_idx],
                         'o', ms=self.marker_size_special)
        if show_guess:
            # plot Lorentzian with initial guess
            ax_dist.plot(self.sweep_points, self.fit_res.init_fit,
                         'k--', linewidth=self.line_width)

        scale = SI_prefix_and_scale_factor(val=max(abs(ax_dist.get_xticks())),
                                           unit=self.sweep_unit[0])[0]

        instr_set = self.data_file['Instrument settings']

        if analyze_ef:
            try:
                old_freq = float(instr_set[self.qb_name].attrs['f_qubit'])
                old_freq_ef = float(
                    instr_set[self.qb_name].attrs['f_ef_qubit'])

                label = 'f0={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                        '\nold f0={:.5f} GHz' \
                        '\nkappa0={:.4f} MHz $\pm$ ({:.2f}) MHz\n' \
                        'f0_gf/2={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                        '\nold f0_gf/2={:.5f} GHz' \
                        '\nkappa_gf={:.4f} MHz $\pm$ ({:.2f}) MHz'.format(
                            self.fit_res.params['f0'].value * scale,
                            self.fit_res.params['f0'].stderr / 1e6,
                            old_freq * scale,
                            self.fit_res.params['kappa'].value / 1e6,
                            self.fit_res.params['kappa'].stderr / 1e6,
                            self.fit_res.params['f0_gf_over_2'].value * scale,
                            self.fit_res.params['f0_gf_over_2'].stderr / 1e6,
                            old_freq_ef * scale,
                            self.fit_res.params['kappa_gf_over_2'].value / 1e6,
                            self.fit_res.params['kappa_gf_over_2'].stderr / 1e6)
            except (TypeError, KeyError, ValueError):
                logging.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
                label = 'f0={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                        '\nkappa0={:.4f} MHz $\pm$ ({:.2f}) MHz\n' \
                        'f0_gf/2={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                        '\nkappa_gf={:.4f} MHz $\pm$ ({:.2f}) MHz'.format(
                            self.fit_res.params['f0'].value * scale,
                            self.fit_res.params['f0'].stderr / 1e6,
                            self.fit_res.params['kappa'].value / 1e6,
                            self.fit_res.params['kappa'].stderr / 1e6,
                            self.fit_res.params['f0_gf_over_2'].value * scale,
                            self.fit_res.params['f0_gf_over_2'].stderr / 1e6,
                            self.fit_res.params['kappa_gf_over_2'].value / 1e6,
                            self.fit_res.params['kappa_gf_over_2'].stderr / 1e6)
        else:
            try:
                old_freq = float(instr_set[self.qb_name].attrs['f_qubit'])

                label = 'f0={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                        '\nold f0={:.5f} GHz' \
                        '\nkappa0={:.4f} MHz $\pm$ ({:.2f}) MHz'.format(
                            self.fit_res.params['f0'].value * scale,
                            self.fit_res.params['f0'].stderr / 1e6,
                            old_freq * scale,
                            self.fit_res.params['kappa'].value / 1e6,
                            self.fit_res.params['kappa'].stderr / 1e6)
            except (TypeError, KeyError, ValueError):
                logging.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
                try: #Dirty fix, should already be fine in Develop
                    label = 'f0={:.5f} GHz $\pm$ ({:.2f}) MHz ' \
                            '\nkappa0={:.4f} MHz $\pm$ ({:.2f}) MHz'.format(
                                self.fit_res.params['f0'].value * scale,
                                self.fit_res.params['f0'].stderr / 1e6,
                                self.fit_res.params['kappa'].value / 1e6,
                                self.fit_res.params['kappa'].stderr / 1e6)
                except:
                    label = None

        fig_dist.text(0.5, 0, label, transform=ax_dist.transAxes,
                      fontsize=self.font_size, verticalalignment='top',
                      horizontalalignment='center', bbox=self.box_props)

        if print_fit_results is True:
            print(self.fit_res.fit_report())

        if kw.get('print_frequency', False):
            if analyze_ef:
                print('f_ge = {:.5} (GHz) \t f_ge Stderr = {:.5} (MHz) \n'
                      'f_gf/2 = {:.5} (GHz) \t f_gf/2 Stderr = {:.5} '
                      '(MHz)'.format(
                          self.fitted_freq * scale,
                          self.fit_res.params['f0'].stderr * 1e-6,
                          self.fitted_freq_gf_over_2 * scale,
                          self.fit_res.params['f0_gf_over_2'].stderr * 1e-6))
            else:
                print('f_ge = {:.5} (GHz) \t '
                      'f_ge Stderr = {:.5} (MHz)'.format(
                          self.fitted_freq * scale,
                          self.fit_res.params['f0'].stderr * 1e-6))

        if show:
            plt.show()
        self.save_fig(fig_dist, figname='Source frequency distance', **kw)

        if close_file:
            self.data_file.close()

    def get_frequency_estimate(self, peak=False):
        best_fit = self.get_best_fit_results(peak=peak)
        frequency_estimate = best_fit['f0'].attrs['value']
        frequency_estimate_stderr = best_fit['f0'].attrs['stderr']

        return frequency_estimate, frequency_estimate_stderr

    def get_linewidth_estimate(self):
        best_fit = self.get_best_fit_results()
        linewidth_estimate = best_fit['kappa'].attrs['value']
        linewidth_estimate_stderr = best_fit['kappa'].attrs['stderr']

        return linewidth_estimate, linewidth_estimate_stderr


class Mixer_Calibration_Analysis(MeasurementAnalysis):
    '''
    Simple analysis that takes the minimum value measured and adds it
    to the analysis datagroup
    '''

    def __init__(self, label='offset', **kw):
        kw['label'] = label
        # Adds the label to the keyword arguments so that it can be passed
        # on in **kw
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, **kw):
        super(self.__class__, self).run_default_analysis(
            close_file=False, **kw)
        # self.add_analysis_datagroup_to_file() #Currently does not write a val here
        # Fit Power to a Lorentzian
        self.measured_powers = self.measured_values[0]
        minimum_index = np.argmin(self.measured_powers)
        minimum_dac_value = self.sweep_points[minimum_index]

        self.fit_results.append(minimum_dac_value)

        fig, ax = self.default_ax()

        self.plot_results_vs_sweepparam(
            x=self.sweep_points, y=self.measured_powers,
            fig=fig, ax=ax,
            xlabel=self.xlabel, ylabel=str('Power (dBm)'),
            save=False)

        self.add_analysis_datagroup_to_file()
        if 'optimization_result' not in self.analysis_group:
            fid_grp = self.analysis_group.create_group('optimization_result')
        else:
            fid_grp = self.analysis_group['optimization_result']
        fid_grp.attrs.create(name='minimum_dac_value',
                             data=minimum_dac_value)

        self.save_fig(fig, xlabel=self.xlabel, ylabel='Power', **kw)
        if close_file:
            self.data_file.close()


class Qubit_Characterization_Analysis(MeasurementAnalysis):

    def __init__(self, label='Qubit_Char', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        figsize = kw.pop('figsize', (11, 10))
        close_file = kw.pop('close_file', True)
        x = self.sweep_points
        x_fine = np.linspace(
            self.sweep_points[0], self.sweep_points[-1], 1000)

        qubit_freq = self.measured_values[2]
        qubit_freq_stderr = self.measured_values[3]

        AWG_Pulse_amp_ch1 = self.measured_values[4]
        AWG_Pulse_amp_ch2 = self.measured_values[5]

        T1 = self.measured_values[6]
        T1_stderr = self.measured_values[7]
        T2_star = self.measured_values[8]
        T2_star_stderr = self.measured_values[9]
        T2_echo = self.measured_values[10]
        T2_echo_stderr = self.measured_values[11]

        self.qubit_freq = qubit_freq

        fit_res = fit_qubit_frequency(sweep_points=x, data=qubit_freq,
                                      data_file=self.data_file,
                                      mode='dac')
        self.save_fitted_parameters(fit_res,
                                    var_name='Qubit_freq_dac')
        self.fit_res = fit_res
        fitted_freqs = fit_mods.QubitFreqDac(
            x_fine, E_c=fit_res.best_values['E_c'],
            f_max=fit_res.best_values['f_max'],
            dac_flux_coefficient=fit_res.best_values['dac_flux_coefficient'],
            dac_sweet_spot=fit_res.best_values['dac_sweet_spot'])

        fig1, ax1 = self.default_ax()
        ax1.errorbar(x=x, y=qubit_freq, yerr=qubit_freq_stderr,
                     label='data', fmt='ob')
        ax1.plot(x_fine, fitted_freqs, '--c', label='fit')
        ax1.legend()
        ax1.set_title(self.timestamp_string + '\n' + 'Qubit Frequency')
        ax1.set_xlabel((str(self.sweep_name + ' (' + self.sweep_unit + ')')))
        ax1.set_ylabel(r'$f_{qubit}$ (GHz)')
        ax1.grid()

        fig2, axarray2 = plt.subplots(2, 1, figsize=figsize)
        axarray2[0].set_title(self.timestamp_string + '\n' + 'Qubit Coherence')
        axarray2[0].errorbar(
            x=x,
            y=T1 * 1e-3, yerr=T1_stderr * 1e-3,
            fmt='o', label='$T_1$')
        axarray2[0].errorbar(
            x=x,
            y=T2_echo * 1e-3, yerr=T2_echo_stderr * 1e-3,
            fmt='o', label='$T_2$-echo')
        axarray2[0].errorbar(
            x=x,
            y=T2_star * 1e-3, yerr=T2_star_stderr * 1e-3,
            fmt='o', label='$T_2$-star')
        axarray2[0].set_xlabel(r'dac voltage')
        axarray2[0].set_ylabel(r'$\tau (\mu s)$ ')
        # axarray[0].set_xlim(-600, 700)
        axarray2[0].set_ylim(0, max([max(T1 * 1e-3), max(T2_echo * 1e-3)])
                             + 3 * max(T1_stderr * 1e-3))
        axarray2[0].legend()
        axarray2[0].grid()

        axarray2[1].errorbar(
            x=qubit_freq * 1e-9,
            y=T1 * 1e-3, yerr=T1_stderr * 1e-3,
            fmt='o', label='$T_1$')
        axarray2[1].errorbar(
            x=qubit_freq * 1e-9,
            y=T2_echo * 1e-3, yerr=T2_echo_stderr * 1e-3,
            fmt='o', label='$T_2$-echo')
        axarray2[1].errorbar(
            x=qubit_freq * 1e-9,
            y=T2_star * 1e-3, yerr=T2_star_stderr * 1e-3,
            fmt='o', label='$T_2^\star$')
        axarray2[1].set_xlabel(r'$f_{qubit}$ (GHz)')
        axarray2[1].set_ylabel(r'$\tau (\mu s)$ ')
        # axarray[1].set_xlim(-600, 700)
        axarray2[1].set_ylim(0, max([max(T1 * 1e-3), max(T2_echo * 1e-3)])
                             + 3 * max(T1_stderr * 1e-3))
        axarray2[1].legend(loc=2)
        axarray2[1].grid()

        fig3, axarray3 = plt.subplots(2, 1, figsize=figsize)
        axarray3[0].set_title(self.timestamp + '\n' + 'AWG pulse amplitude')
        axarray3[0].plot(x, AWG_Pulse_amp_ch1, 'o')
        axarray3[0].plot(x, AWG_Pulse_amp_ch2, 'o')
        axarray3[0].set_xlabel(r'dac voltage')
        axarray3[0].set_ylabel(r'att. (a.u.) ')
        # axarray[0].set_xlim(x[0], x[-1])
        axarray3[0].set_ylim(0, max([max(AWG_Pulse_amp_ch1),
                                     max(AWG_Pulse_amp_ch2)]))
        # Needs to be based on duplexer amplitude controlled by duplexer or not
        axarray3[0].legend()
        axarray3[0].grid()

        axarray3[1].plot(qubit_freq, AWG_Pulse_amp_ch1, 'o')
        axarray3[1].plot(qubit_freq, AWG_Pulse_amp_ch2, 'o')
        axarray3[1].set_xlabel(r'$f_{qubit}$ (GHz)')
        axarray3[1].set_ylabel(r'att. (a.u.) ')
        # axarray[1].set_xlim(qubit_freq[0], qubit_freq[1]+1e6)
        # axarray3[1].set_ylim(0, 65536)
        axarray3[1].grid()

        self.save_fig(fig1, figname=str('Qubit_Frequency'), **kw)
        self.save_fig(fig2, figname=str('Qubit_Coherence'), **kw)
        self.save_fig(fig3, figname=str('Duplex_Attenuation'), **kw)
        if close_file:
            self.finish(**kw)


class Qubit_Sweeped_Spectroscopy_Analysis(Qubit_Characterization_Analysis):

    def __init__(self, qubit_name, label='Qubit_Char', fit_mode='flux', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        self.fit_mode = fit_mode
        self.qubit_name = qubit_name
        super(Qubit_Characterization_Analysis, self).__init__(**kw)

    def run_default_analysis(self, **kw):
        self.add_analysis_datagroup_to_file()
        print_fit_results = kw.pop('print_fit_results', False)
        self.get_naming_and_values()
        show_guess = kw.pop('show_guess', False)
        close_file = kw.pop('close_file', True)
        x = self.sweep_points
        x_fine = np.linspace(
            self.sweep_points[0], self.sweep_points[-1], 1000) * 1e-3

        self.qubit_freq = self.measured_values[2]
        self.qubit_freq_stderr = self.measured_values[3]

        fit_res = fit_qubit_frequency(sweep_points=x * 1e-3,
                                      data=self.qubit_freq * 1e9,
                                      mode=self.fit_mode,
                                      data_file=self.data_file,
                                      qubit_name=self.qubit_name, **kw)
        self.save_fitted_parameters(fit_res,
                                    var_name='Qubit_freq_dac')
        self.fit_res = fit_res

        fitted_freqs = fit_mods.QubitFreqFlux(
            x_fine, E_c=fit_res.best_values['E_c'],
            f_max=fit_res.best_values['f_max'],
            flux_zero=fit_res.best_values['flux_zero'],
            dac_offset=fit_res.best_values['dac_offset'])

        fig1, ax1 = self.default_ax()
        ax1.errorbar(x=x * 1e-3, y=self.qubit_freq * 1e9,
                     yerr=self.qubit_freq_stderr,
                     label='data', fmt='ob')

        if show_guess:
            ax1.plot(x * 1e-3, fit_res.init_fit, 'k--')

        ax1.plot(x_fine, fitted_freqs, '--c', label='fit')
        ax1.legend()
        ax1.set_title(self.timestamp + '\n' + 'Qubit Frequency')
        ax1.set_xlabel((str(self.sweep_name + ' (V)')))
        ax1.set_ylabel(r'$f_{qubit}$ (GHz)')
        ax1.grid()
        self.save_fig(fig1, figname=str('Qubit_Frequency'), **kw)

        if print_fit_results:
            print(fit_res.fit_report())
        if close_file:
            self.finish()


class TwoD_Analysis(MeasurementAnalysis):
    '''
    Analysis for 2D measurements.
    '''

    def run_default_analysis(self, normalize=False, plot_linecuts=True,
                             linecut_log=False, colorplot_log=False,
                             plot_all=True, save_fig=True,
                             transpose=False, figsize=None, filtered=False,
                             subtract_mean_x=False, subtract_mean_y=False,
                             **kw):
        '''
        Args:
            linecut_log (bool):
                log scale for the line cut?
                Remember to set the labels correctly.
            colorplot_log (string/bool):
                True/False for z axis scaling, or any string containing any
                combination of letters x, y, z for scaling of the according axis.
                Remember to set the labels correctly.

        '''
        close_file = kw.pop('close_file', True)

        self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []

        for i, meas_vals in enumerate(self.measured_values):
            # kw["zlabel"] = kw.get("zlabel", self.value_names[i])
            # kw["z_unit"] = kw.get("z_unit", self.value_units[i])

            if filtered:
                if self.value_names[i] == 'Phase':
                    self.measured_values[i] = dm_tools.filter_resonator_visibility(
                        x=self.sweep_points,
                        y=self.sweep_points_2D,
                        z=self.measured_values[i],
                        **kw)

            if (not plot_all) & (i >= 1):
                break
            # Linecuts are above because somehow normalization applies to both
            # colorplot and linecuts otherwise.
            if plot_linecuts:
                fig, ax = plt.subplots(figsize=figsize)
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                savename = 'linecut_{}'.format(self.value_names[i])
                fig_title = '{} {} \nlinecut {}'.format(
                    self.timestamp_string, self.measurementstring,
                    self.value_names[i])
                a_tools.linecut_plot(x=self.sweep_points,
                                     y=self.sweep_points_2D,
                                     z=self.measured_values[i],
                                     y_name=self.parameter_names[1],
                                     y_unit=self.parameter_units[1],
                                     log=linecut_log,
                                     fig=fig, ax=ax, **kw)
                ax.set_title(fig_title)
                set_xlabel(ax, self.parameter_names[0],
                           self.parameter_units[0])
                # ylabel is value units as we are plotting linecuts
                set_ylabel(ax, self.value_names[i],
                           self.value_units[i])

                if save_fig:
                    self.save_fig(fig, figname=savename,
                                  fig_tight=False, **kw)

            fig, ax = plt.subplots(figsize=figsize)
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            if normalize:
                print("normalize on")

            self.ax_array.append(ax)
            savename = 'Heatmap_{}'.format(self.value_names[i])
            fig_title = '{} {} \n{}'.format(
                self.timestamp_string, self.measurementstring,
                self.value_names[i])

            # subtract mean from each row/column if demanded
            plot_zvals = meas_vals.transpose()
            if subtract_mean_x:
                plot_zvals = plot_zvals - np.mean(plot_zvals, axis=1)[:, None]
            if subtract_mean_y:
                plot_zvals = plot_zvals - np.mean(plot_zvals, axis=0)[None, :]

            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=plot_zvals,
                               zlabel=self.value_names[i],
                               z_unit=self.value_units[i],
                               fig=fig, ax=ax,
                               log=colorplot_log,
                               transpose=transpose,
                               normalize=normalize,
                               **kw)

            set_xlabel(ax, self.parameter_names[0], self.parameter_units[0])
            set_ylabel(ax, self.parameter_names[1], self.parameter_units[1])

            if save_fig:
                self.save_fig(fig, figname=savename, **kw)
        if close_file:
            self.finish()


class Mixer_Skewness_Analysis(TwoD_Analysis):

    def run_default_analysis(self, save_fig=True,
                             **kw):
        close_file = kw.pop('close_file', True)
        self.get_naming_and_values_2D()

        self.fig_array = []
        self.ax_array = []
        for i, meas_vals in enumerate(self.measured_values):
            fig, ax = self.default_ax(figsize=(8, 5))
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            fig_title = '{timestamp}_{measurement}_{val_name}'.format(
                timestamp=self.timestamp_string,
                measurement=self.measurementstring,
                val_name=self.zlabels[i])
            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=meas_vals.transpose(),
                               plot_title=fig_title,
                               xlabel=self.xlabel,
                               ylabel=self.ylabel,
                               zlabel=self.zlabels[i],
                               fig=fig, ax=ax, **kw)

            data_arr = self.measured_values[0].T
            ampl_min_lst = np.min(data_arr, axis=1)
            phase_min_idx = np.argmin(ampl_min_lst)
            self.phase_min = self.sweep_points_2D[phase_min_idx]

            ampl_min_idx = np.argmin(data_arr[phase_min_idx])
            self.QI_min = self.sweep_points[ampl_min_idx]

            textstr = 'Q phase of minimum =  %.2f deg' % self.phase_min + '\n' + \
                      'Q/I ratio of minimum = %.2f' % self.QI_min

            ax.text(0.60, 0.95, textstr,
                    transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    horizontalalignment='left',
                    bbox=self.box_props)

            if save_fig:
                self.save_fig(fig, figname=fig_title, **kw)
        if close_file:
            self.finish()

        return self.QI_min, self.phase_min


class Three_Tone_Spectroscopy_Analysis(MeasurementAnalysis):
    '''
    Analysis for 2D measurement Three tone spectroscopy.
    **kwargs:
        f01: fuess for f01
        f12: guess for f12

    '''

    def __init__(self, label='Three_tone', **kw):
        kw['label'] = label
        # kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, f01=None, f12=None,
                             amp_lims=[None, None], line_color='k',
                             phase_lims=[None, None], **kw):
        self.get_naming_and_values_2D()
        # figsize wider for colorbar
        fig1, ax1 = self.default_ax(figsize=(8, 5))
        measured_powers = self.measured_values[0]
        measured_phases = self.measured_values[1]

        fig1_title = self.timestamp_string + \
            self.measurementstring + '_' + 'Amplitude'
        a_tools.color_plot(x=self.sweep_points,
                           y=self.sweep_points_2D,
                           z=measured_powers.transpose(),
                           plot_title=fig1_title,
                           xlabel=self.xlabel,
                           ylabel=self.ylabel,
                           zlabel=self.zlabels[0],
                           clim=amp_lims,
                           fig=fig1, ax=ax1, **kw)

        # figsize wider for colorbar
        fig2, ax2 = self.default_ax(figsize=(8, 5))
        fig2_title = self.timestamp_string + self.measurementstring + '_' + 'Phase'
        if (measured_phases>170).any() and (measured_phases<-170).any():
            measured_phases = np.mod(measured_phases, 360)
        a_tools.color_plot(x=self.sweep_points,
                           y=self.sweep_points_2D,
                           z=measured_phases.transpose(),
                           xlabel=self.xlabel,
                           ylabel=self.ylabel,
                           zlabel=self.zlabels[1],
                           plot_title=fig2_title,
                           fig=fig2, ax=ax2)

        if f01 is not None:
            ax1.vlines(f01, min(self.sweep_points_2D),
                       max(self.sweep_points_2D),
                       linestyles='dashed', lw=2, colors=line_color, alpha=.5)
            ax2.vlines(f01, min(self.sweep_points_2D),
                       max(self.sweep_points_2D),
                       linestyles='dashed', lw=2, colors=line_color, alpha=.5)
        if f12 is not None:
            ax1.plot((min(self.sweep_points),
                      max(self.sweep_points)),
                     (f01 + f12 - min(self.sweep_points),
                      f01 + f12 - max(self.sweep_points)),
                     linestyle='dashed', lw=2, color=line_color, alpha=.5)
            ax2.plot((min(self.sweep_points),
                      max(self.sweep_points)),
                     (f01 + f12 - min(self.sweep_points),
                      f01 + f12 - max(self.sweep_points)),
                     linestyle='dashed', lw=2, color=line_color, alpha=.5)
        if (f01 is not None) and (f12 is not None):
            anharm = f01 - f12
            EC, EJ = a_tools.fit_EC_EJ(f01, f12)
            # EC *= 1000

            textstr = 'f01 = {:.4g} GHz'.format(f01 * 1e-9) + '\n' + \
                      'f12 = {:.4g} GHz'.format(f12 * 1e-9) + '\n' + \
                      'anharm = {:.4g} MHz'.format(anharm * 1e-6) + '\n' + \
                      'EC ~= {:.4g} MHz'.format(EC * 1e-6) + '\n' + \
                      'EJ = {:.4g} GHz'.format(EJ * 1e-9)
            ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes,
                     fontsize=11,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=self.box_props)
            ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes,
                     fontsize=11,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=self.box_props)
        self.save_fig(fig1, figname=fig1_title, **kw)
        self.save_fig(fig2, figname=fig2_title, **kw)
        self.finish()

    def fit_twin_lorentz(self, x, data,
                         f01, f12, **kw):
        vary_f01 = kw.pop('vary_f01', False)
        twin_lor_m = fit_mods.TwinLorentzModel
        twin_lor_m.set_param_hint('center_a', value=f01,
                                  vary=vary_f01)
        twin_lor_m.set_param_hint('center_b', value=f12,
                                  vary=True)
        twin_lor_m.set_param_hint('amplitude_a', value=max(data),
                                  vary=True)
        twin_lor_m.set_param_hint('amplitude_b', value=max(data),
                                  vary=True)
        twin_lor_m.set_param_hint('sigma_a', value=0.001,
                                  vary=True)
        twin_lor_m.set_param_hint('sigma_b', value=0.001,
                                  vary=True)
        twin_lor_m.set_param_hint('background', value=0,
                                  vary=True)
        params = twin_lor_m.make_params()
        fit_res = twin_lor_m.fit(data=data, x=x, params=params)
        return fit_res


class Resonator_Powerscan_Analysis(MeasurementAnalysis):

    def __init__(self, label='powersweep', **kw):
        super(self.__class__, self).__init__(**kw)

    # def run_default_analysis(self,  normalize=True, w_low_power=None,
    #                          w_high_power=None, **kw):
    # super(self.__class__, self).run_default_analysis(close_file=False,
    #     save_fig=False, **kw)
    # close_file = kw.pop('close_file', True)
    def run_default_analysis(self, normalize=True, plot_Q=True, plot_f0=True,
                             plot_linecuts=True, linecut_log=True,
                             plot_all=False, save_fig=True, use_min=False,
                             **kw):
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()

        self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []
        fits = {}  # Dictionary to store the fit results in. Fit results are a
        # dictionary themselfes -> Dictionary of Dictionaries

        f0 = np.zeros(len(self.sweep_points_2D))
        for u, power in enumerate(self.sweep_points_2D):
            fit_res = self.fit_hanger_model(
                self.sweep_points, self.measured_values[0][:, u])
            self.save_fitted_parameters(
                fit_res, var_name='Powersweep' + str(u))
            fits[str(power)] = fit_res
            if use_min:
                min_index = np.argmin(self.measured_values[0][:, u])
                f0[u] = np.min(self.sweep_points[min_index])
            else:
                f0[u] = fits[str(power)].values['f0']
            self.f0 = f0

        self.fit_results = fits

        xlabel = kw.pop("xlabel", self.sweep_name)
        ylabel = kw.pop("ylabel", self.sweep_name_2D)
        x_unit = kw.pop("x_unit", self.sweep_unit)
        y_unit = kw.pop("y_unit", self.sweep_unit_2D)
        z_unit_linecuts = self.value_units[0]

        for i, meas_vals in enumerate(self.measured_values):
            if "zlabel" not in kw:
                kw["zlabel"] = self.value_names[i]
            if "z_unit" not in kw:
                if normalize:
                    kw["z_unit"] = 'normalized'
                else:
                    kw["z_unit"] = self.value_units[i]

            if (not plot_all) & (i >= 1):
                break
            # Linecuts are above because normalization changes the values of the
            # object. Thus it affects both colorplot and linecuts otherwise.
            if plot_Q:
                Q = np.zeros(len(self.sweep_points_2D))
                Qc = np.zeros(len(self.sweep_points_2D))
                for u, power in enumerate(self.sweep_points_2D):
                    Q[u] = self.fit_results[str(power)].values['Q']
                    Qc[u] = self.fit_results[str(power)].values['Qc']
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_QvsPower'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                ax.plot(
                    self.sweep_points_2D, Q, 'blue', label='Loaded Q-Factor')
                ax.plot(
                    self.sweep_points_2D, Qc, 'green', label='Coupling Q-Factor')
                ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
                ax.set_position([0.1, 0.1, 0.5, 0.8])
                set_ylabel(ax, 'Quality Factor')
                set_xlabel(ax, ylabel, y_unit)

                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            if plot_f0:
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_f0vsPower'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                ax.plot(self.sweep_points_2D, f0, 'blue', marker='o',
                        label='Cavity Frequency')
                ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
                ax.set_position([0.15, 0.1, 0.5, 0.8])
                set_ylabel(ax, xlabel, x_unit)
                set_xlabel(ax, ylabel, y_unit)

                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            if plot_linecuts:
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_linecut'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                a_tools.linecut_plot(x=self.sweep_points,
                                     y=self.sweep_points_2D,
                                     z=self.measured_values[i],
                                     plot_title=fig_title,
                                     log=linecut_log,
                                     xlabel=xlabel,
                                     x_unit=x_unit,
                                     y_name=ylabel,
                                     y_unit=y_unit,
                                     z_unit_linecuts=z_unit_linecuts,
                                     fig=fig, ax=ax, **kw)
                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            fig, ax = self.default_ax(figsize=(8, 5))
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            if normalize:
                meas_vals = a_tools.normalize_2D_data(meas_vals)
            fig_title = '{timestamp}_{measurement}_{val_name}'.format(
                timestamp=self.timestamp_string,
                measurement=self.measurementstring,
                val_name=self.zlabels[i])

            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=meas_vals.transpose(),
                               plot_title=fig_title,
                               xlabel=xlabel,
                               x_unit=x_unit,
                               ylabel=ylabel,
                               y_unit=y_unit,
                               fig=fig, ax=ax, **kw)
            if save_fig:
                self.save_fig(fig, figname=fig_title, **kw)

        if close_file:
            self.finish()

        # Find low power regime
        threshold = 0.1e6  # Gotta love hardcoded stuff
        f_low = f0[0]
        P_result = self.sweep_points_2D[0]
        try:
            for u, f in enumerate(f0):
                if np.abs(f0[0] - f0[u+1]) < threshold:
                    f_low = f0[u+1]
                    P_result = self.sweep_points_2D[u]
                else:
                    break
        except IndexError:
            pass

        # High power regime: just use the value at highest power
        f_high = f0[-1]

        if (f_high < f_low):
            shift = f_high - f_low
        else:
            shift = 0
            logging.warning('No power shift found. Consider attenuation')
            # raise Exception('High power regime frequency found to be higher than'
            #                 'low power regime frequency')

        self.f_low = f_low
        self.f_high = f_high
        self.shift = shift
        self.power = P_result

    def fit_hanger_model(self, sweep_values, measured_values):
        HangerModel = fit_mods.SlopedHangerAmplitudeModel

        # amplitude_guess = np.pi*sigma_guess * abs(
        #     max(self.measured_powers)-min(self.measured_powers))

        # Fit Power to a Lorentzian
        measured_powers = measured_values ** 2

        min_index = np.argmin(measured_powers)
        max_index = np.argmax(measured_powers)

        min_frequency = sweep_values[min_index]
        max_frequency = sweep_values[max_index]

        peaks = a_tools.peak_finder((sweep_values),
                                    measured_values)

        if peaks['dip'] is not None:  # look for dips first
            f0 = peaks['dip']
            amplitude_factor = -1.
        elif peaks['peak'] is not None:  # then look for peaks
            f0 = peaks['peak']
            amplitude_factor = 1.
        else:  # Otherwise take center of range
            f0 = np.median(sweep_values)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        amplitude_guess = max(measured_powers) - min(measured_powers)
        # Creating parameters and estimations
        S21min = min(measured_values) / max(measured_values)

        Q = f0 / abs(min_frequency - max_frequency)
        Qe = abs(Q / abs(1 - S21min))

        HangerModel.set_param_hint('f0', value=f0,
                                   min=min(sweep_values),
                                   max=max(sweep_values))
        HangerModel.set_param_hint('A', value=amplitude_guess)
        HangerModel.set_param_hint('Q', value=Q)
        HangerModel.set_param_hint('Qe', value=Qe)
        HangerModel.set_param_hint('Qi', expr='1./(1./Q-1./Qe*cos(theta))',
                                   vary=False)
        HangerModel.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
        HangerModel.set_param_hint('theta', value=0, min=-np.pi / 2,
                                   max=np.pi / 2)
        HangerModel.set_param_hint('slope', value=0, vary=True,
                                   min=-1, max=1)
        params = HangerModel.make_params()
        fit_res = HangerModel.fit(data=measured_powers,
                                  f=sweep_values * 1.e9,
                                  params=params)

        return fit_res

class Resonator_Powerscan_Analysis_test(MeasurementAnalysis):

    def __init__(self, label='powersweep', **kw):
        super(self.__class__, self).__init__(**kw)

    # def run_default_analysis(self,  normalize=True, w_low_power=None,
    #                          w_high_power=None, **kw):
    # super(self.__class__, self).run_default_analysis(close_file=False,
    #     save_fig=False, **kw)
    # close_file = kw.pop('close_file', True)
    def run_default_analysis(self, normalize=True, plot_Q=True, plot_f0=True,
                             plot_linecuts=True, linecut_log=True,
                             plot_all=False, save_fig=True, use_min=False,
                             **kw):
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()

        self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []
        fits = {}  # Dictionary to store the fit results in. Fit results are a
        # dictionary themselfes -> Dictionary of Dictionaries

        f0 = np.zeros(len(self.sweep_points_2D))
        for u, power in enumerate(self.sweep_points_2D):
            fit_res = self.fit_hanger_model(
                self.sweep_points, self.measured_values[0][:, u])
            self.save_fitted_parameters(
                fit_res, var_name='Powersweep' + str(u))
            fits[str(power)] = fit_res
            if use_min:
                min_index = np.argmin(self.measured_values[0][:, u])
                f0[u] = np.min(self.sweep_points[min_index])
            else:
                f0[u] = fits[str(power)].values['f0']
            self.f0 = f0

        self.fit_results = fits

        xlabel = kw.pop("xlabel", self.sweep_name)
        ylabel = kw.pop("ylabel", self.sweep_name_2D)
        x_unit = kw.pop("x_unit", self.sweep_unit)
        y_unit = kw.pop("y_unit", self.sweep_unit_2D)
        z_unit_linecuts = self.value_units[0]

        for i, meas_vals in enumerate(self.measured_values):
            if "zlabel" not in kw:
                kw["zlabel"] = self.value_names[i]
            if "z_unit" not in kw:
                if normalize:
                    kw["z_unit"] = 'normalized'
                else:
                    kw["z_unit"] = self.value_units[i]

            if (not plot_all) & (i >= 1):
                break
            # Linecuts are above because normalization changes the values of the
            # object. Thus it affects both colorplot and linecuts otherwise.
            if plot_Q:
                Q = np.zeros(len(self.sweep_points_2D))
                Qc = np.zeros(len(self.sweep_points_2D))
                for u, power in enumerate(self.sweep_points_2D):
                    Q[u] = self.fit_results[str(power)].values['Q']
                    Qc[u] = self.fit_results[str(power)].values['Qc']
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_QvsPower'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                ax.plot(
                    self.sweep_points_2D, Q, 'blue', label='Loaded Q-Factor')
                ax.plot(
                    self.sweep_points_2D, Qc, 'green', label='Coupling Q-Factor')
                ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
                ax.set_position([0.1, 0.1, 0.5, 0.8])
                set_ylabel(ax, 'Quality Factor')
                set_xlabel(ax, ylabel, y_unit)

                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            if plot_f0:
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_f0vsPower'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                ax.plot(self.sweep_points_2D, f0, 'blue', marker='o',
                        label='Cavity Frequency')
                ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
                ax.set_position([0.15, 0.1, 0.5, 0.8])
                set_ylabel(ax, xlabel, x_unit)
                set_xlabel(ax, ylabel, y_unit)

                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            if plot_linecuts:
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_linecut'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                a_tools.linecut_plot(x=self.sweep_points,
                                     y=self.sweep_points_2D,
                                     z=self.measured_values[i],
                                     plot_title=fig_title,
                                     log=linecut_log,
                                     xlabel=xlabel,
                                     x_unit=x_unit,
                                     y_name=ylabel,
                                     y_unit=y_unit,
                                     z_unit_linecuts=z_unit_linecuts,
                                     fig=fig, ax=ax, **kw)
                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            fig, ax = self.default_ax(figsize=(8, 5))
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            if normalize:
                meas_vals = a_tools.normalize_2D_data(meas_vals)
            fig_title = '{timestamp}_{measurement}_{val_name}'.format(
                timestamp=self.timestamp_string,
                measurement=self.measurementstring,
                val_name=self.zlabels[i])

            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=meas_vals.transpose(),
                               plot_title=fig_title,
                               xlabel=xlabel,
                               x_unit=x_unit,
                               ylabel=ylabel,
                               y_unit=y_unit,
                               fig=fig, ax=ax, **kw)
            if save_fig:
                self.save_fig(fig, figname=fig_title, **kw)

        if close_file:
            self.finish()


        # Find low power regime
        threshold = 0.1e6  # Gotta love hardcoded stuff
        f_low = f0[0]
        P_result = self.sweep_points_2D[0]
        try:
            for u, f in enumerate(f0):
                if np.abs(f0[0] - f0[u+1]) < threshold:
                    f_low = f0[u+1]
                    P_result = self.sweep_points_2D[u]
                else:
                    break
        except IndexError:
            pass

        # High power regime: just use the value at highest power
        f_high = f0[-1]

        shift = 0
        shift_data=[]
        for i in range(len(f0)):
            if (f0[0] - f0[-1]) > 200e3:
                shift = f0[0] - f0[-1]
            elif (f0[0] - f0[i]) > 200e3 :
                shifts= f0[0] - f0[i]
                shift_data.append(shifts)
                shift=np.amax(shift_data)
            else:
                logging.warning('No power shift found. Consider attenuation')

        # if (f_high < f_low):
        #     shift = f_high - f_low
        # else:
        #     shift = 0
        #     logging.warning('No power shift found. Consider attenuation')
        #     # raise Exception('High power regime frequency found to be higher than'
        #     #                 'low power regime frequency')

        self.f_low = f_low
        self.f_high = f_high
        self.shift = shift
        self.power = P_result

    def fit_hanger_model(self, sweep_values, measured_values):
        HangerModel = fit_mods.SlopedHangerAmplitudeModel

        # amplitude_guess = np.pi*sigma_guess * abs(
        #     max(self.measured_powers)-min(self.measured_powers))

        # Fit Power to a Lorentzian
        measured_powers = measured_values ** 2

        min_index = np.argmin(measured_powers)
        max_index = np.argmax(measured_powers)

        min_frequency = sweep_values[min_index]
        max_frequency = sweep_values[max_index]

        peaks = a_tools.peak_finder((sweep_values),
                                    measured_values)

        if peaks['dip'] is not None:  # look for dips first
            f0 = peaks['dip']
            amplitude_factor = -1.
        elif peaks['peak'] is not None:  # then look for peaks
            f0 = peaks['peak']
            amplitude_factor = 1.
        else:  # Otherwise take center of range
            f0 = np.median(sweep_values)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        amplitude_guess = max(measured_powers) - min(measured_powers)
        # Creating parameters and estimations
        S21min = min(measured_values) / max(measured_values)

        Q = f0 / abs(min_frequency - max_frequency)
        Qe = abs(Q / abs(1 - S21min))

        HangerModel.set_param_hint('f0', value=f0,
                                   min=min(sweep_values),
                                   max=max(sweep_values))
        HangerModel.set_param_hint('A', value=amplitude_guess)
        HangerModel.set_param_hint('Q', value=Q)
        HangerModel.set_param_hint('Qe', value=Qe)
        HangerModel.set_param_hint('Qi', expr='1./(1./Q-1./Qe*cos(theta))',
                                   vary=False)
        HangerModel.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
        HangerModel.set_param_hint('theta', value=0, min=-np.pi / 2,
                                   max=np.pi / 2)
        HangerModel.set_param_hint('slope', value=0, vary=True,
                                   min=-1, max=1)
        params = HangerModel.make_params()
        fit_res = HangerModel.fit(data=measured_powers,
                                  f=sweep_values * 1.e9,
                                  params=params)

        return fit_res



class time_trace_analysis(MeasurementAnalysis):
    '''
    Analysis for a binary (+1, -1) time trace
    returns the average length till flip
    '''

    def run_default_analysis(self, flipping_sequence=False, **kw):
        self.get_naming_and_values_2D()

        rsf_lst_mp = []
        rsf_lst_pm = []
        for i in range(np.shape(self.Z)[0]):
            series = self.Z[i, :]
            if flipping_sequence:
                series = dm_tools.binary_derivative_old(series)

            rsf = dm_tools.count_rounds_since_flip_split(series)
            rsf_lst_mp.extend(rsf[0])
            rsf_lst_pm.extend(rsf[1])

        # if self.make_fig:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(13, 6))
        bins = np.linspace(0, 400, 200)
        if flipping_sequence:
            self.average_cycles_flipping = np.mean(rsf_lst_mp)
            self.average_cycles_constant = np.mean(rsf_lst_pm)
            self.ax.hist(rsf_lst_mp, bins, histtype='step', normed=1,
                         label='Avg rounds flipping = %.2f' %
                               np.mean(rsf_lst_mp), color='b')
            self.ax.hist(rsf_lst_pm, bins, histtype='step', normed=1,
                         label='Avg rounds constant = %.2f'
                               % np.mean(rsf_lst_pm), color='r')
            self.ax.set_yscale('log')
            self.ax.set_ylabel('normalized occurence')
            self.ax.set_xlabel('rounds')
            self.ax.set_ylim(.000001, 1)
            self.ax.set_xlim(0, 80)
            self.ax.legend()
            self.ax.set_title(
                self.timestamp_string + '\n' + self.measurementstring)
            self.save_fig(self.fig, xlabel='rounds_flipping',
                          ylabel='normalized occurence', **kw)
            return self.average_cycles_constant, self.average_cycles_flipping
        else:
            self.mean_rnds_since_fl_mp = np.mean(rsf_lst_mp)
            self.mean_rnds_since_fl_pm = np.mean(rsf_lst_pm)
            self.ax.hist(rsf_lst_mp, bins, histtype='step', normed=1,
                         label='Avg rounds till flip -1 to +1 = %.2f' %
                               np.mean(rsf_lst_mp), color='b')
            self.ax.hist(rsf_lst_pm, bins, histtype='step', normed=1,
                         label='Avg rounds till flip +1 to -1 = %.2f'
                               % np.mean(rsf_lst_pm), color='r')
            self.ax.set_yscale('log')
            self.ax.set_ylabel('normalized occurence')
            self.ax.set_xlabel('rounds till flip')
            self.ax.set_ylim(.000001, 1)
            self.ax.set_xlim(0, 80)
            self.ax.legend()
            self.ax.set_title(
                self.timestamp_string + '\n' + self.measurementstring)
            self.save_fig(self.fig, xlabel='rounds_till_flip',
                          ylabel='normalized occurence', **kw)
            return self.mean_rnds_since_fl_pm, self.mean_rnds_since_fl_mp


class time_trace_analysis_initialized(MeasurementAnalysis):
    '''
    Analysis for a binary (+1, -1) time trace
    returns the average length till flip
    '''

    def run_default_analysis(self, flipping_sequence=False, **kw):
        self.get_naming_and_values_2D()
        if flipping_sequence:
            dZ = dm_tools.binary_derivative_2D(np.array(self.Z), axis=0)
            rtf = [dm_tools.count_rounds_to_error(ser) for ser in dZ]
        else:
            rtf = [dm_tools.count_rounds_to_error(ser) for ser in self.Z]
        self.mean_rtf = np.nanmean(rtf)
        self.std_rtf = np.nanstd(rtf)
        self.std_err_rtf = self.std_rtf / np.sqrt(len(self.sweep_points_2D))

        if kw.pop('make_fig', True):
            self.fig, self.ax = plt.subplots(1, 1, figsize=(13, 6))
            bins = np.arange(-.5, 400, 1)
            hist, bins = np.histogram(rtf, bins=bins, density=True)
            self.ax.plot(bins[1:], hist, drawstyle='steps',
                         label='Mean rounds till failure = %.2f'
                               % self.mean_rtf)
            self.ax.set_yscale('log')
            self.ax.set_ylabel('normalized occurence')
            self.ax.set_xlabel('Rounds to failure')
            self.ax.set_ylim(1e-4, 1)
            self.ax.set_xlim(0, 100)
            self.ax.legend()
            self.ax.set_title(self.timestamp_string + '\n' +
                              self.measurementstring)
            self.save_fig(self.fig, xlabel='Rounds to failure',
                          ylabel='normalized occurence', **kw)

        return self.mean_rtf, self.std_err_rtf


class rounds_to_failure_analysis(MeasurementAnalysis):
    '''
    Analysis for a binary (+1, -1) time trace
    returns the average rounds to surprise/failure.
    Additionally also returns the termination fractions.
    If the trace terminates by a 'single event' it counts a flip.
    If the trace terminates by a 'double event' it counts a RO error.
    '''

    def run_default_analysis(self, flipping_sequence=False, **kw):
        self.get_naming_and_values_2D()
        if flipping_sequence:
            dZ = dm_tools.binary_derivative_2D(np.array(self.Z), axis=0)
            rtf_c = [dm_tools.count_rtf_and_term_cond(
                ser, only_count_min_1=True) for ser in dZ]
        else:
            rtf_c = [dm_tools.count_rtf_and_term_cond(ser) for ser in self.Z]
        rtf, term_cond = list(zip(*rtf_c))
        self.mean_rtf = np.nanmean(rtf)
        self.std_rtf = np.nanstd(rtf)
        self.std_err_rtf = self.std_rtf / np.sqrt(len(self.sweep_points_2D))
        term_cts = Counter(term_cond)
        # note that we only take 1 derivative and this is not equal to the
        # notion of detection events as in Kelly et al.
        terminated_by_flip = float(term_cts['single event'])
        terminated_by_RO_err = float(term_cts['double event'])
        total_cts = terminated_by_RO_err + terminated_by_flip + \
            term_cts['unknown']
        self.flip_err_frac = terminated_by_flip / total_cts * 100.
        self.RO_err_frac = terminated_by_RO_err / total_cts * 100.

        if kw.pop('make_fig', True):
            self.fig, self.ax = plt.subplots(1, 1, figsize=(13, 6))
            bins = np.arange(-.5, 400, 1)
            hist, bins = np.histogram(rtf, bins=bins, density=True)
            label = ('Mean rounds to failure = %.2f' % self.mean_rtf +
                     '\n %.1f %% terminated by RO' % self.RO_err_frac +
                     '\n %.1f %% terminated by flip' % self.flip_err_frac)
            self.ax.plot(bins[1:], hist, drawstyle='steps',
                         label=label)
            self.ax.set_yscale('log')
            self.ax.set_ylabel('normalized occurence')
            self.ax.set_xlabel('Rounds to failure')
            self.ax.set_ylim(1e-4, 1)
            self.ax.set_xlim(0, 200)
            self.ax.legend()
            self.ax.set_title(self.timestamp_string + '\n' +
                              self.measurementstring)
            self.save_fig(self.fig, xlabel='Rounds to failure',
                          ylabel='normalized occurence', **kw)

        return self.mean_rtf, self.std_err_rtf, self.RO_err_frac, self.flip_err_frac


class butterfly_analysis(MeasurementAnalysis):
    '''
    Extracts the coefficients for the post-measurement butterfly
    '''

    def __init__(self, auto=True, label='Butterfly', close_file=True,
                 timestamp=None,
                 threshold=None,
                 threshold_init=None,
                 theta_in=0,
                 initialize=False,
                 digitize=True,
                 case=False,
                 # FIXME better variable name for 1>th or 1<th
                 **kw):
        self.folder = a_tools.get_folder(timestamp=timestamp,
                                         label=label, **kw)
        self.load_hdf5data(folder=self.folder, **kw)

        self.get_naming_and_values()

        if theta_in == 0:
            self.data = self.measured_values[0]
            if not digitize:
                # analysis uses +1 for |0> and -1 for |1>
                self.data[self.data == 1] = -1
                self.data[self.data == 0] = +1
        else:
            I_shots = self.measured_values[0]
            Q_shots = self.measured_values[1]

            shots = I_shots + 1j * Q_shots
            rot_shots = dm_tools.rotate_complex(
                shots, angle=theta_in, deg=True)
            I_shots = rot_shots.real
            Q_shots = rot_shots.imag

            self.data = I_shots
        self.initialize = initialize
        if self.initialize:
            if threshold_init is None:
                threshold_init = threshold

            # reshuffling the data to end up with two arrays for the
            # different input states
            shots = np.size(self.data)
            shots_per_mmt = np.floor_divide(shots, 6)
            shots_used = shots_per_mmt * 6
            m0_on = self.data[3:shots_used:6]
            m1_on = self.data[4:shots_used:6]
            m2_on = self.data[5:shots_used:6]

            self.data_rel = np.zeros([np.size(m0_on), 3])
            self.data_rel[:, 0] = m0_on
            self.data_rel[:, 1] = m1_on
            self.data_rel[:, 2] = m2_on
            m0_off = self.data[0:shots_used:6]
            m1_off = self.data[1:shots_used:6]
            m2_off = self.data[2:shots_used:6]
            self.data_exc = np.zeros([np.size(m0_off), 3])
            self.data_exc[:, 0] = m0_off
            self.data_exc[:, 1] = m1_off
            self.data_exc[:, 2] = m2_off

            self.data_exc_post = dm_tools.postselect(threshold=threshold_init,
                                                     data=self.data_exc,
                                                     positive_case=case)[:, 1:]
            self.data_rel_post = dm_tools.postselect(threshold=threshold_init,
                                                     data=self.data_rel,
                                                     positive_case=case)[:, 1:]

            self.data_exc_pre_postselect = self.data_exc
            self.data_rel_pre_postselect = self.data_rel
            # variable is overwritten here, no good.
            self.data_exc = self.data_exc_post
            self.data_rel = self.data_rel_post

            fraction = (np.size(self.data_exc) +
                        np.size(self.data_exc)) * 3 / shots_used / 2

        else:
            m0_on = self.data[2::4]
            m1_on = self.data[3::4]
            self.data_rel = np.zeros([np.size(m0_on), 2])
            self.data_rel[:, 0] = m0_on
            self.data_rel[:, 1] = m1_on
            m0_off = self.data[0::4]
            m1_off = self.data[1::4]
            self.data_exc = np.zeros([np.size(m0_off), 2])
            self.data_exc[:, 0] = m0_off
            self.data_exc[:, 1] = m1_off
        if digitize:
            self.data_exc = dm_tools.digitize(threshold=threshold,
                                              data=self.data_exc,
                                              one_larger_than_threshold=case)
            self.data_rel = dm_tools.digitize(threshold=threshold,
                                              data=self.data_rel,
                                              one_larger_than_threshold=case)
        if close_file:
            self.data_file.close()
        if auto is True:
            self.run_default_analysis(**kw)

    def bar_plot_raw_probabilities(self):
        if self.initialize:
            nr_msmts = 3
            data_exc = self.data_exc_pre_postselect
            data_rel = self.data_rel_pre_postselect
        else:
            data_exc = self.data_exc
            data_rel = self.data_rel
            nr_msmts = 2

        m_on = np.zeros(nr_msmts)
        m_off = np.zeros(nr_msmts)

        for i in range(nr_msmts):
            # Convert pauli eigenvalues to probability of excitation
            # +1 -> 0 and -1 -> 1
            m_off[i] = -(np.mean(data_exc[:, i]) - 1) / 2
            m_on[i] = -(np.mean(data_rel[:, i]) - 1) / 2

        f, ax = plt.subplots()
        ax.set_ylim(0, 1)
        w = .4
        ax.hlines(0.5, -.5, 5, linestyles='--')
        bar0 = ax.bar(np.arange(nr_msmts) + w / 2, m_off, width=w, color='C0',
                      label='No $\pi$-pulse')
        bar1 = ax.bar(np.arange(nr_msmts) - w / 2, m_on, width=w, color='C3',
                      label='$\pi$-pulse')
        pl_tools.autolabel_barplot(ax, bar0)
        pl_tools.autolabel_barplot(ax, bar1)

        ax.set_xlim(-.5, nr_msmts - .5)
        ax.set_xticks([0, 1, 2])
        set_ylabel(ax, 'P (|1>)')
        ax.legend()
        set_xlabel(ax, 'Measurement idx')
        figname = 'Bar plot raw probabilities'
        ax.set_title(figname)

        savename = os.path.abspath(os.path.join(
            self.folder, figname + '.png'))
        print(savename)
        f.savefig(savename, dpi=300, format='png')

    def run_default_analysis(self, verbose=False, **kw):
        self.exc_coeffs = dm_tools.butterfly_data_binning(Z=self.data_exc,
                                                          initial_state=0)
        self.rel_coeffs = dm_tools.butterfly_data_binning(Z=self.data_rel,
                                                          initial_state=1)
        self.butterfly_coeffs = dm_tools.butterfly_matrix_inversion(
            self.exc_coeffs, self.rel_coeffs)
        # eps,declaration,output_input
        F_a_butterfly = (1 - (self.butterfly_coeffs.get('eps00_1') +
                              self.butterfly_coeffs.get('eps01_1') +
                              self.butterfly_coeffs.get('eps10_0') +
                              self.butterfly_coeffs.get('eps11_0')) / 2)

        mmt_ind_rel = (self.butterfly_coeffs.get('eps00_1') +
                       self.butterfly_coeffs.get('eps10_1'))
        mmt_ind_exc = (self.butterfly_coeffs.get('eps11_0') +
                       self.butterfly_coeffs.get('eps01_0'))
        if verbose:
            print('SSRO Fid', F_a_butterfly)
            print('mmt_ind_rel', mmt_ind_rel)
            print('mmt_ind_exc', mmt_ind_exc)
        self.butterfly_coeffs['F_a_butterfly'] = F_a_butterfly
        self.butterfly_coeffs['mmt_ind_exc'] = mmt_ind_exc
        self.butterfly_coeffs['mmt_ind_rel'] = mmt_ind_rel
        self.bar_plot_raw_probabilities()
        self.make_data_tables()

        return self.butterfly_coeffs

    def make_data_tables(self):

        figname1 = 'raw probabilities'

        data_raw_p = [['P(1st m, 2nd m)_(|in>)', 'val'],
                      ['P00_0', '{:.4f}'.format(self.exc_coeffs['P00_0'])],
                      ['P01_0', '{:.4f}'.format(self.exc_coeffs['P01_0'])],
                      ['P10_0', '{:.4f}'.format(self.exc_coeffs['P10_0'])],
                      ['P11_0', '{:.4f}'.format(self.exc_coeffs['P11_0'])],
                      ['P00_1', '{:.4f}'.format(self.rel_coeffs['P00_1'])],
                      ['P01_1', '{:.4f}'.format(self.rel_coeffs['P01_1'])],
                      ['P10_1', '{:.4f}'.format(self.rel_coeffs['P10_1'])],
                      ['P11_1', '{:.4f}'.format(self.rel_coeffs['P11_1'])]]

        savename = os.path.abspath(os.path.join(
            self.folder, figname1))
        data_to_table_png(data=data_raw_p, filename=savename + '.png',
                          title=figname1)

        figname2 = 'inferred states'

        data_inf = [['eps(|out>)_(|in>)', 'val'],
                    ['eps0_0', '{:.4f}'.format(self.exc_coeffs['eps0_0'])],
                    ['eps1_0', '{:.4f}'.format(self.exc_coeffs['eps1_0'])],
                    ['eps0_1', '{:.4f}'.format(self.rel_coeffs['eps0_1'])],
                    ['eps1_1', '{:.4f}'.format(self.rel_coeffs['eps1_1'])]]
        savename = os.path.abspath(os.path.join(
            self.folder, figname2))
        data_to_table_png(data=data_inf, filename=savename + '.png',
                          title=figname2)

        bf = self.butterfly_coeffs
        figname3 = 'Butterfly coefficients'
        data = [['eps(declared, |out>)_(|in>)', 'val'],
                ['eps00_0', '{:.4f}'.format(bf['eps00_0'])],
                ['eps01_0', '{:.4f}'.format(bf['eps01_0'])],
                ['eps10_0', '{:.4f}'.format(bf['eps10_0'])],
                ['eps11_0', '{:.4f}'.format(bf['eps11_0'])],
                ['eps00_1', '{:.4f}'.format(bf['eps00_1'])],
                ['eps01_1', '{:.4f}'.format(bf['eps01_1'])],
                ['eps10_1', '{:.4f}'.format(bf['eps10_1'])],
                ['eps11_1', '{:.4f}'.format(bf['eps11_1'])]]
        savename = os.path.abspath(os.path.join(
            self.folder, figname3))
        data_to_table_png(data=data, filename=savename + '.png',
                          title=figname3)

        figname4 = 'Derived quantities'
        data = [['Measurement induced excitations',
                 '{:.4f}'.format(bf['mmt_ind_exc'])],
                ['Measurement induced relaxation',
                 '{:.4f}'.format(bf['mmt_ind_rel'])],
                ['Readout fidelity',
                 '{:.4f}'.format(bf['F_a_butterfly'])]]
        savename = os.path.abspath(os.path.join(
            self.folder, figname4))
        data_to_table_png(data=data, filename=savename + '.png',
                          title=figname4)


##########################################
### Analysis for data measurement sets ###
##########################################


def fit_qubit_frequency(sweep_points, data, mode='dac',
                        vary_E_c=True, vary_f_max=True,
                        vary_dac_flux_coeff=True,
                        vary_dac_sweet_spot=True,
                        data_file=None, **kw):
    '''
    Function for fitting the qubit dac/flux arc
    Has default values for all the fit parameters, if specied as a **kw it uses
    that value. If a qubit name and a hdf5 data file is specified it uses
    values from the data_file.
    NB! This function could be cleaned up a bit.

    :param sweep_points:
    :param data:
    :param mode:
    :param vary_E_c:
    :param vary_f_max:
    :param vary_dac_flux_coeff:
    :param vary_dac_sweet_spot:
    :param data_file:
    :param kw:
    :return:
    '''

    qubit_name = kw.pop('qubit_name', None)
    if qubit_name is not None and data_file is not None:
        try:
            instrument_settings = data_file['Instrument settings']
            qubit_attrs = instrument_settings[qubit_name].attrs
            print(qubit_attrs)
        except:
            print('Qubit instrument is not in data file')
            qubit_attrs = {}
    else:
        qubit_attrs = {}

    # Extract initial values, first from kw, then data file, then default
    E_c = kw.pop('E_c', qubit_attrs.get('E_c', 0.3e9))
    f_max = kw.pop('f_max', qubit_attrs.get('f_max', np.max(data)))
    dac_flux_coeff = kw.pop('dac_flux_coefficient',
                            qubit_attrs.get('dac_flux_coefficient', 1.))
    dac_sweet_spot = kw.pop('dac_sweet_spot',
                            qubit_attrs.get('dac_sweet_spot', 0))
    flux_zero = kw.pop('flux_zero', qubit_attrs.get('flux_zero', 10))

    if mode == 'dac':
        Q_dac_freq_mod = fit_mods.QubitFreqDacModel
        Q_dac_freq_mod.set_param_hint('E_c', value=E_c, vary=vary_E_c,
                                      min=0, max=500e6)
        Q_dac_freq_mod.set_param_hint('f_max', value=f_max,
                                      vary=vary_f_max)
        Q_dac_freq_mod.set_param_hint('dac_flux_coefficient',
                                      value=dac_flux_coeff,
                                      vary=vary_dac_flux_coeff)
        Q_dac_freq_mod.set_param_hint('dac_sweet_spot',
                                      value=dac_sweet_spot,
                                      vary=vary_dac_sweet_spot)

        fit_res = Q_dac_freq_mod.fit(data=data, dac_voltage=sweep_points)
    elif mode == 'flux':
        Qubit_freq_mod = fit_mods.QubitFreqFluxModel
        Qubit_freq_mod.set_param_hint('E_c', value=E_c, vary=vary_E_c,
                                      min=0, max=100e6)
        Qubit_freq_mod.set_param_hint('f_max', value=f_max, vary=vary_f_max)
        Qubit_freq_mod.set_param_hint('flux_zero', value=flux_zero,
                                      min=0, vary=True)
        Qubit_freq_mod.set_param_hint('dac_offset', value=0, vary=True)

        fit_res = Qubit_freq_mod.fit(data=data, flux=sweep_points)
    return fit_res


# Ramiro's routines


class Chevron_2D(object):

    def __init__(self, auto=True, label='', timestamp=None):
        if timestamp is None:
            self.folder = a_tools.latest_data('Chevron')
            splitted = self.folder.split('\\')
            self.scan_start = splitted[-2] + '_' + splitted[-1][:6]
            self.scan_stop = self.scan_start
        else:
            self.scan_start = timestamp
            self.scan_stop = timestamp
            self.folder = a_tools.get_folder(timestamp=self.scan_start)
        self.pdict = {'I': 'amp',
                      'sweep_points': 'sweep_points'}
        self.opt_dict = {'scan_label': 'Chevron_2D'}
        self.nparams = ['I', 'sweep_points']
        self.label = label
        if auto == True:
            self.analysis()

    def analysis(self):
        chevron_scan = ca.quick_analysis(t_start=self.scan_start,
                                         t_stop=self.scan_stop,
                                         options_dict=self.opt_dict,
                                         params_dict_TD=self.pdict,
                                         numeric_params=self.nparams)
        x, y, z = self.reshape_data(chevron_scan.TD_dict['sweep_points'][0],
                                    chevron_scan.TD_dict['I'][0])
        plot_times = y
        plot_step = plot_times[1] - plot_times[0]

        plot_x = x
        x_step = plot_x[1] - plot_x[0]

        result = z

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        cmin, cmax = 0, 1
        fig_clim = [cmin, cmax]
        out = pl_tools.flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
                                                 xvals=plot_times,
                                                 yvals=plot_x,
                                                 zvals=result)
        ax.set_xlabel(r'AWG Amp (Vpp)')
        ax.set_ylabel(r'Time (ns)')
        ax.set_title('%s: Chevron scan' % self.scan_start)
        # ax.set_xlim(xmin, xmax)
        ax.set_ylim(plot_x.min() - x_step / 2., plot_x.max() + x_step / 2.)
        ax.set_xlim(
            plot_times.min() - plot_step / 2., plot_times.max() + plot_step / 2.)
        #     ax.set_xlim(plot_times.min()-plot_step/2.,plot_times.max()+plot_step/2.)
        # ax.set_xlim(0,50)
        #     print('Bounce %d ns amp=%.3f; Pole %d ns amp=%.3f'%(list_values[iter_idx,0],
        #                                                                list_values[iter_idx,1],
        #                                                                list_values[iter_idx,2],
        # list_values[iter_idx,3]))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('right', size='10%', pad='5%')
        cbar = plt.colorbar(out['cmap'], cax=cax)
        cbar.set_ticks(
            np.arange(fig_clim[0], 1.01 * fig_clim[1], (fig_clim[1] - fig_clim[0]) / 5.))
        cbar.set_ticklabels(
            [str(fig_clim[0]), '', '', '', '', str(fig_clim[1])])
        cbar.set_label('Qubit excitation probability')

        fig.tight_layout()
        self.save_fig(fig)

    def reshape_axis_2d(self, axis_array):
        x = axis_array[0, :]
        y = axis_array[1, :]
        # print(y)
        dimx = np.sum(np.where(x == x[0], 1, 0))
        dimy = len(x) // dimx
        # print(dimx,dimy)
        if dimy * dimx < len(x):
            logging.warning.warn(
                'Data was cut-off. Probably due to an interrupted scan')
            dimy_c = dimy + 1
        else:
            dimy_c = dimy
        # print(dimx,dimy,dimy_c,dimx*dimy)
        return x[:dimy_c], (y[::dimy_c])

    def reshape_data(self, sweep_points, data):
        x, y = self.reshape_axis_2d(sweep_points)
        # print(x,y)
        dimx = len(x)
        dimy = len(y)
        dim = dimx * dimy
        if dim > len(data):
            dimy = dimy - 1
        return x, y[:dimy], (data[:dimx * dimy].reshape((dimy, dimx))).transpose()

    def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            if figname is None:
                figname = (self.scan_start +
                           '_Chevron_2D_' + '.' + plot_format)
            else:
                figname = (figname + '.' + plot_format)
            self.savename = os.path.abspath(os.path.join(
                self.folder, figname))
            if fig_tight:
                try:
                    fig.tight_layout()
                except ValueError:
                    print('WARNING: Could not set tight layout')
            try:
                fig.savefig(
                    self.savename, dpi=300,
                    # value of 300 is arbitrary but higher than default
                    format=plot_format)
            except:
                fail_counter = True
        if fail_counter:
            logging.warning('Figure "%s" has not been saved.' % self.savename)
        if close_fig:
            plt.close(fig)
        return


class DoubleFrequency(TD_Analysis):

    def __init__(self, auto=True, label='Ramsey', timestamp=None, **kw):
        kw['label'] = label
        kw['auto'] = auto
        kw['timestamp'] = timestamp
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self,close_file=False, **kw):
        super().run_default_analysis(
            close_file=close_file,
            close_main_figure=True, save_fig=True, **kw)

        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        x = self.sweep_points
        #y1 are the correct TwoD normalized points
        #y2 is the 1D normalized points, so worse fit
        y1 = self.normalized_data_points
        y2 = a_tools.normalize_data_v3(self.measured_values[0])

        y=y2
        #TODO: implement prony's method and see if it's better
        y[:-4] = y1
        fit_res = self.fit(x[:-4], y[:-4])
        self.fit_res = fit_res

        self.save_fitted_parameters(self.fit_res, var_name='double_fit')

        fig, ax = plt.subplots()
        self.box_props = dict(boxstyle='Square', facecolor='white', alpha=0.8)

        fs = [fit_res.params['freq_1'].value, fit_res.params['freq_2'].value]
        As = [fit_res.params['amp_1'].value, fit_res.params['amp_2'].value]
        taus = [fit_res.params['tau_1'].value, fit_res.params['tau_2'].value]
        min_index=fs.index(np.min(fs))
        max_index=fs.index(np.max(fs))
        self.f1=fs[min_index]
        self.f2=fs[max_index]
        self.A1=As[min_index]
        self.A2=As[max_index]
        self.tau1=taus[min_index]
        self.tau2=taus[max_index]


        textstr = ('$A_1$: {:.3f}       \t$A_2$: {:.3f} \n'.format(self.A1, self.A2) +
                   '$f_1$: {:.3f} MHz\t$f_2$: {:.3f} MHz \n'.format(
                       self.f1 * 1e-6, self.f2 * 1e-6) +
                   r'$\tau _1$: {:.2f} $\mu$s'.format(self.tau1 * 1e6) +
                   '  \t' + r'$\tau _2$: {:.2f}$\mu$s'.format(self.tau2 * 1e6))

        self.ax.text(0.4, 0.95, textstr,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=self.box_props)
        plot_x = np.linspace(self.sweep_points[0],
                        self.sweep_points[-self.NoCalPoints - 1],
                        len(self.sweep_points) * 100)

        best_vals = fit_res.best_values
        plot_y = fit_mods.DoubleExpDampOscFunc(
            plot_x, **best_vals)

        self.ax.set_title('%s: Double Frequency analysis' % self.timestamp)
        self.ax.plot(plot_x, plot_y, '-')
        self.fig.tight_layout()
        self.save_fig(self.fig, **kw)
        self.data_file.close()
        return self.fit_res

    def fit(self, sweep_values, measured_values):
        Double_Cos_Model = fit_mods.DoubleExpDampOscModel
        dt= sweep_values[2]-sweep_values[1]
        zero_mean_values = measured_values-np.mean(measured_values)
        fourier_max_pos = a_tools.peak_finder_v2(
            np.arange(1, len(sweep_values) / 2, 1),
            abs(np.fft.fft(zero_mean_values))[1:len(zero_mean_values) // 2],
            window_len=1, perc=95)
        if (len(fourier_max_pos)==0):
            print('No strong peak found, trying again')
            fourier_max_pos = a_tools.peak_finder_v2(
                np.arange(1, len(sweep_values) / 2, 1),
                abs(np.fft.fft(zero_mean_values))[1:len(zero_mean_values) // 2],
                window_len=1, perc=75)

        # if fourier_max_pos was one the above statement mocks it.
        if len(fourier_max_pos) == 1: #One peak found
            fmin = 1./sweep_values[-1]*\
                (fourier_max_pos[0] - 10)
            fmax = 1./sweep_values[-1]*\
                (fourier_max_pos[0] + 10)
        else:
            fourier_max_pos = fourier_max_pos[0:2]
            fmin = 1./sweep_values[-1]*\
                (np.min(fourier_max_pos) - 10)
            fmax = 1./sweep_values[-1]*\
                (np.max(fourier_max_pos) + 10)
        #Do a ZoomFFT
        if (fmin<0):
            fmin = 0
        [chirp_x, chirp_y] = a_tools.zoom_fft(sweep_values,zero_mean_values,
                                              fmin,fmax)
        fourier_max_pos = a_tools.peak_finder_v2(
            np.arange(0,len(chirp_x)),
            np.abs(chirp_y),
            window_len=1, perc=85)
        #Now do Bertocco's algorithm
        #From [Metrology and Measurement Systems] Frequency and Damping Estimation Methods - An Overview.pdf
        only_one_peak = False
        if (len(fourier_max_pos)==1): #If there is still only one peak
            print('Only one strong frequency found: not a Double Frequency?')
            only_one_peak = True
        else:
            fourier_max_pos = fourier_max_pos[0:2]

        freq_guess = chirp_x[fourier_max_pos]
        n_shift =min(6,len(chirp_y)-1-max(fourier_max_pos))

        Ratio = chirp_y[fourier_max_pos]/chirp_y[fourier_max_pos+n_shift]
        Omega_freq = 2*np.pi*dt*freq_guess
        dOmega_freq = 2*np.pi*dt*(chirp_x[fourier_max_pos + n_shift]-chirp_x[fourier_max_pos])
        expvalue_res = np.exp(1j*Omega_freq)*(Ratio-1)/(Ratio*np.exp(-1j*dOmega_freq)-1)
        #    freq_guess= np.imag(np.log(lambda_result))/(2*np.pi*dt)
        tau_guess= dt/np.real(np.log(expvalue_res))
        # Now get A and phi from a leastsqrs fit since we know f and tau
        # See article above for more information

        while (any(np.array(tau_guess)<0) and n_shift>=2):
            if (n_shift>=3):
                n_shift -=2
            else:
                n_shift -=1
            Ratio = chirp_y[fourier_max_pos]/chirp_y[fourier_max_pos+n_shift]
            Omega_freq = 2*np.pi*dt*freq_guess
            dOmega_freq = 2*np.pi*dt*(chirp_x[fourier_max_pos + n_shift]-chirp_x[fourier_max_pos])
            expvalue_res = np.exp(1j*Omega_freq)*(Ratio-1)/(Ratio*np.exp(-1j*dOmega_freq)-1)
            #    freq_guess= np.imag(np.log(lambda_result))/(2*np.pi*dt)
            tau_guess= dt/np.real(np.log(expvalue_res))
        if (only_one_peak):
            expvals = np.array([2j*np.pi*freq_guess[0] - 1/tau_guess[0],-2j*np.pi*freq_guess[0] - 1/tau_guess[0], 0.])
            E = np.zeros((len(measured_values),3),dtype=complex)
            for ii in range(len(measured_values)):
                for jj in range(3):
                    E[ii,:]=np.exp(expvals*sweep_values[ii])
            # FIXME: when using numpy >1.14 we should change to rcond=None
            coeff = np.linalg.lstsq(E, measured_values, rcond=-1)[0]
            amp_guess = 2*np.abs(coeff[[0]])
            phi_guess = np.angle(coeff[[0]])
        else:
            expvals = np.array([2j*np.pi*freq_guess[0] - 1/tau_guess[0],-2j*np.pi*freq_guess[0] - 1/tau_guess[0],
               2j*np.pi*freq_guess[1] - 1/tau_guess[1],-2j*np.pi*freq_guess[1] - 1/tau_guess[1],0.])
            E = np.zeros((len(measured_values),5),dtype=complex)
            for ii in range(len(measured_values)):
                for jj in range(5):
                    E[ii,:]=np.exp(expvals*sweep_values[ii])
            # FIXME: when using numpy >1.14 we should change to rcond=None
            coeff = np.linalg.lstsq(E, measured_values, rcond=-1)[0]
            amp_guess = 2*np.abs(coeff[[0,2]])
            phi_guess = np.angle(coeff[[0,2]])

        Double_Cos_Model.set_param_hint(
            'tau_1', value=tau_guess[0], vary=True, min=0, max=6*tau_guess[0])
        Double_Cos_Model.set_param_hint(
            'freq_1', value=freq_guess[0], min=0)
        Double_Cos_Model.set_param_hint('phase_1', value=phi_guess[0])
        Double_Cos_Model.set_param_hint('osc_offset', value=np.mean(measured_values), min=0, max=1)
        if (only_one_peak):
            Double_Cos_Model.set_param_hint(
                'tau_2', value=0, vary=False, min=0)
            Double_Cos_Model.set_param_hint(
                'freq_2', value=0, vary=False)
            Double_Cos_Model.set_param_hint('phase_2', value=0, vary=False)
            Double_Cos_Model.set_param_hint(
                'amp_1', value=amp_guess[0], min=0.05, max=0.8, vary=True)
            Double_Cos_Model.set_param_hint(
                'amp_2', value=0, vary=False)

        else:
            Double_Cos_Model.set_param_hint(
                'tau_2', value=tau_guess[1], vary=True, min=0, max=6*tau_guess[1])
            Double_Cos_Model.set_param_hint(
                'freq_2', value=freq_guess[1], min=0)
            Double_Cos_Model.set_param_hint('phase_2', value=phi_guess[1])
            Double_Cos_Model.set_param_hint(
                'amp_1', value=amp_guess[0], min=0.05, max=2*amp_guess[0], vary=True)
            Double_Cos_Model.set_param_hint(
                'amp_2', value=amp_guess[1], min=0.05, max=2*amp_guess[1], vary=True)

        params = Double_Cos_Model.make_params()
        fit_res = Double_Cos_Model.fit(data=measured_values,
                                       t=sweep_values,
                                       params=params)
        return fit_res

    def save_fig(self, fig, figname='_DoubleFreq_', xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            figname = (self.timestamp_string + figname + '.' + plot_format)
            self.savename = os.path.abspath(os.path.join(
                self.folder, figname))
            if fig_tight:
                try:
                    fig.tight_layout()
                except ValueError:
                    print('WARNING: Could not set tight layout')
            try:
                fig.savefig(
                    self.savename, dpi=300,
                    # value of 300 is arbitrary but higher than default
                    format=plot_format)
            except Exception:
                fail_counter = True
                print('Could not save to '+str(self.savename))
        if fail_counter:
            logging.warning('Figure "%s" has not been saved.' % self.savename)
        if close_fig:
            plt.close(fig)
        return


class SWAPN_cost(object):

    def __init__(self, auto=True, label='SWAPN', cost_func='sum', timestamp=None, stepsize=10):
        if timestamp is None:
            self.folder = a_tools.latest_data(label)
            splitted = self.folder.split('\\')
            self.scan_start = splitted[-2] + '_' + splitted[-1][:6]
            self.scan_stop = self.scan_start
        else:
            self.scan_start = timestamp
            self.scan_stop = timestamp
            self.folder = a_tools.get_folder(timestamp=self.scan_start)
        self.pdict = {'I': 'amp',
                      'sweep_points': 'sweep_points'}
        self.opt_dict = {'scan_label': label}
        self.nparams = ['I', 'sweep_points']
        self.stepsize = stepsize
        self.label = label
        self.cost_func = cost_func
        if auto == True:
            self.analysis()

    def analysis(self):
        print(self.scan_start, self.scan_stop,
              self.opt_dict, self.pdict, self.nparams)
        sawpn_scan = ca.quick_analysis(t_start=self.scan_start,
                                       t_stop=self.scan_stop,
                                       options_dict=self.opt_dict,
                                       params_dict_TD=self.pdict,
                                       numeric_params=self.nparams)
        x = sawpn_scan.TD_dict['sweep_points'][0]
        y = sawpn_scan.TD_dict['I'][0]

        if self.cost_func == 'sum':
            self.cost_val = np.sum(
                np.power(y[:-4], np.divide(1, x[:-4]))) / float(len(y[:-4]))
        elif self.cost_func == 'slope':
            self.cost_val = abs(y[0] * (y[1] - y[0])) + abs(y[0])
        elif self.cost_func == 'dumb-sum':
            self.cost_val = (np.sum(y[:-4]) /
                             float(len(y[:-4]))) - y[:-4].min()
        elif self.cost_func == 'until-nonmono-sum':
            i = 0
            y_fil = deepcopy(y)
            lastval = y_fil[0]
            keep_going = 1
            while (keep_going):
                if i > 5:
                    latestthreevals = (
                        y_fil[i] + y_fil[i - 1] + y_fil[i - 2]) / 3
                    threevalsbefore = (
                        y_fil[i - 3] + y_fil[i - 4] + y_fil[i - 5]) / 3
                    if latestthreevals < (threevalsbefore - 0.12) or i > len(y_fil) - 4:
                        keep_going = 0
                i += 1
            y_fil[i - 1:-4] = threevalsbefore
            self.cost_val = (np.sum(y_fil[:-4]) / float(len(y_fil[:-4])))
        self.single_swap_fid = y[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plot_x = x
        plot_step = plot_x[1] - plot_x[0]

        ax.set_xlabel(r'# Swap pulses')
        ax.set_ylabel(r'$F |1\rangle$')
        ax.set_title('%s: SWAPN sequence' % self.scan_start)
        ax.set_xlim(plot_x.min() - plot_step / 2.,
                    plot_x.max() + plot_step / 2.)

        ax.plot(plot_x, y, 'bo')

        fig.tight_layout()
        self.save_fig(fig)

    def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            if figname is None:
                figname = (self.scan_start +
                           '_DoubleFreq_' + '.' + plot_format)
            else:
                figname = (figname + '.' + plot_format)
            self.savename = os.path.abspath(os.path.join(
                self.folder, figname))

            if fig_tight:
                try:
                    fig.tight_layout()
                except ValueError:
                    print('WARNING: Could not set tight layout')
            try:
                fig.savefig(
                    self.savename, dpi=300,
                    # value of 300 is arbitrary but higher than default
                    format=plot_format)
            except:
                fail_counter = True
        if fail_counter:
            logging.warning('Figure "%s" has not been saved.' % self.savename)
        if close_fig:
            plt.close(fig)
        return


class AvoidedCrossingAnalysis(MeasurementAnalysis):
    """
    Performs analysis to fit the avoided crossing
    """

    def __init__(self, auto=True,
                 model='direct_coupling',
                 label=None,
                 timestamp=None,
                 transpose=True,
                 cmap='viridis',
                 filt_func_a=None, filt_func_x0=None, filt_func_y0=None,
                 filter_idx_low=[], filter_idx_high=[], filter_threshold=15e6,
                 force_keep_idx_low=[], force_keep_idx_high=[],
                 f1_guess=None, f2_guess=None, cross_flux_guess=None,
                 g_guess=30e6, coupling_label=r'$J_1/2\pi$',
                 break_before_fitting=False,
                 add_title=True,
                 xlabel=None, ylabel='Frequency (GHz)',
                 weight_function_magn=0,
                 use_distance=True,
                 quadratures=None,
                 blur=None,
                 **kw):
        super().__init__(timestamp=timestamp, label=label, **kw)
        self.get_naming_and_values_2D()
        if quadratures is not None:
            real = np.transpose(self.measured_values[quadratures[0]])
            imag = np.transpose(self.measured_values[quadratures[1]])
        else:
            measured_magns = np.transpose(self.measured_values[weight_function_magn])
            measured_phases = np.transpose(self.measured_values[1+weight_function_magn])
            rad = [(i * np.pi/180) for i in measured_phases]
            real = [measured_magns[j] * np.cos(i) for j, i in enumerate(rad)]
            imag = [measured_magns[j] * np.sin(i) for j, i in enumerate(rad)]
        dists = [a_tools.calculate_distance_ground_state(real[i],imag[i], normalize=True) for i in range(len(real))]

        self.S21dist = dists
        if use_distance:
            self.Z[0]=np.array(self.S21dist)
        if blur is not None:
            self.Z[0] = gaussian_filter(self.Z[0], blur)
        flux = self.Y[:, 0]
        self.make_raw_figure(transpose=transpose, cmap=cmap,
                             add_title=add_title,
                             xlabel=xlabel, ylabel=ylabel)

        self.peaks_low, self.peaks_high = self.find_peaks(**kw)
        self.f, self.ax = self.make_unfiltered_figure(self.peaks_low, self.peaks_high,
                                                      transpose=transpose, cmap=cmap,
                                                      add_title=add_title,
                                                      xlabel=xlabel, ylabel=ylabel)

        self.filtered_dat = self.filter_data(flux, self.peaks_low, self.peaks_high,
                                             a=filt_func_a, x0=filt_func_x0,
                                             y0=filt_func_y0,
                                             filter_idx_low=filter_idx_low,
                                             filter_idx_high=filter_idx_high,
                                             force_keep_idx_low=force_keep_idx_low,
                                             force_keep_idx_high=force_keep_idx_high,
                                             filter_threshold=filter_threshold)
        filt_flux_low, filt_flux_high, filt_peaks_low, filt_peaks_high, \
            filter_func = self.filtered_dat

        self.f, self.ax = self.make_filtered_figure(filt_flux_low, filt_flux_high,
                                                    filt_peaks_low, filt_peaks_high, filter_func,
                                                    add_title=add_title,
                                                    transpose=transpose, cmap=cmap,
                                                    xlabel=xlabel, ylabel=ylabel)
        if break_before_fitting:
            return
        self.fit_res = self.fit_avoided_crossing(
            filt_flux_low, filt_flux_high, filt_peaks_low, filt_peaks_high,
            f1_guess=f1_guess, f2_guess=f2_guess,
            cross_flux_guess=cross_flux_guess, g_guess=g_guess,
            model=model)
        self.add_analysis_datagroup_to_file()
        self.save_fitted_parameters(self.fit_res, var_name='avoided crossing')
        self.f, self.ax = self.make_fit_figure(filt_flux_low, filt_flux_high,
                                               filt_peaks_low, filt_peaks_high,
                                               add_title=add_title,
                                               fit_res=self.fit_res,
                                               coupling_label=coupling_label,
                                               transpose=transpose, cmap=cmap,
                                               xlabel=xlabel, ylabel=ylabel)

    def run_default_analysis(self, **kw):
        # I'm doing this in the init in this function
        pass

    def find_peaks(self, **kw):

        peaks = np.zeros((len(self.X), 2))
        for i in range(len(self.X)):
            p_dict = a_tools.peak_finder_v2(self.X[i], self.Z[0][i], **kw)
            try:
                peaks[i, :] = np.sort(p_dict[:2])
            except Exception as e:
                logging.warning(e)
                peaks[i, :] = np.array([np.NaN, np.NaN])

        peaks_low = peaks[:, 0]
        peaks_high = peaks[:, 1]
        return peaks_low, peaks_high

    def filter_data(self, flux, peaks_low, peaks_high, a, x0=None, y0=None,
                    filter_idx_low=[], filter_idx_high=[],
                    force_keep_idx_low=[], force_keep_idx_high=[],
                    filter_threshold=15e5):
        """
        Filters the input data in three steps.
            1. remove outliers using the dm_tools.get_outliers function
            2. separate data in two branches using a line and filter data on the
                wrong side of the line.
            3. remove any data with indices specified by hand
            4. Keep any data with indeces specified by hand (will overwrite removal)
        """

        if a is None:
            a = -1 * (max(peaks_high) - min(peaks_low)) / \
                (max(flux) - min(flux))
        if x0 is None:
            x0 = np.mean(flux)
        if y0 is None:
            y0 = np.mean(np.concatenate([peaks_low, peaks_high]))

        def filter_func(x): return a * (x - x0) + y0

        filter_mask_high = [True] * len(peaks_high)
        filter_mask_high = ~dm_tools.get_outliers(peaks_high, filter_threshold)
        filter_mask_high = np.where(
            peaks_high < filter_func(flux), False, filter_mask_high)
        filter_mask_high[filter_idx_high] = False  # hand remove 1 datapoint
        filter_mask_high[force_keep_idx_high] = True

        filt_flux_high = flux[filter_mask_high]
        filt_peaks_high = peaks_high[filter_mask_high]

        filter_mask_low = [True] * len(peaks_low)
        filter_mask_low = ~dm_tools.get_outliers(peaks_low, filter_threshold)
        filter_mask_low = np.where(
            peaks_low > filter_func(flux), False, filter_mask_low)
        filter_mask_low[filter_idx_low] = False  # hand remove 2 datapoints
        filter_mask_low[force_keep_idx_low] = True

        filt_flux_low = flux[filter_mask_low]
        filt_peaks_low = peaks_low[filter_mask_low]

        return (filt_flux_low, filt_flux_high,
                filt_peaks_low, filt_peaks_high, filter_func)

    def make_raw_figure(self,  transpose, cmap,
                        xlabel=None, ylabel='Frequency (GHz)',
                        add_title=True):
        flux = self.Y[:, 0]
        title = ' raw data avoided crossing'
        f, ax = plt.subplots()
        if add_title:
            ax.set_title(self.timestamp_string + title)

        pl_tools.flex_colormesh_plot_vs_xy(self.X[0] * 1e-9, flux, self.Z[0],
                                           ax=ax, transpose=transpose,
                                           cmap=cmap)

        # self.ylabel because the axes are transposed
        xlabel = self.ylabel if xlabel is None else xlabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(self.X[0] * 1e-9), max(self.X[0] * 1e-9))
        ax.set_xlim(min(flux), max(flux))
        f.savefig(os.path.join(self.folder, title + '.png'), format='png',
                  dpi=600)
        return f, ax

    def make_unfiltered_figure(self, peaks_low, peaks_high, transpose, cmap,
                               xlabel=None, ylabel='Frequency (GHz)',
                               add_title=True):
        flux = self.Y[:, 0]
        title = ' unfiltered avoided crossing'
        f, ax = plt.subplots()
        if add_title:
            ax.set_title(self.timestamp_string + title)

        pl_tools.flex_colormesh_plot_vs_xy(self.X[0] * 1e-9, flux, self.Z[0],
                                           ax=ax, transpose=transpose,
                                           cmap=cmap)
        ax.plot(flux, peaks_high * 1e-9, 'o', markeredgewidth=1.,
                fillstyle='none', c='r')
        ax.plot(flux, peaks_low * 1e-9, 'o', markeredgewidth=1.,
                fillstyle='none', c='orange')

        # self.ylabel because the axes are transposed
        xlabel = self.ylabel if xlabel is None else xlabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(self.X[0] * 1e-9), max(self.X[0] * 1e-9))
        ax.set_xlim(min(flux), max(flux))
        f.savefig(os.path.join(self.folder, title + '.png'), format='png',
                  dpi=600)
        return f, ax

    def make_filtered_figure(self,
                             filt_flux_low, filt_flux_high,
                             filt_peaks_low, filt_peaks_high, filter_func,
                             transpose, cmap,
                             xlabel=None, ylabel='Frequency (GHz)',
                             add_title=True):
        flux = self.Y[:, 0]
        title = ' filtered avoided crossing'
        f, ax = plt.subplots()
        if add_title:
            ax.set_title(self.timestamp_string + title)

        pl_tools.flex_colormesh_plot_vs_xy(self.X[0] * 1e-9, flux, self.Z[0],
                                           ax=ax, transpose=transpose,
                                           cmap=cmap)
        ax.plot(filt_flux_high, filt_peaks_high * 1e-9,
                'o', fillstyle='none', markeredgewidth=1., c='r',
                label='upper branch peaks')
        ax.plot(filt_flux_low, filt_peaks_low * 1e-9,
                'o', fillstyle='none', markeredgewidth=1., c='orange',
                label='lower branch peaks')

        # self.ylabel because the axes are transposed
        xlabel = self.ylabel if xlabel is None else xlabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(self.X[0] * 1e-9), max(self.X[0] * 1e-9))
        ax.plot(flux, filter_func(flux) * 1e-9, ls='--', c='w',
                label='filter function')
        # ax.legend() # looks ugly, better after matplotlib update?
        f.savefig(os.path.join(self.folder, title + '.png'), format='png',
                  dpi=600)
        return f, ax

    def make_fit_figure(self,
                        filt_flux_low, filt_flux_high,
                        filt_peaks_low, filt_peaks_high, fit_res,
                        transpose, cmap, coupling_label=r'$J_1/2\pi$',
                        xlabel=None, ylabel='Frequency (GHz)',
                        add_title=True):
        flux = self.Y[:, 0]
        title_name = ' avoided crossing fit'
        extratitle = '\n%s' % (self.folder.split('\\')[-1][7:])
        title = title_name + extratitle
        f, ax = plt.subplots()
        if add_title:
            ax.set_title(self.timestamp_string + title)

        colorplot = pl_tools.flex_colormesh_plot_vs_xy(self.X[0] * 1e-9, flux, self.Z[0],
                                                       ax=ax, transpose=transpose,
                                                       cmap=cmap)
        f.colorbar(colorplot['cmap'], ax=colorplot['ax'])

        ax.plot(filt_flux_high, filt_peaks_high * 1e-9,
                'o', fillstyle='none', markeredgewidth=1., c='r',
                label='upper branch peaks')
        ax.plot(filt_flux_low, filt_peaks_low * 1e-9,
                'o', fillstyle='none', markeredgewidth=1., c='orange',
                label='lower branch peaks')

        # self.ylabel because the axes are transposed
        xlabel = self.ylabel if xlabel is None else xlabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(self.X[0] * 1e-9), max(self.X[0] * 1e-9))
        ax.set_xlim(min(flux), max(flux))

        ax.plot(flux, 1e-9 * fit_mods.avoided_crossing_direct_coupling(
            flux, **fit_res.best_values,
            flux_state=False), 'r-', label='fit')
        ax.plot(flux, 1e-9 * fit_mods.avoided_crossing_direct_coupling(
            flux, **fit_res.best_values,
            flux_state=True), 'y-', label='fit')

        g_legend = r'{} = {:.2f}$\pm${:.2f} MHz'.format(
            coupling_label,
            fit_res.params['g'] * 1e-6, fit_res.params['g'].stderr * 1e-6)
        ax.text(.6, .8, g_legend, transform=ax.transAxes, color='white')
        # ax.legend() # looks ugly, better after matplotlib update?
        f.savefig(os.path.join(self.folder, title_name + '.png'), format='png',
                  dpi=600)
        return f, ax

    def fit_avoided_crossing(self,
                             lower_flux, upper_flux, lower_freqs, upper_freqs,
                             f1_guess, f2_guess, cross_flux_guess, g_guess,
                             model='direct'):
        '''
        Fits the avoided crossing to a direct or mediated coupling model.

        models are located in
            fitMods.avoided_crossing_direct_coupling
            fitMods.avoided_crossing_mediated_coupling


        '''

        total_freqs = np.concatenate([lower_freqs, upper_freqs])
        total_flux = np.concatenate([lower_flux, upper_flux])
        total_mask = np.concatenate([np.ones(len(lower_flux)),
                                     np.zeros(len(upper_flux))])

        # Both branches must be combined in a single function for fitting
        # the model is combined in a single function here
        def resized_fit_func(flux, f_center1, f_center2, c1, c2, g):
            return fit_mods.avoided_crossing_direct_coupling(
                flux=flux, f_center1=f_center1, f_center2=f_center2,
                c1=c1, c2=c2,
                g=g, flux_state=total_mask)

        av_crossing_model = lmfit.Model(resized_fit_func)

        if cross_flux_guess is None:
            cross_flux_guess = np.mean(total_flux)
        if f1_guess is None:
            f1_guess = np.mean(total_freqs) - g_guess

        if f2_guess is None:
            # The factor *1000* is a magic number but seems to give a
            # reasonable guess that converges well.
            c1_guess = -1 * ((max(total_freqs) - min(total_freqs)) /
                             (max(total_flux) - min(total_flux))) / 1000

            c2_guess = 1 * ((max(total_freqs) - min(total_freqs)) /
                             (max(total_flux) - min(total_flux))) / 1000

            f2_guess = cross_flux_guess * (c1_guess - c2_guess) + f1_guess
        else:
            c1_guess = (f2_guess - f1_guess) / cross_flux_guess

        av_crossing_model.set_param_hint(
            'g', min=0., max=0.5e9, value=g_guess, vary=True)
        av_crossing_model.set_param_hint(
            'f_center1', min=0, max=20.0e9, value=f1_guess, vary=True)
        av_crossing_model.set_param_hint(
            'f_center2', min=0., max=20.0e9, value=f2_guess, vary=True)
        av_crossing_model.set_param_hint(
            'c1', min=-1.0e12, max=1.0e12, value=c1_guess, vary=True)
        av_crossing_model.set_param_hint(
            'c2', min=-1.0e12, max=1.0e12, value=c2_guess, vary=True)
        params = av_crossing_model.make_params()
        fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                        flux=np.array(total_flux),
                                        params=params)
        return fit_res


class Ram_Z_Analysis(MeasurementAnalysis):

    def __init__(self, timestamp_cos=None, timestamp_sin=None,
                 filter_raw=False, filter_deriv_phase=False, demodulate=True,
                 f_demod=0, f01max=None, E_c=None, flux_amp=None, V_offset=0,
                 V_per_phi0=None, auto=True, make_fig=True, TwoD=False,
                 mean_count=16, close_file=True, **kw):
        super().__init__(timestamp=timestamp_cos, label='cos',
                         TwoD=TwoD, **kw)
        self.cosTrace = np.array(self.measured_values[0])
        super().__init__(timestamp=timestamp_sin, label='sin',
                         TwoD=TwoD, close_file=False, **kw)
        self.sinTrace = np.array(self.measured_values[0])

        self.filter_raw = filter_raw
        self.filter_deriv_phase = filter_deriv_phase
        self.demod = demodulate
        self.f_demod = f_demod

        self.f01max = f01max
        self.E_c = E_c
        self.flux_amp = flux_amp
        self.V_offset = V_offset
        self.V_per_phi0 = V_per_phi0

        self.mean_count = mean_count

        if auto:
            if not TwoD:
                self.run_special_analysis(make_fig=make_fig)
            else:
                self.cosTrace = self.cosTrace.T
                self.sinTrace = self.sinTrace.T
                self.run_dac_arc_analysis(make_fig=make_fig)

        if close_file:
            self.data_file.close()

    def normalize(self, trace):
        # * -1 because cos starts at -1 instead of 1
        # trace *= -1
        # trace -= np.mean(trace)
        # trace /= max(np.abs(trace))
        trace = np.array(trace) * 2 - 1
        trace *= -1
        return trace

    def run_special_analysis(self, make_fig=True):
        self.df, self.raw_phases, self.phases, self.I, self.Q = \
            self.analyze_trace(
                self.cosTrace, self.sinTrace, self.sweep_points,
                filter_raw=self.filter_raw,
                filter_deriv_phase=self.filter_deriv_phase,
                demodulate=self.demod,
                f_demod=self.f_demod,
                return_all=True)

        self.add_dataset_to_analysisgroup('detuning', self.df)
        self.add_dataset_to_analysisgroup('phase', self.phases)
        self.add_dataset_to_analysisgroup('raw phase', self.raw_phases)

        if (self.f01max is not None and self.E_c is not None and
                self.flux_amp is not None and self.V_per_phi0 is not None):

            self.step_response = fit_mods.Qubit_freq_to_dac(
                frequency=self.f01max - self.df,
                f_max=self.f01max,
                E_c=self.E_c,
                dac_sweet_spot=self.V_offset,
                V_per_phi0=self.V_per_phi0,
                asymmetry=0) / self.flux_amp

            self.add_dataset_to_analysisgroup('step_response',
                                              self.step_response)
            plotStep = True
        else:
            print('To calculate step response, f01max, E_c, flux_amp, '
                  'V_per_phi0, and V_offset have to be specified.')
            plotStep = False

        if make_fig:
            self.make_figures(plot_step=plotStep)

    def analyze_trace(self, I, Q, x_pts,
                      filter_raw=False, filter_deriv_phase=False,
                      filter_width=1e-9,
                      demodulate=False, f_demod=0,
                      return_all=False):
        I = self.normalize(I)
        Q = self.normalize(Q)
        dt = x_pts[1] - x_pts[0]

        # Demodulate
        if demodulate:
            I, Q = self.demodulate(I, Q, f_demod, x_pts)

        # Filter raw data
        if filter_raw:
            I = self.gauss_filter(I, filter_width, dt, pad_val=1)
            Q = self.gauss_filter(Q, filter_width, dt, pad_val=0)

        # Calcualte phase and undo phase-wrapping
        raw_phases = np.arctan2(Q, I)
        phases = np.unwrap(raw_phases)

        # Filter phase and/or calculate the derivative
        if filter_deriv_phase:
            df = self.gauss_deriv_filter(phases, filter_width, dt, pad_val=0) \
                / (2 * np.pi)
        else:
            # Calculate central derivative
            df = np.gradient(phases, dt) / (2 * np.pi)

        # If the signal was demodulated df is now the detuning from f_demod
        if demodulate:
            df += f_demod
        df[0] = 0  # detuning must start at 0

        if return_all:
            return (df, np.rad2deg(raw_phases), np.rad2deg(phases), I, Q)
        else:
            return df

    def get_stepresponse(self, df, f01max, E_c, F_amp, V_per_phi0,
                         V_offset=0):
        '''
        Calculates the "volt per phi0" and the step response from the
        detuning.

        Args:
            df (array):     Detuning of the qubit.
            f01max (float): Sweet-spot frequency of the qubit.
            E_c (float):    Charging energy of the qubig.
            F_amp (float):  Amplitude of the applied pulse in V.
            V_per_phi0 (float): Voltage at a flux of phi0.
            V_offset (float): Offset from sweet spot in V.

        Returns:
            s (array):      Normalized step response in voltage space.
        '''
        s = (np.arccos((1 - df / (f01max + E_c)) ** 2) * np.pi / V_per_phi0 +
             V_offset) / F_amp

        return s

    def make_figures(self, plot_step=True):
        '''
        Plot figures. Step response is only plotted if plot_step == True.
        '''
        # Plot data, phases, and detuning
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(self.sweep_points[:len(self.I)], self.I, '-o')
        ax.plot(self.sweep_points[:len(self.Q)], self.Q, '-o')
        pl_tools.set_xlabel(ax, self.parameter_names[0],
                            self.parameter_units[0])
        pl_tools.set_ylabel(ax, 'demodulated normalized trace', 'a.u.')
        ax.set_title(self.timestamp_string + ' demod. norm. data')
        ax.legend(['cos', 'sin'], loc=1)
        self.save_fig(fig, 'Ram-Z_normalized_data.png')

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(self.sweep_points[:len(self.phases)], self.phases, '-o')
        pl_tools.set_xlabel(ax, self.parameter_names[0],
                            self.parameter_units[0])
        pl_tools.set_ylabel(ax, 'phase', 'deg')
        ax.set_title(self.timestamp_string + ' Phase')
        self.save_fig(fig, 'Ram-Z_phase.png')

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(self.sweep_points[:len(self.df)], self.df, '-o')
        pl_tools.set_xlabel(ax, self.parameter_names[0],
                            self.parameter_units[0])
        pl_tools.set_ylabel(ax, 'detuning', 'Hz')
        ax.set_title(self.timestamp_string + ' Detuning')
        self.save_fig(fig, 'Ram-Z_detuning.png')

        if plot_step:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax.plot(self.sweep_points[:len(self.step_response)],
                    self.step_response, '-o')
            ax.axhline(y=1, color='0.75')
            pl_tools.set_xlabel(ax, self.parameter_names[0],
                                self.parameter_units[0])
            pl_tools.set_ylabel(ax, 'step response', '')
            ax.set_title(self.timestamp_string + ' Step Response')
            self.save_fig(fig, 'Ram-Z_step_response.png')
            # fig.savefig('Ram-Z_step_response.png', dpi=300)

    def demodulate(self, I, Q, f_demod, t_pts):
        '''
        Demodulate signal in I and Q, sampled at points t_pts, with frequency
        f_demod.
        '''
        cosDemod = np.cos(2 * np.pi * f_demod * t_pts)
        sinDemod = np.sin(2 * np.pi * f_demod * t_pts)
        Iout = I * cosDemod + Q * sinDemod
        Qout = Q * cosDemod - I * sinDemod

        return Iout, Qout

    def gauss_filter(self, data, sigma, d, nr_sigmas=4, pad_val=None):
        '''
        Convolves data with a normalized Gaussian with width sigma. When used
        as a low-pass filter, the width in the frequency domain is 1/sigma.
        The Gaussian is sampled at the same rate as the data, given by the
        sample distance d. The convolution is calculated only at points of
        complete overlap, and the result will thus contain less points than
        the input array. The data is padded with pad_val (or with data[0] if
        pad_val is not specified) at the front to ensure that the x-axis is
        not changed. No padding is done at the end of the data.

        Args:
            data (array):   Data to be filtered.
            sigma (float):  Width of the Gaussian filter.
            d (float):      Sampling distance of the data, i.e. distance
                            of points on the x-axis of the data.
            nr_sigmas (int): Up to how many sigmas away from the center the
                            Gaussian is sampled.  E.g. if d=1 ns, sigma=.5 ns,
                            nr_sigmas=4 ensure the Gaussian is sampled at
                            least up to +-2 ns, and the filter will have at
                            least nine samples.
            pad_val (float): Value used for padding in front of the data.
        '''
        filterHalfWidth = np.ceil(nr_sigmas * sigma / d)
        tMaxFilter = filterHalfWidth * d
        # upper limit of range has + dt/10 to include endpoint
        tFilter = np.arange(-tMaxFilter, tMaxFilter + d / 10, step=d)

        gaussFilter = np.exp(-tFilter ** 2 / (2 * sigma ** 2))
        gaussFilter /= np.sum(gaussFilter)

        if pad_val is None:
            pad_val = data[0]
        paddedData = np.concatenate((np.ones(int(filterHalfWidth)) *
                                     pad_val, data))
        return np.convolve(paddedData, gaussFilter, mode='valid')

    def gauss_deriv_filter(self, data, sigma, d, nr_sigmas=4, pad_val=None):
        '''
        Convolves data with the derivative of a normalized Gaussian with width
        sigma. This is useful to apply a low-pass filter (with cutoff 1/sigma)
        and simultaneously calculate the derivative. The Gaussian is sampled
        at the same rate as the data, given by the sample distance d. The
        convolution is calculated only at points of complete overlap, and the
        result will thus contain less points than the input array. The data is
        padded with pad_val (or with data[0] if pad_val is not specified) at
        the front to ensure that the x-axis is not changed. No padding is done
        at the end of the data.

        Args:
            data (array):   Data to be filtered.
            sigma (float):  Width of the Gaussian filter.
            d (float):      Sampling distance of the data, i.e. distance of
                            points on the x-axis of the data.
            nr_sigmas (int): Up to how many sigmas away from the center the
                            Gaussian is sampled.  E.g. if d=1 ns, sigma=.5 ns,
                            nr_sigmas=4 ensure the Gaussian is sampled at
                            least up to +-2 ns, and the filter will have at
                            least nine samples.
        '''
        filterHalfWidth = np.ceil(nr_sigmas * sigma / d)
        tMaxFilter = filterHalfWidth * d
        # upper limit of range has + dt/10 to include endpoint
        tFilter = np.arange(-tMaxFilter, tMaxFilter + d / 10, step=d)

        # First calculate normalized Gaussian, then derivative
        gaussFilter = np.exp(-tFilter ** 2 / (2 * sigma ** 2))
        gaussFilter /= np.sum(gaussFilter)
        gaussDerivFilter = gaussFilter * (-tFilter) / (sigma ** 2)

        if pad_val is None:
            pad_val = data[0]
        paddedData = np.concatenate(
            (np.ones(int(filterHalfWidth)) * pad_val, data))
        return np.convolve(paddedData, gaussDerivFilter, mode='valid')

    def run_dac_arc_analysis(self, make_fig=True):
        '''
        Analyze a 2D Ram-Z scan (pulse length vs. pulse amplitude), and
        exctract a dac arc.
        '''
        df = self.analyze_trace(
            self.cosTrace[0], self.sinTrace[0], self.sweep_points,
            filter_raw=self.filter_raw,
            filter_deriv_phase=self.filter_deriv_phase,
            demodulate=self.demod,
            f_demod=self.f_demod,
            return_all=False)

        if self.demod:
            # Take an initial guess for V_per_phi0, if it has not been
            # specified
            # Note: assumes symmetric qubit.
            if self.V_per_phi0 is None:
                self.V_per_phi0 = (
                    np.pi * (self.sweep_points_2D[0] - self.V_offset) /
                    np.arccos(((self.f01max - df + self.E_c) /
                               (self.f01max + self.E_c)) ** 2))

            # Set the demodulation frequencies based on the guess
            self.demod_freqs = [fit_mods.Qubit_dac_to_detun(
                v, f_max=self.f01max,
                E_c=self.E_c,
                dac_sweet_spot=self.V_offset,
                V_per_phi0=self.V_per_phi0) for v in self.sweep_points_2D]
        else:
            self.demod_freqs = np.zeros(len(self.sweep_points_2D))

        # Run analysis on the remaining traces
        self.all_df = np.empty((len(self.sweep_points_2D), len(df)))
        self.all_df[0] = df

        for i in range(1, len(self.sweep_points_2D)):
            df = self.analyze_trace(
                self.cosTrace[i], self.sinTrace[i], self.sweep_points,
                filter_raw=self.filter_raw,
                filter_deriv_phase=self.filter_deriv_phase,
                demodulate=self.demod,
                f_demod=self.demod_freqs[i],
                return_all=False)
            self.all_df[i] = df

        self.mean_freqs = np.array([np.mean(i[-self.mean_count:])
                                    for i in self.all_df])

        self.fit_freqs, self.fit_amps = self.remove_outliers(
            [len(self.sweep_points_2D) // 2])

        self.param_hints = {
            'f_max': self.f01max,
            'E_c': self.E_c,
            'V_per_phi0': self.V_per_phi0,
            'dac_sweet_spot': self.V_offset,
            'asymmetry': 0
        }

        self.fit_res = self.fit_dac_arc(self.fit_freqs,
                                        self.fit_amps,
                                        param_hints=self.param_hints)

        if make_fig:
            self.make_figures_2D()

    def fit_dac_arc(self, df, V, param_hints={}):
        '''
        Fit the model for the dac arc to the detunings df (y-axis) and appplied
        voltages V (x-axis).
        '''
        model = lmfit.Model(fit_mods.Qubit_dac_to_detun)
        model.set_param_hint('f_max', value=param_hints.pop('f_max', 6e9),
                             min=0, vary=False)
        model.set_param_hint('E_c', value=param_hints.pop('E_c', 0.25e9),
                             vary=False)
        model.set_param_hint('V_per_phi0',
                             value=param_hints.pop('V_per_phi0', 1))
        model.set_param_hint('dac_sweet_spot',
                             value=param_hints.pop('V_offset', 0))
        model.set_param_hint('asymmetry',
                             value=param_hints.pop('asymmetry', 0),
                             vary=False)
        params = model.make_params()

        fit_res = model.fit(df, dac_voltage=V, params=params)
        return fit_res

    def remove_outliers(self, indices):
        '''
        Removes the elementes at the given indices from the self.all_df and
        self.sweep_points_2D and returns the resulting arrays.

        Args:
            indices (tuple of ints):
                    Indices of elements which should be removed.
        '''
        fit_freqs = deepcopy(self.mean_freqs)
        fit_amps = deepcopy(self.sweep_points_2D)

        return np.delete(fit_freqs, indices), np.delete(fit_amps, indices)

    def make_figures_2D(self, figsize=(7, 5)):
        xFine = np.linspace(self.sweep_points_2D[0], self.sweep_points_2D[-1],
                            100)
        dacArc = self.fit_res.eval(dac_voltage=xFine)  # fit was in GHz

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(self.sweep_points_2D, self.mean_freqs, '-o')
        ax.plot(xFine, dacArc)
        ax.text(.1, .8,
                '$V_{\mathsf{per }\Phi_0} = $'
                + str(self.fit_res.best_values['V_per_phi0'])
                + '\n$V_\mathsf{offset} = $'
                + str(self.fit_res.best_values['dac_sweet_spot']),
                transform=ax.transAxes)
        set_xlabel(ax, self.parameter_names[1], self.parameter_units[1])
        set_ylabel(ax, 'detuning', 'Hz')

        self.save_fig(fig, 'Ram-Z_dac_arc.png')


class GST_Analysis(TD_Analysis):
    '''
    Analysis for Gate Set Tomography. Extracts data from the files, bins it
    correctly and writes the counts to a file in the format required by
    pyGSTi. The actual analysis is then run using the tools from pyGSTi.
    '''

    def __init__(self, timestamp=None, nr_qubits: int = 1, **kw):
        '''
        Args:
            nr_qubits (int):
                    Number of qubits with which the GST was run.
        '''
        self.exp_per_file = 0
        self.hard_repetitions = 0
        self.soft_repetitions = 0
        self.exp_list = []
        self.gs_target = None
        self.prep_fids = None
        self.meas_fids = None
        self.germs = None
        self.max_lengths = []

        self.nr_qubits = nr_qubits
        self.counts = []
        super().__init__(cal_points=False, make_fig=False,
                         timestamp=timestamp, **kw)

    def run_default_analysis(self, **kw):
        self.close_file = kw.get('close_file', True)
        self.get_naming_and_values()
        self.exp_metadata = h5d.read_dict_from_hdf5(
            {}, self.data_file['Experimental Data']['Experimental Metadata'])

        self.gs_target = pygsti.io.load_gateset(
            self.exp_metadata['gs_target'])
        self.meas_fids = pygsti.io.load_gatestring_list(
            self.exp_metadata['meas_fids'])
        self.prep_fids = pygsti.io.load_gatestring_list(
            self.exp_metadata['prep_fids'])
        self.germs = pygsti.io.load_gatestring_list(
            self.exp_metadata['germs'])
        self.max_lengths = self.exp_metadata['max_lengths']
        self.exp_per_file = self.exp_metadata['exp_per_file']
        self.exp_last_file = self.exp_metadata['exp_last_file']
        self.hard_repetitions = self.exp_metadata['hard_repetitions']
        self.soft_repetitions = self.exp_metadata['soft_repetitions']
        self.nr_hard_segs = self.exp_metadata['nr_hard_segs']

        self.exp_list = pygsti.construction.make_lsgst_experiment_list(
            self.gs_target.gates.keys(), self.prep_fids, self.meas_fids,
            self.germs, self.max_lengths)

        # Count the results. Method depends on how many qubits were used.
        if self.nr_qubits == 1:
            self.counts, self.spam_label_order = self.count_results_1Q()
        elif self.nr_qubits == 2:
            self.counts, self.spam_label_order = self.count_results_2Q()
        else:
            raise NotImplementedError(
                'GST analysis for {} qubits is not implemented.'
                .format(self.nr_qubits))

        # Write extracted counts to file.
        self.pygsti_fn = os.path.join(self.folder, 'pyGSTi_dataset.txt')
        self.write_GST_datafile(self.pygsti_fn, self.counts,
                                self.spam_label_order)

        # Run pyGSTi analysis and create report.
        self.results = pygsti.do_long_sequence_gst(
            self.pygsti_fn, self.gs_target, self.prep_fids, self.meas_fids,
            self.germs, self.max_lengths)

        with open(os.path.join(self.folder,
                               'pyGSTi_results.p'), 'wb') as file:
            pickle.dump(self.results, file)

        self.report_fn = os.path.join(self.folder, 'pyGSTi_report.pdf')
        self.results.create_full_report_pdf(confidenceLevel=95,
                                            filename=self.report_fn,
                                            verbosity=2)

    def write_GST_datafile(self, filepath: str, counts: list, labels=()):
        '''
        Write the measured counts to a file in pyGSTi format.
        Args:
            filepath (string):
                    Full path of the file to be written.
            counts (list):
                    List pf tuples (gate_seq_str, ...), where the first
                    entry is the string representation of the gate sequence,
                    and the following entries are the counts for the counts
                    for the different measurement operators.
            labels (list):
                    List of the labels for the measurement operators. Note
                    that is just for readability and does not affect the
                    order in which the counts are written. The order should
                    be the same as in the pyGSTi gateset definition.
        '''
        # The directive strings tells the pyGSTi parser which column
        # corresponds to which SPAM label.
        directive_string = ('## Columns = ' +
                            ', '.join(['{} count'] * len(labels))
                            .format(*labels))
        with open(filepath, 'w') as file:
            file.writelines(directive_string)
            for tup in counts:
                file.writelines(('\n' + '{}  ' * len(tup)).format(*tup))

    def count_results_1Q(self):
        # Find the results that belong to the same GST sequence and sum up the
        # counts.

        # This determines in which order the results are written.
        # The strings in this list must be the same SPAM labels as defined
        # in the pyGSTi target gateset.
        spam_label_order = ['plus', 'minus']  # plus = |0>, minus = |1>

        # First, reshape data according to soft repetitions.
        counts = []
        data = np.reshape(self.measured_values[0],
                          (self.soft_repetitions, -1))
        # Each row (i.e. data[i, :]) now contains one of the soft repetitions
        # containing all of the required GST experiments. They are however
        # still ordered in segments according to the hard repetitions.
        # d: distance between measurements of same sequence
        # l: length of index range corresponding to one segment
        d = self.exp_per_file
        l = self.exp_per_file * self.hard_repetitions
        for i in range(self.nr_hard_segs - int(self.exp_last_file != 0)):
            # For every segment... subtract 1 from nr_hard_segs to exclude
            # last file if the last file has a different number of experiments
            block_idx = i * l

            for seq_idx in range(self.exp_per_file):
                # For every sequence in the current segment
                # The full index is index of the segment plus index
                # (block_idx) of the sequence in the segment (seq_idx)
                one_count = 0
                for soft_idx in range(self.soft_repetitions):
                    # For all soft repetitions: sum up "1" counts.
                    one_count += np.sum(
                        data[soft_idx, block_idx + seq_idx:block_idx + l:d],
                        dtype=int)
                zero_count = (self.hard_repetitions * self.soft_repetitions -
                              one_count)

                counts.append((self.exp_list[i + seq_idx].str,
                               zero_count, one_count))

        # If the last file has a different number of experiments, count those
        # separately
        if self.exp_last_file != 0:
            d_last = self.exp_last_file
            l_last = self.exp_last_file * self.hard_repetitions
            block_idx = l * (self.hard_repetitions - 1)

            for seq_idx in range(self.exp_last_file):
                one_count = 0
                for soft_idx in range(self.soft_repetitions):
                    one_count += np.sum(
                        data[soft_idx,
                             block_idx + seq_idx:block_idx + l_last:d_last],
                        dtype=int)
                zero_count = (self.hard_repetitions * self.soft_repetitions -
                              one_count)

                counts.append(
                    (self.exp_list[self.nr_hard_segs - 1 + seq_idx].str,
                     zero_count, one_count))
        return counts, spam_label_order

    def count_results_2Q(self):
        # Find the results that belong to the same GST sequence and sum up the
        # counts.

        # This determines in which order the results are written.
        # The strings in this list must be the same SPAM labels as defined
        # in the pyGSTi target gateset.
        # 'up' = |0>, 'dn' = |1>
        spam_label_order = ['upup', 'updn', 'dnup', 'dndn']

        # First, reshape data according to soft repetitions.
        # IMPORTANT NOTE: This assumes that the first column in the measured
        # values is the readout of the least significant qubit. This is
        # important because it has to be consistent with how the pyGSTi spam
        # labels are defined.
        counts = []
        data_q0 = np.reshape(self.measured_values[0],
                             (self.soft_repetitions, -1))
        data_q1 = np.reshape(self.measured_values[1],
                             (self.soft_repetitions, -1))

        # Each row (i.e. data[i, :]) now contains one of the soft repetitions
        # containing all of the required GST experiments. They are however
        # still ordered in segments according to the hard repetitions.
        # d: distance between measurements of same sequence
        # l: length of index range corresponding to one segment
        d = self.exp_per_file
        l = self.exp_per_file * self.hard_repetitions

        for i in range(self.nr_hard_segs - int(self.exp_last_file != 0)):
            # For every segment... subtract 1 from nr_hard_segs to exclude
            # last file if the last file has a different number of experiments
            block_idx = i * l

            for seq_idx in range(self.exp_per_file):
                # For every sequence in the current segment
                new_count = (0, 0, 0, 0)

                for soft_idx in range(self.soft_repetitions):
                    for x in range(0, l, d):
                        q0_bit = data_q0[soft_idx, block_idx + seq_idx + x]
                        q1_bit = data_q1[soft_idx, block_idx + seq_idx + x]
                        if not q0_bit and not q1_bit:
                            new_count[0] += 1
                        elif q0_bit and not q1_bit:
                            new_count[1] += 1
                        elif not q0_bit and q1_bit:
                            new_count[2] += 1
                        else:
                            new_count[3] += 1

                counts.append((self.exp_list[i + seq_idx].str, *new_count))

        # If the last file has a different number of experiments, count those
        # separately
        if self.exp_last_file != 0:
            d_last = self.exp_last_file
            l_last = self.exp_last_file * self.hard_repetitions
            block_idx = l * (self.hard_repetitions - 1)

            for seq_idx in range(self.exp_last_file):
                new_count = (0, 0, 0, 0)
                for soft_idx in range(self.soft_repetitions):
                    for x in range(0, l_last, d_last):
                        q0_bit = data_q0[soft_idx, block_idx + seq_idx + x]
                        q1_bit = data_q1[soft_idx, block_idx + seq_idx + x]
                        if not q0_bit and not q1_bit:
                            new_count[0] += 1
                        elif q0_bit and not q1_bit:
                            new_count[1] += 1
                        elif not q0_bit and q1_bit:
                            new_count[2] += 1
                        else:
                            new_count[3] += 1

                counts.append(
                    (self.exp_list[self.nr_hard_segs - 1 + seq_idx].str,
                     *new_count))

        return counts, spam_label_order


class CZ_1Q_phase_analysis(TD_Analysis):

    def __init__(self, use_diff: bool = True, meas_vals_idx: int = 0, **kw):
        self.use_diff = use_diff
        self.meas_vals_idx = meas_vals_idx
        super().__init__(rotate_and_normalize=False, cal_points=False, **kw)

    def run_default_analysis(self, **kw):
        super().run_default_analysis(make_fig=True, close_file=False)

        model = lmfit.models.QuadraticModel()

        if self.use_diff:
            dat_exc = self.measured_values[self.meas_vals_idx][1::2]
            dat_idx = self.measured_values[self.meas_vals_idx][::2]
            self.full_data = dat_idx - dat_exc
            self.x_points = self.sweep_points[::2]

            # Remove diff points thate are larger than one (parabola won't fit
            # there).
            self.del_indices = np.where(np.array(self.full_data) > 0)[0]
        else:
            self.full_data = self.measured_values[self.meas_vals_idx]
            self.x_points = self.sweep_points
            self.del_indices = np.where(np.array(self.full_data) > 0.5)[0]

        self.fit_data = np.delete(self.full_data, self.del_indices)
        self.x_points_del = np.delete(self.x_points, self.del_indices)

        if self.fit_data.size == 0:
            raise RuntimeError('No points left to fit after removing values '
                               '> 0! Check coarse calibration and adjust '
                               'measurement range.')

        params = model.guess(x=self.x_points_del, data=self.fit_data)

        self.fit_res = model.fit(self.fit_data, params=params,
                                 x=self.x_points_del)

        self.opt_z_amp = (-self.fit_res.best_values['b'] /
                          (2 * self.fit_res.best_values['a']))

        if self.make_fig:
            self.make_figures()

        if kw.get('close_file', True):
            self.data_file.close()

    def make_figures(self, **kw):
        xfine = np.linspace(self.x_points[0], self.x_points[-1], 100)
        fig, ax = plt.subplots()
        ax.plot(self.x_points, self.full_data, '-o')
        ax.plot(self.x_points[self.del_indices],
                self.full_data[self.del_indices], 'rx',
                label='excluded in fit')
        ax.plot(xfine,
                self.fit_res.eval(x=xfine, **self.fit_res.init_values),
                # self.fit_res.init_fit,
                '--',
                label='initial guess', c='k')
        ax.plot(xfine,
                self.fit_res.eval(x=xfine, **self.fit_res.best_values),
                label='best fit')
        set_xlabel(ax, self.parameter_names[0], self.parameter_units[0])
        set_ylabel(ax, 'Z-amp cost', 'a.u.')
        ax.set_title(kw.get('plot_title',
                            textwrap.fill(self.timestamp_string + '_' +
                                          self.measurementstring, 40)))
        ax.text(.1, .9, 'Optimal amplitude: {:.4f}'.format(self.opt_z_amp),
                transform=ax.transAxes)
        ax.legend()
        plt.tight_layout()
        self.save_fig(fig, **kw)


def DAC_scan_analysis_and_plot(**kwargs):
    raise DeprecationWarning(
        'Use FluxFrequency from analysis_v2.dac_scan_analysis instead.')


def time_domain_DAC_scan_analysis_and_plot(**kwargs):
    raise DeprecationWarning(
        'Use FluxFrequency from analysis_v2.dac_scan_analysis instead.')


def Input_average_analysis(IF, fig_format='png', alpha=1, phi=0, I_o=0, Q_o=0,
                           predistort=True, plot=True, timestamp_ground=None,
                           timestamp_excited=None, close_fig=True,
                           optimization_window=None, post_rotation_angle=None,
                           plot_max_time=4096/1.8e9):
    data_file = MeasurementAnalysis(
        label='_0', auto=True, TwoD=False, close_fig=True, timestamp=timestamp_ground)
    temp = data_file.load_hdf5data()
    data_file.get_naming_and_values()

    # using the last x samples for offset subtraction 720 is multiples of 2.5
    # MHz modulation
    offset_calibration_samples = 720

    x = data_file.sweep_points / 1.8
    offset_I = np.mean(data_file.measured_values[
        0][-offset_calibration_samples:])
    offset_Q = np.mean(data_file.measured_values[
        1][-offset_calibration_samples:])
    print('offset I {}, offset Q {}'.format(offset_I, offset_Q))
    y1 = data_file.measured_values[0] - offset_I
    y2 = data_file.measured_values[1] - offset_Q

    I0_no_demod = y1
    Q0_no_demod = y2

    I0, Q0 = SSB_demod(y1, y2, alpha=alpha, phi=phi, I_o=I_o,
                       Q_o=Q_o, IF=IF, predistort=predistort)
    power0 = (I0 ** 2 + Q0 ** 2) / 50

    data_file = MeasurementAnalysis(
        label='_1', auto=True, TwoD=False, close_fig=True, plot=True, timestamp=timestamp_excited)
    temp = data_file.load_hdf5data()
    data_file.get_naming_and_values()

    x = data_file.sweep_points / 1.8
    offset_I = np.mean(data_file.measured_values[
        0][-offset_calibration_samples:])
    offset_Q = np.mean(data_file.measured_values[
        1][-offset_calibration_samples:])
    y1 = data_file.measured_values[0] - offset_I
    y2 = data_file.measured_values[1] - offset_Q
    I1, Q1 = SSB_demod(y1, y2, alpha=alpha, phi=phi, I_o=I_o,
                       Q_o=Q_o, IF=IF, predistort=predistort)

    I1_no_demod = y1
    Q1_no_demod = y2

    power1 = (I1 ** 2 + Q1 ** 2) / 50

    amps = np.sqrt((I1 - I0) ** 2 + (Q1 - Q0) ** 2)
    amp_max = np.max(amps)
    # defining weight functions for postrotation
    weight_I = (I1 - I0) / amp_max
    weight_Q = (Q1 - Q0) / amp_max


    weight_I_no_demod = (I1_no_demod - I0_no_demod) / amp_max
    weight_Q_no_demod = (Q1_no_demod - Q0_no_demod) / amp_max

    # Identical rescaling as is happening in the CCL transmon class
    maxI_no_demod = np.max(np.abs(weight_I_no_demod))
    maxQ_no_demod = np.max(np.abs(weight_Q_no_demod))
    weight_scale_factor = 1./(4*np.max([maxI_no_demod, maxQ_no_demod]))
    weight_I_no_demod = np.array(
        weight_scale_factor*weight_I_no_demod)
    weight_Q_no_demod = np.array(
        weight_scale_factor*weight_Q_no_demod)



    if post_rotation_angle == None:
        arg_max = np.argmax(amps)
        post_rotation_angle = np.arctan2(
            weight_I[arg_max], weight_Q[arg_max]) - np.pi / 2
        # print('found post_rotation angle {}'.format(post_rotation_angle))
    else:
        post_rotation_angle = 2 * np.pi * post_rotation_angle / 360
    I0rot = np.cos(post_rotation_angle) * I0 - np.sin(post_rotation_angle) * Q0
    Q0rot = np.sin(post_rotation_angle) * I0 + np.cos(post_rotation_angle) * Q0
    I1rot = np.cos(post_rotation_angle) * I1 - np.sin(post_rotation_angle) * Q1
    Q1rot = np.sin(post_rotation_angle) * I1 + np.cos(post_rotation_angle) * Q1
    I0 = I0rot
    Q0 = Q0rot
    I1 = I1rot
    Q1 = Q1rot

    # redefining weight functions after rotation
    weight_I = (I1 - I0) / amp_max
    weight_Q = (Q1 - Q0) / amp_max

    edge = 1.05 * max(max(np.sqrt(I0 ** 2 + Q0 ** 2)),
                      max(np.sqrt(I1 ** 2 + Q1 ** 2)))

    def rms(x):
        return np.sqrt(x.dot(x) / x.size)

    if optimization_window != None:
        optimization_start = optimization_window[0]
        optimization_stop = optimization_window[-1]
        start_sample = int(optimization_start * 1.8e9)
        stop_sample = int(optimization_stop * 1.8e9)
        shift_w = 0e-9
        start_sample_w = int((optimization_start - shift_w) * 1.8e9)
        stop_sample_w = int((optimization_stop - shift_w) * 1.8e9)
        depletion_cost_d = np.mean(rms(I0[start_sample:stop_sample]) +
                                   rms(Q0[start_sample:stop_sample]) +
                                   rms(I1[start_sample:stop_sample]) +
                                   rms(Q1[start_sample:stop_sample]))
        depletion_cost_w = 10 * np.mean(rms(I0[start_sample_w:stop_sample_w] - I1[start_sample_w:stop_sample_w]) +
                                        rms(Q0[start_sample_w:stop_sample_w] - Q1[
                                            start_sample_w:stop_sample_w]))  # +abs(np.mean(Q0[start_sample:stop_sample]))+abs(np.mean(I1[start_sample:stop_sample]))+abs(np.mean(Q1[start_sample:stop_sample]))
        depletion_cost = depletion_cost_d + depletion_cost_w
        # print('total {} direct {} weights {}'.format(1000*depletion_cost, 1000*depletion_cost_d, 1000*depletion_cost_w))
    else:
        depletion_cost = 0

    if plot:
        fig, ax = plt.subplots()
        time = np.arange(0, len(weight_I) / 1.8, 1/1.8)
        plt.plot(time, I0, label='I ground')
        plt.plot(time, I1, label='I excited')
        ax.set_ylim(-edge, edge)

        plt.title('Demodulated I')
        plt.xlabel('time (ns)')
        plt.ylabel('Demodulated voltage (V)')

        if optimization_window != None:
            plt.axvline(optimization_start * 1e9, linestyle='--',
                        color='k', label='depletion optimization window')
            plt.axvline(optimization_stop * 1e9, linestyle='--', color='k')
        ax.set_xlim(0, plot_max_time*1e9)
        plt.legend()

        plt.savefig(data_file.folder + '\\' +
                    'transients_I_demodulated.' + fig_format, format=fig_format)
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(time, Q0, label='Q ground')
        plt.plot(time, Q1, label='Q excited')
        ax.set_ylim(-edge, edge)
        plt.title('Demodulated Q')
        plt.xlabel('time (ns)')
        plt.ylabel('Demodulated Q')
        if optimization_window != None:
            plt.axvline(optimization_start * 1e9, linestyle='--',
                        color='k', label='depletion optimization window')
            plt.axvline(optimization_stop * 1e9, linestyle='--', color='k')
        ax.set_xlim(0, plot_max_time*1e9)
        plt.legend()

        plt.savefig(data_file.folder + '\\' +
                    'transients_Q_demodulated.' + fig_format, format=fig_format)
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(time, power0 * 1e6, label='ground', lw=4)
        plt.plot(time, power1 * 1e6, label='excited', lw=4)
        if optimization_window != None:
            plt.axvline(optimization_start * 1e9, linestyle='--',
                        color='k', label='depletion optimization window')
            plt.axvline(optimization_stop * 1e9, linestyle='--', color='k')
        ax.set_xlim(0, plot_max_time*1e9)
        plt.title('Signal power (uW)')
        plt.ylabel('Signal power (uW)')

        plt.savefig(data_file.folder + '\\' + 'transients_power.' +
                    fig_format, format=fig_format)
        plt.close()

    # sampling rate GHz
    A0I = I0
    A0Q = Q0

    A1I = I1
    A1Q = Q1
    Fs = 1.8e9
    f_axis, PSD0I = func.PSD(A0I, 1 / Fs)
    f_axis, PSD1I = func.PSD(A1I, 1 / Fs)
    f_axis, PSD0Q = func.PSD(A0Q, 1 / Fs)
    f_axis, PSD1Q = func.PSD(A1Q, 1 / Fs)

    f_axis_o, PSD0I_o = func.PSD(A0I[-1024:], 1 / Fs)
    f_axis_o, PSD1I_o = func.PSD(A1I[-1024:], 1 / Fs)
    f_axis_o, PSD0Q_o = func.PSD(A0Q[-1024:], 1 / Fs)
    f_axis_o, PSD1Q_o = func.PSD(A1Q[-1024:], 1 / Fs)

    n_spurious = int(round(2 * len(A0I) * abs(IF) / Fs))
    if n_spurious>len(f_axis):
        logging.warning('Calibrate_optimal_weights ANALYSIS: Spurious frequency not in range')
        f_spurious = 0
    else:
        f_spurious = f_axis[n_spurious]
    n_offset = int(round(len(A0I[-1024:]) * abs(IF) / Fs))
    if n_offset>len(f_axis_o):
        logging.warning('Calibrate_optimal_weights ANALYSIS: offset frequency not in range')
        f_offset = 0
    else:
        f_offset = f_axis_o[n_offset]

    # print('f_spurious', f_spurious)
    # print('f_offset', f_offset)
    # print(len(A0I), len(A0I[-1024:]))

    samples = 7
    cost_skew = 0
    cost_offset = 0

    for i in range(samples):
        n_s = np.clip(int(n_spurious - samples / 2 + i),0,len(PSD0I)-1)
        n_o = np.clip(int(n_offset - samples / 2 + i),0,len(PSD0I_o)-1)

        cost_skew = cost_skew + \
            np.abs(PSD0I[n_s]) + np.abs(PSD1I[n_s]) + \
            np.abs(PSD0Q[n_s]) + np.abs(PSD1Q[n_s])
        cost_offset = cost_offset + \
            np.abs(PSD0I_o[n_o]) + np.abs(PSD1I_o[n_o]) + \
            np.abs(PSD0Q_o[n_o]) + np.abs(PSD1Q_o[n_o])

    #         print('freq',f_axis[n])
    #         print('cost_skew', cost_skew)
    if plot:
        fig, ax = plt.subplots(2)
        ax[0].set_xlim(0, 0.4)
        # plotting the spectrum
        ax[0].plot(f_axis * 1e-9, abs(PSD0I), label='ground I')
        # plotting the spectrum
        ax[0].plot(f_axis * 1e-9, abs(PSD1I), label='excited I')
        ax[1].set_xlim(0, 0.4)
        # plotting the spectrum
        ax[1].plot(f_axis * 1e-9, abs(PSD0Q), label='ground Q')
        # plotting the spectrum
        ax[1].plot(f_axis * 1e-9, abs(PSD1Q), label='excited Q')
        ax[1].set_xlabel('Freq (GHz)')
        ax[0].set_ylabel('|PSD|')
        ax[0].set_yscale('log')
        ax[1].set_ylabel('|PSD|')
        ax[1].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title('PSD')

        plt.savefig(data_file.folder + '\\' + 'PSD.' +
                    fig_format, format=fig_format)
        plt.close()

        fig, ax = plt.subplots(2)
        ax[0].set_xlim(0, 0.4)
        # plotting the spectrum
        ax[0].plot(f_axis_o * 1e-9, abs(PSD0I_o), label='ground I')
        # plotting the spectrum
        ax[0].plot(f_axis_o * 1e-9, abs(PSD1I_o), label='excited I')
        ax[1].set_xlim(0, 0.4)
        # plotting the spectrum
        ax[1].plot(f_axis_o * 1e-9, abs(PSD0Q_o), label='ground Q')
        # plotting the spectrum
        ax[1].plot(f_axis_o * 1e-9, abs(PSD1Q_o), label='excited Q')
        ax[1].set_xlabel('Freq (GHz)')
        ax[0].set_ylabel('|PSD|')
        ax[0].set_yscale('log')
        ax[1].set_ylabel('|PSD|')
        ax[1].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title('PSD last quarter')

        plt.savefig(data_file.folder + '\\' + 'PSD_last_quarter.' +
                    fig_format, format=fig_format)
        plt.close()

        fig, ax = plt.subplots(figsize=[8, 7])
        plt.plot(I0, Q0, label='ground', lw=1)
        plt.plot(I1, Q1, label='excited', lw=1)
        ax.set_ylim(-edge, edge)
        ax.set_xlim(-edge, edge)
        plt.legend(frameon=False)
        plt.title('IQ trajectory alpha{} phi{}_'.format(
            alpha, phi) + data_file.timestamp_string)
        plt.xlabel('I (V)')
        plt.ylabel('Q (V)')
        plt.savefig(data_file.folder + '\\' + 'IQ_trajectory.' +
                    fig_format, format=fig_format)
        plt.close()

    fig, ax = plt.subplots(figsize=[8, 7])
    plt.plot(weight_I, weight_Q, label='weights', lw=1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    plt.legend(frameon=False)
    plt.title('IQ trajectory weights')
    plt.xlabel('weight I')
    plt.ylabel('weight Q')
    plt.savefig(data_file.folder + '\\' + 'IQ_trajectory_weights')
    plt.close()

    time = np.arange(0, len(weight_I) / 1.8, 1/1.8)
    fig, ax = plt.subplots()
    plt.plot(time, weight_I, label='weight I')
    plt.plot(time, weight_Q, label='weight Q')
    if optimization_window != None:
        plt.axvline((optimization_start - shift_w) * 1e9, linestyle='--',
                    color='k', label='depletion optimization window')
        plt.axvline((optimization_stop - shift_w) *
                    1e9, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('time (ns)')
    plt.ylabel('Integration weight (V)')
    plt.title('demodulated weight functions_' + data_file.timestamp_string)
    plt.axhline(0, linestyle='--')
    edge = 1.05 * max(max(abs(weight_I)), max(abs(weight_Q)))
    ax.set_xlim(0, plot_max_time*1e9)
    plt.savefig(data_file.folder + '\\' + 'demodulated_weight_functions.' +
                fig_format, format=fig_format)
    plt.close()


    fig, ax = plt.subplots()
    plt.plot(time, weight_I_no_demod, label='weight I')
    plt.plot(time, weight_Q_no_demod, label='weight Q')
    if optimization_window != None:
        plt.axvline((optimization_start - shift_w) * 1e9, linestyle='--',
                    color='k', label='depletion optimization window')
        plt.axvline((optimization_stop - shift_w) *
                    1e9, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('time (ns)')
    plt.ylabel('Integration weight (V)')
    plt.title('weight functions_' + data_file.timestamp_string)
    plt.axhline(0, linestyle='--')
    edge = 1.05 * max(max(abs(weight_I)), max(abs(weight_Q)))
    ax.set_xlim(0, plot_max_time*1e9)
    plt.savefig(data_file.folder + '\\' + 'weight_functions.' +
                fig_format, format=fig_format)
    plt.close()



    # should return a dict for the function detector
    # return cost_skew, cost_offset, depletion_cost, x, y1, y2, I0, Q0, I1, Q1
    return {'cost_skew': cost_skew, 'cost_offset': cost_offset,
            'depletion_cost': depletion_cost, 'x': x, 'y1': y1, 'y2': y2,
            'I0': I0, 'Q0': Q0, 'I1': I1, 'Q1': Q1}


# analysis functions
def SSB_demod(Ivals, Qvals, alpha=1, phi=0, I_o=0, Q_o=0, IF=10e6, predistort=True, sampling_rate=1.8e9):
    # predistortion_matrix = np.array(
    #     ((1,  np.tan(phi*2*np.pi/360)),
    #      (0, 1/alpha * 1/np.cos(phi*2*np.pi/360))))
    predistortion_matrix = np.array(
        ((1, -alpha * np.sin(phi * 2 * np.pi / 360)),
         (0, alpha * np.cos(phi * 2 * np.pi / 360))))

    trace_length = len(Ivals)
    tbase = np.arange(0, trace_length / sampling_rate, 1 / sampling_rate)
    if predistort:
        Ivals = Ivals - I_o
        Qvals = Qvals - Q_o
        [Ivals, Qvals] = np.dot(predistortion_matrix, [Ivals, Qvals])
    cosI = np.array(np.cos(2 * np.pi * IF * tbase))
    sinI = np.array(np.sin(2 * np.pi * IF * tbase))
    I = np.multiply(Ivals, cosI) - np.multiply(Qvals, sinI)
    Q = np.multiply(Ivals, sinI) + np.multiply(Qvals, cosI)
    return I, Q
