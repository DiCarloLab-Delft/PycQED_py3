import os
import logging
log = logging.getLogger(__name__)
import numpy as np
from collections import OrderedDict
import h5py
import matplotlib.lines as mlines
import matplotlib
from matplotlib import pyplot as plt
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods
import pycqed.measurement.hdf5_data as h5d
from pycqed.measurement.calibration_points import CalibrationPoints
import scipy.optimize as optimize
import lmfit
import textwrap
from scipy.interpolate import interp1d
import pylab
from pycqed.analysis.tools import data_manipulation as dm_tools
import importlib
from time import time

try:
    import pygsti
except ImportError as e:
    if str(e).find('pygsti') >= 0:
        log.warning('Could not import pygsti')
    else:
        raise

from scipy.constants import *
from copy import deepcopy
from pprint import pprint
from pycqed.measurement import optimization as opt
try:
    from pycqed.analysis import machine_learning_toolbox as mlt
except: #ModuleNotFoundError:
    log.warning('Machine learning packages not loaded. '
                   'Run from pycqed.analysis import machine_learning_toolbox '
                   'to see errors.')
import pycqed.analysis.tools.plotting as pl_tools
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel,
                                            SI_prefix_and_scale_factor)


try:
    import qutip as qtp
except ImportError as e:
    if str(e).find('qutip') >= 0:
        log.warning('Could not import qutip')
    else:
        raise
importlib.reload(dm_tools)


class MeasurementAnalysis(object):

    def __init__(self, TwoD=False, folder=None, auto=True,
                 cmap_chosen='viridis', no_of_columns=1, qb_name=None, **kw):
        if folder is None:
            self.folder = a_tools.get_folder(**kw)
        else:
            self.folder = folder
        self.load_hdf5data()
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

        # params = {"ytick.color": self.tick_color,
        #           "xtick.color": self.tick_color,
        #           "axes.labelcolor": self.axes_labelcolor, }
        # plt.rcParams.update(params)

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
            print(self.savename,'::loc')
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
            log.warning('Figure "%s" has not been saved.' % self.savename)
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
            values = self.g['Data'].value[:, ind]
        elif key in self.get_key('value_names'):
            names = self.get_key('value_names')
            ind = (names.index(key) +
                   len(self.get_key('sweep_parameter_names')))
            values = self.g['Data'].value[:, ind]
        else:
            values = self.g[key].value
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

        fit_grp.attrs['Fit Report'] = \
            '\n' + '*' * 80 + '\n' + \
            lmfit.fit_report(fit_res) + \
            '\n' + '*' * 80 + '\n\n'

        fit_grp.attrs.create(name='chisqr', data=fit_res.chisqr)
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
            weighted_chisqr = np.sum(weight * (fit_res.data - fit_res.best_fit) ** 2)
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
            log.warning('Nothing to save. Parameters dictionary is empty.')
        else:
            for par_name, par_val in computed_params.items():
                if ('std' or 'stddev' or 'stderr') not in par_name:
                    try:
                        par_group = fit_grp.create_group(par_name)
                    except:  # if it already exists overwrite existing
                        par_group = fit_grp[par_name]
                    par_group.attrs.create(name=par_name, data=par_val)
                else:
                    fit_grp.attrs.create(name=par_name, data=par_val)

    def run_default_analysis(self, TwoD=False, close_file=True,
                             show=False, transpose=False,
                             plot_args=None, **kw):

        new_sweep_points = kw.pop('new_sweep_points', None)

        if plot_args is None:
            plot_args = {}
        if TwoD is False:
            self.get_naming_and_values(new_sweep_points=new_sweep_points)
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
                                                fig=fig, ax=ax,
                                                xlabel=self.sweep_name,
                                                x_unit=self.sweep_unit[0],
                                                ylabel=self.ylabels[i],
                                                save=False,
                                                **plot_args)
                # fig.suptitle(self.plot_title)
            fig.subplots_adjust(hspace=1.5)
            if show:
                plt.show()

        elif TwoD is True:
            self.get_naming_and_values_2D(new_sweep_points=new_sweep_points)
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
                    fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
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
                elif len(self.value_names) == 4 and self.no_of_columns == 2:
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

                ax.set_title(self.zlabels[i], size=self.font_size)
                ax.xaxis.label.set_size(self.font_size)
                ax.yaxis.label.set_size(self.font_size)
                ax.tick_params(labelsize=self.font_size,
                               length=self.tick_length, width=self.tick_width)
                cbar.set_label(self.zlabels[i], size=self.font_size)
                cbar.ax.tick_params(labelsize=self.font_size,
                                    length=self.tick_length,
                                    width=self.tick_width)
                if i == 0:
                    plot_title = '{measurement}\n{timestamp}'.format(
                        timestamp=self.timestamp_string,
                        measurement=self.measurementstring)
                    fig.text(0.5, 1.1, plot_title, fontsize=self.font_size,
                             horizontalalignment='center',
                             verticalalignment='bottom',
                             transform=ax.transAxes)

            fig.subplots_adjust(hspace=1.5)

            # Make space for title
            # fig.tight_layout(h_pad=1.5)
            # fig.subplots_adjust(top=3.0)
            # fig.suptitle(plot_title)
            if show:
                plt.show()

        self.save_fig(fig, fig_tight=True, **kw)

        if close_file:
            self.data_file.close()

    def get_naming_and_values(self, new_sweep_points=None):
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

        if new_sweep_points is not None:
            self.sweep_points_from_file = self.sweep_points
            self.sweep_points = new_sweep_points

        try:
            self.exp_metadata = h5d.read_dict_from_hdf5(
                {}, self.data_file['Experimental Data']['Experimental Metadata'])
        except KeyError:
            self.exp_metadata = {}

    def plot_results_vs_sweepparam(self, x, y, fig, ax, show=False, marker='-o',
                                       log=False, ticks_around=True, label=None,
                                       **kw):

        save = kw.get('save', False)
        font_size = kw.pop('font_size', None)
        cal_zero_points = kw.pop('cal_zero_points', None)
        cal_one_points = kw.pop('cal_one_points', None)
        add_half_line = kw.pop('add_half_line', False)

        if font_size is not None:
            self.font_size = font_size

        self.plot_title = kw.get('plot_title',
                                 self.measurementstring + '\n' +
                                 self.timestamp_string)

        plot_the_title = kw.get('plot_the_title', True)

        # ax.set_title(self.plot_title)
        if plot_the_title:
            fig.text(0.5, 1, self.plot_title, fontsize=self.font_size,
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=ax.transAxes)

        # Plot:
        if cal_zero_points is None and cal_one_points is None:
            line = ax.plot(x, y, marker, markersize=self.marker_size,
                           linewidth=self.line_width, label=label)

        else:
            NoCalPoints = 0
            if cal_zero_points is not None:
                ax.plot(x[cal_zero_points], y[cal_zero_points], '.k')
                ax.hlines(np.mean(y[cal_zero_points]),
                          min(x), max(x),
                          linestyles='--', color='C7')
                ax.text(np.mean(x[cal_zero_points]),
                        np.mean(y[cal_zero_points])+0.05, r'$|g\rangle$',
                        fontsize=font_size, verticalalignment='bottom',
                        horizontalalignment='center', color='k')
                NoCalPoints += len(cal_zero_points)

            if cal_one_points is not None:
                ax.plot(x[cal_one_points], y[cal_one_points], '.k')
                ax.hlines(np.mean(y[cal_one_points]),
                          min(x), max(x),
                          linestyles='--', color='C7')

                l = 'f' if kw.get("for_ef", False) else 'e'
                ax.text(np.mean(x[cal_one_points]),
                        np.mean(y[cal_one_points])-0.05,
                        r'$|{}\rangle$'.format(l),
                        fontsize=font_size, verticalalignment='top',
                        horizontalalignment='center', color='k')
                NoCalPoints += len(cal_one_points)

                # hacky way to get the e level drawn on figure
                # even if not given when doing 3-level readout with 6 calibration points
                if l == 'f' and kw.get("no_cal_points", NoCalPoints) == 6:
                    cal_points_e = np.array(cal_one_points, dtype=np.int) - 2
                    ax.plot(x[cal_points_e], y[cal_points_e], '.k')
                    ax.text(np.mean(x[cal_points_e]),
                            np.mean(y[cal_points_e]) - 0.05,
                            r'$|e\rangle$',
                            fontsize=font_size, verticalalignment='top',
                            horizontalalignment='center', color='k')
            # FIXME: issue when 6 cal points are used, because method above recovers
            #  only 4 although 6 were used. the first two cal points are then
            #  interpreted as data. therefore, for now I allow a dirty override
            #  of the parameter by parent function. Should be fixed more thoroughly !!
            NoCalPoints = kw.get("no_cal_points", NoCalPoints)

            line = ax.plot(x[:-NoCalPoints], y[:-NoCalPoints],
                           marker, markersize=self.marker_size,
                           linewidth=self.line_width, label=label)

        if add_half_line:
            ax.hlines(0.5, min(x), max(x),
                      linestyles='--', color='C7')

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
            ax.xaxis.set_tick_params(labeltop='off', top='on', direction='in')
            ax.yaxis.set_tick_params(labeltop='off', top='on', direction='in')
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
                self.save_fig(fig, xlabel=xlabel, ylabel=(ylabel + '_log'), **kw)
            else:
                self.save_fig(fig, xlabel=xlabel, ylabel=ylabel, **kw)
        if kw.get('return_line', False):
            return line
        else:
            return

    def add_textbox(self, textstring, fig=None, ax=None, **kw):

        x_pos = kw.pop('x_pos', 0.5)
        y_pos = kw.pop('y_pos ', 0)
        horizontalalignment = kw.pop('horizontalalignment', 'center')
        verticalalignment = kw.pop('verticalalignment', 'top')
        font_size = kw.pop('font_size', self.font_size)
        box_props = kw.pop('box_props', self.box_props)
        if fig is None:
            fig = self.fig
        if ax is None:
            ax = self.ax

        fig.text(x_pos, y_pos, textstring,
                 # transform=ax.transAxes,
                 fontsize=font_size,
                 verticalalignment=verticalalignment,
                 horizontalalignment=horizontalalignment,
                 bbox=box_props)


    def plot_complex_results(self, cmp_data, fig, ax, show=False, marker='.',
                             **kw):
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

    def get_naming_and_values_2D(self, new_sweep_points=None):
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
            self.ylabel = str(self.sweep_name_2D + '(' + self.sweep_unit_2D + ')')

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

            # cols = np.unique(x).shape[0]
            cols = np.nonzero(y != y[0])[0][0]

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

        if new_sweep_points is not None:
            self.sweep_points_from_file = self.sweep_points
            self.sweep_points(new_sweep_points)

        try:
            self.exp_metadata = h5d.read_dict_from_hdf5(
                {}, self.data_file['Experimental Data']['Experimental Metadata'])
        except KeyError:
            self.exp_metadata = {}

    def get_naming_and_values_2D_tuples(self):
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
            self.ylabel = str(self.sweep_name_2D + '(' +
                              self.sweep_unit_2D + ')')

        elif datasaving_format == 'Version 2':
            self.parameter_names = self.get_key('sweep_parameter_names')
            self.parameter_units = self.get_key('sweep_parameter_units')
            if len(self.parameter_names) != len(self.parameter_units):
                log.error(' Number of parameter names does not match number'
                              'of parameter units! Check sweep configuration!')
            self.sweep_names = []
            self.sweep_units = []
            for it in range(len(self.parameter_names)):
                self.sweep_names.append(self.parameter_names[it])
                self.sweep_units.append(self.parameter_units[it])
            self.value_names = self.get_key('value_names')
            self.value_units = self.get_key('value_units')
            self.data = self.get_values('Data').transpose()

            x = self.data[0]
            cols = np.unique(x).shape[0]
            nr_missing_values = 0
            if len(x) % cols != 0:
                nr_missing_values = cols - len(x) % cols
            x = np.append(x, np.zeros((1, nr_missing_values))+np.nan)
            self.X = x.reshape(-1, cols)
            self.sweep_points = self.X[0]

            self.sweep_points_2D = []
            self.Y = []
            for i in range(1,len(self.parameter_names)):
                y = self.data[i]
                y = np.append(y,np.zeros((1,nr_missing_values)))
                Y = y.reshape(-1,cols)
                self.Y.append(Y)
                self.sweep_points_2D.append(Y.T[0])

            col_idx_data_start = len(self.sweep_points_2D)+1
            if len(self.value_names) == 1:
                z = self.data[col_idx_data_start]
                z = np.append(z, np.zeros(nr_missing_values)+np.nan)
                self.Z = z.reshape(-1, cols)
                self.measured_values = [self.Z.T]
            else:
                self.Z = []
                self.measured_values = []
                for i in range(len(self.value_names)):
                    z = self.data[col_idx_data_start + i]
                    z = np.append(z, np.zeros(nr_missing_values)+np.nan)
                    Z = z.reshape(-1, cols)
                    self.Z.append(Z)
                    self.measured_values.append(Z.T)
            self.xlabel = self.parameter_names[0] + ' (' + \
                          self.parameter_units[0] + ')'
            self.ylabel = self.parameter_names[1] + ' (' + \
                          self.parameter_units[1] + ')' + '_' + \
                          self.parameter_names[2] + ' (' + \
                          self.parameter_units[2] + ')'

            self.parameter_labels = [a + ' (' + b + ')' for a, b in zip(
                self.parameter_names,
                self.parameter_units)]
            self.zlabels = [a + ' (' + b + ')' for a, b in zip(self.value_names,
                                                               self.value_units)]

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
                chisquares = {k: v for (k, v) in list(
                    normalized_chisquares.items()) if k in haspeak_lst}
                best_key = min(chisquares, key=normalized_chisquares.get)
            else:
                best_key = min(normalized_chisquares,
                               key=normalized_chisquares.get)
            print('Best key: ', best_key)
            best_fit_results = self.data_file['Analysis'][best_key]
            return best_fit_results

    def set_sweep_points(self, sweep_points):
        assert (len(sweep_points) == len(self.sweep_points))
        self.sweep_points = sweep_points


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
                        self.sweep_points[1], '-o', c='grey')
                ax.plot(self.sweep_points[0][-1], self.sweep_points[1][-1],
                        'o', markersize=5, c='w')
                plot_title = kw.pop('plot_title', textwrap.fill(
                    self.timestamp_string + '_' +
                    self.measurementstring, 40))
                ax.set_title(plot_title)
                set_xlabel(ax, self.parameter_names[0], self.parameter_units[0])
                set_ylabel(ax, self.parameter_names[1], self.parameter_units[1])
                self.save_fig(f, figname=base_figname, **kw)


class OptimizationAnalysisNN(MeasurementAnalysis):
    def __init__(self,folder=None,**kw):
        super().__init__(folder=folder,**kw)

    def run_default_analysis(self, close_file=True, show=False, plot_all=False,
                             **kw):

        self.get_naming_and_values()
        try:
            optimization_method = eval(self.data_file['Instrument settings']\
                ['MC'].attrs['optimization_method'])
        except KeyError:
            optimization_method = 'Numerical'
        self.meas_grid = kw.pop('meas_grid')
        self.hyper_parameter_dict = kw.pop('hyper_parameter_dict',None)
        self.two_rounds = kw.pop('two_rounds',False)
        self.round = kw.pop('round',1)
        self.estimator_name = kw.pop('estimator','GRNN_neupy')

        self.accuracy= -np.infty
        self.make_fig = kw.pop('make_fig',True)

        self.train_NN(**kw)

        if self.round > int(self.two_rounds) or self.round == 0:
            #only create figures in the last iteration
            if self.make_fig:
                self.make_figures(**kw)
        if close_file:
            self.data_file.close()
        return self.optimization_result

    def train_NN(self, **kw):
        if np.size(self.measured_values,0) == 1:
            self.abs_vals = deepcopy(self.measured_values[0,:])
        else:
            self.abs_vals = np.sqrt(self.measured_values[0,:]**2 +
                                    self.measured_values[1,:]**2)
        result,est,opti_flag = opt.neural_network_opt(
            None, self.meas_grid, target_values=np.array([self.abs_vals]).T,
            estimator=self.estimator_name,
            hyper_parameter_dict=self.hyper_parameter_dict)
        #test_grid and test_target values. Centered and scaled to [-1,1] since
        #only used for performance estimation of estimator
        self.opti_flag = opti_flag
        self.estimator = est
        # self.accuracy = est.evaluate(self.test_grid,self.test_target)
        self.optimization_result = result

        return result,est

    def make_figures(self, **kw):

        fontsize = kw.pop('label_fontsize',16.)
        try:
            optimization_method = eval(self.data_file['Instrument settings'] \
                ['MC'].attrs['optimization_method'])
        except KeyError:
            optimization_method = 'Numerical'

        pre_proc_dict = self.estimator.pre_proc_dict
        output_scale = pre_proc_dict.get('output',{}).get('scaling',1.)
        output_means = pre_proc_dict.get('output',{}).get('centering',0.)
        input_scale = pre_proc_dict.get('input',{}).get('scaling',1.)
        input_means = pre_proc_dict.get('input',{}).get('centering',0.)
        # for i in range(self.test_grid.ndim):
        #     self.test_grid[:,i] = self.test_grid[:,i]*input_scale[i] + input_means[i]

            #create contour plot
        fig1 = plt.figure(figsize=(10,8))
        #Create data grid for contour plot
        lower_x = np.min(self.sweep_points[0])-0.2*np.std(self.sweep_points[0])
        upper_x = np.max(self.sweep_points[0])+0.2*np.std(self.sweep_points[0])
        lower_y = np.min(self.sweep_points[1])-0.2*np.std(self.sweep_points[1])
        upper_y = np.max(self.sweep_points[1])+0.2*np.std(self.sweep_points[1])
        x_mesh = (np.linspace(lower_x,upper_x,200)-input_means[0])/input_scale[0]
        y_mesh = (np.linspace(lower_y,upper_y,200)-input_means[1])/input_scale[1]
        Xm,Ym = np.meshgrid(x_mesh,y_mesh)
        Zm = np.zeros_like(Xm)
        for k in range(np.size(x_mesh)):
            for l in range(np.size(y_mesh)):
                Zm[k,l] = self.estimator.predict([[Xm[k,l],Ym[k,l]],])
        Zm = Zm*output_scale + output_means
        Xm = Xm*input_scale[0] + input_means[0]
        Ym = Ym*input_scale[1] + input_means[1]
        #Landscape plot of network
        #In case we use tensorflow, add a learning curve plot
        if self.estimator_name=='DNN_Regressor_tf':
            plt_grid = plt.GridSpec(2,10,hspace=0.6)
            ax1 = plt.subplot(plt_grid[0,:])
            ax2 = plt.subplot(plt_grid[1,:8])
            textstr2 = 'Accuracy on test data: %s \n'% np.round(self.accuracy,3)
            textstr2 +='Accuracy on training data, last epoch: %s' \
                        % np.round(self.estimator.learning_acc[-1],3)
            ax2.text(0.95, 0.1, textstr2,
                     transform=ax2.transAxes,
                     fontsize=11, verticalalignment='bottom',
                     horizontalalignment='right',
                     bbox=dict(facecolor='white',edgecolor='black'))
            ax2.set_title('Learning_curve',fontsize=fontsize)
            learning_acc = np.array(self.estimator.learning_acc)
            ax2.plot(learning_acc[:,0],
                     learning_acc[:,1],
                     'g-',
                     linewidth=3)
            ax2.set_ylabel('Coefficient of determination $R^2$'
                           ,fontsize=fontsize)
            ax2.set_xlabel('learning epoch',fontsize=fontsize)
            ax2.grid(True)
        else:
            ax1 = plt.subplot(111)
        base_figname = optimization_method + ' optimization of '
        figname1 = self.timestamp_string+'_'
        for i,meas_vals in enumerate(self.measured_values):
            figname1 += self.value_names[i]
            figname1 += ';'
            base_figname += self.value_names[i]
        textstr = 'Optimization converged to: \n'
        base_figname += '_it_'+str(self.round)
        for i in range(len(self.parameter_names)):
               textstr+='%s: %.3g %s' % (self.parameter_names[i],
                                         self.optimization_result[i],
                                         self.parameter_units[i])
               textstr+='\n'
        textstr+='Empirical error: '+'%.2f' % ((1.-self.estimator.score)*100.) +'%'
        ax1.text(0.98, 0.05, textstr,
             transform=ax1.transAxes,
             fontsize=11, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white',edgecolor='None',
             alpha=0.75, boxstyle='round'))
        figname1+= self.estimator_name+' fitted landscape'
        savename1 = self.timestamp_string + '_' + base_figname

        levels = np.linspace(np.min(Zm),np.max(Zm),30)
        CP = ax1.contourf(Xm,Ym,Zm,levels,extend='both')
        ax1.scatter(self.optimization_result[0],self.optimization_result[1],
                 marker='o',c='white',label='network minimum')
        ax1.scatter(self.sweep_points[0],self.sweep_points[1],
                 marker='o',c='r',label='training data',s=10)
        ax1.tick_params(axis='both',which='minor',labelsize=14)
        ax1.set_ylabel(self.parameter_labels[1],fontsize=fontsize)
        ax1.set_xlabel(self.parameter_labels[0],fontsize=fontsize)
        cbar = plt.colorbar(CP,ax=ax1,orientation='vertical')
        cbar.ax.set_ylabel(self.ylabels[0],fontsize=fontsize)
        ax1.legend(loc='upper left',framealpha=0.75,fontsize=fontsize)
        ax1.set_title(figname1)
        self.save_fig(fig1,figname=savename1,**kw)

        #interpolation plot with only measurement points
        base_figname = 'optimization of '
        for i in range(len(self.value_names)):
            base_figname+= self.value_names[i]
        base_figname += '_it_'+str(self.round)
        if np.shape(self.sweep_points)[0] == 2:
            f, ax = plt.subplots()
            a_tools.color_plot_interpolated(
                x=self.sweep_points[0], y=self.sweep_points[1],
                z=self.abs_vals, ax=ax,N_levels=25,
                zlabel=self.ylabels[0])
            ax.plot(self.sweep_points[0],
                    self.sweep_points[1], 'o', c='grey')
            ax.plot(self.optimization_result[0],
                    self.optimization_result[1],
                    'o', markersize=5, c='w')
            plot_title = self.timestamp_string + '_' +self.measurementstring
            ax.set_title(plot_title)
            textstr = '%s ( %s )' % (self.parameter_names[0],
                                     self.parameter_units[0])
            set_xlabel(ax, textstr)
            textstr = '%s ( %s )' % (self.parameter_names[1],
                                     self.parameter_units[1])
            set_ylabel(ax, textstr)
            self.save_fig(f, figname=base_figname, **kw)


class OptimizationAnalysis(MeasurementAnalysis):

    def run_default_analysis(self, close_file=True, show=False, plot_all=False, **kw):
        self.get_naming_and_values()
        try:
            optimization_method = eval(self.data_file['Instrument settings'] \
                ['MC'].attrs['optimization_method'])
        except KeyError:
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
                axarray[i].plot(self.sweep_points[i], self.measured_values[0],
                                linestyle='--', c='k')
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
            axarray.plot(self.sweep_points, self.measured_values[0],
                         linestyle='--', c='k')
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
                 for_ef=False, qb_name=None, RO_channels=(0, 1), **kw):
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
        self.RO_channels = RO_channels

        super().__init__(qb_name=qb_name, **kw)

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
        self.corr_data = a_tools.rotate_and_normalize_data_1ch(
            self.measured_values[0],
            cal_zero_points=cal_zero_points,
            cal_one_points=cal_one_points)

    def run_default_analysis(self,
                             close_main_fig=True,
                             show=False, **kw):

        save_fig = kw.pop('save_fig', True)
        close_file = kw.pop('close_file', True)
        add_half_line = kw.pop('add_half_line', False)
        super().run_default_analysis(show=show,
            close_file=False, **kw)

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
            plot_title_suffix = kw.get('plot_title_suffix', '')
            self.fig, self.ax = self.default_ax()

            if self.for_ef:
                ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
            else:
                # ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'
                ylabel = r'$F$ $|1 \rangle$'

            plot_title = kw.pop('plot_title', self.measurementstring
                                + plot_title_suffix + '\n' +
                               self.timestamp_string)
            self.plot_results_vs_sweepparam(
                x=self.sweep_points,
                y=self.normalized_values,
                cal_zero_points=self.cal_zero_points,
                cal_one_points=self.cal_one_points,
                fig=self.fig, ax=self.ax,
                xlabel=self.sweep_name,
                x_unit=self.sweep_unit[0],
                ylabel=ylabel,
                marker='o-',
                save=False,
                plot_title=plot_title,
                add_half_line=add_half_line,
                no_cal_points=self.NoCalPoints,
                for_ef=self.for_ef)
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
        print('RO_channels ', self.RO_channels)
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
        if (calsteps == 6) and (not last_ge_pulse):
            # ONLY WORKS FOR SINGLE QUBIT MEASUREMENTS
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
            self.corr_data = a_tools.rotate_and_normalize_data_IQ(
                data=self.measured_values[0:2],
                zero_coord=zero_coord, one_coord=one_coord)[0]
        else:
            if len(self.measured_values) == 1 or len(self.RO_channels) == 1:
                # Only one quadrature was measured
                if cal_zero_points is None and cal_one_points is None:
                    # a_tools.rotate_and_normalize_data_1ch does not work
                    # with 0 cal_points. Use 4 cal_points.
                    log.warning('a_tools.rotate_and_normalize_data_1ch '
                                    'does not have support for 0 cal_points. '
                                    'Setting NoCalPoints to 4.')
                    self.NoCalPoints = 4
                    calsteps = 4
                    cal_zero_points = \
                        list(range(NoPts - int(self.NoCalPoints),
                                   int(NoPts - int(self.NoCalPoints) / 2)))
                    cal_one_points = \
                        list(range(int(NoPts - int(self.NoCalPoints) / 2),
                                   NoPts))
                ch_to_measure = \
                    0 if len(self.measured_values) == 1 else self.RO_channels[0]
                log.debug('ch to measure ', ch_to_measure)
                self.corr_data = a_tools.rotate_and_normalize_data_1ch(
                    self.measured_values[ch_to_measure],
                    cal_zero_points, cal_one_points)
            else:
                self.corr_data = a_tools.rotate_and_normalize_data_IQ(
                    self.measured_values[
                    self.RO_channels[0]:self.RO_channels[1] + 1],
                    cal_zero_points,
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

        self.cal_one_points = cal_one_points
        self.cal_zero_points = cal_zero_points

        return [normalized_values, normalized_data_points, normalized_cal_vals]

    def fit_data(*kw):
        '''
        Exists to be able to include it in the TD_Analysis run default
        '''
        pass


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
        NoCalPoints       (default=4)
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
                                   vary=False)
            cos_mod.set_param_hint('phase',
                                   value=0,
                                   vary=False)
            cos_mod.set_param_hint('frequency',
                                   value=freq_guess,
                                   vary=True,
                                   min=(1 / (100 * self.sweep_pts_wo_cal_pts[-1])),
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
            if (self.fit_result.chisqr > .35) or (init_data_diff > offset_guess):
                log.warning('Fit did not converge, varying phase.')

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
                log.warning(e)

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

                params = fit_mods.Cos_guess(model, data=data, t=self.sweep_points)
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
                    log.warning(e)

            if print_fit_results:
                for fit_res in self.fit_res:
                    print(fit_res.fit_report())

    def run_default_analysis(self, show=False,
                             close_file=False, **kw):

        super().run_default_analysis(show=show,
                                     close_file=close_file,
                                     close_main_fig=True,
                                     save_fig=False, **kw)

        show_guess = kw.get('show_guess', False)
        plot_amplitudes = kw.get('plot_amplitudes', True)
        plot_errorbars = kw.get('plot_errorbars', False)
        print_fit_results = kw.get('print_fit_results', False)
        separate_fits = kw.get('separate_fits', False)
        plot_title_suffix= kw.get('plot_title_suffix', '')

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
                              separate_fits=separate_fits,
                              plot_title_suffix=plot_title_suffix)

        if close_file:
            self.data_file.close()

        return self.fit_result

    def make_figures(self, show=False, show_guess=False, plot_amplitudes=True,
                     plot_errorbars=True, separate_fits=False, **kw):

        plot_title_suffix = kw.pop('plot_title_suffix', '')

        if not separate_fits:
            pi_pulse = self.rabi_amplitudes['piPulse']
            pi_half_pulse = self.rabi_amplitudes['piHalfPulse']

            # Get previously measured values from the data file
            instr_set = self.data_file['Instrument settings']
            try:
                if self.for_ef:
                    pi_pulse_old = eval(
                        instr_set[self.qb_name].attrs['amp180_ef'])
                    pi_half_pulse_old = \
                        pi_pulse_old * eval(
                            instr_set[self.qb_name].attrs['amp90_scale_ef'])
                else:
                    pi_pulse_old = eval(instr_set[self.qb_name].attrs['amp180'])
                    pi_half_pulse_old = \
                        pi_pulse_old * eval(
                            instr_set[self.qb_name].attrs['amp90_scale'])
                old_vals = '\n  $\pi-Amp_{old}$ = %.3g ' % (pi_pulse_old) + \
                           self.parameter_units[0] + \
                           '\n$\pi/2-Amp_{old}$ = %.3g ' % (pi_half_pulse_old) + \
                           self.parameter_units[0]
            except(TypeError, KeyError, ValueError):
                log.warning('qb_name is None. Default value qb_name="qb" is '
                                'used. Old parameter values will not be retrieved.')
                old_vals = ''

            textstr = ('  $\pi-Amp$ = %.3g ' % (pi_pulse) + self.parameter_units[0] +
                       ' $\pm$ (%.3g) ' % (self.rabi_amplitudes['piPulse_std']) +
                       self.parameter_units[0] +
                       '\n$\pi/2-Amp$ = %.3g ' % (pi_half_pulse) +
                       self.parameter_units[0] +
                       ' $\pm$ (%.3g) ' % (self.rabi_amplitudes['piHalfPulse_std']) +
                       self.parameter_units[0] + old_vals)

            self.add_textbox(textstr, fig=self.fig, ax=self.ax)
            # self.fig.text(0.5, 0, textstr,
            #               transform=self.ax.transAxes, fontsize=self.font_size,
            #               verticalalignment='top',
            #               horizontalalignment='center', bbox=self.box_props)

            # Used for plotting the fit (line 1776)
            best_vals = self.fit_result.best_values
            cos_fit_func = lambda a: fit_mods.CosFunc(a,
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
                             self.fit_result.init_fit, 'k--',
                             linewidth=self.line_width)

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
            self.save_fig(self.fig, figname=(self.measurementstring +
                                            '_Rabi_fit' + plot_title_suffix),
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
                log.info('The data could not be fitted correctly. '
                             'The fitted phase "%s" <1, which gives '
                             'negative piHalf '
                             'amplitude.' % phase_fit)

            stepsize = self.sweep_points[1] - self.sweep_points[0]
            # Nyquist: wavelength>2*stepsize
            if (freq_fit) > 2 * stepsize:
                log.info('The data could not be fitted correctly. The '
                             'frequency "%s" is too high.' % freq_fit)

            # Extract pi and pi/2 amplitudes from best fit values
            # if phase_fit == 0:
            #     piPulse = 1/(2*freq_fit)
            #     piHalfPulse = 1/(4*freq_fit)
            #     piPulse_std = freq_std/(2*freq_fit**2)
            #     piHalfPulse_std = freq_std/(4*freq_fit**2)
            #
            # else:
            n = np.arange(-2, 10)

            piPulse_vals = (n*np.pi+phase_fit)/(2*np.pi*freq_fit)
            piHalfPulse_vals = (n*np.pi+np.pi/2+phase_fit)/(2*np.pi*freq_fit)

            # find piHalfPulse
            try:
                piHalfPulse = \
                    np.min(piHalfPulse_vals[piHalfPulse_vals >= self.sweep_points[1]])
                n_piHalf_pulse = n[piHalfPulse_vals==piHalfPulse]
            except ValueError:
                piHalfPulse = np.asarray([])

            if piHalfPulse.size==0 or piHalfPulse>max(self.sweep_points):
                i=0
                while (piHalfPulse_vals[i]<min(self.sweep_points) and
                               i<piHalfPulse_vals.size):
                    i+=1
                piHalfPulse = piHalfPulse_vals[i]
                n_piHalf_pulse = n[i]

            # find piPulse
            try:
                if piHalfPulse.size != 0:
                    piPulse = \
                        np.min(piPulse_vals[piPulse_vals>=piHalfPulse])
                else:
                    piPulse = np.min(piPulse_vals[piPulse_vals>=0.001])
                n_pi_pulse = n[piHalfPulse_vals==piHalfPulse]

            except ValueError:
                piPulse = np.asarray([])

            if piPulse.size==0: #or piPulse>max(self.sweep_points):
                i=0
                while (piPulse_vals[i]<min(self.sweep_points) and
                               i<piPulse_vals.size):
                    i+=1
                piPulse = piPulse_vals[i]
                n_pi_pulse = n[i]

            # piPulse = 1/(2*freq_fit) - phase_fit/(2*np.pi*freq_fit)
            # piHalfPulse = 1/(4*freq_fit) - phase_fit/(2*np.pi*freq_fit)

            #Calculate std. deviation for pi and pi/2 amplitudes based on error
            # propagation theory
            #(http://ugastro.berkeley.edu/infrared09/PDF-2009/statistics1.pdf)
            #Errors were assumed to be uncorrelated.

            #extract cov(phase,freq)
            try:
                freq_idx = self.fit_result.var_names.index('frequency')
                phase_idx = self.fit_result.var_names.index('phase')
                if self.fit_result.covar is not None:
                    cov_freq_phase = self.fit_result.covar[freq_idx, phase_idx]
                else:
                    cov_freq_phase = 0
            except ValueError:
                cov_freq_phase = 0

            piPulse_std = self.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_num=n_pi_pulse,
                cov=cov_freq_phase)
            piHalfPulse_std = self.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_num=n_piHalf_pulse,
                cov=cov_freq_phase)

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
            log.warning("Fitted frequency is zero. The pi-pulse and "
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

    def calculate_pulse_stderr(self, f, phi, f_err, phi_err,
                               period_num, cov=0):
        x = period_num + phi
        return np.sqrt((f_err*x/(2*np.pi*(f**2)))**2 +
                       (phi_err/(2*np.pi*f))**2 -
                       2*(cov**2)*x/((2*np.pi*(f**3))**2))


class Echo_analysis(TD_Analysis):

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
                             t=self.sweep_points[:-self.NoCalPoints])
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

        scale_factor, unit = SI_prefix_and_scale_factor(
            self.fit_res.params['tau'].value, self.parameter_units[0])
        textstr = '$T_2$={:.3g}$\pm$({:.3g}) {} '.format(
            self.fit_res.params['tau'].value * scale_factor,
            self.fit_res.params['tau'].stderr * scale_factor,
            unit)
        if show_guess:
            self.ax.plot(x_fine, self.fit_res.eval(
                t=x_fine, **self.fit_res.init_values), label='guess')
            self.ax.legend(loc='best')

        self.ax.text(0.4, 0.95, textstr, transform=self.ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=self.box_props)
        self.save_fig(self.fig, fig_tight=True, **kw)


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
                                     close_main_fig=True,
                                     save_fig=False, **kw)

        plot_title_suffix = kw.pop('plot_title_suffix', '')

        # Only the unfolding part here is unique to this analysis
        self.sweep_points_xX = self.sweep_points[
                               :len(self.sweep_points)-self.NoCalPoints:3]
        self.sweep_points_xY = self.sweep_points[
                               1:len(self.sweep_points)-self.NoCalPoints:3]
        self.sweep_points_xmY = self.sweep_points[
                                2:len(self.sweep_points)-self.NoCalPoints:3]
        self.corr_data_xX = self.normalized_values[
                            :len(self.sweep_points)-self.NoCalPoints:3]
        self.corr_data_xY = self.normalized_values[
                            1:len(self.sweep_points)-self.NoCalPoints:3]
        self.corr_data_xmY = self.normalized_values[
                             2:len(self.sweep_points)-self.NoCalPoints:3]

        self.fit_data(**kw)

        self.calculate_optimal_qscale(**kw)
        self.save_computed_parameters(self.optimal_qscale,
                                      var_name=self.value_names[0])

        if self.make_fig_qscale:
            fig, ax = self.default_ax()
            self.make_figures(fig=fig, ax=ax,
                              plot_title_suffix =plot_title_suffix, **kw)

            if show:
                plt.show()

            if kw.pop('save_fig', True):
                self.save_fig(fig,
                              figname=self.measurementstring + '_Qscale_fit' +
                                      plot_title_suffix, **kw)

        if close_file:
            self.data_file.close()

        return self.fit_res

    def make_figures(self, fig=None, ax=None,
                     plot_title_suffix ='', **kw):

        # Unique in that it has hardcoded names and points to plot
        show_guess = kw.pop('show_guess', False)

        x_fine = np.linspace(
            min(self.sweep_points[:len(self.sweep_points)-self.NoCalPoints]),
            max(self.sweep_points[:len(self.sweep_points)-self.NoCalPoints]),
            1000)

        # Get old values
        instr_set = self.data_file['Instrument settings']
        try:
            if self.for_ef:
                qscale_old = eval(instr_set[self.qb_name].attrs['motzoi_ef'])
            else:
                qscale_old = eval(instr_set[self.qb_name].attrs['motzoi'])
            old_vals = '\n$qscale_{old} = $%.5g' % (qscale_old)
        except (TypeError, KeyError, ValueError):
            log.warning('qb_name is None. Old parameter values will '
                            'not be retrieved.')
            old_vals = ''

        textstr = ('qscale = %.5g $\pm$ %.5g'
                   % (self.optimal_qscale['qscale'],
                      self.optimal_qscale['qscale_std']) + old_vals)

        if self.for_ef:
            ylabel = r'$F$ $\left(|f \rangle \right) (arb. units)$'
        else:
            ylabel = r'$F$ $\left(|e \rangle \right) (arb. units)$'

        self.add_textbox(textstr, fig=fig, ax=ax)

        plot_title = kw.pop('plot_title', self.measurementstring
                            + plot_title_suffix + '\n' +
                            self.timestamp_string)

        self.plot_results_vs_sweepparam(self.sweep_points_xX, self.corr_data_xX,
                                        fig, ax,
                                        marker='ob',
                                        label=r'$X_{\frac{\pi}{2}}X_{\pi}$',
                                        ticks_around=True,
                                        plot_title='')
        self.plot_results_vs_sweepparam(self.sweep_points_xY, self.corr_data_xY,
                                        fig, ax,
                                        marker='og',
                                        label=r'$X_{\frac{\pi}{2}}Y_{\pi}$',
                                        ticks_around=True,
                                        plot_title='')
        self.plot_results_vs_sweepparam(self.sweep_points_xmY,
                                        self.corr_data_xmY, fig, ax,
                                        marker='or',
                                        label=r'$X_{\frac{\pi}{2}}Y_{-\pi}$',
                                        ticks_around=True,
                                        xlabel=r'$q_{scales}$',
                                        ylabel=ylabel,
                                        plot_title=plot_title)
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
        if (b_vals0['c'] > (0.5 + threshold)) or (b_vals0['c'] < 
                                                  (0.5 - threshold)):
            log.warning('The trace from the X90-X180 pulses is NOT within '
                            '+/-%s of the expected value of 0.5.' % threshold)
        # Warning if optimal_qscale is not within +/-threshold of 0.5
        optimal_qscale_pop = optimal_qscale * b_vals2['slope'] + \
                             b_vals2['intercept']
        if (optimal_qscale_pop > (0.5 + threshold)) or \
                (optimal_qscale_pop < (0.5 - threshold)):
            log.warning('The optimal qscale found gives a population '
                            'that is NOT within +/-%s of the expected value '
                            'of 0.5.' % threshold)

        intercept_diff_mean = self.fit_res[1].params['intercept'].value - \
                              self.fit_res[2].params['intercept'].value
        slope_diff_mean = self.fit_res[2].params['slope'].value - \
                          self.fit_res[1].params['slope'].value

        intercept_diff_std = \
            np.sqrt((self.fit_res[1].params['intercept'].stderr)**2 + \
            (self.fit_res[2].params['intercept'].stderr)**2)
        slope_diff_std = \
            np.sqrt((self.fit_res[2].params['slope'].stderr)**2 + \
            (self.fit_res[1].params['slope'].stderr)**2)

        optimal_qscale_stddev = np.sqrt(
            (intercept_diff_std/slope_diff_mean)**2 +
            (intercept_diff_mean*slope_diff_std/(slope_diff_std**2))**2)

        if print_parameters:
            print('Optimal QScale Parameter = {} \t QScale Stddev = {}'.format(
                optimal_qscale, optimal_qscale_stddev))

        # return as dict for use with "save_computed_parameters"
        self.optimal_qscale = {'qscale': optimal_qscale,
                               'qscale_std': optimal_qscale_stddev}

        return self.optimal_qscale


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

        log.warning('The use of this class is deprectated!' +
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
        self.pge = pge #fixed fraction of ground state in the excited state histogram (relaxation)
        self.peg = peg #fixed fraction of excited state in the ground state hitogram (residual population)
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

    def no_fits_analysis(self, shots_I_1_rot, shots_I_0_rot, min_len,
                         masked=False, **kw):

        plot = kw.get('plot', True)
        show = kw.get('show', False)

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
            self.save_fig(fig, figname=filename, close_fig=self.close_fig, **kw)
            if show:
                plt.show()
            else:
                plt.clf()

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
        show = kw.get('show', False)

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
            t = 1. / (1. + 0.5*z)
            r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196 +
                    t*(.09678418+t*(-.18628806+t*(.27886807 +
                        t*(-1.13520398+t*(1.48851587+t*(-.82215223 +
                            t*.17087277)))))))))
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
        if self.pge==None:
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
        if self.peg==None:
            NormCdf2Model.set_param_hint(
                'frac1', value=0.025, min=0, max=1, vary=True)
        else:
            NormCdf2Model.set_param_hint(
                'frac1', value=self.peg,vary=False)

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
        p_rem = 0

        if plot:
            # plotting s-curves
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title('S-curves (not binned) and fits, determining fidelity '
                         '\nand threshold optimum, %s shots' % min_len)
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
            if show:
                plt.show()
            else:
                plt.clf()

            # plotting the histograms
            fig, axes = plt.subplots(figsize=(10, 4))
            n1, bins1, patches = pylab.hist(shots_I_1_rot, bins=40,
                                            label='1 I', histtype='step',
                                            color='red', normed=False)
            n0, bins0, patches = pylab.hist(shots_I_0_rot, bins=40,
                                            label='0 I', histtype='step',
                                            color='blue', normed=False)
            pylab.clf()
            # n0, bins0 = np.histogram(shots_I_0_rot, bins=int(min_len/50),
            #                          normed=1)
            # n1, bins1 = np.histogram(shots_I_1_rot, bins=int(min_len/50),
            #                          normed=1)

            gdat, = pylab.plot(bins0[:-1] + 0.5 * (bins0[1] - bins0[0]), n0, 'C0o')
            edat, = pylab.plot(bins1[:-1] + 0.5 * (bins1[1] - bins1[0]), n1, 'C3o')

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

            y0 = norm0 * (1 - frac1_0) * pylab.normpdf(bins0, mu0_0, sigma0_0) + \
                 norm0 * frac1_0 * pylab.normpdf(bins0, mu1_0, sigma1_0)
            y1_0 = norm0 * frac1_0 * pylab.normpdf(bins0, mu1_0, sigma1_0)
            y0_0 = norm0 * (1 - frac1_0) * pylab.normpdf(bins0, mu0_0, sigma0_0)

            # building up the histogram fits for on measurements
            y1 = norm1 * (1 - frac1_1) * pylab.normpdf(bins1, mu0_1, sigma0_1) + \
                 norm1 * frac1_1 * pylab.normpdf(bins1, mu1_1, sigma1_1)
            y1_1 = norm1 * frac1_1 * pylab.normpdf(bins1, mu1_1, sigma1_1)
            y0_1 = norm1 * (1 - frac1_1) * pylab.normpdf(bins1, mu0_1, sigma0_1)

            pylab.semilogy(bins0, y0, 'C0', linewidth=1.5)
            pylab.semilogy(bins0, y1_0, 'C0--', linewidth=3.5)
            pylab.semilogy(bins0, y0_0, 'C0--', linewidth=3.5)

            pylab.semilogy(bins1, y1, 'C3', linewidth=1.5)
            pylab.semilogy(bins1, y0_1, 'C3--', linewidth=3.5)
            pylab.semilogy(bins1, y1_1, 'C3--', linewidth=3.5)
            pdf_max = (max(max(y0), max(y1)))
            (pylab.gca()).set_ylim(pdf_max / 1000, 2 * pdf_max)

            plt.title('Histograms of {} shots\n{}-{}'.format(
                min_len, self.timestamp_string, self.qb_name))
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
                       # '$V_{\mathrm{th}}$ = ' + '{:.4f} V'.format(self.V_th_a),
                       'SNR = {:.2f}'.format(SNR),
                       '$p(e|0)$ = {:.4f}'.format(frac1_0),
                       '$p(g|\pi)$ = {:.4f}'.format(1-frac1_1)]
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
            if show:
                plt.show()
            else:
                plt.clf()

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
        fid_grp.attrs.create(name='p_rem', data=p_rem)

        self.sigma0_0 = sigma0_0
        self.sigma1_1 = sigma1_1
        self.mu0_0 = mu0_0
        self.mu1_1 = mu1_1
        self.frac1_0 = frac1_0
        self.frac1_1 = frac1_1
        self.F_d = F_d
        self.SNR = SNR

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
                                              normed=True)
        H1, xedges1, yedges1 = np.histogram2d(shots_I_1, shots_Q_1,
                                              bins=n_bins,
                                              range=[[I_min, I_max, ],
                                                     [Q_min, Q_max, ]],
                                              normed=True)

        fig, axarray = plt.subplots(nrows=1, ncols=2)
        axarray[0].tick_params(axis='both', which='major',
                               labelsize=5, direction='out')
        axarray[1].tick_params(axis='both', which='major',
                               labelsize=5, direction='out')

        plt.subplots_adjust(hspace=20)

        axarray[0].set_title('2D histogram, pi pulse')
        im1 = axarray[0].imshow(np.transpose(H1), interpolation='nearest',
                                origin='low',
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
                                origin='low',
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
                                              value=np.max(
                                                  self.normalized_data_points),
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

        fit_res = fit_mods.ExpDecayModel.fit(
            data=self.normalized_data_points,
            t=self.sweep_points[:len(self.sweep_points) - self.NoCalPoints],
            params=self.params)

        if kw.get('print_fit_results', False):
            print(fit_res.fit_report())

        return fit_res

    def run_default_analysis(self, show=False, close_file=False, **kw):

        super().run_default_analysis(show=show,
                                     close_file=close_file,
                                     close_main_fig=True,
                                     save_fig=False, **kw)

        show_guess = kw.get('show_guess', False)
        # make_fig = kw.get('make_fig',True)

        self.add_analysis_datagroup_to_file()

        # Perform fit and save fitted parameters
        self.fit_res = self.fit_T1(**kw)
        self.save_fitted_parameters(fit_res=self.fit_res, var_name='F|1>')

        # Create self.T1 and self.T1_stderr and save them
        self.get_measured_T1()  # in seconds
        self.save_computed_parameters(self.T1_dict, var_name=self.value_names[0])

        T1_micro_sec = self.T1_dict['T1'] * 1e6
        T1_err_micro_sec = self.T1_dict['T1_stderr'] * 1e6
        # Print T1 and error on screen
        if kw.get('print_parameters', False):
            print('T1 = {:.5f} ('.format(T1_micro_sec) + 'μs) \t '
                                                         'T1 StdErr = {:.5f} ('.format(
                T1_err_micro_sec) + 'μs)')

        # Plot best fit and initial fit + data
        if self.make_fig:
            plot_title_suffix = kw.pop('plot_title_suffix', '')

            units = SI_prefix_and_scale_factor(val=max(abs(self.ax.get_xticks())),
                                               unit=self.sweep_unit[0])[1]
            # Get old values
            instr_set = self.data_file['Instrument settings']
            try:
                if self.for_ef:
                    T1_old = eval(instr_set[self.qb_name].attrs['T1_ef']) * 1e6
                else:
                    T1_old = eval(instr_set[self.qb_name].attrs['T1']) * 1e6
                old_vals = '\nold $T_1$ = {:.5f} '.format(T1_old) + units
            except (TypeError, KeyError, ValueError):
                log.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
                old_vals = ''

            textstr = ('$T_1$ = {:.5f} '.format(T1_micro_sec) +
                       units +
                       ' $\pm$ {:.5f} '.format(T1_err_micro_sec) +
                       units + old_vals)
            self.add_textbox(textstr, fig=self.fig, ax=self.ax)

            if show_guess:
                self.ax.plot(
                    self.sweep_points[:len(self.sweep_points)-self.NoCalPoints],
                    self.fit_res.init_fit, 'k--', linewidth=self.line_width)

            best_vals = self.fit_res.best_values
            t = np.linspace(self.sweep_points[0],
                            self.sweep_points[-self.NoCalPoints-1], 1000)

            y = fit_mods.ExpDecayFunc(
                t, tau=best_vals['tau'],
                n=best_vals['n'],
                amplitude=best_vals['amplitude'],
                offset=best_vals['offset'])

            self.ax.plot(t, y, 'r-', linewidth=self.line_width)

            self.ax.locator_params(axis='x', nbins=6)

            if show:
                plt.show()

            self.save_fig(self.fig, figname=self.measurementstring + '_Fit'+
                                            plot_title_suffix, **kw)

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
            log.warning('Artificial detuning is unknown. Defaults to %s MHz. '
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

        if 'add_half_line' not in kw:
            kw['add_half_line'] = True

        super(Ramsey_Analysis, self).__init__(**kw)

    def fit_Ramsey(self, x, y, **kw):

        print_fit_results = kw.pop('print_fit_results',False)
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
                                          value=0.5*(max(y)-min(y)),
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

        amplitude_guess = 0.5
        if np.all(np.logical_and(y>0, y<1)):
            damped_osc_mod.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.4,
                                          max=4.0,
                                          vary=False)
        else:
            print('data is not normalized, varying amplitude')
            damped_osc_mod.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.4,
                                          max=4.0,
                                          vary=False)
        damped_osc_mod.set_param_hint('tau',
                                      value=x[1]*10,
                                      min=x[1],
                                      max=x[1]*1000)
        damped_osc_mod.set_param_hint('exponential_offset',
                                      value=0.5,
                                      min=0.4,
                                      max=4.0,
                                      vary=False)
        damped_osc_mod.set_param_hint('oscillation_offset',
                                      # expr=
                                      # '{}-amplitude-exponential_offset'.format(
                                      #     y[0]))
                                      value=0,
                                      vary=False)

        self.fit_results_dict = {}
        decay_labels = ['gaussian', 'exponential', ]
        for label, n in zip(decay_labels, [2,1]):
            damped_osc_mod.set_param_hint('n',
                                          value=float('{:.1f}'.format(n)),
                                          vary=False)
            self.params = damped_osc_mod.make_params()

            fit_res = damped_osc_mod.fit(data=y,
                                         t=x,
                                         params=self.params)

            if fit_res.chisqr > .35:
                log.warning('Fit did not converge, varying phase')
                fit_res_lst = []

                for phase_estimate in np.linspace(0, 2*np.pi, 8):
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

            self.fit_results_dict[label] = fit_res
            if print_fit_results:
                print(fit_res.fit_report())

        return fit_res

    def plot_results(self, fit_res, show_guess=False, art_det=0,
                     fig=None, ax=None, textbox=True, plot_gaussian=False):

        self.units = SI_prefix_and_scale_factor(val=max(abs(ax.get_xticks())),
                                                unit=self.sweep_unit[0])[1]  # list

        if isinstance(art_det, list):
            art_det = art_det[0]

        if textbox:
            if plot_gaussian:
                fit_res_gauss = self.fit_results_dict['gaussian']
                fit_res_array = [fit_res, fit_res_gauss]

                textstr = ('$f_{qubit \_ old}$ = %.7g GHz'
                           % (self.qubit_freq_spec*1e-9) +
                           '\n$f_{qubit \_ new \_ exp}$ = %.7g $\pm$ (%.5g) GHz'
                           % (self.qubit_frequency*1e-9,
                              fit_res.params['frequency'].stderr*1e-9) +
                           '\n$f_{qubit \_ new \_ gauss}$ = %.7g $\pm$ (%.5g) GHz'
                           % (self.qubit_frequency_gauss *1e-9,
                              fit_res_gauss.params['frequency'].stderr*1e-9))
                T2_star_str = ('\n$T_{2,exp}^\star$ = %.6g '
                               % (fit_res.params['tau'].value*self.scale)  +
                               self.units + ' $\pm$ (%.6g) '
                               % (fit_res.params['tau'].stderr*self.scale) +
                               self.units +
                               '\n$T_{2,gauss}^\star$ = %.6g '
                               %(fit_res_gauss.params['tau'].value*self.scale) +
                               self.units + ' $\pm$ (%.6g) '
                               %(fit_res_gauss.params['tau'].stderr*self.scale)+
                               self.units)
            else:
                fit_res_array = [fit_res]
                textstr = ('$f_{qubit \_ old}$ = %.7g GHz'
                           % (self.qubit_freq_spec*1e-9) +
                           '\n$f_{qubit \_ new}$ = %.7g $\pm$ (%.5g) GHz'
                           % (self.qubit_frequency*1e-9,
                              fit_res.params['frequency'].stderr*1e-9))
                T2_star_str = ('\n$T_2^\star$ = %.6g '
                               % (fit_res.params['tau'].value*self.scale)  +
                               self.units + ' $\pm$ (%.6g) '
                               % (fit_res.params['tau'].stderr*self.scale) +
                               self.units)

            textstr += ('\n$\Delta f$ = %.5g $ \pm$ (%.5g) MHz'
                        % ((self.qubit_frequency - self.qubit_freq_spec) * 1e-6,
                           fit_res.params['frequency'].stderr * 1e-6) +
                        '\n$f_{Ramsey}$ = %.5g $ \pm$ (%.5g) MHz'
                        % (fit_res.params['frequency'].value*1e-6,
                           fit_res.params['frequency'].stderr*1e-6))
            textstr += T2_star_str
            textstr += ('\nartificial detuning = %.2g MHz'
                        % (art_det * 1e-6))
            self.add_textbox(textstr, fig=fig, ax=ax)

            x = np.linspace(self.sweep_points[0],
                            self.sweep_points[-self.NoCalPoints-1],
                            len(self.sweep_points)*100)

            for i, f_res in enumerate(fit_res_array):
                if i==1:
                    color = 'C4'
                    guess_c = 'C7'
                    label = 'Gaussian decay'
                else:
                    color = 'r'
                    guess_c = 'k'
                    label = 'Exponential decay'

                y = f_res.model.func(x, **f_res.best_values)
                # best_vals = f_res.best_values
                #  tau=best_vals['tau'],
                # n=best_vals['n'],
                # frequency=best_vals['frequency'],
                # phase=best_vals['phase'],
                # amplitude=best_vals['amplitude'],
                # oscillation_offset=best_vals['oscillation_offset'],
                # exponential_offset=best_vals['exponential_offset'])
                ax.plot(x, y, '-', c=color, linewidth=self.line_width,
                        label=label)

                if show_guess:
                    # y_init = fit_mods.ExpDampOscFunc(x, **fit_res.init_values)
                    y_init = f_res.model.func(x, **f_res.init_values)
                    ax.plot(x, y_init, '--', c=guess_c,
                            linewidth=self.line_width)

            ax.legend(frameon=False)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, **kw):

        super().run_default_analysis(
            close_file=close_file,
            close_main_fig=True, save_fig=False,**kw)

        verbose = kw.get('verbose', False)
        # Get old values for qubit frequency
        instr_set = self.data_file['Instrument settings']
        try:
            if self.for_ef:
                self.qubit_freq_spec = \
                    eval(instr_set[self.qb_name].attrs['f_ef_qubit'])
            elif 'freq_qubit' in kw.keys():
                self.qubit_freq_spec = kw['freq_qubit']
            else:
                try:
                    self.qubit_freq_spec = \
                        eval(instr_set[self.qb_name].attrs['f_qubit'])
                except KeyError:
                    self.qubit_freq_spec = \
                        eval(instr_set[self.qb_name].attrs['freq_qubit'])
                    
        except (TypeError, KeyError, ValueError):
            log.warning('qb_name is unknown. Setting previously measured '
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

        self.save_computed_parameters({'artificial_detuning':
                                           self.artificial_detuning},
                                      var_name=self.value_names[0])


        self.save_computed_parameters(self.T2_star,
                                      var_name=self.value_names[0])
        self.save_computed_parameters(self.T2_star_gauss,
                                      var_name=(self.value_names[0]+
                                                ' gaussian decay'))

        self.save_computed_parameters({'qubit_freq': self.qubit_frequency},
                                      var_name=self.value_names[0])
        self.save_computed_parameters({'qubit_freq_gauss':
                                           self.qubit_frequency_gauss},
                                      var_name=(self.value_names[0]+
                                                ' gaussian decay'))

        #Print the T2_star values on screen
        unit = self.parameter_units[0][-1]
        if kw.pop('print_parameters', False):
            print('New qubit frequency exp = {:.7f} (GHz)'.format(
                self.qubit_frequency * 1e-9) +
                  '\t\tqubit frequency stderr = {:.7f} (MHz)'.format(
                      self.ramsey_freq['freq_stderr']*1e-6)+
                  '\nT2* exp = {:.5f} '.format(
                      self.T2_star['T2_star']*self.scale) +'('+'μ'+unit+')'+
                  '\t\tT2* stderr = {:.5f} '.format(
                      self.T2_star['T2_star_stderr']*self.scale) +
                  '('+'μ'+unit+')')

        if close_file:
            self.data_file.close()

        return self.fit_res

    def one_art_det_analysis(self, **kw):

        # Perform fit and save fitted parameters
        self.fit_res = self.fit_Ramsey(
            x=self.sweep_points[:len(self.sweep_points)-self.NoCalPoints],
            y=self.normalized_data_points, **kw)
        self.save_fitted_parameters(self.fit_res,
                                    var_name=self.value_names[0])
        self.save_fitted_parameters(self.fit_results_dict['gaussian'],
                                    var_name=(self.value_names[0]+
                                              ' gaussian decay'))
        self.get_measured_freq(fit_res=self.fit_res, **kw)

        # Calculate new qubit frequency
        self.qubit_frequency = self.qubit_freq_spec + self.artificial_detuning \
                               - self.ramsey_freq['freq']
        self.qubit_frequency_gauss = self.qubit_freq_spec + \
                                     self.artificial_detuning - \
                                     self.fit_results_dict[
                                         'gaussian'].best_values['frequency']

        # Extract T2 star and save it
        self.T2_star = self.get_measured_T2_star(fit_res=self.fit_res, **kw)
        self.T2_star_gauss = \
            self.get_measured_T2_star(
                fit_res=self.fit_results_dict['gaussian'], **kw)
        # the call above defines self.T2_star as a dict; units are seconds

        self.total_detuning = self.fit_res.params['frequency'].value
        self.detuning_stderr = self.fit_res.params['frequency'].stderr
        self.detuning = self.total_detuning - self.artificial_detuning

        if self.make_fig:
            # Plot results
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)
            plot_gaussian = kw.pop('plot_gaussian', True)
            plot_title_suffix = kw.pop('plot_title_suffix', '')

            self.plot_results(self.fit_res, show_guess=show_guess,
                              art_det=self.artificial_detuning,
                              fig=self.fig, ax=self.ax,
                              plot_gaussian=plot_gaussian)

            # dispaly figure
            if show:
                plt.show()

            #save figure
            fig_name_suffix = kw.pop('fig_name_suffix', 'Ramsey_fit')
            self.save_fig(self.fig, figname=(self.measurementstring+'_'+
                                             fig_name_suffix +
                                             plot_title_suffix), **kw)

    def two_art_dets_analysis(self, **kw):

        # Extract the data for each ramsey
        len_art_det = len(self.artificial_detuning)
        sweep_pts_1 = self.sweep_points[
                      0:len(self.sweep_points)-self.NoCalPoints:len_art_det]
        sweep_pts_2 = self.sweep_points[
                      1:len(self.sweep_points)-self.NoCalPoints:len_art_det]
        ramsey_data_1 = self.normalized_values[
                        0:len(self.sweep_points)-self.NoCalPoints:len_art_det]
        ramsey_data_2 = self.normalized_values[
                        1:len(self.sweep_points)-self.NoCalPoints:len_art_det]

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

        #Extract T2 star and save it
        self.T2_star = self.get_measured_T2_star(fit_res=self.fit_res, **kw)
        # units are seconds

        ################
        # Plot results #
        ################
        if self.make_fig_two_dets:
            show_guess = kw.pop('show_guess', False)
            show = kw.pop('show', False)
            plot_title_suffix = kw.pop('plot_title_suffix', '')

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

            #save figure
            fig_name_suffix = kw.pop('fig_name_suffix', 'Ramsey_fit')
            self.save_fig(self.fig, figname=(self.measurementstring+
                                             '_'+fig_name_suffix+
                                             plot_title_suffix), **kw)

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

        T2_star = {'T2_star': T2, 'T2_star_stderr': T2_stderr}

        return T2_star


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
            - 'HalfFeedlineS21' = fit to an apropriate J model

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
            log.warning('No peaks or dips in range')
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
            # added reject outliers to be robust against CBox data acq bug.
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
            data_complex = data_amp * np.cos(data_angle) + 1j * data_amp * np.sin(data_angle)
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

        elif fitting_model == 'HalfFeedlineS21':
            Model = fit_mods.half_Feed_lineS12_J_Model
            fit_mods.half_Feed_lineS12_J_Model.guess(Model,np.transpose([self.sweep_points,self.measured_powers]))

            #checking which choice of PF-RR results in the better fit
            fit=Model.fit(data=self.measured_powers,omega=self.sweep_points)
            fRR=Model.param_hints['omegaRR']['value']
            Model.set_param_hint('omegaRR',value=Model.param_hints['omegaPF']['value'],min=Model.param_hints['omegaPF']['value']-2e7,max=Model.param_hints['omegaPF']['value']+2e7)
            Model.set_param_hint('omegaPF',value=fRR,min=fRR-2e7,max=fRR+2e7)
            fit2=Model.fit(data=self.measured_powers,omega=self.sweep_points)
            if (sum(fit.eval_uncertainty(omega=self.sweep_points))/len(fit.eval_uncertainty(omega=self.sweep_points))>=sum(fit2.eval_uncertainty(omega=self.sweep_points))/len(fit2.eval_uncertainty(omega=self.sweep_points))):
                fit_res = fit2
            else:
                fit_res = fit



        else:
            raise ValueError('fitting model "{}" not recognized'.format(
                fitting_model))

        self.fit_results = fit_res
        self.save_fitted_parameters(fit_res, var_name='HM')

        if print_fit_results is True:
            # print(fit_res.fit_report())
            print(lmfit.fit_report(fit_res))

        ########## Plot results ##########

        fig, ax = self.default_ax()

        if 'hanger' in fitting_model:
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[0],
                                            fig=fig, ax=ax,
                                            xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=str('S21_mag'),
                                            y_unit=self.value_units[0],
                                            save=False)
            # ensures that amplitude plot starts at zero
            ax.set_ylim(ymin=-0.001)

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

        elif fitting_model == 'HalfFeedlineS21':
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_powers,
                                            fig=fig, ax=ax,
                                            xlabel=self.sweep_name,
                                            x_unit=self.sweep_unit[0],
                                            ylabel=str('magn (mVpeak)'),
                                            save=False)

        scale = SI_prefix_and_scale_factor(val=max(abs(ax.get_xticks())),
                                           unit=self.sweep_unit[0])[0]

        instr_set = self.data_file['Instrument settings']
        try:
            old_RO_freq = eval(instr_set[self.qb_name].attrs['f_RO'])
            old_vals = '\n$f_{\mathrm{old}}$ = %.5f GHz' % (old_RO_freq * scale)
        except (TypeError, KeyError, ValueError):
            log.warning('qb_name is None. Old parameter values will '
                            'not be retrieved.')
            old_vals = ''

        if ('hanger' in fitting_model) or ('complex' in fitting_model):
            if kw['custom_power_message'] is None:
                textstr = '$f_{\mathrm{center}}$ = %.5f GHz $\pm$ (%.3g) GHz' % (
                    fit_res.params['f0'].value,
                    fit_res.params['f0'].stderr) + '\n' \
                                                   '$Qc$ = %.1f $\pm$ (%.1f)' % (
                              fit_res.params['Qc'].value,
                              fit_res.params['Qc'].stderr) + '\n' \
                                                             '$Qi$ = %.1f $\pm$ (%.1f)' % (
                              fit_res.params['Qi'].value, fit_res.params['Qi'].stderr) + \
                          old_vals
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
                power_in_w = 10 ** ((custom_power['Power'] - custom_power['Atten']) / 10) * 1e-3
                mean_ph = (2 * (fit_res.params['Q'].value ** 2) / (fit_res.params['Qc'].value * hbar * (
                        2 * pi * fit_res.params['f0'].value * 1e9) ** 2)) * power_in_w
                phase_vel = 4 * custom_power['res_len'] * fit_res.params['f0'].value * 1e9

                textstr = '$f_{\mathrm{center}}$ = %.5f GHz $\pm$ (%.3g) GHz' % (
                    fit_res.params['f0'].value,
                    fit_res.params['f0'].stderr) + '\n' \
                                                   '$Qc$ = %.1f $\pm$ (%.1f)' % (
                              fit_res.params['Qc'].value,
                              fit_res.params['Qc'].stderr) + '\n' \
                                                             '$Qi$ = %.1f $\pm$ (%.1f)' % (
                              fit_res.params['Qi'].value, fit_res.params['Qi'].stderr) + \
                          old_vals + '\n' \
                                     '$< n_{\mathrm{ph} }>$ = %.1f' % (mean_ph) + '\n' \
                                                                                  '$v_{\mathrm{phase}}$ = %.3e m/s' % (
                              phase_vel)

        elif fitting_model == 'lorentzian':
            textstr = '$f_{{\mathrm{{center}}}}$ = %.5f GHz ' \
                      '$\pm$ (%.3g) GHz' % (
                          fit_res.params['f0'].value * scale,
                          fit_res.params['f0'].stderr * scale) + '\n' \
                                                                 '$Q$ = %.1f $\pm$ (%.1f)' % (
                          fit_res.params['Q'].value,
                          fit_res.params['Q'].stderr) + old_vals

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
            self.save_fig(fig, xlabel=self.xlabel, ylabel='Mag', **kw)

        # self.save_fig(fig, xlabel=self.xlabel, ylabel='Mag', **kw)
        if close_file:
            self.data_file.close()
        return fit_res


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

        fig, ax = self.default_ax()
        self.plot_dB_from_linear(x=self.sweep_points,
                                 lin_amp=data_amp,
                                 fig=fig, ax=ax,
                                 save=False)

        self.save_fig(fig, figname='dB_plot', **kw)


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
            self.data_dist = a_tools.rotate_and_normalize_data_no_cal_points(
                np.array([data_real, data_imag]))
            # self.data_dist = a_tools.calculate_distance_ground_state(
            #     data_real=data_real,
            #     data_imag=data_imag,
            #     normalize=False)
        except:
            # Quick fix to make it work with pulsed spec which does not
            # return both I,Q and, amp and phase
            # only using the amplitude!!
            self.data_dist = self.measured_values[0]

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

        # extract highest peak -> ge transition
        if self.peaks['dip'] is None:
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
            log.warning('No peaks or dips have been found. Initial '
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
                                           value=amplitude_guess)  # ,
            # min=4*np.var(self.data_dist))
            LorentzianModel.set_param_hint('offset',
                                           value=0,
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

            if kappa_guess_ef > kappa_guess:
                temp = deepcopy(kappa_guess)
                kappa_guess = deepcopy(kappa_guess_ef)
                kappa_guess_ef = temp

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
                                        label='',
                                        save=False)

        # plot Lorentzian with the fit results
        ax_dist.plot(self.sweep_points, self.fit_res.best_fit,
                     'r-', linewidth=self.line_width, label='')

        # Plot a point for each plot at the chosen best fit f0 frequency (f_ge)
        f0 = self.fit_res.params['f0'].value
        f0_idx = a_tools.nearest_idx(self.sweep_points, f0)
        ax_dist.plot(f0, self.fit_res.best_fit[f0_idx], 'o',
                     ms=self.marker_size_special, label=r'$f_{ge,fit}$')

        if analyze_ef:
            # plot the ef/2 point as well
            f0_gf_over_2 = self.fit_res.params['f0_gf_over_2'].value
            self.fitted_freq_gf_over_2 = f0_gf_over_2
            f0_gf_over_2_idx = a_tools.nearest_idx(self.sweep_points,
                                                   f0_gf_over_2)
            ax_dist.plot(f0_gf_over_2,
                         self.fit_res.best_fit[f0_gf_over_2_idx],
                         'o', ms=self.marker_size_special,
                         label=r'$f_{gf/2,fit}$')
            # plot point at the f_gf_half guess value
            ax_dist.plot(self.fit_res.init_values['f0_gf_over_2'],
                         self.fit_res.best_fit[f0_gf_over_2_idx],
                         'o', ms=self.marker_size_special,
                         label=r'$f_{gf/2,guess}$')
            ax_dist.legend(frameon=False)

        if show_guess:
            # plot Lorentzian with initial guess
            ax_dist.plot(self.sweep_points, self.fit_res.init_fit,
                         'k--', linewidth=self.line_width)

        scale = SI_prefix_and_scale_factor(val=max(abs(ax_dist.get_xticks())),
                                           unit=self.sweep_unit[0])[0]

        instr_set = self.data_file['Instrument settings']

        if analyze_ef:
            try:
                old_freq = eval(instr_set[self.qb_name].attrs['f_qubit'])
                old_freq_ef = eval(instr_set[self.qb_name].attrs['f_ef_qubit'])
                label = 'f0={:.5f} GHz ' \
                        '\nold f0={:.5f} GHz' \
                        '\nkappa0={:.4f} MHz' \
                        'f0_gf/2={:.5f} GHz ' \
                        '\nold f0_gf/2={:.5f} GHz' \
                        '\nguess f0_gf/2={:.5f} GHz' \
                        '\nkappa_gf={:.4f} MHz'.format(
                    self.fit_res.params['f0'].value * scale,
                    old_freq * scale,
                    self.fit_res.params['kappa'].value / 1e6,
                    self.fit_res.params['f0_gf_over_2'].value * scale,
                    old_freq_ef * scale,
                    self.fit_res.init_values['f0_gf_over_2']*scale,
                    self.fit_res.params['kappa_gf_over_2'].value / 1e6)
            except (TypeError, KeyError, ValueError):
                log.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
                label = 'f0={:.5f} GHz ' \
                        '\nkappa0={:.4f} MHz \n' \
                        'f0_gf/2={:.5f} GHz  ' \
                        '\nkappa_gf={:.4f} MHz '.format(
                    self.fit_res.params['f0'].value * scale,
                    self.fit_res.params['kappa'].value / 1e6,
                    self.fit_res.params['f0_gf_over_2'].value * scale,
                    self.fit_res.params['kappa_gf_over_2'].value / 1e6)
        else:
            label = 'f0={:.5f} GHz '.format(
                self.fit_res.params['f0'].value * scale)
            try:
                old_freq = eval(instr_set[self.qb_name].attrs['f_qubit'])
                label += '\nold f0={:.5f} GHz' .format(
                    old_freq * scale)
            except (TypeError, KeyError, ValueError):
                log.warning('qb_name is None. Old parameter values will '
                                'not be retrieved.')
            label += '\nkappa0={:.4f} MHz'.format(
                self.fit_res.params['kappa'].value / 1e6)

        self.add_textbox(label, fig_dist, ax_dist)
        # fig_dist.text(0.5, 0, label, transform=ax_dist.transAxes,
        #               fontsize=self.font_size, verticalalignment='top',
        #               horizontalalignment='center', bbox=self.box_props)

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


class TwoD_Analysis(MeasurementAnalysis):
    '''
    Analysis for 2D measurements.
    '''
    def __init__(self, TwoD=True, **kw):
        super().__init__(TwoD=TwoD, **kw)

    def run_default_analysis(self, normalize=False, plot_linecuts=True,
                             linecut_log=False, colorplot_log=False,
                             plot_all=True, save_fig=True,
                             transpose=False, figsize=None,
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

        super().run_default_analysis(show=False,
                                     close_file=close_file,
                                     close_main_fig=True,
                                     save_fig=True, **kw)
        # self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []

        for i, meas_vals in enumerate(self.measured_values):
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
                                     zlabel=self.zlabels[i],
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
            # print "unransposed",meas_vals
            # print "transposed", meas_vals.transpose()
            self.ax_array.append(ax)
            savename = 'Heatmap_{}'.format(self.value_names[i])
            fig_title = '{} {} \n{}'.format(
                self.timestamp_string, self.measurementstring,
                self.value_names[i])

            if "xlabel" not in kw:
                kw["xlabel"] = self.parameter_names[0]
            if "ylabel" not in kw:
                kw["ylabel"] = self.parameter_names[1]
            if "xunit" not in kw:
                kw["xunit"] = self.parameter_units[0]
            if "yunit" not in kw:
                kw["yunit"] = self.parameter_units[1]

            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=meas_vals.transpose(),
                               zlabel=self.zlabels[i],
                               fig=fig, ax=ax,
                               log=colorplot_log,
                               transpose=transpose,
                               normalize=normalize,
                               **kw)
            ax.set_title(fig_title)

            if save_fig:
                self.save_fig(fig, figname=savename, **kw)


##########################################
### Analysis for data measurement sets ###
##########################################


def fit_qubit_frequency(sweep_points, data, mode='dac',
                        vary_E_c=True, vary_f_max=True,
                        vary_V_per_phi0=True,
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
        except:
            print('Qubit instrument is not in data file')
            qubit_attrs = {}
    else:
        qubit_attrs = {}

    # Extract initial values, first from kw, then data file, then default
    E_c = kw.pop('E_c', qubit_attrs.get('E_c', '0.3e9'))
    f_max = kw.pop('f_max', qubit_attrs.get('f_max', repr(np.max(data))))
    V_per_phi0 = kw.pop('V_per_phi0',
                            eval(qubit_attrs.get('V_per_phi0', '1.')))
    dac_sweet_spot = kw.pop('dac_sweet_spot',
                            eval(qubit_attrs.get('dac_sweet_spot', '0')))
    flux_zero = kw.pop('flux_zero', eval(qubit_attrs.get('flux_zero', '10')))

    if mode == 'dac':
        Q_dac_freq_mod = fit_mods.QubitFreqDacModel
        Q_dac_freq_mod.set_param_hint('E_c', value=E_c, vary=vary_E_c,
                                      min=0, max=500e6)
        Q_dac_freq_mod.set_param_hint('f_max', value=f_max,
                                      vary=vary_f_max)
        Q_dac_freq_mod.set_param_hint('V_per_phi0',
                                      value=V_per_phi0,
                                      vary=vary_V_per_phi0)
        Q_dac_freq_mod.set_param_hint('dac_sweet_spot',
                                      value=dac_sweet_spot,
                                      vary=vary_dac_sweet_spot)
        Q_dac_freq_mod.set_param_hint('asymmetry',
                                      value=0,
                                      vary=False)

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


class AvoidedCrossingAnalysis(MeasurementAnalysis):
    """
    Performs analysis to fit the avoided crossing
    """

    def __init__(self, auto=True,
                 model='direct_coupling',
                 label=None,
                 timestamp=None,
                 transpose=True,
                 cmap='viridis', filter_data=False,
                 filt_func_a=None, filt_func_x0=None, filt_func_y0=None,
                 filter_idx_low=[], filter_idx_high=[], filter_threshold=15e6,
                 f1_guess=None, f2_guess=None, cross_flux_guess=None,
                 g_guess=30e6, coupling_label='g',
                 break_before_fitting=False,
                 add_title=True,
                 xlabel=None, ylabel='Frequency (GHz)', **kw):
        super().__init__(timestamp=timestamp, label=label, **kw)
        self.get_naming_and_values_2D()

        flux = self.Y[:, 0]
        peaks_low, peaks_high = self.find_peaks(**kw)
        self.f, self.ax = self.make_unfiltered_figure(peaks_low, peaks_high,
                                                      transpose=transpose, cmap=cmap,
                                                      add_title=add_title,
                                                      xlabel=xlabel, ylabel=ylabel)
        if filter_data:
            filtered_dat = self.filter_data(flux, peaks_low, peaks_high,
                                            a=filt_func_a, x0=filt_func_x0,
                                            y0=filt_func_y0,
                                            filter_idx_low=filter_idx_low,
                                            filter_idx_high=filter_idx_high,
                                            filter_threshold=filter_threshold)
            filt_flux_low, filt_flux_high, filt_peaks_low, filt_peaks_high, \
            filter_func = filtered_dat
            self.f, self.ax = self.make_filtered_figure(
                filt_flux_low, filt_flux_high,
                filt_peaks_low, filt_peaks_high, filter_func,
                add_title=add_title,
                transpose=transpose, cmap=cmap,
                xlabel=xlabel, ylabel=ylabel)
        else:
            min_freq_sep_estimate = kw.pop('min_freq_sep_estimate', 4e6)
            self.freq_mask = np.abs(peaks_high-peaks_low) > min_freq_sep_estimate
            filt_flux_low = self.Y[:, 0][self.freq_mask ]
            filt_flux_high = self.Y[:, 0]
            filt_peaks_low = peaks_low[self.freq_mask ]
            filt_peaks_high = peaks_high


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
        key = kw.pop('key', 'dip')
        peaks = np.zeros((len(self.X), 2))
        for i in range(len(self.X)):
            # p_dict = a_tools.peak_finder_v2(self.X[i], self.Z[0][i],
            #                                 perc=10, window_len=0)
            # peaks[i, :] = np.sort(p_dict[:2])
            peak_dict = a_tools.peak_finder(self.X[i], self.Z[0][i],
                                            optimize=True,
                                            window_len=0)
            f_high = peak_dict[key]
            # f1_idx = peak_dict['dip_idx']
            f0, f0_gf_over_2, kappa_guess, kappa_guess_ef = \
                a_tools.find_second_peak(self.X[i], self.Z[0][i],
                                         key=key,
                                         peaks=peak_dict,
                                         percentile=10,
                                         verbose=False)
            f_low = [f for f in [f0, f0_gf_over_2] if f != f_high][0]
            # f2_idx = a_tools.nearest_idx(self.X[i], f2)
            peaks[i, :] = np.array([f_low, f_high])

        peaks_low = peaks[:, 0]
        peaks_high = peaks[:, 1]
        return peaks_low, peaks_high

    def filter_data(self, flux, peaks_low, peaks_high, a, x0=None, y0=None,
                    filter_idx_low=[], filter_idx_high=[],
                    filter_threshold=15e5):
        """
        Filters the input data in three steps.
            1. remove outliers using the dm_tools.get_outliers function
            2. separate data in two branches using a line and filter data on the
                wrong side of the line.
            3. remove any data with indices specified by hand
        """
        if a is None:
            a = -1 * (max(peaks_high) - min(peaks_low)) / (max(flux) - min(flux))
        if x0 is None:
            x0 = np.mean(flux)
        if y0 is None:
            y0 = np.mean(np.concatenate([peaks_low, peaks_high]))
        print(filter_threshold)
        filter_func = lambda x: a * (x - x0) + y0

        filter_mask_high = [True] * len(peaks_high)
        filter_mask_high = ~dm_tools.get_outliers(peaks_high, filter_threshold)
        filter_mask_high = np.where(
            peaks_high < filter_func(flux), False, filter_mask_high)
        for i in filter_idx_high: filter_mask_high[i] = False
        # filter_mask_high[-2] = False  # hand remove 1 datapoint

        filt_flux_high = flux[filter_mask_high]
        filt_peaks_high = peaks_high[filter_mask_high]

        filter_mask_low = [True] * len(peaks_low)
        filter_mask_low = ~dm_tools.get_outliers(peaks_low, filter_threshold)
        filter_mask_low = np.where(
            peaks_low > filter_func(flux), False, filter_mask_low)
        for i in filter_idx_low: filter_mask_low[i] = False
        # filter_mask_low[[0, -1]] = False  # hand remove 2 datapoints

        filt_flux_low = flux[filter_mask_low]
        filt_peaks_low = peaks_low[filter_mask_low]

        return (filt_flux_low, filt_flux_high,
                filt_peaks_low, filt_peaks_high, filter_func)

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
                  dpi=self.dpi)
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
                  dpi=self.dpi)
        return f, ax

    def make_fit_figure(self,
                        filt_flux_low, filt_flux_high,
                        filt_peaks_low, filt_peaks_high, fit_res,
                        transpose, cmap, coupling_label='g',
                        xlabel=None, ylabel='Frequency (GHz)',
                        add_title=True):
        flux = self.Y[:, 0]
        title = ' avoided crossing fit'
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
        f.savefig(os.path.join(self.folder, title + '.png'), format='png',
                  dpi=self.dpi)
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

        c2_guess = 0.
        if f2_guess is None:
            # The factor *1000* is a magic number but seems to give a
            # reasonable guess that converges well.
            c1_guess = -1 * ((max(total_freqs) - min(total_freqs)) /
                             (max(total_flux) - min(total_flux))) / 1000

            f2_guess = cross_flux_guess * (c1_guess - c2_guess) + f1_guess
        else:
            c1_guess = c2_guess + (f2_guess - f1_guess) / cross_flux_guess

        g_guess = np.min(np.abs(upper_freqs[self.freq_mask]-lower_freqs))
        f2_guess = np.mean(lower_freqs)
        f1_guess = np.mean(upper_freqs)
        c1_guess = (upper_freqs[-1] - upper_freqs[0]) / \
                   (upper_flux[-1] - upper_flux[0])
        c2_guess = (lower_freqs[-1] - lower_freqs[0]) / \
                   (lower_flux[-1] - lower_flux[0])

        av_crossing_model.set_param_hint(
            'g', min=0., max=0.5e9, value=g_guess, vary=True)
        av_crossing_model.set_param_hint(
            'f_center1', min=0, max=20.0e9, value=f1_guess, vary=True)
        av_crossing_model.set_param_hint(
            'f_center2', min=0., max=20.0e9, value=f2_guess, vary=True)
        av_crossing_model.set_param_hint(
            'c1', min=-1.0e9, max=1.0e9, value=c1_guess, vary=True)
        av_crossing_model.set_param_hint(
            'c2', min=-1.0e9, max=1.0e9, value=c2_guess, vary=True)
        params = av_crossing_model.make_params()
        fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                        flux=np.array(total_flux),
                                        params=params)
        for par in fit_res.params:
            if fit_res.params[par].stderr is None:
                fit_res.params[par].stderr = 0
        return fit_res


class Fluxpulse_Ramsey_2D_Analysis(MeasurementAnalysis):

    def __init__(self, X90_separation=None, flux_pulse_length=None,
                 drive_pulse_length=None,
                 qb_name=None, label=None,
                 cal_points=False,
                 reference_measurements=False,
                 auto=True,
                 **kw):
        """
        Measurement analysis class to analyse Ramsey type measrements
        with an interleaved flux pulse

        Args:
            X90_separation (float): separation between the two X90 pulses
            flux_pulse_length (float): length of the flux pulse in seconds
                                        (used to calculate freq. shifts)
            qb_name (str): qubit name
            label (str): measurement label
            **kw:
        """

        kw['label'] = label
        kw['h5mode'] = 'r+'
        kw['close_file'] = False

        self.label = label
        self.fitted_phases = None
        self.fitted_delay = 0
        self.delay_fit_res = None
        self.X90_separation = X90_separation
        self.flux_pulse_length = flux_pulse_length
        self.drive_pulse_length = drive_pulse_length
        self.return_fit = kw.pop('return_fit', False)
        self.cal_points = cal_points
        self.reference_measurements=reference_measurements

        super(Fluxpulse_Ramsey_2D_Analysis, self).__init__(TwoD=True,
                                                           start_at_zero=True,
                                                           qb_name=qb_name,
                                                           auto=auto,
                                                           **kw)
        self.get_values('Data')
        self.get_naming_and_values_2D()

    def run_default_analysis(self, TwoD=False, close_file=True,
                             show=False, transpose=False,
                             plot_args=None, **kw):
        super().run_default_analysis(TwoD, close_file, show, transpose,
                                     plot_args, **kw)

        self.fit_all(self, **kw)

    def fit_single_cos(self, thetas, ampls,
                       print_fit_results=True, phase_guess=0,
                       cal_points=False):
        if cal_points:
            thetas = thetas[:-4]
            ampls = ampls[:-4]
        cos_mod = fit_mods.CosModel
        average = np.mean(ampls)

        diff = 0.5*(max(ampls) -
                    min(ampls))
        amp_guess = -diff
        # offset guess
        offset_guess = average

        # Set up fit parameters and perform fit
        cos_mod.set_param_hint('amplitude',
                               value=amp_guess,
                               vary=True)
        cos_mod.set_param_hint('phase',
                               value=phase_guess,
                               vary=True)
        cos_mod.set_param_hint('frequency',
                               value=1./(2*np.pi),
                               vary=False)
        cos_mod.set_param_hint('offset',
                               value=offset_guess,
                               vary=True)
        self.params = cos_mod.make_params()
        fit_res = cos_mod.fit(data=ampls,
                              t=thetas,
                              params=self.params)

        if fit_res.chisqr > 0.35:
            log.warning('Fit did not converge, chi-square > 0.35')

        if print_fit_results:
            print(fit_res.fit_report())

        if fit_res.best_values['amplitude'] < 0.:
            fit_res.best_values['phase'] += np.pi
            fit_res.best_values['amplitude'] *= -1

        return fit_res

    def unwrap_phases_extrapolation(self,phases):
        for i in range(2,len(phases)):
            phase_diff_extrapolation = (phases[i-1]
                                        + (phases[i-1]
                                           - phases[i-2]) - phases[i])
            #         print(i,abs(phase_diff_extrapolation)>np.pi)
            #         print(phase_list[i],phase_diff_
            #           extrapolation + phase_list[i])
            if phase_diff_extrapolation > np.pi:
                phases[i] += round(phase_diff_extrapolation/(2*np.pi))*2*np.pi
            #             print('corrected: ',  phase_list[i])
            elif phase_diff_extrapolation < np.pi:
                phases[i] += round(phase_diff_extrapolation/(2*np.pi))*2*np.pi
                #             print('corrected: ',  phase_list[i])
        return phases

    def fit_all(self,
                extrapolate_phase=False,
                return_ampl=False,
                cal_points=None,
                fit_range=None,
                predict_phase=True,
                save_plot=False,
                plot_title=None, **kw):

        only_cos_fits = kw.pop('only_cos_fits', False)
        plot = kw.pop('plot', False)

        if cal_points is None:
            cal_points = self.cal_points
            cal_one_points = [-2, -1]
            cal_zero_points = [-4, -3]
        else:
            cal_one_points = None
            cal_zero_points = None


        phase_list = [0]
        amplitude_list = []

        length_single = len(self.sweep_points)

        if plot:
            if only_cos_fits:
                self.fig, self.ax = plt.subplots()
                ax = self.ax
            else:
                self.fig, self.ax = plt.subplots(2, 1)
                ax = self.ax[0]
        else:
            self.fig, self.ax = (None, None)

        data_rotated = a_tools.rotate_and_normalize_data(
            np.array([self.data[2], self.data[3]]),
            cal_one_points=cal_one_points, cal_zero_points=cal_zero_points)[0]

        if fit_range is None:
            i_start = 0
            i_end = length_single*len(self.sweep_points_2D)
        else:
            i_start = length_single*fit_range[0]
            i_end = length_single*fit_range[1]
        for i in np.arange(i_start, i_end, length_single):

            thetas = self.data[0, i:i+length_single]
            ampls = data_rotated[i:i+length_single]

            if predict_phase:
                phase_guess = phase_list[-1]
            else:
                phase_guess = 0

            fit_res = self.fit_single_cos(thetas, ampls,
                                          print_fit_results=False,
                                          phase_guess=phase_guess,
                                          cal_points=cal_points)

            phase_list.append(fit_res.best_values['phase'])
            amplitude_list.append(fit_res.best_values['amplitude'])

            if plot:
                ax.plot(thetas, ampls, 'k.')
                if cal_points:
                    thetas_fit = np.linspace(thetas[0], thetas[-5], 128)
                    ampls_fit = fit_res.eval(t=thetas_fit)
                else:
                    thetas_fit = np.linspace(thetas[0],thetas[-1], 128)
                    ampls_fit = fit_res.eval(t=thetas_fit)
                ax.plot(thetas_fit, ampls_fit, 'r-')


        phase_list.pop(0)

        phase_list = np.array(phase_list)
        amplitude_list = np.array(amplitude_list)
        if extrapolate_phase:
            phase_list = self.unwrap_phases_extrapolation(phase_list)

        if plot:
            ax.set_title('Cosine fits')
            ax.set_xlabel('theta (rad)')
            ax.set_ylabel('|S21| (arb. units)')
            ax.legend(['data','fits'])

            if not only_cos_fits:
                if fit_range is None:
                    self.ax[1].plot(self.sweep_points_2D,phase_list)
                else:
                    self.ax[1].plot(self.sweep_points_2D[fit_range[0]:fit_range[1]],phase_list)
                self.ax[1].set_title('fitted phases')
                self.ax[1].set_xlabel(self.parameter_names[1]
                                      +' '+self.parameter_units[1])
                self.ax[1].set_ylabel('phase (rad)')

                self.fig.subplots_adjust(hspace=0.7)

            if plot_title is not None:
                ax.set_title(plot_title)

            if save_plot:
                self.fig.savefig(self.folder +
                                 '\\Phase_fits_{}.png'.format(self.timestamp_string))
            plt.show()

        self.fitted_phases = phase_list
        self.fitted_amplitudes = amplitude_list

        if return_ampl:
            return phase_list, amplitude_list
        else:
            return phase_list


class Fluxpulse_Ramsey_2D_Analysis_Predictive(MeasurementAnalysis):

    def __init__(self, X90_separation=None, flux_pulse_length=None,
                 drive_pulse_length=None,
                 qb_name=None, label=None,
                 cal_points=False,
                 reference_measurements=False,
                 plot=False,
                 **kw):
        """
        Measurement analysis class to analyse Ramsey type measrements
        with an interleaved flux pulse

        Args:
            X90_separation (float): separation between the two X90 pulses
            flux_pulse_length (float): length of the flux pulse in seconds
                                        (used to calculate freq. shifts)
            qb_name (str): qubit name
            label (str): measurement label
            **kw:
        """

        kw['label'] = label
        kw['h5mode'] = 'r+'
        kw['close_file'] = False
        self.label = label
        self.fitted_phases = None
        self.fitted_delay = 0
        self.delay_fit_res = None
        self.X90_separation = X90_separation
        self.flux_pulse_length = flux_pulse_length
        self.drive_pulse_length = drive_pulse_length
        self.return_fit = kw.pop('return_fit', False)
        self.cal_points = cal_points
        self.reference_measurements = reference_measurements

        super(Fluxpulse_Ramsey_2D_Analysis_Predictive, self).__init__(TwoD=True,
                                                           start_at_zero=True,
                                                           qb_name=qb_name,
                                                           auto=False,
                                                           **kw)
        self.get_naming_and_values_2D()
        self.fitted_phases, self.fitted_amps = self.fit_all(return_ampl=True,
                                                           plot=plot, **kw)
        fitted_phases_exited = self.fitted_phases[:: 2]
        fitted_phases_ground = self.fitted_phases[1:: 2]
        fitted_amps_exited = self.fitted_amps[:: 2]
        fitted_amps_ground = self.fitted_amps[1:: 2]

        cphases = fitted_phases_exited - fitted_phases_ground
        population_losses = np.abs(fitted_amps_ground - fitted_amps_exited) \
                            /fitted_amps_ground
        self.cphases = cphases
        self.population_losses = population_losses

    def get_naming_and_values_2D(self):
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
            self.ylabel = str(self.sweep_name_2D + '(' + self.sweep_unit_2D + ')')

        elif datasaving_format == 'Version 2':
            self.parameter_names = self.get_key('sweep_parameter_names')
            self.parameter_units = self.get_key('sweep_parameter_units')
            if len(self.parameter_names) != len(self.parameter_units):
                log.error(' Number of parameter names does not match number'
                              'of parameter units! Check sweep configuration!')
            self.sweep_names = []
            self.sweep_units =[]
            for it in range(len(self.parameter_names)):
                self.sweep_names.append(self.parameter_names[it])
                self.sweep_units.append(self.parameter_units[it])
            self.value_names = self.get_key('value_names')
            self.value_units = self.get_key('value_units')
            self.data = self.get_values('Data').transpose()

            x = self.data[0]
            cols = np.unique(x).shape[0]
            nr_missing_values = 0
            if len(x) % cols != 0:
                nr_missing_values = cols - len(x) % cols
            x = np.append(x,np.zeros((1,nr_missing_values))+np.nan)
            self.X = x.reshape(-1,cols)
            self.sweep_points = self.X[0]

            self.sweep_points_2D = []
            self.Y = []
            for i in range(1,len(self.parameter_names)):
                y = self.data[i]
                y = np.append(y,np.zeros((1,nr_missing_values)))
                Y = y.reshape(-1,cols)
                self.Y.append(Y)
                self.sweep_points_2D.append(Y.T[0])

            if len(self.value_names) == 1:
                z = self.data[3]
                z = np.append(z,np.zeros(nr_missing_values)+np.nan)
                self.Z = z.reshape(-1,cols)
                self.measured_values = [self.Z.T]
            else:
                self.Z = []
                self.measured_values = []
                for i in range(len(self.value_names)):
                    z = self.data[3 + i]
                    z = np.append(z,np.zeros(nr_missing_values)+np.nan)
                    Z = z.reshape(-1,cols)
                    self.Z.append(Z)
                    self.measured_values.append(Z.T)
            self.xlabel = self.parameter_names[0] + ' (' + \
                          self.parameter_units[0] + ')'
            self.ylabel = self.parameter_names[1] + ' (' + \
                          self.parameter_units[1] + ')' + '_' + \
                          self.parameter_names[2] + ' (' + \
                          self.parameter_units[2] + ')'

            self.parameter_labels = [a + ' (' + b + ')' for a, b in zip(
                                                        self.parameter_names,
                                                        self.parameter_units)]
            self.zlabels = [a + ' (' + b + ')' for a, b in zip(self.value_names,
                                                               self.value_units)]

    def run_default_analysis(self, TwoD=False, close_file=True,
                             show=False, transpose=False,
                             plot_args=None, **kw):
        super().run_default_analysis(TwoD, close_file, show, transpose,
                                     plot_args, **kw)

        self.fit_all(self, return_ampl=True, **kw)

    def fit_single_cos(self, thetas, ampls,
                       print_fit_results=True, phase_guess=0,
                       cal_points=False):
        if cal_points:
            thetas = thetas[:-4]
            ampls = ampls[:-4]
        cos_mod = fit_mods.CosModel
        average = np.mean(ampls)

        diff = 0.5*(max(ampls) -
                    min(ampls))
        amp_guess = -diff
        # offset guess
        offset_guess = average

        # Set up fit parameters and perform fit
        cos_mod.set_param_hint('amplitude',
                               value=amp_guess,
                               vary=True)
        cos_mod.set_param_hint('phase',
                               value=phase_guess,
                               vary=True)
        cos_mod.set_param_hint('frequency',
                               value=1./(2*np.pi),
                               vary=False)
        cos_mod.set_param_hint('offset',
                               value=offset_guess,
                               vary=True)
        self.params = cos_mod.make_params()
        fit_res = cos_mod.fit(data=ampls,
                              t=thetas,
                              params=self.params)

        if fit_res.chisqr > 0.35:
            log.warning('Fit did not converge, chi-square > 0.35')

        if print_fit_results:
            print(fit_res.fit_report())

        if fit_res.best_values['amplitude'] < 0.:
            fit_res.best_values['phase'] += np.pi
            fit_res.best_values['amplitude'] *= -1

        return fit_res


    def unwrap_phases_extrapolation(self,phases):
        for i in range(2, len(phases)):
            phase_diff_extrapolation = (phases[i-1]
                                        + (phases[i-1]
                                           - phases[i-2]) - phases[i])
            #         print(i,abs(phase_diff_extrapolation)>np.pi)
            #         print(phase_list[i],phase_diff_
            #           extrapolation + phase_list[i])
            if phase_diff_extrapolation > np.pi:
                phases[i] += round(phase_diff_extrapolation/(2*np.pi))*2*np.pi
            #             print('corrected: ',  phase_list[i])
            elif phase_diff_extrapolation < np.pi:
                phases[i] += round(phase_diff_extrapolation/(2*np.pi))*2*np.pi
            #             print('corrected: ',  phase_list[i])
        return phases

    def fit_all(self, plot=False,
                extrapolate_phase=False,
                return_ampl=False,
                cal_points=None,
                fit_range=None,
                predict_phase=False,
                save_plot=False,
                plot_title=None, **kw):

        only_cos_fits = kw.pop('only_cos_fits', False)
        fit_statistics = kw.pop('fit_statistics', False)
        fontsize = 14.
        legend_fontsize = 12.
        if cal_points is None:
            cal_points = self.cal_points

        phase_list = [0]
        amplitude_list = []
        frequency_list = []
        offset_list = []

        length_single = len(self.sweep_points)

        amps_all = np.zeros((len(self.sweep_points_2D[0]), length_single))
        if fit_statistics:
            thetas_fit = self.data[0, :length_single]
        if plot:
            if only_cos_fits:
                self.fig, self.ax = plt.subplots(figsize=(10, 7))
                ax = self.ax
            else:
                self.fig, self.ax = plt.subplots(2, 1)
                ax = self.ax[0]
        else:
            self.fig, self.ax = (None, None)

        data_rotated = a_tools.rotate_and_normalize_data_no_cal_points(
            np.array([self.data[3], self.data[4]]))
        self.data_rotated = data_rotated
        if fit_range is None:
            i_start = 0
            i_end = length_single*len(self.sweep_points_2D[0])
        else:
            i_start = length_single*fit_range[0]
            i_end = length_single*fit_range[1]

        for mod, i in enumerate(np.arange(i_start, i_end, length_single)):
            thetas = self.data[0, i:i+length_single]
            ampls = data_rotated[i:i+length_single]

            if predict_phase:
                phase_guess = phase_list[-1]
            else:
                phase_guess = 0

            fit_res = self.fit_single_cos(thetas, ampls,
                                          print_fit_results=False,
                                          phase_guess=phase_guess,
                                          cal_points=cal_points)
            self.fit_results += [fit_res]
            phase_list.append(fit_res.best_values['phase'])
            amplitude_list.append(fit_res.best_values['amplitude'])
            frequency_list.append(fit_res.best_values['frequency'])
            offset_list.append(fit_res.best_values['offset'])
            amps_all[mod, :] = ampls
            if plot:
                if not fit_statistics:
                    if mod % 2 == 0:
                        linecolor_inter = 'orange'
                    else:
                        linecolor_inter = 'darkblue'
                    linestyle_inter = '-'
                    linestyle_meas = ''
                    linecolor_meas = 'black'
                    linewidth_inter = 2.
                    linewidth_meas = 1.
                    marker_size_meas = 2.
                    marker_style_meas = 'D'
                    ax.plot(thetas, ampls,color=linecolor_meas,
                            linestyle=linestyle_meas,linewidth=linewidth_meas,
                            markersize=marker_size_meas,marker=marker_style_meas,
                            alpha=0.7)
                    thetas_plot = np.linspace(thetas[0],thetas[-1], 128)
                    ampls_plot = fit_res.eval(t=thetas_plot)
                    ax.plot(thetas_plot, ampls_plot,color=linecolor_inter,
                            linestyle=linestyle_inter,linewidth=linewidth_inter)
                else:
                    amps_all[mod, :] = ampls
        self.amps_all = amps_all
        phase_list.pop(0)
        phase_list = np.array(phase_list)
        amplitude_list = np.array(amplitude_list)
        #cphase = phase_list[1]-phase_list[0]

        def fit_cos_and_extract_params(thetas_fit, amps_fit, thetas_eval):
            fit = self.fit_single_cos(thetas_fit,amps_fit,
                                      print_fit_results = False,
                                      phase_guess = phase_guess,
                                      cal_points = cal_points)
            amps_eval = fit.eval(t=thetas_eval)
            phase_fit = fit.best_values['phase']
            amp_fit = fit.best_values['amplitude']

            return amps_eval,[phase_fit,amp_fit]

        if plot and fit_statistics:

            amps_avg_ex = np.mean(amps_all[::2,:],axis=0)
            amps_avg_gr = np.mean(amps_all[1::2,:],axis=0)
            std_amps_ex = np.std(amps_all[::2,:],axis=0)
            std_amps_gr = np.std(amps_all[1::2,:],axis=0)
            thetas_plot = np.linspace(thetas[0],thetas[-1], 128)
            ft_dic ={'excited':[amps_avg_ex +c*std_amps_ex for c in range(-1,2)],
                     'ground' :[amps_avg_gr +c*std_amps_gr for c in range(-1,2)]}
            plt_dic ={'excited': {'amplitudes':[],'fit_params':[]},
                      'ground':  {'amplitudes':[],'fit_params':[]}}
            for key in ft_dic.keys():
                for it, fit_amps in enumerate(ft_dic[key]):
                    amps_fit = ft_dic[key][it]
                    amps_eval,fit_params = fit_cos_and_extract_params(thetas_fit,
                                                                      amps_fit,
                                                                      thetas_plot)
                    plt_dic[key]['amplitudes'].append(amps_eval)
                    plt_dic[key]['fit_params'].append(fit_params)
            cphases = []
            pop_losses =[]
            for it in range(len(plt_dic['excited']['fit_params'])):
                cphases.append(plt_dic['excited']['fit_params'][it][0]
                             - plt_dic['ground']['fit_params'][it][0])
                amps_ex = plt_dic['excited']['fit_params'][it][1]
                amps_gr = plt_dic['ground']['fit_params'][it][1]
                pop_loss  = np.abs(amps_ex-amps_gr)/amps_gr
                pop_losses.append(pop_loss)

            std_cphases = 0.5*np.abs(cphases[0] - cphases[2])
            avg_cphases = np.abs(cphases[1])
            # avg_pop_loss = pop_losses[1]
            # std_pop_loss = 0.5*(pop_losses[0]-pop_losses[2])

            # PLOTTING
            ax.errorbar(thetas_fit,amps_avg_ex,yerr=std_amps_ex,linestyle='',
                    marker='D', color='black',
                    markersize=4., label='Avg Data')
            ax.errorbar(thetas_fit, amps_avg_gr,yerr=std_amps_gr,linestyle='',
                    marker='D', color='black',
                    markersize=4.)
            for key in plt_dic.keys():
                linecolor = 'orange' if key == 'excited' else 'darkblue'
                state_label = ' |e>' if key == 'excited' else ' |g>'
                for it, plot_amps in enumerate(plt_dic[key]['amplitudes']):
                    linestyle = '-' if it == 1 else '--'
                    linewidth = 2.5 if it == 1 else 1.5
                    label = 'Avg. Parameter' + state_label if it == 1 \
                            else r"$1\sigma$ Data fit" + state_label
                    label = None if it == 2 else label
                    ax.plot(thetas_plot, plt_dic[key]['amplitudes'][it],
                            linestyle=linestyle,color=linecolor,
                            linewidth=linewidth, label=label)
            ax.legend(loc='best',fontsize=legend_fontsize)

        if extrapolate_phase:
            phase_list = self.unwrap_phases_extrapolation(phase_list)

        if plot:
            if fit_statistics:
                ax.set_title(r"Cphase $= {:0.2f}\pm {:0.2f}$ deg, "
                             '\n@ {:0.4f}ns ; {:0.4f}V \n date_time: {}'.format(
                    avg_cphases*180/np.pi, std_cphases*180/np.pi,
                    self.sweep_points_2D[0][0]*1e9,
                    self.sweep_points_2D[1][0],
                    self.timestamp_string),fontsize=fontsize)
            else:
                ax.set_title('Single Cosine fits \n' + self.timestamp_string,
                              fontsize=fontsize)
            ax.set_xlabel(r"Phase of $2^{nd}\, \pi/2$ pulse, $\theta$ [rad]",
                          fontsize=fontsize)
            ax.set_ylabel('Response (arb. units)',fontsize=fontsize)

            if not fit_statistics:
                ax.legend(['data', 'fits |e>', 'fits |g>'])

            if not only_cos_fits:
                if fit_range is None:
                    self.ax[1].plot(range(len(self.sweep_points_2D[0])),
                                    phase_list)
                else:
                    self.ax[1].plot(range(len(self.sweep_points_2D[0,
                                              fit_range[0]:fit_range[1]])),
                                    phase_list)
                self.ax[1].set_title('fitted phases')
                self.ax[1].set_xlabel('Date point #')
                self.ax[1].set_ylabel('Phase (rad)')

                self.fig.subplots_adjust(hspace=0.7)

            if plot_title is not None:
                ax.set_title(plot_title)

            if save_plot:
                self.save_fig(self.fig, ('Phase_fits_{}'.format(
                    self.timestamp_string)))
                # self.fig.savefig(self.folder +
                #             '\\Phase_fits_{}.png'.format(self.timestamp_string))
                                 #,dpi=600.)
            plt.show()

        self.fitted_phases = phase_list
        self.fitted_amplitudes = amplitude_list

        if return_ampl:
            return phase_list, amplitude_list,
        else:
            return phase_list


class OptimizationAnalysis_Predictive2D:

    def __init__(self,training_grid : np.ndarray ,
                 target_values : np.ndarray,
                 ma : MeasurementAnalysis,
                 estimator='GRNN_neupy',
                 hyper_parameter_dict=None,
                 x_init = None,
                 target_value_names=None,
                 **kw):
        self.x_init = x_init
        self.ma = ma
        self.save_folder = ma.folder
        self.time_stamp = ma.timestamp_string
        self.training_grid = training_grid
        self.target_values = target_values
        self.output_dim = target_values.ndim
        if self.training_grid.ndim == 1:
            self.training_grid.shape = (np.size(self.training_grid),
                                        self.training_grid.ndim)
        if self.output_dim == 1:
            self.target_values.shape = (np.size(self.target_values),
                                        self.output_dim)
        self.estimator_name = estimator
        self.hyper_parameter_dict = hyper_parameter_dict
        self.target_value_names = target_value_names
        print('OptimizationAnalysis_Predictive initialized.')
        print('Measurement Type: ', ma.measurementstring)
        print('Estimator type: ', self.estimator_name)

        t0 = time()
        self.train_and_minimize()
        t1 = time()

        print('Fitting estimator completed in %.2g sec. \nCreating plots...' \
              % (t1-t0))
        self.make_figures()
        print('Plots created and saved.')
    def train_and_minimize(self):

        result,est,opti_flag \
            = opt.neural_network_opt(None, self.training_grid,self.target_values,
                                     estimator=self.estimator_name,
                                     hyper_parameter_dict=self.hyper_parameter_dict,
                                     x_init=self.x_init)
        self.opti_flag = opti_flag
        self.estimator = est
        self.optimization_result = result

        return result,est

    def make_figures(self,**kw):
        fontsize = kw.pop('label_fontsize',16.)
        try:
            optimization_method = eval(self.ma.data_file['Instrument settings'] \
                ['MC'].attrs['optimization_method'])
        except KeyError:
            optimization_method = 'Numerical'
        if self.target_value_names is None:
            self.target_value_names = ['none_label' for i
                                       in range(self.output_dim)]
        self.target_value_names.append(r"$||z||_2 [a.u]$")
        pre_proc_dict = self.estimator.pre_proc_dict
        output_scale = pre_proc_dict.get('output',{}).get('scaling',1.)
        output_means = pre_proc_dict.get('output',{}).get('centering',0.)
        input_scale = pre_proc_dict.get('input',{}).get('scaling',1.)
        input_means = pre_proc_dict.get('input',{}).get('centering',0.)

        #Create data grid for contour plot in case of more than 3 sweep variables,
        #contour plots are only created for the first two variables, one plot for
        #each output variable.
        lower_x = np.min(self.training_grid[:,0])-\
                         0.2*np.std(self.training_grid[:,0])
        upper_x = np.max(self.training_grid[:,0])+\
                         0.2*np.std(self.training_grid[:,0])
        lower_y = np.min(self.training_grid[:,1])-\
                         0.2*np.std(self.training_grid[:,1])
        upper_y = np.max(self.training_grid[:,1])+\
                         0.2*np.std(self.training_grid[:,1])
        x_mesh = (np.linspace(lower_x,upper_x,200)-input_means[0])/input_scale[0]
        y_mesh = (np.linspace(lower_y,upper_y,200)-input_means[1])/input_scale[1]
        alpha_rescaled = (self.optimization_result[2]-input_means[2])/input_scale[2]
        print(alpha_rescaled)
        Xm,Ym = np.meshgrid(x_mesh,y_mesh)
        Zm = np.zeros((self.output_dim+1,200,200))
        for k in range(np.size(x_mesh)):
            for l in range(np.size(y_mesh)):
                # print(Xm[k, l], Ym[k, l])
                pred = self.estimator.predict([[Xm[k,l],Ym[k,l],alpha_rescaled]])[0]
                for j in range(self.output_dim+1):
                    if j==self.output_dim:
                        new_val = 0.
                        for m in range(1,self.output_dim+1):
                            new_val += Zm[j-m,k,l]**2
                        new_val = np.sqrt(new_val)
                    else:
                        new_val = pred[j]*output_scale[j]+output_means[j]
                    Zm[j,k,l] = new_val
        Xm = Xm*input_scale[0] + input_means[0]
        Ym = Ym*input_scale[1] + input_means[1]
        reminder = (self.output_dim+1) % 2
        div = int((self.output_dim+1)/2.)
        fig = plt.figure(figsize=(20,(reminder+div)*6))
        plt_grid = plt.GridSpec(div+reminder,2,hspace=0.4,wspace=0.3)
        textstr = 'Optimization converged to: \n'
        base_figname = 'predictive optimization of '
        for it in range(len(self.ma.parameter_names)-1):
            textstr+='%s: %.3g %s' % (self.ma.parameter_names[it+1],
                                      self.optimization_result[it],
                                      self.ma.parameter_units[it+1])
            textstr+='\n'
            base_figname += self.ma.parameter_names[it]+'_'
        # textstr+='Empirical error: '+'%.2f' % ((1.-self.estimator.score)*100.) +'%'
        figname = self.ma.timestamp_string+' '
        figname += self.estimator_name+' fitted landscape'
        savename = self.ma.timestamp_string + '_' + base_figname
        tot = 0
        for it in range(div+reminder):
            for jt in range(2):
                if reminder and it == (div+reminder-1) and jt == 1:
                    continue
                ax = plt.subplot(plt_grid[it,jt])
                if it == 0 and jt == 0:
                    ax.text(0.98, 0.05, textstr,
                            transform=ax.transAxes,
                            fontsize=11, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(facecolor='white',edgecolor='None',
                            alpha=0.75, boxstyle='round'))

                levels = np.linspace(np.min(Zm[tot]),np.max(Zm[tot]),25)
                CP = ax.contourf(Xm,Ym,Zm[tot],levels,extend='both')
                ax.scatter(self.optimization_result[0],self.optimization_result[1],
                            marker='*',c='white',label='network minimum',s=14)
                ax.scatter((self.x_init[0]*input_scale[0])+input_means[0],
                           (self.x_init[1]*input_scale[1])+input_means[1],
                           marker='o',c='black',s=14,label='initial point')
                ax.scatter(self.training_grid[:,0],self.training_grid[:,1],
                            marker='o',c='r',label='training data',s=8)
                ax.tick_params(axis='both',which='minor',labelsize=14)
                ax.set_ylabel(self.ma.parameter_labels[2],fontsize=fontsize)
                ax.set_xlabel(self.ma.parameter_labels[1],fontsize=fontsize)
                cbar = plt.colorbar(CP,ax=ax,orientation='vertical')
                cbar.ax.set_ylabel(self.target_value_names[tot],fontsize=fontsize)
                ax.set_title('{} fitted landscape'.format(self.target_value_names[tot]))
                ax.legend(loc='upper left',framealpha=0.75,fontsize=fontsize)
                tot = tot+1

        fig.suptitle(figname, fontsize=16.)
        self.ma.save_fig(fig, figname=savename, **kw)

        base_figname = 'CPhase_interpolated landscapes_'+self.ma.timestamp_string
        f, (ax1,ax2) = plt.subplots(1,2)
        f.subplots_adjust(wspace=0.8)
        a_tools.color_plot_interpolated(
            x=self.training_grid[:,0], y=self.training_grid[:,1],
            z=1.-self.target_values[:,0], ax=ax1,N_levels=25,
            zlabel=self.target_value_names[0])
        ax1.plot(self.training_grid[:,0],
                 self.training_grid[:,1], 'o', c='grey')
        ax1.plot(self.optimization_result[0],
                self.optimization_result[1],
                'o', markersize=5, c='w')
        plot_title = self.ma.timestamp_string + '_' +self.ma.measurementstring
        ax2.set_title('Population loss Interpolated')
        textstr = '%s ( %s )' % (self.ma.parameter_names[1],
                                 self.ma.parameter_units[1])
        set_xlabel(ax1, textstr)
        set_xlabel(ax2, textstr)
        textstr = '%s ( %s )' % (self.ma.parameter_names[2],
                                 self.ma.parameter_units[2])
        set_ylabel(ax1, textstr)
        set_ylabel(ax2, textstr)
        ax1.set_title(r"$|\phi_C/\pi-1|$ Interpolated")
        a_tools.color_plot_interpolated(
            x=self.training_grid[:,0], y=self.training_grid[:,1],
            z=self.target_values[:,1], ax=ax2,N_levels=25,
            zlabel=self.target_value_names[1])
        ax2.plot(self.training_grid[:,0],
                 self.training_grid[:,1], 'o', c='grey')
        ax2.plot(self.optimization_result[0],
                 self.optimization_result[1],
                 'o', markersize=5, c='w')
        fig.suptitle(plot_title,fontsize=16.)
        self.ma.save_fig(f, figname=base_figname, **kw)

class Dynamic_phase_Analysis(MeasurementAnalysis):

    def __init__(self, flux_pulse_amp=None,
                 flux_pulse_length=None, TwoD=False,
                 qb_name=None, label='Dynamic_phase_analysis', **kw):

        kw['label'] = label
        kw['h5mode'] = 'r+'
        kw['close_file'] = False

        self.flux_pulse_amp = flux_pulse_amp
        self.flux_pulse_length = flux_pulse_length

        super().__init__(qb_name=qb_name, TwoD=TwoD, **kw)


    def run_default_analysis(self, TwoD=False, **kw):
        close_file = kw.pop('close_file', True)
        super().run_default_analysis(close_file=False,
                                     close_main_fig=True,
                                     save_fig=False, **kw)
            
        self.NoCalPoints = kw.pop('NoCalPoints', 2)
        self.phases = {}
        self.fit_res = {}
        self.fig, self.ax = plt.subplots()

        if TwoD:
            self.twoD_sweep_analysis(**kw)
        else:
            self.oneD_sweep_analysis(**kw)

        textstr = ''
        for label in self.labels:
            textstr += 'phase_{} = {:0.3f} deg\n'.format(label,
                                                         self.phases[label])

        self.dyn_phase = self.phases['with flux pulse'] - \
                         self.phases['no flux pulse']

        textstr += 'dyn_phase_{} = {:0.3f} deg'.format(self.qb_name,
                                                       self.dyn_phase)
        self.add_textbox(textstr, self.fig, self.ax)
        # self.fig.text(0.5, -0.05, textstr, transform=self.ax.transAxes,
        #               fontsize=self.font_size, verticalalignment='top',
        #               horizontalalignment='center', bbox=self.box_props)

        plt.savefig(self.folder+'\\cos_fit.png', dpi=300, bbox_inches='tight')

        if kw.pop('show', False):
            plt.show()

        if kw.pop('close_fig', True):
            plt.close(self.fig)

        if close_file:
            self.data_file.close()

    def oneD_sweep_analysis(self, **kw):

        self.labels = ['with flux pulse', 'no flux pulse']
        cal_points = kw.pop('cal_points', True)

        self.get_naming_and_values()

        gmetadata = self.data_file['Experimental Data/Experimental Metadata']
        if 'preparation_params' in gmetadata.keys():
            if 'reset' in eval(gmetadata['preparation_params'].attrs[
                                   'preparation_type']):
                nreset = eval(gmetadata['preparation_params'].attrs[
                                  'reset_reps'])
            else:
                nreset = 0
        else:
            nreset = 0
        data_filter = lambda data: data[nreset::nreset + 1]
        self.data_filter = data_filter

        length_single = (len(self.sweep_points)//(nreset+1) - self.NoCalPoints)//2
        thetas = data_filter(self.sweep_points)[0:length_single]
        if 'hard_sweep_params' in gmetadata.keys():
            if 'phase' in gmetadata['hard_sweep_params'].keys():
                if 'values' in gmetadata['hard_sweep_params/phase'].keys():
                    thetas = np.array(gmetadata['hard_sweep_params/phase/values'])
                    thetas = thetas[:length_single]/180*np.pi

        if cal_points:
            cal_points_idxs = [[-2], [-1]]

        for i, start_idx in enumerate([0, length_single]):
            dict_label = self.labels[i]

            if len(self.measured_values) == 1:
                measured_values = np.concatenate((
                    data_filter(self.data[1])[start_idx:start_idx+length_single],
                    data_filter(self.data[1])[cal_points_idxs[0] +
                                              cal_points_idxs[1]]
                ))
                self.ampls = a_tools.rotate_and_normalize_data_1ch(
                    measured_values, cal_zero_points=cal_points_idxs[0],
                    cal_one_points=cal_points_idxs[1])[0:-self.NoCalPoints]

            else:
                measured_values = np.zeros(
                    (len(self.measured_values),
                     length_single + len(cal_points_idxs[0]) +
                                     len(cal_points_idxs[1])))
                for j in range(len(self.measured_values)):
                    measured_values[j] = np.concatenate((
                        data_filter(self.data[j+1])[
                            start_idx:start_idx + length_single],
                            data_filter(self.data[j+1])[cal_points_idxs[0] +
                                                        cal_points_idxs[1]]
                     ))

                self.ampls = a_tools.rotate_and_normalize_data(
                    measured_values, cal_zero_points=cal_points_idxs[0],
                    cal_one_points=cal_points_idxs[1])[0][0:-self.NoCalPoints]
            self.fit_and_plot(thetas=thetas, ampls=self.ampls,
                              ax=self.ax, dict_label=dict_label, **kw)

    def twoD_sweep_analysis(self, **kw):
        self.labels = ['no flux pulse', 'with flux pulse']
        self.get_naming_and_values_2D()
        length_single = len(self.sweep_points)
        textstr = ''

        for index, i in enumerate(np.arange(0, len(self.sweep_points_2D) *
                length_single, length_single)):

            dict_label = self.labels[index]
            data_slice = self.data[:, i:i+length_single-1]
            thetas = data_slice[0]

            self.ampls = a_tools.rotate_and_normalize_data_no_cal_points(
                np.array([data_slice[2], data_slice[3]]))

            self.fit_and_plot(thetas=thetas, ampls=self.ampls,
                              ax=self.ax, dict_label=dict_label, **kw)

    def fit_and_plot(self, thetas, ampls, ax, dict_label, **kw):

        print_fit_results = kw.pop('print_fit_results', False)
        # fit to cosine
        fit_res = self.fit_single_cos(thetas, ampls,
                                    print_fit_results=print_fit_results)

        # plot
        ax.plot(thetas, ampls, 'o', label=dict_label)

        thetas_fine = np.linspace(thetas[0], thetas[-1], len(thetas)*100)
        if fit_res.best_values['amplitude'] > 0:
            data_fit_fine = fit_mods.CosFunc(thetas_fine,
                                             **fit_res.best_values)
        else:
            fit_res_temp = deepcopy(fit_res)
            fit_res_temp.best_values['amplitude'] = \
                -fit_res_temp.best_values['amplitude']
            data_fit_fine = fit_mods.CosFunc(thetas_fine,
                                             **fit_res_temp.best_values)

        ax.plot(thetas_fine, data_fit_fine, 'r')

        if self.flux_pulse_amp is not None:
            ax.set_title('dynamic phase msmt {} at {:0.1f}ns {:0.4f}V {}'.format(
                self.qb_name, self.flux_pulse_length*1e9,
                self.flux_pulse_amp, self.timestamp_string))
        else:
            ax.set_title('dynamic phase msmt {} at {:0.1f}ns {}'.format(
                self.qb_name, self.flux_pulse_length*1e9, self.timestamp_string))
        ax.set_xlabel(r'Phase of 2nd pi/2 pulse, $\theta$[rad]')
        ax.set_ylabel('Response (arb. units)')

        ax.legend()
        self.fit_res[dict_label] = fit_res
        self.phases[dict_label] = fit_res.best_values['phase']*180/np.pi

    def fit_single_cos(self, thetas, ampls,
                       print_fit_results=True, phase_guess=0,
                       cal_points=False):

        cos_mod = fit_mods.CosModel
        average = np.mean(ampls)

        diff = 0.5*(max(ampls) -
                    min(ampls))
        amp_guess = -diff
        # offset guess
        offset_guess = average

        # Set up fit parameters and perform fit
        cos_mod.set_param_hint('amplitude',
                               value=amp_guess,
                               vary=True)
        cos_mod.set_param_hint('phase',
                               value=phase_guess,
                               vary=True)
        cos_mod.set_param_hint('frequency',
                               value=1./(2*np.pi),
                               vary=False)
        cos_mod.set_param_hint('offset',
                               value=offset_guess,
                               vary=True)
        self.params = cos_mod.make_params()
        fit_res = cos_mod.fit(data=ampls,
                              t=thetas,
                              params=self.params)

        if fit_res.chisqr > 0.35:
            log.warning('Fit did not converge, chi-square > 0.35')

        if print_fit_results:
            print(fit_res.fit_report())

        if fit_res.best_values['amplitude'] < 0.:
            fit_res.best_values['phase'] += np.pi
            fit_res.best_values['amplitude'] *= -1

        return fit_res


class FluxPulse_Scope_Analysis(MeasurementAnalysis):

    def __init__(self, qb_name=None,
                 sign_of_peaks=None,
                 label='',
                 auto=True,
                 plot=True,
                  **kw):
        '''
        analysis class to analyse data taken in flux pulse scope measurements

        Args:
            qb_name (str): qubit name
            sign_of_peaks (+1 or -1): optional, +1 if the qubit resonance is a peak and -1 if
                                        it is a dip
            label (str): measurement string
            auto (bool): run default analysis if true
            plot (bool): show plot if true
            **kw (dict): keywords passed to the init of the base class
        '''
        kw['label'] = label
        kw['h5mode'] = 'r+'
        self.sign_of_peaks = sign_of_peaks
        self.plot = plot
        self.qb_name = qb_name
        self.data_rotated = np.array([])
        self.fitted_volts = np.array([])

        super().__init__(TwoD=True, auto=False,qb_name=self.qb_name, **kw)
        if auto:
            self.run_default_analysis()

    def run_default_analysis(self, plot=None, **kw):

        if plot is None:
            plot = self.plot
        self.get_naming_and_values_2D()

        error_occured = False
        if len(self.exp_metadata) != 0:
            try:
                self.delays = self.exp_metadata['sweep_points_dict'][self.qb_name]
                self.freqs = self.exp_metadata['sweep_points_dict_2D'][self.qb_name]
                cp = self.exp_metadata.get('cal_points', None)
                if cp is not None:
                    cp = eval(cp)
                    self.delays = cp.extend_sweep_points(self.delays, self.qb_name)
            except KeyError:
                error_occured = True
        if error_occured:
            self.delays = self.sweep_points
            self.freqs = self.sweep_points_2D

        data_rotated = a_tools.rotate_and_normalize_data_no_cal_points(
            self.data[2:, :])

        data_rotated = data_rotated.reshape(len(self.freqs), len(self.delays))
        self.data_rotated = data_rotated

        if self.sign_of_peaks is None:
            self.sign_of_peaks = np.sign(np.mean(self.data_rotated[:, 0]) -
                                         np.median(self.data_rotated[:, 0]))
        self.fit_all()

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.delays/1e-9, self.freqs/1e9, data_rotated/1e-3,
                           cmap='viridis')
        ax.autoscale(tight=True)

        axc = plt.colorbar(im)
        axc.set_label('transmission, $|S_{21}|$ (mV)')

        ax.set_xlabel(r'delay, $\tau$ (ns)')
        ax.set_ylabel(r'drive frequency, $f_d$ (GHz)')
        ax.set_title('{} {}'.format(self.timestamp_string, self.measurementstring))

        ax.plot(self.delays/1e-9, self.fitted_freqs/1e9, 'r', label='fitted freq.')
        ax.legend()

        plt.savefig('{}//{}_flux_pulse_scope_{}.png'.format(self.folder,
                                                            self.timestamp_string,
                                                            self.qb_name))

        if plot:
            plt.show()
        else:
            plt.close()

    def fit_single_slice(self, data_slice, sigma_guess=10e6,
                         sign_of_peaks=None,
                         plot=False, print_res=False):

        GaussianModel = fit_mods.GaussianModel

        mu_guess = self.sweep_points_2D[np.argmax(data_slice*sign_of_peaks)]
        ampl_guess = (data_slice.max() - data_slice.min())/0.4*sign_of_peaks*sigma_guess
        offset_guess = data_slice[0]

        GaussianModel.set_param_hint('sigma',value=sigma_guess,vary=True)
        GaussianModel.set_param_hint('mu',value=mu_guess,vary=True)
        GaussianModel.set_param_hint('ampl',value=ampl_guess,vary=True)
        GaussianModel.set_param_hint('offset',value=offset_guess,vary=True)

        fit_res = GaussianModel.fit(data_slice, freq=self.sweep_points_2D)
        if plot:
            fit_res.plot()
        if print_res:
            print(fit_res.fit_report())

        return fit_res

    def fit_all(self, plot=False, return_stds=False, sign_of_peaks=None):

        if sign_of_peaks is None:
            sign_of_peaks = self.sign_of_peaks

        delays = self.sweep_points

        fitted_freqs = np.zeros(len(delays))
        fitted_stds = np.zeros(len(delays))

        for i,delay in enumerate(delays):
            data_slice = self.data_rotated[:,i]
            fit_res = self.fit_single_slice(data_slice,
                                            sign_of_peaks=sign_of_peaks,
                                            plot=False, print_res=False)
            self.fit_res = fit_res
            fitted_freqs[i] = fit_res.best_values['mu']
            if fit_res.covar is not None:
                fitted_stds[i] = np.sqrt(fit_res.covar[2, 2])
            else:
                fitted_stds[i] = 0

        if plot:
            fig, ax = plt.subplots()
            if return_stds:
                ax.errorbar(delays/1e-9, fitted_freqs/1e6, yerr=fitted_stds/1e6)
            else:
                ax.plot(delays/1e-9, fitted_freqs/1e6)
            ax.set_xlabel(r'delay, $\tau$ (ns)')
            ax.set_ylabel(r'fitted qubit frequency, $f_q$ (MHz)')
            plt.show()

        self.fitted_freqs = fitted_freqs
        self.fitted_stds = fitted_stds

        if return_stds:
            return fitted_freqs, fitted_stds
        else:
            return fitted_freqs

    def freq_to_volt(self, freq, f_sweet_spot, f_parking, f_pulsed, pulse_amp):
        '''
        frequency to voltage conversion (based an the sqrt(cos(...)) model)

        Args:
            freq: frequency in Hz
            f_sweet_spot: sweet spot frequency in Hz
            f_parking: parking frequency in Hz
            f_pulsed: steady state pulsed frequency in Hz (at the end of the pulse)
            pulse_amp: flux pulse amplitude in volt

        Returns:
            flux pulse voltage
        '''

        phi0 = np.arccos(f_parking**2/f_sweet_spot**2)
        dphi_dV = (phi0 - np.arccos(f_pulsed**2/f_sweet_spot**2))/pulse_amp

        voltage = (phi0 - np.arccos((freq/f_sweet_spot)**2))/dphi_dV
        return voltage

    def convert_freqs_to_volts(self,  f_sweet_spot, f_parking,
                               f_pulsed, pulse_amp, plot=False):

        self.fitted_volts = self.freq_to_volt(self.fitted_freqs,
                                              f_sweet_spot=f_sweet_spot,
                                              f_parking=f_parking,
                                              f_pulsed=f_pulsed,
                                              pulse_amp=pulse_amp)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.sweep_points/1e-9, self.fitted_volts)
            ax.set_xlabel(r'delay, $\tau$ (ns)')
            ax.set_ylabel(r'fitted voltage, $U$ (V)')
            plt.show()

        return self.fitted_volts


class FluxPulse_timing_calibration(FluxPulse_Scope_Analysis):

    def __init__(self, qb_name=None, sign_of_peaks=None, flux_pulse_length=None,
                 return_fit=False,
                 label='',
                 auto=True, plot=True,
                 **kw
                 ):
        '''
        analysis class for the flux pulse timing calibration

        Args:

            qb_name (str): qubit name
            sign_of_peaks (+1 or -1): optional, +1 if the qubit resonance is a peak and -1 if
                                        it is a dip
            flux_pulse_length (float): length of the flux pulse
            return_fit (bool): fit results returned if true
            label (str): measurement string
            auto (bool): run default analysis if true
            plot (bool): show plot if true
            **kw (dict): keywords passed to the init of the base class
        '''
        self.qb_name = qb_name
        self.sign_of_peaks = sign_of_peaks
        self.flux_pulse_length = flux_pulse_length
        self.return_fit = return_fit

        self.label = label
        self.plot = plot

        self.delay_fit_res = None
        self.fitted_delay = None
        self.fig = None
        self.ax = None

        super().__init__(qb_name=self.qb_name,
                         sign_of_peaks=self.sign_of_peaks,
                         label=self.label,
                         auto=True,
                         plot=self.plot,
                         **kw)
        if auto:
            self.run_delay_analysis()

    def run_delay_analysis(self,
                          plot=None,
                          close_file=True, **kw):

        if plot is None:
            plot = self.plot

        self.add_analysis_datagroup_to_file()

        show_guess = kw.get('show_guess', False)
        print_fit_results = kw.get('print_fit_results', True)

        #get the fit results (lmfit.ModelResult) and save them
        self.fitted_delay, self.delay_fit_res = self.fit_delay(
           plot=False,
           print_fit_results=print_fit_results,
           return_fit=True)

        self.save_fitted_parameters(self.delay_fit_res, var_name=self.value_names[0])

        #Plot results
        self.plot_delay_fit(plot=plot, show_guess=show_guess)

        if close_file:
           self.data_file.close()

        return self.delay_fit_res

    def fit_delay(self, flux_pulse_length=None, plot=False,
                  print_fit_results=False, return_fit=None):
        '''
        method to fit the relative delay of the flux pulse to the drive pulses
        '''

        if flux_pulse_length is None:
            flux_pulse_length = self.flux_pulse_length
        if flux_pulse_length is None:
            raise ValueError('Must specify the flux pulse '
                             'length used in the experiment.')
        if return_fit is None:
            return_fit = self.return_fit

        erf_window_model = fit_mods.ErfWindowModel

        offset_guess = self.fitted_freqs[0]

        i_amplitude_guess = np.abs(offset_guess - self.fitted_freqs).argmax()
        amplitude_guess = self.fitted_freqs[i_amplitude_guess] - offset_guess

        i_list_window = np.where(np.abs(self.fitted_freqs - offset_guess) >
                                 np.abs(amplitude_guess/2.))[0]

        t_start_guess = self.sweep_points[i_list_window[0]]
        t_end_guess = self.sweep_points[i_list_window[-1]]


        #Set up fit parameters and perform fit
        erf_window_model.set_param_hint('offset',
                                        value=offset_guess,
                                        vary=True)
        erf_window_model.set_param_hint('amplitude',
                                        value=amplitude_guess,
                                        vary=True)
        erf_window_model.set_param_hint('t_start',
                                        value=t_start_guess,
                                        vary=True)
        erf_window_model.set_param_hint('t_end',
                                        value=t_end_guess,
                                        vary=True)
        erf_window_model.set_param_hint('t_rise',
                                        value=(t_end_guess - t_start_guess)/20.,
                                        vary=True)

        params_delay_fit = erf_window_model.make_params()
        self.delay_fit_res = erf_window_model.fit(data=self.fitted_freqs,
                                                  t=self.sweep_points,
                                                  params=params_delay_fit)

        if self.delay_fit_res.chisqr > 1.:
            log.warning('Fit did not converge, chi-square > 1.')

        self.fitted_delay = (self.delay_fit_res.best_values['t_end'] +
                             self.delay_fit_res.best_values['t_start'])/2. - flux_pulse_length/2.

        if print_fit_results:
            print(self.delay_fit_res.fit_report())
            print('fitted delay  = {}ns'.format(self.fitted_delay/1e-9))



        if plot:
            self.delay_fit_res.plot()

        if return_fit:
            return self.fitted_delay, self.delay_fit_res
        else:
            return self.fitted_delay

    def plot_delay_fit(self, plot=True, show_guess=False):

        self.fig, self.ax = plt.subplots()

        self.ax.plot(self.sweep_points/1e-9, self.delay_fit_res.data/1e6, 'k.',
                     label='data')

        #Used for plotting the fit
        best_vals = self.delay_fit_res.best_values
        erf_window_fit_func = lambda t: fit_mods.ErfWindow(
            t,
            t_start=best_vals['t_start'],
            t_end=best_vals['t_end'],
            t_rise=best_vals['t_rise'],
            amplitude=best_vals['amplitude'],
            offset=best_vals['offset'])


        # plot with initial guess
        if show_guess:
            self.ax.plot(self.sweep_points/1e-9,
                         self.delay_fit_res.init_fit/1e6, 'k--',
                         linewidth=self.line_width,
                         label='initial guess')

        #plot with best fit results
        x = np.linspace(self.sweep_points[0],
                        self.sweep_points[-1],
                        len(self.sweep_points)*100)
        y = erf_window_fit_func(x)
        self.ax.plot(x/1e-9, y/1e6, 'r-', linewidth=self.line_width,label='fit')
        self.ax.set_title('fitted delay = {:4.3f}ns'.format(self.fitted_delay/1e-9))
        self.ax.set_xlabel(r'pulse delay, $\tau$ (ns)')
        self.ax.set_ylabel(r'fitted frequency, $f$ (MHz)')

        if show_guess:
            self.ax.legend(['data', 'guess', 'fit'])
        else:
            self.ax.legend(['data', 'fit'])

        plt.tight_layout()
        plt.savefig('{}//{}_timing_calibration_fit_{}.png'.format(self.folder,
                                                            self.timestamp_string,
                                                            self.qb_name))
        if plot:
            plt.show()
        else:
            plt.close()
