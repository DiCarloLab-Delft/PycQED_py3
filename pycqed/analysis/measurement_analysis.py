import os
import logging
import numpy as np
from scipy import stats
import h5py
from matplotlib import pyplot as plt
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as optimize
import lmfit
from collections import Counter  # used in counting string fractions
import textwrap
from scipy.interpolate import interp1d
import pylab
from pycqed.analysis.tools import data_manipulation as dm_tools
import imp
import math
from math import erfc
from scipy.signal import argrelextrema, argrelmax, argrelmin
from copy import deepcopy
import qutip as qtp
from pycqed.analysis import tomography as tomo_mod

try:
    from nathan_plotting_tools import *
except:
    pass
from pycqed.analysis import composite_analysis as ca

imp.reload(dm_tools)


class MeasurementAnalysis(object):

    def __init__(self, TwoD=False, folder=None, auto=True,
                 cmap_chosen='viridis', **kw):
        if folder is None:
            self.folder = a_tools.get_folder(**kw)
        else:
            self.folder = folder
        self.load_hdf5data(**kw)
        self.fit_results = []
        self.box_props = dict(boxstyle='Square', facecolor='white', alpha=0.8)
        self.cmap_chosen = cmap_chosen
        if auto is True:
            self.run_default_analysis(TwoD=TwoD, **kw)

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
        if not os.path.exists(os.path.join(self.folder, name+'.hdf5')):
            mode = 'w'
        else:
            mode = 'r+'
        return h5py.File(os.path.join(self.folder, name+'.hdf5'), mode)

    def default_fig(self, **kw):
        figsize = kw.pop('figsize', None)
        return plt.figure(figsize=figsize, **kw)

    def default_ax(self, fig=None, *arg, **kw):
        if fig is None:
            fig = self.default_fig(*arg, **kw)
        ax = fig.add_subplot(111)
        ax.set_title(self.timestamp_string+'\n'+self.measurementstring)
        ax.ticklabel_format(useOffset=False)
        return fig, ax

    def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            if figname is None:
                figname = (self.sweep_name+'_'+xlabel +
                           '_vs_'+ylabel+'.'+plot_format)
            else:
                figname = (figname+'.' + plot_format)
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
            self.figarray, self.axarray = plt.subplots(
                val_len, 1, figsize=(min(6*len(self.value_names), 11),
                                     1.5*len(self.value_names)))
        else:
            self.figarray, self.axarray = plt.subplots(
                max(len(self.value_names), 1), 1,
                figsize=(6, 1.5*len(self.value_names)))
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
            '\n'+'*'*80+'\n' + \
            lmfit.fit_report(fit_res) + \
            '\n'+'*'*80 + '\n\n'

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
            weight = ((fit_res.data - mean)/std)**weights
            weighted_chisqr = np.sum(weight*(fit_res.data-fit_res.best_fit)**2)
            fit_grp.attrs.create(name='weighted_chisqr', data=weighted_chisqr)

    def run_default_analysis(self, TwoD=False, close_file=True,
                             show=False, log=False, transpose=False, **kw):
        if TwoD is False:
            self.get_naming_and_values()
            self.sweep_points = kw.pop('sweep_points', self.sweep_points)
            # Preallocate the array of axes in the figure
            # Creates either a 2x2 grid or a vertical list
            if len(self.value_names) == 4:
                fig, axs = plt.subplots(
                    nrows=int(len(self.value_names)/2), ncols=2,
                    figsize=(min(6*len(self.value_names), 11),
                             1.5*len(self.value_names)))
            else:
                fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                        figsize=(5, 3*len(self.value_names)))
                # Add all the sweeps to the plot 1 by 1
                # indices are determined by it's shape/number of sweeps
            for i in range(len(self.value_names)):
                if len(self.value_names) == 1:
                    ax = axs
                elif len(self.value_names) == 2:
                    ax = axs[i % 2]
                elif len(self.value_names) == 4:
                    ax = axs[i/2, i % 2]
                else:
                    ax = axs[i]  # If not 2 or 4 just gives a list of plots
                if i != 0:
                    plot_title = ' '
                else:
                    plot_title = kw.pop('plot_title', textwrap.fill(
                                        self.timestamp_string + '_' +
                                        self.measurementstring, 40))
                ax.ticklabel_format(useOffset=False)
                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=fig, ax=ax, log=log,
                                                xlabel=self.xlabel,
                                                ylabel=self.ylabels[i],
                                                save=False, show=show,
                                                plot_title=plot_title)

        elif TwoD is True:
            self.get_naming_and_values_2D()
            self.sweep_points = kw.pop('sweep_points', self.sweep_points)
            self.sweep_points_2D = kw.pop(
                'sweep_points_2D', self.sweep_points_2D)

            if len(self.value_names) == 4:
                fig, axs = plt.subplots(len(self.value_names)/2, 2,
                                        figsize=(min(6*len(self.value_names),
                                                     11),
                                                 1.5*len(self.value_names)))
            else:
                fig, axs = plt.subplots(max(len(self.value_names), 1), 1,
                                        figsize=(5, 3*len(self.value_names)))

            for i in range(len(self.value_names)):
                if len(self.value_names) == 1:
                    ax = axs
                elif len(self.value_names) == 2:
                    ax = axs[i % 2]
                elif len(self.value_names) == 4:
                    ax = axs[i/2, i % 2]
                else:
                    ax = axs[i]  # If not 2 or 4 just gives a list of plots
                a_tools.color_plot(
                    x=self.sweep_points,
                    y=self.sweep_points_2D,
                    z=self.measured_values[i].transpose(),
                    plot_title=self.zlabels[i],
                    fig=fig, ax=ax,
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    zlabel=self.zlabels[i],
                    save=False,
                    transpose=transpose,
                    cmap_chosen=self.cmap_chosen,
                    **kw)

            fig.tight_layout(h_pad=1.5)
            fig.subplots_adjust(top=0.9)
            plot_title = '{timestamp}_{measurement}'.format(
                timestamp=self.timestamp_string,
                measurement=self.measurementstring)
            fig.suptitle(plot_title, fontsize=18)
            # Make space for title

        self.save_fig(fig, fig_tight=False, **kw)

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
                    self.value_names[i] + '('+value_units[i]+')'))
            self.xlabel = str(self.sweep_name + '('+self.sweep_unit+')')
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

            self.xlabel = self.parameter_names[0] + ' (' +  \
                self.parameter_units[0] + ')'
            self.parameter_labels = [a+' (' + b + ')' for a, b in zip(
                                     self.parameter_names,
                                     self.parameter_units)]

            self.ylabels = [a+' (' + b + ')' for a, b in zip(self.value_names,
                                                             self.value_units)]
        else:
            raise ValueError('datasaving_format "%s " not recognized'
                             % datasaving_format)

    def plot_results_vs_sweepparam(self, x, y, fig, ax, show=False, marker='-o',
                                   log=False, label=None, **kw):
        save = kw.pop('save', False)
        self.plot_title = kw.pop('plot_title',
                                 textwrap.fill(self.timestamp_string + '_' +
                                               self.measurementstring, 40))
        xlabel = kw.pop('xlabel', None)
        ylabel = kw.pop('ylabel', None)
        ax.set_title(self.plot_title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.plot(x, y, marker, label=label)
        if log:
            ax.set_yscale('log')
        if show:
            plt.show()
        if save:
            if log:
                # litle hack to only change savename if logarithmic
                self.save_fig(fig, xlabel=xlabel, ylabel=(ylabel+'_log'), **kw)
            else:
                self.save_fig(fig, xlabel=xlabel, ylabel=ylabel, **kw)
        return

    def plot_complex_results(self, cmp_data, fig, ax, show=False, marker='.', **kw):
        '''
        Plot real and imaginary values measured vs a sweeped parameter
        Example: complex S21 of a resonator

        Author: Stefano Poletto
        Data: November 15, 2016
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
                    self.value_names[i] + '('+value_units[i]+')'))
            self.xlabel = str(self.sweep_name + '('+self.sweep_unit+')')
            self.ylabel = str(self.sweep_name_2D + '('+self.sweep_unit_2D+')')

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
            # X,Y,Z can be put in colormap directly
            self.X = x.reshape(-1, cols)
            self.Y = y.reshape(-1, cols)
            self.sweep_points = self.X[0]
            self.sweep_points_2D = self.Y.T[0]

            if len(self.value_names) == 1:
                z = self.data[2]
                self.Z = z.reshape(-1, cols)
                self.measured_values = [self.Z.T]
            else:
                self.Z = []
                self.measured_values = []
                for i in range(len(self.value_names)):
                    z = self.data[2+i]
                    Z = z.reshape(-1, cols)
                    self.Z.append(Z)
                    self.measured_values.append(Z.T)

            self.xlabel = self.parameter_names[0] + ' (' +  \
                self.parameter_units[0] + ')'
            self.ylabel = self.parameter_names[1] + ' (' +  \
                self.parameter_units[1] + ')'

            self.parameter_labels = [a+' (' + b + ')' for a, b in zip(
                                     self.parameter_names,
                                     self.parameter_units)]

            self.zlabels = [a+' (' + b + ')' for a, b in zip(self.value_names,
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
                norm_chisq = chisqr/np.std(self.measured_values[i])
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
        base_figname = 'optimization of ' + self.value_names[0]
        if np.shape(self.sweep_points)[0] == 2:
            f, ax = plt.subplots()
            a_tools.color_plot_interpolated(
                x=self.sweep_points[0], y=self.sweep_points[1],
                z=self.measured_values[0], ax=ax,
                zlabel=self.value_names[0])
            ax.set_xlabel(self.parameter_labels[0])
            ax.set_ylabel(self.parameter_labels[1])
            ax.plot(self.sweep_points[0], self.sweep_points[1], '-o', c='grey')
            ax.plot(self.sweep_points[0][-1], self.sweep_points[1][-1],
                    'o', markersize=5, c='w')
            plot_title = kw.pop('plot_title', textwrap.fill(
                                self.timestamp_string + '_' +
                                self.measurementstring, 40))
            ax.set_title(plot_title)

            self.save_fig(f, figname=base_figname, **kw)


class OptimizationAnalysis(MeasurementAnalysis):

    def run_default_analysis(self, close_file=True, show=False, plot_all=False, **kw):
        self.get_naming_and_values()
        try:
            optimization_method = self.data_file['Instrument settings']\
                ['MC'].attrs['optimization_method']
        except:
            optimization_method = 'Numerical'
            # This is because the MC is no longer an instrument and thus
            # does not get saved, I have to re add this (MAR 1/2016)
            logging.warning('Could not extract optimization method from' +
                            ' data file')

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
                                              4*len(self.parameter_names)))
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
                                              4*len(self.parameter_names)))
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
                 rotate_and_normalize=True,
                 plot_cal_points=True, **kw):
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
        super(TD_Analysis, self).__init__(**kw)

    # def run_default_analysis(self, close_file=True, **kw):
    #     self.get_naming_and_values()
    #     self.fit_data(**kw)
    #     self.make_figures(**kw)
    #     if close_file:
    #         self.data_file.close()
    #     return self.fit_res

    def rotate_and_normalize_data(self):
        if self.cal_points is None:
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

    def run_default_analysis(self,
                             close_main_fig=True,  **kw):
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        if self.rotate_and_normalize:
            self.rotate_and_normalize_data()
        else:
            self.corr_data = self.measured_values[0]
        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points'.encode('utf-8'))

        # Plotting
        if self.make_fig:
            self.fig1, fig2, self.ax1, axarray = self.setup_figures_and_axes()
            # print(len(self.value_names))
            for i in range(len(self.value_names)):
                if len(self.value_names) >= 4:
                    ax = axarray[i/2, i % 2]
                elif len(self.value_names) == 1:
                    ax = self.ax1
                else:
                    ax = axarray[i]
                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=fig2, ax=ax,
                                                xlabel=self.xlabel,
                                                ylabel=str(
                                                    self.value_names[i]),
                                                save=False)
            ylabel = r'$F$ $\left(|1 \rangle \right)$'

            if self.plot_cal_points:
                x = self.sweep_points
                y = self.corr_data
            else:
                logging.warning('not tested for all types of calpoints')
                if type(self.cal_points[0]) is int:
                    x = self.sweep_points[:-2]
                    y = self.corr_data[:-2]
                else:
                    x = self.sweep_points[:-1*(len(self.cal_points[0])*2)]
                    y = self.corr_data[:-1*(len(self.cal_points[0])*2)]

            self.plot_results_vs_sweepparam(x=x,
                                            y=y,
                                            fig=self.fig1, ax=self.ax1,
                                            xlabel=self.xlabel,
                                            ylabel=ylabel,
                                            save=False)
            self.ax1.set_ylim(min(min(self.corr_data)-.1, -.1),
                              max(max(self.corr_data)+.1, 1.1))
            if not close_main_fig:
                # Hacked in here, good idea to only show the main fig but can
                # be optimized somehow
                self.save_fig(self.fig1, ylabel='Amplitude (normalized)',
                              close_fig=False, **kw)
            else:
                self.save_fig(self.fig1, ylabel='Amplitude (normalized)', **kw)
            self.save_fig(fig2, ylabel='Amplitude', **kw)
        if close_file:
            self.data_file.close()
        return

    def normalize_data_to_calibration_points(self, values, calsteps,
                                             save_norm_to_data_file=True):
        '''
        Rotates and normalizes the data based on the calibration points.

        values: array of measured values, uses only the length of this
        calsteps: number of points that corresponds to calibration points
        '''
        NoPts = len(values)
        cal_zero_points = list(range(NoPts-int(calsteps),
                                     int(NoPts-int(calsteps)/2)))
        cal_one_points = list(range(int(NoPts-int(calsteps)/2), NoPts))

        self.corr_data = a_tools.rotate_and_normalize_data(
            self.measured_values[0:2], cal_zero_points, cal_one_points)[0]
        if save_norm_to_data_file:
            self.add_dataset_to_analysisgroup('Corrected data',
                                              self.corr_data)
            self.analysis_group.attrs.create('corrected data based on',
                                             'calibration points'.encode('utf-8'))
        normalized_values = self.corr_data
        normalized_data_points = normalized_values[:-int(calsteps)]
        normalized_cal_vals = normalized_values[-int(calsteps):]
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
                             close_main_fig=True,  **kw):
        super(chevron_optimization_v1, self).run_default_analysis(**kw)
        sweep_points_wocal = self.sweep_points[:-4]
        measured_values_wocal = self.measured_values[0][:-4]

        output_fft = np.real_if_close(np.fft.rfft(measured_values_wocal))
        ax_fft = np.fft.rfftfreq(len(measured_values_wocal),
                                 d=sweep_points_wocal[1]-sweep_points_wocal[0])
        order_mask = np.argsort(ax_fft)
        y = output_fft[order_mask]
        y = y/np.sum(np.abs(y))

        u = np.where(np.arange(len(y)) == 0, 0, y)
        array_peaks = a_tools.peak_finder(np.arange(len(np.abs(y))),
                                          np.abs(u),
                                          window_len=0)
        if array_peaks['peak_idx'] is None:
            self.period = 0.
            self.cost_value = 100.
        else:
            self.period = 1./ax_fft[order_mask][array_peaks['peak_idx']]
            if self.period == np.inf:
                self.period = 0.
            if self.cost_function == 0:
                self.cost_value = -np.abs(y[array_peaks['peak_idx']])
            else:
                self.cost_value = self.get_cost_value(sweep_points_wocal,
                                                      measured_values_wocal)

    def get_cost_value(self, x, y):
        num_periods = np.floor(x[-1]/self.period)
        if num_periods == np.inf:
            num_periods = 0
        # sum of mins
        sum_min = 0.
        for i in range(int(num_periods)):
            sum_min += np.interp((i+0.5)*self.period, x, y)
            # print(sum_min)

        # sum of maxs
        sum_max = 0.
        for i in range(int(num_periods)):
            sum_max += 1.-np.interp(i*self.period, x, y)
            # print(sum_max)

        return sum_max+sum_min


class chevron_optimization_v2(TD_Analysis):

    def __init__(self, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 plot_cal_points=True, **kw):
        super(chevron_optimization_v2, self).__init__(**kw)

    def run_default_analysis(self,
                             close_main_fig=True,  **kw):
        super(chevron_optimization_v2, self).run_default_analysis(**kw)
        measured_values = a_tools.normalize_data_v3(self.measured_values[0])
        self.cost_value_1, self.period = self.sum_cost(self.sweep_points*1e9,
                                                       measured_values)
        self.cost_value_2 = self.swap_cost(self.sweep_points*1e9,
                                           measured_values)
        self.cost_value = [self.cost_value_1, self.cost_value_2]

        fig, ax = plt.subplots(1, figsize=(8, 6))

        min_idx, max_idx = self.return_max_min(self.sweep_points*1e9,
                                               measured_values, 1)
        ax.plot(self.sweep_points*1e9, measured_values, 'b-')
        ax.plot(self.sweep_points[min_idx]*1e9, measured_values[min_idx], 'r*')
        ax.plot(self.sweep_points[max_idx]*1e9, measured_values[max_idx], 'g*')
        ax.plot(self.period*0.5, self.cost_value_2, 'b*', label='SWAP cost')
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
        self.cost_value_1, self.period = self.sum_cost(self.sweep_points*1e9,
                                                       measured_values)
        self.cost_value_2 = self.swap_cost(self.sweep_points*1e9,
                                           measured_values)
        self.cost_value = [self.cost_value_1, self.cost_value_2]

        min_idx, max_idx = self.return_max_min(self.sweep_points*1e9,
                                               measured_values, 1)
        ax.plot(self.sweep_points*1e9, measured_values, 'b-')
        ax.plot(self.sweep_points[min_idx]*1e9, measured_values[min_idx], 'r*')
        ax.plot(self.sweep_points[max_idx]*1e9, measured_values[max_idx], 'g*')
        ax.plot(self.period*0.5, self.cost_value_2, 'b*', label='SWAP cost')
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
        diff_filtered = np.where(np.abs(diff-avg) < std, diff, np.nan)
        diff_filtered = diff_filtered[~np.isnan(diff_filtered)]
    #     diff_filtered = diff
        return 2.*np.mean(diff_filtered), np.std(diff_filtered)

    def spec_power(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 1)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        f = 1./period

        output_fft = np.real_if_close(np.fft.rfft(y_points))
        ax_fft = np.fft.rfftfreq(len(y_points),
                                 d=x_points[1]-x_points[0])
        order_mask = np.argsort(ax_fft)
        y = output_fft[order_mask]
        y = y/np.sum(np.abs(y))
        return -np.interp(f, ax_fft, np.abs(y))

    def sum_cost(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 4)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        num_periods = np.floor(x_points[-1]/period)

        sum_min = 0.
        for i in range(int(num_periods)):
            sum_min += np.interp((i+0.5)*period, x_points, y_points)
        sum_max = 0.
        for i in range(int(num_periods)):
            sum_max += 1.-np.interp(i*period, x_points, y_points)

        return sum_max+sum_min, period

    def swap_cost(self, data_x, data_y):
        x_points = data_x[:-4]
        y_points = data_y[:-4]
        min_idx, max_idx = self.return_max_min(data_x, data_y, 4)
        period, st = self.get_period(data_x[min_idx], data_x[max_idx])
        return np.interp(period*0.5, x_points, y_points)


class Rabi_Analysis(TD_Analysis):

    def __init__(self, label='Rabi', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        self.fit_data(**kw)
        self.make_figures(**kw)
        if close_file:
            self.data_file.close()
        return self.fit_res

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 6))
        x_fine = np.linspace(min(self.sweep_points), max(self.sweep_points),
                             1000)
        for i in [0, 1]:
            if i == 0:
                plot_title = kw.pop('plot_title', textwrap.fill(
                                    self.timestamp_string + '_' +
                                    self.measurementstring, 40))
            else:
                plot_title = ''
            self.axs[i].ticklabel_format(useOffset=False)
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=self.fig, ax=self.axs[i],
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            save=False,
                                            plot_title=plot_title)

            fine_fit = self.fit_res[i].model.func(
                x_fine, **self.fit_res[i].best_values)
            self.axs[i].plot(x_fine, fine_fit, label='fit')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    x_fine, **self.fit_res[i].init_values)
                self.axs[i].plot(x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)

    def fit_data(self, print_fit_results=False, **kw):
        model = fit_mods.CosModel
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.
        for i in [0, 1]:
            params = model.guess(model, data=self.measured_values[i],
                                 t=self.sweep_points)
            self.fit_res[i] = fit_mods.CosModel.fit(
                data=self.measured_values[i],
                t=self.sweep_points,
                params=params)
            try:
                self.add_analysis_datagroup_to_file()
                self.save_fitted_parameters(fit_res=self.fit_res[i],
                                            var_name=self.value_names[i])
            except Exception as e:
                logging.warning(e)


class TD_UHFQC(TD_Analysis):

    def __init__(self, NoCalPoints=4, center_point=31, make_fig=True,
                 zero_coord=None, one_coord=None, cal_points=None,
                 plot_cal_points=True, **kw):
        super(TD_UHFQC, self).__init__(**kw)

    def run_default_analysis(self,
                             close_main_fig=True,  **kw):
        super(TD_UHFQC, self).run_default_analysis(**kw)
        measured_values = a_tools.normalize_data_v3(self.measured_values[0])

        fig, ax = plt.subplots(1, figsize=(8, 6))

        ax.plot(self.sweep_points*1e9, measured_values, '-o')
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'$F |1\rangle$')

        ax.set_title('%s: TD Scan' % self.timestamp_string)

        self.save_fig(fig, fig_tight=False, **kw)


class Echo_analysis(TD_Analysis):

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        self.rotate_and_normalize_data()
        self.fit_data(**kw)
        self.make_figures(**kw)
        if close_file:
            self.data_file.close()
        return self.fit_res
        pass

    def fit_data(self, print_fit_results=False, **kw):

        self.add_analysis_datagroup_to_file()
        # Instantiating here ensures models have no memory of constraints
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
                                        xlabel=self.xlabel,
                                        ylabel=r'F$|1\rangle$',
                                        save=False,
                                        plot_title=plot_title)

        self.ax.plot(x_fine, self.fit_res.eval(t=x_fine), label='fit')
        textstr = '$T_2$={:.3g}$\pm$({:.3g})s '.format(
            self.fit_res.params['tau'].value,
            self.fit_res.params['tau'].stderr)
        if show_guess:
            self.ax.plot(x_fine, self.fit_res.eval(
                t=x_fine, **self.fit_res.init_values), label='guess')
            self.ax.legend(loc='best')

        self.ax.ticklabel_format(style='sci', scilimits=(0, 0))
        self.ax.text(0.4, 0.95, textstr, transform=self.ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=self.box_props)
        self.save_fig(self.fig, fig_tight=True, **kw)


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


class Motzoi_XY_analysis(TD_Analysis):

    '''
    Analysis for the Motzoi XY sequence (Xy-Yx)
    Extracts the alternating datapoints and then fits two polynomials.
    The intersect of the fits corresponds to the optimum motzoi parameter.
    '''

    def __init__(self, label='Motzoi', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self, close_file=True, close_main_fig=True, **kw):
        self.get_naming_and_values()
        self.add_analysis_datagroup_to_file()
        self.cal_points = kw.pop('cal_point', [[-4, -3], [-2, -1]])
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
        b_vals0 = self.fit_res[0].best_values
        b_vals1 = self.fit_res[1].best_values
        x1, x2 = a_tools.solve_quadratic_equation(
            b_vals1['a']-b_vals0['a'], b_vals1['b']-b_vals0['b'],
            b_vals1['c']-b_vals0['c'])
        self.optimal_motzoi = min(x1, x2, key=lambda x: abs(x))
        return self.optimal_motzoi


class Rabi_Analysis_old(TD_Analysis):

    '''
    This is the old Rabi analysis for the mathematica sequences of 60 points
    '''

    def __init__(self, label='Rabi',  **kw):
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
        fit_res = [None]*len(self.value_names)

        for i, name in enumerate(self.value_names):
            offset_estimate = np.mean(self.measured_values[i])
            if (np.mean(self.measured_values[i][30:34]) <
                    np.mean(self.measured_values[i][34:38])):
                amplitude_sign = -1.
            else:
                amplitude_sign = 1.
            amplitude_estimate = amplitude_sign*abs(max(
                self.measured_values[i])-min(self.measured_values[i]))/2
            w = np.fft.fft(
                self.measured_values[i][:-self.NoCalPoints]-offset_estimate)
            index_of_fourier_maximum = np.argmax(np.abs(w[1:len(w)/2]))+1
            fourier_index_to_freq = 1/abs(self.sweep_points[0] -
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
                                             min=0, max=1/8.)
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
                            len(displaced_fitting_axis)*100)

            y = fit_mods.CosFunc(x,
                                 frequency=best_vals['frequency'],
                                 phase=best_vals['phase'],
                                 amplitude=best_vals['amplitude'],
                                 offset=best_vals['offset'])
            axarray[i].plot(x+self.center_point, y, 'r-')

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
        self.save_fig(fig, figname=self.sweep_name+'Rabi_fit', **kw)
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
            (len(self.sweep_points)-self.NoCalPoints)/1.5
        sorted_swp = np.sort(self.sweep_points)
        # Sorting needed for when data is taken in other than ascending order
        step_per_index = sorted_swp[1] - sorted_swp[0]
        desired_period = desired_period_in_indices * step_per_index
        # calibration points max should be at -20
        # and + 20 from the center -> period of 80
        desired_freq = 1/desired_period
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

    def __init__(self, **kw):
        kw['h5mode'] = 'r+'
        self.rotate = kw.pop('rotate', True)
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, rotate=True,
                             nr_samples=2,
                             sample_0=0,
                             sample_1=1,
                             channels=['I', 'Q'],
                             no_fits=False,
                             print_fit_results=False, **kw):

        self.add_analysis_datagroup_to_file()
        self.no_fits = no_fits
        self.get_naming_and_values()
        # plotting histograms of the raw shots on I and Q axis

        if len(channels) == 1:
            shots_I_data = self.get_values(key=channels[0])
            shots_I_data_0, shots_I_data_1 = a_tools.zigzag(shots_I_data,
                                                            sample_0, sample_1, nr_samples)
            shots_Q_data_0 = shots_I_data_0*0
            shots_Q_data_1 = shots_I_data_1*0

        else:
            # Try getting data by name first and by index otherwise
            try:
                shots_I_data = self.get_values(key=channels[0])
                shots_Q_data = self.get_values(key=channels[1])
            except:
                shots_I_data = self.measured_values[0]
                shots_Q_data = self.measured_values[1]

            shots_I_data_0, shots_I_data_1 = a_tools.zigzag(shots_I_data,
                                                            sample_0, sample_1, nr_samples)
            shots_Q_data_0, shots_Q_data_1 = a_tools.zigzag(shots_Q_data,
                                                            sample_0, sample_1, nr_samples)

        # cutting off half data points (odd number of data points)
        min_len = np.min([np.size(shots_I_data_0), np.size(shots_I_data_1),
                          np.size(shots_Q_data_0), np.size(shots_Q_data_1)])
        shots_I_data_0 = shots_I_data_0[0:min_len]
        shots_I_data_1 = shots_I_data_1[0:min_len]
        shots_Q_data_0 = shots_Q_data_0[0:min_len]
        shots_Q_data_1 = shots_Q_data_1[0:min_len]

        # rotating IQ-plane to transfer all information to the I-axis
        if self.rotate:
            theta, shots_I_data_1_rot, shots_I_data_0_rot = \
                self.optimize_IQ_angle(shots_I_data_1, shots_Q_data_1,
                                       shots_I_data_0, shots_Q_data_0, min_len,
                                       **kw)
            self.theta = theta
        else:
            self.theta = 0
            shots_I_data_1_rot = shots_I_data_1
            shots_I_data_0_rot = shots_I_data_0

            cmap = kw.pop('cmap', 'viridis')
            # plotting 2D histograms of mmts with pulse

            n_bins = 120  # the bins we want to have around our data
            I_min = min(min(shots_I_data_0), min(shots_I_data_1))
            I_max = max(max(shots_I_data_0), max(shots_I_data_1))
            Q_min = min(min(shots_Q_data_0), min(shots_Q_data_1))
            Q_max = max(max(shots_Q_data_0), max(shots_Q_data_1))
            edge = max(abs(I_min), abs(I_max), abs(Q_min), abs(Q_max))
            H0, xedges0, yedges0 = np.histogram2d(shots_I_data_0, shots_Q_data_0,
                                                  bins=n_bins,
                                                  range=[[I_min, I_max],
                                                         [Q_min, Q_max]],
                                                  normed=True)
            H1, xedges1, yedges1 = np.histogram2d(shots_I_data_1, shots_Q_data_1,
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
            im1 = axarray[0].imshow(np.transpose(H1), interpolation='nearest', origin='low',
                                    extent=[xedges1[0], xedges1[-1],
                                            yedges1[0], yedges1[-1]], cmap=cmap)
            axarray[0].set_xlabel('Int. I (V)')
            axarray[0].set_ylabel('Int. Q (V)')
            axarray[0].set_xlim(-edge, edge)
            axarray[0].set_ylim(-edge, edge)

            # plotting 2D histograms of mmts with no pulse
            axarray[1].set_title('2D histogram, no pi pulse')
            im0 = axarray[1].imshow(np.transpose(H0), interpolation='nearest', origin='low',
                                    extent=[xedges0[0], xedges0[-1], yedges0[0],
                                            yedges0[-1]], cmap=cmap)
            axarray[1].set_xlabel('Int. I (V)')
            axarray[1].set_ylabel('Int. Q (V)')
            axarray[1].set_xlim(-edge, edge)
            axarray[1].set_ylim(-edge, edge)

            self.save_fig(fig, figname='SSRO_Density_Plots', **kw)

            self.avg_0_I = np.mean(shots_I_data_0)
            self.avg_1_I = np.mean(shots_I_data_1)
            self.avg_0_Q = np.mean(shots_Q_data_0)
            self.avg_1_Q = np.mean(shots_Q_data_1)
        # making gaussfits of s-curves

        self.no_fits_analysis(shots_I_data_1_rot, shots_I_data_0_rot, min_len,
                              **kw)
        if self.no_fits is False:
            self.s_curve_fits(shots_I_data_1_rot, shots_I_data_0_rot, min_len,
                              **kw)
        self.finish(**kw)

    def optimize_IQ_angle(self, shots_I_1, shots_Q_1, shots_I_0,
                          shots_Q_0, min_len, plot_2D_histograms=True,
                          **kw):
        cmap = kw.pop('cmap', 'viridis')
        # plotting 2D histograms of mmts with pulse

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

        if plot_2D_histograms:
            fig, axarray = plt.subplots(nrows=1, ncols=2)
            axarray[0].tick_params(axis='both', which='major',
                                   labelsize=5, direction='out')
            axarray[1].tick_params(axis='both', which='major',
                                   labelsize=5, direction='out')

            plt.subplots_adjust(hspace=20)

            axarray[0].set_title('2D histogram, pi pulse')
            im1 = axarray[0].imshow(np.transpose(H1), interpolation='nearest', origin='low',
                                    extent=[xedges1[0], xedges1[-1],
                                            yedges1[0], yedges1[-1]], cmap=cmap)
            axarray[0].set_xlabel('Int. I (V)')
            axarray[0].set_ylabel('Int. Q (V)')
            axarray[0].set_xlim(-edge, edge)
            axarray[0].set_ylim(-edge, edge)

            # plotting 2D histograms of mmts with no pulse
            axarray[1].set_title('2D histogram, no pi pulse')
            im0 = axarray[1].imshow(np.transpose(H0), interpolation='nearest', origin='low',
                                    extent=[xedges0[0], xedges0[-1], yedges0[0],
                                            yedges0[-1]], cmap=cmap)
            axarray[1].set_xlabel('Int. I (V)')
            axarray[1].set_ylabel('Int. Q (V)')
            axarray[1].set_xlim(-edge, edge)
            axarray[1].set_ylim(-edge, edge)

            self.save_fig(fig, figname='SSRO_Density_Plots', **kw)

        # this part performs 2D gaussian fits and calculates coordinates of the
        # maxima
        def gaussian(height, center_x, center_y, width_x, width_y):
            width_x = float(width_x)
            width_y = float(width_y)
            return lambda x, y: height*np.exp(-(((center_x-x)/width_x)**2+(
                                              (center_y-y)/width_y)**2)/2)

        def fitgaussian(data):
            params = moments(data)
            errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(
                                               data.shape))-data)
            p, success = optimize.leastsq(errorfunction, params)
            return p

        def moments(data):
            total = data.sum()
            X, Y = np.indices(data.shape)
            x = (X*data).sum()/total
            y = (Y*data).sum()/total
            col = data[:, int(y)]
            eps = 1e-8  # To prevent division by zero
            width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/(
                              col.sum()+eps))
            row = data[int(x), :]
            width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/(
                              row.sum()+eps))
            height = data.max()
            return height, x, y, width_x, width_y

        data0 = H0
        params0 = fitgaussian(data0)
        fit0 = gaussian(*params0)
        data1 = H1
        params1 = fitgaussian(data1)
        fit1 = gaussian(*params1)
        # interpolating to find the gauss top x and y coordinates
        x_lin = np.linspace(0, n_bins, n_bins+1)
        y_lin = np.linspace(0, n_bins, n_bins+1)
        f_x_1 = interp1d(x_lin, xedges1)
        x_1_max = f_x_1(params1[1])
        f_y_1 = interp1d(y_lin, yedges1)
        y_1_max = f_y_1(params1[2])

        f_x_0 = interp1d(x_lin, xedges0)
        x_0_max = f_x_0(params0[1])
        f_y_0 = interp1d(y_lin, yedges0)
        y_0_max = f_y_0(params0[2])

        # following part will calculate the angle to rotate the IQ plane
        # All information is to be rotated to the I channel
        y_diff = y_1_max-y_0_max
        x_diff = x_1_max-x_0_max
        theta = -np.arctan2(y_diff, x_diff)

        shots_I_1_rot = np.cos(theta)*shots_I_1 - np.sin(theta)*shots_Q_1
        shots_Q_1_rot = np.sin(theta)*shots_I_1 + np.cos(theta)*shots_Q_1

        shots_I_0_rot = np.cos(theta)*shots_I_0 - np.sin(theta)*shots_Q_0
        shots_Q_0_rot = np.sin(theta)*shots_I_0 + np.cos(theta)*shots_Q_0

        # # plotting the histograms before rotation
        # fig, axes = plt.subplots()
        # axes.hist(shots_Q_1, bins=40, label='1 Q',
        #           histtype='step', normed=1, color='r')
        # axes.hist(shots_Q_0, bins=40, label='0 Q',
        #           histtype='step', normed=1, color='b')
        # axes.hist(shots_I_1, bins=40, label='1 I',
        #           histtype='step', normed=1, color='m')
        # axes.hist(shots_I_0, bins=40, label='0 I',
        #           histtype='step', normed=1, color='c')

        # axes.set_title('Histograms of shots on IQ plane as measured, %s shots'%min_len)
        # plt.xlabel('DAQ voltage integrated (a.u.)', fontsize=14)
        # plt.ylabel('Fraction', fontsize=14)

        # #plt.hist(SS_Q_data, bins=40,label='0 Q')
        # plt.legend(loc='best')
        # self.save_fig(fig, figname='raw-histograms', **kw)
        # plt.show()
        # #plotting the histograms after rotation
        # fig, axes = plt.subplots()

        # axes.hist(shots_I_1_rot, bins=40, label='|1>',
        #           histtype='step', normed=1, color='r')
        # axes.hist(shots_I_0_rot, bins=40, label='|0>',
        #           histtype='step', normed=1, color='b')

        # axes.set_title('Histograms of shots on rotaded IQ plane, %s shots' %
        #                min_len)
        # plt.xlabel('DAQ voltage integrated (a.u.)', fontsize=14)
        # plt.ylabel('Fraction', fontsize=14)

        # plt.legend()
        # self.save_fig(fig, figname='rotated-histograms', **kw)
        # plt.show()
        return(theta, shots_I_1_rot, shots_I_0_rot)

    def no_fits_analysis(self, shots_I_1_rot, shots_I_0_rot, min_len,
                         **kw):
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
        self.cumsum_1 = cumsum_1/cumsum_1[-1]  # renormalizing

        hist_0, bins = np.histogram(shots_I_0_rot, bins=1000,
                                    range=(min_voltage, max_voltage),
                                    density=1)
        cumsum_0 = np.cumsum(hist_0)
        self.cumsum_0 = cumsum_0/cumsum_0[-1]  # renormalizing

        cumsum_diff = (abs(self.cumsum_1-self.cumsum_0))
        cumsum_diff_list = cumsum_diff.tolist()
        self.index_V_th_a = int(cumsum_diff_list.index(np.max(
            cumsum_diff_list)))
        V_th_a = bins[self.index_V_th_a]+(bins[1]-bins[0])/2
        # adding half a bin size
        F_a = 1-(1-cumsum_diff_list[self.index_V_th_a])/2

        fig, ax = plt.subplots()
        ax.plot(bins[0:-1], self.cumsum_1, label='cumsum_1', color='red')
        ax.plot(bins[0:-1], self.cumsum_0, label='cumsum_0', color='blue')
        ax.axvline(V_th_a, ls='--', label="V_th_a = %.3f" % V_th_a,
                   linewidth=2, color='grey')
        ax.text(.7, .6, '$Fa$ = %.4f' % F_a, transform=ax.transAxes,
                fontsize='large')
        ax.set_title('raw cumulative histograms')
        plt.xlabel('DAQ voltage integrated (AU)', fontsize=14)
        plt.ylabel('Fraction', fontsize=14)

        #plt.hist(SS_Q_data, bins=40,label = '0 Q')
        plt.legend(loc=2)
        self.save_fig(fig, figname='raw-cumulative-histograms', **kw)
        plt.show()

        # saving the results
        if 'SSRO_Fidelity' not in self.analysis_group:
            fid_grp = self.analysis_group.create_group('SSRO_Fidelity')
        else:
            fid_grp = self.analysis_group['SSRO_Fidelity']
        fid_grp.attrs.create(name='V_th_a', data=V_th_a)
        fid_grp.attrs.create(name='F_a', data=F_a)

        self.F_a = F_a
        self.V_th_a = V_th_a

    def s_curve_fits(self, shots_I_1_rot, shots_I_0_rot, min_len,
                     **kw):
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
                    out = 2-r
            return out

        def NormCdf(x, mu, sigma):
            t = x-mu
            y = 0.5*erfcc(-t/(sigma*np.sqrt(2.0)))
            for k in range(np.size(x)):
                if y[k] > 1.0:
                    y[k] = 1.0
            return y

        NormCdfModel = lmfit.Model(NormCdf)

        def NormCdf2(x, mu0, mu1, sigma0, sigma1, frac1):
            t0 = x-mu0
            t1 = x-mu1
            frac0 = 1-frac1
            y = frac1*0.5*erfcc(-t1/(sigma1*np.sqrt(2.0))) + \
                frac0*0.5*erfcc(-t0/(sigma0*np.sqrt(2.0)))
            for k in range(np.size(x)):
                if y[k] > 1.0:
                    y[k] = 1.0
            return y

        NormCdf2Model = lmfit.Model(NormCdf2)
        NormCdfModel.set_param_hint('mu', value=(np.average(shots_I_0_rot)
                                                 + np.average(shots_I_0_rot))/2)
        NormCdfModel.set_param_hint('sigma', value=(np.std(shots_I_0_rot)
                                                    + np.std(shots_I_0_rot))/2, min=0)

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
        NormCdf2Model.set_param_hint('frac1', value=0.9, min=0, max=1)

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
        NormCdf2Model.set_param_hint(
            'frac1', value=0.025, min=0, max=1, vary=True)

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
        print('frac1 in 0: {:.4f}'.format(frac1_0))

        def NormCdf(x, mu, sigma):
            t = x-mu
            y = 0.5*erfcc(-t/(sigma*np.sqrt(2.0)))
            return y

        def NormCdfdiff(x, mu0=mu0, mu1=mu1, sigma0=sigma0, sigma1=sigma1):
            y = -abs(NormCdf(x, mu0, sigma0)-NormCdf(x, mu1, sigma1))
            return y

        V_opt_single = optimize.brent(NormCdfdiff)
        F_single = -NormCdfdiff(x=V_opt_single)
        # print 'V_opt_single', V_opt_single
        # print 'F_single', F_single

        # redefining the function with different variables to avoid problems
        # with arguments in brent optimization
        def NormCdfdiff(x, mu0=mu0_0, mu1=mu1_1, sigma0=sigma0_0, sigma1=sigma1_1):
            y0 = -abs(NormCdf(x, mu0, sigma0)-NormCdf(x, mu1, sigma1))
            return y0

        self.V_th_d = optimize.brent(NormCdfdiff)
        F_d = 1-(1+NormCdfdiff(x=self.V_th_d))/2
        # print 'F_corrected',F_corrected

        def NormCdfdiffDouble(x, mu0_0=mu0_0,
                              sigma0_0=sigma0_0, sigma1_0=sigma1_0,
                              frac1_0=frac1_0, mu1_1=mu1_1,
                              sigma0_1=sigma0_1, sigma1_1=sigma1_1,
                              frac1_1=frac1_1):
            distr0 = (1-frac1_0)*NormCdf(x, mu0_0, sigma0_0) + \
                (frac1_0)*NormCdf(x, mu1_1, sigma1_1)

            distr1 = (1-frac1_1)*NormCdf(x, mu0_0, sigma0_0) + \
                (frac1_1)*NormCdf(x, mu1_1, sigma1_1)
            y = - abs(distr1-distr0)
            return y

        # print "refresh"
        # self.V_th_d = optimize.brent(NormCdfdiffDouble)
        # F_d = -NormCdfdiffDouble(x=self.V_th_d)

        # calculating the signal-to-noise ratio
        signal = abs(mu0_0-mu1_1)
        noise = (sigma0_0 + sigma1_1)/2
        SNR = signal/noise

        # plotting s-curves
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(
            'S-curves (not binned) and fits, determining fidelity and threshold optimum, %s shots' % min_len)
        ax.set_xlabel('DAQ voltage integrated (V)')  # , fontsize=14)
        ax.set_ylabel('Fraction of counts')  # , fontsize=14)
        ax.set_ylim((-.01, 1.01))
        ax.plot(S_sorted_I_0, p_norm_I_0, label='0 I', linewidth=2,
                color='blue')
        ax.plot(S_sorted_I_1, p_norm_I_1, label='1 I', linewidth=2,
                color='red')

        # ax.plot(S_sorted_I_0, fit_res_0.best_fit,
        #         label='0 I single gaussian fit', ls='--', linewidth=3,
        #         color='lightblue')
        # ax.plot(S_sorted_I_1, fit_res_1.best_fit, label='1 I',
        #         linewidth=2, color='red')

        ax.plot(S_sorted_I_0, fit_res_double_0.best_fit,
                label='0 I double gaussfit', ls='--', linewidth=3,
                color='lightblue')
        ax.plot(S_sorted_I_1, fit_res_double_1.best_fit,
                label='1 I double gaussfit', ls='--', linewidth=3,
                color='darkred')
        labelstring = 'V_th_a= %.3f V' % (self.V_th_a)
        labelstring_corrected = 'V_th_d= %.3f V' % (self.V_th_d)

        ax.axvline(self.V_th_a, ls='--', label=labelstring,
                   linewidth=2, color='grey')
        ax.axvline(self.V_th_d, ls='--', label=labelstring_corrected,
                   linewidth=2, color='black')

        leg = ax.legend(loc='best')
        leg.get_frame().set_alpha(0.5)
        self.save_fig(fig, figname='S-curves', **kw)
        plt.show()

        # plotting the histograms
        fig, axes = plt.subplots(figsize=(8, 4))
        n1, bins1, patches = pylab.hist(shots_I_1_rot, bins=int(min_len/50),
                                        label='1 I', histtype='step',
                                        color='red', normed=True)
        n0, bins0, patches = pylab.hist(shots_I_0_rot, bins=int(min_len/50),
                                        label='0 I', histtype='step',
                                        color='blue', normed=True)
        pylab.clf()
        # n0, bins0 = np.histogram(shots_I_0_rot, bins=int(min_len/50), normed=1)
        # n1, bins1 = np.histogram(shots_I_1_rot, bins=int(min_len/50), normed=1)

        pylab.plot(bins1[:-1]+0.5*(bins1[1]-bins1[0]), n1, 'ro')
        pylab.plot(bins0[:-1]+0.5*(bins0[1]-bins0[0]), n0, 'bo')

        # n, bins1, patches = np.hist(shots_I_1_rot, bins=int(min_len/50),
        #                               label = '1 I',histtype='step',
        #                               color='red',normed=1)
        # n, bins0, patches = pylab.hist(shots_I_0_rot, bins=int(min_len/50),
        #                               label = '0 I',histtype='step',
        #                               color='blue',normed=1)

        # add lines showing the fitted distribution
        # building up the histogram fits for off measurements
        y0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0) + \
            frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
        y1_0 = frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
        y0_0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0)

        # building up the histogram fits for on measurements
        y1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1) + \
            frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
        y1_1 = frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
        y0_1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1)

        pylab.semilogy(bins0, y0, 'b', linewidth=1.5)
        pylab.semilogy(bins0, y1_0, 'b--', linewidth=3.5)
        pylab.semilogy(bins0, y0_0, 'b--', linewidth=3.5)

        pylab.semilogy(bins1, y1, 'r', linewidth=1.5)
        pylab.semilogy(bins1, y0_1, 'r--', linewidth=3.5)
        pylab.semilogy(bins1, y1_1, 'r--', linewidth=3.5)
        #(pylab.gca()).set_ylim(1e-6,1e-3)
        pdf_max = (max(max(y0), max(y1)))
        (pylab.gca()).set_ylim(pdf_max/1000, 2*pdf_max)

        plt.title('Histograms of {} shots, {}'.format(
            min_len, self.timestamp_string))
        plt.xlabel('DAQ voltage integrated (V)')  # , fontsize=14)
        plt.ylabel('Fraction of counts')  # , fontsize=14)

        plt.axvline(self.V_th_a, ls='--',
                    linewidth=2, color='grey', label='SNR={0:.2f}\n $F_a$={1:.4f}\n $F_d$={2:.4f}\n $p_e$={3:.4f}'.format(SNR, self.F_a, F_d, frac1_0))
        plt.axvline(self.V_th_d, ls='--',
                    linewidth=2, color='black')
        plt.legend()
        leg2 = ax.legend(loc='best')
        leg2.get_frame().set_alpha(0.5)
        #plt.hist(SS_Q_data, bins=40,label = '0 Q')
        self.save_fig(fig, figname='Histograms', **kw)
        plt.show()

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

        self.sigma0_0 = sigma0_0
        self.sigma1_1 = sigma1_1
        self.mu0_0 = mu0_0
        self.mu1_1 = mu1_1
        self.frac1_0 = frac1_0
        self.frac1_1 = frac1_1
        self.F_d = F_d
        self.SNR = SNR


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
                             current_threshold=None, theta_in=0, **kw):
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        I_shots = self.measured_values[0]
        Q_shots = self.measured_values[1]

        if theta_in != 0:
            shots = I_shots+1j*Q_shots
            rot_shots = dm_tools.rotate_complex(
                shots, angle=theta_in, deg=True)
            I_shots = rot_shots.real
            Q_shots = rot_shots.imag

        # Reshaping the data
        n_bins = 120  # the bins we want to have around our data
        # min min and max max constructions exist so that it also works
        # if one dimension only conatins zeros
        H, xedges, yedges = np.histogram2d(I_shots, Q_shots,
                                           bins=n_bins,
                                           range=[[min(min(I_shots), -1),
                                                   max(max(I_shots), 1)],
                                                  [min(min(Q_shots), -1),
                                                   max(max(Q_shots), 1)]],
                                           normed=True)
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
                ax.ticklabel_format(style='sci', fontsize=4,
                                    scilimits=(0, 0))
                ax.set_xlabel('I')  # TODO: add units
                edge = max(max(abs(xedges)), max(abs(yedges)))
                ax.set_xlim(-edge, edge)
                ax.set_ylim(-edge, edge)
                # ax.set_axis_bgcolor(plt.cm.viridis(0))
            axs[0].set_ylabel('Q')
            #axs[0].ticklabel_format(style = 'sci',  fontsize=4)

            self.save_fig(
                fig, figname='2D-Histograms_{}'.format(theta_in), **kw)

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
        self.relative_separation = abs(diff_vec)/self.mean_sigma
        # relative separation of the gaussians when projected on the I-axis
        self.relative_separation_I = diff_vec.real/self.mean_sigma

        #######################################################
        # Calculating discrimanation fidelities based on erfc #
        #######################################################
        # CDF of gaussian is P(X<=x) = .5 erfc((mu-x)/(sqrt(2)sig))

        # Along the optimal direction
        CDF_a = .5 * math.erfc((abs(diff_vec/2)) /
                               (np.sqrt(2)*sig_a))
        CDF_b = .5 * math.erfc((-abs(diff_vec/2)) /
                               (np.sqrt(2)*sig_b))
        self.F_discr = 1-(1-abs(CDF_a - CDF_b))/2

        # Projected on the I-axis
        CDF_a = .5 * math.erfc((self.mu_a.real - self.opt_I_threshold) /
                               (np.sqrt(2)*sig_a))
        CDF_b = .5 * math.erfc((self.mu_b.real - self.opt_I_threshold) /
                               (np.sqrt(2)*sig_b))

        self.F_discr_I = abs(CDF_a - CDF_b)
        # Current threshold projected on the I-axis
        if current_threshold is not None:
            CDF_a = .5 * math.erfc((self.mu_a.real - current_threshold) /
                                   (np.sqrt(2)*sig_a))
            CDF_b = .5 * math.erfc((self.mu_b.real - current_threshold) /
                                   (np.sqrt(2)*sig_b))
            self.F_discr_curr_t = 1-(1-abs(CDF_a - CDF_b))/2

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
        #axes.hist(shots_Q_data, bins=40, label = '0 Q',histtype='step',normed=1)
        axes.axvline(x=threshold, ls='--', label='threshold')

        axes.set_title(
            'Histogram of I-shots for touch and go measurement and threshold')
        plt.xlabel('DAQ voltage integrated (AU)', fontsize=14)
        plt.ylabel('Fraction', fontsize=14)

        #plt.hist(SS_Q_data, bins=40,label = '0 Q')
        plt.legend()
        self.save_fig(fig, figname='raw-histograms', **kw)
        plt.show()

        self.finish(**kw)


class SSRO_single_quadrature_discriminiation_analysis(MeasurementAnalysis):

    '''
    Analysis that fits two gaussians to a histogram of a dataset.
    Uses this to extract F_discr and the optimal threshold
    '''

    def __init__(self, quadrature='I', **kw):
        # Note: quadrature is a bit of misnomer here
        # it represents the channel/weight of the data we want to bin
        kw['h5mode'] = 'r+'
        self.quadrature = quadrature
        super().__init__(**kw)

    def run_default_analysis(self, close_file=True, **kw):
        self.get_naming_and_values()
        hist, bins, centers = self.histogram_shots(self.shots)
        self.fit_data(hist, centers)
        self.make_figures(hist=hist, centers=centers, **kw)

        self.F_discr, self.opt_threshold = self.calculate_discrimination_fidelity(
            fit_res=self.fit_res)
        if close_file:
            self.data_file.close()
        return

    def get_naming_and_values(self):
        super().get_naming_and_values()
        if type(self.quadrature) is str:
            self.shots = self.get_values(self.quadrature)
            # Potentially bug sensitive!!
            self.units = self.value_units[0]
        elif type(self.quadrature) is int:
            self.shots = self.measured_values[self.quadrature]
            self.units = self.value_units[self.quadrature]

    def histogram_shots(self, shots):
        hist, bins = np.histogram(shots, bins=90, normed=True)
        # 0.7 bin widht is a sensible default for plotting
        centers = (bins[:-1] + bins[1:]) / 2
        return hist, bins, centers

    def fit_data(self, hist, centers):
        self.add_analysis_datagroup_to_file()
        self.model = fit_mods.DoubleGaussModel
        params = self.model.guess(self.model, hist, centers)
        self.fit_res = self.model.fit(data=hist, x=centers, params=params)
        self.save_fitted_parameters(
            fit_res=self.fit_res, var_name='{}shots'.format(self.quadrature))
        return self.fit_res

    def make_figures(self, hist, centers, show_guess=False, **kw):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        width = .7 * (centers[1]-centers[0])
        plot_title = kw.pop('plot_title', textwrap.fill(
                            self.timestamp_string + '_' +
                            self.measurementstring, 40))

        x_fine = np.linspace(min(centers),
                             max(centers), 1000)
        # Plotting the data
        self.ax.bar(centers, hist, align='center', width=width, label='data')
        self.ax.plot(x_fine, self.fit_res.eval(x=x_fine), label='fit', c='r')
        if show_guess:
            self.ax.plot(x_fine, self.fit_res.eval(
                x=x_fine, **self.fit_res.init_values), label='guess', c='g')
            self.ax.legend(loc='best')

        # Prettifying the plot
        self.ax.ticklabel_format(useOffset=False)
        self.ax.set_title(plot_title)
        self.ax.set_xlabel('{} ({})'.format(self.quadrature, self.units))
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

        x_fine = np.linspace(min(mu_a-4*s_a, mu_b-4*s_b),
                             max(mu_b+4*s_a, mu_b+4*s_b), 1000)
        CDF_a = np.zeros(len(x_fine))
        CDF_b = np.zeros(len(x_fine))
        for i, x in enumerate(x_fine):
            CDF_a[i] = .5 * erfc((mu_a-x)/(np.sqrt(2)*s_a))
            CDF_b[i] = .5 * erfc((mu_b-x)/(np.sqrt(2)*s_b))
        F_discr = np.max(abs(CDF_a-CDF_b))
        opt_threshold = x_fine[np.argmax(abs(CDF_a-CDF_b))]
        return F_discr, opt_threshold


class T1_Analysis(TD_Analysis):

    def __init__(self, label='T1', make_fig=True, **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super().__init__(**kw)

    def fit_T1(self, t_arr, data):
        # Guess for params
        fit_mods.ExpDecayModel.set_param_hint('amplitude', value=1,
                                              min=0, max=2)
        fit_mods.ExpDecayModel.set_param_hint(
            'tau',
            value=self.sweep_points[1]*50,  # use index 1
            min=self.sweep_points[1],
            max=self.sweep_points[-1]*1000)
        fit_mods.ExpDecayModel.set_param_hint('offset', value=0, vary=False)
        fit_mods.ExpDecayModel.set_param_hint('n', value=1, vary=False)
        self.params = fit_mods.ExpDecayModel.make_params()

        fit_res = fit_mods.ExpDecayModel.fit(
            data=data,
            t=t_arr,
            params=self.params)
        return fit_res

    def run_default_analysis(self, print_fit_results=False,
                             make_fig=True, **kw):
        show_guess = kw.pop('show_guess', False)
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        fig, figarray, ax, axarray = self.setup_figures_and_axes()
        self.normalized_values = []

        if make_fig:
            for i, name in enumerate(self.value_names):
                if len(self.value_names) < 4:
                    ax2 = axarray[i]
                else:
                    ax2 = axarray[i/2, i % 2]

                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=figarray, ax=ax2,
                                                xlabel=self.xlabel,
                                                ylabel=self.ylabels[i],
                                                save=False)

        if 'I_cal' in self.value_names[i]:  # Fit the data
            norm = self.normalize_data_to_calibration_points(
                self.measured_values[i], self.NoCalPoints)
            self.normalized_values = norm[0]
            self.normalized_data_points = norm[1]
            self.normalized_cal_vals = norm[2]

        else:
            norm = self.normalize_data_to_calibration_points(
                self.measured_values[0], self.NoCalPoints)
            self.normalized_values = norm[0]
            self.normalized_data_points = norm[1]
            self.normalized_cal_vals = norm[2]

        fit_res = self.fit_T1(t_arr=self.sweep_points[:-self.NoCalPoints],
                              data=self.normalized_data_points)

        self.fit_res = fit_res
        best_vals = fit_res.best_values
        self.save_fitted_parameters(fit_res=fit_res, var_name='F|1>')

        self.T1 = best_vals['tau']
        self.T1_stderr = fit_res.params['tau'].stderr

        if print_fit_results:
            print(fit_res.fit_report())
        if make_fig:
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.normalized_values,
                                            fig=fig, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=r'$F$ $|1 \rangle$',
                                            **kw)
            if show_guess:
                ax.plot(self.sweep_points[:-self.NoCalPoints],
                        fit_res.init_fit, 'k--')

            best_vals = fit_res.best_values
            t = np.linspace(self.sweep_points[0],
                            self.sweep_points[-self.NoCalPoints], 1000)

            y = fit_mods.ExpDecayFunc(
                t, tau=best_vals['tau'],
                n=best_vals['n'],
                amplitude=best_vals['amplitude'],
                offset=best_vals['offset'])

            ax.plot(t, y, 'r-')
            textstr = '$T_1$ = %.3g $\pm$ (%.5g) s ' % (
                fit_res.params['tau'].value, fit_res.params['tau'].stderr)

            ax.text(0.4, 0.95, textstr, transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=self.box_props)
            self.save_fig(fig, figname=self.measurementstring+'_Fit', **kw)
            # self.save_fig(fig, figname=self.measurementstring+'_' +
            #               self.value_names[i], **kw)
            self.save_fig(self.figarray, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return fit_res

    def get_measured_T1(self):
        fitted_pars = self.data_file['Analysis']['Fitted Params F|1>']
        T1 = fitted_pars['tau'].attrs['value']
        T1_stderr = fitted_pars['tau'].attrs['stderr']

        return T1, T1_stderr


class Ramsey_Analysis(TD_Analysis):

    def __init__(self, label='Ramsey', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super(self.__class__, self).__init__(**kw)

    def fit_Ramsey(self, print_fit_results=False):
        damped_osc_mod = fit_mods.ExpDampOscModel
        average = np.mean(self.normalized_data_points)

        ft_of_data = np.fft.fft(self.normalized_data_points)
        index_of_fourier_maximum = np.argmax(np.abs(
            ft_of_data[1:len(ft_of_data)//2]))+1
        max_ramsey_delay = self.sweep_points[-self.NoCalPoints] - \
            self.sweep_points[0]

        fft_axis_scaling = 1/(max_ramsey_delay)
        freq_est = fft_axis_scaling*index_of_fourier_maximum
        est_number_of_periods = index_of_fourier_maximum

        if (average > 0.7 or
                (est_number_of_periods < 2) or
                est_number_of_periods > len(ft_of_data)/2.):
            print('the trace is to short to find multiple periods')

            if print_fit_results:
                print('Setting frequency to 0 and ' +
                      'fitting with decaying exponential.')
            damped_osc_mod.set_param_hint('frequency',
                                          value=freq_est,
                                          vary=False)
            damped_osc_mod.set_param_hint('phase',
                                          value=0, vary=False)
        else:
            damped_osc_mod.set_param_hint('frequency',
                                          value=freq_est,
                                          vary=True,
                                          min=(1/(100 *
                                                  self.sweep_points[-1])),
                                          max=(20/self.sweep_points[-1]))

        amplitude_guess = 1
        damped_osc_mod.set_param_hint('amplitude',
                                      value=amplitude_guess,
                                      min=0.4, max=2.0)

        if (np.average(self.normalized_data_points[:4]) >
                np.average(self.normalized_data_points[4:8])):
            phase_estimate = 0
        else:
            phase_estimate = np.pi

        damped_osc_mod.set_param_hint('phase',
                                      value=phase_estimate, vary=True)

        damped_osc_mod.set_param_hint('tau',
                                      value=self.sweep_points[1]*10,
                                      min=self.sweep_points[1],
                                      max=self.sweep_points[1]*1000)

        damped_osc_mod.set_param_hint('exponential_offset',
                                      value=0.5,
                                      min=0.4, max=1.1)
        damped_osc_mod.set_param_hint('oscillation_offset',
                                      value=0, vary=False)

        damped_osc_mod.set_param_hint('n',
                                      value=1,
                                      vary=False)
        self.params = damped_osc_mod.make_params()
        fit_res = damped_osc_mod.fit(data=self.normalized_data_points,
                                     t=self.sweep_points[:-self.NoCalPoints],
                                     params=self.params)
        if fit_res.chisqr > .35:
            logging.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2*np.pi, 8):
                damped_osc_mod.set_param_hint('phase',
                                              value=phase_estimate)
                self.params = damped_osc_mod.make_params()
                fit_res_lst += [damped_osc_mod.fit(
                                data=self.normalized_data_points,
                                t=self.sweep_points[:-self.NoCalPoints],
                                params=self.params)]

            chisqr_lst = [fit_res.chisqr for fit_res in fit_res_lst]
            fit_res = fit_res_lst[np.argmin(chisqr_lst)]
        self.fit_results.append(fit_res)
        if print_fit_results:
            print(fit_res.fit_report())
        return fit_res

    def plot_results(self, fig, ax, fit_res, ylabel, show_guess=False):
        textstr = ('  $f$  \t= %.3g $ \t \pm$ (%.3g) Hz'
                   % (fit_res.params['frequency'].value,
                      fit_res.params['frequency'].stderr) +
                   '\n$T_2^\star$ = %.3g $\t \pm$ (%.3g) s '
                   % (fit_res.params['tau'].value,
                      fit_res.params['tau'].stderr))
        ax.text(0.4, 0.95, textstr,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=self.box_props)

        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.normalized_values,
                                        fig=fig, ax=ax,
                                        xlabel=self.xlabel,
                                        ylabel=ylabel,
                                        save=False)
        if show_guess:
            ax.plot(self.sweep_points[:-self.NoCalPoints],
                    self.fit_res.init_fit, 'k--')
        x = np.linspace(self.sweep_points[0],
                        self.sweep_points[-self.NoCalPoints],
                        len(self.sweep_points)*100)
        best_vals = self.fit_res.best_values
        y = fit_mods.ExpDampOscFunc(
            x, tau=best_vals['tau'],
            n=best_vals['n'],
            frequency=best_vals['frequency'],
            phase=best_vals['phase'],
            amplitude=best_vals['amplitude'],
            oscillation_offset=best_vals['oscillation_offset'],
            exponential_offset=best_vals['exponential_offset'])
        ax.plot(x, y, 'r-')

    def run_default_analysis(self, print_fit_results=False, **kw):

        close_file = kw.pop('close_file', True)
        show_guess = kw.pop('show_guess', False)
        show = kw.pop('show', False)
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()
        fig1, fig2, ax, axarray = self.setup_figures_and_axes()

        norm = self.normalize_data_to_calibration_points(
            self.measured_values[0], self.NoCalPoints)
        self.normalized_values = norm[0]
        self.normalized_data_points = norm[1]
        self.normalized_cal_vals = norm[2]
        self.fit_res = self.fit_Ramsey(print_fit_results)
        self.save_fitted_parameters(self.fit_res, var_name=self.value_names[0])
        self.plot_results(fig1, ax, self.fit_res, show_guess=show_guess,
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

        stepsize = self.sweep_points[1] - self.sweep_points[0]
        self.total_detuning = self.fit_res.params['frequency'].value
        self.detuning_stderr = self.fit_res.params['frequency'].stderr
        self.T2_star = self.fit_res.params['tau'].value
        self.T2_star_stderr = self.fit_res.params['tau'].stderr

        self.artificial_detuning = 4./(60*stepsize)
        self.detuning = self.total_detuning - self.artificial_detuning

        if show:
            plt.show()
        self.save_fig(fig1, figname=self.measurementstring+'_Ramsey_fit', **kw)
        self.save_fig(fig2, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return self.fit_res

    def get_measured_freq(self):
        fitted_pars = self.data_file['Analysis']['Fitted Params I_cal']
        freq = fitted_pars['frequency'].attrs['value']
        freq_stderr = fitted_pars['frequency'].attrs['stderr']

        return freq, freq_stderr

    def get_measured_T2_star(self):
        '''
        Returns measured T2 star from the fit to the Ical data.
         return T2, T2_stderr
        '''
        fitted_pars = self.data_file['Analysis']['Fitted Params I_cal']
        T2 = fitted_pars['tau'].attrs['value']
        T2_stderr = fitted_pars['tau'].attrs['stderr']

        return T2, T2_stderr


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
            1.j*self.measured_values[1][0::2]
        YpX90_data = self.measured_values[0][1::2] + \
            1.j*self.measured_values[1][1::2]

        self.XpY90 = np.mean(XpY90_data)
        self.YpX90 = np.mean(YpX90_data)
        self.detuning = np.abs(self.XpY90 - self.YpX90)

        for i, name in enumerate(self.value_names):
            ax = axarray[i/2, i % 2]
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

        self.time = np.linspace(0, samples/sampling_rate, samples)
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
            dem_cos = np.cos(2*np.pi*self.IF*self.time)
            dem_sin = np.sin(2*np.pi*self.IF*self.time)

            self.demod_transient_I = dem_cos*transients_0[:, 0] + \
                dem_sin * transients_1[:, 0]
            self.demod_transient_Q = -dem_sin*transients_0[:, 0] + \
                dem_cos * transients_1[:, 0]

            fig2, axs2 = plt.subplots(1, 1, figsize=figsize, sharex=True)
            axs2.plot(self.time, self.demod_transient_I, marker='.',
                      label='I demodulated')
            axs2.plot(self.time, self.demod_transient_Q, marker='.',
                      label='Q demodulated')
            axs2.legend()
            self.save_fig(fig2, figname=self.measurementstring+'demod', **kw)
            axs2.set_xlabel('time (ns)')
            axs2.set_ylabel('dac voltage (V)')

            self.power = self.demod_transient_I**2 + self.demod_transient_Q**2
            fig3, ax3 = plt.subplots(1, 1, figsize=figsize, sharex=True)
            ax3.plot(self.time, self.power, marker='.')
            ax3.set_ylabel('Power (a.u.)')
            self.save_fig(fig3, figname=self.measurementstring+'Power', **kw)
            ax3.set_xlabel('time (ns)')

        self.save_fig(fig, figname=self.measurementstring, **kw)
        if close_file:
            self.data_file.close()
        return


class DriveDetuning_Analysis(TD_Analysis):

    def __init__(self, label='DriveDetuning', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False, **kw):

        def sine_fit_data():
            self.fit_type = 'sine'
            damped_osc_mod = fit_mods.CosModel

            # Estimate frequency using Fourier transform
            ft_of_data = np.fft.fft(data)
            freq_est = np.argmax(np.abs(ft_of_data[1:len(ft_of_data)/2]))+1
            print('using chagned')
            slope = stats.linregress(list(range(4)), data[:4])[0]
            if slope > 0:
                amp_sign = 1.
            else:
                amp_sign = -1.
            amp_guess = amp_sign * abs((cal_data_mean[1] - cal_data_mean[0])/2)

            damped_osc_mod.set_param_hint('amplitude', value=amp_guess,
                                          min=-1.2*amp_guess,
                                          max=1.2*amp_guess)
            damped_osc_mod.set_param_hint('frequency', value=freq_est /
                                          sweep_points[-1])
            damped_osc_mod.set_param_hint('phase', value=-np.pi/2, vary=False)
            damped_osc_mod.set_param_hint(
                'offset', value=np.mean(cal_data_mean))
            damped_osc_mod.set_param_hint('tau', value=400)
            self.params = damped_osc_mod.make_params()
            fit_results = damped_osc_mod.fit(data=data, t=sweep_points,
                                             params=self.params)
            return fit_results

        def quadratic_fit_data():
            M = np.array(
                [sweep_points**2, sweep_points, [1]*len(sweep_points)])
            Minv = np.linalg.pinv(M)
            [a, b, c] = np.dot(data, Minv)
            fit_data = (a*sweep_points**2 + b*sweep_points + c)
            return fit_data, (a, b, c)

        close_file = kw.pop('close_file', True)
        figsize = kw.pop('figsize', (11, 5))
        self.add_analysis_datagroup_to_file()
        self.get_naming_and_values()

        if len(self.sweep_points) == 60:
            self.NoCalPoints = 10
        else:
            self.NoCalPoints = 4

        self.normalize_data_to_calibration_points(
            self.measured_values[0], self.NoCalPoints)
        self.add_dataset_to_analysisgroup('Corrected data',
                                          self.corr_data)
        self.analysis_group.attrs.create('corrected data based on',
                                         'calibration points')

        data = self.corr_data[:-self.NoCalPoints]
        cal_data = np.split(self.corr_data[-self.NoCalPoints:], 2)
        cal_data_mean = np.mean(cal_data, axis=1)
        cal_peak_to_peak = abs(cal_data_mean[1] - cal_data_mean[0])

        sweep_points = self.sweep_points[:-self.NoCalPoints]
        data_peak_to_peak = max(data) - min(data)

        self.fit_results_sine = sine_fit_data()
        self.fit_results_quadratic = quadratic_fit_data()

        chisqr_sine = self.fit_results_sine.chisqr
        chisqr_quadratic = np.sum((self.fit_results_quadratic[0] - data)**2)

        if (chisqr_quadratic < chisqr_sine) or \
                (data_peak_to_peak/cal_peak_to_peak < .5):
            self.fit_type = 'quadratic'
            self.slope = self.fit_results_quadratic[1][1]
            amplitude = cal_peak_to_peak / 2

        else:
            self.fit_type = 'sine'
            amplitude = self.fit_results_sine.params['amplitude']
            frequency = self.fit_results_sine.params['frequency']
            self.slope = 2 * np.pi * amplitude * frequency

        self.drive_detuning = self.slope / (2 * np.pi * abs(amplitude))
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
        figsize = kw.pop('figsize', (11, 2*len(self.value_names)))
        if self.idx is not None:
            idx_val = np.where(self.value_names == 'I_cal_%d' % self.idx)[0][0]
        else:
            try:
                idx_val = np.where(self.value_names == 'I_cal')[0][0]
            except:  # Kind of arbitrarily choose axis 0
                idx_val = 0

        fig, axarray = plt.subplots(len(self.value_names)/2, 2,
                                    figsize=figsize)

        I_cal = self.measured_values[idx_val]
        zero_mean = np.mean(I_cal[0::2])
        zero_std = np.std(I_cal[0::2])

        one_mean = np.mean(I_cal[1::2])
        one_std = np.std(I_cal[1::2])

        self.distance = np.power(zero_mean - one_mean, 2)
        distance_error = np.sqrt(
            np.power(2.*(zero_mean - one_mean)*zero_std, 2)
            + np.power(2.*(one_mean - zero_mean)*one_std, 2))
        self.contrast = self.distance/distance_error

        for i, name in enumerate(self.value_names):
            if len(self.value_names) == 4:
                ax = axarray[i/2, i % 2]
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
        ax2.hlines((zero_mean+zero_std, zero_mean-zero_std),
                   0, len(self.sweep_points), linestyle='dashed', color='blue')
        ax2.hlines((one_mean+one_std, one_mean-one_std),
                   0, len(self.sweep_points), linestyle='dashed', color='green')
        ax2.text(2, max(I_cal)+(max(I_cal)-min(I_cal))*.04,
                 "Contrast: %.2f" % self.contrast,
                 bbox=self.box_props)
        self.save_fig(fig, figname=self.measurementstring, **kw)
        self.save_fig(fig2, figname=self.measurementstring+'_calibrated', **kw)
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

        ideal_data = kw.pop('ideal_data', None)
        if ideal_data is None:
            if len(self.measured_values[0]) == 42:
                ideal_data = np.concatenate((0*np.ones(10), 0.5*np.ones(24),
                                             np.ones(8)))
            else:
                ideal_data = np.concatenate((0*np.ones(5), 0.5*np.ones(12),
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
            fig1, fig2, ax1, axarray = self.setup_figures_and_axes()
            for i in range(2):
                if len(self.value_names) >= 4:
                    ax = axarray[i/2, i % 2]
                else:
                    ax = axarray[i]
                self.plot_results_vs_sweepparam(x=self.sweep_points,
                                                y=self.measured_values[i],
                                                fig=fig2, ax=ax,
                                                xlabel=self.xlabel,
                                                ylabel=str(
                                                    self.value_names[i]),
                                                save=False)
            ax1.set_ylim(min(self.corr_data)-.1, max(self.corr_data)+.1)
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
        if close_file:
            self.data_file.close()
        return self.deviation_total


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

        data = self.corr_data[:-1*(len(self.cal_points[0]*2))]
        n_cl = self.sweep_points[:-1*(len(self.cal_points[0]*2))]

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
        F_cl = (1/6*(3 + 2*np.exp(-1*pulse_delay/(2*T1)) +
                     np.exp(-pulse_delay/T1)))**Np
        p = 2*F_cl - 1

        return F_cl, p

    def add_textbox(self, ax, F_T1=None):

        textstr = ('\t$F_{Cl}$'+' \t= {:.4g} $\pm$ ({:.4g})%'.format(
            self.fit_res.params['fidelity_per_Clifford'].value*100,
            self.fit_res.params['fidelity_per_Clifford'].stderr*100) +
            '\n  $1-F_{Cl}$'+'  = {:.4g} $\pm$ ({:.4g})%'.format(
                (1-self.fit_res.params['fidelity_per_Clifford'].value)*100,
                (self.fit_res.params['fidelity_per_Clifford'].stderr)*100) +
            '\n\tOffset\t= {:.4g} $\pm$ ({:.4g})'.format(
                (self.fit_res.params['offset'].value),
                (self.fit_res.params['offset'].stderr)))
        if F_T1 is not None:
            textstr += ('\n\t  $F_{Cl}^{T_1}$  = ' +
                        '{:.6g}%'.format(F_T1*100))

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
                x = self.sweep_points[:-1*(len(self.cal_points[0])*2)]
                y = self.corr_data[:-1*(len(self.cal_points[0])*2)]

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
        self.ax.set_ylim(min(min(self.corr_data)-.1, -.1),
                         max(max(self.corr_data)+.1, 1.1))

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

        data = self.corr_data[:-1*(len(self.cal_points[0]*2))]
        # 1- minus all populations because we measure fidelity to 1
        data_0 = 1 - data[::2]
        data_1 = 1 - data[1::2]
        # 2-state population is just whatever is missing in 0 and 1 state
        # assumes that 2 looks like 1 state
        data_2 = 1 - (data_1) - (data_0)
        n_cl = self.sweep_points[:-1*(len(self.cal_points[0]*2)):2]

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
        numCliff = 2*list(numCliff)
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
            '$F_{\mathrm{Cl}}$'+'= {:.5g} \n\t$\pm$ ({:.2g})%'.format(
                fr0['fidelity_per_Clifford'].value*100,
                fr0['fidelity_per_Clifford'].stderr*100) +
            '\nOffset '+'= {:.4g} \n\t$\pm$ ({:.2g})%'.format(
                fr0['offset'].value*100, fr0['offset'].stderr*100))
        if F_T1 is not None:
            textstr += ('\n\t  $F_{Cl}^{T_1}$  = ' +
                        '{:.5g}%'.format(F_T1*100))
        ax.text(0.95, 0.1, textstr, transform=f.transFigure,
                fontsize=11, verticalalignment='bottom',
                horizontalalignment='right')

    def make_figures(self, n_cl, data_0, data_1, data_2,
                     close_main_fig, **kw):
        f, ax = plt.subplots()
        ax.plot(n_cl, data_0, 'o', color='b', label=r'$|0\rangle$')
        ax.plot(n_cl, data_1, '^', color='r', label=r'$|1\rangle$')
        ax.plot(n_cl, data_2, 'p', color='g', label=r'$|2\rangle$')
        ax.hlines(0, n_cl[0], n_cl[-1]*1.05, linestyle='--')
        ax.hlines(1, n_cl[0], n_cl[-1]*1.05, linestyle='--')
        ax.plot([n_cl[-1]]*4, self.corr_data[-4:], 'o', color='None')
        ax.set_xlabel('Number of Cliffords')
        ax.set_ylabel('State populations')
        plot_title = kw.pop('plot_title', textwrap.fill(
                            self.timestamp_string + '_' +
                            self.measurementstring, 40))
        ax.set_title(plot_title)
        ax.set_xlim(n_cl[0], n_cl[-1]*1.02)
        ax.set_ylim(-.1, 1.1)
        x_fine = np.linspace(0, self.sweep_points[-1]*1.05, 1000)
        fit_0 = fit_mods.double_RandomizedBenchmarkingDecay(
            x_fine, invert=1, ** self.fit_results.best_values)
        fit_1 = fit_mods.double_RandomizedBenchmarkingDecay(
            x_fine, invert=0, ** self.fit_results.best_values)
        fit_2 = 1-fit_1-fit_0

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

    def __init__(self, label='HM', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'
        super().__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             close_file=False, fitting_model='hanger',
                             show_guess=False, show=False,
                             fit_window=None, **kw):
        '''
        Available fitting_models:
            - 'hanger' = amplitude fit with slope
            - 'complex' = complex transmission fit WITHOUT slope

        'fit_window': allows to select the windows of data to fit.
                      Example: fit_window=[100,-100]
        '''
        super(self.__class__, self).run_default_analysis(
            close_file=False, **kw)
        self.add_analysis_datagroup_to_file()

        # Fit Power to a Lorentzian
        self.measured_powers = self.measured_values[0]**2

        min_index = np.argmin(self.measured_powers)
        max_index = np.argmax(self.measured_powers)

        self.min_frequency = self.sweep_points[min_index]
        self.max_frequency = self.sweep_points[max_index]

        self.peaks = a_tools.peak_finder((self.sweep_points),
                                         self.measured_powers)

        # Search for peak
        if self.peaks['dip'] is not None:    # look for dips first
            f0 = self.peaks['dip']
            amplitude_factor = -1.
        elif self.peaks['peak'] is not None:  # then look for peaks
            f0 = self.peaks['peak']
            amplitude_factor = 1.
        else:                                 # Otherwise take center of range
            f0 = np.median(self.sweep_points)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        # Fit data according to the model required
        if 'hanger' in fitting_model:
            if fitting_model == 'hanger':
                HangerModel = fit_mods.SlopedHangerAmplitudeModel
            # this in not working at the moment (need to be fixed)
            elif fitting_model == 'simple_hanger':
                HangerModel = fit_mods.HangerAmplitudeModel
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
            HangerModel.set_param_hint('f0', value=f0*1e-9,
                                       min=min(self.sweep_points)*1e-9,
                                       max=max(self.sweep_points)*1e-9)
            HangerModel.set_param_hint('A', value=amplitude_guess)
            HangerModel.set_param_hint('Q', value=Q, min=1, max=50e6)
            HangerModel.set_param_hint('Qe', value=Qe, min=1, max=50e6)
            # NB! Expressions are broken in lmfit for python 3.5 this has
            # been fixed in the lmfit repository but is not yet released
            # the newest upgrade to lmfit should fix this (MAR 18-2-2016)
            HangerModel.set_param_hint('Qi', expr='1./(1./Q-1./Qe*cos(theta))',
                                       vary=False)
            HangerModel.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
            HangerModel.set_param_hint('theta', value=0, min=-np.pi/2,
                                       max=np.pi/2)
            HangerModel.set_param_hint('slope', value=0, vary=True)
            self.params = HangerModel.make_params()

            if fit_window == None:
                data_x = self.sweep_points
                data_y = self.measured_values[0]
            else:
                data_x = self.sweep_points[fit_window[0]:fit_window[1]]
                data_y_temp = self.measured_values[0]
                data_y = data_y_temp[fit_window[0]:fit_window[1]]

            # make sure that frequencies are in Hz
            if np.floor(data_x[0]/1e8) == 0:  # frequency is defined in GHz
                data_x = data_x*1e9

            fit_res = HangerModel.fit(data=data_y,
                                      f=data_x, verbose=False)

        elif fitting_model == 'complex':
            # this is the fit with a complex transmission curve WITHOUT slope
            data_amp = self.measured_values[0]
            data_angle = self.measured_values[1]
            data_complex = np.add(
                self.measured_values[2], 1j*self.measured_values[3])

            # Initial guesses
            guess_A = max(data_amp)
            # this has to been improved
            guess_Q = f0 / abs(self.min_frequency - self.max_frequency)
            guess_Qe = guess_Q/(1-(max(data_amp)-min(data_amp)))
            # phi_v
            # number of 2*pi phase jumps
            nbr_phase_jumps = (np.diff(data_angle) > 4).sum()
            guess_phi_v = (2*np.pi*nbr_phase_jumps+(data_angle[0]-data_angle[-1]))/(
                self.sweep_points[0] - self.sweep_points[-1])
            # phi_0
            angle_resonance = data_angle[int(len(self.sweep_points)/2)]
            phase_evolution_resonance = np.exp(1j*guess_phi_v*f0)
            angle_phase_evolution = np.arctan2(
                np.imag(phase_evolution_resonance), np.real(phase_evolution_resonance))
            guess_phi_0 = angle_resonance - angle_phase_evolution

            # prepare the parameter dictionary
            P = lmfit.Parameters()
            #           (Name,         Value, Vary,      Min,     Max,  Expr)
            P.add_many(('f0',         f0/1e9, True,     None,    None,  None),
                       ('Q',         guess_Q, True,        1,    50e6,  None),
                       ('Qe',       guess_Qe, True,        1,    50e6,  None),
                       ('A',         guess_A, True,        0,    None,  None),
                       ('theta',           0, True, -np.pi/2, np.pi/2,  None),
                       ('phi_v', guess_phi_v, True,     None,    None,  None),
                       ('phi_0', guess_phi_0, True,   -np.pi,   np.pi,  None))
            P.add('Qi', expr='1./(1./Q-1./Qe*cos(theta))', vary=False)
            P.add('Qc', expr='Qe/cos(theta)', vary=False)

            # Fit
            fit_res = lmfit.minimize(fit_mods.residual_complex_fcn, P,
                                     args=(fit_mods.HangerFuncComplex, self.sweep_points, data_complex))

        elif fitting_model == 'lorentzian':
            LorentzianModel = fit_mods.LorentzianModel

            kappa_guess = 0.005

            amplitude_guess = amplitude_factor * np.pi*kappa_guess * abs(
                max(self.measured_powers)-min(self.measured_powers))

            LorentzianModel.set_param_hint('f0', value=f0,
                                           min=min(self.sweep_points),
                                           max=max(self.sweep_points))
            LorentzianModel.set_param_hint('A', value=amplitude_guess)

            # Fitting
            LorentzianModel.set_param_hint('offset',
                                           value=np.mean(self.measured_powers),
                                           vary=True)
            LorentzianModel.set_param_hint('kappa',
                                           value=kappa_guess,
                                           min=0,
                                           vary=True)
            LorentzianModel.set_param_hint('Q',
                                           expr='f0/kappa',
                                           vary=False)
            self.params = LorentzianModel.make_params()

            fit_res = LorentzianModel.fit(data=self.measured_powers,
                                          f=self.sweep_points*1.e9,
                                          params=self.params)
        else:
            raise ValueError('fitting model "{}" not recognized'.format(
                             fitting_model))

        self.fit_results = fit_res
        self.save_fitted_parameters(fit_res, var_name='HM')

        if print_fit_results is True:
            # print(fit_res.fit_report())
            print(lmfit.fit_report(fit_res))

        fig, ax = self.default_ax()
        textstr = '$f_{\mathrm{center}}$ = %.4f $\pm$ (%.3g) GHz' % (
            fit_res.params['f0'].value, fit_res.params['f0'].stderr) + '\n' \
            '$Qc$ = %.1f $\pm$ (%.1f)' % (fit_res.params['Qc'].value, fit_res.params['Qc'].stderr) + '\n' \
            '$Qi$ = %.1f $\pm$ (%.1f)' % (
                fit_res.params['Qi'].value, fit_res.params['Qi'].stderr)

        # textstr = '$f_{\mathrm{center}}$ = %.4f $\pm$ (%.3g) GHz' % (
        #     fit_res.params['f0'].value, fit_res.params['f0'].stderr)

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=self.box_props)

        if 'hanger' in fitting_model:
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[0],
                                            fig=fig, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=str('S21_mag (arb. units)'),
                                            save=False)

        elif 'complex' in fitting_model:
            self.plot_complex_results(
                data_complex, fig=fig, ax=ax, show=False, save=False)
            # second figure with amplitude
            fig2, ax2 = self.default_ax()
            ax2.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                     verticalalignment='top', bbox=self.box_props)
            self.plot_results_vs_sweepparam(x=self.sweep_points, y=data_amp,
                                            fig=fig2, ax=ax2, show=False, save=False)

        elif fitting_model == 'lorentzian':
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_powers,
                                            fig=fig, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=str('Power (arb. units)'),
                                            save=False)

        if fit_window == None:
            data_x = self.sweep_points
        else:
            data_x = self.sweep_points[fit_window[0]:fit_window[1]]

        if show_guess:
            ax.plot(self.sweep_points, fit_res.init_fit, 'k--')

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
            ax.plot(self.sweep_points, fit_res.best_fit, 'r-')
            f0 = self.fit_results.values['f0']
            plt.plot(f0*1e9, fit_res.eval(f=f0*1e9), 'o', ms=8)

            # save figure
            self.save_fig(fig, xlabel=self.xlabel, ylabel='Mag', **kw)

        if show:
            plt.show()
        # self.save_fig(fig, xlabel=self.xlabel, ylabel='Mag', **kw)
        if close_file:
            self.data_file.close()
        return fit_res


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
        self.measured_powers = self.measured_values[0]**2

        min_index = np.argmin(self.measured_powers)
        max_index = np.argmax(self.measured_powers)

        self.min_frequency = self.sweep_points[min_index]
        self.max_frequency = self.sweep_points[max_index]

        self.peaks = a_tools.peak_finder((self.sweep_points),
                                         self.measured_values[0])

        if self.peaks['dip'] is not None:    # look for dips first
            f0 = self.peaks['dip']
            amplitude_factor = -1.
        elif self.peaks['peak'] is not None:  # then look for peaks
            f0 = self.peaks['peak']
            amplitude_factor = 1.
        else:                                 # Otherwise take center of range
            f0 = np.median(self.sweep_points)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        def poly(x, c0, c1, c2):
            "line"
            return c2 * x**2 + c1 * x + c0

        def cosine(x, amplitude, frequency, phase, offset):
            # Naming convention, frequency should be Hz
            # omega is in radial freq
            return amplitude*np.cos(2*np.pi*frequency*x + phase)+offset

        def hanger_function_amplitude(x, f0, Q, Qe, A, theta):
            '''
            This is the function for a hanger  which does not take into account
            a possible slope.
            This function may be preferred over SlopedHangerFunc if the area around
            the hanger is small.
            In this case it may misjudge the slope
            Theta is the asymmetry parameter
            '''
            return abs(A*(1.-Q/Qe*np.exp(1.j*theta)/(1.+2.j*Q*(x-f0)/f0)))

        HangerModel = lmfit.Model(hanger_function_amplitude)\
            + lmfit.Model(cosine) \
            + lmfit.Model(poly)

        # amplitude_guess = np.pi*sigma_guess * abs(
        #     max(self.measured_powers)-min(self.measured_powers))
        amplitude_guess = max(self.measured_powers)-min(self.measured_powers)

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
        HangerModel.set_param_hint('theta', value=0, min=-np.pi/2,
                                   max=np.pi/2)
        HangerModel.set_param_hint('slope', value=0, vary=True)

        HangerModel.set_param_hint('c0', value=0, vary=False)
        HangerModel.set_param_hint('c1', value=0, vary=True)
        HangerModel.set_param_hint('c2', value=0, vary=True)

        HangerModel.set_param_hint('amplitude', value=0.05, min=0, vary=False)
        HangerModel.set_param_hint(
            'frequency', value=50, min=0, max=300, vary=True)
        HangerModel.set_param_hint(
            'phase', value=0, min=0, max=2*np.pi, vary=True)
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

    def __init__(self, label='Source', **kw):
        kw['label'] = label
        kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, print_fit_results=False,
                             show=False, fit_results_peak=True, **kw):
        def fit_data():
            try:
                self.data_dist = a_tools.calculate_distance_ground_state(
                    data_real=self.measured_values[2],
                    data_imag=self.measured_values[3])
            except:
                # Quick fix to make it work with pulsed spec which does not
                # return both I,Q and, amp and phase
                self.data_dist = a_tools.calculate_distance_ground_state(
                    data_real=self.measured_values[0],
                    data_imag=self.measured_values[1])

            self.peaks = a_tools.peak_finder(
                self.sweep_points, a_tools.smooth(self.data_dist))

            if self.peaks['peak'] is not None:
                f0 = self.peaks['peak']
                kappa_guess = self.peaks['peak_width'] / 4

            else:  # Otherwise take center of range
                f0 = np.median(self.sweep_points)
                kappa_guess = 0.005

            amplitude_guess = np.pi * kappa_guess * \
                abs(max(self.data_dist) - min(self.data_dist))

            LorentzianModel = fit_mods.LorentzianModel
            LorentzianModel.set_param_hint('f0',
                                           min=min(self.sweep_points),
                                           max=max(self.sweep_points),
                                           value=f0)
            LorentzianModel.set_param_hint('A',
                                           value=amplitude_guess,
                                           min=4*np.var(self.data_dist))
            LorentzianModel.set_param_hint('offset',
                                           value=np.mean(self.data_dist),
                                           vary=True)
            LorentzianModel.set_param_hint('kappa',
                                           value=kappa_guess,
                                           min=0,
                                           vary=True)
            LorentzianModel.set_param_hint('Q',
                                           expr='f0/kappa',
                                           vary=False)
            self.params = LorentzianModel.make_params()

            fit_res = LorentzianModel.fit(data=self.data_dist,
                                          f=self.sweep_points*1.e9,
                                          params=self.params)
            print('min ampl', 2*np.var(self.data_dist))
            return fit_res

        self.add_analysis_datagroup_to_file()
        self.savename = kw.pop('save_name', 'Source Frequency')
        show_guess = kw.pop('show_guess', True)
        close_file = kw.pop('close_file', True)
        self.get_naming_and_values()

        if len(self.value_names) == 1:
            fig, axarray = plt.subplots(1, 1, figsize=(12, 10))
            axes = [axarray]
        elif len(self.value_names) == 2:
            fig, axarray = plt.subplots(2, 1, figsize=(12, 10))
            axes = axarray
        elif len(self.value_names) > 2:
            fig, axarray = plt.subplots(2, 2, figsize=(12, 10))
            axes = [axarray[k/2, k % 2] for k in range(len(self.value_names))]

        use_max = kw.get('use_max', False)

        fit_res = fit_data()
        self.fitted_freq = fit_res.params['f0'].value

        self.fit_results.append(fit_res)
        self.save_fitted_parameters(fit_res,
                                    var_name='distance', save_peaks=True)
        if print_fit_results is True:
            print(fit_res.fit_report())

        for k in range(len(self.measured_values)):
            ax = axes[k]
            textstr = '$f_{\mathrm{center}}$ = %.5g $\pm$ (%.3g) GHz\n' % (
                fit_res.params['f0'].value,
                fit_res.params['f0'].stderr)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                    fontsize=11, verticalalignment='top', bbox=self.box_props)

            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[k],
                                            fig=fig, ax=ax,
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[k],
                                            save=False)
            # Plot a point for each plot at the chosen best fit f0 frequency
            f0 = fit_res.params['f0'].value
            f0_idx = a_tools.nearest_idx(self.sweep_points, f0)
            axes[k].plot(f0, self.measured_values[k][f0_idx], 'o', ms=8)

        # Plotting distance from |0>
        label = r'f0={:.5}$\pm$ {:.2} MHz, linewidth={:.4}$\pm${:.2} MHz'.format(
            fit_res.params['f0'].value/1e6, fit_res.params['f0'].stderr/1e6, fit_res.params['kappa'].value/1e6, fit_res.params['kappa'].stderr/1e6)
        fig_dist, ax_dist = self.default_ax()
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.data_dist,
                                        fig=fig_dist, ax=ax_dist,
                                        xlabel=self.xlabel,
                                        ylabel='S21 distance (V)',
                                        label=False,
                                        save=False)
        ax_dist.plot(self.sweep_points, fit_res.best_fit, 'r-')
        ax_dist.plot(f0, fit_res.best_fit[f0_idx], 'o', ms=8)
        if show_guess:
            ax_dist.plot(self.sweep_points, fit_res.init_fit, 'k--')
        ax_dist.text(0.05, 0.95, label, transform=ax_dist.transAxes,
                     fontsize=11, verticalalignment='top', bbox=self.box_props)
        self.save_fig(fig, figname=self.savename, **kw)
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

        return linewidth_estimate


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
        ax1.set_title(self.timestamp_string+'\n' + 'Qubit Frequency')
        ax1.set_xlabel((str(self.sweep_name + ' (' + self.sweep_unit + ')')))
        ax1.set_ylabel(r'$f_{qubit}$ (GHz)')
        ax1.grid()

        fig2, axarray2 = plt.subplots(2, 1, figsize=figsize)
        axarray2[0].set_title(self.timestamp_string+'\n' + 'Qubit Coherence')
        axarray2[0].errorbar(
            x=x,
            y=T1*1e-3, yerr=T1_stderr*1e-3,
            fmt='o', label='$T_1$')
        axarray2[0].errorbar(
            x=x,
            y=T2_echo*1e-3, yerr=T2_echo_stderr*1e-3,
            fmt='o', label='$T_2$-echo')
        axarray2[0].errorbar(
            x=x,
            y=T2_star*1e-3, yerr=T2_star_stderr*1e-3,
            fmt='o', label='$T_2$-star')
        axarray2[0].set_xlabel(r'dac voltage')
        axarray2[0].set_ylabel(r'$\tau (\mu s)$ ')
        # axarray[0].set_xlim(-600, 700)
        axarray2[0].set_ylim(0, max([max(T1*1e-3), max(T2_echo*1e-3)])
                             + 3*max(T1_stderr*1e-3))
        axarray2[0].legend()
        axarray2[0].grid()

        axarray2[1].errorbar(
            x=qubit_freq*1e-9,
            y=T1*1e-3, yerr=T1_stderr*1e-3,
            fmt='o', label='$T_1$')
        axarray2[1].errorbar(
            x=qubit_freq*1e-9,
            y=T2_echo*1e-3, yerr=T2_echo_stderr*1e-3,
            fmt='o', label='$T_2$-echo')
        axarray2[1].errorbar(
            x=qubit_freq*1e-9,
            y=T2_star*1e-3, yerr=T2_star_stderr*1e-3,
            fmt='o', label='$T_2^\star$')
        axarray2[1].set_xlabel(r'$f_{qubit}$ (GHz)')
        axarray2[1].set_ylabel(r'$\tau (\mu s)$ ')
        # axarray[1].set_xlim(-600, 700)
        axarray2[1].set_ylim(0, max([max(T1*1e-3), max(T2_echo*1e-3)])
                             + 3*max(T1_stderr*1e-3))
        axarray2[1].legend(loc=2)
        axarray2[1].grid()

        fig3, axarray3 = plt.subplots(2, 1, figsize=figsize)
        axarray3[0].set_title(self.timestamp+'\n' + 'AWG pulse amplitude')
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
            self.sweep_points[0], self.sweep_points[-1], 1000)*1e-3

        self.qubit_freq = self.measured_values[2]
        self.qubit_freq_stderr = self.measured_values[3]

        fit_res = fit_qubit_frequency(sweep_points=x*1e-3,
                                      data=self.qubit_freq*1e9,
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
        ax1.errorbar(x=x*1e-3, y=self.qubit_freq*1e9,
                     yerr=self.qubit_freq_stderr,
                     label='data', fmt='ob')

        if show_guess:
            ax1.plot(x*1e-3, fit_res.init_fit, 'k--')

        ax1.plot(x_fine, fitted_freqs, '--c', label='fit')
        ax1.legend()
        ax1.set_title(self.timestamp+'\n' + 'Qubit Frequency')
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
                             plot_all=False, save_fig=True,
                             transpose=False,
                             **kw):
        close_file = kw.pop('close_file', True)

        self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []

        for i, meas_vals in enumerate(self.measured_values):
            if (not plot_all) & (i >= 1):
                break
            # Linecuts are above because somehow normalization applies to both
            # colorplot and linecuts otherwise.
            if plot_linecuts:
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_linecut_{i}'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    i=i)
                a_tools.linecut_plot(x=self.sweep_points,
                                     y=self.sweep_points_2D,
                                     z=self.measured_values[i],
                                     plot_title=fig_title,
                                     xlabel=self.xlabel,
                                     y_name=self.sweep_name_2D,
                                     y_unit=self.sweep_unit_2D,
                                     log=linecut_log,
                                     zlabel=self.zlabels[i],
                                     fig=fig, ax=ax, **kw)
                if save_fig:
                    self.save_fig(fig, figname=fig_title,
                                  fig_tight=False, **kw)

            fig, ax = self.default_ax(figsize=(8, 5))
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            if normalize:
                print("normalize on")
            # print "unransposed",meas_vals
            # print "transposed", meas_vals.transpose()
            fig_title = '{timestamp}_{measurement}_{i}'.format(
                timestamp=self.timestamp_string,
                measurement=self.measurementstring,
                i=i)
            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=meas_vals.transpose(),
                               plot_title=fig_title,
                               xlabel=self.xlabel,
                               ylabel=self.ylabel,
                               zlabel=self.zlabels[i],
                               fig=fig, ax=ax,
                               log=colorplot_log,
                               transpose=transpose,
                               normalize=normalize,
                               **kw)
            if save_fig:
                print("saving fig_title", fig_title)
                self.save_fig(fig, figname=fig_title, **kw)
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

            textstr = 'Q phase of minimum =  %.2f deg'  % self.phase_min + '\n' + \
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
    '''

    def __init__(self, label='Three_tone', **kw):
        kw['label'] = label
        # kw['h5mode'] = 'r+'  # Read write mode, file must exist
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, f01=None, f12=None,
                             amp_lims=[None, None], line_color='k',
                             phase_lims=[-180, 180], **kw):
        self.get_naming_and_values_2D()
        # figsize wider for colorbar
        fig1, ax1 = self.default_ax(figsize=(8, 5))
        measured_powers = self.measured_values[0]
        measured_phases = self.measured_values[1]

        fig1_title = self.timestamp_string + \
            self.measurementstring+'_'+'Amplitude'
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
        fig2_title = self.timestamp_string+self.measurementstring+'_'+'Phase'
        a_tools.color_plot(x=self.sweep_points,
                           y=self.sweep_points_2D,
                           z=measured_phases.transpose(),
                           xlabel=self.xlabel,
                           ylabel=self.ylabel,
                           zlabel=self.zlabels[1],
                           clim=phase_lims,
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
                     (f01 + f12-min(self.sweep_points),
                      f01 + f12-max(self.sweep_points)),
                     linestyle='dashed', lw=2, color=line_color, alpha=.5)
            ax2.plot((min(self.sweep_points),
                      max(self.sweep_points)),
                     (f01 + f12-min(self.sweep_points),
                      f01 + f12-max(self.sweep_points)),
                     linestyle='dashed', lw=2, color=line_color, alpha=.5)
        if (f01 is not None) and (f12 is not None):
            anharm = f01-f12
            EC, EJ = a_tools.fit_EC_EJ(f01, f12)
            # EC *= 1000

            textstr = 'f01 = {:.4g} GHz'.format(f01*1e-9) + '\n' + \
                'f12 = {:.4g} GHz'.format(f12*1e-9) + '\n' + \
                'anharm ~= {:.4g} MHz'.format(anharm*1e-6) + '\n' + \
                'EC = {:.4g} MHz'.format(EC*1e-6) + '\n' + \
                'EJ = {:.4g} GHz'.format(EJ*1e-9)
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
        self.save_fig(fig1, figname=ax1.get_title(), **kw)
        self.save_fig(fig2, figname=ax2.get_title(), **kw)
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
    def run_default_analysis(self, normalize=True, plot_Q=True, plot_f0=True, plot_linecuts=True,
                             linecut_log=True, plot_all=False, save_fig=True,
                             **kw):
        close_file = kw.pop('close_file', True)
        self.add_analysis_datagroup_to_file()

        self.get_naming_and_values_2D()
        self.fig_array = []
        self.ax_array = []
        fits = {}  # Dictionary to store the fit results in. Fit results are a
        # dictionary themselfes -> Dictionary of Dictionaries

        for u, power in enumerate(self.sweep_points_2D):
            fit_res = self.fit_hanger_model(
                self.sweep_points, self.measured_values[0][:, u])
            self.save_fitted_parameters(fit_res, var_name='Powersweep'+str(u))
            fits[str(power)] = fit_res
        self.fit_results = fits

        for i, meas_vals in enumerate(self.measured_values):
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
                ax.set_ylabel('Quality Factor')
                ax.set_xlabel('Power [dBm]')

                if save_fig:
                    self.save_fig(
                        fig, figname=fig_title, fig_tight=False, **kw)

            if plot_f0:
                f0 = np.zeros(len(self.sweep_points_2D))
                for u, power in enumerate(self.sweep_points_2D):
                    f0[u] = self.fit_results[str(power)].values['f0']
                fig, ax = self.default_ax(figsize=(8, 5))
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                fig_title = '{timestamp}_{measurement}_{val_name}_f0vsPower'.format(
                    timestamp=self.timestamp_string,
                    measurement=self.measurementstring,
                    val_name=self.zlabels[i])
                ax.plot(
                    self.sweep_points_2D, f0, 'blue', label='Cavity Frequency')
                ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
                ax.set_position([0.15, 0.1, 0.5, 0.8])
                ax.set_ylabel('Frequency [GHz]')
                ax.set_xlabel('Power [dBm]')

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
                                     xlabel=self.xlabel,
                                     y_name=self.sweep_name_2D,
                                     y_unit=self.sweep_unit_2D,
                                     log=linecut_log,
                                     zlabel=self.zlabels[i],
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
                               xlabel=self.xlabel,
                               ylabel=self.ylabel,
                               zlabel=self.zlabels[i],
                               fig=fig, ax=ax, **kw)
            if save_fig:
                self.save_fig(fig, figname=fig_title, **kw)

        if close_file:
            self.finish()

    def fit_hanger_model(self, sweep_values, measured_values):
        HangerModel = fit_mods.SlopedHangerAmplitudeModel

        # amplitude_guess = np.pi*sigma_guess * abs(
        #     max(self.measured_powers)-min(self.measured_powers))

        # Fit Power to a Lorentzian
        measured_powers = measured_values**2

        min_index = np.argmin(measured_powers)
        max_index = np.argmax(measured_powers)

        min_frequency = sweep_values[min_index]
        max_frequency = sweep_values[max_index]

        peaks = a_tools.peak_finder((sweep_values),
                                    measured_values)

        if peaks['dip'] is not None:    # look for dips first
            f0 = peaks['dip']
            amplitude_factor = -1.
        elif peaks['peak'] is not None:  # then look for peaks
            f0 = peaks['peak']
            amplitude_factor = 1.
        else:                                 # Otherwise take center of range
            f0 = np.median(sweep_values)
            amplitude_factor = -1.
            logging.error('No peaks or dips in range')
            # If this error is raised, it should continue the analysis but
            # not use it to update the qubit object

        amplitude_guess = max(measured_powers)-min(measured_powers)
        # Creating parameters and estimations
        S21min = min(measured_values)/max(measured_values)

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
        HangerModel.set_param_hint('theta', value=0, min=-np.pi/2,
                                   max=np.pi/2)
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
                series = dm_tools.binary_derivative(series)

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
                self.timestamp_string+'\n'+self.measurementstring)
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
                self.timestamp_string+'\n'+self.measurementstring)
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
        self.std_err_rtf = self.std_rtf/np.sqrt(len(self.sweep_points_2D))

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
            self.ax.set_title(self.timestamp_string+'\n' +
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
        self.std_err_rtf = self.std_rtf/np.sqrt(len(self.sweep_points_2D))
        term_cts = Counter(term_cond)
        # note that we only take 1 derivative and this is not equal to the
        # notion of detection events as in Kelly et al.
        terminated_by_flip = float(term_cts['single event'])
        terminated_by_RO_err = float(term_cts['double event'])
        total_cts = terminated_by_RO_err + terminated_by_flip + \
            term_cts['unknown']
        self.flip_err_frac = terminated_by_flip/total_cts*100.
        self.RO_err_frac = terminated_by_RO_err/total_cts*100.

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
            self.ax.set_title(self.timestamp_string+'\n' +
                              self.measurementstring)
            self.save_fig(self.fig, xlabel='Rounds to failure',
                          ylabel='normalized occurence', **kw)

        return self.mean_rtf, self.std_err_rtf, self.RO_err_frac, self.flip_err_frac


class butterfly_analysis(MeasurementAnalysis):

    '''
    Extracts the coefficients for the post-measurement butterfly
    '''

    def __init__(self,  auto=True, label='Butterfly', close_file=True,
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
        I_shots = self.measured_values[0]
        Q_shots = self.measured_values[1]

        shots = I_shots+1j*Q_shots
        rot_shots = dm_tools.rotate_complex(shots, angle=theta_in, deg=True)
        I_shots = rot_shots.real
        Q_shots = rot_shots.imag

        self.data = I_shots

        if initialize:
            if threshold_init == None:
                threshold_init = threshold

            # reshuffeling the data to endup with two arrays for the diffeent
            # input states
            shots = np.size(self.data)
            shots_per_mmt = np.floor_divide(shots, 6)
            shots_used = shots_per_mmt*6
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
            self.data_exc = dm_tools.postselect(threshold=threshold_init,
                                                data=self.data_exc,
                                                positive_case=case)
            self.data_rel = dm_tools.postselect(threshold=threshold_init,
                                                data=self.data_rel,
                                                positive_case=case)
            self.data_exc_post = self.data_exc[:, 1:]
            self.data_rel_post = self.data_rel[:, 1:]
            self.data_exc = self.data_exc_post
            self.data_rel = self.data_rel_post
            fraction = (
                np.size(self.data_exc) + np.size(self.data_exc))*3/shots_used/2

            print("postselection fraction", fraction)

            #raise NotImplementedError()
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
                                              positive_case=case)
            self.data_rel = dm_tools.digitize(threshold=threshold,
                                              data=self.data_rel,
                                              positive_case=case)
        if close_file:
            self.data_file.close()
        if auto is True:
            self.run_default_analysis(**kw)

    def run_default_analysis(self,  **kw):
        verbose = kw.pop('verbose', False)
        exc_coeffs = dm_tools.butterfly_data_binning(Z=self.data_exc,
                                                     initial_state=0)
        rel_coeffs = dm_tools.butterfly_data_binning(Z=self.data_rel,
                                                     initial_state=1)
        self.butterfly_coeffs = dm_tools.butterfly_matrix_inversion(exc_coeffs,
                                                                    rel_coeffs)
        # eps,declaration,output_input
        F_a_butterfly = (1-(self.butterfly_coeffs.get('eps00_1') +
                            self.butterfly_coeffs.get('eps01_1') +
                            self.butterfly_coeffs.get('eps10_0') +
                            self.butterfly_coeffs.get('eps11_0'))/2)

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
        return self.butterfly_coeffs


class Tomo_Analysis(MeasurementAnalysis):

    def __init__(self, num_qubits=2, quad='IQ', over_complete_set=False,
                 plot_oper=True, folder=None, auto=True, **kw):
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits
        self.over_complete_set = over_complete_set
        if over_complete_set:
            self.num_measurements = 6**num_qubits
        else:
            self.num_measurements = 4**num_qubits
        self.quad = quad
        self.plot_oper = plot_oper
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, **kw):
        self.get_naming_and_values()
        data_I = self.get_values(key='I')
        data_Q = self.get_values(key='Q')
        measurements_tomo = (np.array([data_I[0:36], data_Q[0:36]])).flatten()
        measurements_cal = np.array([np.average(data_I[36:39]),
                                     np.average(data_I[39:42]),
                                     np.average(data_I[42:45]),
                                     np.average(data_I[45:48]),
                                     np.average(data_Q[36:39]),
                                     np.average(data_Q[39:42]),
                                     np.average(data_Q[42:45]),
                                     np.average(data_Q[45:48])])

        if self.quad == 'IQ':
            self.use_both_quad = True
        else:
            self.use_both_quad = False
            if self.quad == 'Q':
                measurements_tomo[0:self.num_measurements] = measurements_tomo[
                    self.num_measurements:]
                measurements_cal[0:self.num_states] = measurements_cal[
                    self.num_states:]
            elif self.quad != 'I':
                raise Error('Quadrature to use is not clear.')

        beta_I = self.calibrate_beta(
            measurements_cal=measurements_cal[0:self.num_states])
        beta_Q = np.zeros(self.num_states)
        if self.use_both_quad == True:
            beta_Q = self.calibrate_beta(
                measurements_cal=measurements_cal[self.num_states:])

        if self.use_both_quad == True:
            max_idx = 2*self.num_measurements
        else:
            max_idx = self.num_measurements

        results = self.calc_operators(
            measurements_tomo[:max_idx], beta_I, beta_Q)
        self.results = results
        self.dens_mat = self.calc_density_matrix(results)
        if self.plot_oper == True:
            self.plot_operators(**kw)

    def calibrate_beta(self, measurements_cal):
        # calibrates betas for the measurement operator
        cal_matrix = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                cal_matrix[i, j] = (-1)**(self.get_bit_sum(i & j))
        beta = np.dot(np.linalg.inv(cal_matrix), measurements_cal)
        print(beta)
        return beta

    def calc_operators(self, measurements_tomo, beta_I, beta_Q):
        M = self.num_measurements
        K = 4**self.num_qubits - 1
        if self.use_both_quad == False:
            measurement_matrix = np.zeros((M, K))
            measurements_tomo[:M] = measurements_tomo[:M] - beta_I[0]
            measurements_tomo[M:] = measurements_tomo[M:] - beta_Q[0]
        else:
            measurement_matrix = np.zeros((2*M, K))

        for i in range(M):
            for j in range(1, self.num_states):
                place, sign = self.rotate_M0_elem(i, j)
                measurement_matrix[i, place] = sign*beta_I[j]
        if self.use_both_quad == True:
            for i in range(M):
                for j in range(1, self.num_states):
                    place, sign = self.rotate_M0_elem(i, j)
                    measurement_matrix[i+M, place] = sign*beta_Q[j]

        inverse = np.linalg.pinv(measurement_matrix)
        pauli_operators = np.dot(inverse, measurements_tomo)

        p_op = np.zeros(4**self.num_qubits)
        p_op[0] = 1
        p_op[1:] = pauli_operators
        return np.real(p_op)

    def rotate_M0_elem(self, rotation, elem):
        # moves first the first one
        rot_vector = self.get_rotation_vector(rotation)
        # moves first the last one
        elem_op_vector = self.get_m0_elem_vector(elem)

        res_vector = np.zeros(len(rot_vector))
        sign = 1
        for i in range(len(rot_vector)):
            value = elem_op_vector[i]
            res_vector[i] = 0
            if value == 1:
                if rot_vector[i] == 0:
                    res_vector[i] = value
                    sign = sign
                if rot_vector[i] == 1:
                    res_vector[i] = value
                    sign = -1*sign
                if rot_vector[i] == 2:
                    res_vector[i] = 3
                    sign = sign
                if rot_vector[i] == 3:
                    res_vector[i] = 2
                    sign = -1*sign
                if rot_vector[i] == 4:
                    res_vector[i] = 3
                    sign = -1*sign
                if rot_vector[i] == 5:
                    res_vector[i] = 2
                    sign = sign
            else:
                res_vector[i] = value
                sign = sign

        res_number = self.get_pauli_op_number(res_vector) - 1
        # the minus 1 is to not consider the <II> in the pauli vector
        return np.array([res_number, sign])

    def calc_density_matrix(self, pauli_operators):
        Id2 = np.identity(2)
        Z_op = [[1+0.j, 0+0.j], [0+0.j, -1+0.j]]
        X_op = [[0+0.j, 1+0.j], [1+0.j, 0+0.j]]
        Y_op = [[0+0.j, -1.j], [1.j, 0+0.j]]
        rho = np.zeros((self.num_states, self.num_states))
        # np.kron works in the same way as bits (the most signifcant at left)
        for i in range(0, 2**self.num_states):
            vector = self.get_pauli_op_vector(i)
            acum = 1
            for j in range(self.num_qubits-1, -1, -1):
                if vector[j] == 0:
                    temp = np.kron(Id2, acum)
                if vector[j] == 1:
                    temp = np.kron(Z_op, acum)
                if vector[j] == 2:
                    temp = np.kron(X_op, acum)
                if vector[j] == 3:
                    temp = np.kron(Y_op, acum)
                del acum
                acum = temp
                del temp
            rho = rho + acum*pauli_operators[i]
        return rho/self.num_states

    def get_pauli_op_number(self, pauli_vector):
        pauli_number = 0
        N = len(pauli_vector)
        for i in range(0, N, 1):
            pauli_number += pauli_vector[N-i-1] * (4**i)
        return pauli_number

    def get_pauli_op_vector(self, pauli_number):
        N = self.num_qubits
        pauli_vector = np.zeros(N)
        rest = pauli_number
        for i in range(0, N, 1):
            value = rest % 4
            pauli_vector[i] = value
            rest = (rest-value)/4
        return pauli_vector

    def get_m0_elem_vector(self, elem_number):
        elem_vector = np.zeros(self.num_qubits)
        rest = elem_number
        for i in range(self.num_qubits-1, -1, -1):
            value = rest % 2
            elem_vector[i] = value
            rest = (rest-value)/2
        return elem_vector

    def get_rotation_vector(self, rot_number):
        N = self.num_qubits
        rot_vector = np.zeros(N)
        rest = rot_number
        if self.over_complete_set:
            total = 6
        else:
            total = 4
        for i in range(N-1, -1, -1):
            value = rest % total
            rot_vector[i] = value
            rest = (rest-value)/total
        return rot_vector

    def get_bit_sum(self, number):
        N = self.num_qubits
        summ = 0
        rest = number
        for i in range(N-1, -1, -1):
            value = rest % 2
            summ += value
            rest = (rest-value)/2
        return summ

    def get_operators_label(self):
        labels = []
        for i in range(2**self.num_states):
            vector = self.get_pauli_op_vector(i)
            label = ''
            for j in range(self.num_qubits):
                if vector[j] == 0:
                    label = 'I'+label
                if vector[j] == 1:
                    label = 'Z'+label
                if vector[j] == 2:
                    label = 'X'+label
                if vector[j] == 3:
                    label = 'Y'+label
            labels.append(label)

        labels = ['IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX',
                  'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
        return labels

    def plot_operators(self, **kw):
        import qutip as qtip
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(121)
        pauli_1, pauli_2, pauli_cor = self.order_pauli_output2(self.results)
        width = 0.35
        ind1 = np.arange(3)
        ind2 = np.arange(3, 6)
        ind3 = np.arange(6, 15)
        ind = np.arange(15)
        q1 = ax.bar(ind1, pauli_1, width, color='r')
        q1 = ax.bar(ind2, pauli_2, width, color='b')
        q2 = ax.bar(ind3, pauli_cor, width, color='purple')

        ax.set_title('%d Qubit State Tomography' % self.num_qubits)
#         ax.set_ylim(-1,1)
        ax.set_xticks(np.arange(0, 2**self.num_states))
        ax.set_xticklabels(self.get_operators_label())
        ax2 = fig.add_subplot(122, projection='3d')
        qtip.matrix_histogram_complex(qtip.Qobj(self.dens_mat),
                                      xlabels=['00', '01', '10', '11'], ylabels=['00', '01', '10', '11'],
                                      fig=fig, ax=ax2)
        print('working so far')
        self.save_fig(fig, figname=self.measurementstring, **kw)
# print 'Concurrence = %f' %
# qt.concurrence(qt.Qobj(self.dens_mat,dims=[[2, 2], [2, 2]]))
        return

    def order_pauli_output2(self, pauli_op_dis):
        pauli_1 = np.array([pauli_op_dis[2], pauli_op_dis[3], pauli_op_dis[1]])
        pauli_2 = np.array(
            [pauli_op_dis[8], pauli_op_dis[12], pauli_op_dis[4]])
        pauli_corr = np.array([pauli_op_dis[10], pauli_op_dis[11], pauli_op_dis[9],
                               pauli_op_dis[14], pauli_op_dis[
                                   15], pauli_op_dis[13],
                               pauli_op_dis[6], pauli_op_dis[7], pauli_op_dis[5]])
        return pauli_1, pauli_2, pauli_corr

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
            self.scan_start = splitted[-2]+'_'+splitted[-1][:6]
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
        plot_step = plot_times[1]-plot_times[0]

        plot_x = x
        x_step = plot_x[1]-plot_x[0]

        result = z

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        cmin, cmax = 0, 1
        fig_clim = [cmin, cmax]
        out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
                                        xvals=plot_times,
                                        yvals=plot_x,
                                        zvals=result)
        ax.set_xlabel(r'AWG Amp (Vpp)')
        ax.set_ylabel(r'Time (ns)')
        ax.set_title('%s: Chevron scan' % self.scan_start)
        # ax.set_xlim(xmin, xmax)
        ax.set_ylim(plot_x.min()-x_step/2., plot_x.max()+x_step/2.)
        ax.set_xlim(
            plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
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
            np.arange(fig_clim[0], 1.01*fig_clim[1], (fig_clim[1]-fig_clim[0])/5.))
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
        if dimy*dimx < len(x):
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
        dim = dimx*dimy
        if dim > len(data):
            dimy = dimy - 1
        return x, y[:dimy], (data[:dimx*dimy].reshape((dimy, dimx))).transpose()

    def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            if figname is None:
                figname = (self.scan_start+'_Chevron_2D_'+'.'+plot_format)
            else:
                figname = (figname+'.' + plot_format)
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


class DoubleFrequency(object):

    def __init__(self, auto=True, label='Ramsey', timestamp=None):
        if timestamp is None:
            self.folder = a_tools.latest_data(label)
            splitted = self.folder.split('\\')
            self.scan_start = splitted[-2]+'_'+splitted[-1][:6]
            self.scan_stop = self.scan_start
        else:
            self.scan_start = timestamp
            self.scan_stop = timestamp
            self.folder = a_tools.get_folder(timestamp=self.scan_start)
        self.pdict = {'I': 'amp',
                      'sweep_points': 'sweep_points'}
        self.opt_dict = {'scan_label': label}
        self.nparams = ['I', 'sweep_points']
        self.label = label
        if auto == True:
            self.analysis()

    def analysis(self):
        # print(self.scan_start,self.scan_stop,self.opt_dict,self.pdict,self.nparams)
        ramsey_scan = ca.quick_analysis(t_start=self.scan_start,
                                        t_stop=self.scan_stop,
                                        options_dict=self.opt_dict,
                                        params_dict_TD=self.pdict,
                                        numeric_params=self.nparams)
        x = ramsey_scan.TD_dict['sweep_points'][0]*1e6
        y = a_tools.normalize_data_v3(ramsey_scan.TD_dict['I'][0])
        # print(y)

        fit_res = self.fit(x[:-4], y[:-4])
        self.fit_res = fit_res
        # print(fit_res.best_values)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.box_props = dict(boxstyle='Square', facecolor='white', alpha=0.8)

        textstr = ('  $f_1$  \t= %.3g $ \t \pm$ (%.3g) MHz'
                   % (fit_res.params['freq_1'].value,
                      fit_res.params['freq_1'].stderr) +
                   '\n$f_2$ \t= %.3g $ \t \pm$ (%.3g) MHz'
                   % (fit_res.params['freq_2'].value,
                       fit_res.params['freq_2'].stderr) +
                   '\n$T_2^\star (1)$ = %.3g $\t \pm$ (%.3g) s '
                   % (fit_res.params['tau_1'].value,
                       fit_res.params['tau_1'].stderr) +
                   '\n$T_2^\star (2)$ = %.3g $\t \pm$ (%.3g) s '
                   % (fit_res.params['tau_2'].value,
                       fit_res.params['tau_2'].stderr))
        ax.text(0.4, 0.95, textstr,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=self.box_props)

        plot_x = x
        plot_step = plot_x[1]-plot_x[0]

        ax.set_xlabel(r'Time ($\mu s$)')
        ax.set_ylabel(r'$F |1\rangle$')
        ax.set_title('%s: Double Frequency analysis' % self.scan_start)
        ax.set_xlim(plot_x.min()-plot_step/2., plot_x.max()+plot_step/2.)

        ax.plot(plot_x, y, 'bo')
        ax.plot(plot_x[:-4], self.fit_plot, 'r-')

        fig.tight_layout()
        self.save_fig(fig)

    def fit(self, sweep_values, measured_values):
        Double_Cos_Model = fit_mods.DoubleExpDampOscModel
        fourier_max_pos = a_tools.peak_finder_v2(np.arange(1, len(sweep_values)/2, 1),
                                                 abs(np.fft.fft(measured_values))[
                                                     1:len(measured_values)//2],
                                                 window_len=1, perc=95)
        if len(fourier_max_pos) == 1:
            freq_guess = 1./sweep_values[-1] * \
                (fourier_max_pos[0]+np.array([-1, 1]))
        else:
            freq_guess = 1./sweep_values[-1]*fourier_max_pos
        # print(abs(np.fft.fft(measured_values))[1:len(measured_values)/2])
        Double_Cos_Model.set_param_hint(
            'tau_1', value=10., min=1., max=250., vary=True)
        Double_Cos_Model.set_param_hint(
            'tau_2', value=9., min=1., max=250., vary=True)
        # Double_Cos_Model.set_param_hint('tau_2',expr='tau_1', min = 1., max = 250.)
        Double_Cos_Model.set_param_hint(
            'freq_1', value=freq_guess[0], min=0)  # 9.9
        Double_Cos_Model.set_param_hint(
            'freq_2', value=freq_guess[1], min=0)  # 8.26
        # , min = -2*np.pi, max = 2* np.pi)
        Double_Cos_Model.set_param_hint('phase_1', value=1*np.pi/2.)
        # , min = -2*np.pi, max = 2* np.pi)
        Double_Cos_Model.set_param_hint('phase_2', value=-3*np.pi/2.)
        Double_Cos_Model.set_param_hint(
            'amp_1', value=0.25, min=0.1, max=0.4, vary=True)
        Double_Cos_Model.set_param_hint(
            'amp_2', value=0.25, min=0.1, max=0.4, vary=True)
        # Double_Cos_Model.set_param_hint('amp_2',expr='0.5-amp_1', vary=False)
        Double_Cos_Model.set_param_hint('osc_offset', value=0.5, min=0, max=1)
        params = Double_Cos_Model.make_params()
        # if self.fitparams_guess:
        #     for key, val in self.fitparams_guess.items():
        #         if key in params:
        #             params[key].value = val
        fit_res = Double_Cos_Model.fit(data=measured_values,
                                       t=sweep_values,
                                       params=params)
        self.fit_plot = fit_res.model.func(sweep_values, **fit_res.best_values)
        return fit_res

    def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
                 fig_tight=True, **kw):
        plot_formats = kw.pop('plot_formats', ['png'])
        fail_counter = False
        close_fig = kw.pop('close_fig', True)
        if type(plot_formats) == str:
            plot_formats = [plot_formats]
        for plot_format in plot_formats:
            if figname is None:
                figname = (self.scan_start+'_DoubleFreq_'+'.'+plot_format)
            else:
                figname = (figname+'.' + plot_format)
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


class SWAPN_cost(object):

    def __init__(self, auto=True, label='SWAPN', cost_func='sum', timestamp=None, stepsize=10):
        if timestamp is None:
            self.folder = a_tools.latest_data(label)
            splitted = self.folder.split('\\')
            self.scan_start = splitted[-2]+'_'+splitted[-1][:6]
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
                np.power(y[:-4], np.divide(1, x[:-4])))/float(len(y[:-4]))
        elif self.cost_func == 'slope':
            self.cost_val = abs(y[0]*(y[1]-y[0]))+abs(y[0])
        elif self.cost_func == 'dumb-sum':
            self.cost_val = (np.sum(y[:-4])/float(len(y[:-4])))-y[:-4].min()
        elif self.cost_func == 'until-nonmono-sum':
            i = 0
            y_fil = deepcopy(y)
            lastval = y_fil[0]
            keep_going = 1
            while(keep_going):
                if i > 5:
                    latestthreevals = (y_fil[i]+y_fil[i-1] + y_fil[i-2])/3
                    threevalsbefore = (y_fil[i-3]+y_fil[i-4] + y_fil[i-5])/3
                    if latestthreevals < (threevalsbefore-0.12) or i > len(y_fil)-4:
                        keep_going = 0
                i += 1
            y_fil[i-1:-4] = threevalsbefore
            self.cost_val = (np.sum(y_fil[:-4])/float(len(y_fil[:-4])))
        self.single_swap_fid = y[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plot_x = x
        plot_step = plot_x[1]-plot_x[0]

        ax.set_xlabel(r'# Swap pulses')
        ax.set_ylabel(r'$F |1\rangle$')
        ax.set_title('%s: SWAPN sequence' % self.scan_start)
        ax.set_xlim(plot_x.min()-plot_step/2., plot_x.max()+plot_step/2.)

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
                figname = (self.scan_start+'_DoubleFreq_'+'.'+plot_format)
            else:
                figname = (figname+'.' + plot_format)
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


class Tomo_Multiplexed(object):

    def __init__(self, auto=True, label='', timestamp=None,
                 MLE=False, target_cardinal=None, target_bell=None,
                 start_shot=0, end_shot=-1,
                 verbose=0,
                 fig_format='png', q0_label='q0', q1_label='q1', close_fig=True):
        self.label = label
        self.timestamp = timestamp
        self.target_cardinal = target_cardinal
        self.target_bell = target_bell
        self.start_shot = start_shot
        self.end_shot = end_shot
        self.MLE = MLE
        self.verbose = verbose
        self.fig_format = fig_format
        self.q0_label = q0_label
        self.q1_label = q1_label
        self.close_fig = close_fig
        if auto is True:
            self.run_default_analysis()

    def run_default_analysis(self):
        # hard coded number of segments for a 2 qubit state tomography
        # constraint imposed by UHFLI
        a = MeasurementAnalysis(auto=False, label=self.label,
                                timestamp=self.timestamp)
        a.get_naming_and_values()
        self.t_stamp = a.timestamp_string
        self.savefolder = a.folder
        self.nr_segments = 64

        self.shots_q0 = np.zeros(
            (self.nr_segments, int(len(a.measured_values[0])/self.nr_segments)))
        self.shots_q1 = np.zeros(
            (self.nr_segments, int(len(a.measured_values[1])/self.nr_segments)))
        for i in range(self.nr_segments):
            self.shots_q0[i, :] = a.measured_values[0][i::self.nr_segments]
            self.shots_q1[i, :] = a.measured_values[1][i::self.nr_segments]

        # Get correlations between shots
        self.shots_q0q1 = np.multiply(self.shots_q1, self.shots_q0)

        if self.start_shot != 0 or self.end_shot != -1:
            self.shots_q0 = self.shots_q0[:, self.start_shot:self.end_shot]
            self.shots_q1 = self.shots_q1[:, self.start_shot:self.end_shot]
            self.shots_q0q1 = self.shots_q0q1[:, self.start_shot:self.end_shot]
        ##########################################
        # Making  the first figure, tomo shots
        ##########################################
        self.plot_TV_mode()

        avg_h1 = np.mean(self.shots_q0, axis=1)

        # Binning all the points required for the tomo
        h1_00 = np.mean(avg_h1[36:36+7])
        h1_01 = np.mean(avg_h1[43:43+7])
        h1_10 = np.mean(avg_h1[50:50+7])
        h1_11 = np.mean(avg_h1[57:])

        avg_h2 = np.mean(self.shots_q1, axis=1)
        h2_00 = np.mean(avg_h2[36:36+7])
        h2_01 = np.mean(avg_h2[43:43+7])
        h2_10 = np.mean(avg_h2[50:50+7])
        h2_11 = np.mean(avg_h2[57:])

        avg_h12 = np.mean(self.shots_q0q1, axis=1)
        h12_00 = np.mean(avg_h12[36:36+7])
        h12_01 = np.mean(avg_h12[43:43+7])
        h12_10 = np.mean(avg_h12[50:50+7])
        h12_11 = np.mean(avg_h12[57:])

        avg_h12 = np.mean(self.shots_q0q1, axis=1)

        #############################
        # Linear inversion tomo #
        #############################

        measurements_tomo = (
            np.array([avg_h1[0:36], avg_h2[0:36], avg_h12[0:36]])).flatten()  # 108 x 1
        # get the calibration points by averaging over the five measurements taken
        # knowing the initial state we put in
        measurements_cal = np.array(
            [h1_00, h1_01, h1_10, h1_11,
             h2_00, h2_01, h2_10, h2_11,
             h12_00, h12_01, h12_10, h12_11])

        # before we calculate the tomo we need to set the correct order of the
        # rotation matrixes
        tomo_mod.TomoAnalysis_JointRO.rotation_matrixes = [
            qtp.identity(2),
            qtp.sigmax(),
            qtp.rotation(qtp.sigmay(), np.pi / 2),
            qtp.rotation(qtp.sigmay(), -np.pi / 2),
            qtp.rotation(qtp.sigmax(), np.pi / 2),
            qtp.rotation(qtp.sigmax(), -np.pi / 2)]

        # calculate the tomo
        tomo = tomo_mod.TomoAnalysis_JointRO(
            measurements_cal, measurements_tomo, n_qubits=2, n_quadratures=3)
        # operators are expectation values of Pauli operators, rho is density
        # mat
        (self.operators, self.rho) = tomo.execute_linear_tomo()

        self.plot_LI()

        if self.MLE:
            # mle reconstruction of density matrix
            self.rho_2 = tomo.execute_max_likelihood(
                ftol=0.000001, xtol=0.0001)
            # reconstructing the pauli vector
            if self.verbose > 1:
                print(self.rho_2)
            if self.verbose > 0:
                print('Purity %.3f' % (self.rho_2*self.rho_2).tr())
            # calculates the Pauli operator expectation values based on the
            # matrix
            self.operators_mle = tomo_mod.pauli_ops_from_density_matrix(
                self.rho_2)
            if self.verbose > 0:
                print(self.operators_mle)

            self.plot_MLE()
        ########################
        # FIT PHASE CORRECTIONS
        ########################
        if self.MLE:
            self.operators_fit = self.operators_mle
        else:
            self.operators_fit = self.operators

        def rotated_bell_state(dummy_x, angle_MSQ, angle_LSQ, contrast):
            # only works for target_bell=0 for now.
            # to expand, need to figure out the signs in the elements.
            # order is set by looping I,Z,X,Y
            # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            # II IZ IX IY ZI ZZ ZX ZY XI XZ XX XY YI YZ YX YY
            state = np.zeros(16)
            state[0] = 1.
            state[5] = np.cos(angle_LSQ)*contrast
            state[7] = np.sin(angle_LSQ)*contrast
            state[10] = -np.cos(angle_MSQ)*contrast
            state[11] = np.sin(angle_MSQ)*contrast
            state[13] = -np.sin(angle_LSQ)*contrast
            state[14] = np.sin(angle_MSQ)*contrast
            state[15] = np.cos(angle_LSQ)*np.cos(angle_LSQ)*contrast
            return state

        angles_model = lmfit.Model(rotated_bell_state)

        angles_model.set_param_hint(
            'angle_MSQ', value=0., min=-np.pi, max=np.pi, vary=True)
        angles_model.set_param_hint(
            'angle_LSQ', value=0., min=-np.pi, max=np.pi, vary=True)
        angles_model.set_param_hint(
            'contrast', value=1., min=0., max=1., vary=False)
        params = angles_model.make_params()

        self.fit_res = angles_model.fit(data=self.operators_fit,
                                        dummy_x=np.arange(
                                            len(self.operators_fit)),
                                        params=params)
        self.plot_phase_corr()

    def plot_TV_mode(self):
        self.exp_name = os.path.split(self.savefolder)[-1][7:]
        figname = 'Tomography_shots_Exp_{}.{}'.format(self.exp_name,
                                                      self.fig_format)
        fig1, axs = plt.subplots(1, 3, figsize=(17, 4))
        fig1.suptitle(self.exp_name+' ' + self.t_stamp, size=16)
        ax = axs[0]
        ax.plot(np.arange(self.nr_segments), np.mean(self.shots_q0, axis=1),
                'o-')
        ax.set_title('{}'.format(self.q0_label))
        ax = axs[1]
        ax.plot(np.arange(self.nr_segments), np.mean(self.shots_q1, axis=1),
                'o-')
        ax.set_title('{}'.format(self.q1_label))
        ax = axs[2]
        ax.plot(np.arange(self.nr_segments), np.mean(self.shots_q0q1, axis=1),
                'o-')
        ax.set_title('Correlations {}-{}'.format(self.q0_label, self.q1_label))
        savename = os.path.abspath(os.path.join(
            self.savefolder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig1.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig1)

    def plot_LI(self):
        # Making  the second figure, LI tomo
        fig2 = plt.figure(figsize=(15, 5))
        ax = fig2.add_subplot(121)
        if self.target_cardinal is not None:
            fidelity = tomo_mod.calc_fid2_cardinal(self.operators,
                                                   self.target_cardinal)
            target_expectations = tomo_mod.get_cardianal_pauli_exp(
                self.target_cardinal)
            tomo_mod.plot_target_pauli_set(target_expectations, ax)

        if self.target_bell is not None:
            fidelity = tomo_mod.calc_fid2_bell(
                self.operators, self.target_bell)
            target_expectations = tomo_mod.get_bell_pauli_exp(self.target_bell)
            tomo_mod.plot_target_pauli_set(target_expectations, ax)
            txt_x_pos = 0
        else:
            txt_x_pos = 10

        tomo_mod.plot_operators(self.operators, ax)
        ax.set_title('Least squares tomography.')
        if self.verbose > 0:
            print(self.rho)
        qtp.matrix_histogram_complex(self.rho, xlabels=['00', '01', '10', '11'],
                                     ylabels=['00', '01', '10', '11'],
                                     fig=fig2, ax=fig2.add_subplot(
            122, projection='3d'))
        purity = (self.rho*self.rho).tr()
        msg = 'Purity: {:.3f}\nFidelity to target {:.3f}'.format(
            purity, fidelity)
        if self.target_bell is not None:
            theta_vec = np.linspace(0., 2*np.pi, 1001)
            fid_vec = np.zeros(theta_vec.shape)
            for i, theta in enumerate(theta_vec):
                fid_vec[i] = tomo_mod.calc_fid2_bell(self.operators,
                                                     self.target_bell, theta)
            msg += '\nMAX Fidelity {:.3f} at {:.1f} deg'.format(
                np.max(fid_vec),
                theta_vec[np.argmax(fid_vec)]*180./np.pi)
        ax.text(0, .7, msg)

        figname = 'LI-Tomography_Exp_{}.{}'.format(self.exp_name,
                                                   self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.t_stamp, size=16)
        savename = os.path.abspath(os.path.join(
            self.savefolder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)

    def plot_MLE(self):
        # Figure 3 MLE reconstruction
        fig3 = plt.figure(figsize=(15, 5))
        ax = fig3.add_subplot(121)

        if self.target_cardinal is not None:
            fidelity_mle = tomo_mod.calc_fid2_cardinal(self.operators_mle,
                                                       self.target_cardinal)
            target_expectations = tomo_mod.get_cardianal_pauli_exp(
                self.target_cardinal)
            tomo_mod.plot_target_pauli_set(target_expectations, ax)
        if self.target_bell is not None:
            fidelity_mle = tomo_mod.calc_fid2_bell(self.operators_mle,
                                                   self.target_bell)
            target_expectations = tomo_mod.get_bell_pauli_exp(self.target_bell)
            tomo_mod.plot_target_pauli_set(target_expectations, ax)
            txt_x_pos = 0
        else:
            txt_x_pos = 10

        purity = (self.rho_2*self.rho_2).tr()

        msg = 'Purity: {:.3f}\nFidelity to target {:.3f}'.format(
            purity, fidelity_mle)
        if self.target_bell is not None:
            theta_vec = np.linspace(0., 2*np.pi, 1001)
            fid_vec = np.zeros(theta_vec.shape)
            for i, theta in enumerate(theta_vec):
                fid_vec[i] = tomo_mod.calc_fid2_bell(self.operators_mle,
                                                     self.target_bell, theta)
            msg += '\nMAX Fidelity {:.3f} at {:.1f} deg'.format(
                np.max(fid_vec),
                theta_vec[np.argmax(fid_vec)]*180./np.pi)
        ax.text(txt_x_pos, .7, msg)

        tomo_mod.plot_operators(self.operators_mle, ax)
        ax.set_title('Max likelihood estimation tomography')
        qtp.matrix_histogram_complex(self.rho_2, xlabels=['00', '01', '10', '11'],
                                     ylabels=['00', '01', '10', '11'],
                                     fig=fig3,
                                     ax=fig3.add_subplot(122, projection='3d'))

        figname = 'MLE-Tomography_Exp_{}.{}'.format(self.exp_name,
                                                    self.fig_format)
        fig3.suptitle(self.exp_name+' ' + self.t_stamp, size=16)
        savename = os.path.abspath(os.path.join(
            self.savefolder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig3.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig3)

    def plot_phase_corr(self):
        fig2 = plt.figure(figsize=(15, 5))
        ax = fig2.add_subplot(111)
        ordered_fit = np.concatenate(
            ([1], np.concatenate(tomo_mod.order_pauli_output2(self.fit_res.best_fit))))
        tomo_mod.plot_target_pauli_set(ordered_fit, ax)
        tomo_mod.plot_operators(self.operators_fit, ax=ax)
        fidelity = np.dot(self.fit_res.best_fit, self.operators_fit)*0.25
        self.best_fidelity = fidelity
        angle_LSQ_deg = self.fit_res.best_values['angle_LSQ']*180./np.pi
        angle_MSQ_deg = self.fit_res.best_values['angle_MSQ']*180./np.pi
        ax.set_title('Fit of single qubit phase errors')
        msg = r'MAX Fidelity at %.3f $\phi_{MSQ}=$%.1f deg and $\phi_{LSQ}=$%.1f deg' % (
            fidelity,
            angle_LSQ_deg,
            angle_MSQ_deg)
        msg += "\n Chi sqr. %.3f" % self.fit_res.chisqr
        ax.text(0.5, .7, msg)
        figname = 'Fit_report_{}.{}'.format(self.exp_name,
                                            self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.t_stamp, size=16)
        savename = os.path.abspath(os.path.join(
            self.savefolder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)
        angle_LSQ_deg = self.fit_res.best_values['angle_LSQ']*180./np.pi
        angle_MSQ_deg = self.fit_res.best_values['angle_MSQ']*180./np.pi
        ax.set_title('Fit of single qubit phase errors')
        msg = r'MAX Fidelity at %.3f $\phi_{MSQ}=$%.1f deg and $\phi_{LSQ}=$%.1f deg' % (
            fidelity,
            angle_LSQ_deg,
            angle_MSQ_deg)
        msg += "\n Chi sqr. %.3f" % self.fit_res.chisqr
        ax.text(0.5, .7, msg)
        figname = 'Fit_report_{}.{}'.format(self.exp_name,
                                            self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.t_stamp, size=16)
        savename = os.path.abspath(os.path.join(
            self.savefolder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)
