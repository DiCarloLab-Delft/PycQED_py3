"""
File containing the BaseDataAnalyis class.
"""
import os
import numpy as np
import copy
from collections import OrderedDict

import numbers
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.general import NumpyJsonEncoder
from pycqed.analysis.analysis_toolbox import get_color_order as gco
from pycqed.analysis.analysis_toolbox import get_color_list
from pycqed.analysis.tools.plotting import (
    set_axis_label, flex_colormesh_plot_vs_xy, flex_color_plot_vs_x)
import pycqed.analysis_v2.default_figure_settings_analysis as def_fig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import json
import lmfit
import h5py
from pycqed.measurement.hdf5_data import write_dict_to_hdf5

class BaseDataAnalysis(object):
    """
    Abstract Base Class (not intended to be instantiated directly) for
    analysis.

    Children inheriting from this method should specify the following methods
        - __init__      -> specify params to be extracted, set options
                           specific to analysis and call run_analysis method.
        - process_data  -> mundane tasks such as binning and filtering
        - prepare_plots -> specify default plots and set up plotting dicts
        - run_fitting   -> perform fits to data

    The core of this class is the flow defined in run_analysis and should
    be called at the end of the __init__. This executes
    the following code:

        self.extract_data()    # extract data specified in params dict
        self.process_data()    # binning, filtering etc
        if self.do_fitting:
            self.run_fitting() # fitting to models
        self.prepare_plots()   # specify default plots
        if not self.extract_only:
            self.plot(key_list='auto')  # make the plots
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=False):
        '''
        This is the __init__ of the abstract base class.
        It is intended to be called at the start of the init of the child
        classes followed by "run_analysis".

        __init__ of the child classes:
            The __init__ of child classes  should implement the following
            functionality:
                - call the ASB __init__ (this method)
                - define self.params_dict and self.numeric_params
                - specify options specific to that analysis
                - call self.run_analysis

        This method sets several attributes of the analysis class.
        These include assigning the arguments of this function to attributes.
        Other arguments that get created are
            axs (dict)
            figs (dict)
            plot_dicts (dict)

        and a bunch of stuff specified in the options dict
        (TODO: should this not always be extracted from the
        dict to prevent double refs? )

        There are several ways to specify where the data should be loaded
        from.
            data_file_path: directly give the file path of a data file that
                should be loaded.
            t_start, t_stop: give a range of timestamps in where data is
                loaded from. Filtering options can be given through the
                options dictionary. If t_stop is omitted, the extraction
                routine looks for the data with time stamp t_start.
            none of the above: look for the last data which matches the
                filtering options from the options dictionary.
        Note: data_file_path has priority, i.e. if this argument is given
        time stamps are ignored.
        '''
        self.single_timestamp = False
        # initialize an empty dict to store results of analysis
        self.proc_data_dict = OrderedDict()
        if options_dict is None:
            self.options_dict = OrderedDict()
        else:
            self.options_dict = options_dict

        self.ma_type = self.options_dict.get('ma_type', 'MeasurementAnalysis')

        ################################################
        # These options determine what data to extract #
        ################################################
        scan_label = self.options_dict.get('scan_label', label)
        if type(scan_label) is not list:
            self.labels = [scan_label]
        else:
            self.labels = scan_label

        # Initialize to None such that the attribute always exists.
        self.data_file_path = None
        if t_start is None and t_stop is None and data_file_path is None:
            # Nothing specified -> find last file with label
            # This is quite a hacky way to support finding the last file
            # with a certain label, something that was trivial in the old
            # analysis. A better solution should be implemented.
            self.t_start = a_tools.latest_data(scan_label,
                                               return_timestamp=True)[0]
        elif data_file_path is not None:
            # Data file path specified ignore timestamps
            self.extract_from_file = True
            self.t_start = None
            self.data_file_path = data_file_path
        elif t_start is not None:
            # No data file specified -> use timestamps
            self.t_start = t_start
        else:
            raise ValueError('Either t_start or data_file_path must be '
                             'given.')

        if t_stop is None:
            self.t_stop = self.t_start
        else:
            self.t_stop = t_stop
        self.do_timestamp_blocks = self.options_dict.get('do_blocks', False)
        self.do_individual_traces = self.options_dict.get(
            'do_individual_traces', False)
        self.filter_no_analysis = self.options_dict.get(
            'filter_no_analysis', False)
        self.exact_label_match = self.options_dict.get(
            'exact_label_match', False)

        ########################################
        # These options relate to the plotting #
        ########################################
        self.plot_dicts = OrderedDict()
        self.axs = OrderedDict()
        self.figs = OrderedDict()
        self.presentation_mode = self.options_dict.get(
            'presentation_mode', False)
        self.tight_fig = self.options_dict.get('tight_fig', True)
        # used in self.plot_text, here for future compatibility
        self.fancy_box_props = dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5)

        self.options_dict['plot_init'] = self.options_dict.get('plot_init',
                                                               False)
        self.options_dict['save_figs'] = self.options_dict.get(
            'save_figs', True)
        self.options_dict['close_figs'] = self.options_dict.get(
            'close_figs', True)
        ####################################################
        # These options relate to what analysis to perform #
        ####################################################
        self.extract_only = extract_only
        self.do_fitting = do_fitting

        self.verbose = self.options_dict.get('verbose', False)
        self.auto_keys = self.options_dict.get('auto_keys', None)

        if type(self.auto_keys) is str:
            self.auto_keys = [self.auto_keys]

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.
        This function is typically called after the __init__.
        """
        self.extract_data()    # extract data specified in params dict
        self.process_data()    # binning, filtering etc
        if self.do_fitting:
            self.prepare_fitting()      # set up fit_dicts
            self.run_fitting()          # fitting to models
            self.save_fit_results()
            self.analyze_fit_results()  # analyzing the results of the fits

        self.prepare_plots()   # specify default plots
        if not self.extract_only:
            self.plot(key_list='auto')  # make the plots
            self.save_figures(close_figs=self.options_dict['close_figs'])

    def get_timestamps(self):
        """
        Extracts timestamps based on variables
            self.t_start
            self.t_stop
            self.labels # currently set from options dict
        """
        if type(self.t_start) is list:
            if (type(self.t_stop) is list and
                    len(self.t_stop) == len(self.t_start)):
                self.timestamps = []
                for tt in range(len(self.t_start)):
                    if self.do_timestamp_blocks:
                        self.timestamps.append(
                            a_tools.get_timestamps_in_range(
                                self.t_start[tt], self.t_stop[tt],
                                label=self.labels,
                                exact_label_match=self.exact_label_match))
                    else:
                        self.timestamps.extend(
                            a_tools.get_timestamps_in_range(
                                self.t_start[tt], self.t_stop[tt],
                                label=self.labels,
                                exact_label_match=self.exact_label_match))
            else:
                if self.do_timestamp_blocks:
                    self.do_timestamp_blocks = False
                raise ValueError("Invalid number of t_stop timestamps.")
        else:
            self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels,
                exact_label_match=self.exact_label_match)

        if self.verbose:
            print(len(self.timestamps), type(self.timestamps[0]))

        if len(np.ravel(self.timestamps)) < 1:
            raise ValueError(
                "No timestamps in range! Check the labels and other filters.")

    def extract_data(self):
        """
        Extracts the data specified in
            self.params_dict
            self.numeric_params
        and stores it into: self.raw_data_dict

        Data extraction is now supported in three different ways
            - using json
            - using timestamp blocks
            - single timestamp only
        """

        if self.data_file_path is not None:
            extension = self.data_file_path.split('.')[-1]
            if extension == 'json':
                with open(self.data_file_path, 'r') as file:
                    self.raw_data_dict = json.load(file)
            else:
                raise RuntimeError('Cannot load data from file "{}". '
                                   'Unknown file extension "{}"'
                                   .format(self.data_file_path, extension))
            return

        self.get_timestamps()
        # this disables the data extraction for other files if there is only
        # one file being used to load data from
        if self.single_timestamp:
            self.timestamps = [self.timestamps[0]]
        TwoD = self.options_dict.get('TwoD', False)
        # this should always be extracted as it is used to determine where
        # the file is as required for datasaving
        self.params_dict['folder'] = 'folder'

        if self.do_timestamp_blocks:
            self.raw_data_dict = {key: []
                                  for key in list(self.params_dict.keys())}
            self.raw_data_dict['timestamps'] = []
            for tstamps in self.timestamps:
                if self.verbose:
                    print(len(tstamps), type(tstamps))
                temp_dict = a_tools.get_data_from_timestamp_list(
                    tstamps, param_names=self.params_dict,
                    ma_type=self.ma_type,
                    TwoD=TwoD, numeric_params=self.numeric_params,
                    filter_no_analysis=self.filter_no_analysis)
                for key in self.raw_data_dict:
                    self.raw_data_dict[key].append(temp_dict[key])

            # Use timestamps to calculate datetimes and add to dictionary
            self.raw_data_dict['datetime'] = []
            for tstamps in self.raw_data_dict['timestamps']:
                self.raw_data_dict['datetime'].append(
                    [a_tools.datetime_from_timestamp(ts) for ts in tstamps])

            # Convert temperature data to dictionary form and extract Tmc
            # N.B. This has the hardcoded temperature names from pycqed_py2
            if 'temperatures' in self.raw_data_dict:
                self.raw_data_dict['Tmc'] = []
                for ii, temperatures in enumerate(
                        self.raw_data_dict['temperatures']):
                    temp = []
                    self.raw_data_dict['Tmc'].append([])
                    for ii in range(len(temperatures)):
                        exec("temp.append(%s)" % (temperatures[ii]))
                        self.raw_data_dict['Tmc'][ii].append(
                            temp[ii].get('T_MClo', None))
                    self.raw_data_dict['temperatures'][ii] = temp

        else:
            self.raw_data_dict = a_tools.get_data_from_timestamp_list(
                self.timestamps, param_names=self.params_dict,
                ma_type=self.ma_type,
                TwoD=TwoD, numeric_params=self.numeric_params,
                filter_no_analysis=self.filter_no_analysis)

            # Use timestamps to calculate datetimes and add to dictionary
            self.raw_data_dict['datetime'] = [a_tools.datetime_from_timestamp(
                timestamp) for timestamp in self.raw_data_dict['timestamps']]

            # Convert temperature data to dictionary form and extract Tmc
            if 'temperatures' in self.raw_data_dict:
                temp = []
                self.raw_data_dict['Tmc'] = []
                for ii in range(len(self.raw_data_dict['temperatures'])):
                    exec("temp.append(%s)" %
                         (self.raw_data_dict['temperatures'][ii]))
                    self.raw_data_dict['Tmc'].append(
                        temp[ii].get('T_MClo', None))
                self.raw_data_dict['temperatures'] = temp

        # this is a hacky way to use the same data extraction when there is
        # many files as when there is few files.
        if self.single_timestamp:
            new_dict = {}
            for key, value in self.raw_data_dict.items():
                if key != 'timestamps':
                    new_dict[key] = value[0]
            self.raw_data_dict = new_dict
            self.raw_data_dict['timestamp'] = self.timestamps[0]
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['nr_experiments'] = len(self.timestamps)

        # Converts a multi file 'measured_values' dict to an ordered dict
        # from which values can be easily extracted
        if ('measured_values' in self.raw_data_dict and
                'value_names' in self.raw_data_dict and
                not self.single_timestamp):
            # the not self.single_timestamp is there for legacy reasons
            measured_values_dict = OrderedDict()
            for key in (self.raw_data_dict['value_names'][0]):
                measured_values_dict[key] = []
            for dset in self.raw_data_dict['measured_values']:
                for i, col in enumerate(dset):
                    measured_values_dict[self.raw_data_dict[
                        'value_names'][0][i]].append(col)
            self.raw_data_dict[
                'measured_values_ord_dict'] = measured_values_dict

    def process_data(self):
        """
        process_data: overloaded in child classes,
        takes care of mundane tasks such as binning filtering etc
        """
        pass

    def prepare_plots(self):
        """
        Defines a default plot by setting up the plotting dictionaries to
        specify what is to be plotted
        """
        pass

    def analyze_fit_results(self):
        """
        Do analysis on the results of the fits to extract quantities of
        interest.
        """
        pass

    def save_figures(self, savedir: str=None, savebase: str =None,
                     tag_tstamp: bool=True,
                     fmt: str ='png', key_list: list='auto',
                     close_figs: bool=True):
        if savedir is None:
            savedir = self.raw_data_dict.get('folder', '')
            if isinstance(savedir, list):
                savedir = savedir[0]
            if isinstance(savedir, list):
                savedir = savedir[0]
        if savebase is None:
            savebase = ''
        if tag_tstamp:
            tstag = '_'+self.raw_data_dict['timestamps'][0]
        else:
            tstag = ''

        if key_list == 'auto' or key_list is None:
            key_list = self.figs.keys()
        for key in key_list:
            if self.presentation_mode:
                savename = os.path.join(savedir, savebase+key+tstag+'presentation'+'.'+fmt)
                self.axs[key].figure.savefig(savename, bbox_inches='tight', fmt=fmt)
                savename = os.path.join(savedir, savebase+key+tstag+'presentation'+'.svg')
                self.axs[key].figure.savefig(savename, bbox_inches='tight', fmt='svg')
            else:
                savename = os.path.join(savedir, savebase+key+tstag+'.'+fmt)
                self.axs[key].figure.savefig(savename, bbox_inches='tight', fmt=fmt)
            if close_figs:
                plt.close(self.axs[key].figure)

    def save_data(self, savedir: str=None, savebase: str=None,
                  tag_tstamp: bool=True,
                  fmt: str='json', key_list='auto'):
        '''
        Saves the data from self.raw_data_dict to file.

        Args:
            savedir (string):
                    Directory where the file is saved. If this is None, the
                    file is saved in self.raw_data_dict['folder'] or the
                    working directory of the console.
            savebase (string):
                    Base name for the saved file.
            tag_tstamp (bool):
                    Whether to append the timestamp of the first to the base
                    name.
            fmt (string):
                    File extension for the format in which the file should
                    be saved.
            key_list (list or 'auto'):
                    Specifies which keys from self.raw_data_dict are saved.
                    If this is 'auto' or None, all keys-value pairs are
                    saved.
        '''
        if savedir is None:
            savedir = self.raw_data_dict.get('folder', '')
            if isinstance(savedir, list):
                savedir = savedir[0]
        if savebase is None:
            savebase = ''
        if tag_tstamp:
            tstag = '_'+self.raw_data_dict['timestamps'][0]
        else:
            tstag = ''

        if key_list == 'auto' or key_list is None:
            key_list = self.raw_data_dict.keys()

        save_dict = {}
        for k in key_list:
            save_dict[k] = self.raw_data_dict[k]

        filepath = os.path.join(savedir, savebase + tstag + '.' + fmt)
        with open(filepath, 'w') as file:
            json.dump(save_dict, file, cls=NumpyJsonEncoder, indent=4)
        print('Data saved to "{}".'.format(filepath))

    def prepare_fitting(self):
        # initialize everything to an empty dict if not overwritten
        self.fit_dicts = OrderedDict()

    def run_fitting(self):
        '''
        This function does the fitting and saving of the parameters
        based on the fit_dict options.
        Only model fitting is implemented here. Minimizing fitting should
        be implemented here.
        '''
        self.fit_res = {}
        for key, fit_dict in self.fit_dicts.items():
            guess_dict = fit_dict.get('guess_dict', None)
            guess_pars = fit_dict.get('guess_pars', None)
            fit_yvals = fit_dict['fit_yvals']
            fit_xvals = fit_dict['fit_xvals']

            model = fit_dict.get('model', None)
            if model is None:
                fit_fn = fit_dict.get('fit_fn', None)
                model = fit_dict.get('model', lmfit.Model(fit_fn))
            fit_guess_fn = fit_dict.get('fit_guess_fn', None)
            if fit_guess_fn is None:
                fit_guess_fn = model.guess

            if self.do_timestamp_blocks:
                fit_dict['fit_res'] = []
                for tt in range(len(fit_yvals)):
                    if guess_pars is None:
                        if guess_dict is None:
                            guess_dict = fit_guess_fn(**fit_yvals[tt],
                                                      **fit_xvals[tt])
                        for key, val in list(guess_dict.items()):
                            model.set_param_hint(key, **val)
                        guess_pars = model.make_params()
                    fit_dict['fit_res'].append(
                        model.fit(params=guess_pars, **fit_xvals[tt],
                                  **fit_yvals[tt]))
            else:
                if guess_pars is None:
                    if fit_guess_fn is not None:
                        # a fit function should return lmfit parameter objects
                        # but can also work by returning a dictionary of guesses
                        guess_pars = fit_guess_fn(**fit_yvals, **fit_xvals)
                        if not isinstance(guess_pars, lmfit.Parameters):
                            for gd_key, val in list(guess_pars.items()):
                                model.set_param_hint(gd_key, **val)
                            guess_pars = model.make_params()

                        if guess_dict is not None:
                            for gd_key, val in guess_dict.items():
                                for attr, attr_val in val.items():
                                    # e.g. setattr(guess_pars['frequency'], 'value', 20e6)
                                    setattr(guess_pars[gd_key], attr, attr_val)
                        # A guess can also be specified as a dictionary.
                        # additionally this can be used to overwrite values
                        # from the guess functions.
                    else:
                        for key, val in list(guess_dict.items()):
                            model.set_param_hint(key, **val)
                        guess_pars = model.make_params()

                fit_dict['fit_res'] = model.fit(
                    params=guess_pars, **fit_xvals, **fit_yvals)

            self.fit_res[key] = fit_dict['fit_res']

    def save_fit_results(self):
        """
        Saves the fit results
        """
        fn = a_tools.measurement_filename(a_tools.get_folder(self.timestamps[0]))
        with h5py.File(fn, 'r+') as data_file:
            try:
                analysis_group = data_file.create_group('Analysis')
            except ValueError:
                # If the analysis group already exists.
                # Delete the old group and create a new group (overwrite).
                del data_file['Analysis']
                analysis_group = data_file.create_group('Analysis')

            # Iterate over all the fit result dicts
            for fr_key, fit_res in self.fit_res.items():
                fr_group = analysis_group.create_group(fr_key)
                # TODO: convert the params object to a simple dict
                # write_dict_to_hdf5(fit_res.params, entry_point=fr_group)
                write_dict_to_hdf5(fit_res.best_values, entry_point=fr_group)


    def plot(self, key_list=None, axs_dict=None,
             presentation_mode=None, no_label=False):
        """
        Goes over the plots defined in the plot_dicts and creates the
        desired figures.
        """
        if presentation_mode is None:
            presentation_mode = self.presentation_mode
        if axs_dict is not None:
            for key, val in list(axs_dict.items()):
                self.axs[key] = val
        if key_list is 'auto':
            key_list = self.auto_keys
        if key_list is None:
            key_list = self.plot_dicts.keys()
        if type(key_list) is str:
            key_list = [key_list]
        self.key_list = key_list

        for key in key_list:
            # go over all the plot_dicts
            pdict = self.plot_dicts[key]
            pdict['no_label'] = no_label
            # Use the key of the plot_dict if no ax_id is specified
            pdict['ax_id'] = pdict.get('ax_id', key)

            if pdict['ax_id'] not in self.axs:
                # This fig variable should perhaps be a different
                # variable for each plot!!
                # This might fix a bug.
                self.figs[pdict['ax_id']], self.axs[pdict['ax_id']] = plt.subplots(
                    pdict.get('numplotsy', 1), pdict.get('numplotsx', 1),
                    sharex=pdict.get('sharex', False),
                    sharey=pdict.get('sharey', False),
                    figsize=pdict.get('plotsize', None))  # (8, 6)))
                if pdict.get('3d', False):
                    self.axs[pdict['ax_id']].remove()
                    self.axs[pdict['ax_id']] = Axes3D(
                        self.figs[pdict['ax_id']],
                        azim=pdict.get('3d_azim', -35),
                        elev=pdict.get('3d_elev', 35))
                    self.axs[pdict['ax_id']].patch.set_alpha(0)

                # transparent background around axes for presenting data
                self.figs[pdict['ax_id']].patch.set_alpha(0)

        if presentation_mode:
            self.plot_for_presentation(key_list=key_list, no_label=no_label)
        else:
            for key in key_list:
                pdict = self.plot_dicts[key]
                if type(pdict['plotfn']) is str:
                    plotfn = getattr(self, pdict['plotfn'])
                else:
                    plotfn = pdict['plotfn']
                plotfn(pdict, axs=self.axs[pdict['ax_id']])
            self.format_datetime_xaxes(key_list)
            self.add_to_plots(key_list=key_list)

    def add_to_plots(self, key_list=None):
        pass

    def format_datetime_xaxes(self, key_list):
        for key in key_list:
            pdict = self.plot_dicts[key]
            # this check is needed as not all plots have xvals e.g., plot_text
            if 'xvals' in pdict.keys():
                if (type(pdict['xvals'][0]) is datetime.datetime and
                        key in self.axs.keys()):
                    self.axs[key].figure.autofmt_xdate()

    def plot_for_presentation(self, key_list=None, no_label=False):
        if key_list is None:
            key_list = list(self.plot_dicts.keys())
        for key in key_list:
            self.plot_dicts[key]['title'] = None

        self.plot(key_list=key_list, presentation_mode=False,
                  no_label=no_label)

    def plot_bar(self, pdict, axs):
        pfunc = getattr(axs, pdict.get('func', 'bar'))
        # xvals interpreted as edges for a bar plot
        plot_xedges = pdict.get('xvals', None)
        if plot_xedges is None:
            plot_centers = pdict['xcenters']
            plot_xwidth = pdict['xwidth']
        else:
            plot_xwidth = (plot_xedges[1:] - plot_xedges[:-1])
            # center is left edge + width/2
            plot_centers = plot_xedges[:-1] + plot_xwidth / 2
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict.get('xlabel', None)
        plot_ylabel = pdict.get('ylabel', None)
        plot_xunit = pdict.get('xunit', None)
        plot_yunit = pdict.get('yunit', None)
        plot_xtick_loc = pdict.get('xtick_loc', None)
        plot_ytick_loc = pdict.get('ytick_loc', None)
        plot_xtick_labels = pdict.get('xtick_labels', None)
        plot_ytick_labels = pdict.get('ytick_labels', None)
        plot_title = pdict.get('title', None)
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_barkws = pdict.get('bar_kws', {})
        plot_multiple = pdict.get('multiple', False)
        dataset_desc = pdict.get('setdesc', '')
        dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
        do_legend = pdict.get('do_legend', False)

        if plot_multiple:
            p_out = []
            for ii, this_yvals in enumerate(plot_yvals):
                p_out.append(pfunc(plot_centers, this_yvals, width=plot_xwidth,
                                   color=gco(ii, len(plot_yvals)-1),
                                   label='%s%s' % (
                                       dataset_desc, dataset_label[ii]),
                                   **plot_barkws))

        else:
            p_out = pfunc(plot_centers, plot_yvals, width=plot_xwidth,
                          label='%s%s' % (dataset_desc, dataset_label),
                          **plot_barkws)

        if plot_xrange is not None:
            axs.set_xlim(*plot_xrange)
        if plot_yrange is not None:
            axs.set_ylim(*plot_yrange)
        if plot_xlabel is not None:
            set_axis_label('x', axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_axis_label('y', axs, plot_ylabel, plot_yunit)
        if plot_xtick_labels is not None:
            axs.xaxis.set_ticklabels(plot_xtick_labels)
        if plot_ytick_labels is not None:
            axs.yaxis.set_ticklabels(plot_ytick_labels)
        if plot_xtick_loc is not None:
            axs.xaxis.set_ticks(plot_xtick_loc)
        if plot_ytick_loc is not None:
            axs.yaxis.set_ticks(plot_ytick_loc)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_ncol = pdict.get('legend_ncol', 1)
            legend_title = pdict.get('legend_title', None)
            legend_pos = pdict.get('legend_pos', 'best')
            axs.legend(title=legend_title, loc=legend_pos, ncol=legend_ncol)

        if self.tight_fig:
            axs.figure.tight_layout()

        pdict['handles'] = p_out

    def plot_bar3D(self, pdict, axs):
        pfunc = axs.bar3d
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_zvals = pdict['zvals']
        plot_xlabel = pdict.get('xlabel', None)
        plot_ylabel = pdict.get('ylabel', None)
        plot_zlabel = pdict.get('zlabel', None)
        plot_xunit = pdict.get('xunit', None)
        plot_yunit = pdict.get('yunit', None)
        plot_zunit = pdict.get('zunit', None)
        plot_color = pdict.get('color', None)
        plot_colormap = pdict.get('colormap', None)
        plot_title = pdict.get('title', None)
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_zrange = pdict.get('zrange', None)
        plot_barkws = pdict.get('bar_kws', {})
        plot_barwidthx = pdict.get('bar_widthx', None)
        plot_barwidthy = pdict.get('bar_widthy', None)
        plot_xtick_loc = pdict.get('xtick_loc', None)
        plot_ytick_loc = pdict.get('ytick_loc', None)
        plot_xtick_labels = pdict.get('xtick_labels', None)
        plot_ytick_labels = pdict.get('ytick_labels', None)
        do_legend = pdict.get('do_legend', False)

        xpos, ypos = np.meshgrid(plot_xvals, plot_yvals)
        xpos = xpos.T.flatten()
        ypos = ypos.T.flatten()
        zpos = np.zeros_like(xpos)
        if plot_barwidthx is None:
            plot_barwidthx = plot_xvals[1] - plot_xvals[0]
        if not hasattr(plot_barwidthx, '__iter__'):
            plot_barwidthx = np.ones_like(zpos)*plot_barwidthx
        if plot_barwidthy is None:
            plot_barwidthy = plot_yvals[1] - plot_yvals[0]
        if not hasattr(plot_barwidthy, '__iter__'):
            plot_barwidthy = np.ones_like(zpos) * plot_barwidthy
        plot_barheight = plot_zvals.flatten()

        if plot_colormap is not None:
            # plot_color assumed to be floats
            if hasattr(plot_color, '__iter__') and \
                    hasattr(plot_color[0], '__iter__'):
                plot_color = np.array(plot_color).flatten()
            plot_color = plot_colormap(plot_color)
        else:
            # plot_color assumed to be RGBA tuple(s)
            if hasattr(plot_color[0], '__iter__') and \
                    hasattr(plot_color[0][0], '__iter__'):
                plot_color = np.array(plot_color)
                plot_color = plot_color.reshape((-1, plot_color.shape[-1]))
            elif not hasattr(plot_color[0], '__iter__'):
                plot_color = np.array(plot_color)
                n = plot_zvals.size
                plot_color = np.repeat(plot_color, n).reshape(-1, n).T

        zsort = plot_barkws.pop('zsort', 'max')
        p_out = pfunc(xpos - plot_barwidthx/2, ypos - plot_barwidthy/2, zpos,
                      plot_barwidthx, plot_barwidthy, plot_barheight,
                      color=plot_color, zsort=zsort, **plot_barkws)

        if plot_xtick_labels is not None:
            axs.xaxis.set_ticklabels(plot_xtick_labels)
        if plot_ytick_labels is not None:
            axs.yaxis.set_ticklabels(plot_ytick_labels)
        if plot_xtick_loc is not None:
            axs.xaxis.set_ticks(plot_xtick_loc)
        if plot_ytick_loc is not None:
            axs.yaxis.set_ticks(plot_ytick_loc)

        if plot_xrange is not None:
            axs.set_xlim(*plot_xrange)
        if plot_yrange is not None:
            axs.set_ylim(*plot_yrange)
        if plot_zrange is not None:
            axs.set_zlim3d(*plot_zrange)
        if plot_xlabel is not None:
            set_axis_label('x', axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_axis_label('y', axs, plot_ylabel, plot_yunit)
        if plot_zlabel is not None:
            set_axis_label('z', axs, plot_zlabel, plot_zunit)
        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_kws = pdict.get('legend_kws', {})
            legend_entries = pdict.get('legend_entries', [])
            legend_artists = [entry[0] for entry in legend_entries]
            legend_labels = [entry[1] for entry in legend_entries]
            axs.legend(legend_artists, legend_labels, **legend_kws)

        if self.tight_fig:
            axs.figure.tight_layout()

        if pdict.get('colorbar', True) and plot_colormap is not None:
            self.plot_colorbar(axs=axs, pdict=pdict)

        pdict['handles'] = p_out

    def plot_line(self, pdict, axs):
        """
        Basic line plotting function.
        Takes either an x and y array or a list of x and y arrays.
        Detection happens based on types of the data
        """
        pfunc = getattr(axs, pdict.get('func', 'plot'))
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict.get('xlabel', None)
        plot_ylabel = pdict.get('ylabel', None)
        plot_xunit = pdict.get('xunit', None)
        plot_yunit = pdict.get('yunit', None)
        plot_title = pdict.get('title', None)
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_linekws = pdict.get('line_kws', {})

        # plot_multiple = pdict.get('multiple', False)
        plot_linestyle = pdict.get('linestyle', '-')
        plot_marker = pdict.get('marker', 'o')
        dataset_desc = pdict.get('setdesc', '')
        # Fixme, this default creates a nasty bug when not plotting a set of
        # lines.
        dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
        do_legend = pdict.get('do_legend', False)

        # Detect if two arrays/lists of x and yvals are passed or a list
        # of x-arrays and a list of y-arrays
        if (isinstance(plot_xvals[0], numbers.Number) or
                isinstance(plot_xvals[0], datetime.datetime)):
            plot_multiple = False
        else:
            plot_multiple = True
            assert(len(plot_xvals) == len(plot_yvals))
            assert(len(plot_xvals[0]) == len(plot_yvals[0]))

        if plot_multiple:
            p_out = []
            len_color_cycle = pdict.get('len_color_cycle', len(plot_yvals))
            # Default gives max contrast
            cmap = pdict.get('cmap', 'tab10')  # Default matplotlib cycle
            colors = get_color_list(len_color_cycle, cmap)
            if cmap == 'tab10':
                len_color_cycle = min(10, len_color_cycle)

            # plot_*vals is the list of *vals arrays
            for i, (xvals, yvals) in enumerate(zip(plot_xvals, plot_yvals)):
                p_out.append(pfunc(xvals, yvals,
                                   linestyle=plot_linestyle,
                                   marker=plot_marker,
                                   color=colors[i % len_color_cycle],
                                   label='%s%s' % (
                                       dataset_desc, dataset_label[i]),
                                   **plot_linekws))

        else:
            p_out = pfunc(plot_xvals, plot_yvals,
                          linestyle=plot_linestyle, marker=plot_marker,
                          label='%s%s' % (dataset_desc, dataset_label),
                          **plot_linekws)

        if plot_xrange is None:
            pass  # Do not set xlim if xrange is None as the axs gets reused
        else:
            xmin, xmax = plot_xrange
            axs.set_xlim(xmin, xmax)

        if plot_xlabel is not None:
            set_axis_label('x', axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_axis_label('y', axs, plot_ylabel, plot_yunit)
        if plot_yrange is not None:
            ymin, ymax = plot_yrange
            axs.set_ylim(ymin, ymax)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_ncol = pdict.get('legend_ncol', 1)
            legend_title = pdict.get('legend_title', None)
            legend_pos = pdict.get('legend_pos', 'best')
            axs.legend(title=legend_title, loc=legend_pos, ncol=legend_ncol)

        if self.tight_fig:
            axs.figure.tight_layout()

            # Need to set labels again, because tight_layout can screw them up
            if plot_xlabel is not None:
                set_axis_label('x', axs, plot_xlabel, plot_xunit)
            if plot_ylabel is not None:
                set_axis_label('y', axs, plot_ylabel, plot_yunit)

        pdict['handles'] = p_out

    def plot_yslices(self, pdict, axs):
        pfunc = getattr(axs, pdict.get('func', 'plot'))
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_slicevals = pdict['slicevals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_nolabel = pdict.get('no_label', False)
        plot_title = pdict['title']
        slice_idxs = pdict['sliceidxs']
        slice_label = pdict.get('slicelabel', '')
        slice_units = pdict.get('sliceunits', '')
        do_legend = pdict.get('do_legend', True)
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]

        for ii, idx in enumerate(slice_idxs):
            if len(slice_idxs) == 1:
                pfunc(plot_xvals, plot_yvals[idx], '-bo',
                      label='%s = %.2f %s' % (
                    slice_label, plot_slicevals[idx], slice_units))
            else:
                if ii == 0 or ii == len(slice_idxs)-1:
                    pfunc(plot_xvals, plot_yvals[idx], '-o',
                          color=gco(ii, len(slice_idxs)-1),
                          label='%s = %.2f %s' % (
                        slice_label, plot_slicevals[idx], slice_units))
                else:
                    pfunc(plot_xvals, plot_yvals[idx], '-o',
                          color=gco(ii, len(slice_idxs)-1))
        if plot_xrange is None:
            xmin, xmax = np.min(plot_xvals)-plot_xvals_step / \
                2., np.max(plot_xvals)+plot_xvals_step/2.
        else:
            xmin, xmax = plot_xrange
        axs.set_xlim(xmin, xmax)

        if not plot_nolabel:
            axs.set_axis_label('x', plot_xlabel)
            axs.set_axis_label('y', plot_ylabel)

        if plot_yrange is not None:
            ymin, ymax = plot_yrange
            axs.set_ylim(ymin, ymax)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_ncol = pdict.get('legend_ncol', 1)
            legend_title = pdict.get('legend_title', None)
            legend_pos = pdict.get('legend_pos', 'best')
            axs.legend(title=legend_title, loc=legend_pos, ncol=legend_ncol)
            legend_pos = pdict.get('legend_pos', 'best')
            # box_props = dict(boxstyle='Square', facecolor='white', alpha=0.6)
            legend = axs.legend(loc=legend_pos, frameon=1)
            frame = legend.get_frame()
            frame.set_alpha(0.8)
            frame.set_linewidth(0)
            frame.set_edgecolor(None)
            legend_framecol = pdict.get('legend_framecol', 'white')
            frame.set_facecolor(legend_framecol)

        if self.tight_fig:
            axs.figure.tight_layout()

    def plot_colorxy(self, pdict, axs):
        self.plot_color2D(flex_colormesh_plot_vs_xy, pdict, axs)

    def plot_colorx(self, pdict, axs):
        self.plot_color2D(flex_color_plot_vs_x, pdict, axs)

    def plot_color2D_grid_idx(self, pfunc, pdict, axs, idx):
        pfunc(pdict, np.ravel(axs)[idx])

    def plot_color2D_grid(self, pdict, axs):
        color2D_pfunc = pdict.get('pfunc', self.plot_colorxy)
        num_elements = len(pdict['zvals'])
        num_axs = axs.size
        if num_axs > num_elements:
            max_plot = num_elements
        else:
            max_plot = num_axs
        plot_idxs = pdict.get('plot_idxs', None)
        if plot_idxs is None:
            plot_idxs = list(range(max_plot))
        else:
            plot_idxs = plot_idxs[:max_plot]
        this_pdict = {key: val for key, val in list(pdict.items())}
        if pdict.get('sharex', False):
            this_pdict['xlabel'] = ''
        if pdict.get('sharey', False):
            this_pdict['ylabel'] = ''

        box_props = dict(boxstyle='Square', facecolor='white', alpha=0.7)
        plot_axlabels = pdict.get('axlabels', None)

        for ii, idx in enumerate(plot_idxs):
            this_pdict['zvals'] = np.squeeze(pdict['zvals'][idx])
            if ii != 0:
                this_pdict['title'] = None
            else:
                this_pdict['title'] = pdict['title']
            self.plot_color2D_grid_idx(color2D_pfunc, this_pdict, axs, ii)
            if plot_axlabels is not None:
                np.ravel(axs)[idx].text(
                    0.95, 0.9, plot_axlabels[idx],
                    transform=np.ravel(axs)[idx].transAxes, fontsize=16,
                    verticalalignment='center', horizontalalignment='right',
                    bbox=box_props)
        if pdict.get('sharex', False):
            for ax in axs[-1]:
                ax.set_axis_label('x', pdict['xlabel'])
        if pdict.get('sharey', False):
            for ax in axs:
                ax[0].set_axis_label('y', pdict['ylabel'])

    def plot_color2D(self, pfunc, pdict, axs):
        """

        """
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_cbar = pdict.get('plotcbar', True)
        plot_cmap = pdict.get('cmap', 'YlGn')
        plot_zrange = pdict.get('zrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_xrange = pdict.get('xrange', None)
        plot_xwidth = pdict.get('xwidth', None)
        plot_transpose = pdict.get('transpose', False)
        plot_nolabel = pdict.get('no_label', False)
        plot_normalize = pdict.get('normalize', False)
        plot_logzscale = pdict.get('logzscale', False)
        if plot_logzscale:
            plot_zvals = np.log10(pdict['zvals']/plot_logzscale)
        else:
            plot_zvals = pdict['zvals']

        if plot_xwidth is not None:
            plot_xvals_step = 0
            plot_yvals_step = 0
        else:
            plot_xvals_step = (abs(np.max(plot_xvals)-np.min(plot_xvals))/
                len(plot_xvals))
            plot_yvals_step = (abs(np.max(plot_yvals)-np.min(plot_yvals))/
                len(plot_yvals))
            # plot_yvals_step = plot_yvals[1]-plot_yvals[0]

        if plot_zrange is not None:
            fig_clim = plot_zrange
        else:
            fig_clim = [None, None]

        trace = {}
        block = {}
        if self.do_individual_traces:
            trace['xvals'] = plot_xvals
            trace['yvals'] = plot_yvals
            trace['zvals'] = plot_zvals
        else:
            trace['yvals'] = [plot_yvals]
            trace['xvals'] = [plot_xvals]
            trace['zvals'] = [plot_zvals]

        # FIXME: we should get rid of do_timestamps_blocks
        if self.do_timestamp_blocks:
            block['xvals'] = trace['xvals']
            block['yvals'] = trace['yvals']
            block['zvals'] = trace['zvals']
        else:
            block['xvals'] = [trace['xvals']]
            block['yvals'] = [trace['yvals']]
            block['zvals'] = [trace['zvals']]

        for ii in range(len(block['zvals'])):
            traces = {}
            for key, vals in block.items():
                traces[key] = vals[ii]
            for tt in range(len(traces['zvals'])):
                if self.verbose:
                    (print(t_vals[tt].shape) for key, t_vals in traces.items())
                if plot_xwidth is not None:
                    xwidth = plot_xwidth[tt]
                else:
                    xwidth = None
                out = pfunc(ax=axs,
                            xwidth=xwidth,
                            clim=fig_clim, cmap=plot_cmap,
                            xvals=traces['xvals'][tt],
                            yvals=traces['yvals'][tt],
                            zvals=traces['zvals'][tt],  # .transpose(),
                            transpose=plot_transpose,
                            normalize=plot_normalize)

        # else:
        #     out = pfunc(ax=axs, clim=fig_clim, cmap=plot_cmap,
        #                 xvals=plot_xvals,
        #                 yvals=plot_yvals,
        #                 zvals=plot_zvals.transpose(),
        #                 transpose=plot_transpose,
        #                 normalize=plot_normalize)

        if plot_xrange is None:
            if plot_xwidth is not None:
                xmin, xmax = min([min(xvals)-plot_xwidth[tt]/2
                                  for tt, xvals in enumerate(plot_xvals)]), \
                    max([max(xvals)+plot_xwidth[tt]/2
                         for tt, xvals in enumerate(plot_xvals)])
            else:
                xmin = np.min(plot_xvals) - plot_xvals_step/2
                xmax = np.max(plot_xvals) + plot_xvals_step/2
        else:
            xmin, xmax = plot_xrange
        if plot_transpose:
            axs.set_ylim(xmin, xmax)
        else:
            axs.set_xlim(xmin, xmax)

        if plot_yrange is None:
            if plot_xwidth is not None:
                ymin, ymax = min([min(yvals[0])
                                  for tt, yvals in enumerate(plot_yvals)]), \
                    max([max(yvals[0])
                         for tt, yvals in enumerate(plot_yvals)])
            else:
                ymin = np.min(plot_yvals) - plot_yvals_step / 2.
                ymax = np.max(plot_yvals) + plot_yvals_step/2.
        else:
            ymin, ymax = plot_yrange
        if plot_transpose:
            axs.set_xlim(ymin, ymax)
        else:
            axs.set_ylim(ymin, ymax)

        if not plot_nolabel:
            self.label_color2D(pdict, axs)

        axs.cmap = out['cmap']
        if plot_cbar:
            self.plot_colorbar(axs=axs, pdict=pdict)

    def label_color2D(self, pdict, axs):
        plot_transpose = pdict.get('transpose', False)
        plot_xlabel = pdict['xlabel']
        plot_xunit = pdict['xunit']
        plot_ylabel = pdict['ylabel']
        plot_yunit = pdict['yunit']
        plot_title = pdict['title']
        if plot_transpose:
            # transpose switches X and Y
            set_axis_label('x', axs, plot_ylabel, plot_yunit)
            set_axis_label('y', axs, plot_xlabel, plot_xunit)
        else:
            set_axis_label('x', axs, plot_xlabel, plot_xunit)
            set_axis_label('y', axs, plot_ylabel, plot_yunit)
        if plot_title is not None:
            axs.set_title(plot_title)

    def plot_colorbar(self, cax=None, key=None, pdict=None, axs=None,
                      orientation='vertical'):
        if key is not None:
            pdict = self.plot_dicts[key]
            axs = self.axs[key]
        else:
            if pdict is None or axs is None:
                raise ValueError(
                    'pdict and axs must be specified'
                    ' when no key is specified.')
        plot_nolabel = pdict.get('no_label', False)
        plot_clabel = pdict.get('clabel', None)
        plot_cbarwidth = pdict.get('cbarwidth', '10%')
        plot_cbarpad = pdict.get('cbarpad', '5%')
        plot_ctick_loc = pdict.get('ctick_loc', None)
        plot_ctick_labels = pdict.get('ctick_labels', None)
        if cax is None:
            if not isinstance(axs, Axes3D):
                axs.ax_divider = make_axes_locatable(axs)
                axs.cax = axs.ax_divider.append_axes(
                    'right', size=plot_cbarwidth, pad=plot_cbarpad)
                cmap = axs.cmap
            else:
                plot_cbarwidth = str_to_float(plot_cbarwidth)
                plot_cbarpad = str_to_float(plot_cbarpad)
                axs.cax, _ = mpl.colorbar.make_axes(
                    axs, shrink=1-plot_cbarwidth-plot_cbarpad, pad=plot_cbarpad,
                    orientation=orientation)
                cmap = pdict.get('colormap')
        else:
            axs.cax = cax
        if hasattr(cmap, 'autoscale_None'):
            axs.cbar = plt.colorbar(cmap, cax=axs.cax, orientation=orientation)
        else:
            norm = mpl.colors.Normalize(0, 1)
            axs.cbar = mpl.colorbar.ColorbarBase(axs.cax, cmap=cmap, norm=norm)
        if plot_ctick_loc is not None:
            axs.cbar.set_ticks(plot_ctick_loc)
        if plot_ctick_labels is not None:
            axs.cbar.set_ticklabels(plot_ctick_labels)
        if not plot_nolabel and plot_clabel is not None:
            axs.cbar.set_label(plot_clabel)

        if self.tight_fig:
            axs.figure.tight_layout()

    def plot_fit(self, pdict, axs):
        """
        Plots an lmfit fit result object using the plot_line function.
        """
        model = pdict['fit_res'].model
        plot_init = pdict.get('plot_init', False)  # plot the initial guess
        pdict['marker'] = pdict.get('marker', '')  # different default
        plot_linestyle_init = pdict.get('init_linestyle', '--')
        plot_numpoints = pdict.get('num_points', 1000)

        if len(model.independent_vars) == 1:
            independent_var = model.independent_vars[0]
        else:
            raise ValueError('Fit can only be plotted if the model function'
                             ' has one independent variable.')

        x_arr = pdict['fit_res'].userkws[independent_var]
        pdict['xvals'] = np.linspace(np.min(x_arr), np.max(x_arr),
                                     plot_numpoints)
        pdict['yvals'] = model.eval(pdict['fit_res'].params,
                                    **{independent_var: pdict['xvals']})
        self.plot_line(pdict, axs)

        if plot_init:
            # The initial guess
            pdict_init = copy.copy(pdict)
            pdict_init['linestyle'] = plot_linestyle_init
            pdict_init['yvals'] = model.eval(
                **pdict['fit_res'].init_values,
                **{independent_var: pdict['xvals']})
            pdict_init['setlabel'] += ' init'
            self.plot_line(pdict_init, axs)

    def plot_text(self, pdict, axs):
        """
        Helper function that adds text to a plot
        """
        pfunc = getattr(axs, pdict.get('func', 'text'))
        plot_text_string = pdict['text_string']
        plot_xpos = pdict.get('xpos', .98)
        plot_ypos = pdict.get('ypos', .98)
        verticalalignment = pdict.get('verticalalignment', 'top')
        horizontalalignment = pdict.get('verticalalignment', 'right')

        # fancy box props is based on the matplotlib legend
        box_props = pdict.get('box_props', 'fancy')
        if box_props == 'fancy':
            box_props = self.fancy_box_props

        # pfunc is expected to be ax.text
        pfunc(x=plot_xpos, y=plot_ypos, s=plot_text_string,
              transform=axs.transAxes,
              verticalalignment=verticalalignment,
              horizontalalignment=horizontalalignment,
              bbox=box_props)

    def plot_vlines(self, pdict, axs):
        """
        Helper function to add vlines to a plot
        """
        pfunc = getattr(axs, pdict.get('func', 'vlines'))
        x = pdict['x']
        ymin = pdict['ymin']
        ymax = pdict['ymax']
        label = pdict.get('setlabel', None)
        colors = pdict.get('colors', None)
        linestyles = pdict.get('linestyles', '--')

        axs.vlines(x, ymin, ymax, colors,
                   linestyles=linestyles, label=label, **pdict['line_kws'])
        axs.legend()

    def plot_matplot_ax_method(self, pdict, axs):
        """
        Used to use any of the methods of a matplotlib axis object through
        the pdict interface.

        An example pdict would be:
            {'func': 'axhline',
             'plot_kw': {'y': 0.5, 'mfc': 'green'}}
        which would call
            ax.axhline(y=0.5, mfc='green')
        to plot a horizontal green line at y=0.5

        """
        pfunc = getattr(axs, pdict.get('func'))
        pfunc(**pdict['plot_kws'])

def str_to_float(s):
    if s[-1] == '%':
        return float(s.strip('%'))/100
    else:
        return float(s)