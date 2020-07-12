"""
File containing the BaseDataAnalyis class.
"""
import warnings
from inspect import signature
import os
import numpy as np
import copy
import logging
from collections import OrderedDict
import numbers
from matplotlib import pyplot as plt
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.general import NumpyJsonEncoder
from pycqed.analysis.analysis_toolbox import get_color_order as gco
from pycqed.analysis.analysis_toolbox import get_color_list
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.plotting import (
    flex_colormesh_plot_vs_xy, flex_color_plot_vs_x)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import json
import lmfit
import h5py
from pycqed.measurement.hdf5_data import write_dict_to_hdf5
from collections.abc import Iterable

import importlib
importlib.reload(a_tools)


class BaseDataAnalysis(object):
    """
    Abstract Base Class for analysis.

    Children inheriting from this method should specify the following methods
        - __init__      -> specify params to be extracted, set options
                           specific to analysis and call run_analysis method.
        - process_data  -> mundane tasks such as binning and filtering
        - run_fitting   -> perform fits to data
        - prepare_plots -> specify default plots and set up plotting dicts

    The core of this class is the flow defined in run_analysis and should
    be called at the end of the __init__. This executes
    the following code:

        self.extract_data()    # extract data specified in params dict
        self.process_data()    # binning, filtering etc
        if self.do_fitting:
            self.run_fitting() # fitting to models
        self.prepare_plots()   # specify default plots
        if self.save_qois:
            self.save_quantities_of_interest()
        if not self.extract_only:
            self.plot(key_list='auto')  # make the plots

    """

    fit_res = None
    '''
    Dictionary containing fitting objects
    '''
    fit_dict = None
    '''
    Dictionary containing fitting results
    '''

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '', data_file_path: str = None,
                 close_figs: bool = True, options_dict: dict = None,
                 extract_only: bool = False, do_fitting: bool = False,
                 save_qois: bool = True):
        '''
        This is the __init__ of the abstract base class.
        It is intended to be called at the start of the init of the child
        classes followed by "run_analysis".

        __init__ of the child classes:
            The __init__ of child classes  should implement the following
            functionality:
                - call the AbstractBaseClass __init__ (this method)
                - define self.params_dict and self.numeric_params
                - specify options specific to that analysis
                - call self.run_analysis

        Running analysis will add the following (important) attributes
        to the analysis object:

        Data and derived quantities:
            self.raw_data_dict          (dict)
            self.proc_data_dict         (dict)
                self.proc_data_dict['quantities_of_interest'] (dict)
            self.fit_res                (dict)

        Figures:
            axs (dict)
            figs (dict)
            plot_dicts (dict)


        and a bunch of stuff specified in the options dict
        (TODO: should this not always be extracted from the
        dict to prevent double refs? )

        There are several ways to specify where the data should be loaded
        from.

        none of the below parameters: look for the last data which matches the
                filtering options from the options dictionary.

        :param t_start, t_stop: give a range of timestamps in where data is
                                loaded from. Filtering options can be given
                                through the options dictionary. If t_stop is
                                omitted, the extraction routine looks for
                                the data with time stamp t_start.
        :param label: Only process datasets with this label.
        :param data_file_path: directly give the file path of a data file that
                                should be loaded. Note: data_file_path has
                                priority, i.e. if this argument is given time
                                stamps are ignored.
        :param close_figs: Close the figure (do not display)
        :param options_dict: available options are:
                                -'presentation_mode'
                                -'tight_fig'
                                -'plot_init'
                                -'save_figs'
                                -'close_figs'
                                -'verbose'
                                -'auto-keys'
                                -'twoD'
                                -'ma_type'
                                -'scan_label'
                                -'do_individual_traces'
                                -'filter_no_analysis'
                                -'exact_label_match'
                                -'filter_dict'
                                    dictionary of parameter names as keys and
                                    values as values. Only datasets with specified values
                                    of parameters will be extracted and used in analysis
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param save_qois: Should the save save_quantities_of_interest method be executed?
        '''
        # todo: what exactly does this flag do? May 2018 (Adriaan/Rene)
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
        if scan_label is None:
            scan_label = ''
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
        if self.presentation_mode:
            warnings.warn("presentation_mode is deprecated",
                          DeprecationWarning)
        self.tight_fig = self.options_dict.get('tight_fig', True)
        # used in self.plot_text, here for future compatibility
        self.fancy_box_props = dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5)

        self.options_dict['plot_init'] = self.options_dict.get('plot_init',
                                                               False)
        self.options_dict['save_figs'] = self.options_dict.get(
            'save_figs', True)
        self.options_dict['close_figs'] = self.options_dict.get(
            'close_figs', close_figs)
        ####################################################
        # These options relate to what analysis to perform #
        ####################################################
        self.extract_only = extract_only
        self.do_fitting = do_fitting
        if self.do_fitting:
            self.fit_dicts = OrderedDict()

        self.verbose = self.options_dict.get('verbose', False)
        self.auto_keys = self.options_dict.get('auto_keys', None)

        if type(self.auto_keys) is str:
            self.auto_keys = [self.auto_keys]
        ####################################################
        # Save quantities of interest switch               #
        ####################################################
        self.save_qois = save_qois

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.

        This function is typically called after the __init__.
        """
        self.extract_data()  # extract data specified in params dict
        self.process_data()  # binning, filtering etc
        if self.do_fitting:
            self.prepare_fitting()  # set up fit_dicts
            self.run_fitting()  # fitting to models
            self.save_fit_results()
            self.analyze_fit_results()  # analyzing the results of the fits
        if self.save_qois:
            self.save_quantities_of_interest()

        if not self.extract_only:
            self.run_post_extract()

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto')  # make the plots

        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

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
                    self.timestamps.extend(
                        a_tools.get_timestamps_in_range(
                            self.t_start[tt], self.t_stop[tt],
                            label=self.labels,
                            exact_label_match=self.exact_label_match))
            else:
                raise ValueError("Invalid number of t_stop timestamps.")
        else:
            self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels,
                exact_label_match=self.exact_label_match)

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
        filter_dict = self.options_dict.get('filter_dict', None)
        self.raw_data_dict = a_tools.get_data_from_timestamp_list(
            self.timestamps, param_names=self.params_dict,
            ma_type=self.ma_type,
            TwoD=TwoD, numeric_params=self.numeric_params,
            filter_no_analysis=self.filter_no_analysis,
            filter_dict=filter_dict)

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

    def extract_data_json(self):
        file_name = self.t_start
        with open(file_name, 'r') as f:
            raw_data_dict = json.load(f)
        # print [[key, type(val[0]), len(val)] for key, val in
        # raw_data_dict.items()]
        self.raw_data_dict = {}
        for key, val in list(raw_data_dict.items()):
            if type(val[0]) is dict:
                self.raw_data_dict[key] = val[0]
            else:
                self.raw_data_dict[key] = np.double(val)
        # print [[key, type(val), len(val)] for key, val in
        # self.raw_data_dict.items()]
        self.raw_data_dict['timestamps'] = [self.t_start]

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

    def save_figures(
        self,
        savedir: str = None,
        tag_tstamp: bool = True,
        fmt: str = 'png', key_list: list = 'auto',
        close_figs: bool = True
    ):
        """
        Save figures self.figs attribute.

        Args:
            savedir (str)       : directory to save figures to.
                    raw_data_dict['folder'] if not specified
                    raw_data_dict['folder'][-1] if folder is a list
            tag_tstamp (bool)   : appends a timstamp in the figure filename
            fmt (str)           : the format to save the figures in
                    e.g., png, svg, pdf etc.
            key_list (list)      : keys of figures to save, if 'auto',
                saves all figures in self.figs.


        """
        if savedir is None:
            savedir = self.raw_data_dict.get('folder')
            # for analyses that have more than one
            if isinstance(savedir, list):
                savedir = savedir[0]

        if tag_tstamp:
            tstag = '_' + self.raw_data_dict['timestamps'][0]
        else:
            tstag = ''

        # FIXME: remove either auto or None as an option.
        if key_list == 'auto' or key_list is None:
            key_list = self.figs.keys()

        try:
            os.mkdir(savedir)
        except FileExistsError:
            pass

        if self.verbose:
            print('Saving figures to %s' % savedir)

        for key in key_list:
            if self.presentation_mode:
                savename = os.path.join(
                    savedir, key + tstag + 'presentation' + '.' + fmt)
                self.figs[key].savefig(savename, bbox_inches='tight', fmt=fmt)
                savename = os.path.join(
                    savedir, key + tstag + 'presentation' + '.svg')
                self.figs[key].savefig(
                    savename, bbox_inches='tight', fmt='svg')
            else:
                savename = os.path.join(
                    savedir, key + tstag + '.' + fmt)
                self.figs[key].savefig(savename, bbox_inches='tight', fmt=fmt)
            if close_figs:
                plt.close(self.figs[key])

    def save_data(
        self, savedir: str = None, savebase: str = None,
        tag_tstamp: bool = True,
        fmt: str = 'json', key_list='auto'
    ):
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
            tstag = '_' + self.raw_data_dict['timestamps'][0]
        else:
            tstag = ''

        if key_list == 'auto' or key_list is None:
            key_list = self.raw_data_dict.keys()

        save_dict = {}
        for k in key_list:
            save_dict[k] = self.raw_data_dict[k]

        try:
            os.mkdir(savedir)
        except FileExistsError:
            pass

        filepath = os.path.join(savedir, savebase + tstag + '.' + fmt)
        if self.verbose:
            print('Saving raw data to %s' % filepath)
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


        There are two ways of fitting, specified in fit_dict['fitting_type']

        - Using the model-fit procedure of lmfit, this is the default
                fit_dict['fitting_type'] = 'model'
        - Using the minimizer routine of lmfit, this needs to be specified by
                fit_dict['fitting_type'] = 'minimize'


        Initial guesses can be passed on in several different ways.

        - as fit_dict['guess_pars'] directly as the model with guess parameters,
                that needs to be made in the respective analysis and passed on
                like fit_dict['guess_pars'] = model.make_params()
                If this argument is passed on, no other guesses will be performed.
                This is not implemented yet for the 'minimize' fitting type.

        - as a guess function that will be run. This can be passed explicitly as
                fit_dict['fit_guess_fn']
                or also by giving the model specified in fit_dict['model'] an
                argument .guess
                The guess function can be given parameters in
                fit_dict['guessfn_pars']

        - as fit_dict['guess_dict'], which is a dictionary containing the guess
                parameters. These guess parameters will converted into the parameter
                objects required by either model fit or minimize.

        '''
        self.fit_res = {}
        for key, fit_dict in self.fit_dicts.items():
            guess_dict = fit_dict.get('guess_dict', None)
            guess_pars = fit_dict.get('guess_pars', None)
            guessfn_pars = fit_dict.get('guessfn_pars', {})
            fit_yvals = fit_dict['fit_yvals']
            fit_xvals = fit_dict['fit_xvals']

            fitting_type = fit_dict.get('fitting_type', 'model')

            model = fit_dict.get('model', None)
            if model is None:
                fit_fn = fit_dict.get('fit_fn', None)
                model = fit_dict.get('model', lmfit.Model(fit_fn))
            fit_guess_fn = fit_dict.get('fit_guess_fn', None)
            if fit_guess_fn is None:
                if fitting_type == 'model' and fit_dict.get('fit_guess', True):
                    fit_guess_fn = model.guess

            if guess_pars is None:  # if you pass on guess_pars, immediately go to the fitting
                if fit_guess_fn is not None:  # Run the guess funtions here
                    if fitting_type == 'minimize':
                        guess_pars = fit_guess_fn(**fit_yvals, **fit_xvals, **guessfn_pars)
                        params = lmfit.Parameters()
                        for gd_key, val in guess_pars.items():
                            params.add(gd_key)
                            for attr, attr_val in val.items():
                                setattr(params[gd_key], attr, attr_val)

                    # a fit function should return lmfit parameter objects
                    # but can also work by returning a dictionary of guesses
                    elif fitting_type == 'model':
                        guess_pars = fit_guess_fn(**fit_yvals, **fit_xvals, **guessfn_pars)
                        if not isinstance(guess_pars, lmfit.Parameters):
                            for gd_key, val in list(guess_pars.items()):
                                model.set_param_hint(gd_key, **val)
                            guess_pars = model.make_params()

                        # A guess can also be specified as a dictionary.
                        # additionally this can be used to overwrite values
                        # from the guess functions.
                        if guess_dict is not None:
                            for gd_key, val in guess_dict.items():
                                for attr, attr_val in val.items():
                                    # e.g. setattr(guess_pars['frequency'], 'value', 20e6)
                                    setattr(guess_pars[gd_key], attr, attr_val)
                elif guess_dict is not None:
                    if fitting_type == 'minimize':
                        params = lmfit.Parameters()
                        for gd_key, val in list(guess_dict.items()):
                            params.add(gd_key)
                            for attr, attr_val in val.items():
                                setattr(params[gd_key], attr, attr_val)

                    elif fitting_type == 'model':
                        for gd_key, val in list(guess_dict.items()):
                            model.set_param_hint(gd_key, **val)
                        guess_pars = model.make_params()
            else:
                if fitting_type == 'minimize':
                    raise NotImplementedError(
                        'Conversion from guess_pars to params with lmfit.Parameters() needs to be implemented')
                    # TODO: write a method that converts the type model.make_params() to a lmfit.Parameters() object
            if fitting_type == 'model':  # Perform the fitting
                fit_dict['fit_res'] = model.fit(**fit_xvals, **fit_yvals,
                                                params=guess_pars)
                self.fit_res[key] = fit_dict['fit_res']
            elif fitting_type == 'minimize':  # Perform the fitting

                fit_dict['fit_res'] = lmfit.minimize(fcn=_complex_residual_function,
                                                     params=params,
                                                     args=(fit_fn, fit_xvals, fit_yvals))
                # save the initial params
                fit_dict['fit_res'].initial_params = params
                fit_dict['fit_res'].userkws = fit_xvals  # save the x values
                fit_dict['fit_res'].fit_fn = fit_fn  # save the fit function
                self.fit_res[key] = fit_dict['fit_res']

    def save_fit_results(self):
        """
        Save fit_results that are part of self.fit_res.

        Fit results from the self.fit_res dict are stored in the hdf5 file
        under
            Analysis/fr_key, where fr_key is the key in self.fit_res.

        Fit results overwrite previously stored data if there is a conflict
        in naming. This is so the most recent analysis ran is stored.
        """

        # Check weather there is any data to save
        if hasattr(self, 'fit_res') and self.fit_res is not None:
            # Find the file to save to
            fn = self.options_dict.get('analysis_result_file', False)
            if not fn:
                fn = a_tools.measurement_filename(
                    a_tools.get_folder(self.timestamps[0]))

            try:
                os.mkdir(os.path.dirname(fn))
            except FileExistsError:
                pass

            if self.verbose:
                print('Saving fitting results to %s' % fn)

            # Save data to file
            with h5py.File(fn, 'a') as data_file:
                try:
                    analysis_group = data_file.create_group('Analysis')
                except ValueError:
                    # If the analysis group already exists, re-use it
                    # (as not to overwrite previous/other fits)
                    analysis_group = data_file['Analysis']

                # Iterate over all the fit result dicts as not to overwrite old/other analysis
                for fr_key, fit_res in self.fit_res.items():
                    try:
                        fr_group = analysis_group.create_group(fr_key)
                    except ValueError:
                        # If the analysis sub group already exists (each fr_key should be unique)
                        # Delete the old group and create a new group (overwrite).
                        del analysis_group[fr_key]
                        fr_group = analysis_group.create_group(fr_key)

                    d = self._convert_dict_rec(copy.deepcopy(fit_res))
                    write_dict_to_hdf5(d, entry_point=fr_group)

    def save_quantities_of_interest(self):
        """
        Save quantities of interest.

        If self.proc_data_dict['quantities_of_interest'] exists, and it is
        a dictionary it will attempt to store the contents in the datafile in
        "Analysis/quantities_of_interest"

        Previously stored quantities of interest are overwritten.
        """

        # Check weather there is any data to save
        if 'quantities_of_interest' in self.proc_data_dict and isinstance(
                self.proc_data_dict['quantities_of_interest'], dict):
            # Find the file to save to
            fn = self.options_dict.get('analysis_result_file', False)
            if not fn:
                fn = a_tools.measurement_filename(
                    a_tools.get_folder(self.timestamps[0]))

            try:
                os.mkdir(os.path.dirname(fn))
            except FileExistsError:
                pass

            if self.verbose:
                print('Saving quantities of interest to %s' % fn)

            qoi_name = 'quantities_of_interest'
            # Save data to file
            with h5py.File(fn, 'a') as data_file:
                a_key = 'Analysis'
                if a_key not in data_file.keys():
                    analysis_group = data_file.create_group(a_key)
                else:
                    analysis_group = data_file[a_key]

                # [2020-07-11 Victor] some analysis can be called several
                # times on the same datafile, e.g. single qubit RB,
                # in that case the `qois_group` should not be overwritten!
                # level = 0 => Overwrites the entire qois_group
                # level = 1 => Overwrites only the entries in the `qois_group`
                # present in the `qois_dict`
                overwrite_qois = getattr(self, "overwrite_qois", True)
                group_overwrite_level = 0 if overwrite_qois else 1

                qois_dict = {qoi_name: self.proc_data_dict['quantities_of_interest']}
                write_dict_to_hdf5(
                    qois_dict,
                    entry_point=analysis_group,
                    group_overwrite_level=group_overwrite_level)

    @staticmethod
    def _convert_dict_rec(obj):
        try:
            # is iterable?
            for k in obj:
                obj[k] = BaseDataAnalysis._convert_dict_rec(obj[k])
        except TypeError:
            if isinstance(obj, lmfit.model.ModelResult) or isinstance(obj, lmfit.minimizer.MinimizerResult):
                obj = BaseDataAnalysis._flatten_lmfit_modelresult(obj)
            else:
                obj = str(obj)
        return obj

    @staticmethod
    def _flatten_lmfit_modelresult(model):
        assert (type(model) is lmfit.model.ModelResult or
                type(model) is lmfit.minimizer.MinimizerResult)
        dic = OrderedDict()
        dic['success'] = model.success
        dic['message'] = model.message
        dic['params'] = {}
        for param_name in model.params:
            dic['params'][param_name] = {}
            param = model.params[param_name]
            for k in param.__dict__:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    dic['params'][param_name][k] = getattr(param, k)
            dic['params'][param_name]['value'] = getattr(param, 'value')

        return dic

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
                    figsize=pdict.get('plotsize', None)
                    # plotsize None uses .rc_default of matplotlib
                )

                # transparent background around axes for presenting data
                self.figs[pdict['ax_id']].patch.set_alpha(0)

        if presentation_mode:
            self.plot_for_presentation(key_list=key_list, no_label=no_label)
        else:
            for key in key_list:
                pdict = self.plot_dicts[key]
                plot_touching = pdict.get('touching', False)

                if type(pdict['plotfn']) is str:
                    plotfn = getattr(self, pdict['plotfn'])
                else:
                    plotfn = pdict['plotfn']

                # used to ensure axes are touching
                if plot_touching:
                    self.axs[pdict['ax_id']].figure.subplots_adjust(wspace=0,
                                                                    hspace=0)

                # Check if pdict is one of the accepted arguments, these are
                # the plotting functions in the analysis base class.
                if 'pdict' in signature(plotfn).parameters:
                    plotfn(pdict=pdict, axs=self.axs[pdict['ax_id']])

                # most normal plot functions also work, it is required
                # that these accept an "ax" argument to plot on and **kwargs
                # the pdict is passed in as kwargs to such a function
                elif 'ax' in signature(plotfn).parameters:
                    # Calling the function passing along anything
                    # defined in the specific plot dict as kwargs
                    plotfn(ax=self.axs[pdict['ax_id']], **pdict)
                else:
                    raise ValueError(
                        '"{}" is not a valid plot function'.format(plotfn))

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
        plot_xedges = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict.get('xlabel', None)
        plot_ylabel = pdict.get('ylabel', None)
        plot_xunit = pdict.get('xunit', None)
        plot_yunit = pdict.get('yunit', None)
        plot_title = pdict.get('title', None)
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_barkws = pdict.get('bar_kws', {})
        plot_multiple = pdict.get('multiple', False)
        dataset_desc = pdict.get('setdesc', '')
        dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
        do_legend = pdict.get('do_legend', False)
        plot_touching = pdict.get('touching', False)

        plot_xwidth = (plot_xedges[1:] - plot_xedges[:-1])
        # center is left edge + widht /2
        plot_centers = plot_xedges[:-1] + plot_xwidth / 2

        if plot_multiple:
            p_out = []
            for ii, this_yvals in enumerate(plot_yvals):
                p_out.append(pfunc(plot_centers, this_yvals, width=plot_xwidth,
                                   color=gco(ii, len(plot_yvals) - 1),
                                   label='%s%s' % (
                                       dataset_desc, dataset_label[ii]),
                                   **plot_barkws))

        else:
            p_out = pfunc(plot_centers, plot_yvals, width=plot_xwidth,
                          label='%s%s' % (dataset_desc, dataset_label),
                          **plot_barkws)
        if plot_xrange is None:
            xmin, xmax = plot_xedges.min(), plot_xedges.max()
        else:
            xmin, xmax = plot_xrange
        if plot_xlabel is not None:
            set_xlabel(axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_ylabel(axs, plot_ylabel, plot_yunit)
        axs.set_xlim(xmin, xmax)
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

        if plot_touching:
            axs.figure.subplots_adjust(wspace=0, hspace=0)

        if self.tight_fig:
            axs.figure.tight_layout()

        pdict['handles'] = p_out

    def plot_line(self, pdict, axs):
        """
        Basic line plotting function.
        Takes either an x and y array or a list of x and y arrays.
        Detection happens based on types of the data
        """

        # if a y or xerr is specified, used the errorbar-function
        plot_linekws = pdict.get('line_kws', {})
        xerr = pdict.get('xerr', None)
        yerr = pdict.get('yerr', None)
        if xerr is not None or yerr is not None:
            pdict['func'] = pdict.get('func', 'errorbar')
            if yerr is not None:
                plot_linekws['yerr'] = plot_linekws.get('yerr', yerr)
            if xerr is not None:
                plot_linekws['xerr'] = plot_linekws.get('xerr', xerr)

        pdict['line_kws'] = plot_linekws

        axs.set_aspect(pdict.get('aspect', 'auto'))
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
            assert (len(plot_xvals) == len(plot_yvals))
            assert (len(plot_xvals[0]) == len(plot_yvals[0]))

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
            if pdict.get('color', False):
                plot_linekws['color'] = pdict.get('color')
            # "setlabel": "NONE" allows to disable the label
            p_out = pfunc(plot_xvals, plot_yvals,
                          linestyle=plot_linestyle,
                          marker=plot_marker,
                          label=(None if dataset_label == "NONE"
                            else '%s%s' % (dataset_desc, dataset_label)),
                          **plot_linekws)

        if plot_xrange is None:
            pass  # Do not set xlim if xrange is None as the axs gets reused
        else:
            xmin, xmax = plot_xrange
            axs.set_xlim(xmin, xmax)

        if plot_xlabel is not None:
            set_xlabel(axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_ylabel(axs, plot_ylabel, plot_yunit)
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
                set_xlabel(axs, plot_xlabel, plot_xunit)
            if plot_ylabel is not None:
                set_ylabel(axs, plot_ylabel, plot_yunit)

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

        plot_xvals_step = plot_xvals[1] - plot_xvals[0]

        for ii, idx in enumerate(slice_idxs):
            if len(slice_idxs) == 1:
                pfunc(plot_xvals, plot_yvals[idx], '-bo',
                      label='%s = %.2f %s' % (
                          slice_label, plot_slicevals[idx], slice_units))
            else:
                if ii == 0 or ii == len(slice_idxs) - 1:
                    pfunc(plot_xvals, plot_yvals[idx], '-o',
                          color=gco(ii, len(slice_idxs) - 1),
                          label='%s = %.2f %s' % (
                              slice_label, plot_slicevals[idx], slice_units))
                else:
                    pfunc(plot_xvals, plot_yvals[idx], '-o',
                          color=gco(ii, len(slice_idxs) - 1))
        if plot_xrange is None:
            xmin, xmax = np.min(plot_xvals) - plot_xvals_step / \
                2., np.max(plot_xvals) + plot_xvals_step / 2.
        else:
            xmin, xmax = plot_xrange
        axs.set_xlim(xmin, xmax)

        if not plot_nolabel:
            axs.set_xlabel(plot_xlabel)
            axs.set_ylabel(plot_ylabel)

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
        """
        This wraps flex_colormesh_plot_vs_xy which excepts data of shape
            x -> 1D array
            y -> 1D array
            z -> 2D array (shaped (xl, yl))
        """
        self.plot_color2D(flex_colormesh_plot_vs_xy, pdict, axs)

    def plot_colorx(self, pdict, axs):
        """
        This wraps flex_color_plot_vs_x which excepts data of shape
            x -> 1D array
            y -> list "xl" 1D arrays
            z -> list "xl" 1D arrays
        """

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
                ax.set_xlabel(pdict['xlabel'])
        if pdict.get('sharey', False):
            for ax in axs:
                ax[0].set_ylabel(pdict['ylabel'])

    def plot_color2D(self, pfunc, pdict, axs):
        """

        """
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_cbar = pdict.get('plotcbar', True)
        plot_cmap = pdict.get('cmap', 'viridis')
        plot_zrange = pdict.get('zrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_xrange = pdict.get('xrange', None)
        plot_xwidth = pdict.get('xwidth', None)
        plot_transpose = pdict.get('transpose', False)
        plot_nolabel = pdict.get('no_label', False)
        plot_normalize = pdict.get('normalize', False)
        plot_logzscale = pdict.get('logzscale', False)
        if plot_logzscale:
            plot_zvals = np.log10(pdict['zvals'] / plot_logzscale)
        else:
            plot_zvals = pdict['zvals']

        if plot_xwidth is not None:
            plot_xvals_step = 0
            plot_yvals_step = 0
        else:
            plot_xvals_step = (abs(np.max(plot_xvals) - np.min(plot_xvals)) /
                               len(plot_xvals))
            plot_yvals_step = (abs(self._globalmax(plot_yvals) - self._globalmin(plot_yvals)) /
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
                            zvals=traces['zvals'][tt],
                            transpose=plot_transpose,
                            normalize=plot_normalize)

        if plot_xrange is None:
            if plot_xwidth is not None:
                xmin, xmax = min([min(xvals) - plot_xwidth[tt] / 2
                                  for tt, xvals in enumerate(plot_xvals)]), \
                    max([max(xvals) + plot_xwidth[tt] / 2
                         for tt, xvals in enumerate(plot_xvals)])
            else:
                xmin = np.min(plot_xvals) - plot_xvals_step / 2
                xmax = np.max(plot_xvals) + plot_xvals_step / 2
        else:
            xmin, xmax = plot_xrange
        if plot_transpose:
            axs.set_ylim(xmin, xmax)
        else:
            axs.set_xlim(xmin, xmax)

        if plot_yrange is None:
            if plot_xwidth is not None:
                ymin_list, ymax_list = [], []
                for ytraces in block['yvals']:
                    ymin_trace, ymax_trace = [], []
                    for yvals in ytraces:
                        ymin_trace.append(min(yvals))
                        ymax_trace.append(max(yvals))
                    ymin_list.append(min(ymin_trace))
                    ymax_list.append(max(ymax_trace))
                ymin = min(ymin_list)
                ymax = max(ymax_list)
            else:
                ymin = self._globalmin(plot_yvals) - plot_yvals_step / 2.
                ymax = self._globalmax(plot_yvals) + plot_yvals_step / 2.
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
        plot_title = pdict.get('title', None)
        if plot_transpose:
            # transpose switches X and Y
            set_xlabel(axs, plot_ylabel, plot_yunit)
            set_ylabel(axs, plot_xlabel, plot_xunit)
        else:
            set_xlabel(axs, plot_xlabel, plot_xunit)
            set_ylabel(axs, plot_ylabel, plot_yunit)
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
        plot_zlabel = pdict['zlabel']
        plot_cbarwidth = pdict.get('cbarwidth', '10%')
        plot_cbarpad = pdict.get('cbarpad', '5%')
        plot_numcticks = pdict.get('numcticks', 5.)
        if cax is None:
            axs.ax_divider = make_axes_locatable(axs)
            axs.cax = axs.ax_divider.append_axes(
                'right', size=plot_cbarwidth, pad=plot_cbarpad)
        else:
            axs.cax = cax
        axs.cbar = plt.colorbar(axs.cmap, cax=axs.cax, orientation=orientation)
        # cmin, cmax = axs.cbar.get_clim()
        # cbarticks = np.arange(cmin,1.01*cmax,(cmax-cmin)/plot_numcticks)
        # cbar.set_ticks(cbarticks)
        # cbar.set_ticklabels(['%.0f'%(val) for val in cbarticks])
        if not plot_nolabel and plot_zlabel is not None:
            axs.cbar.set_label(plot_zlabel)

        if self.tight_fig:
            axs.figure.tight_layout()

    def plot_fit(self, pdict, axs):
        """
        Plots an lmfit fit result object using the plot_line function.
        """
        if "ax_row" in pdict.keys() and "ax_col" in pdict.keys():
            # This covers the case of being able to plot fits on
            # specific subplot
            axs = axs[pdict["ax_row"]][pdict["ax_col"]]

        if pdict['fit_res'] == {}:
            # This is an implicit way of indicating a failed fit.
            # We can probably do better by for example plotting the initial
            # guess.
            logging.warning('fit_res is an empty dictionary, cannot plot.')
            return

        plot_init = pdict.get('plot_init', False)  # plot the initial guess
        plot_normed = pdict.get('plot_normed', False)
        pdict['marker'] = pdict.get('marker', '')  # different default
        plot_linestyle_init = pdict.get('init_linestyle', '--')
        plot_numpoints = pdict.get('num_points', 1000)

        if hasattr(pdict['fit_res'], 'model'):
            model = pdict['fit_res'].model
            if not (isinstance(model, lmfit.model.Model) or
                    isinstance(model, lmfit.model.ModelResult)):
                raise TypeError(
                    'The passed item in "fit_res" needs to be'
                    ' a fitting model, but is {}'.format(type(model)))

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
        else:
            '''
            This is the case for the minimizier fit
            '''

            fit_xvals = pdict['fit_res'].userkws
            if len(fit_xvals.keys()) == 1:
                independent_var = list(fit_xvals.keys())[0]
            else:
                raise ValueError('Fit can only be plotted if the model function'
                                 ' has one independent variable.')
            x_arr = pdict['fit_res'].userkws[independent_var]
            pdict['xvals'] = np.linspace(np.min(x_arr), np.max(x_arr),
                                         plot_numpoints)
            fit_fn = pdict['fit_res'].fit_fn
            output = fit_fn(**pdict['fit_res'].params,
                            **{independent_var: pdict['xvals']})
            output_mod_fn = pdict.get('output_mod_fn', None)
            if output_mod_fn is not None:
                pdict['yvals'] = output_mod_fn(output)
            else:
                pdict['yvals'] = output

            # plot parametrically
            output_mod_fn_x = pdict.get('output_mod_fn_x', None)
            if output_mod_fn_x is not None:
                pdict['xvals'] = output_mod_fn_x(output)

        if plot_normed:
            pdict['yvals'] = pdict['yvals'] / pdict['yvals'][0]

        self.plot_line(pdict, axs)

        if plot_init:
            pdict_init = copy.copy(pdict)
            pdict_init['linestyle'] = plot_linestyle_init
            if hasattr(pdict_init['fit_res'], 'model'):
                # The initial guess
                pdict_init['yvals'] = model.eval(
                    **pdict_init['fit_res'].init_values,
                    # This is probably a bug .init_values should be .init_params
                    # not changing as I cannot test it right now.
                    **{independent_var: pdict_init['xvals']})
            else:
                output = fit_fn(**pdict_init['fit_res'].initial_params,
                                **{independent_var: pdict_init['xvals']})
                output_mod_fn = pdict_init.get('output_mod_fn', None)
                if output_mod_fn is not None:
                    pdict_init['yvals'] = output_mod_fn(output)
                else:
                    pdict_init['yvals'] = output

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
        horizontalalignment = pdict.get('horizontalalignment', 'right')

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
        # axs.legend()

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
        pdict['plot_args'] = pdict.get('plot_args', [])
        pfunc(*pdict['plot_args'], **pdict['plot_kws'])

    @staticmethod
    def _sort_by_axis0(arr, sorted_indices, type=None):
        '''
        Sorts the array (possibly a list of unequally long lists) by a list of indicies
        :param arr: array (possibly a list of unequally long lists)
        :param sorted_indices:  list of indicies
        :param type: the datatype of the contained values
        :return: Sorted array
        '''
        if type is None:
            return [np.array(arr[i]) for i in sorted_indices]
        else:
            return [np.array(arr[i], dtype=type) for i in sorted_indices]

    @staticmethod
    def _globalmin(array):
        '''
        Gives the global minimum of an array (possibly a list of unequally long lists)
        :param array: array (possibly a list of unequally long lists)
        :return: Global minimum
        '''
        return np.min([np.min(v) for v in array])

    @staticmethod
    def _globalmax(array):
        '''
        Gives the global maximum of an array (possibly a list of unequally long lists)
        :param array: array (possibly a list of unequally long lists)
        :return: Global maximum
        '''
        return np.max([np.max(v) for v in array])

    def plot_vlines_auto(self, pdict, axs):
        xs = pdict.get('xdata')
        for i, x in enumerate(xs):
            d = {}
            for k in pdict:
                lk = k[:-1]
                # if lk in signature(axs.axvline).parameters:
                if k not in ['xdata', 'plotfn', 'ax_id', 'do_legend']:
                    try:
                        d[lk] = pdict[k][i]
                    except:
                        pass
            axs.axvline(x=x, **d)

    def plot_errorbar(self, x, y, yerr, title, xlabel='', xunit='',
                      marker='o',
                      ylabel='', yunit='', ax=None, **kw):
        """
        FIXME, move to base data analysis class
        """
        if ax is None:
            f, ax = plt.subplots()

        ax.errorbar(x, y, yerr, marker=marker)
        ax.set_title(title)
        set_xlabel(ax, xlabel, xunit)
        set_ylabel(ax, ylabel, yunit)


def plot_scatter_errorbar(self, ax_id, xdata, ydata,
                          xerr=None, yerr=None, pdict=None):
    pdict = pdict or {}

    pds = {
        'ax_id': ax_id,
        'plotfn': self.plot_line,
        'zorder': 10,
        'xvals': xdata,
        'yvals': ydata,
        'marker': 'x',
        'linestyle': 'None',
        'yerr': yerr,
        'xerr': xerr,
    }

    if xerr is not None or yerr is not None:
        pds['func'] = 'errorbar'
        pds['marker'] = None
        pds['line_kws'] = {'fmt': 'none'}
        if pdict.get('marker', False):
            pds['line_kws'] = {'fmt': pdict['marker']}
        else:
            ys = 0 if yerr is None else np.min(yerr) / np.max(ydata)
            xs = 0 if xerr is None else np.min(xerr) / np.max(xdata)
            if ys < 1e-2 and xs < 1e-2:
                pds['line_kws'] = {'fmt': 'o'}
            else:
                pds['line_kws'] = {'fmt': '.'}
    else:
        pds['func'] = 'scatter'

    pds = _merge_dict_rec(pds, pdict)

    return pds


def plot_scatter_errorbar_fit(self, ax_id, xdata, ydata, fitfunc, xerr=None, yerr=None, fitextra=0.1,
                              fitpoints=1000, pdict_scatter=None, pdict_fit=None):
    pdict_fit = pdict_fit or {}
    pds = plot_scatter_errorbar(self=self, ax_id=ax_id, xdata=xdata, ydata=ydata, xerr=xerr, yerr=yerr,
                                pdict=pdict_scatter)

    mi, ma = np.min(xdata), np.max(xdata)
    ex = (ma - mi) * fitextra
    xdata_fit = np.linspace(mi - ex, ma + ex, fitpoints)
    ydata_fit = fitfunc(xdata_fit)

    pdf = {
        'ax_id': ax_id,
        'zorder': 5,
        'plotfn': self.plot_line,
        'xvals': xdata_fit,
        'yvals': ydata_fit,
        'linestyle': '-',
        'marker': '',
    }

    pdf = _merge_dict_rec(pdf, pdict_fit)

    return pds, pdf


def _merge_dict_rec(dict_a: dict, dict_b: dict):
    for k in dict_a:
        if k in dict_b:
            if dict_a[k] is dict or dict_b[k] is dict:
                a = dict_a[k] or {}
                dict_a[k] = _merge_dict_rec(a, dict_b[k])
            else:
                dict_a[k] = dict_b[k]
    for k in dict_b:
        if k not in dict_a:
            dict_a[k] = dict_b[k]
    return dict_a


def _complex_residual_function(pars, fit_function, x, data):
    '''
    Residual of a complex function with complex results dictionary data and
    real input values dictionary 'x'
    data should be of the format data = {'data':[np.array]}
    x should be of the format x = {'x':[np.array}} where x is the variable
    name in the function fit_function
    For resonators 'x' is the the frequency, 'data' the complex transmission
    '''

    cmp_values = fit_function(**x, **pars)

    res = cmp_values-data['data']
    res = np.append(res.real, res.imag)

    return res
