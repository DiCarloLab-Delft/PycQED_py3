"""
File containing the BaseDataAnalyis class.
This is based on the REM analysis from PycQED_py2 (as of July 7 2017)



"""
import numpy as np


from matplotlib import pyplot as plt
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.default_figure_settings_analysis as def_fig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
# import SSRO_generic
import json

gco = a_tools.get_color_order


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

    def __init__(self, t_start, t_stop=None,
                 options_dict=None,
                 extract_only=False,
                 do_fitting=False):
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
        '''
        self.t_start = t_start
        if t_stop is None:
            self.t_stop = t_start
        else:
            self.t_stop = t_stop
        self.options_dict = options_dict
        self.ma_type = self.options_dict.get('ma_type', 'MeasurementAnalysis')
        # FIXME: shouldn't scan_label be a keyword fo the init?
        scan_label = options_dict.get('scan_label', '')
        if type(scan_label) is not list:
            self.labels = [scan_label]
        else:
            self.labels = scan_label
        def_fig.apply_default_figure_settings()
        self.plot_dicts = dict()
        self.axs = dict()
        self.figs = dict()
        self.extract_only = extract_only
        self.do_fitting = do_fitting
        self.presentation_mode = options_dict.get('presentation_mode', False)
        self.tight_fig = options_dict.get('tight_fig', True)
        self.do_timestamp_blocks = options_dict.get('do_blocks', False)
        self.filter_no_analysis = options_dict.get('filter_no_analysis', False)
        self.verbose = options_dict.get('verbose', False)
        self.auto_keys = options_dict.get('auto_keys', None)
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
            self.run_fitting()  # fitting to models
        self.prepare_plots()   # specify default plots
        if not self.extract_only:
            self.plot(key_list='auto')  # make the plots

    def get_timestamps(self):
        """
        Extracts timestamps based on variables
            self.t_start
            self.t_stop
            self.labels # currently set from options dict
        """
        print(self.labels)
        if type(self.t_start) is list:
            if (type(self.t_stop) is list and
                    len(self.t_stop) == len(self.t_start)):
                self.timestamps = []
                for tt in range(len(self.t_start)):
                    if self.do_timestamp_blocks:
                        self.timestamps.append(
                            a_tools.get_timestamps_in_range(
                                self.t_start[tt], self.t_stop[tt],
                                label=self.labels))
                    else:
                        self.timestamps.extend(
                            a_tools.get_timestamps_in_range(
                                self.t_start[tt], self.t_stop[tt],
                                label=self.labels))
            else:
                if self.do_timestamp_blocks:
                    self.do_timestamp_blocks = False
                raise ValueError("Invalid number of t_stop timestamps.")
        else:
            self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

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
        and stores it into
            self.data_dict
        """
        if self.t_start[-5:] == '.json':
            self.use_json = True
            self.extract_data_json()
            return
        else:
            self.use_json = False

        self.get_timestamps()
        TwoD = self.params_dict.pop('TwoD', False)

        if self.do_timestamp_blocks:
            self.data_dict = {key: [] for key in list(self.params_dict.keys())}
            self.data_dict['timestamps'] = []
            for tstamps in self.timestamps:
                if self.verbose:
                    print(len(tstamps), type(tstamps))
                temp_dict = a_tools.get_data_from_timestamp_list(
                    tstamps, param_names=self.params_dict,
                    ma_type=self.ma_type,
                    TwoD=TwoD, numeric_params=self.numeric_params,
                    filter_no_analysis=self.filter_no_analysis)
                for key in self.data_dict:
                    self.data_dict[key].append(temp_dict[key])

            # Use timestamps to calculate datetimes and add to dictionary
            self.data_dict['datetime'] = []
            for tstamps in self.data_dict['timestamps']:
                self.data_dict['datetime'].append(
                    [a_tools.datetime_from_timestamp(ts) for ts in tstamps])

            # Convert temperature data to dictionary form and extract Tmc
            if 'temperatures' in self.data_dict:
                self.data_dict['Tmc'] = []
                for ii, temperatures in enumerate(
                        self.data_dict['temperatures']):
                    temp = []
                    self.data_dict['Tmc'].append([])
                    for ii in range(len(temperatures)):
                        exec("temp.append(%s)" % (temperatures[ii]))
                        self.data_dict['Tmc'][ii].append(
                            temp[ii].get('T_MClo', None))
                    self.data_dict['temperatures'][ii] = temp

        else:
            self.data_dict = a_tools.get_data_from_timestamp_list(
                self.timestamps, param_names=self.params_dict,
                ma_type=self.ma_type,
                TwoD=TwoD, numeric_params=self.numeric_params,
                filter_no_analysis=self.filter_no_analysis)

            # Use timestamps to calculate datetimes and add to dictionary
            self.data_dict['datetime'] = [a_tools.datetime_from_timestamp(
                timestamp) for timestamp in self.data_dict['timestamps']]

            # Convert temperature data to dictionary form and extract Tmc
            if 'temperatures' in self.data_dict:
                temp = []
                self.data_dict['Tmc'] = []
                for ii in range(len(self.data_dict['temperatures'])):
                    exec("temp.append(%s)" %
                         (self.data_dict['temperatures'][ii]))
                    self.data_dict['Tmc'].append(temp[ii].get('T_MClo', None))
                self.data_dict['temperatures'] = temp

    def extract_data_json(self):
        file_name = self.t_start
        with open(file_name, 'r') as f:
            data_dict = json.load(f)
        # print [[key, type(val[0]), len(val)] for key, val in
        # data_dict.items()]
        self.data_dict = {}
        for key, val in list(data_dict.items()):
            if type(val[0]) is dict:
                self.data_dict[key] = val[0]
            else:
                self.data_dict[key] = np.double(val)
        # print [[key, type(val), len(val)] for key, val in
        # self.data_dict.items()]
        self.data_dict['timestamps'] = [self.t_start]

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

    def run_fitting(self):
        """
        Perform fits and save results of the fits to file.
        """
        pass

    def save_figures(self, savedir='', savebase=None, tag_tstamp=True,
                     fmt='pdf', key_list='auto'):
        if savebase is None:
            savebase = self.data_dict['timestamps'][0]
        else:
            if tag_tstamp:
                tstag = '_'+self.data_dict['timestamps'][0]
            else:
                tstag = ''
        if key_list is 'auto':
            key_list = self.auto_keys
        if key_list is None:
            key_list = list(self.plot_dicts.keys())
        for key in key_list:
            self.axs[key].figure.savefig(
                savedir+savebase+'_'+key+tstag+'.'+fmt, fmt=fmt)

    def plot(self, key_list=None, axs_dict=None,
             presentation_mode=None, no_label=False):
        if presentation_mode is None:
            presentation_mode = self.presentation_mode
        if axs_dict is not None:
            for key, val in list(axs_dict.items()):
                self.axs[key] = val
        if key_list is 'auto':
            key_list = self.auto_keys
        if key_list is None:
            key_list = list(self.plot_dicts.keys())
        if type(key_list) is str:
            key_list = [key_list]
        self.key_list = key_list

        for key in key_list:
            pdict = self.plot_dicts[key]
            pdict['no_label'] = no_label
            if key not in self.axs:
                # This fig variable should perhaps be a different
                # variable for each plot!!
                # This might fix a bug.
                self.figs[key], self.axs[key] = plt.subplots(
                    pdict.get('numplotsy', 1), pdict.get('numplotsx', 1),
                    sharex=pdict.get('sharex', False),
                    sharey=pdict.get('sharey', False),
                    figsize=pdict.get('plotsize', (8, 6)))
        if presentation_mode:
            self.plot_for_presentation(key_list=key_list, no_label=no_label)
        else:
            for key in key_list:
                pdict = self.plot_dicts[key]
                if type(pdict['plotfn']) is str:
                    plotfn = getattr(self, pdict['plotfn'])
                else:
                    plotfn = pdict['plotfn']
                plotfn(pdict, axs=self.axs[key])
            self.format_datetime_xaxes(key_list)
            self.add_to_plots(key_list=key_list)

    def add_to_plots(self, key_list=None):
        pass

    def format_datetime_xaxes(self, key_list):
        for key in key_list:
            pdict = self.plot_dicts[key]
            if type(pdict['xvals'][0]) is datetime.datetime:
                self.axs[key].figure.autofmt_xdate()

    def plot_for_presentation(self, key_list=None, no_label=False):
        if key_list is None:
            key_list = list(self.plot_dicts.keys())
        for key in key_list:
            self.plot_dicts[key]['title'] = None
            self.axs[key].clear()
        self.plot(key_list=key_list, presentation_mode=False,
                  no_label=no_label)

    def plot_bar(self, pdict, axs):
        pfunc = getattr(axs, pdict.get('func', 'bar'))
        # xvals interpreted as edges for a bar plot
        plot_xedges = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_title = pdict['title']
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_barkws = pdict.get('bar_kws', {})
        plot_multiple = pdict.get('multiple', False)
        dataset_desc = pdict.get('setdesc', '')
        dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
        do_legend = pdict.get('do_legend', False)

        plot_xleft = plot_xedges[:-1]
        plot_xwidth = (plot_xedges[1:]-plot_xedges[:-1])

        if plot_multiple:
            p_out = []
            for ii, this_yvals in enumerate(plot_yvals):
                p_out.append(pfunc(plot_xleft, this_yvals, width=plot_xwidth,
                                   color=gco(ii, len(plot_yvals)-1),
                                   label='%s%s' % (
                                       dataset_desc, dataset_label[ii]),
                                   **plot_barkws))

        else:
            p_out = pfunc(plot_xleft, plot_yvals, width=plot_xwidth,
                          label='%s%s' % (dataset_desc, dataset_label),
                          **plot_barkws)
        if plot_xrange is None:
            xmin, xmax = plot_xedges.min(), plot_xedges.max()
        else:
            xmin, xmax = plot_xrange
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        axs.set_ylabel(plot_ylabel)
        if plot_yrange is not None:
            ymin, ymax = plot_yrange
            axs.set_ylim(ymin, ymax)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_pos = pdict.get('legend_pos', 'best')
            # box_props = dict(boxstyle='Square', facecolor='white', alpha=0.6)
            legend = axs.legend(loc=legend_pos, frameon=1)
            frame = legend.get_frame()
            frame.set_alpha(0.8)
            frame.set_linewidth(0)
            frame.set_edgecolor(None)
            frame.set_facecolor('white')

        if self.tight_fig:
            axs.figure.tight_layout()

        pdict['handles'] = p_out

    def plot_line(self, pdict, axs):
        pfunc = getattr(axs, pdict.get('func', 'plot'))
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_title = pdict['title']
        plot_xrange = pdict.get('xrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_linekws = pdict.get('line_kws', {})
        plot_multiple = pdict.get('multiple', False)
        plot_linestyle = pdict.get('linestyle', '-')
        plot_marker = pdict.get('marker', 'o')
        dataset_desc = pdict.get('setdesc', '')
        dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
        do_legend = pdict.get('do_legend', False)

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]

        if plot_multiple:
            p_out = []
            for ii, this_yvals in enumerate(plot_yvals):
                p_out.append(pfunc(plot_xvals, this_yvals,
                                   linestyle=plot_linestyle,
                                   marker=plot_marker,
                                   color=gco(ii, len(plot_yvals)-1),
                                   label='%s%s' % (
                                       dataset_desc, dataset_label[ii]),
                                   **plot_linekws))

        else:
            p_out = pfunc(plot_xvals, plot_yvals,
                          linestyle=plot_linestyle, marker=plot_marker,
                          label='%s%s' % (dataset_desc, dataset_label),
                          **plot_linekws)
        if plot_xrange is None:
            xmin, xmax = np.min(plot_xvals)-plot_xvals_step / \
                2., np.max(plot_xvals)+plot_xvals_step/2.
        else:
            xmin, xmax = plot_xrange
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        axs.set_ylabel(plot_ylabel)
        if plot_yrange is not None:
            ymin, ymax = plot_yrange
            axs.set_ylim(ymin, ymax)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
            legend_pos = pdict.get('legend_pos', 'best')
            # box_props = dict(boxstyle='Square', facecolor='white', alpha=0.6)
            legend = axs.legend(loc=legend_pos, frameon=1)
            frame = legend.get_frame()
            frame.set_alpha(0.8)
            frame.set_linewidth(0)
            frame.set_edgecolor(None)
            frame.set_facecolor('white')

        if self.tight_fig:
            axs.figure.tight_layout()

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
            axs.set_xlabel(plot_xlabel)
            axs.set_ylabel(plot_ylabel)

        if plot_yrange is not None:
            ymin, ymax = plot_yrange
            axs.set_ylim(ymin, ymax)

        if plot_title is not None:
            axs.set_title(plot_title)

        if do_legend:
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
        self.plot_color2D(a_tools.flex_colormesh_plot_vs_xy, pdict, axs)

    def plot_imagexy(self, pdict, axs):
        self.plot_color2D(a_tools.flex_image_plot_vs_xy, pdict, axs)

    def plot_colorx(self, pdict, axs):
        self.plot_color2D(a_tools.flex_color_plot_vs_x, pdict, axs)

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
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_cbar = pdict.get('plotcbar', True)
        plot_cmap = pdict.get('cmap', 'YlGn')
        plot_zrange = pdict.get('zrange', None)
        plot_yrange = pdict.get('yrange', None)
        plot_xrange = pdict.get('xrange', None)
        plot_transpose = pdict.get('transpose', False)
        plot_nolabel = pdict.get('no_label', False)
        plot_normalize = pdict.get('normalize', False)
        plot_logzscale = pdict.get('logzscale', False)
        if plot_logzscale:
            plot_zvals = np.log10(pdict['zvals']/plot_logzscale)
        else:
            plot_zvals = pdict['zvals']

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]
        plot_yvals_step = plot_yvals[1]-plot_yvals[0]

        if plot_zrange is not None:
            fig_clim = plot_zrange
        else:
            fig_clim = [None, None]

        if self.do_timestamp_blocks:
            for tt in len(plot_zvals):
                if self.verbose:
                    print(plot_xvals[tt].shape, plot_yvals[
                          tt].shape, plot_zvals[tt].shape)
                out = pfunc(ax=axs, clim=fig_clim, cmap=plot_cmap,
                            xvals=plot_xvals[tt],
                            yvals=plot_yvals[tt],
                            zvals=plot_zvals[tt].transpose(),
                            transpose=plot_transpose,
                            normalize=plot_normalize)
        else:
            out = pfunc(ax=axs, clim=fig_clim, cmap=plot_cmap,
                        xvals=plot_xvals,
                        yvals=plot_yvals,
                        zvals=plot_zvals.transpose(),
                        transpose=plot_transpose,
                        normalize=plot_normalize)

        if plot_xrange is None:
            xmin, xmax = plot_xvals.min()-plot_xvals_step / \
                2., plot_xvals.max()+plot_xvals_step/2.
        else:
            xmin, xmax = plot_xrange
        if plot_transpose:
            axs.set_ylim(xmin, xmax)
        else:
            axs.set_xlim(xmin, xmax)

        if plot_yrange is None:
            ymin, ymax = plot_yvals.min()-plot_yvals_step / \
                2., plot_yvals.max()+plot_yvals_step/2.
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
        plot_ylabel = pdict['ylabel']
        plot_title = pdict['title']
        if plot_transpose:
            axs.set_xlabel(plot_ylabel)
            axs.set_ylabel(plot_xlabel)
        else:
            axs.set_xlabel(plot_xlabel)
            axs.set_ylabel(plot_ylabel)
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
