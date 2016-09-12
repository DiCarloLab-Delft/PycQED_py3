import sys
import numpy as np
sys.path.append('d:\GitHubRepos\PycQED_v3')
from matplotlib import pyplot as plt
from analysis import analysis_toolbox as a_tools
from analysis import fitting_models as fit_mods
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis.plotting_tools import *



class Standard_MA(object):
    """
    This analysis object imports all the timestamps between t_start and t_stop
        If not given, t_stop is assumed to be t_start (one folder measurement).
        extract_only disables the plot.
        do_fitting: enables/disables fitting call
        options_dictionary is a dictionary containing:
                        scan_label: label to filter timestamps
                        other variables which are analysis dependant

    This class is meant to be a parent class for any analysis. Each analysis needs
    to specify the analysis flow:
        params_dict_TD: dictionary containing the values needed from the files
        numeric_params: list containing keys that are meant to be transformed to numbers
        process_data(): particular function for each analysis
        run_fitting()(): particular function for each analysis
        prepare_plots(): particular function for each analysis

    """
    def __init__(self, t_start, t_stop=None, options_dict=None,
                 extract_only=False, do_fitting=False):
        '''
        to be defined
        '''
        self.t_start = t_start
        if t_stop is None:
            self.t_stop = t_start
        else:
            self.t_stop = t_stop
        self.options_dict = options_dict
        self.labels = options_dict['scan_label']
        self.plot_dicts = dict()
        self.axs = dict()
        self.extract_only = extract_only
        self.do_fitting = do_fitting

    def extract_data(self):
        self.TD_timestamps = a_tools.get_timestamps_in_range(self.t_start, self.t_stop, label=self.labels)

        if len(self.TD_timestamps) < 1:
            raise ValueError("No timestamps in range! Check the labels and other filters.")

        self.TD_dict = a_tools.get_data_from_timestamp_list(self.TD_timestamps, self.params_dict_TD, numeric_params=self.numeric_params)

        # Use timestamps to calculate datetimes and add to dictionary
        self.TD_dict['datetime'] = [a_tools.datetime_from_timestamp(timestamp) for timestamp in self.TD_dict['timestamps']]

        # Convert temperature data to dictionary form and extract Tmc
        # temp = []
        # self.TD_dict['Tmc'] = []
        # for ii in range(len(self.TD_dict['temperatures'])):
        #     exec("temp.append(%s)"%(self.TD_dict['temperatures'][ii]))
        #     self.TD_dict['Tmc'].append(temp[ii].get('T_MClo',None))
        # self.TD_dict['temperatures'] = temp

    def run_analysis(self):
        self.extract_data()
        self.process_data()
        if self.do_fitting:
            self.run_fitting()
        self.prepare_plots()
        if not self.extract_only:
            self.plot()

    def prepare_plots(self):
        pass

    def run_fitting(self):
        pass

    def plot(self, key_list=None, axs_dict=None):
        if axs_dict is not None:
            for key, val in axs_dict.items():
                self.axs[key] = val
        if key_list is None:
            key_list = self.plot_dicts.keys()
        if type(key_list) is str:
            key_list = [key_list]
        for key in key_list:
            pdict = self.plot_dicts[key]
            if key not in self.axs:
                fig, self.axs[key] = plt.subplots(pdict.get('numplotsx', 1),
                                                  pdict.get('numplotsy', 1),
                                                  figsize=pdict.get('plotsize', (8,6)))

            pdict['plotfn'](pdict, axs=self.axs[key])
            # if pdict['type'] == 'colorxy':
            #     self.plot_colorxy(pdict, axs=self.axs[key])
            # elif pdict['type'] == 'colorx':
            #     self.plot_colorx(pdict, axs=self.axs[key])
            # elif pdict['type'] == 'line':
            #     self.plot_line(pdict, axs=self.axs[key])

    def plot_line(self, pdict, axs=None):
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_title = pdict['title']

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]

        axs.plot(plot_xvals, plot_yvals, '-bo')
        xmin, xmax = plot_xvals.min()-plot_xvals_step/2., plot_xvals.max()+plot_xvals_step/2.
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        ymin, ymax = plot_yvals.min()-plot_yvals_step/2.,plot_yvals.max()+plot_yvals_step/2.
        axs.set_ylabel(plot_ylabel)
        axs.set_ylim(ymin, ymax)

        axs.set_title(plot_title)

        axs.figure.tight_layout()

    def plot_yslices(self, pdict, axs=None):
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_slicevals = pdict['slicevals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_title = pdict['title']
        slice_idxs = pdict['sliceidxs']
        slice_label = pdict['slicelabel']
        slice_units = pdict['sliceunits']

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]

        for ii, idx in enumerate(slice_idxs):
            if len(slice_idxs) == 1:
                axs.plot(plot_xvals, plot_yvals[idx], '-bo',
                         label='%s = %.3f %s'%(slice_label, plot_slicevals[idx], slice_units))
            else:
                if ii==0 or ii==len(slice_idxs)-1:
                    axs.plot(plot_xvals, plot_yvals[idx], '-o',
                             color=get_color_order(ii, len(slice_idxs)-1),
                             label='%s = %.3f %s'%(slice_label, plot_slicevals[idx], slice_units))
                else:
                    axs.plot(plot_xvals, plot_yvals[idx], '-o',
                             color=get_color_order(ii, len(slice_idxs)-1))
        xmin, xmax = pdict['xrange']
        # xmin, xmax = plot_xvals.min()-plot_xvals_step/2., plot_xvals.max()+plot_xvals_step/2.
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        ymin, ymax = pdict['yrange']
        axs.set_ylabel(plot_ylabel)
        axs.set_ylim(ymin, ymax)

        axs.set_title(plot_title)

        axs.legend(loc='best')

        axs.figure.tight_layout()


    def plot_colorxy(self, pdict, axs=None):
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_zvals = pdict['zvals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_zlabel = pdict['zlabel']
        plot_title = pdict['title']
        plot_cmap = pdict.get('cmap','YlGn')

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]
        plot_yvals_step = plot_yvals[1]-plot_yvals[0]

        fig_clim = pdict['zrange']
        out = flex_colormesh_plot_vs_xy(ax=axs,clim=fig_clim,cmap=plot_cmap,
                             xvals=plot_xvals,
                             yvals=plot_yvals,
                             zvals=plot_zvals.transpose())

        xmin, xmax = plot_xvals.min()-plot_xvals_step/2., plot_xvals.max()+plot_xvals_step/2.
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        ymin, ymax = plot_yvals.min()-plot_yvals_step/2.,plot_yvals.max()+plot_yvals_step/2.
        axs.set_ylabel(plot_ylabel)
        axs.set_ylim(ymin, ymax)

        axs.set_title(plot_title)

        ax_divider = make_axes_locatable(axs)
        cax = ax_divider.append_axes('right',size='5%', pad='2.5%')
        cbar = plt.colorbar(out['cmap'],cax=cax)
        cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
        cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
        cbar.set_label(plot_zlabel)

        self.cbar = cbar

        axs.figure.tight_layout()



    def plot_colorx(self, pdict, axs=None):
        plot_xvals = pdict['xvals']
        plot_yvals = pdict['yvals']
        plot_zvals = pdict['zvals']
        plot_xlabel = pdict['xlabel']
        plot_ylabel = pdict['ylabel']
        plot_zlabel = pdict['zlabel']
        plot_title = pdict['title']
        plot_cmap = pdict.get('cmap','YlGn')

        plot_xvals_step = plot_xvals[1]-plot_xvals[0]
        plot_yvals_step = plot_yvals[1,0]-plot_yvals[0,0]

        fig_clim = pdict['zrange']
        out = flex_color_plot_vs_x(ax=axs,cmap=plot_cmap,normalize=False,
                             xvals=plot_xvals,
                             yvals=plot_yvals,
                             zvals=plot_zvals.transpose())

        xmin, xmax = plot_xvals.min()-plot_xvals_step/2., plot_xvals.max()+plot_xvals_step/2.
        axs.set_xlabel(plot_xlabel)
        axs.set_xlim(xmin, xmax)

        ymin, ymax = plot_yvals.min()-plot_yvals_step/2.,plot_yvals.max()+plot_yvals_step/2.
        axs.set_ylabel(plot_ylabel)
        axs.set_ylim(ymin, ymax)

        axs.set_title(plot_title)

        # ax_divider = make_axes_locatable(axs)
        # cax = ax_divider.append_axes('right',size='5%', pad='2.5%')
        # cbar = plt.colorbar(out['cmap'], cax=cax)
        # cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
        # cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
        # cbar.set_label(plot_zlabel)

        # self.cbar = cbar

        axs.figure.tight_layout()

class quick_analysis(Standard_MA):
    def __init__(self, t_start, t_stop,
                 options_dict,
                 extract_only=False,
                 auto=True,
                 params_dict_TD=None,
                 numeric_params=None):
        super(quick_analysis, self).__init__(t_start, t_stop=t_stop,
                                             options_dict=options_dict,
                                             extract_only=extract_only)
        self.params_dict_TD = params_dict_TD

        self.numeric_params = numeric_params
        if auto is True:
            self.run_analysis()
    def process_data(self):
        pass
