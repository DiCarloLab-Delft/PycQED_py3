"""
This file contains examples for the most basic/simplest of analyses.
They only do plotting of the data and can be used as sort of a template
when making more complex analyses.

We distinguish 3 cases (for the most trivial analyses)
- 1D (single or multi-file)
- 2D_single_file
- 2D_multi_file (which inherits from 1D single file)

"""
import numpy as np
import pycqed.analysis_v2.base_analysis as ba


class Basic1DAnalysis(ba.BaseDataAnalysis):
    """
    Basic 1D analysis.

    Creates a line plot for every parameter measured in a set of datafiles.
    Creates a single plot for each parameter measured.

    Supported options_dict keys
        x2 (str)  : name of a parameter that is varied if multiple datasets
                    are combined.
        average_sets (bool)  : if True averages all datasets together.
            requires shapes of the different datasets to be the same.
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
        # self.single_timestamp = False
        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        # x2 is whatever parameter is varied between sweeps
        x2 = self.options_dict.get('x2', None)
        if x2 is not None:
            self.params_dict['x2'] = x2

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_plots(self):
        # assumes that value names are unique in an experiment

        setlabel = self.raw_data_dict.get('x2', self.timestamps)
        if 'x2' in self.options_dict.keys():
            legend_title = self.options_dict.get('x2_label',
                                                 self.options_dict['x2'])
        else:
            legend_title = 'timestamp'

        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):

            yvals = self.raw_data_dict['measured_data'][val_name]

            if self.options_dict.get('average_sets', False):
                xvals =  self.raw_data_dict['xvals'][0]
                yvals = np.mean(yvals, axis=0)
                setlabel = ['Averaged data']
            else:
                xvals =  self.raw_data_dict['xvals']

            if len(np.shape(yvals)) == 1:
                do_legend = False
            else:
                do_legend = True

            self.plot_dicts[val_name] = {
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': yvals,
                'ylabel': val_name,
                'yrange': self.options_dict.get('yrange', None),
                'xrange': self.options_dict.get('xrange', None),
                'yunit': self.raw_data_dict['value_units'][0][i],
                'setlabel': setlabel,
                'legend_title': legend_title,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': do_legend,
                'legend_pos': 'upper right'}


class Basic2DAnalysis(Basic1DAnalysis):
    """
    Extracts a 2D dataset from a set of 1D scans and plots the data.

    Special options dict kwargs
        "x2"  specifies the name of the parameter varied between the different
              linescans.
    """

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        super().prepare_plots()
        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):
            self.plot_dicts[val_name]['cmap'] = 'viridis'

            if 'x2' in self.raw_data_dict.keys():
                xvals = self.raw_data_dict['x2']
                x2 = self.options_dict['x2']
                xlabel = self.options_dict.get('x2_label', x2)
                xunit = self.options_dict.get('x2_unit', '')
            else:
                xvals = np.arange(len(self.raw_data_dict['xvals']))
                xlabel = 'Experiment idx'
                xunit = ''

            self.plot_dicts[val_name+"_heatmap"] = {
                'plotfn': self.plot_colorx,
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,

                'yvals': self.raw_data_dict['xvals'],
                'ylabel': self.raw_data_dict['xlabel'][0],
                'yunit': self.raw_data_dict['xunit'][0][0],

                'zvals': self.raw_data_dict['measured_data']
                [val_name],
                'clabel': val_name,
                'zunit': self.raw_data_dict['value_units'][0][i],

                'cmap': 'viridis',
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': True,
                'legend_pos': 'upper right'}
