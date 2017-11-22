"""
This file contains examples for the most basic/simplest of analyses.
They only do plotting of the data and can be used as sort of a template
when making more complex analyses.

We distinguish 3 cases (for the most trivial analyses)
- 1D (single or multi-file)
- 2D_single_file
- 2D_multi_file (which inherits from 1D single file)

"""
import pycqed.analysis_v2.base_analysis as ba


class Basic1DAnalysis(ba.BaseDataAnalysis):
    """
    Basic 1D analysis.

    Creates a line plot for every parameter measured in a set of datafiles.
    Creates a single plot for each parameter measured.
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
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
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):
            self.plot_dicts[val_name] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['xvals'],
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': self.raw_data_dict['measured_values_ord_dict'][val_name],
                'ylabel': val_name,
                'yunit': self.raw_data_dict['value_units'][0][i],
                'setlabel': self.timestamps,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': True,
                'legend_pos': 'upper right'}


class Basic2DAnalysis(Basic1DAnalysis):
    def prepare_plots(self):
        super.prepare_plots()
        # assumes that value names are unique in an experiment
        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):
            self.plot_dicts[val_name] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['xvals'],
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': self.raw_data_dict['measured_values_ord_dict'][val_name],
                'ylabel': val_name,
                'yunit': self.raw_data_dict['value_units'][0][i],
                'setlabel': self.timestamps,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': True,
                'legend_pos': 'upper right'}


