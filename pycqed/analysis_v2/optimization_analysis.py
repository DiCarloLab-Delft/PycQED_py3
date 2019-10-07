import numpy as np
import pycqed.analysis_v2.base_analysis as ba


class OptimizationAnalysis(ba.BaseDataAnalysis):
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



        self.numeric_params = []
        if auto:
            self.run_analysis()


    def prepare_plots(self):
        # assumes that value names are unique in an experiment


        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):
            yvals = self.raw_data_dict['measured_data'][val_name][0]

            self.plot_dicts['{}_vs_iteration'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': np.arange(len(yvals)),
                'xlabel': 'Iteration',
                'xunit': '#',
                'yvals': yvals,
                'ylabel': val_name,
                'yunit': self.raw_data_dict['value_units'][0][i]}

            legend_title = 'timestamp'

        # for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):


        #     self.plot_dicts[val_name] = {
        #         'plotfn': self.plot_line,
        #         'xvals': xvals,
        #         'xlabel': self.raw_data_dict['xlabel'][0],
        #         'xunit': self.raw_data_dict['xunit'][0][0],
        #         'yvals': yvals,
        #         'ylabel': val_name,
        #         'yrange': self.options_dict.get('yrange', None),
        #         'xrange': self.options_dict.get('xrange', None),
        #         'yunit': self.raw_data_dict['value_units'][0][i],
        #         'setlabel': setlabel,
        #         'legend_title': legend_title,
        #         'title': (self.raw_data_dict['timestamps'][0]+' - ' +
        #                   self.raw_data_dict['timestamps'][-1] + '\n' +
        #                   self.raw_data_dict['measurementstring'][0]),
        #         'do_legend': do_legend,
        #         'legend_pos': 'upper right'}