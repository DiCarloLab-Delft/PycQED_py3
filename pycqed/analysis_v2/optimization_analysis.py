import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.measurement.hdf5_data as h5d
import matplotlib.pyplot as plt
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import os
from pycqed.analysis import measurement_analysis as ma


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
            yvals = self.raw_data_dict['measured_values_ord_dict'][val_name][0]

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

class Gaussian_OptimizationAnalysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True, minimize: bool=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.options_dict['save_figs'] = False
        self.minimize = minimize
        self.numeric_params = []
        if auto:
            self.run_analysis()
            self.save_figures(savedir=self.raw_data_dict['folder'][-1], key_list=[list(self.figs)[-1]], tag_tstamp=None)
            for i, ts in enumerate(self.timestamps):
                self.save_figures(savedir=self.raw_data_dict['folder'][i], key_list=[list(self.figs)[i]], tag_tstamp=None)

    def extract_data(self):
        self.raw_data_dict = dict()
        self.raw_data_dict['timestamps'] = list()
        self.raw_data_dict['folder'] = list()
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start, self.t_stop,label=self.labels)
        for ts in self.timestamps:
            data_fp = ma.a_tools.get_datafilepath_from_timestamp(ts)
            param_spec = {'data_settings': ('Experimental Data', 'attr:all_attr'),
                          'data': ('Experimental Data/Data', 'dset'),
                         'optimization_settings': ('Optimization settings', 'attr:all_attr')}
            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(data_fp, param_spec)
            # Parts added to be compatible with base analysis data requirements
            self.raw_data_dict['timestamps'].append(ts)
            self.raw_data_dict['folder'].append(os.path.split(data_fp)[0])

    def process_data(self):
        self.proc_data_dict = dict()
        for ts in self.timestamps:
            self.proc_data_dict[ts] = dict()
            self.proc_data_dict[ts]['function_values'] = dict()
            self.proc_data_dict[ts]['parameter_values'] = dict()
            for i, func_name in enumerate(self.raw_data_dict[ts]['data_settings']['value_names']):
                self.proc_data_dict[ts]['function_values'][func_name.decode("utf-8") ] = self.raw_data_dict[ts]['data'][:,i+len(self.raw_data_dict[ts]['data_settings']['sweep_parameter_names'])]
            for i, parameter in enumerate(self.raw_data_dict[ts]['data_settings']['sweep_parameter_names']):
                self.proc_data_dict[ts]['parameter_values'][parameter.decode("utf-8") ] = self.raw_data_dict[ts]['data'][:,i]
            self.proc_data_dict[ts]['function_units'] = [unit.decode("utf-8") for unit in list(self.raw_data_dict[ts]['data_settings']['value_units'])]
            self.proc_data_dict[ts]['parameter_units'] = [unit.decode("utf-8") for unit in list(self.raw_data_dict[ts]['data_settings']['sweep_parameter_units'])]
            if self.minimize:
                self.proc_data_dict[ts]['optimal_values'] = {func_name:min(self.proc_data_dict[ts]['function_values'][func_name]) for func_name in self.proc_data_dict[ts]['function_values'].keys()}
                func_idx = np.argmin(self.proc_data_dict[ts]['function_values'][list(self.proc_data_dict[ts]['function_values'])[0]])
            else:
                self.proc_data_dict[ts]['optimal_values'] = {func_name:max(self.proc_data_dict[ts]['function_values'][func_name]) for func_name in self.proc_data_dict[ts]['function_values'].keys()}
                func_idx = np.argmax(self.proc_data_dict[ts]['function_values'][list(self.proc_data_dict[ts]['function_values'])[0]])
            self.proc_data_dict[ts]['optimal_parameters'] = {param_name:self.proc_data_dict[ts]['parameter_values'][param_name][func_idx] for param_name in self.proc_data_dict[ts]['parameter_values'].keys()}

    def prepare_plots(self):
        self.plot_dicts = dict()
        # assumes that value names are unique in an experiment
        for i, ts in enumerate(self.timestamps):
            self.plot_dicts['Gaussian_optimization_{}'.format(ts)] = {
                            'plotfn': plot_gaussian_optimization,
                            'optimization_dict': self.proc_data_dict[ts],
                            'numplotsy': len(list(self.proc_data_dict[ts]['function_values'])+list(self.proc_data_dict[ts]['parameter_values'])),
                            'presentation_mode': True
                            }
        self.plot_dicts['Compare_optimizations'] = {
                        'plotfn': plot_gaussian_optimization,
                        'optimization_dict': self.proc_data_dict,
                        'compare': True,
                        'compare_labels':self.options_dict.get('compare_labels'),
                        'numplotsy': 1,
                        'presentation_mode': True
                        }

def plot_gaussian_optimization(optimization_dict, ax=None, figsize=None, compare=False, compare_labels=None, **kw):
    if 'function_values' not in list(optimization_dict):
        compare = True
    if compare:
        timestamps = list(optimization_dict)
        parameters = []
        functions = list(optimization_dict[timestamps[-1]]['function_values'])
    else:
        parameters = list(optimization_dict['parameter_values'])
        functions = list(optimization_dict['function_values'])
    if figsize == None:
        figsize = (10, 5*len(parameters+functions))
    if ax is None:
        fig, ax = plt.subplots(len(functions+parameters), figsize=figsize)
    else:
        if isinstance(ax, np.ndarray):
            fig = ax[0].get_figure()
            if len(ax) != len(parameters+functions):
                for i in range(len(ax),len(parameters+functions)):
                    fig.add_subplot(i+1,1,i+1)
        else:
            fig = ax.get_figure()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

    for i, axis in enumerate(fig.get_axes()):
        if i < len(functions):
            if compare:
                for l, ts in enumerate(timestamps):
                    if compare_labels == None:
                        label = ts
                    else:
                        label = compare_labels[l]
                    y_val = optimization_dict[ts]['function_values'][functions[i]]
                    x_val = np.arange(len(y_val))
                    median_range=max(int(len(x_val)*0.01),2)
                    opt_idx = np.where(y_val==optimization_dict[ts]['optimal_values'][functions[i]])[0][0]
                    axis.plot(x_val, np.array([np.median(y_val[max(k-median_range,0):k+median_range]) for k in range(len(x_val))]), zorder=3)
                    axis.scatter(opt_idx, optimization_dict[ts]['optimal_values'][functions[i]],
                               color=axis.get_lines()[-1].get_color(), edgecolor='black', s=100,
                               marker='*', zorder=4, label='{}_{}={}'.format(functions[i], label, round(optimization_dict[ts]['optimal_values'][functions[i]],2)))
                axis.set_ylabel('{} ({})'.format(functions[i], optimization_dict[timestamps[-1]]['function_units'][i]))
            else:
                y_val = optimization_dict['function_values'][functions[i]]
                x_val = np.arange(len(y_val))
                median_range=max(int(len(x_val)*0.01),2)
                opt_idx = np.where(y_val==optimization_dict['optimal_values'][functions[i]])[0][0]
                axis.plot(x_val, y_val, zorder=1)
                axis.scatter(x_val, y_val, s=20, zorder=2)
                axis.plot(x_val, np.array([np.median(y_val[max(k-median_range,0):k+median_range]) for k in range(len(x_val))]), zorder=3)
                axis.scatter(opt_idx, optimization_dict['optimal_values'][functions[i]],
                           color='yellow', edgecolor='black', s=100, marker='*', zorder=4, label='{}={}'.format(functions[i],round(optimization_dict['optimal_values'][functions[i]],2)))
                axis.set_ylabel('{} ({})'.format(functions[i], optimization_dict['function_units'][i]))
            axis.set_xlabel('iterations (#)')
            axis.legend()
        else:
            j = i-len(functions)
            y_val = optimization_dict['parameter_values'][parameters[j]]
            axis.plot(x_val, y_val)
            axis.scatter(x_val, y_val, s=20, zorder=2)
            axis.plot(x_val, np.array([np.median(y_val[max(k-median_range,0):k+median_range]) for k in range(len(x_val))]), zorder=3)
            axis.scatter(opt_idx, optimization_dict['optimal_parameters'][parameters[j]],
                       color='yellow', edgecolor='black', s=100, marker='*', zorder=4, label='{}={}'.format(parameters[j],round(optimization_dict['optimal_parameters'][parameters[j]],2)))
            axis.set_xlabel('iterations (#)')
            axis.set_ylabel('{} ({})'.format(parameters[j], optimization_dict['parameter_units'][j]))
            axis.legend()
    return fig, fig.get_axes()