import lmfit
import numpy as np
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from copy import deepcopy


class Scope_Trace_analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract the intercept of two parameters.

    relevant options_dict parameters
        x_ch_idx (int): specifies x channel default = 1
        y_ch_idx (int): specifies y channel default = 2

        edge_time (float)     : time corresponding to rising edge
        square_length (float) : duration of the square_pulse
        shortest_timescale(float): timescale after rising edge and falling edge
            up to which to ignore when calculating the deviation
        longest_timescale(float): timescale after rising edge and falling edge
            from which to ignore when calculating the deviation
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xvals': 'sweep_points',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()


    def process_data(self):
        """
        selects the relevant acq channel based on "x_ch_idx" and "y_ch_idx"
        specified in the options dict. If x_ch_idx and y_ch_idx are the same
        it will unzip the data.
        """

        # Extracting the basic x and y values
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        x_ch_idx = self.options_dict.get('x_ch_idx', 0)
        y_ch_idx = self.options_dict.get('y_ch_idx', 1)
        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][y_ch_idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][y_ch_idx]

        self.proc_data_dict['xlabel'] = self.raw_data_dict['value_names'][0][x_ch_idx]
        self.proc_data_dict['xunit'] = self.raw_data_dict['value_units'][0][x_ch_idx]

        self.proc_data_dict['xvals'] = list(self.raw_data_dict
            ['measured_data'].values())[x_ch_idx][0]
        self.proc_data_dict['yvals'] = list(self.raw_data_dict
            ['measured_data'].values())[y_ch_idx][0]


        # Detect the rising edge and shift the time axis
        self.proc_data_dict['square_length'] = self.options_dict.get('square_length', 1e-6)
        r_edge_idx = detect_threshold_crossing(
            self.proc_data_dict['yvals'], 0.05)
        edge_time = self.proc_data_dict['xvals'][r_edge_idx]

        print(edge_time)

        stop_time = edge_time + self.proc_data_dict['square_length']
        r_edge_idx = np.argmin(abs(self.proc_data_dict['xvals'] - edge_time))
        f_edge_idx = np.argmin(abs(self.proc_data_dict['xvals'] - stop_time))

        self.proc_data_dict['tvals'] = self.proc_data_dict['xvals'] - edge_time

        # Setting which part of the experiment to ignore when calculating difference
        shortest_timescale = self.options_dict.get('shortest_timescale', 0)
        sh_ign_idx = np.argmin(abs(self.proc_data_dict['xvals'] -
                                shortest_timescale))
        longest_timescale = self.options_dict.get('longest_timescale', 40e-6)
        lo_ign_idx = np.argmin(abs(self.proc_data_dict['xvals'] -
                                longest_timescale))


        # Determine the mean amplitude of the square pulse
        end_of_sq_idx = min(r_edge_idx+lo_ign_idx, f_edge_idx)
        self.proc_data_dict['sq_amp'] = np.mean(
            self.proc_data_dict['yvals'][r_edge_idx+sh_ign_idx:end_of_sq_idx])
        self.proc_data_dict['background_amp'] = np.mean(
            self.proc_data_dict['yvals'][:r_edge_idx])

        # Determine the expected waveform while ignoring the
        self.proc_data_dict['expected_wf'] = np.ones(
            len(self.proc_data_dict['tvals']))*self.proc_data_dict['background_amp']
        self.proc_data_dict['expected_wf'][r_edge_idx:f_edge_idx] = self.proc_data_dict['sq_amp']


        diff_to_exp = self.proc_data_dict['yvals'] - self.proc_data_dict['expected_wf']
        # parts in cost function to ignore
        diff_to_exp[:r_edge_idx+sh_ign_idx] = 0 # part at short timescale
        diff_to_exp[r_edge_idx+lo_ign_idx:f_edge_idx] = 0 # part in pulse for long timescale
        diff_to_exp[f_edge_idx:] = 0  # part before the pulse

        deviation = np.sqrt(np.sum((diff_to_exp)**2))/len(diff_to_exp)
        self.proc_data_dict['diff_to_exp'] = diff_to_exp

        self.proc_data_dict['deviation'] = deviation



    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['tvals'],
            'xlabel': self.proc_data_dict['xlabel'],
            'xunit': self.proc_data_dict['xunit'],
            'yvals': self.proc_data_dict['yvals'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'Scope Trace',
            'marker':'',
            'title': (self.proc_data_dict['timestamps'][0] + ' \n' +
                      self.proc_data_dict['measurementstring'][0]),
            'do_legend': True,
            'legend_pos': 'upper right'}

        self.plot_dicts['expected_wf'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['tvals'],
            'xlabel': self.proc_data_dict['xlabel'],
            'xunit': self.proc_data_dict['xunit'],
            'yvals': self.proc_data_dict['expected_wf'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'Desired waveform',
            'marker':'',
            'do_legend': True,
            'legend_pos': 'upper right'}

        dev_msg = "Deviation from expected {:.4g}".format(
            self.proc_data_dict['deviation'])
        self.plot_dicts['text_msg'] = {
            'ax_id': 'main',
            'ypos': 0.8,
            'plotfn': self.plot_text,
            'box_props': 'fancy',
            'text_string': dev_msg}



        self.plot_dicts['fine'] = deepcopy(self.plot_dicts['main'])
        self.plot_dicts['fine']['ax_id'] = 'fine'
        self.plot_dicts['fine']['yrange'] = (self.proc_data_dict['sq_amp']*.95,
                                             self.proc_data_dict['sq_amp']*1.05)

        self.plot_dicts['fine']['xrange'] = (
            -0.05*self.proc_data_dict['square_length'],
            1.05*self.proc_data_dict['square_length'])

        self.plot_dicts['fine_exp'] = deepcopy(self.plot_dicts['expected_wf'])
        self.plot_dicts['fine_exp']['ax_id'] = 'fine'


        self.plot_dicts['plus_percent'] = {
            'plotfn': self.plot_matplot_ax_method,
            'ax_id': 'fine',
            'func': 'axhline',
            'plot_kws': {'y': self.proc_data_dict['sq_amp']*1.01,
                         'ls':':', 'c':'grey', 'label':r'$\pm$ 1\%'}}
        self.plot_dicts['minus_percent'] = {
            'plotfn': self.plot_matplot_ax_method,
            'ax_id': 'fine',
            'func': 'axhline',
            'plot_kws': {'y': self.proc_data_dict['sq_amp']*0.99,
                         'ls':':', 'c':'grey'}}

        self.plot_dicts['plus_tenth_percent'] = {
            'plotfn': self.plot_matplot_ax_method,
            'ax_id': 'fine',
            'func': 'axhline',
            'plot_kws': {'y': self.proc_data_dict['sq_amp']*1.001,
                         'ls':'--', 'c':'grey', 'label':r'$\pm$ 0.1\%'}}
        self.plot_dicts['minus_tenth_percent'] = {
            'plotfn': self.plot_matplot_ax_method,
            'ax_id': 'fine',
            'func': 'axhline',
            'plot_kws': {'y': self.proc_data_dict['sq_amp']*0.999,
                         'ls':'--', 'c':'grey'}}


        self.plot_dicts['diff_to_exp'] = {
            'plotfn': self.plot_line,
            'ax_id': 'diff_to_exp',
            'xvals': self.proc_data_dict['tvals'],
            'xlabel': self.proc_data_dict['xlabel'],
            'xunit': self.proc_data_dict['xunit'],
            'yvals': self.proc_data_dict['diff_to_exp'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'Difference to desired waveform',
            'marker':'',
            'do_legend': True,
            'legend_pos': 'upper right'}




def detect_threshold_crossing(signal, frac_of_max=0.10):
    """
    Detects the first crossing of some threshold and returns the index
    """

    th = signal > frac_of_max*np.max(signal)
    # marks all but the first occurence of True to False
    th[1:][th[:-1] & th[1:]] = False
    return np.where(th)[0][0]