"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis
"""

import logging
import numpy as np
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis.tools.data_manipulation as dm_tools


class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Responsible for creating the histograms based on the raw data
        """
        # Determine the shape of the data to extract wheter to rotate or not
        preselect = self.options_dict.get('preselect', False)
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)

        nr_expts = self.raw_data_dict['nr_experiments']
        self.proc_data_dict['shots_0'] = [''] * nr_expts
        self.proc_data_dict['shots_1'] = [''] * nr_expts
        for i, meas_val in enumerate(self.raw_data_dict['measured_values']):
            dat = meas_val[0]  # hardcoded for 1 channel for now
            sh_0, sh_1 = get_shots_zero_one(
                dat, preselect=preselect, nr_samples=nr_samples,
                sample_0=sample_0, sample_1=sample_1)

            self.proc_data_dict['shots_0'][i] = sh_0
            self.proc_data_dict['shots_1'][i] = sh_1

        self.proc_data_dict['hist_0'] = np.histogram(
            self.proc_data_dict['shots_0'][0], bins=100)
        self.proc_data_dict['hist_1'] = np.histogram(
            self.proc_data_dict['shots_1'][0], bins=100)

        self.proc_data_dict['shots_xlabel'] = \
            self.raw_data_dict['value_names'][0][0]
        self.proc_data_dict['shots_xunit'] = \
            self.raw_data_dict['value_units'][0][0]

    def analyze_fit_results(self):
        # uses the fit results to set values like V_th
        pass

    def prepare_plots(self):
        # # The histograms
        self.plot_dicts['1D_histogram'] = {
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_0'][1],
            'yvals': self.proc_data_dict['hist_0'][0],
            'bar_kws': {'alpha': .4, 'edgecolor': 'C0'},
            'setlabel': 'Shots 0',
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts',
            'title': '1D Histograms'}
        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': self.proc_data_dict['hist_1'][1],
            'yvals': self.proc_data_dict['hist_1'][0],
            'bar_kws': {'alpha': .4, 'edgecolor': 'C3'},
            'setlabel': 'Shots 1', 'do_legend': True,
            'xlabel': self.proc_data_dict['shots_xlabel'],
            'xunit': self.proc_data_dict['shots_xunit'],
            'ylabel': 'Counts'}
        # self.plot_dicts['2D_histogram'] = {
        #     'plotfn': self.plot_line, }

        # self.plot_dicts['1D_cumulative_histogram'] = {
        #     'plotfn': self.plot_line, }


class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
    pass


def get_shots_zero_one(data, preselect: bool=False,
                       nr_samples: int=2, sample_0: int=0, sample_1: int=1):
    if not preselect:
        data_0, data_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)
    else:
        presel_0, presel_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)

        data_0, data_1 = a_tools.zigzag(
            data, sample_0+1, sample_1+1, nr_samples)
    if preselect:
        raise NotImplementedError()
        presel_0 = presel_0*0
        presel_1 = presel_1*0
    return data_0, data_1
