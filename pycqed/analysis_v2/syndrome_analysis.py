import lmfit
import numpy as np
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis.tools.data_manipulation as dm_tools


class Single_Qubit_RoundsToEvent_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False
        self.params_dict = {'measurementstring': 'measurementstring',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        """
        # FIXME: this should be processed_data_dict instead of data_dict
        exp_pattern = self.options_dict.get('exp_pattern', 'alternating')
        raw_dat = self.raw_data_dict['measured_values']
        nr_expts = np.shape(raw_dat)[0]
        raw_pat = [raw_dat[i][0] for i in range(nr_expts)]

        if exp_pattern == 'alternating':
            pat = [dm_tools.binary_derivative(raw_pat[i])
                   for i in range(nr_expts)]
        elif exp_pattern == 'identical':
            pat = raw_pat
        else:
            raise ValueError('exp_pattern {} should '.format(exp_pattern) +
                             'be either alternating or identical')

        events = [dm_tools.binary_derivative(pat[i])
                  for i in range(nr_expts)]
        double_events = [dm_tools.binary_derivative(events[i])
                         for i in range(nr_expts)]

        self.proc_data_dict['pattern'] = pat
        self.proc_data_dict['raw_pattern'] = raw_pat
        self.proc_data_dict['events'] = events
        self.proc_data_dict['double_events'] = double_events
        self.proc_data_dict['frac_zero'] = 1 - np.mean(raw_pat, axis=1)
        self.proc_data_dict['frac_one'] = np.mean(raw_pat, axis=1)
        self.proc_data_dict['frac_no_error'] = 1-np.mean(events, axis=1)
        self.proc_data_dict['frac_single'] = np.mean(events, axis=1)
        self.proc_data_dict['frac_double'] = np.mean(double_events, axis=1)

    def prepare_plots(self):
        pass
        self.plot_dicts['err_frac'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['datetime'],
            'yvals': self.proc_data_dict['frac_single'],
            'ylabel': 'Fraction',
            'yunit': '',
            'yrange': (0, 1),  # FIXME change to ylim in base analysis
            'setlabel': 'Single errors',
            'title': (self.raw_data_dict['timestamps'][0] + ' -  ' +
                      self.raw_data_dict['timestamps'][-1] + '\n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            'legend_pos': 'upper right'}
        self.plot_dicts['double_err_frac'] = {
            'ax_id': 'err_frac',
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['datetime'],
            'yvals': self.proc_data_dict['frac_double'],
            'ylabel': 'Fraction',
            'yunit': '',
            'setlabel': 'Double errors',
            'do_legend': True,
            'legend_pos': 'upper right'}
