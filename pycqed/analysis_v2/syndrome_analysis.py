import logging
import numpy as np
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
        exp_meta_str = 'Experimental Data.Experimental Metadata.'
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'depletion_time': exp_meta_str+'depletion_time',
            'net_gate': exp_meta_str+'net_gate',
            'sequence_type': exp_meta_str+'sequence_type',
            'feedback': exp_meta_str+'feedback'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        """
        # N.B. flipping is the default (net_gate = pi-pulse)
        net_pulse_pat = 'flipping'
        if not len(set(self.raw_data_dict['net_gate'])) == 1:
            logging.warning('Different net pulses in dataset.')
        if self.raw_data_dict['net_gate'][0] == 'i':
            net_pulse_pat = 'constant'
        if self.raw_data_dict['feedback'][0]:
            net_pulse_pat = 'FB_to_ground'

        exp_pattern = self.options_dict.get('exp_pattern', net_pulse_pat)

        raw_dat = self.raw_data_dict['measured_values']
        nr_expts = np.shape(raw_dat)[0]
        raw_pat = [raw_dat[i][0] for i in range(nr_expts)]

        s_events = []
        d_events = []
        for i in range(nr_expts):
            err_marker = getattr(dm_tools,
                                 'mark_errors_{}'.format(exp_pattern))
            events = err_marker(raw_pat[i])
            s_events.append(events[0])
            d_events.append(events[1])

        self.proc_data_dict['raw_pattern'] = raw_pat
        self.proc_data_dict['exp_pattern'] = exp_pattern

        self.proc_data_dict['single_events'] = s_events
        self.proc_data_dict['double_events'] = d_events

        self.proc_data_dict['frac_zero'] = 1 - np.mean(raw_pat, axis=1)
        self.proc_data_dict['frac_one'] = np.mean(raw_pat, axis=1)
        self.proc_data_dict['frac_no_error'] = 1-np.mean(s_events, axis=1)
        self.proc_data_dict['frac_single'] = np.mean(s_events, axis=1)
        self.proc_data_dict['frac_double'] = np.mean(d_events, axis=1)

    def prepare_plots(self):
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
                      'Rounds to event analysis'),
            'do_legend': True,
            'legend_pos': 'upper right'}
        self.plot_dicts['double_err_frac'] = {
            'ax_id': 'err_frac',
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['datetime'],
            'yvals': self.proc_data_dict['frac_double'],
            'setlabel': 'Double errors',
            'do_legend': True,
            'legend_pos': 'upper right'}

        dataset_idx = self.options_dict.get('typ_data_idx', 0)
        start_idx = self.options_dict.get('typ_start_idx', 10)
        stop_idx = self.options_dict.get('typ_stop_idx', 70)
        raw_pat = self.proc_data_dict['raw_pattern'][
            dataset_idx][start_idx:stop_idx]
        s_events = self.proc_data_dict['single_events'][
            dataset_idx][start_idx:stop_idx]
        d_events = self.proc_data_dict['double_events'][
            dataset_idx][start_idx-1:stop_idx-1]

        self.plot_dicts['typical_trace'] = {
            'plotfn': self.plot_line,
            'xvals': np.arange(len(raw_pat)),
            'yvals': raw_pat,
            'xlabel': 'Measurement #',
            'ylabel': 'Declared state',
            'yrange': (-0.05, 1.05),  # FIXME change to ylim in base analysis
            'setlabel': 'Raw trace',
            'title': 'Typical trace',
            'do_legend': True,
            'legend_pos': 'right'}


        if self.proc_data_dict['exp_pattern'] == 'constant':
            event_pat = .5 * np.ones(len(raw_pat))
        else:
            event_pat = raw_pat

        self.plot_dicts['typical_pat_single'] = {
            'ax_id': 'typical_trace',
            'plotfn': self.plot_line,
            'xvals': np.arange(len(raw_pat))[s_events > .5]+.5,
            'yvals': event_pat[s_events > .5],
            'marker': 'x',
            'line_kws': {'color': 'r', 'markersize': 7},
            'linestyle': '',
            'setlabel': 'Single events', 'do_legend': True,
            'legend_pos': 'right'}

        self.plot_dicts['typical_pat_double'] = {
            'ax_id': 'typical_trace',
            'plotfn': self.plot_line,
            'xvals': np.arange(len(raw_pat))[d_events > .5],
            'yvals': raw_pat[d_events > .5],
            'marker': '|',
            'linestyle': '',
            'line_kws': {'markerfacecolor': 'None', 'markeredgecolor': 'red',
                         'markersize': 15},
            'setlabel': 'Double events', 'do_legend': True, 'legend_pos': 'right'}
