import logging
from copy import copy
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis.tools.data_manipulation as dm_tools


class Single_Qubit_RoundsToEvent_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 extract_metadata: bool=True,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False
        self.extract_metadata = extract_metadata
        exp_meta_str = 'Experimental Data.Experimental Metadata.'
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values'}
        if self.extract_metadata:
            self.params_dict['depletion_time'] = exp_meta_str+'depletion_time'
            self.params_dict['net_gate'] = exp_meta_str+'net_gate'
            self.params_dict['sequence_type'] = exp_meta_str+'sequence_type'
            self.params_dict['feedback'] = exp_meta_str+'feedback'

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        """
        # this option is used when the data is not extracted from the datafiles
        net_gate = self.raw_data_dict.get('net_gate', ['pi'])
        feedback = self.raw_data_dict.get('feedback', [False])[0]

        # N.B. flipping is the default (net_gate = pi-pulse)
        net_pulse_pat = 'flipping'
        if not len(set(net_gate)) == 1:
            logging.warning('Different net pulses in dataset.')
        if net_gate[0] == 'i':
            net_pulse_pat = 'constant'
        if feedback:
            net_pulse_pat = 'FB_to_ground'

        exp_pattern = self.options_dict.get('exp_pattern', net_pulse_pat)

        raw_dat = self.raw_data_dict['measured_values']
        nr_expts = self.raw_data_dict['nr_experiments']
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

        title_map = {'constant': 'Typical trace constant',
                     'FB_to_ground': 'Typical trace feedback',
                     'flipping': 'Typical trace alternating'}

        self.plot_dicts['typical_trace'] = {
            'plotfn': self.plot_line,
            'xvals': np.arange(len(raw_pat)),
            'yvals': raw_pat,
            'xlabel': 'Measurement #',
            'ylabel': 'Declared state',
            'yrange': (-0.05, 1.05),  # FIXME change to ylim in base analysis
            'setlabel': 'Raw trace',
            'title': title_map[self.proc_data_dict['exp_pattern']],
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
            'setlabel': 'Error', 'do_legend': True,
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






class One_Qubit_Paritycheck_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for the (repeated) parity check or indirect measurement
    of a single qubit corresponding to the following circuit

                          ^(N)
    Data: M -|-(x)-0-----|-M
             |     |     |
    Anc.: M -|--X--0-X-M-|-M

    This analysis gives

    relevant options_dict parameters
        ch_idx_anc (int) specifies the readout channel for the ancilla qubit
        ch_idx_data (int) specifies the readout channel for the data qubit.

        post_sel_th_anc (float)
        post_sel_th_data (float)
        nr_of_measurements (int) The number of repetitions of the circuit,
            used in the data binning.


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

    def process_data(self):
        ch_idx_anc = self.options_dict.get('ch_idx_anc', 0)
        ch_idx_data = self.options_dict.get('ch_idx_data', 1)
        nr_of_measurements = self.options_dict.get('nr_of_measurements', 11)
        post_sel_th_anc = self.options_dict.get('post_sel_th_anc', 0)
        post_sel_th_data = self.options_dict.get('post_sel_th_data', 0)

        ass_th_anc = self.options_dict.get('ass_th_anc', post_sel_th_anc)


        post_select = self.options_dict.get('post_select', True)

        nr_bins = self.options_dict.get('nr_bins', 100)



        shots_anc = list(self.raw_data_dict['measured_values_ord_dict'].values())\
            [ch_idx_anc][0]
        shots_data = list(self.raw_data_dict['measured_values_ord_dict'].values())\
            [ch_idx_data][0]

        self.proc_data_dict['shots_anc'] = shots_anc
        self.proc_data_dict['shots_data'] = shots_data

        # Data binning
        prep0_anc = copy(shots_anc[::nr_of_measurements*2])
        meas0_anc = copy(shots_anc[1::nr_of_measurements*2])
        prep1_anc = copy(shots_anc[nr_of_measurements::nr_of_measurements*2])
        meas1_anc = copy(shots_anc[nr_of_measurements+1::nr_of_measurements*2])


        prep0_data = copy(shots_data[::nr_of_measurements*2])
        meas0_data = copy(shots_data[1::nr_of_measurements*2])
        prep1_data = copy(shots_data[nr_of_measurements::nr_of_measurements*2])
        meas1_data = copy(shots_data[nr_of_measurements+1::nr_of_measurements*2])

        # post selection
        if post_select:
            meas0_anc[np.where(prep0_anc > post_sel_th_anc)] = np.nan
            meas0_anc[np.where(prep0_data > post_sel_th_data)] = np.nan
            meas1_anc[np.where(prep1_anc > post_sel_th_anc)] = np.nan
            meas1_anc[np.where(prep1_data > post_sel_th_data)] = np.nan

            meas0_data[np.where(prep0_anc > post_sel_th_anc)] = np.nan
            meas0_data[np.where(prep0_data > post_sel_th_data)] = np.nan
            meas1_data[np.where(prep1_anc > post_sel_th_anc)] = np.nan
            meas1_data[np.where(prep1_data > post_sel_th_data)] = np.nan

        # Histogramming
        self.proc_data_dict['prep0_anc_hist'] = np.histogram(
            prep0_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))
        self.proc_data_dict['prep1_anc_hist'] = np.histogram(
            prep1_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))

        self.proc_data_dict['meas0_anc_hist'] = np.histogram(
            meas0_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))
        self.proc_data_dict['meas1_anc_hist'] = np.histogram(
            meas1_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))

        self.proc_data_dict['prep0_data_hist'] = np.histogram(
            prep0_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        self.proc_data_dict['prep1_data_hist'] = np.histogram(
            prep1_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        self.proc_data_dict['meas0_data_hist'] = np.histogram(
            meas0_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        self.proc_data_dict['meas1_data_hist'] = np.histogram(
            meas1_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))


        F_ass_1 = np.sum(meas1_anc<ass_th_anc) /(np.sum(meas1_anc>ass_th_anc) + np.sum(meas1_anc<ass_th_anc))
        F_ass_0 = np.sum(meas0_anc>ass_th_anc) /(np.sum(meas0_anc>ass_th_anc) + np.sum(meas0_anc<ass_th_anc))

        F_ass_avg = 1-(1-F_ass_0+1-F_ass_1)/2
        self.proc_data_dict['F_ass']=F_ass_avg


    def prepare_plots(self):
        self.plot_dicts['meas0_anc'] = {
                    'xlabel': 'anc dac value',
                    'title': 'Ancilla qubit histograms' + '\n'+self.timestamps[0],
                    'ylabel': 'Counts',
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas0_anc_hist'][1],
                    'yvals': self.proc_data_dict['meas0_anc_hist'][0],
                    'ax_id': '1D_histogram_anc',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                                'edgecolor': 'C0'},
                    'setlabel': 'Data prep. in: |0>'}

        self.plot_dicts['meas1_anc'] = {
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas1_anc_hist'][1],
                    'yvals': self.proc_data_dict['meas1_anc_hist'][0],
                    'ax_id': '1D_histogram_anc',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                                'edgecolor': 'C3'},
                    'setlabel': 'Data prep. in: |1>'}


        self.plot_dicts['text_msg'] = {
            'ax_id': '1D_histogram_anc',
            'ypos': 0.75,
            'plotfn': self.plot_text,
            'box_props': 'fancy',
            'text_string': 'F avg. ass.: {:.2f} %'.format(self.proc_data_dict['F_ass']*100)}

        self.plot_dicts['meas0_data'] = {
                    'xlabel': 'data dac value',
                    'title': 'Data qubit histograms' + '\n'+self.timestamps[0],
                    'ylabel': 'Counts',
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas0_data_hist'][1],
                    'yvals': self.proc_data_dict['meas0_data_hist'][0],
                    'ax_id': '1D_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                                'edgecolor': 'C0'},
                    'setlabel': 'Data prep. in: |0>'}
        self.plot_dicts['meas1_data'] = {
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas1_data_hist'][1],
                    'yvals': self.proc_data_dict['meas1_data_hist'][0],
                    'ax_id': '1D_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                                'edgecolor': 'C3'},
                    'setlabel': 'Data prep. in: |1>'}


        # a.plot_dicts = plot_dicts
        # # Clear the plots using this snippet
        # a.axs_dict = None
        # a.pdict = None
        # a.axs ={}
        # a.figs = {}
        # # end of clear plots snippet

        # a.plot()
        # axanc = a.axs['1D_histogram_anc']
        # axanc.legend()

        # axanc = a.axs['1D_histogram_anc']
        # axanc.legend()

        # a.save_figures(close_figs=False)
