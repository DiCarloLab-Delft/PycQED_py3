import logging
from copy import copy
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis.tools.data_manipulation as dm_tools


class Single_Qubit_RoundsToEvent_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 extract_metadata: bool=True,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
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

        post_sel_th_anc  (float)
        post_sel_th_data (float)
        dig_th_anc       (float)
        dig_th_data      (float)

        exp_pat_0       (str)
        exp_pat_1       (str)


        nr_of_meas (int) : number of measurement per prepared state.
            used to determine the period for data binning. Includes
            the initialization measurement.


    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label:str ='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, data_file_path=data_file_path,
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

        # Shorthand reference to proc_data_dict to keep code more readable
        dat_dict = self.proc_data_dict

        ch_idx_anc = self.options_dict.get('ch_idx_anc', 0)
        ch_idx_data = self.options_dict.get('ch_idx_data', 1)
        nr_of_meas = self.options_dict.get('nr_of_meas', 5)
        post_sel_th_anc = self.options_dict.get('post_sel_th_anc', 0)
        post_sel_th_data = self.options_dict.get('post_sel_th_data', 0)

        dig_th_anc = self.options_dict.get('dig_th_anc', post_sel_th_anc)
        dig_th_data = self.options_dict.get('dig_th_anc', post_sel_th_data)


        post_select = self.options_dict.get('post_select', True)

        nr_bins = self.options_dict.get('nr_bins', 100)

        shots_anc = list(self.raw_data_dict['measured_data'].values())\
            [ch_idx_anc][0]
        shots_data = list(self.raw_data_dict['measured_data'].values())\
            [ch_idx_data][0]

        dat_dict['shots_anc'] = shots_anc
        dat_dict['shots_data'] = shots_data

        # Data binning
        (prep_0_anc, meas_0_anc, trace_0_anc, prep_1_anc, meas_1_anc,
            trace_1_anc) = repeated_parity_data_binning(
            dat_dict['shots_anc'], nr_of_meas)
        (prep_0_data, meas_0_data, trace_0_data, prep_1_data, meas_1_data,
            trace_1_data) = repeated_parity_data_binning(
            dat_dict['shots_data'], nr_of_meas)


        dat_dict['trace_0_anc'] = trace_0_anc
        # Digitizing traces
        dat_dict['trace_0_anc_dig'] = dm_tools.digitize(
            trace_0_anc, dig_th_anc, zero_state=0)
        dat_dict['trace_0_data_dig'] = dm_tools.digitize(
            trace_0_data, dig_th_data, zero_state=0)
        dat_dict['trace_1_anc_dig'] = dm_tools.digitize(
            trace_1_anc, dig_th_anc, zero_state=0)
        dat_dict['trace_1_data_dig'] = dm_tools.digitize(
            trace_1_data, dig_th_data, zero_state=0)

        # post selection
        if post_select:
            dat_dict['post_sel_idx_0'] = dm_tools.get_post_select_indices(
                (post_sel_th_anc, post_sel_th_data), (prep_0_anc, prep_0_data))
            dat_dict['post_sel_idx_1'] = dm_tools.get_post_select_indices(
                (post_sel_th_anc, post_sel_th_data), (prep_1_anc, prep_1_data))

            for arr in [meas_0_anc, meas_0_data, trace_0_anc, trace_0_data]:
                arr[dat_dict['post_sel_idx_0']] = np.nan
            for arr in [meas_1_anc, meas_1_data, trace_1_anc, trace_1_data]:
                arr[dat_dict['post_sel_idx_1']] = np.nan

        # Counting the rounds to event
        dat_dict['RTE_0'] = count_RTE(dat_dict['trace_0_anc_dig'],
                                      exp_pattern='alternating', init_state=0)
        dat_dict['RTE_1'] = count_RTE(dat_dict['trace_1_anc_dig'],
                                      exp_pattern='constant', init_state=0)

        dat_dict['RTE_0_ps'] = copy(dat_dict['RTE_0'])
        dat_dict['RTE_0_ps'][dat_dict['post_sel_idx_0']] = np.nan
        dat_dict['RTE_1_ps'] = copy(dat_dict['RTE_1'])
        dat_dict['RTE_1_ps'][dat_dict['post_sel_idx_1']] = np.nan

        dat_dict['mRTE_0_ps'] = np.nanmean(dat_dict['RTE_0_ps'])
        dat_dict['mRTE_1_ps'] = np.nanmean(dat_dict['RTE_1_ps'])



        # Histogramming
        dat_dict['prep_0_anc_hist'] = np.histogram(
            prep_0_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))
        dat_dict['prep_1_anc_hist'] = np.histogram(
            prep_1_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))

        dat_dict['meas_0_anc_hist'] = np.histogram(
            meas_0_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))
        dat_dict['meas_1_anc_hist'] = np.histogram(
            meas_1_anc, bins=nr_bins,
            range=(np.min(shots_anc), np.max(shots_anc)))

        dat_dict['prep_0_data_hist'] = np.histogram(
            prep_0_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        dat_dict['prep_1_data_hist'] = np.histogram(
            prep_1_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        dat_dict['meas_0_data_hist'] = np.histogram(
            meas_0_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))
        dat_dict['meas_1_data_hist'] = np.histogram(
            meas_1_data, bins=nr_bins,
            range=(np.min(shots_data), np.max(shots_data)))


        # Histogramming RTE

        dat_dict['prep_0_RTE_hist'] = np.histogram(
            dat_dict['RTE_0_ps'], bins=nr_of_meas+1,
            range=(-0.5, nr_of_meas+0.5))
        dat_dict['prep_1_RTE_hist'] = np.histogram(
            dat_dict['RTE_1_ps'], bins=nr_of_meas+1,
            range=(-0.5, nr_of_meas+0.5))


        F_ass_1 = np.sum(meas_1_anc<dig_th_anc) /(
            np.sum(meas_1_anc>dig_th_anc) + np.sum(meas_1_anc<dig_th_anc))
        F_ass_0 = np.sum(meas_0_anc>dig_th_anc) /(
            np.sum(meas_0_anc>dig_th_anc) + np.sum(meas_0_anc<dig_th_anc))

        F_ass_avg = 1-(1-F_ass_0+1-F_ass_1)/2
        self.proc_data_dict['F_ass'] = F_ass_avg

    def prepare_plots(self):
        self.prepare_ancilla_ro_histogram()
        self.prepare_data_ro_histogram()
        self.prepare_typical_trace_plot()
        self.prepare_RTE_histogram()




    def prepare_ancilla_ro_histogram(self):
        #########################################
        # Ancilla qubit figures
        #########################################
        self.plot_dicts['meas_0_anc'] = {
                    'xlabel': 'anc dac value',
                    'title': 'Ancilla qubit histograms' + '\n'+self.timestamps[0],
                    'ylabel': 'Counts',
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas_0_anc_hist'][1],
                    'yvals': self.proc_data_dict['meas_0_anc_hist'][0],
                    'ax_id': '1D_histogram_anc',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                                'edgecolor': 'C0'},
                    'setlabel': 'Data prep. in: |0>'}

        self.plot_dicts['meas_1_anc'] = {
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas_1_anc_hist'][1],
                    'yvals': self.proc_data_dict['meas_1_anc_hist'][0],
                    'ax_id': '1D_histogram_anc',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                                'edgecolor': 'C3'},
                    'setlabel': 'Data prep. in: |1>'}

        if not self.presentation_mode:
            self.plot_dicts['text_msg'] = {
                'ax_id': '1D_histogram_anc',
                'ypos': 0.6,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'text_string': 'F avg. ass.: {:.2f} %'.format(
                    self.proc_data_dict['F_ass']*100)}
            max_cnts = np.max([np.max(self.proc_data_dict['meas_0_anc_hist'][0]),
                               np.max(self.proc_data_dict['meas_1_anc_hist'][0])])
            ps_th = self.options_dict.get('post_sel_th_anc', 0)
            dig_th = self.options_dict.get('dig_th_anc', ps_th)
            self.plot_dicts['post_sel_th_anc'] = {
                'ax_id': '1D_histogram_anc',
                'plotfn': self.plot_vlines,
                'x': ps_th,
                'ymin': 0,
                'ymax': max_cnts*1.05,
                'colors': '.3',
                'linestyles': 'dashed',
                'line_kws': {'linewidth': .8},
                'setlabel': 'Post sel. thres. {:.2f}'.format(ps_th),
                'do_legend': True}

            ps_th = self.options_dict.get('dig_th_anc', 0)
            self.plot_dicts['dig_th_anc'] = {
                'ax_id': '1D_histogram_anc',
                'plotfn': self.plot_vlines,
                'x': dig_th,
                'ymin': 0,
                'ymax': max_cnts*1.05,
                'colors': '.3',
                'linestyles': 'solid',
                'line_kws': {'linewidth': .8},
                'setlabel': 'Digitization thres. {:.2f}'.format(dig_th),
                'do_legend': True}



    def prepare_data_ro_histogram(self):
        #########################################
        # Data qubit figures
        #########################################
        self.plot_dicts['meas_0_data'] = {
                    'xlabel': 'data dac value',
                    'title': 'Data qubit histograms' + '\n'+self.timestamps[0],
                    'ylabel': 'Counts',
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas_0_data_hist'][1],
                    'yvals': self.proc_data_dict['meas_0_data_hist'][0],
                    'ax_id': '1D_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                                'edgecolor': 'C0'},
                    'setlabel': 'Data prep. in: |0>'}
        self.plot_dicts['meas_1_data'] = {
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['meas_1_data_hist'][1],
                    'yvals': self.proc_data_dict['meas_1_data_hist'][0],
                    'ax_id': '1D_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                                'edgecolor': 'C3'},
                    'setlabel': 'Data prep. in: |1>'}

        max_cnts = np.max([np.max(self.proc_data_dict['meas_0_data_hist'][0]),
                           np.max(self.proc_data_dict['meas_1_data_hist'][0])])
        ps_th = self.options_dict.get('post_sel_th_data', 0)
        dig_th = self.options_dict.get('dig_th_data', ps_th)
        self.plot_dicts['post_sel_th_data'] = {
            'ax_id': '1D_histogram_data',
            'plotfn': self.plot_vlines,
            'x': ps_th,
            'ymin': 0,
            'ymax': max_cnts*1.05,
            'colors': '.3',
            'linestyles': 'dashed',
            'line_kws': {'linewidth': .8},
            'setlabel': 'Post sel. thres. {:.2f}'.format(ps_th),
            'do_legend': True}

        ps_th = self.options_dict.get('dig_th_data', 0)
        self.plot_dicts['dig_th_data'] = {
            'ax_id': '1D_histogram_data',
            'plotfn': self.plot_vlines,
            'x': dig_th,
            'ymin': 0,
            'ymax': max_cnts*1.05,
            'colors': '.3',
            'linestyles': 'solid',
            'line_kws': {'linewidth': .8},
            'setlabel': 'Digitization thres. {:.2f}'.format(dig_th),
            'do_legend': True}

    def prepare_typical_trace_plot(self):

        idx = self.options_dict.get('idx_typ_0', 0)
        raw_pat = self.proc_data_dict['trace_0_anc_dig'][idx, :]
        self.plot_dicts['typical_trace_0'] = {
            'plotfn': self.plot_line,
            'ax_id': 'typical_trace',
            'xvals': np.arange(len(raw_pat))+1,
            'yvals': raw_pat,
            'xlabel': 'Measurement #',
            'ylabel': 'Declared state',
            'yrange': (-.05, 1.05),  # FIXME change to ylim in base analysis
            'setlabel': 'Initial state |0>',
            'title': 'Typical trace of Ancilla outcomes',
            'do_legend': True,
            'legend_pos': 'right'}

        idx = self.options_dict.get('idx_typ_1', 1)
        raw_pat = self.proc_data_dict['trace_1_anc_dig'][idx, :]
        self.plot_dicts['typical_trace_1'] = {
            'plotfn': self.plot_line,
            'ax_id': 'typical_trace',
            'xvals': np.arange(len(raw_pat))+1,
            'yvals': raw_pat,
            'xlabel': 'Measurement #',
            'ylabel': 'Declared state',
            'yrange': (-.05, 1.05),  # FIXME change to ylim in base analysis
            'setlabel': 'Initial state |1>',
            'title': 'Typical trace of Ancilla outcomes',
            'do_legend': True,
            'legend_pos': 'right'}

    def prepare_RTE_histogram(self):
        self.plot_dicts['RTE_0'] = {
                    'xlabel': 'Measurement #',
                    'title': 'RTE histogram' + '\n'+self.timestamps[0],
                    'ylabel': 'Counts',
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['prep_0_RTE_hist'][1],
                    'yvals': self.proc_data_dict['prep_0_RTE_hist'][0],
                    'ax_id': 'RTE_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C0',
                                'edgecolor': 'C0'},
                    'setlabel': 'Data prep. in: |0>\n Mean RTE: {:.2f}'.format(
                        self.proc_data_dict['mRTE_0_ps'])}
        self.plot_dicts['RTE_1'] = {
                    'plotfn': self.plot_bar,
                    'xvals': self.proc_data_dict['prep_1_RTE_hist'][1],
                    'yvals': self.proc_data_dict['prep_1_RTE_hist'][0],
                    'ax_id': 'RTE_histogram_data',
                    'bar_kws': {'log': False, 'alpha': .4, 'facecolor': 'C3',
                                'edgecolor': 'C3'},
                    'setlabel': 'Data prep. in: |1>\n Mean RTE: {:.2f}'.format(
                        self.proc_data_dict['mRTE_1_ps']),
                    'do_legend':True}






def repeated_parity_data_binning(shots, nr_of_meas:int):
    """
    Used for data binning of the repeated parity check experiment.
    Assumes the data qubit is alternatively prepared in 0 and 1.

    Args:
        shots (1D array) : array containing all measured values of 1 qubit
        nr_of_meas (int) : number of measurement per prepared state.
            used to determine the period for data binning. Includes
            the initialization measurement.

    Returns
        prep_0  (1D array)  outcomes of the initialization measurement
        meas_0  (1D array)  outcomes of the first measurement
        trace_0 (2D array)  traces

        prep_1  (1D array)
        meas_1  (1D array)
        trace_1 (2D array)
    """

    prep_0 = copy(shots[::nr_of_meas*2])
    meas_0 = copy(shots[1::nr_of_meas*2])

    prep_1 = copy(shots[nr_of_meas::nr_of_meas*2])
    meas_1 = copy(shots[nr_of_meas+1::nr_of_meas*2])

    trace_0 = np.zeros((len(prep_0), nr_of_meas-1))
    trace_1 = np.zeros((len(prep_1), nr_of_meas-1))
    for i in range(len(prep_0)):
        trace_0[i, :] = shots[1+(2*i)*nr_of_meas: (2*i+1)*nr_of_meas]

        trace_1[i, :] = shots[1+(2*i+1)*nr_of_meas: (2*i+2)*nr_of_meas]

    return (prep_0, meas_0, trace_0, prep_1, meas_1, trace_1)


def count_RTE(traces, exp_pattern: str, init_state: int):
    """
    Args:
        traces (list of 1D arrays): of declared states as 0 and 1
        exp_pattern (str) : "constant" or "alternating"
    """
    RTE = np.zeros(len(traces))

    RTE_larger_than_length = False
    if init_state not in (0, 1):
        raise ValueError('Initial state should be 0 or 1')
    if exp_pattern not in ['constant', 'alternating']:
        raise ValueError("exp_pattern should be 'constant' or 'alternating'")

    for i, t in enumerate(traces):
        if exp_pattern == 'constant':
            if init_state == 0:
                try:
                    RTE[i] = min(np.where( t >0.5)[0])+1
                except ValueError:
                    RTE[i] = len(t) + 1
                    RTE_larger_than_length = True

            else:
                try:
                    RTE[i] = min(np.where(t < 0.5)[0])+1
                except ValueError:
                    RTE[i] = len(t)+1
                    RTE_larger_than_length = True


        elif exp_pattern == 'alternating':
            try:
                t = np.append([init_state], t)
                dt = dm_tools.binary_derivative(t)
                RTE[i] = min(np.where(dt<0.5)[0])+1 # Detect first time there is no flip
            except ValueError:
                RTE[i] = len(t)+1
                RTE_larger_than_length = True
    if RTE_larger_than_length:
        print('Sequences with no errors measured')
    return RTE

