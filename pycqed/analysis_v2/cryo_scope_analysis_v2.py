"""
Same as cryo_scope analysis but slightly different in data extraction.
In the end there should be only one of these two files.
"""

import lmfit
from collections import OrderedDict
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.waveform_control_CC.waveform as wf
import pycqed.analysis.fitting_models as fit_mods
import numpy as np
from numpy.fft import fft, ifft, fftfreq


# This is the version from before Nov 22 2017
# Should be replaced by the Brian's cryoscope tools (analysis/tools/cryoscope_tools)

class RamZFluxArc(ba.BaseDataAnalysis):
    """
    Analysis for the 2D scan that is used to calibrate the FluxArc.

    Works for linecuts of RamZ experiments.
    important options in options_dict
        x2:      (str) : specifies the parameter to extract for the x-axis
        x2_label (str)
        x2_unit  (str)

        data_key (int) : specifies the
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True,
                 f_demod: float=0, demodulate: bool=False):
        if options_dict is None:
            options_dict = dict()

        self.numeric_params = []
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        # Now actually extract the parameters and the data
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

    def process_data(self):
        data_key = self.options_dict.get(
            'data_key', self.raw_data_dict['value_names'][0][0])
        raw_dat = self.raw_data_dict['measured_values_ord_dict'][data_key]

        self.proc_data_dict['RamZ_Xdat'] = []
        self.proc_data_dict['RamZ_Ydat'] = []
        self.proc_data_dict['times'] = []
        for i, col in enumerate(raw_dat):
            self.proc_data_dict['RamZ_Xdat'].append(col[::2])
            self.proc_data_dict['RamZ_Ydat'].append(col[1::2])
            self.proc_data_dict['times'].append(
                self.raw_data_dict['xvals'][i][::2])

        for phase, dat in zip(['X', 'Y'],
                              [self.proc_data_dict['RamZ_Xdat'],
                               self.proc_data_dict['RamZ_Ydat']]):
            FFT = np.zeros(np.shape(dat))
            for i, vals in enumerate(dat):
                FFT[i, 1:] = np.abs(np.fft.fft(vals))[1:]
            dt = (self.proc_data_dict['times'][0][1] -
                  self.proc_data_dict['times'][0][0])
            freqs = np.fft.fftfreq(len(vals), dt)

            self.proc_data_dict['FFT_{}'.format(phase)] = FFT
            # FIXME: This reshaping is required for the plot 2D
            self.proc_data_dict['FFT_{}_freqs'.format(phase)] = [freqs]*len(self.raw_data_dict['x2'])
            self.proc_data_dict['FFT_{}_freq_unit'.format(phase)] = 'Hz'
            self.proc_data_dict['FFT_{}_zlabel'.format(phase)] = 'Magnitude'
            self.proc_data_dict['FFT_{}_zunit'.format(phase)] = 'a.u.'

            FFT_peak_idx = []
            for i in range(np.shape(self.proc_data_dict['FFT_{}'.format(phase)])[0]):
                FFT_peak_idx.append(np.argmax(
                    self.proc_data_dict['FFT_{}'.format(phase)][i, :len(freqs)//2]))
            self.proc_data_dict['FFT_{}_peak_idx'.format(phase)] = FFT_peak_idx
            self.proc_data_dict['FFT_{}_peak_freqs'.format(phase)] = freqs[FFT_peak_idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        start_fit_idx = self.options_dict.get('start_fit_idx', 0)
        stop_fit_idx = self.options_dict.get('stop_fit_idx', -1)

        # self.fit_dicts['parabola_fit'] = {
        #     'model': lmfit.models.PolynomialModel(degree=2),
        #     'fit_xvals': {'x': self.raw_data_dict['xvals'][start_fit_idx:stop_fit_idx]},
        #     'fit_yvals': {'data':
        #                   self.proc_data_dict['FFT_peak_freqs'][start_fit_idx:stop_fit_idx]}}

    def prepare_plots(self):
        if 'x2' in self.raw_data_dict.keys():
            xvals = self.raw_data_dict['x2']
            x2 = self.options_dict['x2']
            xlabel = self.options_dict.get('x2_label', x2)
            xunit = self.options_dict.get('x2_unit', '')
        else:
            xvals = np.arange(len(self.raw_data_dict['xvals']))
            xlabel = 'Experiment idx'
            xunit = ''

        data_key = self.options_dict.get(
            'data_key', self.raw_data_dict['value_names'][0][0])

        for phase in ['X', 'Y']:
            self.plot_dicts["{}_data_heatmap".format(phase)] = {
                'plotfn': self.plot_colorx,
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,

                'yvals': self.proc_data_dict['times'],
                'ylabel': self.raw_data_dict['xlabel'][0],
                'yunit': self.raw_data_dict['xunit'][0][0],

                'zvals': self.proc_data_dict['RamZ_{}dat'.format(phase)],
                'clabel': data_key,
                'zunit': self.raw_data_dict['value_units'][0][1],

                'cmap': 'viridis',
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          'RamZ_{}_heatmap'.format(phase)),
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts["{}_FFT_heatmap".format(phase)] = {
                'plotfn': self.plot_colorx,
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,

                'yvals': self.proc_data_dict['FFT_{}_freqs'.format(phase)],
                'ylabel': 'Frequency',
                'yunit': self.proc_data_dict['FFT_{}_freq_unit'.format(phase)],

                'zvals': self.proc_data_dict['FFT_{}'.format(phase)],
                'clabel': data_key,
                'zunit': self.raw_data_dict['value_units'][0][1],

                'cmap': 'viridis',
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          'RamZ_{}_heatmap'.format(phase)),
                'do_legend': True,
                'legend_pos': 'upper right'}


        # self.plot_dicts['raw_data'] = {
        #     'plotfn': self.plot_colorxy,
        #     'title': self.timestamps[0] + ' raw data',
        #     'xvals': self.raw_data_dict['xvals'],
        #     'xlabel': self.raw_data_dict['xlabel'],
        #     'xunit': self.raw_data_dict['xunit'],
        #     'yvals': self.raw_data_dict['yvals'],
        #     'ylabel': self.raw_data_dict['ylabel'],
        #     'yunit': self.raw_data_dict['yunit'],
        #     'zvals': raw_dat,
        #     'clabel': self.raw_data_dict['value_names'][0],
        #     'zunit': self.raw_data_dict['value_units'][0],
        #     'do_legend': True, }

        # self.plot_dicts['fourier_data'] = {
        #     'plotfn': self.plot_colorxy,
        #     'title': self.timestamps[0] + ' fourier transformed data',
        #     'xvals': self.raw_data_dict['xvals'],
        #     'xlabel': self.raw_data_dict['xlabel'],
        #     'xunit': self.raw_data_dict['xunit'],

        #     'yvals': self.proc_data_dict['FFT_yvals'],
        #     'ylabel': self.proc_data_dict['FFT_ylabel'],
        #     'yunit': self.proc_data_dict['FFT_yunit'],
        #     'zvals': self.proc_data_dict['FFT'],
        #     'clabel': self.proc_data_dict['FFT_zlabel'],
        #     'zunit': self.proc_data_dict['FFT_zunit'],
        #     'do_legend': True, }

        # self.plot_dicts['fourier_peaks'] = {
        #     'plotfn': self.plot_line,
        #     'xvals': self.raw_data_dict['xvals'],
        #     'yvals': self.proc_data_dict['FFT_peak_freqs'],
        #     'ax_id': 'fourier_data',
        #     'marker': 'o',
        #     'line_kws': {'color': 'C1', 'markersize': 5,
        #                  'markeredgewidth': .2,
        #                  'markerfacecolor': 'None'},
        #     'linestyle': '',
        #     'setlabel': 'Fourier maxima', 'do_legend': True,
        #     'legend_pos': 'right'}

        # if self.do_fitting:
        #     self.plot_dicts['parabola_fit'] = {
        #         'ax_id': 'fourier_data',
        #         'plotfn': self.plot_fit,
        #         'fit_res': self.fit_dicts['parabola_fit']['fit_res'],
        #         'plot_init': self.options_dict['plot_init'],
        #         'setlabel': 'parabola fit',
        #         'do_legend': True,
        #         'line_kws': {'color': 'C3'},

        #         'legend_pos': 'upper right'}
