'''
Hacked together by Rene Vollmer
'''

import datetime
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar

import numpy as np
import lmfit

from pycqed.analysis import analysis_toolbox as a_tools



class RamseyAnalysisSingle(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_Ramsey',
                 options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True, close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )

        self.params_dict = {'clear_scaling_amp': 'RO_lutman.M_amp_RO',
                            'dephasing': 'Analysis.SNR.value',
                            'F_a': 'Analysis.F_a.value',
                            'F_r': 'Analysis.R_r.value',
                            }
        self.numeric_params = ['tau', 'tau_stderr']

        if auto:
            self.run_analysis()

    def run_fitting(self):
        dephasing_data = self.raw_data['dephasing']
        clear_scaling_amp_dephasing = self.raw_data['clear_scaling_amp']
        coherence = 2 * dephasing_data[1, :]

        def gaussian(x, sigma, scale):
            return scale * np.exp(-(x) ** 2 / (2 * sigma ** 2))

        gmodel = lmfit.models.Model(gaussian)
        coherence_fit = gmodel.fit(coherence, sigma=0.07, scale=0.9, x=clear_scaling_amp_dephasing)

        self.fit_res = coherence_fit

    def prepare_plots(self):
        self.plot_dicts['CLEAR_vs_SNR'] = {
            'plotfn': self.plot_fit,
            'fit_res': self.fit_res,
            'xvals': self.raw_data['clear_scaling_amp_SNR'],
            'yvals': self.raw_data['SNR'],
            'marker': 'x',
            'linestyle': '-',
        }
