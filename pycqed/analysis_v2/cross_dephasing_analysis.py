'''
Toolset to analyse measurement-induced Dephasing of qubits

Hacked together by Rene Vollmer
'''

import pycqed
from pycqed.analysis_v2.quantum_efficiency_analysis import RamseyAnalysisSweep
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from collections import OrderedDict
import copy

class CrossDephasingAnalysis(ba.BaseDataAnalysis):
    '''
    Analyses measurement-induced Dephasing of qubits
    '''

    def __init__(self, qubit_labels: list,
    			 t_start: str = None, t_stop: str = None,
                 label_pattern: str = 'ro_amp_sweep_ramsey_trgt_{TQ}_measured_{RQ}',
                 options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label_pattern,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)

        self.qubit_labels = qubit_labels
        self.ra = np.array([[None] * len(qubit_labels)] * len(qubit_labels))
        d = copy.deepcopy(self.options_dict)
        d['save_figs'] = False
        for i, tq in enumerate(qubit_labels):
            for j, rq in enumerate(qubit_labels):
                label = label_pattern.replace('{TQ}', tq).replace('{RQ}', rq)
                self.ra[i, j] = RamseyAnalysisSweep(
                                                t_start=t_start,
                                                t_stop=t_stop,
                                                label=label, options_dict=d,
                                                auto=False, extract_only=True)

        if auto:
            self.run_analysis()

    def extract_data(self):
        pass

    def fit_data(self):
        qubit_labels = self.qubit_labels
        self.fit_dicts = OrderedDict()
        self.fit_dicts['sigmas'] = np.array(
            [[None] * len(qubit_labels)] * len(qubit_labels))
        for i, tq in enumerate(qubit_labels):
            for j, rq in enumerate(qubit_labels):
                ra = self.ra[i, j]
                ra.run_analysis()
                self.fit_dicts['sigmas'][i, j] = ra.fit_dicts['coherence_fit']['sigma']
            self.fit_dicts['sigmas_norm'][i] = self.fit_dicts['sigmas'][i] / self.fit_dicts['sigmas'][i, i]

    def prepare_plots(self):
        self.plot_dicts['sigmas'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': self.qubit_labels, 'ylabel': 'Targeted Qubit', 'yunit': '',
            'xvals': self.qubit_labels, 'xlabel': 'Dephased Qubit', 'xunit': '',
            'zvals': self.fit_dicts['sigmas'],
            'zlabel': r'Ramsey Gauss width $\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['sigmas_norm'] = {
            'plotfn': self.plot_colorxy,
            'title': 'Normalized by targetted Qubit',  # todo
            'yvals': self.qubit_labels, 'ylabel': 'Targeted Qubit', 'yunit': '',
            'xvals': self.qubit_labels, 'xlabel': 'Dephased Qubit', 'xunit': '',
            'zvals': self.fit_dicts['sigmas_norm'],
            'zlabel': r'Ramsey Gauss width $\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
