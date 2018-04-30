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
import datetime
import os
from pycqed.analysis import analysis_toolbox as a_tools

import numpy as np
import matplotlib.pyplot as plt


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

        self.label_pattern = label_pattern
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
        ts = []
        for i, tq in enumerate(self.qubit_labels):
            for j, rq in enumerate(self.qubit_labels):
                ra = self.ra[i, j]
                ra.extract_data()
                ts.append(np.max(ra.raw_data_dict['datetime']))

        youngest = np.max(ts)
        youngest += datetime.timedelta(seconds=1)

        self.raw_data_dict = OrderedDict()
        self.raw_data_dict['datetime'] = [youngest]
        self.raw_data_dict['timestamps'] = [youngest.strftime("%Y%m%d_%H%M%S")]
        self.timestamps = [youngest.strftime("%Y%m%d_%H%M%S")]

        f = '%s_measurement_cross_dephasing_analysis' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')

    def run_fitting(self):
        qubit_labels = self.qubit_labels
        self.fit_dicts = OrderedDict()
        self.fit_res = OrderedDict()
        self.fit_dicts['sigmas'] = np.array(
            [[None] * len(qubit_labels)] * len(qubit_labels), dtype=float)
        self.fit_dicts['sigmas_norm'] = np.array(
            [[None] * len(qubit_labels)] * len(qubit_labels), dtype=float)
        self.fit_dicts['deph_norm'] = np.array(
            [[None] * len(qubit_labels)] * len(qubit_labels), dtype=float)
        self.fit_res['coherence_fit'] = np.array(
            [[None] * len(qubit_labels)] * len(qubit_labels), dtype=object)

        for i, tq in enumerate(qubit_labels):
            for j, rq in enumerate(qubit_labels):
                ra = self.ra[i, j]
                ra.run_analysis()
                self.fit_res['coherence_fit'] = ra.fit_res['coherence_fit']
                self.fit_dicts['sigmas'][i, j] = ra.fit_dicts['coherence_fit']['sigma']
            self.fit_dicts['sigmas_norm'][i,:] = self.fit_dicts['sigmas'][i,:] / self.fit_dicts['sigmas'][i, i]
            self.fit_dicts['deph_norm'][i,:] = self.fit_dicts['sigmas'][i, i] / self.fit_dicts['sigmas'][i,:]

    def prepare_plots(self):
        self.plot_dicts['sigmas'] = {
            'plotfn': self.plot_labeled_2d,
            'title': '',  # todo
            'yvals': self.qubit_labels, 'ylabel': 'Targeted Qubit', 'yunit': '',
            'xvals': self.qubit_labels, 'xlabel': 'Dephased Qubit', 'xunit': '',
            'zvals': self.fit_dicts['sigmas'],
            'zlabel': r'Ramsey Gauss width $\sigma$',
            #'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['sigmas_norm'] = {
            'plotfn': self.plot_labeled_2d,
            'title': 'Normalized by targetted Qubit',  # todo
            'yvals': self.qubit_labels, 'ylabel': 'Targeted Qubit', 'yunit': '',
            'xvals': self.qubit_labels, 'xlabel': 'Dephased Qubit', 'xunit': '',
            'zvals': self.fit_dicts['sigmas_norm'],
            'zlabel': r'Normalized Ramsey Gauss width $\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['deph_norm'] = {
            'plotfn': self.plot_labeled_2d,
            'title': 'Normalized by targetted Qubit',  # todo
            'yvals': self.qubit_labels, 'ylabel': 'Targeted Qubit', 'yunit': '',
            'xvals': self.qubit_labels, 'xlabel': 'Dephased Qubit', 'xunit': '',
            'zvals': self.fit_dicts['deph_norm'],
            'zlabel': r'Normalized Inverse Ramsey Gauss width $\sigma^{-1}$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        for i, tq in enumerate(self.qubit_labels):
            for j, rq in enumerate(self.qubit_labels):
                ra = self.ra[i, j]
                label = self.label_pattern.replace('{TQ}', tq)
                label = label.replace('{RQ}', rq)
                for p in ra.plot_dicts:
                    self.plot_dicts[p+label] = ra.plot_dicts[p]
                    self.plot_dicts[p+label]['ax_id'] = self.plot_dicts[p+label].get('ax_id', '')+label
                    self.plot_dicts[p+label]['title'] = self.plot_dicts[p+label].get('title', 'Coherence ')+label

    def plot_labeled_2d(self, pdict, axs):
        xl = pdict.get('xvals')
        yl = pdict.get('yvals')
        z = pdict.get('zvals')

        xn = np.array(range(len(xl)))+0.5
        yn = np.array(range(len(yl)))+0.5
        pdict['xvals'] = xn
        pdict['yvals'] = yn
        pdict['zrange'] = (0,np.max(z))

        self.plot_colorxy(pdict=pdict, axs=axs)

        axs.yaxis.set_ticklabels(yl)
        axs.yaxis.set_ticks(yn)
        axs.xaxis.set_ticklabels(xl)
        axs.xaxis.set_ticks(xn)

        axs.cbar.set_label(pdict.get('zlabel', ''))