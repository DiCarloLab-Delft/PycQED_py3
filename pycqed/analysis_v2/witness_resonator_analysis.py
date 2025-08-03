
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2 import measurement_analysis as ma2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as MA
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis import fitting_models as fit_mods
import lmfit
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)
import logging
import pycqed.measurement.hdf5_data as h5d
import os

import importlib
importlib.reload(ba)
importlib.reload(fit_mods)


class WitnessResAnalysis(ba.BaseDataAnalysis):

    def __init__(self,
                 feedline_attenuation: float = 0,
                 t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.feedline_attenuation = feedline_attenuation
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.raw_data_dict = {}

        param_spec = {'A': ('Analysis/reso_fit/params/A', 'attr:value'),
        'Q': ('Analysis/reso_fit/params/Q', 'attr:value'),
        'Qe': ('Analysis/reso_fit/params/Qe', 'attr:value'),
        'f0': ('Analysis/reso_fit/params/f0', 'attr:value'),
        'phi_0': ('Analysis/reso_fit/params/phi_0', 'attr:value'),
        'phi_v': ('Analysis/reso_fit/params/phi_v', 'attr:value'),
        'slope': ('Analysis/reso_fit/params/slope', 'attr:value'),
        'theta': ('Analysis/reso_fit/params/theta', 'attr:value'),
        'power': ('Instrument settings/VNA', 'attr:power')}

        fit_individual_linescans = self.options_dict.get('fit_individual_linescans', False)

        for ts in self.timestamps:

            if fit_individual_linescans:
                a = ma2.VNA_analysis(auto=True,
                     t_start=ts,
                     options_dict={'fit_options':{'model':'complex'},
                                   'plot_init': False})

            data_fp = a_tools.get_datafilepath_from_timestamp(ts)

            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):

        self.proc_data_dict['Power'] = []
        self.proc_data_dict['Intrinsic_Q'] = []
        self.proc_data_dict['Coupling_Q'] = []
        self.proc_data_dict['Photon_number'] = []

        for ts in self.timestamps:

            freq = self.raw_data_dict[ts]['f0']
            Q = self.raw_data_dict[ts]['Q']
            Qe = self.raw_data_dict[ts]['Qe']
            theta = self.raw_data_dict[ts]['theta']
            power = float(self.raw_data_dict[ts]['power'])

            if power <= self.options_dict.get('low_pow_thrs', -500):
                continue

            Qc = 1/np.real(1/(Qe*np.exp(1j*theta)))
            Qi = 1/(1/Q - 1/Qc)

            if self.options_dict.get('photon_number_plot', False):
                att = self.feedline_attenuation
                h = 6.62e-34
                power_att = power + att
                pow_watts = 10**(power_att/10)/1000
                photon_number = 2/(2*np.pi*h*freq**2) * Q**2/Qc * pow_watts

                self.proc_data_dict['Photon_number'].append(photon_number)

            self.proc_data_dict['Intrinsic_Q'].append(Qi)
            self.proc_data_dict['Coupling_Q'].append(Qc)
            self.proc_data_dict['Power'].append(power)

        self.proc_data_dict['Intrinsic_Q'] = np.array(self.proc_data_dict['Intrinsic_Q'])
        self.proc_data_dict['Coupling_Q'] = np.array(self.proc_data_dict['Coupling_Q'])
        self.proc_data_dict['Power'] = np.array(self.proc_data_dict['Power'])
        self.proc_data_dict['Photon_number'] = np.array(self.proc_data_dict['Photon_number'])

        self.proc_data_dict['freq'] = self.raw_data_dict[self.timestamps[np.argmax(self.proc_data_dict['Power'])]]['f0']

        self.proc_data_dict['Qi_low_pow'] = self.proc_data_dict['Intrinsic_Q'][np.argmin(self.proc_data_dict['Power'])]
        self.proc_data_dict['Qi_high_pow'] = self.proc_data_dict['Intrinsic_Q'][np.argmax(self.proc_data_dict['Power'])]
        self.proc_data_dict['Qc'] = self.proc_data_dict['Coupling_Q'][np.argmax(self.proc_data_dict['Power'])]

        if self.options_dict.get('photon_number_plot', False):

            self.proc_data_dict['Qi_single_photon'] = self.proc_data_dict['Intrinsic_Q'][np.argmin(np.abs(self.proc_data_dict['Photon_number'] - 1))]
            self.proc_data_dict['Qc_single_photon'] = self.proc_data_dict['Coupling_Q'][np.argmin(np.abs(self.proc_data_dict['Photon_number'] - 1))]
            self.proc_data_dict['Pow_single_photon'] = self.proc_data_dict['Power'][np.argmin(np.abs(self.proc_data_dict['Photon_number'] - 1))]     

    def prepare_plots(self):

        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(5,5), nrows=1, ncols=1, dpi=100)
        self.axs_dict['Q_plots_pow'] = axs
        self.figs['Q_plots_pow'] = fig
        self.plot_dicts['Q_plots_pow'] = {
            'plotfn': power_Q_plots,
            'ax_id': 'Q_plots_pow',
            'Power': self.proc_data_dict['Power'],
            'Intrinsic_Q': self.proc_data_dict['Intrinsic_Q'],
            'Coupling_Q': self.proc_data_dict['Coupling_Q'],
            'freq': self.proc_data_dict['freq'],
            'Qi_low_pow': self.proc_data_dict['Qi_low_pow'],
            'Qi_high_pow': self.proc_data_dict['Qi_high_pow'],
            'Qc': self.proc_data_dict['Qc'],
            't_start': self.timestamps[0],
            't_stop': self.timestamps[-1],
            'label': self.labels[0]
        }

        if self.options_dict.get('photon_number_plot', False):

            fig, axs = plt.subplots(figsize=(5,5), nrows=1, ncols=1, dpi=100)
            self.axs_dict['Q_plots_photnum'] = axs
            self.figs['Q_plots_photnum'] = fig
            self.plot_dicts['Q_plots_photnum'] = {
                'plotfn': photnum_Q_plots,
                'ax_id': 'Q_plots_photnum',
                'Photon_number': self.proc_data_dict['Photon_number'],
                'att': self.feedline_attenuation,
                'freq': self.proc_data_dict['freq'],
                'Qi_single_photon': self.proc_data_dict['Qi_single_photon'],
                'Qc_single_photon': self.proc_data_dict['Qc_single_photon'],
                'Pow_single_photon': self.proc_data_dict['Pow_single_photon'],
                'Intrinsic_Q': self.proc_data_dict['Intrinsic_Q'],
                'Coupling_Q': self.proc_data_dict['Coupling_Q'],
                't_start': self.timestamps[0],
                't_stop': self.timestamps[-1],
                'label': self.labels[0]
            }



def power_Q_plots(Power,
                  Intrinsic_Q,
                  Coupling_Q,
                  freq,
                  Qi_low_pow,
                  Qi_high_pow,
                  Qc,
                  t_start,
                  t_stop,
                  label,
                  ax, **kw):
    
    fig = ax.get_figure()
    axs = fig.get_axes()[0]

    axs.plot(Power[:],Intrinsic_Q[:],marker='o',label = 'Qi')
    axs.plot(Power,Coupling_Q,marker='o',label = 'Qc')
    axs.set_ylim(.5e5,3e6)
    axs.set_yscale('log')
    axs.legend(fontsize=12)
    axs.set_xlabel('VNA power (dBm)', fontsize=13)
    axs.set_ylabel('Quality factor', fontsize=13)
    axs.tick_params(which='both', axis='both', labelsize=12, direction='in', top=True, right=True, labelright=False, labelleft=True)
    axs.yaxis.set_label_position('left')
    fig.suptitle(f'Power dependency of quality factor for {label}\nts_start = {t_start} to ts_stop = {t_stop}')
    text_str = f'$f_0$ = {freq/1e9:.4f} GHz\n$Q_i$ @ {min(Power)} dBm = {Qi_low_pow/1e3:.1f}k\n$Q_i$ @ {max(Power)} dBm = {Qi_high_pow/1e3:.1f}k\n$Q_c$ = {Qc/1e3:.1f}k'
    func_str = r'$\mathrm{S}_{21}(f) = A \left(1+\alpha\frac{f-f_0}{f_0}\right)\left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i(\phi_v f + \phi_0)}$'
    fig.text(0.8, 0, text_str, horizontalalignment='center', verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5))
    fig.text(0.03, -0.05, func_str, horizontalalignment='left', verticalalignment='top', fontsize=11)
    fig.tight_layout()


def photnum_Q_plots(Photon_number,
                    att,
                    Intrinsic_Q,
                    Coupling_Q,
                    freq,
                    Qi_single_photon,
                    Qc_single_photon,
                    Pow_single_photon,
                    t_start,
                    t_stop,
                    label,
                    ax, **kw):
    
    fig = ax.get_figure()
    axs = fig.get_axes()[0]

    axs.plot(Photon_number[:],Intrinsic_Q[:],marker='o',label = 'Qi')
    axs.plot(Photon_number,Coupling_Q,marker='o',label = 'Qc')
    axs.set_ylim(.5e5,3e6)
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend(fontsize=12)
    axs.set_xlabel('Photon number', fontsize=13)
    axs.set_ylabel('Quality factor', fontsize=13)
    axs.tick_params(which='both', axis='both', labelsize=12, direction='in', top=True, right=True, labelright=False, labelleft=True)
    axs.yaxis.set_label_position('left')
    fig.suptitle(f'Photon number dependency of quality factor for {label}\nTotal feedline attenuation = {att} dBm\nts_start = {t_start} to ts_stop = {t_stop}')
    text_str = f'$f_0$ = {freq/1e9:.4f} GHz\n$Q_i$ @ 1 photon = {Qi_single_photon/1e3:.1f}k\n$Q_c$ @ 1 photon = {Qc_single_photon/1e3:.1f}k\n'+r'$P_{VNA}$'+f' @ 1 photon = {Pow_single_photon} dBm'
    func_str = r'$\mathrm{S}_{21}(f) = A \left(1+\alpha\frac{f-f_0}{f_0}\right)\left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i(\phi_v f + \phi_0)}$'
    fig.text(0.8, 0, text_str, horizontalalignment='center', verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5))
    fig.text(0.03, -0.05, func_str, horizontalalignment='left', verticalalignment='top', fontsize=10.5)
    
    fig.tight_layout()