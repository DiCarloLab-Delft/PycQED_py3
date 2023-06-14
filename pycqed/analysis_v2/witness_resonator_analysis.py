
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
        # 'phi_v': ('Analysis/reso_fit/params/phi_v', 'attr:value'),
        # 'slope': ('Analysis/reso_fit/params/slope', 'attr:value'),
        'theta': ('Analysis/reso_fit/params/theta', 'attr:value'),
        # 'power': ('Instrument settings/VNA', 'attr:power')
                      }

        fit_individual_linescans = self.options_dict.get('fit_individual_linescans', False)

        for ts in self.timestamps:

            if fit_individual_linescans:
                a = VNA_analysis(auto=True,
                     t_start=ts,
                     options_dict={'fit_options':{'model':'complex'},
                                   'plot_init': False})

            data_fp = a_tools.get_datafilepath_from_timestamp(ts)

            if data_fp.split('_')[-3][0] == 'm' and data_fp.split('_')[-3][1] == '0':
                power = 0
            elif data_fp.split('_')[-3][0] == 'm':
                power = -1*float(data_fp.split('_')[-3][1:3])
            elif data_fp.split('_')[-3][0] == '0':
                power = 0
            else:
                power = float(data_fp.split('_')[-3][:2])

            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(data_fp, param_spec)
            self.raw_data_dict[ts]['power'] = power



            # power = data_fp

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

class VNA_analysis(ma2.complex_spectroscopy):
    '''
    Performs as fit to the resonance using data acquired using VNA R&S ZNB 20

    eg. to use complex model:
    ma2.VNA_analysis(t_start='20010101_213600',
            options_dict={'fit_options': {'model': 'complex'}})
    '''
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=True,
                 extract_only=False,
                 auto=True):
        super(VNA_analysis, self).__init__(t_start, t_stop=t_stop,
                                           options_dict=options_dict,
                                           extract_only=extract_only,
                                           auto=auto,
                                           do_fitting=do_fitting)

    def process_data(self):
        super(VNA_analysis, self).process_data()

    def run_fitting(self):
        super().run_fitting()

        freq = self.fit_dicts['reso_fit']['fit_res'].params['f0']
        Q = self.fit_dicts['reso_fit']['fit_res'].params['Q']
        Qe = self.fit_dicts['reso_fit']['fit_res'].params['Qe']
        theta = self.fit_dicts['reso_fit']['fit_res'].params['theta']

        Qc = 1/np.real(1/(Qe.value*np.exp(1j*theta.value)))
        Qi = 1/(1/Q.value - 1/Qc)
        # FIXME: replace if statement with .format_value string which can handle None values as stderr
        if ((freq.stderr==None)):
            msg = '$f_0 = {:.6g}\pm{:.2g}$ MHz\n'.format(freq.value/1e6,freq.value/1e6)
            msg += r'$Q = {:.4g}\pm{:.2g}$ $\times 10^3$'.format(Q.value/1e3,Q.value/1e3)
            msg += '\n'
            msg += r'$Q_c = {:.4g}$ $\times 10^3$'.format(Qc/1e3)
            msg += '\n'
            msg += r'$Q_e = {:.4g}$ $\times 10^3$'.format(Qe.value/1e3)
            msg += '\n'
            msg += r'$Q_i = {:.4g}$ $\times 10^3$'.format(Qi/1e3)

            self.proc_data_dict['complex_fit_msg'] = msg
            # print('Fitting went wrong')
        else:
            msg = '$f_0 = {:.6g}\pm{:.2g}$ MHz\n'.format(freq.value/1e6,freq.stderr/1e6)
            msg += r'$Q = {:.4g}\pm{:.2g}$ $\times 10^3$'.format(Q.value/1e3,Q.stderr/1e3)
            msg += '\n'
            msg += r'$Q_c = {:.4g}$ $\times 10^3$'.format(Qc/1e3)
            msg += '\n'
            msg += r'$Q_e = {:.4g}$ $\times 10^3$'.format(Qe.value/1e3)
            msg += '\n'
            msg += r'$Q_i = {:.4g}$ $\times 10^3$'.format(Qi/1e3)

            self.proc_data_dict['complex_fit_msg'] = msg

    def prepare_plots(self):
        super(VNA_analysis, self).prepare_plots()
        if self.do_fitting:
            self.plot_dicts['reso_fit_amp'] = {
                'ax_id': 'amp',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'output_mod_fn':np.abs,
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

            self.plot_dicts['reso_fit_real'] = {
                'ax_id': 'real',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'output_mod_fn':np.real,
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

            self.plot_dicts['reso_fit_imag'] = {
                'ax_id': 'imag',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'output_mod_fn':np.imag,
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

            self.plot_dicts['reso_fit_phase'] = {
                'ax_id': 'phase',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'output_mod_fn':lambda a: np.unwrap(np.angle(a)),
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

            self.plot_dicts['reso_fit_plane'] = {
                'ax_id': 'plane',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'output_mod_fn': np.imag,
                'output_mod_fn_x': np.real,
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

            self.plot_dicts['plane_text'] = {
                                        'plotfn': self.plot_text,
                                        'text_string': self.proc_data_dict['complex_fit_msg'],
                                        'xpos': 1.05, 'ypos': .6, 'ax_id': 'plane',
                                        'horizontalalignment': 'left'}


    def prepare_fitting(self):
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex.
        fit_options = self.options_dict.get('fit_options', None)
        subtract_background = self.options_dict.get(
            'subtract_background', False)
        if fit_options is None:
            fitting_model = 'hanger'
        else:
            fitting_model = fit_options['model']
        if subtract_background:
            self.do_subtract_background(thres=self.options_dict['background_thres'],
                                        back_dict=self.options_dict['background_dict'])
        if fitting_model == 'hanger':
            fit_fn = fit_mods.SlopedHangerFuncAmplitude
            fit_guess_fn = fit_mods.SlopedHangerFuncAmplitudeGuess
        elif fitting_model == 'simple_hanger':
            fit_fn = fit_mods.HangerFuncAmplitude
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            # TODO HangerFuncAmplitude Guess
        elif fitting_model == 'lorentzian':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.Lorentzian
            # TODO LorentzianGuess
        elif fitting_model == 'complex':
            ## TODO makes something that returns guesspars

            hanger_fit = VNA_analysis(self.timestamps,
                                         do_fitting= True,
                                         options_dict= {'fit_options':
                                                        {'model':'hanger'}},
                                         extract_only= True)
            hanger_fit_res = hanger_fit.fit_dicts['reso_fit']['fit_res']

            complex_guess = {key:{'value':val} for key,val in hanger_fit_res.best_values.items()}
            complex_guess['f0'] = {'value':complex_guess['f0']['value']*1e9,
                        'min':self.proc_data_dict['plot_frequency'][0],
                        'max':self.proc_data_dict['plot_frequency'][-1], 'vary': False}
            complex_guess['Q'] = {'value':complex_guess['Q']['value'],
                        'min':complex_guess['Q']['value']*0.5,
                        'max':complex_guess['Q']['value']*2, 'vary': True}

            complex_guess['Qe']['vary'] = True
            complex_guess['theta']['vary'] = True
            complex_guess['A']['vary'] = True

            # '''
            # Here we estimate the phase velocity and the initial phase based
            # on the first and last 5% of data points in the scan
            # '''
            # num_fit_points = int(len(self.proc_data_dict['plot_frequency'])/100.)
            # phase_data = np.hstack([np.unwrap(self.proc_data_dict['plot_phase'])[:num_fit_points],
            #                 np.unwrap(self.proc_data_dict['plot_phase'][-num_fit_points:])])
            # freq_data = np.hstack([self.proc_data_dict['plot_frequency'][:num_fit_points],
            #                 self.proc_data_dict['plot_frequency'][-num_fit_points:]])
            # lin_pars = np.polyfit(freq_data,phase_data,1)
            # phase_v, phase_0 = lin_pars
            #
            # if np.sign(phase_v)==-1:
            #     min_phase_v = -1.05*np.abs(phase_v)
            #     max_phase_v = -0.95*np.abs(phase_v)
            # else:
            #     min_phase_v = 0.95*np.abs(phase_v)
            #     max_phase_v = 1.05*np.abs(phase_v)
            complex_guess['phi_0'] = {'value':-np.pi/2, 'min':-2*np.pi,
                                            'max':2*np.pi, 'vary': True}

            # complex_guess['phi_v'] = {'value': phase_v, 'min': min_phase_v,
            #                           'max': max_phase_v, 'vary':False}
            # complex_guess['phi_0'] = {'value': phase_0%(2*np.pi), 'min': -2 * np.pi,
            #                           'max': 2 * np.pi, 'vary':False}
            del complex_guess['slope']

            def test_func(f, f0, Q, Qe,A, theta, phi_0):

                slope_corr = 1# (1 + slope * (f - f0) / f0)
                # propagation_delay_corr = np.exp(1j * (phi_v * f + phi_0))
                propagation_delay_corr = np.exp(1j * (phi_0))
                hanger_contribution = (1 - Q / Qe * np.exp(1j * theta) /
                                       (1 + 2.j * Q * (f - f0) / f0))
                S21 = A * slope_corr * hanger_contribution * propagation_delay_corr

                return S21

            # fit_fn = fit_mods.hanger_func_complex_SI
            fit_fn = test_func


        if len(self.raw_data_dict['timestamps']) == 1:
            if fitting_model == 'complex':
                ### do the initial guess
                complex_data = np.add(self.proc_data_dict['real'],1.j*self.proc_data_dict['imag'])
                self.fit_dicts['reso_fit'] = {'fit_fn': fit_fn,
                                              'guess_dict':complex_guess,
                                              'fit_yvals': {'data': complex_data},
                                              'fit_xvals': {'f': self.proc_data_dict['plot_frequency']},
                                              'fitting_type':'minimize'}
            else:
                self.fit_dicts['reso_fit'] = {'fit_fn': fit_fn,
                                              'fit_guess_fn': fit_guess_fn,
                                              'fit_yvals': {'data': self.proc_data_dict['plot_amp']},
                                              'fit_xvals': {'f': self.proc_data_dict['plot_frequency']}
                                              }
        else:
            self.fit_dicts['reso_fit'] = {'fit_fn': fit_fn,
                                          'fit_guess_fn': fit_guess_fn,
                                          'fit_yvals': [{'data': np.squeeze(tt)} for tt in self.plot_amp],
                                          'fit_xvals': np.squeeze([{'f': tt[0]} for tt in self.plot_frequency])}

    def analyze_fit_results(self):
        pass



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
    axs.set_ylim(.5e5,10e6)
    axs.set_yscale('log')
    axs.legend(fontsize=12)
    axs.set_xlabel('VNA power (dBm)', fontsize=13)
    axs.set_ylabel('Quality factor', fontsize=13)
    axs.tick_params(which='both', axis='both', labelsize=12, direction='in', top=True, right=True, labelright=False, labelleft=True)
    axs.yaxis.set_label_position('left')
    fig.suptitle(f'Power dependency of quality factor for {label}\nts_start = {t_start} to ts_stop = {t_stop}')
    text_str = f'$f_0$ = {freq/1e9:.4f} GHz\n$Q_i$ @ {min(Power)} dBm = {Qi_low_pow/1e3:.1f}k\n$Q_i$ @ {max(Power)} dBm = {Qi_high_pow/1e3:.1f}k\n$Q_c$ = {Qc/1e3:.1f}k'
    # func_str = r'$\mathrm{S}_{21}(f) = A \left(1+\alpha\frac{f-f_0}{f_0}\right)\left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i(\phi_v f + \phi_0)}$'
    func_str = r'$\mathrm{S}_{21}(f) = A \left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i\phi_0}$'
    fig.text(0.8, 0, text_str, horizontalalignment='center', verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5))
    fig.text(0.1, -0.05, func_str, horizontalalignment='left', verticalalignment='top', fontsize=11)
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
    axs.set_ylim(.5e5,10e6)
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend(fontsize=12)
    axs.set_xlabel('Photon number', fontsize=13)
    axs.set_ylabel('Quality factor', fontsize=13)
    axs.tick_params(which='both', axis='both', labelsize=12, direction='in', top=True, right=True, labelright=False, labelleft=True)
    axs.yaxis.set_label_position('left')
    fig.suptitle(f'Photon number dependency of quality factor for {label}\nTotal feedline attenuation = {att} dBm\nts_start = {t_start} to ts_stop = {t_stop}')
    text_str = f'$f_0$ = {freq/1e9:.4f} GHz\n$Q_i$ @ 1 photon = {Qi_single_photon/1e3:.1f}k\n$Q_c$ @ 1 photon = {Qc_single_photon/1e3:.1f}k\n'+r'$P_{VNA}$'+f' @ 1 photon = {Pow_single_photon} dBm'
    # func_str = r'$\mathrm{S}_{21}(f) = A \left(1+\alpha\frac{f-f_0}{f_0}\right)\left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i(\phi_v f + \phi_0)}$'
    func_str = r'$\mathrm{S}_{21}(f) = A \left(1 - \frac{Q\mathrm{e}^{i\theta}}{Q_e(1+2iQ(f-f_0)/f_0)}\right)\mathrm{e}^{i\phi_0}$'
    fig.text(0.8, 0, text_str, horizontalalignment='center', verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', pad=.4,
                                    facecolor='white', alpha=0.5))
    fig.text(0.1, -0.05, func_str, horizontalalignment='left', verticalalignment='top', fontsize=10.5)
    
    fig.tight_layout()