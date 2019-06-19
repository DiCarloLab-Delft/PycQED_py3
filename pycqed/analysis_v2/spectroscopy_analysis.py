"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of all the spectroscopy measurement analyses.
"""

import pycqed.analysis_v2.base_analysis as ba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as MA
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis import fitting_models as fit_mods
import pycqed.analysis.fit_toolbox.geometry as geo
import lmfit
import logging
from collections import OrderedDict
from scipy import integrate
import importlib
importlib.reload(ba)


class Spectroscopy(ba.BaseDataAnalysis):

    def __init__(self, t_start: str,
                 t_stop: str = None,
                 options_dict: dict = None,
                 label: str = None,
                 extract_only: bool = False,
                 auto: bool = True,
                 do_fitting: bool = False):
        super(Spectroscopy, self).__init__(t_start=t_start, t_stop=t_stop,
                                           label=label,
                                           options_dict=options_dict,
                                           extract_only=extract_only,
                                           do_fitting=do_fitting)
        self.extract_fitparams = self.options_dict.get('fitparams', False)
        self.params_dict = {'freq_label': 'sweep_name',
                            'freq_unit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'freq': 'sweep_points',
                            'amp': 'amp',
                            'phase': 'phase'}
        
        self.options_dict.get('xwidth', None)
        # {'xlabel': 'sweep_name',
        # 'xunit': 'sweep_unit',
        # 'measurementstring': 'measurementstring',
        # 'sweep_points': 'sweep_points',
        # 'value_names': 'value_names',
        # 'value_units': 'value_units',
        # 'measured_values': 'measured_values'}

        if self.extract_fitparams:
            self.params_dict.update({'fitparams': self.options_dict.get('fitparams_key', 'fit_params')})

        self.numeric_params = ['freq', 'amp', 'phase']
        if 'qubit_label' in self.options_dict:
            self.labels.extend(self.options_dict['qubit_label'])
        sweep_param = self.options_dict.get('sweep_param', None)
        if sweep_param is not None:
            self.params_dict.update({'sweep_param': sweep_param})
            self.numeric_params.append('sweep_param')
        if auto is True:
            self.run_analysis()

    def process_data(self):
        proc_data_dict = self.proc_data_dict
        proc_data_dict['freq_label'] = 'Frequency (GHz)'
        proc_data_dict['amp_label'] = 'Transmission amplitude (arb. units)'

        proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        proc_data_dict['freq_range'] = self.options_dict.get(
            'freq_range', None)
        proc_data_dict['amp_range'] = self.options_dict.get('amp_range', None)
        proc_data_dict['phase_range'] = self.options_dict.get(
            'phase_range', None)
        proc_data_dict['plotsize'] = self.options_dict.get('plotsize', (8, 5))
        if len(self.raw_data_dict['timestamps']) == 1:
            proc_data_dict['plot_frequency'] = np.squeeze(
                self.raw_data_dict['freq'])
            proc_data_dict['plot_amp'] = np.squeeze(self.raw_data_dict['amp'])
            proc_data_dict['plot_phase'] = np.squeeze(
                self.raw_data_dict['phase'])
        else:
            # TRANSPOSE ALSO NEEDS TO BE CODED FOR 2D
            sweep_param = self.options_dict.get('sweep_param', None)
            if sweep_param is not None:
                proc_data_dict['plot_xvals'] = np.array(
                    self.raw_data_dict['sweep_param'])
                proc_data_dict['plot_xvals'] = np.reshape(proc_data_dict['plot_xvals'],
                                                          (len(proc_data_dict['plot_xvals']), 1))
                proc_data_dict['plot_xlabel'] = self.options_dict.get(
                    'xlabel', sweep_param)
            else:
                xvals = np.array([[tt] for tt in range(
                    len(self.raw_data_dict['timestamps']))])
                proc_data_dict['plot_xvals'] = self.options_dict.get(
                    'xvals', xvals)
                proc_data_dict['plot_xlabel'] = self.options_dict.get(
                    'xlabel', 'Scan number')
            proc_data_dict['plot_xwidth'] = self.options_dict.get(
                'xwidth', None)
            if proc_data_dict['plot_xwidth'] == 'auto':
                x_diff = np.diff(np.ravel(proc_data_dict['plot_xvals']))
                dx1 = np.concatenate(([x_diff[0]], x_diff))
                dx2 = np.concatenate((x_diff, [x_diff[-1]]))
                proc_data_dict['plot_xwidth'] = np.minimum(dx1, dx2)
                proc_data_dict['plot_frequency'] = np.array(
                    self.raw_data_dict['freq'])
                proc_data_dict['plot_phase'] = np.array(
                    self.raw_data_dict['phase'])
                proc_data_dict['plot_amp'] = np.array(
                    self.raw_data_dict['amp'])

            else:
                # manual setting of plot_xwidths
                proc_data_dict['plot_frequency'] = self.raw_data_dict['freq']
                proc_data_dict['plot_phase'] = self.raw_data_dict['phase']
                proc_data_dict['plot_amp'] = self.raw_data_dict['amp']

    def prepare_plots(self):
        proc_data_dict = self.proc_data_dict
        plotsize = self.options_dict.get('plotsize')
        if len(self.raw_data_dict['timestamps']) == 1:
            plot_fn = self.plot_line
            self.plot_dicts['amp'] = {'plotfn': plot_fn,
                                      'xvals': proc_data_dict['plot_frequency'],
                                      'yvals': proc_data_dict['plot_amp'],
                                      'title': 'Spectroscopy amplitude: %s' % (self.timestamps[0]),
                                      'xlabel': proc_data_dict['freq_label'],
                                      'ylabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize
                                      }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                                        'xvals': proc_data_dict['plot_frequency'],
                                        'yvals': proc_data_dict['plot_phase'],
                                        'title': 'Spectroscopy phase: %s' % (self.timestamps[0]),
                                        'xlabel': proc_data_dict['freq_label'],
                                        'ylabel': proc_data_dict['phase_label'],
                                        'yrange': proc_data_dict['phase_range'],
                                        'plotsize': plotsize
                                        }
        else:
            self.plot_dicts['amp'] = {'plotfn': self.plot_colorx,
                                      'xvals': proc_data_dict['plot_xvals'],
                                      'xwidth': proc_data_dict['plot_xwidth'],
                                      'yvals': proc_data_dict['plot_frequency'],
                                      'zvals': proc_data_dict['plot_amp'],
                                      'title': 'Spectroscopy amplitude: %s' % (self.timestamps[0]),
                                      'xlabel': proc_data_dict['plot_xlabel'],
                                      'ylabel': proc_data_dict['freq_label'],
                                      'zlabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['freq_range'],
                                      'zrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize,
                                      'plotcbar': self.options_dict.get('colorbar', False),
                                      }

            self.plot_dicts['amp'] = {'plotfn': self.plot_colorx,
                                      'xvals': proc_data_dict['plot_xvals'],
                                      'yvals': proc_data_dict['plot_frequency'],
                                      'zvals': proc_data_dict['plot_amp'],
                                      }

    def plot_for_presentation(self, key_list=None, no_label=False):
        super(Spectroscopy, self).plot_for_presentation(
            key_list=key_list, no_label=no_label)
        for key in key_list:
            pdict = self.plot_dicts[key]
            if key == 'amp':
                if pdict['plotfn'] == self.plot_line:
                    ymin, ymax = 0, 1.2 * np.max(np.ravel(pdict['yvals']))
                    self.axs[key].set_ylim(ymin, ymax)
                    self.axs[key].set_ylabel('Transmission amplitude (V rms)')


class complex_spectroscopy(Spectroscopy):
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True):
        super(complex_spectroscopy, self).__init__(t_start, t_stop=t_stop,
                                                   options_dict=options_dict,
                                                   extract_only=extract_only,
                                                   auto=False,
                                                   do_fitting=do_fitting)
        self.params_dict = {'freq_label': 'sweep_name',
                            'freq_unit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'measured_values': 'measured_values',
                            'freq': 'sweep_points',
                            'amp': 'amp',
                            'phase': 'phase',
                            'real': 'real',
                            'imag': 'imag'}
        self.options_dict.get('xwidth', None)

        if self.extract_fitparams:
            self.params_dict.update({'fitparams': 'fit_params'})

        self.numeric_params = ['freq', 'amp', 'phase', 'real', 'imag']
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess', {})
        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(complex_spectroscopy, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        if len(self.raw_data_dict['timestamps']) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(np.pi / 180. * self.proc_data_dict['plot_phase']) * 180 / np.pi
            self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        else:
            pass
        self.raw_data_dict['real'] = [
            self.raw_data_dict['measured_values'][0][2]]
        self.raw_data_dict['imag'] = [
            self.raw_data_dict['measured_values'][0][3]]
        self.proc_data_dict['real'] = self.raw_data_dict['real'][0]
        self.proc_data_dict['imag'] = self.raw_data_dict['imag'][0]
        self.proc_data_dict['plot_real'] = self.proc_data_dict['real']
        self.proc_data_dict['plot_imag'] = self.proc_data_dict['imag']
        self.proc_data_dict['real_label'] = 'Real{S21} (V rms)'
        self.proc_data_dict['imag_label'] = 'Imag{S21} (V rms)'
        if len(self.raw_data_dict['timestamps']) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(np.pi / 180. * self.proc_data_dict['plot_phase']) * 180 / np.pi
            self.proc_data_dict['plot_xlabel'] = 'Frequency (Hz)'
        else:
            pass

    def prepare_plots(self):
        super(complex_spectroscopy, self).prepare_plots()
        proc_data_dict = self.proc_data_dict
        plotsize = self.options_dict.get('plotsize')
        if len(self.raw_data_dict['timestamps']) == 1:
            plot_fn = self.plot_line
            self.plot_dicts['amp']['title'] = 'S21 amp: %s' % (
                self.timestamps[0])
            self.plot_dicts['amp']['setlabel'] = 'amp'
            self.plot_dicts['phase']['title'] = 'S21 phase: %s' % (
                self.timestamps[0])
            self.plot_dicts['phase']['setlabel'] = 'phase'
            self.plot_dicts['real'] = {'plotfn': plot_fn,
                                       'xvals': proc_data_dict['plot_frequency'],
                                       'yvals': proc_data_dict['plot_real'],
                                       'title': 'S21 amp: %s' % (self.timestamps[0]),
                                       'xlabel': proc_data_dict['freq_label'],
                                       'ylabel': proc_data_dict['real_label'],
                                       'yrange': proc_data_dict['amp_range'],
                                       'plotsize': plotsize
                                       }
            self.plot_dicts['imag'] = {'plotfn': plot_fn,
                                       'xvals': proc_data_dict['plot_frequency'],
                                       'yvals': proc_data_dict['plot_imag'],
                                       'title': 'S21 phase: %s' % (self.timestamps[0]),
                                       'xlabel': proc_data_dict['freq_label'],
                                       'ylabel': proc_data_dict['imag_label'],
                                       'yrange': proc_data_dict['amp_range'],
                                       'plotsize': plotsize
                                       }
            pdict_names = ['amp', 'phase', 'real', 'imag']

            self.figs['combined'], axs = plt.subplots(
                nrows=4, ncols=1, sharex=True, figsize=(8, 6))

            for i, name in enumerate(pdict_names):
                combined_name = 'combined_' + name
                self.axs[combined_name] = axs[i]
                self.plot_dicts[combined_name] = self.plot_dicts[name].copy()
                self.plot_dicts[combined_name]['ax_id'] = combined_name

                # shorter label as the axes are now shared
                self.plot_dicts[combined_name]['ylabel'] = name
                self.plot_dicts[combined_name]['xlabel'] = None if i in [
                    0, 1, 2, 3] else self.plot_dicts[combined_name]['xlabel']
                self.plot_dicts[combined_name]['title'] = None if i in [
                    0, 1, 2, 3] else self.plot_dicts[combined_name]['title']
                self.plot_dicts[combined_name]['touching'] = True


        else:
            raise NotImplementedError('Not coded up yet for multiple traces')


class VNA_analysis(complex_spectroscopy):
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True):
        super(VNA_analysis, self).__init__(t_start, t_stop=t_stop,
                                           options_dict=options_dict,
                                           extract_only=extract_only,
                                           auto=auto,
                                           do_fitting=do_fitting)

    def process_data(self):
        super(VNA_analysis, self).process_data()

    def prepare_plots(self):
        super(VNA_analysis, self).prepare_plots()
        if self.do_fitting:
            self.plot_dicts['reso_fit'] = {
                'ax_id': 'amp',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['reso_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'hanger',
                'line_kws': {'color': 'r'},
                'do_legend': True}

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
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            # hanger_fit = VNA_analysis(self.timestamps,
            #                              do_fitting= True,
            #                              options_dict= {'fit_options':
            #                                             {'model':'hanger'}},
            #                              extract_only= True)
            # hanger_fit_res = hanger_fit.fit_dicts['reso_fit']['fit_res']
            # complex_guess = hanger_fit_res.best_values

            # delta_phase = np.unwrap(self.proc_data_dict['plot_phase'])[-1] - /
            #               np.unwrap(self.proc_data_dict['plot_phase'])[0]
            # delta_freq = self.proc_data_dict['plot_frequency'][-1] - /
            #              self.proc_data_dict['plot_frequency'][0]
            # phase_v = delta_phase/delta_freq
            # fit_fn = fit_mods.SlopedHangerFuncComplex2

            # TODO HangerFuncComplexGuess

        if len(self.raw_data_dict['timestamps']) == 1:
            if fitting_model == 'complex':
                self.fit_dicts['reso_fit'] = {'fit_fn': fit_fn,
                                              'fit_guess_fn': fit_guess_fn,
                                              'fit_yvals': {'data': self.proc_data_dict['plot_amp']},
                                              'fit_xvals': {'f': self.proc_data_dict['plot_frequency']}
                                              }
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


class ResonatorSpectroscopy(Spectroscopy):
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True):
        super(ResonatorSpectroscopy, self).__init__(t_start, t_stop=t_stop,
                                                    options_dict=options_dict,
                                                    extract_only=extract_only,
                                                    auto=False,
                                                    do_fitting=do_fitting)
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess', {})
        self.simultan = self.options_dict.get('simultan', False)

        if self.simultan:
            if not (len(t_start) == 2 and t_stop is None):
                raise ValueError('Exactly two timestamps need to be passed for'
                             ' simultan resonator spectroscopy in ground '
                             'and excited state as: t_start = [t_on, t_off]')


        if auto is True:
            self.run_analysis()


    def process_data(self):
        super(ResonatorSpectroscopy, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        if len(self.raw_data_dict['timestamps']) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(np.pi / 180. *
                              self.proc_data_dict['plot_phase']) * 180 / np.pi
            self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        else:
            pass

    def prepare_fitting(self):
        super().prepare_fitting()
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex, hanger_with_pf.
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
            guess_pars = None
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
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.HangerFuncComplex
            # TODO HangerFuncComplexGuess
        elif fitting_model == 'hanger_with_pf':
            if self.simultan:
                fit_fn = fit_mods.simultan_hanger_with_pf
                self.sim_fit = fit_mods.fit_hanger_with_pf(
                            fit_mods.SimHangerWithPfModel,[
                            np.transpose([self.proc_data_dict['plot_frequency'][0],
                                          self.proc_data_dict['plot_amp'][0]]),
                            np.transpose([self.proc_data_dict['plot_frequency'][1],
                                          self.proc_data_dict['plot_amp'][1]])],
                            simultan=True)
                guess_pars = None
                fit_guess_fn = None
                x_fit_0 = self.proc_data_dict['plot_frequency'][0]

                self.chi = (self.sim_fit[1].params['omega_ro'].value -
                            self.sim_fit[0].params['omega_ro'].value)/2
                self.f_RO_res = (self.sim_fit[0].params['omega_ro'].value+
                                 self.sim_fit[1].params['omega_ro'].value)/2
                self.f_PF = self.sim_fit[0].params['omega_pf'].value
                self.kappa = self.sim_fit[0].params['kappa_pf'].value
                self.J_ = self.sim_fit[0].params['J'].value

               
            else:
                fit_fn = fit_mods.hanger_with_pf
                fit_temp = fit_mods.fit_hanger_with_pf(
                            fit_mods.HangerWithPfModel,
                            np.transpose([self.proc_data_dict['plot_frequency'],
                                          self.proc_data_dict['plot_amp']]))
                guess_pars = fit_temp.params
                self.proc_data_dict['fit_params'] = fit_temp.params
                fit_guess_fn = None

        if (len(self.raw_data_dict['timestamps']) == 1) or self.simultan:
            self.fit_dicts['reso_fit'] = {
                              'fit_fn': fit_fn,
                              'fit_guess_fn': fit_guess_fn,
                              'guess_pars': guess_pars,
                              'fit_yvals': {
                                  'data': self.proc_data_dict['plot_amp']
                                           },
                              'fit_xvals': {
                                  'f': self.proc_data_dict['plot_frequency']}
                                         }
        else:
            self.fit_dicts['reso_fit'] = {
                              'fit_fn': fit_fn,
                              'fit_guess_fn': fit_guess_fn,
                              'guess_pars': guess_pars,
                              'fit_yvals': [{'data': np.squeeze(tt)}
                                               for tt in self.plot_amp],
                              'fit_xvals': np.squeeze([{'f': tt[0]}
                                               for tt in self.plot_frequency])}

    def run_fitting(self):
        if not self.simultan:
            super().run_fitting()

    def do_subtract_background(self, thres=None, back_dict=None, ):
        if len(self.raw_data_dict['timestamps']) == 1:
            pass
        else:
            x_filtered = []
            y_filtered = []
            for tt in range(len(self.raw_data_dict['timestamps'])):
                y = np.squeeze(self.plot_amp[tt])
                x = np.squeeze(self.plot_frequency)[tt]
                # print(self.plot_frequency)
                # [print(x.shape) for x in self.plot_frequency]
                # print(x)
                # print(y)
                # print(len(x),len(y))
                guess_dict = SlopedHangerFuncAmplitudeGuess(y, x)
                Q = guess_dict['Q']['value']
                f0 = guess_dict['f0']['value']
                df = 2 * f0 / Q
                fmin = f0 - df
                fmax = f0 + df
                indices = np.logical_or(x < fmin * 1e9, x > fmax * 1e9)
                
                x_filtered.append(x[indices])
                y_filtered.append(y[indices])
            self.background = pd.concat([pd.Series(y_filtered[tt], index=x_filtered[tt])
                                         for tt in range(len(self.raw_data_dict['timestamps']))], axis=1).mean(axis=1)
            background_vals = self.background.reset_index().values
            freq = background_vals[:, 0]
            amp = background_vals[:, 1]
            # thres = 0.0065
            indices = amp < thres
            freq = freq[indices] * 1e-9
            amp = amp[indices]
            fit_fn = double_cos_linear_offset
            model = lmfit.Model(fit_fn)
            fit_yvals = amp
            fit_xvals = {'t': freq}
            # fit_guess_fn = double_cos_linear_offset_guess
            # guess_dict = fit_guess_fn(fit_yvals, **fit_xvals)
            for key, val in list(back_dict.items()):
                model.set_param_hint(key, **val)
            params = model.make_params()
            print(fit_xvals)
            fit_res = model.fit(fit_yvals,
                                params=params,
                                **fit_xvals)
            self.background_fit = fit_res

            for tt in range(len(self.raw_data_dict['timestamps'])):
                divide_vals = fit_fn(np.squeeze(self.plot_frequency)[tt] * 1e-9, **fit_res.best_values)
                self.plot_amp[tt] = np.array(
                    [np.array([np.divide(np.squeeze(self.plot_amp[tt]), divide_vals)])]).transpose()

    def prepare_plots(self):
        if not self.simultan:
            super(ResonatorSpectroscopy, self).prepare_plots()
        else:
            proc_data_dict = self.proc_data_dict
            plotsize = self.options_dict.get('plotsize')
            plot_fn = self.plot_line
            amp_diff = np.abs(proc_data_dict['plot_amp'][0]*np.exp(
                                  1j*np.pi*proc_data_dict['plot_phase'][0]/180)-
                              proc_data_dict['plot_amp'][1]*np.exp(
                                  1j*np.pi*proc_data_dict['plot_phase'][1]/180))
            # FIXME: Nathan 2019.05.08 I don't think this is the right place to adapt
            #  the ro fequency (i.e. in prepare_plot)... I had a hard time finding
            #  where it happened !
            self.f_RO = proc_data_dict['plot_frequency'][0][np.argmax(amp_diff)]
            self.plot_dicts['amp1'] = {'plotfn': plot_fn,
                                      'ax_id': 'amp',
                                      'xvals': proc_data_dict['plot_frequency'][0],
                                      'yvals': proc_data_dict['plot_amp'][0],
                                      'title': 'Spectroscopy amplitude: \n'
                                               '%s-%s' % (
                                          self.raw_data_dict[
                                              'measurementstring'][0],
                                          self.timestamps[0]),
                                      'xlabel': proc_data_dict['freq_label'],
                                      'xunit': 'Hz',
                                      'ylabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize,
                                      'color': 'b',
                                      'linestyle': '',
                                      'marker': 'o',
                                      'setlabel': '|g> data',
                                      'do_legend': True
                                       }
            self.plot_dicts['amp2'] = {'plotfn': plot_fn,
                                       'ax_id': 'amp',
                                       'xvals': proc_data_dict['plot_frequency'][1],
                                       'yvals': proc_data_dict['plot_amp'][1],
                                       'color': 'r',
                                       'linestyle': '',
                                       'marker': 'o',
                                       'setlabel': '|e> data',
                                       'do_legend': True
                                       }
            self.plot_dicts['diff'] = {'plotfn': plot_fn,
                                       'ax_id': 'amp',
                                       'xvals': proc_data_dict['plot_frequency'][0],
                                       'yvals': amp_diff,
                                       'color': 'g',
                                       'linestyle': '',
                                       'marker': 'o',
                                       'setlabel': 'diff',
                                       'do_legend': True
                                       }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                                        'xvals': proc_data_dict['plot_frequency'],
                                        'yvals': proc_data_dict['plot_phase'],
                                        'title': 'Spectroscopy phase: '
                                                 '%s' % (self.timestamps[0]),
                                        'xlabel': proc_data_dict['freq_label'],
                                        'ylabel': proc_data_dict['phase_label'],
                                        'yrange': proc_data_dict['phase_range'],
                                        'plotsize': plotsize
                                        }


    def plot_fitting(self):
        if self.do_fitting:
            fit_options = self.options_dict.get('fit_options', None)
            if fit_options is None:
                fitting_model = 'hanger'
            else:
                fitting_model = fit_options['model']
            for key, fit_dict in self.fit_dicts.items():
                if not self.simultan:
                    fit_results = fit_dict['fit_res']
                else:
                    fit_results = self.sim_fit
                ax = self.axs['amp']
                if len(self.raw_data_dict['timestamps']) == 1 or self.simultan:
                    if fitting_model == 'hanger':
                        ax.plot(list(fit_dict['fit_xvals'].values())[0],
                                fit_results.best_fit, 'r-', linewidth=1.5)
                        textstr = 'f0 = %.5f $\pm$ %.1g GHz' % (
                              fit_results.params['f0'].value,
                              fit_results.params['f0'].stderr) + '\n' \
                                           'Q = %.4g $\pm$ %.0g' % (
                              fit_results.params['Q'].value,
                              fit_results.params['Q'].stderr) + '\n' \
                                           'Qc = %.4g $\pm$ %.0g' % (
                              fit_results.params['Qc'].value,
                              fit_results.params['Qc'].stderr) + '\n' \
                                           'Qi = %.4g $\pm$ %.0g' % (
                              fit_results.params['Qi'].value,
                              fit_results.params['Qi'].stderr)
                        box_props = dict(boxstyle='Square',
                                         facecolor='white', alpha=0.8)
                        self.box_props = {key: val for key,
                                                       val in box_props.items()}
                        self.box_props.update({'linewidth': 0})
                        self.box_props['alpha'] = 0.
                        ax.text(0.03, 0.95, textstr, transform=ax.transAxes,
                                verticalalignment='top', bbox=self.box_props)
                    elif fitting_model == 'simple_hanger':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'lorentzian':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'complex':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'hanger_with_pf':
                        if not self.simultan:
                            ax.plot(list(fit_dict['fit_xvals'].values())[0],
                                    fit_results.best_fit, 'r-', linewidth=1.5)

                            par = ["%.3f" %(fit_results.params['omega_ro'].value*1e-9),
                                   "%.3f" %(fit_results.params['omega_pf'].value*1e-9),
                                   "%.3f" %(fit_results.params['kappa_pf'].value*1e-6),
                                   "%.3f" %(fit_results.params['J'].value*1e-6),
                                   "%.3f" %(fit_results.params['gamma_ro'].value*1e-6)]
                            textstr = str('f_ro = '+par[0]+' GHz'
                                      +'\n\nf_pf = '+par[1]+' GHz'
                                      +'\n\nkappa = '+par[2]+' MHz'
                                      +'\n\nJ = '+par[3]+' MHz'
                                      +'\n\ngamma_ro = '+par[4]+' MHz')
                            ax.plot([0],
                                    [0],
                                    'w',
                                    label=textstr)
                        else:
                            x_fit_0 = np.linspace(min(
                                self.proc_data_dict['plot_frequency'][0][0],
                                self.proc_data_dict['plot_frequency'][1][0]),
                                max(self.proc_data_dict['plot_frequency'][0][-1],
                                    self.proc_data_dict['plot_frequency'][1][-1]),
                                len(self.proc_data_dict['plot_frequency'][0]))
                            x_fit_1 = np.linspace(min(
                                self.proc_data_dict['plot_frequency'][0][0],
                                self.proc_data_dict['plot_frequency'][1][0]),
                                max(self.proc_data_dict['plot_frequency'][0][-1],
                                    self.proc_data_dict['plot_frequency'][1][-1]),
                                len(self.proc_data_dict['plot_frequency'][1]))

                            ax.plot(x_fit_0,
                                    fit_results[0].eval(
                                        fit_results[0].params,
                                        f=x_fit_0),
                                    'b--', linewidth=1.5, label='|g> fit')
                            ax.plot(x_fit_1,
                                    fit_results[1].eval(
                                        fit_results[1].params,
                                        f=x_fit_1),
                                    'r--', linewidth=1.5, label='|e> fit')
                            f_RO = self.f_RO
                            ax.plot([f_RO, f_RO],
                                    [0,max(max(self.raw_data_dict['amp'][0]),
                                           max(self.raw_data_dict['amp'][1]))],
                                    'k--', linewidth=1.5)

                            par = ["%.3f" %(fit_results[0].params['gamma_ro'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['omega_pf'].value*1e-9),
                                   "%.3f" %(fit_results[0].params['kappa_pf'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['J'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['omega_ro'].value*1e-9),
                                   "%.3f" %(fit_results[1].params['omega_ro'].value*1e-9),
                                   "%.3f" %((fit_results[1].params['omega_ro'].value-
                                             fit_results[0].params['omega_ro'].value)
                                             /2*1e-6)]
                            textstr = str('\n\nkappa = '+par[2]+' MHz'
                                          +'\n\nJ = '+par[3]+' MHz'
                                          +'\n\nchi = '+par[6]+' MHz'
                                          +'\n\nf_pf = '+par[1]+' GHz'
                                          +'\n\nf_rr |g> = '+par[4]+' GHz'
                                          +'\n\nf_rr |e> = '+par[5]+' GHz'
                                          +'\n\nf_RO = '+"%.3f" %(f_RO*1e-9)+''
                                          ' GHz'
                                         )
                            ax.plot([f_RO],
                                    [0],
                                    'w--', label=textstr)
                        # box_props = dict(boxstyle='Square',
                        #                  facecolor='white', alpha=0.8)
                        # self.box_props = {key: val for key,
                        #                                val in box_props.items()}
                        # self.box_props.update({'linewidth': 0})
                        # self.box_props['alpha'] = 0.
                        #
                        ax.legend(loc='upper left', bbox_to_anchor=[1, 1])

                else:
                    reso_freqs = [fit_results[tt].params['f0'].value *
                                  1e9 for tt in range(len(self.raw_data_dict['timestamps']))]
                    ax.plot(np.squeeze(self.plot_xvals),
                            reso_freqs,
                            'o',
                            color='m',
                            markersize=3)

    def plot(self, key_list=None, axs_dict=None, presentation_mode=None, no_label=False):
        super(ResonatorSpectroscopy, self).plot(key_list=key_list,
                                                axs_dict=axs_dict,
                                                presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()


class ResonatorSpectroscopy_v2(Spectroscopy):
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True):
        """
        FIXME: Nathan: the dependency on the # of timestamps is carried
         through the entire class and is horrible. We should loop and make fits
         separately, instead of using the simultan parameter.
         It would be much simpler!
        Args:
            t_start:
            options_dict:
                ref_state: reference state timestamp when comparing several
                    spectra. Most of the time it will be timestamp of ground
                    state.
                # TODO: Nathan: merge with fit_options (?)
                qutrit_fit_options: dict with options for qutrit RO frequency
                    fitting.
                        sigma_init: initial noise standard deviation assumed
                            for distribution of point in IQ plane. Assumed to
                            be large and algorithm will reduce it.
                        target_fidelity: target fidelity
                        max_width_at_max_fid: maximum width (in Hz) when
                        searching for appropriate sigma
            t_stop:
            do_fitting:
            extract_only:
            auto:
        """
        super(ResonatorSpectroscopy_v2, self).__init__(t_start, t_stop=t_stop,
                                                    options_dict=options_dict,
                                                    extract_only=extract_only,
                                                    auto=False,
                                                    do_fitting=do_fitting)
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess', {})

        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorSpectroscopy_v2, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        n_spectra = len(self.raw_data_dict['timestamps'])
        self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        if self.options_dict.get('ref_state', None) is None:
            default_ref_state = 'g'
            message = "Analyzing spectra of {} states but no ref_state " \
                      "was passed. Assuming timestamp[0]: {} is the " \
                      "timestamp of reference state with label {}"
            logging.warning(
                message.format(n_spectra, self.raw_data_dict['timestamps'][0],
                               default_ref_state))
            self.ref_state = default_ref_state
        else:
            self.ref_state = self.options_dict['ref_state']

        spectra_mapping = \
            self.options_dict.get("spectra_mapping",
                                  self._default_spectra_mapping())

        spectra = {state: self.raw_data_dict["amp"][i] *
                      np.exp(1j * np.pi *
                           self.raw_data_dict["phase"][i] / 180.)
                    for i, state in enumerate(spectra_mapping.keys())}

        iq_distance = {state + self.ref_state:
                           np.abs(spectra[state] - spectra[self.ref_state])
                       for state in spectra_mapping.keys()
                       if state != self.ref_state}
        for state_i in spectra_mapping:
            for state_j in spectra_mapping:
                if not state_i + state_j  in iq_distance and \
                        state_i != state_j:
                    # both ij and ji will have entries which will have
                    # the same values but this is not a problem per se.
                    iq_distance[state_i + state_j] = \
                        np.abs(spectra[state_i] - spectra[state_j])

        self.proc_data_dict["spectra_mapping"] = spectra_mapping
        self.proc_data_dict["spectra"] = spectra
        self.proc_data_dict["iq_distance"] = iq_distance
        self.proc_data_dict["fit_raw_results"] = OrderedDict()

    def _default_spectra_mapping(self):
        default_levels_order = ('g', 'e', 'f')
        tts = self.raw_data_dict['timestamps']
        spectra_mapping = {default_levels_order[i]: tt
                           for i, tt in enumerate(tts)}
        msg = "Assuming following mapping templates of spectra: {}." \
              "\nspectra_mapping can be used in options_dict to modify" \
              "this behavior."
        logging.warning(msg.format(spectra_mapping))
        return spectra_mapping


    def prepare_fitting(self):
        super().prepare_fitting()
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex, hanger_with_pf.
        fit_options = self.options_dict.get('fit_options', dict())
        subtract_background = \
            self.options_dict.get('subtract_background', False)
        fitting_model = fit_options.get('model', 'hanger')
        self.proc_data_dict['fit_results'] = OrderedDict()
        self.fit_res = dict()
        if subtract_background:
            logging.warning("Substract background might not work and has "
                            "not been tested.")
            self.do_subtract_background(
                thres=self.options_dict['background_thres'],
                back_dict=self.options_dict['background_dict'])

        if fitting_model == 'hanger':
            fit_fn = fit_mods.SlopedHangerFuncAmplitude
            fit_guess_fn = fit_mods.SlopedHangerFuncAmplitudeGuess
            guess_pars = None
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
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.HangerFuncComplex
            # TODO HangerFuncComplexGuess
        elif fitting_model == 'hanger_with_pf':
            if len(self.raw_data_dict['timestamps']) == 1:
                # single fit
                fit_fn = fit_mods.hanger_with_pf
                fit_temp = fit_mods.fit_hanger_with_pf(
                    fit_mods.HangerWithPfModel,
                    np.transpose([self.proc_data_dict['plot_frequency'],
                                  self.proc_data_dict['plot_amp']]))
                guess_pars = fit_temp.params
                self.proc_data_dict['fit_params'] = fit_temp.params
                self.proc_data_dict['fit_raw_results'][self.ref_state] = \
                    fit_temp.params
                fit_guess_fn = None
            else:
                # comparative fit to reference state
                # FIXME: Nathan: I guess here only fit dicts should be created
                #  and then passed to run_fitting() of basis class but this is
                #  not done here. Instead, fitting seems to be done here.
                ref_spectrum = self.proc_data_dict['spectra'][self.ref_state]
                for state, spectrum in self.proc_data_dict['spectra'].items():
                    if state == self.ref_state:
                        continue
                    key = self.ref_state + state
                    fit_fn = fit_mods.simultan_hanger_with_pf
                    fit_results = fit_mods.fit_hanger_with_pf(
                        fit_mods.SimHangerWithPfModel, [
                            np.transpose(
                                [self.proc_data_dict['plot_frequency'][0],
                                 np.abs(ref_spectrum)]),
                            np.transpose(
                                [self.proc_data_dict['plot_frequency'][0],
                                 np.abs(spectrum)])],
                        simultan=True)
                    self.proc_data_dict['fit_raw_results'][key] = fit_results


                    guess_pars = None
                    fit_guess_fn = None

                    chi = (fit_results[1].params['omega_ro'].value -
                                fit_results[0].params['omega_ro'].value) / 2
                    f_RO_res = (fit_results[0].params['omega_ro'].value +
                                     fit_results[1].params['omega_ro'].value) / 2
                    f_PF = fit_results[0].params['omega_pf'].value
                    kappa = fit_results[0].params['kappa_pf'].value
                    J_ = fit_results[0].params['J'].value
                    f_RO = self.find_f_RO([self.ref_state, state])
                    self.fit_res[key] = \
                        dict(chi=chi, f_RO_res=f_RO_res, f_PF=f_PF,
                             kappa=kappa, J_=J_, f_RO=f_RO)

        if (len(self.raw_data_dict['timestamps']) == 1):
            self.fit_dicts['reso_fit'] = {
                'fit_fn': fit_fn,
                'fit_guess_fn': fit_guess_fn,
                'guess_pars': guess_pars,
                'fit_yvals': {'data': self.proc_data_dict['plot_amp']},
                'fit_xvals': { 'f': self.proc_data_dict['plot_frequency']}}

    def find_f_RO(self, states):
        """
        Finds the best readout frequency of the list of states.
            If one state is passed, the resonator frequency is returned.
            If two states are passed, the frequency with maximal difference
        between the two states in IQ plane is returned (optimal qubit nRO freq).
            If three states are passed, optimal frequency is found by finding
            the highest variance allowing a target fidelity to be reached on a
            narrow frequency interval. (optimal qutrit RO_freq)
        Args:
            states: list of states between which readout frequency
                should be found

        Returns:

        """
        key = "".join(states)
        if len(states) == 1:
            f_RO = self.proc_data_dict['plot_frequency'][0][
            np.argmax(self.proc_data_dict['spectra'][key])]
        elif len(states) == 2:
            f_RO = self.proc_data_dict['plot_frequency'][0][
                np.argmax(self.proc_data_dict['iq_distance'][key])]
        elif len(states) == 3:
            f_RO = self._find_f_RO_qutrit(
                self.proc_data_dict['spectra'],
                self.proc_data_dict['plot_frequency'][0],
                **self.options_dict.get('qutrit_fit_options', dict()))
        else:
            raise ValueError("{} states were given but method expects 1, "
                             "2 or 3 states.")
        return f_RO

    def _find_f_RO_qutrit(self, spectra, freqs, sigma_init=0.01, **kw):
        n_iter = 0
        avg_fidelities = OrderedDict()
        single_level_fidelities = OrderedDict()
        optimal_frequency = []
        sigmas = [sigma_init]

        logging.debug("###### Starting Analysis to find qutrit f_RO ######")

        while self.update_sigma(avg_fidelities, sigmas, freqs,
                                optimal_frequency, n_iter, **kw):
            logging.debug("Iteration {}".format(n_iter))
            sigma = sigmas[-1]
            if sigma in avg_fidelities.keys():
                continue
            else:
                avg_fidelity, single_level_fidelity = \
                    self.evaluate_fidelity(spectra, sigma)
                avg_fidelities[sigma] = avg_fidelity
                single_level_fidelities[sigma] = single_level_fidelity
                n_iter += 1
        raw_results = dict(avg_fidelities=avg_fidelities,
                           single_level_fidelities=single_level_fidelities,
                           sigmas=sigmas, optimal_frequency=optimal_frequency)
        qutrit_key = "".join(list(spectra))
        self.proc_data_dict["fit_raw_results"][qutrit_key] = \
            raw_results

        logging.debug("###### Finished Analysis. Optimal f_RO: {} ######"
                      .format(optimal_frequency[-1]))
        return optimal_frequency[-1]

    def update_sigma(self, avg_fidelities, sigmas, freqs,
                     optimal_frequency, n_iter, n_iter_max=20,
                     target_fidelity=0.999, max_width_at_max_fid=0.2e6, **kw):
        continue_search = True
        if n_iter >= n_iter_max:
            logging.warning("Could not converge to a proper RO frequency" \
                  "within {} iterations. Returning best frequency found so far. "
                  "Consider changing log_bounds".format(n_iter_max))
            continue_search = False
        elif len(avg_fidelities.keys()) == 0:
            # search has not started yet
            continue_search = True
        else:
            delta_freq = freqs[1] - freqs[0]
            if max_width_at_max_fid < delta_freq:
                msg = "max_width_at_max_fid cannot be smaller than the " \
                      "difference between two frequency data points.\n" \
                      "max_width_at_max_fid: {}\nDelta freq: {}"
                raise ValueError(msg.format(max_width_at_max_fid, delta_freq))

            sigma_current = sigmas[-1]
            fid, idx_width = self.fidelity_and_width(
                avg_fidelities[sigma_current], target_fidelity)
            width = idx_width * delta_freq
            logging.debug("sigmas " + str(sigmas) + " width (MHz): "
                          + str(width / 1e6))
            f_opt = freqs[np.argmax(avg_fidelities[sigma_current])]
            optimal_frequency.append(f_opt)

            if len(sigmas) == 1:
                sigma_previous = 10 ** (np.log10(sigma_current) + 1)
            else:
                sigma_previous = sigmas[-2]
            log_diff = np.log10(sigma_previous) - np.log10(sigma_current)

            if fid >= target_fidelity and width <= max_width_at_max_fid:
                # succeeded
                continue_search = False
            elif fid >= target_fidelity and width > max_width_at_max_fid:
                # sigma is too small, update lower bound
                if log_diff < 0:
                    sigma_new = \
                        10 ** (np.log10(sigma_current) - np.abs(log_diff) / 2)
                else:
                    sigma_new = \
                        10 ** (np.log10(sigma_current) + np.abs(log_diff))
                msg = "Width > max_width, update sigma to: {}"
                logging.debug(msg.format(sigma_new))
                sigmas.append(sigma_new)
            elif fid < target_fidelity:
                # sigma is too high, update higher bound
                if np.all(np.diff(sigmas) < 0):
                    sigma_new = 10 ** (np.log10(sigma_current) - log_diff)
                else:
                    sigma_new = 10 ** (np.log10(sigma_current) -
                                       np.abs(log_diff) / 2)
                msg = "Fidelity < target fidelity, update sigma to: {}"
                logging.debug(msg.format(sigma_new))
                sigmas.append(sigma_new)

        return continue_search

    def fidelity_and_width(self, avg_fidelity, target_fidelity):
        avg_fidelity = np.array(avg_fidelity)
        max_fid = np.max(avg_fidelity)
        idx_width = np.sum(
            (avg_fidelity >= target_fidelity) * (avg_fidelity <= 1.))
        return max_fid, idx_width

    def evaluate_fidelity(self, spectra, sigma):
        def g(x, d, sigma=0.1):
            while (x < - np.pi or x > np.pi):
                if x < np.pi:
                    x += 2 * np.pi
                elif x > np.pi:
                    x -= 2 * np.pi
            return np.exp(-d ** 2 / np.cos(x) ** 2 / (2 * sigma ** 2))

        def integral_part1(angle):
            return integrate.quad(lambda x: 1 / (2 * np.pi),
                                  angle - np.pi, angle)[0]

        def integral_part2(angle, sigma):
            const = 1 / (2 * np.pi)
            return const * integrate.quad(lambda x: g(x, d1, sigma=sigma),
                                          angle - np.pi, angle)[0]

        assert len(spectra) == 3, "3 spectra required for qutrit F_RO " \
                                  "analysis. Found {}".format((len(spectra)))
        i1s, i2s, i3s = [], [], []
        # in most cases, states will be ['g', 'e', 'f'] but to ensure not to
        # be dependent on labels we take indices of keys
        states = list(spectra.keys())
        for i in range(len(spectra[states[0]])):
            pt1 = (spectra[states[0]][i].real, spectra[states[0]][i].imag)
            pt2 = (spectra[states[1]][i].real, spectra[states[1]][i].imag)
            pt3 = (spectra[states[2]][i].real, spectra[states[2]][i].imag)
            d1 = geo.distance(pt1, pt2) / 2
            d2 = geo.distance(pt2, pt3) / 2
            d3 = geo.distance(pt1, pt3) / 2
            # translate to point1
            pt2 = tuple(np.asarray(pt2) - np.asarray(pt1))
            pt3 = tuple(np.asarray(pt3) - np.asarray(pt1))
            pt1 = (0., 0.)
            c, R = geo.circumcenter(pt2, pt3, pt1, show=False)
            gamma1 = np.arccos(d1 / R)
            gamma2 = np.arccos(d2 / R)
            gamma3 = np.arccos(d3 / R)
            i1 = integral_part1(gamma1) - integral_part2(gamma1, sigma)
            i2 = integral_part1(gamma2) - integral_part2(gamma2, sigma)
            i3 = integral_part1(gamma3) - integral_part2(gamma3, sigma)
            i1s.append(i1)
            i2s.append(i2)
            i3s.append(i3)

        i1s, i2s, i3s = np.array(i1s), np.array(i2s), np.array(i3s)
        total_area = 2 * i1s + 2 * i2s + 2 * i3s
        avg_fidelity = total_area / 3
        fid_state_0 = i1s + i3s
        fid_state_1 = i1s + i2s
        fid_state_2 = i2s + i3s

        single_level_fid = {states[0]: fid_state_0,
                            states[1]: fid_state_1,
                            states[2]: fid_state_2}

        return avg_fidelity, single_level_fid

    def run_fitting(self):
        # FIXME: Nathan: for now this is left as written previously but
        #  ultimately all fitting should be done in base class if possible
        states = list(self.proc_data_dict['spectra'])
        if len(states) == 1:
            super().run_fitting()

        if len(states) == 3:
            f_RO_qutrit =  self.find_f_RO(states)
            self.fit_res["".join(states)] = dict(f_RO=f_RO_qutrit)


    def do_subtract_background(self, thres=None, back_dict=None, ):
        if len(self.raw_data_dict['timestamps']) == 1:
            pass
        else:
            x_filtered = []
            y_filtered = []
            for tt in range(len(self.raw_data_dict['timestamps'])):
                y = np.squeeze(self.plot_amp[tt])
                x = np.squeeze(self.plot_frequency)[tt]
                # print(self.plot_frequency)
                # [print(x.shape) for x in self.plot_frequency]
                # print(x)
                # print(y)
                # print(len(x),len(y))
                guess_dict = SlopedHangerFuncAmplitudeGuess(y, x)
                Q = guess_dict['Q']['value']
                f0 = guess_dict['f0']['value']
                df = 2 * f0 / Q
                fmin = f0 - df
                fmax = f0 + df
                indices = np.logical_or(x < fmin * 1e9, x > fmax * 1e9)

                x_filtered.append(x[indices])
                y_filtered.append(y[indices])
            self.background = pd.concat([pd.Series(y_filtered[tt], index=x_filtered[tt])
                                         for tt in
                                         range(len(self.raw_data_dict['timestamps']))],
                                        axis=1).mean(axis=1)
            background_vals = self.background.reset_index().values
            freq = background_vals[:, 0]
            amp = background_vals[:, 1]
            # thres = 0.0065
            indices = amp < thres
            freq = freq[indices] * 1e-9
            amp = amp[indices]
            fit_fn = double_cos_linear_offset
            model = lmfit.Model(fit_fn)
            fit_yvals = amp
            fit_xvals = {'t': freq}
            # fit_guess_fn = double_cos_linear_offset_guess
            # guess_dict = fit_guess_fn(fit_yvals, **fit_xvals)
            for key, val in list(back_dict.items()):
                model.set_param_hint(key, **val)
            params = model.make_params()
            print(fit_xvals)
            fit_res = model.fit(fit_yvals,
                                params=params,
                                **fit_xvals)
            self.background_fit = fit_res

            for tt in range(len(self.raw_data_dict['timestamps'])):
                divide_vals = fit_fn(np.squeeze(self.plot_frequency)[tt] * 1e-9,
                                     **fit_res.best_values)
                self.plot_amp[tt] = np.array(
                    [np.array([np.divide(np.squeeze(self.plot_amp[tt]),
                                         divide_vals)])]).transpose()

    def prepare_plots(self):
        self.get_default_plot_params(set_pars=True)
        proc_data_dict = self.proc_data_dict
        spectra = proc_data_dict['spectra']
        plotsize = self.options_dict.get('plotsize')
        plot_fn = self.plot_line
        for state, spectrum in spectra.items():
            all_freqs = proc_data_dict['plot_frequency']
            freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
            self.plot_dicts['amp_{}'
                .format(state)] = {
                'plotfn': plot_fn,
                'ax_id': 'amp',
                'xvals': freqs,
                'yvals': np.abs(spectrum),
                'title': 'Spectroscopy amplitude: \n'
                        '%s-%s' % (
                            self.raw_data_dict['measurementstring'][0],
                            self.timestamps[0]),
                'xlabel': proc_data_dict['freq_label'],
                'xunit': 'Hz',
                'ylabel': proc_data_dict['amp_label'],
                'yrange': proc_data_dict['amp_range'],
                'plotsize': plotsize,
                # 'color': 'b',
                'linestyle': '',
                'marker': 'o',
                'setlabel': '$|{}\\rangle$'.format(state),
                'do_legend': True }
            if state != self.ref_state and len(spectra) == 2.:
                # if comparing two stattes we are interested in the
                # difference between the two responses
                label = "iq_distance_{}{}".format(state, self.ref_state)
                self.plot_dicts[label] = {
                    'plotfn': plot_fn,
                    'ax_id': 'amp',
                    'xvals': proc_data_dict['plot_frequency'][0],
                    'yvals': proc_data_dict['iq_distance'][
                        state + self.ref_state],
                    #'color': 'g',
                    'linestyle': '',
                    'marker': 'o',
                    'markersize': 5,
                    'setlabel': label,
                    'do_legend': True}

    def plot_fitting(self):
        fit_options = self.options_dict.get('fit_options', None)
        if fit_options is None:
            fitting_model = 'hanger'
        else:
            fitting_model = fit_options['model']

        if len(self.raw_data_dict['timestamps']) == 1:
            fit_results = self.fit_dict['fit_res']
        else:
            fit_results = self.proc_data_dict['fit_raw_results']
        ax = self.axs['amp']
        if fitting_model == 'hanger':
            raise NotImplementedError(
                'Plotting hanger is not supported in this class.')
        elif fitting_model == 'simple_hanger':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'lorentzian':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'complex':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'hanger_with_pf':
            label = "$|{}\\rangle$ {}"
            all_freqs = self.proc_data_dict['plot_frequency']
            freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
            for state, spectrum in self.proc_data_dict['spectra'].items():
                if len(self.proc_data_dict['spectra']) == 1:
                    # then also add single fit parameters to the legend
                    # else the coupled params will be added from fit results
                    textstr = "f_ro = {:.3f} GHz\nf_pf = {:3f} GHz\n" \
                        "kappa = {:3f} MHz\nJ = {:3f} MHz\ngamma_ro = " \
                        "{:3f} MHz".format(
                            fit_results.params['omega_ro'].value * 1e-9,
                            fit_results.params['omega_pf'].value * 1e-9,
                            fit_results.params['kappa_pf'].value * 1e-6,
                            fit_results.params['J'].value * 1e-6,
                            fit_results.params['gamma_ro'].value * 1e-6)
                    # note: next line will have to be removed when
                    # cleaning up the # timestamps dependency
                    ax.plot(freqs,
                            fit_results.best_fit, 'r-', linewidth=1.5)
                    ax.plot([], [], 'w', label=textstr)

            if len(self.proc_data_dict['spectra']) != 1 :
                for states, params in self.fit_res.items():
                    f_r = fit_results[states]
                    if len(states) == 3:
                        ax.plot([params["f_RO"], params["f_RO"]],
                                [0, np.max(self.raw_data_dict['amp'])],
                                'm--', linewidth=1.5, label="F_RO_{}"
                                .format(states))
                        ax2 = ax.twinx()
                        last_fit_key = list(f_r["avg_fidelities"].keys())[-1]
                        ax2.scatter(freqs, f_r["avg_fidelities"][last_fit_key],
                                    color='c',
                                    label= "{} fidelity".format(states),
                                    marker='.')
                        ax2.set_ylabel("Fidelity")
                        label = "f_RO_{} = {:.5f} GHz".format(states,
                                                          params['f_RO'] * 1e-9)
                        ax.plot([],[], label=label)
                        fig, ax3 = plt.subplots()
                        for sigma, avg_fid in f_r['avg_fidelities'].items():
                            ax3.plot(self.proc_data_dict['plot_frequency'][0],
                                     avg_fid, label=sigma)
                        ax3.plot([f_r["optimal_frequency"][-1]],
                                 [f_r["optimal_frequency"][-1]], "k--")
                        t_f = self.options_dict.get('qutrit_fit_options', dict())
                        ax3.set_ylim()

                    elif len(states) == 2:
                        c = "r--"
                        ax.plot(freqs, f_r[0].eval(f_r[0].params, f=freqs),
                                c, label=label.format(states[0], "fit"),
                                linewidth=1.5)
                        ax.plot(freqs, f_r[1].eval(f_r[1].params, f=freqs),
                                c, label=label.format(states[1], "fit"),
                                linewidth=1.5)
                        ax.plot([params['f_RO'], params['f_RO']],
                                [0, np.max(self.raw_data_dict['amp'])],
                                'r--', linewidth=2)

                        params_str = 'states: {}' \
                            '\n kappa = {:.3f} MHz\n J = {:.3f} MHz' \
                            '\n chi = {:.3f} MHz\n f_pf = {:.3f} GHz' \
                            '\n f_rr $|{}\\rangle$ = {:.3f} GHz' \
                            '\n f_rr $|{}\\rangle$ = {:.3f} GHz' \
                            '\n f_RO = {:.3f} GHz'.format(
                            states,
                            f_r[0].params['kappa_pf'].value * 1e-6,
                            f_r[0].params['J'].value * 1e-6,
                            (f_r[1].params['omega_ro'].value -
                             f_r[0].params['omega_ro'].value) / 2 * 1e-6,
                            f_r[0].params['omega_pf'].value * 1e-9,
                            states[0], f_r[0].params['omega_ro'].value * 1e-9,
                            states[1], f_r[1].params['omega_ro'].value * 1e-9,
                            params['f_RO'] * 1e-9)
                        ax.plot([],[], 'w', label=params_str)
            ax.legend(loc='upper left', bbox_to_anchor=[1.1, 1])


    def plot(self, key_list=None, axs_dict=None, presentation_mode=None, no_label=False):
        super(ResonatorSpectroscopy_v2, self).plot(key_list=key_list,
                                                axs_dict=axs_dict,
                                                presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()



class ResonatorDacSweep(ResonatorSpectroscopy):
    def __init__(self, t_start,
                 options_dict,
                 t_stop=None,
                 do_fitting=True,
                 extract_only=False,
                 auto=True):
        super(ResonatorDacSweep, self).__init__(t_start, t_stop=t_stop,
                                                options_dict=options_dict,
                                                do_fitting=do_fitting,
                                                extract_only=extract_only,
                                                auto=False)
        self.params_dict['dac_value'] = 'IVVI.dac12'

        self.numeric_params = ['freq', 'amp', 'phase', 'dac_value']
        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorDacSweep, self).process_data()
        # self.plot_xvals = self.options_dict.get('xvals',np.array([[tt] for tt in range(len(self.raw_data_dict['timestamps']))]))
        conversion_factor = self.options_dict.get('conversion_factor', 1)

        # self.plot_xvals =
        self.plot_xlabel = self.options_dict.get('xlabel', 'Gate voltage (V)')
        self.plot_xwidth = self.options_dict.get('xwidth', None)
        for tt in range(len(self.plot_xvals)):
            print(self.plot_xvals[tt][0])

        if self.plot_xwidth == 'auto':
            x_diff = np.diff(np.ravel(self.plot_xvals))
            dx1 = np.concatenate(([x_diff[0]], x_diff))
            dx2 = np.concatenate((x_diff, [x_diff[-1]]))
            self.plot_xwidth = np.minimum(dx1, dx2)
            self.plot_frequency = np.array(
                [[tt] for tt in self.raw_data_dict['freq']])
            self.plot_phase = np.array(
                [[tt] for tt in self.raw_data_dict['phase']])
            self.plot_amp = np.array(
                [np.array([tt]).transpose() for tt in self.raw_data_dict['amp']])
