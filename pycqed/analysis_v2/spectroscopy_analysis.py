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
import lmfit

from importlib import reload  # Useful for reloading while testing

import importlib
importlib.reload(ba)
importlib.reload(fit_mods)


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
                                      'do_legend':True,
                                      'plotsize': plotsize
                                      }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                                        'xvals': proc_data_dict['plot_frequency'],
                                        'yvals': proc_data_dict['plot_phase'],
                                        'title': 'Spectroscopy phase: %s' % (self.timestamps[0]),
                                        'xlabel': proc_data_dict['freq_label'],
                                        'ylabel': proc_data_dict['phase_label'],
                                        'yrange': proc_data_dict['phase_range'],
                                        'do_legend':True,
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
            self.proc_data_dict['plot_phase'] = np.unwrap(self.proc_data_dict['plot_phase'],discont=3.141592653589793)
            self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        else:
            pass
        self.raw_data_dict['real'] = [
            self.raw_data_dict['measured_values'][0][2]]
        self.raw_data_dict['imag'] = [
            self.raw_data_dict['measured_values'][0][3]]
        self.proc_data_dict['real'] = np.array(self.raw_data_dict['real'][0])
        self.proc_data_dict['imag'] = np.array(self.raw_data_dict['imag'][0])
        self.proc_data_dict['plot_real'] = self.proc_data_dict['real']
        self.proc_data_dict['plot_imag'] = self.proc_data_dict['imag']
        self.proc_data_dict['real_label'] = 'Real{S21} (V rms)'
        self.proc_data_dict['imag_label'] = 'Imag{S21} (V rms)'
        if len(self.raw_data_dict['timestamps']) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(self.proc_data_dict['plot_phase'],discont=3.141592653589793)
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
                                       'setlabel':'real',
                                       'plotsize': plotsize,
                                       'do_legend':True
                                       }
            self.plot_dicts['imag'] = {'plotfn': plot_fn,
                                       'xvals': proc_data_dict['plot_frequency'],
                                       'yvals': proc_data_dict['plot_imag'],
                                       'title': 'S21 phase: %s' % (self.timestamps[0]),
                                       'xlabel': proc_data_dict['freq_label'],
                                       'ylabel': proc_data_dict['imag_label'],
                                       'yrange': proc_data_dict['amp_range'],
                                       'setlabel':'imag',
                                       'plotsize': plotsize,
                                       'do_legend':True
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
                        'max':self.proc_data_dict['plot_frequency'][-1]}
            complex_guess['Q'] = {'value':complex_guess['Q']['value'],
                        'min':complex_guess['Q']['value']*0.5,
                        'max':complex_guess['Q']['value']*2.}

            '''
            Here we estimate the phase velocity and the initial phase based
            on the first and last 5% of data points in the scan
            '''
            num_fit_points = int(len(self.proc_data_dict['plot_frequency'])/20.)
            phase_data = np.hstack([np.unwrap(self.proc_data_dict['plot_phase'])[:num_fit_points],
                            np.unwrap(self.proc_data_dict['plot_phase'][-num_fit_points:])])
            freq_data = np.hstack([self.proc_data_dict['plot_frequency'][:num_fit_points],
                            self.proc_data_dict['plot_frequency'][-num_fit_points:]])
            lin_pars = np.polyfit(freq_data,phase_data,1)
            phase_v, phase_0 = lin_pars

            if np.sign(phase_v)==-1:
                min_phase_v = -1.05*np.abs(phase_v)
                max_phase_v = -0.95*np.abs(phase_v)
            else:
                min_phase_v = 0.95*np.abs(phase_v)
                max_phase_v = 1.05*np.abs(phase_v)
            complex_guess['phi_v'] = {'value':phase_v,'min':min_phase_v,
                                            'max':max_phase_v}
            complex_guess['phi_0'] = {'value':phase_0%(2*np.pi), 'min':0,
                                            'max':2*np.pi}

            fit_fn = fit_mods.hanger_func_complex_SI



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
        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorSpectroscopy, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        if len(self.raw_data_dict['timestamps']) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(self.proc_data_dict['plot_phase'],discont=3.141592653589793)
            self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        else:
            pass

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
            fit_fn = fit_mods.HangerFuncComplex
            # TODO HangerFuncComplexGuess

        if len(self.raw_data_dict['timestamps']) == 1:
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
        super(ResonatorSpectroscopy, self).prepare_plots()

    def plot_fitting(self):
        if self.do_fitting:
            for key, fit_dict in self.fit_dicts.items():
                fit_results = fit_dict['fit_res']
                ax = self.axs['amp']
                if len(self.raw_data_dict['timestamps']) == 1:
                    ax.plot(list(fit_dict['fit_xvals'].values())[0], fit_results.best_fit, 'r-', linewidth=1.5)
                    textstr = 'f0 = %.5f $\pm$ %.1g GHz' % (
                        fit_results.params['f0'].value, fit_results.params['f0'].stderr) + '\n' \
                                                                                           'Q = %.4g $\pm$ %.0g' % (
                                  fit_results.params['Q'].value, fit_results.params['Q'].stderr) + '\n' \
                                                                                                   'Qc = %.4g $\pm$ %.0g' % (
                                  fit_results.params['Qc'].value, fit_results.params['Qc'].stderr) + '\n' \
                                                                                                     'Qi = %.4g $\pm$ %.0g' % (
                                  fit_results.params['Qi'].value, fit_results.params['Qi'].stderr)
                    box_props = dict(boxstyle='Square',
                                     facecolor='white', alpha=0.8)
                    self.box_props = {key: val for key,
                                                   val in box_props.items()}
                    self.box_props.update({'linewidth': 0})
                    self.box_props['alpha'] = 0.
                    ax.text(0.03, 0.95, textstr, transform=ax.transAxes,
                            verticalalignment='top', bbox=self.box_props)
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
                                                axs_dict=axs_dict, presentation_mode=presentation_mode)
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
        print(self.raw_data_dict['dac_value'])
        # self.plot_xvals =
        self.plot_xlabel = self.options_dict.get('xlabel', 'Gate voltage (V)')
        self.plot_xwidth = self.options_dict.get('xwidth', None)
        for tt in range(len(self.plot_xvals)):
            print(self.plot_xvals[tt][0])
        print(self.plot_xvals)
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
