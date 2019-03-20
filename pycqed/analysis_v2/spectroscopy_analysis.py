"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of all
the spectroscopy measurement analyses.
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
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)


import importlib
importlib.reload(ba)
importlib.reload(fit_mods)


class Spectroscopy(ba.BaseDataAnalysis):

    def __init__(self, t_start: str,
                 t_stop: str = None,
                 options_dict: dict = None,
                 label: str = '',
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
            self.proc_data_dict['plot_phase'] = np.unwrap(self.proc_data_dict['plot_phase'],discont=np.pi)
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
            self.proc_data_dict['plot_phase'] = np.unwrap(self.proc_data_dict['plot_phase'],discont=np.pi)
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
                                       'title': 'S21 real: %s' % (self.timestamps[0]),
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
                                       'title': 'S21 imaginary: %s' % (self.timestamps[0]),
                                       'xlabel': proc_data_dict['freq_label'],
                                       'ylabel': proc_data_dict['imag_label'],
                                       'yrange': proc_data_dict['amp_range'],
                                       'setlabel':'imag',
                                       'plotsize': plotsize,
                                       'do_legend':True
                                       }
            self.plot_dicts['plane'] = {'plotfn': plot_fn,
                                       'xvals': proc_data_dict['plot_real'],
                                       'yvals': proc_data_dict['plot_imag'],
                                       'title': 'S21 parametric plot: %s' % (self.timestamps[0]),
                                       'xlabel': proc_data_dict['real_label'],
                                       'ylabel': proc_data_dict['imag_label'],
                                       'yrange': proc_data_dict['amp_range'],
                                       'xrange': proc_data_dict['amp_range'],
                                       'setlabel':'plane',
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
    '''
    Performs as fit to the resonance using data acquired using VNA R&S ZNB 20

    eg. to use complex model:
    ma2.VNA_analysis(t_start='20010101_213600', options_dict={'fit_options': {'model': 'complex'}})
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

            self.plot_dicts['reso_fit_phase'] = {
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


class VNA_TwoD_Analysis(MA.TwoD_Analysis):
  """
  TwoD analysis of a 2D-VNA resonator scan. Plot points on all resonator frequencies,
  by fitting resonators to the linecuts
  """
  def __init__(self, t_start,
                 options_dict=None,
                 do_fitting=True,
                 extract_only=False,
                 auto=True):
        super(MA.TwoD_Analysis, self).__init__(timestamp = timestamp,
                                           options_dict=options_dict,
                                           extract_only=extract_only,
                                           auto=auto,
                                           do_fitting=do_fitting)

        linecut_fit_result = self.fit_linecuts()
        self.linecut_fit_result = linecut_fit_result
        f0s = []
        for res in self.linecut_fit_result:
            f0s.append(res.values['f0']*1e9)
        self.f0s = np.array(f0s)
        self.run_full_analysis()

  # Better: try to define the fitting dictionaries like in the complex VNA fit MA

          # if len(self.raw_data_dict['timestamps']) == 1:
          #   if fitting_model == 'complex':
          #       ### do the initial guess
          #       complex_data = np.add(self.proc_data_dict['real'],1.j*self.proc_data_dict['imag'])
          #       self.fit_dicts['reso_fit'] = {'fit_fn': fit_fn,
          #                                     'guess_dict':complex_guess,
          #                                     'fit_yvals': {'data': complex_data},
          #                                     'fit_xvals': {'f': self.proc_data_dict['plot_frequency']},
          #                                     'fitting_type':'minimize'}





  def fit_linecuts(self):

    linecut_mag = np.array(self.measured_values)[0].T
    sweep_points = self.sweep_points
    fit_result = []
    for linecut in linecut_mag:
      fit_result.append(self.resonator_fit(sweep_points,linecut))

    return fit_result

  def resonator_fit(self,sweep_points,linecut_mag):
            ########## Fit data ##########

    # Note that not the full functionality of the fit function of resonators is implemented yet

      # Fit Power to a Lorentzian

      min_index = np.argmin(linecut_mag)
      max_index = np.argmax(linecut_mag)

      min_frequency = sweep_points[min_index]
      max_frequency = sweep_points[max_index]

      measured_powers_smooth = a_tools.smooth(linecut_mag,
                                              window_len=11)
      peaks = a_tools.peak_finder((sweep_points),
                                       measured_powers_smooth,
                                       window_len=0)

      # Search for peak
      if peaks['dip'] is not None:  # look for dips first
          f0 = peaks['dip']
          amplitude_factor = -1.
      elif peaks['peak'] is not None:  # then look for peaks
          f0 = peaks['peak']
          amplitude_factor = 1.
      else:  # Otherwise take center of range
          f0 = np.median(sweep_points)
          amplitude_factor = -1.
          logging.warning('No peaks or dips in range')
          # If this error is raised, it should continue the analysis but
          # not use it to update the qubit object
          # N.B. This not updating is not implemented as of 9/2017

          # f is expected in Hz but f0 in GHz!
      Model = fit_mods.SlopedHangerAmplitudeModel
      # added reject outliers to be robust agains CBox data acq bug.
      # this should have no effect on regular data acquisition and is
      # only used in the guess.
      amplitude_guess = max(
          dm_tools.reject_outliers(np.sqrt(linecut_mag)))

      # Creating parameters and estimations
      S21min = (min(dm_tools.reject_outliers(np.sqrt(linecut_mag))) /
                max(dm_tools.reject_outliers(np.sqrt(linecut_mag))))

      Q = f0 / abs(min_frequency - max_frequency)
      Qe = abs(Q / abs(1 - S21min))

      # Note: input to the fit function is in GHz for convenience
      Model.set_param_hint('f0', value=f0 * 1e-9,
                           min=min(sweep_points) * 1e-9,
                           max=max(sweep_points) * 1e-9)
      Model.set_param_hint('A', value=amplitude_guess)
      Model.set_param_hint('Q', value=Q, min=1, max=50e6)
      Model.set_param_hint('Qe', value=Qe, min=1, max=50e6)
      # NB! Expressions are broken in lmfit for python 3.5 this has
      # been fixed in the lmfit repository but is not yet released
      # the newest upgrade to lmfit should fix this (MAR 18-2-2016)
      Model.set_param_hint('Qi', expr='abs(1./(1./Q-1./Qe*cos(theta)))',
                           vary=False)
      Model.set_param_hint('Qc', expr='Qe/cos(theta)', vary=False)
      Model.set_param_hint('theta', value=0, min=-np.pi / 2,
                           max=np.pi / 2)
      Model.set_param_hint('slope', value=0, vary=True)

      params = Model.make_params()

      
      data_x = sweep_points
      data_y = np.sqrt(linecut_mag)

      # # make sure that frequencies are in Hz
      # if np.floor(data_x[0]/1e8) == 0:  # frequency is defined in GHz
      #     data_x = data_x*1e9

      fit_res = Model.fit(data=data_y,
                          f=data_x, verbose=False)
      return fit_res

  def run_full_analysis(self, normalize=False, plot_linecuts=True,
                               linecut_log=False, colorplot_log=False,
                               plot_all=True, save_fig=True,
                               transpose=False, figsize=None, filtered=False,
                               subtract_mean_x=False, subtract_mean_y=False,
                               **kw):
          '''
          Args:
              linecut_log (bool):
                  log scale for the line cut?
                  Remember to set the labels correctly.
              colorplot_log (string/bool):
                  True/False for z axis scaling, or any string containing any
                  combination of letters x, y, z for scaling of the according axis.
                  Remember to set the labels correctly.

          '''
          close_file = kw.pop('close_file', True)
          self.fig_array = []
          self.ax_array = []

          for i, meas_vals in enumerate(self.measured_values[:1]):
              if filtered:
                  # print(self.measured_values)
                  # print(self.value_names)
                  if self.value_names[i] == 'Phase':
                      self.measured_values[i] = dm_tools.filter_resonator_visibility(
                                                          x=self.sweep_points,
                                                          y=self.sweep_points_2D,
                                                          z=self.measured_values[i],
                                                          **kw)

              if (not plot_all) & (i >= 1):
                  break
              # Linecuts are above because somehow normalization applies to both
              # colorplot and linecuts otherwise.
              if plot_linecuts:
                  fig, ax = plt.subplots(figsize=figsize)
                  self.fig_array.append(fig)
                  self.ax_array.append(ax)
                  savename = 'linecut_{}'.format(self.value_names[i])
                  fig_title = '{} {} \nlinecut {}'.format(
                      self.timestamp_string, self.measurementstring,
                      self.value_names[i])
                  a_tools.linecut_plot(x=self.sweep_points,
                                       y=self.sweep_points_2D,
                                       z=self.measured_values[i],
                                       y_name=self.parameter_names[1],
                                       y_unit=self.parameter_units[1],
                                       log=linecut_log,
                                       zlabel=self.zlabels[i],
                                       fig=fig, ax=ax, **kw)
                  ax.set_title(fig_title)
                  set_xlabel(ax, self.parameter_names[0],
                             self.parameter_units[0])
                  # ylabel is value units as we are plotting linecuts
                  set_ylabel(ax, self.value_names[i],
                             self.value_units[i])

                  if save_fig:
                      self.save_fig(fig, figname=savename,
                                    fig_tight=False, **kw)

              fig, ax = plt.subplots(figsize=figsize)
              self.fig_array.append(fig)
              self.ax_array.append(ax)
              if normalize:
                  print("normalize on")
              self.ax_array.append(ax)
              savename = 'Heatmap_{}'.format(self.value_names[i])
              fig_title = '{} {} \n{}'.format(
                  self.timestamp_string, self.measurementstring,
                  self.value_names[i])

              if "xlabel" not in kw:
                  kw["xlabel"] = self.parameter_names[0]
              if "ylabel" not in kw:
                  kw["ylabel"] = self.parameter_names[1]
              if "xunit" not in kw:
                  kw["xunit"] = self.parameter_units[0]
              if "yunit" not in kw:
                  kw["yunit"] = self.parameter_units[1]

              # subtract mean from each row/column if demanded
              plot_zvals = meas_vals.transpose()
              if subtract_mean_x:
                  plot_zvals = plot_zvals - np.mean(plot_zvals,axis=1)[:,None]
              if subtract_mean_y:
                  plot_zvals = plot_zvals - np.mean(plot_zvals,axis=0)[None,:]

              a_tools.color_plot(x=self.sweep_points,
                                 y=self.sweep_points_2D,
                                 z=plot_zvals,
                                 zlabel=self.zlabels[i],
                                 fig=fig, ax=ax,
                                 log=colorplot_log,
                                 transpose=transpose,
                                 normalize=normalize,
                                 **kw)
              ax.plot(self.f0s,self.sweep_points_2D,'ro-')
              
              ax.set_title(fig_title)

              if save_fig:
                  self.save_fig(fig, figname=savename, **kw)
          if close_file:
              self.finish()

class VNA_DAC_Analysis(VNA_TwoD_Analysis):
  """
  This function can be called with timestamp as its only argument. It will fit
  a cosine to any VNA DAC arc. The fit is stored in dac_fit_res.
  Use .sweetspotvalue to get a guess for the qubit sweetspot current,
  and use .current_to_flux to get the current requiired for one flux period
  """

  def __init__(self, timestamp,
             options_dict=None,
             do_fitting=True,
             extract_only=False,
             auto=True):
        super(VNA_TwoD_Analysis, self).__init__(timestamp = timestamp,
                                       options_dict=options_dict,
                                       extract_only=extract_only,
                                       auto=auto,
                                       do_fitting=do_fitting)

        linecut_fit_result = self.fit_linecuts()
        self.linecut_fit_result = linecut_fit_result
        f0s = []
        for res in self.linecut_fit_result:
            f0s.append(res.values['f0']*1e9)
        self.f0s = np.array(f0s)
        self.run_full_analysis()
        self.dac_fit_res = self.fit_dac_arc()
        self.sweet_spot_value = self.dac_fit_res.values['phase']
        self.current_to_flux = 1/self.dac_fit_res.values['frequency']
        self.plot_fit_result()

  def fit_dac_arc(self):
      DAC_values = self.sweep_points_2D
      f0s = self.f0s

      fit_res_list = []
      min_freq = 0
      max_freq = 4/(max(DAC_values)-min(DAC_values))
      test_freqs = np.linspace(min_freq,max_freq,21)
      DAC_range = max(DAC_values)-min(DAC_values)
      test_phases = np.linspace(min(DAC_values)+0.25*DAC_range,min(DAC_values)+0.75*DAC_range,21)
      for freq in test_freqs:
        for phase in test_phases:
          print(phase)
          Model = fit_mods.CosModel2
          Model.set_param_hint('frequency', value=freq,min=0, max = max_freq)
          Model.set_param_hint('offset', value=np.mean(f0s), min=min(f0s), max=max(f0s))
          Model.set_param_hint('amplitude', value=(max(f0s)-min(f0s))/2,min=0,max=max(f0s)-min(f0s))
          Model.set_param_hint('phase', value=phase, min = min(DAC_values), max = max(DAC_values))
          params = Model.make_params()
          fit_res_list.append(Model.fit(data=f0s,t=DAC_values, verbose=False))
      chi_sqrs = []
      for fit_res in fit_res_list:
        chi_sqrs.append(fit_res.chisqr)
      best_fit_index = np.argmin(np.array(chi_sqrs))
      return fit_res_list[best_fit_index]

  def plot_fit_result(self,normalize=False, plot_linecuts=True,
                               linecut_log=False, colorplot_log=False,
                               plot_all=True, save_fig=True,
                               transpose=False, figsize=None, filtered=False,
                               subtract_mean_x=False, subtract_mean_y=False,
                               **kw):
      fig, ax = plt.subplots(figsize=figsize)
      self.fig_array.append(fig)
      self.ax_array.append(ax)
      # print "unransposed",meas_vals
      # print "transposed", meas_vals.transpose()
      self.ax_array.append(ax)
      savename = 'Heatmap_{}'.format(self.value_names[0])
      fig_title = '{} {} \n{}'.format(
          self.timestamp_string, self.measurementstring,
          self.value_names[0])

      if "xlabel" not in kw:
          kw["xlabel"] = self.parameter_names[0]
      if "ylabel" not in kw:
          kw["ylabel"] = self.parameter_names[1]
      if "xunit" not in kw:
          kw["xunit"] = self.parameter_units[0]
      if "yunit" not in kw:
          kw["yunit"] = self.parameter_units[1]

      # subtract mean from each row/column if demanded
      plot_zvals = self.measured_values[0].transpose()
      if subtract_mean_x:
          plot_zvals = plot_zvals - np.mean(plot_zvals,axis=1)[:,None]
      if subtract_mean_y:
          plot_zvals = plot_zvals - np.mean(plot_zvals,axis=0)[None,:]

      a_tools.color_plot(x=self.sweep_points,
                         y=self.sweep_points_2D,
                         z=plot_zvals,
                         zlabel=self.zlabels[0],
                         fig=fig, ax=ax,
                         log=colorplot_log,
                         transpose=transpose,
                         normalize=normalize,
                         **kw)

      plot_dacs = np.linspace(min(self.sweep_points_2D),max(self.sweep_points_2D),101)
      plot_freqs = fit_mods.CosFunc(plot_dacs,**self.dac_fit_res.params)

      ax.plot(self.f0s,self.sweep_points_2D,'ro-')
      ax.plot(plot_freqs,plot_dacs,'b')
      
      ax.set_title(fig_title)

      if save_fig:
          self.save_fig(fig, figname=savename, **kw)



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
