"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of all the spectroscopy measurement analyses.
"""
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
import pandas as pd
from pycqed.analysis import measurement_analysis as MA
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis import fitting_models as fit_mods
import lmfit

class Spectroscopy(ba.BaseDataAnalysis):
    def __init__(self, t_start,
                 options_dict,
                 t_stop = None,
                 extract_only=False,
                 auto=True):
        super(Spectroscopy, self).__init__(t_start, t_stop=t_stop,
                                             options_dict=options_dict,
                                             extract_only=extract_only)
        self.extract_fitparams = self.options_dict.get('fitparams',True)
        self.params_dict = {'freq':'sweep_points',
                            'amp':'amp',
                            'phase':'phase'}
        if self.extract_fitparams:
            self.params_dict.update({'fitparams':'fit_params'})

        self.numeric_params = ['freq', 'amp', 'phase']
        if 'qubit_label' in self.options_dict:
            self.labels.extend(self.options_dict['qubit_label'])
        if auto is True:
            self.run_analysis()

    def process_data(self):
        self.freq_label = 'Frequency (GHz)'
        self.amp_label = 'Transmission amplitude (arb. units)'

        self.phase_label = 'Transmission phase (degrees)'
        self.freq_range = self.options_dict.get('freq_range',None)
        self.amp_range = self.options_dict.get('amp_range',None)
        self.phase_range = self.options_dict.get('phase_range',None)
        self.plotsize = self.options_dict.get('plotsize',(8,5))
        if len(self.data_dict['timestamps']) == 1:
            self.plot_frequency = np.squeeze(self.data_dict['freq'])
            self.plot_amp = np.squeeze(self.data_dict['amp'])
            self.plot_phase = np.squeeze(self.data_dict['phase'])
            print(self.plot_frequency.shape, self.plot_amp.shape, self.plot_phase.shape)

        else:
            #TRANSPOSE ALSO NEEDS TO BE CODED FOR 2D
            self.plot_xvals = self.options_dict.get('xvals',np.array([[tt] for tt in range(len(self.data_dict['timestamps']))]))
            self.plot_xwidth = self.options_dict.get('xwidth',None)
            self.plot_xlabel = self.options_dict.get('xlabel','Scan number')
            if self.plot_xwidth == 'auto':
                x_diff = np.diff(np.ravel(self.plot_xvals))
                dx1 = np.concatenate(([x_diff[0]],x_diff))
                dx2 = np.concatenate((x_diff,[x_diff[-1]]))
                self.plot_xwidth = np.maximum(dx1,dx2)
                self.plot_frequency = np.array([[tt] for tt in self.data_dict['freq']])
                self.plot_phase = np.array([[tt] for tt in self.data_dict['phase']])
                self.plot_amp = np.array([np.array([tt]).transpose() for tt in self.data_dict['amp']])

            else:
                self.plot_frequency = self.data_dict['freq']
                self.plot_phase = self.data_dict['phase']
                self.plot_amp = self.data_dict['amp']







    def prepare_plots(self):

        if len(self.data_dict['timestamps']) == 1:
            plot_fn = self.plot_line
            self.plot_dicts['amp'] = {'plotfn': plot_fn,
                            'xvals': self.plot_frequency,
                            'yvals': self.plot_amp,
                            'title': 'Spectroscopy amplitude: %s'%(self.timestamps[0]),
                            'xlabel': self.plot_xlabel,
                            'ylabel': self.amp_label,
                            'yrange': self.amp_range,
                            'plotsize': self.plotsize
                            }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                            'xvals': self.plot_frequency,
                            'yvals': self.plot_phase,
                            'title': 'Spectroscopy phase: %s'%(self.timestamps[0]),
                            'xlabel': self.plot_xlabel,
                            'ylabel': self.phase_label,
                            'yrange': self.phase_range,
                            'plotsize': self.plotsize
                            }
        else:
            print('plotting not yet coded up for multiple traces')
            self.plotcbar = self.options_dict.get('colorbar',True)
            plot_fn = self.plot_colorx #(self, pdict, axs)
            self.plot_dicts['amp'] = {'plotfn': plot_fn,
                            'xvals': self.plot_xvals,
                            'xwidth': self.plot_xwidth,
                            'yvals': self.plot_frequency,
                            'zvals': self.plot_amp,
                            'title': 'Spectroscopy amplitude: %s'%(self.timestamps[0]),
                            'xlabel': self.plot_xlabel,
                            'ylabel': self.freq_label,
                            'zlabel': self.amp_label,
                            'yrange': self.freq_range,
                            'zrange': self.amp_range,
                            'plotsize': self.plotsize,
                            'plotcbar': self.plotcbar
                            }
            # bla



    def plot_for_presentation(self, key_list=None, no_label=False):
        super(Spectroscopy, self).plot_for_presentation(key_list=key_list, no_label=no_label)
        for key in key_list:
            pdict = self.plot_dicts[key]
            if key == 'amp':
                if pdict['plotfn'] == self.plot_line:
                    ymin, ymax = 0, 1.2*np.max(np.ravel(pdict['yvals']))
                    self.axs[key].set_ylim(ymin, ymax)
                    self.axs[key].set_ylabel('Transmission amplitude (V rms)')




class ResonatorSpectroscopy(Spectroscopy):
    def __init__(self, t_start,
                 options_dict,
                 t_stop = None,
                 do_fitting = True,
                 extract_only=False,
                 auto=True):
        super(ResonatorSpectroscopy, self).__init__(t_start, t_stop=t_stop,
                                             options_dict=options_dict,
                                             extract_only=extract_only,
                                             auto=False)
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess',{})
        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorSpectroscopy, self).process_data()
        self.subtract_background = self.options_dict.get('subtract_background',False)
        self.amp_label = 'Transmission amplitude (V rms)'
        self.phase_label = 'Transmission phase (degrees)'
        if len(self.data_dict['timestamps']) == 1:
            self.plot_phase = np.unwrap(np.pi/180.*self.plot_phase)*180/np.pi
            self.plot_xlabel = 'Readout Frequency (GHz)'
        else:
            pass

    def prepare_fitting(self):
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex.
        fit_options = self.options_dict['fit_options']
        fitting_model = fit_options['model']
        if self.options_dict['subtract_background'] == True:
            self.do_subtract_background(thres=self.options_dict['background_thres'],
                                        back_dict=self.options_dict['background_dict'])
        if fitting_model == 'hanger':
            fit_fn = fit_mods.SlopedHangerFuncAmplitude
            fit_guess_fn = SlopedHangerFuncAmplitudeGuess
        elif fitting_model == 'simple_hanger':
            fit_fn = fit_mods.HangerFuncAmplitude
            raise NotImplementedError('This functions guess function is not coded up yet')
            #TODO HangerFuncAmplitude Guess
        elif fitting_model == 'lorentzian':
            print('This functions guess function is not coded up yet')
            fit_fn = fit_mods.Lorentzian
            #TODO LorentzianGuess
        elif fitting_model == 'complex':
            print('This functions guess function is not coded up yet')
            fit_fn = fit_mods.HangerFuncComplex
            #TODO HangerFuncComplexGuess

        if len(self.data_dict['timestamps']) == 1:
            self.fit_dict = {'fit_fn': fit_fn,
                             'fit_guess_fn': fit_guess_fn,
                             'fit_yvals': self.plot_amp,
                             'fit_xvals': {'f': self.plot_frequency}
                             }
        else:
            self.fit_dict = {'fit_fn': fit_fn,
                             'fit_guess_fn': fit_guess_fn,
                             'fit_yvals': [np.squeeze(tt) for tt in self.plot_amp],
                             'fit_xvals': np.squeeze([{'f': tt[0]} for tt in self.plot_frequency])}

    def do_subtract_background(self,thres=None,back_dict=None,):
        if len(self.data_dict['timestamps']) == 1:
            pass
        else:
            x_filtered = []
            y_filtered = []
            for tt in range(len(self.data_dict['timestamps'])):
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
                df = 2*f0/Q
                fmin = f0-df
                fmax = f0+df
                indices = np.logical_or(x < fmin*1e9, x > fmax*1e9)
                x_filtered.append(x[indices])
                y_filtered.append(y[indices])
            self.background = pd.concat([pd.Series(y_filtered[tt], index=x_filtered[tt])
                       for tt in range(len(self.data_dict['timestamps']))], axis=1).mean(axis=1)
            background_vals = self.background.reset_index().values
            freq = background_vals[:,0]
            amp = background_vals[:,1]
            # thres = 0.0065
            indices = amp < thres
            freq = freq[indices]*1e-9
            amp = amp[indices]
            fit_fn = double_cos_linear_offset
            model = lmfit.Model(fit_fn)
            fit_yvals = amp
            fit_xvals = {'t':freq}
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

            for tt in range(len(self.data_dict['timestamps'])):
                divide_vals = fit_fn(np.squeeze(self.plot_frequency)[tt]*1e-9,**fit_res.best_values)
                self.plot_amp[tt] = np.array([np.array([np.divide(np.squeeze(self.plot_amp[tt]),divide_vals)])]).transpose()



    def plot_fitting(self):
        if self.do_fitting:
            fit_results = self.fit_dict['fit_results']
            ax = self.axs['amp']
            if len(self.data_dict['timestamps']) == 1:
                ax.plot(list(self.fit_dict['fit_xvals'].values())[0], fit_results.best_fit, 'r-', linewidth=1.5)
                textstr = 'f0 = %.5f $\pm$ %.1g GHz' % (fit_results.params['f0'].value, fit_results.params['f0'].stderr) + '\n' \
                    'Q = %.4g $\pm$ %.0g' % (fit_results.params['Q'].value, fit_results.params['Q'].stderr) + '\n' \
                    'Qc = %.4g $\pm$ %.0g' % (fit_results.params['Qc'].value, fit_results.params['Qc'].stderr) + '\n' \
                    'Qi = %.4g $\pm$ %.0g' % (fit_results.params['Qi'].value, fit_results.params['Qi'].stderr)
                box_props = dict(boxstyle='Square', facecolor='white', alpha=0.8)
                self.box_props = {key:val for key, val in box_props.items()}
                self.box_props.update({'linewidth':0})
                self.box_props['alpha'] = 0.
                ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=self.box_props)
            else:
                reso_freqs = [fit_results[tt].params['f0'].value*1e9 for tt in range(len(self.data_dict['timestamps']))]
                ax.plot(np.squeeze(self.plot_xvals),
                        reso_freqs,
                        'o',
                        color='m',
                        markersize=3)

    def plot(self, key_list=None, axs_dict=None, presentation_mode=None,no_label=False):
        super(ResonatorSpectroscopy, self).plot(key_list=key_list, axs_dict=axs_dict, presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()

