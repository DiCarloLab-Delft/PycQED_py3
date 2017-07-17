"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of all the spectroscopy measurement analyses.
"""
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis import measurement_analysis as MA


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
        if len(self.data_dict['timestamps']) == 1:
            self.plot_frequency = np.squeeze(self.data_dict['freq'])
            self.plot_amp = np.squeeze(self.data_dict['amp'])
            self.plot_phase = np.squeeze(self.data_dict['phase'])
            print(self.plot_frequency.shape, self.plot_amp.shape, self.plot_phase.shape)
            self.plot_xlabel = 'Frequency (GHz)'
            self.amp_label = 'Transmission amplitude (arb. units)'
            self.phase_label = 'Transmission phase (degrees)'
            self.amp_range = self.options_dict.get('amp_range',None)
            self.phase_range = self.options_dict.get('phase_range',None)
            self.plotsize = self.options_dict.get('plotsize',(8,5))
        else:
            #TRANSPOSE ALSO NEEDS TO BE CODED FOR 2D
            self.plot_xvals = self.options_dict.get('xvals',np.array([[tt] for tt in range(len(self.data_dict['timestamps']))]) )
            self.plot_xwidth = self.options_dict.get('xwidth',None)
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

            # print(len(self.plot_frequency.shape), self.plot_amp.shape, len(self.plot_phase.shape))
            self.plot_xlabel = self.options_dict.get('xlabel','Scan number')
            self.freq_label = 'Frequency (GHz)'
            self.amp_label = 'Transmission amplitude (arb. units)'
            self.phase_label = 'Transmission phase (degrees)'
            self.freq_range = self.options_dict.get('freq_range',None)
            self.amp_range = self.options_dict.get('amp_range',None)
            self.phase_range = self.options_dict.get('phase_range',None)
            self.plotsize = self.options_dict.get('plotsize',(8,5))




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
                            'plotsize': self.plotsize
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
        self.plot_phase = np.unwrap(np.pi/180.*self.plot_phase)*180/np.pi
        self.amp_label = 'Transmission amplitude (V rms)'
        self.phase_label = 'Transmission phase (degrees)'
        self.plot_xlabel = 'Readout Frequency (GHz)'

    def run_fitting(self):
        if len(self.data_dict['timestamps']) == 1:
            self.fitma=MA.Homodyne_Analysis(timestamp=self.data_dict['timestamps'][0],auto=True,close_file=False)
            update_fit = False
            if self.fitparams_guess:
                update_fit = True
                for key, val in self.fitparams_guess.items():
                    self.fitma.fit_results.params = self.fitma.params
                    if key in self.fitma.fit_results.params:
                        self.fitma.fit_results.params[key].value = val
                self.fitma.fit_results.fit()
            self.fitma.finish()

        else:
            self.do_fitting = False

    def plot_fitting(self):
        if self.do_fitting:
            ax = self.axs['amp']
            ax.plot(self.fitma.sweep_points, self.fitma.fit_results.best_fit, 'r-', linewidth=1.5)
            textstr = 'f0 = %.5f $\pm$ %.1g GHz' % (self.fitma.fit_results.params['f0'].value, self.fitma.fit_results.params['f0'].stderr) + '\n' \
                'Q = %.4g $\pm$ %.0g' % (self.fitma.fit_results.params['Q'].value, self.fitma.fit_results.params['Q'].stderr) + '\n' \
                'Qc = %.4g $\pm$ %.0g' % (self.fitma.fit_results.params['Qc'].value, self.fitma.fit_results.params['Qc'].stderr) + '\n' \
                'Qi = %.4g $\pm$ %.0g' % (self.fitma.fit_results.params['Qi'].value, self.fitma.fit_results.params['Qi'].stderr)
            self.box_props = {key:val for key, val in self.fitma.box_props.items()}
            self.box_props.update({'linewidth':0})
            self.box_props['alpha'] = 0.
            ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=self.box_props)

    def plot(self, key_list=None, axs_dict=None, presentation_mode=None,no_label=False):
        super(ResonatorSpectroscopy, self).plot(key_list=key_list, axs_dict=axs_dict, presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()

