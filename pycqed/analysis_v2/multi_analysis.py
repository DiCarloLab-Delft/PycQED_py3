import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
import pycqed.measurement.hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
import pandas as pd
from scipy import linalg
import cmath as cm
from pycqed.analysis import fitting_models as fit_mods
import lmfit
from copy import deepcopy

class Multi_AllXY_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        t_start: str = None,
        t_stop: str = None,
        label: str = "Multi_AllXY",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        auto=True,
        qubits: list = None
    ):
        super().__init__(
            label=label,
            t_start = t_start,
            t_stop = t_stop
        )
        self.qubits = qubits
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_stop, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['points'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['points'] = self.raw_data_dict['points']
        nm = len(self.proc_data_dict['points']) # number of measurement points
        
        
        ### Creating the ideal data ###
        if nm == 42:
            self.proc_data_dict['ideal_data'] = np.concatenate((0 * np.ones(10), 0.5 * np.ones(24), np.ones(8)))
            self.proc_data_dict['locs'] = np.arange(1, 42, 2)
        else:
            self.proc_data_dict['ideal_data'] = np.concatenate((0 * np.ones(5), 0.5 * np.ones(12),np.ones(4)))
            self.proc_data_dict['locs'] = np.arange(0, 21, 1)
            
        for q in self.qubits:
            
            ### callibration points for normalization ###
            if nm == 42:
                zero = (self.raw_data_dict['{}_data'.format(q)][0]+self.raw_data_dict['{}_data'.format(q)][1])/2 ## II is set as 0 cal point
                one = (self.raw_data_dict['{}_data'.format(q)][-5]+self.raw_data_dict['{}_data'.format(q)][-6]+self.raw_data_dict['{}_data'.format(q)][-7]+self.raw_data_dict['{}_data'.format(q)][-8])/4  ## average of XI and YI is set as the 1 cal point
                data_normalized = (self.raw_data_dict['{}_data'.format(q)]-zero)/(one-zero)
            else:
                zero = self.raw_data_dict['{}_data'.format(q)][0] ## II is set as 0 cal point
                one = (self.raw_data_dict['{}_data'.format(q)][-3]+self.raw_data_dict['{}_data'.format(q)][-4])/2  ## average of XI and YI is set as 1 cal point
                data_normalized = (self.raw_data_dict['{}_data'.format(q)]-zero)/(one-zero)    

            ### Analyzing Data ###

            data_error = data_normalized-self.proc_data_dict['ideal_data']
            deviation = np.mean(abs(data_error))  

            self.proc_data_dict['normalized_data_{}'.format(q)] = data_normalized
            self.proc_data_dict['deviation_{}'.format(q)] = deviation
        
            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['AllXY_'+q] = {
                                        'plotfn': plot_Multi_AllXY,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'AllXY_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0]
                                        }

def plot_Multi_AllXY(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,1))
    
    labels = ['II', 'XX', 'YY', 'XY', 'YX','xI', 'yI', 'xy', 'yx', 'xY', 'yX',
              'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy','XI', 'YI', 'xx', 'yy']    
    q = qubit
    ax.plot(data['points'],data['normalized_data_{}'.format(q)],'o-',label='Qubit '+q)
    ax.plot(data['points'],data['ideal_data'],label='Ideal data')
    deviation_text = r'Deviation: %.5f' %data['deviation_{}'.format(q)]
    ax.text(1, 1, deviation_text, fontsize=11)
    ax.xaxis.set_ticks(data['locs'])
    ax.set_xticklabels(labels, rotation=60)
    ax.set(ylabel=r'$F$ $|1 \rangle$', title='AllXY for Qubit {}'.format(q))
    ax.legend(loc=4)
    ax.set_title(title)


class Multi_Rabi_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        auto=True,
        qubits: list = None,
    ):
        super().__init__(
            label=label,
            t_start = ts
        )
        self.qubits = qubits
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_stop, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['amps'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['quantities_of_interest'] = {}
        self.proc_data_dict['amps'] = self.raw_data_dict['amps']
        amps = self.proc_data_dict['amps']
        for q in self.qubits:
            data = self.raw_data_dict['{}_data'.format(q)]
            nor_data = data - (max(data)+min(data))/2
            self.proc_data_dict['{}_nor_data'.format(q)] = nor_data
            
            cos_mod = fit_mods.CosModel
            
            fft_of_data = np.fft.fft(nor_data, norm='ortho')
            power_spectrum = np.abs(fft_of_data) ** 2
            index_of_fourier_maximum = np.argmax(
                    power_spectrum[1:len(fft_of_data) // 2]) + 1

            top_x_val = np.take(amps, np.argmax(nor_data))
            bottom_x_val = np.take(amps,np.argmin(nor_data))
            if index_of_fourier_maximum == 1:
                freq_guess = 1.0 / (2.0 * np.abs(bottom_x_val - top_x_val))
            else:
                fft_scale = 1.0 / (amps[-1] -
                                   amps[0])
                freq_guess = fft_scale * index_of_fourier_maximum
                
            diff = 0.5 * (max(nor_data) -
                          min(nor_data))
            amp_guess = -diff

                
                
            cos_mod.set_param_hint('amplitude',
                                   value=amp_guess,
                                   vary=True)
            cos_mod.set_param_hint('phase',
                                   value=0,
                                   vary=False)
            cos_mod.set_param_hint('frequency',
                                   value=freq_guess,
                                   vary=True,
                                   min=(
                                       1 / (100 * amps[-1])),
                                   max=(20 / amps[-1]))
            offset_guess = 0
            cos_mod.set_param_hint('offset',
                                   value=offset_guess,
                                   vary=True)
            cos_mod.set_param_hint('period',
                                   expr='1/frequency',
                                   vary=False)
            params = cos_mod.make_params()
            fit_res = cos_mod.fit(data=nor_data,
                                          t=amps,
                                          params=params)
            
            self.proc_data_dict['{}_fitted_data'.format(q)] = fit_res.best_fit
            self.proc_data_dict['{}_fit_res'.format(q)] = fit_res
            self.proc_data_dict['quantities_of_interest'][q] = fit_res.best_values
            f = fit_res.best_values['frequency']
            self.proc_data_dict['quantities_of_interest'][q]['pi_amp'] = 1/(2*f)
        
            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['rabi_'+q] = {
                                        'plotfn': plot_Multi_Rabi,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'Rabi_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0],
                                        'plotsize': (10,10)
                                        }

def plot_Multi_Rabi(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    q = qubit
    amps = data['amps']
    nor_data = data['{}_nor_data'.format(q)]
    fit_data = data['{}_fitted_data'.format(q)]
    frequency = data['quantities_of_interest'][q]['frequency']
    pi = 1/(2*frequency)
    frequency_error = data['{}_fit_res'.format(q)].params['frequency'].stderr
    pi_error = 2*pi**2*frequency_error
    pi_text = r'$\mathrm{\pi}$ = %.3f' %pi + '- amp a.u. $\pm$ '
    pi_error_text = r'%.6f' %pi_error + 'a.u.\n'
    pi_2_text = r'$\mathrm{\pi}$ = %.3f' %(pi/2) + '- amp a.u. $\pm$ '
    pi_2_error_text = r'%.6f' %pi_error + 'a.u.'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ypos = min(nor_data)*1.4
    ax.text(0.25, ypos, pi_text+pi_error_text+pi_2_text+pi_2_error_text, fontsize=11,bbox = props)
    ax.plot(amps,nor_data,'-o')
    ax.plot(amps,fit_data,'-r')
    ax.set(ylabel=r'V_homodyne (a.u.)')
    ax.set(xlabel= r'Channel_amp (a.u.)')
    ax.set_title(title)

class Multi_Ramsey_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto=True,
        qubits: list = None,
        times: list = None,
        artificial_detuning: float = None
    ):
        super().__init__(
            label=label,
            t_start = ts
        )
        
        self.qubits = qubits
        self.times= times
        if artificial_detuning is None:
            artificial_detuning = 0
        self.artificial_detuning = artificial_detuning
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.raw_data_dict['artificial_detuning'] = self.artificial_detuning
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_start, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['points'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
            self.raw_data_dict['{}_times'.format(q)] = self.times[i]
            param_spec_old_freq = {'{}_freq_old'.format(q): ('Instrument settings/{}'.format(q), 'attr:freq_qubit')}
            old_freq = h5d.extract_pars_from_datafile(data_fp, param_spec_old_freq)
            self.raw_data_dict['{}_freq_old'.format(q)] = float(old_freq['{}_freq_old'.format(q)])
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['quantities_of_interest'] = {}
        self.proc_data_dict['quantities_of_interest']['artificial_detuning'] = self.artificial_detuning
        for i, q in enumerate(self.qubits):
             ### normalize data using cal points ###
            self.proc_data_dict['{}_times'.format(q)]=self.raw_data_dict['{}_times'.format(q)]
            data = self.raw_data_dict['{}_data'.format(q)]
            zero = (data[-4]+data[-3])/2
            one = (data[-2]+data[-1])/2
            nor_data =  (data-zero)/(one-zero)
            self.proc_data_dict['{}_nor_data'.format(q)] = nor_data
            old_freq = self.raw_data_dict['{}_freq_old'.format(q)]
            self.proc_data_dict['{}_freq_old'.format(q)]=old_freq
            ### fit to normalized data ###
            x = self.proc_data_dict['{}_times'.format(q)][0:-4]
            y = self.proc_data_dict['{}_nor_data'.format(q)][0:-4]

            damped_osc_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
            average = np.mean(y)

            ft_of_data = np.fft.fft(y)
            index_of_fourier_maximum = np.argmax(np.abs(ft_of_data[1:len(ft_of_data) // 2])) + 1
            max_ramsey_delay = x[-1] - x[0]

            fft_axis_scaling = 1 / (max_ramsey_delay)
            freq_est = fft_axis_scaling * index_of_fourier_maximum
            est_number_of_periods = index_of_fourier_maximum

            damped_osc_mod.set_param_hint('frequency',
                                          value=freq_est,
                                          min=(1/(100 * x[-1])),
                                          max=(20/x[-1]))

            if (np.average(y[:4]) >
                    np.average(y[4:8])):
                phase_estimate = 0
            else:
                phase_estimate = np.pi
            damped_osc_mod.set_param_hint('phase',
                                          value=phase_estimate, vary=True)

            amplitude_guess = 1
            damped_osc_mod.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.4,
                                          max=4.0)
            damped_osc_mod.set_param_hint('tau',
                                          value=x[1]*10,
                                          min=x[1],
                                          max=x[1]*1000)
            damped_osc_mod.set_param_hint('exponential_offset',
                                          value=0.5,
                                          min=0.4,
                                          max=4.0)
            damped_osc_mod.set_param_hint('oscillation_offset',
                                          value=0,
                                          vary=False)
            damped_osc_mod.set_param_hint('n',
                                          value=1,
                                          vary=False)
            params = damped_osc_mod.make_params()

            fit_res = damped_osc_mod.fit(data=y,
                                         t=x,
                                         params=params)
            self.proc_data_dict['{}_fitted_data'.format(q)] = fit_res.best_fit
            self.proc_data_dict['{}_fit_res'.format(q)] = fit_res
            self.proc_data_dict['quantities_of_interest'][q] = fit_res.best_values
            new_freq = old_freq + self.artificial_detuning - fit_res.best_values['frequency']
            self.proc_data_dict['quantities_of_interest'][q]['freq_new'] = new_freq
            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['Ramsey_'+q] = {
                                        'plotfn': plot_Multi_Ramsey,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'Ramsey_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0],
                                        'plotsize': (10,10)
                                        }

def plot_Multi_Ramsey(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
           
    q = qubit
    times = data['{}_times'.format(q)]*1e6
    nor_data = data['{}_nor_data'.format(q)]
    fit_data = data['{}_fitted_data'.format(q)]
    
    
    fq_old = data['{}_freq_old'.format(q)]*1e-9
    fq_new = data['quantities_of_interest'][q]['freq_new']*1e-9
    fq_new_error = data['{}_fit_res'.format(q)].params['frequency'].stderr*1e-9
    df = (fq_new-fq_old)*1e3
    df_error = fq_new_error*1e3
    T2 = data['quantities_of_interest'][q]['tau']*1e6
    T2_error = data['{}_fit_res'.format(q)].params['tau'].stderr*10e6
    art_det = data['quantities_of_interest']['artificial_detuning']*1e-6
    
    fq_old_text = r'$f_{qubit_old}$ = %.9f' %fq_old + 'GHz'
    fq_new_text = r'$f_{qubit_new}$ = %.9f' %fq_new + 'GHz $\pm$ '
    fq_new_error_text = '%.9f' %fq_new_error + 'GHz'
    df_text = r'$\mathrm{\Delta}f_{qubit_new}$ = %.9f' %df+ 'MHz $\pm$ '
    df_error_text = '%.9f' %df_error + 'MHz'
    T2_text = r'T2_star = %.9f' %T2 + '$\mathrm{\mu}$s $\pm$ '
    T2_error_text = '%.9f' %T2_error + '$\mathrm{\mu}$s'
    art_det_text = r'artificial detuning = %.2f' %art_det + 'MHz'
    text = fq_old_text+ '\n'+ fq_new_text+fq_new_error_text+'\n'+df_text+df_error_text+'\n'+T2_text+T2_error_text+'\n'+art_det_text
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    xpos = (times[-1]+times[0])*0.3
    ypos = min(nor_data)-0.35
    ax.text(xpos, ypos, text, fontsize=11,bbox = props)
    
    ax.plot(times,nor_data,'-o')
    ax.plot(times[:-4],fit_data,'-r')
    ax.set(ylabel=r'$F$ $|1 \rangle$')
    ax.set(xlabel= r'Time ($\mathrm{\mu}$s)')
    ax.set_title(title)

class Multi_T1_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto=True,
        qubits: list = None,
        times: list = None
    ):
        super().__init__(
            label=label,
            t_start = ts
        )
        
        self.qubits = qubits
        self.times= times
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_start, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['points'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
            self.raw_data_dict['{}_times'.format(q)] = self.times[i]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['quantities_of_interest'] = {}
        for i, q in enumerate(self.qubits):
             ### normalize data using cal points ###
            self.proc_data_dict['{}_times'.format(q)] = self.raw_data_dict['{}_times'.format(q)]
            data = self.raw_data_dict['{}_data'.format(q)]
            zero = (data[-4]+data[-3])/2
            one = (data[-2]+data[-1])/2
            nor_data =  (data-zero)/(one-zero)
            self.proc_data_dict['{}_nor_data'.format(q)] = nor_data
            
            ### fit to normalized data ###
            times = self.proc_data_dict['{}_times'.format(q)]
            dt=times[1]-times[0]

            fit_mods.ExpDecayModel.set_param_hint('amplitude',
                                              value=1,
                                              min=0,
                                              max=2)
            fit_mods.ExpDecayModel.set_param_hint('tau',
                                                  value=dt * 50,
                                                  min=dt,
                                                  max=dt * 1000)
            fit_mods.ExpDecayModel.set_param_hint('offset',
                                                  value=0,
                                                  vary=False)
            fit_mods.ExpDecayModel.set_param_hint('n',
                                                  value=1,
                                                  vary=False)
            params = fit_mods.ExpDecayModel.make_params()

            fit_res = fit_mods.ExpDecayModel.fit(data=nor_data[:-4],
                                                 t=times[:-4],
                                                 params=params)
            
            self.proc_data_dict['{}_fitted_data'.format(q)] = fit_res.best_fit
            self.proc_data_dict['{}_fit_res'.format(q)] = fit_res
            self.proc_data_dict['quantities_of_interest'][q] = fit_res.best_values

            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['T1_'+q] = {
                                        'plotfn': plot_Multi_T1,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'T1_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0],
                                        'plotsize': (5,4)
                                        }
            

def plot_Multi_T1(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
           
    q = qubit
    times = data['{}_times'.format(q)]*1e6
    nor_data = data['{}_nor_data'.format(q)]
    fit_data = data['{}_fitted_data'.format(q)]
    T1 = data['quantities_of_interest'][q]['tau']*1e6
    T1_error = data['{}_fit_res'.format(q)].params['tau'].stderr*1e6
    T1_text = r'T1 = %.3f' %T1 + ' $\mathrm{\mu}$s $\pm$ '
    T1_error_text = r'%.3f' %T1_error + ' $\mathrm{\mu}$s'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    xpos = (times[-1]+times[0])*0.5
    ypos = -0.3 #min(nor_data)-0.5
    ax.text(xpos, ypos, T1_text+T1_error_text, 
        fontsize=11,
        bbox = props, 
        horizontalalignment='center',
        verticalalignment='center')
    ax.plot(times,nor_data,'-o')
    ax.plot(times[:-4],fit_data,'-r')
    ax.set(ylabel=r'$F$ $|1 \rangle$')
    ax.set(xlabel= r'Time ($\mathrm{\mu}$s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    
class Multi_Echo_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto=True,
        qubits: list = None,
        times: list = None
    ):
        super().__init__(
            label=label,
            t_start = ts
        )
        
        self.qubits = qubits
        self.times= times
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_start, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['points'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
            self.raw_data_dict['{}_times'.format(q)] = self.times[i]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['quantities_of_interest'] = {}
        for i, q in enumerate(self.qubits):
             ### normalize data using cal points ###
            self.proc_data_dict['{}_times'.format(q)]=self.raw_data_dict['{}_times'.format(q)]
            data = self.raw_data_dict['{}_data'.format(q)]
            zero = (data[-4]+data[-3])/2
            one = (data[-2]+data[-1])/2
            nor_data =  (data-zero)/(one-zero)
            self.proc_data_dict['{}_nor_data'.format(q)] = nor_data
            
            ### fit to normalized data ###
            x = self.proc_data_dict['{}_times'.format(q)][0:-4]
            dx= x[1]-x[0]

            y = self.proc_data_dict['{}_nor_data'.format(q)][0:-4]

            #damped_osc_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
            

            # make frequency estimate
            ft_of_data = np.fft.fft(y)
            index_of_fourier_maximum = np.argmax(np.abs(ft_of_data[1:len(ft_of_data) // 2])) + 1
            max_echo_delay = x[-1] - x[0]

            fft_axis_scaling = 1 / (max_echo_delay)
            freq_est = fft_axis_scaling * index_of_fourier_maximum
            est_number_of_periods = index_of_fourier_maximum

            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('frequency',
                                          value=freq_est,
                                          min=(1/(100 * x[-1])),
                                          max=(20/x[-1]),
                                          vary=True)

            # make phase estimate
            if (np.average(y[:3]) >
                    np.average(y[3:6])):
                phase_estimate = 0
            else:
                phase_estimate = np.pi

            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('phase',
                                          value=phase_estimate,
                                          min=0,
                                          max=2*np.pi,
                                          vary=True)

            amplitude_guess = 0.4
            #damped_osc_mod.
            fit_mods.ExpDampOscModel.set_param_hint('amplitude',
                                          value=amplitude_guess,
                                          min=0.2,
                                          max=0.8,
                                          vary=True)
            
            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('tau',
                                          value=max_echo_delay/2,
                                          min=dx,
                                          max=dx*1000,
                                          vary=True)

            average = np.mean(y)
            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('exponential_offset',
                                          value=average,
                                          min=0.3,
                                          max=0.7,
                                          vary=True)

            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('oscillation_offset',
                                          value=0,
                                          vary=False)
            
            #damped_osc_mod
            fit_mods.ExpDampOscModel.set_param_hint('n',
                                          value=1,
                                          vary=False)
            
            #params = damped_osc_mod.make_params()
            params = fit_mods.ExpDampOscModel.make_params()


            # perform the fit
            #fit_res = damped_osc_mod.fit(data=y, t=x, params=params)
            fit_res = fit_mods.ExpDampOscModel.fit(data=y, t=x, params=params)

            # for diagnostics only
            #print(fit_res.fit_report())

            self.proc_data_dict['{}_fitted_data'.format(q)] = fit_res.best_fit
            self.proc_data_dict['{}_fit_res'.format(q)] = fit_res
            self.proc_data_dict['quantities_of_interest'][q] = fit_res.best_values
            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['Echo_'+q] = {
                                        'plotfn': plot_Multi_Echo,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'Echo_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0],
                                        'plotsize': (5,4)
                                        }

def plot_Multi_Echo(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
           
    q = qubit
    times = data['{}_times'.format(q)]*1e6
    nor_data = data['{}_nor_data'.format(q)]
    fit_data = data['{}_fitted_data'.format(q)]
    
    T2 = data['quantities_of_interest'][q]['tau']*1e6
    T2_error = data['{}_fit_res'.format(q)].params['tau'].stderr*1e6

    T2_text = r'T2_echo = %.3f' %T2 + ' $\mathrm{\mu}$s $\pm$ '
    T2_error_text = r'%.3f' %T2_error + ' $\mathrm{\mu}$s'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    xpos = (times[-1]+times[0])*0.5
    ypos = -0.3     #min(nor_data)-0.25
    ax.text(xpos, ypos, T2_text+T2_error_text, 
        fontsize=10, bbox = props, 
        horizontalalignment='center',
        verticalalignment='center')
    ax.plot(times,nor_data,'-o')
    ax.plot(times[:-4],fit_data,'-r')
    ax.set(ylabel= r'$F$ $|1 \rangle$')
    ax.set(xlabel= r'Time ($\mathrm{\mu}$s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)

class Multi_Flipping_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto=True,
        qubits: list = None,
    ):
        super().__init__(
            label=label,
            t_start = ts
        )
        
        self.qubits = qubits
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_start, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['number_flips'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data'.format(q)] = data['data'][:,i+1]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['quantities_of_interest'] = {}
        number_flips = self.raw_data_dict['number_flips']
        self.proc_data_dict['number_flips'] = number_flips
        for i, q in enumerate(self.qubits):
             ### normalize data using cal points ###
            data = self.raw_data_dict['{}_data'.format(q)]
            zero = (data[-4]+data[-3])/2
            one = (data[-2]+data[-1])/2
            nor_data =  (data-zero)/(one-zero)
            self.proc_data_dict['{}_nor_data'.format(q)] = nor_data
            self.proc_data_dict['quantities_of_interest'][q]={}
            
            ### fit to normalized data ###
            x = number_flips[:-4]
            y = self.proc_data_dict['{}_nor_data'.format(q)][0:-4]
            
            ### cos fit ###
            cos_fit_mod = fit_mods.CosModel
            params = cos_fit_mod.guess(cos_fit_mod,data=y,t=x)
            cos_mod = lmfit.Model(fit_mods.CosFunc)
            fit_res_cos = cos_mod.fit(data=y,t=x,params = params)
            
            t = np.linspace(x[0],x[-1],200)
            cos_fit = fit_mods.CosFunc(t = t ,amplitude = fit_res_cos.best_values['amplitude'],
                                   frequency = fit_res_cos.best_values['frequency'],
                                   phase = fit_res_cos.best_values['phase'],
                                   offset = fit_res_cos.best_values['offset'])
            self.proc_data_dict['{}_cos_fit_data'.format(q)] = cos_fit
            self.proc_data_dict['{}_cos_fit_res'.format(q)] = fit_res_cos
            self.proc_data_dict['quantities_of_interest'][q]['cos_fit'] = fit_res_cos.best_values
            
            
                
            ### line fit ###
            poly_mod = lmfit.models.PolynomialModel(degree=1)
            c0_guess = x[0]
            c1_guess = (y[-1]-y[0])/(x[-1]-x[0])
            poly_mod.set_param_hint('c0',value=c0_guess,vary=True)
            poly_mod.set_param_hint('c1',value=c1_guess,vary=True)
            poly_mod.set_param_hint('frequency', expr='-c1/(2*pi)')
            params = poly_mod.make_params()
            fit_res_line = poly_mod.fit(data=y,x=x,params = params)
            self.proc_data_dict['{}_line_fit_data'.format(q)] = fit_res_line.best_fit
            self.proc_data_dict['{}_line_fit_res'.format(q)] = fit_res_line
            self.proc_data_dict['quantities_of_interest'][q]['line_fit'] = fit_res_line.best_values
            ### calculating scale factors###
            sf_cos = (1+fit_res_cos.params['frequency'])**2
            phase = np.rad2deg(fit_res_cos.params['phase'])%360
            if phase > 180:
                sf_cos = 1/sf_cos
            self.proc_data_dict['quantities_of_interest'][q]['cos_fit']['sf'] = sf_cos
            
            sf_line = (1+fit_res_line.params['frequency'])**2
            self.proc_data_dict['quantities_of_interest'][q]['line_fit']['sf'] = sf_line
            ### choose correct sf ###
            msg = 'Scale factor based on '
            if fit_res_line.bic<fit_res_cos.bic:
                scale_factor = sf_line
                msg += 'line fit\n'   
            else:
                scale_factor = sf_cos
                msg += 'cos fit\n'
            msg += 'line fit: {:.4f}\n'.format(sf_line)
            msg += 'cos fit: {:.4f}'.format(scale_factor)
            self.proc_data_dict['{}_scale_factor'.format(q)] = scale_factor
            self.proc_data_dict['{}_scale_factor_msg'.format(q)] = msg
            
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['flipping_'+q] = {
                                        'plotfn': plot_Multi_flipping,
                                        'data': self.proc_data_dict,
                                        'qubit': q,
                                        'title': 'flipping_'+q+'_'
                                        +self.raw_data_dict['timestamps'][0],
                                        'plotsize': (10,7)
                                        }

def plot_Multi_flipping(qubit, data,title, ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
           
    q = qubit
    number_flips = data['number_flips']
    t = np.linspace(number_flips[0],number_flips[-5],200)
    nor_data = data['{}_nor_data'.format(q)]
    fit_data_line = data['{}_line_fit_data'.format(q)]
    fit_data_cos = data['{}_cos_fit_data'.format(q)]
    text = data['{}_scale_factor_msg'.format(q)]
    xpos = (number_flips[-1]+number_flips[0])*0.3
    ypos = -0.3
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(xpos, ypos, text, fontsize=11,bbox = props)
    
    ax.plot(number_flips,nor_data,'-o')
    ax.plot(number_flips[:-4],fit_data_line,'-',label='line fit')
    ax.plot(t,fit_data_cos,'-',label='cosine fit')
    ax.legend()
    ax.set(ylabel=r'$F$ $|1 \rangle$')
    ax.set(xlabel= r'number of flips (#)')
    ax.set_title(title)
    4444444

class Multi_Motzoi_Analysis(ba.BaseDataAnalysis):
    def __init__(
        self,
        ts: str = None,
        label: str = "",
        data_file_path: str = None,
        options_dict: dict = None,
        extract_only: bool = False,
        close_figs=False,
        do_fitting: bool = False,
        save_qois: bool = True,
        auto=True,
        qubits: list = None,
                ):
        super().__init__(
                        label=label,
                        t_start = ts
                        )
        
        self.qubits = qubits
        if auto:
            self.run_analysis()
            
    def extract_data(self):
        self.raw_data_dict = {} 
        self.timestamps = a_tools.get_timestamps_in_range(self.t_start,self.t_start, label = self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps
        data_fp = a_tools.get_datafilepath_from_timestamp(self.timestamps[0])
        param_spec = {'data': ('Experimental Data/Data', 'dset')}
        data = h5d.extract_pars_from_datafile(data_fp, param_spec)
        self.raw_data_dict['amps'] = data['data'][:,0]
        for i, q in enumerate(self.qubits):
            self.raw_data_dict['{}_data_A'.format(q)] = data['data'][:,2*i+1]
            self.raw_data_dict['{}_data_B'.format(q)] = data['data'][:,2*i+2]
        self.raw_data_dict['folder'] = os.path.dirname(data_fp)
        
    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        self.proc_data_dict['quantities_of_interest'] = {}
        amps = self.proc_data_dict['amps']
        for q in self.qubits:
            self.proc_data_dict['quantities_of_interest'][q] = {}
            A = self.raw_data_dict['{}_data_A'.format(q)]
            B = self.raw_data_dict['{}_data_B'.format(q)]
            ### for dset A ###
            poly_mod_A = lmfit.models.PolynomialModel(degree=2)
            c0_guess = amps[0]
            c1_guess = (A[-1]-A[0])/(amps[-1]-amps[0])
            c2_guess = 0
            poly_mod_A.set_param_hint('c0',value=c0_guess,vary=True)
            poly_mod_A.set_param_hint('c1',value=c1_guess,vary=True)
            poly_mod_A.set_param_hint('c2',value=c2_guess,vary=True)
            params = poly_mod_A.make_params()
            fit_res_line_A = poly_mod_A.fit(data=A,x=amps,params = params)
            self.proc_data_dict['{}_fit_A'.format(q)] = fit_res_line_A

            ### for dset B ###
            poly_mod_B = lmfit.models.PolynomialModel(degree=2)
            c0_guess = amps[0]
            c1_guess = (B[-1]-B[0])/(amps[-1]-amps[0])
            c2_guess = 0
            poly_mod_B.set_param_hint('c0',value=c0_guess,vary=True)
            poly_mod_B.set_param_hint('c1',value=c1_guess,vary=True)
            poly_mod_B.set_param_hint('c2',value=c2_guess,vary=True)
            params = poly_mod_B.make_params()
            fit_res_line_B = poly_mod_B.fit(data=B,x=amps,params = params)
            self.proc_data_dict['{}_fit_B'.format(q)] = fit_res_line_B

            af = fit_res_line_A.best_values
            bf = fit_res_line_B.best_values
            c0 = af['c0']-bf['c0']
            c1 = af['c1']-bf['c1']
            c2 = af['c2']-bf['c2']
            poly_coeff = [c0, c1, c2]
            self.proc_data_dict['quantities_of_interest'][q]['poly_coeff'] = poly_coeff
            
            poly = np.polynomial.polynomial.Polynomial([af['c0'],af['c1'],af['c2']])
            ic = np.polynomial.polynomial.polyroots(poly_coeff)
            intersect_L = ic[0], poly(ic[0])
            intersect_R = ic[1], poly(ic[1])
            if ((min(amps)<ic[0]) and (ic[0] < max(amps))):
                intersect = intersect_L
            else:
                intersect = intersect_R
            self.proc_data_dict['{}_intersect'.format(q)] = intersect
        
    def prepare_plots(self):
        for q in self.qubits:
            self.plot_dicts['Motzoi_'+q] = {
                                            'plotfn': self.plot_line,
                                            'plotsize': (10,7),
                                            'xvals': self.proc_data_dict['amps'],
                                            'xlabel': 'Motzoi amp',
                                            'yvals': self.proc_data_dict['{}_data_A'.format(q)],
                                            'ylabel': 'Raw w0 yX',
                                            'yunit': 'V peak',
                                            'setlabel': 'A',
                                            'title': 'Motzoi_'+q+'_'
                                                +self.raw_data_dict['timestamps'][0],
                                            'do_legend': True,
                                            'legend_pos': 'upper right'}
            self.plot_dicts['B_'+q] = {
                                            'ax_id': 'Motzoi_'+q,
                                            'plotfn': self.plot_line,
                                            'xvals': self.proc_data_dict['amps'],
                                            'xlabel': 'Motzoi amp',
                                            'yvals': self.proc_data_dict['{}_data_B'.format(q)],
                                            'ylabel': 'Raw w0 yX',
                                            'yunit': 'V peak',
                                            'setlabel': 'B',
                                            'do_legend': True,
                                            'legend_pos': 'upper right'}
            self.plot_dicts['Fit_A_'+q] = {
                                            'ax_id': 'Motzoi_'+q,
                                            'plotfn': self.plot_line,
                                            'xvals': self.proc_data_dict['amps'],
                                            'xlabel': 'Motzoi amp',
                                            'yvals': self.proc_data_dict['{}_fit_A'.format(q)].best_fit,
                                            'ylabel': 'Raw w0 yX',
                                            'yunit': 'V peak',
                                            'setlabel': 'Fit A',
                                            'marker': ',',
                                            'do_legend': True,
                                            'legend_pos': 'upper right'}
            self.plot_dicts['Fit_B_'+q] = { 
                                            'ax_id': 'Motzoi_'+q,
                                            'plotfn': self.plot_line,
                                            'xvals': self.proc_data_dict['amps'],
                                            'xlabel': 'Motzoi amp',
                                            'yvals': self.proc_data_dict['{}_fit_B'.format(q)].best_fit,
                                            'ylabel': 'Raw w0 yX',
                                            'yunit': 'V peak',
                                            'setlabel': 'Fit B',
                                            'marker': ',',
                                            'do_legend': True,
                                            'legend_pos': 'upper right'}
            self.plot_dicts['intercept_message'] = {
                                                    'ax_id': 'Motzoi_'+q,
                                                    'plotfn': self.plot_line,
                                                    'xvals': [self.proc_data_dict['{}_intersect'.format(q)][0]],
                                                    'yvals': [self.proc_data_dict['{}_intersect'.format(q)][1]],
                                                    'line_kws': {'alpha': .5, 'color': 'gray',
                                                                 'markersize': 15},
                                                    'marker': 'o',
                                                    'setlabel': 'Intercept: {:.3f}'.format(self.proc_data_dict['{}_intersect'.format(q)][0]),
                                                    'do_legend': True}
