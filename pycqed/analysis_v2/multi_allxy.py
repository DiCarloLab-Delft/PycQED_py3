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
        fig, ax = plt.subplots()
    
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