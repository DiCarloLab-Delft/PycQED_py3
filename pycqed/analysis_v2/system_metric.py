import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.hdf5_data as h5d
import os

class System_Metric(ba.BaseDataAnalysis):
    """
    System analysis plots data from several qubit objects to visualize total system metric.

    """ 
    def __init__(self, qubit_objects = None, qubit_list: list = None, t_start: str = None, metric: str = None, label: str = '',
                options_dict: dict = None, parameter_list = None,
                auto=True):
        
        super().__init__(t_start=t_start,
                         label=label,
                         options_dict=options_dict)
        self.qubit_objects = qubit_objects
        self.qubit_list = qubit_list
        if parameter_list is None:
            self.parameter_list = ['F_RB', 'F_discr', 'F_ssro', 'T1', 'T2_echo', 'T2_star', 'anharmonicity', 'asymmetry']
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = {}
        if t_start is not None:
            data_fp = get_datafilepath_from_timestamp(t_start)
            if self.qubit_objects is not None and self.qubit_list is None:
                qubit_list = [qubit.getname for qubit in qubit_objects]
            else:
                qubit_list = self.qubit_list
            for qubit in qubit_list:
                self.raw_data_dict[qubit] = {}
                param_dict = {}
                for param in self.parameter_list:
                    param_spec = {'{}'.format(param): ('Instrument settings/{}'.format(qubit), 'attr:{}'.format(param))}
                    param_dict[param] = list(h5d.extract_pars_from_datafile(data_fp, param_spec).values())[0]
                self.raw_data_dict[qubit] = param_dict 
        self.raw_data_frame = pd.DataFrame.from_dict(self.raw_data_dict).T
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.t_start
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = self.raw_data_dict.copy()
        del self.proc_data_dict['timestamps']
        del self.proc_data_dict['folder']
        coords = [(0,1), (2,1), (-1,0), (1,0), (3,0), (0,-1), (2,-1)]
        for i, qubit in enumerate(self.qubit_list):
            self.proc_data_dict[qubit]['coords'] = coords[i]
        self.proc_data_frame = pd.DataFrame.from_dict(self.proc_data_dict).T

    
    def prepare_plots(self):
        self.plot_dicts = {}
        self.plot_dicts['metric'] = {
                            'plotfn': plot_system_metric,
                            'df_1Q': self.proc_data_frame
        }




def plot_system_metric(f, ax, df_1Q, df_2Q = None, vmin=None, vmax=None, unit=1, norm=None,
                     plot:str='gate', main_color='black', axis=False):
    """
    Plots device performance
    
    plot (str): 
        {"leakage", "gate", "readout"}
    """
    
    val_fmt_str = '{0:1.1e}'
    # Decide metric
    if plot == 'leakage' or plot == 'L1':
        plot_key = 'L1'
        cmap='hot'
        clabel= 'Leakage'
    elif plot=='gate': 
        plot_key = 'gate_infid'
        cmap='viridis'
        clabel= 'Gate infidelity'
    elif plot == 'readout infidelity':
        cmap = 'ocean'
        clabel= 'Readout infidelity'
        plot_key = 'ro_fid'
    elif plot == 'F_ssro':
        cmap = 'ocean'
        clabel= 'Assignment readout fidelity'
        plot_key = 'F_ssro'
        val_fmt_str = '{:.3f}'
    elif plot == 'F_discr':
        cmap = 'ocean'
        clabel= 'Discriminated readout fidelity'
        plot_key = 'F_ssro'
        val_fmt_str = '{:.3f}'
    elif plot == 'readout_QND':
        cmap='cividis'
        clabel= 'Readout QNDness'
        plot_key = 'ro_QND'
    elif plot == 'freq_max':
        cmap='PuOr'
        clabel= 'Frequency (GHz)'
        plot_key = 'freq_max'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1e9
    elif plot == 'freq_target':
        cmap='nipy_spectral_r'
        clabel= 'Frequency (GHz)'
        plot_key = 'freq_target_GHz'
        val_fmt_str = '{:.3f}'
        norm = None
    elif plot == 'T1':
        cmap='RdYlGn'
        clabel= 'T1 ($\mu$s)'
        plot_key = 'T1'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1e-6
    elif plot == 'T2_echo':
        cmap='RdYlGn'
        clabel= r"T2 echo ($\mu$s)"
        plot_key = 'T2_echo'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1e-6
    elif plot == 'T2_star':
        cmap='RdYlGn'
        clabel= r"T2 star ($\mu$s)"
        plot_key = 'T2_star'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1e-6
    elif plot == 'anharmonicity':
        cmap='RdYlGn'
        clabel= 'Anharmonicity (GHz)'
        plot_key = 'anharmonicity'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1e9
    elif plot == 'asymmetry':
        cmap='RdYlGn'
        clabel= 'asymmetry'
        plot_key = 'asymetry'
        val_fmt_str = '{:.3f}'
        norm = None
        unit = 1
    else: 
        raise KeyError
    
    # Plot qubit locations
    x = [c[0] for c in df_1Q['coords']]
    y = [c[1] for c in df_1Q['coords']]
    ax.scatter(x, y, s=1500, edgecolors=main_color, color='None')
    
    # Extract metric values from dictionary
    values = [float(v) for v in df_1Q[plot_key]]
    values = np.array(values)
    
    # Plot qubits colors based on metric value
    if vmin == None:
        vmin = min(values/unit)-np.mean(values/unit)/10
    if vmax == None:
        vmax = max(values/unit)+np.mean(values/unit)/10
#   sc = ax.scatter(x, y, s=1500, c=df[plot_key], vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm())
    sc = ax.scatter(x, y, s=1500, c=values/unit, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)

    #Plot qubit labels and corresponding metric values as text
    qubit_list = [qubit for qubit,i in df_1Q.iterrows()]
    for i, qubit in enumerate(qubit_list):
        ax.text(x[i], y[i]+.4, s=qubit, va='center', ha='center', color=main_color)
        ax.text(x[i], y[i], s=val_fmt_str.format(values[i]/unit), 
                color='black',
                va='center', ha='center')
    # Main figure
    ax.set_ylim(-1.5, 1.6)
    ax.set_xlim(-1.5, 3.5)
    ax.set_aspect('equal')
    
    # Color bar
    cax = f.add_axes([.95, 0.13, .03, .75])
    plt.colorbar(sc, cax=cax)
    cax.set_ylabel(clabel)
    set_axeslabel_color(cax, main_color)
    if not axis:
        ax.set_axis_off()
    
    # Two qubit part 
    if df_2Q is not None:

        x = np.array([c[0] for c in dict_2Q['coords']])
        y = np.array([c[1] for c in dict_2Q['coords']])


        ax.scatter(x, y, s=2000, edgecolors=main_color, color='None', 
                   marker=(4, 0, i*90),)
        if plot in {'gate', 'leakage'}:
            sc = ax.scatter(x, y, s=2000, c=dict_2Q[plot_key], vmin=vmin, vmax=vmax, cmap=cmap, 
                            marker=(4, 0, i*90),
                            norm=norm)
            for ind, row in dict_2Q.iterrows():
                if row[plot_key]>1e-3 and plot=='leakage':
                    c = 'black'
                else: 
                    c='white'
                ax.text(row['coords'][0], row['coords'][1], s=val_fmt_str.format(row[plot_key]), 
                        color=c,
                        va='center', ha='center')