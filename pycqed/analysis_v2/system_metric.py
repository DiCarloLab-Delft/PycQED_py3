import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
# import pycqed.analysis.measurement_analysis as ma
import h5py
from pycqed.analysis import analysis_toolbox as a_tools
# from os.path import join
# import pycqed.analysis_v2.measurement_analysis as ma2
# import lmfit
# from pycqed.analysis import fitting_models as fit_mods
# import logging
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel,
                                            data_to_table_png,
                                            SI_prefix_and_scale_factor,
                                            set_axeslabel_color)
from matplotlib.colors import LogNorm, ListedColormap, LinearSegmentedColormap
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.hdf5_data as h5d
import os


class System_Metric(ba.BaseDataAnalysis):
    """
    System analysis plots data from several qubit objects to visualize total
    system metric.

    """

    def __init__(self, feedline=None, qubit_list: list = None,
                 t_start: str = None, metric: str = None, label: str = '',
                 options_dict: dict = None, parameter_list=None,
                 pairs: list = None, parameter_list_2Q: list = None, auto=True):

        super().__init__(t_start=t_start,
                         label=label,
                         options_dict=options_dict)
        if qubit_list is None and pairs is None:
            if feedline == '1':
                qubit_list = ['D1', 'Z1', 'X', 'D3', 'D4']
                pairs = [('D1', 'Z1'), ('Z1', 'D3'), ('X', 'D3'), ('D1', 'X'),
                         ('X', 'D4')]
            elif feedline == '2':
                qubit_list = ['D2', 'Z2']
                pairs = [('D2', 'Z2')]
                # in case feedline 2
            elif feedline == 'both':
                qubit_list = ['D1', 'D2', 'Z1', 'X', 'Z2', 'D3', 'D4']
                pairs = [('D1', 'Z1'), ('Z1', 'D3'), ('X', 'D3'), ('D1', 'X'),
                         ('X', 'D4'), ('Z2', 'D4'), ('D2', 'Z2'), ('D2', 'X')]
                # Both feedlines
            else:
                raise KeyError
        else:
            raise KeyError

        if t_start is None:
            t_start = a_tools.latest_data(return_timestamp=True)[0]

        self.qubit_list = qubit_list
        self.pairs = pairs
        self.feedline = feedline  # as for GBT we work/feedline
        self.t_start = t_start

        if parameter_list is None:
            # params you want to report. All taken from the qubit object.
            self.parameter_list = ['freq_res', 'freq_qubit',
                                   'anharmonicity', 'fl_dc_I0', 'T1',
                                   'T2_echo', 'T2_star', 'F_RB', 'F_ssro',
                                   'F_discr', 'ro_rel_events', 'ro_res_ext']
        if parameter_list_2Q is None:
            # params you want to report. All taken from the device object.
            self.parameter_list_2Q = ['ro_lo_freq', 'ro_pow_LO']
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = {}
        data_fp = get_datafilepath_from_timestamp(self.t_start)
        for qubit in self.qubit_list:
            self.raw_data_dict[qubit] = {}
            param_dict = {}
            for param in self.parameter_list:
                param_spec = {'{}'.format(param): (
                    'Instrument settings/{}'.format(qubit),
                    'attr:{}'.format(param))}
                param_dict[param] = list(h5d.extract_pars_from_datafile(
                    data_fp, param_spec).values())[0]
            self.raw_data_dict[qubit] = param_dict
            for key in param_dict.keys():
                if param_dict[key] == 'None' or param_dict[key] == '0':
                    param_dict[key] = np.NaN
            param_dict['F_RB'] = 1-float(param_dict['F_RB'])
            param_dict['F_RB'] = str(param_dict['F_RB'])
            param_dict['F_ssro'] = 1-float(param_dict['F_ssro'])
            param_dict['F_ssro'] = str(param_dict['F_ssro'])
            # for fgate in param_dict['F_RB'`]:
            #     param_dict[F_RB]
        # Two qubit gates dic in pairs
        self.raw_data_dict_2Q = {}
        for pair in self.pairs:
            self.raw_data_dict_2Q[pair] = {}
            param_dict_2Q = {}
            for param_2Q in self.parameter_list_2Q:
                param_spec = {'{}'.format(param_2Q): (
                    'Instrument settings/device',
                    'attr:{}'.format(param_2Q))}
                param_dict_2Q[param_2Q] = list(h5d.extract_pars_from_datafile(
                    data_fp, param_spec).values())[0]
            self.raw_data_dict_2Q[pair] = param_dict_2Q
            # create a dic for each Qb
        # convert from dic to pd data frame
        self.raw_data_frame = pd.DataFrame.from_dict(self.raw_data_dict).T
        self.raw_data_frame_2Q = pd.DataFrame.from_dict(
            self.raw_data_dict_2Q).T
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.t_start
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = self.raw_data_dict.copy()
        self.proc_data_dict_2Q = self.raw_data_dict_2Q.copy()
        del self.proc_data_dict['timestamps']
        del self.proc_data_dict['folder']
        # choose the co-ordinations based on the feedline you are having
        if self.feedline == '1':
            coords = [(0, 1), (-1, 0), (1, 0), (0, -1), (2, -1)]
            # feedline 1 coords
        elif self.feedline == '2':
            coords = [(2, 1), (3, 0)]  # feedline 2 coords
        elif self.feedline == 'both':
            coords = [(0, 1), (2, 1), (-1, 0), (1, 0), (3, 0), (0, -1),
                      (2, -1)]  # Both feedlines coords
        else:
            raise KeyError
        for i, qubit in enumerate(self.qubit_list):
            self.proc_data_dict[qubit]['coords'] = coords[i]
        self.proc_data_frame = pd.DataFrame.from_dict(self.proc_data_dict).T
        self.proc_data_frame_2Q = pd.DataFrame.from_dict(
            self.proc_data_dict_2Q).T
        for i, pair in enumerate(self.pairs):
            x = np.mean([self.proc_data_frame.loc[pair[0]]['coords'][0],
                         self.proc_data_frame.loc[pair[1]]['coords'][0]])
            y = np.mean([self.proc_data_frame.loc[pair[0]]['coords'][1],
                         self.proc_data_frame.loc[pair[1]]['coords'][1]])
            coords = (x, y)
            self.proc_data_frame_2Q.loc['-'.join(pair)] = {'coords': coords,
                                                           'cliff_fid': np.NaN}

    def prepare_plots(self):
        self.plot_dicts = {}
        self.plot_dicts['metric'] = {
            'plotfn': self.plot_system_metric,
            'df_1Q': self.proc_data_frame,
            'df_2Q': self.proc_data_frame_2Q
        }

    def plot_system_metric(self, df_1Q, ax=None, df_2Q=None, vmin=None,
                           vmax=None, unit=1, norm=None, plot: str = 'freq_max',
                           main_color='black', figdir=None, axis=False, **kw):
        """
        Plots device performance

        plot (str):
            {"leakage", "gate", "readout"}
        """
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.get_figure()

        val_fmt_str = '{0:1.1e}'
        norm = LogNorm()

        cool_colormap = LinearSegmentedColormap.from_list("", ["green","yellow","red"])
        # Decide metric
        if plot == 'leakage' or plot == 'L1':
            plot_key = 'L1'
            cmap = 'hot'
            clabel = 'Leakage'
        elif plot == 'gate':
            plot_key = 'gate_infid'
            cmap = 'viridis'
            clabel = 'Gate infidelity'
        elif plot == 'readout infidelity':
            cmap = 'ocean'
            clabel = 'Readout infidelity'
            plot_key = 'ro_fid'
        elif plot == 'F_RB':
            cmap = cool_colormap 
            clabel = 'Single Gate infidelity'
            plot_key = 'F_RB'
            # val_fmt_str = '{:.3g}'
        elif plot == 'F_ssro':
            cmap = cool_colormap 
            clabel = 'Assignment readout fidelity'
            plot_key = 'F_ssro'
            norm = None
            # val_fmt_str = '{:.3f}'
        elif plot == 'ro_res_ext':
            cmap = cool_colormap 
            clabel = 'Residual Excitation'
            plot_key = 'ro_res_ext'
            norm = None
            # val_fmt_str = '{:.3f}'
        elif plot == 'F_discr':
            cmap = 'ocean'
            clabel = 'Discriminated readout fidelity'
            plot_key = 'F_ssro'
            val_fmt_str = '{:.3f}'
        elif plot == 'readout_QND':
            cmap = 'cividis'
            clabel = 'Readout QNDness'
            plot_key = 'ro_QND'
        elif plot == 'freq_max':
            cmap = 'PuOr'
            clabel = 'Frequency (GHz)'
            plot_key = 'freq_qubit'
            norm = None
            val_fmt_str = '{:.3f}'
            unit = 1e9
        elif plot == 'freq_target':
            cmap = 'nipy_spectral_r'
            clabel = 'Frequency (GHz)'
            norm = None
            plot_key = 'freq_target_GHz'
            val_fmt_str = '{:.3f}'
            norm = None
        elif plot == 'T1':
            cmap = 'RdYlGn'
            clabel = r"T1 ($\mu$s)"
            norm = None
            plot_key = 'T1'
            val_fmt_str = '{:.3f}'
            norm = None
            unit = 1e-6
        elif plot == 'T2_echo':
            cmap = 'RdYlGn'
            norm = None
            clabel = r"T2 echo ($\mu$s)"
            plot_key = 'T2_echo'
            val_fmt_str = '{:.3f}'
            norm = None
            unit = 1e-6
        elif plot == 'T2_star':
            cmap = 'RdYlGn'
            norm = None
            clabel = r"T2 star ($\mu$s)"
            plot_key = 'T2_star'
            val_fmt_str = '{:.3f}'
            norm = None
            unit = 1e-6
        elif plot == 'anharmonicity':
            cmap = 'RdYlGn'
            clabel = 'Anharmonicity (GHz)'
            plot_key = 'anharmonicity'
            val_fmt_str = '{:.3f}'
            norm = None
            unit = 1e9
        elif plot == 'asymmetry':
            cmap = 'RdYlGn'
            clabel = 'asymmetry'
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
        if vmin is None:
            vmin = min(values/unit)-np.mean(values/unit)/10
        if vmax is None:
            vmax = max(values/unit)+np.mean(values/unit)/10
        sc = ax.scatter(x, y, s=1500, c=values/unit, vmin=vmin,
                        vmax=vmax, cmap=cmap, norm=norm)

        # Plot qubit labels and corresponding metric values as text
        qubit_list = [qubit for qubit, i in df_1Q.iterrows()]
        for i, qubit in enumerate(qubit_list):
            ax.text(x[i], y[i]+.4, s=qubit, va='center',
                    ha='center', color=main_color)
            ax.text(x[i], y[i], s=val_fmt_str.format(values[i]/unit),
                    color='black',
                    va='center', ha='center')
            if plot == 'freq_max':
                T = []
                val_fmt_str = '{:g}'
                T1 = round(float(self.proc_data_frame.at[qubit, 'T1'])*1e6, 1)
                T2s = round(
                    float(self.proc_data_frame.at[qubit, 'T2_star'])*1e6, 1)
                T2e = round(
                    float(self.proc_data_frame.at[qubit, 'T2_echo'])*1e6, 1)
                T.extend([T1, T2e])
                ax.text(x[i], y[i]-0.38, s=T,
                        color='black',
                        va='center', ha='center')
                ax.text(1,1.5, s='[T1 , T2_echo] in $\mu$s',ha='center',
                        va='center', color='black')
        # Main figure
        ax.set_ylim(-1.5, 1.6)
        ax.set_xlim(-1.5, 3.5)
        ax.set_aspect('equal')
        # feedline positions
        ax.text(-1., 1.2, 'Feedline 1', rotation=-45,
                ha='center', va='center', color=main_color)
        ax.text(3., 1.2, 'Feedline 2', rotation=-45,
                ha='center', va='center', color=main_color)
        ax.plot([-1.5, 2], [1.5, -2], c='C0')
        ax.plot([1.5, 3.5], [2.5, .5])
        ax.set_title('System metric GBT', color=main_color)

        # Color bar
        cax = f.add_axes([.95, 0.13, .03, .75])
        plt.colorbar(sc, cax=cax)
        cax.set_ylabel(clabel)
        set_axeslabel_color(cax, main_color)
        if not axis:
            ax.set_axis_off()

        # # # Two qubit part
        # if df_2Q is None:
        #     df_2Q = self.proc_data_frame_2Q
        #     x = np.array([c[0] for c in df_2Q['coords']])
        #     y = np.array([c[1] for c in df_2Q['coords']])

        #     ax.scatter(x, y, s=2000, edgecolors=main_color, color='None',
        #                marker=(4, 0, i*90),)
        #     if plot in {'gate', 'leakage'}:
        #         sc = ax.scatter(x, y, s=2000, c=df_2Q[plot_key], vmin=vmin, vmax=vmax, cmap=cmap,
        #                         marker=(4, 0, i*90),
        #                         norm=norm)
        #         for ind, row in df_2Q.iterrows():
        #             if row[plot_key] > 1e-3 and plot == 'leakage':
        #                 c = 'black'
        #             else:
        #                 c = 'white'
        #             ax.text(row['coords'][0], row['coords'][1], s=val_fmt_str.format(row[plot_key]),
        #                     color=c,
        #                     va='center', ha='center')

        # # plotting saving
        # if figdir is None:
        #     figdir = os.path.dirname(
        #         get_datafilepath_from_timestamp(self.t_start))

        # f.patch.set_alpha(0)
        # f.savefig(os.path.join(figdir, os.path.basename(figdir) + 'System_Metric_SW_freqs{}.png'.format(main_color)),
        #           dpi=1200, bbox_inches='tight')
        # f.savefig(os.path.join(figdir, os.path.basename(figdir) + 'System_Metric_SW_freqs{}.svg'.format(main_color)),
        #           bbox_inches='tight')
        # f.savefig(os.path.join(figdir, os.path.basename(figdir) + 'System_Metric_SW_freqs{}.pdf'.format(
        #     main_color)), bbox_inches='tight')
