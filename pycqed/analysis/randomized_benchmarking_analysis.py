import os
import h5py
import lmfit
import logging
import numpy as np
import scipy.optimize as optimize

from copy import deepcopy

from matplotlib import pyplot as plt
from pycqed.measurement import hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)

font_size = 18
marker_size = 5
fig_size_dim = 10
line_width = 2
axes_line_width = 1
golden_ratio = (1+np.sqrt(5))/2
params = {'figure.figsize': (fig_size_dim, fig_size_dim/golden_ratio),
          'figure.dpi': 300,
          'savefig.dpi': 300,
          'figure.titlesize': font_size,
          'legend.fontsize': font_size,
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'axes.linewidth': axes_line_width,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'lines.markersize': marker_size,
          'lines.linewidth': line_width,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'axes.formatter.useoffset': False,
          }
plt.rcParams.update(params)

class RandomizedBenchmarking_Analysis(ma.TD_Analysis):

    '''
    Rotates and normalizes the data before doing a fit with a decaying
    exponential to extract the Clifford fidelity.
    By optionally specifying T1 and the pulse separation (time between start
    of pulses) the T1 limited fidelity will be given and plotted in the
    same figure.
    '''

    def __init__(self, T1=None, T2=None, pulse_length=None,
                 gate_decomp='HZ', TwoD=True, **kw):

        self.T1 = T1
        self.T2 = T2
        if self.T1 == 0:
            self.T1 = None
        if self.T2 == 0:
            self.T2 = None

        self.gate_decomp = gate_decomp
        self.pulse_length = pulse_length

        super().__init__(TwoD=TwoD, **kw)

    def run_default_analysis(self, **kw):

        if not kw.pop('skip', False):
            close_main_fig = kw.pop('close_main_fig', True)
            close_file = kw.pop('close_file', True)

            self.data_RB = self.extract_data(**kw)
            #data = self.corr_data[:-1*(len(self.cal_points[0]*2))]
            #n_cl = self.sweep_points[:-1*(len(self.cal_points[0]*2))]
            self.add_analysis_datagroup_to_file()
            self.add_dataset_to_analysisgroup('Corrected data', self.data_RB)
            self.analysis_group.attrs.create(
                'corrected data based on', 'calibration points'.encode('utf-8'))

            self.fit_res = self.fit_data(self.data_RB, self.n_cl, **kw)
            self.fit_results = [self.fit_res]

            self.save_fitted_parameters(fit_res=self.fit_res, var_name='F|1>')
            if self.make_fig:
                self.make_figures(close_main_fig=close_main_fig, **kw)

            if close_file:
                self.data_file.close()
        return

    def extract_data(self, **kw):

        qb_RO_channel = kw.pop('qb_RO_channel', None)

        if self.cal_points is None:
            self.cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if qb_RO_channel is None:
            ma.MeasurementAnalysis.run_default_analysis(
                self, close_file=False, **kw)
        else:
            self.get_naming_and_values_2D()

        if len(self.measured_values) == 1 or qb_RO_channel is not None:

            if qb_RO_channel is not None:
                ch = qb_RO_channel
            else:
                ch = 0

            logging.warning('Data is assumed to be thresholded!')
            self.n_cl = np.unique(self.sweep_points_2D)
            self.nr_seeds = self.sweep_points.size

            data_raw = self.measured_values[ch]

            data = np.zeros((self.n_cl.size))
            for i in range(self.n_cl.size):
                data[i] = np.mean([data_raw[j][i] for j in
                                   range(self.nr_seeds)])

            data = 1 - data
            # data = np.mean(self.measured_values)

        else:
            self.n_cl = np.unique(self.sweep_points_2D)
            nr_sweep_pts = self.sweep_points.size #nr_seeds+NoCalPts
            self.nr_seeds = nr_sweep_pts - 2*len(self.cal_points[0])
            data = np.zeros(self.n_cl.size)

            data_rearranged = [np.zeros((self.n_cl.size, nr_sweep_pts)),
                               np.zeros((self.n_cl.size, nr_sweep_pts))]
            I = self.measured_values[0]
            Q = self.measured_values[1]
            for i in range(self.n_cl.size):
                for j in range(nr_sweep_pts):
                    data_rearranged[0][i, j] = I[j, i]
                    data_rearranged[1][i, j] = Q[j, i]
            self.data_rearranged = data_rearranged
            a = np.zeros((2, nr_sweep_pts))  # this is an array with the
            # same shape as measured_values
            # for TwoD==False
            self.data_calibrated = deepcopy(self.data_rearranged[0])
            for i in range(self.n_cl.size):
                a[0] = data_rearranged[0][i]
                a[1] = data_rearranged[1][i]
                data_calibrated = a_tools.rotate_and_normalize_data(
                    a, self.cal_points[0], self.cal_points[1])[0]
                self.data_calibrated[i] = data_calibrated
                data_calibrated = data_calibrated[:-int(self.NoCalPoints)]
                data[i] = np.mean(data_calibrated)

            self.calibrated_data_points = np.zeros(
                shape=(self.data_calibrated.shape[0], self.nr_seeds))
            for i,d in enumerate(self.data_calibrated):
                self.calibrated_data_points[i] = \
                    d[:-int(2*len(self.cal_points[0]))]

            # we want prob to be in gnd state
            data = 1 - data

        return data

    def add_textbox(self, ax, F_T1=None, plot_T1_lim=True, **kw):

        if not hasattr(self, 'fit_res'):
            fit_res = kw.pop('fit_res', None)
        else:
            fit_res = self.fit_res

        textstr = ('$F_{Cl}$'+' = {:.6g} $\pm$ ({:.4g})%'.format(
            fit_res.params['fidelity_per_Clifford'].value*100,
            fit_res.params['fidelity_per_Clifford'].stderr*100) +
                   '\n$1-F_{Cl}$'+'  = {:.4g} $\pm$ ({:.4g})%'.format(
            (1-fit_res.params['fidelity_per_Clifford'].value)*100,
            (fit_res.params['fidelity_per_Clifford'].stderr)*100) +
                   '\nOffset = {:.4g} $\pm$ ({:.3g})'.format(
                       (fit_res.params['offset'].value),
                       (fit_res.params['offset'].stderr)))
        if F_T1 is not None and plot_T1_lim:
            textstr += ('\n$F_{Cl}^{T_1}$  = ' +
                        '{:.6g}%'.format(F_T1*100))

        horizontal_alignment = kw.pop('horizontal_alignment', 'left')
        horiz_place = 0.025
        if horizontal_alignment == 'right':
            horiz_place = 0.975

        vertical_alignment = kw.pop('horizontal_alignment', 'bottom')
        vert_place = 0.025
        if vertical_alignment == 'top':
            vert_place = 0.975

        ax.text(horiz_place, vert_place, textstr, transform=ax.transAxes,
                fontsize=self.font_size, verticalalignment=vertical_alignment,
                horizontalalignment=horizontal_alignment, bbox=self.box_props)

    def make_figures(self, close_main_fig, **kw):

        xlabel = 'Number of Cliffords, m'
        ylabel = r'Probability, $P$ $\left(|g \rangle \right)$'
        self.fig, self.ax = self.default_ax()

        self.plot_results_vs_sweepparam(x=self.n_cl,
                                        y=self.data_RB,
                                        fig=self.fig, ax=self.ax,
                                        marker='o',
                                        xlabel=xlabel,
                                        ylabel=ylabel,
                                        save=False)

        if kw.pop('plot_errorbars', True):
            if kw.pop('std_err', False):
                err_bars = []
                for data in self.calibrated_data_points:
                    err_bars.append(np.std(data)/np.sqrt(data.size))
                err_bars = np.asarray(err_bars)
                err_label = 'stderr'

            else:
                err_bars = self.epsilon
                err_label = str(int(self.conf_level*100)) + '% CI'

            a_tools.plot_errorbars(self.n_cl,
                                   self.data_RB,
                                   ax=self.ax,
                                   err_bars=err_bars,
                                   label=err_label,
                                   linewidth=self.line_width//2,
                                   marker='none',
                                   markersize=self.marker_size,
                                   capsize=self.line_width,
                                   capthick=self.line_width)

        x_fine = np.linspace(0, self.n_cl[-1], 1000)
        for fit_res in self.fit_results:
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **fit_res.best_values)
            self.ax.plot(x_fine, best_fit, 'C0', label='Fit')
        # self.ax.set_ylim(min(min(self.data_RB)-.1, -.1),
        #                  max(max(self.dadata_RBta)+.1, 1.1))
        # self.ax.set_ylim(0.4, max(max(self.data_RB)+.1, 1.1))

        # Here we add the line corresponding to T1 limited fidelity
        plot_T1_lim = kw.pop('plot_T1_lim', True)
        F_T1 = None

        if self.pulse_length is not None:
            if self.T1 is not None and self.T2 is not None and plot_T1_lim:
                F_T1, p_T1 = calc_T1_limited_fidelity(
                    self.T1, self.T2, self.pulse_length, self.gate_decomp)
                T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                    x_fine, self.fit_res.best_values['Amplitude'], p_T1,
                    self.fit_res.best_values['offset'])
                self.ax.plot(x_fine, T1_limited_curve, '-.', color='C1',
                             linewidth=self.line_width,
                             label='decoherence-limit')

        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       fontsize=self.font_size)

        # Add a textbox
        self.add_textbox(self.ax, F_T1, plot_T1_lim=plot_T1_lim)

        if kw.pop('show_RB', False):
            plt.show()

        if not close_main_fig:
            # Hacked in here, good idea to only show the main fig but can
            # be optimized somehow
            self.save_fig(self.fig, ylabel='Amplitude (normalized)',
                          close_fig=False, **kw)
        else:
            self.save_fig(self.fig, ylabel='Amplitude (normalized)', **kw)

    def fit_data(self, data, numCliff,
                 print_fit_results=False,
                 show_guess=False,
                 plot_results=False, **kw):

        RBModel = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
        # RBModel = fit_mods.RBModel
        RBModel.set_param_hint('Amplitude', value=0)#-0.5)
        RBModel.set_param_hint('p', value=.99)
        RBModel.set_param_hint('offset', value=.5)
        # From Magesan et al., Scalable and Robust Randomized Benchmarking
        # of Quantum Processes
        RBModel.set_param_hint('fidelity_per_Clifford',  # vary=False,
                               expr='(p + (1-p)/2)')
        RBModel.set_param_hint('error_per_Clifford',  # vary=False,
                               expr='1-fidelity_per_Clifford')
        if self.gate_decomp == 'XY':
            RBModel.set_param_hint('fidelity_per_gate',  # vary=False,
                                   expr='fidelity_per_Clifford**(1./1.875)')
        elif self.gate_decomp == 'HZ':
            RBModel.set_param_hint('fidelity_per_gate',  # vary=False,
                                   expr='fidelity_per_Clifford**(1./1.125)')
        else:
            raise ValueError('Gate decomposition not recognized.')
        RBModel.set_param_hint('error_per_gate',  # vary=False,
                               expr='1-fidelity_per_gate')

        params = RBModel.make_params()

        # Run once to get an estimate for the error per Clifford
        fit_res = RBModel.fit(data, numCliff=numCliff, params=params)

        # Use the found error per Clifford to standard errors for the data
        # points fro Helsen et al. (2017)
        self.conf_level = kw.pop('conf_level', 0.68)
        epsilon_guess = kw.pop('epsilon_guess', 0.1)
        eta = kw.pop('eta', 0)
        epsilon = calculate_confidence_intervals(
            nr_seeds=self.nr_seeds,
            nr_cliffords=self.n_cl,
            infidelity=fit_res.params['error_per_Clifford'].value,
            conf_level=self.conf_level,
            epsilon_guess=epsilon_guess,
            eta=eta)

        self.epsilon = epsilon
        # Run fit again with scale_covar=False, and weights = 1/epsilon

        # if an entry in epsilon_sqrd is 0, replace it with half the minimum
        # value in the epsilon_sqrd array
        idxs = np.where(epsilon==0)[0]
        epsilon[idxs] = min([eps for eps in epsilon if eps!=0])/2

        fit_res = RBModel.fit(data, numCliff=numCliff, params=params,
                              scale_covar=False, weights=1/epsilon)

        if print_fit_results:
            print(fit_res.fit_report())
        if plot_results:
            plt.plot(fit_res.data, 'o-', label='data')
            plt.plot(fit_res.best_fit, label='best fit')
            if show_guess:
                plt.plot(fit_res.init_fit, '--', label='init fit')

        return fit_res


class Interleaved_RB_Analysis(RandomizedBenchmarking_Analysis):

    def __init__(self, folders_dict=None, qb_name=None, **kw):
        """
        folders_dict (dict): of the form {msmt_name: folder}
        qb_name (str): name of qb on which experiment has been done
        """
        self.folders_dict = folders_dict
        if folders_dict is None:
            labels = kw.pop('labels', None)
            if labels is not None:
                self.folders_dict = {}
                for label in labels:
                    self.folders_dict[label] = \
                        a_tools.latest_data(contains=label)
            else:
                raise ValueError('folders_dict is unspecified. Please specify'
                                 'at least the msmt_name keys in the '
                                 'folders_dic, or a list of labels.')

        for msmt_name, folder in self.folders_dict.items():
            if folder is None or folder == '':
                # if only msmt_name keys were given, take latest
                # data files with that label
                self.folders_dict[msmt_name] = \
                    a_tools.latest_data(contains=msmt_name)
            else:
                try:
                    # maybe the folder is a timestamp
                    folder = a_tools.get_folder(timestamp=folder)
                    self.folders_dict[msmt_name] = folder
                except Exception:
                    pass

        if qb_name is None:
            try:
                folder = self.folders_dict[list(self.folders_dict)[0]]
                q_idx = folder[-10::].index('q')
                qb_name = folder[-10::][q_idx::]
            except ValueError:
                pass

        folder = list(self.folders_dict.values())[0]
        super().__init__(qb_name=qb_name, folder=folder, **kw)

    def run_default_analysis(self, **kw):

        make_fig_IRB = kw.pop('make_fig_IRB', True)
        close_file = kw.pop('close_file', True)

        self.T1s = {}
        self.T2s = {}
        self.pulse_lengths = {}

        self.T1s, self.T2s, self.pulse_lengths = \
            load_T1_T2_pulse_length(self.folders_dict, self.qb_name,
                                    T1s=self.T1s, T2s=self.T2s,
                                    pulse_lengths=self.pulse_lengths)

        self.fit_res_dict = {}
        self.data_dict = {}
        self.data_files_dict = {}
        self.ncl_dict = {}
        self.epsilon_dict = {}
        for msmt_name in self.folders_dict:
            self.folder = self.folders_dict[msmt_name]
            self.load_hdf5data(folder=self.folder, **kw)
            self.T1 = self.T1s[msmt_name][self.qb_name]
            self.T2 = self.T2s[msmt_name][self.qb_name]
            self.pulse_length = self.pulse_lengths[msmt_name][self.qb_name]

            self.data_dict[msmt_name] = self.extract_data(**kw)
            self.fit_res_dict[msmt_name] = \
                self.fit_data(self.data_dict[msmt_name], self.n_cl, **kw)
            self.data_files_dict[msmt_name] = self.data_file
            self.ncl_dict[msmt_name] = self.n_cl
            self.epsilon_dict[msmt_name] = self.epsilon
            #
        # Add the corrected data in self.data_dict to the 'IRB Analysis' group
        # in the HDF5 file of the last measurement
        # (i.e., in list(self.folders_dict)[-1])
        try:
            data_dict_to_save = {'nr_cliffords': self.n_cl}
            data_dict_to_save.update(self.data_dict)
            names_units_dict = {'sweep_par_names': 'nr_cliffords',
                                'value_names': ['Prob(g)_'+i for i in
                                                list(self.folders_dict)]}

            self.IRB_analysis_group = \
                create_experimentaldata_dataset(data_object=self.data_file,
                                                data_dict=data_dict_to_save,
                                                group_name='IRB Analysis',
                                                shape=(len(self.n_cl),
                                                       1+
                                                       len(self.folders_dict)),
                                                column_names=
                                                    list(data_dict_to_save),
                                                names_units_dict=
                                                    names_units_dict)
            self.IRB_analysis_group.attrs.create(
                'nr_seeds', str(self.nr_seeds).encode('utf-8'))
            self.IRB_analysis_group.attrs.create(
                'gate_decomposition', self.gate_decomp.encode('utf-8'))
            self.IRB_analysis_group.attrs.create(
                'max_nr_Cliffords', str(max(self.n_cl)).encode('utf-8'))
            self.IRB_analysis_group.attrs.create(
                'qubit_names', self.qb_name.encode('utf-8'))
            self.IRB_analysis_group.attrs.create(
                'interleaved_gates',  h5d.encode_to_utf8(
                    [i for i in list(self.folders_dict)]))

            # Save fitted parameters in the self.IRB_analysis_group
            for msmt_name, fit_res in self.fit_res_dict.items():
                save_fitted_parameters(fit_res,
                                       fit_group=self.IRB_analysis_group,
                                       fit_name='Fit Results ' +
                                                msmt_name + ' RB')
        except Exception:
            print('Could not save fit parameters to HDF file.')
            pass

        # Estimate gate errors
        self.gate_errors = {}
        self.interleaved_gates = list(self.folders_dict)[1::]
        regular_RB_key = list(self.folders_dict)[0]
        for interleaved_gate in self.interleaved_gates:
            self.gate_errors[interleaved_gate] = {}

            self.gate_errors[interleaved_gate]['val'], \
            self.gate_errors[interleaved_gate]['stderr'] = \
                estimate_gate_error(
                p0=self.fit_res_dict[regular_RB_key].best_values['p'],
                p_gate=self.fit_res_dict[interleaved_gate].best_values['p'],
                p0_stderr=self.fit_res_dict[regular_RB_key].params['p'].stderr,
                p_gate_stderr=self.fit_res_dict[
                    interleaved_gate].params['p'].stderr,)

        try:
            # Save the gate errors in the self.IRB_analysis_group
            save_computed_parameters(self.gate_errors, name='Gate Errors',
                                     group=self.IRB_analysis_group)
        except Exception:
            print('Could not save gate errors to HDF file.')
            pass

        if make_fig_IRB:
            self.plot_IRB(**kw)

        # if close_file:
        #     self.finish(**kw)

    def plot_IRB(self,  plot_errorbars=True,
                 plot_T1_lim=True, **kw):

        if kw.pop('fig', None) is None:
            self.fig, self.ax = plt.subplots()
        else:
            if kw.pop('ax', None) is None:
                self.ax = self.fig.add_subplot(111)

        xlabel = 'Number of Cliffords, m'
        ylabel = r'Probability, $P$ $\left(|g \rangle \right)$'
        self.title = 'IRB_HZ'
        for msmt_name in self.data_dict:
            self.title += ('_' + msmt_name)

        FT1 = []
        pT1 = []
        textstr = ''
        # c = ['b', 'C2', 'C4']
        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c = [i for i in default_color_cycle if i not in ['#ff7f0e', '#2ca02c',
                                                         '#d62728']]

        for gate_name, gate_error_dict in self.gate_errors.items():
            textstr += ('$r_{C,' + gate_name + '}$'+' = '
                '{:.4g} $\pm$ ({:.4g})% \n'. format(
                gate_error_dict['val']*100, gate_error_dict['stderr']*100))

        for i, msmt_name in enumerate(self.data_dict):

            # Plot fit lines
            x_fine = np.linspace(0, self.ncl_dict[msmt_name][-1], 1000)
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **self.fit_res_dict[msmt_name].best_values)
            self.ax.plot(x_fine, best_fit, c=c[i])

            # Plot data points
            plotting_function(x=self.ncl_dict[msmt_name],
                             y=self.data_dict[msmt_name],
                             fig=self.fig, ax=self.ax,
                             label=msmt_name,
                             marker='o',
                             line_color=c[i],
                             plot_title=self.title,
                             save=False)

            # Plot error bars
            if plot_errorbars:
                err_bars = self.epsilon_dict[msmt_name]
                err_label = str(int(self.conf_level*100)) + '% CI'
                a_tools.plot_errorbars(self.ncl_dict[msmt_name],
                                       self.data_dict[msmt_name],
                                       ax=self.ax,
                                       err_bars=err_bars,
                                       label=err_label,
                                       color=c[i],
                                       linewidth=line_width//2,
                                       marker='none',
                                       capsize=line_width,
                                       capthick=line_width)

            self.ax.set_ylim(0.4,
                             max(max(self.data_dict[msmt_name])+.1, 1.1))

            textstr += \
                ('$r_{perCl,' + msmt_name + '}$'+' = '
                    '{:.6g} $\pm$ ({:.4g})% \n'.format(
                    (1-self.fit_res_dict[msmt_name].params[
                    'fidelity_per_Clifford'].value)*100,
                    (1-self.fit_res_dict[msmt_name].params[
                        'fidelity_per_Clifford'].stderr)*100))

            # Here we get the line corresponding to T1 limited fidelity
            if self.pulse_lengths[msmt_name] is not None:
                if self.T1s[msmt_name][self.qb_name] is not None \
                        and plot_T1_lim:
                    F_T1, p_T1 = calc_T1_limited_fidelity(
                        self.T1s[msmt_name][self.qb_name],
                        self.T2s[msmt_name][self.qb_name],
                        self.pulse_lengths[msmt_name][self.qb_name],
                        self.gate_decomp)
                    FT1.append(F_T1)
                    pT1.append(p_T1)

        # plot T1-limited curve
        if plot_T1_lim:
            F_T1 = np.mean(FT1)
            p_T1 = np.mean(pT1)
            amp_mean = np.mean(
                [self.fit_res_dict[msmt_name].best_values['Amplitude'] for
                 msmt_name in self.fit_res_dict])
            offset_mean = np.mean(
                [self.fit_res_dict[msmt_name].best_values['offset'] for
                 msmt_name in self.fit_res_dict])
            T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, amp_mean, p_T1, offset_mean)
            self.ax.plot(x_fine, T1_limited_curve, '-.',
                         color='C1',
                         label='decoh-limit (avg)')
            textstr += ('$r_{Cl,avg}^{T_1}$  = ' +
                        '{:.6g}%\t'.format((1-F_T1)*100))

        # Set legend
        handles, labels = self.ax.get_legend_handles_labels()
        if plot_errorbars:
            T1_lim_handle = handles.pop(labels.index('decoh-limit (avg)'))
            T1_lim_label = labels.pop(labels.index('decoh-limit (avg)'))
            handles_new = []
            labels_new = []
            n = len(self.folders_dict)
            for i in range(n):
                handles_new.extend([handles[i], handles[i+n]])
                labels_new.extend([labels[i], labels[i+n]])
            handles_new.extend([T1_lim_handle])
            labels_new.extend([T1_lim_label])
        else:
            handles_new = handles
            labels_new = labels

        self.ax.legend(handles_new, labels_new,
                       loc='center left', fontsize=font_size,
                       bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)

        # Set textbox
        # self.ax.text(0.025, 0.95, textstr, transform=self.ax.transAxes,
        #         verticalalignment='top', fontsize=font_size)
        self.ax.text(0.5, 0.95, textstr, transform=self.ax.transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     fontsize=font_size)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # show figure
        if kw.pop('show_IRB', False):
            plt.show()

        # Save figure
        if kw.pop('save_fig', True):
            save_folder = self.folders_dict[list(self.folders_dict)[-1]]
            filename = os.path.abspath(os.path.join(save_folder,
                                                    self.title+'.png'))
            self.fig.savefig(filename, format='png',
                             bbox_inches='tight')

        if kw.pop('close_fig', True):
            plt.close(self.fig)


class Simultaneous_RB_Analysis(RandomizedBenchmarking_Analysis):

    def __init__(self, measurement_dict=None, qb_names=None, **kw):
        """
        folders_dict (dict): of the form {msmt_name: folder} if experiment was
            done for 2 qubits (used correlation mode o UHFQC), or of the form
            {msmt_name: (ts_start, ts_end)} if nr qubits > 2 (each file
            if for one value of the nr_cliffords).
        qb_names (tuple or list): of the names of the qubits used
        qb_clifford_name (str): name of the qb on which the Cliffords are
            applied in the CxI_IxC measurements

        For 2 qubits and no CZ interleaved, this analysis expects 3 data files
        corresponding to msmt_name = ['CxC', 'IxC', 'CxI'].
        For 2 qubits and CZ interleaved, msmt_name = ['CZ', 'ICZ']
        """

        self.qb_names = qb_names
        if self.qb_names is None:
            raise ValueError('qb_names is not specified.')
        if type(self.qb_names) != list:
            self.qb_names = list(self.qb_names)

        self.measurement_dict = measurement_dict
        if self.measurement_dict is None:
            raise ValueError('Specify the measurement_dict.')

        try:
            self.folders_dict = \
                {msmt_name: measurement_dict[msmt_name]['file']
                 for msmt_name in measurement_dict}
        except Exception:
            self.folders_dict = None


        if self.folders_dict is None:
            labels = kw.pop('labels', None)
            if len(self.qb_names) == 2 and (labels is not None):
                self.folders_dict = {}
                for label in labels:
                    self.folders_dict[label] = \
                        a_tools.latest_data(contains=label)
            else:
                raise ValueError('folders_dict is unspecified. '
                                 'Please specify at least the msmt_name '
                                 'keys in the folders_dic, or a list '
                                 'of labels.')

        for msmt_name, folder in self.folders_dict.items():
            if len(self.qb_names) == 2:
                if folder is None or folder == '':
                    # if only msmt_name keys were given, take latest
                    # data files with that label
                    self.folders_dict[msmt_name] = \
                        a_tools.latest_data(contains=msmt_name)
                else:
                    try:
                        # maybe the folder is a timestamp
                        folder = a_tools.get_folder(timestamp=folder)
                        self.folders_dict[msmt_name] = folder
                    except Exception:
                        pass
            else:
                # We have multiple data files for the same msmt_name
                if type(folder) == str:
                    try:
                        # folder is a timestamp
                        self.folders_dict[msmt_name] = \
                            [a_tools.get_folder(timestamp=folder)]
                    except Exception:
                        # folder is a path
                        pass
                else:
                    if len(folder) == 2 and len(folder[0]) <= 15:
                        try:
                            # (ts_start, ts_stop) is given
                             ts_list = a_tools.get_timestamps_in_range(
                                 timestamp_start=folder[0],
                                 timestamp_end=folder[1],
                                 label=msmt_name)

                             self.folders_dict[msmt_name] = \
                                [a_tools.get_folder(timestamp=ts)
                                 for ts in ts_list]
                        except Exception:
                            raise ValueError('No files were found for the label '
                                            '"{}". Make sure the keys in '
                                            'folders_dict match the measurement'
                                            ' label.'.format(msmt_name))
                    else:
                        try:
                            # a list of timestamps is given
                            self.folders_dict[msmt_name] = \
                                [a_tools.get_folder(timestamp=ts)
                                 for ts in folder]
                        except Exception:
                            try:
                                # a list of timestamps is given
                                self.folders_dict[msmt_name] = \
                                    [a_tools.get_folder(timestamp=ts)
                                     for ts in folder]
                            except Exception:
                                # a list of paths is given
                                pass

            # print(msmt_name)
            # print(self.folders_dict[msmt_name])

        self.clifford_qbs = {}
        if not np.any(['CZ' in msmt_name for msmt_name in self.folders_dict]):

            try:
                for msmt_name in self.measurement_dict:
                    if 'clifford_qb_idx' in self.measurement_dict[msmt_name]:
                        self.clifford_qbs[msmt_name] = \
                            self.qb_names[
                                self.measurement_dict[msmt_name][
                                    'clifford_qb_idx']]
                if len(self.clifford_qbs) == 0:
                    raise ValueError('Specify "clifford_qb" entires in the'
                                     ' measurement_dict.')
            except ValueError:
                pass


        folder = list(self.folders_dict.values())[0]
        # print(folder)
        if not isinstance(folder, str):
            folder = folder[0]
        # print(folder)
        super().__init__(TwoD=True, folder=folder, **kw)


    def run_default_analysis(self, **kw):

        close_file = kw.pop('close_file', True)
        make_fig_SRB_base = kw.pop('make_fig_SRB_base', True)
        make_fig_cross_talk = kw.pop('make_fig_cross_talk', True)
        make_fig_addressab = kw.pop('make_fig_addressab', True)

        self.T1s, self.T2s, self.pulse_lengths = \
            load_T1_T2_pulse_length(self.folders_dict, self.qb_names)

        if len(self.qb_names) == 2:
            self.correlator_analysis(**kw)
            self.ylabel_RB = r'Probability, $P$ $\left(|g \rangle \right)$'
            self.ylabel_corr = (r'Probability, '
                                r'$P$ $\left(|{n}\rangle \right)$ + '.
                                format(n='g'*len(self.qb_names)) +
                                r'$P$ $\left(|{n}\rangle '
                                r'\right)$'.format(n='e'*len(self.qb_names)))
        else:
            self.single_shot_analysis(**kw)
            self.ylabel_RB = r'Expectation value $\langle \sigma_z \rangle$'
            self.ylabel_corr = (r'Expectation value $\langle '
                                r'\sigma_z^{{\otimes {{{n}}} }} '
                                r'\rangle$'.format(n=len(self.qb_names)))

        # Save fitted params
        for msmt_name, data_file in self.data_files_dict.items():
            # Save fitted parameters in the Analysis group
            for var_name, fit_res in self.fit_res_dict[msmt_name].items():
                if fit_res is not None:
                    save_fitted_parameters(
                        fit_res,
                        fit_group=data_file['Analysis'],
                        fit_name='Fitted Params ' + var_name)

        # Get depolarization parameters and infidelities
        self.depolariz_params = {}
        self.infidelities = {}
        for msmt_name, fit_params_dict in self.fit_res_dict.items():
            self.depolariz_params[msmt_name] = {}
            self.infidelities[msmt_name] = {}
            for var_name, fit_res in fit_params_dict.items():
                try:
                    self.depolariz_params[msmt_name][var_name] = \
                        {'val': fit_res.best_values['p'],
                         'stderr': fit_res.params['p'].stderr}
                    self.infidelities[msmt_name][var_name] = \
                        {'val': 1-fit_res.params[
                            'fidelity_per_Clifford'].value,
                         'stderr': fit_res.params[
                             'fidelity_per_Clifford'].stderr}
                except Exception:
                    pass

        if 'CxC' in self.data_dict:
            fit_res_dict_CxC = self.fit_res_dict['CxC']
            # get delta_alpha (delta depolarization params)
            # delta_alpha_stderr^2 = stderr_corr^2 - (stderr_qb0^2 * p_qb0^2 +
            #   stderr_qb1^2 * p_qb1^2 + ... + stderr_qbN^2 * p_qbN^2)
            product = 1
            std_err_product_squared = 1
            for qb_name in self.qb_names:
                product *= fit_res_dict_CxC[qb_name].best_values['p']

                std_err_product_squared *= \
                    (fit_res_dict_CxC[qb_name].params['p'].stderr)**2
                for other_qb_names in self.qb_names:
                    if other_qb_names != qb_name:
                        std_err_product_squared *= \
                            (fit_res_dict_CxC[
                                 other_qb_names].best_values['p'])**2

            delta_alpha = \
                np.abs(fit_res_dict_CxC['corr'].best_values['p'] - product)
            delta_alpha_stderr = np.sqrt(
                (fit_res_dict_CxC['corr'].params['p'].stderr)**2 -
                std_err_product_squared)

            self.delta_alpha = {'val': delta_alpha,
                                'stderr': delta_alpha_stderr}

        save_fig = kw.pop('save_fig', True)
        plot_errorbars = kw.pop('plot_errorbars', True)
        fig_title_suffix = kw.pop('fig_title_suffix', '')
        self.save_folder = kw.pop('save_folder', None)
        if self.save_folder is None:
            msmt_name_CxC = [msmt_name for msmt_name in
                             self.folders_dict if msmt_name not in
                             self.clifford_qbs][0]
            self.save_folder = self.folders_dict[msmt_name_CxC]
            if not isinstance(self.save_folder, str):
                self.save_folder = self.save_folder[0]

        self.xlabel_RB = 'Number of Cliffords, m'

        if make_fig_SRB_base:
            self.plot_SRB(
                plot_errorbars=plot_errorbars,
                save_fig=save_fig,  **kw)

        if np.any(['CZ' in msmt_name for msmt_name in self.folders_dict]):
            # Estimate gate error
            self.CZ_gate_error = {}
            # self.interleaved_gates = list(self.folders_dict)[1::]
            # regular_RB_key = list(self.folders_dict)[0]
            # for interleaved_gate in self.interleaved_gates:
            # self.gate_errors[interleaved_gate] = {}
            ICZ_msmt_name = [msmt_name for msmt_name in self.folders_dict if
                            'ICZ' in msmt_name][0]
            CZ_msmt_name = [msmt_name for msmt_name in self.folders_dict if
                            msmt_name!=ICZ_msmt_name][0]

            try:
                self.CZ_gate_error['val'], \
                self.CZ_gate_error['stderr'] = \
                    estimate_gate_error(
                        p0=self.fit_res_dict[
                            ICZ_msmt_name]['corr'].best_values['p'],
                        p_gate=self.fit_res_dict[
                            CZ_msmt_name]['corr'].best_values['p'],
                        p0_stderr=self.fit_res_dict[
                            ICZ_msmt_name]['corr'].params['p'].stderr,
                        p_gate_stderr=self.fit_res_dict[
                            CZ_msmt_name]['corr'].params['p'].stderr)

                # Save the gate errors in the self.IRB_analysis_group
                group = self.data_files_dict[CZ_msmt_name]['Analysis']
                save_computed_parameters(self.CZ_gate_error,
                                         name='CZ Gate Error',
                                         group=group)
            except Exception:
                pass

            self.plot_CZ_ISRB_two_qubits(
                plot_errorbars=plot_errorbars,
                save_fig=save_fig,
                fig_title_suffix=fig_title_suffix, **kw)
        else:
            if make_fig_cross_talk:
                self.plot_cross_talk(
                    plot_errorbars=plot_errorbars,
                    save_fig=save_fig,
                    fig_title_suffix=fig_title_suffix, **kw)

            if make_fig_addressab:
                self.plot_addressability(
                    save_fig=save_fig,
                    fig_title_suffix=fig_title_suffix, **kw)

        # if close_file:
        #     self.finish(**kw)

    def correlator_analysis(self, **kw):

        msmt_to_not_fit = kw.get('msmt_to_not_fit', [])

        self.fit_res_dict = {}
        self.data_dict = {}
        self.msmt_strings = {}
        self.data_files_dict = {}
        self.epsilon_dict = {}
        for msmt_name in self.folders_dict:
            self.folder = self.folders_dict[msmt_name]
            self.load_hdf5data(folder=self.folder, **kw)

            self.data_dict[msmt_name] = self.extract_data(two_qubits=True, **kw)
            self.add_analysis_datagroup_to_file()

            self.msmt_strings[msmt_name] = self.measurementstring
            self.data_files_dict[msmt_name] = self.data_file

            self.fit_res_dict[msmt_name] = {}
            self.epsilon_dict[msmt_name] = {}
            self.n_cl = self.data_dict[msmt_name]['n_cl']
            self.nr_seeds = self.data_dict[msmt_name]['nr_seeds']

            # fit data
            for var_name, dset in self.data_dict[msmt_name]['data'].items():
                if msmt_name in msmt_to_not_fit:
                    self.fit_res_dict[msmt_name][var_name] = None
                else:
                    if msmt_name in self.clifford_qbs:
                        if var_name == self.clifford_qbs[msmt_name] or \
                                var_name == 'corr':
                            self.fit_res_dict[msmt_name][var_name] = \
                                self.fit_data(dset, self.n_cl, **kw)
                            self.epsilon_dict[msmt_name][var_name] = \
                                self.epsilon
                        else:
                            self.fit_res_dict[msmt_name][var_name] = None
                            self.epsilon_dict[msmt_name][var_name] = None
                    else:
                        self.fit_res_dict[msmt_name][var_name] = \
                            self.fit_data(dset, self.n_cl, **kw)
                        self.epsilon_dict[msmt_name][var_name] = self.epsilon

    def single_shot_analysis(self, **kw):

        msmt_to_not_fit = kw.get('msmt_to_not_fit', [])

        self.msmt_strings = {}
        self.data_files_dict = {}
        self.fit_res_dict = {}
        self.epsilon_dict = {}

        self.data_dict = {}
        for msmt_name in self.folders_dict:
            self.data_dict[msmt_name] = {}
            self.data_dict[msmt_name]['n_cl'] = np.array([], dtype=int)
            self.data_dict[msmt_name]['data'] = {}
            for var_name in self.qb_names+['corr']:
                self.data_dict[msmt_name]['data'][var_name] = np.array([])

            for i, folder in enumerate(self.folders_dict[msmt_name]):
                self.folder = folder
                self.load_hdf5data(folder=self.folder, **kw)
                # # get folder from timestamp
                # self.folder = a_tools.get_folder(timestamp=ts)
                self.extract_data(two_qubits=False, msmt_name=msmt_name, **kw)

                if i == 0:
                    self.data_dict[msmt_name]['nr_seeds'] = self.nr_seeds
                # get the nr_cliffords for the sequence from the
                # measurementstring
                try:
                    idx_cliffs = self.measurementstring.index('cliffords')
                except TypeError:
                    try:
                        idx_cliffs = self.measurementstring.index('Cliffords')
                    except TypeError:
                        raise ValueError('Could not find the nr_cliffords in '
                                         'the measurement string.')
                nr_cliffs_string = self.measurementstring[
                                   idx_cliffs-7:idx_cliffs-1]
                try:
                    idx_underscore = nr_cliffs_string.index('_')
                except TypeError:
                    raise ValueError('Could not find the underscore in '
                                     'the nr_cliffs substring.')

                self.data_dict[msmt_name]['n_cl'] = np.append(
                    self.data_dict[msmt_name]['n_cl'],
                    int(float(nr_cliffs_string[idx_underscore+1::])))
                self.add_analysis_datagroup_to_file()

            self.msmt_strings[msmt_name] = self.measurementstring
            self.data_files_dict[msmt_name] = self.data_file

            # fit data
            self.fit_res_dict[msmt_name] = {}
            self.epsilon_dict[msmt_name] = {}
            self.n_cl = self.data_dict[msmt_name]['n_cl']
            for var_name, dset in self.data_dict[msmt_name]['data'].items():
                if msmt_name in msmt_to_not_fit:
                    self.fit_res_dict[msmt_name][var_name] = None
                else:
                    print(self.clifford_qbs)
                    if msmt_name in self.clifford_qbs:
                        if var_name == self.clifford_qbs[msmt_name] or \
                                        var_name == 'corr':
                            print('notNone ', msmt_name, var_name)
                            self.fit_res_dict[msmt_name][var_name] = \
                                self.fit_data(dset, self.n_cl, **kw)
                            self.epsilon_dict[msmt_name][var_name] = \
                                self.epsilon
                        else:
                            print('None ', msmt_name, var_name)
                            self.fit_res_dict[msmt_name][var_name] = None
                            self.epsilon_dict[msmt_name][var_name] = None
                    else:
                        print('notNone ', msmt_name, var_name)
                        self.fit_res_dict[msmt_name][var_name] = \
                            self.fit_data(dset, self.n_cl, **kw)
                        self.epsilon_dict[msmt_name][var_name] = \
                            self.epsilon


    def extract_data(self, **kw):

        two_qubits = kw.pop('two_qubits', True)
        if self.cal_points is None:
            self.cal_points = [[-2], [-1]]

        # ma.MeasurementAnalysis.run_default_analysis(
        #     self, close_file=False, **kw)
        self.get_naming_and_values_2D()

        if two_qubits:
            n_cl = np.unique(self.sweep_points_2D)
            nr_seeds = self.sweep_points.size
            data = {'n_cl': n_cl,
                    'nr_seeds': nr_seeds,
                    'data': {}}

            qb0_raw = self.measured_values[0]
            qb1_raw = self.measured_values[1]
            corr_raw = self.measured_values[2]

            qb0 = np.zeros(n_cl.size)
            qb1 = np.zeros(n_cl.size)
            corr = np.zeros(n_cl.size)
            for i in range(n_cl.size):
                qb0[i] = np.mean([qb0_raw[j][i] for j in range(nr_seeds)])
                qb1[i] = np.mean([qb1_raw[j][i] for j in range(nr_seeds)])
                corr[i] = np.mean([corr_raw[j][i] for j in range(nr_seeds)])

            data['data'][self.qb_names[0]] = 1 - qb0
            data['data'][self.qb_names[1]] = 1 - qb1
            data['data']['corr'] = corr

            return data
        else:
            msmt_name = kw.pop('msmt_name', None)
            if msmt_name is None:
                raise ValueError('extract_data needs msmt_name when '
                                 'nr_qubits>2.')

            self.nr_seeds = self.sweep_points.size
            self.nr_shots = self.sweep_points_2D.size

            mean_data_array = np.zeros(len(self.qb_names)+1)

            measurement_data = deepcopy(self.data[2::])
            # rescale data to be between [-1, 1]
            for dim in range(len(self.qb_names)):
                measurement_data[dim] = -2*measurement_data[dim] + 1

            raw_correl_data = deepcopy(measurement_data[0])
            for i in np.arange(1, len(self.qb_names)):
                raw_correl_data = \
                    np.multiply(raw_correl_data,
                                measurement_data[i])
            mean_data_array[-1] = np.mean(raw_correl_data)

            # get averaged results for each qubit measurement
            for i in np.arange(len(self.qb_names)):
                mean_data_array[i] = np.mean(measurement_data[i])

            for var_name, data_point in zip(self.qb_names+['corr'],
                                            mean_data_array):

                self.data_dict[msmt_name]['data'][var_name] = np.append(
                    self.data_dict[msmt_name]['data'][var_name], data_point)
                    # 0.5*data_point+0.5) #put data between [0.5, 1]

            return

    def plot_SRB(self, save_fig=True, **kw):

        show_SRB_base = kw.pop('show_SRB_base', False)
        plot_T1_lim_base = kw.pop('plot_T1_lim_base', True)
        close_fig = kw.pop('close_fig', True)

        # from pprint import pprint
        # pprint(self.fit_res_dict)

        for msmt_idx, msmt_name in enumerate(self.data_dict):

            # nrows = 1 + len(self.qb_names)
            # fig, axs = plt.subplots(nrows=nrows,
            #                         sharex=True,
            #                         figsize=(fig_size_dim,
            #                                  nrows*fig_size_dim/golden_ratio))

            for var_idx, var_name in enumerate(
                    self.data_dict[msmt_name]['data']):

                fig, ax = plt.subplots(figsize=(
                    fig_size_dim, fig_size_dim/golden_ratio))

                fig_title = ('{}_{}_SRB_{}-{}seeds'.format(
                    var_name,
                    msmt_name,
                    self.gate_decomp,
                    int(self.data_dict[msmt_name]['nr_seeds'])))

                if var_name == 'corr':
                    if len(self.qb_names) == 2:
                        subplot_title = \
                            r'$\langle \sigma_{{z,{qb0}}} ' \
                            r'\sigma_{{z,{qb1}}} \rangle$'.format(
                                qb0=self.qb_names[0],
                                qb1=self.qb_names[1])
                    else:
                        subplot_title = r'$\langle ' \
                                        r'\sigma_z^{{\otimes {{{n}}} }}$'\
                            .format(n=len(self.qb_names)) + r'$\rangle$'

                    ylabel = self.ylabel_corr
                    plot_T1_lim = False
                    horizontal_alignment='right'
                    T1 = None
                    T2 = None
                    pulse_length = None
                else:
                    subplot_title = \
                        r'$\langle \sigma_{{z,{qb0}}} ' \
                        r'\rangle$'.format(qb0=var_name)
                    ylabel = self.ylabel_RB
                    plot_T1_lim = plot_T1_lim_base
                    horizontal_alignment='right'
                    T1 = self.T1s[msmt_name][var_name]
                    T2 = self.T2s[msmt_name][var_name]
                    pulse_length = self.pulse_lengths[msmt_name][var_name]

                line = plotting_function(x=self.data_dict[msmt_name]['n_cl'],
                                         y=self.data_dict[msmt_name]['data'][
                                             var_name],
                                         fig=fig,
                                         ax=ax,
                                         marker='o',
                                         xlabel=self.xlabel_RB,
                                         ylabel=ylabel,
                                         return_line=True)

                if self.fit_res_dict[msmt_name][var_name] is None:
                    pass
                else:
                    self.add_errorbars_T1_lim_simultaneous_RB(
                        ax,
                        line=line,
                        msmt_name=msmt_name,
                        var_name=var_name,
                        T1=T1, T2=T2, pulse_length=pulse_length,
                        plot_T1_lim=plot_T1_lim,
                        horizontal_alignment=horizontal_alignment, **kw)

                ax.set_title(subplot_title)
                if len(self.qb_names) == 2:
                    ax.set_ylim(
                        [0.3,
                         max(max(self.data_dict[msmt_name]['data'][
                                     var_name])+.1, 1.1)])

                # fig.subplots_adjust(hspace=0.2)
                # set fig title
                fig.text(0.5, 1.05, fig_title, fontsize=font_size,
                         horizontalalignment='center',
                         verticalalignment='bottom',
                         transform=ax.transAxes)

                # show figure
                if show_SRB_base:
                    plt.show()

                # Save figure
                if save_fig:
                    filename = os.path.abspath(os.path.join(
                        self.save_folder, fig_title+'.png'))
                    fig.savefig(filename, format='png',
                                     bbox_inches='tight')

                if close_fig:
                    plt.close(fig)

    def plot_CZ_ISRB_two_qubits(self, plot_errorbars=True, save_fig=True,
                                fig_title_suffix='', **kw):

        show_CZ_ISRB = kw.pop('show_CZ_ISRB', False)
        plot_T1_lim_CZ_ISRB = kw.pop('plot_T1_lim_CZ_ISRB', True)
        add_textbox_CZ_ISRB = kw.pop('add_textbox_CZ_ISRB', True)

        fig, ax = plt.subplots()

        var_name = 'corr'

        for msmt_name, msmt_dict in self.data_dict.items():
            # make legend labels
            legend_label = \
                (r'f($\langle \sigma_{{ z,{{{qb0}}} }} '
                 r'\sigma_{{ z, {{{qb1}}} }} \rangle$)'.format(
                    qb0=self.qb_names[0], qb1=self.qb_names[1]))
            legend_label = msmt_name[0:-4] + ' ' + legend_label

            corr_data = msmt_dict['data'][var_name]
            n_cl = msmt_dict['n_cl']

            line = plotting_function(x=n_cl,
                                     y=corr_data,
                                     fig=fig,
                                     ax=ax,
                                     marker='o',
                                     xlabel=self.xlabel_RB,
                                     ylabel=self.ylabel_RB,
                                     label=legend_label,
                                     return_line=True)

            self.add_errorbars_T1_lim_simultaneous_RB(
                ax,
                line=line,
                msmt_name=msmt_name,
                var_name=var_name,
                plot_T1_lim=False,
                add_textbox=False,
                plot_errorbars=plot_errorbars,
                **kw)


        if add_textbox_CZ_ISRB:
            textstr = ('$r_{CZ}$'+' = {:.4g} $\pm$ ({:.4g})%'. format(
                self.CZ_gate_error['val']*100,
                self.CZ_gate_error['stderr']*100))
            for msmt_name in self.folders_dict:
                textstr += \
                    ('\n$r_{{perCl,{{{m_name}}} }}$'.format(
                        m_name=msmt_name[0:-4])+
                     ' = {:.6g} $\pm$ ({:.4g})%'.format(
                        (1-self.fit_res_dict[msmt_name]['corr'].params[
                            'fidelity_per_Clifford'].value)*100,
                        (1-self.fit_res_dict[msmt_name]['corr'].params[
                            'fidelity_per_Clifford'].stderr)*100))

        if plot_T1_lim_CZ_ISRB:
            default_color_cycle = \
                plt.rcParams['axes.prop_cycle'].by_key()['color']
            # msmt_name = 'ICZ_CxC'
            for idx, qb_name in enumerate(self.qb_names):

                #set T1 limit color
                T1_lim_color = default_color_cycle[
                    default_color_cycle.index(line[0].get_color())+idx+1]
                T1_lim_label = 'decoh-limit ' + qb_name

                T1 = self.T1s[msmt_name][qb_name]
                T2 = self.T2s[msmt_name][qb_name]
                pulse_length = self.pulse_lengths[msmt_name][qb_name]

                F_T1, p_T1 = calc_T1_limited_fidelity(T1, T2, pulse_length)

                n_cl = self.data_dict[msmt_name]['n_cl']
                fit_res = self.fit_res_dict[msmt_name]['corr']
                x_fine = np.linspace(0, n_cl[-1], 1000)
                T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                    x_fine, fit_res.best_values['Amplitude'], p_T1,
                    fit_res.best_values['offset'])

                ax.plot(x_fine, T1_limited_curve, '-.', color=T1_lim_color,
                        linewidth=line_width,
                        label=T1_lim_label)

                if add_textbox_CZ_ISRB:
                    textstr += ('\n$r_{{Cl,{{{qb}}}}}^{{T_1}}$  = '.format(
                                    qb=qb_name) +
                                '{:.6g}%'.format((1-F_T1)*100))


        # set legend
        handles, labels = ax.get_legend_handles_labels()

        msmt_idxs = [labels.index(i) for i in labels if 'CZ' in i]
        msmt_handles = [handles[i] for i in msmt_idxs]
        msmt_labels = [labels[i] for i in msmt_idxs]
        try:
            decoh_lim_idx = [labels.index(i) for i in labels if
                              'decoh' in i][0]
            decoh_lim_idxs = [decoh_lim_idx, decoh_lim_idx + 1]
            decoh_lim_handles = [handles[i] for i in decoh_lim_idxs]
            decoh_lim_labels = [labels[i] for i in decoh_lim_idxs]
        except Exception:
            decoh_lim_handles = [None, None]
            decoh_lim_labels = [None, None]
        try:
            CI_idx = [labels.index(i) for i in labels if 'CI' in i][0]
            CI_idxs = [CI_idx, CI_idx + 1]
            CI_handles = [handles[i] for i in CI_idxs]
            CI_labels = [labels[i] for i in CI_idxs]
        except Exception:
            CI_handles = [None, None]
            CI_labels = [None, None]

        handles = [msmt_handles[0], CI_handles[0],
                   msmt_handles[1], CI_handles[1],
                   decoh_lim_handles[0], decoh_lim_handles[1]]
        labels = [msmt_labels[0], CI_labels[0],
                  msmt_labels[1], CI_labels[1],
                  decoh_lim_labels[0], decoh_lim_labels[1]]
        [handles.remove(item) for item in handles if item is None]
        [labels.remove(item) for item in labels if item is None]

        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                  ncol=1, frameon=False, fontsize=font_size)

        # ax.set_ylim((0.3, 1))
        # ax.set_yticks(0.1*np.arange(ax.get_ylim()[0]*10,
        #                             1+ax.get_ylim()[1]*10))

        # find n_cl array which goes up to the highest value and use
        # that to set xticks
        ax.get_xaxis().get_major_formatter().set_scientific(False)

        msmt_name_0 = list(self.data_dict)[0]
        msmt_name_1 = list(self.data_dict)[1]
        temp_list = [self.data_dict[msmt_name_0]['n_cl'],
                     self.data_dict[msmt_name_1]['n_cl']]
        max_n_cl = temp_list[np.argmax(
            [self.data_dict[msmt_name_0]['n_cl'][-1],
             self.data_dict[msmt_name_1]['n_cl'][-1]])]
        if max_n_cl[-1] >=100:
            ax.set_xticks(max_n_cl[0::2])

        # Add textbox
        if add_textbox_CZ_ISRB:
            ax.text(0.975, 0.95, textstr, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=font_size)

        # show figure
        if show_CZ_ISRB:
            plt.show()

        # Save figure
        if save_fig:
            fig_title = 'CZ_ISRB_plot'
            if plot_errorbars:
                fig_title += '_errorbars'
            fig_title += '_' + fig_title_suffix
            # for save_folder in self.folders_dict.values():
            filename = os.path.abspath(os.path.join(self.save_folder,
                                                    fig_title+'.png'))
            fig.savefig(filename, format='png',
                        bbox_inches='tight')

        if kw.pop('close_fig', True):
            plt.close(fig)

    def plot_cross_talk(self, plot_errorbars=True, save_fig=True,
                        fig_title_suffix=None, **kw):

        # ONLY FOR 2 qubits
        show_SRB_cross_talk = kw.pop('show_SRB_cross_talk', False)
        plot_T1_lim_cross_talk = kw.pop('plot_T1_lim_cross_talk', False)
        add_textbox_cross_talk = kw.pop('add_textbox_cross_talk', True)
        n = len(self.qb_names)


        fig, ax = plt.subplots()
        ax_mirror = ax.twinx()

        # make legend labels
        legend_labels = []
        if n == 2:
            for qb_name in self.qb_names:
                legend_labels.extend(
                    [r'C$\otimes$C f($\langle \sigma_{{z,{qb}}} \rangle$)'.format(
                        qb=qb_name)])
            legend_labels.extend(
                [r'C$\otimes$C f($\langle \sigma_z \sigma_z \rangle$)',
                 r'f($\langle \sigma_z \rangle \langle \sigma_z \rangle$)'])
        else:
            for qb_name in self.qb_names:
                legend_labels.extend(
                    [r'$\langle \sigma_{{z,{qb}}} \rangle$'.format(
                qb=qb_name)])
            legend_labels.extend(
                [r'$\langle '
                 r'\sigma_z^{{\otimes {{{n}}} }} \rangle$'.format(n=n),
                 r'$\langle \sigma_z '
                 r'\rangle ^{{\otimes {{{n}}} }}$'.format(n=n)])

        msmt_name = [msmt_name for msmt_name in
                     self.folders_dict if msmt_name not in
                     self.clifford_qbs][0]

        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_colors = [i for i in default_color_cycle if ('#2ca02c' not in i)
                       and ('#d62728' not in i)]
        for var_idx, var_name in enumerate(self.data_dict[msmt_name]['data']):

            if var_name=='corr':
                #set color of the corr line to a darker green
                line_color = 'g'
            else:
                line_color = line_colors[var_idx]

            line = plotting_function(x=self.data_dict[msmt_name]['n_cl'],
                                     y=self.data_dict[msmt_name]['data'][
                                         var_name],
                                     fig=fig,
                                     ax=ax,
                                     marker='o',
                                     xlabel=self.xlabel_RB,
                                     ylabel=self.ylabel_RB,
                                     label=legend_labels[var_idx],
                                     return_line=True,
                                     line_color=line_color)

            self.add_errorbars_T1_lim_simultaneous_RB(
                ax,
                line=line,
                msmt_name=msmt_name,
                var_name=var_name,
                plot_T1_lim=plot_T1_lim_cross_talk,
                add_textbox=False,
                plot_errorbars=plot_errorbars,
                **kw)

        # plot the product \sigma_qb2 * \signam_qb7 which should equal
        # \sigma_corr in the ideal case
        self.find_and_plot_alpha_product(msmt_name=msmt_name, ax=ax,
                                         legend_labels=legend_labels, n=n)

        if plot_errorbars:
            # set legend
            handles, labels = ax.get_legend_handles_labels()
            alpha_product_handle = handles.pop(labels.index(legend_labels[-1]))
            alpha_product_label = labels.pop(labels.index(legend_labels[-1]))
            handles_new = []
            labels_new = []

            for i in range(n+1):
                handles_new.extend([handles[i], handles[i+n+1]])
                labels_new.extend([labels[i], labels[i+n+1]])
            handles_new.extend([alpha_product_handle])
            labels_new.extend([alpha_product_label])
            ax.legend(handles_new, labels_new,
                      loc='center left', bbox_to_anchor=(1.15, 0.5),
                      ncol=1, frameon=False, fontsize=font_size)
        else:
            ax.legend(loc='upper right', ncol=1, fancybox=True,
                      frameon=False, fontsize=font_size)

        # ax.set_ylim((0.3, 1))
        # ax.set_yticks(0.1*np.arange(ax.get_ylim()[0]*10, 1+ax.get_ylim()[1]*10))
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        if self.data_dict[msmt_name]['n_cl'][-1] >= 100:
            ax.set_xticks(self.data_dict[msmt_name]['n_cl'][0::2])

        ax_mirror.set_ylabel(self.ylabel_corr)

        ax_mirror.yaxis.label.set_color(line[0].get_color())
        ax_mirror.set_yticks(ax.get_yticks())
        ax_mirror.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]))
        ax_mirror.tick_params(axis='y', colors=line[0].get_color(),
                              direction='in')
        # turn off ax ticks on the right
        ax.tick_params(axis='y', which='both', labelleft='on',
                       labelright='off')

        if add_textbox_cross_talk:
            self.add_textbox_cross_talk(
                fit_res_dict_CxC=self.fit_res_dict[msmt_name], ax=ax, **kw)

        # show figure
        if show_SRB_cross_talk:
            plt.show()

        # Save figure
        if save_fig:
            fig_title = 'cross-talk_plot'
            if plot_errorbars:
                fig_title += '_errorbars'

            if fig_title_suffix is not None:
                fig_title += '_' + fig_title_suffix

            filename = os.path.abspath(os.path.join(
                self.save_folder, fig_title+'.png'))
            fig.savefig(filename, format='png',
                        bbox_inches='tight')

        if kw.pop('close_fig', True):
            plt.close(fig)

    def find_and_plot_alpha_product(self, msmt_name, ax, legend_labels, n=None):
        # plot the product \sigma_qb2 * \signam_qb7 which should equal
        # \sigma_corr in the ideal case
        if n is None:
            n = len(self.qb_names)

        if n == 2:
            self.alpha_product = \
                1 + 2*self.data_dict[msmt_name]['data'][self.qb_names[0]] * \
                    self.data_dict[msmt_name]['data'][self.qb_names[1]] - \
                self.data_dict[msmt_name]['data'][self.qb_names[0]] - \
                self.data_dict[msmt_name]['data'][self.qb_names[1]]
        else:
            self.alpha_product = 1
            for qb_name in self.qb_names:
                self.alpha_product *= self.data_dict[msmt_name]['data'][qb_name]

        alpha_fit_res = self.fit_data(self.alpha_product,
                                      self.data_dict[msmt_name]['n_cl'])
        x_fine = np.linspace(self.data_dict[msmt_name]['n_cl'][0],
                             self.data_dict[msmt_name]['n_cl'][-1], 100)
        alpha_data_fine = fit_mods.RandomizedBenchmarkingDecay(
            x_fine, **alpha_fit_res.best_values)
        ax.plot(x_fine, alpha_data_fine, 'm--',
                label=legend_labels[-1], linewidth=line_width+2, dashes=(2, 2))
        ax.plot(self.data_dict[msmt_name]['n_cl'], self.alpha_product, 'mo')

    def add_textbox_cross_talk(self, fit_res_dict_CxC, ax, **kw):
        # pring infidelities
        textstr = ''
        for qb_name in self.qb_names:
            textstr += ('$r_{{{qb}}}$'.format(qb=qb_name) +
                            ' = {:.3g} $\pm$ ({:.3g})%'.format(
                        (1-fit_res_dict_CxC[qb_name].params[
                            'fidelity_per_Clifford'].value)*100,
                        fit_res_dict_CxC[qb_name].params[
                            'fidelity_per_Clifford'].stderr*100) + '\n')

        textstr += ('$r_{corr}$' + ' = {:.3g} $\pm$ ({:.3g})%'.format(
                (1-fit_res_dict_CxC['corr'].params[
                    'fidelity_per_Clifford'].value)*100,
                fit_res_dict_CxC['corr'].params[
                    'fidelity_per_Clifford'].stderr*100))

        ax.text(0.27, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=font_size)

        # Print alphas
        textstr = ''
        for qb_name in self.qb_names:
            textstr += (r'$\alpha_{{{qb}}}$'.format(qb=qb_name) +
                        ' = {:.3g} $\pm$ ({:.3g})%'.format(
                fit_res_dict_CxC[qb_name].best_values['p']*100,
                fit_res_dict_CxC[qb_name].params['p'].stderr*100) + '\n')

        textstr += (r'$\alpha_{corr}$' +
                    ' = {:.3g} $\pm$ ({:.3g})%'.format(
                fit_res_dict_CxC['corr'].best_values['p']*100,
                fit_res_dict_CxC['corr'].params['p'].stderr*100))

        textstr += ('\n' + r'$\delta \alpha$' +
                    ' = {:.3g} $\pm$ ({:.3g})%'.format(
                        self.delta_alpha['val']*100,
                        self.delta_alpha['stderr']*100))

        ax.text(0.64, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=font_size)

    def plot_addressability(self, plot_errorbars=True, save_fig=True,
                            fig_title_suffix='', **kw):

        # ONLY FOR 2 QUBITS NOW!
        show_SRB_addressab = kw.pop('show_SRB_addressab', False)
        plot_T1_lim_addressab = kw.pop('plot_T1_lim_addressab', True)
        add_textbox_addressab = kw.pop('add_textbox_addressab', True)

        self.delta_r = {}
        for qb_idx, qb_name in enumerate(self.qb_names):

            fig, ax = plt.subplots()

            # get the msmt_name (CxI or IxC) such that qb_name was the one
            # that had Cliffords applied to it
            msmt_name_CxI_IxC = [m_name for (m_name, name) in
                                 self.clifford_qbs.items() if name==qb_name][0]
            # msmt_name_CxI_IxC = [m_name for (m_name, var_dict) in
            #                      self.fit_res_dict.items() if
            #                      (var_dict[qb_name] is not None
            #                       and m_name != 'CxC')][0]
            msmt_name_CxC = [msmt_name for msmt_name in
                             self.folders_dict if msmt_name not in
                             self.clifford_qbs][0]
            legend_labels = \
                [r'{}$\otimes${} {}'.format(
                    msmt_name_CxI_IxC[0], msmt_name_CxI_IxC[-1], qb_name),
                 r'C$\otimes$C {}'.format(qb_name)]

            for msmt_idx, msmt_name in enumerate([msmt_name_CxI_IxC,
                                                  msmt_name_CxC]):

                line = plotting_function(x=self.data_dict[msmt_name]['n_cl'],
                                         y=self.data_dict[msmt_name]['data'][
                                             qb_name],
                                         fig=fig,
                                         ax=ax,
                                         marker='o',
                                         xlabel=self.xlabel_RB,
                                         ylabel=self.ylabel_RB,
                                         label=legend_labels[msmt_idx],
                                         return_line=True)

                if msmt_idx == 0:
                    #set T1 limit color
                    default_color_cycle = \
                        plt.rcParams['axes.prop_cycle'].by_key()['color']
                    T1_lim_color = default_color_cycle[
                        default_color_cycle.index(line[0].get_color())+2]
                    T1 = self.T1s[msmt_name][qb_name]
                    T2 = self.T2s[msmt_name][qb_name]
                    pulse_length = self.pulse_lengths[msmt_name][qb_name]
                    FT1 = self.add_errorbars_T1_lim_simultaneous_RB(
                        ax=ax,
                        line=line,
                        msmt_name=msmt_name,
                        var_name=qb_name,
                        plot_T1_lim=plot_T1_lim_addressab,
                        T1=T1, T2=T2, pulse_length=pulse_length,
                        T1_lim_color=T1_lim_color,
                        add_textbox=False,
                        plot_errorbars=plot_errorbars,
                        return_FT1=True,
                        **kw)
                else:
                    self.add_errorbars_T1_lim_simultaneous_RB(
                        ax=ax,
                        line=line,
                        msmt_name=msmt_name,
                        var_name=qb_name,
                        plot_T1_lim=False,
                        add_textbox=False,
                        plot_errorbars=plot_errorbars,
                        return_FT1=False,
                        **kw)

            # add legend
            handles, labels = ax.get_legend_handles_labels()
            if plot_errorbars:
                handles_new = [handles.pop(1)]
                labels_new = [labels.pop(1)]
                for i in range(2):
                    handles_new.extend([handles[i], handles[i+2]])
                    labels_new.extend([labels[i], labels[i+2]])
                ax.legend(handles_new, labels_new, loc='upper right', ncol=1,
                          frameon=False, fontsize=font_size)
            else:
                handles_new = [h for h in handles if h!=handles[1]]
                handles_new.extend([handles[1]])
                labels_new = [l for l in labels if l!=labels[1]]
                labels_new.extend([labels[1]])
                ax.legend(handles_new, labels_new, loc='upper right', ncol=1,
                          fancybox=True, frameon=False, fontsize=font_size)

            ax.set_ylim((0.3, 1))
            ax.set_yticks(0.1*np.arange(ax.get_ylim()[0]*10,
                                        1+ax.get_ylim()[1]*10))

            # find n_cl array which goes up to the highest value and use
            # that to set xticks
            temp_list = [self.data_dict[msmt_name_CxI_IxC]['n_cl'],
                          self.data_dict[msmt_name_CxC]['n_cl']]
            max_n_cl = temp_list[np.argmax(
                [self.data_dict[msmt_name_CxI_IxC]['n_cl'][-1],
                 self.data_dict[msmt_name_CxC]['n_cl'][-1]])]
            ax.set_xticks(max_n_cl[0::2])
            ax.get_xaxis().get_major_formatter().set_scientific(False)

            # Calculate delta_r = abs(r_CxC - r_CxI_IxC)
            fit_res_IxC_CxI = self.fit_res_dict[msmt_name_CxI_IxC][qb_name]
            fit_res_CxC = self.fit_res_dict[msmt_name_CxC][qb_name]
            self.delta_r[qb_name] = {}
            self.delta_r[qb_name]['val'] = np.abs(
                1 - fit_res_IxC_CxI.params['fidelity_per_Clifford'].value -
                (1 - fit_res_CxC.params['fidelity_per_Clifford'].value))*100
            self.delta_r[qb_name]['stderr'] = np.abs(
                1 - fit_res_IxC_CxI.params['fidelity_per_Clifford'].stderr -
                (1 - fit_res_CxC.params['fidelity_per_Clifford'].stderr))*100
            # add textbox -> ONLY FOR 2 QUBITS
            if add_textbox_addressab:
                self.add_textbox_addressability(
                    ax=ax,
                    FT1=FT1,
                    qbC_name=qb_name,
                    qbI_name=[qb for qb in self.qb_names if qb!=qb_name][0],
                    fit_res_IxC_CxI=fit_res_IxC_CxI,
                    fit_res_CxC=fit_res_CxC,
                    delta_r=self.delta_r)

            # Save figure
            if save_fig:
                fig_title = 'addressability_plot_{}'.format(qb_name)
                if plot_errorbars:
                    fig_title += '_errorbars'
                fig_title += '_' + fig_title_suffix
                # for save_folder in [self.folders_dict[msmt_name_CxC],
                #                     self.folders_dict[msmt_name_CxI_IxC]]:
                filename = os.path.abspath(os.path.join(self.save_folder,
                                                        fig_title+'.png'))
                fig.savefig(filename, format='png',
                            bbox_inches='tight')

        # show figure
        if show_SRB_addressab:
            plt.show()

        if kw.pop('close_fig', True):
            plt.close(fig)

    def add_textbox_addressability(self, ax,  FT1, qbC_name, qbI_name,
                                   fit_res_IxC_CxI, fit_res_CxC, delta_r):

        textstr = \
            ('$r_{{{qbc}}}$'.format(qbc=qbC_name[-1]) +
                ' = {:.3g} $\pm$ ({:.3g})%'.format(
                (1 - fit_res_IxC_CxI.params['fidelity_per_Clifford'].value)*100,
                fit_res_IxC_CxI.params['fidelity_per_Clifford'].stderr*100) +
            '\n' + '$r_{{ {qbc}|{{{qbi}}} }}$'.format(
                qbc=qbC_name[-1], qbi=qbI_name[-1]) +
            ' = {:.3g} $\pm$ ({:.3g})%'.format(
               (1 - fit_res_CxC.params['fidelity_per_Clifford'].value)*100,
               fit_res_CxC.params['fidelity_per_Clifford'].stderr*100))

        textstr += \
            ('\n'+r'$\delta r_{{ {qbc}|{{{qbi}}} }}$'.format(
                qbc=qbC_name[-1], qbi= qbI_name[-1]) +
             ' = {:.3g} $\pm$ ({:.3g})%'.format(
                delta_r[qbC_name]['val'], delta_r[qbC_name]['stderr']))

        textstr += ('\n'+'$r_{T_1}$' + ' = {:.3g}%'.format((1 - FT1)*100))
        ax.text(1.05, 0.5, textstr, transform=ax.transAxes,
                fontsize=font_size, verticalalignment='center',
                horizontalalignment='left')

    def add_errorbars_T1_lim_simultaneous_RB(self, ax,
                                             msmt_name, var_name,
                                             line=None,
                                             T1=None, T2=None,
                                             pulse_length=None, **kw):

        plot_T1_lim = kw.pop('plot_T1_lim', True)
        return_FT1 = kw.pop('return_FT1', False)
        T1_lim_color = kw.pop('T1_lim_color', None)
        T1_lim_label = kw.pop('T1_lim_label', None)
        plot_errorbars = kw.get('plot_errorbars', True)
        add_textbox = kw.pop('add_textbox', True)
        plot_fit = kw.pop('plot_fit', True)

        data = self.data_dict[msmt_name]['data'][var_name]
        n_cl = self.data_dict[msmt_name]['n_cl']
        fit_res = self.fit_res_dict[msmt_name][var_name]
        epsilon = self.epsilon_dict[msmt_name][var_name]

        if line is not None:
            try:
                c = line[0].get_color()
            except:
                c = line.get_color()
        else:
            c = None

        if plot_T1_lim and T1_lim_color is None:
            default_color_cycle = \
                plt.rcParams['axes.prop_cycle'].by_key()['color']
            try:
                T1_lim_color = \
                    default_color_cycle[default_color_cycle.index(c)+1]
            except:
                T1_lim_color = 'b'

        if plot_errorbars:
            err_bars = epsilon
            err_label = str(int(self.conf_level*100)) + '% CI'
            a_tools.plot_errorbars(n_cl,
                                   data,
                                   ax=ax,
                                   err_bars=err_bars,
                                   label=err_label,
                                   linewidth=line_width//2,
                                   marker='none',
                                   capsize=line_width,
                                   capthick=line_width,
                                   color=c)

        if plot_fit:
            x_fine = np.linspace(0, n_cl[-1], 1000)
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **fit_res.best_values)
            ax.plot(x_fine, best_fit, color=c)


        textstr = \
            ('$r_{perCl,' + var_name + '}$'+' = '
                    '{:.6g} $\pm$ ({:.4g})% \n'.format(
                (1-fit_res.params[
                    'fidelity_per_Clifford'].value)*100,
                (1-fit_res.params[
                    'fidelity_per_Clifford'].stderr)*100))

        # Here we add the line corresponding to T1 limited fidelity
        F_T1 = None
        if pulse_length is not None:
            if T1 is not None and T2 is not None and plot_T1_lim:
                F_T1, p_T1 = calc_T1_limited_fidelity(
                    T1, T2, pulse_length)
                x_fine = np.linspace(0, n_cl[-1], 1000)
                T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                    x_fine, fit_res.best_values['Amplitude'], p_T1,
                    fit_res.best_values['offset'])

                if T1_lim_label is None:
                    T1_lim_label = 'decoh-limit'
                ax.plot(x_fine, T1_limited_curve, '-.', color=T1_lim_color,
                        linewidth=line_width,
                        label=T1_lim_label)
                textstr += ('$r_{Cl,avg}^{T_1}$  = ' +
                            '{:.6g}%\t'.format((1-F_T1)*100))

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=font_size)

        # Add a textbox
        if add_textbox:
            ax.text(0.975, 0.975, textstr, transform=ax.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         fontsize=font_size)

        if plot_T1_lim and return_FT1:
            return F_T1
        else:
            return

def load_T1_T2_pulse_length(folders_dict, qb_names,
                            T1s=None, T2s=None, pulse_lengths=None):
    """
    Loads T1, T2, and DRAG pulse length from folders for all the qubits
    in qb_names.

    Args:
        folders_dict (dict): folders_dict (dict): of the form
            {msmt_name: folder}
        qb_names (list): list of qubit names used in the experiment
        T1s (dict): an already-existing T1s_dict
        T2s (dict): an already-existing T2s_dict
        pulse_lengths (dict): an already-existing pulse_lengths_dict
    Returns:
        T1, T2, pulse_length dicts of the form:
            {msmt_name: {'qb_name': parameter_value}}

    Example: for IRB for qb2, if the folders correspond to the IRB measurements
        where the gate of interest was interleaved, then T1 will look like:
        T1 = {'interleaved_gate_name': {'qb2': T1_qb2}}
        where the used passed in msmt='interleaved_gate_name'.

    """

    if type(qb_names) != list and type(qb_names) != tuple:
        qb_names = [qb_names]

    if T1s is None:
        T1s = {}
    if T2s is None:
        T2s = {}
    if pulse_lengths is None:
        pulse_lengths = {}

    count = 0
    for msmt_name, folder in folders_dict.items():
        T1s[msmt_name] = {}
        T2s[msmt_name] = {}
        pulse_lengths[msmt_name] = {}
        if type(folder) != str:
            if len(folder)>1:
                print('Qubit coherence times are taken from the first '
                      'folder for the measurement "{}".'.format(msmt_name))
            folder = folder[0]

        try:
            h5filepath = a_tools.measurement_filename(folder)
            data_file = h5py.File(h5filepath, 'r+')
            instr_set = data_file['Instrument settings']
        except Exception:
            logging.warning('Could not open data file "{}".'.format(folder))
            count += 1

        for qb in qb_names:
            try:
                T1s[msmt_name][qb] = float(instr_set[qb].attrs['T1'])
            except Exception:
                print('Could not load T1 for {}.'.format(qb))
            try:
                T2s[msmt_name][qb] = float(instr_set[qb].attrs['T2'])
            except Exception:
                print('Could not load T2 for {}.'.format(qb))
            try:
                pulse_lengths[msmt_name][qb] = \
                    float(instr_set[qb].attrs['nr_sigma']) * \
                    float(instr_set[qb].attrs['gauss_sigma'])
            except Exception:
                print('Could not load pulse_length for {}.'.format(qb))

    if count == len(folders_dict):
        raise ValueError('Could not open any of the data files.')

    return T1s, T2s, pulse_lengths

def add_final_dataset_to_analysisgroup(analysis_group, data_dict,
                                 datasetname, shape, column_names):

    try:
        dset = analysis_group.create_dataset(
            datasetname, shape,
            dtype='float64')
    except:
        del analysis_group[datasetname]
        dset = analysis_group.create_dataset(
            datasetname, shape,
            dtype='float64')

    for i, data in enumerate(data_dict.values()):
        dset[:, i] = data

    dset.attrs['column_names'] = h5d.encode_to_utf8(column_names)


def create_experimentaldata_dataset(data_object, data_dict, group_name, shape,
                                    column_names, names_units_dict):

    sweep_par_names = names_units_dict.get('sweep_par_names', '')
    sweep_par_units = names_units_dict.get('sweep_par_units', '#')
    value_names = names_units_dict.get('value_names', '')
    value_units = names_units_dict.get('value_units', '#')

    try:
        data_group = data_object.create_group(group_name)
    except Exception:
        del data_object[group_name]
        data_group = data_object.create_group(group_name)
    try:
        dset = data_group.create_dataset(
            'Corrected data', shape,
            dtype='float64')
    except Exception:
        del data_group['Corrected data']
        dset = data_group.create_dataset(
            'Corrected data', shape,
            dtype='float64')

    for i, data in enumerate(data_dict.values()):
        dset[:, i] = data

    dset.attrs['column_names'] = h5d.encode_to_utf8(column_names)
    # Added to tell analysis how to extract the data
    data_group.attrs['datasaving_format'] = h5d.encode_to_utf8('Version 2')
    data_group.attrs['sweep_parameter_names'] = h5d.encode_to_utf8(
        sweep_par_names)
    data_group.attrs['sweep_parameter_units'] = h5d.encode_to_utf8(
        sweep_par_units)
    data_group.attrs.create(
        'corrected data based on', 'calibration points'.encode('utf-8'))
    data_group.attrs['value_names'] = h5d.encode_to_utf8(value_names)
    data_group.attrs['value_units'] = h5d.encode_to_utf8(value_units)

    return data_group

def create_experimentaldata_dataset_v2(
        data_object, data_dict, group_name, shape,
        column_names, names_units_dict):

    sweep_par_names = names_units_dict.get('sweep_par_names', '')
    sweep_par_units = names_units_dict.get('sweep_par_units', '#')
    value_names = names_units_dict.get('value_names', '')
    value_units = names_units_dict.get('value_units', '#')

    try:
        data_group = data_object.create_group(group_name)
    except Exception:
        del data_object[group_name]
        data_group = data_object.create_group(group_name)
    try:
        dset = data_group.create_dataset(
            'Corrected data', shape,
            dtype='float64')
    except Exception:
        del data_group['Corrected data']
        dset = data_group.create_dataset(
            'Corrected data', shape,
            dtype='float64')

    for i, data in enumerate(data_dict.values()):
        dset[:, i] = data

    dset.attrs['column_names'] = h5d.encode_to_utf8(column_names)
    # Added to tell analysis how to extract the data
    data_group.attrs['datasaving_format'] = h5d.encode_to_utf8('Version 2')
    data_group.attrs['sweep_parameter_names'] = h5d.encode_to_utf8(
        sweep_par_names)
    data_group.attrs['sweep_parameter_units'] = h5d.encode_to_utf8(
        sweep_par_units)
    data_group.attrs.create(
        'corrected data based on', 'calibration points'.encode('utf-8'))
    data_group.attrs['value_names'] = h5d.encode_to_utf8(value_names)
    data_group.attrs['value_units'] = h5d.encode_to_utf8(value_units)

    return data_group

def save_computed_parameters(computed_params, name, group):

    """ Allows to save additional parameters computed from fit results,
    such as the pi pulse and pi/2 pulse amplitudes. Each will be
    saved as a new attribute in the FittedParams+var_name group created
    in "save_fitted_parameters."

    Input parameters:
        computed_params:       DICTIONARY of parameters to be saved; first
                               value of main variable, then
                               its statistical data such as stddev.
                               Ex: {'piPulse':piPulse_val,
                               'piPulse_std':piPulse_std_val}.
        var_name:              same var_name used in
                              'save_fitted_parameters'
    """

    if name not in group:
        fit_grp = group.create_group(name)
    else:
        fit_grp = group[name]

    if len(computed_params) == 0:
        logging.warning('Nothing to save. Parameters dictionary is empty.')
    else:
        for msmt_name, par_vals in computed_params.items():
            try:
                par_group = fit_grp.create_group(msmt_name)
            except:  # if it already exists overwrite existing
                par_group = fit_grp[msmt_name]
            if type(par_vals) == dict:
                for par_name, par_val in par_vals.items():
                        par_group.attrs.create(name=par_name, data=par_val)
            else:
                try:
                    for par_val in par_vals:
                        par_group.attrs.create(name=msmt_name, data=par_val)
                except TypeError:
                    par_group.attrs.create(name=msmt_name, data=par_vals)


def save_fitted_parameters(fit_res, fit_name, fit_group,
                           var_name='nr_cliffords'):

    if fit_name not in fit_group:
        fit_grp = fit_group.create_group(fit_name)
    else:
        fit_grp = fit_group[fit_name]

    try:
        fit_grp.attrs['Fit Report'] = \
            '\n'+'*'*80+'\n' + \
            lmfit.fit_report(fit_res) + \
            '\n'+'*'*80 + '\n\n'
    except Exception:
        pass

    fit_grp.attrs.create(name='chisqr', data=fit_res.chisqr)
    fit_grp.attrs.create(name='var_name', data=var_name.encode('utf-8'))

    if fit_res.covar is not None:
        if 'covar' in list(fit_grp.keys()):
            del fit_grp['covar']
        fit_grp.create_dataset(name='covar', data=fit_res.covar)

    for parname, par in fit_res.params.items():
        try:
            par_group = fit_grp.create_group(parname)
        except:  # if it already exists overwrite existing
            par_group = fit_grp[parname]
        par_dict = vars(par)
        for val_name, val in par_dict.items():
            if val_name == '_val':
                val_name = 'value'
            if val_name == 'correl' and val is not None:
                try:
                    correl_group = par_group.create_group(val_name)
                except:
                    correl_group = par_group[val_name]
                for cor_name, cor_val in val.items():
                    correl_group.attrs.create(name=cor_name, data=cor_val)
            else:
                try:
                    par_group.attrs.create(name=val_name, data=val)
                except:
                    pass


def calc_T1_limited_fidelity(T1, T2, pulse_length, gate_decomp='HZ'):
        '''
        Formula from Asaad et al.
        pulse separation is time between start of pulses

        Returns:
            F_cl (float): decoherence limited fildelity
            p (float): decoherence limited depolarization parameter
        '''
        #Np = 1.875  # Avg. number of gates per Clifford for XY decomposition
        #Np = 0.9583  # Avg. number of gates per Clifford for HZ decomposition
        if gate_decomp == 'HZ':
            Np = 1.125
        elif gate_decomp == 'XY':
            Np = 1.875
        else:
            raise ValueError('Gate decomposition not recognized.')

        F_cl = (1/6*(3 + 2*np.exp(-1*pulse_length/(T2)) +
                     np.exp(-pulse_length/T1)))**Np
        p = 2*F_cl - 1

        return F_cl, p

def calculate_confidence_intervals(nr_seeds, nr_cliffords,
                                   conf_level=0.68, infidelity=0,
                                   epsilon_guess=0.1, eta=0):

    # From Helsen et al. (2017)
    # For each number of cliffords in nr_cliffords (array), finds epsilon
    # such that with probability greater than conf_level, the true value of
    # the survival probability, p_N_m, for a given N=nr_seeds and
    # m=nr_cliffords, is in the interval
    # [p_N_m_measured-epsilon, p_N_m_measured+epsilon]
    # See Helsen et al. (2017) for more details.

    # eta is the SPAM-dependent prefactor defined in Helsen et al. (2017)
    epsilon = []
    delta = 1-conf_level
    for n_cl in nr_cliffords:
        if n_cl == 0:
            epsilon.append(0)
        else:
            V = min((13*n_cl*infidelity**2)/2, 7*infidelity/2)

            H = lambda eps: (1/(1-eps))**((1-eps)/(V+1)) * \
                            (V/(V+eps))**((V+eps)/(V+1)) - \
                            (delta/2)**(1/nr_seeds)
            epsilon.append(optimize.fsolve(H, epsilon_guess)[0])

    return np.asarray(epsilon)

def estimate_gate_error(p0, p_gate,  p0_stderr, p_gate_stderr, d=2, std=True):
    """
    Finds the gate error of the interleaved gate in the IRB experiment.
    From Magesan et. al (2012); DOI: 10.1103/PhysRevLett.109.080505

    Args:
        p0 (float): depolarization parameter from the IRB measurement with the
            identity gate interleaved
        p_gate (float): depolarization parameter from the IRB measurement
            with the gate of interest interleaved
        p0_stderr (float): standard error of p0
        p_gate_stderr (float): standard error of p_gate
        d (int): number of qubits
        std (boold): whether to calculate the stderr for the gate error

    Returns:
        rc (float): error of the interleaved gate
        std_err (float): if std==True, standard error of rc
    """

    rc = (d-1)*(1-p_gate/p0)/d
    if std:
        stderr = np.sqrt((p_gate_stderr/(2*p0))**2 +
                          (p0_stderr*p_gate/(2*p0**2))**2)
        return (rc, stderr)
    else:
        return rc

def check_limits_gate_error(rc, p0, p_gate, d=2):
    """
    Checks whether the error of the interleaved gate is in the correct range
    given in Magesan et. al (2012); DOI: 10.1103/PhysRevLett.109.080505.

    Args:
        rc (float): error of interleaved gate
        p0 (float): depolarization parameter from the IRB measurement with the
            identity gate interleaved
        p_gate (float): depolarization parameter from the IRB measurement
            with the gate of interest interleaved
        d (int): number of qubits

    Returns:
        (bool): whether rc is withing the correct limits
    """
    x1 = (d-1)*(np.abs(p0-p_gate/p0)+(1-p0))/d
    x2 = 2*(d**2-1)*(1-p0)/(p0*(d**2)) + 4*np.sqrt(1-p0)*np.sqrt(d**2-1)/p0
    E = min(x1,x2)

    return (rc>=(rc-E) and rc<=(rc+E))

def plotting_function(x, y, fig, ax, marker='-o',
                      ticks_around=True, label=None, **kw):

    # font_size = kw.pop('font_size', 18)
    # marker_size = kw.pop('marker_size', 5)
    # fig_size_dim = kw.pop('fig_size_dim', 10)
    # line_width = kw.pop('line_width', 2)
    # axes_line_width = kw.pop('axes_line_width', 0.5)
    # golden_ratio = (1+np.sqrt(5))/2

    line_color = kw.pop('line_color', None)

    params = {'figure.figsize': (fig_size_dim, fig_size_dim/golden_ratio),
              'figure.dpi': 300,
              'savefig.dpi': 300,
              'figure.titlesize': font_size,
              'legend.fontsize': font_size,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.linewidth': axes_line_width,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'lines.markersize': marker_size,
              'lines.linewidth': line_width,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              }

    if ticks_around:
        params['xtick.top'] = True
        params['ytick.right'] = True
        plt.rcParams.update(params)

    # Plot:
    if line_color is None:
        line = ax.plot(x, y, marker, label=label)
    else:
        line = ax.plot(x, y, marker, label=label, color=line_color)
    # Adjust ticks
    # set axes labels format to scientific when outside interval [0.01,99]
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter()
    fmt.set_powerlimits((-4, 4))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # Set axis labels
    xlabel = kw.get('xlabel', None)
    ylabel = kw.get('ylabel', None)
    x_unit = kw.get('x_unit', None)
    y_unit = kw.get('y_unit', None)

    if xlabel is not None:
        set_xlabel(ax, xlabel, unit=x_unit)
    if ylabel is not None:
        set_ylabel(ax, ylabel, unit=y_unit)

    fig.tight_layout()

    if kw.get('return_line', False):
        return line
    else:
        return

