import os
import h5py
import lmfit
import logging
import itertools
import numpy as np
import scipy.optimize as optimize

from copy import deepcopy

from matplotlib import pyplot as plt
from pycqed.measurement import hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font_size = 18
marker_size = 5
fig_size_dim = 10
line_width = 2
axes_line_width = 1
golden_ratio = (1+np.sqrt(5))/2
# params = {'figure.figsize': (fig_size_dim, fig_size_dim/golden_ratio),
#           'figure.dpi': 300,
#           'savefig.dpi': 300,
#           'font.size': font_size,
#           'figure.titlesize': font_size,
#           'legend.fontsize': font_size,
#           'axes.labelsize': font_size,
#           'axes.titlesize': font_size,
#           'axes.linewidth': axes_line_width,
#           'xtick.labelsize': font_size,
#           'ytick.labelsize': font_size,
#           'lines.markersize': marker_size,
#           'lines.linewidth': line_width,
#           'xtick.direction': 'in',
#           'ytick.direction': 'in',
#           'axes.formatter.useoffset': False,
#           }
# plt.rcParams.update(params)

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

        make_fig_RB = kw.pop('make_fig_RB', True)
        if not kw.pop('skip', False):
            close_main_fig = kw.pop('close_main_fig', True)
            close_file = kw.pop('close_file', True)

            self.data_RB = self.extract_data(**kw)
            #data = self.corr_data[:-1*(len(self.cal_points[0]*2))]
            #n_cl = self.sweep_points[:-1*(len(self.cal_points[0]*2))]
            self.fit_res = self.fit_data(self.data_RB, self.n_cl, **kw)
            self.fit_results = [self.fit_res]
            self.data_RB_raw = deepcopy(self.data_RB)
            self.fit_res_raw = deepcopy(self.fit_res)
            self.add_analysis_datagroup_to_file()
            self.add_dataset_to_analysisgroup('RB data',
                                              self.data_RB_raw)
            # self.analysis_group.attrs.create(
            #     'corrected data based on', 'calibration points'.encode('utf-8'))
            self.save_fitted_parameters(fit_res=self.fit_res_raw,
                                        var_name='F|1>')

            if kw.pop('add_correction', False):
                A = self.fit_res.best_values['Amplitude']
                B = self.fit_res.best_values['offset']
                A_scaled = A+B
                # overwrite the values of the Amplitude and offset
                self.fit_res.best_values['Amplitude'] = A_scaled
                self.fit_res.best_values['offset'] = 0
                self.data_RB = (A_scaled/A)*(self.data_RB - B)
                self.add_dataset_to_analysisgroup('Corrected data - rescaled',
                                                  self.data_RB)
                self.save_fitted_parameters(fit_res=self.fit_res,
                                            var_name='F|1>')

            if make_fig_RB:
                self.make_figures(close_main_fig=close_main_fig, **kw)

            if close_file:
                self.data_file.close()
        return

    def extract_data(self, **kw):

        qb_RO_channel = kw.pop('qb_RO_channel', None)
        find_empirical_variance = kw.get('find_empirical_variance',
                                         True)
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

            print('Data is assumed to be thresholded!')
            self.n_cl = np.unique(self.sweep_points_2D)
            self.nr_seeds = self.sweep_points.size

            data_raw = self.measured_values[ch]

            data = np.zeros((self.n_cl.size))
            if find_empirical_variance:
                self.epsilon = np.zeros((self.n_cl.size))
            for i in range(self.n_cl.size):
                y = [data_raw[j][i] for j in range(self.nr_seeds)]
                data[i] = np.mean(y)

                if find_empirical_variance:
                    y = 2*(1 - np.asarray(y)) - 1
                    self.epsilon[i] = np.std(y)
            # data = 1-data #Prob of |1>
            data = 2*(1 - data) - 1 # <sigma_z>

        else:
            print('cal_points analysis')
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
            if find_empirical_variance:
                self.epsilon = np.zeros((self.n_cl.size))
            for i in range(self.n_cl.size):
                a[0] = data_rearranged[0][i]
                a[1] = data_rearranged[1][i]
                data_calibrated = a_tools.rotate_and_normalize_data(
                    a, self.cal_points[0], self.cal_points[1])[0]
                self.data_calibrated[i] = data_calibrated
                data_calibrated = data_calibrated[:-int(self.NoCalPoints)]
                data[i] = np.mean(data_calibrated)

                if find_empirical_variance:
                    y = 2*(1 - np.asarray(data_calibrated)) - 1
                    self.epsilon[i] = np.std(y)

            self.calibrated_data_points = np.zeros(
                shape=(self.data_calibrated.shape[0], self.nr_seeds))
            for i,d in enumerate(self.data_calibrated):
                self.calibrated_data_points[i] = \
                    d[:-int(2*len(self.cal_points[0]))]

            # we want prob to be in gnd state
            data = 2*(1 - data) - 1

        return data

    def add_textbox(self, ax, F_T1=None, plot_T1_lim=True, **kw):

        if not hasattr(self, 'fit_res'):
            fit_res = kw.pop('fit_res', None)
        else:
            fit_res = self.fit_res

        textstr = ('$r_{\mathrm{Cl}}$' + ' = {:.3f}% $\pm$ {:.2f}%'.format(
            (1-fit_res.params['fidelity_per_Clifford'].value)*100,
            (fit_res.params['fidelity_per_Clifford'].stderr)*100))
        if F_T1 is not None and plot_T1_lim:
            textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                        '{:.3f}%'.format((1-F_T1)*100))
        textstr += ('\n' + r'$\langle \sigma_z \rangle _{m=0}$ = ' +
                    '{:.2f} $\pm$ {:.2f}'.format(
            fit_res.params['Amplitude'].value + fit_res.params['offset'].value,
            np.sqrt(fit_res.params['offset'].stderr**2 +
                    fit_res.params['Amplitude'].stderr**2)))

        horizontal_alignment = kw.pop('horizontal_alignment', 'right')
        horiz_place = 0.45
        if horizontal_alignment == 'left':
            horiz_place = 0.025

        vertical_alignment = kw.pop('horizontal_alignment', 'top')
        vert_place = 0.95
        if vertical_alignment == 'bottom':
            vert_place = 0.025

        ax.text(horiz_place, vert_place, textstr, transform=ax.transAxes,
                verticalalignment=vertical_alignment,
                horizontalalignment='left')

    def make_figures(self, close_main_fig, **kw):

        xlabel = 'Number of Cliffords, $m$'
        # ylabel = r'Probability, $P$ $\left(|g \rangle \right)$'
        ylabel = r'Expectation value $\langle \sigma_z \rangle$'
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

        x_fine = np.linspace(self.n_cl[0], self.n_cl[-1], 100)
        for fit_res in self.fit_results:
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **fit_res.best_values)
            self.ax.plot(x_fine, best_fit, 'C0')
        self.ax.set_ylim(min(min(self.data_RB)-.1, -.1),
                         max(max(self.data_RB)+.1, 1.1))
        # self.ax.set_ylim(0.4, max(max(self.data_RB)+.1, 1.1))
        # self.ax.set_ylim([0, 1])
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


        if kw.pop('add_legend', True):
            self.ax.legend(loc='lower left', frameon=False)

        # Add a textbox
        if kw.pop('add_textbox_RB', True):
            self.add_textbox(self.ax, F_T1, plot_T1_lim=plot_T1_lim)

        if kw.pop('show_RB', False):
            plt.show()

        if kw.pop('save_fig', True):
            if not close_main_fig:
                # Hacked in here, good idea to only show the main fig but can
                # be optimized somehow
                self.save_fig(self.fig, ylabel='Amplitude (normalized)',
                              xlabel='Nr. of Cliffords, m',
                              close_fig=False, **kw)
            else:
                self.save_fig(self.fig, ylabel='Amplitude (normalized)',
                              xlabel='Nr. of Cliffords, m', **kw)

    def fit_data(self, data, numCliff,
                 print_fit_results=False,
                 show_guess=False,
                 plot_results=False, **kw):

        find_empirical_variance = kw.pop('find_empirical_variance', True)
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


        self.conf_level = kw.pop('conf_level', 0.68)
        if find_empirical_variance:
            # print('epsilon ', self.epsilon)
            fit_res = RBModel.fit(data, numCliff=numCliff, params=params,
                                  scale_covar=False, weights=1/self.epsilon)
        else:
            print('old fit')
            # Run once to get an estimate for the error per Clifford
            fit_res = RBModel.fit(data, numCliff=numCliff, params=params)

            # Use the found error per Clifford to standard errors for the data
            # points fro Helsen et al. (2017)
            epsilon_guess = kw.pop('epsilon_guess', 0.01)
            epsilon = calculate_confidence_intervals(
                nr_seeds=self.nr_seeds,
                nr_cliffords=self.n_cl,
                depolariz_param=fit_res.best_values['p'],
                conf_level=self.conf_level,
                epsilon_guess=epsilon_guess, d=2)

            self.epsilon = epsilon
            print('epsilon ', self.epsilon)
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
        ylabel = r'Expectation value $\langle \sigma_z \rangle$'
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
            x_fine = np.linspace(self.ncl_dict[msmt_name][0],
                                 self.ncl_dict[msmt_name][-1], 1000)
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

            self.ax.set_ylim(0,
                             max(max(self.data_dict[msmt_name])+.1, 1.1))

            textstr += \
                ('$r_{perCl,' + msmt_name + '}$'+' = '
                    '{:.6g} $\pm$ ({:.4g})% \n'.format(
                    (1-self.fit_res_dict[msmt_name].params[
                    'fidelity_per_Clifford'].value)*100,
                    (self.fit_res_dict[msmt_name].params[
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
            self.F_T1 = np.mean(FT1)
            self.p_T1 = np.mean(pT1)
            amp_mean = np.mean(
                [self.fit_res_dict[msmt_name].best_values['Amplitude'] for
                 msmt_name in self.fit_res_dict])
            offset_mean = np.mean(
                [self.fit_res_dict[msmt_name].best_values['offset'] for
                 msmt_name in self.fit_res_dict])
            T1_limited_curve = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, amp_mean, self.p_T1, offset_mean)
            self.ax.plot(x_fine, T1_limited_curve, '-.',
                         color='C1',
                         label='decoh-limit (avg)')
            textstr += ('$r_{Cl,avg}^{T_1}$  = ' +
                        '{:.6g}%\t'.format((1-self.F_T1)*100))

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

    def __init__(self, timestamp=None, qb_names=None,
                 use_cal_points=False, **kw):
        """
        timestamp (dict): string or length-1 list if experiment was done for
            2 qubits (used correlation mode of UHFQC), or either of the forms
            [ts_start, ts_end] or [ts0, ..., tsk] if nr qubits > 2 (each file
            if for one value of the nr_cliffords).
        qb_names (tuple or list): with the names of the qubits used
        """
        self.use_cal_points = use_cal_points
        self.qb_names = qb_names
        use_latest_data = kw.pop('use_latest_data', False)

        if self.qb_names is None:
            raise ValueError('qb_names is not specified.')
        if type(self.qb_names) != list:
            self.qb_names = list(self.qb_names)

        if use_latest_data:
            folder = a_tools.latest_data()
            self.folders = [folder]
        else:
            if timestamp is None:
                raise ValueError('timestamp is not specified.')
            if not isinstance(timestamp, list):
                timestamp = [timestamp]

            self.folders = []
            if len(timestamp)==2:
                qb_idxs = ''.join([i[-1] for i in self.qb_names])
                msmt_label = 'qubits' + qb_idxs
                timestamp = a_tools.get_timestamps_in_range(
                                timestamp_start=timestamp[0],
                                timestamp_end=timestamp[1],
                                label=msmt_label, exact_label_match=True)

            for ts in timestamp:
                folder = a_tools.get_folder(timestamp=ts)
                self.folders.append(folder)
            print(self.folders)
            folder = self.folders[0]
            if not isinstance(folder, str):
                folder = folder[0]

        super().__init__(TwoD=True, folder=folder, **kw)


    def run_default_analysis(self, **kw):

        close_file = kw.pop('close_file', True)
        make_fig_SRB_base = kw.pop('make_fig_SRB_base', True)
        make_fig_cross_talk = kw.pop('make_fig_cross_talk', True)
        add_correction = kw.get('add_correction', False)

        self.T1s, self.T2s, self.pulse_lengths = \
            load_T1_T2_pulse_length(self.folders, self.qb_names)

        if len(self.qb_names) == 2:
            self.correlator_analysis(**kw)
        else:
            self.single_shot_analysis(**kw)

        self.data_dict = deepcopy(self.data_dict_raw)
        self.fit_res_dict = deepcopy(self.fit_res_dict_raw)
        if add_correction:
            d_dict = self.data_dict['data']
            for var_name in d_dict:
                fit_res = self.fit_res_dict_raw[var_name]
                A = fit_res.best_values['Amplitude']
                B = fit_res.best_values['offset']
                A_scaled = A+B
                # overwrite the values of the Amplitude and offset
                self.fit_res_dict[var_name].best_values[
                    'Amplitude'] = A_scaled
                self.fit_res_dict[var_name].best_values[
                    'offset'] = 0

                self.data_dict['data'][var_name] = \
                    (A_scaled/A)*(self.data_dict_raw['data'][var_name] - B)

        # Save fitted params
        for data_file in self.data_files:
            # Save fitted parameters in the Analysis group
            for var_name, fit_res in self.fit_res_dict.items():
                if fit_res is not None:
                    save_fitted_parameters(
                        fit_res,
                        fit_group=data_file['Analysis'],
                        fit_name='Fitted Params ' + var_name)

        # Get depolarization parameters and infidelities
        self.depolariz_params = {}
        self.infidelities = {}
        for var_name, fit_res in self.fit_res_dict.items():
            if fit_res is not None:
                self.depolariz_params[var_name] = \
                    {'val': fit_res.best_values['p'],
                     'stderr': fit_res.params['p'].stderr}
                if var_name == 'corr':
                    d = 2**len(self.qb_names)
                else:
                    d = 2
                self.infidelities[var_name] = \
                    {'val': (d-1)*(1-fit_res.best_values['p'])/d,
                     'stderr': np.abs((d-1)*fit_res.params['p'].stderr/d)}
            else:
                self.depolariz_params[var_name] = None
                self.infidelities[var_name] = None

        # get delta_alpha (delta depolarization params)
        depolariz_params_dict = self.depolariz_params
        self.alpha_product = 1
        std_err_product_squared = 0
        for qb_name in self.qb_names:
            self.alpha_product *= depolariz_params_dict[qb_name]['val']

            std_err_term_squared = \
                (depolariz_params_dict[qb_name]['stderr'])**2
            for other_qb_names in self.qb_names:
                if other_qb_names != qb_name:
                    std_err_term_squared *= \
                        (depolariz_params_dict[ other_qb_names]['val'])**2

            std_err_product_squared += std_err_term_squared

        self.alpha_product_err = np.sqrt(std_err_product_squared)
        delta_alpha = \
            depolariz_params_dict['corr']['val'] - self.alpha_product
        delta_alpha_stderr = np.sqrt(
            (depolariz_params_dict['corr']['stderr'])**2 +
            std_err_product_squared)

        self.delta_alpha = {'val': delta_alpha,
                            'stderr': delta_alpha_stderr}

        save_fig = kw.pop('save_fig', True)
        plot_errorbars = kw.pop('plot_errorbars', True)
        fig_title_suffix = kw.pop('fig_title_suffix', '')
        self.save_folder = kw.pop('save_folder', None)
        if self.save_folder is None:
            self.save_folder = self.folders[0]
            if not isinstance(self.save_folder, str):
                self.save_folder = self.save_folder[0]

        self.xlabel_RB = 'Number of Cliffords, m'
        self.ylabel_RB = r'Expectation value $\langle \sigma_z \rangle$'
        self.ylabel_corr = (r'Expectation value $\langle '
                            r'\sigma_z^{{\otimes {{{n}}} }} '
                            r'\rangle$'.format(n=len(self.qb_names)))
        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.line_colors = [i for i in default_color_cycle if
                            ('#2ca02c' not in i) and ('#d62728' not in i)]

        if make_fig_SRB_base:
            self.plot_SRB(
                plot_errorbars=plot_errorbars,
                save_fig=save_fig,
                fig_title_suffix=fig_title_suffix, **kw)

        if make_fig_cross_talk:
            self.plot_cross_talk(
                plot_errorbars=plot_errorbars,
                save_fig=save_fig,
                fig_title_suffix=fig_title_suffix, **kw)

        # if close_file:
        #     self.finish(**kw)

    def correlator_analysis(self, **kw):

        self.fit_res_dict_raw = {}
        self.data_dict_raw = {}
        self.msmt_strings = {}
        self.data_files = []
        self.epsilon_dict = {}
        self.var_data_dict = {}
        # needed for consistent colors for each qubit
        self.RO_channels = {}
        for folder in self.folders:
            self.folder = folder
            self.load_hdf5data(folder=self.folder, **kw)

            self.data_dict_raw = self.extract_data(
                two_qubits=True, **kw)
            self.add_analysis_datagroup_to_file()

            self.msmt_strings = self.measurementstring
            self.data_files += [self.data_file]

            self.n_cl = self.data_dict_raw['n_cl']
            self.nr_seeds = self.data_dict_raw['nr_seeds']

            # get RO channels and fit data
            for var_name, dset in self.data_dict_raw['data'].items():
                if var_name != 'corr':
                    instr_set = self.data_file[
                        'Instrument settings']
                    self.RO_channels[var_name] = int(instr_set[var_name].attrs[
                        'RO_acq_weight_function_I'])

                if var_name == 'corr':
                    d = 2**(len(self.qb_names))
                else:
                    d = 2

                self.fit_res_dict_raw[var_name] = \
                    self.fit_data(dset, self.n_cl, d=d,
                                  epsilon=self.var_data_dict[var_name],
                                  **kw)
                if self.epsilon is None:
                    self.epsilon_dict[var_name] = self.var_data_dict[var_name]
                else:
                    self.epsilon_dict[var_name] = self.epsilon


    def single_shot_analysis(self, **kw):

        find_empirical_variance = kw.get('find_empirical_variance', True)

        self.msmt_strings = []
        self.data_files = []
        self.fit_res_dict_raw = {}
        self.epsilon_dict = {}
        self.var_data_dict = {}
        for name in self.qb_names+['corr']:
            if find_empirical_variance:
                self.var_data_dict[name] = np.array([])
            else:
                self.var_data_dict[name] = None

        # needed for consistent colors for each qubit
        self.RO_channels = {}

        self.data_dict_raw = {}
        self.data_dict_raw['n_cl'] = np.array([], dtype=int)
        self.data_dict_raw['data'] = {}

        for var_name in self.qb_names+['corr']:
            self.data_dict_raw['data'][var_name] = np.array([])

        single_file = (len(self.folders) == 1)
        for i, folder in enumerate(self.folders):
            print(folder)
            self.folder = folder
            self.load_hdf5data(folder=self.folder, **kw)
            self.extract_data(two_qubits=False,
                              single_file=single_file, **kw)

            if i == 0:
                self.data_dict_raw['nr_seeds'] = self.nr_seeds
            if not single_file:
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

                self.data_dict_raw['n_cl'] = np.append(
                    self.data_dict_raw['n_cl'],
                    int(float(nr_cliffs_string[idx_underscore+1::])))
            self.add_analysis_datagroup_to_file()

            self.msmt_strings += [self.measurementstring]
            self.data_files += [self.data_file]

        # get RO channels and fit data
        self.n_cl = self.data_dict_raw['n_cl']
        for var_name, dset in self.data_dict_raw['data'].items():
            if var_name != 'corr':
                instr_set = self.data_file['Instrument settings']
                self.RO_channels[var_name] = int(instr_set[var_name].attrs[
                                        'RO_acq_weight_function_I'])

            if var_name == 'corr':
                d = 2**(len(self.qb_names))
            else:
                d = 2

            self.fit_res_dict_raw[var_name] = \
                self.fit_data(dset, self.n_cl, d=d,
                              epsilon=self.var_data_dict[var_name], **kw)
            if self.epsilon is None:
                self.epsilon_dict[var_name] = self.var_data_dict[var_name]
            else:
                self.epsilon_dict[var_name] = self.epsilon

    def extract_data(self, **kw):

        two_qubits = kw.pop('two_qubits', True)
        find_empirical_variance = kw.pop('find_empirical_variance', True)
        scaling_factor = kw.pop('scaling_factor', 1)
        print(scaling_factor)
        self.get_naming_and_values_2D()
        self.data[2::] = self.data[2:]/scaling_factor
        for i in range(len(self.measured_values)):
            self.measured_values[i] = self.measured_values[i]/scaling_factor

        if two_qubits:
            n_cl = np.unique(self.sweep_points_2D)
            data = {'n_cl': n_cl,
                    'nr_seeds': 0,
                    'data': {}}

            if self.use_cal_points:
                if self.cal_points is None:
                    self.cal_points = [[-2], [-1]]

                nr_sweep_pts = self.sweep_points.size #nr_seeds+NoCalPts
                nr_seeds = nr_sweep_pts - 2*len(self.cal_points[0])
                data['nr_seeds'] = nr_seeds

                data_temp = np.zeros(n_cl.size)

                for idx, qb_data in enumerate(self.data[2::]):
                    for i in range(len(n_cl)):
                        qb_data_rearranged = \
                            qb_data[i*nr_sweep_pts:i*nr_sweep_pts+nr_sweep_pts]

                        corr_data = a_tools.normalize_data_v3(
                            qb_data_rearranged,
                            self.cal_points[0],
                            self.cal_points[1])[0:self.cal_points[0][0]]

                        data_temp[i] = np.mean(corr_data)

                    if idx == len(self.data[2::])-1:
                        data['data']['corr'] = deepcopy(data_temp)
                    else:
                        data['data'][self.qb_names[idx]] = deepcopy(data_temp)

            else:
                print('here')
                nr_seeds = self.sweep_points.size
                data['nr_seeds'] = nr_seeds

                qb0_raw = self.measured_values[0]
                qb1_raw = self.measured_values[1]
                corr_raw = self.measured_values[2]

                qb0 = np.zeros(n_cl.size)
                qb1 = np.zeros(n_cl.size)
                corr = np.zeros(n_cl.size)

                if find_empirical_variance:
                    self.var_data_dict[self.qb_names[0]] = np.array([])
                    self.var_data_dict[self.qb_names[1]] = np.array([])
                    self.var_data_dict['corr'] = np.array([])
                else:
                    self.var_data_dict[self.qb_names[0]] = None
                    self.var_data_dict[self.qb_names[1]] = None
                    self.var_data_dict['corr'] = None

                for i in range(n_cl.size):
                    y = [qb0_raw[j][i] for j in range(nr_seeds)]
                    qb0[i] = np.mean(y)
                    if find_empirical_variance:
                        y = 2*(1 - np.asarray(y)) - 1
                        self.var_data_dict[self.qb_names[0]] = np.append(
                            self.var_data_dict[self.qb_names[0]], np.std(y))

                    y = [qb1_raw[j][i] for j in range(nr_seeds)]
                    qb1[i] = np.mean(y)
                    if find_empirical_variance:
                        y = 2*(1 - np.asarray(y)) - 1
                        self.var_data_dict[self.qb_names[1]] =np.append(
                            self.var_data_dict[self.qb_names[1]], np.std(y))

                    y = [corr_raw[j][i] for j in range(nr_seeds)]
                    corr[i] = np.mean(y)
                    if find_empirical_variance:
                        y = 2*np.asarray(y) - 1
                        self.var_data_dict['corr'] = np.append(
                            self.var_data_dict['corr'], np.std(y))

                data['data'][self.qb_names[0]] = 2*(1 - qb0) - 1
                data['data'][self.qb_names[1]] = 2*(1 - qb1) - 1
                data['data']['corr'] = 2*corr - 1

            return data
        else:
            single_file = kw.pop('single_file', True)
            measurement_data = deepcopy(self.data[2::])
            raw_correl_data = deepcopy(measurement_data[0])

            for col in range(measurement_data.shape[1]):
                raw_correl_data[col] = 1 - np.count_nonzero(
                    measurement_data[:, col]) % 2

            if single_file:
                n_cl = self.sweep_points_2D
                self.nr_seeds = np.unique(self.sweep_points).size
                assert (self.data.shape[1] % (self.nr_seeds * n_cl.size) == 0)
                nr_shots = self.data.shape[1] // (self.nr_seeds * n_cl.size)

                self.data_dict_raw['n_cl'] = n_cl
                self.data_dict_raw['nr_shots'] = nr_shots

                for cliff_idx in range(n_cl.size):
                    measurement_data_subset = \
                        measurement_data[:,
                        cliff_idx*nr_shots*self.nr_seeds:
                        (cliff_idx+1)*nr_shots*self.nr_seeds]
                    raw_correl_data_subset = \
                        raw_correl_data[cliff_idx*nr_shots*self.nr_seeds:
                                        (cliff_idx+1)*nr_shots*self.nr_seeds]
                    mean_data_array = np.zeros(len(self.qb_names)+1)

                    if find_empirical_variance: # here for corr data only
                        y = np.array([])
                        for j in range(self.nr_seeds):
                            y = np.append(y, np.mean(
                                raw_correl_data_subset[j::self.nr_seeds]))
                        y = 2*y-1
                        self.var_data_dict['corr'] = np.append(
                            self.var_data_dict['corr'], np.std(y))

                    mean_data_array[-1] = np.mean(raw_correl_data_subset)

                    # get averaged results for each qubit measurement
                    for i in np.arange(len(self.qb_names)):
                        mean_data_array[i] = 1 - np.mean(
                            measurement_data_subset[i])

                        if find_empirical_variance:
                            dcol = 1 - measurement_data_subset[i]
                            y = np.array([])
                            for j in range(self.nr_seeds):
                                y = np.append(y, np.mean(
                                    dcol[j::self.nr_seeds]))
                            y = 2*y-1
                            self.var_data_dict[self.qb_names[i]] = np.append(
                                self.var_data_dict[self.qb_names[i]], np.std(y))

                    for var_name, data_point in zip(self.qb_names+['corr'],
                                                    mean_data_array):
                        self.data_dict_raw['data'][var_name] = np.append(
                            self.data_dict_raw['data'][var_name],
                            2*data_point-1) # <sigma_z>
                            # data_point) # prob of |1>

            else:
                self.nr_seeds = self.sweep_points.size
                # self.nr_shots = self.sweep_points_2D.size
                mean_data_array = np.zeros(len(self.qb_names)+1)

                if find_empirical_variance: # here for corr data only
                    y = np.array([])
                    for j in range(self.nr_seeds):
                        y = np.append(y, np.mean(
                            raw_correl_data[j::self.nr_seeds]))
                    y = 2*y-1
                    self.var_data_dict['corr'] = np.append(
                            self.var_data_dict['corr'], np.std(y))

                mean_data_array[-1] = np.mean(raw_correl_data)

                # get averaged results for each qubit measurement
                for i in np.arange(len(self.qb_names)):
                    mean_data_array[i] = 1 - np.mean(measurement_data[i])

                    if find_empirical_variance:
                        dcol = 1 - measurement_data[i]
                        y = np.array([])
                        for j in range(self.nr_seeds):
                            y = np.append(y, np.mean(dcol[j::self.nr_seeds]))
                        y = 2*y-1
                        self.var_data_dict[self.qb_names[i]] = np.append(
                            self.var_data_dict[self.qb_names[i]], np.std(y))

                for var_name, data_point in zip(self.qb_names+['corr'],
                                                mean_data_array):
                    self.data_dict_raw['data'][var_name] = np.append(
                        self.data_dict_raw['data'][var_name],
                        2*data_point-1)

            return

    def fit_data(self, data, numCliff, print_fit_results=False,
                 show_guess=False,  plot_results=False, **kw):

        guess_pars_dict = kw.pop('guess_pars_dict', {})
        d = kw.pop('d', 2)
        print('d in fit_data ', d)

        RBModel = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
        RBModel.set_param_hint('Amplitude',
                               min=0, max=1,
                               value=guess_pars_dict.get('Amplitude', 0.9))
        RBModel.set_param_hint('p',
                               value=guess_pars_dict.get('p', 0.95),
                               min=0, max=1)
        RBModel.set_param_hint('offset',
                               value=guess_pars_dict.get('offset', 0),
                               vary=True)
        # From Magesan et al., Scalable and Robust Randomized Benchmarking
        # of Quantum Processes
        RBModel.set_param_hint('fidelity_per_Clifford',  # vary=False,
                               expr='1-(({}-1)*(1-p)/{})'.format(d, d))
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
        self.conf_level = kw.pop('conf_level', 0.68)

        if kw.pop('single_fit', False):
            print('single fit')
            fit_res = RBModel.fit(data, numCliff=numCliff, params=params)
            self.epsilon = None
        else:
            self.epsilon = kw.pop('epsilon', None)
            if self.epsilon is None:
                print('old fit')
                # Run once to get an estimate for the error per Clifford
                fit_res = RBModel.fit(data, numCliff=numCliff, params=params)

                # Use the found error per Clifford to standard errors for the data
                # points fro Helsen et al. (2017)
                epsilon_guess = kw.pop('epsilon_guess', 0.01)
                epsilon = calculate_confidence_intervals(
                    nr_seeds=self.nr_seeds,
                    nr_cliffords=self.n_cl,
                    depolariz_param=fit_res.best_values['p'],
                    conf_level=self.conf_level,
                    epsilon_guess=epsilon_guess, d=d)

                self.epsilon = epsilon
                # Run fit again with scale_covar=False, and weights = 1/epsilon

                # if an entry in epsilon_sqrd is 0, replace it with half the minimum
                # value in the epsilon_sqrd array
                idxs = np.where(epsilon==0)[0]
                epsilon[idxs] = min([eps for eps in epsilon if eps!=0])/2
                print(epsilon)
                fit_res = RBModel.fit(data, numCliff=numCliff, params=params,
                                      scale_covar=False, weights=1/epsilon)

            else:
                print('empirical variance fit')
                self.conf_level = kw.pop('conf_level', 0.68)
                fit_res = RBModel.fit(data, numCliff=numCliff, params=params,
                                      scale_covar=False, weights=1/self.epsilon)

        return fit_res

    def find_multi_qubit_error(self, fit_results_dict=None, **kw):
        print('in multi qubit error')
        if fit_results_dict is None:
            fit_results_dict = self.fit_res_dict

        n = kw.pop('n', None)
        assume_uncorrelated = kw.pop('assume_uncorrelated', False)
        if n is None:
            n = len(self.qb_names)

        # dimension of full n qubit Hilbert space
        d = 2**n

        self.total_error = {}

        if assume_uncorrelated:
            total_depolariz_param = get_total_uncorr_depolariz_param(
                qb_names=self.qb_names, fit_results_dict=fit_results_dict)
        else:
            if n == 2:
                total_depolariz_param = get_total_corr_depolariz_param(
                    qb_names=self.qb_names, fit_results_dict=fit_results_dict)
            else:
                find_empirical_variance = kw.pop('find_empirical_variance',
                                                 True)
                print(find_empirical_variance)
                correl_data_dict = {}
                correl_variance_dict = {}
                # find depolarization params from all possible combinations of
                # qubit pairs/triplets from the single shots
                if len(self.folders) > 1:
                    for folder in self.folders:
                        self.load_hdf5data(folder=folder, **kw)
                        self.get_naming_and_values_2D()
                        measurement_data = deepcopy(self.data[2::])
                        self.fit_data_subsets(
                            n, measurement_data,
                            correl_data_dict, correl_variance_dict,
                            find_empirical_variance=find_empirical_variance)
                else:
                    self.load_hdf5data(folder=self.folders[0], **kw)
                    self.get_naming_and_values_2D()
                    measurement_data = deepcopy(self.data[2::])
                    for cliff_idx in range(self.data_dict['n_cl'].size):
                        measurement_data_subset = \
                            measurement_data[:,
                            cliff_idx*self.data_dict['nr_shots']*self.nr_seeds:
                            (cliff_idx+1)*self.data_dict[
                                'nr_shots']*self.nr_seeds]
                        self.fit_data_subsets(
                            n, measurement_data_subset,
                            correl_data_dict, correl_variance_dict,
                            find_empirical_variance=find_empirical_variance)

                # fit each data set and get depolarization params
                depolariz_params_dict = {}
                for key, dset in correl_data_dict.items():
                    print('correl_variance_dict['+str(key)+'] ',
                          correl_variance_dict[key])
                    d_subspace = 2**(len(key))
                    depolariz_params_dict[key] = {}
                    fit_res = self.fit_data(dset,
                                            self.data_dict['n_cl'],
                                            d=d_subspace,
                                            epsilon=correl_variance_dict[key])
                    depolariz_params_dict[key]['val'] = \
                        fit_res.best_values['p']
                    depolariz_params_dict[key]['stderr'] = \
                        fit_res.params['p'].stderr

                self.depolariz_params_dict_mq_err = depolariz_params_dict
                self.correlated_data_dict_mq_err = correl_data_dict
                self.correl_variance_dict_mq_err = correl_variance_dict
                # calculate total depolarization param
                from pprint import pprint
                pprint(depolariz_params_dict)
                total_depolariz_param = get_total_corr_depolariz_param(
                    qb_names=self.qb_names, fit_results_dict=fit_results_dict,
                    subspace_depolariz_params_dict=depolariz_params_dict, **kw)

        print('d in multi-qubit error ', d)
        self.total_depolariz_param = total_depolariz_param
        self.total_error['val'] = (d-1) * \
                                  (1-total_depolariz_param['val'])/d
        self.total_error['stderr'] = np.abs((d-1) * \
                                            total_depolariz_param[
                                                'stderr']/d)

    def fit_data_subsets(self, nr_qubits, measurement_data,
                         correl_data_dict, correl_variance_dict,
                         find_empirical_variance=True):

        for qb_combs in np.arange(2, nr_qubits):
            # get list of tuples representing all combinations of
            # qb idxs
            idxs_list = list(itertools.combinations(range(nr_qubits),
                                                    qb_combs))
            raw_correl_data = np.zeros(len(measurement_data[0]))

            # calculate correlators for each index tuple by
            # multiplying all shots
            for idxs_combs in idxs_list:
                subspace_data = measurement_data[
                                np.asarray(idxs_combs), :]
                key = ''.join([str(i) for i in idxs_combs])
                if key not in correl_data_dict:
                    correl_data_dict[key] = np.array([])
                if (key not in correl_variance_dict):
                    if find_empirical_variance:
                        correl_variance_dict[key] = np.array([])
                    else:
                        correl_variance_dict[key] = None

                for col in range(subspace_data.shape[1]):
                    raw_correl_data[col] = 1 - np.count_nonzero(
                        subspace_data[:, col]) % 2

                if find_empirical_variance:
                    # if we use empirical variance, then we need to
                    # find the std of the nr_seeds distribution
                    y = np.array([])
                    for j in range(self.nr_seeds):
                        y = np.append(
                            y,
                            np.mean(raw_correl_data[j::self.nr_seeds]))
                    y = 2*y-1
                    correl_variance_dict[key] = np.append(
                        correl_variance_dict[key], np.std(y))

                #get <sigma_z^{\otimes s}>
                mean_correl_data = 2*np.mean(raw_correl_data) - 1
                correl_data_dict[key] = np.append(
                    correl_data_dict[key],
                    deepcopy(mean_correl_data))


    def plot_SRB(self, save_fig=True, fig_title_suffix=None, **kw):

        show_SRB_base = kw.pop('show_SRB_base', False)
        plot_T1_lim_base = kw.pop('plot_T1_lim_base', True)
        close_fig = kw.pop('close_fig', True)

        for var_idx, var_name in enumerate(self.data_dict['data']):

            fig, ax = plt.subplots(figsize=(
                fig_size_dim, fig_size_dim/golden_ratio))

            fig_title = ('{}_SRB_{}-{}seeds'.format(
                var_name,
                self.gate_decomp,
                int(self.data_dict['nr_seeds'])))

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
                plot_T1_lim_temp = False
                horizontal_alignment='right'
                T1 = None
                T2 = None
                pulse_length = None
            else:
                subplot_title = \
                    r'$\langle \sigma_{{z,{qb0}}} ' \
                    r'\rangle$'.format(qb0=var_name)
                ylabel = self.ylabel_RB
                plot_T1_lim_temp = plot_T1_lim_base
                horizontal_alignment='right'
                T1 = self.T1s[var_name]
                T2 = self.T2s[var_name]
                pulse_length = self.pulse_lengths[var_name]

            line = plotting_function(x=self.data_dict['n_cl'],
                                     y=self.data_dict['data'][
                                         var_name],
                                     fig=fig,
                                     ax=ax,
                                     marker='o',
                                     xlabel=self.xlabel_RB,
                                     ylabel=ylabel,
                                     return_line=True)

            if self.fit_res_dict[var_name] is None:
                pass
            else:
                self.add_errorbars_T1_lim_simultaneous_RB(
                    ax,
                    line=line,
                    var_name=var_name,
                    T1=T1, T2=T2, pulse_length=pulse_length,
                    plot_T1_lim=plot_T1_lim_temp,
                    horizontal_alignment=horizontal_alignment, **kw)

            ax.set_title(subplot_title)

            fig.text(0.5, 1.05, fig_title, fontsize=font_size,
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=ax.transAxes)

            # show figure
            if show_SRB_base:
                plt.show()

            # Save figure
            if save_fig:
                if fig_title_suffix is not None:
                    fig_title += '_' + fig_title_suffix
                filename = os.path.abspath(os.path.join(
                    self.save_folder, fig_title+'.png'))
                fig.savefig(filename, format='png',
                                 bbox_inches='tight')

            if close_fig:
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
        for qb_name in self.qb_names:
            legend_labels.extend(
                [r'$\langle \sigma_{{z,{qb}}} \rangle$'.format(
            qb=qb_name)])
        legend_labels.extend(
            [r'$\langle '
             r'\sigma_z^{{\otimes {{{n}}} }} \rangle$'.format(n=n),
             r'$\langle \sigma_z '
             r'\rangle ^{{\otimes {{{n}}} }}$'.format(n=n)])

        for var_idx, var_name in enumerate(self.data_dict['data']):

            if var_name=='corr':
                #set color of the corr line to a darker green
                line_color = 'g'
            else:
                line_color = self.line_colors[self.RO_channels[var_name]]

            line = plotting_function(x=self.data_dict['n_cl'],
                                     y=self.data_dict['data'][
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
                var_name=var_name,
                plot_T1_lim=plot_T1_lim_cross_talk,
                add_textbox=False,
                plot_errorbars=plot_errorbars,
                **kw)

            if self.epsilon_dict[var_name] is None:
                plot_errorbars = False

        # plot the product \sigma_qb2 * \signam_qb7 which should equal
        # \sigma_corr in the ideal case
        self.find_and_plot_alpha_product(ax=ax, legend_labels=legend_labels,
                                         n=n, **kw)

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
            self.add_textbox_cross_talk(ax=ax, **kw)

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

    def find_and_plot_alpha_product(self, ax, legend_labels, n=None, **kw):
        # plot the product \sigma_qb2 * \signam_qb7 which should equal
        # \sigma_corr in the ideal case

        add_correction = kw.pop('add_correction', False)

        if add_correction:
            data_dict = self.data_dict
            fit_res_dict = self.fit_res_dict
        else:
            data_dict = self.data_dict_raw
            fit_res_dict = self.fit_res_dict_raw
            print('use unscaled data')

        A = fit_res_dict['corr'].best_values['Amplitude']
        x_fine = np.linspace(data_dict['n_cl'][0],
                             data_dict['n_cl'][-1], 100)
        ax.plot(x_fine, A*self.alpha_product**x_fine, 'm--',
                label=legend_labels[-1], linewidth=line_width+2, dashes=(2, 2))
        ax.plot(data_dict['n_cl'],
                A*self.alpha_product**data_dict['n_cl'], 'mo')

    def add_textbox_cross_talk(self, ax, **kw):
        # pring infidelities
        textstr = ''
        for qb_name in self.qb_names:
            textstr += ('$r_{{{qb}}}$'.format(qb=qb_name) +
                            ' = {:.2f} $\pm$ ({:.1f})%'.format(
                        self.infidelities[qb_name]['val']*100,
                        self.infidelities[qb_name]['stderr']*100)
                        + '\n')

        textstr += ('$r_{corr}$' + ' = {:.2f} $\pm$ ({:.1f})%'.format(
                self.infidelities['corr']['val']*100,
                self.infidelities['corr']['stderr']*100))

        x_position = 0.2
        ax.text(x_position, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=font_size)

        # Print alphas
        textstr = ''
        for qb_name in self.qb_names:
            textstr += (r'$\alpha_{{{qb}}}$'.format(qb=qb_name) +
                        ' = {:.1f} $\pm$ ({:.1f})%'.format(
                            self.fit_res_dict[qb_name].best_values['p']*100,
                            self.fit_res_dict[qb_name].params['p'].stderr*100)
                        + '\n')

        textstr += (r'$\alpha_{corr}$' +
                    ' = {:.1f} $\pm$ ({:.1f})%'.format(
                        self.fit_res_dict['corr'].best_values['p']*100,
                        self.fit_res_dict['corr'].params['p'].stderr*100))

        textstr += ('\n' + '  ' + r'$\delta \alpha$' +
                    ' = {:.1f} $\pm$ ({:.1f})%'.format(
                        self.delta_alpha['val']*100,
                        self.delta_alpha['stderr']*100))

        ax.text(x_position+0.4, 0.975, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=font_size)

    def add_errorbars_T1_lim_simultaneous_RB(self, ax,
                                             var_name,
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

        data = self.data_dict['data'][var_name]
        n_cl = self.data_dict['n_cl']
        fit_res = self.fit_res_dict[var_name]
        epsilon = self.epsilon_dict[var_name]

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
            if epsilon is not None:
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
            x_fine = np.linspace(n_cl[0], n_cl[-1], 1000)
            best_fit = fit_mods.RandomizedBenchmarkingDecay(
                x_fine, **fit_res.best_values)
            ax.plot(x_fine, best_fit, color=c)


        textstr = \
            ('$r_{perCl,' + var_name + '}$'+' = '
                    '{:.6g} $\pm$ ({:.4g})% \n'.format(
                self.infidelities[var_name]['val']*100,
                self.infidelities[var_name]['stderr']*100))

        # Here we add the line corresponding to T1 limited fidelity
        F_T1 = None
        if pulse_length is not None:
            if T1 is not None and T2 is not None and plot_T1_lim:
                F_T1, p_T1 = calc_T1_limited_fidelity(
                    T1, T2, pulse_length)
                x_fine = np.linspace(n_cl[0], n_cl[-1], 1000)
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

def load_T1_T2_pulse_length(folders, qb_names,
                            T1s=None, T2s=None, pulse_lengths=None):
    """
    Loads T1, T2, and DRAG pulse length from folders for all the qubits
    in qb_names.

    Args:
        folders (list): list of folders
        qb_names (list): list of qubit names used in the experiment
        T1s (dict): an already-existing T1s_dict
        T2s (dict): an already-existing T2s_dict
        pulse_lengths (dict): an already-existing pulse_lengths_dict
    Returns:
        T1, T2, pulse_length dicts of the form:
            {'qb_name': parameter_value}

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
    for folder in folders:
        T1s = {}
        T2s = {}
        pulse_lengths = {}
        if type(folder) != str:
            if len(folder)>1:
                print('Qubit coherence times are taken from the first folder.')
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
                T1s[qb] = float(instr_set[qb].attrs['T1'])
            except Exception:
                print('Could not load T1 for {}.'.format(qb))
            try:
                T2s[qb] = float(instr_set[qb].attrs['T2'])
            except Exception:
                print('Could not load T2 for {}.'.format(qb))
            try:
                pulse_lengths[qb] = \
                    float(instr_set[qb].attrs['nr_sigma']) * \
                    float(instr_set[qb].attrs['gauss_sigma'])
            except Exception:
                print('Could not load pulse_length for {}.'.format(qb))

    if count == len(folders):
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
                                   conf_level=0.68, depolariz_param=1,
                                   epsilon_guess=0.01, d=2):

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
    infidelity = (d-1)*(1-depolariz_param)/d

    for n_cl in nr_cliffords:
        if n_cl == 0:
            epsilon.append(0)
        else:
            # if np.abs(n_cl*infidelity - 1) < 1:
            #     epsilon_guess = 0.01
            # else:
            #     epsilon_guess = 0.1
            # print('epsilon_guess ', epsilon_guess)
            if d==2:
                # print('1 qubit')
                # print('n_cl ', n_cl)
                # if n_cl < 20:
                V_short_n_cl = (13*n_cl*infidelity**2)/2
                    # V = V_short_n_cl
                # else:
                V_long_n_cl = 7*infidelity/2
                    # V = V_long_n_cl
                V = min(V_short_n_cl, V_long_n_cl)
                # V = (6 + 0.25*
                #      ((1-depolariz_param**(2*n_cl))/
                #       (1-depolariz_param**2)))*infidelity**2
            else:
                # print('n qubits')
                # print('n_cl ', n_cl)
                # if np.abs(n_cl*infidelity - 1) < 1:#n_cl<20:
                    # V_short_n_cl = (2*(d+1)/(d-1) + 0.25*(-2+d**2)/((d-1)**2))\
                    #                *n_cl*(infidelity**2)
                V_short_n_cl = \
                    (0.25*(-2+d**2)/((d-1)**2)) * (infidelity**2) + \
                    (0.5*n_cl*(n_cl-1)*(d**2)/((d-1)**2)) * (infidelity**2)
                    # V = V_short_n_cl
                # else:
                    # V = 2*((d+1)/(d-1))*(infidelity**2) + \
                    #     0.25*((1-depolariz_param**(2*n_cl))/
                    #           (1-depolariz_param**2))*((-2+d**2)/((d-1)**2))*\
                    #     infidelity**2
                V1 = 0.25*((-2+d**2)/((d-1)**2))*n_cl*(infidelity**2) * \
                    depolariz_param**(n_cl-1) + ((d/(d-1))**2) * \
                    (infidelity**2)*( (1+(n_cl-1) *
                        (depolariz_param**(2*n_cl)) -
                        n_cl*(depolariz_param**(2*n_cl-2))) /
                        (1-depolariz_param**2)**2 )
                V = min(V1, V_short_n_cl)
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
        stderr = ((d-1)/d)*np.sqrt((p_gate_stderr/p0)**2 +
                          (p0_stderr*p_gate/(p0**2))**2)
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

    return (rc>=(rc-E) and rc<=(rc+E)), E

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

    ax.set_ylim([-0.15, 1.15])
    majorLocator = MultipleLocator(0.2)
    majorFormatter = FormatStrFormatter('%1.1f')
    minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)

    fig.tight_layout()

    if kw.get('return_line', False):
        return line
    else:
        return

def get_total_uncorr_depolariz_param(qb_names, fit_results_dict,
                                     total_uncorr_depolariz_param=None):

    # Assume p_ij..z = p_i*p_j*...*p_z
    # Example: if n = 4 qubits:
    # p_total = 3*p_0 + 3*p_1 + 3*p_2 + 3*p_3 +
    #           3^2*p_0*p_1 + ... + 3^2*p_2*p_3 +
    #           3^3*p_0*p_1*p_2 + ... + 3^3*p_1*p_2*p_3 +
    #           3^4*p_0*p_1*p_2*p_3
    # where all the p's must be obtained from individual fits of the
    # shot-by-shot products from the data for the respective qubits.

    if total_uncorr_depolariz_param is None:
        total_uncorr_depolariz_param = {}
        total_uncorr_depolariz_param['val'] = 0
        total_uncorr_depolariz_param['stderr'] = 0

    n = len(qb_names)

    prefactors = []
    for i in range(n+1)[1::]:
        k = len(list(itertools.combinations(n*[''], i)))
        prefactors += k*[3**i]

    if n == 2:
        for idx, var_name in enumerate((qb_names+['corr'])):
            # print(var_name)
            if var_name == 'corr':
                val = 1
                stderr = 0
                for qb_name in qb_names:
                    val *= fit_results_dict[qb_name].best_values['p']
                    # calculate stderr using propagation of errors
                    std_err_product_squared = 1
                    std_err_product_squared *= \
                        (fit_results_dict[qb_name].params[
                             'p'].stderr)**2
                    for other_qb_names in qb_names:
                        if other_qb_names != qb_name:
                            std_err_product_squared *= \
                                (fit_results_dict[
                                     other_qb_names].best_values[
                                     'p'])**2
                    stderr += std_err_product_squared
                # stderr = np.sqrt(stderr)
            else:
                val = fit_results_dict[var_name].best_values['p']
                stderr = fit_results_dict[var_name].params['p'].stderr**2

            total_uncorr_depolariz_param['val'] += prefactors[idx] * val
            total_uncorr_depolariz_param['stderr'] += \
                (prefactors[idx]**2) * stderr

        total_uncorr_depolariz_param['val'] /= np.sum(prefactors)
        total_uncorr_depolariz_param['stderr'] = \
            np.sqrt(total_uncorr_depolariz_param['stderr'])/np.sum(prefactors)

    else:
        list_of_idxs = list(range(n))
        for qb_combs in np.arange(2, n+1):
            # get list of tuples representing all combinations of qb idxs
            list_of_idxs += list(itertools.combinations(range(n),
                                                     qb_combs))

        for idx, qb_idxs in enumerate(list_of_idxs):
            if not isinstance(qb_idxs, tuple):
                # single qb RB fits
                qb_name = qb_names[qb_idxs]
                total_uncorr_depolariz_param['val'] += \
                    prefactors[idx] * fit_results_dict[
                        qb_name].best_values['p']
                total_uncorr_depolariz_param['stderr'] += \
                    (prefactors[idx] * fit_results_dict[
                        qb_name].params['p'].stderr)**2
            else:
                # multi qb RB fits
                val = 1
                stderr = 0
                for qb_idx in qb_idxs:
                    qb_name = qb_names[qb_idx]
                    val *= fit_results_dict[qb_name].best_values['p']
                    # calculate stderr using propagation of errors
                    std_err_product_squared = 1
                    std_err_product_squared *= \
                        (fit_results_dict[qb_name].params[
                             'p'].stderr)**2
                    for other_qb_names in [qb_names[g] for
                                           g in qb_idxs]:
                        if other_qb_names != qb_name:
                            std_err_product_squared *= \
                                (fit_results_dict[
                                     other_qb_names].best_values[
                                     'p'])**2

                    stderr += std_err_product_squared

                total_uncorr_depolariz_param['val'] += prefactors[idx] * val
                total_uncorr_depolariz_param['stderr'] += \
                    (prefactors[idx]**2) * stderr

        total_uncorr_depolariz_param['val'] /= np.sum(prefactors)
        total_uncorr_depolariz_param['stderr'] = \
            np.sqrt(total_uncorr_depolariz_param['stderr'])/np.sum(prefactors)

    return total_uncorr_depolariz_param

def get_multi_qb_error_from_single_qb_RB(qb_names, d,
                                         single_qb_RB_fit_results_dict):

    # Total r_n assuming we have neither crosstalk nor correlated errors.
    # Example: if n = 4 qubits:
    # p_total = 3*p_0 + 3*p_1 + 3*p_2 + 3*p_3 +
    #           3^2*p_0*p_1 + ... + 3^2*p_2*p_3 +
    #           3^3*p_0*p_1*p_2 + ... + 3^3*p_1*p_2*p_3 +
    #           3^4*p_0*p_1*p_2*p_3
    # r_n = (d-1)(1-p_total)/d, d=2^n

    # where the p's are obtained from fits to the SINGLE QUBIT RB data for
    # each qubit.

    total_error = {}
    total_depolariz_param = get_total_uncorr_depolariz_param(
        qb_names=qb_names, fit_results_dict=single_qb_RB_fit_results_dict)

    total_error['val'] = (d-1) * (1-total_depolariz_param['val'])/d
    total_error['stderr'] = np.abs((d-1) * total_depolariz_param['stderr']/d)

    return total_error

def get_total_corr_depolariz_param(qb_names, fit_results_dict,
                                   total_corr_depolariz_param=None,
                                   subspace_depolariz_params_dict=None, **kw):

    """
    Args:
        qb_names (list): list or tuple with names of qubits used in experiment
        fit_results_dict (dict): fit results from fits to each single qubit
            data and to the n-qubit correlator
        total_corr_depolariz_param (dict): dict with keys 'val' and 'stderr'
             for the total depolarization parameter (p_total in description
             below); in case you want to append to an existing one; if None,
             it will generate a new dict
        subspace_depolariz_params_dict (dict): dict of the form
             {qb_idxs_str: {'val': p_value, 'stderr': p_error}}, with the
             depolarizing parameters obtained from fits to all combinations of
             s<n subsets from the n qubits.
             Example: if n=3 (qb0, qb1, qb2), qb_idxs_str \in {'01', '02', '12'}

    Keyword args:
        correlated_subspace (int): highest degree of correlated errors
            counterintuitive for correlated_subspace=1: corresponds to all
            degrees of correlations

    Returns:
        total_corr_depolariz_param (dict): dict with keys 'val' and 'stderr'
            for the total depolarization parameter (p_total in description
            below)
    """
    # Example: if n = 4 qubits and correlated_subspace = 1 (assume all degrees
    # of correlations up to n qubits are present):
    # p_total = 3*p_0 + 3*p_1 + 3*p_2 + 3*p_3 +
    #           3^2*p_01 + 3^2*p_02 + ... + 3^2*p_23 +
    #           3^3*p_012 + ... + 3^3*p_123 +
    #           3^4*p_0123

    # If n = 4 qubits and correlated_subspace = 2 (assume single qubit errors
    # and only 2-qubit correlated errors are present, and find upper bound):
    # p_total = 3*p_0 + 3*p_1 + 3*p_2 + 3*p_3 +
    #           3^2*p_01 + 3^2*p_02 + ... + 3^2*p_23 +
    #           3^3*max(p_0*p_12, p_1*p_02, p_2*p_01, p_0*p_1*p_2) + ... +
    #               3^3*max(p_1*p_23, p_2*p_13, p_3*p_12, p_1*p_2*p_3) +
    #           3^4*max(p_01*p_23, p_02*p_13, p_03*p_12, p_0*p_1*p_23,
    #               p_1*p_0*p_23, p_2*p_0*p_13, p_3*p_0*p_12, p_0*p_1*p_2*p_3)

    # If n = 4 qubits and correlated_subspace = 3  (assume single qubit errors
    # and only 2- and 3-qubit correlated errors are present,
    # and find upper bound):
    # p_total = 3*p_0 + 3*p_1 + 3*p_2 + 3*p_3 +
    #           3^2*p_01 + 3^2*p_02 + ... + 3^2*p_23 +
    #           3^3*p_012 + ... + 3^3*p_123 +
    #           3^4*max(p_0*p_123, p_1*p_023, p_2*p_013, p_3*p_012,
    #               p_0*p_1*p_23, p_1*p_0*p_23, p_2*p_0*p_13,
    #               p_3*p_0*p_12, p_01*p_23, p_02*p_13, p_03*p_12,
    #               p_0*p_1*p_2*p_3)

    # All the p's must be obtained from individual fits of the
    # shot-by-shot products from the data for the respective qubits.

    if total_corr_depolariz_param is None:
        total_corr_depolariz_param  = {}
        total_corr_depolariz_param ['val'] = 0
        total_corr_depolariz_param ['stderr'] = 0

    n = len(qb_names)

    prefactors = []
    for i in range(n+1)[1::]:
        k = len(list(itertools.combinations(n*[''], i)))
        prefactors += k*[3**i]

    if n == 2:
        for idx, var_name in enumerate((qb_names+['corr'])):
            total_corr_depolariz_param['val'] += \
                prefactors[idx] * fit_results_dict[
                    var_name].best_values['p']
            total_corr_depolariz_param['stderr'] += \
                (prefactors[idx] * fit_results_dict[
                    var_name].params['p'].stderr)**2
        total_corr_depolariz_param['val'] /= np.sum(prefactors)
        total_corr_depolariz_param['stderr'] = \
            np.sqrt(total_corr_depolariz_param ['stderr'])/np.sum(prefactors)

    else:
        if subspace_depolariz_params_dict is None:
            correl_subspace_keys = []
        else:
            correl_subspace_keys = list(subspace_depolariz_params_dict)

        correlated_subspace = kw.pop('correlated_subspace', 1)
        print('correlated_subspace ', correlated_subspace)
        single_qb_idxs = [str(i) for i in range(n)]
        for idx, var_name in enumerate((
                        single_qb_idxs + correl_subspace_keys + ['corr'])):
            if correlated_subspace == 1 or len(var_name) <= correlated_subspace:

                # if correlated_subspace == 1, do not decompose anything -> look
                # at all correlators
                try:
                    if len(var_name) == 1:
                        var_name = qb_names[int(var_name)]
                    print('if try: ', var_name)
                    # take the depolarization params from the single qb RB
                    print(fit_results_dict[var_name].best_values['p'])
                    total_corr_depolariz_param['val'] += prefactors[idx] * \
                                                    fit_results_dict[
                                                        var_name]. \
                                                        best_values['p']
                    total_corr_depolariz_param['stderr'] += \
                        (prefactors[idx] * fit_results_dict[
                            var_name].params['p'].stderr)**2
                except KeyError:
                    # additionally:
                    # if correlated_subspace == 2, do not decompose the
                    # depolarization params of the 2 qb subspace;
                    # if correlated_subspace == 3, do not decompose the
                    # depolarization params of the 3 qb subspace; etc.
                    print('if except: ', var_name)
                    print(subspace_depolariz_params_dict[var_name]['val'])
                    total_corr_depolariz_param['val'] += \
                        prefactors[idx] * subspace_depolariz_params_dict[
                            var_name]['val']
                    total_corr_depolariz_param['stderr'] += \
                        (prefactors[idx] * subspace_depolariz_params_dict[
                            var_name]['stderr'])**2
            else:
                if var_name == 'corr':
                    # make var_name = '123..n'
                    var_name = ''.join([str(i) for i in range(n)])

                print('else: ', var_name)

                # get indices of results that must be summed to get one term
                # in the total_corr_depolariz_param dict
                smallest_correlated_subspace = \
                    kw.pop('smallest_correlated_subspace ', 1)
                keys = get_keys(corr_string=var_name,
                                corr_s=correlated_subspace,
                                smallest_corr_s=smallest_correlated_subspace)
                print('keys ', keys)
                print('len keys ', len(keys))

                depolariz_param_term = 0
                depolariz_param_term_stderr = 0

                max_alpha_product = 0
                for keys_tuple in keys:
                    alpha_temp = []
                    alpha_temp_stderr = []
                    for k in keys_tuple:
                        if len(k) == 1:
                            alpha_temp += [fit_results_dict[
                                qb_names[int(k)]].best_values['p']]
                            alpha_temp_stderr += [fit_results_dict[
                                       qb_names[int(k)]].params['p'].stderr]
                        else:
                            alpha_temp += \
                                [subspace_depolariz_params_dict[k]['val']]
                            alpha_temp_stderr += \
                                [subspace_depolariz_params_dict[k]['stderr']]
                    print(alpha_temp)
                    print(alpha_temp_stderr)
                    alpha_product_temp = np.product(np.asarray(alpha_temp))
                    print(alpha_product_temp)

                    if alpha_product_temp > max_alpha_product:
                        print(keys_tuple)
                        max_alpha_product = alpha_product_temp
                        # get std error
                        std_err_term = 0
                        for index, err in enumerate(alpha_temp_stderr):
                            std_err_temp = 0
                            # print('err**2 ', err**2)
                            std_err_temp += err**2
                            print('std_err_temp ', std_err_temp)
                            # p = 1
                            for val in [v for v in alpha_temp
                                        if v!=alpha_temp[index]]:
                                # print('val**2 ', val**2)
                                std_err_temp *= val**2
                            # std_err_temp *= p

                            print('std_err_temp ', std_err_temp)
                            std_err_term += std_err_temp
                            print('std_err_term ', std_err_term)

                print('max ', max_alpha_product)
                print('max ', std_err_term)
                depolariz_param_term += max_alpha_product
                # depolariz_param_term_stderr += np.sqrt(std_err_term)
                depolariz_param_term_stderr += std_err_term

                # # i am now picking the max alpha_product so I will have only
                # # one term so no need to divide by len(keys)
                # depolariz_param_term /= len(keys)
                # depolariz_param_term_stderr /= len(keys)

                # print(depolariz_param_term)
                # print(depolariz_param_term_stderr)
                print('idx ', idx)
                total_corr_depolariz_param['val'] += \
                    prefactors[idx] * max_alpha_product
                total_corr_depolariz_param['stderr'] += \
                    (prefactors[idx]**2) * std_err_term

                # from pprint import pprint
                # pprint(total_corr_depolariz_param)

        total_corr_depolariz_param['val'] /= np.sum(prefactors)
        total_corr_depolariz_param['stderr'] = \
            np.sqrt(total_corr_depolariz_param['stderr'])/np.sum(prefactors)

        # pprint(total_corr_depolariz_param)

    return total_corr_depolariz_param


def get_keys(corr_string, corr_s, smallest_corr_s=1):
    if len(corr_string)<smallest_corr_s or corr_s<smallest_corr_s:
        return []
    else:
        var_name = corr_string
        corr_qbs_idxs_sep = list(itertools.combinations(corr_string, corr_s))

        corr_strings = []
        keys = []
        for corr_tuple in corr_qbs_idxs_sep:
            corr_string = ''.join(corr_tuple)
            corr_strings += [corr_string]

            other_qb_string = \
                ''.join([c for c in var_name if c not in corr_string])

            if other_qb_string not in corr_strings:
                keys += [(corr_string, other_qb_string)]

            temp = get_keys(corr_string=corr_string, corr_s=corr_s-1,
                     smallest_corr_s=smallest_corr_s)
            keys += [tup+(other_qb_string,) for tup in temp]

            if len(other_qb_string)>smallest_corr_s:
                temp = get_keys(corr_string=other_qb_string, corr_s=corr_s-1,
                         smallest_corr_s=smallest_corr_s)
                keys += [corr_tuple+tup for tup in temp]

        return keys