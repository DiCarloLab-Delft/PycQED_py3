import datetime
from datetime import timedelta

import numpy as np
import pycqed.analysis_v2.base_analysis as ba
import matplotlib.dates as mdates

from copy import deepcopy

import numpy as np
import lmfit
from matplotlib import pyplot as plt
import os
from pycqed.analysis import analysis_toolbox as a_tools


class CoherenceTimesAnalysisSingle(ba.BaseDataAnalysis):
    # todo docstring
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '', data_file_path: str = None,
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 tau_key='Analysis.Fitted Params F|1>.tau.value',
                 tau_stderr_key='Analysis.Fitted Params F|1>.tau.stderr',
                 plot_versus_dac=True,
                 dac_key='Instrument settings.fluxcurrent.Q',
                 plot_versus_frequency=True,
                 frequency_key='Instrument settings.Q.freq_qubit',
                 ):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=False)
        # self.single_timestamp = False
        self.params_dict = {'tau': tau_key,
                            'tau_stderr': tau_stderr_key,
                            }
        self.plot_versus_dac = plot_versus_dac
        if plot_versus_dac:
            self.params_dict['dac'] = dac_key

        self.plot_versus_frequency = plot_versus_frequency
        if plot_versus_frequency:
            self.params_dict['qfreq'] = frequency_key

        self.numeric_params = []

        if auto:
            self.run_analysis()

    def extract_data(self):
        # load data
        if not (hasattr(self, 'raw_data_dict')):
            super().extract_data()
            tau = np.array(self.raw_data_dict['tau'], dtype=float)
            tau_std = np.array(self.raw_data_dict['tau_stderr'], dtype=float)
            # sort data
            if self.plot_versus_dac:
                dacs = np.array(self.raw_data_dict['dac'], dtype=float)
                sorted_indices = dacs.argsort()
                self.raw_data_dict['dac_sorted'] = dacs[sorted_indices]
                self.raw_data_dict['dac_sorted_tau'] = tau[sorted_indices]
                self.raw_data_dict['dac_sorted_tau_stderr'] = tau_std[sorted_indices]
            if self.plot_versus_frequency:
                freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
                sorted_indices = freqs.argsort()
                self.raw_data_dict['freq_sorted'] = freqs[sorted_indices]
                self.raw_data_dict['freq_sorted_tau'] = tau[sorted_indices]
                self.raw_data_dict['freq_sorted_tau_stderr'] = tau_std[sorted_indices]

    def prepare_plots(self):
        if not ("time_stability" in self.plot_dicts):
            self._prepare_plot(ax_id='time_stability', xk='datetime', yk='tau', ye='tau_stderr',
                               xlabel='Time in Delft', xunit=None)
            if self.plot_versus_frequency:
                self._prepare_plot(ax_id="freq_relation", xk='freq_sorted', yk='freq_sorted_tau',
                                   ye='freq_sorted_tau_stderr',
                                   xlabel='Qubit Frequency', xunit='Hz')
            if self.plot_versus_dac:
                self._prepare_plot(ax_id="dac_relation", xk='dac_sorted', yk='dac_sorted_tau',
                                   ye='dac_sorted_tau_stderr',
                                   xlabel='DAC Value', xunit='A')

    def _prepare_plot(self, ax_id, xk, yk, ye, xlabel, xunit):
        xvals = self.raw_data_dict[xk]
        yvals = self.raw_data_dict[yk]
        # fixme: add error bars
        yerr = self.raw_data_dict[ye]

        plot_dict = {
            'xlabel': self.options_dict.get('xlabel', xlabel),
            'xunit': self.options_dict.get('xunit', xunit),
            'ylabel': self.options_dict.get('ylabel', 'Coherence'),
            'yunit': self.options_dict.get('yunit', 's'),
            'yrange': self.options_dict.get('yrange', (0, 1.1 * np.max(yvals))),
            'xrange': self.options_dict.get('xrange', None),
            'yunit': 's',
            # 'setlabel': setlabel,
            # 'legend_title': legend_title,
            # 'title': (self.raw_data_dict['timestamps'][0]+' - ' +
            #          self.raw_data_dict['timestamps'][-1] + '\n' +
            #          self.raw_data_dict['measurementstring'][0]),
            # 'do_legend': do_legend,
            # 'legend_pos': 'upper right'
        }

        self.plot_dicts[ax_id] = plot_scatter_errorbar(self=self, ax_id=ax_id, xdata=xvals, ydata=yvals,
                                                       xerr=None, yerr=yerr, pdict=plot_dict)


class CoherenceTimesAnalysis(ba.BaseDataAnalysis):
    T1 = 't1'
    T2 = 't2'
    T2_star = 't2s'

    def __init__(self, dac_instr_names: list, qubit_instr_names: list,
                 t_start: str = None, t_stop: str = None,
                 label: str = '', labels=None,
                 data_file_path: str = None,
                 options_dict: dict = None, extract_only: bool = False,
                 auto: bool = True,
                 tau_keys: dict = None,
                 tau_std_keys: dict = None,
                 plot_versus_dac: bool = True,
                 dac_key_pattern: str = 'Instrument settings.fluxcurrent.{DAC}',
                 plot_versus_frequency: bool = True,
                 frequency_key_pattern: str = 'Instrument settings.{Q}.freq_qubit',
                 res_freq: list = None, res_Qc: list = None, chi_shift: list = None,
                 do_fitting: bool = True, verbose: bool = True,
                 ):

        ## Check data and apply default values
        assert (len(qubit_instr_names) == len(dac_instr_names))
        if plot_versus_dac:
            assert (len(qubit_instr_names) == len(dac_instr_names))
        if res_Qc or res_freq or chi_shift:
            assert (len(qubit_instr_names) == len(res_Qc))
            assert (len(qubit_instr_names) == len(res_freq))
            assert (len(qubit_instr_names) == len(chi_shift))

        # Find the keys for the coherence times and their errors
        tau_keys = tau_keys or {
            self.T1: 'Analysis.Fitted Params F|1>.tau.value',
            self.T2: 'Analysis.Fitted Params raw w0.tau.value',
            self.T2_star: 'Analysis.Fitted Params corr_data.tau.value',
        }
        tau_std_keys = tau_std_keys or {
            self.T1: 'Analysis.Fitted Params F|1>.tau.stderr',
            self.T2: 'Analysis.Fitted Params raw w0.tau.stderr',
            self.T2_star: 'Analysis.Fitted Params corr_data.tau.stderr',
        }

        labels = labels or {
            self.T1: '_T1_{Q}',
            self.T2: '_echo_{Q}',
            self.T2_star: '_ramsey_{Q}',
        }

        assert (len(tau_keys) == len(labels))
        assert (len(tau_keys) == len(tau_std_keys))
        assert (len(tau_keys) >= 3)

        req = (self.T1, self.T2, self.T2_star)
        if not all(k in tau_keys for k in req):
            raise KeyError("You need to at least specify ", req, " for parameters tau_keys.")
        if not all(k in tau_std_keys for k in req):
            raise KeyError("You need to at least specify ", req, " for parameters tau_std_keys.")
        if not all(k in labels for k in req):
            raise KeyError("You need to at least specify ", req, " for parameters labels.")

        # Call abstract init
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         extract_only=extract_only)

        # Save some data for later use
        self.raw_data_dict = {}
        self.fit_res = {}
        self.res_Qc = res_Qc
        self.res_freq = res_freq
        self.chi_shift = chi_shift
        self.qubit_names = qubit_instr_names
        self.verbose = verbose

        # Find the dac and frequency keys
        self.dac_keys = [] if plot_versus_dac else None
        self.freq_keys = [] if plot_versus_frequency else None
        for i, dac_instr_name in enumerate(dac_instr_names):
            qubit_instr_name = qubit_instr_names[i]
            if plot_versus_dac:
                dac_key = self._parse(dac=dac_instr_name, qubit=qubit_instr_name, pattern=dac_key_pattern)
                self.dac_keys.append(dac_key)
            if plot_versus_frequency:
                freq_key = self._parse(dac=dac_instr_name, qubit=qubit_instr_name, pattern=frequency_key_pattern)
                self.freq_keys.append(freq_key)

        # Create all slave objects
        self.all_analysis = {}
        for i, dac_instr_name in enumerate(dac_instr_names):
            qubit = self.qubit_names[i]
            dac_key = self.dac_keys[i] if plot_versus_dac else None
            freq_key = self.freq_keys[i] if plot_versus_frequency else None
            self.all_analysis[self.qubit_names[i]] = {}
            for typ in tau_keys:
                tau_key = tau_keys[typ]
                tau_std_key = tau_std_keys[typ]

                self.all_analysis[qubit][typ] = CoherenceTimesAnalysisSingle(
                    t_start=t_start, t_stop=t_stop,
                    label=self._parse(dac=dac_instr_name, qubit=qubit, pattern=labels[typ]),
                    auto=False, extract_only=False,
                    tau_key=tau_key,
                    tau_stderr_key=tau_std_key,
                    plot_versus_dac=plot_versus_dac,
                    dac_key=dac_key,
                    plot_versus_frequency=plot_versus_frequency,
                    frequency_key=freq_key,
                    data_file_path=data_file_path,
                    options_dict=options_dict
                )

        if auto:
            self.run_analysis()

    @staticmethod
    def _parse(qubit, dac, pattern):
        key = pattern.replace('{DAC}', dac)
        key = key.replace('{Q}', qubit)
        return key

    def extract_data(self):
        youngest = None
        for qubit in self.all_analysis:
            self.raw_data_dict[qubit] = {}
            for typ in self.all_analysis[qubit]:
                a = self.all_analysis[qubit][typ]
                a.extract_data()
                self.raw_data_dict[qubit][typ] = a.raw_data_dict
                tmp_y = self.max_time(a.raw_data_dict["datetime"])
                if not youngest or youngest < tmp_y:
                    youngest = tmp_y
        youngest += datetime.timedelta(seconds=1)

        self.raw_data_dict['datetime'] = [youngest]
        self.raw_data_dict['timestamps'] = [youngest.strftime("%Y%m%d_%H%M%S")]
        self.timestamps = [youngest]
        folder = a_tools.datadir + '/%s_coherence_analysis' % (youngest.strftime("%Y%m%d/%H%M%S"))
        self.raw_data_dict['folder'] = [folder]

    @staticmethod
    def max_time(times):
        youngest = None
        for tmp in times:
            if not youngest or youngest < tmp:
                youngest = tmp
        return youngest

    def run_fitting(self):
        if self.freq_keys and self.dac_keys:
            for qubit in self.all_analysis:
                if self.verbose:
                    print("Analysing qubit: %s" % qubit)
                # sort the data by dac (just in case some samples are missing)
                qo = self.all_analysis[qubit]
                dac = []
                for typ in qo:
                    a = qo[typ]
                    data = a.raw_data_dict
                    data['dac'] = [float(d) for d in data['dac']]
                    for d in data['dac']:
                        dac.append(d)
                dac = [float(d) for d in set(dac)]

                if len(dac) > 4:
                    freq = [None] * len(dac)
                    t1 = [None] * len(dac)
                    a = qo[self.T1]
                    qf = a.raw_data_dict['qfreq']
                    tau = a.raw_data_dict['tau']
                    for i, d in enumerate(a.raw_data_dict['dac']):
                        index = dac.index(float(d))
                        freq[index] = qf[i]
                        t1[index] = tau[i]

                    t2s = [None] * len(dac)
                    a = qo[self.T2_star]
                    tau = a.raw_data_dict['tau']
                    for i, d in enumerate(a.raw_data_dict['dac']):
                        t2s[dac.index(d)] = tau[i]

                    t2e = [None] * len(dac)
                    a = qo[self.T2]
                    for i, d in enumerate(a.raw_data_dict['dac']):
                        t2e[dac.index(d)] = tau[i]

                    t1 = np.array(t1)
                    t2s = np.array(t2s)
                    t2e = np.array(t2e)
                    # Ignore unset values
                    t1_mask = (t1 == None)
                    t2s_mask = (t2s == None)
                    t2e_mask = (t2e == None)
                    # t1[t1_mask] = 0
                    # t2s[t2s_mask] = 0
                    # t2e[t2e_mask] = 0
                    it = prepare_input_table(dac=dac, frequency=freq, T1=t1, T2_star=t2s, T2_echo=t2e,
                                             T1_mask=t1_mask, T2_star_mask=t2s_mask,
                                             T2_echo_mask=t2e_mask)
                    chi = self.chi_shift[qubit] if self.chi_shift else None
                    res_freq = self.res_freq[qubit] if self.res_freq else None
                    res_Qc = self.res_Qc[qubit] if self.res_Qc else None
                    analysis_data_dict = PSD_Analysis(table=it, freq_resonator=res_freq, Qc=res_Qc, chi_shift=chi,
                                                      verbose=self.verbose)

                    self.raw_data_dict[qubit]['analysis'] = analysis_data_dict
                    self.fit_res[qubit] = analysis_data_dict
                    self.raw_data_dict[qubit]['processed_data'] = {
                        'dac': dac, 'qfreq': freq,
                        self.T1: t1, self.T2: t2e, self.T2_star: t2s,
                        self.T1 + '_mask': t1_mask, self.T2 + '_mask': t2e_mask, self.T2_star + '_mask': t2s_mask,
                    }

                else:
                    # fixme: make this a proper warning
                    print('Found %d dac values. I need at least 4 dac values to run the PSD analysis.' % len(dac))
        else:
            # fixme: make this a proper warning
            print('You have to enable plot_versus_frequency and plot_versus_dac to execute the PSD analysis.')

    def save_fit_results(self):
        # todo: if you want to save some results to a hdf5, do it here
        pass

    def prepare_plots(self):
        # prepare axis
        cr_all_base = "coherence_ratios"
        ct_all_base = 'coherence_times'
        cg_all_base = 'coherence_gamma'
        # f, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        # self.figs[cg_base] = f
        # self.plot_dicts[cg_base] = {}
        # self.axs[cg_base + "_flux"] = axs[0]
        # self.axs[cg_base + "_frequency"] = axs[1]
        # self.axs[cg_base + "_sensitivity"] = axs[2]

        self.plot_dicts = {}
        for qubit in self.all_analysis:
            # if the analysis was succesful
            dat = self.raw_data_dict[qubit]
            if dat['analysis']:
                flux = dat['analysis']['flux']
                freq = dat['analysis']['qfreq']
                sensitivity = dat['analysis']['sensitivity']
                gamma_phi_echo = dat['analysis']['gamma_phi_echo']
                gamma_phi_echo_f = dat['analysis']['gamma_phi_echo_f']
                gamma_phi_ramsey = dat['analysis']['gamma_phi_ramsey']
                gamma_phi_ramsey_f = dat['analysis']['gamma_phi_ramsey_f']
                # t1 = dat['analysis'][self.T1]
                # t2 = dat['analysis'][self.T2]
                # t2s = dat['analysis'][self.T2_star]

                ############
                # coherence_gamma
                pdict_scatter = {
                    'xlabel': 'Sensitivity',
                    'xunit': 'GHz/$\Phi_0$',
                    'ylabel': '$\Gamma_{\phi}$',
                    'yunit': '$s^{-1}$',
                    'setlabel': '$\Gamma_{\phi,\mathrm{Ramsey}}$',
                }
                pdict_fit = {}
                cg_base = qubit + "_" + cg_all_base
                pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id=cg_base, xdata=np.abs(sensitivity),
                                                     ydata=gamma_phi_ramsey, fitfunc=gamma_phi_ramsey_f,
                                                     xerr=None, yerr=None, fitextra=0.1, fitpoints=1000,
                                                     pdict_scatter=pdict_scatter, pdict_fit=pdict_fit)
                self.plot_dicts[cg_base + '_ramsey_fit'] = pdf
                self.plot_dicts[cg_base + '_ramsey_scatter'] = pds

                pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id=cg_base, xdata=np.abs(sensitivity),
                                                     ydata=gamma_phi_echo, fitfunc=gamma_phi_echo_f,
                                                     xerr=None, yerr=None, fitextra=0.1, fitpoints=1000,
                                                     pdict_scatter=pdict_scatter, pdict_fit=pdict_fit)
                self.plot_dicts[cg_base + '_echo_fit'] = pdf
                self.plot_dicts[cg_base + '_echo_scatter'] = pds

                ############
                # coherence_ratios
                cr_base = qubit + '_' + cr_all_base

                ratio_gamma = gamma_phi_ramsey / gamma_phi_echo

                pdict_scatter = {
                    'xlabel': 'Flux',
                    'xunit': 'm$\Phi_0$',
                    'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                }
                pds = plot_scatter_errorbar(self=self, ax_id=cr_all_base + '_flux', xdata=flux,
                                            ydata=ratio_gamma,
                                            xerr=None, yerr=None,
                                            pdict=pdict_scatter)
                self.plot_dicts[cr_base + '_flux'] = pds

                pdict_scatter = {
                    'xlabel': 'Qubit Frequency',
                    'xunit': 'Hz',
                    'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                }
                pds = plot_scatter_errorbar(self=self, ax_id=cr_all_base + '_frequency', xdata=freq,
                                            ydata=ratio_gamma,
                                            xerr=None, yerr=None,
                                            pdict=pdict_scatter)
                self.plot_dicts[cr_base + '_frequency'] = pds

                pdict_scatter = {
                    'xlabel': r'Sensitivity $|\partial\nu/\partial\Phi|$',
                    'xunit': 'Hz/$\Phi_0$',
                    'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                }
                pds = plot_scatter_errorbar(self=self, ax_id=cr_all_base + '_sensitivity', xdata=sensitivity,
                                            ydata=ratio_gamma,
                                            xerr=None, yerr=None,
                                            pdict=pdict_scatter)
                self.plot_dicts[cr_base + '_sensitivity'] = pds

                ############
                # coherence_times
                ct_base = qubit + '_' + ct_all_base

                yrange = self.options_dict.get('yrange', (0, 60e-6))
                for typ in self.all_analysis[qubit]:
                    a = self.all_analysis[qubit][typ]
                    a.prepare_plots()
                    label = '%s_%s' % (qubit, typ)

                    self.plot_dicts[ct_base + '_time_stability_' + typ] = a.plot_dicts["time_stability"]
                    self.plot_dicts[ct_base + '_time_stability_' + typ]['ax_id'] = ct_base + '_time_stability'
                    self.plot_dicts[ct_base + '_time_stability_' + typ]['setlabel'] = label
                    self.plot_dicts[ct_base + '_time_stability_' + typ]['do_legend'] = True
                    self.plot_dicts[ct_base + '_time_stability_' + typ]['yrange'] = None

                    if self.freq_keys:
                        self.plot_dicts[ct_base + '_freq_relation_' + typ] = a.plot_dicts["freq_relation"]
                        self.plot_dicts[ct_base + '_freq_relation_' + typ]['ax_id'] = ct_base + '_freq_relation'
                        self.plot_dicts[ct_base + '_freq_relation_' + typ]['setlabel'] = label
                        self.plot_dicts[ct_base + '_freq_relation_' + typ]['do_legend'] = True
                        self.plot_dicts[ct_base + '_freq_relation_' + typ]['yrange'] = None
                    if self.dac_keys:
                        self.plot_dicts[ct_base + '_dac_relation_' + typ] = a.plot_dicts["dac_relation"]
                        self.plot_dicts[ct_base + '_dac_relation_' + typ]['ax_id'] = ct_base + '_dac_relation'
                        self.plot_dicts[ct_base + '_dac_relation_' + typ]['setlabel'] = label
                        self.plot_dicts[ct_base + '_dac_relation_' + typ]['do_legend'] = True
                        self.plot_dicts[ct_base + '_dac_relation_' + typ]['yrange'] = None

                    if self.dac_keys and self.freq_keys:
                        pdict_scatter = {
                            'xlabel': r'Sensitivity $|\partial\nu/\partial\Phi|$',
                            'xunit': 'Hz/$\Phi_0$',
                            'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                            'setlabel': label
                        }
                        pds = plot_scatter_errorbar(self=self, ax_id=ct_base + '_flux_relation', xdata=flux,
                                                    ydata=dat['analysis'][typ],
                                                    xerr=None, yerr=None,
                                                    pdict=pdict_scatter)
                        self.plot_dicts[ct_base + '_flux_relation_' + typ] = pds

                # self.raw_data_dict[qubit][typ] = a.raw_data_dict

        d = np.array([1, 2, 3, 4]) * 1e-6
        self.plot_dicts['test2'] = plot_scatter_errorbar(self, ax_id='test2', xdata=d, ydata=d, xerr=d / 10,
                                                         yerr=d / 10, pdict=None)
        self.plot_dicts['test'] = {
            'ax_id': 'test',
            'plotfn': self.plot_line,
            'xvals': d,
            'xlabel': 'x',
            'xunit': 's',
            'yvals': d,
            'ylabel': 'y',
            'yunit': 'yu',
            'setlabel': 'setlabel',
            'marker': 'x',
            'yerr': d / 10,
            'xerr': d / 10,
        }


def plot_scatter_errorbar(self, ax_id, xdata, ydata, xerr=None, yerr=None, pdict=None):
    pdict = pdict or {}

    pds = {
        'ax_id': ax_id,
        'plotfn': self.plot_line,
        'xvals': xdata,
        'xlabel': 'x',
        'xunit': 'xu',
        'yvals': ydata,
        'ylabel': 'y',
        'yunit': 'yu',
        'setlabel': 'setlabel',
        'marker': 'x',
        'yerr': yerr,
        'xerr': xerr,
    }

    if xerr is not None or yerr is not None:
        pds['func'] = 'errorbar'
        pds['marker'] = None
        pds['line_kws'] = {'fmt': 'none'}
    else:
        pds['func'] = 'scatter'

    for k in pdict:
        pds[k] = pdict[k]

    return pds


def plot_scatter_errorbar_fit(self, ax_id, xdata, ydata, fitfunc, xerr=None, yerr=None, fitextra=0.1,
                              fitpoints=1000, pdict_scatter=None, pdict_fit=None):
    pdict_fit = pdict_fit or {}
    pds = plot_scatter_errorbar(self=self, ax_id=ax_id, xdata=xdata, ydata=ydata, xerr=xerr, yerr=xerr,
                                pdict=pdict_scatter)

    mi, ma = np.min(xdata), np.max(xdata)
    ex = (ma - mi) * fitextra
    xdata_fit = np.linspace(mi - ex, ma + ex, fitpoints)
    ydata_fit = fitfunc(xdata_fit)

    pdf = {
        'ax_id': ax_id,
        'plotfn': self.plot_line,
        'xvals': xdata_fit,
        'yvals': ydata_fit,
        'linestyle': '-',
        'marker': '',
    }
    for k in pdict_fit:
        pdf[k] = pdict_fit[k]
    return pds, pdf


def PSD_Analysis(table, freq_resonator: float = None, Qc: float = None, chi_shift: float = None, verbose: bool = True):
    """
    Requires a table as input:
           Row  | Content
        --------+--------
            1   | dac values
            2   | frequency
            3   | T1
            4   | T2 star
            5   | T2 echo
            6   | Exclusion mask (True where data is to be excluded)

    Generates 7 datasets:
        > T1, T2, Echo vs flux
        > T1, T2, Echo vs frequency
        > T1, T2, Echo vs flux sensitivity

        > ratio Ramsey/Echo vs flux
        > ratio Ramsey/Echo vs frequency
        > ratio Ramsey/Echo vs flux sensitivity

        > Dephasing rates Ramsey and Echo vs flux sensitivity

        If properties of resonator are provided (freq_resonator, Qc, chi_shift),
        it also calculates the number of noise photons.
    """
    dac, freq, T1, Tramsey, Techo, exclusion_mask = table
    exclusion_mask = np.array(exclusion_mask, dtype=bool)

    # Extract the dac arcs required for getting the sensitivities
    fit_result_arch = fit_frequencies(dac, freq)

    # convert dac in flux as unit of Phi_0
    flux = (dac - fit_result_arch.best_values['offset']) \
           / fit_result_arch.best_values['dac0']

    # calculate the derivative vs flux
    sensitivity_angular = partial_omega_over_flux(
        flux, fit_result_arch.best_values['Ec'],
        fit_result_arch.best_values['Ej'])
    sensitivity = sensitivity_angular / (2 * np.pi)

    # Pure dephasing times
    # Calculate pure dephasings
    Gamma_1 = 1.0 / T1[~exclusion_mask]
    Gamma_ramsey = 1.0 / Tramsey[~exclusion_mask]
    Gamma_echo = 1.0 / Techo[~exclusion_mask]

    Gamma_phi_ramsey = Gamma_ramsey - Gamma_1 / 2.0
    Gamma_phi_echo = Gamma_echo - Gamma_1 / 2.0

    fit_res_gammas = fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo)

    intercept = fit_res_gammas.params['intercept'].value
    slope_ramsey = fit_res_gammas.params['slope_ramsey'].value
    slope_echo = fit_res_gammas.params['slope_echo'].value

    # after fitting gammas
    # from flux noise
    # Martinis PRA 2003
    sqrtA_rams = slope_ramsey / (np.pi * np.sqrt(30))
    sqrtA_echo = slope_echo / (np.pi * np.sqrt(1.386))

    if verbose:
        print('Amplitude echo PSD = (%s u\Phi_0)^2' % (sqrtA_echo / 1e-6))
        print('Amplitude rams PSD = (%s u\Phi_0)^2' % (sqrtA_rams / 1e-6))

    data_dict = {}
    # from white noise
    # using Eq 5 in Nat. Comm. 7,12964 (The flux qubit revisited to enhance
    # coherence and reproducability)
    if not ((freq_resonator is None) and (Qc is None) and (chi_shift is None)):
        n_avg = calculate_n_avg(freq_resonator, Qc, chi_shift, intercept)
        data_dict['avg_noise_photons'] = n_avg
        if verbose:
            print('Estimated residual photon number: %s' % n_avg)

    Gamma_phi_echo_f = lambda x: slope_echo * x + intercept
    Gamma_phi_ramsey_f = lambda x: slope_ramsey * x + intercept
    data_dict['qfreq'] = freq
    data_dict['t1'] = T1
    data_dict['t2'] = Techo
    data_dict['t2s'] = Tramsey
    data_dict['flux'] = flux
    data_dict['gamma_intercept'] = intercept
    data_dict['gamma_slope_ramsey'] = slope_ramsey
    data_dict['gamma_slope_echo'] = slope_echo
    data_dict['gamma_phi_ramsey'] = Gamma_phi_ramsey
    data_dict['gamma_phi_ramsey_f'] = Gamma_phi_ramsey_f
    data_dict['gamma_phi_echo'] = Gamma_phi_echo
    data_dict['gamma_phi_echo_f'] = Gamma_phi_echo_f
    data_dict['sensitivity'] = sensitivity
    data_dict['sqrtA_echo'] = (sqrtA_echo / 1e-6)
    data_dict['fit_res'] = fit_res_gammas

    return data_dict


def calculate_n_avg(freq_resonator, Qc, chi_shift, intercept):
    """
    Returns the avg photon of white noise,assuming photon shot noise from the RO hanger.
    """
    k_r = 2 * np.pi * freq_resonator / Qc
    eta = k_r ** 2 / (k_r ** 2 + 4 * chi_shift ** 2)
    n_avg = intercept * k_r / (4 * chi_shift ** 2 * eta)
    return n_avg


def prepare_input_table(dac: list, frequency: list, T1: list, T2_star: list, T2_echo: list,
                        T1_mask: list = None, T2_star_mask: list = None, T2_echo_mask: list = None):
    """
    Returns a table ready for PSD_Analysis input
    If sizes are different, it adds nans on the end.
    """
    assert (len(dac) == len(frequency))
    assert (len(dac) == len(T1))
    assert (len(dac) == len(T2_star))
    assert (len(dac) == len(T2_echo))

    if T1_mask is None:
        T1_mask = np.zeros(len(T1), dtype=bool)
    if T2_star_mask is None:
        T2_star_mask = np.zeros(len(T2_star), dtype=bool)
    if T2_echo_mask is None:
        T2_echo_mask = np.zeros(len(T2_echo), dtype=bool)

    assert (len(T1) == len(T1_mask))
    assert (len(T2_star) == len(T2_star_mask))
    assert (len(T2_echo) == len(T2_echo_mask))

    table = np.ones((6, len(dac)))
    table = table * np.nan
    table[0, :] = dac
    table[1, :len(frequency)] = frequency
    table[2, :len(T1)] = T1
    table[3, :len(T2_star)] = T2_star
    table[4, :len(T2_echo)] = T2_echo
    table[5, :len(T1_mask)] = np.logical_or(np.logical_or(T1_mask,
                                                          T2_star_mask),
                                            T2_echo_mask)

    return table


def arch(dac, Ec, Ej, offset, dac0):
    '''
    Function for frequency vs flux (in dac) for the transmon

    Input:
        - dac: voltage used in the DAC to generate the flux
        - Ec (Hz): Charging energy of the transmon in Hz
        - Ej (Hz): Josephson energy of the transmon in Hz
        - offset: voltage offset of the arch (same unit of the dac)
        - dac0: dac value to generate 1 Phi_0 (same unit of the dac)

    Note: the Phi_0 (periodicity) dac0
    '''
    model = np.sqrt(8 * Ec * Ej * np.abs(np.cos((np.pi * (dac - offset)) / dac0))) - Ec

    return model


# define the model (from the function) used to fit data
arch_model = lmfit.Model(arch)


# derivative of arch vs flux (in unit of Phi0)
# this is the sensitivity to flux noise
def partial_omega_over_flux(flux, Ec, Ej):
    '''
    Note: flux is in unit of Phi0
    Ej and Ec are in Hz

    Output: angular frequency over Phi_0
    '''
    model = -np.sign(np.cos(np.pi * flux)) * (np.pi ** 2) * np.sqrt(8 * Ec * Ej) * \
            np.sin(np.pi * flux) / np.sqrt(np.abs(np.cos(np.pi * flux)))
    return model


def fit_frequencies(dac, freq):
    arch_model.set_param_hint('Ec', value=250e6, min=200e6, max=300e6)
    arch_model.set_param_hint('Ej', value=18e9, min=0)
    arch_model.set_param_hint('offset', value=0)
    arch_model.set_param_hint('dac0', value=2000, min=0)

    arch_model.make_params()

    fit_result_arch = arch_model.fit(freq, dac=dac)
    return fit_result_arch


def residual_Gamma(pars_dict, sensitivity, Gamma_phi_ramsey, Gamma_phi_echo):
    slope_ramsey = pars_dict['slope_ramsey']
    slope_echo = pars_dict['slope_echo']
    intercept = pars_dict['intercept']

    gamma_values_ramsey = slope_ramsey * np.abs(sensitivity) + intercept
    residual_ramsey = Gamma_phi_ramsey - gamma_values_ramsey

    gamma_values_echo = slope_echo * np.abs(sensitivity) + intercept
    residual_echo = Gamma_phi_echo - gamma_values_echo

    return np.concatenate((residual_ramsey, residual_echo))


def super_residual(p):
    data = residual_Gamma(p)
    # print(type(data))
    return data.astype(float)


def fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo, verbose=0):
    # create a parametrrer set for the initial guess
    p = lmfit.Parameters()
    p.add('slope_ramsey', value=100.0, vary=True)
    p.add('slope_echo', value=100.0, vary=True)
    p.add('intercept', value=100.0, vary=True)

    # mi = lmfit.minimize(super_residual, p)
    wrap_residual = lambda p: residual_Gamma(p,
                                             sensitivity=sensitivity,
                                             Gamma_phi_ramsey=Gamma_phi_ramsey,
                                             Gamma_phi_echo=Gamma_phi_echo)
    fit_result_gammas = lmfit.minimize(wrap_residual, p)
    verbose = 1
    if verbose > 0:
        lmfit.printfuncs.report_fit(fit_result_gammas.params)
    return fit_result_gammas


def plot_gamma_fit(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo, slope_ramsey, slope_echo, intercept, ax=None,
                   **kwargs):
    if ax is None:
        f, ax = plt.subplots()

    ax.plot(np.abs(sensitivity) / 1e9, Gamma_phi_ramsey,
            'o', color='C2', label='$\Gamma_{\phi,\mathrm{Ramsey}}$')
    ax.plot(np.abs(sensitivity) / 1e9, slope_ramsey *
            np.abs(sensitivity) + intercept, color='C2')

    ax.plot(np.abs(sensitivity) / 1e9, Gamma_phi_echo,
            'o', color='C0', label='$\Gamma_{\phi,\mathrm{Echo}}$')
    ax.plot(np.abs(sensitivity) / 1e9, slope_echo *
            np.abs(sensitivity) + intercept, color='C0')

    ax.legend(loc=0)
    ax.set_title('Pure dephasing vs flux sensitivity')
    ax.set_xlabel(r'$|\partial f/\partial\Phi|$ (GHz/$\Phi_0$)')
    ax.set_ylabel('$\Gamma_{\phi}$ (1/s)')
    ax.set_ylim(0, np.max(Gamma_phi_ramsey) * 1.05)


def plot_coherence_times(flux, freq, sensitivity, T1, Tramsey, Techo, ax, **kwargs):
    # if not ax:
    f, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    ax[0].plot(flux / 1e-3, T1 / 1e-6, 'o', color='C3', label='$T_1$')
    ax[0].plot(flux / 1e-3, Tramsey / 1e-6, 'o', color='C2', label='$T_2^*$')
    ax[0].plot(flux / 1e-3, Techo / 1e-6, 'o', color='C0', label='$T_2$')
    ax[0].set_title('$T_1$, $T_2^*$, $T_2$ vs flux', size=16)
    ax[0].set_ylabel('Coherence time ($\mu$s)', size=16)
    ax[0].set_xlabel('Flux (m$\Phi_0$)', size=16)

    ax[1].plot(freq / 1e9, T1 / 1e-6, 'o', color='C3', label='$T_1$')
    ax[1].plot(freq / 1e9, Tramsey / 1e-6, 'o', color='C2', label='$T_2^*$')
    ax[1].plot(freq / 1e9, Techo / 1e-6, 'o', color='C0', label='$T_2$')
    ax[1].set_title('$T_1$, $T_2^*$, $T_2$ vs frequency', size=16)
    ax[1].set_xlabel('Frequency (GHz)', size=16)
    ax[1].legend(loc=0)

    ax[2].plot(np.abs(sensitivity) / 1e9, T1 / 1e-6, 'o', color='C3', label='$T_1$')
    ax[2].plot(np.abs(sensitivity) / 1e9, Tramsey / 1e-6,
               'o', color='C2', label='$T_2^*$')
    ax[2].plot(
        np.abs(sensitivity) / 1e9, Techo / 1e-6, 'o', color='C0', label='$T_2$')
    ax[2].set_title('$T_1$, $T_2^*$, $T_2$ vs sensitivity', size=16)
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)', size=16)


def plot_ratios(flux, freq, sensitivity, Gamma_phi_ramsey, Gamma_phi_echo, ax=None, **kwargs):
    # Pure dephasing times
    f, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    ratio_gamma = Gamma_phi_ramsey / Gamma_phi_echo

    ax[0].plot(flux / 1e-3, ratio_gamma, 'o', color='C0')
    ax[0].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs flux', size=16)
    ax[0].set_ylabel('Ratio', size=16)
    ax[0].set_xlabel('Flux (m$\Phi_0$)', size=16)

    ax[1].plot(freq / 1e9, ratio_gamma, 'o', color='C0')
    ax[1].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs frequency', size=16)
    ax[1].set_xlabel('Frequency (GHz)', size=16)

    ax[2].plot(np.abs(sensitivity) / 1e9, ratio_gamma, 'o', color='C0')
    ax[2].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs sensitivity', size=16)
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)', size=16)


def plot_coherence_times_freq(freq, T1, Tramsey, Techo):
    if not ax:
        f, ax = plt.subplots()

    ax.plot(freq / 1e9, T1 / 1e-6, 'o', color='C3', label='$T_1$')
    ax.plot(freq / 1e9, Tramsey / 1e-6, 'o', color='C2', label='$T_2^*$')
    ax.plot(freq / 1e9, Techo / 1e-6, 'o', color='C0', label='$T_2$')
    ax.set_title('$T_1$, $T_2^*$, $T_2$ vs frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.legend(loc=0)
