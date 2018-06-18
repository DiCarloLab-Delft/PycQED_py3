'''
Hacked together by Rene Vollmer
'''

import datetime
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar

import numpy as np
import lmfit

from pycqed.analysis import analysis_toolbox as a_tools


class CoherenceTimesAnalysisSingle(ba.BaseDataAnalysis):
    # todo docstring
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 tau_key='Analysis.Fitted Params F|1>.tau.value',
                 tau_std_key='Analysis.Fitted Params F|1>.tau.stderr',
                 plot_versus_dac=True,
                 dac_key='Instrument settings.fluxcurrent.Q',
                 plot_versus_frequency=True,
                 frequency_key='Instrument settings.Q.freq_qubit',
                 ):
        '''
        Plots and Analyses the coherence time (e.g. T1, T2 OR T2*) of one measurement series.

        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param label: the label that was used to name the measurements (only necessary if non-relevant measurements are in the time range)
        :param options_dict: Available options are the ones from the base_analysis and:
                                - (todo)
        :param auto: Execute all steps automatically
        :param close_figs: Close the figure (do not display)
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param tau_key: key for the tau (time) fit result, e.g. 'Analysis.Fitted Params F|1>.tau.value'
        :param tau_std_key: key for the tau (time) standard deviation fit result,
                            e.g. 'Analysis.Fitted Params F|1>.tau.stderr'
        :param plot_versus_dac: Extract and plot dac value?
                                    E.g. set False if you did not vary the dac for this measurement.
        :param dac_key: key for the dac current values, e.g. 'Instrument settings.fluxcurrent.Q'
        :param plot_versus_frequency: Extract and plot frequency value?
                                      E.g. set False if you did not use the Qubit object.
        :param frequency_key: key for the dac current values, e.g. 'Instrument settings.Q.freq_qubit'
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)
        # self.single_timestamp = False
        if 'F|1>' in tau_key:
            chisquared_key = 'Analysis.Fitted Params F|1>.chisqr'
        elif 'raw' in tau_key:
            chisquared_key = 'Analysis.Fitted Params raw w0.chisqr'
        elif 'corr_data' in tau_key:
            chisquared_key = 'Analysis.Fitted Params corr_data.chisqr'
        self.params_dict = {'tau': tau_key,
                            'tau_stderr': tau_std_key,
                            'chisquared' : chisquared_key
                            }
        self.numeric_params = ['tau', 'tau_stderr', 'chisquared']

        self.plot_versus_dac = plot_versus_dac
        if plot_versus_dac:
            self.params_dict['dac'] = dac_key

        self.plot_versus_frequency = plot_versus_frequency
        if plot_versus_frequency:
            self.params_dict['qfreq'] = frequency_key

        self.numeric_params = []

        if auto:
            self.run_analysis()
        # return self.proc_data_dict

    def extract_data(self):
        # load data
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
                self.raw_data_dict['dac_sorted_freq'] = freqs[sorted_indices]

        if self.plot_versus_frequency:
            freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
            sorted_indices = freqs.argsort()
            self.raw_data_dict['freq_sorted'] = freqs[sorted_indices]
            self.raw_data_dict['freq_sorted_tau'] = tau[sorted_indices]
            self.raw_data_dict['freq_sorted_tau_stderr'] = tau_std[sorted_indices]
            if self.plot_versus_dac:
                freqs = np.array(self.raw_data_dict['dac'], dtype=float)
                self.raw_data_dict['freq_sorted_dac'] = freqs[sorted_indices]

    def run_fitting(self):
        # This is not the proper way to do this!
        # TODO: move this to prepare_fitting
        if hasattr(self, 'raw_data_dict'):
            self.fit_res = {}
            if self.plot_versus_dac and self.plot_versus_frequency:
                dac = self.raw_data_dict['dac_sorted']
                freq = self.raw_data_dict['dac_sorted_freq']
                # dac = self.raw_data_dict['freq_sorted_dac']
                # freq = self.raw_data_dict['freq_sorted']
                # Extract the dac arcs required for getting the sensitivities
                fit_object = fit_frequencies(dac=dac, freq=freq)
                self.fit_res['dac_arc_object'] = fit_object
                self.fit_res['dac_arc_fitfct'] = lambda x: fit_object.model.eval(fit_object.params, dac=x)

                # convert dac in flux as unit of Phi_0
                flux = (dac - fit_object.best_values['offset']) / fit_object.best_values['dac0']
                self.fit_res['flux_values'] = flux

                # calculate the derivative vs flux
                sensitivity_angular = partial_omega_over_flux(flux, fit_object.best_values['Ec'],
                                                              fit_object.best_values['Ej'])
                self.fit_res['Ec'] = fit_object.best_values['Ec']
                self.fit_res['Ej'] = fit_object.best_values['Ej']
                self.fit_res['sensitivity_values'] = sensitivity_angular / (2 * np.pi)
                if self.verbose:
                    # todo: print EC and EJ
                    pass
        else:
            print('Warning: first run extract_data!')

    def save_fit_results(self):
        # todo: if you want to save some results to a hdf5, do it here
        pass

    def prepare_plots(self):
        if not ("time_stability" in self.plot_dicts):
            self._prepare_plot(ax_id='time_stability', xvals=self.raw_data_dict['datetime'],
                               yvals=self.raw_data_dict['tau'], yerr=self.raw_data_dict['tau_stderr'],
                               xlabel='Time in Delft', xunit=None)
            if self.plot_versus_frequency:
                self._prepare_plot(ax_id='freq_relation', xvals=self.raw_data_dict['freq_sorted'],
                                   yvals=self.raw_data_dict['freq_sorted_tau'],
                                   yerr=self.raw_data_dict['freq_sorted_tau_stderr'],
                                   xlabel='Qubit Frequency', xunit='Hz')

            if self.plot_versus_dac:
                # dac vs frequency (with fit if possible)
                plot_dict = {
                    'xlabel': 'DAC Current', 'xunit': 'A',
                    'ylabel': 'Qubit Frequency', 'yunit': 'Hz',
                }
                if hasattr(self, 'fit_res'):
                    pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id='dac_freq_relation',
                                                         xdata=self.raw_data_dict['dac_sorted'],
                                                         ydata=self.raw_data_dict['dac_sorted_freq'],
                                                         fitfunc=self.fit_res['dac_arc_fitfct'],
                                                         pdict_scatter=plot_dict, pdict_fit=plot_dict)
                    self.plot_dicts["dac_freq_relation_scatter"] = pds
                    self.plot_dicts["dac_freq_relation_fit"] = pdf
                else:
                    pds = plot_scatter_errorbar(self=self, ax_id='dac_freq_relation',
                                                xdata=self.raw_data_dict['dac_sorted'],
                                                ydata=self.raw_data_dict['dac_sorted_freq'],
                                                pdict=plot_dict)
                    self.plot_dicts["dac_freq_relation"] = pds

                # coherence time vs dac
                self._prepare_plot(ax_id="dac_relation", xvals=self.raw_data_dict['dac_sorted'],
                                   yvals=self.raw_data_dict['dac_sorted_tau'],
                                   yerr=self.raw_data_dict['dac_sorted_tau_stderr'],
                                   xlabel='DAC Value', xunit='A')

            if hasattr(self, 'fit_res'):
                if 'sensitivity_values' in self.fit_res:
                    # sensitivity vs tau
                    self._prepare_plot(ax_id="sensitivity_relation",
                                       xvals=self.fit_res['sensitivity_values'] * 1e-9,
                                       yvals=self.raw_data_dict['dac_sorted_tau'],
                                       yerr=self.raw_data_dict['dac_sorted_tau_stderr'],
                                       xlabel=r'Sensitivity $|\partial\nu/\partial\Phi|$',
                                       xunit=r'GHz/$\Phi_0$')
                if 'flux_values' in self.fit_res:
                    # flux vs tau
                    self._prepare_plot(ax_id="flux_relation",
                                       xvals=self.fit_res['flux_values'],
                                       yvals=self.raw_data_dict['dac_sorted_tau'],
                                       yerr=self.raw_data_dict['dac_sorted_tau_stderr'],
                                       xlabel='Flux Value', xunit='$\Phi_0$')

    def _prepare_plot(self, ax_id, xvals, yvals, xlabel, xunit, yerr=None):
        plot_dict = {
            'xlabel': xlabel,
            'xunit': xunit,
            'ylabel': 'Coherence',
            'yrange': (0, 1.1 * np.max(yvals)),
            'yunit': 's',
            # 'marker': 'x',
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
    T2 = 't2'  # e.g. echo
    T2_star = 't2s'  # e.g. ramsey

    def __init__(self, dac_instr_names: list, qubit_instr_names: list,
                 t_start: str = None, t_stop: str = None,
                 label: str = '', labels=None,
                 options_dict: dict = None, extract_only: bool = False,
                 auto: bool = True,
                 tau_keys: dict = None,
                 tau_std_keys: dict = None,
                 plot_versus_dac: bool = True,
                 dac_key_pattern: str = 'Instrument settings.fluxcurrent.{DAC}',
                 plot_versus_frequency: bool = True,
                 frequency_key_pattern: str = 'Instrument settings.{Q}.freq_qubit',
                 res_freq: list = None, res_Qc: list = None, chi_shift: list = None,
                 do_fitting: bool = True, close_figs: bool = True,
                 ):
        '''
        Plots and Analyses the coherence times (i.e. T1, T2 OR T2*) of one or several measurements.

        :param t_start:
        :param t_stop:
        :param label:
        :param options_dict:
        :param auto:
        :param close_figs: Close the figure (do not display)
        :param extract_only:
        :param do_fitting:
        :param tau_key:
        :param tau_std_key: key for the tau (time) standard deviation fit result,
                            e.g. 'Analysis.Fitted Params F|1>.tau.stderr'
        :param plot_versus_dac: Extract and plot dac value?
                                    E.g. set False if you did not vary the dac for this measurement.
        :param dac_key:
        :param plot_versus_frequency: Extract and plot frequency value?
                                      E.g. set False if you did not use the Qubit object.
        :param frequency_key:


        :param dac_instr_names:
        :param qubit_instr_names:
        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param labels: a dict of the labels that were used to name the measurements (only necessary if non-relevant measurements are in the time range)
        :param options_dict:  Available options are the ones from the base_analysis and:
                                - (todo)
        :param extract_only: Should we also do the plots?
        :param auto: Execute all steps automatically
        :param tau_keys: dict of keys for the tau (time) fit results,
                            e.g. {CoherenceTimesAnalysis.T1 : 'Analysis.Fitted Params F|1>.tau.value', ...}
        :param tau_std_keys: dict of keys for the tau standard deviation (time) fit results,
                            e.g. {CoherenceTimesAnalysis.T1 : 'Analysis.Fitted Params F|1>.tau.stderr', ...}
        :param plot_versus_dac: Extract and plot dac value?
                                    E.g. set False if you did not vary the dac for this measurement.
        :param dac_key_pattern: key pattern for the dac current values, e.g. 'Instrument settings.fluxcurrent.{DAC}'
                                    use {Q} to replace by qubit_instr_names and {DAC} to replace by dac_instr_names.
        :param plot_versus_frequency: Extract and plot frequency value?
                                      E.g. set False if you did not use the Qubit object.
        :param frequency_key_pattern: keys for the dac current values, e.g. 'Instrument settings.{Q}.freq_qubit'
                                    use {Q} to replace by qubit_instr_names and {DAC} to replace by dac_instr_names.
        :param res_freq: Frequency of the resonator
        :param res_Qc: Quality factor of the resonator
        :param chi_shift: Qubit-induced dispersive shift
        :param do_fitting: Should the run_fitting method be executed?
        :param close_figs:  Close the figure (do not display)
        '''

        if dac_instr_names is str:
            dac_instr_names = [dac_instr_names, ]
        if qubit_instr_names is str:
            qubit_instr_names = [qubit_instr_names, ]

        ## Check data and apply default values
        assert (len(qubit_instr_names) == len(dac_instr_names))
        if plot_versus_dac:
            assert (len(qubit_instr_names) == len(dac_instr_names))
        if res_Qc or res_freq or chi_shift:
            assert (len(qubit_instr_names) == len(res_Qc))
            assert (len(qubit_instr_names) == len(res_freq))
            assert (len(qubit_instr_names) == len(chi_shift))

        # Find the keys for the coherence times and their errors
        # todo merge instead of overwrite!
        tau_keys = tau_keys or {
            self.T1: 'Analysis.Fitted Params F|1>.tau.value',
            self.T2: 'Analysis.Fitted Params corr_data.tau.value',
            # self.T2: 'Analysis.Fitted Params raw w0.tau.value',
            self.T2_star: 'Analysis.Fitted Params raw w0.tau.value',
        }
        tau_std_keys = tau_std_keys or {
            self.T1: 'Analysis.Fitted Params F|1>.tau.stderr',
            self.T2: 'Analysis.Fitted Params corr_data.tau.stderr',
            # self.T2: 'Analysis.Fitted Params raw w0.tau.stderr',
            self.T2_star: 'Analysis.Fitted Params raw w0.tau.stderr',
        }

        if len(qubit_instr_names) == 1:
            s = ''
        else:
            s = '_{Q}'

        labels = labels or {
            self.T1: '_T1' + s,
            self.T2: '_echo' + s,
            self.T2_star: '_ramsey' + s,
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
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         extract_only=extract_only,
                         close_figs=close_figs)

        # Save some data for later use
        self.raw_data_dict = {}
        self.fit_res = {}
        self.res_Qc = res_Qc
        self.res_freq = res_freq
        self.chi_shift = chi_shift
        self.qubit_names = qubit_instr_names

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
                    auto=False, extract_only=True,
                    tau_key=tau_key,
                    tau_std_key=tau_std_key,
                    plot_versus_dac=plot_versus_dac,
                    dac_key=dac_key,
                    plot_versus_frequency=plot_versus_frequency,
                    frequency_key=freq_key,
                    options_dict=options_dict,
                    close_figs=close_figs,
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

    @staticmethod
    def _put_data_into_scheme(scheme, scheme_mess, other_mess):
        scheme = list(scheme)
        other = [None] * len(scheme)
        for i, o in enumerate(other_mess):
            try:
                j = scheme.index(float(scheme_mess[i]))
                other[j] = o
            except:
                pass
        return other

    def process_data(self):
        for qubit in self.all_analysis:
            self.proc_data_dict[qubit] = {}
            qo = self.all_analysis[qubit]
            # print(qo.keys())
            # collect all dac values
            all_dac = np.array([])
            for typ in qo:
                a = self.all_analysis[qubit][typ]
                all_dac = np.append(all_dac, np.array(a.raw_data_dict['dac'], dtype=float))
            all_dac = np.unique(all_dac)
            all_dac.sort()
            self.proc_data_dict[qubit]['all_dac'] = all_dac

            # Sort the measurement data (taus) into the dac values
            qubit_mask = np.array([True] * len(all_dac), dtype=bool)
            for typ in qo:
                a = self.all_analysis[qubit][typ]
                d = self.raw_data_dict[qubit][typ]
                self.proc_data_dict[qubit][typ] = {}

                sorted_taus = self._put_data_into_scheme(scheme=all_dac, scheme_mess=d['dac'],
                                                         other_mess=d['tau'])

                sorted_taus = np.array(sorted_taus)
                # sorted_chis = self._put_data_into_scheme(scheme=all_dac, scheme_mess=d['dac'],
                                                         # other_mess=d['chisquared'])
                # sorted_chis = np.array(sorted_chis)
                # thold = 0.5
                # mask = sorted_chis > thold
                # self.proc_data_dict[qubit][typ]['all_dac_sorted_chisquared_mask'] = mask
                # qubit_mask = (qubit_mask * 1 + mask * 1) == 2

                self.proc_data_dict[qubit][typ]['all_dac_sorted_tau'] = sorted_taus
                mask = np.equal(sorted_taus, None)
                self.proc_data_dict[qubit][typ]['all_dac_sorted_tau_mask'] = mask
                # print('premask', qubit_mask, mask)
                qubit_mask = (qubit_mask * 1 + mask * 1) == 2
                self.proc_data_dict[qubit]['all_dac']= all_dac[~qubit_mask]
                # print(qubit_mask)
                # self.option_dict.get('shall i')
                    # if yes, sort chi^2
                    # mask = np.where self.option_dict.get('threshold')
                    # qubit_mask = (qubit_mask * 1 + mask * 1) == 2
                    

                self.proc_data_dict[qubit][typ]['qubit_mask'] = qubit_mask
            self.proc_data_dict[qubit]['all_mask'] = qubit_mask

            # Calculate gamma = 1/Tau where appropriate
            for typ in qo:
                sorted_taus = self.proc_data_dict[qubit][typ]['all_dac_sorted_tau']
                mask = np.equal(sorted_taus, None)
                # print(typ)
                # print('sortedtau', sorted_taus)
                # print('mask', mask)
                # print('qubitmask', self.proc_data_dict[qubit][typ]['qubit_mask'])
                # print('calc', sorted_taus[~self.proc_data_dict[qubit][typ]['qubit_mask']])
                self.proc_data_dict[qubit][typ]['all_dac_sorted_gamma'] = 1.0 / sorted_taus[~mask]

            # print(self.raw_data_dict['timestamps'])
            gamma_1 = self.proc_data_dict[qubit][self.T1]['all_dac_sorted_gamma']
            gamma_ramsey = self.proc_data_dict[qubit][self.T2_star]['all_dac_sorted_gamma']
            gamma_echo = self.proc_data_dict[qubit][self.T2]['all_dac_sorted_gamma']
            gamma_phi_ramsey = (gamma_ramsey - gamma_1 / 2.0)
            gamma_phi_echo = (gamma_echo - gamma_1 / 2.0)
            # print('gamma', gamma_ramsey, gamma_echo, gamma_phi_ramsey, gamma_phi_echo)
            self.proc_data_dict[qubit]['gamma_phi_ramsey'] = gamma_phi_ramsey
            self.proc_data_dict[qubit]['gamma_phi_echo'] = gamma_phi_echo

    def run_fitting(self):
        # This is not the proper way to do this!
        # TODO: move this to prepare_fitting
        if self.freq_keys and self.dac_keys:
            for qubit in self.all_analysis:
                self.fit_res[qubit] = {}
                if self.verbose:
                    print("Fitting qubit: %s" % qubit)

                all_dac = self.proc_data_dict[qubit]['all_dac']
                for typ in self.all_analysis[qubit]:
                    self.fit_res[qubit][typ] = {}
                    a = self.all_analysis[qubit][typ]
                    # Sort the flux and sensitivity data
                    if self.freq_keys and self.dac_keys:
                        # sort the data by dac (just in case some samples are missing)
                        a.run_fitting()
                        self.fit_res[qubit][typ] = a.fit_res

                        sorted_sens = self._put_data_into_scheme(scheme=all_dac, scheme_mess=a.raw_data_dict['dac'],
                                                                 other_mess=a.fit_res['sensitivity_values'])
                        sorted_flux = self._put_data_into_scheme(scheme=all_dac, scheme_mess=a.raw_data_dict['dac'],
                                                                 other_mess=a.fit_res['flux_values'])
                        sorted_sens = np.array(sorted_sens, dtype=float)
                        sorted_flux = np.array(sorted_flux, dtype=float)
                        self.fit_res[qubit][typ]['sorted_sensitivity'] = sorted_sens
                        self.fit_res[qubit][typ]['sorted_flux'] = sorted_flux

                # these should all be the same for all types, so chose one (here T1)
                self.fit_res[qubit]['sorted_flux'] = self.fit_res[qubit][self.T1]['sorted_flux']
                self.fit_res[qubit]['sorted_sensitivity'] = self.fit_res[qubit][self.T1][
                    'sorted_sensitivity']

                # Make the PSD fit, if we have enough data
                exclusion_mask = self.proc_data_dict[qubit]['all_mask']
                masked_dac = all_dac #[~exclusion_mask]
                if len(masked_dac) > 4:
                    # Fit gamma vs sensitivity
                    sensitivity = self.fit_res[qubit]['sorted_sensitivity']
                    gamma_phi_ramsey = self.proc_data_dict[qubit]['gamma_phi_ramsey']
                    gamma_phi_echo = self.proc_data_dict[qubit]['gamma_phi_echo']
                    fit_res_gammas = fit_gammas(sensitivity=sensitivity, Gamma_phi_ramsey=gamma_phi_ramsey,
                                                Gamma_phi_echo=gamma_phi_echo, verbose=self.verbose)
                    self.fit_res[qubit]['fit_res'] = fit_res_gammas
                    intercept = fit_res_gammas.params['intercept'].value
                    slope_ramsey = fit_res_gammas.params['slope_ramsey'].value
                    slope_echo = fit_res_gammas.params['slope_echo'].value

                    # after fitting gammas
                    # from flux noise
                    # Martinis PRA 2003
                    sqrtA_rams = slope_ramsey / (np.pi * np.sqrt(30))
                    sqrtA_echo = slope_echo / (np.pi * np.sqrt(1.386))

                    if self.verbose:
                        print('Amplitude echo PSD = (%s u\Phi_0)^2' % (sqrtA_echo / 1e-6))
                        print('Amplitude rams PSD = (%s u\Phi_0)^2' % (sqrtA_rams / 1e-6))

                    chi = self.chi_shift[qubit] if self.chi_shift else None
                    res_freq = self.res_freq[qubit] if self.res_freq else None
                    res_Qc = self.res_Qc[qubit] if self.res_Qc else None

                    # from white noise
                    # using Eq 5 in Nat. Comm. 7,12964 (The flux qubit revisited to enhance
                    # coherence and reproducability)
                    if not ((res_freq is None) and (res_Qc is None) and (chi is None)):
                        n_avg = calculate_n_avg(res_freq, res_Qc, chi, intercept)
                        self.fit_res[qubit]['avg_noise_photons'] = n_avg
                        if self.verbose:
                            print('Estimated residual photon number: %s' % n_avg)

                    self.fit_res[qubit]['gamma_intercept'] = intercept
                    self.fit_res[qubit]['gamma_slope_ramsey'] = slope_ramsey
                    self.fit_res[qubit]['gamma_slope_echo'] = slope_echo
                    self.fit_res[qubit]['gamma_phi_ramsey_f'] = lambda x: slope_ramsey * x * 1e9 + intercept
                    self.fit_res[qubit]['gamma_phi_echo_f'] = lambda x: slope_echo * x * 1e9 + intercept
                    self.fit_res[qubit]['sqrtA_echo'] = (sqrtA_echo / 1e-6)
                    self.fit_res[qubit]['fit_res'] = fit_res_gammas

                    self.fit_res[qubit]['gamma_intercept_std'] = fit_res_gammas.params['intercept'].stderr
                    self.fit_res[qubit]['gamma_slope_ramsey_std'] = fit_res_gammas.params['slope_ramsey'].stderr
                    self.fit_res[qubit]['gamma_slope_echo_std'] = fit_res_gammas.params['slope_echo'].stderr


                else:
                    # fixme: make this a proper warning
                    print(
                        'Found %d dac values. I need at least 4 dac values to run the PSD analysis.' % len(masked_dac))
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
            cm = self.options_dict.get('current_multiplier', 1)

            dat = self.raw_data_dict[qubit]
            if hasattr(self, 'fit_res'):
                sensitivity = self.fit_res[qubit]['sorted_sensitivity']
                flux = self.fit_res[qubit]['sorted_flux']

                # freq = self.fit_res[qubit][self.T1]['qfreq']
                gamma_phi_echo = self.proc_data_dict[qubit]['gamma_phi_echo']
                gamma_phi_echo_f = self.fit_res[qubit]['gamma_phi_echo_f']
                gamma_phi_ramsey = self.proc_data_dict[qubit]['gamma_phi_ramsey']
                gamma_phi_ramsey_f = self.fit_res[qubit]['gamma_phi_ramsey_f']

                ############
                # coherence_gamma
                pdict_scatter = {
                    'xlabel': r'Sensitivity $|\partial\nu/\partial\Phi|$',
                    'xunit': r'GHz/$\Phi_0$',
                    'ylabel': r'$\Gamma_{\phi}$',
                    'yunit': r'$s^{-1}$',
                    'setlabel': '$\Gamma_{\phi,\mathrm{Ramsey}}$',
                }
                pdict_fit = {}
                cg_base = qubit + "_" + cg_all_base
                pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id=cg_base, xdata=np.abs(sensitivity) * 1e-9,
                                                     ydata=gamma_phi_ramsey, fitfunc=gamma_phi_ramsey_f,
                                                     xerr=None, yerr=None, fitextra=0.1, fitpoints=1000,
                                                     pdict_scatter=pdict_scatter, pdict_fit=pdict_fit)
                self.plot_dicts[cg_base + '_ramsey_fit'] = pdf
                self.plot_dicts[cg_base + '_ramsey_scatter'] = pds

                pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id=cg_base, xdata=np.abs(sensitivity) * 1e-9,
                                                     ydata=gamma_phi_echo, fitfunc=gamma_phi_echo_f,
                                                     xerr=None, yerr=None, fitextra=0.1, fitpoints=1000,
                                                     pdict_scatter=pdict_scatter, pdict_fit=pdict_fit)
                self.plot_dicts[cg_base + '_echo_fit'] = pdf
                self.plot_dicts[cg_base + '_echo_scatter'] = pds

                if self.options_dict.get('print_fit_result_plot', True):
                    dac_fit_text = '$\Gamma = %.5f(\pm %.5f)$\n' % (
                    self.fit_res[qubit]['gamma_intercept'], self.fit_res[qubit]['gamma_intercept_std'])
                    # dac_fit_text += '$\Gamma/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (self.fit_res[qubit]['gamma_intercept'], self.fit_res[qubit]['gamma_intercept_std'])
                    # dac_fit_text += '$\Gamma/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (self.fit_res[qubit]['gamma_intercept'], self.fit_res[qubit]['gamma_intercept_std'])

                    self.fit_res[qubit]['gamma_slope_ramsey_std']
                    self.fit_res[qubit]['gamma_slope_echo_std']

                    self.plot_dicts[cg_base + '_text_msg'] = {
                        'ax_id': cg_base,
                        # 'ypos': 0.15,
                        'plotfn': self.plot_text,
                        'box_props': 'fancy',
                        'text_string': dac_fit_text,
                    }

                ############
                # coherence_ratios
                cr_base = qubit + '_' + cr_all_base
                ratio_gamma = (gamma_phi_ramsey / gamma_phi_echo)

                pdict_scatter = {
                    'xlabel': 'Flux',
                    'xunit': 'm$\Phi_0$',
                    'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                }
                pds = plot_scatter_errorbar(self=self, ax_id=cr_all_base + '_flux',
                                            xdata=flux * 1e3,
                                            ydata=ratio_gamma,
                                            xerr=None, yerr=None,
                                            pdict=pdict_scatter)
                self.plot_dicts[cr_base + '_flux'] = pds

                pdict_scatter = {
                    'xlabel': r'Sensitivity $|\partial\nu/\partial\Phi|$',
                    'xunit': r'GHz/$\Phi_0$',
                    'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                }
                pds = plot_scatter_errorbar(self=self, ax_id=cr_all_base + '_sensitivity',
                                            xdata=np.abs(sensitivity) * 1e-9,
                                            ydata=ratio_gamma,
                                            xerr=None, yerr=None,
                                            pdict=pdict_scatter)
                self.plot_dicts[cr_base + '_sensitivity'] = pds

            ############
            # coherence_times
            ct_base = qubit + '_' + ct_all_base

            plot_types = ['time_stability', 'freq_relation', 'dac_relation', 'flux_relation', 'sensitivity_relation', ]
            ymax = [0] * len(plot_types)
            ymin = [0] * len(plot_types)
            markers = ('x', 'o', '+')
            for typi, typ in enumerate(self.all_analysis[qubit]):
                a = self.all_analysis[qubit][typ]
                a.prepare_plots()
                label = '%s_%s' % (qubit, typ)
                for pti, plot_type in enumerate(plot_types):  # 'dac_freq_relation

                    if plot_type in ['freq_relation', ]:
                        if self.freq_keys:
                            plot = True
                        else:
                            plot = False
                    elif plot_type in ['dac_relation', ]:
                        if self.dac_keys:
                            plot = True
                        else:
                            plot = False
                    elif plot_type in ['flux_relation', 'sensitivity_relation',
                                       'dac_freq_relation']:
                        if self.freq_keys and self.dac_keys:
                            plot = True
                        else:
                            plot = False
                    else:
                        plot = True

                    if plot:
                        if a.plot_dicts[plot_type]['yrange']:
                            ymin[pti] = min(ymin[pti], a.plot_dicts[plot_type]['yrange'][0])
                            ymax[pti] = max(ymax[pti], a.plot_dicts[plot_type]['yrange'][1])

                        key = ct_base + '_' + plot_type + '_' + typ
                        self.plot_dicts[key] = a.plot_dicts[plot_type]
                        self.plot_dicts[key]['ax_id'] = ct_base + '_' + plot_type
                        self.plot_dicts[key]['setlabel'] = label
                        self.plot_dicts[key]['do_legend'] = True
                        self.plot_dicts[key]['yrange'] = None
                        # self.plot_dicts[key]['xrange'] = None
                        self.plot_dicts[key]['marker'] = markers[typi % len(markers)]
                        if self.plot_dicts[key]['func'] == 'errorbar':
                            self.plot_dicts[key]['line_kws'] = {'fmt': markers[typi % len(markers)]}

                    if 'analysis' in dat and dat['analysis']:
                        if self.dac_keys and self.freq_keys:
                            pdict_scatter = {
                                'xlabel': r'Sensitivity $|\partial\nu/\partial\Phi|$',
                                'xunit': 'm$\Phi_0$',
                                'ylabel': '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$',
                                'setlabel': label
                            }
                            pds = plot_scatter_errorbar(self=self, ax_id=ct_base + '_flux_gamma_relation',
                                                        xdata=flux * 1e3,
                                                        ydata=dat['analysis'][typ],
                                                        xerr=None, yerr=None,
                                                        pdict=pdict_scatter)
                            self.plot_dicts[ct_base + '_flux_gamma_relation_' + typ] = pds
            for pti, plot_type in enumerate(plot_types):
                key = ct_base + '_' + plot_type + '_' + typ
                if key in self.plot_dicts:
                    self.plot_dicts[key]['yrange'] = [ymin[pti], ymax[pti]]
            # self.raw_data_dict[qubit][typ] = a.raw_data_dict


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
    d = np.abs(np.cos((np.pi * (dac - offset)) / dac0))
    model = np.sqrt(8 * Ec * Ej * d) - Ec

    return model


# define the model (from the function above) used to fit data
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
    arch_model.set_param_hint('Ec', value=260e6, min=100e6, max=350e6)
    arch_model.set_param_hint('Ej', value=19e9, min=0.1e9, max=30e9)
    arch_model.set_param_hint('offset', value=0, min=-0.05, max=0.05)
    arch_model.set_param_hint('dac0', value=0.1, min=0)

    arch_model.make_params()
    # print('freq, dac', freq, dac)
    fit_result_arch = arch_model.fit(freq, dac=dac)
    return fit_result_arch


def residual_Gamma(pars_dict, sensitivity, Gamma_phi_ramsey, Gamma_phi_echo):
    slope_ramsey = pars_dict['slope_ramsey']
    slope_echo = pars_dict['slope_echo']
    intercept = pars_dict['intercept']

    gamma_values_ramsey = slope_ramsey * np.abs(sensitivity) + intercept
    # print(len(Gamma_phi_ramsey), len(gamma_values_ramsey))
    residual_ramsey = Gamma_phi_ramsey - gamma_values_ramsey

    gamma_values_echo = slope_echo * np.abs(sensitivity) + intercept
    residual_echo = Gamma_phi_echo - gamma_values_echo

    return np.concatenate((residual_ramsey, residual_echo))


def fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo, verbose: bool = False):
    # create a parametrrer set for the initial guess
    p = lmfit.Parameters()
    p.add('slope_ramsey', value=100.0, vary=True)
    p.add('slope_echo', value=100.0, vary=True)
    p.add('intercept', value=100.0, vary=True)

    wrap_residual = lambda p: residual_Gamma(p,
                                             sensitivity=sensitivity,
                                             Gamma_phi_ramsey=Gamma_phi_ramsey,
                                             Gamma_phi_echo=Gamma_phi_echo)
    fit_result_gammas = lmfit.minimize(wrap_residual, p)
    if verbose:
        lmfit.printfuncs.report_fit(fit_result_gammas.params)
    return fit_result_gammas
