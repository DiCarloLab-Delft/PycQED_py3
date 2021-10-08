'''
Hacked together by Rene Vollmer
Cleaned up (a bit) by Adriaan
'''
import os
import matplotlib.pyplot as plt
import datetime
from collections import OrderedDict
from copy import deepcopy
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.utilities.general import SafeFormatter, format_value_string
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit,\
    plot_scatter_errorbar, set_xlabel, set_ylabel

import numpy as np
import lmfit
from pycqed.analysis.tools.plotting import SI_val_to_msg_str, \
    format_lmfit_par, plot_lmfit_res

from pycqed.analysis import analysis_toolbox as a_tools


class CoherenceAnalysis(ba.BaseDataAnalysis):
    """
    Power spectral density analysis of transmon coherence.

    Note, analysis of the coherence times is separated from data extraction
    as that can be a complicated process that is highly experiment dependent.

    Args:
        coherence_table : table containing the data, see below for
            specification.
        freq_resonator: readout resonator frequency (in Hz)
        Qc:             coupling Q of the readout resonator
        chi_shift       in Hz
        savename:       figures are saved in a folder created based on the
            provided t_stop in the datadir. `savename` is used to name
            this folder.


    Coherence_table specification:
           Row  | Content (arrays)
        --------+--------
            1   | dac
            2   | frequency (Hz)
            3   | T1 (s)
            4   | T2 star (s)
            5   | T2 echo (s)
            6   | Exclusion mask (True where data is to be excluded)

    Generates 8 plots:
        > T1, T2, Echo vs flux
        > T1, T2, Echo vs frequency
        > T1, T2, Echo vs flux sensitivity
        > ratio Ramsey/Echo vs flux
        > ratio Ramsey/Echo vs frequency
        > ratio Ramsey/Echo vs flux sensitivity
        > Dephasing rates Ramsey and Echo vs flux sensitivity
        > Dac arc fit, use to assess if sensitivity calculation is correct.

    If properties of resonator are provided (freq_resonator, Qc, chi_shift),
    it also calculates the number of noise photons.
    """

    def __init__(self, coherence_table,
                 t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto: bool=True, close_figs=True,
                 freq_resonator: float=None, Qc: float=None,
                 chi_shift: float=None, savename: str= 'coherence_analysis',
                 **kwargs):

        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs,
                         **kwargs)
        if t_stop is not None:
            self.options_dict['save_figs'] = False
        self._coherence_table = coherence_table
        self._freq_resonator = freq_resonator
        self._Qc = Qc
        self._chi_shift = chi_shift

        if auto:
            self.run_analysis()

    def extract_data(self):
        """Put data from the coherence table in the raw data dict."""
        self.raw_data_dict = OrderedDict()
        rdd = self.raw_data_dict
        rdd['dac'], rdd['freq'], rdd['T1'], rdd['Tramsey'],\
            rdd['Techo'], rdd['exclusion_mask'] = self._coherence_table
        # Externally loaded tables my be cast to ints.
        rdd['exclusion_mask'] = rdd['exclusion_mask'].astype(bool)

        # Resonator information
        rdd['freq_resonator'] = self._freq_resonator
        rdd['Qc'] = self._Qc
        rdd['chi_shift'] = self._chi_shift

        # Extra info for datasaving etc.
        rdd['timestamps'] = [self.t_start, self.t_stop]
        t_stop = self.t_stop.split('_')
        folder = os.path.join(
            a_tools.datadir, t_stop[0],
            '{}_coherence_analysis'.format(t_stop[1]))
        self.raw_data_dict['folder'] = [folder]
        print(folder)

    def process_data(self):
        """
        Process data.

        Performs the following:
            - fit dac-arc
            - convert dac to flux
            - calculate sensitivity δf/δΦ
            - calculate dephasing rates Γ
            - fit dephasing rates
            - determine derived quantities
        """
        self.fit_res = OrderedDict()
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        pdd = self.proc_data_dict

        # Extract the dac arcs required for getting the sensitivities
        # FIXME: add a proper guess function to make the fit more robust
        self.fit_res['dac_arc'] = fit_frequencies(pdd['dac'], pdd['freq'],
                                                  dac0_guess=.1)

        # convert dac in flux as unit of Phi_0
        flux = (pdd['dac'] - self.fit_res['dac_arc'].best_values['offset'])\
            / self.fit_res['dac_arc'].best_values['dac0']
        pdd['flux'] = flux

        # calculate the derivative vs flux
        sensitivity_angular = partial_omega_over_flux(
            flux, self.fit_res['dac_arc'].best_values['Ec'],
            self.fit_res['dac_arc'].best_values['Ej'])
        pdd['sensitivity'] = sensitivity_angular/(2*np.pi)

        # Calculate pure dephasing rates
        pdd['Gamma_phi_ramsey'] = calc_dephasing_rate(
            T1=pdd['T1'], T2=pdd['Tramsey'])[~pdd['exclusion_mask']]
        pdd['Gamma_phi_echo'] = calc_dephasing_rate(
            T1=pdd['T1'], T2=pdd['Techo'])[~pdd['exclusion_mask']]

        # Often, flux noise is well described by a power spectral density
        # PSD of the form S_Φ = A*1/f (single sided), where A is a scaling
        # factor (units of Φ_0^2).

        # Fit dephasing rates as a function of sensitivity to a linear model.
        self.fit_res['gammas'] = fit_gammas(
            pdd['sensitivity'], pdd['Gamma_phi_ramsey'], pdd['Gamma_phi_echo'])

        # There is special significance in the slope and the intercept of
        # these distributions.
        # The slope can be related to the magnitude (scale factor) of the
        # flux noise. Both are typically expressed in units of μΦ_0.
        pdd['slope_ramsey'] = \
            self.fit_res['gammas'].params['slope_ramsey'].value
        pdd['slope_echo'] = self.fit_res['gammas'].params['slope_echo'].value
        # The intercept with the y-axis (zero sensitivity) relates to the
        # flux insensitive contribution to the spectrum. The flux insensitive
        # contribution can be linked to the photon number in the resonator.
        # White noise shows up as a quadratic
        pdd['intercept'] = self.fit_res['gammas'].params['intercept'].value

        # Based on Martinis PRB 2003. Magic numbers are found by evaluating
        # integral of eq. 10 with different filter spectra.
        # filter spectrum of Ramsey is described in eq. 11, Echo in eq. 35
        # Note that magic numbers are for *single* sided PSDs.
        pdd['sqrtA_rams'] = pdd['slope_ramsey']/(np.pi*np.sqrt(30))
        pdd['sqrtA_echo'] = pdd['slope_echo']/(np.pi*np.sqrt(1.386))
        # Note: both methods that are expected to give the same number.
        # More details can be found in Luthi PRL (2018) and month report M11

        # from white noise
        # using Eq 5 Yan et al. Nat. Comm. 7,12964  (The flux qubit revisited
        # to enhance coherence and reproducability)
        if not ((pdd['freq_resonator'] is None) and (pdd['Qc'] is None)
                and (pdd['chi_shift'] is None)):
            pdd['n_avg'] = calculate_n_avg(pdd['freq_resonator'], pdd['Qc'],
                                           pdd['chi_shift'], pdd['intercept'])
            print('Estimated residual photon number: %s' % pdd['n_avg'])
        self._create_gamma_message()

    def _create_gamma_message(self):
        pdd = self.proc_data_dict
        text_msg = 'Summary: \n'

        text_msg += 'Slope Ramsey: {:.2f} {}\n'.format(*SI_val_to_msg_str(
            pdd['slope_ramsey'], unit=r'$\Phi_0$', return_type=float))
        text_msg += 'Slope echo: {:.2f} {}\n'.format(*SI_val_to_msg_str(
            pdd['slope_echo'], unit=r'$\Phi_0$', return_type=float))
        text_msg += r'$\sqrt{A}$' + \
            ' Ramsey: {:.2f} {}\n'.format(*SI_val_to_msg_str(
                pdd['sqrtA_rams'], unit=r'$\Phi_0$', return_type=float))
        text_msg += r'$\sqrt{A}$'+'echo: {:.2f} {}\n'.format(*SI_val_to_msg_str(
            pdd['sqrtA_echo'], unit=r'$\Phi_0$', return_type=float))

        if 'n_avg' in self.proc_data_dict.keys():
            text_msg += r'$n_\gamma$'+': {:.2f} {}\n'.format(*SI_val_to_msg_str(
                pdd['n_avg'], unit='', return_type=float))
        self.proc_data_dict['gamma_fit_msg'] = text_msg

    def prepare_fitting(self):
        pass

    def run_fitting(self):
        # Overwritten to prevent overwriting fit_res dict.
        pass

    def analyze_fit_results(self):
        pass

    def prepare_plots(self):
        """
        Prepare data for the plots.

        The plots that are created are:
            - dac-arc fit
            - coherence times (3x)
            - dephasing ratios (3x)
            - dephasing rates fit
        """
        self.plot_dicts['dac_arc'] = {
            'plotfn': plot_dac_arc,
            'dac': self.proc_data_dict['dac'],
            'freq': self.proc_data_dict['freq'],
            'fit_res': self.fit_res['dac_arc']
        }

        fs = plt.rcParams['figure.figsize']

        # # define figure and axes here to have custom layout
        self.figs['coherence_times'], axs = plt.subplots(
            nrows=1, ncols=3, sharey=True, figsize=(fs[0]*3, fs[1]))
        self.figs['coherence_times'].patch.set_alpha(0)

        self.axs['coherence_flux'] = axs[0]
        self.axs['coherence_freq'] = axs[1]
        self.axs['coherence_sens'] = axs[2]

        self.plot_dicts['coherence_times'] = {
            'plotfn': plot_coherence_times,
            'axs': [axs[0], axs[1], axs[2]],
            'ax_id': 'coherence_flux',
            'flux': self.proc_data_dict['flux'],
            'freq': self.proc_data_dict['freq'],
            'sensitivity': self.proc_data_dict['sensitivity'],
            'T1': self.proc_data_dict['T1'],
            'Tramsey': self.proc_data_dict['Tramsey'],
            'Techo': self.proc_data_dict['Techo'],
        }

        self.figs['coherence_ratios'], axs = plt.subplots(
            nrows=1, ncols=3, sharey=True, figsize=(fs[0]*3, fs[1]))
        self.figs['coherence_ratios'].patch.set_alpha(0)

        self.axs['ratios_flux'] = axs[0]
        self.axs['ratios_freq'] = axs[1]
        self.axs['ratios_sens'] = axs[2]

        self.plot_dicts['coherence_ratios'] = {
            'plotfn': plot_ratios,
            'axs': [axs[0], axs[1], axs[2]],
            'ax_id': 'ratios_flux',
            'flux': self.proc_data_dict['flux'],
            'freq': self.proc_data_dict['freq'],
            'sensitivity': self.proc_data_dict['sensitivity'],
            'Gamma_phi_ramsey': self.proc_data_dict['Gamma_phi_ramsey'],
            'Gamma_phi_echo': self.proc_data_dict['Gamma_phi_echo'],
        }

        self.plot_dicts['gamma_fit'] = {
            'plotfn': plot_gamma_fit,
            'sensitivity': self.proc_data_dict['sensitivity'],
            'Gamma_phi_ramsey': self.proc_data_dict['Gamma_phi_ramsey'],
            'Gamma_phi_echo': self.proc_data_dict['Gamma_phi_echo'],
            'slope_echo': self.proc_data_dict['slope_echo'],
            'slope_ramsey': self.proc_data_dict['slope_ramsey'],
            'intercept': self.proc_data_dict['intercept'],
            'freq': self.proc_data_dict['freq'],
            'fit_res': self.fit_res['gammas']
        }
        self.plot_dicts['gamma_msg'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['gamma_fit_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'gamma_fit',
            'horizontalalignment': 'left'}


class CoherenceTimesAnalysisSingle(ba.BaseDataAnalysis):
    """
    Plots and Analyses the coherence time (e.g. T1, T2 OR T2*) of one measurement series.
    Makes several plots, such as Tx vs time, Tx vs frequency, Tx vs flux, freq vs current etc.
    Ferforms only up to two fits:
        - frequency vs current (used to concert current to flux and to calculate
            flux sensitivity |df/dI|; if requested)
        - A/f to Tx vs frequency (to extract Q-factor from T1 measurements; if requested)

    Args:
        t_start (str):
            start time of scan as a string of format YYYYMMDD_HHmmss

        t_stop (str):
            end time of scan as a string of format YYYYMMDD_HHmmss

        label (str):
            the label that was used to name the measurements
            (only necessary if non-relevant measurements are in the time range)

        options_dict (dict):
            Guesses of resonator frequencits for purcell effect can be specified:
                'guess_mode_frequency': float OR
                'guess_mode_frequency': [float, float]
            only up to two modes are supported at this point.
            ...
            Other available options are the ones from the base_analysis and:
                                - (todo)

        auto (bool):
            Execute all steps automatically

        close_figs (bool):
            Close the figure (do not display)

        extract_only (bool):
            Should we also do the plots?

        do_fitting (bool):
            Should the run_fitting method be executed?

        fit_T1_vs_freq (bool):
            Should fitting T1 vs freq be done?
            in options dict. If not specified only 1/f dependence is fitted.

        tau_key (str):
            key for the tau (time) fit result, e.g. 'Analysis.Fitted Params F|1>.tau.value'

        tau_std_key (str):
            key for the tau (time) standard deviation fit result,
            e.g. 'Analysis.Fitted Params F|1>.tau.stderr'

        plot_versus_dac (bool): Extract and plot dac value?
            E.g. set False if you did not vary the dac for this measurement.

        dac_key (str):
            key for the dac current values, e.g. 'Instrument settings.fluxcurrent.Q'

        plot_versus_frequency (bool):
            Extract and plot frequency value?
            E.g. set False if you did not use the Qubit object.

        frequency_key (bool):
            key for the dac current values, e.g. 'Instrument settings.Q.freq_qubit'
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False, auto: bool=True,
                 close_figs: bool=True, do_fitting: bool=True,
                 tau_key='Analysis.Fitted Params F|1>.tau.value',
                 tau_std_key='Analysis.Fitted Params F|1>.tau.stderr',
                 use_chisqr=False,
                 plot_versus_dac=True,
                 dac_key='Instrument settings.fluxcurrent.Q',
                 plot_versus_frequency=True,
                 frequency_key='Instrument settings.Q.freq_qubit',
                 fit_T1_vs_freq=False,
                 mean_and_std=False
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)
        # self.single_timestamp = False

        if use_chisqr:
            if 'F|1>' in tau_key:
                chisquared_key = 'Analysis.Fitted Params F|1>.chisqr'
            elif 'raw' in tau_key:
                chisquared_key = 'Analysis.Fitted Params raw w0.chisqr'
            elif 'corr_data' in tau_key:
                chisquared_key = 'Analysis.Fitted Params corr_data.chisqr'
            self.params_dict = {'tau': tau_key,
                                'tau_stderr': tau_std_key,
                                'chisquared': chisquared_key
                                }
            self.numeric_params = ['tau', 'tau_stderr', 'chisquared']
        else:
            self.params_dict = {'tau': tau_key,
                                'tau_stderr': tau_std_key,
                                # 'chisquared' : chisquared_key
                                }
            self.numeric_params = ['tau', 'tau_stderr']  # , 'chisquared'

        self.fit_T1_vs_freq = fit_T1_vs_freq

        self.plot_versus_dac = plot_versus_dac
        if plot_versus_dac:
            self.params_dict['dac'] = dac_key

        self.plot_versus_frequency = plot_versus_frequency
        if plot_versus_frequency:
            self.params_dict['qfreq'] = frequency_key

        self.numeric_params = []

        self.mean_and_std = mean_and_std

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
                # FIXME hardcoded guess should be gone!
                fit_object = fit_frequencies(dac=dac, freq=freq, dac0_guess=.1)
                self.fit_res['dac_arc_object'] = fit_object
                self.fit_res['dac_arc_fitfct'] = lambda x: fit_object.model.eval(
                    fit_object.params, dac=x)

                # convert dac in flux as unit of Phi_0
                flux = (
                    dac - fit_object.best_values['offset']) / fit_object.best_values['dac0']
                self.fit_res['flux_values'] = flux

                # calculate the derivative vs flux
                sensitivity_angular = partial_omega_over_flux(flux, fit_object.best_values['Ec'],
                                                              fit_object.best_values['Ej'])
                self.fit_res['Ec'] = fit_object.best_values['Ec']
                self.fit_res['Ej'] = fit_object.best_values['Ej']
                self.fit_res['sensitivity_values'] = sensitivity_angular / \
                    (2 * np.pi)
                if self.verbose:
                    # todo: print EC and EJ
                    pass

            # Fit T1 vs frequency dependence using model of decay due to dielectric losses
            # and (if 'guess_mode_frequency' are specified in options dict) due to
            # purcell effect
            # Refs:
            #     Luthi PRL 2018
            #     Krantz arXiv:1904.06560 pp. 48
            if self.fit_T1_vs_freq:
                guess_freq = self.options_dict.get('guess_mode_frequency', None)

                freq = self.raw_data_dict['dac_sorted_freq']
                tau = self.raw_data_dict['dac_sorted_tau']

                if guess_freq is None:
                    fit_object_T1_vs_freq = fit_fixed_Q_factor(freq, tau)
                    self.fit_res['Q_qubit'] = fit_object_T1_vs_freq.best_values['Q']
                    msg = '$Q$={:.3g}'.format(self.fit_res['Q_qubit'])
                elif type(guess_freq) is float:
                    fit_object_T1_vs_freq = fit_T1_purcell_diel(freq, tau, guess_freq)
                    self.fit_res['Q_qubit'] = fit_object_T1_vs_freq.best_values['Q']
                    self.fit_res['fres'] = fit_object_T1_vs_freq.best_values['fres']
                    self.fit_res['gres'] = np.abs(fit_object_T1_vs_freq.best_values['g'])
                    self.fit_res['kappares'] = fit_object_T1_vs_freq.best_values['kappa']
                    msg = '$Q$={:.3g} \n'.format(self.fit_res['Q_qubit'])
                    msg += '$f$={:.3g} GHz\n'.format(self.fit_res['fres']/1e9)
                    msg += '$g$={:.3g} MHz\n'.format(self.fit_res['gres']/1e6)
                    msg += '$\kappa$={:.3g} MHz'.format(self.fit_res['kappares']/1e6)
                else:
                    fit_object_T1_vs_freq = fit_T1_purcell_diel(freq, tau, guess_freq)
                    self.fit_res['Q_qubit'] = fit_object_T1_vs_freq.best_values['Q']
                    self.fit_res['fres1'] = fit_object_T1_vs_freq.best_values['fres1']
                    self.fit_res['gres1'] = np.abs(fit_object_T1_vs_freq.best_values['g1'])
                    self.fit_res['kappares1'] = fit_object_T1_vs_freq.best_values['kappa1']
                    self.fit_res['fres2'] = fit_object_T1_vs_freq.best_values['fres2']
                    self.fit_res['gres2'] = np.abs(fit_object_T1_vs_freq.best_values['g2'])
                    self.fit_res['kappares2'] = fit_object_T1_vs_freq.best_values['kappa2']
                    msg = '$Q$={:.3g} \n'.format(self.fit_res['Q_qubit'])
                    msg += '$f_1$={:.3g} GHz\n'.format(self.fit_res['fres1']/1e9)
                    msg += '$g_1$={:.2g} MHz\n'.format(self.fit_res['gres1']/1e6)
                    msg += '$\kappa_1$={:.2g} MHz\n'.format(self.fit_res['kappares1']/1e6)
                    msg += '$f_2$={:.3g} GHz\n'.format(self.fit_res['fres2']/1e9)
                    msg += '$g_2$={:.2g} MHz\n'.format(self.fit_res['gres2']/1e6)
                    msg += '$\kappa_2$={:.2g} MHz'.format(self.fit_res['kappares2']/1e6)

                self.proc_data_dict['freq_relation_msg'] = msg

                self.fit_res['Q_qubit_fitfct'] = lambda x: fit_object_T1_vs_freq.model.eval(
                    fit_object_T1_vs_freq.params, freq=x)
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
            if self.mean_and_std:
                if 'T1' in self.labels[0]:
                    measured_param = '$T_1$'
                elif 'Ramsey' in self.labels[0]:
                    measured_param = '$T_2^*$'
                elif 'echo' in self.labels[0]:
                    measured_param = '$T_2^{echo}$'
                else:
                    measured_param = 'Coherence'
                t_msg = measured_param+' = {:.1f}$\pm${:.1f} $\mu$s'.format(np.mean(self.raw_data_dict['tau'])*1e6,
                                                   np.std(self.raw_data_dict['tau'])*1e6)
                self.plot_dicts['mean_and_std'] = {
                'plotfn': self.plot_text,
                'text_string': t_msg,
                'xpos': 0.05, 'ypos': 0.05, 'ax_id': 'time_stability',
                'horizontalalignment': 'left', 'verticalalignment': 'bottom'}

            if self.plot_versus_frequency and self.fit_T1_vs_freq:
                plot_dict = {
                    'xlabel': 'Qubit Frequency', 'xunit': 'Hz',
                    'ylabel': 'T1', 'yunit': 's'
                }
                pds, pdf = plot_scatter_errorbar_fit(self=self, ax_id='freq_relation',
                                                     xdata=self.raw_data_dict['freq_sorted'],
                                                     ydata=self.raw_data_dict['freq_sorted_tau'],
                                                     yerr=self.raw_data_dict['freq_sorted_tau_stderr'],
                                                     fitfunc=self.fit_res['Q_qubit_fitfct'],
                                                     pdict_scatter=plot_dict, pdict_fit=plot_dict)
                self.plot_dicts["freq_relation_scatter"] = pds
                self.plot_dicts["freq_relation_fit"] = pdf

                self.plot_dicts['freq_relation_text'] = {
                    'plotfn': self.plot_text,
                    'text_string': self.proc_data_dict['freq_relation_msg'],
                    'xpos': 1.05, 'ypos': .6, 'ax_id': 'freq_relation',
                    'horizontalalignment': 'left'}
            else:
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
        if 'T1' in self.labels[0]:
            ylab = '$T_1$'
        else:
            ylab = 'Coherence'
        plot_dict = {
            'xlabel': xlabel,
            'xunit': xunit,
            'ylabel': ylab,
            'yrange': (0, 1.1 * np.max(yvals)),
            'yunit': 's',
            'marker': 'x',
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


class AliasedCoherenceTimesAnalysisSingle(ba.BaseDataAnalysis):
    """
    Analysis for aliased Ramsey type experiments.

    Assumes final measurement is performed in both the x and y-basis.
    """

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '', data_file_path: str = None,
                 options_dict: dict = None, extract_only: bool = False,
                 do_fitting: bool = True, auto=True,
                 ch_idxs: list= [0, 1],
                 vary_offset: bool=True):
        """

        Args:
            ch_idxs (list): correspond to column containing data in the x- and
                y-basis respectively. If the figure shows no signal, be sure
                to check this setting.
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'detuning': 'Experimental Data.Experimental Metadata.sq_eps',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = ['detuning']
        self.ch_idxs = ch_idxs
        self.vary_offset = vary_offset
        if auto:
            self.run_analysis()


    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        xs = self.raw_data_dict['measured_values'][0][self.ch_idxs[0]]
        ys = self.raw_data_dict['measured_values'][0][self.ch_idxs[1]]

        mn = (np.mean(xs) + np.mean(ys))/2
        self.proc_data_dict['mean'] = mn
        amp = np.sqrt((xs-mn)**2 + (ys-mn)**2)*2
        self.proc_data_dict['amp'] = amp

    def run_fitting(self):
        super().run_fitting()

        decay_fit = lmfit.Model(lambda t, tau, A, n,
                                o: A*np.exp(-(t/tau)**n)+o)

        tau0 = self.raw_data_dict['xvals'][0][-1]/3
        decay_fit.set_param_hint('tau', value=tau0, min=0, vary=True)
        decay_fit.set_param_hint('A', value=0.7, vary=True)
        decay_fit.set_param_hint('n', value=1.2, min=1, max=2, vary=True)
        decay_fit.set_param_hint(
            'o', value=0.01, min=0, max=0.3, vary=self.vary_offset)
        params = decay_fit.make_params()
        decay_fit = decay_fit.fit(data=self.proc_data_dict['amp'],
                                  t=self.raw_data_dict['xvals'][0],
                                  params=params)
        self.fit_res['coherence_decay'] = decay_fit

        text_msg = 'Summary\n'

        det, unit = SI_val_to_msg_str(self.raw_data_dict['detuning'][0], 'Hz',
                                      return_type=float)
        text_msg += 'Square pulse detuning {:.3f} {}\n'.format(det, unit)

        text_msg += r'Fitting to : $A e^{(-(t/\tau)^n)}+o$' + '\n\t'
        text_msg += format_lmfit_par(r'$A$', decay_fit.params['A'], '\n\t')
        text_msg += format_lmfit_par(r'$\tau$',
                                     decay_fit.params['tau'], '\n\t')
        text_msg += format_lmfit_par(r'$n$', decay_fit.params['n'], '\n\t')
        text_msg += format_lmfit_par(r'$o$', decay_fit.params['o'], '')

        self.proc_data_dict['decay_fit_msg'] = text_msg


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['xvals'][0],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': 's',
            'yvals': self.proc_data_dict['amp'],
            'ylabel': r'$\sqrt{\langle \sigma_X \rangle^2 + \langle \sigma_Y \rangle^2}$',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'setlabel': 'data',
            'color': 'C0',
        }

        self.plot_dicts['decay_fit'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main',
            'fit_res': self.fit_res['coherence_decay'],
            'setlabel': 'Decay fit',
            'do_legend': True,
            'color': 'C1',
        }

        self.plot_dicts['decay_text'] = {
            'plotfn': self.plot_text,
            'ax_id': 'main',
            'text_string': self.proc_data_dict['decay_fit_msg'],
            'xpos': 1.05, 'ypos': .5,
            'horizontalalignment': 'left'}


class CoherenceTimesAnalysis_old(ba.BaseDataAnalysis):
    """
    Old version of Coherence times analysis.

    This is in essence a messy combination of data extraction and the PSD
    analysis of the coherence.

    I am now separating these two out so that it is easier to use.
    """
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
                 do_fitting: bool = True, close_figs: bool = True, use_chisqr=False,
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

        # Check data and apply default values
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
            self.T2_star: '_Ramsey' + s,
        }

        assert (len(tau_keys) == len(labels))
        assert (len(tau_keys) == len(tau_std_keys))
        assert (len(tau_keys) >= 3)

        req = (self.T1, self.T2, self.T2_star)
        if not all(k in tau_keys for k in req):
            raise KeyError("You need to at least specify ",
                           req, " for parameters tau_keys.")
        if not all(k in tau_std_keys for k in req):
            raise KeyError("You need to at least specify ",
                           req, " for parameters tau_std_keys.")
        if not all(k in labels for k in req):
            raise KeyError("You need to at least specify ",
                           req, " for parameters labels.")

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
                dac_key = self._parse(
                    dac=dac_instr_name, qubit=qubit_instr_name, pattern=dac_key_pattern)
                self.dac_keys.append(dac_key)
            if plot_versus_frequency:
                freq_key = self._parse(
                    dac=dac_instr_name, qubit=qubit_instr_name, pattern=frequency_key_pattern)
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
                    label=self._parse(dac=dac_instr_name,
                                      qubit=qubit, pattern=labels[typ]),
                    auto=False, extract_only=True,
                    tau_key=tau_key,
                    tau_std_key=tau_std_key,
                    plot_versus_dac=plot_versus_dac,
                    dac_key=dac_key,
                    plot_versus_frequency=plot_versus_frequency,
                    frequency_key=freq_key,
                    options_dict=options_dict,
                    close_figs=close_figs,
                    use_chisqr=use_chisqr
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
        folder = a_tools.datadir + \
            '/%s_coherence_analysis' % (youngest.strftime("%Y%m%d/%H%M%S"))
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
                all_dac = np.append(all_dac, np.array(
                    a.raw_data_dict['dac'], dtype=float))
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
                self.proc_data_dict[qubit]['all_dac'] = all_dac[~qubit_mask]
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
                self.proc_data_dict[qubit][typ]['all_dac_sorted_gamma'] = 1.0 / \
                    sorted_taus[~mask]

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

                        sorted_sens = self._put_data_into_scheme(scheme=all_dac, scheme_mess=a.raw_data_dict['dac_sorted'],
                                                                 other_mess=a.fit_res['sensitivity_values'])
                        sorted_flux = self._put_data_into_scheme(scheme=all_dac, scheme_mess=a.raw_data_dict['dac_sorted'],
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
                masked_dac = all_dac  # [~exclusion_mask]
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
                        print('Amplitude echo PSD = (%s u\Phi_0)^2' %
                              (sqrtA_echo / 1e-6))
                        print('Amplitude rams PSD = (%s u\Phi_0)^2' %
                              (sqrtA_rams / 1e-6))

                    chi = self.chi_shift[qubit] if self.chi_shift else None
                    res_freq = self.res_freq[qubit] if self.res_freq else None
                    res_Qc = self.res_Qc[qubit] if self.res_Qc else None

                    # from white noise
                    # using Eq 5 in Nat. Comm. 7,12964 (The flux qubit revisited to enhance
                    # coherence and reproducability)
                    if not ((res_freq is None) and (res_Qc is None) and (chi is None)):
                        n_avg = calculate_n_avg(
                            res_freq, res_Qc, chi, intercept)
                        self.fit_res[qubit]['avg_noise_photons'] = n_avg
                        if self.verbose:
                            print('Estimated residual photon number: %s' % n_avg)

                    self.fit_res[qubit]['gamma_intercept'] = intercept
                    self.fit_res[qubit]['gamma_slope_ramsey'] = slope_ramsey
                    self.fit_res[qubit]['gamma_slope_echo'] = slope_echo
                    self.fit_res[qubit]['gamma_phi_ramsey_f'] = lambda x: slope_ramsey * \
                        x * 1e9 + intercept
                    self.fit_res[qubit]['gamma_phi_echo_f'] = lambda x: slope_echo * \
                        x * 1e9 + intercept
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
            print(
                'You have to enable plot_versus_frequency and plot_versus_dac to execute the PSD analysis.')

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
                    'yunit': r'Hz',
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
                        'xpos': 0.6,
                        'ypos': 0.95,
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

            plot_types = ['time_stability', 'freq_relation',
                          'dac_relation', 'flux_relation', 'sensitivity_relation', ]
            ymax = [0] * len(plot_types)
            ymin = [0] * len(plot_types)
            markers = ('x', 'o', '+')
            for typi, typ in enumerate(self.all_analysis[qubit]):
                a = self.all_analysis[qubit][typ]
                a.prepare_plots()
                label = '%s_%s' % (qubit, typ)
                # 'dac_freq_relation
                for pti, plot_type in enumerate(plot_types):

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
                            ymin[pti] = min(
                                ymin[pti], a.plot_dicts[plot_type]['yrange'][0])
                            ymax[pti] = max(
                                ymax[pti], a.plot_dicts[plot_type]['yrange'][1])

                        key = ct_base + '_' + plot_type + '_' + typ
                        self.plot_dicts[key] = a.plot_dicts[plot_type]
                        self.plot_dicts[key]['ax_id'] = ct_base + \
                            '_' + plot_type
                        self.plot_dicts[key]['setlabel'] = label
                        self.plot_dicts[key]['do_legend'] = True
                        self.plot_dicts[key]['yrange'] = None
                        # self.plot_dicts[key]['xrange'] = None
                        self.plot_dicts[key]['marker'] = markers[typi % len(
                            markers)]
                        if self.plot_dicts[key]['func'] == 'errorbar':
                            self.plot_dicts[key]['line_kws'] = {
                                'fmt': markers[typi % len(markers)]}

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
                            self.plot_dicts[ct_base +
                                            '_flux_gamma_relation_' + typ] = pds
            for pti, plot_type in enumerate(plot_types):
                key = ct_base + '_' + plot_type + '_' + typ
                if key in self.plot_dicts:
                    self.plot_dicts[key]['yrange'] = [ymin[pti], ymax[pti]]
            # self.raw_data_dict[qubit][typ] = a.raw_data_dict


class CoherenceAnalysisDataExtractor(ba.BaseDataAnalysis):
    """
    Extractor of dac values, frequency, T1, T2star and T2echo for use in
    CoherenceAnalysis.
    """
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
             options_dict: dict=None, auto: bool=True, close_figs=True,
             params_dict_TD: dict=None, numeric_params: list=None,
             chi_shift: float=None, savename: str= 'coherence_analysis',
             **kwargs):

        # super().__init__(t_start=t_start, t_stop=t_stop, label=label,
        #                  options_dict=options_dict, close_figs=close_figs,
        #                  **kwargs)

        dac_key = 'Instrument settings.fluxcurrent.FBL_X'
        frequency_key = 'Instrument settings.X.freq_qubit'


        T1_analysis = CoherenceTimesAnalysisSingle(t_start=self.t_start,t_stop=self.t_stop,
            label='T1',options_dict=options_dict,extract_only=True,do_fitting=False,
            tau_key='Analysis.Fitted Params F|1>.tau.value',
             tau_std_key='Analysis.Fitted Params F|1>.tau.stderr',
             use_chisqr=False,
             plot_versus_dac=False,
             dac_key=dac_key,
             plot_versus_frequency=False,
             frequency_key=frequency_key,
             fit_T1_vs_freq=False)








#             if 'F|1>' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params F|1>.chisqr'
#             elif 'raw' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params raw w0.chisqr'
#             elif 'corr_data' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params corr_data.chisqr'
#             self.params_dict = {'tau_T1': tau_key,
#                                 'tau_stderr': tau_std_key,
#                                 'chisquared': chisquared_key
#                                 }
#             self.numeric_params = ['tau', 'tau_stderr', 'chisquared']

#         if params_dict_TD is None:
#             params_dict_TD = {'tau':'Fitted Params F|1>.tau.value',
#                       'tau_err':'Fitted Params F|1>.tau.stderr',
#                       'chi_squared_reduced':'Fitted Params F|1>.redchi',
#                       'dac':'IVVI.dac1', # We do nothing with dac
#                       'frequency':qubit_label+'.f_qubit',
#                       'plot_data':'Corrected data',
#                       'plot_times':'sweep_points',
# #                       'temperatures':'LaDucati.temperatures', # Does not exist any more
#                       'field':'Magnet.field'}

#         if numeric_params is None:
#             numeric_params = ['tau','tau_err','dac','frequency',
#             'plot_data','plot_times','Attenuation','field']

#         if t_stop is not None:
#             self.options_dict['save_figs'] = False
#         self._coherence_table = coherence_table
#         self._freq_resonator = freq_resonator
#         self._Qc = Qc
#         self._chi_shift = chi_shift

#         if auto:
#             self.run_analysis()

#     # Define a function that extracts all the T1, Ramsey and Echo
#     # def extract_data(self,t_start,t_stop,T1_label,TEcho_label,TRamsey_label,
#     #                     params_dict_TD,numeric_params,filter_no_analysis):
#     #     '''
#     #     Extract all the T1, TEcho and TRamsey data in the timestamp range
#     #     '''
#     #     T1_timestamps = a_tools.get_timestamp_in_range(t_start,t_stop,label=T1_label)
#     #     TEcho_timestamps = a_tools.get_timestamp_in_range(t_start,t_stop,label=TEcho_label)
#     #     TRamsey_timestamps = a_tools.get_timestamp_in_range(t_start,t_stop,label=TRamsey_label)
#     #     self.T1_dict = a_tools.get_data_from_timestamp_list(T1_timestamps,
#     #                 params_dict_TD, numeric_params=numeric_params,
#     #                 filter_no_analysis=filter_no_analysis)
#     #     self.TEcho_dict = a_tools.get_data_from_timestamp_list(TEcho_timestamps,
#     #         params_dict_TD, numeric_params=numeric_params,
#     #         filter_no_analysis=filter_no_analysis)
#     #     self.TRamsey_dict = a_tools.get_data_from_timestamp_list(TRamsey_timestamps,
#     #         params_dict_TD, numeric_params=numeric_params,
#     #         filter_no_analysis=filter_no_analysis)


#     def extract_data(self):
#         # load data
#         super().extract_data()
#         tau = np.array(self.raw_data_dict['tau_T1'], dtype=float)
#         tau_std = np.array(self.raw_data_dict['tau_T1_stderr'], dtype=float)
#         # sort data

#         if self.plot_versus_dac:
#             dacs = np.array(self.raw_data_dict['dac'], dtype=float)
#             sorted_indices = dacs.argsort()
#             self.raw_data_dict['dac_sorted'] = dacs[sorted_indices]
#             self.raw_data_dict['dac_sorted_tau'] = tau[sorted_indices]
#             self.raw_data_dict['dac_sorted_tau_stderr'] = tau_std[sorted_indices]
#             if self.plot_versus_frequency:
#                 freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
#                 self.raw_data_dict['dac_sorted_freq'] = freqs[sorted_indices]

#         if self.plot_versus_frequency:
#             freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
#             sorted_indices = freqs.argsort()
#             self.raw_data_dict['freq_sorted'] = freqs[sorted_indices]
#             self.raw_data_dict['freq_sorted_tau'] = tau[sorted_indices]
#             self.raw_data_dict['freq_sorted_tau_stderr'] = tau_std[sorted_indices]
#             if self.plot_versus_dac:
#                 freqs = np.array(self.raw_data_dict['dac'], dtype=float)
#                 self.raw_data_dict['freq_sorted_dac'] = freqs[sorted_indices]




#     def filter_data(self,filter_dics):















#         if use_chisqr:
#             if 'F|1>' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params F|1>.chisqr'
#             elif 'raw' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params raw w0.chisqr'
#             elif 'corr_data' in tau_key:
#                 chisquared_key = 'Analysis.Fitted Params corr_data.chisqr'
#             self.params_dict = {'tau': tau_key,
#                                 'tau_stderr': tau_std_key,
#                                 'chisquared': chisquared_key
#                                 }
#             self.numeric_params = ['tau', 'tau_stderr', 'chisquared']
#         else:
#             self.params_dict = {'tau': tau_key,
#                                 'tau_stderr': tau_std_key,
#                                 # 'chisquared' : chisquared_key
#                                 }
#             self.numeric_params = ['tau', 'tau_stderr']  # , 'chisquared'

#         self.fit_T1_vs_freq = fit_T1_vs_freq

#         self.plot_versus_dac = plot_versus_dac
#         if plot_versus_dac:
#             self.params_dict['dac'] = dac_key

#         self.plot_versus_frequency = plot_versus_frequency
#         if plot_versus_frequency:
#             self.params_dict['qfreq'] = frequency_key

#         self.numeric_params = []

#         if auto:
#             self.run_analysis()
#         # return self.proc_data_dict

#     def extract_data(self):
#         # load data
#         super().extract_data()
#         tau = np.array(self.raw_data_dict['tau'], dtype=float)
#         tau_std = np.array(self.raw_data_dict['tau_stderr'], dtype=float)
#         # sort data

#         if self.plot_versus_dac:
#             dacs = np.array(self.raw_data_dict['dac'], dtype=float)
#             sorted_indices = dacs.argsort()
#             self.raw_data_dict['dac_sorted'] = dacs[sorted_indices]
#             self.raw_data_dict['dac_sorted_tau'] = tau[sorted_indices]
#             self.raw_data_dict['dac_sorted_tau_stderr'] = tau_std[sorted_indices]
#             if self.plot_versus_frequency:
#                 freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
#                 self.raw_data_dict['dac_sorted_freq'] = freqs[sorted_indices]

#         if self.plot_versus_frequency:
#             freqs = np.array(self.raw_data_dict['qfreq'], dtype=float)
#             sorted_indices = freqs.argsort()
#             self.raw_data_dict['freq_sorted'] = freqs[sorted_indices]
#             self.raw_data_dict['freq_sorted_tau'] = tau[sorted_indices]
#             self.raw_data_dict['freq_sorted_tau_stderr'] = tau_std[sorted_indices]
#             if self.plot_versus_dac:
#                 freqs = np.array(self.raw_data_dict['dac'], dtype=float)
#                 self.raw_data_dict['freq_sorted_dac'] = freqs[sorted_indices]















    # Define a function that generates the filter mask
    # Filters could e.g. be given as a



def calculate_n_avg(freq_resonator: float, Qc: float,
                    chi_shift: float, intercept: float):
    """
    Calculate the avg. photon number of white noise.

    args:
        freq_resonator: resonator frequency (in Hz)
        Qc  :       coupling quality factor
        chi_shift : dispersive shift (includes 2pi? )
        intercept :  ?? (FIXME ask someone who made this)

    return:
        n_avg : average photon number due to white noise effects.


    Assumes photon shot noise from the RO resonator.
    """
    k_r = 2*np.pi*freq_resonator/Qc
    eta = k_r**2/(k_r**2 + 4*chi_shift**2)
    n_avg = intercept*k_r/(4*chi_shift**2*eta)
    return n_avg


def arch(dac, Ec, Ej, offset, dac0):
    """
    Convert flux (in dac) to frequency for a transmon.

    Args:
        - dac: voltage used in the DAC to generate the flux
        - Ec (Hz): Charging energy of the transmon in Hz
        - Ej (Hz): Josephson energy of the transmon in Hz
        - offset: voltage offset of the arch (same unit of the dac)
        - dac0: dac value to generate 1 Phi_0 (same unit of the dac)

    Note: the Phi_0 (periodicity) dac0.

    N.B. I believe we have a cleaner version of this somewhere...
    """
    d = np.abs(np.cos((np.pi * (dac - offset)) / dac0))
    model = np.sqrt(8 * Ec * Ej * d) - Ec

    return model


# # define the model (from the function above) used to fit data
# arch_model = lmfit.Model(arch)


def partial_omega_over_flux(flux, Ec, Ej):
    """
    Calculate derivative of flux arc in units of omega/Phi0.

    Args:
        flux  (units of phi0)
        Ec: charging energy (Hz)
        Ej: josephson energy (Hz)

    Return:
        frequency/flux in units of omega/Phi0
    """
    model = -np.sign(np.cos(np.pi * flux)) * (np.pi ** 2) * \
        np.sqrt(8 * Ec * Ej) * \
        np.sin(np.pi * flux) / np.sqrt(np.abs(np.cos(np.pi * flux)))
    return model


def fixed_Q_factor_model(freq, Q):
    '''
    inverse proportional dependence of the qubit T1 on frequrncy can be described
    with a constant Q-factor of a qubit: Q = 2*pi*f*T1
    Here is a function calculating T1(f) that can be fitted to data
    '''
    model = Q/(2*np.pi*freq)
    return model


def fit_fixed_Q_factor(freq, tau):
    Q_factor_model = lmfit.Model(fixed_Q_factor_model)
    Q_factor_model.set_param_hint('Q', value=50e4, min=100e3, max=100e6)
    fit_result_Q_factor = Q_factor_model.fit(tau, freq=freq)
    return fit_result_Q_factor


def fixed_purcell_diel(freq, Q, fres, g, kappa):
    '''
    Inverse proportional dependence of the qubit T1 on frequrncy can be described
    with a constant Q-factor of a qubit: Q = 2*pi*f*T1
    Except for that I include decay rate due to purcell effect.
    Refs:
        Luthi PRL 2018
        Krantz arXiv:1904.06560 pp. 48
    Here is a function calculating T1(f) that can be fitted to data
    '''
    Qfact_rate = (2*np.pi*freq)/Q
    purcell_rate = fres/freq*g**2/kappa/(1+8*np.pi*(freq-fres)**2/kappa**2)
    model = 1/(Qfact_rate + purcell_rate)
    return model

def fixed_purcell_diel2(freq, Q, fres1, g1, kappa1, fres2, g2, kappa2):
    '''
    T1 with purcell decay due to two resonators
    '''
    Qfact_rate = (2*np.pi*freq)/Q
    purcell_rate1 = fres1/freq*g1**2/kappa1/(1+8*np.pi*(freq-fres1)**2/kappa1**2)
    purcell_rate2 = fres2/freq*g2**2/kappa2/(1+8*np.pi*(freq-fres2)**2/kappa2**2)
    model = 1/(Qfact_rate + purcell_rate1 + purcell_rate2)
    return model

def fit_T1_purcell_diel(freq, tau, fres_guess):
    Q_guess = 2*np.pi*np.mean(freq)*np.mean(tau)
    if type(fres_guess) is float:
        Q_factor_model = lmfit.Model(fixed_purcell_diel)
        Q_factor_model.set_param_hint('Q', value=Q_guess)
        Q_factor_model.set_param_hint('fres', value=fres_guess)
        Q_factor_model.set_param_hint('g', value=10e6)
        Q_factor_model.set_param_hint('kappa', value=1e8)
    else:
        Q_factor_model = lmfit.Model(fixed_purcell_diel2)
        Q_factor_model.set_param_hint('Q', value=Q_guess)
        Q_factor_model.set_param_hint('fres1', value=fres_guess[0])
        Q_factor_model.set_param_hint('g1', value=10e6)
        Q_factor_model.set_param_hint('kappa1', value=1e8)
        Q_factor_model.set_param_hint('fres2', value=fres_guess[1])
        Q_factor_model.set_param_hint('g2', value=10e6)
        Q_factor_model.set_param_hint('kappa2', value=1e8)

    fit_result_Q_factor = Q_factor_model.fit(tau, freq=freq)
    return fit_result_Q_factor


def fit_frequencies(dac, freq,
                    Ec_guess=260e6, Ej_guess=None, offset_guess=0,
                    dac0_guess=None):
    """
    Perform fit against the transmon flux arc model.

    Args:
        dac: flux in units of φ0
        freq: 01 transition frequency in Hz

    return:
        fit_result_arch: an lmfit fit_result object.
    """
    # define the model (from the function) used to fit data
    arch_model = lmfit.Model(arch)

    if dac0_guess is None:
        dac0_guess = np.max(np.abs(dac))

    if Ej_guess is None:
        f_max = np.max(freq)
        Ej_guess = (f_max+Ec_guess)**2/(8*Ec_guess)

    # set some hardcoded guesses
    arch_model.set_param_hint('Ec', value=Ec_guess, min=1e6, max=350e6)
    arch_model.set_param_hint('Ej', value=Ej_guess, min=0.1e9, max=50e12)
    arch_model.set_param_hint('offset', value=offset_guess, min=-0.5, max=0.5)
    arch_model.set_param_hint('dac0', value=dac0_guess, min=0)

    params = arch_model.make_params()
    fit_result_arch = arch_model.fit(freq, dac=dac, params=params)
    return fit_result_arch


def residual_Gamma(pars_dict, sensitivity, Gamma_phi_ramsey, Gamma_phi_echo):
    """
    Residual function for fitting dephasing rates (Gamma).

    Two separate linear models are used for the ramsey and echo dephasing
    rates.
    """
    # FIXME: this really needs a dostring to explain what it does
    slope_ramsey = pars_dict['slope_ramsey']
    slope_echo = pars_dict['slope_echo']
    intercept = pars_dict['intercept']

    gamma_values_ramsey = slope_ramsey*np.abs(sensitivity) + intercept
    residual_ramsey = Gamma_phi_ramsey - gamma_values_ramsey

    gamma_values_echo = slope_echo*np.abs(sensitivity) + intercept
    residual_echo = Gamma_phi_echo - gamma_values_echo

    return np.concatenate((residual_ramsey, residual_echo))


def fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo,
               verbose: int=0):
    """
    Perform a fit to the residual_Gamma using hardcoded guesses.

    Args:
        sensitivity         x-values
        Gamma_phi_ramsey    dephasing rate of Ramsey experiment
        gamma_phi_echo      dephasing rate of echo experiment
        verbose (int)       verbosity level
    Returns:
        fit_result_gammas  an lmfit fit_res object
    """
    # create a parameter set for the initial guess
    p = lmfit.Parameters()
    p.add('slope_ramsey', value=100.0, vary=True)
    p.add('slope_echo', value=100.0, vary=True)
    p.add('intercept', value=100.0, vary=True)

    # mi = lmfit.minimize(super_residual, p)
    def wrap_residual(p): return residual_Gamma(
        p,
        sensitivity=sensitivity,
        Gamma_phi_ramsey=Gamma_phi_ramsey,
        Gamma_phi_echo=Gamma_phi_echo)
    fit_result_gammas = lmfit.minimize(wrap_residual, p)
    if verbose > 0:
        lmfit.printfuncs.report_fit(fit_result_gammas.params)
    return fit_result_gammas


def calc_dephasing_rate(T1, T2):
    """Calculate pure dephasing rate based on T1 and T2."""
    gamma = 1/T2 - 1/(2*T1)
    return gamma


def PSD_Analysis(table, freq_resonator=None, Qc=None, chi_shift=None,
                 path=None):
    """
    Power spectral density analysis of transmon coherence.

    Args:
        table : table containing the data, see below for specification.
        freq_resonator: readout resonator frequency (in Hz)
        Qc:             coupling Q of the readout resonator
        chi_shift       in Hz
        path:           filepath, if provided is used for saving the plots



    Input table specification:
           Row  | Content
        --------+--------
            1   | dac
            2   | frequency
            3   | T1
            4   | T2 star
            5   | T2 echo
            6   | Exclusion mask (True where data is to be excluded)

    Generates 8 plots:
        > T1, T2, Echo vs flux
        > T1, T2, Echo vs frequency
        > T1, T2, Echo vs flux sensitivity
        > ratio Ramsey/Echo vs flux
        > ratio Ramsey/Echo vs frequency
        > ratio Ramsey/Echo vs flux sensitivity
        > Dephasing rates Ramsey and Echo vs flux sensitivity
        > Dac arc fit, use to assess if sensitivity calculation is correct.

    If properties of resonator are provided (freq_resonator, Qc, chi_shift),
    it also calculates the number of noise photons.
    """
    dac, freq, T1, Tramsey, Techo, exclusion_mask = table
    exclusion_mask = np.array(exclusion_mask, dtype=bool)

    # Extract the dac arcs required for getting the sensitivities
    fit_result_arch = fit_frequencies(dac, freq, dac0_guess=.1)

    # convert dac in flux as unit of Phi_0
    flux = (dac-fit_result_arch.best_values['offset'])\
        / fit_result_arch.best_values['dac0']

    # calculate the derivative vs flux
    sensitivity_angular = partial_omega_over_flux(
        flux, fit_result_arch.best_values['Ec'],
        fit_result_arch.best_values['Ej'])
    sensitivity = sensitivity_angular/(2*np.pi)

    # Pure dephasing times
    # Calculate pure dephasings
    Gamma_1 = 1.0/T1[~exclusion_mask]
    Gamma_ramsey = 1.0/Tramsey[~exclusion_mask]
    Gamma_echo = 1.0/Techo[~exclusion_mask]

    Gamma_phi_ramsey = Gamma_ramsey - Gamma_1/2.0
    Gamma_phi_echo = Gamma_echo - Gamma_1/2.0

    plot_dac_arc(dac, freq, fit_result_arch)
    plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo, path)
    plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo, path)

    fit_res_gammas = fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo)

    intercept = fit_res_gammas.params['intercept'].value
    slope_ramsey = fit_res_gammas.params['slope_ramsey'].value
    slope_echo = fit_res_gammas.params['slope_echo'].value

    plot_gamma_fit(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo,
                   slope_ramsey, slope_echo, intercept, path)

    # after fitting gammas
    # from flux noise
    # Martinis PRA 2003
    sqrtA_rams = slope_ramsey/(np.pi*np.sqrt(30))
    sqrtA_echo = slope_echo/(np.pi*np.sqrt(1.386))

    print('Amplitude echo PSD = (%s u\Phi_0)^2' % (sqrtA_echo/1e-6))
    print('Amplitude rams PSD = (%s u\Phi_0)^2' % (sqrtA_rams/1e-6))

    # from white noise
    # using Eq 5 in Nat. Comm. 7,12964 (The flux qubit revisited to enhance
    # coherence and reproducability)
    if not ((freq_resonator is None) and (Qc is None) and (chi_shift is None)):
        n_avg = calculate_n_avg(freq_resonator, Qc, chi_shift, intercept)
        print('Estimated residual photon number: %s' % n_avg)
    else:
        n_avg = np.nan

    return (sqrtA_echo/1e-6), n_avg


def prepare_input_table(dac, frequency, T1, T2_star, T2_echo,
                        T1_mask=None, T2_star_mask=None, T2_echo_mask=None):
    """
    Returns a table ready for PSD_Analysis input
    If sizes are different, it adds nans on the end.
    """
    assert(len(dac) == len(frequency))
    assert(len(dac) == len(T1))
    assert(len(dac) == len(T2_star))
    assert(len(dac) == len(T2_echo))

    if T1_mask is None:
        T1_mask = np.zeros(len(T1), dtype=bool)
    if T2_star_mask is None:
        T2_star_mask = np.zeros(len(T2_star), dtype=bool)
    if T2_echo_mask is None:
        T2_echo_mask = np.zeros(len(T2_echo), dtype=bool)

    assert(len(T1) == len(T1_mask))
    assert(len(T2_star) == len(T2_star_mask))
    assert(len(T2_echo) == len(T2_echo_mask))

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


def plot_dac_arc(dac, freq, fit_res, title='', ax=None, **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(dac, freq, 'o', label='data')
    plot_lmfit_res(ax=ax, fit_res=fit_res, plot_init=True,
                   plot_kw={'label': 'arc-fit'},
                   plot_init_kw={'label': 'init-fit', 'ls': '--'})

    ax.legend(loc=0)
    # FIXME, no units for xlabel
    set_xlabel(ax, 'Dac', '')
    set_ylabel(ax, 'Frequency', 'Hz')


def plot_coherence_times_freq(flux, freq, sensitivity,
                              T1, Tramsey, Techo, path,
                              figname='Coherence_times.PNG'):
    f, ax = plt.subplots()

    ax.plot(freq/1e9, T1/1e-6, 'o', color='C3', label='$T_1$')
    ax.plot(freq/1e9, Tramsey/1e-6, 'o', color='C2', label='$T_2^*$')
    ax.plot(freq/1e9, Techo/1e-6, 'o', color='C0', label='$T_2$')
    ax.set_title('$T_1$, $T_2^*$, $T_2$ vs frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.legend(loc=0)


def plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo, axs, ax=None, **kw):
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    axs[0].plot(flux, T1, 'o', color='C3', label='$T_1$')
    axs[0].plot(flux, Tramsey, 'o', color='C2', label='$T_2^*$')
    axs[0].plot(flux, Techo, 'o', color='C0', label='$T_2$')
    axs[0].set_title('$T_1$, $T_2^*$, $T_2$ vs flux')
    set_ylabel(axs[0], 'Coherence time', 's')
    set_xlabel(axs[0], 'Flux',  '$\Phi_0$')

    axs[1].plot(freq, T1, 'o', color='C3', label='$T_1$')
    axs[1].plot(freq, Tramsey, 'o', color='C2', label='$T_2^*$')
    axs[1].plot(freq, Techo, 'o', color='C0', label='$T_2$')
    axs[1].set_title('$T_1$, $T_2^*$, $T_2$ vs frequency')
    set_xlabel(axs[1], 'Frequency', 'Hz')
    axs[1].legend(loc=0)

    axs[2].plot(np.abs(sensitivity)/1e9, T1,
                'o', color='C3', label='$T_1$')
    axs[2].plot(np.abs(sensitivity)/1e9, Tramsey,
                'o', color='C2', label='$T_2^*$')
    axs[2].plot(
        np.abs(sensitivity)/1e9, Techo, 'o', color='C0', label='$T_2$')
    axs[2].set_title('$T_1$, $T_2^*$, $T_2$ vs sensitivity')
    axs[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)')


def plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo, axs, ax=None, **kw):
    # Pure dephasing times
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    ratio_gamma = Gamma_phi_ramsey/Gamma_phi_echo

    axs[0].plot(flux/1e-3, ratio_gamma, 'o', color='C0')
    axs[0].set_title(
        '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs flux', size=16)
    axs[0].set_ylabel('Ratio', size=16)
    axs[0].set_xlabel('Flux (m$\Phi_0$)', size=16)

    axs[1].plot(freq/1e9, ratio_gamma, 'o', color='C0')
    axs[1].set_title(
        '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs frequency', size=16)
    axs[1].set_xlabel('Frequency (GHz)', size=16)

    axs[2].plot(np.abs(sensitivity)/1e9, ratio_gamma, 'o', color='C0')
    axs[2].set_title(
        '$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs sensitivity', size=16)
    axs[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)', size=16)


def super_residual(p):
    data = residual_Gamma(p)
    return data.astype(float)


def plot_gamma_fit(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo,
                   slope_ramsey, slope_echo, intercept, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_ramsey,
            'o', color='C2', label='$\Gamma_{\phi,\mathrm{Ramsey}}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_ramsey *
            np.abs(sensitivity)+intercept, color='C2')

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_echo,
            'o', color='C0', label='$\Gamma_{\phi,\mathrm{Echo}}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_echo *
            np.abs(sensitivity)+intercept, color='C0')

    ax.legend(loc=0)
    ax.set_title('Pure dephasing vs flux sensitivity')
    ax.set_xlabel(r'$|\partial f/\partial\Phi|$ (GHz/$\Phi_0$)')
    set_ylabel(ax, '$\Gamma_{\phi}$', 'Hz')
    ax.set_ylim(0, np.max(Gamma_phi_ramsey)*1.05)



