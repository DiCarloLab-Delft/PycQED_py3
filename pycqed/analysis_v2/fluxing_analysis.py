import lmfit
from uncertainties import ufloat
from pycqed.analysis import measurement_analysis as ma
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis.tools.data_manipulation import \
    populations_using_rate_equations
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, plot_fit
import matplotlib.pyplot as plt
from pycqed.analysis.fitting_models import CosFunc, Cos_guess, \
    avoided_crossing_freq_shift


class Chevron_Analysis(ba.BaseDataAnalysis):
    def __init__(self, ts: str=None, label=None,
                 ch_idx=0,
                 coupling='g', min_fit_amp=0, auto=True):
        """
        Analyzes a Chevron and fits the avoided crossing.

        Parameters
        ----------
        ts: str
            timestamp of the datafile
        label: str
            label to find the datafile (optional)
        ch_idx: int
            channel to use when fitting the avoided crossing
        coupling: Enum("g", "J1", "J2")
            used to label the avoided crossing and calculate related quantities
        min_fit_amp:
            minimal maplitude of the fitted cosine for each line cut.
            Oscillations with a smaller amplitude will be ignored in the fit
            of the avoided crossing.
        auto: bool
            if True run all parts of the analysis.

        """
        super().__init__(do_fitting=True)
        self.ts = ts
        self.label = label
        self.coupling = coupling
        self.ch_idx = ch_idx
        self.min_fit_amp = min_fit_amp
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        a = ma.MeasurementAnalysis(
            timestamp=self.ts, label=self.label, auto=False)
        a.get_naming_and_values_2D()
        a.finish()
        self.timestamps = [a.timestamp_string]
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['timestamp_string'] = a.timestamp
        for attr in ['sweep_points', 'sweep_points_2D', 'measured_values',
                     'parameter_names', 'parameter_units', 'value_names',
                     'value_units']:
            self.raw_data_dict[attr] = getattr(a, attr)
        self.raw_data_dict['folder'] = a.folder

    def process_data(self):
        self.proc_data_dict = OrderedDict()

        # select the relevant data
        x = self.raw_data_dict['sweep_points']
        t = self.raw_data_dict['sweep_points_2D']
        Z = self.raw_data_dict['measured_values'][self.ch_idx].T

        # fit frequencies to each individual cut (time trace)
        freqs = []
        freqs_std = []
        fit_results = []
        amps = []
        for xi, z in zip(x, Z.T):
            CosModel = lmfit.Model(CosFunc)
            CosModel.guess = Cos_guess
            pars = CosModel.guess(CosModel, z, t)
            fr = CosModel.fit(data=z, t=t, params=pars)
            amps.append(fr.params['amplitude'].value)
            freqs.append(fr.params['frequency'].value)
            freqs_std.append(fr.params['frequency'].stderr)
            fit_results.append(fr)
        # N.B. the fit results are not saved in self.fit_res as this would
        # bloat the datafiles.
        self.proc_data_dict['fit_results'] = np.array(fit_results)
        self.proc_data_dict['amp_fits'] = np.array(amps)
        self.proc_data_dict['freq_fits'] = np.array(freqs)
        self.proc_data_dict['freq_fits_std'] = np.array(freqs_std)

        # take a Fourier transform (nice for plotting)
        fft_data = abs(np.fft.fft(Z.T).T)
        fft_freqs = np.fft.fftfreq(len(t), d=t[1]-t[0])
        sort_vec = np.argsort(fft_freqs)

        fft_data_sorted = fft_data[sort_vec, :]
        fft_freqs_sorted = fft_freqs[sort_vec]
        self.proc_data_dict['fft_data_sorted'] = fft_data_sorted
        self.proc_data_dict['fft_freqs_sorted'] = fft_freqs_sorted

    def run_fitting(self):
        super().run_fitting()

        fit_mask = np.where(self.proc_data_dict['amp_fits'] > self.min_fit_amp)

        avoided_crossing_mod = lmfit.Model(avoided_crossing_freq_shift)
        # hardcoded guesses! Bad practice, needs a proper guess func
        avoided_crossing_mod.set_param_hint('a', value=3e9)
        avoided_crossing_mod.set_param_hint('b', value=-2e9)
        avoided_crossing_mod.set_param_hint('g', value=20e6, min=0)
        params = avoided_crossing_mod.make_params()

        self.fit_res['avoided_crossing'] = avoided_crossing_mod.fit(
            data=self.proc_data_dict['freq_fits'][fit_mask],
            flux=self.raw_data_dict['sweep_points'][fit_mask],
            params=params)

    def analyze_fit_results(self):
        self.proc_data_dict['quantities_of_interest'] = {}
        # Extract quantities of interest from the fit
        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']
        g = self.fit_res['avoided_crossing'].params['g']
        qoi['g'] = ufloat(g.value, g.stderr)

        self.coupling_msg = ''
        if self.coupling == 'J1':
            qoi['J1'] = qoi['g']
            qoi['J2'] = qoi['g']*np.sqrt(2)
            self.coupling_msg += r'Measured $J_1$ = {} MHz'.format(
                qoi['J1']*1e-6)+'\n'
            self.coupling_msg += r'Expected $J_2$ = {} MHz'.format(
                qoi['J2']*1e-6)
        elif self.coupling == 'J2':
            qoi['J1'] = qoi['g']/np.sqrt(2)
            qoi['J2'] = qoi['g']
            self.coupling_msg += r'Expected $J_1$ = {} MHz'.format(
                qoi['J1']*1e-6)+'\n'
            self.coupling_msg += r'Measured $J_2$ = {} MHz'.format(
                qoi['J2']*1e-6)
        else:
            self.coupling_msg += 'g = {}'.format(qoi['g'])

    def prepare_plots(self):
        for i, val_name in enumerate(self.raw_data_dict['value_names']):
            self.plot_dicts['chevron_{}'.format(val_name)] = {
                'plotfn': plot_chevron,
                'x': self.raw_data_dict['sweep_points'],
                'y': self.raw_data_dict['sweep_points_2D'],
                'Z': self.raw_data_dict['measured_values'][i].T,
                'xlabel': self.raw_data_dict['parameter_names'][0],
                'ylabel': self.raw_data_dict['parameter_names'][1],
                'zlabel': self.raw_data_dict['value_names'][i],
                'xunit': self.raw_data_dict['parameter_units'][0],
                'yunit': self.raw_data_dict['parameter_units'][1],
                'zunit': self.raw_data_dict['value_units'][i],
                'title': self.raw_data_dict['timestamp_string']+'\n' +
                'Chevron {}'.format(val_name)
            }

        self.plot_dicts['chevron_fft'] = {
            'plotfn': plot_chevron_FFT,
            'x': self.raw_data_dict['sweep_points'],
            'xunit': self.raw_data_dict['parameter_units'][0],
            'fft_freqs': self.proc_data_dict['fft_freqs_sorted'],
            'fft_data': self.proc_data_dict['fft_data_sorted'],
            'freq_fits': self.proc_data_dict['freq_fits'],
            'freq_fits_std': self.proc_data_dict['freq_fits_std'],
            'fit_res': self.fit_res['avoided_crossing'],
            'coupling_msg': self.coupling_msg,
            'title': self.raw_data_dict['timestamp_string']+'\n' +
            'Fourier transform of Chevron'}


def plot_chevron(x, y, Z, xlabel, xunit, ylabel, yunit,
                 zlabel, zunit,
                 title, ax, **kw):
    colormap = ax.pcolormesh(x, y, Z, cmap='viridis',  # norm=norm,
                             linewidth=0, rasterized=True,
                             # assumes digitized readout
                             vmin=0, vmax=1)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='5%', pad='2%')
    cbar = plt.colorbar(colormap, cax=cax, orientation='vertical')
    cax.set_ylabel('L1 (%)')

    set_ylabel(cax, zlabel, zunit)


def plot_chevron_FFT(x, xunit,  fft_freqs, fft_data, freq_fits, freq_fits_std,
                     fit_res, coupling_msg, title, ax, **kw):

    colormap = ax.pcolormesh(x,
                             fft_freqs, fft_data, cmap='viridis',  # norm=norm,
                             linewidth=0, rasterized=True, vmin=0, vmax=5)

    ax.errorbar(x=x, y=freq_fits, yerr=freq_fits_std, ls='--', c='r', alpha=.5,
                label='Extracted freqs')
    x_fine = np.linspace(x[0], x[-1], 200)
    plot_fit(x, fit_res, ax=ax, c='C1', label='Avoided crossing fit', ls=':')

    set_xlabel(ax, 'Flux bias', xunit)
    set_ylabel(ax, 'Frequency', 'Hz')
    ax.legend(loc=(1.05, .7))
    ax.text(1.05, 0.5, coupling_msg, transform=ax.transAxes)
