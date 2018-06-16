import lmfit
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
import logging
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.plotting import SI_val_to_msg_str


class efT1_analysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idx: int =1,
                 ):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs,
                         do_fitting=True)
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idx = rates_ch_idx
        self.d1 = 2
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        time = a.sweep_points[:-6:2]

        self.raw_data_dict['time'] = time
        self.raw_data_dict['time units'] = a.sweep_unit[0]

        self.raw_data_dict['value_names'] = a.value_names
        self.raw_data_dict['value_units'] = a.value_units
        self.raw_data_dict['measurementstring'] = a.measurementstring
        self.raw_data_dict['timestamp_string'] = a.timestamp_string

        self.raw_data_dict['cal_pts_zero'] = OrderedDict()
        self.raw_data_dict['cal_pts_one'] = OrderedDict()
        self.raw_data_dict['cal_pts_two'] = OrderedDict()
        self.raw_data_dict['measured_values_I'] = OrderedDict()
        self.raw_data_dict['measured_values_X'] = OrderedDict()

        for i, val_name in enumerate(a.value_names):
            self.raw_data_dict['cal_pts_zero'][val_name] = \
                a.measured_values[i][-6:-4]
            self.raw_data_dict['cal_pts_one'][val_name] = \
                a.measured_values[i][-4:-2]
            self.raw_data_dict['cal_pts_two'][val_name] = \
                a.measured_values[i][-2:]
            self.raw_data_dict['measured_values_I'][val_name] = \
                a.measured_values[i][::2][:-3]
            self.raw_data_dict['measured_values_X'][val_name] = \
                a.measured_values[i][1::2][:-3]

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        for key in ['V0', 'V1', 'V2', 'SI', 'SX', 'P0', 'P1', 'P2', 'M_inv']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            V0 = np.mean(
                self.raw_data_dict['cal_pts_zero'][val_name])
            V1 = np.mean(
                self.raw_data_dict['cal_pts_one'][val_name])
            V2 = np.mean(
                self.raw_data_dict['cal_pts_two'][val_name])

            self.proc_data_dict['V0'][val_name] = V0
            self.proc_data_dict['V1'][val_name] = V1
            self.proc_data_dict['V2'][val_name] = V2

            SI = self.raw_data_dict['measured_values_I'][val_name]
            SX = self.raw_data_dict['measured_values_X'][val_name]

            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            P0, P1, P2, M_inv = populations_using_rate_equations(
                SI, SX, V0, V1, V2)
            self.proc_data_dict['P0'][val_name] = P0
            self.proc_data_dict['P1'][val_name] = P1
            self.proc_data_dict['P2'][val_name] = P2
            self.proc_data_dict['M_inv'][val_name] = M_inv

    def run_fitting(self):
        super().run_fitting()
        self.fit_res['fit_res_P2'] = OrderedDict()
        self.fit_res['fit_res_P1'] = OrderedDict()
        self.fit_res['fit_res_P0'] = OrderedDict()
        decay_mod = lmfit.Model(ExpDecayFunclocal, independent_vars='t')
        decay_mod.set_param_hint('tau', value=15e-6, min=0, vary=True)
        decay_mod.set_param_hint('amplitude', value=1, min=0, vary=True)
        decay_mod.set_param_hint('offset', value=0, vary=True)
        params1 = decay_mod.make_params()

        try:
            for value_name in self.raw_data_dict['value_names']:
                fit_res_P2 = decay_mod.fit(
                    data=self.proc_data_dict['P2'][value_name],
                    t=self.proc_data_dict['time'], params=params1)
            self.fit_res['fit_res_P2'] = fit_res_P2
            tau2_best = fit_res_P2.best_values['tau']
            text_msg = (
                r'$T_1^{fe}$  : ' +
                SI_val_to_msg_str(
                    round(fit_res_P2.params['tau'].value, 6), 's')[0]
                + SI_val_to_msg_str(fit_res_P2.params['tau'].value, 's')[1]
            )
        except Exception as e:
            logging.warning("Fitting failed")
            logging.warning(e)
            self.fit_res['fit_res_P2'] = {}

        doubledecay_mod = lmfit.Model(
            DoubleExpDecayFunclocal, independent_vars='t')
        doubledecay_mod.set_param_hint('tau1', value=10e-6, min=0, vary=True)
        doubledecay_mod.set_param_hint(
            'tau2', value=tau2_best, min=0, vary=False)
        doubledecay_mod.set_param_hint(
            'amplitude1', value=1.0, min=0, vary=True)
        doubledecay_mod.set_param_hint(
            'amplitude2', value=0.5, min=0, vary=True)
        doubledecay_mod.set_param_hint('offset', value=.0, vary=True)

        params2 = doubledecay_mod.make_params()
        try:
            for value_name in self.raw_data_dict['value_names']:
                fit_res_P1 = doubledecay_mod.fit(
                             data=self.proc_data_dict['P1'][value_name],
                             t=self.proc_data_dict['time'],
                             params=params2)

            self.fit_res['fit_res_P1'] = fit_res_P1
            tau1_best = fit_res_P1.best_values['tau1']
            amplitude1_best = fit_res_P1.best_values['amplitude1']
            amplitude2_best = fit_res_P1.best_values['amplitude2']
            text_msg += ('\n' +
                         r'$T_1^{eg}$  : ' + SI_val_to_msg_str
                         (round(fit_res_P1.params['tau1'].value, 6), 's')[0]
                         + SI_val_to_msg_str
                         (fit_res_P1.params['tau1'].value, 's')[1])
        except Exception as e:
            logging.warning("Doulbe Fitting failed")
            logging.warning(e)
            self.fit_res['fit_res_P1'] = {}

        doubledecay_mod = lmfit.Model(
            DoubleExpDecayFunclocal2, independent_vars='t')
        doubledecay_mod.set_param_hint(
            'tau1', value=tau1_best, min=0, vary=False)
        doubledecay_mod.set_param_hint(
            'tau2', value=tau2_best, min=0, vary=False)
        doubledecay_mod.set_param_hint(
            'amplitude1', value=amplitude1_best, min=0, vary=False)
        doubledecay_mod.set_param_hint(
            'amplitude2', value=amplitude2_best, min=0, vary=False)
        doubledecay_mod.set_param_hint('offset', value=.0, vary=True)

        params3 = doubledecay_mod.make_params()
        try:
            for value_name in self.raw_data_dict['value_names']:
                fit_res_P0 = doubledecay_mod.fit(
                             data=self.proc_data_dict['P0'][value_name],
                             t=self.proc_data_dict['time'],
                             params=params3)

            self.fit_res['fit_res_P0'] = fit_res_P0
        except Exception as e:
            logging.warning("Doulbe Fitting failed")
            logging.warning(e)
            self.fit_res['fit_res_P0'] = {}

        self.proc_data_dict['fit_msg'] = text_msg

    def prepare_plots(self):
        val_names = self.raw_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['plot_populations_{}'.format(val_name)] = {
                'plotfn': plot_populations,
                'time': self.proc_data_dict['time'],
                'P0': self.proc_data_dict['P0'][val_name],
                'P1': self.proc_data_dict['P1'][val_name],
                'P2': self.proc_data_dict['P2'][val_name],

                'xlabel': 'Time',
                'xunit': self.raw_data_dict['time units'],
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string']
                          + '\n' +
                          self.proc_data_dict['measurementstring']
                }

            # define figure and axes here to have custom layout

            self.plot_dicts['fit_res_P0'] = {
                'plotfn': self.plot_fit,
                # 'plot_init': True,
                'ax_id': 'plot_populations_{}'.format(val_name),
                'fit_res': self.fit_res['fit_res_P0'],
                'setlabel': r'P($|g\rangle$) fit',
                'do_legend': True,
                'color': 'C0',
            }

            self.plot_dicts['fit_res_P1'] = {
                'plotfn': self.plot_fit,
                # 'plot_init': True,
                'ax_id': 'plot_populations_{}'.format(val_name),
                'fit_res': self.fit_res['fit_res_P1'],
                'setlabel': r'P($|e\rangle$) fit',
                'do_legend': True,
                'color': 'C1',

            }

            self.plot_dicts['fit_res_P2'] = {
                'plotfn': self.plot_fit,
                'ax_id': 'plot_populations_{}'.format(val_name),
                'fit_res': self.fit_res['fit_res_P2'],
                'setlabel': r'P($|f\rangle$) fit',
                'do_legend': True,
                'color': 'C2',
            }

            self.plot_dicts['fit_msg'] = {
                'plotfn': self.plot_text,
                'text_string': self.proc_data_dict['fit_msg'],
                'xpos': 0.1, 'ypos': .9,
                'ax_id': 'plot_populations_{}'.format(val_name),
                'horizontalalignment': 'left'}


def plot_populations(time, P0, P1, P2, ax,
                     xlabel='Time', xunit='s',
                     ylabel='Population', yunit='',
                     title='', **kw):
    ax.plot(time, P0, c='C0', linestyle='',
            label=r'P($|g\rangle$)', marker='v')
    ax.plot(time, P1, c='C1', linestyle='',
            label=r'P($|e\rangle$)', marker='^')
    ax.plot(time, P2, c='C2', linestyle='',
            label=r'P($|f\rangle$)', marker='d')

    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel)
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def populations_using_rate_equations(SI: np.array, SX: np.array,
                                     V0: float, V1: float, V2: float):
    """
    Args:
        SI (array): signal value for signal with I (Identity) added
        SX (array): signal value for signal with X (π-pulse) added
        V0 (float):
        V1 (float):
        V2 (float):
    returns:
        P0 (array): population of the |0> state
        P1 (array): population of the |1> state
        P2 (array): population of the |2> state
        M_inv (2D array) :  Matrix inverse to find populations

    Based on equation (S1) from Asaad & Dickel et al. npj Quant. Info. (2016)

    To quantify leakage, we monitor the populations Pi of the three lowest
    energy states (i ∈ {0, 1, 2}) and calculate the average
    values <Pi>. To do this, we calibrate the average signal levels Vi for
    the transmons in level i, and perform each measurement twice, the second
    time with an added final π pulse on the 0–1 transition. This final π
    pulse swaps P0 and P1, leaving P2 unaffected. Under the assumption that
    higher levels are unpopulated (P0 +P1 +P2 = 1),

     [V0 −V2,   V1 −V2] [P0]  = [S −V2]
     [V1 −V2,   V0 −V2] [P1]  = [S' −V2]

    where S (S') is the measured signal level without (with) final π pulse.
    The populations are extracted by matrix inversion.
    """
    M = np.array([[V0-V2, V1-V2], [V1-V2, V0-V2]])
    M_inv = np.linalg.inv(M)

    P0 = np.zeros(len(SI))
    P1 = np.zeros(len(SX))
    for i, (sI, sX) in enumerate(zip(SI, SX)):
        p0, p1 = np.dot(np.array([sI-V2, sX-V2]), M_inv)
        p0, p1 = np.dot(M_inv, np.array([sI-V2, sX-V2]))
        P0[i] = p0
        P1[i] = p1

    P2 = 1 - P0 - P1

    return P0, P1, P2, M_inv


def ExpDecayFunclocal(t, tau, amplitude, offset):
    return amplitude * np.exp(-(t / tau)) + offset


def DoubleExpDecayFunclocal(t, tau1, tau2, amplitude1, amplitude2, offset):
    return amplitude1 * np.exp(-(t / tau1)) - \
        amplitude2 * np.exp(-(t / tau2)) + offset


def DoubleExpDecayFunclocal2(t, tau1, tau2, amplitude1, amplitude2, offset):
    return - amplitude1 * np.exp(-(t / tau1)) + \
        amplitude2 * np.exp(-(t / tau2)) * tau2 / tau1 + offset
