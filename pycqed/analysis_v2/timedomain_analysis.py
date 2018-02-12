import lmfit
import numpy as np
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from copy import deepcopy

class Single_Qubit_TimeDomainAnalysis(ba.BaseDataAnalysis):

    def process_data(self):
        '''
        This takes care of rotating and normalizing the data if required.
        this should work for several input types.
            - I/Q values (2 quadratures + cal points)
            - weight functions (1 quadrature + cal points)
            - counts (no cal points)

        There are several options possible to specify the normalization
        using the options dict.
            cal_points (tuple) of indices of the calibration points

            zero_coord, one_coord
        '''

        cal_points = self.options_dict.get('cal_points', None)
        zero_coord = self.options_dict.get('zero_coord', None)
        one_coord = self.options_dict.get('one_coord', None)

        if cal_points is None:
            # default for all standard Timedomain experiments
            cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if len(self.raw_data_dict['measured_values']) == 1:
            # if only one weight function is used rotation is not required
            self.proc_data_dict['corr_data'] = a_tools.normalize_data_v3(
                self.raw_data_dict['measured_values'][0],
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1])
        else:
            self.proc_data_dict['corr_data'], zero_coord, one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.raw_data_dict['measured_values'][0:2],
                    zero_coord=zero_coord,
                    one_coord=one_coord,
                    cal_zero_points=cal_points[0],
                    cal_one_points=cal_points[1])

        # This should be added to the hdf5 datafile but cannot because of the
        # way that the "new" analysis works.

        # self.add_dataset_to_analysisgroup('Corrected data',
        #                                   self.proc_data_dict['corr_data'])


class Idling_Error_Rate_Analyisis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        post_sel_th = self.options_dict.get('post_sel_th', 0.5)
        raw_shots = self.raw_data_dict['measured_values'][0][0]
        post_sel_shots = raw_shots[::2]
        data_shots = raw_shots[1::2]
        data_shots[np.where(post_sel_shots > post_sel_th)] = np.nan

        states = ['0', '1', '+']
        self.proc_data_dict['xvals'] = np.unique(self.raw_data_dict['xvals'])
        for i, state in enumerate(states):
            self.proc_data_dict['shots_{}'.format(state)] =data_shots[i::3]

            self.proc_data_dict['yvals_{}'.format(state)] = \
                np.nanmean(np.reshape(self.proc_data_dict['shots_{}'.format(state)],
                               (len(self.proc_data_dict['xvals']), -1),
                               order='F'), axis=1)


    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            self.plot_dicts['Prepare in {}'.format(state)] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': yvals,
                'ylabel': 'Counts',
                'yrange': [0, 1],
                'xrange': self.options_dict.get('xrange', None),
                'yunit': 'frac',
                'setlabel': 'Prepare in {}'.format(state),
                'do_legend':True,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'legend_pos': 'upper right'}
        if self.do_fitting:
            for state in ['0', '1', '+']:
                self.plot_dicts['fit_{}'.format(state)] = {
                    'ax_id': 'main',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['fit {}'.format(state)]['fit_res'],
                    'plot_init': self.options_dict['plot_init'],
                    'setlabel': 'fit |{}>'.format(state),
                    'do_legend': True,
                    'legend_pos': 'upper right'}

                self.plot_dicts['fit_text']={
                    'ax_id':'main',
                    'box_props': 'fancy',
                    'xpos':1.05,
                    'horizontalalignment':'left',
                    'plotfn': self.plot_text,
                    'text_string': self.proc_data_dict['fit_msg']}



    def analyze_fit_results(self):
        fit_msg =''
        states = ['0', '1', '+']
        for state in states:
            fr = self.fit_res['fit {}'.format(state)]
            N1 = fr.params['N1'].value, fr.params['N1'].stderr
            N2 = fr.params['N2'].value, fr.params['N2'].stderr
            fit_msg += ('Prep |{}> : \n\tN_1 = {:.2g} $\pm$ {:.2g}'
                    '\n\tN_2 = {:.2g} $\pm$ {:.2g}\n').format(
                state, N1[0], N1[1], N2[0], N2[1])

        self.proc_data_dict['fit_msg'] = fit_msg

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            mod = lmfit.Model(fit_mods.idle_error_rate_exp_decay)
            mod.guess = fit_mods.idle_err_rate_guess.__get__(mod, mod.__class__)

            # Done here explicitly so that I can overwrite a specific guess
            guess_pars = mod.guess(N=xvals, data=yvals)
            vary_N2 = self.options_dict.get('vary_N2', True)

            if not vary_N2:
                guess_pars['N2'].value = 1e21
                guess_pars['N2'].vary = False
            # print(guess_pars)
            self.fit_dicts['fit {}'.format(states[i])] = {
                'model': mod,
                'fit_xvals': {'N': xvals},
                'fit_yvals': {'data': yvals},
                'guess_pars': guess_pars}
            # Allows fixing the double exponential coefficient




class FlippingAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # Even though we expect an exponentially damped oscillation we use
        # a simple cosine as this gives more reliable fitting and we are only
        # interested in extracting the frequency of the oscillation
        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.raw_data_dict['sweep_points'][:-4],
            data=self.proc_data_dict['corr_data'][:-4])

        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars['amplitude'].value = 0.5
        guess_pars['amplitude'].vary = False
        guess_pars['offset'].value = 0.5
        guess_pars['offset'].vary = False

        self.fit_dicts['cos_fit'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

        # In the case there are very few periods we fall back on a small
        # angle approximation to extract the drive detuning
        poly_mod = lmfit.models.PolynomialModel(degree=1)
        # the detuning can be estimated using on a small angle approximation
        # c1 = d/dN (cos(2*pi*f N) ) evaluated at N = 0 -> c1 = -2*pi*f
        poly_mod.set_param_hint('frequency', expr='-c1/(2*pi)')
        guess_pars = poly_mod.guess(x=self.raw_data_dict['sweep_points'][:-4],
                                    data=self.proc_data_dict['corr_data'][:-4])
        # Constraining the line ensures that it will only give a good fit
        # if the small angle approximation holds
        guess_pars['c0'].vary = False
        guess_pars['c0'].value = 0.5

        self.fit_dicts['line_fit'] = {
            'model': poly_mod,
            'fit_xvals': {'x': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        sf_line = self._get_scale_factor_line()
        sf_cos = self._get_scale_factor_cos()
        self.proc_data_dict['scale_factor'] = self.get_scale_factor()

        msg = 'Scale fact. based on '
        if self.proc_data_dict['scale_factor'] == sf_cos:
            msg += 'cos fit\n'
        else:
            msg += 'line fit\n'
        msg += 'cos fit: {:.4f}\n'.format(sf_cos)
        msg += 'line fit: {:.4f}'.format(sf_line)

        self.raw_data_dict['scale_factor_msg'] = msg
        # TODO: save scale factor to file

    def get_scale_factor(self):
        """
        Returns the scale factor that should correct for the error in the
        pulse amplitude.
        """
        # Model selection based on the Bayesian Information Criterion (BIC)
        # as  calculated by lmfit
        if (self.fit_dicts['line_fit']['fit_res'].bic <
                self.fit_dicts['cos_fit']['fit_res'].bic):
            scale_factor = self._get_scale_factor_line()
        else:
            scale_factor = self._get_scale_factor_cos()
        return scale_factor

    def _get_scale_factor_cos(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['cos_fit']['fit_res'].params['frequency']

        # the square is needed to account for the difference between
        # power and amplitude
        scale_factor = (1+frequency)**2

        phase = np.rad2deg(self.fit_dicts['cos_fit']['fit_res'].params['phase']) % 360
        # phase ~90 indicates an under rotation so the scale factor
        # has to be larger than 1. A phase ~270 indicates an over
        # rotation so then the scale factor has to be smaller than one.
        if phase > 180:
            scale_factor = 1/scale_factor

        return scale_factor

    def _get_scale_factor_line(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['line_fit']['fit_res'].params['frequency']
        scale_factor = (1+frequency)**2
        # no phase sign check is needed here as this is contained in the
        # sign of the coefficient

        return scale_factor

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'],  # does not do anything yet
            'yvals': self.proc_data_dict['corr_data'],
            'ylabel': 'Excited state population',
            'yunit': '',
            'setlabel': 'data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'line fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'cos fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['text_msg'] = {
                'ax_id': 'main',
                'ypos': 0.15,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'text_string': self.raw_data_dict['scale_factor_msg']}

class Intersect_Analysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analysis to extract the intercept of two parameters.

    relevant options_dict parameters
        ch_idx_A (int) specifies first channel for intercept
        ch_idx_B (int) specifies second channel for intercept if same as first
            it will assume data was taken interleaved.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xvals': 'sweep_points',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()


    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_A" and "ch_idx_B"
        specified in the options dict. If ch_idx_A and ch_idx_B are the same
        it will unzip the data.
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        ch_idx_A = self.options_dict.get('ch_idx_A', 0)
        ch_idx_B = self.options_dict.get('ch_idx_B', 0)


        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx_A]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx_A]

        if ch_idx_A == ch_idx_B:
            yvals = list(self.raw_data_dict['measured_values_ord_dict'].values())[ch_idx_A][0]
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0][::2]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0][1::2]
            self.proc_data_dict['yvals_A'] = yvals[::2]
            self.proc_data_dict['yvals_B'] = yvals[1::2]
        else:
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0]

            self.proc_data_dict['yvals_A'] = list(self.raw_data_dict
                ['measured_values_ord_dict'].values())[ch_idx_A][0]
            self.proc_data_dict['yvals_B'] = list(self.raw_data_dict
                ['measured_values_ord_dict'].values())[ch_idx_B][0]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_A'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_A']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_A']}}

        self.fit_dicts['line_fit_B'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_B']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_B']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_A'].best_values
        fr_1 = self.fit_res['line_fit_B'].best_values

        c0 = (fr_0['c0'] - fr_1['c0'])
        c1 = (fr_0['c1'] - fr_1['c1'])
        c2 = (fr_0['c2'] - fr_1['c2'])
        poly_coeff = [c0, c1, c2]
        poly = np.polynomial.polynomial.Polynomial([fr_0['c0'],
                                                   fr_0['c1'], fr_0['c2']])
        ic = np.polynomial.polynomial.polyroots(poly_coeff)

        self.proc_data_dict['intersect_L'] = ic[0], poly(ic[0])
        self.proc_data_dict['intersect_R'] = ic[1], poly(ic[1])

        if (((np.min(self.proc_data_dict['xvals']))< ic[0]) and
                ( ic[0] < (np.max(self.proc_data_dict['xvals'])))):
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_L']
        else:
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_R']

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_A'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_A'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'A',
            'title': (self.proc_data_dict['timestamps'][0] + ' \n' +
                      self.proc_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_B'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_B'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'B',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_A'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_A']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit A',
                'do_legend': True}
            self.plot_dicts['line_fit_B'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_B']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit B',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['intersect'][0],
                 self.proc_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['intersect'][0]],
                'yvals': [self.proc_data_dict['intersect'][1]],
                'line_kws': {'alpha': .5, 'color':'gray',
                            'markersize':15},
                'marker': 'o',
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_intersect(self):

        return self.proc_data_dict['intersect']



class CZ_1QPhaseCal_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract the intercept for a single qubit phase calibration
    experiment

    N.B. this is a less generic version of "Intersect_Analysis" and should
    be deprecated (MAR Dec 2017)
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx" in options dict and
        then splits the data for th
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx = self.options_dict['ch_idx']

        yvals = list(self.raw_data_dict['measured_values_ord_dict'].values())[ch_idx][0]

        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx]
        self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
        self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]
        self.proc_data_dict['yvals_off'] = yvals[::2]
        self.proc_data_dict['yvals_on'] = yvals[1::2]


    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_off'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_off']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_off']}}

        self.fit_dicts['line_fit_on'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_on']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_on']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_off'].best_values
        fr_1 = self.fit_res['line_fit_on'].best_values
        ic = -(fr_0['c0'] - fr_1['c0'])/(fr_0['c1'] - fr_1['c1'])

        self.proc_data_dict['zero_phase_diff_intersect'] = ic


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_off'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_on'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['line_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['zero_phase_diff_intersect'],
                 self.raw_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['zero_phase_diff_intersect']],
                'yvals': [np.mean(self.proc_data_dict['xvals_on'])],
                'line_kws': {'alpha': 0},
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_zero_phase_diff_intersect(self):

        return self.proc_data_dict['zero_phase_diff_intersect']


class Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Very basic analysis to determine the phase of a single oscillation
    that has an assumed period of 360 degrees.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        idx = 1

        self.proc_data_dict['yvals'] = list(self.raw_data_dict['measured_values_ord_dict'].values())[idx][0]
        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        self.fit_dicts['cos_fit'] = {
            'model': cos_mod,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.raw_data_dict['xvals'][0]},
            'fit_yvals': {'data': self.proc_data_dict['yvals']}}

    def analyze_fit_results(self):
        fr = self.fit_res['cos_fit'].best_values
        self.proc_data_dict['phi'] =  np.rad2deg(fr['phase'])


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['xvals'][0],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit',
                'do_legend': True}


class Conditional_Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract quantities from a conditional oscillation.

    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_osc" and
        "ch_idx_spec" in the options dict and then splits the data for the
        off and on cases
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx_spec = self.options_dict.get('ch_idx_spec', 0)
        ch_idx_osc = self.options_dict.get('ch_idx_osc', 1)
        normalize_to_cal_points = self.options_dict.get('normalize_to_cal_points', True)
        cal_points = [
                        [[-4, -3], [-2, -1]],
                        [[-4, -2], [-3, -1]],
                       ]


        i = 0
        for idx, type_str in zip([ch_idx_osc, ch_idx_spec], ['osc', 'spec']):
            yvals = list(self.raw_data_dict['measured_values_ord_dict'].values())[idx][0]
            self.proc_data_dict['ylabel_{}'.format(type_str)] = self.raw_data_dict['value_names'][0][idx]
            self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.normalize_data_v3(yvals,
                    cal_zero_points=cal_points[i][0],
                    cal_one_points=cal_points[i][1])
                i +=1

                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]
                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]

            else:
                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]


                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]



    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod0 = lmfit.Model(fit_mods.CosFunc)
        cos_mod0.guess = fit_mods.Cos_guess.__get__(cos_mod0, cos_mod0.__class__)
        self.fit_dicts['cos_fit_off'] = {
            'model': cos_mod0,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.proc_data_dict['xvals_off'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_off'][:-2]}}

        cos_mod1 = lmfit.Model(fit_mods.CosFunc)
        cos_mod1.guess = fit_mods.Cos_guess.__get__(cos_mod1, cos_mod1.__class__)
        self.fit_dicts['cos_fit_on'] = {
            'model': cos_mod1,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.proc_data_dict['xvals_on'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_on'][:-2]}}

    def analyze_fit_results(self):
        fr_0 = self.fit_res['cos_fit_off'].params
        fr_1 = self.fit_res['cos_fit_on'].params

        phi0 = np.rad2deg(fr_0['phase'].value)
        phi1 = np.rad2deg(fr_1['phase'].value)

        phi0_stderr = np.rad2deg(fr_0['phase'].stderr)
        phi1_stderr = np.rad2deg(fr_1['phase'].stderr)

        self.proc_data_dict['phi_0'] = phi0, phi0_stderr
        self.proc_data_dict['phi_1'] = phi1, phi1_stderr
        phi_cond_stderr = (phi0_stderr**2+phi1_stderr**2)**.5
        self.proc_data_dict['phi_cond'] = (phi1 -phi0), phi_cond_stderr


        osc_amp = np.mean([fr_0['amplitude'], fr_1['amplitude']])
        osc_amp_stderr = np.sqrt(fr_0['amplitude'].stderr**2 +
                                 fr_1['amplitude']**2)/2

        self.proc_data_dict['osc_amp_0'] = (fr_0['amplitude'].value,
                                            fr_0['amplitude'].stderr)
        self.proc_data_dict['osc_amp_1'] = (fr_1['amplitude'].value,
                                            fr_1['amplitude'].stderr)

        # self.proc_data_dict['osc_amp'] = (osc_amp, osc_amp_stderr)
        self.proc_data_dict['missing_fraction'] = (
            np.mean(self.proc_data_dict['yvals_spec_on'][:-2]) -
            np.mean(self.proc_data_dict['yvals_spec_off'][:-2]))


    def prepare_plots(self):
        self._prepare_main_oscillation_figure()
        self._prepare_spectator_qubit_figure()

    def _prepare_main_oscillation_figure(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_off'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_on'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['cos_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}

            # offset as a guide for the eye
            y = self.fit_res['cos_fit_off'].params['offset'].value
            self.plot_dicts['cos_off_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C0', 'linestyle': 'dotted'}
                    }

            phase_message = (
                'Phase diff.: {:.1f} $\pm$ {:.1f} deg\n'
                'Phase off: {:.1f} $\pm$ {:.1f}deg\n'
                'Phase on: {:.1f} $\pm$ {:.1f}deg\n'
                'Osc. amp. off: {:.4f} $\pm$ {:.4f}\n'
                'Osc. amp. on: {:.4f} $\pm$ {:.4f}'.format(
                    self.proc_data_dict['phi_cond'][0],
                    self.proc_data_dict['phi_cond'][1],
                    self.proc_data_dict['phi_0'][0],
                    self.proc_data_dict['phi_0'][1],
                    self.proc_data_dict['phi_1'][0],
                    self.proc_data_dict['phi_1'][1],
                    self.proc_data_dict['osc_amp_0'][0],
                    self.proc_data_dict['osc_amp_0'][1],
                    self.proc_data_dict['osc_amp_1'][0],
                    self.proc_data_dict['osc_amp_1'][1]))
            self.plot_dicts['phase_message'] = {
                'ax_id': 'main',
                'ypos': 0.9,
                'xpos': 1.45,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': phase_message}

    def _prepare_spectator_qubit_figure(self):

        self.plot_dicts['spectator_qubit'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_off'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['spec_on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'spectator_qubit',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_on'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            leak_msg = (
                'Missing fraction: {:.2f} % '.format(
                    self.proc_data_dict['missing_fraction']*100))
            self.plot_dicts['leak_msg'] = {
                'ax_id': 'spectator_qubit',
                'ypos': 0.7,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': leak_msg}
            # offset as a guide for the eye
            y = self.fit_res['cos_fit_on'].params['offset'].value
            self.plot_dicts['cos_on_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C1', 'linestyle': 'dotted'}
                    }
