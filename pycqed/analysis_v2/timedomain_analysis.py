import lmfit
import numpy as np
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba


class Single_Qubit_TimeDomainAnalysis(ba.BaseDataAnalysis):
    pass

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
            # Implicit in AllXY experiments
            if len(self.data_dict['measured_values'][0]) == 42:
                cal_points = [list(range(2)), list(range(-8, -4))]
            elif len(self.data_dict['measured_values'][0]) == 21:
                cal_points = [list(range(1)), list(range(-4, -2))]
            # default for other experiments
            else:
                cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if len(self.data_dict['measured_values']) == 1:
            # if only one weight function is used rotation is not required
            self.data_dict['corr_data'] = a_tools.normalize_data_v3(
                self.data_dict['measured_values'][0],
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1])
        else:
            self.data_dict['corr_data'], zero_coord, one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.data_dict['measured_values'][0:2],
                    zero_coord=zero_coord,
                    one_coord=one_coord,
                    cal_zero_points=cal_points[0],
                    cal_one_points=cal_points[1])

        # This should be added to the hdf5 datafile but cannot because of the
        # way that the "new" analysis works.

        # self.add_dataset_to_analysisgroup('Corrected data',
        #                                   self.data_dict['corr_data'])


class FlippingAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str, t_stop: str=None,
                 options_dict: dict={}, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
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

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.data_dict['sweep_points'],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'],  # does not do anything yet
            'yvals': self.data_dict['corr_data'],
            'ylabel': 'Excited state population',
            'yunit': '',
            'setlabel': 'data',
            'title': self.data_dict['timestamp'] + ' Flipping',
            'do_legend': True}

        fr_poly = self.fit_res['poly_fit']
        fr_cos = self.fit_res['cos_fit']
        xv = np.linspace(np.min(self.data_dict['sweep_points']),
                         np.max(self.data_dict['sweep_points']), 1000)

        self.plot_dicts['poly_fit'] = {
            'ax_id': 'main',
            'plotfn': self.plot_line,
            'xvals': xv,
            'marker': '',
            'setlabel': 'poly_fit',
            'yvals': fr_poly.model.func(x=xv, **fr_poly.best_values),
            'do_legend': True}

        self.plot_dicts['cos_fit'] = {
            'ax_id': 'main',
            'plotfn': self.plot_line,
            'marker': '',
            'xvals': xv,
            'setlabel': 'cos_fit',
            'yvals': fr_cos.model.func(t=xv, **fr_cos.best_values),
            'do_legend': True}

    def run_fitting(self):
        self.fit_res = {}

        poly_mod = lmfit.models.PolynomialModel(degree=2)
        guess_pars = poly_mod.guess(x=self.data_dict['sweep_points'][:-4],
                                    data=self.data_dict['corr_data'][:-4])
        self.fit_res['poly_fit'] = poly_mod.fit(
            x=self.data_dict['sweep_points'][:-4],
            data=self.data_dict['corr_data'][:-4],
            params=guess_pars)

        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(model=cos_mod,
                                        t=self.data_dict['sweep_points'][:-4],
                                        data=self.data_dict['corr_data'][:-4])
        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars['phase'].value = np.deg2rad(90)
        guess_pars['phase'].vary = False

        self.fit_res['cos_fit'] = cos_mod.fit(
            t=self.data_dict['sweep_points'][:-4],
            data=self.data_dict['corr_data'][:-4],
            params=guess_pars)

    def get_scaling_factor(self):
        pass
