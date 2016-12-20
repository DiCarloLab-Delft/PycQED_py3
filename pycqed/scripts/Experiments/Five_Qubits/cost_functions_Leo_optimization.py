from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement import detector_functions as det
from pycqed.measurement import awg_sweep_functions as awg_swf
import numpy as np
import lmfit
import pycqed.measurement.sweep_functions as swf


class Leo_optim_det(det.Soft_Detector):

    def __init__(self,
                 qubit,
                 MC_nested,
                 phases=np.arange(0, 780, 20),
                 inter_swap_wait=50e-9,
                 upload=True,  **kw):

        self.name = 'swap_wait_leo_optim'
        self.value_names = ['Cost function']
        self.value_units = ['a.u.']
        self.detector_control = 'soft'
        self.upload = upload
        self.AWG = qubit.AWG
        self.qubit = qubit
        self.phases = phases

        mw_pulse_pars, RO_pars = qubit.get_pulse_pars()
        flux_pulse_pars = qubit.get_flux_pars()
        dist_dict = qubit.get_dist_dict()
        self.s = awg_swf.swap_swap_wait(
            mw_pulse_pars, RO_pars, flux_pulse_pars, dist_dict,
            AWG=self.AWG, inter_swap_wait=inter_swap_wait)
        if not upload:
            self.s.upload = False

        self.AWG.ch3_amp(qubit.swap_amp())

        self.MC_nested = MC_nested

    def acquire_data_point(self, **kw):
        # # Update kernel from kernel object

        self.MC_nested.set_sweep_function(self.s)
        self.MC_nested.set_detector_function(self.qubit.int_avg_det_rot)
        self.MC_nested.set_sweep_points(self.phases)
        self.MC_nested.run('swap-swap-wait')
        self.s.upload = False

        a = SWAP_Cost_Leo1(show_guess=True)
        return a.cost_func_val


class SWAP_Cost_Leo1(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):

        self.get_naming_and_values()
        self.measured_values = self.measured_values
        self.sweep_points = self.sweep_points
        self.fit_data(**kw)
        self.make_figures(**kw)

        osc_amp = self.fit_res[0].best_values['amplitude']
        self.cost_func_val = 1-2 * abs(osc_amp)

        if close_file:
            self.data_file.close()

    def fit_data(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.
        for i in [0, 1]:
            model = lmfit.Model(ma.fit_mods.CosFunc)
            params = simple_cos_guess(model,
                                      data=self.measured_values[i][:-4],
                                      t=self.sweep_points[:-4])
            self.fit_res[i] = model.fit(
                data=self.measured_values[i][:-4],
                t=self.sweep_points[:-4],
                params=params)
            self.save_fitted_parameters(fit_res=self.fit_res[i],
                                        var_name=self.value_names[i])


def simple_cos(t, amplitude, frequency, phase, offset):
    return amplitude*np.cos(frequency*t+phase)+offset


def simple_cos_guess(model, data, t):
    '''
    Guess for a cosine fit using FFT, only works for evenly spaced points
    '''
    amp_guess = abs(max(data)-min(data))/2  # amp is positive by convention
    offs_guess = np.mean(data)

    freq_guess = 1/360
    ph_guess = -2*np.pi*t[data == max(data)]/360

    model.set_param_hint('period', expr='1/frequency')
    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp
    return params


class CPhase_cost_func_det(det.Soft_Detector):

    def __init__(self,
                 qCP,
                 qS,
                 dist_dict,
                 MC_nested,
                 lambda1,
                 int_avg_det,
                 phases=np.arange(0, 780, 20),
                 inter_swap_wait=100e-9,
                 upload=True,  **kw):

        self.name = 'CPhase_cost_func_det'
        self.value_names = [
            'Cost function', 'phase_diff', 'amp_single_exc', 'amp_no_exc']
        self.value_units = ['a.u.', 'deg', '', '']
        self.detector_control = 'soft'
        self.upload = upload
        self.AWG = qCP.AWG
        self.phases = phases
        self.MC = MC_nested
        self.int_avg_det = int_avg_det
        self.MC.soft_avg(1)
        self.inter_swap_wait = inter_swap_wait
        self.qCP = qCP
        self.qS = qS
        self.lambda1 = lambda1
        qCP.swap_amp(1.132)

        qCP_pulse_pars, RO_pars_qCP = qCP.get_pulse_pars()
        qS_pulse_pars, RO_pars_qS = qS.get_pulse_pars()
        flux_pulse_pars_qCP = qCP.get_flux_pars()
        flux_pulse_pars_qCP['pulse_type'] = 'MartinisFluxPulse'
        flux_pulse_pars_qCP['lambda0'] = 1
        flux_pulse_pars_qCP['lambda1'] = lambda1

        flux_pulse_pars_qS = qS.get_flux_pars()
        self.s = awg_swf.swap_CP_swap_2Qubits(
            qCP_pulse_pars, qS_pulse_pars,
            flux_pulse_pars_qCP, flux_pulse_pars_qS, RO_pars_qCP,
            dist_dict, AWG=qS.AWG,  inter_swap_wait=self.inter_swap_wait,
            excitations='both')
        self.s.sweep_points = phases
        self.MC_nested = MC_nested

        self.pars_changed = True

    def acquire_data_point(self, **kw):
        if self.pars_changed:
            self.s.upload = True
            self.s.flux_pulse_pars_qCP['lambda1'] = self.lambda1
            self.s.prepare()
            self.s.upload = False
        self.pars_changed = False
        self.AWG.ch3_amp(self.qS.swap_amp())
        self.AWG.ch4_amp(self.qCP.swap_amp())

        self.MC_nested.set_sweep_function(self.s)
        self.MC_nested.set_detector_function(self.int_avg_det)
        self.MC_nested.set_sweep_points(self.phases)
        self.MC_nested.run('swap-CP-swap')
        a = SWAP_Cost_v2(show_guess=False, label='swap-CP-swap')

        return a.cost_func_val, a.dphi, a.osc_amp_0, a.osc_amp_1


class lambda1_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda1'
        self.parameter_name = 'lambda 1'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda1 = self.comp_det.lambda1
        if old_lambda1 != val:
            self.comp_det.lambda1 = val
            self.comp_det.pars_changed = True


class SWAP_Cost_v2(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):

        self.get_naming_and_values()

        cal_0I = self.measured_values[0][-4]
        cal_0Q = self.measured_values[1][-4]
        cal_1I = self.measured_values[0][-1]
        cal_1Q = self.measured_values[1][-1]

        self.measured_values[0][:] = (
            self.measured_values[0] - cal_0I)/(cal_1I-cal_0I)
        self.measured_values[1][:] = (
            self.measured_values[1] - cal_0Q)/(cal_1Q-cal_0Q)
        self.measured_values = self.measured_values
        self.sweep_points = self.sweep_points
        self.fit_data(**kw)
        self.make_figures(**kw)

        self.osc_amp_0 = self.fit_res[0].best_values['amplitude']
        self.osc_amp_1 = self.fit_res[1].best_values['amplitude']
        self.phi_0 = self.fit_res[0].best_values['phase']
        self.phi_1 = self.fit_res[1].best_values['phase']
        self.dphi = ((self.phi_1-self.phi_0)/(2*np.pi)*360) % 360

        if close_file:
            self.data_file.close()

    def fit_data(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.

        num_points = len(self.sweep_points)-4
        id_dat = self.measured_values[0][:num_points//2]
        id_swp = self.sweep_points[:num_points//2]
        ex_dat = self.measured_values[0][num_points//2:-4]
        ex_swp = self.sweep_points[num_points//2:-4]

        self.cost_func_val = max(id_dat-ex_dat) + max(ex_dat-id_dat)

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=id_dat, t=id_swp)
        self.fit_res[0] = model.fit(data=id_dat, t=id_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[0],
                                    var_name='id_fit')

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=ex_dat, t=ex_swp)
        self.fit_res[1] = model.fit(data=ex_dat, t=ex_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[1],
                                    var_name='ex_fit')

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))
        x_fine = np.linspace(min(self.sweep_points), max(self.sweep_points),
                             1000)
        self.ylabels = ['q_CP', 'q_S']
        for i in [0, 1]:
            if i == 0:
                plot_title = kw.pop('plot_title', ma.textwrap.fill(
                                    self.timestamp_string + '_' +
                                    self.measurementstring, 40))
            else:
                plot_title = ''
            self.axs[i].ticklabel_format(useOffset=False)
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=self.fig, ax=self.axs[i],
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            save=False,
                                            plot_title=plot_title)

            fine_fit = self.fit_res[i].model.func(
                x_fine, **self.fit_res[i].best_values)
            self.axs[0].plot(x_fine, fine_fit, label='fit')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    x_fine, **self.fit_res[i].init_values)
                self.axs[0].plot(x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)
