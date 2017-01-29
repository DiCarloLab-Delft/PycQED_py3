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

        self.AWG.ch4_amp(qubit.SWAP_amp())

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
    #function ment to only tune-up CPhase pulse
    def __init__(self,
                 qCP,
                 qS,
                 dist_dict,
                 MC_nested,
                 int_avg_det,
                 flux_pulse_pars_qCP,
                 flux_pulse_pars_qS,
                 phases=np.arange(0, 780, 20),
                 inter_swap_wait=100e-9,
                 upload=True, CPhase=True,
                 single_qubit_phase_cost=False,
                 reverse_control_target=False, **kw):

        self.name = 'CPhase_cost_func_det'
        self.detector_control = 'soft'
        self.upload = upload
        self.AWG = qCP.AWG
        self.phases = phases
        self.MC = MC_nested
        self.int_avg_det = int_avg_det
        self.inter_swap_wait = inter_swap_wait
        self.qCP = qCP
        self.qS = qS
        self.CPhase = CPhase
        self.reverse_control_target = reverse_control_target
        self.single_qubit_phase_cost = single_qubit_phase_cost
        if self.reverse_control_target:
            self.value_names = [
               'Cost function', 'phase_diff', 'phi_0_qCP', 'phi_0_qS', 'amp1_qCP-amp0_qCP']
            self.value_units = ['a.u.', 'deg', 'deg', 'deg', 'a.u.']
        else:
            self.value_names = [
                'Cost function', 'phase_diff',  'amp_no_exc', 'amp_single_exc', 'phi_0', 'amp1-amp0']
            self.value_units = ['a.u.', 'deg', '', '', 'deg', 'a.u.']

        qCP_pulse_pars, RO_pars_qCP = qCP.get_pulse_pars()
        qS_pulse_pars, RO_pars_qS = qS.get_pulse_pars()
        #flux_pulse_pars_qCP = qCP.get_flux_pars()

        #flux_pulse_pars_qS = qS.get_flux_pars()
        self.theta_f = flux_pulse_pars_qCP['theta_f']
        self.lambda1 = flux_pulse_pars_qCP['lambda_coeffs'][0]
        self.lambda2 = flux_pulse_pars_qCP['lambda_coeffs'][1]
        self.lambda3 = flux_pulse_pars_qCP['lambda_coeffs'][2]
        self.amplitude = flux_pulse_pars_qCP['amplitude']
        self.phase_corr_pulse_length_qCP = flux_pulse_pars_qCP['phase_corr_pulse_length']
        self.phase_corr_pulse_amp_qCP = flux_pulse_pars_qCP['phase_corr_pulse_amp']
        self.phase_corr_pulse_length_qS = flux_pulse_pars_qS['phase_corr_pulse_length']
        self.phase_corr_pulse_amp_qS = flux_pulse_pars_qS['phase_corr_pulse_amp']

        self.s = awg_swf.swap_CP_swap_2Qubits(
            qCP_pulse_pars, qS_pulse_pars,
            flux_pulse_pars_qCP, flux_pulse_pars_qS, RO_pars_qCP,
            dist_dict, AWG=qS.AWG,  inter_swap_wait=self.inter_swap_wait,
            excitations='both', CPhase=self.CPhase,
            reverse_control_target=self.reverse_control_target)
        self.s.sweep_points = phases
        self.MC_nested = MC_nested

        self.pars_changed = True

    def acquire_data_point(self, **kw):

        if self.pars_changed:
            self.s.upload = True
            self.s.flux_pulse_pars_qCP['theta_f'] = self.theta_f
            self.s.flux_pulse_pars_qCP['lambda_coeffs'] = np.array([
                self.lambda1, self.lambda2, self.lambda3])
            self.s.flux_pulse_pars_qCP['amplitude'] = self.amplitude
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qCP
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qCP

            self.s.flux_pulse_pars_qS['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qS
            self.s.flux_pulse_pars_qS['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qS
            self.s.prepare()
            self.s.upload = False
        self.pars_changed = False
        self.AWG.ch4_amp(self.qS.SWAP_amp())
        self.MC_nested.set_sweep_function(self.s)
        self.MC_nested.set_detector_function(self.int_avg_det)
        self.MC_nested.set_sweep_points(self.phases)
        label='swap_CP_swap_amp_{0:.4f}_l1_{1:.2f}_l2_{2:.2f}'.format(
            self.AWG.ch3_amp(), self.lambda1, self.lambda2)
        self.MC_nested.run(label)
        if not self.reverse_control_target:
            a = SWAP_Cost(show_guess=False, label=label)
            return a.cost_func_val, a.dphi, a.osc_amp_0, a.osc_amp_1, a.phi_0, a.osc_amp_1-a.osc_amp_0

        else:
            a = SWAP_Cost_reverse_control_target(show_guess=False,
                                                 label=label, single_qubit_phase_cost=self.single_qubit_phase_cost)
            return a.cost_func_val, a.dphi, a.phi_0_qCP, a.phi_0_qS, a.osc_amp_1_qCP-a.osc_amp_0_qCP


class CPhase_cost_func_det_Ramiro(det.Soft_Detector):
    #function ment to only tune-up CPhase pulse
    def __init__(self,
                 qCP,
                 qS,
                 dist_dict,
                 MC_nested,
                 int_avg_det,
                 flux_pulse_pars_qCP,
                 flux_pulse_pars_qS,
                 timings_dict,
                 sphasesweep=np.arange(0.75, 1.251+0.04, 0.01),
                 inter_swap_wait=100e-9,
                 upload=True, CPhase=True,
                 single_qubit_phase_cost=False,
                 reverse_control_target=False,
                 cost_function_choice=0, sweep_q=0, **kw):

        self.name = 'CPhase_cost_func_det'
        self.detector_control = 'soft'
        self.upload = upload
        self.AWG = qCP.AWG
        self.sphasesweep = sphasesweep
        self.MC = MC_nested
        self.int_avg_det = int_avg_det
        self.inter_swap_wait = inter_swap_wait
        self.qCP = qCP
        self.qS = qS
        self.CPhase = CPhase
        self.reverse_control_target = reverse_control_target
        self.single_qubit_phase_cost = single_qubit_phase_cost
        self.cost_function_choice = cost_function_choice
        self.sweep_q = sweep_q

        if self.cost_function_choice==0:
            self.value_names = ['Cost function']
            self.value_units = ['a.u.']
        elif self.cost_function_choice==1:
            self.value_names = ['Cost function']
            self.value_units = ['a.u.']
        elif self.cost_function_choice==2:
            self.value_names = ['Cost function', 'Fringe distance', 'phase_0']
            self.value_units = ['a.u.', 'deg', 'deg']
        qCP_pulse_pars, RO_pars_qCP = qCP.get_pulse_pars()
        qS_pulse_pars, RO_pars_qS = qS.get_pulse_pars()
        self.theta_f = flux_pulse_pars_qCP['theta_f']
        self.lambda1 = flux_pulse_pars_qCP['lambda_coeffs'][0]
        self.lambda2 = flux_pulse_pars_qCP['lambda_coeffs'][1]
        self.lambda3 = flux_pulse_pars_qCP['lambda_coeffs'][2]
        self.amplitude = flux_pulse_pars_qCP['amplitude']
        self.phase_corr_pulse_length_qCP = flux_pulse_pars_qCP['phase_corr_pulse_length']
        self.phase_corr_pulse_amp_qCP = flux_pulse_pars_qCP['phase_corr_pulse_amp']
        self.phase_corr_pulse_length_qS = flux_pulse_pars_qS['phase_corr_pulse_length']
        self.phase_corr_pulse_amp_qS = flux_pulse_pars_qS['phase_corr_pulse_amp']


        if self.cost_function_choice==0:
            self.s = awg_swf.swap_CP_swap_2Qubits_1qphasesweep(
                qCP_pulse_pars, qS_pulse_pars,
                flux_pulse_pars_qCP, flux_pulse_pars_qS, RO_pars_qCP,
                dist_dict, timings_dict=timings_dict,
                AWG=qS.AWG, inter_swap_wait=self.inter_swap_wait,
                upload=False,
                excitations='both', CPhase=self.CPhase,
                reverse_control_target=self.reverse_control_target,
                sweep_q=self.sweep_q)
        elif self.cost_function_choice==1:
            self.s = awg_swf.swap_CP_swap_2Qubits_1qphasesweep_amp(
                qCP_pulse_pars, qS_pulse_pars,
                flux_pulse_pars_qCP, flux_pulse_pars_qS, RO_pars_qCP,
                dist_dict, timings_dict=timings_dict,
                AWG=qS.AWG,  inter_swap_wait=self.inter_swap_wait,
                upload=False,
                excitations='both', CPhase=self.CPhase,
                reverse_control_target=self.reverse_control_target,
                sweep_q=self.sweep_q)
        else:
            raise Exception

        self.s.sweep_points = sphasesweep
        self.s.upload = True
        self.last_seq = self.s.prepare()
        self.s.upload = False
        self.MC_nested = MC_nested

        self.pars_changed = True
        self.cost_function_choice = cost_function_choice

    def acquire_data_point(self, **kw):
        if self.pars_changed:
            self.s.upload = False
            self.s.flux_pulse_pars_qCP['theta_f'] = self.theta_f
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][0] = self.lambda1
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][1] = self.lambda2
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][2] = self.lambda3
            self.s.flux_pulse_pars_qCP['amplitude'] = self.amplitude
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qCP
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qCP

            self.s.flux_pulse_pars_qS['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qS
            self.s.flux_pulse_pars_qS['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qS
            self.last_seq = self.s.prepare()
            self.s.upload = False
        self.pars_changed = False
        self.AWG.ch4_amp(self.qS.SWAP_amp())
        self.MC_nested.set_sweep_function(self.s)
        self.MC_nested.set_detector_function(self.int_avg_det)
        self.MC_nested.set_sweep_points(self.sphasesweep)
        label='swap_CP_swap_amp_{0:.4f}_l1_{1:.2f}_l2_{2:.2f}'.format(
            self.AWG.ch3_amp(), self.lambda1, self.lambda2)
        self.MC_nested.run(label)
        if self.cost_function_choice==0:
            a = SWAP_Cost_Ramiro(show_guess=False, label=label)
            return 1.-np.abs(a.cost_func_val)
            # return a.cost_func_val, a.phi_0, a.phi_1
        elif self.cost_function_choice==1:
            a = SWAP_Cost_Ramiro_amp(show_guess=False, label=label)
            return a.cost_func_val



#analysis functions
class SWAP_Cost(ma.Rabi_Analysis):

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
        self.phi_0_rad = self.fit_res[0].best_values['phase']
        self.phi_1_rad = self.fit_res[1].best_values['phase']

        self.dphi = ((((self.phi_1_rad-self.phi_0_rad)/(2*np.pi)*360)+180)% 360)-180
        # wrapping the phase at +270/-90 degrees
        self.phi_0 = ((((self.phi_0_rad)/(2*np.pi)*360)+180 )% 360)-180
        self.phi_1 = ((self.phi_1_rad)/(2*np.pi)*360) % 360
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

        #self.cost_func_val = 2-(max(id_dat-ex_dat) + max(ex_dat-id_dat))
        # we can calculate the cost function disreclty from the fit-parameters

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
        self.x_fine = np.linspace(min(self.sweep_points[:-4]), max(self.sweep_points[:-4]),
                             1000)
        self.fine_fit_0 = self.fit_res[0].model.func(
                self.x_fine, **self.fit_res[0].best_values)
        self.fine_fit_1 = self.fit_res[1].model.func(
                self.x_fine, **self.fit_res[1].best_values)
        #cost funcations penalizes extra for deviations from 180 degrees phase
        #difference and also penalizes the single-qubit error in the q_CPhase
        self.cost_func_val = (2-(max(self.fine_fit_0-self.fine_fit_1) +
                                max(self.fine_fit_1-self.fine_fit_0)))/2

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))

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
                                            plot_title=plot_title, marker='--o')

            fine_fit = self.fit_res[i].model.func(
                self.x_fine, **self.fit_res[i].best_values)
            self.axs[0].plot(self.x_fine, fine_fit, label='fit')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    self.x_fine, **self.fit_res[i].init_values)
                self.axs[0].plot(self.x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)

class SWAP_Cost_reverse_control_target(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True,
                             single_qubit_phase_cost=False, **kw):

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
        self.fit_data(single_qubit_phase_cost, **kw)
        self.make_figures(**kw)

        #analysing the qCP fits
        self.osc_amp_0_qCP = self.fit_res[0].best_values['amplitude']
        self.osc_amp_1_qCP = self.fit_res[1].best_values['amplitude']
        self.phi_0_rad_qCP = self.fit_res[0].best_values['phase']
        self.phi_1_rad_qCP = self.fit_res[1].best_values['phase']

        self.dphi_qCP = ((((self.phi_1_rad_qCP-self.phi_0_rad_qCP)/(2*np.pi)*360)+180)% 360)-180
        # wrapping the phase at +270/-90 degrees
        self.phi_0_qCP = ((((self.phi_0_rad_qCP)/(2*np.pi)*360)+90 )% 360)-90
        self.phi_1_qCP = ((self.phi_1_rad_qCP)/(2*np.pi)*360) % 360

        self.osc_amp_0_qS = self.fit_res[2].best_values['amplitude']
        self.osc_amp_1_qS = self.fit_res[3].best_values['amplitude']
        self.phi_0_rad_qS = self.fit_res[2].best_values['phase']
        self.phi_1_rad_qS = self.fit_res[3].best_values['phase']

        #analysing the qS fits
        self.dphi_qS = ((((self.phi_1_rad_qS-self.phi_0_rad_qS)/(2*np.pi)*360)+180)% 360)-180
        # wrapping the phase at +270/-90 degrees
        self.phi_0_qS = ((((self.phi_0_rad_qS)/(2*np.pi)*360)+90 )% 360)-90
        self.phi_1_qS = ((self.phi_1_rad_qS)/(2*np.pi)*360) % 360

        self.dphi = (self.dphi_qS+self.dphi_qCP)/2
        #constructing the cost function with subparts that scale from 0 to 1
        # contrast_cost = (self.cost_CPHASE_qCP+self.cost_CPHASE_qS)/2
        # cost_2Q_phase_error =  abs(self.dphi-180)/180
        # single_qubit_phase_cost = abs(self.phi_0_qCP-90)/90 + abs(self.phi_0_qS-90)/90
        # self.cost_func_val = 3*cost_2Q_phase_error + contrast_cost + single_qubit_phase_cost
        self.cost_func_val = (self.cost_CPHASE_qCP+self.cost_CPHASE_qS)/2

        if close_file:
            self.data_file.close()

    def fit_data(self, single_qubit_phase_cost, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['', '', '', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.

        num_points = len(self.sweep_points)-4
        #extracting the data for qCP (first half of the data)
        qCP_id_dat = self.measured_values[0][:num_points//4]
        qCP_id_swp = self.sweep_points[:num_points//4]
        qCP_ex_dat = self.measured_values[0][num_points//4:num_points//2]
        qCP_ex_swp = self.sweep_points[num_points//4:num_points//2]
        #extracting the data for qS (second half of the data)
        qS_id_dat = self.measured_values[1][num_points//2:num_points*3//4]
        qS_id_swp = self.sweep_points[num_points//2:num_points*3//4]
        qS_ex_dat = self.measured_values[1][num_points*3//4:-4]
        qS_ex_swp = self.sweep_points[num_points*3//4:-4]

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=qCP_id_dat, t=qCP_id_swp)
        self.fit_res[0] = model.fit(data=qCP_id_dat, t=qCP_id_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[0],
                                    var_name='qCP_id_fit')

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=qCP_ex_dat, t=qCP_ex_swp)
        self.fit_res[1] = model.fit(data=qCP_ex_dat, t=qCP_ex_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[1],
                                    var_name='qCP_ex_fit')

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=qS_id_dat, t=qS_id_swp)
        self.fit_res[2] = model.fit(data=qS_id_dat, t=qS_id_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[2],
                                    var_name='qS_id_fit')

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=qS_ex_dat, t=qS_ex_swp)
        self.fit_res[3] = model.fit(data=qS_ex_dat, t=qS_ex_swp, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res[3],
                                    var_name='qS_ex_fit')

        self.x_fine = np.linspace(min(self.sweep_points[:-4]), max(self.sweep_points[:-4]),
                             721)
        self.fine_fit_0 = self.fit_res[0].model.func(
                self.x_fine, **self.fit_res[0].best_values)
        self.fine_fit_1 = self.fit_res[1].model.func(
                self.x_fine, **self.fit_res[1].best_values)
        self.fine_fit_2 = self.fit_res[2].model.func(
                self.x_fine, **self.fit_res[2].best_values)
        self.fine_fit_3 = self.fit_res[3].model.func(
                self.x_fine, **self.fit_res[3].best_values)
        #cost funcations penalizes extra for deviations from 180 degrees phase
        #difference and also penalizes the single-qubit error in the q_CPhase
        # self.cost_CPHASE_qCP= (2-(max(self.fine_fit_0-self.fine_fit_1) +
        #                         max(self.fine_fit_1-self.fine_fit_0)))/2
        # self.cost_CPHASE_qS= (2-(max(self.fine_fit_2-self.fine_fit_3) +
        #                         max(self.fine_fit_2-self.fine_fit_3)))/2
        if single_qubit_phase_cost:
            self.cost_CPHASE_qCP = (abs(self.fine_fit_0[0]-0.5)+2*abs(self.fine_fit_0[90])+
                                    abs(self.fine_fit_1[0]-0.5)+
                                    abs(self.fine_fit_0[180]-0.5)+
                                    abs(self.fine_fit_1[180]-0.5)+2*abs(self.fine_fit_1[270]))/8
            self.cost_CPHASE_qS = (abs(self.fine_fit_2[0]-0.5)+2*abs(self.fine_fit_2[90])+
                                    abs(self.fine_fit_3[0]-0.5)+
                                    abs(self.fine_fit_2[180]-0.5)+
                                    abs(self.fine_fit_3[180]-0.5)+2*abs(self.fine_fit_3[270]))/8
        else:
            self.cost_CPHASE_qCP = (2-(max(self.fine_fit_0-self.fine_fit_1) +
                        max(self.fine_fit_1-self.fine_fit_0)))/2
            self.cost_CPHASE_qS = (2-(max(self.fine_fit_2-self.fine_fit_3) +
                        max(self.fine_fit_3-self.fine_fit_2)))/2




    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))

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
                                            plot_title=plot_title,
                                            marker='--o')

            fine_fit_0 = self.fit_res[i*2].model.func(
                self.x_fine, **self.fit_res[i*2].best_values)
            self.axs[i].plot(self.x_fine, fine_fit_0, label='fit_0')
            fine_fit_1 = self.fit_res[i*2+1].model.func(
                self.x_fine, **self.fit_res[i*2+1].best_values)
            self.axs[i].plot(self.x_fine, fine_fit_1, label='fit_1')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    self.x_fine, **self.fit_res[i].init_values)
                self.axs[0].plot(self.x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc=0)
        self.save_fig(self.fig, fig_tight=False, **kw)

class SWAP_Cost_Ramiro(ma.Rabi_Analysis):

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
        self.sweep_points = self.sweep_points
        self.make_figures(**kw)

        self.cost_func_val = self.sweep_points[np.argmin(self.measured_values[0][:-4])]
        if close_file:
            self.data_file.close()

    def fit_data(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.

        num_points = len(self.sweep_points)-4
        id_dat = self.measured_values[1][num_points//2:3*num_points//4]
        id_swp = self.sweep_points[num_points//2:3*num_points//4]
        ex_dat = self.measured_values[1][3*num_points//4:-4]
        ex_swp = self.sweep_points[3*num_points//4:-4]

        # self.cost_func_val = 2-(max(id_dat-ex_dat) + max(ex_dat-id_dat))
        # we can calculate the cost function disreclty from the fit-parameters

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
        self.x_fine = np.linspace(min(self.sweep_points[:-4]),
                                  max(self.sweep_points[:-4]),
                                  1000)
        self.fine_fit_0 = self.fit_res[0].model.func(
                self.x_fine, **self.fit_res[0].best_values)
        self.fine_fit_1 = self.fit_res[1].model.func(
                self.x_fine, **self.fit_res[1].best_values)
        # cost funcations penalizes extra for deviations from 180 degrees phase
        # difference and also penalizes the single-qubit error in the q_CPhase
        # self.cost_func_val = max(abs(self.fine_fit_0-self.fine_fit_1))
        self.cost_func_val = max(abs(id_dat-ex_dat))

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))

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
                                            plot_title=plot_title, marker='--o')

            # fine_fit = self.fit_res[i].model.func(
            #     self.x_fine, **self.fit_res[i].best_values)
            # self.axs[0].plot(self.x_fine, fine_fit, label='fit')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    self.x_fine, **self.fit_res[i].init_values)
                self.axs[0].plot(self.x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)


class SWAP_Cost_Ramiro_amp(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):

        self.get_naming_and_values()

        cal_0I = np.mean([self.measured_values[0][-4],self.measured_values[0][-3]])
        cal_0Q = np.mean([self.measured_values[0][-4],self.measured_values[0][-2]])
        cal_1I = np.mean([self.measured_values[0][-1],self.measured_values[0][-2]])
        cal_1Q = np.mean([self.measured_values[0][-1],self.measured_values[0][-3]])

        self.measured_values[0][:] = (
            self.measured_values[0] - cal_0I)/(cal_1I-cal_0I)
        self.measured_values[1][:] = (
            self.measured_values[1] - cal_0Q)/(cal_1Q-cal_0Q)
        # self.measured_values = self.measured_values
        self.sweep_points = self.sweep_points
        self.fit_data(**kw)
        self.make_figures(**kw)

        if close_file:
            self.data_file.close()

    def fit_data(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['', '']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.

        # num_points = len(self.sweep_points)-4
        # id_dat = self.measured_values[1][num_points//2:3*num_points//4]
        # id_swp = self.sweep_points[num_points//2:3*num_points//4]
        # ex_dat = self.measured_values[1][3*num_points//4:-4]
        # ex_swp = self.sweep_points[3*num_points//4:-4]

        num_points = len(self.sweep_points)-4
        id_dat = self.measured_values[0][:num_points//2]
        id_swp = self.sweep_points[:num_points//2]
        ex_dat = self.measured_values[0][num_points//2:-4]
        ex_swp = self.sweep_points[num_points//2:-4]

        # self.cost_func_val = 2-(max(id_dat-ex_dat) + max(ex_dat-id_dat))
        # we can calculate the cost function disreclty from the fit-parameters

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
        self.x_fine = np.linspace(min(self.sweep_points[:-4]),
                                  max(self.sweep_points[:-4]),
                                  1000)
        self.fine_fit_0 = self.fit_res[0].model.func(
                self.x_fine, **self.fit_res[0].best_values)
        self.fine_fit_1 = self.fit_res[1].model.func(
                self.x_fine, **self.fit_res[1].best_values)
        # cost funcations penalizes extra for deviations from 180 degrees phase
        # difference and also penalizes the single-qubit error in the q_CPhase
        # self.cost_func_val = max(abs(self.fine_fit_0-self.fine_fit_1))
        self.cost_func_val = max(abs(id_dat-ex_dat))

    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))

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
                                            plot_title=plot_title, marker='--o')

            # fine_fit = self.fit_res[i].model.func(
            #     self.x_fine, **self.fit_res[i].best_values)
            # self.axs[0].plot(self.x_fine, fine_fit, label='fit')
            if show_guess:
                fine_fit = self.fit_res[i].model.func(
                    self.x_fine, **self.fit_res[i].init_values)
                self.axs[0].plot(self.x_fine, fine_fit, label='guess')
                self.axs[i].legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)


# sweep functions
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



class lambda2_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda2'
        self.parameter_name = 'lambda 2'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda2 = self.comp_det.lambda2
        if old_lambda2 != val:
            self.comp_det.lambda2 = val
            self.comp_det.pars_changed = True

class lambda3_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda3'
        self.parameter_name = 'lambda 3'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda3 = self.comp_det.lambda3
        if old_lambda3 != val:
            self.comp_det.lambda3 = val
            self.comp_det.pars_changed = True

class phase_corr_pulse_amp_sweep_qCP(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'phase_corr_pulse_amp_qCP'
        self.parameter_name = 'phase correction pulse amp qCP'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_phase_corr_pulse_amp = self.comp_det.phase_corr_pulse_amp_qCP
        if old_phase_corr_pulse_amp != val:
            self.comp_det.phase_corr_pulse_amp_qCP = val
            self.comp_det.pars_changed = True

class phase_corr_pulse_length_sweep_qCP(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'phase_corr_pulse_length_qCP'
        self.parameter_name = 'phase correction pulse length qCP'
        self.unit = 'ns'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_phase_corr_pulse_length = self.comp_det.phase_corr_pulse_length_qCP
        if old_phase_corr_pulse_length != val:
            self.comp_det.phase_corr_pulse_length_qCP = val
            self.comp_det.pars_changed = True

class phase_corr_pulse_amp_sweep_qS(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'phase_corr_pulse_amp_qS'
        self.parameter_name = 'phase correction pulse amp qS'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_phase_corr_pulse_amp = self.comp_det.phase_corr_pulse_amp_qS
        if old_phase_corr_pulse_amp != val:
            self.comp_det.phase_corr_pulse_amp_qS = val
            self.comp_det.pars_changed = True

class phase_corr_pulse_length_sweep_qS(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'phase_corr_pulse_length_qS'
        self.parameter_name = 'phase correction pulse length qS'
        self.unit = 'ns'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_phase_corr_pulse_length = self.comp_det.phase_corr_pulse_length_qS
        if old_phase_corr_pulse_length != val:
            self.comp_det.phase_corr_pulse_length_qS = val
            self.comp_det.pars_changed = True


class theta_f_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'theta_f'
        self.parameter_name = 'theta_f'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_theta_f = self.comp_det.theta_f
        if old_theta_f != val:
            self.comp_det.theta_f = val
            self.comp_det.pars_changed = True

class phase_corr_pulse_amp_sweep_qCP(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'phase_corr_pulse_amp_qCP'
        self.parameter_name = 'phase correction pulse amp qCP'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_phase_corr_pulse_amp = self.comp_det.phase_corr_pulse_amp_qCP
        if old_phase_corr_pulse_amp != val:
            self.comp_det.phase_corr_pulse_amp_qCP = val
            self.comp_det.pars_changed = True
