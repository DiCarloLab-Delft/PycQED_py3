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
                 q0,
                 q1,
                 dist_dict,
                 MC_nested,
                 lambda1,
                 phases=np.arange(0, 780, 20),
                 inter_swap_wait=50e-9,
                 upload=True,  **kw):

        self.name = 'CPhase_cost_func_det'
        self.value_names = ['Cost function', 'phase_diff', 'amp_no_id', 'amp_id']
        self.value_units = ['a.u.', 'deg', '', '']
        self.detector_control = 'soft'
        self.upload = upload
        self.AWG = q0.AWG
        self.phases = phases
        self.MC = MC_nested
        self.MC.soft_avg(20)

        self.q0 = q0
        self.q1 = q1
        self.lambda1 = lambda1
        q0.swap_amp(1.132)

        q0_pulse_pars, RO_pars_q0 = q0.get_pulse_pars()
        q1_pulse_pars, RO_pars_q1 = q1.get_pulse_pars()
        flux_pulse_pars_q0 = q0.get_flux_pars()
        flux_pulse_pars_q0['pulse_type'] = 'MartinisFluxPulse'
        flux_pulse_pars_q0['lambda0'] = 1
        flux_pulse_pars_q0['lambda1'] = lambda1


        flux_pulse_pars_q1 = q1.get_flux_pars()
        self.s0 = awg_swf.swap_CP_swap_2Qubits(
            q0_pulse_pars, q1_pulse_pars,
            flux_pulse_pars_q0, flux_pulse_pars_q1, RO_pars_q0,
            dist_dict, AWG=self.AWG,  inter_swap_wait=100e-9-q0.swap_time(),
            identity=False)
        self.s0.sweep_points = phases
        self.s1 = awg_swf.swap_CP_swap_2Qubits(
            q0_pulse_pars, q1_pulse_pars,
            flux_pulse_pars_q0, flux_pulse_pars_q1, RO_pars_q0,
            dist_dict, AWG=self.AWG,  inter_swap_wait=100e-9-q0.swap_time(),
            identity=True)
        self.s1.sweep_points = phases
        self.MC_nested = MC_nested

        self.pars_changed = True

    def acquire_data_point(self, **kw):
        if self.pars_changed:
            self.s0.upload = True
            self.s0.flux_pulse_pars_q0['lambda1'] = self.lambda1
            self.s0.prepare()
            self.s0.upload=False
            self.s1.upload = True
            self.s1.flux_pulse_pars_q0['lambda1'] = self.lambda1
            self.s1.prepare()
            self.s1.upload=False
        self.pars_changed = False
        self.MC_nested.set_sweep_function(self.s0)
        self.AWG.load_awg_file('swap_CP_swap_2Qubits_Id{}_FILE.AWG'.format(False))
        self.AWG.ch3_amp(self.q1.swap_amp())
        self.AWG.ch4_amp(self.q0.swap_amp())
        for ch in range(1, 5):
            self.AWG.set('ch{}_state'.format(ch), 1)
        self.MC_nested.set_detector_function(self.q0.int_avg_det_rot)
        self.MC_nested.set_sweep_points(self.phases)
        self.MC_nested.run('swap-CP-swap_no_identity')

        self.MC_nested.set_sweep_function(self.s1)
        self.AWG.load_awg_file('swap_CP_swap_2Qubits_Id{}_FILE.AWG'.format(True))
        self.AWG.ch3_amp(self.q1.swap_amp())
        self.AWG.ch4_amp(self.q0.swap_amp())
        for ch in range(1, 5):
            self.AWG.set('ch{}_state'.format(ch), 1)
        self.MC_nested.set_detector_function(self.q0.int_avg_det_rot)
        self.MC_nested.set_sweep_points(self.phases)
        self.MC_nested.run('swap-CP-swap_identity')

        a = SWAP_Cost_Leo1(show_guess=True, label='swap_no_identity')
        b = SWAP_Cost_Leo1(show_guess=True, label='swap_identity')
        mab = max(a.measured_values[0][:-4] - b.measured_values[0][:-4])
        mba = max(b.measured_values[0][:-4] - a.measured_values[0][:-4])
        cf = mab+mba
        amp_a = a.fit_res[0].best_values['amplitude']
        amp_b = b.fit_res[0].best_values['amplitude']
        phi_a = a.fit_res[0].best_values['phase']
        phi_b = b.fit_res[0].best_values['phase']
        phase_diff = ((phi_b-phi_a)/(2*np.pi)*360)%360

        return cf, phase_diff, amp_a, amp_b

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
