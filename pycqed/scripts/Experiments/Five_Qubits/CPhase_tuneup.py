from pycqed.Experiments.Five_Qubits import cost_functions_Leo_optimizations as cl




awg_swf.swap_CP_swap_2Qubits(
    qCP_pulse_pars, qS_pulse_pars,
    flux_pulse_pars_qCP, flux_pulse_pars_qS, RO_pars_qCP,
    dist_dict, AWG=qS.AWG,  inter_swap_wait=self.inter_swap_wait,
    excitations='both', CPhase=self.CPhase,
    reverse_control_target=self.reverse_control_target)





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
        self.MC.soft_avg(1)
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
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][0] = self.lambda1
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][1] = self.lambda2
            self.s.flux_pulse_pars_qCP['lambda_coeffs'][2] = self.lambda3
            self.s.flux_pulse_pars_qCP['amplitude'] = self.amplitude
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qCP
            self.s.flux_pulse_pars_qCP['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qCP

            self.s.flux_pulse_pars_qS['phase_corr_pulse_length'] = self.phase_corr_pulse_length_qS
            self.s.flux_pulse_pars_qS['phase_corr_pulse_amp'] = self.phase_corr_pulse_amp_qS
            self.s.prepare()
            self.s.upload = False
        self.pars_changed = False
        self.AWG.ch4_amp(self.qS.swap_amp())
        self.MC_nested.set_sweep_function(self.s)
        self.MC_nested.set_detector_function(self.int_avg_det)
        self.MC_nested.set_sweep_points(self.phases)
        label='swap_CP_swap_amp_{0:.4f}_l1_{1:.2f}_l2_{2:.2f}'.format(self.AWG.ch3_amp(),self.lambda1, self.lambda2)
        self.MC_nested.run(label)
        if not self.reverse_control_target:
            a = cl.SWAP_Cost(show_guess=False, label=label)
            return a.cost_func_val, a.dphi, a.osc_amp_0, a.osc_amp_1, a.phi_0, a.osc_amp_1-a.osc_amp_0

        else:
            a = cl.SWAP_Cost_reverse_control_target(show_guess=False,
                                                 label=label, single_qubit_phase_cost=self.single_qubit_phase_cost)
            return a.cost_func_val, a.dphi, a.phi_0_qCP, a.phi_0_qS, a.osc_amp_1_qCP-a.osc_amp_0_qCP
