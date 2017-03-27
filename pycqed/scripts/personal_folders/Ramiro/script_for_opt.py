

        sweep_functions = []
        x0 = []
        init_steps = []
        for parameter:
            sweep_functions.append(cb_swf.LutMan_amp180_90(self.LutMan))
            x0.append(amp_guess)
            init_steps.append(a_step)

        MC.set_sweep_functions(sweep_functions)

        if len(sweep_functions) != 1:
            # noise ensures no_improv_break sets the termination condition
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': x0,
                            'initial_step': init_steps,
                            'no_improv_break': 10,
                            'minimize': False,
                            'maxiter': 500}
        elif len(sweep_functions) == 1:
            # Powell does not work for 1D, use brent instead
            brack = (x0[0]-5*init_steps[0], x0[0])
            # Ensures relative change in parameter is relevant
            if parameters == ['frequency']:
                tol = 1e-9
            else:
                tol = 1e-3
            print('Tolerance:', tol, init_steps[0])
            print(brack)
            ad_func_pars = {'adaptive_function': brent,
                            'brack': brack,
                            'tol': tol,  # Relative tolerance in brent
                            'minimize': False}
        MC.set_adaptive_function_parameters(ad_func_pars)

        MC.run(name=name, mode='adaptive')


chevron_det = cdet.Chevron_optimization_v1()
MC.set_detector_function(chevron_det)

# Defining amp parameter
def get_chevron_amp_par(det):
    return det.chevron_amp
def set_chevron_amp_par(val, det):
    det.chevron_amp = val
    return det.chevron_amp
chevron_amp_par = qc.Parameter(name='AWG_amplitude',
                              unit='Vpp',
                              label='AWG amplitude')
chevron_amp_par.get = lambda: get_chevron_amp_par(chevron_det)
chevron_amp_par.set = lambda val: set_chevron_amp_par(val, chevron_det)
