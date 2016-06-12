exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())

#calibrating phases and Dux default attenuations
calibrate_duplexer_phase_2D(pulse_pars)
DUX_1_default = VIP_mon_2_dux.Mux_G_att()
DUX_2_default = VIP_mon_2_dux.Mux_D_att()


nr_cliffords = [80, 300]
nr_seeds= 200#200
DUX_1_init_steps = [+0.02, +0.01]
DUX_2_init_steps = [+0.1, +0.05]


detector_restless=det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional=det.CBox_single_qubit_frac1_counter(CBox)

methods=['traditional', 'restless']
#methods=['restless']

init1=[-0.033, 0.033, -0.033, 0.033]
init2=[-0.2, -0.2, 0.33, 0.33]
# init1=[-0.033]
# init2=[-0.2]

for i in range(100):
        for j in range(len(init1)):
            DUX_1_init= init1[j]+DUX_1_default
            DUX_2_init= init2[j]+DUX_2_default
            for method in methods:
                CBox.nr_averages(2048)
                set_trigger_slow()
                #returning duplexer to original values for phase calibration
                Dux.in1_out1_attenuation(DUX_1_default)
                Dux.in2_out1_attenuation(DUX_2_default)
                VIP_mon_2_dux.Mux_G_att(DUX_1_default)
                VIP_mon_2_dux.Mux_D_att(DUX_2_default)

                #measure T1, tunne frequency, calibrate optimal weight-functions for RB comparisson
                VIP_mon_2_dux.measure_T1(np.linspace(0, 120e-6, 41))
                VIP_mon_2_dux.find_frequency(method='ramsey',
                                             steps=[30,100,300], update=True)
                VIP_mon_2_dux.measure_ssro(set_integration_weights=True) #calibrating SSRO (sets Dux parameters to default)

                #tuning motzoi and amplitude numerically
                sweep_pars = [pw.wrap_par_remainder(Dux.in1_out1_attenuation, remainder=1),
                            pw.wrap_par_remainder(Dux.in2_out1_attenuation, remainder=1)]

                x0 = [DUX_1_init, DUX_2_init] # setting x0 for the first round of optimization
                for nr_clifford, DUX_1_init_step, DUX_2_init_step in zip(nr_cliffords, DUX_1_init_steps, DUX_2_init_steps):

                    steps = [DUX_1_init_step, DUX_2_init_step]
                    ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': x0,
                            'initial_step': steps,
                            'no_improv_break': 35,
                            'sigma':.5,
                            'minimize': True,
                            'maxiter': 500}
                    if method is 'restless':
                        sq.Randomized_Benchmarking_seq(pulse_pars, RO_pars, [nr_clifford], nr_seeds=nr_seeds,
                                       net_clifford=3, post_msmt_delay=3e-6, cal_points=False, resetless=True)
                        detector= detector_restless
                        name='restless_RB_optimization_{}Cl'.format(nr_clifford)
                        set_trigger_fast()

                    elif method is 'traditional':
                        sq.Randomized_Benchmarking_seq(pulse_pars, RO_pars, [nr_clifford], nr_seeds=nr_seeds,
                                                   net_clifford=0, post_msmt_delay=3e-6, cal_points=False, resetless=True)
                        detector= detector_traditional
                        name = 'traditional_RB_optimization_{}Cl'.format(nr_clifford)
                        set_trigger_slow()
                    AWG.start()
                    calibrate_RO_threshold_no_rotation()
                    ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': x0,
                        'initial_step': steps,
                        'no_improv_break': 35,
                        'sigma':.5,
                        'minimize': True,
                        'maxiter': 500}
                    MC.set_sweep_functions(sweep_pars)
                    MC.set_adaptive_function_parameters(ad_func_pars)
                    MC.set_detector_function(detector)
                    AWG.start()
                    MC.run(name=name,
                       mode='adaptive')
                    ma.OptimizationAnalysis(close_fig=True)
                    ma.OptimizationAnalysis_v2(close_fig=True)
                    #overwriting x0 for the second round of optimization
                    x0 = [Dux.in1_out1_attenuation(), Dux.in2_out1_attenuation()]


                #verification of tuned pulses after the second round
                a = ma.T1_Analysis(label='T1', auto=True)
                T1 = a.T1
                set_trigger_slow()
                set_CBox_cos_sine_weigths(VIP_mon_2_dux.f_RO_mod())
                CBox.nr_averages(4096)
                measure_allXY(pulse_pars, RO_pars)
                measure_RB(pulse_pars, RO_pars, upload=True, T1=T1, close_fig=True)


