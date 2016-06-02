exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())
try:
    SH.closeDevice()
except:
    pass
reload(sh)
SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None)
station.add_component(SH)

f_qubit_LO_default=VIP_mon_2_dux.f_qubit()-VIP_mon_2_dux.f_pulse_mod()
DUX_1_default=0.3
DUX_2_default=0.7
Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('ON')
pulse_pars, RO_pars = VIP_mon_2_dux.get_pulse_pars()

#parameters for tuning
nr_cliffords = 400
nr_seeds=200#200
DUX_1_init_step = +0.01
DUX_2_init_step = +0.02
DUX_3_init_step = +1000

#f_qubit_LO_init=f_qubit_LO_default+0.05e6
#f_qubit_LO_init_step=0.05e6

detector_restless=det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional=det.CBox_single_integration_average_det(CBox)

methods=['traditional']
init1=[0.01]
init2=[0.01]
init3=[10693, 10693]

for i in range(1):
    for j in range(len(init1)):
        DUX_1_init= init1[j]+DUX_1_default
        DUX_2_init= init2[j]+DUX_2_default
        DUX_3_init= init3[j]
        for method in methods:
            CBox.nr_averages(2048)
            #returning duplexer to original values for phase calibration
            Dux.in1_out1_attenuation(DUX_1_default)
            Dux.in2_out1_attenuation(DUX_2_default)
            VIP_mon_2_dux.Mux_G_att(DUX_1_default)
            VIP_mon_2_dux.Mux_D_att(DUX_2_default)
            #VIP_mon_2_dux.measure_T1(np.linspace(0, 120e-6, 41))

            #measure T1
            a = ma.T1_Analysis(label='T1', auto=True)
            T1 = a.T1
            #starting with calibration of frequency and duplexer phase
            Dux.in1_out1_phase(init3)
            VIP_mon_2_dux.find_frequency(method='ramsey',steps=[30,100,300], update=True)
            #exec(open(PyCQEDpath+'\scripts\Experiments\Restless\duplexer_phase_cal.py').read())

            #tuning motzoi and amplitude numerically
            # sweep_pars = [Qubit_LO.frequency, Dux.in1_out1_attenuation, Dux.in2_out1_attenuation, ]
            # x0 = [f_qubit_LO_init, DUX_1_init DUX_2_init]
            # steps = [f_qubit_LO_init_step, DUX_1_init_step DUX_2_init_step ]
            sweep_pars = [Dux.in1_out1_attenuation, Dux.in2_out1_attenuation]
            x0 = [DUX_1_init, DUX_2_init]
            steps = [DUX_1_init_step, DUX_2_init_step]
            # sweep_pars = [Dux.in1_out1_attenuation, Dux.in2_out1_attenuation, Dux.in2_out1_phase]
            # x0 = [DUX_1_init, DUX_2_init, DUX_3_init]
            # steps = [DUX_1_init_step, DUX_2_init_step, DUX_3_init_step]

            set_trigger_slow()
            VIP_mon_2_dux.measure_ssro(set_integration_weights=True) #calibrating SSRO (sets Dux parameters to default)
            set_trigger_fast()
            sq.Randomized_Benchmarking_seq(pulse_pars, RO_pars, [nr_cliffords], nr_seeds=nr_seeds,
                                           net_clifford=3, post_msmt_delay=3e-6, cal_points=False, resetless=True)
            AWG.start()
            calibrate_RO_threshold_no_rotation()

            if method is 'restless':
                ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': x0,
                    'initial_step': steps,
                    'no_improv_break': 35,
                    'sigma':.5,
                    'minimize': True,
                    'maxiter': 500}
                AWG.start()
                MC.set_sweep_functions(sweep_pars)
                MC.set_adaptive_function_parameters(ad_func_pars)
                set_trigger_fast()
                MC.set_detector_function(detector_restless)
                MC.run(name='restless_RB_optimization_{}Cl_{}sds'.format(nr_cliffords, nr_seeds),
                   mode='adaptive')
            elif method is 'traditional':
                ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': x0,
                    'initial_step': steps,
                    'no_improv_break': 35,
                    'sigma':.5,
                    'minimize': False,
                    'maxiter': 500}
                AWG.start()
                MC.set_sweep_functions(sweep_pars)
                MC.set_adaptive_function_parameters(ad_func_pars)
                set_trigger_slow()
                MC.set_detector_function(detector_traditional)
                MC.run(name='traditional_RB_optimization_{}Cl_{}sds'.format(nr_cliffords, nr_seeds),
                   mode='adaptive')
            ma.OptimizationAnalysis(close_fig=True)
            ma.OptimizationAnalysis_v2(close_fig=True)

            #verification resetless pulses
            set_trigger_slow()
            set_CBox_cos_sine_weigths(VIP_mon_2_dux.f_RO_mod())
            CBox.nr_averages(4096)
            measure_allXY(pulse_pars, RO_pars)
            #measure_RB(pulse_pars, RO_pars, upload=True, T1=T1, close_fig=True)


#old stuff

 # #tuning motzoi and amplitude traditionally
        # VIP_mon_2_dux.measure_motoi_XY(motzois=np.linspace(-.6, -.3, 21))
        # VIP_mon_2_dux.find_pulse_amplitude(amps=np.linspace(-.5, .5, 31), N_steps=[3,7,19], max_n=100, take_fit_I=False)
        # CBox.nr_averages(2048*2)
