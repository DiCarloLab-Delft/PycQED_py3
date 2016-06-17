exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())
VIP_mon_2_dux = VIP_mon_2_dux
pw = pw
Dux = Dux
import numpy as np


#calibrating phases and Dux default attenuations
pulse_pars, RO_pars = calibrate_pulse_pars_conventional()
DUX_1_default = VIP_mon_2_dux.Mux_G_att()
DUX_2_default = VIP_mon_2_dux.Mux_D_att()
f_default = VIP_mon_2_dux.f_qubit()-VIP_mon_2_dux.f_pulse_mod()

#RB tuning sequence parameters
nr_cliffords = [80, 300]
nr_seeds= 200#200

# Ansatz parameters for numerical optimization
init1=[-0.033, 0.033] #duplexer 1
init2=[-0.2,  0.33] # duplexer 2
init3=[-0.5e6, 0.5e6] # frequency

#ansatz steps, 80 cliffords/300 cliffords
DUX_1_init_steps = [+0.02, +0.01]
DUX_2_init_steps = [+0.1, +0.05]
f_init_steps = [+0.5e6, +0.05e6]

#tuning methods
methods=['traditional', 'restless']
two_par=True
GST=True
#methods = ['restless']

#defingin the detectors
detector_restless=det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional=det.CBox_single_qubit_frac1_counter(CBox)


# Generating the required optimization sequences with the tuned updated pulse pars
for nr_clifford in nr_cliffords:
    sq.Randomized_Benchmarking_seq(
        pulse_pars, RO_pars, [nr_clifford],
        seq_name='RB_rstl_opt_seq_{}'.format(nr_clifford),
        nr_seeds=nr_seeds,
        net_clifford=3, post_msmt_delay=3e-6, cal_points=False, resetless=True)

    sq.Randomized_Benchmarking_seq(
        pulse_pars, RO_pars, [nr_clifford], nr_seeds=nr_seeds,
        seq_name='RB_conv_opt_seq_{}'.format(nr_clifford),
        net_clifford=0, post_msmt_delay=3e-6, cal_points=False, resetless=True)

#ensures a file with the name 'RB_verification_seq' is created
measure_RB(pulse_pars, RO_pars, upload=True, T1=25e-6, close_fig=True)
#ensures a file with the name 'GST_seq_FILE' is created
if GST:
 measure_GST(upload=True,  l=512, nr_elts=6103, nr_logs=100)

for i in range(10000):
    for j in range(len(init1)):
        for k in range(len(init2)):
            for l in range(len(init3)):
                DUX_1_init = init1[j] + DUX_1_default
                DUX_2_init = init2[k] + DUX_2_default
                f_init = init3[l]+f_default
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
                    if two_par:
                        VIP_mon_2_dux.find_frequency(method='ramsey',
                                                  steps=[30,100,300], update=True)
                    VIP_mon_2_dux.measure_ssro(set_integration_weights=True) #calibrating SSRO (sets Dux parameters to default)

                    #tuning motzoi and amplitude numerically
                    # sweep_pars = [pw.wrap_par_remainder(Dux.in1_out1_attenuation, remainder=1),
                    #             pw.wrap_par_remainder(Dux.in2_out1_attenuation, remainder=1)]
                    if two_par:
                        sweep_pars = [pw.wrap_par_remainder(Dux.in1_out1_attenuation, remainder=1),
                                    pw.wrap_par_remainder(Dux.in2_out1_attenuation, remainder=1)]
                    else:
                        sweep_pars = [pw.wrap_par_remainder(Dux.in1_out1_attenuation, remainder=1),
                                    pw.wrap_par_remainder(Dux.in2_out1_attenuation, remainder=1),
                                    pw.wrap_par_set_get(Qubit_LO.frequency)]

                    #x0 = [DUX_1_init, DUX_2_init] # setting x0 for the first round of optimization
                    if two_par:
                        [DUX_1_init, DUX_2_init]
                    else:
                        x0 = [DUX_1_init, DUX_2_init, f_init]
                    #for nr_clifford, DUX_1_init_step, DUX_2_init_step in zip(nr_cliffords, DUX_1_init_steps, DUX_2_init_steps):
                    for nr_clifford, DUX_1_init_step, DUX_2_init_step, f_init_step in zip(nr_cliffords, DUX_1_init_steps, DUX_2_init_steps, f_init_steps):

                        #teps = [DUX_1_init_step, DUX_2_init_step]
                        if two_par:
                            steps = [DUX_1_init_step, DUX_2_init_step]
                        else:
                            steps = [DUX_1_init_step, DUX_2_init_step, f_init_step]
                        ad_func_pars = {'adaptive_function': nelder_mead,
                                        'x0': x0,
                                        'initial_step': steps,
                                        'no_improv_break': 35,
                                        'sigma':.5,
                                        'minimize': True,
                                        'maxiter': 500}
                        if method is 'restless':
                            station.pulsar.load_awg_file('RB_rstl_opt_seq_{}_FILE.AWG'.format(nr_clifford))
                            # sq.Randomized_Benchmarking_seq(pulse_pars, RO_pars, [nr_clifford], nr_seeds=nr_seeds,
                            #                net_clifford=3, post_msmt_delay=3e-6, cal_points=False, resetless=True)
                            detector= detector_restless
                            name='restless_RB_optimization_{}Cl'.format(nr_clifford)
                            set_trigger_fast()

                        elif method is 'traditional':
                            station.pulsar.load_awg_file('RB_conv_opt_seq_{}_FILE.AWG'.format(nr_clifford))
                            # sq.Randomized_Benchmarking_seq(pulse_pars, RO_pars, [nr_clifford], nr_seeds=nr_seeds,
                            #                            net_clifford=0, post_msmt_delay=3e-6, cal_points=False, resetless=True)
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
                        if two_par:
                            x0 = [Dux.in1_out1_attenuation(), Dux.in2_out1_attenuation()]
                        else:
                            x0 = [Dux.in1_out1_attenuation(), Dux.in2_out1_attenuation(), Qubit_LO.frequency()]
                        # x0 = [Dux.in1_out1_attenuation(), Dux.in2_out1_attenuation()]

                    #verification of tuned pulses after the second round
                    a = ma.T1_Analysis(label='T1', auto=True)
                    T1 = a.T1
                    set_trigger_slow()
                    CBox.nr_averages(4096)
                    station.pulsar.load_awg_file('RB_verification_seq_FILE.AWG')
                    measure_RB(pulse_pars, RO_pars, upload=False, T1=T1, close_fig=True)
                    measure_allXY(pulse_pars, RO_pars)
                    # added GST to loop (still needs a retest)
                    if GST:
                        station.pulsar.load_awg_file('GST_seq_FILE.AWG')
                        measure_GST(upload=False,  l=512, nr_elts=6103, nr_logs=100)
                    # 75s loading, 120s msmt

