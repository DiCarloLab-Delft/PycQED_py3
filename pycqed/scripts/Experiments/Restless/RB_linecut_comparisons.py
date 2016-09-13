'''
Collection of small functions that perform basic experiments
Relies on the global namespace of the notebook!
'''


def OneD_linecuts():
    #resetless RB, 1D cuts
    AWG.visa_handle.timeout = 180000
    nr_seeds=400
    f0=VIP_mon_4_tek.f_qubit.get()-VIP_mon_4_tek.f_pulse_mod.get()
    set_trigger_fast()
    Dux.in2_out1_attenuation.set(1-.2)

    Dux.in1_out1_switch('ON')
    Dux.in2_out1_switch('ON')

    pulse_pars_duplex['amplitude'] = 0.5
    pulse_pars_duplex['motzoi'] = -.5


    scan_ranges = [[1-0, 1-1],
                   [1-.45, 1- .95],
                   [1-.55, 1- .85],
                   [1-.6, 1- .8]]

    for j in range(100):
        # Linecuts
        for i, nr_cliffords in enumerate([10, 20, 40,80]): #,120,160]:
            set_trigger_fast()
            scan_range = scan_ranges[i]
            sq.Randomized_Benchmarking_seq(pulse_pars_duplex, RO_pars, nr_cliffords=[nr_cliffords],
                                           nr_seeds=nr_seeds, post_msmt_delay=5e-6,
                                          cal_points=False, net_clifford=3, resetless=True)

            Dux.in1_out1_attenuation.set(.4) # ensure the RB seq will screw up so there is discr analysis possible
            CBox.lin_trans_coeffs.set([1, 0, 0, 1])

            AWG.start()
            # Find and rotate to optimal angle (inserted because set integration weights doesn't work)
            CBox.lin_trans_coeffs.set([1, 0, 0, 1])
            theta = 2*np.pi*VIP_mon_4_tek.measure_discrimination_fid()[2]/360
            rot_mat = [np.cos(-1*theta), -np.sin(-1*theta),
               np.sin(-1*theta), np.cos(-1*theta)]
            CBox.lin_trans_coeffs.set(rot_mat)

            # Determine threshold
            d=det.CBox_integration_logging_det(CBox, AWG)
            MC.set_sweep_function(swf.None_Sweep(sweep_control='hard'))
            MC.set_sweep_points(np.arange(8000))
            MC.set_detector_function(d)
            MC.run('threshold_determination')
            a=ma.SSRO_single_quadrature_discriminiation_analysis()
            CBox.sig0_threshold_line.set(int(a.opt_threshold))
            AWG.start()
            MC.set_sweep_function(Dux.in1_out1_attenuation)
            MC.set_sweep_points(np.linspace(scan_range[0], scan_range[1], 51))
            MC.set_detector_function(det.CBox_single_qubit_event_s_fraction(CBox))
            data = MC.run('rstls_RB_cut_{}cl_{}sds'.format(nr_cliffords, nr_seeds))
            ma.MeasurementAnalysis()

            # Take RB landscape (conventional)
            set_trigger_slow()
            MC.set_detector_function(det.CBox_single_qubit_fidelity_counter(CBox))
            data = MC.run('trad_RB_cut_{}cl_{}sds'.format(nr_cliffords, nr_seeds))
            ma.MeasurementAnalysis()


    # Restore settings from before the sweep
    Dux.in1_out1_attenuation.set(DUX_1_default)
    Dux.in2_out1_attenuation.set(DUX_2_default)

    CBox.lin_trans_coeffs.set([1, 0, 0, 1])