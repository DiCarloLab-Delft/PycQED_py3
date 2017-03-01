T1_lengths = [4.8e-6, 12e-6, 30e-6, 60e-6, 90e-6, 120e-6]
Echo_lengths = [4.8e-6, 12e-6, 30e-6, 60e-6, 90e-6, 120e-6]
qubit = QR # qubit
# dac scan parameters
dac_span = 140
dac_center = -16
dac_Step = 10
dac_param = IVVI.dac1 # parameter for dac
do_spec = False #do spectroscopy?
freq_calc = f_dac1 # function to predict the frequency
mixer_cal_period = 5
pi_pulse_period = 2


dac_range = np.concatenate((np.arange(dac_center, dac_center+dac_span/2, dac_step), np.arange(dac_center, dac_center-dac_span/2, -dac_step)))
dac_param(0.)
for idx, d in enumerate(dac_range):
    # sets dac value
    dac_param(d)
    # does resonator scan
    try:
        qubit.find_resonator_frequency(freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    if do_spec:
        # does spectroscopy scan
        try:
            qubit.find_frequency(freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
        except:
            print('Failed qubit fit')
    else:
        # if no spec, assigns qubit frequency through the method provided
        qubit.f_qubit(freq_calc(d))
    if ((idx % mixer_cal_period) == 0):
        # calibrates mixers
        qubit.calibrate_mixer_offsets(SH)
    if ((idx % pi_pulse_period) == 0):
        # calibrates pi pulse
        qubit.find_pulse_amplitude(amps=np.linspace(-0.5, 0.5, 51),
                                        N_steps=[1], max_n=100, take_fit_I=False)
    # fine-tunes the frequency
    qubit.find_frequency(method='ramsey', update=True)
    # calibrates pi pulse
    if (idx % pi_pulse_period) == 0:
        qubit.find_pulse_amplitude(amps=np.linspace(-0.3, 0.3, 31),
                                        N_steps=[1], max_n=100, take_fit_I=False)
    # does T1s
    try:
        i=0
        T1_val=1000e-9
        t_final_vec=T1_lengths
        while(((t_final_vec[i]/T1_val) < 5.) and i < len(t_final_vec)):
            t_final=t_final_vec[i]
            qubit.measure_T1(times=np.linspace(0, t_final, 61))
            ma_obj=ma.T1_Analysis(auto=True, close_fig=True)
            T1_val=ma_obj.T1
            i=i+1
    except:
        print('Failed T1 scans')
    # does Echoes
    try:
        i=0
        T2_val=1000.
        t_final_vec=Echo_lengths
        while(((t_final_vec[i]/T2_val) < 5.) and i < len(t_final_vec)):
            t_final=t_final_vec[i]
            qubit.measure_echo(
                times=np.linspace(0, t_final, 61), artificial_detuning=4./t_final)
            ma_obj=ma.Ramsey_Analysis(
                auto=True, label='Echo', close_fig=True)
            T2_val=ma_obj.T2_star
            i=i+1
    except:
        print('Failed Echo scans')
dac_param(0.)
