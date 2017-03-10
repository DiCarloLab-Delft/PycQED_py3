dac_range = np.concatenate((np.arange(0,161,5),np.arange(0,-200,-5)))
IVVI.dac1(0.)
for idx,j in enumerate(dac_range):
    IVVI.dac1(d)
    try:
        QR.RO_acq_averages(1024)
        QR.find_resonator_frequency(freqs=np.arange(7.188e9,7.195e9,0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    # gets guess of qubit frequency
    f_guess = f_dac1(d)
    QR.f_qubit(f_guess)

    # SH = sh.SignalHound_USB_SA124B('SH')
    if (idx % 5) == 0:
        QR.calibrate_mixer_offsets(SH)

    if (idx % 2) == 0:
        QR.find_pulse_amplitude(amps=np.linspace(-0.3, 0.3, 31),
                                        N_steps=[1], max_n=100, take_fit_I=False)

    QR.find_frequency(method='ramsey', update=True)

    QR.find_pulse_amplitude(amps=np.linspace(-0.3, 0.3, 31),
                                    N_steps=[1], max_n=100, take_fit_I=False)

    try:
        i = 0
        T1_val = 1000e-9
        t_final_vec = [4.8e-6,12e-6,30e-6,60e-6,90e-6,120e-6]
        while(((t_final_vec[i]/T1_val)<4.) and i<len(t_final_vec)):
            t_final = t_final_vec[i]
            QR.measure_T1(times=np.linspace(0,t_final,61))
            ma_obj = ma.T1_Analysis(auto=True)
            T1_val = ma_obj.T1
            i = i+1
    except:
        print('Failed T1 scan')
    try:
        i = 0
        T2_val = 1000e-9
        t_final_vec = [4.8e-6,12e-6,30e-6,60e-6,90e-6,120e-6]
        while(((t_final_vec[i]/T2_val)<4.) and i<len(t_final_vec)):
            t_final = t_final_vec[i]
            QR.measure_ramsey(times=np.linspace(0,t_final,61), artificial_detuning=4./t_final)
            ma_obj = ma.Ramsey_Analysis(auto=True)
            T2_val = ma_obj.T2_star
            i = i+1
    except:
        print('Failed T2s scan')
    try:
        i = 0
        T2_val = 1000e-9
        t_final_vec = [4.8e-6,12e-6,30e-6,60e-6,90e-6,120e-6]
        while(((t_final_vec[i]/T2_val)<4.) and i<len(t_final_vec)):
            t_final = t_final_vec[i]
            QR.measure_echo(times=np.linspace(0,t_final,61), artificial_detuning=4./t_final)
            ma_obj = ma.Ramsey_Analysis(auto=True,label='Echo')
            T2_val = ma_obj.T2_star
            i = i+1
    except:
        print('Failed T2s scan')

IVVI.dac1(0.)