import qcodes as qc

q0 = qc.station.components['q0']
q1 = qc.station.components['q1']
q2 = qc.station.components['q2']
q3 = qc.station.components['q3']
q4 = qc.station.components['q4']

MC = qc.station.MC
LutManMan = qc.station.components['LutManMan']

list_qubits = [q0, q1, q2, q3, q4]
for q in list_qubits:
    q.RO_acq_averages(2**10)
    q.pulse_I_offset(0.001)
    q.pulse_Q_offset(-0.006)
    q.RO_I_offset(11e-3)
    q.RO_Q_offset(15e-3)


# preparing the multiplexed readout pulse
integration_length = 550e-9
pulse_length = 450e-9
chi = 2.5e6
M_amps = [0.2, 0.2, 0.0, 0.00, 0.0]
f_ROs = [7.599e9, 7.095e9, 7.787e9, 7.085e9, 7.998e9]

# configuring the readout LO frequency
LO_frequency = f_ROs[1]+380e6

for i in range(5):
    LutMan = eval('LutMan{}'.format(i))
    q = list_qubits[i]
    # Defined in the init
    switch_to_IQ_mod_RO_UHFQC(q)
    q.RO_acq_weight_function_I(i)
    q.RO_acq_weight_function_Q(i)
    q.RO_acq_integration_length(integration_length)
    q.RO_amp(M_amps[i])
    q.f_RO(f_ROs[i])
    q.f_RO_mod(q.f_RO()-LO_frequency)
    LutMan.M_modulation(q.f_RO()-LO_frequency)
    LutMan.M0_modulation(LutMan.M_modulation()+chi)
    LutMan.M1_modulation(LutMan.M_modulation()-chi)
    LutMan.Q_modulation(0)
    LutMan.Q_amp180(0.8/2)
    LutMan.M_up_amp(M_amps[i])
    LutMan.Q_gauss_width(100e-9)
    LutMan.Q_motzoi_parameter(0)
    LutMan.M_amp(M_amps[i])
    LutMan.M_down_amp0(0.0/2)
    LutMan.M_down_amp1(0.0/2)
    LutMan.M_length(pulse_length-100e-9)
    LutMan.M_up_length(100.0e-9)
    LutMan.M_down_length(1e-9)
    LutMan.Q_gauss_nr_sigma(5)
    LutMan.acquisition_delay(270e-9)

multiplexed_wave = [['LutMan0', 'M_up_mid_double_dep'],
                    ['LutMan1', 'M_up_mid_double_dep'],
                    ['LutMan2', 'M_up_mid_double_dep'],
                    ['LutMan3', 'M_up_mid_double_dep'],
                    ['LutMan4', 'M_up_mid_double_dep'],
                    ]

LutManMan.generate_multiplexed_pulse(multiplexed_wave)
LutManMan.render_wave('Multiplexed_pulse', time_units='s')
LutManMan.render_wave_PSD(
    'Multiplexed_pulse', f_bounds=[00e6, 1000e6], y_bounds=[1e-18, 1e-6])
LutManMan.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')

# configuring a joint qubit gate LO
Qubit_LO_frequency = q0.f_qubit()+0.1e9
q0.f_pulse_mod(q0.f_qubit()-Qubit_LO_frequency)
q1.f_pulse_mod(q1.f_qubit()-Qubit_LO_frequency)
q2.f_pulse_mod(q2.f_qubit()-Qubit_LO_frequency)
q3.f_pulse_mod(q3.f_qubit()-Qubit_LO_frequency)
q4.f_pulse_mod(q4.f_qubit()-Qubit_LO_frequency)

MC.soft_avg(1)



###########################################
# integration weight function tune-up
###########################################
cal_shots = 4094
MC.soft_avg(1)
# calibrating the weight functions and measuring individual SSRO
for q in [q0, q1]:

    q.measure_ssro(SSB=True, optimized_weights=True,
                    nr_shots=cal_shots, one_weight_function_UHFQC=True)
    print('LO freq: {:.4f} GHz'.format(LO.frequency()*1e-9))
    print('Drive freq: {:.4f} GHz'.format(Qubit_LO.frequency()*1e-9))


# measuring the multiplexed SSRO result of all four states and uploading
# the cross-talk suippression matrix
tune_up_shots = 2**14
# reload(awg_swf_m)
# reload(sq_m)
sq_m.station = station # multi qubit sequences
q0_pulse_pars, RO_pars = q0.get_pulse_pars()
q1_pulse_pars, RO_pars_q1 = q1.get_pulse_pars()

detector = det.UHFQC_integration_logging_det(
    UHFQC=UHFQC_1, AWG=AWG,
    channels=[0, 1, 2, 3],
    integration_length=integration_length, nr_shots=2048)
sweep = awg_swf_m.two_qubit_off_on(
    q0_pulse_pars=q0_pulse_pars, q1_pulse_pars=q1_pulse_pars, RO_pars=RO_pars)
MC.set_sweep_function(sweep)
MC.set_sweep_points(np.arange(tune_up_shots))
MC.set_detector_function(detector)
label = 'Two_qubit_SSRO_tuneup'
MC.run(label)
mu_matrix,  V_th, mu_matrix_inv, V_th_cor,  V_offset_cor = Niels.two_qubit_ssro_fidelity(
    label, fig_format='png')

UHFQC_1.quex_trans_offset_weightfunction_0(V_offset_cor[0])
UHFQC_1.quex_trans_offset_weightfunction_1(V_offset_cor[1])
UHFQC_1.upload_transformation_matrix(mu_matrix_inv)


###################################################
# Verification of the results
###################################################
tune_up_shots=2**14
sq_m.station=station
q0_pulse_pars, RO_pars=q0.get_pulse_pars()
q1_pulse_pars, RO_pars_q1=q1.get_pulse_pars()

detector = det.UHFQC_integration_logging_det(
                UHFQC=UHFQC_1, AWG=AWG,
                channels=[0,1,2,3],
                integration_length=integration_length, nr_shots=2048, cross_talk_suppression=True)
sweep = awg_swf_m.two_qubit_off_on(
    q0_pulse_pars=q0_pulse_pars,q1_pulse_pars=q1_pulse_pars, RO_pars=RO_pars)
MC.set_sweep_function(sweep)
MC.set_sweep_points(np.arange(tune_up_shots))
MC.set_detector_function(detector)
label ='Two_qubit_SSRO_check'
MC.run(label)
mu_matrix,  V_th, mu_matrix_inv, V_th_cor,  V_offset_cor=Niels.two_qubit_ssro_fidelity(label, fig_format='png')
print("residual cross-talk matrix",  mu_matrix)


###########################################################
# Two qubit AllXY to verify multiplexed driving and RO
#############################################################

# S5 is the device object
mq_mod.measure_two_qubit_AllXY(S5, q0.name, q1.name)