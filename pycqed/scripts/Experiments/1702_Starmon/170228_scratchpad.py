for q in [q, QR]:
    q.RF_RO_source(RF.name)

    q.RO_pulse_type('MW_IQmod_pulse_UHFQC')
    q.spec_pow(-60)
    # q.LO_

    q.RO_pulse_length(2e-6)
    q.RO_acq_marker_delay(100e-9)
    q.RO_pulse_delay(100e-9)
    q.AWG = None
    # q.heterodyne_instr = HS
    q.f_RO_mod(20e6)

    q.RO_acq_weight_function_I(0)
    q.RO_acq_weight_function_Q(1)

    switch_to_pulsed_RO_UHFQC(q)
    RF.power(-36)
    UHFQC_1.prepare_DSB_weight_and_rotation(IF=q.f_RO_mod())
    q.RO_power_cw(-50)


# Testing the resonator scan

QR.find_resonator_frequency()
QL.find_resonator_frequency()



q = QR
q.RO_pulse_length()
q.RO_amp(0.2)
q.AWG = None
LutMan = q.RO_LutMan
switch_to_IQ_mod_RO_UHFQC(q)
q.RO_acq_weight_function_I(0)
q.RO_acq_weight_function_Q(1)
q.RO_acq_integration_length(integration_length)

LutMan.M_modulation(q.f_RO_mod())
LutMan.M0_modulation(LutMan.M_modulation()+chi)
LutMan.M1_modulation(LutMan.M_modulation()-chi)
LutMan.Q_modulation(0)
LutMan.Q_amp180(0.8/2)
LutMan.M_up_amp(0.2)
LutMan.Q_gauss_width(100e-9)
LutMan.Q_motzoi_parameter(0)
LutMan.M_amp(0.2)
LutMan.M_down_amp0(0.0/2)
LutMan.M_down_amp1(0.0/2)
LutMan.M_length(pulse_length-100e-9)
LutMan.M_up_length(100.0e-9)
LutMan.M_down_length(1e-9)
LutMan.Q_gauss_nr_sigma(5)
LutMan.acquisition_delay(270e-9)

multiplexed_wave = [['LutMan0', 'M_up_mid_double_dep'],
                    ]

LutManMan.generate_multiplexed_pulse(multiplexed_wave)
LutManMan.render_wave('Multiplexed_pulse', time_unit='s')
LutManMan.render_wave_PSD(
    'Multiplexed_pulse', f_bounds=[00e6, 1000e6], y_bounds=[1e-18, 1e-6])
LutManMan.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')

# Enabling the AWG channel on the UHFQC
UHFQC_1.awgs_0_enable(1)