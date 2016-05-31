AWG=AWG
SH = SH
Qubit_LO = Qubit_LO
MC= MC
Dux = Dux

#qubit drive mixer calibrations
reload(cal_tools)
old_att_1 = Dux.in1_out1_attenuation.get()
old_att_2 = Dux.in2_out1_attenuation.get()
Dux.in1_out1_attenuation.set(0.0)
Dux.in2_out1_attenuation.set(0.0)
Dux.in1_out1_switch.set('ON')
Dux.in2_out1_switch.set('OFF')
ch1_min, ch2_min = cal_tools.mixer_carrier_cancellation_5014(
    AWG, SH, source=Qubit_LO, MC=MC, AWG_channel1=1, AWG_channel2=2)
#G_phi, G_alpha = cal_tools.mixer_skewness_calibration_5014(SH, source=S2, station=station, I_ch=1, Q_ch=2)
Dux.in1_out1_switch.set('OFF')
Dux.in2_out1_switch.set('ON')
ch3_min, ch4_min = cal_tools.mixer_carrier_cancellation_5014(
    AWG, SH, source=Qubit_LO, MC=MC, AWG_channel1=3, AWG_channel2=4)
#D_phi, D_alpha = cal_tools.mixer_skewness_calibration_5014(SH, source=S2,station=station, I_ch=3, Q_ch=4)
Dux.in1_out1_switch.set('ON')
Dux.in2_out1_switch.set('ON')
Dux.in1_out1_attenuation.set(old_att_1)
Dux.in2_out1_attenuation.set(old_att_2)
#print(G_alpha, G_phi, D_alpha, D_phi)
print(ch1_min, ch2_min, ch3_min, ch4_min)
#VIP_mon_2_tek.phi_skew.set(G_phi)
#VIP_mon_2_tek.alpha.set(G_alpha)
VIP_mon_2_tek.pulse_I_offset.set(ch1_min)
VIP_mon_2_tek.pulse_Q_offset.set(ch2_min)