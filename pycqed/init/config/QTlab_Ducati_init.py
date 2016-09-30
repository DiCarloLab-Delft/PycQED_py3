import qt
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pycqed.utilities import general as gen
# plt.ion()
sys.path.append(os.path.join(qt.config['PycQEDdir'],
                'instrument_drivers/meta_instruments/PyLabVIEW'))
sys.path.append(os.path.join(qt.config['PycQEDdir'], 'modules/measurement'))
sys.path.append(os.path.join(qt.config['PycQEDdir'], 'modules/analysis'))
sys.path.append(os.path.join(qt.config['PycQEDdir'], 'scripts'))
import measurement_analysis as MA
import analysis_toolbox as a_tools
import fitting_models as fit_mods
import sweep_functions as swf
import AWG_sweep_functions as awg_swf
import detector_functions as det
from pycqed.measurement import CBox_sweep_functions as CB_swf
import composite_detector_functions as cdet
import calibration_toolbox as cal_tools

qt.config['scriptdirs'] = [
    str(os.path.join(qt.config['PycQEDdir']+'/scripts/testing')),
    str(os.path.join(
        qt.config['PycQEDdir']+'/scripts/personal_folders/Adriaan')),
    str(os.path.join(
        qt.config['PycQEDdir']+'/scripts/personal_folders/Niels'))]

# Fridge
LaDucati = qt.instruments.create('LaDucati', 'Fridge_Monitor',
                                 fridge_name='LaDucati')

Plotmon = qt.instruments.create('Plotmon', 'Plot_Monitor')
MC = qt.instruments.create('MC', 'MeasurementControl')
MC_nest1 = qt.instruments.create('MC_nest1', 'MeasurementControl')

# Sources
RF = qt.instruments.create('RF', 'RS_SGS100A', address='GPIB0::11::INSTR')
LO = qt.instruments.create('LO', 'Agilent_E8257D', address='GPIB0::24::INSTR')
S1 = qt.instruments.create('S1', 'RS_SGS100A', address='TCPIP0::192.168.0.77')
S2 = qt.instruments.create('S2', 'RS_SGS100A', address='TCPIP0::192.168.0.78')
S3 = qt.instruments.create('S3', 'Agilent_E8257D', step_attenuator=True,
                           address='GPIB0::13::INSTR')


AWG = qt.instruments.create('AWG', 'Tektronix_AWG5014',
                            setup_folder='LaDucati', address='GPIB0::9::INSTR')
AWG.set_trigger_source("Internal")
for jj in [1, 2, 3, 4]:
    for kk in [1, 2]:
        exec('AWG.set_ch%s_status("On")' % jj)
        exec('AWG.set_ch%s_marker%s_high(2.)' % (jj, kk))
        exec('AWG.set_ch%s_marker%s_low(-1.)' % (jj, kk))
#presetting marker 2 Ch lower level to be near the R&S threshold
exec('AWG.set_ch1_marker2_low(1.2)')
exec('AWG.set_ch1_marker2_high(2.7)')
# Calibrated Mixer Offsets
AWG.set_ch1_offset(-0.008)
AWG.set_ch2_offset(-0.014)

# Duplexer = qt.instruments.create('Duplexer', 'QuTech_Duplexer',
#                                  address='TCPIP0::192.168.0.100')
# Duplexer.set_in1_out2_attenuation(2**15)  # arbitrary fixed value in arbitrary units
# Duplexer.set_in2_out2_attenuation(2**15)
# Duplexer.set_in1_out2_switch('On')
# Duplexer.set_in2_out2_switch('On')

ATS = qt.instruments.create('ATS', 'Alazar_ATS9870')

AWG.start()
ATS_CW = qt.instruments.create('ATS_CW', 'ATS_CW')
AWG.stop()
AWG.set_trigger_slope("Positive")

ATS_TD = qt. instruments.create('ATS_TD', 'ATS_TD')

TD_Meas = qt.instruments.create('TD_Meas', 'TimeDomainMeasurement')
TD_Meas.set_RF_source('RF')
TD_Meas.set_RF_power(-18)
TD_Meas.set_LO_source('LO')
TD_Meas.set_NoSweeps(1024)
TD_Meas.set_points_per_trace(2048)
TD_Meas.set_int_start(200)
TD_Meas.set_t_int(460)
TD_Meas.set_IF(40e6)
TD_Meas.set_Navg(5)
TD_Meas.set_single_channel_IQ(1)

#TD_Meas.set_CBox_tape_mode(False)

ATS_TD.set_sample_rate(1000)  # MSPS
ATS_TD.set_range(.4)

AWG.start()

HM = qt.instruments.create('HM', 'Homodyne_source')
HM.set_RF_source('RF')
HM.set_LO_source('LO')
HM.set_IF(20e6)
HM.set_t_int(20)
HM.set_RF_power(-37)
HM.set_LO_power(16)
HM.init(optimize=True, silent=True)

AWG.stop()

Pulsed_Spec = qt.instruments.create('Pulsed_Spec', 'PulsedSpectroscopy')
Pulsed_Spec.set_NoReps(3000)
Pulsed_Spec.set_points_per_trace(6000)
Pulsed_Spec.set_int_start(1200)
Pulsed_Spec.set_t_int(4000)
Pulsed_Spec.set_LO_source('LO')
Pulsed_Spec.set_RF_source('RF')
Pulsed_Spec.set_IF(10e6)
Pulsed_Spec.set_Navg(1)
Pulsed_Spec.set_RF_power(-32)
Pulsed_Spec.set_cal_mode('IQ')
Pulsed_Spec.set_single_channel_IQ(1)
Pulsed_Spec.set_AWG_seq_filename('Spec_5014')


IVVI = qt.instruments.create('IVVI', 'IVVI',
                             address='ASRL1', reset=False, numdacs=16)
IVVI.set_parameter_rate('dac1', 1, 20)
IVVI.set_parameter_rate('dac2', 1, 20)
IVVI.set_parameter_rate('dac3', 1, 20)

CBox = qt.instruments.create('CBox', 'QuTech_ControlBox', address='Com3')
CBox.set_signal_delay(40)
CBox.set_integration_length(140) # 280=1400 ns
CBox.set_acquisition_mode(0)
CBox.set_lin_trans_coeffs(1, 0, 0, 1)
CBox.set_log_length(200)
CBox.set_dac_offset(0, 1, 40) #I channel qubit drive AWG
CBox.set_dac_offset(0, 0, 48) #Q channel
CBox.set_dac_offset(1, 1, 18) #I channel
CBox.set_dac_offset(1, 0, -36) #Q channel readout AWG

t_base = np.arange(512)*5e-9
IF = 4e7
cosI = np.cos(2*np.pi * t_base*IF)
sinI = np.sin(2*np.pi * t_base*IF)
w0 = np.round(cosI*120)
w1 = np.round(sinI*120)

CBox.set_integration_weights(line=0, weights=w0)
CBox.set_integration_weights(line=1, weights=w1)
CBox.set_averaging_parameters(70, 11)
print("CBox initialized")

CBox_lut_man = qt.instruments.create(
    'CBox_lut_man', 'QuTech_ControlBox_LookuptableManager')

CBox_lut_man_2 = qt.instruments.create(
    'CBox_lut_man_2', 'QuTech_ControlBox_LookuptableManager')

# Reload the settings
gen.load_settings_onto_instrument(CBox_lut_man)
gen.load_settings_onto_instrument(CBox_lut_man_2)

# switching on pulsemode
RF.set_pulsemod_state('ON')
RF.set_parameter_bounds('power', -120, 20)

print("pulsed RF on")
HS = qt.instruments.create('HS', 'HeterodyneSource',
                           RF='RF', LO='LO', IF=.01)

ATT = qt.instruments.create('ATT', 'Aeroflex_8320', address='TCPIP0::192.168.0.98')

# print 'Initializing Signal Hound'
SH = qt.instruments.create('SH', 'SignalHound_USB_SA124B')

JPA = qt.instruments.create('JPA', 'JPA_object')
gen.load_settings_onto_instrument(JPA)


for qubit_num in range(1, 11):
    qt.instruments.create('VIP_mon_%d' % qubit_num, 'qubit_object')
    exec('VIP_mon_%d = qt.instruments["VIP_mon_%d"]' % (qubit_num, qubit_num))

qubit_lst = [VIP_mon_1, VIP_mon_2, VIP_mon_3, VIP_mon_4,
             VIP_mon_5, VIP_mon_6, VIP_mon_7, VIP_mon_8, VIP_mon_9, VIP_mon_10]
for vipmon in qubit_lst:
    gen.load_settings_onto_instrument(vipmon) #,timestamp='20150624_103352'



# VIPmon 6 experiment specific

S1.set_frequency(VIP_mon_2.get_current_frequency()*10**9-VIP_mon_2.get_sideband_modulation_frequency()*10**9)
S2.set_frequency(VIP_mon_2.get_current_frequency()*10**9-VIP_mon_2.get_sideband_modulation_frequency()*10**9)
S3.set_frequency(VIP_mon_2.get_current_RO_frequency()*10**9+10000000)
S3.set_power(-22.2)

TD_Meas.set_f_readout(VIP_mon_2.get_current_RO_frequency()*10**9)
TD_Meas.set_NoSegments(1)
TD_Meas.set_Navg(1)

AWG.set_trigger_source("external")




