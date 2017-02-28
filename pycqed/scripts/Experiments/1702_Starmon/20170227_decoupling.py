def switch_to_pulsed_RO_UHFQC(qubit):
    UHFQC_1.awg_sequence_acquisition()
    qubit.RO_pulse_type('Gated_MW_RO_pulse')
    qubit.RO_acq_marker_delay(75e-9)
    qubit.acquisition_instr('UHFQC_1')
    qubit.RO_acq_marker_channel('ch3_marker2')
    qubit.RO_acq_weight_function_I(0)
    qubit.RO_acq_weight_function_Q(1)

switch_to_pulsed_RO_UHFQC(QL)

QL.RO_power_cw(-40)
QL.f_RO(7.08e9)
QL.spec_pow(-40)
QL.f_qubit(4.8e9)
QL.RO_pulse_power(-50)
QL.f_RO_mod(10e6)
QL.RO_acq_averages(256)
QL.RO_acq_marker_channel('ch3_marker2')
QL.RO_pulse_type('Gated_MW_RO_pulse')
QL.RO_pulse_marker_channel('ch4_marker1')
QL.RO_acq_marker_delay(0)
QL.RO_pulse_delay(0)
HS.AWG = station.AWG
QL.RO_pulse_length(2e6)

QL.RO_acq_integration_length(2e-6)
QL.spec_pulse_marker_channel('ch4_marker2')
QL.RO_pulse_length(2e-6)
QL.td_source_pow(-20)
QL.spec_pow_pulsed(-20)
QL.spec_pulse_depletion_time(5e-6)
QL.spec_pulse_length(2e-6)

QL.RO_acq_marker_delay(0)  # 1.1e-6)
QL.RO_pulse_length(2e-6)  # 1.25e-6)

QR.td_source_pow(-20)

QR.RO_power_cw(-50)
QR.f_RO(7.18e9)
QR.spec_pow(-40)
QR.f_qubit(4.8e9)
QR.RO_pulse_power(-35)
QR.f_RO_mod(10e6)
QR.RO_acq_averages(256)
QR.RO_acq_marker_channel('ch3_marker2')
QR.RO_pulse_type('Gated_MW_RO_pulse')
QR.RO_pulse_marker_channel('ch4_marker1')
QR.RO_acq_marker_delay(0)
QR.RO_pulse_delay(0)
HS.AWG = station.AWG
QR.RO_pulse_length(2e6)

QR.RO_acq_integration_length(2e-6)
QR.spec_pulse_marker_channel('ch3_marker1')
QR.RO_pulse_length(2e-6)
QR.td_source_pow(-20)
QR.spec_pow_pulsed(-20)
QR.spec_pulse_depletion_time(5e-6)
QR.spec_pulse_length(2e-6)

switch_to_pulsed_RO_UHFQC(QR)

########################################

QL.spec_pow(-54)
QL.RO_power_cw(-62)
QL.RO_acq_averages(4096)

QR.spec_pow(-48)
QR.RO_power_cw(-60)


########
# REF for reloading settings
########
# QR
# scan_start = '20170226_213551'
# scan_stop = '20170227_030743'
# QL
# scan_start = '20170227_105823'
# scan_stop = '20170227_115924'

dac_range = np.concatenate((np.arange(0, 501, 10), np.arange(0, -501, -10)))
IVVI.dac1(0.)
for d in dac_range:
    IVVI.dac1(d)
    try:
        QR.RO_acq_averages(1024)
        QR.find_resonator_frequency(
            freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QR.RO_acq_averages(1024)
        QR.find_frequency(freqs=np.arange(6.11e9, 5.95e9, -.5e6))
    except:
        print('Failed resonator fit')
IVVI.dac1(0.)


dac_range = np.concatenate((np.arange(0, 801, 25), np.arange(0, -801, -25)))
for d in dac_range:
    IVVI.dac2(d)
    try:
        QL.find_resonator_frequency(
            freqs=np.arange(7.085e9, 7.09e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QL.find_frequency(freqs=np.arange(4.69e9, 4.58e9, -.2e6))
    except:
        print('Failed resonator fit')

############################################


def measure_sweetspot(dac_channel, dac_range, RO_freq, window_len=3):
    HS.frequency(RO_freq)
    HS.RF_power(-50)
    MC.set_sweep_function(dac_channel)
    MC.set_sweep_points(dac_range)
    MC.set_detector_function(det.Heterodyne_probe(HS, trigger_separation=4e-6))
    label_scan = 'Biasing_scan_%s' % dac_channel.name
    MC.run(label_scan)
    ma.MeasurementAnalysis(label=label_scan, auto=True)
    return analyze_last_biasing_scan(window_len)

import os


def analyze_last_biasing_scan(window_len):
    label = 'Biasing_scan'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[1:-2]
    y = a.measured_values[0, 1:-2]
    mid_point = len(y)//2
    peaks_x_low = a_tools.peak_finder_v2(
        x[:mid_point], -y[:mid_point], window_len=window_len)
    peaks_x_high = a_tools.peak_finder_v2(
        x[mid_point:], -y[mid_point:], window_len=window_len)

    savefolder = a.folder
    figname = 'Sweet_spot_fit.PNG'
    savename = os.path.abspath(os.path.join(savefolder, figname))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-o')
    ax.axvline(peaks_x_low[0], linestyle='--', color='k')
    ax.axvline(peaks_x_high[0], linestyle='--', color='k')
    ax.axvline((peaks_x_low[0]+peaks_x_high[0])/2, linestyle='--', color='r')
    ax.set_title('Sweetspot Fit')
#     ax.set_xlabel(str(a.sweep_name + '('+a.sweep_unit+')'))
#     ax.set_ylabel(str(a.value_names[0] + '('+a.value_units[0]+')'))
    fig.savefig(savename, format='PNG', dpi=450)
    plt.close(fig)

    return (peaks_x_low[0]+peaks_x_high[0])/2

RO_freq_QR = 7.190e9
RO_freq_QL = 7.0867e9
dac_range_QL = np.arange(-1000, 1001, 10)
dac_range_QR = dac_range_QL

ss_iter_QL = []
ss_iter_QR = []
max_iter = 30
for i in range(max_iter):
    new_ss_QL = measure_sweetspot(IVVI.dac2, dac_range_QL, RO_freq_QL, 7)
    ss_iter_QL.append(new_ss_QL)
    IVVI.dac2(new_ss_QL)
    new_ss_QR = measure_sweetspot(IVVI.dac1, dac_range_QR, RO_freq_QR)
    ss_iter_QR.append(new_ss_QR)
    IVVI.dac1(new_ss_QR)

from matplotlib import pyplot as plt
%matplotlib inline
plt.plot(ss_iter_QR)
plt.plot(ss_iter_QL)
ax = plt.gca()
ax.set_xlabel('Iteration')
ax.set_ylabel('Sweetspot (mV)')
#####################################################
IVVI.dac1(-280)
IVVI.dac2(-400)

QL.spec_pow(-54)
QL.RO_power_cw(-60)
QL.RO_acq_averages(4096)

try:
    QL.find_resonator_frequency(
        freqs=np.arange(7.085e9, 7.09e9, 0.1e6), use_min=True)
except:
    print('Failed resonator fit')
try:
    QL.find_frequency(freqs=np.arange(4.66e9, 4.6e9, -.2e6))
except:
    print('Failed resonator fit')

QR.RO_acq_averages(4096)
try:
    QR.find_resonator_frequency(
        freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
except:
    print('Failed resonator fit')
try:
    QR.find_frequency(freqs=np.arange(6.e9, 5.8e9, -.5e6))
except:
    print('Failed resonator fit')

########################################################
dac1_center = -280
dac2_center = -400
f_center1 = 5.822e9
f_center2 = 4.628e9

dac_range_1 = np.arange(-20, 20, 5) + dac1_center
# dac1 scan
IVVI.dac1(dac1_center)
IVVI.dac2(dac2_center)
f_q_1 = np.arange(-50e6, 50.5e6, .5e6)+f_center1
f_q_2 = np.arange(-50e6, 50.5e6, .5e6)+f_center2
for d in dac_range_1:
    IVVI.dac1(d)
    # Qubit Right scan
    QR.msmt_suffix = '_%d_%d_QR' % (IVVI.dac1(), IVVI.dac2())
    try:
        QR.find_resonator_frequency(
            freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QR.find_frequency(freqs=f_q_1)
    except:
        print('Failed resonator fit')

    # Qubit Left scan
    QL.msmt_suffix = '_%d_%d_QL' % (IVVI.dac1(), IVVI.dac2())
    try:
        QL.find_resonator_frequency(
            freqs=np.arange(7.085e9, 7.09e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QL.find_frequency(freqs=f_q_2)
    except:
        print('Failed resonator fit')

IVVI.dac1(dac1_center)
IVVI.dac2(dac2_center)

dac_range_2 = np.arange(-20, 20, 5) + dac2_center
# dac1 scan
IVVI.dac1(dac1_center)
IVVI.dac2(dac2_center)
f_q_1 = np.arange(-50e6, 50.5e6, .5e6)+f_center1
f_q_2 = np.arange(-50e6, 50.5e6, .5e6)+f_center2
for d in dac_range_2:
    IVVI.dac2(d)
    # Qubit Right scan
    QR.msmt_suffix = '_%d_%d_QR' % (IVVI.dac1(), IVVI.dac2())
    try:
        QR.find_resonator_frequency(
            freqs=np.arange(7.188e9, 7.195e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QR.find_frequency(freqs=f_q_1)
    except:
        print('Failed resonator fit')

    # Qubit Left scan
    QL.msmt_suffix = '_%d_%d_QL' % (IVVI.dac1(), IVVI.dac2())
    try:
        QL.find_resonator_frequency(
            freqs=np.arange(7.085e9, 7.09e9, 0.1e6), use_min=True)
    except:
        print('Failed resonator fit')
    try:
        QL.find_frequency(freqs=f_q_2)
    except:
        print('Failed resonator fit')

IVVI.dac1(dac1_center)
IVVI.dac2(dac2_center)

################################

RO_freq = [7.0867e9, 7.190e9]
qubit_name = ['QL', 'QR']
for q_idx in [0, 1]:
    HS.frequency(RO_freq[q_idx])
    HS.RF_power(-50)
    dac_range = np.arange(-1000, 1001, 50)
    MC.set_sweep_function(IVVI.dac1)
    MC.set_sweep_points(dac_range)
    MC.set_sweep_function_2D(IVVI.dac2)
    MC.set_sweep_points_2D(dac_range)
    MC.set_detector_function(det.Heterodyne_probe(HS, trigger_separation=4e-6))
    label_scan = 'Biasing_scan_%s' % qubit_name[q_idx]
    MC.run(label_scan, mode='2D')

IVVI.dac1(0.)
IVVI.dac2(0.)
