
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

import pycqed.measurement.kernel_functions_ZI as ZI_kf

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 'medium'

# Settings
fs = 2.4e9
time_start = -100e-9
time_start = np.around(time_start*fs)/fs
time_end = 100e-9
time = np.arange(time_start, time_end, 1/fs)

bounce_delay = 10.1e-9
bounce_amp = 0.1

# Get filter kernel for bounce
a = ZI_kf.first_order_bounce_kern(bounce_delay, bounce_amp, fs)

# Construct impulse
impulse = np.zeros(len(time))
zero_ind = np.argmin(np.abs(time))
impulse[zero_ind] = 1.0

# Construct step
step = np.zeros(len(time))
step[time >= 0.0] = 1.0

# Apply forward filter to impulse and step response
impulse_response = signal.lfilter([1.0], a, impulse)
step_response = signal.lfilter([1.0], a, step)

# Compute ideal inverted filter kernel
b_ideal = ZI_kf.ideal_inverted_fir_kernel(impulse_response, zero_ind)

# Apply ideal inverted filter to impulse response and step response
impulse_response_corr = signal.lfilter(b_ideal, 1.0, impulse_response)
step_response_corr = signal.lfilter(b_ideal, 1.0, step_response)

# Apply hardware-friendly filter to impulse response and step response
impulse_response_corr_hw = ZI_kf.first_order_bounce_corr(impulse_response, bounce_delay, bounce_amp, fs)
step_response_corr_hw = ZI_kf.first_order_bounce_corr(step_response, bounce_delay, bounce_amp, fs)

# Plot impulse response comparison
plt.figure(1, figsize=(10,14))

plt.subplot(3, 1, 1)
plt.plot(time*1e9, impulse_response)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Impulse response')

plt.subplot(3, 1, 2)
plt.plot(time*1e9, impulse_response_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Ideal corrected impulse response')

plt.subplot(3, 1, 3)
plt.plot(time*1e9, impulse_response_corr_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) Simulated hardware-corrected impulse response')

plt.tight_layout()
plt.savefig('impulse_response.png',dpi=600,bbox_inches='tight')
plt.show()

# Plot step response comparison
plt.figure(1, figsize=(10,14))

plt.subplot(3, 1, 1)
plt.plot(time*1e9, step_response)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Step response')

plt.subplot(3, 1, 2)
plt.plot(time*1e9, step_response_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Ideal corrected step response')

plt.subplot(3, 1, 3)
plt.plot(time*1e9, step_response_corr_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) Simulated hardware-corrected step response')

plt.tight_layout()
plt.savefig('step_response.png',dpi=600,bbox_inches='tight')
plt.show()

# Sawtooth test waveform
sawtooth_period = 50e-9
ideal_waveform = np.remainder(2*time/sawtooth_period, 1)
distorted_waveform = signal.lfilter([1.0], a, ideal_waveform)

# Apply ideal inverted filter to the waveform
distorted_waveform_corr = signal.lfilter(b_ideal, 1.0, distorted_waveform)

# Apply hardware-friendly filter to the waveform
distorted_waveform_corr_hw = ZI_kf.first_order_bounce_corr(distorted_waveform, bounce_delay, bounce_amp, fs)

# Compute errors with respect to the ideal waveform
err = ideal_waveform - distorted_waveform_corr
err_hw = ideal_waveform - distorted_waveform_corr_hw

# Plot the test waveform comparison
plt.figure(1, figsize=(18,10))

plt.subplot(2, 3, 1)
plt.plot(time*1e9, ideal_waveform)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Ideal waveform')

plt.subplot(2, 3, 4)
plt.plot(time*1e9, distorted_waveform)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Distorted waveform')

plt.subplot(2, 3, 2)
plt.plot(time*1e9, distorted_waveform_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) Ideal corrected waveform')

plt.subplot(2, 3, 5)
plt.plot(time*1e9, err)
plt.ylim([-0.1,0.1])
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(d) Error of ideal correction')

plt.subplot(2, 3, 3)
plt.plot(time*1e9, distorted_waveform_corr_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(e) Simulated hardware-frienldy corrected waveform')

plt.subplot(2, 3, 6)
plt.plot(time*1e9, err_hw)
plt.ylim([-0.1,0.1])
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(f) Error of hardware-friendly correction')

plt.tight_layout()
plt.savefig('test_waveform.png', dpi=600, bbox_inches='tight')
plt.show()

# Compare to backward appliction of the filter kernel
hw_corr = ZI_kf.first_order_bounce_corr(distorted_waveform, bounce_delay, bounce_amp, fs)
b = ZI_kf.first_order_bounce_kern(bounce_delay, ZI_kf.coef_round(bounce_amp, force_bshift=0), fs)
first_order_corr = signal.lfilter(b, 1.0, distorted_waveform)

err_bw = hw_corr - first_order_corr

plt.figure(1, figsize=(18,10))
plt.subplot(1, 2, 1)
plt.plot(time*1e9, hw_corr, label='Hardware-friendly correction')
plt.plot(time*1e9, first_order_corr, label='Expected backward appliction')
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.legend()
plt.title('(a) Comparison')

plt.subplot(1, 2, 2)
plt.plot(time*1e9, err_bw)
plt.ylim([-0.1,0.1])
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Error')

plt.tight_layout()
plt.savefig('test_waveform.png', dpi=600, bbox_inches='tight')
plt.show()
