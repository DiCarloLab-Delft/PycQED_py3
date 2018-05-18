
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal


import pycqed.measurement.kernel_functions_ZI as ZI_kern


mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 'medium'


# Settings
fs = 2.4e9
time_start = -100e-9
time_start = np.around(time_start*fs)/fs
time_end = 100e-9
time = np.arange(time_start, time_end, 1/fs)

delay = 10.1e-9
amplitude = 0.1


# Construct impulse_response
impulse = np.zeros(len(time))
zero_ind = np.argmin(np.abs(time))
impulse[zero_ind] = 1.0
delay_ind = np.argmin(np.abs(time-delay))
impulse_response = np.copy(impulse)
impulse_response[delay_ind] = amplitude


# Derive step response
step = np.zeros(len(time))
step[time >= 0.0] = 1.0
step_response = signal.lfilter(impulse_response[zero_ind:], 1.0, step)


# Compute ideal inverted filter kernel
a = ZI_kern.ideal_inverted_fir_kernel(impulse_response, zero_ind)
a1 = ZI_kern.first_order_bounce_kern(delay, -amplitude, fs)

# Apply ideal inverted filter to impulse response and step response
impulse_response_corr = signal.lfilter(a, 1.0, impulse_response)
step_response_corr = signal.lfilter(a, 1.0, step_response)

# Apply first-order inverted filter to impulse response and step response
impulse_response_corr1 = signal.lfilter(a1, 1.0, impulse_response)
step_response_corr1 = signal.lfilter(a1, 1.0, step_response)

# Apply hardware-friendly filter to impulse response and step response
impulse_response_corr_hw = ZI_kern.multipath_first_order_bounce_correction(impulse_response, round(delay*fs), amplitude)
step_response_corr_hw = ZI_kern.multipath_first_order_bounce_correction(step_response, round(delay*fs), amplitude)


# Plot impulse response comparison
plt.figure(1, figsize=(14,10))

plt.subplot(2, 2, 1)
plt.plot(time*1e9, impulse_response)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Impulse response')

plt.subplot(2, 2, 2)
plt.plot(time*1e9, impulse_response_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Ideal corrected impulse response')

plt.subplot(2, 2, 3)
plt.plot(time*1e9, impulse_response_corr1)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) First-order corrected impulse response')

plt.subplot(2, 2, 4)
plt.plot(time*1e9, impulse_response_corr_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) Simulated hardware-corrected impulse response')

plt.tight_layout()
plt.savefig('impulse_response.png',dpi=600,bbox_inches='tight')
plt.show()


# Plot step response comparison
plt.figure(1, figsize=(14,10))

plt.subplot(2, 2, 1)
plt.plot(time*1e9, step_response)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Step response')

plt.subplot(2, 2, 2)
plt.plot(time*1e9, step_response_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Ideal corrected step response')

plt.subplot(2, 2, 3)
plt.plot(time*1e9, step_response_corr1)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) First-order corrected step response')

plt.subplot(2, 2, 4)
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
distorted_waveform = signal.lfilter(impulse_response[zero_ind:], 1.0, ideal_waveform)

# Apply ideal inverted filter to the waveform
distorted_waveform_corr = signal.lfilter(a, 1.0, distorted_waveform)

# Apply first-order filter to the waveform
distorted_waveform_corr1 = signal.lfilter(a1, 1.0, distorted_waveform)

# Apply hardware-friendly filter to the waveform
distorted_waveform_corr_hw = ZI_kern.multipath_first_order_bounce_correction(distorted_waveform, round(delay*fs), amplitude)

# Compute errors with respect to the ideal waveform
err = ideal_waveform - distorted_waveform_corr
err1 = ideal_waveform - distorted_waveform_corr1
err_hw = ideal_waveform - distorted_waveform_corr_hw

# Plot the test waveform comparison
plt.figure(1, figsize=(14,14))

plt.subplot(4, 2, 1)
plt.plot(time*1e9, ideal_waveform)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(a) Ideal waveform')

plt.subplot(4, 2, 2)
plt.plot(time*1e9, distorted_waveform)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(b) Distorted waveform')

plt.subplot(4, 2, 3)
plt.plot(time*1e9, distorted_waveform_corr)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) Ideal corrected waveform')

plt.subplot(4, 2, 4)
plt.plot(time*1e9, err)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(d) Error of ideal correction')

plt.subplot(4, 2, 5)
plt.plot(time*1e9, distorted_waveform_corr1)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(c) First-order correction')

plt.subplot(4, 2, 6)
plt.plot(time*1e9, err1)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(d) Error of first-order correction')

plt.subplot(4, 2, 7)
plt.plot(time*1e9, distorted_waveform_corr_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(e) Simulated hardware-frienldy first-order corrected waveform')

plt.subplot(4, 2, 8)
plt.plot(time*1e9, err_hw)
plt.xlabel('Time, t (ns)')
plt.ylabel('Amplitude (a.u)')
plt.title('(f) Error of  hardware-friendly correction')

plt.tight_layout()
plt.savefig('test_waveform.png', dpi=600, bbox_inches='tight')
plt.show()
