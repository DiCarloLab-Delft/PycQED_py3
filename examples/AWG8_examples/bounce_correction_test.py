
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
time_start = -50e-9
time_start = np.around(time_start*fs)/fs
time_end = 50e-9
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


# Apply ideal inverted filter to impulse response and step response
impulse_response_corr = signal.lfilter(a, 1.0, impulse_response)
step_response_corr = signal.lfilter(a, 1.0, step_response)

# Apply hardware-friendly filter to impulse response and step response
impulse_response_corr_hw = ZI_kern.multipath_bounce_correction(impulse_response, round(delay*fs), -amplitude)
step_response_corr_hw = ZI_kern.multipath_bounce_correction(step_response, round(delay*fs), -amplitude)


# Plot impulse response comparison
plt.figure(1, figsize=(7,10))

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
plt.title('(c) Harware-corrected impulse response')

plt.tight_layout()
plt.savefig('impulse_response.png',dpi=600,bbox_inches='tight')
plt.show()


# Plot step response comparison
plt.figure(1, figsize=(7,10))

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
plt.title('(c) Harware-corrected step response')

plt.tight_layout()
plt.savefig('step_response.png',dpi=600,bbox_inches='tight')
plt.show()

