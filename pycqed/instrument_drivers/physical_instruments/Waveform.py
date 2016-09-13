'''
	File:				Waveform.py
	Author:				Wouter Vlothuizen, TNO/QuTech
	Purpose:			generate Waveforms
	Based on:			pulse.py, pulse_library.py
    Prerequisites:
	Usage:
    Bugs:
'''

import numpy as np

class Waveform():
	# complex waveforms
	@staticmethod
	def exp(fs, nrSamples, frequency, initialPhase=0, amplitude=1):
	    return amplitude * np.exp(2*np.pi * frequency/fs * np.array(range(nrSamples)) + initialPhase)


	# real (i.e. non-complex) waveforms
	@staticmethod
	def cos(fs, nrSamples, frequency, initialPhase=0, amplitude=1):
	    return amplitude * np.cos(2*np.pi * frequency/fs * np.array(range(nrSamples)) + initialPhase)

	@staticmethod
	def sin(fs, nrSamples, frequency, initialPhase=0, amplitude=1):
	    return amplitude * np.sin(2*np.pi * frequency/fs * np.array(range(nrSamples)) + initialPhase)

	@staticmethod
	def DC(fs, nrSamples, offset=0):
	    return np.zeros(nrSamples) + offset

	@staticmethod
	def gauss(fs, nrSamples, mu, sigma, amplitude=1):
		t = 1/fs * np.array(range(nrSamples))
		return amplitude*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))

	@staticmethod
	def derivGauss(fs, nrSamples, mu, sigma, amplitude=1, motzoi=1):
		t = 1/fs * np.array(range(nrSamples))
		gauss = amplitude*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
		return motzoi * -1 * (t-mu)/(sigma**1) * gauss
     
	@staticmethod
	def block(fs, nrSamples, offset=0):
		negative = np.zeros(nrSamples/2)
		positive = np.zeros(nrSamples/2) + offset
		return np.concatenate((negative, positive), axis=0)
	