#!/usr/bin/python

import qcodes

from QWG import QWG
from QWG import IPTransport
from Waveform import Waveform


# create waveforms
sampleCnt = 96
fs = 1e9
f = fs/32
# mu=30, sigma=10, dirAmpl=1.0 fits 64 samples nicely
mu = 30e-9
sigma = 10e-9
dirAmpl = 1.0
# mu2=15, sigma2=5, dirAmpl2=1.0 fits 64 samples nicely
mu2 = 15e-9
sigma2 = 5e-9
dirAmpl2 = 1.0

wvCos = Waveform.cos(fs, sampleCnt, f)
wvSin = Waveform.sin(fs, sampleCnt, f)
wvZero = Waveform.DC(fs, sampleCnt)
wvHi = Waveform.DC(fs, sampleCnt, 1.0)
wvLo = Waveform.DC(fs, sampleCnt, -1.0)
wvGauss = Waveform.gauss(fs, sampleCnt, mu, sigma)
wvDerivGauss = Waveform.derivGauss(fs, sampleCnt, mu, sigma, dirAmpl)
wvGauss2 = Waveform.gauss(fs, sampleCnt, mu2, sigma2)
wvDerivGauss2 = Waveform.derivGauss(fs, sampleCnt, mu2, sigma2, dirAmpl2)
marker1 = []
marker2 = []



if 0:
	qwg1 = QWG('QWG-1', IPTransport('192.168.42.10', 5025))
else:
	# local variant, in combination with 'nc -l 5025' run locally from a terminal
	qwg1 = QWG('QWG-1', IPTransport('127.0.0.1', 5025))

qwg1.reset()


if 1:	# continuous
	# 
	qwg1.createWaveformReal('cos', wvCos, marker1, marker2)
	qwg1.createWaveformReal('sin', wvSin, marker1, marker2)
	qwg1.createWaveformReal('zero', wvZero, marker1, marker2)
	qwg1.createWaveformReal('hi', wvHi, marker1, marker2)
	qwg1.createWaveformReal('lo', wvLo, marker1, marker2)
	qwg1.createWaveformReal('gauss', wvGauss, marker1, marker2)
	qwg1.createWaveformReal('derivGauss', wvDerivGauss, marker1, marker2)

	if 1:	# dc
		qwg1.setWaveform(1, 'hi')
		qwg1.setWaveform(2, 'zero')
		qwg1.setWaveform(3, 'hi')
		qwg1.setWaveform(4, 'zero')
	else:
		qwg1.setWaveform(1, 'derivGauss')
		qwg1.setWaveform(2, 'zero')
		qwg1.setWaveform(3, 'zero')
		qwg1.setWaveform(4, 'gauss')
	qwg1.setRunModeContinuous()

else:	# codeword based
	qwg1.createWaveformReal('zero', wvZero, marker1, marker2)
	qwg1.createWaveformReal('hi', wvHi, marker1, marker2)
	qwg1.createWaveformReal('lo', wvLo, marker1, marker2)
	qwg1.createWaveformReal('gauss', wvGauss, marker1, marker2)
	qwg1.createWaveformReal('derivGauss', wvDerivGauss, marker1, marker2)
	qwg1.createWaveformReal('gauss2', wvGauss2, marker1, marker2)
	qwg1.createWaveformReal('derivGauss2', wvDerivGauss2, marker1, marker2)

	qwg1.createWaveformReal('gaussNeg', -wvGauss, marker1, marker2)


	# segment 0: idle
	qwg1.setWaveform(1, 'zero')
	qwg1.setWaveform(2, 'zero')
	qwg1.setWaveform(3, 'zero')
	qwg1.setWaveform(4, 'zero')

	# segment 1: (NB: multi-segment handling is a temporary hack)
	seg=0

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'hi')
	qwg1.setSeqElemWaveform(seg, 2, 'hi')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss')
	qwg1.setSeqElemWaveform(seg, 4, 'derivGauss')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'gauss')
	qwg1.setSeqElemWaveform(seg, 2, 'gaussNeg')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
	qwg1.setSeqElemWaveform(seg, 4, 'derivGauss2')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'gauss')
	qwg1.setSeqElemWaveform(seg, 2, 'derivGauss')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
	qwg1.setSeqElemWaveform(seg, 4, 'derivGauss2')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'gauss2')
	qwg1.setSeqElemWaveform(seg, 2, 'derivGauss2')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss')
	qwg1.setSeqElemWaveform(seg, 4, 'derivGauss')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'zero')
	qwg1.setSeqElemWaveform(seg, 2, 'zero')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss')
	qwg1.setSeqElemWaveform(seg, 4, 'gaussNeg')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'zero')
	qwg1.setSeqElemWaveform(seg, 2, 'zero')
	qwg1.setSeqElemWaveform(seg, 3, 'derivGauss')
	qwg1.setSeqElemWaveform(seg, 4, 'zero')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'zero')
	qwg1.setSeqElemWaveform(seg, 2, 'gauss2')
	qwg1.setSeqElemWaveform(seg, 3, 'zero')
	qwg1.setSeqElemWaveform(seg, 4, 'zero')

	seg = seg+1
	qwg1.setSeqElemWaveform(seg, 1, 'zero')
	qwg1.setSeqElemWaveform(seg, 2, 'zero')
	qwg1.setSeqElemWaveform(seg, 3, 'gauss2')
	qwg1.setSeqElemWaveform(seg, 4, 'zero')

	qwg1.setRunModeCodeword()


qwg1.ch_pair1_sideband_frequency.set(100e6)
qwg1.ch_pair3_sideband_frequency.set(100e6)
qwg1.syncSidebandGenerators()

qwg1.ch1_state.set(1)
qwg1.ch2_state.set(1)
qwg1.ch3_state.set(1)
qwg1.ch4_state.set(1)

qwg1.run()


print(qwg1.getIdentity())

print(qwg1.getError())

