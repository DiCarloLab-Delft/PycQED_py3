# module for visualizing sequences.
#
# author: Wolfgang Pfaff
# modified by: Adriaan Rol

import numpy as np
from matplotlib import pyplot as plt


def show_wf(tvals, wf, name='', ax=None, ret=None, dt=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if dt is None:
        dt = tvals[1]-tvals[0]
    ax.plot(tvals, wf, ls='-', marker='.')

    ax.set_xlim(tvals[0], 2*tvals[-1]-tvals[-2])
    ax.set_ylabel(name + ' Amplitude')

    if ret == 'ax':
        return ax
    else:
        return None


def show_element(element, delay=True):
    tvals, wfs = element.waveforms()
    cnt = len(wfs)
    i = 0

    fig, axs = plt.subplots(cnt, 1, sharex=True)
    t0 = 0
    t1 = 0

    for wf in wfs:
        i += 1
        hi = element._channels[wf]['high']
        lo = element._channels[wf]['low']
        # some prettifying
        ax = axs[i-1]
        ax.set_axis_bgcolor('gray')
        ax.axhspan(lo, hi, facecolor='w', linewidth=0)
        # the waveform
        if delay:
            t = tvals
        else:
            t = element.real_times(tvals, wf)

        t0 = min(t0, t[0])
        t1 = max(t1, t[-1])
        # TODO style options
        show_wf(t, wfs[wf], name=wf, ax=ax, dt=1./element.clock)

        ax.set_ylim(lo*1.1, hi*1.1)

        if i == cnt:
            ax.set_xlabel('Time')
            ax.set_xlim(t0, t1)


def show_fourier_of_element_channels(element, channels, units='Hz'):
    '''
    Shows a fourier transform of a waveform.
    element : from which a waveform needs to be displayed
    channels (str): names of the channels on which the waveform is defined
                    in time domain. If the lenght of the channels is 2 it
                    interprets the first as being the I quadrature and the
                    second as the q quadrature.
    '''
    tvals, wfs = element.waveforms()
    fig, ax = plt.subplots(1, 1)
    dt = tvals[1]-tvals[0]

    if len(channels) == 2:
        compl_data = wfs[channels[0]] + 1j * wfs[channels[1]]
        trans_dat = np.fft.fft(compl_data)*dt
        n = len(compl_data)
    elif len(channels) == 1:
        trans_dat = np.fft.fft(wfs[channels[0]])*dt
        n = len(wfs[channels[0]])
    else:
        trans_dat = np.fft.fft(wfs[channels])*dt
        n = len(wfs[channels])

    freqs = np.fft.fftfreq(n, d=dt)

    if units == 'MHz':
        freqs *= 1e-6
    elif units == 'GHz':
        freqs *= 1e-9
    elif units == 'Hz':
        pass
    else:
        raise Exception('units "%s" not recognized, valid options' +
                        ' are GHz, MHz and Hz')
    ax.plot(freqs, trans_dat, ls='-o', marker='.')
    ax.set_xlabel('Frequency (%s)' % units)
