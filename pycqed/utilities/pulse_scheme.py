import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


def new_pulse_fig(figsize):
    '''
    Open a new figure and configure it to plot pulse schemes.
    '''
    fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=False)
    ax.axis('off')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    ax.axhline(0, color='0.75')

    return fig, ax


def mwPulse(ax, pos, width=1.5, label=None, phase=0, labelHeight=1.3,
            color='C0'):
    '''
    Draw a microwave pulse: Gaussian envelope with modulation.
    '''
    x = np.linspace(pos, pos + width, 100)
    envPos = np.exp(-(x - (pos + width / 2))**2 / (width / 4)**2)
    envNeg = -np.exp(-(x - (pos + width / 2))**2 / (width / 4)**2)
    mod = envPos * np.sin(2 * np.pi * 3 / width * x + phase)

    ax.plot(x, envPos, '--', color=color)
    ax.plot(x, envNeg, '--', color=color)
    ax.plot(x, mod, '-', color=color)

    if label is not None:
        ax.text(pos + width / 2, labelHeight, label,
                horizontalalignment='center', color=color)

    return pos + width


def fluxPulse(ax, pos, width=2.5, s=.1, amp=1.5, color='C1'):
    '''
    Draw a smooth flux pulse, where the rising and falling edges are given by
    Fermi-Dirac functions.
    s: smoothness of edge
    '''
    x = np.linspace(pos, pos + width, 100)
    y = amp / ((np.exp(-(x - (pos + 5.5 * s)) / s) + 1) *
               (np.exp((x - (pos + width - 5.5 * s)) / s) + 1))

    ax.fill_between(x, y, color=color, alpha=0.3)
    ax.plot(x, y, color=color)

    return pos + width


def ramZPulse(ax, pos, width=2.5, s=0.1, amp=1.5, sep=1.5, color='C1'):
    '''
    Draw a Ram-Z flux pulse, i.e. only part of the pulse is shaded, to indicate
    cutting off the pulse at some time.
    '''
    xLeft = np.linspace(pos, pos + sep, 100)
    xRight = np.linspace(pos + sep, pos + width, 100)
    xFull = np.concatenate((xLeft, xRight))
    y = amp / ((np.exp(-(xFull - (pos + 5.5 * s)) / s) + 1) *
               (np.exp((xFull - (pos + width - 5.5 * s)) / s) + 1))
    yLeft = y[:len(xLeft)]

    ax.fill_between(xLeft, yLeft, alpha=0.3, color=color, linewidth=0.0)
    ax.plot(xFull, y, color=color)

    return pos + width


def interval(ax, start, stop, height=1.5, label=None, labelHeight=None,
             vlines=True, color='k', arrowstyle='<|-|>'):
    '''
    Draw an arrow to indicate an interval.
    '''
    if labelHeight is None:
        labelHeight = height + 0.2

    arrow = matplotlib.patches.FancyArrowPatch(
        posA=(start, height), posB=(stop, height), arrowstyle=arrowstyle,
        color=color, mutation_scale=7)
    ax.add_patch(arrow)

    if vlines:
        ax.plot([start, start], [0, labelHeight], '--', color=color)
        ax.plot([stop, stop], [0, labelHeight], '--', color=color)

    if label is not None:
        ax.text((start + stop) / 2, labelHeight, label, color=color,
                horizontalalignment='center')
