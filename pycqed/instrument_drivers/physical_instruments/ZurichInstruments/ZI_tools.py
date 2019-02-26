# Zurich Instrument goodies extracted from IPython notebooks
# author: Niels H

import numpy as np
import matplotlib.pyplot as plt

# from: AWG8_staircase_test
def plot_timing_diagram(data, bits, line_length=30):
    def _plot_lines(ax, pos, *args, **kwargs):
        if ax == 'x':
            for p in pos:
                plt.axvline(p, *args, **kwargs)
        else:
            for p in pos:
                plt.axhline(p, *args, **kwargs)

    def _plot_timing_diagram(data, bits):
        plt.figure(figsize=(20, 0.5 * len(bits)))

        t = np.arange(len(data))
        _plot_lines('y', 2 * np.arange(len(bits)), color='.5', linewidth=2)
        _plot_lines('x', t, color='.5', linewidth=0.5)

        for n, i in enumerate(reversed(bits)):
            line = [((x >> i) & 1) for x in data]
            plt.step(t, np.array(line) + 2 * n, 'r', linewidth=2, where='post')
            plt.text(-0.5, 2 * n, str(i))

        plt.xlim([t[0], t[-1]])
        plt.ylim([0, 2 * len(bits) + 1])

        plt.gca().axis('off')
        plt.show()

    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []

        _plot_timing_diagram(d, bits)


# From: iPython-Notebooks/Experiments/1709_M18/170918_UHFQC - LutMan testing
def print_timing_diagram(data, bits, line_length=30):
    def _print_timing_diagram(data, bits):
        line_length = 0

        for i in bits:
            print('       ', end='')
            last = (data[0] >> i) & 1
            for d in data:
                curr = (d >> i) & 1
                if last == 0 and curr == 0:
                    print('   ', end='')
                elif last == 0 and curr == 1:
                    print(' __', end='')
                elif last == 1 and curr == 0:
                    print('   ', end='')
                elif last == 1 and curr == 1:
                    print('___', end='')
                last = curr
            print('')
            print('Bit {:2d}:'.format(i), end='')

            last = (data[0] >> i) & 1
            for d in data:
                curr = (d >> i) & 1
                if last == 0 and curr == 0:
                    print('___', end='')
                elif last == 0 and curr == 1:
                    print('/  ', end='')
                elif last == 1 and curr == 0:
                    print('\\__', end='')
                elif last == 1 and curr == 1:
                    print('   ', end='')
                last = curr
            print('')

    done = False
    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []
            done = True

        _print_timing_diagram(d, bits)
        if not done:
            print('')
            print('       ', end='')
            print('-' * 3 * line_length)


# simple version of above to better view timing
def print_timing_diagram_simple(data, bits, line_length=30):
    def _print_timing_diagram(data, bits):
        line_length = 0

        for i in bits:
            print('Bit {:2d}:'.format(i), end='')

            last = (data[0] >> i) & 1
            for d in data:
                curr = (d >> i) & 1
                if curr == 0:
                    print('0', end='')
                elif curr == 1:
                    print('1', end='')
                last = curr
            print('')

    done = False
    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []
            done = True

        _print_timing_diagram(d, bits)
        if not done:
            print('')
            print('       ', end='')
            print('-' * line_length)