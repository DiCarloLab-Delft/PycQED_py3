# Zurich Instrument utilities extracted from IPython notebooks

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

#    last = False
    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []
#            last = True

        _plot_timing_diagram(d, bits)

