import numpy as np
import os
import seaborn.apionly as sns
sns.set_palette('muted')
cls = (sns.color_palette())
import matplotlib.pyplot as plt


save_folder = '/Users/Adriaan/GitHubRepos/DiCarloLab_Repositories/Paper_16_RestlessTuning/Figures'
save_format = 'pdf'


def make_figure(fig_name='timing_tech',
                Acquisition_time=np.array([1.6,  0.122, 0.122]),
                AWG_overhead=np.array([0.093,  0.093, 0]),
                Processing_overhead=np.array([.164+.063,  .164+.063, 0]),
                Overhead=np.array([0.04, 0.04, 0.04]),
                methods=('Conventional-', 'Restless-', 'Restless+'),
                ):
    cls = (sns.color_palette('muted'))

    plt.rcParams.update({'font.size': 9, 'legend.labelspacing': 0,
                         'legend.columnspacing': .3,
                         'legend.handletextpad': .2})
    f, ax = plt.subplots(figsize=(3.3, .9))
    y_pos = np.array([.8, 0, -.8])+.1

    lefts = np.array([0, 0, 0])
    ax.barh(y_pos, AWG_overhead, color=cls[2], align='center',
            height=0.6, label='Set pars.')
    lefts = lefts+np.array(AWG_overhead)

    ax.barh(y_pos, Acquisition_time, color=cls[0], align='center',
            height=0.6, label='Experiment', left=lefts)
    lefts = lefts + Acquisition_time
    ax.barh(y_pos, Processing_overhead, color=cls[3], align='center',
            height=0.6, label='Processing', left=lefts)
    lefts = lefts + Processing_overhead
    ax.barh(y_pos, Overhead, color=cls[1], align='center',
            height=0.6, label='Misc.', left=lefts)
    lefts = lefts + Overhead

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, rotation=0)  # , va='top')
    ax.set_xlabel('Time per iteration (s)')
    ax.set_xlim(0, 2.)
    ax.set_ylim(-1.2, 1.3)

    ax.legend(frameon=False, ncol=4, loc=(-0.2, 1),
              labelspacing=0, prop={'size': 7})
    ax.tick_params(pad=1.5)
    ax.xaxis.labelpad = 0

    plt.subplots_adjust(
        left=0.28, bottom=0.3, right=.96, top=.8, wspace=0.1, hspace=0.1)

    for fmt in ['pdf']:
        if f is not None:
            save_name = os.path.abspath(
                os.path.join(save_folder, fig_name+'.{}'.format(fmt)))
            f.savefig(save_name, format=fmt, dpi=400)
