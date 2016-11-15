import numpy as np
import matplotlib.gridspec as gridspec
from pycqed.analysis import analysis_toolbox as a_tools

import seaborn.apionly as sns
sns.set_palette('muted')
cls = (sns.color_palette())
import matplotlib.pyplot as plt

from pycqed.scripts.Experiments.Restless.paper_figs.helper_funcs import latexify, cm2inch

import os

a_tools.datadir = '/Users/Adriaan/Dropbox/PhD-Delft/Experiments_and_projects/Restless tuning/Restless_data'
save_folder = '/Users/Adriaan/GitHubRepos/DiCarloLab_Repositories/Paper_16_RestlessTuning/Figures'
save_format = 'pdf'


params = {  # 'backend': 'ps',
    'axes.labelpad': 0.1,
    'legend.fontsize': 7,
    'legend.handlelength': 3,
    'legend.columnspacing': 1,
    'legend.handletextpad': .2,
    'ytick.major.pad': 2,
    'ytick.minor.pad': 2,
}

# latexify(fig_height=5)


def make_figure(F_vec, idxs, Ncl,
                mn_eps, sem_eps, std_eps, sem_std,
                fig_name='signal_noise',
                Ncl_cont=None,
                simple_std=None,
                model_avg=None, model_std=None,
                proj_avg=None, proj_std=None):

    if Ncl_cont is None:
        Ncl_cont = Ncl
    if model_avg is not None:
        avg1, avg2, avg3 = model_avg
    if model_std is not None:
        std1, std2, std3 = model_std
    if simple_std is not None:
        std_fix1, std_fix2, std_fix3 = simple_std

    latexify()
    plt.rcParams.update(params)
    f, ax = plt.subplots(figsize=cm2inch(8.6, 7))
    ncols = 1
    nrows = 2
    height_ratios = np.ones(nrows)
    height_ratios[1] = .6
    width_ratios = np.ones(ncols)
    gs = gridspec.GridSpec(nrows, ncols, wspace=.05, hspace=0,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)

    ax = plt.subplot(gs[0])

    colors = sns.color_palette()

    labels = [
        '$F_\mathrm{Cl}^\mathrm{a}='+' {:.4f}$, '.format((F_vec[idxs[0][0]])) +
        r'$\Delta F_\mathrm{Cl} =' +
        ' {:.4f} $'.format((F_vec[idxs[0][1]]-F_vec[idxs[0][0]])),
        '$F_\mathrm{Cl}^\mathrm{a}='+' {:.4f}$, '.format((F_vec[idxs[1][0]])) +
        r'$\Delta F_\mathrm{Cl} =' +
        ' {:.4f} $'.format((F_vec[idxs[1][1]]-F_vec[idxs[1][0]])),
        '$F_\mathrm{Cl}^\mathrm{a}='+' {:.4f}$, '.format((F_vec[idxs[2][0]])) +
        r'$\Delta F_\mathrm{Cl} ='+' {:.4f} $'.format((F_vec[idxs[2][1]]-F_vec[idxs[2][0]]))]

    markers = ['o', 'd', '^']
    # Model averages
    if model_avg is not None:
        ax.plot(Ncl_cont, np.array(avg1),
                color=colors[0])
        ax.plot(Ncl_cont, np.array(avg2),
                color=colors[1])
        ax.plot(Ncl_cont, np.array(avg3),
                color=colors[2])
    if proj_avg is not None:
        ax.plot(Ncl_cont, np.array(proj_avg),
                color=colors[3])

    # Data averages
    print(np.shape(mn_eps))
    for i, id_pair in enumerate(idxs):
        ax.errorbar(Ncl, (mn_eps[:, id_pair[0]]-mn_eps[:, id_pair[1]])/100.,
                    (sem_eps[:, id_pair[1]] **
                     2+sem_eps[:, id_pair[0]]**2)**.5/100.,
                    marker=markers[i],
                    ls='',
                    label=labels[i],)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc=(.0, .63))
    ax.text(.65, .85, r'$\Delta F_\mathrm{Cl} = F_\mathrm{Cl}^\mathrm{b} - F_\mathrm{Cl}^\mathrm{a}$',
            transform=ax.transAxes, fontsize=7)
    ax.hlines(0, 0, 1600, linestyle='dotted')
    ax.set_ylim(-0.02, 0.17)

    ax.set_ylabel('')
    ax.set_xlabel(r'$N_{\mathrm{Cl}}$')
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    plt.setp(ax.get_xticklabels(), visible=False)
    # Model sigmas
    ax2 = plt.subplot(gs[1], sharex=ax)
    if model_std is not None:
        ax2.plot(Ncl_cont, np.array(std1),
                 color=colors[0])
        ax2.plot(Ncl_cont, np.array(std2),
                 color=colors[1])
        ax2.plot(Ncl_cont, np.array(std3),
                 color=colors[2])
    if proj_std is not None:
        ax2.plot(Ncl_cont, np.array(proj_std),
                 color=colors[3])

    # simple model
    if simple_std is not None:
        ax2.plot(Ncl_cont, np.array(std_fix1),
                 color=colors[0], linestyle='--')
        ax2.plot(Ncl_cont, np.array(std_fix2),
                 color=colors[1], linestyle='--')
        ax2.plot(Ncl_cont, np.array(std_fix3),
                 color=colors[2], linestyle='--')
    # data sigma
    for i, id_pair in enumerate(idxs):
        ax2.errorbar(Ncl,
                     (std_eps[:, id_pair[1]]+std_eps[:, id_pair[0]])*.5/100,
                     (sem_std[:, id_pair[1]]+sem_std[:, id_pair[0]])*.5/100,
                     marker=markers[i], ls='')

    # Dummy lines for legend
    if simple_std is not None:
        ax2.plot([1e6, 1e7], [0, 0], color='grey',
                 linestyle='--', label='Simple model')
    if model_avg is not None:
        ax2.plot([1e6, 1e7], [0, 0], color='k',
                 linestyle='-', label='Extensive model')
    ax2.legend(frameon=False, loc=(.2, .05), ncol=2)

    ax2.set_ylim(0.,  0.0155)
    ax2.set_xlabel('Number of Cliffords, $N_{\mathrm{Cl}}$')
    ax2.set_ylabel('')
    ax2.set_yticks([0, 0.005, 0.01, 0.015])
    ax.set_xlim(1, 2000)
    ax.set_xscale('log')
    ax.text(0.92, .88, '(a)', transform=ax.transAxes)
    ax2.text(0.92, .88, '(b)', transform=ax2.transAxes)

    ax.text(-0.14, .5, r'Signal, $ {\Delta\overline{\varepsilon_\mathrm{R}}}$',
            color='k', transform=ax.transAxes,
            ha='center', va='center',
            rotation='90')
    ax2.text(-0.14, .5, r'Noise, $\overline{\sigma_{\varepsilon _\mathrm{R}}}$',
             color='k', transform=ax2.transAxes,
             ha='center', va='center',
             rotation='90')

    plt.gcf().subplots_adjust(bottom=0.17)
    plt.subplots_adjust(
        left=0.14, bottom=0.11, right=.98, top=.99, wspace=0.1, hspace=0.1)
    for fmt in ['pdf']:
        if f is not None:
            save_name = os.path.abspath(
                os.path.join(save_folder, fig_name+'.{}'.format(fmt)))
            f.savefig(save_name, format=fmt, dpi=1200)
