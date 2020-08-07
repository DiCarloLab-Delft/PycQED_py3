# Based on Niels analysis
from os.path import join
from pycqed.analysis import measurement_analysis as ma
from scipy import stats
import numpy as np
import pylab
# %matplotlib inline
from matplotlib import pyplot as plt
from numpy.linalg import inv


def two_qubit_ssro_fidelity(label, fig_format='png',
                            qubit_labels=('q0', 'q1')):
    # extracting data sets
    states = ['00', '01', '10', '11']
    nr_states = len(states)
    namespace = globals()

    data = ma.MeasurementAnalysis(auto=False, label=label)
    data.get_naming_and_values()

    # extract fit parameters for q0

    w0_data = data.measured_values[0]
    w1_data = data.measured_values[1]
    lengths = []
    i = 0
    for nr_state, state in enumerate(states):
        if i == 0:
            namespace['w0_data_r{}'.format(state)] = []
            namespace['w1_data_r{}'.format(state)] = []

    for nr_state, state in enumerate(states):
        namespace['w0_data_sub_{}'.format(state)] = w0_data[
            nr_state::nr_states]
        namespace['w1_data_sub_{}'.format(state)] = w1_data[
            nr_state::nr_states]
        lengths.append(len(w0_data[nr_state::nr_states]))

    # capping off the maximum lengths
    min_len = np.min(lengths)
    for nr_state, state in enumerate(states):
        namespace['w0_data_sub_{}'.format(state)] = namespace[
            'w0_data_sub_{}'.format(state)][0:min_len]
        namespace['w1_data_sub_{}'.format(state)] = namespace[
            'w1_data_sub_{}'.format(state)][0:min_len]
    for nr_state, state in enumerate(states):
        namespace['w0_data_r{}'.format(
            state)] += list(namespace['w0_data_sub_{}'.format(state)])
        namespace['w1_data_r{}'.format(
            state)] += list(namespace['w1_data_sub_{}'.format(state)])

    for nr_state, state in enumerate(states):
        namespace['w0_data_{}'.format(state)] = namespace[
            'w0_data_r{}'.format(state)]
        namespace['w1_data_{}'.format(state)] = namespace[
            'w1_data_r{}'.format(state)]

    min_len_all = min_len/2

    ###########################################################################
    # Extracting and plotting the results for q0 (first weight function)
    ###########################################################################

    ma.SSRO_Analysis(label=label, auto=True, channels=[data.value_names[0]],
                     sample_0=0,
                     sample_1=1, nr_samples=4, rotate=False)
    ana = ma.MeasurementAnalysis(label=label, auto=False)
    ana.load_hdf5data()
    Fa_q0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_a']
    Fd_q0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_d']
    mu0_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_0']
    mu1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_0']
    mu0_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_1']
    mu1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_1']

    sigma0_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_0']
    sigma1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_1']
    sigma0_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_1']
    sigma1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_0']
    frac1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_0']
    frac1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_1']
    V_opt = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_a']
    V_opt_d = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_d']

    SNR_q0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['SNR']

    n, bins0, patches = plt.hist(namespace['w0_data_00'],
                                 bins=int(min_len_all/50),
                                 label='input state {}'.format(state),
                                 histtype='step',
                                 color='red', density=True, visible=False)
    n, bins1, patches = plt.hist(namespace['w0_data_01'],
                                 bins=int(min_len_all/50),
                                 label='input state {}'.format(state),
                                 histtype='step',
                                 color='red', density=True, visible=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'red', 'grey', 'magenta']
    markers = ['o', 'o', 'o', 'v']

    for marker, color, state in zip(markers, colors, states):

        n, bins, patches = ax.hist(namespace['w0_data_{}'.format(state)],
                                   bins=int(min_len_all/50),
                                   histtype='step',  density=True,
                                   visible=False)
        ax.plot(bins[:-1]+0.5*(bins[1]-bins[0]), n, color=color,
                linestyle='None', marker=marker, label='|{}>'.format(state))

    y0 = (1-frac1_0)*stats.norm.pdf(bins0, mu0_0, sigma0_0) + \
        frac1_0*stats.norm.pdf(bins0, mu1_0, sigma1_0)
    # y1_0 = frac1_0*stats.norm.pdf(bins0, mu1_0, sigma1_0)
    # y0_0 = (1-frac1_0)*stats.norm.pdf(bins0, mu0_0, sigma0_0)

    # building up the histogram fits for on measurements
    y1 = (1-frac1_1)*stats.norm.pdf(bins1, mu0_1, sigma0_1) + \
        frac1_1*stats.norm.pdf(bins1, mu1_1, sigma1_1)
    # y1_1 = frac1_1*stats.norm.pdf(bins1, mu1_1, sigma1_1)
    # y0_1 = (1-frac1_1)*stats.norm.pdf(bins1, mu0_1, sigma0_1)

    ax.semilogy(bins0, y0, 'b', linewidth=1.5, label='fit |00>')
    ax.semilogy(bins1, y1, 'r', linewidth=1.5, label='fit |01>')
    ax.set_ylim(0.2e-6, 1e-3)

    pdf_max = (max(max(y0), max(y1)))
    ax.set_ylim(pdf_max/100, 2*pdf_max)

    ax.set_title('Histograms for {}'.format(qubit_labels[0]))
    ax.set_xlabel('Integration result {} (a.u.)'.format(qubit_labels[0]))
    ax.set_ylabel('Fraction of counts')
    ax.axvline(V_opt, ls='--',
               linewidth=2, color='grey',
               label='SNR={0:.2f}\n $F_a$={1:.5f}\n $F_d$={2:.5f}'.format(
                   SNR_q0, Fa_q0, Fd_q0))
    ax.axvline(V_opt_d, ls='--', linewidth=2, color='black')
    ax.legend(frameon=False, loc='upper right')
    a = ax.get_xlim()
    ax.set_xlim(a[0], a[0]+(a[1]-a[0])*1.2)
    plt.savefig(join(ana.folder, 'histogram_w0.' +
                     fig_format), format=fig_format)
    plt.close()

    V_th = np.zeros(len(qubit_labels))
    V_th_d = np.zeros(len(qubit_labels))

    V_th[0] = V_opt
    V_th_d[0] = V_opt_d


    ###########################################################################
    # Extracting and plotting the results for q1 (second weight function)
    ###########################################################################

    ma.SSRO_Analysis(label=label, auto=True, channels=[data.value_names[1]],
                     sample_0=0, sample_1=2, nr_samples=4, rotate=False)
    ana = ma.MeasurementAnalysis(label=label, auto=False)
    ana.load_hdf5data()
    Fa_q1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_a']
    Fd_q1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_d']
    mu0_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_0']
    mu1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_0']
    mu0_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_1']
    mu1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_1']

    sigma0_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_0']
    sigma1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_1']
    sigma0_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_1']
    sigma1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_0']
    frac1_0 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_0']
    frac1_1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_1']
    V_opt = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_a']
    V_opt_d = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_d']
    SNR_q1 = ana.data_file['Analysis']['SSRO_Fidelity'].attrs['SNR']

    n, bins0, patches = plt.hist(namespace['w1_data_00'],
                                 bins=int(min_len_all/50),
                                 label='input state {}'.format(state),
                                 histtype='step', color='red', density=True,
                                 visible=False)
    n, bins1, patches = plt.hist(namespace['w1_data_10'],
                                 bins=int(min_len_all/50),
                                 label='input state {}'.format(state),
                                 histtype='step',
                                 color='red', density=True, visible=False)
    fig, axes = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'red', 'grey', 'magenta']
    markers = ['o', 'o', 'o', 'v']
    for marker, color, state in zip(markers, colors, states):

        n, bins, patches = plt.hist(namespace['w1_data_{}'.format(state)],
                                    bins=int(min_len_all/50),
                                    histtype='step',  density=True,
                                    visible=False)
        pylab.plot(bins[:-1]+0.5*(bins[1]-bins[0]), n,
                   color=color, linestyle='None', marker=marker)

    y0 = (1-frac1_0)*stats.norm.pdf(bins0, mu0_0, sigma0_0) + \
        frac1_0*stats.norm.pdf(bins0, mu1_0, sigma1_0)

    # building up the histogram fits for on measurements
    y1 = (1-frac1_1)*stats.norm.pdf(bins1, mu0_1, sigma0_1) + \
        frac1_1*stats.norm.pdf(bins1, mu1_1, sigma1_1)
    # y1_1 = frac1_1*stats.norm.pdf(bins1, mu1_1, sigma1_1)
    # y0_1 = (1-frac1_1)*stats.norm.pdf(bins1, mu0_1, sigma0_1)

    plt.semilogy(bins0, y0, 'b', linewidth=1.5, label='fit |00>')

    plt.semilogy(bins1, y1, 'r', linewidth=1.5, label='fit |10>')
    (pylab.gca()).set_ylim(0.2e-6, 1e-3)
    pdf_max = (max(max(y0), max(y1)))
    (pylab.gca()).set_ylim(pdf_max/100, 2*pdf_max)

    axes.set_title('Histograms for {}'.format(qubit_labels[1]))
    plt.xlabel('Integration result {} (a.u.)'.format(qubit_labels[1]))
    plt.ylabel('Fraction of counts')
    plt.axvline(V_opt, ls='--',
                linewidth=2, color='grey',
                label='SNR={0:.2f}\n $F_a$={1:.5f}\n $F_d$={2:.5f}'.format(
                    SNR_q1, Fa_q1, Fd_q1))
    plt.axvline(V_opt_d, ls='--',
                linewidth=2, color='black')
    plt.legend(frameon=False, loc='upper right')
    a = plt.xlim()
    plt.xlim(a[0], a[0]+(a[1]-a[0])*1.2)
    plt.savefig(join(ana.folder, 'histogram_w1.' +
                     fig_format), format=fig_format)
    plt.close()
    V_th[1] = V_opt
    V_th_d[1] = V_opt_d

    # calculating cross-talk matrix and inverting
    ground_state = '00'
    weights = [0, 1]
    cal_states = ['01', '10']
    ground_state = '00'
    mu_0_vec = np.zeros(len(weights))
    for j, weight in enumerate(weights):
        mu_0_vec[j] = np.average(
            namespace['w{}_data_{}'.format(weight, ground_state)])

    mu_matrix = np.zeros((len(cal_states), len(weights)))
    for i, state in enumerate(cal_states):
        for j, weight in enumerate(weights):
            mu_matrix[i, j] = np.average(
                namespace['w{}_data_{}'.format(weight, state)])-mu_0_vec[j]

    mu_matrix_inv = inv(mu_matrix)
    V_th_cor = np.dot(mu_matrix_inv, V_th)
    V_offset_cor = np.dot(mu_matrix_inv, mu_0_vec)
    res_dict = {'mu_matrix': mu_matrix,  'V_th': V_th,
                'V_th_d': V_th_d,
                'mu_matrix_inv': mu_matrix_inv,
                'V_th_cor': V_th_cor,  'V_offset_cor': V_offset_cor,
                'Fa_q0': Fa_q0, 'Fa_q1': Fa_q1, 'Fd_q0': Fd_q0, 'Fd_q1': Fd_q1,
                'SNR_q0': SNR_q0, 'SNR_q1': SNR_q1}
    return res_dict
