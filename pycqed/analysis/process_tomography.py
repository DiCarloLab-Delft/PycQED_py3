import numpy as np
import qutip as qtp
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import composite_analysis as ca
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting_tools import *
import time
import os

rotation_matrixes = [qtp.qeye(2).full(),
                     qtp.sigmax().full(),
                     qtp.rotation(qtp.sigmay(), np.pi / 2).full(),
                     qtp.rotation(qtp.sigmay(), -np.pi / 2).full(),
                     qtp.rotation(qtp.sigmax(), np.pi / 2).full(),
                     qtp.rotation(qtp.sigmax(), -np.pi / 2).full()]
pauli_matrixes = [qtp.qeye(2).full(),
                  qtp.sigmax().full(),
                  qtp.sigmay().full(),
                  qtp.sigmaz().full()]
pauli_ro = [qtp.qeye(2).full(),
            qtp.sigmaz().full()]


def get_rotation(idx):
    # j inner loop
    # i outer loop
    j = idx % 6
    i = ((idx - j)//6) % 6
    return np.kron(rotation_matrixes[i], rotation_matrixes[j])
#     return qtp.tensor(rotation_matrixes[i],rotation_matrixes[j])
#     return i,j


def get_pauli(idx):
    # j inner loop
    # i outer loop
    j = idx % 4
    i = ((idx - j)//4) % 4
    return np.kron(pauli_matrixes[i], pauli_matrixes[j])
#     return qtp.tensor(pauli_matrixes[i],pauli_matrixes[j])


def get_measurement_pauli(idx):
    # j inner loop
    # i outer loop
    j = idx % 2
    i = ((idx - j)//2) % 2
    return np.kron(pauli_ro[i], pauli_ro[j])
#     return qtp.tensor(pauli_ro[i],pauli_ro[j])


def unroll_mn(idx):
    # j inner loop
    # i outer loop
    j = idx % 16
    i = ((idx - j)//16) % 16
    return i, j


def unroll_lk(idx):
    # j inner loop
    # i outer loop
    j = idx % 36
    i = ((idx - j)//36) % 36
    return i, j


def get_pauli_txt(idx):
    # j inner loop
    # i outer loop
    j = idx % 4
    i = ((idx - j)//4) % 4
    return pauli_matrixes_txt[i]+pauli_matrixes_txt[j]

pauli_matrixes_txt = ['I', 'X', 'Y', 'Z']


def qpt_matrix_term(l, n, k, j, m):
    #     l preparation index
    #     k tomo index
    #     j beta index
    #     m,n process index
    #     rho00 = qtp.Qobj(np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),dims=[[2,2],[2,2]])
    ul = get_rotation(l)
    pm = get_pauli(m)
    uk = get_rotation(k)
    pj = get_measurement_pauli(j)
    pn = get_pauli(n)
#     trace = (ul.dag()*pn.dag()*uk.dag()*pj*uk*pm*ul*rho00).tr()
    trace = np.dot(dagger(ul), np.dot(
        dagger(pn), np.dot(dagger(uk), np.dot(pj, np.dot(uk, np.dot(pm, ul))))))[0, 0]
#     print(trace)
    return trace


def dagger(op):
    return np.conjugate(np.transpose(op))


def qtp_matrix_element(mn, lk, beta):
    #     beta is wrong!
    m, n = unroll_mn(mn)
    l, k = unroll_lk(lk)
    element = 0.
    for j in range(4):
        element += beta[j]*qpt_matrix_term(l, n, k, j, m)
#     print(mn,element)
    return element


def calc_fidelity1(dens_mat1, dens_mat2):
    sqrt_2 = qtp.Qobj(dens_mat2).sqrtm()
    fid = ((sqrt_2 * qtp.Qobj(dens_mat1) * sqrt_2).sqrtm()).tr()
    return np.real(fid)


def analyze_qpt(t_start, t_stop, label):  # identity tomo

    opt_dict = {'scan_label': label}

    pdict = {'I': 'I',
             'Q': 'Q',
             'times': 'sweep_points'}
    nparams = ['I', 'Q', 'times']

    tomo_scans = ca.quick_analysis(t_start=t_start, t_stop=t_stop,
                                   options_dict=opt_dict,
                                   params_dict_TD=pdict,
                                   numeric_params=nparams)
    assert(len(tomo_scans.TD_timestamps[:]) == 36)

    nr_segments = 64
    measurement_number = 36

    shots_q0 = np.zeros(
        (measurement_number, nr_segments, int(len(tomo_scans.TD_dict['I'][0])/nr_segments)))
    shots_q1 = np.zeros(
        (measurement_number, nr_segments, int(len(tomo_scans.TD_dict['Q'][0])/nr_segments)))
    for j in range(measurement_number):
        for i in range(nr_segments):
            shots_q0[j, i, :] = tomo_scans.TD_dict['I'][j][i::nr_segments]
            shots_q1[j, i, :] = tomo_scans.TD_dict['Q'][j][i::nr_segments]

    shots_q0q1 = np.multiply(shots_q1, shots_q0)

    avg_h1 = np.mean(shots_q0, axis=2)

    h1_00 = np.mean(avg_h1[:, 36:36+7], axis=1)
    h1_01 = np.mean(avg_h1[:, 43:43+7], axis=1)
    h1_10 = np.mean(avg_h1[:, 50:50+7], axis=1)
    h1_11 = np.mean(avg_h1[:, 57:], axis=1)

    avg_h2 = np.mean(shots_q1, axis=2)
    h2_00 = np.mean(avg_h2[:, 36:36+7], axis=1)
    h2_01 = np.mean(avg_h2[:, 43:43+7], axis=1)
    h2_10 = np.mean(avg_h2[:, 50:50+7], axis=1)
    h2_11 = np.mean(avg_h2[:, 57:], axis=1)

    avg_h12 = np.mean(shots_q0q1, axis=2)
    h12_00 = np.mean(avg_h12[:, 36:36+7], axis=1)
    h12_01 = np.mean(avg_h12[:, 43:43+7], axis=1)
    h12_10 = np.mean(avg_h12[:, 50:50+7], axis=1)
    h12_11 = np.mean(avg_h12[:, 57:], axis=1)

    avg_h12 = np.mean(shots_q0q1, axis=2)

    measurements_tomo = np.zeros((measurement_number*measurement_number*3))
    for i in range(measurement_number):
        #     measurements_tomo[i*measurement_number*3:(i+1)*measurement_number*3] = (
        # np.array([avg_h1[i,0:36], avg_h2[i,0:36],
        # avg_h12[i,0:36]])).flatten()
        measurements_tomo[i*36:(i+1)*36] = avg_h1[i, 0:36]
        measurements_tomo[i*36+measurement_number*measurement_number:
                          (i+1)*36+measurement_number*measurement_number] = avg_h2[i, 0:36]
        measurements_tomo[i*36+2*measurement_number*measurement_number:
                          (i+1)*36+2*measurement_number*measurement_number] = avg_h12[i, 0:36]
    measurements_cal = np.array([[h1_00, h1_01, h1_10, h1_11],
                                 [h2_00, h2_01, h2_10, h2_11],
                                 [h12_00, h12_01, h12_10, h12_11]])

    t0 = time.time()
    # get the betas
    betas = np.zeros((3, 4, measurement_number))
    matrix = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    for i in range(measurement_number):
        betas[0, :, i] = np.dot(
            np.linalg.inv(matrix), measurements_cal[0, :, i])
        betas[1, :, i] = np.dot(
            np.linalg.inv(matrix), measurements_cal[1, :, i])
        betas[2, :, i] = np.dot(
            np.linalg.inv(matrix), measurements_cal[2, :, i])
    # define the matrix
    qtp_matrix = np.zeros(
        (measurement_number*measurement_number*3, 16*16), dtype=np.complex128)
    # fill the matrix
    for i in range(measurement_number*measurement_number):
        if ((i % 50) == 0):
            print(i/(measurement_number*measurement_number))
        l, k = unroll_lk(i)
        for s in range(16*16):
            qtp_matrix[i, s] = qtp_matrix_element(s, i, betas[0, :, l])
            qtp_matrix[i+measurement_number*measurement_number,
                       s] = qtp_matrix_element(s, i, betas[1, :, l])
            qtp_matrix[i+2*measurement_number*measurement_number,
                       s] = qtp_matrix_element(s, i, betas[2, :, l])
        t1 = time.time()
    #     print((t1-t0)/(i+1))

    inv_matrix = np.linalg.pinv(qtp_matrix)
    chi_mat = np.dot(inv_matrix, measurements_tomo)
    t2 = time.time()

    chi_mat = chi_mat.reshape((16, 16))

    def fid(chi_mat, phi1, phi2, phi_2Q=np.pi, option=0):
        # fidelity calculation
        chi_mat_theory = np.zeros(chi_mat.shape, dtype=np.complex128)
        chi_mat_theory[0, 0] = 0.25
        chi_mat_theory[0, 3] = 0.25*np.exp(-1j*phi1)
        chi_mat_theory[0, 12] = 0.25*np.exp(-1j*phi2)
        chi_mat_theory[0, 15] = 0.25*np.exp(-1j*(phi1+phi2+phi_2Q))
        chi_mat_theory[3, 0] = 0.25*np.exp(1j*phi1)
        chi_mat_theory[3, 3] = 0.25
        chi_mat_theory[3, 12] = 0.25*np.exp(-1j*(phi1-phi2))
        chi_mat_theory[3, 15] = 0.25*np.exp(1j*(phi2+phi_2Q))
        chi_mat_theory[12, 0] = 0.25*np.exp(-1j*phi2)
        chi_mat_theory[12, 3] = 0.25*np.exp(-1j*(-phi1+phi2))
        chi_mat_theory[12, 12] = 0.25
        chi_mat_theory[12, 15] = 0.25*np.exp(1j*(phi1+phi_2Q))
        chi_mat_theory[15, 0] = 0.25*np.exp(1j*(phi1+phi2+phi_2Q))
        chi_mat_theory[15, 3] = 0.25*np.exp(-1j*(phi2+phi_2Q))
        chi_mat_theory[15, 12] = 0.25*np.exp(1j*(phi1+phi_2Q))
        chi_mat_theory[15, 15] = 0.25

        d = 4
        f_pro = calc_fidelity1(chi_mat, chi_mat_theory)
        f_avg = (((d*f_pro)+1)/(d+1))

        f_pro, f_avg = np.real_if_close(f_pro), np.real_if_close(f_avg)
        if option == 0:
            return np.real(f_avg)
        else:
            return np.real(f_pro)

    phi1_vec = np.linspace(-20, 20, 200)*np.pi/180.
    phi2_vec = np.linspace(-20, 20, 200)*np.pi/180.
    fid_mat = np.zeros((200, 200))

    for i, phi1 in enumerate(phi1_vec):
        for j, phi2 in enumerate(phi2_vec):
            fid_mat[i, j] = fid(chi_mat, phi1, phi2, np.pi)
    f_ave_opt = fid_mat.max()
    f_pro_opt = (f_ave_opt*5-1)/4

    # figures
    plot_times = np.arange(16)
    plot_step = plot_times[1]-plot_times[0]

    plot_x = np.arange(16)
    x_step = plot_x[1]-plot_x[0]

    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax = axs[0]
    cmin, cmax = -0.3, 0.3  # chi_mat.min(),chi_mat.max()
    fig_clim = [cmin, cmax]
    out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='RdBu',
                                    xvals=plot_times,
                                    yvals=plot_x,
                                    zvals=np.real(chi_mat))
    ax.set_xlabel(r'Operators')
    ax.set_ylabel(r'Operators')
    # ax.set_xlim(xmin, xmax)
    ax.set_ylim(plot_x.min()-x_step/2., plot_x.max()+x_step/2.)
    ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
    ax.set_xticks(plot_times)
    ax.set_xticklabels([get_pauli_txt(i) for i in range(16)])
    ax.set_yticks(plot_x)
    ax.set_yticklabels([get_pauli_txt(i) for i in range(16)])
    # ax.set_xlim(0,50)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='10%', pad='5%')
    cbar = plt.colorbar(out['cmap'], cax=cax)
    cbar.set_ticks(
        np.arange(fig_clim[0], 1.01*fig_clim[1], (fig_clim[1]-fig_clim[0])/5.))
    cbar.set_ticklabels([str(fig_clim[0]), '', '', '', '', str(fig_clim[1])])
    cbar.set_label('Process Tomography')

    ax = axs[1]
    out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='RdBu',
                                    xvals=plot_times,
                                    yvals=plot_x,
                                    zvals=np.imag(chi_mat))
    ax.set_xlabel(r'Operators')
    ax.set_ylabel(r'Operators')
    # ax.set_xlim(xmin, xmax)
    ax.set_ylim(plot_x.min()-x_step/2., plot_x.max()+x_step/2.)
    ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
    ax.set_xticks(plot_times)
    ax.set_xticklabels([get_pauli_txt(i) for i in range(16)])
    ax.set_yticks(plot_x)
    ax.set_yticklabels([get_pauli_txt(i) for i in range(16)])
    # ax.set_xlim(0,50)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='10%', pad='5%')
    cbar = plt.colorbar(out['cmap'], cax=cax)
    cbar.set_ticks(
        np.arange(fig_clim[0], 1.01*fig_clim[1], (fig_clim[1]-fig_clim[0])/5.))
    cbar.set_ticklabels([str(fig_clim[0]), '', '', '', '', str(fig_clim[1])])
    cbar.set_label('Process Tomography')
    fig.tight_layout()

    fig.suptitle('%s - %s: Quantum Process Tomography. F_avg = %.4f; F_opt = %.4f' % (tomo_scans.TD_timestamps[0],
                                                                                      tomo_scans.TD_timestamps[
                                                                                          -1],
                                                                                      f_ave_opt, f_pro_opt))

    figname = '%s_QTP_manhattan.PNG' % tomo_scans.TD_timestamps[0]

#     savename = os.path.abspath(os.path.join(
#         savefolder, figname))
#     # value of 450dpi is arbitrary but higher than default
#     fig.savefig(savename, format='png', dpi=450)
    return chi_mat, f_ave_opt


def chi2PTM():
    return


def PTM2chi():
    return


def chi_PTM_matrices():
    return
