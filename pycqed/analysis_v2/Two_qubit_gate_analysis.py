import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
from matplotlib.colors import to_rgba, LogNorm
from pycqed.analysis.tools.plotting import hsluv_anglemap45


class Two_qubit_gate_tomo_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for the two qubit gate tomography calibration experiment.
    
    """

    def __init__(self, n_pairs: int,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.n_pairs = n_pairs
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):

        self.proc_data_dict = {}
        self.qoi = {}

        for n in range(self.n_pairs):
            # Raw I Q shots from Ramsey qubit
            I_data = self.raw_data_dict['data'][:,1+2*n]
            Q_data = self.raw_data_dict['data'][:,2+2*n]
            C_data = self.raw_data_dict['data'][:,1+2*(self.n_pairs+n)]
            # Calibration points shots
            Cal_0 = {'I': I_data[12::16], 'Q': Q_data[12::16]}
            Cal_1 = {'I': I_data[13::16], 'Q': Q_data[13::16]}
            Cal_2 = {'I': I_data[14::16], 'Q': Q_data[14::16]}
            # Average RO level extracted from calibration points
            avg_0 = np.array([np.mean(Cal_0['I']), np.mean(Cal_0['Q'])])
            avg_1 = np.array([np.mean(Cal_1['I']), np.mean(Cal_1['Q'])])
            avg_2 = np.array([np.mean(Cal_2['I']), np.mean(Cal_2['Q'])])
            # Raw Ramsey qubit shots
            Pauli_Z_off_raw = {'I': np.concatenate((I_data[0::16], I_data[6::16])), 
                               'Q': np.concatenate((Q_data[0::16], Q_data[6::16])),
                               'C': np.concatenate((C_data[0::16], C_data[6::16]))}
            Pauli_Z_on_raw  = {'I': np.concatenate((I_data[1::16], I_data[7::16])), 
                               'Q': np.concatenate((Q_data[1::16], Q_data[7::16])),
                               'C': np.concatenate((C_data[1::16], C_data[7::16]))}
            Pauli_X_off_raw = {'I': np.concatenate((I_data[2::16], I_data[8::16])), 
                               'Q': np.concatenate((Q_data[2::16], Q_data[8::16])),
                               'C': np.concatenate((C_data[2::16], C_data[8::16]))}
            Pauli_X_on_raw  = {'I': np.concatenate((I_data[3::16], I_data[9::16])), 
                               'Q': np.concatenate((Q_data[3::16], Q_data[9::16])),
                               'C': np.concatenate((C_data[3::16], C_data[9::16]))}
            Pauli_Y_off_raw = {'I': np.concatenate((I_data[4::16], I_data[10::16])), 
                               'Q': np.concatenate((Q_data[4::16], Q_data[10::16])),
                               'C': np.concatenate((C_data[4::16], C_data[10::16]))}
            Pauli_Y_on_raw  = {'I': np.concatenate((I_data[5::16], I_data[11::16])), 
                               'Q': np.concatenate((Q_data[5::16], Q_data[11::16])),
                               'C': np.concatenate((C_data[5::16], C_data[11::16]))}
            # Assigning shots based on readout levels
            def state_assignment(P):
                '''
                Takes dictionary of input vector shots and returns digitized vector
                of shots.
                '''
                N = len(P['I'])
                P_dig = np.zeros(N)
                P_state = np.zeros(N)
                P2 = 0
                for i in range(N):
                    P_vec = np.array([P['I'][i], P['Q'][i]])
                    dist_0 = np.linalg.norm(P_vec-avg_0)
                    dist_1 = np.linalg.norm(P_vec-avg_1)
                    dist_2 = np.linalg.norm(P_vec-avg_2)
                    P_dig[i] = np.argmin([dist_0, dist_1])*-2+1
                    P_state[i] = np.argmin([dist_0, dist_1, dist_2])
                    if P_state[i] == 2:
                        P2 += 1/N
                return P_dig, P2
            Pauli_X_off_dig, P2_X_off = state_assignment(Pauli_X_off_raw)
            Pauli_X_on_dig , P2_X_on  = state_assignment(Pauli_X_on_raw)
            Pauli_Y_off_dig, P2_Y_off = state_assignment(Pauli_Y_off_raw)
            Pauli_Y_on_dig , P2_Y_on  = state_assignment(Pauli_Y_on_raw)
            Pauli_Z_off_dig, P2_Z_off = state_assignment(Pauli_Z_off_raw)
            Pauli_Z_on_dig , P2_Z_on  = state_assignment(Pauli_Z_on_raw)
            ####################################
            # Calculate quantities of interest
            ####################################
            # Pauli vectors for contorl qubit in On or Off
            avg_X_off = np.mean(Pauli_X_off_dig)
            avg_Y_off = np.mean(Pauli_Y_off_dig)
            avg_Z_off = np.mean(Pauli_Z_off_dig)
            avg_X_on = np.mean(Pauli_X_on_dig)
            avg_Y_on = np.mean(Pauli_Y_on_dig)
            avg_Z_on = np.mean(Pauli_Z_on_dig)
            # Projection of Bloch vector onto the equator
            r_off = np.sqrt(avg_Y_off**2+avg_X_off**2)
            r_on  = np.sqrt(avg_Y_on**2+avg_X_on**2)
            phi_off = np.mod(np.arctan2(avg_Y_off, avg_X_off), 2*np.pi)
            phi_on  = np.mod(np.arctan2(avg_Y_on, avg_X_on), 2*np.pi)
            # Calculate purity of the state (magnitude of bloch vector)
            Purity_off = np.sqrt(avg_X_off**2+avg_Y_off**2+avg_Z_off**2)
            Purity_on = np.sqrt(avg_X_on**2+avg_Y_on**2+avg_Z_on**2)
            # Average Leakage over all Pauli components
            Leakage_off = np.mean([P2_X_off, P2_Y_off, P2_Z_off])*100
            Leakage_on  = np.mean([P2_X_on, P2_Y_on, P2_Z_on])*100

            # Save quantities of interest
            self.proc_data_dict[f'Cal_shots_{n}'] = [Cal_0, Cal_1, Cal_2]
            self.proc_data_dict[f'Pauli_vector_off_{n}'] = [avg_X_off, avg_Y_off, avg_Z_off]
            self.proc_data_dict[f'Pauli_vector_on_{n}']  = [avg_X_on, avg_Y_on, avg_Z_on]
            self.proc_data_dict[f'R_off_{n}'] = r_off
            self.proc_data_dict[f'R_on_{n}']  = r_on
            self.proc_data_dict[f'Phi_off_{n}'] = phi_off
            self.proc_data_dict[f'Phi_on_{n}']  = phi_on
            self.proc_data_dict[f'Purity_off_{n}'] = Purity_off
            self.proc_data_dict[f'Purity_on_{n}']  = Purity_on
            self.proc_data_dict[f'Leakage_off_{n}'] = Leakage_off
            self.proc_data_dict[f'Leakage_on_{n}']  = Leakage_on

            self.qoi[f'Leakage_diff_{n}'] = Leakage_on-Leakage_off
            self.qoi[f'Phase_diff_{n}'] = np.mod(phi_on-phi_off, 2*np.pi)*180/np.pi

    def prepare_plots(self):

        self.axs_dict = {}
        for n in range(self.n_pairs):

            self.figs[f'Main_figure_{n}'] = plt.figure(figsize=(8,8), dpi=100)
            axs = [self.figs[f'Main_figure_{n}'].add_subplot(231), 
                   self.figs[f'Main_figure_{n}'].add_subplot(232),
                   self.figs[f'Main_figure_{n}'].add_subplot(222),
                   self.figs[f'Main_figure_{n}'].add_subplot(223, projection='polar'),
                   self.figs[f'Main_figure_{n}'].add_subplot(224),
                   self.figs[f'Main_figure_{n}'].add_subplot(233)]
            self.figs[f'Main_figure_{n}'].patch.set_alpha(0)
            
            self.axs_dict[f'Tomo_off_{n}'] = axs[0]
            self.axs_dict[f'Tomo_on_{n}']  = axs[1]
            self.axs_dict[f'Calibration_points_{n}'] = axs[2]
            self.axs_dict[f'Equator_{n}'] = axs[3]
            self.axs_dict[f'Leakage_{n}'] = axs[4]
            self.axs_dict[f'Param_table_{n}'] = axs[5]

            self.plot_dicts[f'Pauli_off_plot_{n}']={
                'plotfn': Tomo_plotfn_1,
                'data': self.proc_data_dict[f'Pauli_vector_off_{n}'],
                'ax_id': f'Tomo_off_{n}'
            }
            self.plot_dicts[f'Pauli_on_plot_{n}']={
                'plotfn': Tomo_plotfn_2,
                'data': self.proc_data_dict[f'Pauli_vector_on_{n}'],
                'ax_id': f'Tomo_on_{n}'
            }
            self.plot_dicts[f'Calibration_points_{n}']={
                'plotfn': Calibration_plotfn,
                'Cal_0': self.proc_data_dict[f'Cal_shots_{n}'][0],
                'Cal_1': self.proc_data_dict[f'Cal_shots_{n}'][1],
                'Cal_2': self.proc_data_dict[f'Cal_shots_{n}'][2],
                'labels': self.raw_data_dict['value_names'][2*n:],
                'ax_id': f'Calibration_points_{n}'
            }
            self.plot_dicts[f'Equator_{n}']={
                'plotfn': Equator_plotfn,
                'r_off': self.proc_data_dict[f'R_off_{n}'],
                'r_on': self.proc_data_dict[f'R_on_{n}'],
                'phi_off': self.proc_data_dict[f'Phi_off_{n}'],
                'phi_on': self.proc_data_dict[f'Phi_on_{n}'],            
                'ax_id': f'Equator_{n}'
            }
            self.plot_dicts[f'Leakage_{n}']={
                'plotfn': Leakage_plotfn,
                'Leakage_off': self.proc_data_dict[f'Leakage_off_{n}'],
                'Leakage_on': self.proc_data_dict[f'Leakage_on_{n}'],           
                'ax_id': f'Leakage_{n}'
            }
            self.plot_dicts[f'Param_table_{n}']={
                'plotfn': Param_table_plotfn,
                'phi_off': self.proc_data_dict[f'Phi_off_{n}'],
                'phi_on': self.proc_data_dict[f'Phi_on_{n}'],
                'Purity_off': self.proc_data_dict[f'Purity_off_{n}'],
                'Purity_on': self.proc_data_dict[f'Purity_on_{n}'], 
                'Leakage_off': self.proc_data_dict[f'Leakage_off_{n}'],
                'Leakage_on': self.proc_data_dict[f'Leakage_on_{n}'],           
                'ax_id': f'Param_table_{n}'
            }


    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))



def Tomo_plotfn_1(ax, data, **kw):
    ax.set_position((.0, .76, 0.4, .14))
    ax.bar([0], [1], ls='--', ec='k', fc=to_rgba('purple', alpha=.1))
    ax.bar([0,1,2], data, fc=to_rgba('purple', alpha=.8))
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel(r'$\langle m_{\sigma}\rangle$', labelpad=-5)
    ax.set_xlim(-.5, 2.5)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['','', '', ''])
    ax.text(1.65, .75, r'Control $|0\rangle$')
    ax.set_title('Pauli expectation values')


def Tomo_plotfn_2(ax, data, **kw):
    ax.set_position((.0, .6, 0.4, .14))
    ax.bar([0], [-1], ls='--', ec='k', fc=to_rgba('purple', alpha=.1))
    ax.bar([0,1,2], data, fc=to_rgba('purple', alpha=.8))
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel(r'$\langle m_{\sigma}\rangle$', labelpad=-5)
    ax.set_xlim(-.5, 2.5)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.text(1.65, .75, r'Control $|1\rangle$')

def Calibration_plotfn(ax, Cal_0, Cal_1, Cal_2, labels, **kw):
    ax.set_position((.49, .6, 0.3, 0.3))
    ax.scatter(Cal_0['I'], Cal_0['Q'], color='C0', 
                   marker='.', alpha=.05, label=r'$|0\rangle$')
    ax.scatter(Cal_1['I'], Cal_1['Q'], color='C3', 
                   marker='.', alpha=.05, label=r'$|1\rangle$')
    ax.scatter(Cal_2['I'], Cal_2['Q'], color='C2', 
                   marker='.', alpha=.05, label=r'$|2\rangle$')
    ax.set_xlabel(labels[0].decode())
    ax.set_ylabel(labels[1].decode())
    ax.set_title('Calibration points')
    leg = ax.legend(frameon=False, ncol=3, columnspacing=1.)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)


def Equator_plotfn(ax, r_off, phi_off, r_on, phi_on, **kw):
    ax.set_position((0.02, .25, 0.23, 0.23))
    ax.set_rlim(0, 1)
    ax.set_rticks([.5])
    ax.set_yticklabels([''])
    ax.plot([0, phi_off], [0, r_off], 'C0--', alpha=.5, lw=1)
    ax.plot([0, phi_on], [0, r_on], 'C3--', alpha=.5, lw=1)
    ax.plot([phi_off], [r_off], 'C0o', label=r'Control $|0\rangle$')
    ax.plot([phi_on], [r_on], 'C3o', label=r'Control $|1\rangle$')
    ax.set_title('Projection onto equator', pad=20)
    ax.legend(loc=8, frameon=False, fontsize=7)


def Leakage_plotfn(ax, Leakage_off, Leakage_on, **kw):
    ax.set_position((0.35, .27, 0.15, 0.24))
    ax.bar([0,1], [Leakage_off, Leakage_on], fc=to_rgba('C2', alpha=1))
    ax.bar([0], [Leakage_on], fc=to_rgba('C2', alpha=.2))
    ax.set_xticks([0,1])
    ax.set_xticklabels([r'$|0\rangle$', r'$|1\rangle$'])
    ax.set_xlabel(r'Control state')
    ax.set_ylabel(r'P$(|2\rangle)$ (%)')
    ax.set_title(r'Leakage $|2\rangle$')


def Param_table_plotfn(ax,
                       phi_off,
                       phi_on,
                       Purity_off,
                       Purity_on,
                       Leakage_off,
                       Leakage_on,
                       **kw):

    ax.set_position((0.6, .37, 0.2, 0.1))
    collabel=(r'$|0\rangle_C$', r'$|1\rangle_C$')
    ax.axis('off')
    tab_values=[['{:.2f}'.format(phi_off*180/np.pi), '{:.2f}'.format(phi_on*180/np.pi)],
                ['{:.3f}'.format(Purity_off), '{:.3f}'.format(Purity_on)],
                ['{:.2f}'.format(Leakage_off), '{:.2f}'.format(Leakage_on)]]

    table = ax.table(cellText=tab_values,
                         colLabels=collabel,
                         rowLabels=[r'$\phi_\mathrm{Ramsey}$',
                                    r'Purity',
                                    r'$P(|2\rangle)$'],
                         colWidths=[.3] * 2,
                         loc='center')

    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    ax.text(-.4,-.5, 'Cphase: {:.2f}$^o$'.format((phi_on-phi_off)*180/np.pi), fontsize=14)
    ax.text(-.4,-.9, 'Leakage diff: {:.2f} %'.format(Leakage_on-Leakage_off), fontsize=14)


class VCZ_tmid_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q0,
                 Q1,
                 A_ranges,
                 Q_parks: str = None,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q0 = Q0
        self.Q1 = Q1
        self.Q_parks = Q_parks
        self.ranges = A_ranges
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}

        Amps_idxs = np.unique(self.raw_data_dict['data'][:,0])
        Tmid = np.unique(self.raw_data_dict['data'][:,1])
        nx, ny = len(Amps_idxs), len(Tmid)
        Amps_list = [ np.linspace(r[0], r[1], nx) for r in self.ranges ] 
        self.proc_data_dict['Amps'] = Amps_list
        self.proc_data_dict['Tmid'] = Tmid

        for i, q0 in enumerate(self.Q0):
            CP = self.raw_data_dict['data'][:,2*i+2].reshape(ny, nx)
            MF = self.raw_data_dict['data'][:,2*i+3].reshape(ny, nx)
            self.proc_data_dict[f'CP_{i}'] = CP
            self.proc_data_dict[f'MF_{i}'] = MF

    def prepare_plots(self):
        self.axs_dict = {}
        n = len(self.Q0)
        self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'] = plt.figure(figsize=(9,4*n), dpi=100)
        # self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].patch.set_alpha(0)
        axs = []
        for i, q0 in enumerate(self.Q0):
            axs.append(self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].add_subplot(n,2,2*i+1))
            axs.append(self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].add_subplot(n,2,2*i+2))

            self.axs_dict[f'plot_{i}'] = axs[0]

            self.plot_dicts[f'VCZ_landscape_{self.Q0}_{self.Q1}_{i}']={
                'plotfn': VCZ_Tmid_landscape_plotfn,
                'ax_id': f'plot_{i}',
                'Amps' : self.proc_data_dict['Amps'][i], 
                'Tmid' : self.proc_data_dict['Tmid'], 
                'CP' : self.proc_data_dict[f'CP_{i}'],
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'q0' : self.Q0[i], 'q1' : self.Q1[i],
                'ts' : self.timestamp, 'n': i,
                'title' : f'Qubits {" ".join(self.Q0)}, {" ".join(self.Q1)}',
            }

        for i, q0 in enumerate(self.Q0):
            self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'] = plt.figure(figsize=(9,4), dpi=100)
            # self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].patch.set_alpha(0)
            axs = [self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].add_subplot(121),
                   self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].add_subplot(122)]

            self.axs_dict[f'conditional_phase_{i}'] = axs[0]
            self.axs_dict[f'missing_fraction_{i}'] = axs[0]

            self.plot_dicts[f'VCZ_landscape_{self.Q0[i]}_{self.Q1[i]}']={
                'plotfn': VCZ_Tmid_landscape_plotfn,
                'ax_id': f'conditional_phase_{i}',
                'Amps' : self.proc_data_dict['Amps'][i], 
                'Tmid' : self.proc_data_dict['Tmid'], 
                'CP' : self.proc_data_dict[f'CP_{i}'], 
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'q0' : self.Q0[i], 'q1' : self.Q1[i],
                'ts' : self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def VCZ_Tmid_landscape_plotfn(
    ax, 
    Amps, Tmid, 
    CP, MF, 
    q0, q1,
    ts, n=0,
    title=None, **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()

    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        if rang:
            X = X/np.max(vals) * (rang[1]-rang[0]) + rang[0]
        return X

    X = get_plot_axis(Amps)
    Y = get_plot_axis(Tmid)
    a1 = axs[0+2*n].pcolormesh(X, Y, CP, cmap=hsluv_anglemap45, vmin=0, vmax=360)
    fig.colorbar(a1, ax=axs[0+2*n], label='conditional phase', ticks=[0, 90, 180, 270, 360])
    a2 = axs[1+2*n].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a2, ax=axs[1+2*n], label='missing fraction')

    def get_contours(cphase, phase):
        n = len(cphase)
        x = []
        y = np.arange(n)
        for i in range(n):
            x.append(np.argmin(abs(cphase[i]-phase)))
        dx = np.array(x)[1:]-np.array(x)[:-1]
        k = 0
        contours = {'0': {'x':[x[0]], 'y':[0]}}
        for i, s in enumerate(dx):
            if s > 0:
                contours[f'{k}']['x'].append(x[i+1])
                contours[f'{k}']['y'].append(i+1)
            else:
                k += 1
                contours[f'{k}'] = {'x':[x[i+1]], 'y':[i+1]}
        return contours
    CT = get_contours(CP, phase=180)
    for c in CT.values():
        c['x'] = Amps[c['x']]
        c['y'] = Tmid[c['y']]
        axs[1+2*n].plot(c['x'], c['y'], marker='', ls='--', color='white')

    for i in range(2):
        axs[i+2*n].set_xlabel('Amplitude')
        axs[i+2*n].set_ylabel(r'$\tau_\mathrm{mid}$')
    axs[0+2*n].set_title(f'Conditional phase')
    axs[1+2*n].set_title(f'Missing fraction')

    if title:
        fig.suptitle(title+'\n'+ts, y=1.01)
        axs[0+2*n].set_title(f'Conditional phase {q0} {q1}')
        axs[1+2*n].set_title(f'Missing fraction {q0} {q1}')
    else:
        fig.suptitle(f'Qubits {q0} {q1}\n'+ts, y=1.02)
        axs[0].set_title(f'Conditional phase')
        axs[1].set_title(f'Missing fraction')

    fig.tight_layout()



class VCZ_B_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q0,
                 Q1,
                 A_ranges,
                 directions,
                 Q_parks: str = None,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q0 = Q0
        self.Q1 = Q1
        self.Q_parks = Q_parks
        self.ranges = A_ranges
        self.directions = directions
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.qoi = {}
        Amps_idxs = np.unique(self.raw_data_dict['data'][:,0])
        Bamps = np.unique(self.raw_data_dict['data'][:,1])
        nx, ny = len(Amps_idxs), len(Bamps)
        Amps_list = [ np.linspace(r[0], r[1], nx) for r in self.ranges ] 

        self.proc_data_dict['Amps'] = Amps_list
        self.proc_data_dict['Bamps'] = Bamps

        def cost_function(CP, MF,
                          phase=180,
                          cp_coef=1, l1_coef=1):
            A = ((np.abs(CP)-180)/180)**2
            B = ((MF-np.min(MF))/.5)**2
            C = (np.mean(MF-np.min(MF), axis=0)/.5)**2
            return cp_coef*A + l1_coef*(B+C)

        for i, q0 in enumerate(self.Q0):
            CP = self.raw_data_dict['data'][:,2*i+2].reshape(nx, ny)
            MF = self.raw_data_dict['data'][:,2*i+3].reshape(nx, ny)
            CF = cost_function(CP, MF)

            idxs_min = np.unravel_index(np.argmin(CF), CF.shape)
            A_min, B_min = Amps_list[i][idxs_min[1]], Bamps[idxs_min[0]]
            CP_min, L1_min = CP[idxs_min], MF[idxs_min]/2

            self.proc_data_dict[f'CP_{i}'] = CP
            self.proc_data_dict[f'MF_{i}'] = MF
            self.proc_data_dict[f'CF_{i}'] = CF
            self.qoi[f'Optimal_amps_{q0}'] = A_min, B_min
            self.qoi[f'Gate_perf_{q0}'] = CP_min, L1_min

    def prepare_plots(self):
        self.axs_dict = {}

        n = len(self.Q0)
        self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'] = plt.figure(figsize=(15,4*n), dpi=100)
        # self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].patch.set_alpha(0)
        axs = []
        for i, q0 in enumerate(self.Q0):
            axs.append(self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].add_subplot(n,3,3*i+1))
            axs.append(self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].add_subplot(n,3,3*i+2))
            axs.append(self.figs[f'VCZ_landscape_{self.Q0}_{self.Q1}'].add_subplot(n,3,3*i+3))

            self.axs_dict[f'plot_{i}'] = axs[0]

            self.plot_dicts[f'VCZ_landscape_{self.Q0}_{self.Q1}_{i}']={
                'plotfn': VCZ_B_landscape_plotfn,
                'ax_id': f'plot_{i}',
                'Amps' : self.proc_data_dict['Amps'][i], 
                'Bamps' : self.proc_data_dict['Bamps'], 
                'CP' : self.proc_data_dict[f'CP_{i}'],
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'CF' : self.proc_data_dict[f'CF_{i}'],
                'q0' : self.Q0[i], 'q1' : self.Q1[i],
                'opt' : self.qoi[f'Optimal_amps_{q0}'],
                'ts' : self.timestamp,
                'n': i,
                'direction' : self.directions[i][0],
                'title' : f'Qubits {" ".join(self.Q0)}, {" ".join(self.Q1)}',
                'gate_perf' : self.qoi[f'Gate_perf_{q0}']
            }

        for i, q0 in enumerate(self.Q0):
            self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'] = plt.figure(figsize=(15,4), dpi=100)
            # self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].patch.set_alpha(0)
            axs = [self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].add_subplot(131),
                   self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].add_subplot(132),
                   self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'].add_subplot(133)]
            self.axs_dict[f'conditional_phase_{i}'] = axs[0]

            self.plot_dicts[f'VCZ_landscape_{self.Q0[i]}_{self.Q1[i]}']={
                'plotfn': VCZ_B_landscape_plotfn,
                'ax_id': f'conditional_phase_{i}',
                'Amps' : self.proc_data_dict['Amps'][i], 
                'Bamps' : self.proc_data_dict['Bamps'], 
                'CP' : self.proc_data_dict[f'CP_{i}'],
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'CF' : self.proc_data_dict[f'CF_{i}'],
                'q0' : self.Q0[i], 'q1' : self.Q1[i],
                'opt' : self.qoi[f'Optimal_amps_{q0}'],
                'ts' : self.timestamp,
                'gate_perf' : self.qoi[f'Gate_perf_{q0}']
            }

            self.figs[f'VCZ_Leakage_contour_{q0}_{self.Q1[i]}'] = plt.figure(figsize=(9,4), dpi=100)
            # self.figs[f'VCZ_Leakage_contour_{q0}_{self.Q1[i]}'].patch.set_alpha(0)
            axs = [self.figs[f'VCZ_Leakage_contour_{q0}_{self.Q1[i]}'].add_subplot(121),
                   self.figs[f'VCZ_Leakage_contour_{q0}_{self.Q1[i]}'].add_subplot(122)]
            self.axs_dict[f'contour_{i}'] = axs[0]

            self.plot_dicts[f'VCZ_Leakage_contour_{q0}_{self.Q1[i]}']={
                'plotfn': VCZ_L1_contour_plotfn,
                'ax_id': f'contour_{i}',
                'Amps' : self.proc_data_dict['Amps'][i], 
                'Bamps' : self.proc_data_dict['Bamps'], 
                'CP' : self.proc_data_dict[f'CP_{i}'],
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'CF' : self.proc_data_dict[f'CF_{i}'],
                'q0' : self.Q0[i],
                'q1' : self.Q1[i],
                'opt' : self.qoi[f'Optimal_amps_{q0}'],
                'ts' : self.timestamp,
                'gate_perf' : self.qoi[f'Gate_perf_{q0}'],
                'direction' : self.directions[i][0]
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def VCZ_B_landscape_plotfn(
    ax,
    Amps, Bamps,
    CP, MF, CF,
    q0, q1, ts,
    gate_perf,
    opt, direction=None,
    n=0, title=None, **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()

    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        if rang:
            X = X/np.max(vals) * (rang[1]-rang[0]) + rang[0]
        return X

    X = get_plot_axis(Amps)
    Y = get_plot_axis(Bamps)
    a1 = axs[0+3*n].pcolormesh(X, Y, CP, cmap=hsluv_anglemap45, vmin=0, vmax=360)
    fig.colorbar(a1, ax=axs[0+3*n], label='conditional phase', ticks=[0, 90, 180, 270, 360])
    a2 = axs[1+3*n].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a2, ax=axs[1+3*n], label='missing fraction')
    a3 = axs[2+3*n].pcolormesh(X, Y, CF, cmap='viridis',
                   norm=LogNorm(vmin=CF.min(), vmax=CF.max()))
    fig.colorbar(a3, ax=axs[2+3*n], label='cost function')

    text_str = 'Optimal parameters\n'+\
               f'gate: {q0} CZ_{direction}\n'+\
               f'$\phi$: {gate_perf[0]:.2f} \t $L_1$: {gate_perf[1]*100:.1f}%\n'+\
               f'A amp: {opt[0]:.4f}\n'+\
               f'B amp: {opt[1]:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[2+3*n].text(1.35, 0.98, text_str, transform=axs[2+3*n].transAxes, fontsize=10,
            verticalalignment='top', bbox=props, linespacing=1.6)

    ##########################
    # Calculate contours
    ##########################
    # unwrap phase so contour is correctly estimated
    AUX = np.zeros(CP.shape)
    for i in range(len(CP)):
        AUX[i] = np.deg2rad(CP[i])*1
        AUX[i] = np.unwrap(AUX[i])
        AUX[i] = np.rad2deg(AUX[i])
    for i in range(len(CP[:,i])):
        AUX[:,i] = np.deg2rad(AUX[:,i])
        AUX[:,i] = np.unwrap(AUX[:,i])
        AUX[:,i] = np.rad2deg(AUX[:,i])
    cs = axs[1+3*n].contour(Amps, Bamps, AUX, levels=[180, 180+360],
                        colors='white', linestyles='--')
    axs[1+3*n].clabel(cs, inline=True, fontsize=10, fmt='$180^o$')

    for i in range(3):
        axs[i+3*n].plot(opt[0], opt[1], 'o', mfc='white', mec='grey', mew=.5)
        axs[i+3*n].set_xlabel('Amplitude')
        axs[i+3*n].set_ylabel(r'B amplitude')
    if title:
        fig.suptitle(title+'\n'+ts, y=1)
        axs[0+3*n].set_title(f'Conditional phase {q0} {q1}')
        axs[1+3*n].set_title(f'Missing fraction {q0} {q1}')
        axs[2+3*n].set_title(f'Cost function {q0} {q1}')
    else:
        fig.suptitle(f'Qubits {q0} {q1}\n'+ts, y=1)
        axs[0].set_title(f'Conditional phase')
        axs[1].set_title(f'Missing fraction')
        axs[2].set_title(f'Cost function')
    fig.tight_layout()

def VCZ_L1_contour_plotfn(
    ax,
    Amps, Bamps,
    CP, MF, CF,
    q0, q1, ts,
    gate_perf,
    opt, direction=None,
    title=None, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        if rang:
            X = X/np.max(vals) * (rang[1]-rang[0]) + rang[0]
        return X
    def get_contour_idxs(CP):
        phase = 180
        if np.mean(CP) > 300:
            phase += 360
        idxs_i = []
        idxs_j = []
        for i in range(len(CP)):
            idx = np.argmin(np.abs(CP[:,i]-phase))
            if np.abs(CP[idx, i]-phase) < 10:
                idxs_i.append(i)
                idxs_j.append(idx)
        return idxs_i, idxs_j
    ##########################
    # Calculate contours
    ##########################
    # unwrap phase so contour is correctly estimated
    AUX = np.zeros(CP.shape)
    for i in range(len(CP)):
        AUX[i] = np.deg2rad(CP[i])*1
        AUX[i] = np.unwrap(AUX[i])
        AUX[i] = np.rad2deg(AUX[i])
    for i in range(len(CP[:,i])):
        AUX[:,i] = np.deg2rad(AUX[:,i])
        AUX[:,i] = np.unwrap(AUX[:,i])
        AUX[:,i] = np.rad2deg(AUX[:,i])
    idxs = get_contour_idxs(AUX)
    
    X = get_plot_axis(Amps)
    Y = get_plot_axis(Bamps)
    a1 = axs[0].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a1, ax=axs[0], label='missing fraction')
    cs = axs[0].contour(Amps, Bamps, AUX, levels=[180, 180+360, 180+720],
                        colors='white', linestyles='--')
    axs[0].clabel(cs, inline=True, fontsize=10, fmt='$180^o$')
    
    axs[1].axvline(opt[0], color='k', ls='--', alpha=.5)
    axs[1].plot(Amps[idxs[0]], MF[idxs][::-1]/2*100)
    
    axs[0].plot(opt[0], opt[1], 'o', mfc='white', mec='grey', mew=.5)
    axs[0].set_xlabel('Amplitude')
    axs[0].set_ylabel(r'B amplitude')
    
    axs[1].set_xlabel('Amplitude')
    axs[1].set_ylabel(r'$L_1$ (%)')

    fig.suptitle(f'Qubits {q0} {q1}\n'+ts, y=.95)
    axs[0].set_title(f'Missing fraction')
    axs[1].set_title(f'$L_1$ along contour')
    fig.tight_layout()


class Parity_check_ramsey_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q_target,
                 Q_control,
                 control_cases,
                 angles,
                 solve_for_phase_gate_model:bool = False,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q_target = Q_target
        self.Q_control = Q_control
        self.control_cases = control_cases
        self.angles = angles
        self.solve_for_phase_gate_model = solve_for_phase_gate_model
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.qoi = {}
        # Processing
        n = len(self.Q_target+self.Q_control)
        detector_list = [ name.decode().split(' ')[-1] for name in 
                          self.raw_data_dict['value_names']]
        calibration_points = ['{:0{}b}'.format(i, n) for i in range(2**n)]
        self.calibration_points = calibration_points
        Ramsey_curves = {}
        Cal_points = {}
        for k, q in enumerate(self.Q_target+self.Q_control):
            # Sort raw data
            q_idx = detector_list.index(q)
            Cal_points_raw = self.raw_data_dict['data'][-len(calibration_points):,1+q_idx]
            Ramsey_curves_raw = { case : self.raw_data_dict['data'][i*len(self.angles):(i+1)*len(self.angles),1+q_idx]\
                                  for i, case in enumerate(self.control_cases) }
            # Sort and calculate calibration point levels
            selector = np.tile(np.concatenate([np.zeros(2**(n-k-1)),
                               np.ones(2**(n-k-1))]), 2**(k))
            Cal_0 = np.mean(Cal_points_raw[~np.ma.make_mask(selector)])
            Cal_1 = np.mean(Cal_points_raw[np.ma.make_mask(selector)])
            # Convert to probability
            Cal_points[q] = (Cal_points_raw-Cal_0)/(Cal_1-Cal_0)
            Ramsey_curves[q] = { case : (Ramsey_curves_raw[case]-Cal_0)/(Cal_1-Cal_0)\
                                 for case in self.control_cases }

        # Fit phases
        from scipy.optimize import curve_fit
        def func(x, phi, A, B):
            return A*(np.cos( (x+phi)/360 *2*np.pi )+1)/2 + B
        Fit_res = { q : {} for q in self.Q_target}
        for q in self.Q_target:
            for case in self.control_cases:
                # print(Ramsey_curves[q][case])
                popt, pcov = curve_fit(func, self.angles, Ramsey_curves[q][case],
                                       p0 = [90, .9, 0],
                                       bounds=[(-100, 0, -np.inf), (300, np.inf, np.inf)])
                Fit_res[q][case] = popt

        # Missing fraction
        P_excited = {}
        Missing_fraction = {}
        L_0 = {}
        L_1 = {}
        n_c = len(self.Q_control)
        for i, q in enumerate(self.Q_control):
            P_excited[q] = { case : np.mean(Ramsey_curves[q][case]) for case in self.control_cases }
            L_0[q] = []
            L_1[q] = []
            for case in self.control_cases:
                if case[i] == '0':
                    L_0[q].append( P_excited[q][case] )
                elif case[i] == '1':
                    L_1[q].append( P_excited[q][case] )
                else:
                    raise(f'Control case {case} not valid.')
            L_0[q] = np.mean(L_0[q])
            L_1[q] = np.mean(L_1[q])
            Missing_fraction[q] = L_1[q]-L_0[q]

        # Solve for Phase gate model
        if self.solve_for_phase_gate_model:
            q = self.Q_target[0]
            n_c = len(self.Q_control)
            Phase_vec = np.array([Fit_res[q][c][0] for c in self.control_cases])
            Phase_model = get_phase_model_values(n_c, Phase_vec)

        self.proc_data_dict['Ramsey_curves'] = Ramsey_curves
        self.proc_data_dict['Cal_points'] = Cal_points
        self.proc_data_dict['Fit_res'] = Fit_res
        self.proc_data_dict['P_excited'] = P_excited
        self.proc_data_dict['L_0'] = L_0
        self.proc_data_dict['L_1'] = L_1
        self.proc_data_dict['Missing_fraction'] = Missing_fraction

        self.qoi['Missing_fraction'] = Missing_fraction
        # self.qoi['L_0'] = L_0
        # self.qoi['L_1'] = L_1
        self.qoi['P_excited'] = P_excited
        self.qoi['Phases'] = {}
        q = self.Q_target[0]
        self.qoi['Phases'][q] = { c:Fit_res[q][c][0] for c in self.control_cases }
        if self.solve_for_phase_gate_model:
            self.qoi['Phase_model'] = Phase_model

    def prepare_plots(self):
        self.axs_dict = {}

        Q_total = self.Q_target+self.Q_control
        n = len(Q_total)
        fig, axs = plt.subplots(figsize=(7,2*n), nrows=n, sharex=True, dpi=100)
        self.figs[f'Parity_check_Ramsey_{"_".join(Q_total)}'] = fig
        self.axs_dict[f'plot_1'] = axs[0]
        # fig.patch.set_alpha(0)

        self.plot_dicts[f'Parity_check_Ramsey_{"_".join(Q_total)}']={
                'plotfn': Ramsey_curves_plotfn,
                'ax_id': f'plot_1',
                'Q_target': self.Q_target,
                'Q_control': self.Q_control,
                'angles': self.angles,
                'calibration_points': self.calibration_points,
                'control_cases': self.control_cases,
                'Ramsey_curves': self.proc_data_dict['Ramsey_curves'],
                'Cal_points': self.proc_data_dict['Cal_points'],
                'Fit_res': self.proc_data_dict['Fit_res'],
                'L_0': self.proc_data_dict['L_0'],
                'L_1': self.proc_data_dict['L_1'],
                'Missing_fraction': self.proc_data_dict['Missing_fraction'],
                'timestamp': self.timestamps[0]}

        fig, axs = plt.subplots(figsize=(9,4), ncols=2, dpi=100)
        self.figs[f'Parity_check_phases_{"_".join(Q_total)}'] = fig
        self.axs_dict[f'plot_2'] = axs[0]

        self.plot_dicts[f'Parity_check_phases_{"_".join(Q_total)}']={
                'plotfn': Phases_plotfn,
                'ax_id': f'plot_2',
                'Q_target': self.Q_target,
                'Q_control': self.Q_control,
                'control_cases': self.control_cases,
                'Phases': self.qoi['Phases'],
                'timestamp': self.timestamps[0]}

        n = len(self.Q_control)
        fig, axs = plt.subplots(figsize=(5,2*n), nrows=n, sharex=True, dpi=100)
        if type(axs) != np.ndarray:
            axs = [axs]
        self.figs[f'Parity_check_missing_fraction_{"_".join(Q_total)}'] = fig
        self.axs_dict[f'plot_3'] = axs[0]

        self.plot_dicts[f'Parity_check_missing_fraction_{"_".join(Q_total)}']={
                'plotfn': Missing_fraction_plotfn,
                'ax_id': f'plot_3',
                'Q_target': self.Q_target,
                'Q_control': self.Q_control,
                'P_excited': self.proc_data_dict['P_excited'],
                'control_cases': self.control_cases,
                'timestamp': self.timestamps[0]}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Ramsey_curves_plotfn(
    ax, 
    Q_target,
    Q_control,
    angles,
    calibration_points,
    control_cases,
    Ramsey_curves,
    Cal_points,
    Fit_res,
    L_0,
    L_1,
    Missing_fraction,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    def func(x, phi, A, B):
        return A*(np.cos( (x+phi)/360 *2*np.pi )+1)/2 + B

    cal_ax = np.arange(len(calibration_points))*10+360
    from matplotlib.cm import hsv 
    Colors = { case : hsv(x) for x, case in \
               zip(np.linspace(0,1,len(control_cases)), control_cases)}
    for i, q in enumerate(Q_target+Q_control):
        for case in control_cases:
            if q in Q_target:
                axs[i].plot(angles, func(angles, *Fit_res[q][case]),
                            '--', color=Colors[case], alpha=.5,
                            label=rf'$|{case}\rangle$ : {Fit_res[q][case][0]:.1f}')
            axs[i].plot(angles, Ramsey_curves[q][case],
                        '.', color=Colors[case], alpha=.5)
        axs[i].plot(cal_ax, Cal_points[q], 'C0.-')
        if q in Q_control:
            axs[i].plot([angles[0], angles[-1]], [L_0[q], L_0[q]], 'k--')
            axs[i].plot([angles[0], angles[-1]], [L_1[q], L_1[q]], 'k--',
                        label = f'Missing fac. : {Missing_fraction[q]*100:.1f} %')
            axs[i].legend(loc=2, frameon=False)
        axs[i].set_ylabel(f'Population {q}')
    axs[-1].set_xticks(np.arange(0, 360, 60))
    axs[-1].set_xlabel('Phase (deg), calibration points')
    axs[0].legend(frameon=False, bbox_to_anchor=(1.04,1), loc="upper left")
    axs[0].set_title(f'{timestamp}\nParity check ramsey '+\
                     f'{" ".join(Q_target)} with control qubits {" ".join(Q_control)}')

def Phases_plotfn(
    ax,
    Q_target,
    Q_control,
    control_cases, 
    Phases,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Sort control cases by number of excitations
    # and get ideal phases vector "vec"
    cases_sorted = [control_cases[0]]
    vec = [0]
    for n in range(len(control_cases[0])):
        for c in control_cases:
            if c.count('1') == n+1:
                cases_sorted.append(c)
                vec.append(180*np.mod(n+1%2,2))
    # Phase error vector
    q = Q_target[0]
    phase_err_sorted = np.array([Phases[q][c] for c in cases_sorted])-np.array(vec)

    axs[0].plot(cases_sorted, np.zeros(len(cases_sorted))+180, 'k--')
    axs[0].plot(cases_sorted, np.zeros(len(cases_sorted)), 'k--')
    axs[0].plot(cases_sorted, [Phases[q][c] for c in cases_sorted], 'o-')
    axs[0].set_xticklabels([fr'$|{c}\rangle$' for c in cases_sorted], rotation=90)
    axs[0].set_yticks([0, 45, 90, 135, 180])
    axs[0].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
    axs[0].set_ylabel(f'{"".join(Q_target)} Phase (deg)')
    axs[0].grid(ls='--')
    
    axs[1].bar(cases_sorted, phase_err_sorted)
    axs[1].set_xticklabels([fr'$|{c}\rangle$' for c in cases_sorted], rotation=90)
    axs[1].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
    axs[1].set_ylabel(f'{"".join(Q_target)} Phase error (deg)')
    fig.suptitle(f'{timestamp}\nParity check ramsey '+\
                 f'{" ".join(Q_target)} with control qubits {" ".join(Q_control)}', y=1.1)
    fig.tight_layout()


def Missing_fraction_plotfn(
    ax,
    Q_target,
    Q_control,
    P_excited,
    control_cases,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    for i, q in enumerate(Q_control):    
        axs[i].plot([P_excited[q][case]*100 for case in control_cases], 'C0o-')

        axs[i].grid(ls='--')
        axs[i].set_xticks(np.arange(len(control_cases)))
        axs[i].set_xticklabels([fr'$|{c}\rangle$' for c in control_cases], rotation=90)
        
        axs[-1].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
        axs[i].set_ylabel(f'$P_\{"mathrm{exc}"}$ {q} (%)')

    axs[0].set_title(f'{timestamp}\nParity check ramsey '+\
                     f'{" ".join(Q_target)} with control qubits {" ".join(Q_control)}')

def get_phase_model_values(n, Phase_vec):
    # Get Operator matrix dictionary
    I = np.array([[1, 0],
                  [0, 1]])
    Z = np.array([[1, 0],
                  [0,-1]])
    Operators = {}
    for s in ['{:0{}b}'.format(i, n) for i in range(2**n)]:
        op_string = ''
        op_matrix = 1
        for i in s:
            if i == '0':
                op_string += 'I'
                op_matrix = np.kron(op_matrix, I)
            else:
                op_string += 'Z'
                op_matrix = np.kron(op_matrix, Z)
        Operators[op_string] = op_matrix 
    # Calculate M matrix
    M = np.zeros((2**n,2**n))
    for i, Op in enumerate(Operators.values()):
        for j in range(2**n):
            # create state vector
            state = np.zeros((1,2**n))
            state[0][j] = 1
            M[i, j] = np.dot(state, np.dot(Op, state.T))
    # Get ideal phase vector
    states = ['{:0{}b}'.format(i, n) for i in range(2**n)]
    Phase_vec_ideal = np.array([s.count('1')*180 for s in states])
    ########################################
    # Correct rotations for modulo of phase
    ########################################
    state_idxs_sorted_by_exc = {i:[] for i in range(n+1)}
    for i, s in enumerate(states):
        nr_exc = s.count('1')
        state_idxs_sorted_by_exc[nr_exc].append(i)
    for i in range(n):
        phi_0 = Phase_vec[state_idxs_sorted_by_exc[i][0]]
        for idx in state_idxs_sorted_by_exc[i+1]:
            while Phase_vec[idx] < phi_0:
                Phase_vec[idx] += 360
    # Calculate Phase gate model coefficients
    M_inv = np.linalg.inv(M)
    vector_ideal = np.dot(M_inv, Phase_vec_ideal)
    vector = np.dot(M_inv, Phase_vec)

    Result = {op:vector[i]-vector_ideal[i] for i, op in enumerate(Operators.keys())}

    return Result

class Parity_check_calibration_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q_ancilla: list,
                 Q_control: list,
                 Q_pair_target: list,
                 B_amps: list,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q_ancilla = Q_ancilla
        self.Q_control = Q_control
        self.Q_pair_target = Q_pair_target
        self.B_amps = B_amps
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.qoi = {}

        n_c = len(self.Q_control)
        Operators = [ name.decode()[-n_c:] for name in self.raw_data_dict['value_names']\
                      if 'Phase_model' in name.decode() ]
        Phases = self.raw_data_dict['data'][:,1:-n_c]
        for q in self.Q_pair_target:
            if q in self.Q_control:
                q_idx = self.Q_control.index(q)
        # Sort phases
        Operators_sorted = []
        idx_sorted = []
        for n in range(len(Operators)+1):
            for i, op in enumerate(Operators):
                if op.count('Z') == n:
                    Operators_sorted.append(op)
                    idx_sorted.append(i)
        Phases_sorted = Phases[:,idx_sorted]
        # Fit linear curves to two body term
        Two_body_phases = Phases_sorted[:,n_c-q_idx]
        Single_body_phases = Phases_sorted[:,0]
        from scipy.optimize import curve_fit
        def func(x, A, B):
            return A*x+B
        popt, pcov = curve_fit(func, self.B_amps, Two_body_phases)
        Opt_B = -popt[1]/popt[0]
        # Fit single body phase
        popt_0, pcov_0 = curve_fit(func, self.B_amps, Single_body_phases)
        Phase_offset = func(Opt_B, *popt_0)
        # Get Missing fraction relevant
        Missing_fraction = self.raw_data_dict['data'][:,q_idx-n_c]

        self.proc_data_dict['Phases'] = Phases_sorted
        self.proc_data_dict['Operators'] = Operators_sorted
        self.proc_data_dict['Two_body_phases'] = Two_body_phases
        self.proc_data_dict['Missing_fraction'] = Missing_fraction
        self.proc_data_dict['Fit_res'] = popt

        self.qoi['Optimal_B'] = Opt_B
        self.qoi['Phase_offset'] = Phase_offset

    def prepare_plots(self):
        self.axs_dict = {}
        fig = plt.figure(figsize=(10,4))
        axs = [fig.add_subplot(121),
               fig.add_subplot(222),
               fig.add_subplot(224)]
        self.figs[f'Parity_check_calibration_{"_".join(self.Q_pair_target)}'] = fig
        self.axs_dict['plot_1'] = axs[0]
        # fig.patch.set_alpha(0)
        self.plot_dicts[f'Parity_check_calibration_{"_".join(self.Q_pair_target)}']={
                'plotfn': gate_calibration_plotfn,
                'ax_id': 'plot_1',
                'B_amps': self.B_amps,
                'Phases': self.proc_data_dict['Phases'],
                'Operators': self.proc_data_dict['Operators'],
                'Q_control': self.Q_control,
                'Q_pair_target': self.Q_pair_target,
                'Two_body_phases': self.proc_data_dict['Two_body_phases'],
                'Missing_fraction': self.proc_data_dict['Missing_fraction'],
                'Fit_res': self.proc_data_dict['Fit_res'],
                'Opt_B': self.qoi['Optimal_B'],
                'timestamp': self.timestamps[0]}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def gate_calibration_plotfn(
    ax,
    B_amps,
    Phases,
    Operators,
    Q_control,
    Q_pair_target,
    Two_body_phases,
    Missing_fraction,
    Opt_B,
    Fit_res,
    timestamp,
    **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()

    def func(x, A, B):
            return A*x+B
    from matplotlib.cm import viridis
    Colors = [ viridis(x) for x in np.linspace(0, 1, len(B_amps)) ]

    for i, b_amp in enumerate(B_amps):
        axs[0].plot(Phases[i], color=Colors[i], marker='o')
    axs[0].grid(ls='--')
    axs[0].set_xticks(np.arange(len(Operators)))
    axs[0].set_xticklabels(Operators, rotation=90)
    axs[0].set_xlabel(f'Operators (${"".join(["U_{"+s+"} " for s in Q_control[::1]])}$)')
    axs[0].set_ylabel('Phase error (deg)')

    axs[1].plot(B_amps, func(B_amps, *Fit_res), 'C0--')
    axs[1].plot(B_amps, Two_body_phases, 'C0o')
    axs[1].axhline(0, ls='--', color='k', alpha=.5)
    axs[1].axvline(Opt_B, ls='--', color='k', alpha=.5, label=f'Optimal B : {Opt_B:.3f}')
    axs[1].legend(frameon=False)
    axs[1].set_ylabel('Phase error (deg)')
    axs[1].set_xticks(B_amps)
    axs[1].set_xticklabels([])

    axs[2].plot(B_amps, Missing_fraction*100, 'C2o-')
    axs[2].set_xticks(B_amps)
    axs[2].set_xlabel('B values')
    axs[2].set_ylabel('Missing fraction (%)')

    fig.suptitle(f'{timestamp}\nParity check phase calibration gate {" ".join(Q_pair_target)}', y=1.01)
    fig.tight_layout()


class Parity_check_park_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q_park_target: str,
                 Q_spectator: str,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):
        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q_park_target = Q_park_target
        self.Q_spectator = Q_spectator
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names'),
                      'poly_coefs': (f'Instrument settings/flux_lm_{self.Q_park_target}',
                                     'attr:q_polycoeffs_freq_01_det'),
                      'channel_range': (f'Instrument settings/flux_lm_{self.Q_park_target}',
                                        'attr:cfg_awg_channel_range'),
                      'channel_amp': (f'Instrument settings/flux_lm_{self.Q_park_target}',
                                      'attr:cfg_awg_channel_amplitude')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.qoi = {}
        # Sort data
        Amps = self.raw_data_dict['data'][:,0]
        L0 = self.raw_data_dict['data'][:,1]
        L1 = self.raw_data_dict['data'][:,2]
        # Calculate frequency axis
        poly_coefs = [float(n) for n in self.raw_data_dict['poly_coefs'][1:-1].split(' ') if n != '' ]
        channel_range = float(self.raw_data_dict['channel_range'])
        channel_amp = float(self.raw_data_dict['channel_amp'])
        Freqs = np.poly1d(poly_coefs)(Amps*channel_amp*channel_range/2)
        # Calculate optimal parking amp
        idx_opt = np.argmin(L0+L1)
        Amp_opt = Amps[idx_opt]
        Freq_opt = Freqs[idx_opt]
        # Save stuff
        self.proc_data_dict['L0'] = L0
        self.proc_data_dict['L1'] = L1
        self.proc_data_dict['Amps'] = Amps
        self.proc_data_dict['Freqs'] = Freqs
        self.qoi['Freq_opt'] = Freq_opt
        self.qoi['Amp_opt'] = Amp_opt

    def prepare_plots(self):
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(6,6), nrows=2, dpi=100)
        self.figs[f'Park_amplitude_sweep_{self.Q_park_target}'] = fig
        self.axs_dict['plot_1'] = axs[0]
        # fig.patch.set_alpha(0)
        self.plot_dicts[f'Park_amplitude_sweep_{self.Q_park_target}']={
                'plotfn': park_sweep_plotfn,
                'ax_id': 'plot_1',
                'Amps' : self.proc_data_dict['Amps'],
                'Freqs' : self.proc_data_dict['Freqs'],
                'L0' : self.proc_data_dict['L0'],
                'L1' : self.proc_data_dict['L1'],
                'Amp_opt' : self.qoi['Amp_opt'],
                'Freq_opt' : self.qoi['Freq_opt'],
                'qubit' : self.Q_park_target,
                'q_spec' : self.Q_spectator,
                'timestamp': self.timestamps[0]}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def park_sweep_plotfn(
    ax,
    Amps,
    Freqs,
    L0,
    L1,
    Amp_opt,
    Freq_opt,
    qubit,
    q_spec,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    qubit_string = '{'+qubit[0]+'_'+qubit[1]+'}'
    spec_string = '{'+q_spec[0]+'_'+q_spec[1]+'}'
    q2_string = '{'+q_spec[0]+'_'+q_spec[1]+qubit[0]+'_'+qubit[1]+'}'
    axs[0].plot(Freqs*1e-6, L0, 'C0.-', label=rf'$|01\rangle_\mathrm{q2_string}$')
    axs[0].plot(Freqs*1e-6, L1, 'C3.-', label=rf'$|11\rangle_\mathrm{q2_string}$')
    axs[0].axvline(Freq_opt*1e-6, color='k', ls='--', lw=1, label='Opt. freq.')
    axs[0].set_xlabel(f'Parking frequency $\mathrm{qubit_string}$ (MHz)')
    axs[0].set_ylabel(rf'Qubit $\mathrm{qubit_string}$'+' $\mathrm{P_{excited}}$')
    axs[0].set_title(f'{timestamp}\nParking amplitude sweep $\mathrm{qubit_string}$ '+\
                     f'with spectator $\mathrm{spec_string}$')

    axs[1].plot(Amps, L0, 'C0.-', label=rf'$|01\rangle_\mathrm{q2_string}$')
    axs[1].plot(Amps, L1, 'C3.-', label=rf'$|11\rangle_\mathrm{q2_string}$')
    axs[1].axvline(Amp_opt, color='k', ls='--', lw=1, label='Opt. amp.')
    axs[1].set_xlabel(f'Parking Amplitude $\mathrm{qubit_string}$ (MHz)')
    axs[1].set_ylabel(rf'Qubit $\mathrm{qubit_string}$'+' $\mathrm{P_{excited}}$')

    fig.tight_layout()
    axs[0].legend(frameon=False, bbox_to_anchor=(1.02, 1))
    axs[1].legend(frameon=False, bbox_to_anchor=(1.02, 1))


class Parity_check_fidelity_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q_ancilla: str,
                 Q_control: list,
                 control_cases: list,
                 post_selection: bool,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True):
        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q_ancilla = Q_ancilla
        self.Q_control = Q_control
        self.control_cases = control_cases
        self.post_selection = post_selection
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        qubit_list = [self.Q_ancilla] + self.Q_control
        _data = {'data': ('Experimental Data/Data', 'dset'),
                 'value_names': ('Experimental Data', 'attr:value_names')}
        _thrs = {f'threshold_{q}': (f'Instrument settings/{q}', 'attr:ro_acq_threshold')
                 for q in qubit_list}
        param_spec = {**_data, **_thrs}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
                             data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.qoi = {}
        # Process data
        qubit_list = [self.Q_ancilla]
        if self.post_selection:
            qubit_list.append(self.Q_control)

        nr_cases = len(self.control_cases)
        nr_shots_per_case = len(self.raw_data_dict['data'][:,0])//nr_cases//(1+self.post_selection)
        Threshold = { q : float(self.raw_data_dict[f'threshold_{q}'])
                      for q in qubit_list}
        Shots_raw = {}
        Shots_dig = { q: {} for q in qubit_list }
        PS_mask = { case : np.ones(nr_shots_per_case)
                    for case in self.control_cases }
        for i, q in enumerate(qubit_list):
            shots_raw = self.raw_data_dict['data'][:,i+1]
            shots_dig = np.array([ 0 if s<Threshold[q] else 1
                                   for s in shots_raw ])
            Shots_raw[q] = shots_raw
            for j, case in enumerate(self.control_cases):
                if self.post_selection:
                    PS_mask[case] *= np.array([1 if s == 0 else np.nan for s
                                               in shots_dig[2*j::2*nr_cases] ])
                    Shots_dig[q][case] = shots_dig[2*j+1::2*nr_cases]
                else:
                    Shots_dig[q][case] = shots_dig[j::nr_cases]
        # Apply post selection
        if self.post_selection:
            ps_fraction = 0
            for case in self.control_cases:
                for q in qubit_list:
                    Shots_dig[q][case] = Shots_dig[q][case][~np.isnan(PS_mask[case])]
                ps_fraction += np.sum(~np.isnan(PS_mask[case]))
            ps_fraction /= (nr_shots_per_case*len(self.control_cases))
            self.qoi['ps_fraction'] = ps_fraction
        # Calculate distribution
        P = { case : np.mean(Shots_dig[self.Q_ancilla][case]) for case in self.control_cases }
        P_ideal = { case : case.count('1')%2 for case in self.control_cases }
        fidelity = 1-np.mean([ np.abs(P[case]-P_ideal[case]) for case in self.control_cases ])
        # Save stuff
        self.proc_data_dict['Shots_raw'] = Shots_raw
        self.proc_data_dict['Shots_dig'] = Shots_dig
        self.proc_data_dict['Threshold'] = Threshold
        self.proc_data_dict['P'] = P
        self.proc_data_dict['P_ideal'] = P_ideal
        self.qoi['fidelity'] = fidelity

    def prepare_plots(self):
        self.axs_dict = {}
        fig, ax = plt.subplots(figsize=(1+0.4*len(self.control_cases),3), dpi=100)
        self.figs['Parity_check_fidelity'] = fig
        self.axs_dict['plot_1'] = ax
        # fig.patch.set_alpha(0)
        self.plot_dicts['Parity_check_fidelity']={
                'plotfn': parity_fidelity_plotfn,
                'ax_id': 'plot_1',
                'P_dist': self.proc_data_dict['P'],
                'P_dist_ideal': self.proc_data_dict['P_ideal'],
                'control_cases': self.control_cases,
                'Q_ancilla': self.Q_ancilla,
                'Q_control': self.Q_control,
                'fidelity': self.qoi['fidelity'],
                'ps_fraction': self.qoi['ps_fraction']
                     if self.post_selection else None,
                'timestamp': self.timestamps[0]}

        fig, ax = plt.subplots(figsize=(1+0.4*len(self.control_cases),3), dpi=100)
        self.figs['Parity_check_error'] = fig
        self.axs_dict['plot_2'] = ax
        # fig.patch.set_alpha(0)
        self.plot_dicts['Parity_check_error']={
                'plotfn': parity_error_plotfn,
                'ax_id': 'plot_2',
                'P_dist': self.proc_data_dict['P'],
                'P_dist_ideal': self.proc_data_dict['P_ideal'],
                'control_cases': self.control_cases,
                'Q_ancilla': self.Q_ancilla,
                'Q_control': self.Q_control,
                'fidelity': self.qoi['fidelity'],
                'ps_fraction': self.qoi['ps_fraction']
                     if self.post_selection else None,
                'timestamp': self.timestamps[0]}

        if not self.post_selection:
            n_plots = 1
        else:
            n_plots = len(self.Q_control)+1
        fig, axs = plt.subplots(figsize=(n_plots*2.5,2), ncols=n_plots, sharey=True)
        if not self.post_selection:
            axs = [axs]
        self.figs['Raw_shots'] = fig
        self.axs_dict['plot_3'] = axs[0]
        # fig.patch.set_alpha(0)
        self.plot_dicts['Raw_shots']={
                'plotfn': raw_shots_plotfn,
                'ax_id': 'plot_3',
                'Shots_raw': self.proc_data_dict['Shots_raw'],
                'Threshold': self.proc_data_dict['Threshold'],
                'Q_ancilla': self.Q_ancilla,
                'Q_control': self.Q_control,
                'post_selection': self.post_selection,
                'timestamp': self.timestamps[0]}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def parity_fidelity_plotfn(
    ax,
    P_dist,
    P_dist_ideal,
    control_cases,
    Q_ancilla,
    Q_control,
    fidelity,
    ps_fraction,
    timestamp,
    **kw):
    fig = ax.get_figure()

    n = len(P_dist)
    idx_sort = np.argsort([ s.count('1') for s in control_cases ])
    P_dist = np.array([ P_dist[case] for case in control_cases])[idx_sort]
    P_dist_ideal = np.array([ P_dist_ideal[case] for case in control_cases])[idx_sort]

    ax.bar(np.arange(n), P_dist)
    ax.bar(np.arange(n), P_dist_ideal, lw=1, edgecolor='black', fc=(0,0,0,0))
    text_str = f'Fidelity: {fidelity*100:.1f}%'
    if ps_fraction:
        text_str = text_str + f'\nPS fraction: {ps_fraction*100:.1f}%'
    ax.text(1.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.array(control_cases)[idx_sort], rotation=90, fontsize=14)
    label = '{'+' '.join([q[0]+'_'+q[1] for q in Q_control])+'}'
    ax.set_xlabel(rf'Control state, $|\mathrm{label}\rangle$', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels([0, 0.5, 1], size=14)
    ax.set_ylabel(r'$|1\rangle_A$ probability', fontsize=14)
    ax.tick_params(axis='y', direction='in')
    ax.set_title(f'{timestamp}\n'+f'{Q_ancilla},{",".join(Q_control)} parity check')

def parity_error_plotfn(
    ax,
    P_dist,
    P_dist_ideal,
    control_cases,
    Q_ancilla,
    Q_control,
    fidelity,
    ps_fraction,
    timestamp,
    **kw):
    fig = ax.get_figure()
    n = len(P_dist)
    idx_sort = np.argsort([ s.count('1') for s in control_cases ])
    P_dist = np.array([ P_dist[case] for case in control_cases])[idx_sort]
    P_dist_ideal = np.array([ P_dist_ideal[case] for case in control_cases])[idx_sort]
    error = abs(P_dist-P_dist_ideal)
    ax.bar(np.arange(n), error*100)
    text_str = f'Fidelity: {fidelity*100:.1f}%'
    if ps_fraction:
        text_str = text_str + f'\nPS fraction: {ps_fraction*100:.1f}%'
    ax.text(1.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.array(control_cases)[idx_sort], rotation=90, fontsize=14)
    label = '{'+' '.join([q[0]+'_'+q[1] for q in Q_control])+'}'
    ax.set_xlabel(rf'Control state, $|\mathrm{label}\rangle$', fontsize=14)
    ax.set_ylabel(r'Parity assignement error (%)', fontsize=14)
    ax.tick_params(axis='y', direction='in')
    ax.set_title(f'{timestamp}\n'+f'{Q_ancilla},{",".join(Q_control)} parity check')

def raw_shots_plotfn(
    ax,
    Shots_raw,
    Threshold,
    Q_ancilla,
    Q_control,
    timestamp,
    post_selection,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    q_list = [Q_ancilla]
    if post_selection:
        q_list.append(Q_control)
    for i, q in enumerate(q_list):
        axs[i].hist(Shots_raw[q], bins=100)
        axs[i].axvline(Threshold[q], color='k', ls='--', lw=1)
        axs[i].set_title(f'Shots {q}')
        axs[i].set_xlabel('Integrated voltage')
    fig.suptitle(timestamp, y=1.1)