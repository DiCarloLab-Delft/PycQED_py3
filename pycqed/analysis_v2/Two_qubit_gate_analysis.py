import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
from matplotlib.colors import to_rgba


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