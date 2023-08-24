import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp,\
                                             get_timestamps_in_range
import pycqed.measurement.hdf5_data as h5d
from matplotlib.colors import to_rgba, LogNorm
from pycqed.analysis.tools.plotting import hsluv_anglemap45
import itertools
from pycqed.analysis.analysis_toolbox import set_xlabel
from pycqed.utilities.general import get_gate_directions, get_parking_qubits
from pycqed.utilities.general import print_exception


def Chevron(delta, t, g, delta_0, a, b, phi):
    '''
    Fit function for chevron function.
    Args:
        delta : Detuning of qubit
        t : duration of pulse
        g : coupling of avoided crossing
        delta_0 : detuning at avoided crossing
        a : scale factor used for fitting
        b : offset factor used for fitting
        phi : phase offset used for fitting (this 
              accounts for pulse distortion)
    '''
    g_rad = g*2*np.pi
    delta_rad = delta*2*np.pi
    delta_0_rad = delta_0*2*np.pi
    # Frequency of Chevron oscillation
    Omega = np.sqrt((delta_rad-delta_0_rad)**2+(2*g_rad)**2)
    # Amplitude of Chevron oscillation
    Osc_amp = (2*g_rad)**2 / ((delta_rad-delta_0_rad)**2+(2*g_rad)**2)
    # Population of Chevron oscillation
    pop = Osc_amp*(1-np.cos(Omega*t+phi))/2
    return a*pop + b

class Chevron_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for Chevron routine
    """
    def __init__(self,
                 Poly_coefs: float,
                 QH_freq: float,
                 QL_det: float,
                 avoided_crossing: str = "11-02",
                 Out_range: float = 5,
                 DAC_amp: float = 0.5,
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
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Poly_coefs = Poly_coefs
        self.QH_freq = QH_freq
        self.QL_det = QL_det
        self.avoided_crossing = avoided_crossing
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
        # Get qubit names
        self.QH = self.raw_data_dict['folder'].split(' ')[-3]
        self.QL = self.raw_data_dict['folder'].split(' ')[-2]
        self.proc_data_dict = {}
        # Sort data
        Amps = np.unique(self.raw_data_dict['data'][:,0])
        Times = np.unique(self.raw_data_dict['data'][:,1])
        Pop_H = self.raw_data_dict['data'][:,2]
        Pop_L = self.raw_data_dict['data'][:,3]
        nx, ny = len(Amps), len(Times)
        Pop_H = Pop_H.reshape(ny, nx)
        Pop_L = Pop_L.reshape(ny, nx)
        # Convert amplitude to detuning (frequency)
        P_func = np.poly1d(self.Poly_coefs)
        Out_voltage = Amps*self.DAC_amp*self.Out_range/2
        Detunings = P_func(Out_voltage)
        # Fit Chevron
        from scipy.optimize import curve_fit
        def fit_func(xy, g, delta_0, a, b, phi):
            delta, time = xy
            outcome = Chevron(delta, time, g, delta_0, a, b, phi)
            return outcome.ravel()
        # perform fit
        x, y = np.meshgrid(Detunings, Times)
        z = Pop_L
        # initial guess
        idx_det0 = np.argmax(np.mean(z, axis=0))
        p0 = [11e6,                 # g
              Detunings[idx_det0],  # delta_0
              np.max(z)-np.min(z),  # a
              np.min(z),            # b
              0,                    # phi
              ]
        popt, pcov = curve_fit(fit_func, (x,y), z.ravel(), p0=p0)
        detuning_freq = popt[1]
        detuning_amp = np.max((P_func-detuning_freq).roots)
        T_p = abs((np.pi-popt[4])/(2*np.pi*popt[0]))
        # Save data
        self.proc_data_dict['Out_voltage'] = Out_voltage
        self.proc_data_dict['Detunings'] = Detunings
        self.proc_data_dict['Times'] = Times
        self.proc_data_dict['Pop_H'] = Pop_H
        self.proc_data_dict['Pop_L'] = Pop_L
        self.proc_data_dict['Fit_params'] = popt
        self.qoi = {'coupling': popt[0],
                    'detuning_freq': detuning_freq,
                    'detuning_amp': detuning_amp,
                    'Tp': T_p}
    
    def prepare_plots(self):
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(12*.8,5*.8), ncols=3, dpi=100)
        self.figs[f'Chevron'] = fig
        self.axs_dict[f'Chevron'] = axs[0]
        self.plot_dicts[f'Chevron']={
            'plotfn': Chevron_plotfn,
            'ax_id': f'Chevron',
            'Detunings' : self.proc_data_dict['Detunings'],
            'Out_voltage' : self.proc_data_dict['Out_voltage'],
            'Times' : self.proc_data_dict['Times'],
            'Pop_H' : self.proc_data_dict['Pop_H'],
            'Pop_L' : self.proc_data_dict['Pop_L'],
            'f0' : self.qoi['detuning_freq'],
            'a0' : self.qoi['detuning_amp'],
            'tp' : self.qoi['Tp'],
            'ts' : self.timestamp,
            'qH' : self.QH, 'qL' : self.QL,
            'qH_freq' : self.QH_freq, 'qL_det' : self.QL_det,
            'poly_coefs' : self.Poly_coefs,
            'avoided_crossing' : self.avoided_crossing,
            'Fit_params' : self.proc_data_dict['Fit_params'],
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Chevron_plotfn(
    ax,
    qH, qL, 
    qH_freq, qL_det,
    poly_coefs,
    Detunings,
    Out_voltage,
    avoided_crossing,
    Fit_params,
    Times,
    Pop_H,
    Pop_L,
    f0, a0, tp,
    ts, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    # Avoided crossing plot
    p_func = np.poly1d(poly_coefs)
    
    Voltage_axis = np.linspace(0, Out_voltage[-1]*1.2, 201)
    Frequency_axis = qH_freq-p_func(Voltage_axis)
    axs[2].plot(Voltage_axis, Frequency_axis*1e-9, 'C0-')
    if qL_det != 0 :
        axs[2].axhline([(qH_freq-f0+qL_det)*1e-9], color='k', ls='--', alpha=.25)
    axs[2].axhline([(qH_freq-f0)*1e-9], color='k', ls='--')
    axs[2].text((Voltage_axis[0]+(Voltage_axis[-1]-Voltage_axis[0])*.02),
                (qH_freq-f0+(Frequency_axis[-1]-Frequency_axis[0])*.03)*1e-9,
                f'$f_{{{avoided_crossing}}}$', color='k', size=12, va='top')
    axs[2].plot([a0], [(qH_freq-f0)*1e-9], 'C3.')
    axs[2].set_xlabel('Output voltage (V)')
    axs[2].set_ylabel(f'{qH} frequency (GHz)')
    axs[2].set_xlim(Voltage_axis[0], Voltage_axis[-1])
    axs[2].set_title('Frequency scheme')
    # axt = axs[2].twinx()

    # Chevrons plot
    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        return X
    Detunings = get_plot_axis(Detunings)
    Times = get_plot_axis(Times)
    # High frequency qubit population
    axs[0].pcolormesh(Detunings*1e-6, Times*1e9, Pop_H)
    axs[0].set_xlabel(f'{qH} detuning (MHz)')
    axs[0].set_ylabel('Duration (ns)')
    axs[0].set_title(f'Population {qH}')
    axs[0].axvline(f0*1e-6, color='w', ls='--')
    axs[0].axhline(tp/2*1e9, color='w', ls='--')
    axs[0].plot([f0*1e-6], [tp/2*1e9], 'C3.')
    axt0 = axs[0].twiny()
    axt0.set_xlim((qH_freq*1e-6-np.array(axs[0].get_xlim()))*1e-3)
    axt0.set_xlabel(f'{qH} Frequency (GHz)')
    # Low frequency qubit population
    axs[1].pcolormesh(Detunings*1e-6, Times*1e9, Pop_L)
    axs[1].set_xlabel(f'{qH} detuning (MHz)')
    axs[1].axvline(f0*1e-6, color='w', ls='--')
    axs[1].axhline(tp/2*1e9, color='w', ls='--')
    axs[1].plot([f0*1e-6], [tp/2*1e9], 'C3.')
    axs[1].text((Detunings[0]+(Detunings[-1]-Detunings[0])*.02)*1e-6, (tp/2+(Times[-1]-Times[0])*.03)*1e9,
                f'$t_p/2={tp/2*1e9:.2f}$ ns', color='w', size=12)
    axs[1].text((Detunings[0]+(Detunings[-1]-Detunings[0])*.02)*1e-6, (Times[0]+(Times[-1]-Times[0])*.03)*1e9,
                f'$\\Delta={f0*1e-6:.2f}$ MHz', color='w', size=12)
    axs[1].text((Detunings[0]+(Detunings[-1]-Detunings[0])*.02)*1e-6, (Times[-1]-(Times[-1]-Times[0])*.03)*1e9,
                f'$J_2={Fit_params[0]*1e-6:.2f}$ MHz', color='w', size=12, va='top')
    axs[1].set_title(f'Population {qL}')
    axt1 = axs[1].twiny()
    axt1.set_xlim((qH_freq*1e-6-np.array(axs[1].get_xlim()))*1e-3)
    axt1.set_xlabel(f'{qH} Frequency (GHz)')
    # Add Chevron fit contours
    X = np.linspace(Detunings[0], Detunings[-1], 201)
    Y = np.linspace(Times[0], Times[-1], 201)
    _X, _Y = np.meshgrid(X, Y)
    Z = Chevron(_X, _Y, *Fit_params)
    Z = (Z - np.min(Z))/(np.max(Z)-np.min(Z))
    for c_lvl, alpha in zip([.05, .2, .5], [.1, .2, .5]):
        axs[0].contour(X*1e-6, Y*1e9, Z, [c_lvl], colors=['w'],
                     linewidths=[1], linestyles=['--'], alpha=alpha)
        axs[1].contour(X*1e-6, Y*1e9, Z, [c_lvl], colors=['w'],
                     linewidths=[1], linestyles=['--'], alpha=alpha)

    fig.suptitle(f'{ts}\nChevron {qH}, {qL}', y=.95)
    fig.tight_layout()


class TLS_landscape_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for TLS landscape
    """
    def __init__(self,
                 Q_freq: float,
                 Poly_coefs: float,
                 Out_range: float = 5,
                 DAC_amp: float = 0.5,
                 interaction_freqs: dict = None,
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
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Poly_coefs = Poly_coefs
        self.Q_freq = Q_freq
        self.interaction_freqs = interaction_freqs
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
        # Get qubit names
        self.Q_name = self.raw_data_dict['folder'].split(' ')[-3]
        self.proc_data_dict = {}
        # Sort data
        Amps = np.unique(self.raw_data_dict['data'][:,0])
        Times = np.unique(self.raw_data_dict['data'][:,1])
        Pop = self.raw_data_dict['data'][:,2]
        nx, ny = len(Amps), len(Times)
        Pop = Pop.reshape(ny, nx)
        # Convert amplitude to detuning (frequency)
        P_func = np.poly1d(self.Poly_coefs)
        Out_voltage = Amps*self.DAC_amp*self.Out_range/2
        Detunings = P_func(Out_voltage)
        # Save data
        self.proc_data_dict['Out_voltage'] = Out_voltage
        self.proc_data_dict['Detunings'] = Detunings
        self.proc_data_dict['Times'] = Times
        self.proc_data_dict['Pop'] = Pop
    
    def prepare_plots(self):
        self.axs_dict = {}
        fig, ax = plt.subplots(figsize=(10,4), dpi=100)
        self.figs[f'TLS_landscape'] = fig
        self.axs_dict[f'TLS_landscape'] = ax
        self.plot_dicts[f'TLS_landscape']={
            'plotfn': TLS_landscape_plotfn,
            'ax_id': f'TLS_landscape',
            'Detunings' : self.proc_data_dict['Detunings'],
            'Out_voltage' : self.proc_data_dict['Out_voltage'],
            'Times' : self.proc_data_dict['Times'],
            'Pop' : self.proc_data_dict['Pop'],
            'Q_name' : self.Q_name,
            'Q_freq' : self.Q_freq,
            'interaction_freqs' : self.interaction_freqs,
            'ts' : self.timestamp,
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def TLS_landscape_plotfn(
    ax,
    Q_name, 
    Q_freq,
    Detunings,
    Out_voltage,
    Times,
    Pop,
    ts,
    interaction_freqs=None,
    **kw):
    fig = ax.get_figure()
    # Chevrons plot
    def get_plot_axis(vals, rang=None):
        if len(vals)>1:
            dx = vals[1]-vals[0]
            X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        else:
            X = vals
        return X
    Detunings = get_plot_axis(Detunings)
    Times = get_plot_axis(Times)
    # Frequency qubit population
    vmax = min([1, np.max(Pop)])
    vmax = max([vmax, 0.15])
    im = ax.pcolormesh(Detunings*1e-6, Times*1e9, Pop, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Population')
    # plot two-qubit gate frequencies:
    if interaction_freqs:
        for gate, freq in interaction_freqs.items():
            if freq > 10e6:
                ax.axvline(freq*1e-6, color='w', ls='--')
                ax.text(freq*1e-6, np.mean(Times)*1e9,
                        f'CZ {gate}', va='center', ha='right',
                        color='w', rotation=90)
    ax.set_xlabel(f'{Q_name} detuning (MHz)')
    ax.set_ylabel('Duration (ns)')
    ax.set_title(f'Population {Q_name}')
    axt0 = ax.twiny()
    axt0.set_xlim((Q_freq*1e-6-np.array(ax.get_xlim()))*1e-3)
    axt0.set_xlabel(f'{Q_name} Frequency (GHz)')
    fig.suptitle(f'{ts}\nTLS landscape {Q_name}', y=.95)
    fig.tight_layout()


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


def _rotate_and_center_data(I, Q, vec0, vec1, phi=0):
    vector = vec1-vec0
    angle = np.arctan(vector[1]/vector[0])
    rot_matrix = np.array([[ np.cos(-angle+phi),-np.sin(-angle+phi)],
                           [ np.sin(-angle+phi), np.cos(-angle+phi)]])
    proc = np.array((I, Q))
    proc = np.dot(rot_matrix, proc)
    return proc.transpose()

def _calculate_fid_and_threshold(x0, n0, x1, n1):
    """
    Calculate fidelity and threshold from histogram data:
    x0, n0 is the histogram data of shots 0 (value and occurences),
    x1, n1 is the histogram data of shots 1 (value and occurences).
    """
    # Build cumulative histograms of shots 0 
    # and 1 in common bins by interpolation.
    all_x = np.unique(np.sort(np.concatenate((x0, x1))))
    cumsum0, cumsum1 = np.cumsum(n0), np.cumsum(n1)
    ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
    necumsum0 = ecumsum0/np.max(ecumsum0)
    ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
    necumsum1 = ecumsum1/np.max(ecumsum1)
    # Calculate optimal threshold and fidelity
    F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
    opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
    opt_idx = int(round(np.average(opt_idxs)))
    F_assignment_raw = F_vs_th[opt_idx]
    threshold_raw = all_x[opt_idx]
    return F_assignment_raw, threshold_raw

def _fit_double_gauss(x_vals, hist_0, hist_1,
                      _x0_guess=None, _x1_guess=None):
    '''
    Fit two histograms to a double gaussian with
    common parameters. From fitted parameters,
    calculate SNR, Pe0, Pg1, Teff, Ffit and Fdiscr.
    '''
    from scipy.optimize import curve_fit
    # Double gaussian model for fitting
    def _gauss_pdf(x, x0, sigma):
        return np.exp(-((x-x0)/sigma)**2/2)
    global double_gauss
    def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
        _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
        return _dist0
    # helper function to simultaneously fit both histograms with common parameters
    def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
        _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
        _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
        return np.concatenate((_dist0, _dist1))
    # Guess for fit
    pdf_0 = hist_0/np.sum(hist_0) # Get prob. distribution
    pdf_1 = hist_1/np.sum(hist_1) # 
    if _x0_guess == None:
        _x0_guess = np.sum(x_vals*pdf_0) # calculate mean
    if _x1_guess == None:
        _x1_guess = np.sum(x_vals*pdf_1) #
    _sigma0_guess = np.sqrt(np.sum((x_vals-_x0_guess)**2*pdf_0)) # calculate std
    _sigma1_guess = np.sqrt(np.sum((x_vals-_x1_guess)**2*pdf_1)) #
    _r0_guess = 0.01
    _r1_guess = 0.05
    _A0_guess = np.max(hist_0)
    _A1_guess = np.max(hist_1)
    p0 = [_x0_guess, _x1_guess, _sigma0_guess, _sigma1_guess, _A0_guess, _A1_guess, _r0_guess, _r1_guess]
    # Bounding parameters
    _x0_bound = (-np.inf,np.inf)
    _x1_bound = (-np.inf,np.inf)
    _sigma0_bound = (0,np.inf)
    _sigma1_bound = (0,np.inf)
    _r0_bound = (0,1)
    _r1_bound = (0,1)
    _A0_bound = (0,np.inf)
    _A1_bound = (0,np.inf)
    bounds = np.array([_x0_bound, _x1_bound, _sigma0_bound, _sigma1_bound, _A0_bound, _A1_bound, _r0_bound, _r1_bound])
    # Fit parameters within bounds
    popt, pcov = curve_fit(
        _double_gauss_joint, x_vals,
        np.concatenate((hist_0, hist_1)),
        p0=p0, bounds=bounds.transpose())
    popt0 = popt[[0,1,2,3,4,6]]
    popt1 = popt[[1,0,3,2,5,7]]
    # Calculate quantities of interest
    SNR = abs(popt0[0] - popt1[0])/((abs(popt0[2])+abs(popt1[2]))/2)
    P_e0 = popt0[5]*popt0[2]/(popt0[2]*popt0[5] + popt0[3]*(1-popt0[5]))
    P_g1 = popt1[5]*popt1[2]/(popt1[2]*popt1[5] + popt1[3]*(1-popt1[5]))
    # Fidelity from fit
    _range = x_vals[0], x_vals[-1]
    _x_data = np.linspace(*_range, 10001)
    _h0 = double_gauss(_x_data, *popt0)# compute distrubition from
    _h1 = double_gauss(_x_data, *popt1)# fitted parameters.
    Fid_fit, threshold_fit = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
    # Discrimination fidelity
    _h0 = double_gauss(_x_data, *popt0[:-1], 0)# compute distrubition without residual
    _h1 = double_gauss(_x_data, *popt1[:-1], 0)# excitation of relaxation.
    Fid_discr, threshold_discr = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
    # return results
    qoi = { 'SNR': SNR,
            'P_e0': P_e0, 'P_g1': P_g1, 
            'Fid_fit': Fid_fit, 'Fid_discr': Fid_discr }
    return popt0, popt1, qoi

def _decision_boundary_points(coefs, intercepts):
    '''
    Find points along the decision boundaries of 
    LinearDiscriminantAnalysis (LDA).
    This is performed by finding the interception
    of the bounds of LDA. For LDA, these bounds are
    encoded in the coef_ and intercept_ parameters
    of the classifier.
    Each bound <i> is given by the equation:
    y + coef_i[0]/coef_i[1]*x + intercept_i = 0
    Note this only works for LinearDiscriminantAnalysis.
    Other classifiers might have diferent bound models.
    '''
    points = {}
    # Cycle through model coeficients
    # and intercepts.
    for i, j in [[0,1], [1,2], [0,2]]:
        c_i = coefs[i]
        int_i = intercepts[i]
        c_j = coefs[j]
        int_j = intercepts[j]
        x =  (- int_j/c_j[1] + int_i/c_i[1])/(-c_i[0]/c_i[1] + c_j[0]/c_j[1])
        y = -c_i[0]/c_i[1]*x - int_i/c_i[1]
        points[f'{i}{j}'] = (x, y)
    # Find mean point
    points['mean'] = np.mean([ [x, y] for (x, y) in points.values()], axis=0)
    return points

class Repeated_CZ_experiment_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for LRU experiment.
    """
    def __init__(self,
                 rounds: int,
                 heralded_init: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.rounds = rounds
        self.heralded_init = heralded_init
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
        ######################################
        # Sort shots and assign them
        ######################################
        _cycle = self.rounds*2 + 3**2
        # Get qubit names in channel order
        names = [ name.decode().split(' ')[-2] for name in self.raw_data_dict['value_names'] ]
        self.Qubits = names[::2]
        # Dictionary that will store raw shots
        # so that they can later be sorted.
        raw_shots = {q: {} for q in self.Qubits}
        for q_idx, qubit in enumerate(self.Qubits):
            self.proc_data_dict[qubit] = {}
            _ch_I, _ch_Q = 2*q_idx+1, 2*q_idx+2
            _raw_shots = self.raw_data_dict['data'][:,[_ch_I, _ch_Q]]
            _shots_0 = _raw_shots[2*self.rounds+0::_cycle]
            _shots_1 = _raw_shots[2*self.rounds+4::_cycle]
            _shots_2 = _raw_shots[2*self.rounds+8::_cycle]
            # Rotate data
            center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
            center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
            center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
            raw_shots[qubit] = _rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
            # Sort different combinations of input states
            states = ['0','1', '2']
            combinations = [''.join(s) for s in itertools.product(states, repeat=2)]
            self.combinations = combinations
            Shots_state = {}
            for i, comb in enumerate(combinations):
                Shots_state[comb] = raw_shots[qubit][2*self.rounds+i::_cycle]
            Shots_0 = np.vstack([Shots_state[comb] for comb in combinations if comb[q_idx]=='0'])
            Shots_1 = np.vstack([Shots_state[comb] for comb in combinations if comb[q_idx]=='1'])
            Shots_2 = np.vstack([Shots_state[comb] for comb in combinations if comb[q_idx]=='2'])
            self.proc_data_dict[qubit]['Shots_0'] = Shots_0
            self.proc_data_dict[qubit]['Shots_1'] = Shots_1
            self.proc_data_dict[qubit]['Shots_2'] = Shots_2
            self.proc_data_dict[qubit]['Shots_state'] = Shots_state
            # Use classifier for data
            data = np.concatenate((Shots_0, Shots_1, Shots_2))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            clf.fit(data, labels)
            dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2'],
                                    [Shots_0, Shots_1, Shots_2]):
                _res = clf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((3,3))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
                for j, state in enumerate(['0', '1', '2']):
                    _res = clf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            self.proc_data_dict[qubit]['dec_bounds'] = dec_bounds
            self.proc_data_dict[qubit]['classifier'] = clf
            self.proc_data_dict[qubit]['Fid_dict'] = Fid_dict
            self.proc_data_dict[qubit]['Assignment_matrix'] = M
            #########################################
            # Project data along axis perpendicular
            # to the decision boundaries.
            #########################################
            ############################
            # Projection along 01 axis.
            ############################
            # Rotate shots over 01 axis
            shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_1 = shots_1[:,0]
            n_shots_1 = len(shots_1)
            # find range
            _all_shots = np.concatenate((shots_0, shots_1))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x1, n1 = np.unique(shots_1, return_counts=True)
            Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
            # Save processed data
            self.proc_data_dict[qubit]['projection_01'] = {}
            self.proc_data_dict[qubit]['projection_01']['h0'] = h0
            self.proc_data_dict[qubit]['projection_01']['h1'] = h1
            self.proc_data_dict[qubit]['projection_01']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_01']['popt0'] = popt0
            self.proc_data_dict[qubit]['projection_01']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_01']['SNR'] = params_01['SNR']
            self.proc_data_dict[qubit]['projection_01']['Fid'] = Fid_01
            self.proc_data_dict[qubit]['projection_01']['threshold'] = threshold_01
            ############################
            # Projection along 12 axis.
            ############################
            # Rotate shots over 12 axis
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            # Take relavant quadrature
            shots_1 = shots_1[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_1, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x1, n1 = np.unique(shots_1, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
            # Histogram of shots for 1 and 2
            h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
            # Save processed data
            self.proc_data_dict[qubit]['projection_12'] = {}
            self.proc_data_dict[qubit]['projection_12']['h1'] = h1
            self.proc_data_dict[qubit]['projection_12']['h2'] = h2
            self.proc_data_dict[qubit]['projection_12']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_12']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_12']['popt2'] = popt2
            self.proc_data_dict[qubit]['projection_12']['SNR'] = params_12['SNR']
            self.proc_data_dict[qubit]['projection_12']['Fid'] = Fid_12
            self.proc_data_dict[qubit]['projection_12']['threshold'] = threshold_12
            ############################
            # Projection along 02 axis.
            ############################
            # Rotate shots over 02 axis
            shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_0, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
            # Save processed data
            self.proc_data_dict[qubit]['projection_02'] = {}
            self.proc_data_dict[qubit]['projection_02']['h0'] = h0
            self.proc_data_dict[qubit]['projection_02']['h2'] = h2
            self.proc_data_dict[qubit]['projection_02']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_02']['popt0'] = popt0
            self.proc_data_dict[qubit]['projection_02']['popt2'] = popt2
            self.proc_data_dict[qubit]['projection_02']['SNR'] = params_02['SNR']
            self.proc_data_dict[qubit]['projection_02']['Fid'] = Fid_02
            self.proc_data_dict[qubit]['projection_02']['threshold'] = threshold_02
        ############################################
        # Calculate Mux assignment fidelity matrix #
        ############################################
        # Get assignment fidelity matrix
        M = np.zeros((9,9))
        states = ['0','1', '2']
        combinations = [''.join(s) for s in itertools.product(states, repeat=2)]
        # Calculate population vector for each input state
        for i, comb in enumerate(combinations):
            _res = []
            # Assign shots for each qubit
            for q in self.Qubits:
                _clf = self.proc_data_dict[q]['classifier']
                _res.append(_clf.predict(self.proc_data_dict[q]['Shots_state'][comb]).astype(str))
            # <res> holds the outcome of shots for each qubit
            res = np.array(_res).T
            for j, comb in enumerate(combinations):
                M[i][j] = np.mean(np.logical_and(*(res == list(comb)).T))
        self.proc_data_dict['Mux_assignment_matrix'] = M
        ##############################
        # Analyze experimental shots #
        ##############################
        self.raw_shots = raw_shots
        _shots_ref = {}
        _shots_exp = {}
        for q in self.Qubits:
            _clf = self.proc_data_dict[q]['classifier']
            _shots_ref[q] = np.array([ _clf.predict(self.raw_shots[q][i+self.rounds::_cycle]) for i in range(self.rounds) ])
            _shots_exp[q] = np.array([ _clf.predict(self.raw_shots[q][i::_cycle]) for i in range(self.rounds) ])
            # convert to string
            _shots_ref[q] = _shots_ref[q].astype(str)
            _shots_exp[q] = _shots_exp[q].astype(str)
        # Concatenate strings of different outcomes
        Shots_ref = _shots_ref[self.Qubits[0]]
        Shots_exp = _shots_exp[self.Qubits[0]]
        for q in self.Qubits[1:]:
            Shots_ref = np.char.add(Shots_ref, _shots_ref[q])
            Shots_exp = np.char.add(Shots_exp, _shots_exp[q])
        '''
        Shots_ref and Shots_exp is an array
        of shape (<rounds>, <nr_shots>).
        We will use them to calculate the 
        population vector at each round.
        '''
        Pop_vec_exp = np.zeros((self.rounds, len(combinations)))
        Pop_vec_ref = np.zeros((self.rounds, len(combinations)))
        for i in range(self.rounds):
            Pop_vec_ref[i] = [np.mean(Shots_ref[i]==comb) for comb in combinations]
            Pop_vec_exp[i] = [np.mean(Shots_exp[i]==comb) for comb in combinations]
        # Apply readout corrections
        M = self.proc_data_dict['Mux_assignment_matrix']
        M_inv = np.linalg.inv(M)
        Pop_vec_ref = np.dot(Pop_vec_ref, M_inv)
        Pop_vec_exp = np.dot(Pop_vec_exp, M_inv)
        self.proc_data_dict['Pop_vec_ref'] = Pop_vec_ref
        self.proc_data_dict['Pop_vec_exp'] = Pop_vec_exp
        # Calculate 2-qubit leakage probability
        _leak_idxs = np.where([ '2' in comb for comb in combinations])[0]
        P_leak_ref = np.sum(Pop_vec_ref[:,_leak_idxs], axis=1)
        P_leak_exp = np.sum(Pop_vec_exp[:,_leak_idxs], axis=1)
        self.proc_data_dict['P_leak_ref'] = P_leak_ref
        self.proc_data_dict['P_leak_exp'] = P_leak_exp
        # Fit leakage and seepage
        from scipy.optimize import curve_fit
        def func(n, L, S):
            return (1-np.exp(-n*(L+S)))*L/(L+S)
        _x = np.arange(self.rounds+1)
        _y = [0]+list(self.proc_data_dict['P_leak_ref'])
        p0 = [.1, .1]
        popt_ref, pcov_ref = curve_fit(func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
        _y = [0]+list(self.proc_data_dict['P_leak_exp'])
        popt_exp, pcov_exp = curve_fit(func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
        self.proc_data_dict['fit_ref'] = popt_ref, pcov_ref
        self.proc_data_dict['fit_exp'] = popt_exp, pcov_exp
        # Calculate individual leakage probability
        for i, q in enumerate(self.Qubits):
            _leak_idxs = np.where([ '2' == comb[i] for comb in combinations])[0]
            P_leak_ref = np.sum(Pop_vec_ref[:,_leak_idxs], axis=1)
            P_leak_exp = np.sum(Pop_vec_exp[:,_leak_idxs], axis=1)
            self.proc_data_dict[f'P_leak_ref_{q}'] = P_leak_ref
            self.proc_data_dict[f'P_leak_exp_{q}'] = P_leak_exp
            # Fit leakage and seepage rates
            _x = np.arange(self.rounds)
            _y = list(self.proc_data_dict[f'P_leak_ref_{q}'])
            p0 = [.1, .1]
            popt_ref, pcov_ref = curve_fit(func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
            _y = list(self.proc_data_dict[f'P_leak_exp_{q}'])
            popt_exp, pcov_exp = curve_fit(func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
            self.proc_data_dict[f'fit_ref_{q}'] = popt_ref, pcov_ref
            self.proc_data_dict[f'fit_exp_{q}'] = popt_exp, pcov_exp

    def prepare_plots(self):
        self.axs_dict = {}
        for qubit in self.Qubits:
            fig = plt.figure(figsize=(8,4), dpi=100)
            axs = [fig.add_subplot(121),
                   fig.add_subplot(322),
                   fig.add_subplot(324),
                   fig.add_subplot(326)]
            # fig.patch.set_alpha(0)
            self.axs_dict[f'IQ_readout_histogram_{qubit}'] = axs[0]
            self.figs[f'IQ_readout_histogram_{qubit}'] = fig
            self.plot_dicts[f'IQ_readout_histogram_{qubit}'] = {
                'plotfn': ssro_IQ_projection_plotfn,
                'ax_id': f'IQ_readout_histogram_{qubit}',
                'shots_0': self.proc_data_dict[qubit]['Shots_0'],
                'shots_1': self.proc_data_dict[qubit]['Shots_1'],
                'shots_2': self.proc_data_dict[qubit]['Shots_2'],
                'projection_01': self.proc_data_dict[qubit]['projection_01'],
                'projection_12': self.proc_data_dict[qubit]['projection_12'],
                'projection_02': self.proc_data_dict[qubit]['projection_02'],
                'classifier': self.proc_data_dict[qubit]['classifier'],
                'dec_bounds': self.proc_data_dict[qubit]['dec_bounds'],
                'Fid_dict': self.proc_data_dict[qubit]['Fid_dict'],
                'qubit': qubit,
                'timestamp': self.timestamp
            }
            fig, ax = plt.subplots(figsize=(3,3), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict[f'Assignment_matrix_{qubit}'] = ax
            self.figs[f'Assignment_matrix_{qubit}'] = fig
            self.plot_dicts[f'Assignment_matrix_{qubit}'] = {
                'plotfn': assignment_matrix_plotfn,
                'ax_id': f'Assignment_matrix_{qubit}',
                'M': self.proc_data_dict[qubit]['Assignment_matrix'],
                'qubit': qubit,
                'timestamp': self.timestamp
            }
        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict[f'Mux_assignment_matrix'] = ax
        self.figs[f'Mux_assignment_matrix'] = fig
        self.plot_dicts[f'Mux_assignment_matrix'] = {
            'plotfn': mux_assignment_matrix_plotfn,
            'ax_id': f'Mux_assignment_matrix',
            'M': self.proc_data_dict['Mux_assignment_matrix'],
            'Qubits': self.Qubits,
            'timestamp': self.timestamp
        }
        fig = plt.figure(figsize=(6,6.5))
        gs = fig.add_gridspec(7, 2)
        axs = [fig.add_subplot(gs[0:3,0]),
               fig.add_subplot(gs[0:3,1]),
               fig.add_subplot(gs[3:5,0]),
               fig.add_subplot(gs[3:5,1]),
               fig.add_subplot(gs[5:7,:])]
        # fig.patch.set_alpha(0)
        self.axs_dict['Population_plot'] = axs[0]
        self.figs['Population_plot'] = fig
        self.plot_dicts['Population_plot'] = {
            'plotfn': population_plotfn,
            'ax_id': 'Population_plot',
            'rounds': self.rounds,
            'combinations': self.combinations,
            'Pop_vec_ref' : self.proc_data_dict['Pop_vec_ref'],
            'Pop_vec_exp' : self.proc_data_dict['Pop_vec_exp'],
            'P_leak_ref' : self.proc_data_dict['P_leak_ref'],
            'P_leak_exp' : self.proc_data_dict['P_leak_exp'],
            'P_leak_ref_q0' : self.proc_data_dict[f'P_leak_ref_{self.Qubits[0]}'],
            'P_leak_exp_q0' : self.proc_data_dict[f'P_leak_exp_{self.Qubits[0]}'],
            'P_leak_ref_q1' : self.proc_data_dict[f'P_leak_ref_{self.Qubits[1]}'],
            'P_leak_exp_q1' : self.proc_data_dict[f'P_leak_exp_{self.Qubits[1]}'],
            'fit_ref' : self.proc_data_dict['fit_ref'],
            'fit_exp' : self.proc_data_dict['fit_exp'],
            'fit_ref_q0' : self.proc_data_dict[f'fit_ref_{self.Qubits[0]}'],
            'fit_exp_q0' : self.proc_data_dict[f'fit_exp_{self.Qubits[0]}'],
            'fit_ref_q1' : self.proc_data_dict[f'fit_ref_{self.Qubits[1]}'],
            'fit_exp_q1' : self.proc_data_dict[f'fit_exp_{self.Qubits[1]}'],
            'Qubits': self.Qubits,
            'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def ssro_IQ_projection_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    projection_01,
    projection_12,
    projection_02,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'IQ plot qubit {qubit}')
    fig.suptitle(f'{timestamp}\n')
    ##########################
    # Plot projections
    ##########################
    # 01 projection
    _bin_c = projection_01['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[1].bar(_bin_c, projection_01['h0'], bin_width, fc='C0', alpha=0.4)
    axs[1].bar(_bin_c, projection_01['h1'], bin_width, fc='C3', alpha=0.4)
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt0']), '-C0')
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt1']), '-C3')
    axs[1].axvline(projection_01['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_01["Fid"]*100:.1f}%',
                      f'SNR : {projection_01["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[1].text(.775, .9, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[1].text(projection_01['popt0'][0], projection_01['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[1].text(projection_01['popt1'][0], projection_01['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[1].set_xticklabels([])
    axs[1].set_xlim(_bin_c[0], _bin_c[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Projection of data')
    # 12 projection
    _bin_c = projection_12['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[2].bar(_bin_c, projection_12['h1'], bin_width, fc='C3', alpha=0.4)
    axs[2].bar(_bin_c, projection_12['h2'], bin_width, fc='C2', alpha=0.4)
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt1']), '-C3')
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt2']), '-C2')
    axs[2].axvline(projection_12['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_12["Fid"]*100:.1f}%',
                      f'SNR : {projection_12["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[2].text(.775, .9, text, transform=axs[2].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[2].text(projection_12['popt1'][0], projection_12['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[2].text(projection_12['popt2'][0], projection_12['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[2].set_xticklabels([])
    axs[2].set_xlim(_bin_c[0], _bin_c[-1])
    axs[2].set_ylim(bottom=0)
    # 02 projection
    _bin_c = projection_02['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[3].bar(_bin_c, projection_02['h0'], bin_width, fc='C0', alpha=0.4)
    axs[3].bar(_bin_c, projection_02['h2'], bin_width, fc='C2', alpha=0.4)
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt0']), '-C0')
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt2']), '-C2')
    axs[3].axvline(projection_02['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_02["Fid"]*100:.1f}%',
                      f'SNR : {projection_02["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[3].text(projection_02['popt0'][0], projection_02['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[3].text(projection_02['popt2'][0], projection_02['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[3].set_xticklabels([])
    axs[3].set_xlim(_bin_c[0], _bin_c[-1])
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlabel('Integrated voltage')
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def assignment_matrix_plotfn(
    M,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    im = ax.imshow(M, cmap=plt.cm.Reds, vmin=0, vmax=1)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                             color = 'white')
            else:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$'])
    ax.set_xlabel('Assigned state')
    ax.set_yticks([0,1,2])
    ax.set_yticklabels([r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$'])
    ax.set_ylabel('Prepared state')
    ax.set_title(f'{timestamp}\nQutrit assignment matrix qubit {qubit}')
    cbar_ax = fig.add_axes([.95, .15, .03, .7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('assignment probability')

def mux_assignment_matrix_plotfn(
    M,
    Qubits,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()

    states = ['0','1', '2']
    combinations = [''.join(s) for s in itertools.product(states, repeat=2)]
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                        color = 'white', size=10)
            elif abs(c)>.01:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                        size=8)
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels([f'${c0}_\mathrm{{{Qubits[0]}}}{c1}_\mathrm{{{Qubits[1]}}}$' for c0, c1 in combinations], size=8)
    ax.set_yticklabels([f'${c0}_\mathrm{{{Qubits[0]}}}{c1}_\mathrm{{{Qubits[1]}}}$' for c0, c1 in combinations], size=8)
    ax.set_xlabel('Assigned state')
    ax.set_ylabel('Input state')
    cb = fig.colorbar(im, orientation='vertical', aspect=35)
    pos = ax.get_position()
    pos = [ pos.x0+.65, pos.y0, pos.width, pos.height ]
    fig.axes[-1].set_position(pos)
    cb.set_label('Assignment probability', rotation=-90, labelpad=15)
    ax.set_title(f'{timestamp}\nMultiplexed qutrit assignment matrix {" ".join(Qubits)}')

def population_plotfn(
    rounds,
    combinations,
    Pop_vec_ref,
    Pop_vec_exp,
    P_leak_ref,
    P_leak_exp,
    P_leak_ref_q0,
    P_leak_exp_q0,
    P_leak_ref_q1,
    P_leak_exp_q1,
    fit_ref,
    fit_exp,
    fit_ref_q0,
    fit_exp_q0,
    fit_ref_q1,
    fit_exp_q1,
    Qubits,
    timestamp,
    ax, **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()

    Rounds = np.arange(rounds)
    color = {'00' : '#bbdefb',
             '01' : '#64b5f6',
             '10' : '#1e88e5',
             '11' : '#0d47a1',
             '02' : '#003300',
             '20' : '#1b5e20',
             '12' : '#4c8c4a',
             '21' : '#81c784',
             '22' : '#b2fab4'}
    # Plot probabilities
    for i, comb in enumerate(combinations):
        label = f'${comb[0]}_\mathrm{{{Qubits[0]}}}{comb[1]}_\mathrm{{{Qubits[1]}}}$'
        axs[0].plot(Rounds, Pop_vec_ref[:,i], color=color[comb], label=label)
        axs[1].plot(Rounds, Pop_vec_exp[:,i], color=color[comb], label=label)
    # Plot qubit leakage probability
    def func(n, L, S):
        return (1-np.exp(-n*(L+S)))*L/(L+S)
    axs[2].plot(Rounds, func(Rounds, *fit_ref_q0[0]), '--k')
    axs[2].plot(Rounds, func(Rounds, *fit_exp_q0[0]), '--k')
    axs[2].plot(Rounds, P_leak_ref_q0, 'C8', label='Ref.')
    axs[2].plot(Rounds, P_leak_exp_q0, 'C4', label='Gate')
    axs[2].legend(frameon=False, loc=4)
    axs[2].text(.05, .8, Qubits[0], transform=axs[2].transAxes)
    axs[3].plot(Rounds, func(Rounds, *fit_ref_q1[0]), '--k')
    axs[3].plot(Rounds, func(Rounds, *fit_exp_q1[0]), '--k')
    axs[3].plot(Rounds, P_leak_ref_q1, 'C8', label='Ref.')
    axs[3].plot(Rounds, P_leak_exp_q1, 'C4', label='Gate')
    axs[3].legend(frameon=False, loc=4)
    axs[3].text(.05, .8, Qubits[1], transform=axs[3].transAxes)
    # Plot total leakage probability
    axs[4].plot(Rounds, func(Rounds, *fit_ref[0]), '--k')
    axs[4].plot(Rounds, func(Rounds, *fit_exp[0]), '--k')
    axs[4].plot(Rounds, P_leak_ref, 'C8', label='Ref.')
    axs[4].plot(Rounds, P_leak_exp, 'C4', label='Gate')
    # Set common yaxis
    _lim = (*axs[0].get_ylim(), *axs[1].get_ylim())
    axs[0].set_ylim(min(_lim), max(_lim))
    axs[1].set_ylim(min(_lim), max(_lim))
    _lim = (*axs[2].get_ylim(), *axs[3].get_ylim())
    axs[2].set_ylim(min(_lim), max(_lim))
    axs[3].set_ylim(min(_lim), max(_lim))
    axs[4].set_xlabel('Rounds')
    axs[0].set_ylabel('Population')
    axs[2].set_ylabel('Leak. population')
    axs[4].set_ylabel('Leak. population')
    axs[1].set_yticklabels([])
    axs[3].set_yticklabels([])
    axs[1].legend(frameon=False, bbox_to_anchor=(1.01, 1.1), loc=2)
    axs[4].legend(frameon=False)
    axs[0].set_title('Reference')
    axs[1].set_title('Gate')
    fig.suptitle(f'{timestamp}\nRepeated CZ experiment {" ".join(Qubits)}')
    fig.tight_layout()
    popt_ref_q0, pcov_ref_q0 = fit_ref_q0
    perr_ref_q0 = np.sqrt(np.diag(pcov_ref_q0))
    popt_exp_q0, pcov_exp_q0 = fit_exp_q0
    perr_exp_q0 = np.sqrt(np.diag(pcov_exp_q0))
    perr_CZ_q0 = np.sqrt(perr_ref_q0[0]**2+perr_exp_q0[0]**2)
    popt_ref_q1, pcov_ref_q1 = fit_ref_q1
    perr_ref_q1 = np.sqrt(np.diag(pcov_ref_q1))
    popt_exp_q1, pcov_exp_q1 = fit_exp_q1
    perr_exp_q1 = np.sqrt(np.diag(pcov_exp_q1))
    perr_CZ_q1 = np.sqrt(perr_ref_q1[0]**2+perr_exp_q1[0]**2)
    text_str = 'Qubit leakage\n'+\
               f'CZ $L_1^\\mathrm{{{Qubits[0]}}}$: ${(popt_exp_q0[0]-popt_ref_q0[0])*100:.2f}\\pm{perr_CZ_q0*100:.2f}$%\n'+\
               f'CZ $L_1^\\mathrm{{{Qubits[1]}}}$: ${(popt_exp_q1[0]-popt_ref_q1[0])*100:.2f}\\pm{perr_CZ_q1*100:.2f}$%'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[3].text(1.06, 1, text_str, transform=axs[3].transAxes, fontsize=8.5,
                verticalalignment='top', bbox=props)
    popt_ref, pcov_ref = fit_ref
    perr_ref = np.sqrt(np.diag(pcov_ref))
    popt_exp, pcov_exp = fit_exp
    perr_exp = np.sqrt(np.diag(pcov_exp))
    perr_CZ = np.sqrt(perr_ref[0]**2+perr_exp[0]**2)
    text_str = 'Ref. curve\n'+\
               f'$L_1$: ${popt_ref[0]*100:.2f}\\pm{perr_ref[0]*100:.2f}$%\n'+\
               f'$L_2$: ${popt_ref[1]*100:.2f}\\pm{perr_ref[1]*100:.2f}$%\n'+\
               'Gate curve\n'+\
               f'$L_1$: ${popt_exp[0]*100:.2f}\\pm{perr_exp[0]*100:.2f}$%\n'+\
               f'$L_2$: ${popt_exp[1]*100:.2f}\\pm{perr_exp[1]*100:.2f}$%\n\n'+\
               f'CZ $L_1$: ${(popt_exp[0]-popt_ref[0])*100:.2f}\\pm{perr_CZ*100:.2f}$%'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[4].text(1.03, 1, text_str, transform=axs[4].transAxes, fontsize=8.5,
                verticalalignment='top', bbox=props)


def SNZ(delta, tmid, tp, g, delta_0, det11_02, n_dist, B_amp):
    '''
    Function parametrizing the SNZ landscape.
    Args:
        delta : Detuning of high freq. qubit
        tp : duration of pulse.
        tmid :  SNZ tmid parameter.
        n_dist : # of sampling points that model pulse distortion.
        B_amp : SNZ B amplitude parameter.
        g : coupling of avoided crossing.
        delta_0 : detuning at avoided crossing.
        det11_02 :  detuning of 11-02 levels at sweetspot
    '''
    g_rad = g*2*np.pi
    det11_02_rad = det11_02*2*np.pi
    delta_rad = delta*2*np.pi
    delta_0_rad = delta_0*2*np.pi
    delta_rad -= delta_0_rad
    # Convert B_amp to frequency detuning
    B_det_rad = (1-B_amp)*det11_02_rad
    # Frequency of Chevron oscillation
    Omega = np.sqrt(delta_rad**2+(2*g_rad)**2)
    # Population of first Chevron oscillation
    _term1 = -np.exp(+1j*Omega/2*tp)*((delta_rad+Omega)/(2*g_rad))*( -g_rad/Omega*1 )
    _term2 = -np.exp(-1j*Omega/2*tp)*((delta_rad-Omega)/(2*g_rad))*( +g_rad/Omega*1 )
    _term3 = np.exp(1j*Omega/2*tp)*( -g_rad/Omega*1 )
    _term4 = np.exp(-1j*Omega/2*tp)*( +g_rad/Omega*1 )
    c11 = _term1+_term2
    c20 = _term3+_term4
    # Population after evolving in B amp
    tB = .5/2.4e9
    c11 = c11*np.exp(1j*-B_det_rad/2*tB)
    c20 = c20*np.exp(1j*+B_det_rad/2*tB)
    # Population after evolving in the sweetspot
    # We account for pulse distortion using an offset in tmid
    t_mid_distorted = (tmid-n_dist/2.4e9)
    c11 = c11*np.exp(1j*-det11_02_rad/2*t_mid_distorted)
    c20 = c20*np.exp(1j*+det11_02_rad/2*t_mid_distorted)
    # Population after evolving in B amp
    tB = .5/2.4e9
    c11 = c11*np.exp(1j*-B_det_rad/2*tB)
    c20 = c20*np.exp(1j*+B_det_rad/2*tB)
    # Population after second Chevron
    _term1 = -np.exp(1j*Omega/2*tp)*((delta_rad+Omega)/(2*g_rad))*( c20/2*(1-delta_rad/Omega)-g_rad/Omega*c11 )
    _term2 = -np.exp(-1j*Omega/2*tp)*((delta_rad-Omega)/(2*g_rad))*( c20/2*(1+delta_rad/Omega)+g_rad/Omega*c11 )
    _term3 = np.exp(1j*Omega/2*tp)*( c20/2*(1-delta_rad/Omega)-g_rad/Omega*c11 )
    _term4 = np.exp(-1j*Omega/2*tp)*( c20/2*(1+delta_rad/Omega)+g_rad/Omega*c11 )
    c11 = _term1+_term2
    c20 = _term3+_term4
    # Calculate state populations
    pop11 = np.abs(c11)**2
    pop20 = np.abs(c20)**2
    # Calculate conditional phase
    phase11 = np.angle(c11) 
    phase20 = np.angle(c20)
    cphase = np.angle(c11) - delta_rad*tp + det11_02_rad*t_mid_distorted/2 + B_det_rad*tB
    cphase *= -1
    phase11 = np.mod(phase11*180/np.pi, 360)
    phase20 = np.mod(phase20*180/np.pi, 360)
    cphase = np.mod(cphase*180/np.pi, 360)
    return pop20, pop11, cphase

def get_optimal_SNZ_params(xy, pop20, cphase):
    '''
    Extracts optimal SNZ parameters detuning,
    tmid [ xy = (det, tmid) ] using corresponding
    20 population and cphase from SNZ lanscape. 
    '''
    x, y = xy
    assert x.shape == y.shape
    assert (x.shape == pop20.shape) and (x.shape == cphase.shape)
    # build cost function for finding optimal parameters
    _a = ((cphase-180)/180)**2    # cphase term
    _b = pop20**2                # leakage term
    _c = np.mean(pop20, axis=0)**2
    Cost_function = _a+_b+_c
    nx, ny = np.unravel_index(np.argmin(Cost_function, axis=None), Cost_function.shape)
    opt_detuning = x[nx, ny]
    opt_tmid = y[nx, ny]
    return opt_detuning, opt_tmid*2.4e9

def get_Tmid_parameters_from_SNZ_landscape(XY, fit_params):
    '''
    Extracts multiple optimal parameters from
    different 180 cphase contours of SNZ lanscape.
    Args:
        XY : (Dets, Tmids) Tuple of 1D arrays with 
        detunings (Hz) and Tmids (# sampling points)
        of landscape.
        fit_params: output SNZ fit params.
    '''
    # Sort fit parameters
    tp_factor, g, delta_0, det11_02, n_dist, _a, _b = fit_params
    tp = tp_factor/(4*g)
    # Get interpolated landscape limits
    Det, Tmid = XY
    X = np.linspace(Det[0], Det[-1], 201)
    Y = np.linspace(Tmid[0], Tmid[-1], 201)/2.4e9
    # Number of suitable 180 cphase contour levels (+1)
    n_contours = int((Y[-1]-n_dist/2.4e9)*det11_02)
    # Calculate optimal parameters for each contour section
    Opt_params = []
    for i in range(n_contours+2):
        # Calculate interpolated SNZ landscape for contour section
        X_range = X*1
        Y_range = np.linspace((i-.5)/det11_02+n_dist/2.4e9,
                              (i+.5)/det11_02+n_dist/2.4e9, 201)
        _x, _y = np.meshgrid(X_range, Y_range)
        z0, z1, z2 = SNZ(delta=_x, tmid=_y, g=g, tp=tp, B_amp=0.5,
             delta_0=delta_0, det11_02=det11_02, n_dist=n_dist)
        # Compute optimal det and tmid for contour section
        opt_params = get_optimal_SNZ_params((_x, _y), z0, z2)
        # If value is within range
        if (opt_params[1]>Tmid[0]) and (opt_params[1]<Tmid[-1]+.5):
            Opt_params.append(opt_params)
    return Opt_params

def get_value_at_coordinate(arr, x_arr, y_arr, x, y):
    """
    Get the value at the provided coordinates from the 2D numpy array.

    Parameters:
    arr (numpy.ndarray): input 2D array
    x (float): x-coordinate (x_arr[0] <= x <= x_arr[-1])
    y (float): y-coordinate (y_arr[0] <= y <= y_arr[-1])

    Returns:
    int/float: value at the given coordinates
    """
    # Ensure input is a numpy array
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # Ensure x, y are within 0 and 1
    if not x_arr[0] <= x <= x_arr[-1]:
        raise ValueError("Both x and y must be within the range")
    if not y <= y_arr[-1]:
        y -= .5
    # Convert to percentage of range
    x = (x - x_arr[0]) / (x_arr[-1]*(1+1e-6)-x_arr[0])
    y = (y - y_arr[0]) / (y_arr[-1]*(1+1e-6)-y_arr[0])
    # Calculate the indices that correspond to the x and y coordinates
    x_index = (x * (arr.shape[1]) ).astype(int)
    y_index = (y * (arr.shape[0]) ).astype(int)
    return arr[y_index, x_index]

class VCZ_tmid_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q0, Q1,
                 A_ranges,
                 Q_parks: str = None,
                 Poly_coefs: list = None,
                 Out_range: float = 5,
                 DAC_amp: float = 0.5,
                 Q0_freq:float = None,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True,
                 asymmetry: float = 0):

        super().__init__(t_start=t_start, 
                         t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.Q0 = Q0
        self.Q1 = Q1
        self.Q_parks = Q_parks
        self.ranges = A_ranges
        self.Poly_coefs = Poly_coefs
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Q0_freq = Q0_freq
        self.asymmetry = asymmetry
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
        # Sort data
        Amps_idxs = np.unique(self.raw_data_dict['data'][:,0])
        Tmid = np.unique(self.raw_data_dict['data'][:,1])
        nx, ny = len(Amps_idxs), len(Tmid)
        Amps_list = [ np.linspace(r[0], r[1], nx) for r in self.ranges ] 
        self.proc_data_dict['Amps'] = Amps_list
        self.proc_data_dict['Tmid'] = Tmid
        if self.Poly_coefs:
            P_funcs = [ np.poly1d(coefs) for coefs in self.Poly_coefs ]
            Detunings = [ P_funcs[i](Amps_list[i]*self.DAC_amp*self.Out_range/2*(1+self.asymmetry)) \
                          for i in range(len(self.Q0)) ]
            self.proc_data_dict['Detunings'] = Detunings
        for i, q0 in enumerate(self.Q0):
            CP = self.raw_data_dict['data'][:,2*i+2].reshape(ny, nx)
            MF = self.raw_data_dict['data'][:,2*i+3].reshape(ny, nx)
            self.proc_data_dict[f'CP_{i}'] = CP
            self.proc_data_dict[f'MF_{i}'] = MF
        # Fit SNZ landscapes using SNZ landscape parametrization
        if self.Poly_coefs:
            self.qoi = {}
            for i, q0 in enumerate(self.Q0):
                # Define fit function
                from scipy.optimize import curve_fit
                def fit_func(xy, tp_factor, g, delta_0, det11_02, n_dist, a, b):
                    '''
                    Fit function helper for SNZ gate landscape.
                    '''
                    delta, tmid = xy
                    tp = tp_factor/(4*g)
                    pop20, pop11, cphase = SNZ(delta, tmid, tp, g, delta_0,
                                               det11_02, n_dist, B_amp=0.5)
                    pop20_scaled = a*pop20 + b
                    outcome = pop20_scaled.ravel()
                    # outcome = cphase.ravel()
                    return outcome
                # sort fit data
                _detunings = self.proc_data_dict['Detunings'][i]
                _tmids = self.proc_data_dict['Tmid']/2.4e9
                x, y = np.meshgrid(_detunings, _tmids)
                # Multiply missing fraction by two to get population.
                mf = 2*self.proc_data_dict[f'MF_{i}']
                cp = self.proc_data_dict[f'CP_{i}']
                z = mf
                # Assess center of fit
                idx = np.argmin(np.mean(self.proc_data_dict[f'MF_{i}'], axis=0))
                mean_det = self.proc_data_dict['Detunings'][i][idx]
                # Fit data using measured missing fraction
                # We attempt fit with different initial guesses and look
                # at best fit of parameters based on measured cphase.
                Fit_params = []
                Fit_error = [] 
                fit_attempts = 11
                for k in range(fit_attempts):
                    # Fit guess of frequency
                    det11_02 = mean_det*(1 + 0.025*k -(fit_attempts-1)*0.025)
                    #            tp,    g,  delta_0, det_11_02, n_dist,   a,   b
                    p0 =      [   1, 12e6, mean_det,  det11_02,      0,   1,   0]
                    bounds = (( 0.9, 10e6,        0,         0,     -2, 0.1, -.1),
                              ( 1.1, 13e6,   np.inf,    np.inf,      2, 1.1,  .1))
                    popt, pcov = curve_fit(fit_func, (x,y), z.ravel(), p0=p0, bounds=bounds)
                    # Check if fit was successful by looking at difference
                    # between measured and predicted cphase.
                    _x, _y = np.meshgrid(_detunings, _tmids)
                    tp_factor, g, delta_0, det11_02, n_dist, _a, _b = popt
                    pop20, pop11, cphase = SNZ(_x, _y,
                                               tp = tp_factor/(4*g),
                                               g = g,
                                               delta_0 = delta_0,
                                               det11_02 = det11_02,
                                               n_dist = n_dist,
                                               B_amp=0.5)
                    avg_dif = np.mean(np.abs(cp - cphase))
                    Fit_error.append(abs(avg_dif))
                    Fit_params.append(popt)
                # Fit results with lowest error
                popt = Fit_params[np.argmin(Fit_error)]
                self.proc_data_dict[f'Fit_params_{i}'] = popt
                # Get optimal det and tmid parameters
                XY = self.proc_data_dict['Detunings'][i], self.proc_data_dict['Tmid']
                Opt_params = get_Tmid_parameters_from_SNZ_landscape(XY, popt)
                # Chose highest tmid parameter
                # This is done by penalizing distance to an integer
                # tmid value (this is relevant because of sampling).
                # We also favor higher tmids as these pulses are less 
                # impacted by distortion errors.
                def cost_func_tmid(det, tmid, tmid_max):
                    '''
                    Cost function for realizing tmids
                    '''
                    # term penalizing distance to allowed tmid
                    a = abs(np.round(tmid)-tmid)/.5
                    # term penalizing high tmids
                    b = tmid/tmid_max
                    # term penalizing actual phase at point
                    phase_at_point = get_value_at_coordinate(
                        self.proc_data_dict[f'CP_{i}'], 
                        self.proc_data_dict['Detunings'][i], 
                        self.proc_data_dict['Tmid'],
                        det, tmid)
                    c = np.abs(phase_at_point-180)/180
                    return a + b + c
                Opt_tmids = [tmid for d, tmid in Opt_params]
                _tmid_cf = [cost_func_tmid(det, tmid, Opt_params[-1][1]) for det, tmid in Opt_params]
                idx = np.argmin(_tmid_cf)
                opt_tmid = Opt_tmids[idx]
                # print(get_value_at_coordinate(cp, XY[0], XY[1], *Opt_params[idx])-180)
                # Put optimal values in last element of list (for plot purposes)
                _Opt_params = list(Opt_params)
                _Opt_params.remove(Opt_params[idx])
                _Opt_params.append(Opt_params[idx])
                # Save optimal SNZ parameters
                self.proc_data_dict[f'Opt_params_{i}'] = _Opt_params
                tp_factor, g, delta_0, det11_02, n_dist, a, b = popt
                self.qoi[f'opt_params_{i}'] = (delta_0, opt_tmid)
                self.qoi[f'tp_factor_{i}'] = tp_factor

    def prepare_plots(self):
        self.axs_dict = {}
        n = len(self.Q0)
        if n>1:
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
            self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'] = plt.figure(figsize=(8,4.25), dpi=100)
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
                'ts' : self.timestamp,
                'q0_freq' : self.Q0_freq,
                'Dets' : self.proc_data_dict['Detunings'][i] if self.Poly_coefs\
                         else None,
                'fit_params' : self.proc_data_dict[f'Fit_params_{i}'] if self.Poly_coefs\
                               else None,
                'Opt_params' : self.proc_data_dict[f'Opt_params_{i}'] if self.Poly_coefs\
                               else None,
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def find_contours(Array, value=180):
    '''
    array: 2D array on which to search
    value: values of the contours desired
    '''
    # Find points close to value
    _points = []
    for i in range(Array.shape[0]):
        idxs = np.where(np.abs(Array[i,:]-value)<0.99)
        for j in idxs[0]:
            _points.append([i,j])
    # Sort points in different contours
    _contours = [[_points[0]]]
    for point in _points[1:]:
        p_distance = Array.shape[0]
        for contour in _contours:
            for p in contour:
                _distance = np.sqrt( np.sum( (np.array(point) - np.array(p))**2 ) )
                p_distance = min(_distance, _distance)
            if p_distance < 10:
                contour.append(point)
                break
        if p_distance < 10:
            pass
        else:
            _contours.append([point])
    return _contours

def VCZ_Tmid_landscape_plotfn(
    ax, 
    Amps, Tmid, 
    CP, MF, 
    q0, q1,
    ts, n=0,
    Dets=None,
    q0_freq=None,
    fit_params=None,
    Opt_params=None,
    title=None, **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()
    # Plot leakage and conditional phase landscapes
    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        if rang:
            X = X/np.max(vals) * (rang[1]-rang[0]) + rang[0]
        return X
    # Plot versus transmon detuning
    if type(Dets) != type(None):
        X = get_plot_axis(Dets)
    # Plot versus gain
    else:
        X = get_plot_axis(Amps)
    Y = get_plot_axis(Tmid)
    a1 = axs[0+2*n].pcolormesh(X, Y, CP, cmap=hsluv_anglemap45, vmin=0, vmax=360)
    fig.colorbar(a1, ax=axs[0+2*n], label='conditional phase', ticks=[0, 90, 180, 270, 360])
    a2 = axs[1+2*n].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a2, ax=axs[1+2*n], label='missing fraction')
    # Set axis labels
    for i in range(2):
        axs[i+2*n].set_xlabel('Amplitude')
        axs[i+2*n].set_ylabel(r'$\tau_\mathrm{mid}$ (#)')
        if type(Dets) != type(None):
            set_xlabel(axs[i+2*n], f'{q0} detuning', unit='Hz')
    axs[0+2*n].set_title(f'Conditional phase')
    axs[1+2*n].set_title(f'Missing fraction')
    # Set figure title
    if title:
        fig.suptitle(title+'\n'+ts, y=1.01)
        axs[0+2*n].set_title(f'Conditional phase {q0} {q1}')
        axs[1+2*n].set_title(f'Missing fraction {q0} {q1}')
    else:
        fig.suptitle(ts+f'\nQubits {q0} {q1}', fontsize=14, y=.9)
        axs[0].set_title(f'Conditional phase')
        axs[1].set_title(f'Missing fraction')
    # Add qubit frequency axis and SNZ leakage fit contours
    if type(Dets) != type(None):
        # Add qubit frequency axis
        if q0_freq:
            axt0 = axs[0+2*n].twiny()
            axt0.set_xlim((q0_freq-np.array(axs[0+2*n].get_xlim()))*1e-9)
            axt0.set_xlabel(f'{q0} Frequency (GHz)')
            axt1 = axs[1+2*n].twiny()
            axt1.set_xlim((q0_freq-np.array(axs[1+2*n].get_xlim()))*1e-9)
            axt1.set_xlabel(f'{q0} Frequency (GHz)')
            # Plot SNZ leakage fitting contours
            _x = np.linspace(X[0], X[-1], 201)
            _y = np.linspace(Y[0], Y[-1], 201)
            _X, _Y = np.meshgrid(_x, _y)
            # Get interpolated landscape from fit
            # fit params
            tp_factor, g, delta_0, det11_02, n_dist, a, b = fit_params
            Pop20, Pop11, Cphase = SNZ(_X, _Y/2.4e9,
                                       tp = tp_factor/(4*g),
                                       g = g,
                                       delta_0 = delta_0,
                                       det11_02 = det11_02,
                                       n_dist = n_dist,
                                       B_amp=0.5
                                       )
            for i in range(2):
                # Plot leakage contours
                for c, a_ in zip([.2, .6, .8], [.7, .85, 1]):
                    axs[i+2*n].contour(_X, _Y, Pop20, [c], colors=['w'],
                                  linewidths=[1], linestyles=['--'], alpha=a_)
                # Plot 180 cphase contours
                # CS = axs[i+2*n].contour(_X, _Y, Cphase, [180], colors=['w'],
                #                    linewidths=[1.5], linestyles=['--'], alpha=1)
                # axs[i+2*n].clabel(CS, CS.levels, inline=True, fmt='$%i^\\circ$', fontsize=10)
                Contours_180 = find_contours(Cphase, value=180)   
                for c, contour in enumerate(Contours_180):
                    axs[i+2*n].plot(_x[np.array(contour)[:,1]], _y[np.array(contour)[:,0]], f'w--', lw=2)
                # Plot optimal parameters
                for opt_det, opt_tmid in Opt_params[:-1]:
                    axs[i+2*n].plot([opt_det], [opt_tmid], '*', markersize=12, 
                                    markerfacecolor='cornflowerblue', markeredgewidth=1,
                                    markeredgecolor='w', zorder=100)
                for opt_det, opt_tmid in Opt_params[-1:]:
                    axs[i+2*n].plot([opt_det], [opt_tmid], '*', markersize=12, 
                                    markerfacecolor='blue', markeredgewidth=1,
                                    markeredgecolor='w', zorder=100)
    else:
        # Plot 180 phase contours
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
            if type(Dets) != type(None):
                c['x'] = Dets[c['x']]
            else:
                c['x'] = Amps[c['x']]
            c['y'] = Tmid[c['y']]
            axs[1+2*n].plot(c['x'], c['y'], marker='', ls='--', color='white')
    fig.tight_layout()


def SNZ2(delta, tmid, tp, g, delta_0, det11_02, n_dist, B_amp):
    '''
    Function parametrizing the SNZ landscape.
    Args:
        delta : Detuning of high freq. qubit
        tp : duration of pulse
        tmid :  SNZ tmid parameter
        g : coupling of avoided crossing
        delta_0 : detuning at avoided crossing.
        det11_02 :  detuning of 11-02 levels at sweetspot
    '''
    g_rad = g*2*np.pi
    det11_02_rad = det11_02*2*np.pi
    delta_rad = delta*2*np.pi
    delta_0_rad = delta_0*2*np.pi
    delta_rad -= delta_0_rad
    # Convert B_amp to frequency detuning
    B_det_rad = (1-B_amp)*det11_02_rad
    # Frequency of Chevron oscillation
    Omega = np.sqrt(delta_rad**2+(2*g_rad)**2)
    # Population of first Chevron oscillation
    _term1 = -np.exp(+1j*Omega/2*tp)*((delta_rad+Omega)/(2*g_rad))*( -g_rad/Omega*1 )
    _term2 = -np.exp(-1j*Omega/2*tp)*((delta_rad-Omega)/(2*g_rad))*( +g_rad/Omega*1 )
    _term3 = np.exp(1j*Omega/2*tp)*( -g_rad/Omega*1 )
    _term4 = np.exp(-1j*Omega/2*tp)*( +g_rad/Omega*1 )
    c11 = _term1+_term2
    c20 = _term3+_term4
    # Population after evolving in B amp
    tB = 1/2.4e9
    c11 = c11*np.exp(1j*-B_det_rad/2*tB)
    c20 = c20*np.exp(1j*+B_det_rad/2*tB)
    # Population after evolving in the sweetspot
    # We account for pulse distortion using an offset in tmid
    t_mid_distorted = (tmid - n_dist/2.4e9)
    c11 = c11*np.exp(1j*-det11_02_rad/2*t_mid_distorted)
    c20 = c20*np.exp(1j*+det11_02_rad/2*t_mid_distorted)
    # Population after evolving in B amp
    tB = 1/2.4e9
    c11 = c11*np.exp(1j*-B_det_rad/2*tB)
    c20 = c20*np.exp(1j*+B_det_rad/2*tB)
    # Population after second Chevron
    _term1 = -np.exp(1j*Omega/2*tp)*((delta_rad+Omega)/(2*g_rad))*( c20/2*(1-delta_rad/Omega)-g_rad/Omega*c11 )
    _term2 = -np.exp(-1j*Omega/2*tp)*((delta_rad-Omega)/(2*g_rad))*( c20/2*(1+delta_rad/Omega)+g_rad/Omega*c11 )
    _term3 = np.exp(1j*Omega/2*tp)*( c20/2*(1-delta_rad/Omega)-g_rad/Omega*c11 )
    _term4 = np.exp(-1j*Omega/2*tp)*( c20/2*(1+delta_rad/Omega)+g_rad/Omega*c11 )
    c11 = _term1+_term2
    c20 = _term3+_term4
    # Calculate state populations
    pop11 = np.abs(c11)**2
    pop20 = np.abs(c20)**2
    # Calculate conditional phase
    phase11 = np.angle(c11) 
    phase20 = np.angle(c20)
    cphase = np.angle(c11) - delta_rad*tp + det11_02_rad*t_mid_distorted/2 + B_det_rad*tB
    cphase *= -1
    phase11 = np.mod(phase11*180/np.pi, 360)
    phase20 = np.mod(phase20*180/np.pi, 360)
    cphase = np.mod(cphase*180/np.pi, 360)
    return pop20, pop11, cphase

class VCZ_B_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q0,
                 Q1,
                 A_ranges,
                 directions,
                 Poly_coefs: list = None,
                 Out_range: float = 5,
                 DAC_amp: float = 0.5,
                 tmid: float = None,
                 Q0_freq:float = None,
                 Q_parks: str = None,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True,
                 asymmetry: float = 0):

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
        self.Poly_coefs = Poly_coefs
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Q0_freq = Q0_freq
        self.tmid = tmid
        self.asymmetry = asymmetry
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
        if self.Poly_coefs:
            P_funcs = [ np.poly1d(coefs) for coefs in self.Poly_coefs ]
            Detunings = [ P_funcs[i](Amps_list[i]*self.DAC_amp*self.Out_range/2*(1+self.asymmetry)) \
                          for i in range(len(self.Q0)) ]
            self.proc_data_dict['Detunings'] = Detunings
        # Calculate cost function to find optimal
        # parameters of amplitude and B amp
        def cost_function(CP, MF,
                          phase=180,
                          cp_coef=1, l1_coef=1):
            '''
            Cost function for minimizing cphase
            error and leakage simultaneously.
            '''
            A = ((np.abs(CP)-180)/180)**2
            B = ((MF-np.min(MF))/.5)**2
            C = (np.mean(MF-np.min(MF), axis=0)/.5)**2
            return cp_coef*A + l1_coef*(B+C)
        for i, q0 in enumerate(self.Q0):
            CP = self.raw_data_dict['data'][:,2*i+2].reshape(ny, nx)
            MF = self.raw_data_dict['data'][:,2*i+3].reshape(ny, nx)
            CF = cost_function(CP, MF)
            # Find minimum of cost function
            idxs_min = np.unravel_index(np.argmin(CF), CF.shape)
            A_min, B_min = Amps_list[i][idxs_min[1]], Bamps[idxs_min[0]]
            CP_min, L1_min = CP[idxs_min], MF[idxs_min]/2
            if self.Poly_coefs:
                Det_min = Detunings[i][idxs_min[1]]
                self.qoi[f'Optimal_det_{q0}'] = Det_min
            # Save quantities of interest
            self.proc_data_dict[f'CP_{i}'] = CP
            self.proc_data_dict[f'MF_{i}'] = MF
            self.proc_data_dict[f'CF_{i}'] = CF
            self.qoi[f'Optimal_amps_{q0}'] = A_min, B_min
            self.qoi[f'Gate_perf_{q0}'] = CP_min, L1_min
        # Fit SNZ landscapes using SNZ 
        # landscape parametrization
        if self.Poly_coefs:
            for i, q0 in enumerate(self.Q0):
                # Define fit function
                from scipy.optimize import curve_fit
                def fit_func(xy, tp_factor, tmid, g, delta_0, det11_02, n_dist, a, b):
                    '''
                    Fit function helper for SNZ gate landscape.
                    '''
                    delta, bamp = xy
                    tp = tp_factor/(4*g)
                    pop20, pop11, cphase = SNZ2(delta, tmid, tp, g, delta_0, det11_02, n_dist, B_amp=bamp)
                    outcome = a*pop20 + b
                    return outcome.ravel()
                # sort fit data
                _detunings = self.proc_data_dict['Detunings'][i]
                _Bamps = self.proc_data_dict['Bamps']
                x, y = np.meshgrid(_detunings, _Bamps)
                # Multiply missing fraction by two to get population.
                z = 2*self.proc_data_dict[f'MF_{i}']
                # initial fit guess
                #    tp_factor,      tmid,    g,            delta_0,              det_11_02, n_dist,   a,   b
                p0 = [       1, self.tmid, 12e6, np.mean(Detunings), np.mean(Detunings)*1.1,     .5,   1,   0]
                bounds = ((0.9,         0, 10e6,                  0,                      0,      0, 0.1, -.1),
                          (1.1,  12/2.4e9, 13e6,             np.inf,                 np.inf,      2, 1.1,  .1))
                popt, pcov = curve_fit(fit_func, (x,y), z.ravel(), p0=p0, bounds=bounds)
                self.proc_data_dict[f'Fit_params_{i}'] = popt
                self.qoi[f'tp_factor_{i}'] = popt[0]

    def prepare_plots(self):
        self.axs_dict = {}

        n = len(self.Q0)
        if n>1:
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
            if self.Poly_coefs:
                fig = plt.figure(figsize=(13,4), dpi=100)
            else:
                fig = plt.figure(figsize=(15,4), dpi=100)
            self.figs[f'VCZ_landscape_{q0}_{self.Q1[i]}'] = fig
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
                'ts' : self.timestamp,
                'gate_perf' : self.qoi[f'Gate_perf_{q0}'],
                'direction' : self.directions[i][0],
                'q0_freq' : self.Q0_freq,
                'Dets' : self.proc_data_dict['Detunings'][i] if self.Poly_coefs\
                         else None,
                'opt' : (self.qoi[f'Optimal_det_{q0}'], self.qoi[f'Optimal_amps_{q0}'][1])\
                        if self.Poly_coefs else self.qoi[f'Optimal_amps_{q0}'],
                'fit_params' : self.proc_data_dict[f'Fit_params_{i}'] if self.Poly_coefs\
                               else None,
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
                'ts' : self.timestamp,
                'gate_perf' : self.qoi[f'Gate_perf_{q0}'],
                'direction' : self.directions[i][0],
                'q0_freq' : self.Q0_freq,
                'Dets' : self.proc_data_dict['Detunings'][i] if self.Poly_coefs\
                         else None,
                'opt' : (self.qoi[f'Optimal_det_{q0}'], self.qoi[f'Optimal_amps_{q0}'][1])\
                        if self.Poly_coefs else self.qoi[f'Optimal_amps_{q0}'],
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
    opt,
    Dets=None,
    q0_freq=None,
    direction=None,
    fit_params=None,
    n=0, title=None, **kw):

    fig = ax.get_figure()
    axs = fig.get_axes()
    # Plot leakage and conditional phase landscapes
    def get_plot_axis(vals, rang=None):
        dx = vals[1]-vals[0]
        X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
        if rang:
            X = X/np.max(vals) * (rang[1]-rang[0]) + rang[0]
        return X
    # Plot versus transmon detuning
    if type(Dets) != type(None):
        X = get_plot_axis(Dets)
    # Plot versus gain
    else:
        X = get_plot_axis(Amps)
    Y = get_plot_axis(Bamps)
    a1 = axs[0+3*n].pcolormesh(X, Y, CP, cmap=hsluv_anglemap45, vmin=0, vmax=360)
    fig.colorbar(a1, ax=axs[0+3*n], label='conditional phase', ticks=[0, 90, 180, 270, 360])
    a2 = axs[1+3*n].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a2, ax=axs[1+3*n], label='missing fraction')
    a3 = axs[2+3*n].pcolormesh(X, Y, CF, cmap='viridis',
                   norm=LogNorm(vmin=CF.min(), vmax=CF.max()))
    fig.colorbar(a3, ax=axs[2+3*n], label='cost function')
    # Plot gate parameters and metrics
    text_str = 'Optimal parameters\n'+\
               f'gate: {q0} CZ_{direction}\n'+\
               f'$\\phi$: {gate_perf[0]:.2f} \t $L_1$: {gate_perf[1]*100:.1f}%\n'
    if type(Dets) != type(None):
        text_str += f'Detuning: {opt[0]*1e-6:.1f}MHz\n'
    else:
        text_str += f'A amp: {opt[0]:.4f}\n'
    text_str += f'B amp: {opt[1]:.4f}'
    # Add fit params
    if type(fit_params) != type(None):
        tp_factor, tmid, g, delta_0, det11_02, n_dist, a, b = fit_params
        text_str += '\nFit params'
        text_str += f'\n$t_p^\\mathrm{{factor}}$: {tp_factor:.3f}'
        text_str += f'\n$t_\\mathrm{{mid}}$: {tmid*2.4e9:.3f} (#)'
        text_str += f'\n$J_2/2\\pi$: {g*1e-6:.2f} MHz'

    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[2+3*n].text(1.45, 0.98, text_str, transform=axs[2+3*n].transAxes, fontsize=10,
            verticalalignment='top', bbox=props, linespacing=1.6)
    # Set axis labels and titles
    for i in range(3):
        axs[i+3*n].plot(opt[0], opt[1], 'o', mfc='white', mec='grey', mew=.5)
        axs[i+3*n].set_xlabel('Amplitude')
        axs[i+3*n].set_ylabel(r'B amplitude')
        if type(Dets) != type(None):
            set_xlabel(axs[i+3*n], f'{q0} detuning', unit='Hz')
    if title:
        fig.suptitle(title+'\n'+ts, y=1)
        axs[0+3*n].set_title(f'Conditional phase {q0} {q1}')
        axs[1+3*n].set_title(f'Missing fraction {q0} {q1}')
        axs[2+3*n].set_title(f'Cost function {q0} {q1}')
    else:
        fig.suptitle(ts+f'\nQubits {q0} {q1}', y=.95, size=14)
        axs[0].set_title(f'Conditional phase')
        axs[1].set_title(f'Missing fraction')
        axs[2].set_title(f'Cost function')
    # Add qubit frequency axis and SNZ leakage fit contours
    if type(Dets) != type(None):
        # Add qubit frequency axis
        axt0 = axs[0+3*n].twiny()
        axt0.set_xlim((q0_freq-np.array(axs[0+3*n].get_xlim()))*1e-9)
        axt0.set_xlabel(f'{q0} Frequency (GHz)')
        axt1 = axs[1+3*n].twiny()
        axt1.set_xlim((q0_freq-np.array(axs[1+3*n].get_xlim()))*1e-9)
        axt1.set_xlabel(f'{q0} Frequency (GHz)')
        axt2 = axs[2+3*n].twiny()
        axt2.set_xlim((q0_freq-np.array(axs[2+3*n].get_xlim()))*1e-9)
        axt2.set_xlabel(f'{q0} Frequency (GHz)')
        # # This fit is not accurate !
        # # Plot SNZ leakage fitting contours
        # _X = np.linspace(X[0], X[-1], 201)
        # _Y = np.linspace(Y[0], Y[-1], 201)
        # _X, _Y = np.meshgrid(_X, _Y)
        # # Get interpolated landscape from fit
        # # fit params
        # # print(fit_params)
        # tp_factor, tmid, g, delta_0, det11_02, n_dist, a, b = fit_params
        # Pop20, Pop11, Cphase = SNZ2(delta=_X, B_amp=_Y,
        #                             tp=tp_factor/(4*g),
        #                             tmid=tmid,
        #                             g=g,
        #                             delta_0=delta_0,
        #                             det11_02=det11_02,
        #                             n_dist=n_dist)
        # for i in range(2):
        #     # Plot leakage contours
        #     for c, a_ in zip([.05, .2, .6, .8], [.5, .7, .85, 1]):
        #         axs[i+2*n].contour(_X, _Y, Pop20, [c], colors=['w'],
        #                       linewidths=[1], linestyles=['--'], alpha=a_)
        #     # # Plot 180 cphase contours
        #     # CS = axs[i+2*n].contour(_X, _Y, Cphase, [180], colors=['w'],
        #     #                    linewidths=[1.5], linestyles=['--'], alpha=1)
        #     # axs[i+2*n].clabel(CS, CS.levels, inline=True, fmt='$%i^\\circ$', fontsize=10)

    # Plot 180 cphase contour
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
    if type(Dets) != type(None):
        _x_axis = Dets
    else:
        _x_axis = Amps
    cs = axs[1+3*n].contour(_x_axis, Bamps, AUX, levels=[180, 180+360],
                        colors='white', linestyles='--')
    # axs[1+3*n].clabel(cs, inline=True, fontsize=10, fmt='$180^o$')
    fig.tight_layout()

def VCZ_L1_contour_plotfn(
    ax,
    Amps, Bamps,
    CP, MF, CF,
    q0, q1, ts,
    gate_perf,
    opt, direction=None,
    q0_freq=None, Dets=None,
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
    # Plot versus transmon detuning
    if type(Dets) != type(None):
        X = get_plot_axis(Dets)
    # Plot versus gain
    else:
        X = get_plot_axis(Amps)
    Y = get_plot_axis(Bamps)
    a1 = axs[0].pcolormesh(X, Y, MF, cmap='hot')
    fig.colorbar(a1, ax=axs[0], label='missing fraction')
    if type(Dets) != type(None):
        _x_axis = Dets
    else:
        _x_axis = Amps
    cs = axs[0].contour(_x_axis, Bamps, AUX, levels=[180, 180+360, 180+720],
                        colors='white', linestyles='--')
    # axs[0].clabel(cs, inline=True, fontsize=10, fmt='$180^o$')
    # Plot optimal points
    axs[0].plot(opt[0], opt[1], 'o', mfc='white', mec='grey', mew=.5)
    axs[1].axvline(opt[0], color='k', ls='--', alpha=.5)
    axs[1].plot(_x_axis[idxs[0]], MF[idxs][::-1]/2*100)
    # Set axis label and title
    axs[0].set_xlabel('Amplitude')
    axs[1].set_xlabel('Amplitude')
    if type(Dets) != type(None):
        set_xlabel(axs[0], f'{q0} detuning', unit='Hz')
        set_xlabel(axs[1], f'{q0} detuning', unit='Hz')
    axs[0].set_ylabel(r'B amplitude')
    axs[1].set_ylabel(r'$L_1$ (%)')
    # Add qubit frequency axis
    if type(Dets) != type(None):
        # Add qubit frequency axis
        if q0_freq:
            axt0 = axs[0].twiny()
            axt0.set_xlim((q0_freq-np.array(axs[0].get_xlim()))*1e-9)
            axt0.set_xlabel(f'{q0} Frequency (GHz)')
            axt1 = axs[1].twiny()
            axt1.set_xlim((q0_freq-np.array(axs[1].get_xlim()))*1e-9)
            axt1.set_xlabel(f'{q0} Frequency (GHz)')
    # Set title
    fig.suptitle(ts+f'\nQubits {q0} {q1}', y=.9, size=14)
    axs[0].set_title(f'Missing fraction')
    axs[1].set_title(f'$L_1$ along contour')
    fig.tight_layout()


class VCZ_flux_offset_sweep_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
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

        self.Q0 = self.raw_data_dict['folder'].split('_')[-3]
        self.Q1 = self.raw_data_dict['folder'].split('_')[-2]
        self.Q_parks = eval(self.raw_data_dict['folder'].split('_')[-1])

    def process_data(self):
        self.proc_data_dict = {}
        # Sort data
        Offset = self.raw_data_dict['data'][:,0]
        self.proc_data_dict['Offset'] = Offset
        CP = self.raw_data_dict['data'][:,1]
        MF = self.raw_data_dict['data'][:,2]
        self.proc_data_dict[f'CP'] = CP
        self.proc_data_dict[f'MF'] = MF
        # Fit data
        self.qoi = {}
        p_coef = np.polyfit(Offset, self.proc_data_dict[f'MF'], deg=2)
        # Find minimum of leakage using derivative
        p_func = np.poly1d(p_coef)
        crit = p_func.deriv().roots
        r_crit = crit[crit.imag==0].real
        opt_offset = r_crit[0]
        self.proc_data_dict[f'p_coef'] = p_coef
        self.qoi[f'offset_opt'] = opt_offset

    def prepare_plots(self):
        self.axs_dict = {}
        self.figs[f'Offset_sweep_{self.Q0}_{self.Q1}'] = plt.figure(figsize=(8,3), dpi=100)
        # self.figs[f'Offset_sweep_{self.Q0}_{self.Q1}'].patch.set_alpha(0)
        axs = [self.figs[f'Offset_sweep_{self.Q0}_{self.Q1}'].add_subplot(121),
               self.figs[f'Offset_sweep_{self.Q0}_{self.Q1}'].add_subplot(122)]
        self.axs_dict[f'conditional_phase'] = axs[0]
        self.axs_dict[f'missing_fraction'] = axs[1]
        self.plot_dicts[f'Offset_sweep_{self.Q0}_{self.Q1}']={
            'plotfn': Offset_sweep_plotfn,
            'ax_id': f'conditional_phase',
            'Offset' : self.proc_data_dict['Offset'], 
            'CP' : self.proc_data_dict[f'CP'], 
            'MF' : self.proc_data_dict[f'MF'],
            'p_coef' : self.proc_data_dict[f'p_coef'],
            'opt_offset' : self.qoi[f'offset_opt'],
            'q0' : self.Q0, 'q1' : self.Q1,
            'timestamp' : self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Offset_sweep_plotfn(
    ax,
    Offset,
    CP, MF,
    p_coef,
    opt_offset,
    q0, q1,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    axs[0].plot(Offset*1e3, CP, 'o')
    axs[0].set_xlabel('Current offset (mA)')
    axs[0].set_ylabel('Conditional phase (deg)')
    lim = axs[0].get_ylim()
    axs[0].set_ylim(lim[0]-10, lim[1]+10)

    p_func = np.poly1d(p_coef)
    _offset = np.linspace(Offset[0], Offset[-1], 101)
    axs[1].plot(_offset*1e3, p_func(_offset), 'C0--', label='Fit')
    axs[1].plot(Offset*1e3, MF, 'C3o', label='data')
    axs[1].axvline(opt_offset*1e3, color='k', ls='--', label=f'{opt_offset*1e3:.3f} mA')
    axs[1].set_xlabel('Current offset (mA)')
    axs[1].set_ylabel('Missing fration')
    axs[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    fig.suptitle(f'{timestamp}\nFlux offset sweep {q0} {q1}', y=1.0)
    fig.tight_layout()


class VCZ_asymmetry_sweep_Analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
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

        self.Q0 = eval(self.raw_data_dict['folder'].split('_')[-3])
        self.Q1 = eval(self.raw_data_dict['folder'].split('_')[-2])
        self.Q_parks = eval(self.raw_data_dict['folder'].split('_')[-1])

    def process_data(self):
        self.proc_data_dict = {}
        # Sort data
        Asymmetry = self.raw_data_dict['data'][:,0]
        self.proc_data_dict['Asymmetry'] = Asymmetry
        for i, q0 in enumerate(self.Q0):
            CP = self.raw_data_dict['data'][:,2*i+1]
            MF = self.raw_data_dict['data'][:,2*i+2]
            self.proc_data_dict[f'CP_{i}'] = CP
            self.proc_data_dict[f'MF_{i}'] = MF
        # Fit data
        self.qoi = {}
        for i, q0 in enumerate(self.Q0):
            p_coef = np.polyfit(Asymmetry, 
                        self.proc_data_dict[f'MF_{i}'], deg=2)
            # Find minimum of leakage using derivative
            p_func = np.poly1d(p_coef)
            crit = p_func.deriv().roots
            r_crit = crit[crit.imag==0].real
            opt_asymmetry = r_crit[0]
            self.proc_data_dict[f'p_coef_{i}'] = p_coef
            self.qoi[f'asymmetry_opt_{i}'] = opt_asymmetry

    def prepare_plots(self):
        self.axs_dict = {}

        for i, q0 in enumerate(self.Q0):
            self.figs[f'Asymmetry_sweep_{q0}_{self.Q1[i]}'] = plt.figure(figsize=(8,3), dpi=100)
            # self.figs[f'Asymmetry_sweep_{q0}_{self.Q1[i]}'].patch.set_alpha(0)
            axs = [self.figs[f'Asymmetry_sweep_{q0}_{self.Q1[i]}'].add_subplot(121),
                   self.figs[f'Asymmetry_sweep_{q0}_{self.Q1[i]}'].add_subplot(122)]
            self.axs_dict[f'conditional_phase_{i}'] = axs[0]
            self.axs_dict[f'missing_fraction_{i}'] = axs[1]
            self.plot_dicts[f'Asymmetry_sweep_{self.Q0[i]}_{self.Q1[i]}']={
                'plotfn': Asymmetry_sweep_plotfn,
                'ax_id': f'conditional_phase_{i}',
                'Asymmetry' : self.proc_data_dict['Asymmetry'], 
                'CP' : self.proc_data_dict[f'CP_{i}'], 
                'MF' : self.proc_data_dict[f'MF_{i}'],
                'p_coef' : self.proc_data_dict[f'p_coef_{i}'],
                'opt_asymmetry' : self.qoi[f'asymmetry_opt_{i}'],
                'q0' : self.Q0[i], 'q1' : self.Q1[i],
                'timestamp' : self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Asymmetry_sweep_plotfn(
    ax,
    Asymmetry,
    CP, MF,
    p_coef,
    opt_asymmetry,
    q0, q1,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    axs[0].plot(Asymmetry*100, CP, 'o')
    axs[0].set_xlabel('Pulse asymmetry (%)')
    axs[0].set_ylabel('Conditional phase (deg)')
    lim = axs[0].get_ylim()
    axs[0].set_ylim(lim[0]-10, lim[1]+10)

    p_func = np.poly1d(p_coef)
    _asym = np.linspace(Asymmetry[0], Asymmetry[-1], 101)
    axs[1].plot(_asym*100, p_func(_asym), 'C0--', label='Fit')
    axs[1].plot(Asymmetry*100, MF, 'C3o', label='data')
    axs[1].axvline(opt_asymmetry*100, color='k', ls='--', label=f'${opt_asymmetry*100:.3f}$%')
    axs[1].set_xlabel('Pulse asymmetry, (%)')
    axs[1].set_ylabel('Missing fration')
    axs[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc=2)

    fig.suptitle(f'{timestamp}\nAsymmetry sweep {q0} {q1}', y=1.0)
    fig.tight_layout()


def avoided_crossing_fit_func(x, alpha, J):
    x_rad = x*2*np.pi
    J_rad = J*2*np.pi
    alpha_rad = alpha*2*np.pi
    w_err = 2*J_rad**2/(x_rad-alpha_rad)
    # rad_err = np.pi*w_err/(2*np.sqrt(2)*J_rad)
    rad_err = w_err/(2*J_rad)
    deg_err = rad_err*180/np.pi
    return np.mod(deg_err+180 , 360) - 180

class Park_frequency_sweep_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 qH: str,
                 qL: str,
                 qP: str,
                 Parking_distances: list,
                 freq_qH: float = None,
                 alpha_qH: float = None,
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
        self.qH = qH
        self.qL = qL
        self.qP = qP
        self.Parking_distances = Parking_distances
        self.alpha_qH = alpha_qH
        self.freq_qH = freq_qH
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
        # Sort data
        Amps = self.raw_data_dict['data'][:,0]
        # qH single qubit phases with qP in 0 or 1 ("_s") 
        Phi   = self.raw_data_dict['data'][:,1]
        Phi_s = self.raw_data_dict['data'][:,2]
        Delta_phi = self.raw_data_dict['data'][:,3]
        # Conditional phases between qH and qL with qP in 0 or 1 ("_s")
        Phi_cond   = self.raw_data_dict['data'][:,4]
        Phi_cond_s = self.raw_data_dict['data'][:,5]
        Delta_phi_cond = self.raw_data_dict['data'][:,6]
        # Missing fraction of qL with qP in 0 or 1 ("_s")
        Miss_frac   = self.raw_data_dict['data'][:,7]
        Miss_frac_s = self.raw_data_dict['data'][:,8]
        Delta_miss_frac = self.raw_data_dict['data'][:,9]
        # Fit avoided crossing
        from scipy.optimize import curve_fit
        _x = self.Parking_distances[30:]*1+0
        _y = Delta_phi_cond[30:]*1+0
        p0 = [600e6, 20e6, 20e6]
        popt, pcov = curve_fit(avoided_crossing_fit_func, _x, _y,
                               p0 = p0, bounds=([ _x[0],  5e6,  5e6],
                                                [_x[-1], 50e6, 50e6]))
        print(pcov)
        print(popt)
        # popt = p0
        self.proc_data_dict['popt'] = popt
        # Save data in processed data dict
        self.proc_data_dict['Phi'] = Phi
        self.proc_data_dict['Phi_s'] = Phi_s
        self.proc_data_dict['Delta_phi'] = Delta_phi
        self.proc_data_dict['Phi_cond'] = Phi_cond
        self.proc_data_dict['Phi_cond_s'] = Phi_cond_s
        self.proc_data_dict['Delta_phi_cond'] = Delta_phi_cond
        self.proc_data_dict['Miss_frac'] = Miss_frac
        self.proc_data_dict['Miss_frac_s'] = Miss_frac_s
        self.proc_data_dict['Delta_miss_frac'] = Delta_miss_frac

    def prepare_plots(self):
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(5,5), nrows=2, ncols=2, dpi=100)
        axs = axs.flatten()
        self.figs[f'Park_sweep_gate_{self.qH}_{self.qL}_park_{self.qP}'] = fig
        self.axs_dict['plot_1'] = axs[0]
        # fig.patch.set_alpha(0)
        self.plot_dicts[f'Park_sweep_gate_{self.qH}_{self.qL}_park_{self.qP}']={
                'plotfn': park_sweep_plotfn,
                'ax_id': 'plot_1',
                'qH': self.qH,
                'qL': self.qL,
                'qP': self.qP,
                'Parking_distances': self.Parking_distances,
                'Phi' : self.proc_data_dict['Phi'],
                'Phi_s' : self.proc_data_dict['Phi_s'],
                'Delta_phi' : self.proc_data_dict['Delta_phi'],
                'Phi_cond' : self.proc_data_dict['Phi_cond'],
                'Phi_cond_s' : self.proc_data_dict['Phi_cond_s'],
                'Delta_phi_cond' : self.proc_data_dict['Delta_phi_cond'],
                'Miss_frac' : self.proc_data_dict['Miss_frac'],
                'Miss_frac_s' : self.proc_data_dict['Miss_frac_s'],
                'Delta_miss_frac' : self.proc_data_dict['Delta_miss_frac'],
                'alpha_qH': self.alpha_qH,
                'popt': self.proc_data_dict['popt'],
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
    qH, qL, qP,
    Parking_distances,
    Phi, Phi_s, Delta_phi,
    Phi_cond, Phi_cond_s, Delta_phi_cond,
    Miss_frac, Miss_frac_s, Delta_miss_frac,
    timestamp, alpha_qH, popt,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Plot of single-qubit phase of qH
    axs[0].plot(Parking_distances*1e-6, Phi_cond, 'C0.')
    if alpha_qH:
        axs[0].axvline(-alpha_qH*1e-6, ls='--', color='k', lw=1)
        axs[0].text(-alpha_qH*1e-6, 180, f'$-\\alpha_{{{qH}}}$',
                    va='center', ha='center', size=8,
                    bbox=dict(boxstyle='round', facecolor='w', alpha=1, lw=0))
    axs[0].set_ylim(-90+180, 90+180)
    axs[0].set_ylabel(f'$\\phi_\\mathrm{{cond}}^\\mathrm{{{qH},{qL}}}$ (deg)')
    axs[0].axhline(180, ls='--', color='k', lw=1, alpha=.25, zorder=10)
    # Plot of qH-qL conditional phase
    axs[2].plot(Parking_distances*1e-6, Delta_phi, 'C0.')
    if alpha_qH:
        axs[2].axvline(-alpha_qH*1e-6, ls='--', color='k', lw=1)
        axs[2].text(-alpha_qH*1e-6, 0, f'$-\\alpha_{{{qH}}}$',
                    va='center', ha='center', size=8,
                    bbox=dict(boxstyle='round', facecolor='w', alpha=1, lw=0))
    axs[2].set_ylim(-90, 90)
    axs[2].set_ylabel(f'$\\delta \\phi_\\mathrm{{{qH}}}$ (deg)')
    axs[2].set_xlabel(f'$\\Delta_\\mathrm{{{qH},{qP}}}$ (MHz)')
    axs[2].axhline(0, ls='--', color='k', lw=1, alpha=.25, zorder=10)
    # Plot of qH-qL conditional phase difference for different qP states
    axs[1].plot(Parking_distances*1e-6, 
                avoided_crossing_fit_func(Parking_distances, *popt), 'k--')
    axs[1].plot(Parking_distances*1e-6, Delta_phi_cond, 'C0.')
    axs[1].set_ylim(-90, 90)
    axs[1].set_ylabel('$\\delta \\phi_\\mathrm{cond}}$ (deg)')
    axs[1].axhline(0, ls='--', color='k', lw=1, alpha=.25, zorder=10)
    # Plot of Missing fractions
    axs[3].plot(Parking_distances*1e-6, Miss_frac/2, 'C0-', alpha=.25, label='$L_{{1_{{|0\\rangle_P}}}}$')
    axs[3].plot(Parking_distances*1e-6, Miss_frac_s/2, 'C3-', alpha=.25, label='$L_{{1_{{|1\\rangle_P}}}}$')
    axs[3].plot(Parking_distances*1e-6, np.abs(Delta_miss_frac)/2, 'C0.')
    axs[3].set_xlabel(f'$\\Delta_\\mathrm{{{qH},{qP}}}$ (MHz)')
    axs[3].set_ylabel('$|\\delta L_1|$')
    axs[3].legend(frameon=False)
    # twin axes for qL-qP detuning
    ax0 = axs[0].twiny()
    ax0.set_xlim(np.array(axs[0].get_xlim())-300)
    ax0.set_xlabel(f'$\\Delta_\\mathrm{{{qL},{qP}}}$ (MHz)')
    ax1 = axs[1].twiny()
    ax1.set_xlim(np.array(axs[1].get_xlim())-300)
    ax1.set_xlabel(f'$\\Delta_\\mathrm{{{qL},{qP}}}$ (MHz)')
    ax2 = axs[2].twiny()
    ax2.set_xlim(np.array(axs[2].get_xlim()))
    ax2.set_xticklabels([])
    ax3 = axs[3].twiny()
    ax3.set_xlim(np.array(axs[3].get_xlim()))
    ax3.set_xticklabels([])
    # Adjust positions of axis
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0, pos.y0, pos.width, pos.height])
    pos = axs[1].get_position()
    axs[1].set_position([pos.x0+.1, pos.y0, pos.width, pos.height])
    pos = axs[2].get_position()
    axs[2].set_position([pos.x0, pos.y0+.02, pos.width, pos.height])
    pos = axs[3].get_position()
    axs[3].set_position([pos.x0+.1, pos.y0+.02, pos.width, pos.height])
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    # Drawing of two-qubit gate scheme
    from matplotlib.patches import Circle
    ax = fig.add_subplot(221)
    pos = ax.get_position()
    ax.set_position([pos.x0+pos.width*(1-.425*1.1-.05), pos.y0+pos.height*(1-.45*1.1+.03),
                     pos.width*.425*1.1, pos.height*.45*1.1])
    patch = Circle((0, 0.5), radius=.3, color='C0', lw=1, ec='k')
    ax.add_patch(patch)
    patch = Circle((0.75, -0.5), radius=.3, color='C0', lw=1, ec='k')
    ax.add_patch(patch)
    patch = Circle((-0.75, -0.5), radius=.3, color='C3', lw=1, ec='k')
    ax.add_patch(patch)
    ax.plot([0, .75], [.5, -.5], c='k', zorder=-1, lw=3)
    ax.plot([0, -.75], [.5, -.5], c='k', zorder=-1, lw=3, ls=(.1,(1,.5)), alpha=.5)
    ax.text(0, .5, qH, va='center', ha='center', color='w')
    ax.text(.75, -.5, qL, va='center', ha='center', color='w')
    ax.text(-.75, -.5, qP, va='center', ha='center', color='w')
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.axis('off')
    # Title
    fig.suptitle(f'{timestamp}\nPark sweep {qP} gate {qH},{qL}', y=1.075)


def convert_amp_to_freq(poly_coefs, ch_range, ch_amp, dac_amp):
    '''
    Helper function to convert flux pulse amp to frequency detuning.
    '''
    poly_func = np.poly1d(poly_coefs)
    out_volt = dac_amp*ch_amp*ch_range/2
    freq_det = poly_func(out_volt)
    return freq_det

def vcz_waveform(sampling_rate,
                 amp_at_int_11_02,
                 norm_amp_fine,
                 amp_pad,
                 asymmetry,
                 time_sqr,
                 time_middle,
                 time_pad,
                 use_asymmety,
                 use_net_zero_pulse
    ):
    '''
    Trace SNZ waveform.
    '''
    amp_at_sweetspot = 0.0
    dt = 1
    norm_amp_sq = 1
    time_sqr = time_sqr * sampling_rate
    time_middle = time_middle * sampling_rate
    time_pad = time_pad * sampling_rate
    # This is to avoid numerical issues when the user would run sweeps with
    # e.g. `time_at_swtspt = np.arange(0/2.4e9, 10/ 2.4e9, 2/2.4e9)`
    # instead of `time_at_swtspt = np.arange(0, 42, 2) / 2.4e9` and get
    # bad results for specific combinations of parameters
    time_middle = np.round(time_middle / dt) * dt
    time_sqr = np.round(time_sqr / dt) * dt
    time_pad = np.round(time_pad / dt) * dt
    # build padding part of waveform
    pad_amps = np.full(int(time_pad / dt), 0) + amp_pad*2
    for _i in range(len(pad_amps)):
        if _i<12:
            pad_amps[_i] = 0
    # pad_amps = np.full(int(time_pad / dt), 0)
    sq_amps = np.full(int(time_sqr / dt), norm_amp_sq)
    amps_middle = np.full(int(time_middle / dt), amp_at_sweetspot)
    # build asymmetric SNZ amplitudes
    if use_asymmety:
        norm_amp_pos = 1+asymmetry
        norm_amp_neg = 1-asymmetry
    else:
        norm_amp_pos = 1
        norm_amp_neg = 1
    pos_sq_amps = np.full(int(time_sqr / dt), norm_amp_pos)
    neg_sq_amps = np.full(int(time_sqr / dt), norm_amp_neg)
    # slope amp will be using the same scaling factor as in the symmetric case, 
    # but relative to pos and neg amplitudes 
    # such that this amp is in the range [0, 1]
    slope_amp_pos = np.array([norm_amp_fine * norm_amp_pos])
    slope_amp_neg = np.array([norm_amp_fine * norm_amp_neg])
    pos_NZ_amps = np.concatenate((pos_sq_amps, slope_amp_pos))
    neg_NZ_amps = np.concatenate((slope_amp_neg, neg_sq_amps))
    amp = np.concatenate(
        ([amp_at_sweetspot],
        pad_amps,
        pos_NZ_amps,
        amps_middle,
        (1-use_net_zero_pulse*2)*neg_NZ_amps,
        pad_amps[::-1],
        [amp_at_sweetspot])
    )
    amp = amp_at_int_11_02 * amp
    tlist = np.cumsum(np.full(len(amp) - 1, dt))
    tlist = np.concatenate(([0.0], tlist))  # Set first point to have t=0
    return amp

def gen_park(sampling_rate, park_length, park_pad_length, park_amp,
             park_double_sided):
    '''
    Trace parking waveform.
    '''
    zeros = np.zeros(int(park_pad_length * sampling_rate))
    if park_double_sided:
        ones = np.ones(int(park_length * sampling_rate / 2))
        pulse_pos = park_amp * ones
        return np.concatenate((zeros, pulse_pos, - pulse_pos, zeros))
    else:
        pulse_pos = park_amp*np.ones(int(park_length*sampling_rate))
        return np.concatenate((zeros, pulse_pos, zeros))

class TwoQubitGate_frequency_trajectory_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Qubit_pairs: list,
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
        self.Qubit_pairs = Qubit_pairs
        self.Qubits = np.unique(Qubit_pairs)
        for qH, qL in Qubit_pairs:
            Q_parks = get_parking_qubits(qH, qL)
            for q in Q_parks:
                if not (q in self.Qubits):
                    self.Qubits = np.concatenate((self.Qubits, [q]))

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        '''
        Extract all relevant waveform information to trace
        frequency trajectory of qubit during two-qubit gate.
        '''
        data = {q:{} for q in self.Qubits}
        for q in self.Qubits:
            # Qubit parameters
            param_spec =  {'poly_coefs': (f'Instrument settings/flux_lm_{q}', 'attr:q_polycoeffs_freq_01_det'),
                           'ch_range': (f'Instrument settings/flux_lm_{q}', 'attr:cfg_awg_channel_range'),
                           'ch_amp': (f'Instrument settings/flux_lm_{q}', 'attr:cfg_awg_channel_amplitude'),
                           'frequency': (f'Instrument settings/{q}', f'attr:freq_qubit'),
                           'anharmonicity': (f'Instrument settings/{q}', f'attr:anharmonicity')}
            # Gate parameters
            for d in ['NW', 'NE', 'SW', 'SE']:
                # Amplitudes
                param_spec[f'amp_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_amp_dac_at_11_02_{d}')
                param_spec[f'B_amp_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_amp_fine_{d}')
                param_spec[f'asymmetry_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_asymmetry_{d}')
                param_spec[f'amp_pad_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_amp_pad_{d}')
                param_spec[f'use_asymmetry_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_use_asymmetric_amp_{d}')
                param_spec[f'use_net_zero_pulse_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_use_net_zero_pulse_{d}')
                # Durations
                param_spec[f'tp_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_time_single_sq_{d}')
                param_spec[f'tmid_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_time_middle_{d}')
                param_spec[f'tpad_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_time_pad_{d}')
            # Park parameters
            param_spec['park_double_sided'] = (f'Instrument settings/flux_lm_{q}', f'attr:park_double_sided')
            param_spec['park_amp'] = (f'Instrument settings/flux_lm_{q}', f'attr:park_amp')
            param_spec['t_park'] = (f'Instrument settings/flux_lm_{q}', f'attr:park_length')
            param_spec['tpad_park'] = (f'Instrument settings/flux_lm_{q}', f'attr:park_pad_length')
            # extract data
            data[q] = h5d.extract_pars_from_datafile(data_fp, param_spec)
            # Sort and parse extracted quantities
            p_coefs = data[q]['poly_coefs'][1:-1].split(' ')
            while '' in p_coefs:
                p_coefs.remove('')
            data[q]['poly_coefs'] = list(eval(','.join(p_coefs)))
            data[q]['ch_range'] = eval(data[q]['ch_range'])
            data[q]['ch_amp'] = eval(data[q]['ch_amp'])
            data[q]['frequency'] = eval(data[q]['frequency'])
            data[q]['anharmonicity'] = eval(data[q]['anharmonicity'])
            for d in ['NW', 'NE', 'SW', 'SE']:
                data[q][f'amp_{d}'] = eval(data[q][f'amp_{d}'])
                data[q][f'amp_pad_{d}'] = eval(data[q][f'amp_pad_{d}'])
                data[q][f'B_amp_{d}'] = eval(data[q][f'B_amp_{d}'])
                data[q][f'asymmetry_{d}'] = eval(data[q][f'asymmetry_{d}'])
                data[q][f'tp_{d}'] = eval(data[q][f'tp_{d}'])
                data[q][f'tmid_{d}'] = eval(data[q][f'tmid_{d}'])
                data[q][f'tpad_{d}'] = eval(data[q][f'tpad_{d}'])
                data[q][f'use_asymmetry_{d}'] = eval(data[q][f'use_asymmetry_{d}'])
                data[q][f'use_net_zero_pulse_{d}'] = eval(data[q][f'use_net_zero_pulse_{d}'])
            data[q]['park_double_sided'] = eval(data[q]['park_double_sided'])
            data[q]['park_amp'] = eval(data[q]['park_amp'])
            data[q]['t_park'] = eval(data[q]['t_park'])
            data[q]['tpad_park'] = eval(data[q]['tpad_park'])
        # Get TLS landscapes
        self.TLS_analysis = {}
        for q in self.Qubits:
            label = f'Chevron {q} D1 ground'
            try:
                # Try to find TLS landscapes for relevant qubits
                TS = get_timestamps_in_range(
                        timestamp_start='20000101_000000',
                        label=label)
                for ts in TS[::-1]:
                    # Try runing TLS analysis for each timestamp
                    # until it is successful.
                    try:
                        a = TLS_landscape_Analysis(
                                t_start = ts,
                                Q_freq = data[q]['frequency'],
                                Poly_coefs = data[q]['poly_coefs'],
                                extract_only=True)
                        assert len(a.proc_data_dict['Times'])>3, \
                            'Not enough time steps in Chevron\nTrying other timestamp...'
                        self.TLS_analysis[q] = a
                        break
                    except:
                        print_exception()
            except:
                print_exception()
                print(f'No valid TLS landscape data found for {q}')
        # save data in raw data dictionary
        self.raw_data_dict = data
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        data = self.raw_data_dict
        self.proc_data_dict = {q : {} for q in self.Qubits}
        for q in self.Qubits:
            self.proc_data_dict[q]['frequency'] = data[q]['frequency']
            self.proc_data_dict[q]['anharmonicity'] = data[q]['anharmonicity']
            # estimate detunings at each amplitude
            for d in ['NW', 'NE', 'SW', 'SE']:
                # Trace CZ waveform
                _wf = vcz_waveform(
                    sampling_rate = 2.4e9,
                    amp_at_int_11_02 = data[q][f'amp_{d}'],
                    norm_amp_fine = data[q][f'B_amp_{d}'],
                    amp_pad = data[q][f'amp_pad_{d}'],
                    asymmetry = data[q][f'asymmetry_{d}'],
                    time_sqr = data[q][f'tp_{d}'],
                    time_middle = data[q][f'tmid_{d}'],
                    time_pad = data[q][f'tpad_{d}'],
                    use_asymmety = data[q][f'use_asymmetry_{d}'],
                    use_net_zero_pulse = data[q][f'use_net_zero_pulse_{d}'])
                self.proc_data_dict[q][f'cz_waveform_{d}'] = _wf
                # Convert CZ waveform into frequency trajectory
                _Ftrajectory = -convert_amp_to_freq(data[q]['poly_coefs'],
                                                    data[q]['ch_range'],
                                                    data[q]['ch_amp'], _wf)
                _Ftrajectory += data[q]['frequency']
                self.proc_data_dict[q][f'cz_freq_trajectory_{d}'] = _Ftrajectory
            # Parking trajectories
            _wf = gen_park(sampling_rate = 2.4e9,
                           park_length = data[q]['t_park'],
                           park_pad_length = data[q]['tpad_park'],
                           park_amp = data[q]['park_amp'],
                           park_double_sided = data[q]['park_double_sided'])
            self.proc_data_dict[q]['park_waveform'] = _wf
            _Ftrajectory = -convert_amp_to_freq(data[q]['poly_coefs'],
                                                data[q]['ch_range'],
                                                data[q]['ch_amp'], _wf)
            _Ftrajectory += data[q]['frequency']
            self.proc_data_dict[q]['park_freq_trajectory'] = _Ftrajectory
            # Idling trajectory
            n_points = len(_Ftrajectory)
            self.proc_data_dict[q]['idle_freq_trajectory'] = np.full(n_points, data[q]['frequency'])

    def prepare_plots(self):
        self.axs_dict = {}
        for qH, qL in self.Qubit_pairs:

            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            self.figs[f'{qH}_{qL}_Gate_frequency_trajectory'] = fig
            self.axs_dict[f'plot_{qH}_{qL}'] = ax
            # fig.patch.set_alpha(0)
            self.plot_dicts[f'{qH}_{qL}_Gate_frequency_trajectory']={
                    'plotfn': CZ_frequency_trajectory_plotfn,
                    'ax_id': f'plot_{qH}_{qL}',
                    'data': self.proc_data_dict,
                    'qH': qH,
                    'qL': qL,

                    'TLS_analysis_dict': self.TLS_analysis,
                    'timestamp': self.timestamps[0]}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def CZ_frequency_trajectory_plotfn(
    ax,
    data, qH, qL,
    timestamp,
    TLS_analysis_dict,
    include_TLS_landscape=True,
    **kw):
    fig = ax.get_figure()
    # Compile all relevant freq. trajectories
    directions = get_gate_directions(qH, qL)
    parked_qubits = get_parking_qubits(qH, qL)
    wf = { qH: f'cz_freq_trajectory_{directions[0]}',
           qL: f'cz_freq_trajectory_{directions[1]}' }
    for q in parked_qubits:
        wf[q] = 'park_freq_trajectory'
    # Draw CZ trajectories
    for q, _wf in wf.items():
        if q in parked_qubits:
            ax.plot(data[q][_wf]*1e-9, '--', markersize=3, lw=1, label=f'{q}')
        else:
            ax.plot(data[q][_wf]*1e-9, '.-', markersize=3, lw=1, label=f'{q}')
            # if q == qH: # plot 02 level
            #     ax.plot((data[q][_wf]+data[q]['anharmonicity'])*1e-9, 'C0.-', 
            #             alpha=.5, markersize=3, lw=1, label=f'{q}')
        # labels
        ax.text(5, data[q][_wf][5]*1e-9+.015, f'{q}')
    # settings of plot
    ax.set_title(f'{timestamp}\n{qH}, {qL} Gate')
    ax.set_ylabel('Frequency (GHz)')
    ax.set_xlabel('Time (# samples)')
    ax.grid(ls='--', alpha=.5)
    # Side plots for TLS landscapes
    if include_TLS_landscape:
        axR = fig.add_subplot(111)
        pos = axR.get_position()
        axR.set_position([pos.x0+pos.width*1.005, pos.y0, pos.width*0.2, pos.height])
        def get_plot_axis(vals, rang=None):
            if len(vals)>1:
                dx = vals[1]-vals[0]
                X = np.concatenate((vals, [vals[-1]+dx])) - dx/2
            else:
                X = vals
            return X
        Detunings = data[qH]['frequency'] - get_plot_axis(TLS_analysis_dict[qH].proc_data_dict['Detunings'])
        Times = get_plot_axis(TLS_analysis_dict[qH].proc_data_dict['Times'])
        Pop = TLS_analysis_dict[qH].proc_data_dict['Pop'] 
        # Frequency qubit population
        vmax = min([1, np.max(Pop)])
        vmax = max([vmax, 0.15])
        im = axR.pcolormesh(Times*1e9, Detunings*1e-9, Pop.transpose(), vmax=vmax)
        axR.text(Times[len(Times)//2]*1e9, Detunings[0]*1e-9-.05, qH, ha='center', va='top', color='w')
        axR.set_title('Gate qubits', size=7)
        if qL in TLS_analysis_dict.keys():
            Detunings = data[qL]['frequency'] - get_plot_axis(TLS_analysis_dict[qL].proc_data_dict['Detunings'])
            Pop = TLS_analysis_dict[qL].proc_data_dict['Pop'] 
            # Frequency qubit population
            vmax = min([1, np.max(Pop)])
            vmax = max([vmax, 0.15])
            im = axR.pcolormesh(Times*1e9, Detunings*1e-9, Pop.transpose(), vmax=vmax)
            axR.text(Times[len(Times)//2]*1e9, Detunings[0]*1e-9-.05, qL, ha='center', va='top', color='w')
            axR.axhline(Detunings[0]*1e-9, color='w')
        axR.set_ylim(ax.get_ylim())
        axR.yaxis.tick_right()
        axR.set_xticks([])
        axR.axis('off')
        # Parked qubit plots
        i = 0
        for q in parked_qubits:
            if q in TLS_analysis_dict.keys():
                axP = fig.add_subplot(221+i)
                # using previous axis position <pos>
                axP.set_position([pos.x0+pos.width*(1.21 + i*.205), pos.y0,
                                  pos.width*0.2, pos.height])
                
                Detunings = data[q]['frequency'] - get_plot_axis(TLS_analysis_dict[q].proc_data_dict['Detunings'])
                Pop = TLS_analysis_dict[q].proc_data_dict['Pop'] 
                # Frequency qubit population
                vmax = min([1, np.max(Pop)])
                vmax = max([vmax, 0.15])
                im = axP.pcolormesh(Times*1e9, Detunings*1e-9, Pop.transpose(), vmax=vmax)
                axP.text(Times[len(Times)//2]*1e9, Detunings[0]*1e-9-.05, q, ha='center', va='top', color='w')
                
                axP.set_title('Park qubits', size=7)
                axP.set_ylim(ax.get_ylim())
                axP.yaxis.tick_right()
                axP.set_xticks([])
                axP.axis('off')
                i += 1


class Parity_check_ramsey_analysis(ba.BaseDataAnalysis):
    """
    Analysis 
    """
    def __init__(self,
                 Q_target,
                 Q_control,
                 Q_spectator,
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
        self.Q_spectator = Q_spectator
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
        L_2 = {}
        n_c = len(self.Q_control)
        for i, q in enumerate(self.Q_control):
            P_excited[q] = { case : np.mean(Ramsey_curves[q][case]) for case in self.control_cases }
            L_0[q] = []
            L_1[q] = []
            L_2[q] = []
            for case in self.control_cases:
                if case[i] == '0':
                    L_0[q].append( P_excited[q][case] )
                elif case[i] == '1':
                    L_1[q].append( P_excited[q][case] )
                elif case[i] == '2':
                    L_2[q].append( P_excited[q][case] )
                else:
                    raise(f'Control case {case} not valid.')
            L_0[q] = np.mean(L_0[q])
            L_1[q] = np.mean(L_1[q])
            L_2[q] = np.mean(L_2[q])
            Missing_fraction[q] = L_1[q]-L_0[q]

        # Solve for Phase gate model
        Phase_model = {}
        if self.solve_for_phase_gate_model:
            for q in self.Q_target:
                n_c = len(self.Q_control)
                Phase_vec = np.array([Fit_res[q][c][0] for c in self.control_cases])
                if self.Q_spectator:
                    n_spec = len(self.Q_spectator)
                    Phase_model[q] = get_phase_model_values(n_c, Phase_vec, n_spec)
                else:
                    Phase_model[q] = get_phase_model_values(n_c, Phase_vec)
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
        for q in self.Q_target:
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

        for i, q in enumerate(self.Q_target):
            Q_total = [q]+self.Q_control
            fig, axs = plt.subplots(figsize=(9,4), ncols=2, dpi=100)
            self.figs[f'Parity_check_phases_{"_".join(Q_total)}'] = fig
            self.axs_dict[f'plot_phases_{i}'] = axs[0]

            self.plot_dicts[f'Parity_check_phases_{"_".join(Q_total)}']={
                    'plotfn': Phases_plotfn,
                    'ax_id': f'plot_phases_{i}',
                    'q_target': q,
                    'Q_control': self.Q_control,
                    'Q_spectator': self.Q_spectator,
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

        if self.solve_for_phase_gate_model:
            for i, q in enumerate(self.Q_target):
                Q_total = [q]+self.Q_control
                fig, ax = plt.subplots(figsize=(5,4))
                self.figs[f'Phase_gate_model_{"_".join(Q_total)}'] = fig
                self.axs_dict[f'plot_phase_gate_{i}'] = ax
                self.plot_dicts[f'Phase_gate_model_{"_".join(Q_total)}']={
                        'plotfn': Phase_model_plotfn,
                        'ax_id': f'plot_phase_gate_{i}',
                        'q_target': q,
                        'Q_control': self.Q_control,
                        'Q_spectator': self.Q_spectator,
                        'Phase_model': self.qoi['Phase_model'][q],
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
    if len(control_cases) == 2:
        Colors = { case : color for color, case in zip(['C2', 'C3'],control_cases)}
    else:
        from matplotlib.cm import hsv
        Colors = { case : hsv(x) for x, case in \
                   zip(np.linspace(0,1,len(control_cases)), control_cases)}
    for i, q in enumerate(Q_target+Q_control):
        for case in control_cases:
            if q in Q_target:
                _case_str = ''
                for j, _q in enumerate(Q_control):
                    _case_str += f'{case[j]}_'+'{'+_q+'}'
                _label = '$|'+_case_str+rf'\rangle$ : {Fit_res[q][case][0]:.1f}'
                _angles = np.linspace(angles[0], angles[-1], 101)
                axs[i].plot(_angles, func(_angles, *Fit_res[q][case]),
                            '--', color=Colors[case], alpha=1 if len(control_cases)==2 else .5,
                            label=_label)
            axs[i].plot(angles, Ramsey_curves[q][case],
                        '.', color=Colors[case], alpha=1 if len(control_cases)==2 else .5)
        axs[i].plot(cal_ax, Cal_points[q], 'C0.-')
        axs[i].legend(frameon=False, bbox_to_anchor=(1.04,1), loc="upper left")
        if q in Q_control:
            axs[i].plot([angles[0], angles[-1]], [L_0[q], L_0[q]], 'k--')
            axs[i].plot([angles[0], angles[-1]], [L_1[q], L_1[q]], 'k--',
                        label = f'Missing fac. : {Missing_fraction[q]*100:.1f} %')
            axs[i].legend(loc=2, frameon=False)
        axs[i].set_ylabel(f'Population {q}')
    axs[-1].set_xticks(np.arange(0, 360, 60))
    axs[-1].set_xlabel('Phase (deg), calibration points')
    axs[0].set_title(f'{timestamp}\nParity check ramsey '+\
                     f'{" ".join(Q_target)} with control qubits {" ".join(Q_control)}')

def Phases_plotfn(
    ax,
    q_target,
    Q_control,
    Q_spectator,
    control_cases, 
    Phases,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Sort control cases by number of excitations
    # and get ideal phases vector "vec"
    if Q_spectator:
        n_spec = len(Q_spectator)
        cases_sorted = [control_cases[0], control_cases[1]]
        vec = [0, 0]
        for n in range(len(control_cases[0])):
            for c in control_cases:
                if c[:-n_spec].count('1') == n+1:
                    cases_sorted.append(c)
                    vec.append(180*np.mod(n+1%2,2))
    else:
        cases_sorted = [control_cases[0]]
        vec = [0]
        for n in range(len(control_cases[0])):
            for c in control_cases:
                if c.count('1') == n+1:
                    cases_sorted.append(c)
                    vec.append(180*np.mod(n+1%2,2))
    # Phase error vector
    q = q_target
    phase_err_sorted = np.array([Phases[q][c] for c in cases_sorted])-np.array(vec)

    axs[0].plot(cases_sorted, np.zeros(len(cases_sorted))+180, 'k--')
    axs[0].plot(cases_sorted, np.zeros(len(cases_sorted)), 'k--')
    axs[0].plot(cases_sorted, [Phases[q][c] for c in cases_sorted], 'o-')
    axs[0].set_xticks(axs[0].get_xticks())
    axs[0].set_xticklabels([fr'$|{c}\rangle$' for c in cases_sorted], rotation=90, fontsize=7)
    axs[0].set_yticks([0, 45, 90, 135, 180])
    axs[0].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
    axs[0].set_ylabel(f'{q_target} Phase (deg)')
    axs[0].grid(ls='--')
    
    axs[1].bar(cases_sorted, phase_err_sorted, zorder=10)
    axs[1].grid(ls='--', zorder=-10)
    axs[1].set_xticks(axs[1].get_xticks())
    axs[1].set_xticklabels([fr'$|{c}\rangle$' for c in cases_sorted], rotation=90, fontsize=7)
    axs[1].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
    axs[1].set_ylabel(f'{q_target} Phase error (deg)')
    fig.suptitle(f'{timestamp}\nParity check ramsey '+\
                 f'{q_target} with control qubits {" ".join(Q_control)}', y=1.0)
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

def get_phase_model_values(n, Phase_vec, n_spec=None):
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
    if n_spec:
        Phase_vec_ideal = np.array([s[:-n_spec].count('1')*180 for s in states])
    else:
        Phase_vec_ideal = np.array([s.count('1')*180 for s in states])
    ########################################
    # Correct rotations for modulo of phase
    ########################################
    state_idxs_sorted_by_exc = {i:[] for i in range(n+1)}
    for i, s in enumerate(states):
        if n_spec:
            nr_exc = s[:-n_spec].count('1')
        else:
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

def Phase_model_plotfn(
    ax,
    q_target,
    Q_control,
    Q_spectator,
    Phase_model,
    timestamp,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    Ops = np.array([ op for op in Phase_model.keys() ])
    
    if Q_spectator:
        n_spec = len(Q_spectator)
        Ops_sorted = [Ops[0], Ops[1]]
        Phases_sorted = [Phase_model[Ops[0]], Phase_model[Ops[1]]]
        for n in range(len(Ops[0])):
            for c in Ops:
                if c[:-n_spec].count('Z') == n+1:
                    Ops_sorted.append(c)
                    Phases_sorted.append(Phase_model[c])
    else:
        Ops_sorted = [Ops[0]]
        Phases_sorted = [Phase_model[Ops[0]]]
        for n in range(len(Ops[0])):
            for c in Ops:
                if c.count('Z') == n+1:
                    Ops_sorted.append(c)
                    Phases_sorted.append(Phase_model[c])

    axs[0].bar(Ops_sorted, Phases_sorted, color='C0', zorder=10)
    axs[0].set_xticks(Ops_sorted)
    axs[0].set_xticklabels(Ops_sorted, rotation=90, fontsize=7)
    axs[0].set_xlabel('Operator $U_{'+fr'{"}U_{".join(Q_control)}'+'}$')
    axs[0].set_ylabel(f'Phase model coefficient error (deg)')
    axs[0].grid(ls='--', zorder=0)
    fig.suptitle(f'{timestamp}\nPhase gate model coefficients\n'+\
             f'{q_target} with control qubits {" ".join(Q_control)}', y=1.0)
    fig.tight_layout()


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
        # param_spec = {**_data, **_thrs}
        param_spec = {**_data}
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
            qubit_list += self.Q_control

        n = len(self.Q_control)+1
        nr_cases = len(self.control_cases)
        total_shots = len(self.raw_data_dict['data'][:,0])
        nr_shots_per_case = total_shots//((1+self.post_selection)*nr_cases+2**n)
        Threshold = {}
        RO_fidelity = {}
        # Sort calibration shots and calculate threshold
        Cal_shots = { q: {} for q in qubit_list }
        states = ['0','1']
        if self.post_selection:
            combinations = [''.join(s) for s in itertools.product(states, repeat=n)]
        else:
            combinations = ['0', '1']
        for i, q in enumerate(qubit_list):
            for j, comb in enumerate(combinations):
                if self.post_selection:
                    _shots = self.raw_data_dict['data'][:,i+1]
                    Cal_shots[q][comb] = _shots[2*nr_cases+j::2*nr_cases+2**n]
                else:
                    _shots = self.raw_data_dict['data'][:,i+1]
                    Cal_shots[q][comb] = _shots[nr_cases+j::nr_cases+2]
            shots_0 = []
            shots_1 = []
            for comb in combinations:
                if comb[i] == '0':
                    shots_0 += list(Cal_shots[q][comb])
                else:
                    shots_1 += list(Cal_shots[q][comb])
            def _calculate_threshold(shots_0, shots_1):
                s_max = np.max(list(shots_0)+list(shots_1))
                s_min = np.min(list(shots_0)+list(shots_1))
                s_0, bins_0 = np.histogram(shots_0, bins=100, range=(s_min, s_max))
                s_1, bins_1 = np.histogram(shots_1, bins=100, range=(s_min, s_max))
                bins = (bins_0[:-1]+bins_0[1:])/2
                th_idx = np.argmax(np.cumsum(s_0) - np.cumsum(s_1))
                threshold = bins[th_idx]
                return threshold
            Threshold[q] = _calculate_threshold(shots_0, shots_1)
            RO_fidelity[q] = \
                (np.mean([1 if s < Threshold[q] else 0 for s in shots_0])+
                 np.mean([0 if s < Threshold[q] else 1 for s in shots_1]))/2
        # Sort experiment shots
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
                                               in shots_dig[2*j::2*nr_cases+2**n] ])
                    Shots_dig[q][case] = shots_dig[2*j+1::2*nr_cases+2**n]
                else:
                    Shots_dig[q][case] = shots_dig[j::nr_cases+2]
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
        self.proc_data_dict['RO_fidelity'] = RO_fidelity
        self.proc_data_dict['P'] = P
        self.proc_data_dict['P_ideal'] = P_ideal
        self.proc_data_dict['Cal_shots'] = Cal_shots
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
        fig, axs = plt.subplots(figsize=(n_plots*2.5,4),
                                ncols=n_plots, nrows=2,
                                sharex='col', sharey='row')
        if not self.post_selection:
            axs = [axs]
        self.figs['Raw_shots'] = fig
        self.axs_dict['plot_3'] = np.array(axs).flatten()[0]
        # fig.patch.set_alpha(0)
        self.plot_dicts['Raw_shots']={
                'plotfn': raw_shots_plotfn,
                'ax_id': 'plot_3',
                'Shots_raw': self.proc_data_dict['Shots_raw'],
                'Cal_shots': self.proc_data_dict['Cal_shots'],
                'Threshold': self.proc_data_dict['Threshold'],
                'RO_fidelity': self.proc_data_dict['RO_fidelity'],
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
    Cal_shots,
    Threshold,
    RO_fidelity,
    Q_ancilla,
    Q_control,
    timestamp,
    post_selection,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    q_list = [Q_ancilla]
    if post_selection:
        q_list += Q_control
    n = len(q_list)
    for i, q in enumerate(q_list):
        shots_0 = []
        shots_1 = []
        for case in Cal_shots[q].keys():
            if case[i] == '0':
                shots_0 += list(Cal_shots[q][case])
            else:
                shots_1 += list(Cal_shots[q][case])

        s_max = np.max(list(shots_0)+list(shots_1))
        s_min = np.min(list(shots_0)+list(shots_1))
        s_0, bins_0 = np.histogram(shots_0, bins=100, range=(s_min, s_max))
        s_1, bins_1 = np.histogram(shots_1, bins=100, range=(s_min, s_max))
        bins = (bins_0[:-1]+bins_0[1:])/2
        axs[i].fill_between(bins, s_0, 0, alpha=.25, color='C0')
        axs[i].fill_between(bins, s_1, 0, alpha=.25, color='C3')
        axs[i].plot(bins, s_0, 'C0-')
        axs[i].plot(bins, s_1, 'C3-')
        axs[i].axvline(Threshold[q], color='k', ls='--', lw=1)
        axs[i].set_title(f'Shots {q}\n{RO_fidelity[q]*100:.1f} %')

        axs[n+i].hist(Shots_raw[q], bins=100)
        axs[n+i].axvline(Threshold[q], color='k', ls='--', lw=1)
        axs[n+i].set_xlabel('Integrated voltage')
    fig.suptitle(timestamp, y=1.05)
    