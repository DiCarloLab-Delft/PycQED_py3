import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
from matplotlib.colors import to_rgba, LogNorm
from pycqed.analysis.tools.plotting import hsluv_anglemap45
import itertools
from pycqed.analysis.analysis_toolbox import set_xlabel, get_timestamps_in_range
from pycqed.utilities.general import get_gate_directions



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
        p0 = [11e6,                 # g
              np.mean(Detunings),   # delta_0
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
    axs[0].set_title(f'Population {qH}', pad = 40)
    axs[0].axvline(f0*1e-6, color='w', ls='--')
    axs[0].axhline(tp/2*1e9, color='w', ls='--')
    axs[0].plot([f0*1e-6], [tp/2*1e9], 'C3.')
    axt0 = axs[0].twiny()
    axt0.set_xlim((qH_freq*1e-6-np.array(axs[0].get_xlim()))*1e-3)
    axt0.set_xlabel(f'{qH} Frequency (GHz)', labelpad = 2)
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
    axs[1].set_title(f'Population {qL}', pad = 40)
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

    fig.suptitle(f'{ts}\nChevron {qH}, {qL}', y=1.2)
    # fig.tight_layout()




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
                 auto=True,
                 flux_lm_qpark = None,
                 isparked: bool = False):

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
        self.flux_lm_qpark = flux_lm_qpark
        self.isparked = isparked
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
        self.proc_data_dict['Detunings'] = np.real(Detunings)
        self.proc_data_dict['Times'] = Times
        self.proc_data_dict['Pop'] = Pop
        self.proc_data_dict['park_detuning'] = None
        if self.isparked:
            poly = self.flux_lm_qpark.q_polycoeffs_freq_01_det()
            ch_amp_park = self.flux_lm_qpark.park_amp()
            sq_amp_park = self.flux_lm_qpark.sq_amp()
            out_range_park = self.flux_lm_qpark.cfg_awg_channel_range()
            out_voltage_park = (sq_amp_park * ch_amp_park * out_range_park) / 2
            park_detuning = poly[0] * out_voltage_park ** 4 + poly[1] * out_voltage_park ** 3 + poly[2] * out_voltage_park ** 2 + poly[3] * out_voltage_park + poly[4]
            self.proc_data_dict['park_detuning'] = park_detuning

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
            'isparked' : self.isparked,
            'park_detuning': self.proc_data_dict['park_detuning'],
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
    interaction_freqs = None,
    isparked = None,
    park_detuning = None,
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
    vmax = 1 #min([1, np.max(Pop)])
    vmax = max([vmax, 0.15])
    vmin = 0
    im = ax.pcolormesh(Detunings*1e-6, Times*1e9, Pop, vmax=vmax, vmin = vmin)
    fig.colorbar(im, ax=ax, label='Population')
    # plot two-qubit gate frequencies:
    if interaction_freqs:
        for gate, freq in interaction_freqs.items():
            if freq > 10e6:
                ax.axvline(freq*1e-6, color='w', ls='--')
                ax.text(freq*1e-6, np.mean(Times)*1e9,
                        f'CZ {gate}', va='center', ha='right',
                        color='w', rotation=90)
    if isparked:
        ax.axvline(park_detuning*1e-6, color='w', ls='--')
        ax.text(park_detuning*1e-6, np.mean(Times)*1e9,
                f'parking freq', va='center', ha='right',
                color='w', rotation=90)
    ax.set_xlabel(f'{Q_name} detuning (MHz)')
    ax.set_ylabel('Duration (ns)')
    ax.set_title(f'Population {Q_name}', pad = 35)
    axt0 = ax.twiny()
    axt0.set_xlim((Q_freq*1e-6-np.array(ax.get_xlim()))*1e-3) # removing this for the TLS
    axt0.set_xlabel(f'{Q_name} Frequency (GHz)', labelpad = 4)
    fig.tight_layout()
    fig.suptitle(f'{ts}\nTLS landscape {Q_name}', y=1.07)


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
                 Q0,
                 Q1,
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
        self.Poly_coefs = Poly_coefs
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Q0_freq = Q0_freq
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
            Detunings = [ P_funcs[i](Amps_list[i]*self.DAC_amp*self.Out_range/2) \
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
        if n > 1:
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
            _X = np.linspace(X[0], X[-1], 201)
            _Y = np.linspace(Y[0], Y[-1], 201)
            _X, _Y = np.meshgrid(_X, _Y)
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
                CS = axs[i+2*n].contour(_X, _Y, Cphase, [180], colors=['w'],
                                   linewidths=[1.5], linestyles=['--'], alpha=1)
                axs[i+2*n].clabel(CS, CS.levels, inline=True, fmt='$%i^\\circ$', fontsize=10)
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
        self.Poly_coefs = Poly_coefs
        self.Out_range = Out_range
        self.DAC_amp = DAC_amp
        self.Q0_freq = Q0_freq
        self.tmid = tmid
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
            Detunings = [ P_funcs[i](Amps_list[i]*self.DAC_amp*self.Out_range/2) \
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
        if n > 1:
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
        # Plot SNZ leakage fitting contours
        _X = np.linspace(X[0], X[-1], 201)
        _Y = np.linspace(Y[0], Y[-1], 201)
        _X, _Y = np.meshgrid(_X, _Y)
        # Get interpolated landscape from fit
        # fit params
        print(fit_params)
        tp_factor, tmid, g, delta_0, det11_02, n_dist, a, b = fit_params
        Pop20, Pop11, Cphase = SNZ2(delta=_X, B_amp=_Y,
                                    tp=tp_factor/(4*g),
                                    tmid=tmid,
                                    g=g,
                                    delta_0=delta_0,
                                    det11_02=det11_02,
                                    n_dist=n_dist)
        for i in range(2):
            # Plot leakage contours
            for c, a_ in zip([.05, .2, .6, .8], [.5, .7, .85, 1]):
                axs[i+2*n].contour(_X, _Y, Pop20, [c], colors=['w'],
                              linewidths=[1], linestyles=['--'], alpha=a_)
            # # Plot 180 cphase contours
            # CS = axs[i+2*n].contour(_X, _Y, Cphase, [180], colors=['w'],
            #                    linewidths=[1.5], linestyles=['--'], alpha=1)
            # axs[i+2*n].clabel(CS, CS.levels, inline=True, fmt='$%i^\\circ$', fontsize=10)

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
                axs[i].plot(angles, func(angles, *Fit_res[q][case]),
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
    axs[0].set_xticklabels([fr'$|{c}\rangle$' for c in cases_sorted], rotation=90, fontsize=7)
    axs[0].set_yticks([0, 45, 90, 135, 180])
    axs[0].set_xlabel(fr'Control qubit states $|${",".join(Q_control)}$\rangle$')
    axs[0].set_ylabel(f'{q_target} Phase (deg)')
    axs[0].grid(ls='--')
    
    axs[1].bar(cases_sorted, phase_err_sorted, zorder=10)
    axs[1].grid(ls='--', zorder=-10)
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



Park_dict = {
     ('QNW', 'QC'): [],
     ('QNE', 'QC'): [],
     ('QC', 'QSW'): ['QSE'],
     ('QC', 'QSE'): ['QSW'],
 }

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
                 amp_pad_samples,
                 asymmetry,
                 time_sqr,
                 time_middle,
                 time_pad,
                 use_asymmety,
                 use_net_zero_pulse,
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
        if _i<amp_pad_samples:
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
            Q_parks = Park_dict[(qH, qL)]
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
                # param_spec[f'asymmetry_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_asymmetry_{d}')
                # param_spec[f'amp_pad_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_amp_pad_{d}')
                # param_spec[f'amp_pad_samples_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_amp_pad_samples_{d}')
                # param_spec[f'use_asymmetry_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_use_asymmetric_amp_{d}')
                # param_spec[f'use_net_zero_pulse_{d}'] = (f'Instrument settings/flux_lm_{q}', f'attr:vcz_use_net_zero_pulse_{d}')
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
                data[q][f'amp_pad_{d}'] = 0#eval(data[q][f'amp_pad_{d}'])
                data[q][f'amp_pad_samples_{d}'] = 0#eval(data[q][f'amp_pad_samples_{d}'])
                data[q][f'B_amp_{d}'] = eval(data[q][f'B_amp_{d}'])
                data[q][f'asymmetry_{d}'] = 1#eval(data[q][f'asymmetry_{d}'])
                data[q][f'tp_{d}'] = eval(data[q][f'tp_{d}'])
                data[q][f'tmid_{d}'] = eval(data[q][f'tmid_{d}'])
                data[q][f'tpad_{d}'] = eval(data[q][f'tpad_{d}'])
                # data[q][f'use_asymmetry_{d}'] = eval(data[q][f'use_asymmetry_{d}'])
                # data[q][f'use_net_zero_pulse_{d}'] = eval(data[q][f'use_net_zero_pulse_{d}'])
            data[q]['park_double_sided'] = eval(data[q]['park_double_sided'])
            data[q]['park_amp'] = eval(data[q]['park_amp'])
            data[q]['t_park'] = eval(data[q]['t_park'])
            data[q]['tpad_park'] = eval(data[q]['tpad_park'])
        # Get TLS landscapes
        self.TLS_analysis = {}
        for q in self.Qubits:
            label = f'Chevron {q} QC ground'
            print(label)
            try:
                # Try to find TLS landscapes for relevant qubits
                TS = get_timestamps_in_range(
                        timestamp_start='20220101_000000',
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
                        # print_exception()
                        print('No TLS landscape found')
            except:
                # print_exception()
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
                    amp_pad_samples = data[q][f'amp_pad_samples_{d}'],
                    asymmetry = 1,#data[q][f'asymmetry_{d}'],
                    time_sqr = data[q][f'tp_{d}'],
                    time_middle = data[q][f'tmid_{d}'],
                    time_pad = data[q][f'tpad_{d}'],
                    use_asymmety = False,#data[q][f'use_asymmetry_{d}'],
                    use_net_zero_pulse = True,)#data[q][f'use_net_zero_pulse_{d}'])
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

            print(self.TLS_analysis)
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
    parked_qubits = Park_dict[(qH, qL)]
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
        Detunings = data[qH]['frequency'] 
        Detunings -= get_plot_axis(TLS_analysis_dict[qH].proc_data_dict['Detunings'])
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
    