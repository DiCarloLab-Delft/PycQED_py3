"""
- Extract data
- Filter data
- Format in table
- Run PSD analysis
"""
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Extract data
from pycqed.analysis.PSD import standard_arches_psd as std_psd
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import composite_analysis as ca


"""
- Extract data
- Filter data
- Format in table
- Run PSD analysis
"""
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Extract data
from pycqed.analysis.PSD import standard_arches_psd as std_psd

from pycqed.analysis import composite_analysis as ca


def extract_T1_data(t_start='20170301_180710',
                    t_stop='20170302_042802',
                    flux_variable_name='IVVI.dac2',
                    qubit_name='QL'):
    label = 'T1'
    pdict = {'f_q': '{}.f_qubit'.format(qubit_name),
             'flux': flux_variable_name,
             'T1': 'Analysis.Fitted Params F|1>.tau.value',
             'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
    opt_dict = {'scan_label': 'T1_{}'.format(qubit_name)}
    nparams = ['f_q', 'T1', 'T1_std', 'flux']
    T1_scans = ca.quick_analysis(t_start=t_start, t_stop=t_stop,
                                 options_dict=opt_dict,
                                 params_dict_TD=pdict, numeric_params=nparams)

    freqs = T1_scans.TD_dict['f_q']
    flux = T1_scans.TD_dict['flux']
    T1s = T1_scans.TD_dict['T1']
    T1s_std = T1_scans.TD_dict['T1_std']

    T1_timestamps = T1_scans.TD_dict['timestamps']

    return (T1_timestamps, flux, freqs, T1s, T1s_std)


def extract_T2_data(T1_timestamps,
                    flux_variable_name='IVVI.dac2',
                    scan_label='Echo_QL'):
    T2_timestamps = []
    for T1_stamp in T1_timestamps:
        timestamp = a_tools.latest_data(
            scan_label, older_than=T1_stamp, return_timestamp=True)[0]
        T2_timestamps.append(timestamp)
    pdict = {'T2': 'Analysis.Fitted Params w0.tau.value',
             'T2_std': 'Analysis.Fitted Params w0.tau.stderr', }
    opt_dict = {'scan_label': ''}
    nparams = ['T2', 'T2_std']
    echo_scans = ca.quick_analysis_list(T2_timestamps,
                                        options_dict=opt_dict, extract_only=True,
                                        params_dict_TD=pdict, numeric_params=nparams)

    T2s = echo_scans.TD_dict['T2']
    T2s_std = echo_scans.TD_dict['T2_std']
    return T2s, T2s_std

T1_timestamps, flux, freqs, T1s, T1s_std = extract_T1_data()
T2e, T2e_std = extract_T2_data(T1_timestamps=T1_timestamps)
T2s, T2s_std = extract_T2_data(
    T1_timestamps=T1_timestamps, scan_label='Ramsey')
# Filter data

# Plot unfiltered
f, ax = plt.subplots()
ax.errorbar(flux, T1s*1e6, T1s_std*1e6,
            marker='o', linestyle='', c='C3',
            label=r'$T_1$', markerfacecolor='none')
ax.errorbar(flux, T2e*1e6, T2e_std*1e6,
            marker='o', linestyle='', c='C2',
            label=r'$T_2$-echo', markerfacecolor='none')
ax.errorbar(flux, T2s*1e6, T2s_std*1e6,
            marker='o', linestyle='', c='C0',
            label=r'$T_2^\star$', markerfacecolor='none')
ax.set_ylim(0, 40)
ax.set_xlabel('Dac (mV)')
ax.set_ylabel(r'$\tau$ ($\mu$s)')
# ax.set_
ax.legend()

plt.show()

# Format in table

# Run PSD analysis

PSD_input_table = std_psd.prepare_input_table(flux, freqs, T1s, T2s, T2e)

std_psd.PSD_Analysis(PSD_input_table)


# Filter data

# Format in table

# f, ax =plt.subplot()
# ax.errorbar(

# Run PSD analysis

# std_psd.prepare_input_table(dac, frequency=)
# std_psd.PSD_Analysis(


filter_mask = [True]*len(flux)
# filter_mask = dm_tools.get_outliers(T1s, .1e-6)
filter_mask = np.where(T1s_std > 2e-6, False, filter_mask)
filter_mask = np.where(T2s_std > 2e-6, False, filter_mask)
filter_mask = np.where(T1s > 30e-6, False, filter_mask)
filter_mask = np.where(flux < -350, False, filter_mask)
filter_mask = np.where(flux > 350, False, filter_mask)
filter_mask


flux_filt = flux[filter_mask]
freqs_filt = freqs[filter_mask]
T1s_filt = T1s[filter_mask]
T1s_std_filt = T1s_std[filter_mask]
T2e_filt = T2e[filter_mask]
T2e_std_filt = T2e_std[filter_mask]
T2s_filt = T2s[filter_mask]
T2s_std_filt = T2s_std[filter_mask]

# Plot unfiltered
f, ax = plt.subplots()
ax.errorbar(flux_filt, T1s_filt*1e6, T1s_std_filt*1e6,
            marker='o', linestyle='', c='C3',
            label=r'$T_1$', markerfacecolor='none')
ax.errorbar(flux_filt, T2e_filt*1e6, T2e_std_filt*1e6,
            marker='o', linestyle='', c='C2',
            label=r'$T_2$-echo', markerfacecolor='none')
ax.errorbar(flux_filt, T2s_filt*1e6, T2s_std_filt*1e6,
            marker='o', linestyle='', c='C0',
            label=r'$T_2^\star$', markerfacecolor='none')
# ax.set_ylim(0, 40)
ax.set_xlabel('Dac (mV)')
ax.set_ylabel(r'$\tau$ ($\mu$s)')


PSD_input_table = std_psd.prepare_input_table(flux_filt, freqs_filt,
                                              T1s_filt, T2s_filt, T2e_filt)


std_psd.PSD_Analysis(PSD_input_table, path='',
                     freq_resonator=7.08e9, Qc=8200, chi_shift=0.1e6)


#############################
from pycqed.analysis import fitting_models as fit_mods
import lmfit
omega = lambda flux, f_max, EC, asym: (f_max + EC) * (asym**2 + (1-asym**2)*np.cos(np.pi*flux)**2)**0.25 - EC
f_flux = lambda flux: omega(flux=flux,
                            f_max=6.10,
                            EC=0.28,
                            asym=0.)-0.28

def ChevFourierFunc(delta, alpha, beta, g, branch):
    assert(len(delta)==len(branch))
    freqs = alpha*np.sqrt(4*g*g+beta*beta*delta*delta)
    return np.where(branch,freqs,-freqs)
def ChevFourierFunc2(delta, alpha, beta, f_res, g, branch):
    assert(len(delta)==len(branch))
    freqs = alpha*np.sqrt(4*g*g+4.*np.pi*np.pi*(f_flux(beta*delta)-f_res)**2)
    return np.where(branch,freqs,-freqs)

import warnings
def reshape_axis_2d(axis_array):
    x = axis_array[0,:]
    y = axis_array[1,:]
#     print(y)
    dimx = np.sum(np.where(x==x[0],1,0))
    dimy = len(x) // dimx
#     print(dimx,dimy)
    if dimy*dimx<len(x):
        warnings.warn('Data was cut-off. Probably due to an interrupted scan')
        dimy_c = dimy + 1
    else:
        dimy_c = dimy
#     print(dimx,dimy,dimy_c,dimx*dimy)
    return x[:dimy_c],(y[::dimy_c])
def reshape_data(sweep_points,data):
    x, y = reshape_axis_2d(sweep_points)
#     print(x,y)
    dimx = len(x)
    dimy = len(y)
    dim  = dimx*dimy
    if dim>len(data):
        dimy = dimy - 1
    return x,y[:dimy],(data[:dimx*dimy].reshape((dimy,dimx))).transpose()

def save_sim(distortion, save_file, f_max, e_min, e_max, e_points, g, time_stop, time_step):
    time_vec = np.arange(0., time_stop, time_step)
    freq_vec = np.linspace(e_min, e_max, e_points)
#     result = chevron_olli.chevron(2.*np.pi*(5.94 - 4.8), e_min, e_max, e_points, np.pi*0.0239*2., time_stop, time_step, distortion)
    result = chevron_olli.chevron(2.*np.pi*(f_max - 4.8), e_min, e_max, e_points, g, time_stop, time_step, distortion)
    np.savetxt(save_file,result.flatten())

from pycqed.analysis import composite_analysis as ca

scan_start = '20161119_191625'
scan_stop = '20161119_191625'

scan_start = '20170309_222001'
scan_stop = '20170309_222001'

pdict = {'I':'amp',
         'sweep_points':'sweep_points'}
opt_dict = {'scan_label':'Chevron_2D'}
nparams = ['I', 'sweep_points']
spec_scans = ca.quick_analysis(t_start=scan_start,t_stop=scan_stop, options_dict=opt_dict,
                  params_dict_TD=pdict,numeric_params=nparams)
x,y,z = reshape_data(spec_scans.TD_dict['sweep_points'][0],spec_scans.TD_dict['I'][0])

from pycqed.analysis.tools.plotting import flex_colormesh_plot_vs_xy
from mpl_toolkits.axes_grid1 import make_axes_locatable

%matplotlib inline
fig, axs = plt.subplots(1,2, figsize=(15,6))

ax = axs[0]
plot_times = y
plot_step = plot_times[1]-plot_times[0]

plot_x = x*1e9
x_step = plot_x[1]-plot_x[0]

result = z
cmin, cmax = 0, 1.
fig_clim = [cmin, cmax]
out = flex_colormesh_plot_vs_xy(ax=ax,clim=fig_clim,cmap='viridis',
                     xvals=plot_times,
                     yvals=plot_x,
                     zvals=result)
ax.set_xlabel(r'AWG Amp (Vpp)')
ax.set_ylabel(r'Time (ns)')
# ax.set_xlim(xmin, xmax)
ax.set_ylim(plot_x.min()-x_step/2.,plot_x.max()+x_step/2.)
ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
#     ax.set_xlim(plot_times.min()-plot_step/2.,ploft_times.max()+plot_step/2.)
# ax.set_xlim(0,50)
#     print('Bounce %d ns amp=%.3f; Pole %d ns amp=%.3f'%(list_values[iter_idx,0],
#                                                                list_values[iter_idx,1],
#                                                                list_values[iter_idx,2],
#                                                                list_values[iter_idx,3]))
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right',size='10%', pad='5%')
cbar = plt.colorbar(out['cmap'],cax=cax)
cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
cbar.set_label('Qubit excitation probability')
#     ax.plot(u[:,0],u[:,1],'ro')

#     ax.xaxis.label.set_fontsize(14)
#     ax.yaxis.label.set_fontsize(14)
ax.title.set_fontsize(14)
#     fig.savefig(filename=save_name+'.png',format='png')

fig.tight_layout()




ax = axs[1]
plot_fft = np.fft.fft(result[:-4,:],axis=0)
plot_fft_f = np.fft.fftfreq(len(plot_x[:-4]),x_step)
fft_step = plot_fft_f[1]-plot_fft_f[0]
sort_vec = np.argsort(plot_fft_f)
print(plot_fft.shape,sort_vec)
plot_fft_abs = np.abs(plot_fft[sort_vec,:])

y = plot_fft_f[sort_vec]/2*np.pi
mask_higher = np.where(y>2.*(y[1]-y[0]),True,False)
mask_lower = np.where(y<2.*(y[0]-y[1]),True,False)

peaks_higher = np.zeros(len(plot_times))
peaks_lower = np.zeros(len(plot_times))
for i,p in enumerate(plot_times):
    u = y[mask_higher]
    peaks_higher[i] = u[np.argmax(plot_fft_abs[mask_higher,i])]
    u = y[mask_lower]
    peaks_lower[i] = u[np.argmax(plot_fft_abs[mask_lower,i])]

cmin, cmax = 0, 10.
fig_clim = [cmin, cmax]
out = flex_colormesh_plot_vs_xy(ax=ax,clim=fig_clim,cmap='viridis',
                     xvals=plot_times,
                     yvals=y,
                     zvals=plot_fft_abs)
ax.plot(plot_times,peaks_lower, 'o', fillstyle='none', c='orange')
ax.plot(plot_times,peaks_higher, 'o', fillstyle='none', c='orange')
ax.set_xlabel(r'Amplitude (Vpp)')
ax.set_ylabel(r'Time (ns)')
# ax.set_xlim(xmin, xmax)
ax.set_ylim(plot_fft_f.min()-fft_step/2.,plot_fft_f.max()+fft_step/2.)
ax.set_xlim(plot_times.min()-plot_step/2.,plot_times.max()+plot_step/2.)
# ax.set_xlim(0,50)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right',size='10%', pad='5%')
cbar = plt.colorbar(out['cmap'],cax=cax)
cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
cbar.set_label('Fourier transform')

fig.tight_layout()
fit_mask_lower = np.array([True]*len(peaks_lower))
fit_mask_higher = np.array([True]*len(peaks_higher))
# fit_mask_lower[:25] = False
# fit_mask_higher[:25] = False

my_fit_points = np.concatenate((plot_times[fit_mask_lower],plot_times[fit_mask_higher]))
my_fit_data = np.concatenate((peaks_lower[fit_mask_lower],peaks_higher[fit_mask_higher]))
mask_branch = np.concatenate((np.zeros(len(peaks_lower[fit_mask_lower]),dtype=np.bool),
                              np.ones(len(peaks_higher[fit_mask_higher]),dtype=np.bool)))

fit_func = lambda delta, alpha, beta, f_res, g: ChevFourierFunc2(delta, alpha, beta, f_res, g, mask_branch)
ChevFourierModel = lmfit.Model(fit_func)

ChevFourierModel.set_param_hint('alpha', value=1., min = 0., max = 10., vary=True)
ChevFourierModel.set_param_hint('beta', value=0.14, min = -10., max = 10., vary=True)
ChevFourierModel.set_param_hint('f_res', value=4.68, min = 0., max = 50., vary=False)
ChevFourierModel.set_param_hint('g', value=np.pi*0.0239*2., min = 0, max = 2000, vary=True)

my_fit_params = ChevFourierModel.make_params()

fit_res = ChevFourierModel.fit(data=my_fit_data, delta=my_fit_points, params=my_fit_params)
eval_fit_lower = lambda d: ChevFourierFunc2(delta=d, **fit_res.best_values, branch=np.zeros(d.shape,dtype=np.bool))
eval_fit_higher = lambda d: ChevFourierFunc2(delta=d, **fit_res.best_values, branch=np.ones(d.shape,dtype=np.bool))
fit_res, fit_res.best_values




# ax.plot(plot_times[fit_mask_lower],peaks_lower[fit_mask_lower],'ro')
# ax.plot(plot_times[fit_mask_higher],peaks_higher[fit_mask_higher],'ro')
ax.plot(plot_times,eval_fit_lower(plot_times),'-', c='orange', label='fit')
ax.plot(plot_times,eval_fit_higher(plot_times),'-', c='orange')
ax.set_xlim(plot_times.min(),plot_times.max())

coupling_label = '$J_2$'
g_legend = r'{} = {:.2f}$\pm${:.2f} MHz'.format(
    coupling_label,
    fit_res.params['g']/(2*np.pi)*1e3, fit_res.params['g'].stderr/(2*np.pi)*1e3)
ax.text(.6, .8, g_legend, transform=ax.transAxes, color='white')