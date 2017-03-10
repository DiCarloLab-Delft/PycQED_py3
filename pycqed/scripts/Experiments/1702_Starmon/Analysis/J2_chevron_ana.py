from pycqed.analysis import fitting_models as fit_mods
import lmfit
omega = lambda flux, f_max, EC, asym: (
    f_max + EC) * (asym**2 + (1-asym**2)*np.cos(np.pi*flux)**2)**0.25 - EC
f_flux = lambda flux: omega(flux=flux,
                            f_max=6.10,
                            EC=0.28,
                            asym=0.)-0.28


def ChevFourierFunc(delta, alpha, beta, g, branch):
    assert(len(delta) == len(branch))
    freqs = alpha*np.sqrt(4*g*g+beta*beta*delta*delta)
    return np.where(branch, freqs, -freqs)


def ChevFourierFunc2(delta, alpha, beta, f_res, g, branch):
    assert(len(delta) == len(branch))
    freqs = alpha*np.sqrt(4*g*g+4.*np.pi*np.pi*(f_flux(beta*delta)-f_res)**2)
    return np.where(branch, freqs, -freqs)

import warnings


def reshape_axis_2d(axis_array):
    x = axis_array[0, :]
    y = axis_array[1, :]
#     print(y)
    dimx = np.sum(np.where(x == x[0], 1, 0))
    dimy = len(x) // dimx
#     print(dimx,dimy)
    if dimy*dimx < len(x):
        warnings.warn('Data was cut-off. Probably due to an interrupted scan')
        dimy_c = dimy + 1
    else:
        dimy_c = dimy
#     print(dimx,dimy,dimy_c,dimx*dimy)
    return x[:dimy_c], (y[::dimy_c])


def reshape_data(sweep_points, data):
    x, y = reshape_axis_2d(sweep_points)
#     print(x,y)
    dimx = len(x)
    dimy = len(y)
    dim = dimx*dimy
    if dim > len(data):
        dimy = dimy - 1
    return x, y[:dimy], (data[:dimx*dimy].reshape((dimy, dimx))).transpose()


def save_sim(distortion, save_file, f_max, e_min, e_max, e_points, g, time_stop, time_step):
    time_vec = np.arange(0., time_stop, time_step)
    freq_vec = np.linspace(e_min, e_max, e_points)
#     result = chevron_olli.chevron(2.*np.pi*(5.94 - 4.8), e_min, e_max, e_points, np.pi*0.0239*2., time_stop, time_step, distortion)
    result = chevron_olli.chevron(
        2.*np.pi*(f_max - 4.8), e_min, e_max, e_points, g, time_stop, time_step, distortion)
    np.savetxt(save_file, result.flatten())

from pycqed.analysis import composite_analysis as ca

scan_start = '20161119_191625'
scan_stop = '20161119_191625'

scan_start = '20170309_222001'
scan_stop = '20170309_222001'

pdict = {'I': 'amp',
         'sweep_points': 'sweep_points'}
opt_dict = {'scan_label': 'Chevron_2D'}
nparams = ['I', 'sweep_points']
spec_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop, options_dict=opt_dict,
                               params_dict_TD=pdict, numeric_params=nparams)
x, y, z = reshape_data(
    spec_scans.TD_dict['sweep_points'][0], spec_scans.TD_dict['I'][0])

from pycqed.analysis.tools.plotting import flex_colormesh_plot_vs_xy
from mpl_toolkits.axes_grid1 import make_axes_locatable

%matplotlib inline
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

ax = axs[0]
plot_times = y
plot_step = plot_times[1]-plot_times[0]

plot_x = x*1e9
x_step = plot_x[1]-plot_x[0]

result = z
cmin, cmax = 0, 1.
fig_clim = [cmin, cmax]
out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
                                xvals=plot_times,
                                yvals=plot_x,
                                zvals=result)
ax.set_xlabel(r'AWG Amp (Vpp)')
ax.set_ylabel(r'Time (ns)')
# ax.set_xlim(xmin, xmax)
ax.set_ylim(plot_x.min()-x_step/2., plot_x.max()+x_step/2.)
ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
#     ax.set_xlim(plot_times.min()-plot_step/2.,plot_times.max()+plot_step/2.)
# ax.set_xlim(0,50)
#     print('Bounce %d ns amp=%.3f; Pole %d ns amp=%.3f'%(list_values[iter_idx,0],
#                                                                list_values[iter_idx,1],
#                                                                list_values[iter_idx,2],
#                                                                list_values[iter_idx,3]))
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right', size='10%', pad='5%')
cbar = plt.colorbar(out['cmap'], cax=cax)
cbar.set_ticks(
    np.arange(fig_clim[0], 1.01*fig_clim[1], (fig_clim[1]-fig_clim[0])/5.))
cbar.set_ticklabels([str(fig_clim[0]), '', '', '', '', str(fig_clim[1])])
cbar.set_label('Qubit excitation probability')
#     ax.plot(u[:,0],u[:,1],'ro')

#     ax.xaxis.label.set_fontsize(14)
#     ax.yaxis.label.set_fontsize(14)
ax.title.set_fontsize(14)
#     fig.savefig(filename=save_name+'.png',format='png')

fig.tight_layout()

ax = axs[1]
plot_fft = np.fft.fft(result[:-4, :], axis=0)
plot_fft_f = np.fft.fftfreq(len(plot_x[:-4]), x_step)
fft_step = plot_fft_f[1]-plot_fft_f[0]
sort_vec = np.argsort(plot_fft_f)
print(plot_fft.shape, sort_vec)
plot_fft_abs = np.abs(plot_fft[sort_vec, :])

y = plot_fft_f[sort_vec]
mask_higher = np.where(y > 2.*(y[1]-y[0]), True, False)
mask_lower = np.where(y < 2.*(y[0]-y[1]), True, False)

peaks_higher = np.zeros(len(plot_times))
peaks_lower = np.zeros(len(plot_times))
for i, p in enumerate(plot_times):
    #         peaks_higher[i] = a_tools.peak_finder(y[mask_higher],plot_fft_abs[mask_higher,i])['peak']
    #         peaks_lower[i] = a_tools.peak_finder(y[mask_lower],plot_fft_abs[mask_lower,i])['peak']
    u = y[mask_higher]
    peaks_higher[i] = u[np.argmax(plot_fft_abs[mask_higher, i])]
    u = y[mask_lower]
    peaks_lower[i] = u[np.argmax(plot_fft_abs[mask_lower, i])]

cmin, cmax = 0, 10.
fig_clim = [cmin, cmax]
out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
                                xvals=plot_times,
                                yvals=y,
                                zvals=plot_fft_abs)
ax.plot(plot_times, peaks_lower, 'ro')
ax.plot(plot_times, peaks_higher, 'ro')
ax.set_xlabel(r'Amplitude (Vpp)')
ax.set_ylabel(r'Time (ns)')
# ax.set_xlim(xmin, xmax)
ax.set_ylim(plot_fft_f.min()-fft_step/2., plot_fft_f.max()+fft_step/2.)
ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
# ax.set_xlim(0,50)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right', size='10%', pad='5%')
cbar = plt.colorbar(out['cmap'], cax=cax)
cbar.set_ticks(
    np.arange(fig_clim[0], 1.01*fig_clim[1], (fig_clim[1]-fig_clim[0])/5.))
cbar.set_ticklabels([str(fig_clim[0]), '', '', '', '', str(fig_clim[1])])
cbar.set_label('Fourier transform')

fig.tight_layout()

fit_mask_lower = np.array([True]*len(peaks_lower))
fit_mask_higher = np.array([True]*len(peaks_higher))
# fit_mask_lower[:25] = False
# fit_mask_higher[:25] = False

my_fit_points = np.concatenate(
    (plot_times[fit_mask_lower], plot_times[fit_mask_higher]))
my_fit_data = np.concatenate(
    (peaks_lower[fit_mask_lower], peaks_higher[fit_mask_higher]))
mask_branch = np.concatenate((np.zeros(len(peaks_lower[fit_mask_lower]), dtype=np.bool),
                              np.ones(len(peaks_higher[fit_mask_higher]), dtype=np.bool)))

fit_func = lambda delta, alpha, beta, f_res, g: ChevFourierFunc2(
    delta, alpha, beta, f_res, g, mask_branch)
ChevFourierModel = lmfit.Model(fit_func)

ChevFourierModel.set_param_hint('alpha', value=1., min=0., max=10., vary=True)
ChevFourierModel.set_param_hint(
    'beta', value=0.14, min=-10., max=10., vary=True)
ChevFourierModel.set_param_hint(
    'f_res', value=4.68, min=0., max=50., vary=False)
ChevFourierModel.set_param_hint(
    'g', value=np.pi*0.0239*2., min=0, max=2000, vary=True)

my_fit_params = ChevFourierModel.make_params()

my_fit_res = ChevFourierModel.fit(
    data=my_fit_data, delta=my_fit_points, params=my_fit_params)
eval_fit_lower = lambda d: ChevFourierFunc2(delta=d, **my_fit_res.best_values, branch=np.zeros(d.shape, dtype=np.bool))
eval_fit_higher = lambda d: ChevFourierFunc2(delta=d, **my_fit_res.best_values, branch=np.ones(d.shape, dtype=np.bool))
my_fit_res, my_fit_res.best_values

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

ax = axs[0]
ax.plot(plot_times[fit_mask_lower], peaks_lower[fit_mask_lower], 'ro')
ax.plot(plot_times[fit_mask_higher], peaks_higher[fit_mask_higher], 'ro')
ax.plot(plot_times, eval_fit_lower(plot_times), 'r-')
ax.plot(plot_times, eval_fit_higher(plot_times), 'r-')
ax.set_xlim(plot_times.min(), plot_times.max())
ax.set_xlabel(r'Amplitude (Vpp)')
ax.set_ylabel(r'Time (ns)')

ax = axs[1]
ax.plot((f_flux(my_fit_res.best_values['beta']*plot_times)-my_fit_res.best_values[
        'f_res'])[fit_mask_lower], peaks_lower[fit_mask_lower], 'ro')
ax.plot((f_flux(my_fit_res.best_values['beta']*plot_times)-my_fit_res.best_values[
        'f_res'])[fit_mask_higher], peaks_higher[fit_mask_higher], 'ro')
ax.plot((f_flux(my_fit_res.best_values[
        'beta']*plot_times)-my_fit_res.best_values['f_res']), eval_fit_lower(plot_times), 'r-')
ax.plot((f_flux(my_fit_res.best_values[
        'beta']*plot_times)-my_fit_res.best_values['f_res']), eval_fit_higher(plot_times), 'r-')
ax.set_xlim((f_flux(my_fit_res.best_values['beta']*plot_times)-my_fit_res.best_values['f_res']).min(
), (f_flux(my_fit_res.best_values['beta']*plot_times)-my_fit_res.best_values['f_res']).max())
ax.set_xlabel(r'$\Delta$ (GHz)')
ax.set_ylabel(r'Time (ns)')
