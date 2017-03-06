import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.analysis_toolbox import peak_finder_v2
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis.plotting_tools import flex_colormesh_plot_vs_xy
import lmfit

# import data
ac = ma.MeasurementAnalysis(
    timestamp='20170304_151058', close_fig=True, TwoD=True, transpose=True)

# find peaks
peaks = np.zeros((len(ac.X), 2))
for i in range(len(ac.X)):
    p_dict = peak_finder_v2(ac.X[i], ac.Z[0][i])
    peaks[i, :] = np.sort(p_dict[:2])

peaks_low = peaks[:, 0]
peaks_high = peaks[:, 1]

#############
# Making the figure w/o filtering
#############
flux = ac.Y[:, 0]
f, ax = plt.subplots()
ax.set_title(ac.timestamp_string + ' avoided crossing')
flex_colormesh_plot_vs_xy(ac.X[0], flux, ac.Z[0], ax=ax,
                          transpose=True, cmap='viridis')
ax.plot(flux, peaks_low, 'o', c='r')
ax.plot(flux, peaks_high, 'o', c='y')
ax.set_xlabel(ac.ylabel)  # the axes are transposed
ax.set_ylabel(ac.xlabel)


#
# Setting the filter mask
a = -.00610e9
x_ac = 520
y_ac = 4.972e9

filter_func = lambda x: a*(x-x_ac)+y_ac

filter_mask_high = [True] * len(peaks_high)
filter_mask_high = ~dm_tools.get_outliers(peaks_high, 15e6)
filter_mask_high = np.where(
    peaks_high < filter_func(flux), False, filter_mask_high)
filter_mask_high[-2] = False  # hand remove 1 datapoint

filt_flux_high = flux[filter_mask_high]
filt_peaks_high = peaks_high[filter_mask_high]


filter_mask_low = [True] * len(peaks_low)
filter_mask_low = ~dm_tools.get_outliers(peaks_low, 15e6)
filter_mask_low = np.where(
    peaks_low > filter_func(flux), False, filter_mask_low)
filter_mask_low[[0, -1]] = False  # hand remove 2 datapoints

filt_flux_low = flux[filter_mask_low]
filt_peaks_low = peaks_low[filter_mask_low]

#############
# Plotting filtered data
#############
f, ax = plt.subplots()
ax.set_title(ac.timestamp_string + ' filtered avoided crossing')
flex_colormesh_plot_vs_xy(ac.X[0], flux, ac.Z[0],
                          ax=ax,
                          transpose=True, cmap='viridis')

ax.plot(flux, filter_func(flux),  ls='--', c='w')
ax.plot(filt_flux_high, filt_peaks_high, 'o', c='r')
ax.plot(filt_flux_low, filt_peaks_low, 'o', c='y')
ax.set_xlabel(ac.ylabel)  # the axes are transposed
ax.set_ylabel(ac.xlabel)

ax.plot(flux, filter_func(flux),  ls='--', c='w')


############
# Sets up the model
############
def avoided_crossing_mediated_coupling(dacs, f_bus, f_center1, f_center2,
                                       coeff_1, coeff_2, g, dacs_state=None):
    if type(dacs_state) == bool:
        dacs_state = [dacs_state]*len(dacs)

    frequencies = np.zeros([len(dacs), 2])
    for kk, dac in enumerate(dacs):
        f_1 = dac * coeff_1 + f_center1
        f_2 = dac * coeff_2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(dacs_state, frequencies[:, 0], frequencies[:, 1])
    return result


def avoided_crossing_direct_coupling(dacs, f_center1, f_center2,
                                     coeff_1, coeff_2, g, dacs_state=None):
    if type(dacs_state) == bool:
        dacs_state = [dacs_state]*len(dacs)

    frequencies = np.zeros([len(dacs), 2])
    for kk, dac in enumerate(dacs):
        f_1 = dac * coeff_1 + f_center1
        f_2 = dac * coeff_2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(dacs_state, frequencies[:, 0], frequencies[:, 1])
    return result

########
# Doing the fit
########


def avoided_crossing_dac_fit(dacs_lower, dacs_upper, lower_freqs, upper_freqs,
                             f1_guess, f2_guess, cross_guess):

    total_freqs = np.concatenate([lower_freqs, upper_freqs])
    total_dacs = np.concatenate([dacs_lower, dacs_upper])
    total_mask = np.concatenate([np.ones(len(dacs_lower)),
                                 np.zeros(len(dacs_upper))])

    def resized_fit_func(dacs,
                         f_center1, f_center2, coeff_1, coeff_2,
                         g):
        # removes the mask from the function call to allow fitting
        return avoided_crossing_direct_coupling(dacs=dacs,
                                                f_center1=f_center1,
                                                f_center2=f_center2,
                                                coeff_1=coeff_1,
                                                coeff_2=coeff_2,
                                                g=g,
                                                dacs_state=total_mask)

    av_crossing_model = lmfit.Model(resized_fit_func)

    c2_guess = 0.
    c1_guess = c2_guess + (f2_guess-f1_guess)/cross_guess

    av_crossing_model.set_param_hint(
        'g', min=0., max=0.2e9, value=0.003e9, vary=True)
    av_crossing_model.set_param_hint(
        'f_center1', min=0, max=12.0e9, value=f1_guess, vary=True)
    av_crossing_model.set_param_hint(
        'f_center2', min=0., max=12.0e9, value=f2_guess, vary=True)
    av_crossing_model.set_param_hint(
        'coeff_1', min=-100.0e6, max=100.0e6, value=c1_guess, vary=True)
    av_crossing_model.set_param_hint(
        'coeff_2', min=-100.0e6, max=100.0e6, value=c2_guess, vary=True)
    params = av_crossing_model.make_params()
    fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                    dacs=np.array(total_dacs),
                                    params=params)
    return fit_res

fit_res = avoided_crossing_dac_fit(filt_flux_low,
                                   filt_flux_high,
                                   filt_peaks_low,
                                   filt_peaks_high,
                                   f1_guess=8.2e9, f2_guess=4.68e9,
                                   cross_guess=520)

########
# Plot results
########


########
# Plot results
########
f, ax = plt.subplots()
ax.set_title(ac.timestamp_string + ' filtered avoided crossing')
flex_colormesh_plot_vs_xy(1e-9*ac.X[0], flux, ac.Z[0],
                          ax=ax,
                          transpose=True, cmap='viridis')
ax.plot(flux, 1e-9*filter_func(flux),  ls='--', c='w')
ax.plot(filt_flux_high, 1e-9*filt_peaks_high, 'o',fillstyle='none', c='r')
ax.plot(filt_flux_low, 1e-9*filt_peaks_low, 'o', fillstyle='none', c='y')
ax.set_xlabel(ac.ylabel)  # the axes are transposed
ax.set_ylabel('Frequency (GHz)')

ax.plot(flux, 1e-9*avoided_crossing_direct_coupling(
    flux, **fit_res.best_values,
    dacs_state=False), 'r-', label='fit')
ax.plot(flux, 1e-9*avoided_crossing_direct_coupling(
    flux, **fit_res.best_values,
    dacs_state=True), 'y-', label='fit')

g_legend = r'$J_2$ = {:.2f}$\pm${:.2f} MHz'.format(
    fit_res.params['g']*1e-6, fit_res.params['g'].stderr*1e-6)
# , weight='bold')
ax.text(.6, .8, g_legend, transform=ax.transAxes, color='white')
ax.set_xlim(min(flux), max(flux))
ax.set_ylim(min(ac.X[0]*1e-9), max(ac.X[0]*1e-9))
f.savefig(ac.folder+'\\avoided_crossing.png', format='png', dpi=600)