import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.analysis_toolbox import peak_finder_v2
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis.plotting_tools import flex_colormesh_plot_vs_xy
import lmfit

# import data
a = ma.MeasurementAnalysis(
    timestamp='20170302_224401', close_fig=False, TwoD=True, transpose=True)

# find peaks
peaks = np.zeros((len(a.X), 2))
for i in range(len(a.X)):
    p_dict = peak_finder_v2(a.X[i], a.Z[0][i])
    peaks[i, :] = np.sort(p_dict[:2])

#############
# Making the figure w/o filtering
#############
flux = a.Y[:, 0]
f, ax = plt.subplots()
ax.set_title(a.timestamp_string + ' avoided crossing')
flex_colormesh_plot_vs_xy(a.X[0], flux, a.Z[0], ax=ax,
                          transpose=True, cmap='viridis')
ax.plot(flux, peaks[:, 0], 'o', c='r')
ax.plot(flux, peaks[:, 1], 'o', c='y')
ax.set_xlabel(a.ylabel)  # the axes are transposed
ax.set_ylabel(a.xlabel)


# Setting the filter mask
filter_mask_high = [True]*len(flux)
filter_mask_high = np.where(flux < 578, False, filter_mask_high)
# filter_mask_high =~ ~dm_tools.get_outliers(peaks[:,0], .1e6)

filter_mask_low = [True]*len(flux)
filter_mask_low = ~dm_tools.get_outliers(peaks[:, 1], 2e6)
filter_mask_low = np.where(flux > 581, False, filter_mask_low)

#############
# Plotting filtered data
#############
flux = a.Y[:, 0]
f, ax = plt.subplots()
ax.set_title(a.timestamp_string + ' filtered avoided crossing')
flex_colormesh_plot_vs_xy(a.X[0], flux, a.Z[0],
                          ax=ax,
                          transpose=True, cmap='viridis')
ax.plot(flux[filter_mask_high], peaks[filter_mask_high, 0], 'o', c='r')
ax.plot(flux[filter_mask_low], peaks[filter_mask_low, 1], 'o', c='y')
ax.set_xlabel(a.ylabel)  # the axes are transposed
ax.set_ylabel(a.xlabel)


############
# Sets up the model
############
# Model with bus mediating
def high_freq(dac, f_bus, f_center1, f_center2, coeff_1, coeff_2, g):
    ff = np.zeros(len(dac))
    for kk, d in enumerate(dac):
        f_1 = d * coeff_1 + f_center1
        f_2 = d * coeff_2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        ff[kk] = np.linalg.eigvalsh(matrix)[1]
    return ff


def low_freq(dac, f_bus, f_center1, f_center2, coeff_1, coeff_2, g):
    ff = np.zeros(len(dac))
    for kk, d in enumerate(dac):
        f_1 = d * coeff_1 + f_center1
        f_2 = d * coeff_2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        ff[kk] = np.linalg.eigvalsh(matrix)[0]
    return ff


def avoided_crossing_dac_1d(dacs, f_bus, f_center1, f_center2,
                            coeff_1, coeff_2, g, dacs_state):

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

# model with direct coupling


def high_freq(dac, f_bus, f_center1, f_center2, coeff_1, coeff_2, g):
    ff = np.zeros(len(dac))
    for kk, d in enumerate(dac):
        f_1 = d * coeff_1 + f_center1
        f_2 = d * coeff_2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        ff[kk] = np.linalg.eigvalsh(matrix)[1]
    return ff


def low_freq(dac, f_bus, f_center1, f_center2, coeff_1, coeff_2, g):
    ff = np.zeros(len(dac))
    for kk, d in enumerate(dac):
        f_1 = d * coeff_1 + f_center1
        f_2 = d * coeff_2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        ff[kk] = np.linalg.eigvalsh(matrix)[0]
    return ff


def avoided_crossing_dac_1d(dacs, f_bus, f_center1, f_center2,
                            coeff_1, coeff_2, g, dacs_state):

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


def avoided_crossing_dac_fit(dacs_lower, dacs_upper, lower_freqs, upper_freqs, f1_guess, f2_guess, f_bus_guess, cross_guess):
    total_freqs = []
    total_freqs.extend(lower_freqs)
    total_freqs.extend(upper_freqs)
    total_dacs = []
    total_dacs.extend(dacs_lower)
    total_dacs.extend(dacs_upper)
    total_mask = []
    total_mask.extend([True]*np.ones(len(dacs_lower)))
    total_mask.extend([False]*np.zeros(len(dacs_upper)))
    resized_fit_func = lambda dacs, f_bus, f_center1, f_center2, coeff_1, coeff_2, g: avoided_crossing_dac_1d(dacs=dacs,
                                                                                                              f_bus=f_bus,
                                                                                                              f_center1=f_center1,
                                                                                                              f_center2=f_center2,
                                                                                                              coeff_1=coeff_1,
                                                                                                              coeff_2=coeff_2,
                                                                                                              g=g,
                                                                                                              dacs_state=total_mask)
    av_crossing_model = lmfit.Model(resized_fit_func)

    c2_guess = 0.
    c1_guess = c2_guess + (f2_guess-f1_guess)/cross_guess

    av_crossing_model.set_param_hint('f_bus', value=f_bus_guess,
                                     min=7.8, max=8.5, vary=False)
    av_crossing_model.set_param_hint(
        'g', min=0., max=0.2, value=0.003, vary=True)
    av_crossing_model.set_param_hint(
        'f_center1', min=0, max=12., value=f1_guess, vary=True)
    av_crossing_model.set_param_hint(
        'f_center2', min=0., max=5., value=f2_guess, vary=True)
    av_crossing_model.set_param_hint(
        'coeff_1', min=-100., max=100., value=c1_guess, vary=True)
    av_crossing_model.set_param_hint(
        'coeff_2', min=-10., max=10., value=c2_guess, vary=True)
    params = av_crossing_model.make_params()
    fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                    dacs=np.array(total_dacs),
                                    params=params)
    return fit_res

fit_res = avoided_crossing_dac_fit(flux[filter_mask_high], flux[filter_mask_low],
                                   peaks[filter_mask_high, 0] *
                                   1e-9, peaks[filter_mask_low, 1]*1e-9,
                                   f1_guess=8.2, f2_guess=4.68,
                                   f_bus_guess=8.148809251400813, cross_guess=583)

########
# Plot results
########

#############
# Plotting filtered data
flux = a.Y[:, 0]
f, ax = plt.subplots()
ax.set_title(a.timestamp_string + ' filtered avoided crossing')
flex_colormesh_plot_vs_xy(a.X[0], flux, a.Z[0],
                          ax=ax,
                          transpose=True, cmap='viridis')
ax.plot(flux[filter_mask_high], peaks[filter_mask_high, 0], 'o', c='r')
ax.plot(flux[filter_mask_low], peaks[filter_mask_low, 1], 'o', c='y')
ax.set_xlabel(a.ylabel)  # the axes are transposed
ax.set_ylabel(a.xlabel)

ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
ax.title.set_fontsize(14)
ax.plot(flux, low_freq(flux, **fit_res.best_values)*1e9, 'r--', linewidth=2)
ax.plot(flux, high_freq(flux, **fit_res.best_values)*1e9, 'y--', linewidth=2)


# ax.plot(flux[filter_mask_high], peaks[filter_mask_high, 0],'wo',fillstyle='full', markeredgewidth=2,markersize=10, markeredgecolor='red')
# ax.plot(flux[filter_mask_low], peaks[filter_mask_low, 1],'wo',fillstyle='full', markeredgewidth=2,markersize=10, markeredgecolor='blue')

cross_point = (fit_res.best_values['f_center1']-fit_res.best_values['f_center2'])/(
    fit_res.best_values['coeff_2']-fit_res.best_values['coeff_1'])
g_legend = 'g = %.2f MHz' % (fit_res.best_values['g']*1e3)
ax.text(581, 4.7e9, g_legend, color='r', weight='bold')
ax.axvline(cross_point, linestyle='dashed', color='k')
ax.set_xlim(np.min(flux), np.max(flux))
ax.set_ylim(np.min(a.X[0]), np.max(a.X[0]))
print(cross_point)
ax.set_xlabel(r'Dac (mV)')
