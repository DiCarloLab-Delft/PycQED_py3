import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis.analysis_toolbox import peak_finder_v2
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis.plotting_tools import flex_colormesh_plot_vs_xy
import lmfit

a = ma.MeasurementAnalysis(close_fig=False, TwoD=True, transpose=True)


peaks = np.zeros((len(a.X), 2))
for i in range(len(a.X)):
    p_dict = peak_finder_v2(a.X[i], a.Z[0][i])
    peaks[i, :] = np.sort(p_dict[:2])


# Making the figure
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
filter_mask_high = np.where(flux < 577, False, filter_mask_high)
# filter_mask_high =~ ~dm_tools.get_outliers(peaks[:,0], .1e6)

filter_mask_low = [True]*len(flux)
filter_mask_low = ~dm_tools.get_outliers(peaks[:, 1], 2e6)
filter_mask_low = np.where(flux > 581, False, filter_mask_low)

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


def qubit_freq_cos(f_max, dac, dac_coeff, dac_offset):
    return f_max*np.sqrt(np.abs(np.cos(np.pi*(dac-dac_offset)*dac_coeff)))


def avoided_crossing_dac_1d(dacs, f_cte, g, f_max, dac_coeff,
                            dac_offset, dacs_state):
    result_f = []

    frequencies = np.zeros([len(dacs), 2])
    f2 = f_cte
    for kk, dac in enumerate(dacs):
        f1 = qubit_freq_cos(f_max, dac, dac_coeff, dac_offset)
        df = f1 - f2
        determinant = np.sqrt(df**2+4*g**2)
        frequencies[kk, :] = 0.5*(f1+f2-determinant), 0.5*(f1+f2+determinant)
    result = np.where(dacs_state, frequencies[:, 0], frequencies[:, 1])
    return result


def avoided_crossing_dac_fit(dacs_lower, dacs_upper, lower_freqs,
                             upper_freqs, f_max_guess, f_cte_guess,
                             dac_offset_guess):
    total_freqs = []
    total_freqs.extend(lower_freqs)
    total_freqs.extend(upper_freqs)
    total_dacs = []
    total_dacs.extend(dacs_lower)
    total_dacs.extend(dacs_upper)
    total_mask = []
    total_mask.extend([True]*np.ones(len(dacs_lower)))
    total_mask.extend([False]*np.zeros(len(dacs_upper)))
    resized_fit_func = lambda dacs, f_cte, g, f_max, dac_coeff, dac_offset: avoided_crossing_dac_1d(dacs,
                                                                                                    f_cte,
                                                                                                    g,
                                                                                                    f_max,
                                                                                                    dac_coeff,
                                                                                                    dac_offset,
                                                                                                    total_mask)
    av_crossing_model = lmfit.Model(resized_fit_func)

    coeff_guess = 1/2000.

    av_crossing_model.set_param_hint('f_cte', value=f_cte_guess,
                                     min=np.min(total_freqs), max=np.max(total_freqs))
    av_crossing_model.set_param_hint('g', min=0, value=0.053e9)
    av_crossing_model.set_param_hint('f_max', vary=True, value=f_max_guess)
    av_crossing_model.set_param_hint('dac_coeff', value=coeff_guess)
    av_crossing_model.set_param_hint('dac_offset', value=dac_offset_guess)
    params = av_crossing_model.make_params()
    fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                    dacs=np.array(total_dacs),
                                    params=params)
    return fit_res

fit_res = avoided_crossing_dac_fit(flux[filter_mask_low], flux[filter_mask_high],
                                   peaks[filter_mask_low, 0],
                                   peaks[filter_mask_high, 1],
                                   f_max_guess=6.108392498189497e9,
                                   f_cte_guess=5.51e9,
                                   dac_offset_guess=40)


def lowlvl_avcross(v, a1, b1, a2, b2, g):
    """
    Return the low level of an avoided crossing
    """
    e1 = a1*v + b1
    e2 = a2*v + b2
    dE = e1 - e2
    return 0.5*((e1+e2)-np.sqrt(dE**2+4*g*g))


def highlvl_avcross(v, a1, b1, a2, b2, g):
    """
    Return the high level of an avoided crossing
    """
    e1 = a1*v + b1
    e2 = a2*v + b2
    dE = e1 - e2
    return 0.5*((e1+e2)+np.sqrt(dE**2+4*g*g))


def mixlvl_avcross(v, a1, b1, a2, b2, g):
    """
    Returns the low/high level of an avoided crossing for v below/above 0
    """
    if v > 0:
        return 0.5*((e1+e2)+np.sqrt(dE**2+4*g*g))
    else:
        return 0.5*((e1+e2)-np.sqrt(dE**2+4*g*g))

mixAC_Model = lmfit.Model(mixlvl_avcross)