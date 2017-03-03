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


def qubit_freq_cos(f_max, flux, flux_offset, flux_coeff, asymmetry, E_c):
    #     return f_max-flux_coeff*flux-flux_offset*flux*flux
    # return
    # f_max*np.sqrt(np.abs(np.cos(np.pi*(flux_coeff*flux-flux_offset))))
    return (f_max + E_c)*(asymmetry**2 + (1-asymmetry**2)*np.cos(np.pi*(flux-flux_offset)*flux_coeff)**2)**0.25-E_c


def avoided_crossing_flux_1d(fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c, fluxes_state):
    result_f = []

    frequencies = np.zeros([len(fluxes), 2])
    f2 = f_cte
    for kk, flux in enumerate(fluxes):
        # requirement for black box: takes in parameters, returns frequencies for the solutions of the hamiltonian
        # start of black-box
        f1 = qubit_freq_cos(
            f_max, flux, flux_offset, flux_coeff, asymmetry, E_c)
        df = f1 - f2
        Ham = np.matrix([[f1, g], [g, f2]])
        frequencies[kk, :] = np.sort(np.linalg.eigvals(Ham))
#         determinant = np.sqrt(df**2+4*g**2)
# frequencies[kk,:] = 0.5*(f1+f2-determinant),0.5*(f1+f2+determinant)
# #low,high

        # end of black-box
    result = np.where(fluxes_state, frequencies[:, 0], frequencies[:, 1])
    return result


def high_sol(fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c):
    return avoided_crossing_flux_1d(fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c, np.array([False]*len(fluxes)))


def low_sol(fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c):
    return avoided_crossing_flux_1d(fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c, np.array([True]*len(fluxes)))


def avoided_crossing_flux_fit(fluxes_lower, fluxes_upper, lower_freqs, upper_freqs, f_max_guess, f_cte_guess, flux_offset_guess, g_guess, asymm_guess, flux_coeff, E_c):
    total_freqs = []
    total_freqs.extend(lower_freqs)
    total_freqs.extend(upper_freqs)
    total_fluxes = []
    total_fluxes.extend(fluxes_lower)
    total_fluxes.extend(fluxes_upper)
    total_mask = []
    total_mask.extend([True]*np.ones(len(fluxes_lower)))
    total_mask.extend([False]*np.zeros(len(fluxes_upper)))
    resized_fit_func = lambda fluxes, f_cte, g, f_max, flux_offset, flux_coeff, asymmetry, E_c: avoided_crossing_flux_1d(fluxes,
                                                                                                                         f_cte,
                                                                                                                         g,
                                                                                                                         f_max,
                                                                                                                         flux_offset,
                                                                                                                         flux_coeff,
                                                                                                                         asymmetry,
                                                                                                                         E_c,
                                                                                                                         total_mask)
    av_crossing_model = lmfit.Model(resized_fit_func)

    av_crossing_model.set_param_hint('f_cte', value=f_cte_guess,
                                     min=np.min(total_freqs), max=np.max(total_freqs))
    av_crossing_model.set_param_hint('g', min=0, value=g_guess)
    av_crossing_model.set_param_hint(
        'asymmetry', min=0.5, max=0.7, vary=True, value=asymm_guess)
    av_crossing_model.set_param_hint(
        'E_c', min=0.2, max=0.26, vary=False, value=E_c)
    av_crossing_model.set_param_hint(
        'f_max', min=6.4, max=6.9, vary=True,  value=f_max_guess)
    av_crossing_model.set_param_hint(
        'flux_coeff', min=0.9, max=1.1, vary=False, value=flux_coeff)
    av_crossing_model.set_param_hint(
        'flux_offset', vary=False, value=flux_offset_guess)
    params = av_crossing_model.make_params()
    fit_res = av_crossing_model.fit(data=np.array(total_freqs),
                                    fluxes=np.array(total_fluxes),
                                    params=params)
    return fit_res


# Make the fit

low_flux = flux[filter_mask_low]
high_flux = flux[filter_mask_high]
low_freqs = peaks[filter_mask_low, 0]
high_freqs = peaks[filter_mask_high, 1]

low_flux = flux[filter_mask_low]
high_flux = flux[filter_mask_high]
low_freqs = peaks[filter_mask_low, 0]
high_freqs = peaks[filter_mask_high, 1]


fit_res = avoided_crossing_flux_fit(
    low_flux, high_flux, low_freqs, high_freqs,
    f_max_guess=6.089e9, f_cte_guess=4.68e9, flux_offset_guess=0,
    g_guess=100e6, asymm_guess=0, flux_coeff=0.0004, E_c=280e6)
