"""
This example shows how to combine data from multiple linescans into a combined
heatmap, detect peaks and fit a simple cosine arc model to the data as required
for the standard curve analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis import composite_analysis as ca
from pycqed.analysis import plotting_tools as pt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools

# Extract your data
scan_start = '20170302_153551'
scan_stop = '20170302_180743'
qubit_name = 'QR'
opt_dict = {'scan_label': 'spectroscopy_QR'}
pdict = {'amp': 'amp',
         'swp': 'sweep_points',
         'dac': 'IVVI.dac1'}

nparams = ['amp', 'swp', 'dac']
spec_scans = ca.quick_analysis(t_start=scan_start,
                               t_stop=scan_stop,
                               options_dict=opt_dict,
                               params_dict_TD=pdict,
                               numeric_params=nparams)


y = (spec_scans.TD_dict['swp'])
x = (spec_scans.TD_dict['dac'])
z = (spec_scans.TD_dict['amp'])

# Create your figure
f, ax = plt.subplots()
image_dict = pt.flex_color_plot_vs_x(xvals=x,
                                     yvals=y*1e-9,
                                     zvals=z,  # clim=fig_clim,
                                     ax=ax, cmap='viridis', normalize=True)
ax.set_xlabel('Dac 1 (mV)')
ax.set_ylabel('Frequency (GHz)')


# # Optionally add a colorbar
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right', size='5%', pad='5%')
cbar = plt.colorbar(image_dict['cmap'], cax=cax)
cbar.set_label('Normalized homodine voltage (a.u.)')


#######################################
# Find peaks in the data
z_norm = np.zeros(spec_scans.TD_dict['amp'].shape)
for i, d in enumerate(spec_scans.TD_dict['dac']):
    z_norm[i, :] = (spec_scans.TD_dict['amp'][i, :] /
                    max(spec_scans.TD_dict['amp'][:, 1:-2].flatten()))

peaks = np.zeros(len(spec_scans.TD_dict['dac']))
for i in range(len(spec_scans.TD_dict['dac'])):
    p_dict = a_tools.peak_finder(spec_scans.TD_dict['swp'][i], z_norm[i, :])
#     peaks[i] = p_dict['dip']
    peaks[i] = p_dict['peak']


# Perform the fit


dacmod = fit_mods.QubitFreqDacModel

assymetry_on = False
my_fit_points = spec_scans.TD_dict['dac'][~np.isnan(peaks)]
my_fit_data = peaks[~np.isnan(peaks)]
f_max_guess = my_fit_data.max()
dac_max_guess = my_fit_points[np.argmax(my_fit_data)]

dacmod.set_param_hint('f_max', value=f_max_guess, min=0., max=8e9, vary=True)
dacmod.set_param_hint('E_c', value=0.28e9, min=0.2e9, max=0.5e9, vary=False)
dacmod.set_param_hint(
    'dac_sweet_spot', value=dac_max_guess, min=-2000, max=2000, vary=True)
dacmod.set_param_hint('dac_flux_coefficient', value=1/2000, vary=True)
dacmod.set_param_hint('asymmetry', value=0, vary=assymetry_on)

my_fit_params = dacmod.make_params()

my_fit_res = dacmod.fit(
    data=my_fit_data, dac_voltage=my_fit_points, params=my_fit_params)


