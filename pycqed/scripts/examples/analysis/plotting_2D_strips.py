"""
This example shows how to combine data from multiple linescans into a combined
heatmap that shows strips.
"""

import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis import composite_analysis as ca
from pycqed.analysis import plotting_tools as pt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                                     zvals=z,
                                     ax=ax, cmap='viridis', normalize=True)
ax.set_xlabel('Dac 1 (mV)')
ax.set_ylabel('Frequency (GHz)')


# # Optionally add a colorbar
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right', size='5%', pad='5%')
cbar = plt.colorbar(image_dict['cmap'], cax=cax)
cbar.set_label('Normalized homodine voltage (a.u.)')