import numpy as np
from pycqed.analysis import composite_analysis as ca
import matplotlib.pyplot as plt

# # initial arches
scan_start = '20170123_191225'
scan_stop = '20170123_231225'

for i, scan_label in enumerate(['AncT', 'AncB', 'DataT', 'DataM', 'DataB']):

    pdict = {  # 'Heterodyne frequency': 'Heterodyne frequency',
        'frequencies': 'sweep_points',
        'flux_vec': 'FluxControl.flux_vector',
        'amp': 'amp',
        'phase': 'phase',
        '|S21|': 'measured_values'}
    opt_dict = {'scan_label': scan_label}
    nparams = ['|S21|', 'frequencies']
    spec_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                                   options_dict=opt_dict,
                                   params_dict_TD=pdict, numeric_params=nparams)
    # Extracting a param from an instrument that is not a simple value

    if i == 0:
        flux_vals = []
        for s in spec_scans.TD_dict['flux_vec']:
            start = s.find('[')+1+5*i
            end = start+5  # s.find('0. ')
            try:
                val = float(s[start:end])
            except:
                val = 0  # because regex does not catch that
            flux_vals.append(val)
        flux_vals = np.array(flux_vals)
    plot_z = np.array(spec_scans.TD_dict['amp'])

    spec_scans.plot_dicts['arches'] = {
        'plotfn': spec_scans.plot_colorx,
        'xvals': flux_vals,
        'yvals': spec_scans.TD_dict['frequencies']*1e-9,
        'zvals': plot_z.transpose(),

        'title': '%s - %s: %s Arches' % (spec_scans.TD_timestamps[0],
                                         spec_scans.TD_timestamps[-1],
                                         scan_label),
        'xlabel': r'Flux ()',
        'ylabel': r'Frequency (GHz)',
        'zlabel': 'Homodyne amplitude (mV)',
        'zrange': [0.5, 1],
        'plotsize': (8, 8),
        'cmap': 'YlGn_r'
    }

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    spec_scans.axs['arches'] = ax
    spec_scans.plot()
    ax.set_xlim(-.2, .2)

    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.title.set_fontsize(14)
    plt.savefig(scan_label+'.png')
