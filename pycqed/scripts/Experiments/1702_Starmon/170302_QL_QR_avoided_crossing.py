

for dac_voltage in (0, 600, 21):
    IVVI.dac1(dac_voltage)
    QR.find_resonator_frequency()
    QR.find_frequency()


# Analyzing the data

scan_start = '20170226_213551'
scan_stop = '20170227_030743'
qubit_name = 'QR'
opt_dict = {'scan_label': 'spectroscopy_QR'}
pdict = {'amp':'amp',
         'swp':'sweep_points',
         'dac':'IVVI.dac1'}

nparams = ['amp', 'swp','dac']
spec_scans = ca.quick_analysis(t_start=scan_start,t_stop=scan_stop, options_dict=opt_dict,
                  params_dict_TD=pdict,numeric_params=nparams)

spec_scans.TD_dict['swp'].shape,spec_scans.TD_dict['dac'].shape, spec_scans.TD_dict['amp'].shape

z_norm = np.zeros(spec_scans.TD_dict['amp'].shape)
for i,d in enumerate(spec_scans.TD_dict['dac']):
    z_norm[i,:] = spec_scans.TD_dict['amp'][i,:]/max(spec_scans.TD_dict['amp'][:,1:-2].flatten())



result = z_norm[order_mask,:]

plot_times = spec_scans.TD_dict['dac'][order_mask]
plot_step = plot_times[1]-plot_times[0]

plot_x = spec_scans.TD_dict['swp'][0]*1e-9
x_step = plot_x[1]-plot_x[0]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
cmin, cmax = 0, 1#result.max()
fig_clim = [cmin, cmax]
out = flex_colormesh_plot_vs_xy(ax=ax,clim=fig_clim,cmap='YlGn_r',
                     xvals=plot_times,
                     yvals=plot_x,
                     zvals=result.transpose())
ax.set_xlabel(r'Biasing (mV)')
ax.set_ylabel(r'Frequency (GHz)')
# ax.set_xlim(xmin, xmax)
ax.set_title('%s :%s resonator dac scan'%(spec_scans.TD_timestamps[0],qubit_name))
ax.set_ylim(plot_x.min()-x_step/2.,plot_x.max()+x_step/2.)
ax.set_xlim(plot_times.min()-plot_step/2.,plot_times.max()+plot_step/2.)
# ax.set_xlim(0,50)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('right',size='10%', pad='5%')
cbar = plt.colorbar(out['cmap'],cax=cax)
cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
cbar.set_label('Homodine voltage (mV)')

fig.tight_layout()

##################################


QR.find_frequency(pulsed=True)
QR.cw_source.on()
freqs = np.arange(4.66e9, 4.71e9, 1e6)
dac_voltages= np.arange(576, 584, .2)

MC.set_sweep_function(QR.cw_source.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_function_2D(FC.flux0)
MC.set_sweep_points_2D(dac_voltages)
MC.set_detector_function(
    det.Heterodyne_probe(QR.heterodyne_instr, trigger_separation=2.8e-6))
MC.run('QR-avoided crossing', mode='2D')
QR.cw_source.off()
