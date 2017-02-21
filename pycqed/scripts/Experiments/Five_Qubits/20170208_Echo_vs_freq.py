import numpy as np
from qcodes.plots.pyqtgraph import QtPlot
from pycqed.instrument_drivers.physical_instruments import QuTech_AWG_Module as qwg
QWG = qwg.QuTech_AWG_Module(
                'QWG', address='192.168.0.10',
                port=5025, server_name=None)


vw = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))

omega = lambda flux, f_max, EC, asym: (
    f_max + EC) * (asym**2 + (1-asym**2)*np.cos(np.pi*flux)**2)**0.25 - EC

f_flux = [None] * 5

f_flux[0] = lambda flux: omega(flux=flux,
                               f_max=5.9433029198374685,
                               EC=0.28,
                               asym=0.)


f_flux[1] = lambda flux: omega(flux=flux,
                               f_max=6.3774980297713189,
                               EC=0.28,
                               asym=0.)


f_flux[2] = lambda flux: omega(flux=flux,
                               f_max=5.688399095,
                               EC=0.28,
                               asym=0.)


f_flux[3] = lambda flux: omega(flux=flux,
                               f_max=6.1113712558694182,
                               EC=0.28,
                               asym=0.)


f_flux[4] = lambda flux: omega(flux=flux,
                               f_max=6.7138650690678894,
                               EC=0.28,
                               asym=0.)


def sweep_flux(fl, idx):
    this_flux = Flux_Control.flux_vector()
    this_flux[idx] = fl
    Flux_Control.flux_vector(this_flux)


fluxes = np.linspace(-.2, .2, 101)
freqs = f_flux[2](fluxes*1.4+0.02)
vw.clear()
vw.add(x=fluxes, y=freqs, xlabel=('Flux (Phi0)'), ylabel='Frequency (GHz)')
vw.add(x=fluxes, y=(5.688399095-.05)*np.ones(len(fluxes)))


##############################
# Plotting the flux arcs
##############################


# Will now start the loop interleaving
#     - Calculate frequency
#     - Find frequency using Ramsey
#     - Calibrate amp
#     - Measure echo

flux_func = f_flux[2]

fluxes = np.linspace(-.07, .04, 31)
for flux in fluxes:
    # Factor 1.2 is to correct for more steeper flux curves
    Flux_Control.flux2(flux)
    DataT.f_qubit(flux_func(Flux_Control.flux2()*1.4 + 0.02)*1e9)
    print('Estimated qubit freq to {:.4g}'.format(DataT.f_qubit()))
    times = np.arange(0, .1e-6, 1e-9)
    artificial_detuning = 2/times[-1]

    for i in range(2):
        DataT.measure_ramsey(times, artificial_detuning=artificial_detuning)
        a = ma.Ramsey_Analysis(auto=True, close_fig=True)
        fitted_freq = a.fit_res.params['frequency'].value
        detuning = fitted_freq-artificial_detuning
        DataT.f_qubit(DataT.f_qubit()-detuning)

    DataT.find_frequency(method='Ramsey')
    times = np.arange(0, 35e-6, .2e-6)
    DataT.measure_echo(times=times, artificial_detuning=4/times[-1])
    times = np.arange(0, 40e-6, .5e-6)
    DataT.measure_T1(times)


####################
# Extracting data
####################

# Extraction of T2e
scan_start = '20170208_195807'
scan_stop = '20170208_213000'
qubit_name = 'DataT'

###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T1': 'Analysis.Fitted Params F|1>.tau.value',
         'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
opt_dict = {'scan_label': 'T1_DataT'}
nparams = ['f_q', 'T1', 'T1_std', 'flux']
T1_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                             options_dict=opt_dict,
                             params_dict_TD=pdict, numeric_params=nparams)

freqs = T1_scans.TD_dict['f_q']
flux = T1_scans.TD_dict['flux']
T1s = T1_scans.TD_dict['T1']
T1s_std = T1_scans.TD_dict['T1_std']

###############################################
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T2e': 'Analysis.Fitted Params w0.tau.value',
         'T2e_std': 'Analysis.Fitted Params w0.tau.stderr', }
opt_dict = {'scan_label': 'Echo_DataT'}
nparams = ['f_q', 'T2e', 'T2e_std', 'flux']
echo_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                               options_dict=opt_dict,
                               params_dict_TD=pdict, numeric_params=nparams)

freqs = echo_scans.TD_dict['f_q']
flux = echo_scans.TD_dict['flux']
T2s = echo_scans.TD_dict['T2e']
T2s_std = echo_scans.TD_dict['T2e_std']

scan_start = '20170208_213000'
scan_stop = '20170209_083000'
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T2e': 'Analysis.Fitted Params w0.tau.value',
         'T2e_std': 'Analysis.Fitted Params w0.tau.stderr', }
opt_dict = {'scan_label': 'Echo_DataT'}
nparams = ['f_q', 'T2e', 'T2e_std', 'flux']
echo_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                               options_dict=opt_dict,
                               params_dict_TD=pdict, numeric_params=nparams)
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T1': 'Analysis.Fitted Params F|1>.tau.value',
         'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
opt_dict = {'scan_label': 'T1_DataT'}
nparams = ['f_q', 'T1', 'T1_std', 'flux']
T1_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                             options_dict=opt_dict,
                             params_dict_TD=pdict, numeric_params=nparams)


freqs_d = echo_scans.TD_dict['f_q']
flux_d = echo_scans.TD_dict['flux']
T2s_d = echo_scans.TD_dict['T2e']
T2s_std_d = echo_scans.TD_dict['T2e_std']

T1s_d = T1_scans.TD_dict['T1']
T1s_std_d = T1_scans.TD_dict['T1_std']


vw.add(x=flux_d, y=freqs_d*1e-9, symbol='o',
       subplot=1,
       xlabel='Flux (a.u.)', ylabel='Frequency (GHz)')

vw.win.nextRow()
vw.add(x=flux_d, y=T2s_d*1e6, symbol='o', subplot=2,
       xlabel='Flux (a.u.)', ylabel='T2e (us)')

f, ax = plt.subplots()
ax.plot(flux_d, freqs_d*1e-9, 'o', label='AWG Flux dicsonnected')
ax.plot(flux, freqs*1e-9, 'd', label='Normal')
ax.set_xlabel('Flux (a.u.)')
ax.set_ylabel('Frequency (GHz)')
ax.set_xlim(-0.08, 0.05)
ax.set_ylim(5.55, 5.7)






scan_start = '20170209_093000'
scan_stop = '20170209_123000'
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T2e': 'Analysis.Fitted Params w0.tau.value',
         'T2e_std': 'Analysis.Fitted Params w0.tau.stderr', }
opt_dict = {'scan_label': 'Echo_DataT'}
nparams = ['f_q', 'T2e', 'T2e_std', 'flux']
echo_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                               options_dict=opt_dict,
                               params_dict_TD=pdict, numeric_params=nparams)
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T1': 'Analysis.Fitted Params F|1>.tau.value',
         'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
opt_dict = {'scan_label': 'T1_DataT'}
nparams = ['f_q', 'T1', 'T1_std', 'flux']
T1_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                             options_dict=opt_dict,
                             params_dict_TD=pdict, numeric_params=nparams)


freqs_r = echo_scans.TD_dict['f_q']
flux_r = echo_scans.TD_dict['flux']
T2s_r = echo_scans.TD_dict['T2e']
T2s_std_r = echo_scans.TD_dict['T2e_std']

T1s_r = T1_scans.TD_dict['T1']
T1s_std_r = T1_scans.TD_dict['T1_std']



scan_start = '20170209_143000'
scan_stop = '20170209_183000'
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T2e': 'Analysis.Fitted Params w0.tau.value',
         'T2e_std': 'Analysis.Fitted Params w0.tau.stderr', }
opt_dict = {'scan_label': 'Echo_DataT'}
nparams = ['f_q', 'T2e', 'T2e_std', 'flux']
echo_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                               options_dict=opt_dict,
                               params_dict_TD=pdict, numeric_params=nparams)
qubit_name = 'DataT'
###############################################
pdict = {'f_q': 'DataT.f_qubit',
         'flux': 'FluxControl.flux2',
         'T1': 'Analysis.Fitted Params F|1>.tau.value',
         'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
opt_dict = {'scan_label': 'T1_DataT'}
nparams = ['f_q', 'T1', 'T1_std', 'flux']
T1_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                             options_dict=opt_dict,
                             params_dict_TD=pdict, numeric_params=nparams)


freqs_q = echo_scans.TD_dict['f_q']
flux_q = echo_scans.TD_dict['flux']
T2s_q = echo_scans.TD_dict['T2e']
T2s_std_q = echo_scans.TD_dict['T2e_std']

T1s_q = T1_scans.TD_dict['T1']
T1s_std_q = T1_scans.TD_dict['T1_std']





#####################################
# Preparing the QWG settings
#####################################
for ch in range(4):
    QWG.set('ch{}_amp'.format(ch+1), 2)
    QWG.set('ch{}_state'.format(ch+1), True)
