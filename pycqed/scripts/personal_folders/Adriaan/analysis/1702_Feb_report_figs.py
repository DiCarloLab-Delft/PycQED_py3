"""
- Extract data
- Filter data
- Format in table
- Run PSD analysis
"""
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Extract data
from pycqed.analysis.PSD import standard_arches_psd as std_psd
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import composite_analysis as ca


"""
- Extract data
- Filter data
- Format in table
- Run PSD analysis
"""
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Extract data
from pycqed.analysis.PSD import standard_arches_psd as std_psd

from pycqed.analysis import composite_analysis as ca


def extract_T1_data(t_start='20170301_180710',
                    t_stop='20170302_042802',
                    flux_variable_name='IVVI.dac2',
                    qubit_name='QL'):
    label = 'T1'
    pdict = {'f_q': '{}.f_qubit'.format(qubit_name),
             'flux': flux_variable_name,
             'T1': 'Analysis.Fitted Params F|1>.tau.value',
             'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
    opt_dict = {'scan_label': 'T1_{}'.format(qubit_name)}
    nparams = ['f_q', 'T1', 'T1_std', 'flux']
    T1_scans = ca.quick_analysis(t_start=t_start, t_stop=t_stop,
                                 options_dict=opt_dict,
                                 params_dict_TD=pdict, numeric_params=nparams)

    freqs = T1_scans.TD_dict['f_q']
    flux = T1_scans.TD_dict['flux']
    T1s = T1_scans.TD_dict['T1']
    T1s_std = T1_scans.TD_dict['T1_std']

    T1_timestamps = T1_scans.TD_dict['timestamps']

    return (T1_timestamps, flux, freqs, T1s, T1s_std)



def extract_T2_data(T1_timestamps,
                    flux_variable_name='IVVI.dac2',
                    scan_label='Echo_QL'):
    T2_timestamps = []
    for T1_stamp in T1_timestamps:
        timestamp = a_tools.latest_data(
            scan_label, older_than=T1_stamp, return_timestamp=True)[0]
        T2_timestamps.append(timestamp)
    pdict = {'T2': 'Analysis.Fitted Params w0.tau.value',
             'T2_std': 'Analysis.Fitted Params w0.tau.stderr', }
    opt_dict = {'scan_label': ''}
    nparams = ['T2', 'T2_std']
    echo_scans = ca.quick_analysis_list(T2_timestamps,
                                   options_dict=opt_dict, extract_only=True,
                                   params_dict_TD=pdict, numeric_params=nparams)

    T2s = echo_scans.TD_dict['T2']
    T2s_std = echo_scans.TD_dict['T2_std']
    return T2s, T2s_std

T1_timestamps, flux, freqs, T1s, T1s_std = extract_T1_data()
T2e, T2e_std = extract_T2_data(T1_timestamps=T1_timestamps)
T2s, T2s_std = extract_T2_data(T1_timestamps=T1_timestamps, scan_label='Ramsey')
# Filter data

# Plot unfiltered
f, ax = plt.subplots()
ax.errorbar(flux, T1s*1e6, T1s_std*1e6,
            marker='o', linestyle='',c='C3',
            label=r'$T_1$', markerfacecolor='none')
ax.errorbar(flux, T2e*1e6, T2e_std*1e6,
            marker='o', linestyle = '', c='C2',
            label=r'$T_2$-echo', markerfacecolor='none')
ax.errorbar(flux, T2s*1e6, T2s_std*1e6,
            marker='o', linestyle = '',c='C0',
            label=r'$T_2^\star$', markerfacecolor='none')
ax.set_ylim(0, 40)
ax.set_xlabel('Dac (mV)')
ax.set_ylabel(r'$\tau$ ($\mu$s)')
# ax.set_
ax.legend()

plt.show()

# Format in table

# Run PSD analysis

PSD_input_table = std_psd.prepare_input_table(flux, freqs, T1s, T2s, T2e)

std_psd.PSD_Analysis(PSD_input_table)


# Filter data

# Format in table

# f, ax =plt.subplot()
# ax.errorbar(

# Run PSD analysis

# std_psd.prepare_input_table(dac, frequency=)
# std_psd.PSD_Analysis(
