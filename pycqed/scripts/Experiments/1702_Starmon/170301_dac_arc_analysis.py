from pycqed.analysis import composite_analysis as ca
# Extraction of T2e
scan_start = '20170301_171007'
scan_stop = '20170301_180000'
qubit_name = 'QL'

###############################################
pdict = {'f_q': 'QL.f_qubit',
         'flux': 'IVVI.dac2',
         'T1': 'Analysis.Fitted Params F|1>.tau.value',
         'T1_std': 'Analysis.Fitted Params F|1>.tau.stderr', }
opt_dict = {'scan_label': 'T1_QL'}
nparams = ['f_q', 'T1', 'T1_std', 'flux']

T1_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop,
                             options_dict=opt_dict,
                             params_dict_TD=pdict, numeric_params=nparams)

freqs = T1_scans.TD_dict['f_q']
flux = T1_scans.TD_dict['flux']
T1s = T1_scans.TD_dict['T1']
T1s_std = T1_scans.TD_dict['T1_std']
