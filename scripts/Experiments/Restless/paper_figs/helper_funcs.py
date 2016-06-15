from modules.analysis import measurement_analysis as ma
from modules.analysis import analysis_toolbox as a_tools
from modules.analysis import fitting_models as fit_mods
import time
import lmfit
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def extract_data(start_timestamp, end_timestamp, label):
    data_dict = {}
    t0 = time.time()
    timestamps = a_tools.get_timestamps_in_range(start_timestamp, end_timestamp,
                                                  label=label)
    if len(timestamps) == 0:
        raise ValueError('No timestamps in range')
    # for tst in conv_stamps:
    for tst in timestamps:
        a = ma.MeasurementAnalysis(timestamp = tst, auto=False)
        cl_idx1 = a.measurementstring.find('cl')

        if 'conv' in label:
            cl_idx0 = a.measurementstring.find('l_') # for conventional
        else:
            cl_idx0 = a.measurementstring.find('s_') # for restless
        n_cl = int(a.measurementstring[cl_idx0+2:cl_idx1])
        a.get_naming_and_values()
        a.sweep_points
        vals = a.measured_values[0]
        if n_cl in data_dict.keys():
            data_dict[n_cl] += [vals]
        else:
            data_dict[n_cl] = [vals]

    for key in data_dict.keys():
        data_dict[key] = (np.mean(data_dict[key], axis=0))
    t1 = time.time()
    print('extracting data took {:.2f}s'.format(t1-t0))
    n_cl = np.sort(list(data_dict.keys()))
    y = a.sweep_points
    Z = np.zeros((len(y), len(n_cl)))
    for i, n in enumerate(n_cl):
        Z[:, i] = data_dict[n]

    return n_cl, y, Z


def extract_fidelities_from_eps(eps, n_cl):
    rbmod = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
    rbmod.set_param_hint('Amplitude', value=-50)
    rbmod.set_param_hint('p', value=.99)
    rbmod.set_param_hint('offset', value=50)
    rbmod.set_param_hint('fidelity_per_Clifford',  # vary=False,
                         expr='(p + (1-p)/2)')
    rbmod.set_param_hint('error_per_Clifford',  # vary=False,
                         expr='1-fidelity_per_Clifford')
    rbmod.set_param_hint('fidelity_per_gate',  # vary=False,
                         expr='fidelity_per_Clifford**(1./1.875)')
    rbmod.set_param_hint('error_per_gate',  # vary=False,
                         expr='1-fidelity_per_gate')
    pars = rbmod.make_params()
    F = np.zeros(np.shape(eps)[0])
    for i in range(np.shape(eps)[0]):
        fit_res = rbmod.fit(eps[i, :], numCliff=n_cl, params=pars)
        F[i] = fit_res.params['fidelity_per_Clifford'].value

    return F


def extract_linecuts(start_timestamp, end_timestamp, label):
    data_dict = {}
    t0 = time.time()
    timestamps = a_tools.get_timestamps_in_range(start_timestamp, end_timestamp,
                                                 label=label)
    if len(timestamps) == 0:
        raise ValueError('No timestamps in range')
    # for tst in conv_stamps:
    for tst in timestamps:
        a = ma.MeasurementAnalysis(timestamp=tst, auto=False)
        cl_idx0 = a.measurementstring.find('att_')  # for restless
        cl_idx1 = a.measurementstring.find('cl')
        n_cl = int(a.measurementstring[cl_idx0+4:cl_idx1])

        att_idx0 = a.measurementstring.find('noise_')  # for restless
        att_idx1 = a.measurementstring.find('att')
        att = (a.measurementstring[att_idx0+6:att_idx1])

        a.get_naming_and_values()
        a.sweep_points
        vals = a.measured_values[0]
        # if np.mean(vals) < 55 and np.std(vals) < 3:
        if n_cl not in data_dict.keys():
            data_dict[n_cl] = {}
        if att not in data_dict[n_cl].keys():
            data_dict[n_cl][att] = []
        data_dict[n_cl][att] += [vals]
    n_cls = np.sort(list(data_dict.keys()))
    atts = np.sort(list(data_dict[n_cls[0]].keys()))

    mean_eps = np.zeros((len(n_cls), len(atts)))
    std_eps = np.zeros((len(n_cls), len(atts)))
    for i, n_cl in enumerate(n_cls):
        for j, att in enumerate(atts):
            mean_eps[i, j] = np.mean(data_dict[n_cl][att])
            std_eps[i, j] = np.std(data_dict[n_cl][att])
    return n_cls, atts, mean_eps, std_eps



