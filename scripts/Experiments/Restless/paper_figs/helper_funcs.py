from modules.analysis import measurement_analysis as ma
from modules.analysis import analysis_toolbox as a_tools
from modules.analysis import fitting_models as fit_mods
import matplotlib.pyplot as plt
import time

import lmfit
import numpy as np
import uncertainties
from copy import deepcopy

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def calc_T1_limited_fidelity(T1, pulse_delay):
        '''
        Formula from Asaad et al.
        pulse separation is time between start of pulses
        '''
        Np = 1.875  # Number of gates per Clifford
        F_cl = (1/6*(3 + 2*np.e**(-1*pulse_delay/(2*T1)) +
                     np.e**(-pulse_delay/T1)))**Np
        p = 2*F_cl - 1

        return F_cl, p


def t_stampt_to_hours(timestamp, start_timestamp):
    return (float(timestamp[-2:])/3600+float(timestamp[-4:-2])/60
            + float(timestamp[-6:-4])
            + 24*float(timestamp[-9:-7]))-24*float(start_timestamp[-9:-7])


def t_stampt_to_seconds(timestamp, start_timestamp):
    return 3600*(float(timestamp[-2:])/3600+float(timestamp[-4:-2])/60
                  + float(timestamp[-6:-4])
            + 24*float(timestamp[-9:-7]))-24*float(start_timestamp[-9:-7])


def extract_data(start_timestamp, end_timestamp, label):
    data_dict = {}
    t0 = time.time()
    timestamps = a_tools.get_timestamps_in_range(start_timestamp, end_timestamp,
                                                 label=label)
    if len(timestamps) == 0:
        raise ValueError('No timestamps in range')
    # for tst in conv_stamps:
    for tst in timestamps:
        a = ma.MeasurementAnalysis(timestamp=tst, auto=False)
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
        try:
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
            if np.mean(vals) < 52 and np.std(vals) < 2.5:
                if n_cl not in data_dict.keys():
                    data_dict[n_cl] = {}
                if att not in data_dict[n_cl].keys():
                    data_dict[n_cl][att] = []
                data_dict[n_cl][att] += [vals]
        except Exception as e:
            print(tst)
            raise(e)
    n_cls = np.sort(list(data_dict.keys()))
    atts = np.sort(list(data_dict[n_cls[0]].keys()))

    mean_eps = np.zeros((len(n_cls), len(atts)))
    std_eps = np.zeros((len(n_cls), len(atts)))
    for i, n_cl in enumerate(n_cls):
        for j, att in enumerate(atts):
            mean_eps[i, j] = np.mean(data_dict[n_cl][att])
            # mean_eps[i, j, :] = np.mean(data_dict[n_cl][att], axis=1)
            std_eps[i, j] = np.mean(np.std(data_dict[n_cl][att], axis=1))
            # std_eps[i, j, :] = np.std(data_dict[n_cl][att], axis=1)
    return n_cls, atts, mean_eps, std_eps, data_dict


def extract_verification_data(start_timestamp, end_timestamp):

    # T1 calibration
    timestamps_T1 = a_tools.get_timestamps_in_range(
        start_timestamp, end_timestamp, label='T1')
    # RB assessment
    timestamps_RB = a_tools.get_timestamps_in_range(
        start_timestamp, end_timestamp, label='RB_30')
    cycles = len(timestamps_T1)

    T1_dict = {'time': np.zeros(cycles),
               'mean': np.zeros(cycles),
               'stderr': np.zeros(cycles),
               'F': np.zeros(cycles),
               'F_std': np.zeros(cycles)}

    RB_rstl_dict = {'time': np.zeros([cycles/2]),
                    'F': np.zeros([cycles/2]),
                    'F_std': np.zeros([cycles/2]),
                    'offset': np.zeros([cycles/2]),
                    'offset_std': np.zeros([cycles/2])}
    RB_trad_dict = deepcopy(RB_rstl_dict)

    Dux_trad_dict = {'in1_out1_attenuation': np.zeros(cycles/2),
                     'in2_out1_attenuation': np.zeros(cycles/2),
                     'in1_out1_phase': np.zeros(cycles/2)}
    Dux_rstl_dict = deepcopy(Dux_trad_dict)
    m = ma.MeasurementAnalysis(auto=False, timestamp=timestamps_T1[0])
    starting_time = t_stampt_to_hours(m.timestamp, start_timestamp=start_timestamp)

    # extracting T1 data
    for j in range(cycles):
        m = ma.MeasurementAnalysis(auto=False, timestamp=timestamps_T1[j])
        T1_dict['time'][j] = t_stampt_to_hours(m.timestamp, start_timestamp=start_timestamp) - starting_time
        T1_dict['mean'][j] = m.data_file['Analysis']['Fitted Params F|1>']['tau'].attrs['value']
        T1_dict['stderr'][j] = m.data_file['Analysis']['Fitted Params F|1>']['tau'].attrs['stderr']
        T1_val = uncertainties.ufloat(T1_dict['mean'][j], T1_dict['stderr'][j])
        F_T1, p_T1 = calc_T1_limited_fidelity(T1_val, 20e-9)
        T1_dict['F'][j] = 100*F_T1.nominal_value
        T1_dict['F_std'][j] = 100*F_T1.std_dev
        m.finish()

    # extracting the RB traditional values
    for j in range(int(cycles/2)):
        m = ma.MeasurementAnalysis(auto=False, timestamp=timestamps_RB[2*j])
        RB_trad_dict['time'][j] = t_stampt_to_hours(
            m.timestamp, start_timestamp=start_timestamp) - starting_time
        RB_trad_dict['F'][j] = 100*m.data_file['Analysis']['Fitted Params Double_curve_RB']['fidelity_per_Clifford'].attrs['value']
        RB_trad_dict['F_std'][j] = 100*m.data_file['Analysis']['Fitted Params Double_curve_RB']['fidelity_per_Clifford'].attrs['stderr']
        RB_trad_dict['offset'][j] = m.data_file['Analysis']['Fitted Params Double_curve_RB']['offset'].attrs['value']
        RB_trad_dict['offset_std'][j] = m.data_file['Analysis']['Fitted Params Double_curve_RB']['offset'].attrs['stderr']
        Dux_trad_dict['in1_out1_attenuation'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in1_out1_attenuation']
        Dux_trad_dict['in2_out1_attenuation'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in2_out1_attenuation']
        Dux_trad_dict['in1_out1_phase'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in1_out1_phase']

    #     print(Dux_trad[j, 0])
        # extracting the RB resetless values
        m = ma.MeasurementAnalysis(auto=False, timestamp=timestamps_RB[2*j+1])
        RB_rstl_dict['time'][j] = t_stampt_to_hours(
            m.timestamp, start_timestamp=start_timestamp) - starting_time
        RB_rstl_dict['F'][j] = 100*m.data_file['Analysis']['Fitted Params Double_curve_RB']['fidelity_per_Clifford'].attrs['value']
        RB_rstl_dict['F_std'][j] = 100*m.data_file['Analysis']['Fitted Params Double_curve_RB']['fidelity_per_Clifford'].attrs['stderr']
        RB_rstl_dict['offset'][j] = m.data_file['Analysis']['Fitted Params Double_curve_RB']['offset'].attrs['value']
        RB_rstl_dict['offset_std'][j] = m.data_file['Analysis']['Fitted Params Double_curve_RB']['offset'].attrs['stderr']
        Dux_rstl_dict['in1_out1_attenuation'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in1_out1_attenuation']
        Dux_rstl_dict['in2_out1_attenuation'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in2_out1_attenuation']
        Dux_rstl_dict['in1_out1_phase'][j] = \
            m.data_file['Instrument settings']['Dux'].attrs['in1_out1_phase']

    return T1_dict, RB_rstl_dict, RB_trad_dict, Dux_rstl_dict, Dux_trad_dict


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}

    code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    Width and max height in inches for IEEE journals taken from
    computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    """
    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {#'backend': 'ps',
#               'text.latex.preamble': [r'\usepackage{gensymb}'],
              'lines.markersize': 5,
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'axes.labelpad': 0,
#               'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'legend.numpoints': 1,
              'legend.frameon': False,
              'legend.markerscale': .75
    }

    plt.rcParams.update(params)
