import logging
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

import lmfit
import numpy as np
import scipy as sp
import itertools
import matplotlib as mpl
from collections import OrderedDict
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.readout_analysis as roa
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from copy import deepcopy


def get_data_to_process(data_dict, data_keys_in, data_keys_out=None):
    """
    Finds data to be processed in unproc_data_dict based on data_keys_in, and
    creates data_keys_out if data_keys_out is None.
    :param unproc_data_dict: OrderedDict containing data to be processed
    :param data_keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: measured_data.raw w0.
    :param data_keys_out: list of channel names or dictionary paths where the
            processed data is to be saved
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in} data_keys_out
    """
    # if data_keys_in is None:
    #     if 'measured_data' in data_dict:
    #         data_to_proc_dict = data_dict['measured_data']
    #     else:
    #         data_to_proc_dict = list(data_dict.values())[-1]
    # else:
    data_to_proc_dict = OrderedDict()
    key_found = True
    for keyi in data_keys_in:
        all_keys = keyi.split('.')
        if len(all_keys) == 1:
            try:
                if isinstance(data_dict[all_keys[0]], dict):
                    data_to_proc_dict.update(data_dict[all_keys[0]])
                else:
                    data_to_proc_dict[keyi] = data_dict[all_keys[0]]

            except KeyError:
                try:
                    data_to_proc_dict[keyi] = data_dict[
                        'measured_data'][keyi]
                except KeyError:
                    key_found = False
        else:
            try:
                data = deepcopy(data_dict)
                for k in all_keys:
                    data = data[k]
                if isinstance(data, dict):
                    data_to_proc_dict.update({k: data[k] for k in data})
                else:
                    data_to_proc_dict[all_keys[-1]] = data
            except KeyError:
                key_found = False
        if not key_found:
            raise ValueError(f'Channel {keyi} was not found.')

    if data_keys_out is None:
        data_keys_out = list(data_to_proc_dict)
    return data_to_proc_dict, data_keys_out


def get_param(name, data_dict, default_value=None, **func_pars):
    value = func_pars.get(name,
                          data_dict.get('exp_metadata',
                                        dict()).get(name,
                                                    default_value))
    return value


def filter(data_dict, data_keys_in, data_keys_out, **params):
    """
    Filters data in raw_data_dict for each ch_in according to data_filter
    in params. Puts the filtered data in ch_out

    To be used for example for filtering:
        - reset readouts
        - data with and without flux pulse/ pi pulse etc.

    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param data_keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param data_keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments:
        data_filter (function, default: lambda data: data): filtering condition
    """
    data_to_proc_dict, data_keys_out = get_data_to_process(
        data_dict, data_keys_in, data_keys_out)
    if len(data_keys_out) != len(data_to_proc_dict):
        raise ValueError('data_keys_out and data_keys_in must have the same length.')
    data_filter_func = get_param('data_filter', data_dict,
                                 default_value=lambda data: data, **params)
    for keyo, keyi in zip(data_keys_out, list(data_to_proc_dict)):
        data = data_dict
        all_keys = keyo.split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]
        data[all_keys[-1]] = data_filter_func(data_to_proc_dict[keyi])
    return data_dict


def do_preselection(data_dict, classified_data, data_keys_out, **params):
    """
    Keeps only the data for which the preselection readout data in
    classified_data satisfies the preselection condition.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param classified_data: array of 0,1 for qubit, and 0,1,2 for qutrit
    :param data_keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        presel_ro_idxs (function, default: lambda idx: idx % 2 == 0):
            specifies which (classified) data entry is a preselection ro
        data_keys_in (list): list of key names or dictionary keys paths in
            data_dict for the data to be processed
        presel_condition (int, default: 0): 0, 1 (, or 2 for qutrit). Keeps
            data for which the data in classified data corresponding to
            preselection readouts satisfies presel_condition.
    """
    data_keys_in = params.get('data_keys_in', None)
    presel_ro_idxs = get_param('presel_ro_idxs', data_dict,
                               default_value=lambda idx: idx % 2 == 0, **params)
    presel_condition = get_param('presel_condition', data_dict,
                                 default_value=0, **params)
    if data_keys_in is not None:
        data_to_proc_dict, data_keys_out = get_data_to_process(
            data_dict, data_keys_in, data_keys_out)
        for keyo, keyi in zip(data_keys_out, list(data_to_proc_dict)):
            mask = np.zeros(len(data_to_proc_dict[keyi]))
            val = True
            for idx in range(len(data_to_proc_dict[keyi])):
                if presel_ro_idxs(idx):
                    val = (classified_data[idx] == presel_condition)
                    mask[idx] = False
                else:
                    mask[idx] = val
            preselected_data = data_to_proc_dict[keyi][mask]
            data = data_dict
            all_keys = keyo.split('.')
            for k in range(len(all_keys)-1):
                data[all_keys[k]] = OrderedDict()
                data = data[all_keys[k]]
            data[all_keys[-1]] = preselected_data
    else:
        assert (len(data_keys_out) == 1)
        mask = np.zeros(len(classified_data))
        val = True
        for idx in range(len(classified_data)):
            if presel_ro_idxs(idx):
                val = (classified_data[idx] == 0)
                mask[idx] = False
            else:
                mask[idx] = val
        data_dict[data_keys_out[0]] = classified_data[mask]
    return data_dict


def average():
    pass


def rotate_cal_states():
    """
    Rotates data based on calibration points object (done in this object?)
    :return: {p_state0: state_probabilities_state0,
              p_state1: state_probabilities_state1,
              ...}
    """
    pass




