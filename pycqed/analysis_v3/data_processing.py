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


def get_data_to_process(unproc_data_dict, keys_in=None, keys_out=None):
    """
    Finds data to be processed in unproc_data_dict based on keys_in, and
    creates keys_out if keys_out is None.
    :param unproc_data_dict: OrderedDict containing data to be processed
    :param keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: measured_data.raw w0.
    :param keys_out: list of channel names or dictionary paths where the
            processed data is to be saved
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in} keys_out
    """
    if keys_in is None:
        if 'measured_data' in unproc_data_dict:
            data_to_proc_dict = unproc_data_dict['measured_data']
        else:
            data_to_proc_dict = list(unproc_data_dict.values())[-1]
    else:
        data_to_proc_dict = OrderedDict()
        key_found = True
        for keyi in keys_in:
            all_keys = keyi.split('.')
            if len(all_keys) == 1:
                try:
                    if isinstance(unproc_data_dict[all_keys[0]], dict):
                        data_to_proc_dict.update(unproc_data_dict[
                                                        all_keys[0]])
                    else:
                        data_to_proc_dict[keyi] = unproc_data_dict[
                            all_keys[0]]

                except KeyError:
                    try:
                        data_to_proc_dict[keyi] = unproc_data_dict[
                            'measured_data'][keyi]
                    except KeyError:
                        key_found = False
            else:
                try:
                    data = deepcopy(unproc_data_dict)
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

    if keys_out is None:
        keys_out = list(data_to_proc_dict)
    return data_to_proc_dict, keys_out


def get_param(name, data_dict, default_value=None, **func_pars):
    value = func_pars.get(name,
                          data_dict.get('exp_metadata',
                                        dict()).get(name,
                                                    default_value))
    return value


def filter_data(data_dict, keys_in, keys_out, **params):
    """
    Filters data in raw_data_dict for each ch_in according to data_filter
    in params. Puts the filtered data in ch_out

    To be used for example for filtering:
        - reset readouts
        - data with and without flux pulse/ pi pulse etc.

    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in data_dict for 
                    the data to be processed
    :param keys_out: list of key names or dictionary keys paths in data_dict for 
                    the processed data to be saved into
    :param params: keyword arguments, for ex: data_filter function
    """
    # keys_in = params.get('keys_in', None)
    # keys_out = params.get('keys_out', None)
    data_to_proc_dict, keys_out = get_data_to_process(
        data_dict, keys_in, keys_out)
    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in must have the same length.')
    data_filter_func = get_param('data_filter', data_dict,
                                 default_value=lambda data: data, **params)
    for keyo, keyi in zip(keys_out, list(data_to_proc_dict)):
        data = data_dict
        all_keys = keyo.split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]
        data[all_keys[-1]] = data_filter_func(data_to_proc_dict[keyi])
    return data_dict


def do_preselection(data_dict, keys_in=None, keys_out=None, **params):
    data_to_proc_dict, keys_out = get_data_to_process(
        data_dict, keys_in, keys_out)
    presel_ros = get_param('presel_ros', data_dict,
                           default_value=lambda data: data[0::2], **params)
    ro_thresholds = get_param('ro_thresholds', data_dict, **params)

    for i, keyi in enumerate(data_to_proc_dict):
        presel_data = presel_ros(data_to_proc_dict[keyi])
        idx0 = list(data_to_proc_dict[keyi]).index(presel_data[0])
        step = list(data_to_proc_dict[keyi]).index(presel_data[1]) - idx0
        msmt_data = [data_to_proc_dict[keyi][i::step] for i in
                     range(idx0+1, step)]
        msmt_data = np.vstack(msmt_data).reshape((-1,), order='F')

        out_keys = keys_out[2*i:2*i + 2]
        for j, keyo in enumerate(out_keys):
            data = data_dict
            all_keys = keyo.split('.')
            for i in range(len(all_keys)-1):
                data[all_keys[i]] = OrderedDict()
                data = data[all_keys[i]]
            data[all_keys[-1]] = presel_data if j == 0 else msmt_data
    return data_dict


def cal_points_rotation():
    """
    Rotates based on calibration points object (done in this object?)
    :return: {p_state0: state_probabilities_state0,
              p_state1: state_probabilities_state1,
              ...}
    """
    pass




