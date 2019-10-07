import logging
log = logging.getLogger(__name__)
import os
import h5py
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.hdf5_data import read_dict_from_hdf5
from pycqed.measurement.calibration_points import CalibrationPoints


def get_hdf_param_value(group, param_name):
    '''
    Returns an attribute "key" of the group "Experimental Data"
    in the hdf5 datafile.
    '''
    s = group.attrs[param_name]
    # converts byte type to string because of h5py datasaving
    if type(s) == bytes:
        s = s.decode('utf-8')
    # If it is an array of value decodes individual entries
    if type(s) == np.ndarray:
        s = [s.decode('utf-8') for s in s]
    return s


def get_params_from_hdf_file(data_dict, **params):
    params_dict = get_param('params_dict', data_dict, **params)
    numeric_params = get_param('numeric_params', data_dict,
                               default_value=[], **params)
    if params_dict is None:
        raise ValueError('params_dict was not specified.')

    raw_data_dict = []
    raw_data_dict_ts = OrderedDict([(param, []) for param in
                                    params_dict])

    folder = params.get('folder', data_dict.get('folder', None))
    if folder is None:
        raise ValueError('No folder was found.')
    h5mode = get_param('h5mode', data_dict, default_value='r+', **params)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, h5mode)

    if 'measurementstring' in raw_data_dict_ts:
        raw_data_dict_ts['measurementstring'] = \
            os.path.split(folder)[1][7:]
    if 'measured_data' in raw_data_dict_ts:
        raw_data_dict_ts['measured_data'] = \
            np.array(data_file['Experimental Data']['Data']).T

    for save_par, file_par in params_dict.items():
        if len(file_par.split('.')) == 1:
            par_name = file_par.split('.')[0]
            for group_name in data_file.keys():
                if par_name in list(data_file[group_name].attrs):
                    raw_data_dict_ts[save_par] = \
                        get_hdf_param_value(data_file[group_name], par_name)
        else:
            group_name = '/'.join(file_par.split('.')[:-1])
            par_name = file_par.split('.')[-1]
            if group_name in data_file:
                if par_name in list(data_file[group_name].attrs):
                    raw_data_dict_ts[save_par] = \
                        get_hdf_param_value(data_file[group_name], par_name)
                elif par_name in list(data_file[group_name].keys()):
                    raw_data_dict_ts[save_par] = read_dict_from_hdf5(
                        {}, data_file[group_name][par_name])
        if isinstance(raw_data_dict_ts[save_par], list) and \
                len(raw_data_dict_ts[save_par]) == 1:
            raw_data_dict_ts[save_par] = raw_data_dict_ts[save_par][0]
    raw_data_dict.append(raw_data_dict_ts)

    if len(raw_data_dict) == 1:
        raw_data_dict = raw_data_dict[0]
    for par_name in raw_data_dict:
        if par_name in numeric_params:
            raw_data_dict[par_name] = np.double(raw_data_dict[par_name])
    data_dict.update(raw_data_dict)
    return data_dict


def get_data_to_process(data_dict, keys_in=None):
    """
    Finds data to be processed in unproc_data_dict based on keys_in.

    :param unproc_data_dict: OrderedDict containing data to be processed
    :param keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: measured_data.raw w0.
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in}
    """
    if keys_in is None:
        if 'measured_data' in data_dict:
            keys_in = list(data_dict['measured_data'])
        else:
            raise ValueError('"keys_in" were not specified.')

    data_to_proc_dict = OrderedDict()
    key_found = True
    for keyi in keys_in:
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
    return data_to_proc_dict


def get_param(name, data_dict, default_value=None, raise_error=False, **params):
    value = params.get(name,
                          data_dict.get(name,
                                        data_dict.get('exp_metadata',
                                                      dict()).get(name,
                                                                  default_value)
                                        )
                       )
    if raise_error and value is None:
        raise ValueError(f'{name} was not found in either exp_metadata or '
                         f'input params.')
    return value


def get_cp_sp_spmap_measobjn(data_dict, **params):
    """
    Extracts cal_points, sweep_points, meas_obj_sweep_points_map and
    meas_obj_name from experiment metadata or from params.
    :param data_dict: OrderedDict containing experiment metadata (exp_metadata)
    :param params: keyword arguments
    :return: cal_points, sweep_points, meas_obj_sweep_points_map and
    meas_obj_name

    Assumptions:
        - if cp or sp are strings, then it assumes they can be evaluated
    """
    cp = get_param('cal_points', data_dict, raise_error=True, **params)
    if isinstance(cp, str):
        cp = eval(cp)
    sp = get_param('sweep_points', data_dict, raise_error=True, **params)
    if isinstance(sp, str):
        sp = eval(sp)
    meas_obj_sweep_points_map = get_param('meas_obj_sweep_points_map',
                                          data_dict, raise_error=True, **params)
    mobjn = get_param('meas_obj_name', data_dict, raise_error=True, **params)
    return cp, sp, meas_obj_sweep_points_map, mobjn


def get_qb_channel_map_from_file(data_dict, data_keys, **params):
    file_type = params.get('file_type', 'hdf')
    qb_names = get_param('qb_names', data_dict, **params)
    if qb_names is None:
        raise ValueError('Either channel_map or qb_names must be specified.')

    folder = get_param('folder', data_dict, **params)
    if folder is None:
        if 'folder' in data_dict:
            folder = data_dict['folder']
        else:
            raise ValueError('Path to file must be saved in '
                             'data_dict[folder] in order to extract '
                             'channel_map.')

    if file_type == 'hdf':
        qb_channel_map = a_tools.get_qb_channel_map_from_hdf(
            qb_names, value_names=data_keys, file_path=folder)
    else:
        raise ValueError('Only "hdf" files supported at the moment.')
    return qb_channel_map


## Helper functions ##
def get_msmt_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the measurement
    points (without calibration points data).
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: measured data without calibration points data
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return all_data
        else:
            return deepcopy(all_data[:-n_cal_pts])
    else:
        return all_data


def get_cal_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the calibration points
    data.
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: Calibration points data
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            return deepcopy(all_data[-n_cal_pts:])
    else:
        return np.array([])


def get_cal_sweep_points(sweep_points_array, cal_points, qb_name):
    """
    Creates the sweep points corresponding to the calibration points data as
    equally spaced number_of_cal_states points, with the spacing given by the
    spacing in sweep_points_array.
    :param sweep_points_array: array of physical sweep points
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            step = np.abs(sweep_points_array[-1] - sweep_points_array[-2])
            return np.array([sweep_points_array[-1] + i * step for
                             i in range(1, n_cal_pts + 1)])
    else:
        return np.array([])


## Plotting nodes ##
def get_cal_state_color(cal_state_label):
    if cal_state_label == 'g' or cal_state_label == r'$|g\rangle$':
        return 'k'
    elif cal_state_label == 'e' or cal_state_label == r'$|e\rangle$':
        return 'gray'
    elif cal_state_label == 'f' or cal_state_label == r'$|f\rangle$':
        return 'C8'
    else:
        return 'C4'


def get_latex_prob_label(prob_label):
    if 'pg ' in prob_label.lower():
        return r'$|g\rangle$ state population'
    elif 'pe ' in prob_label.lower():
        return r'$|e\rangle$ state population'
    elif 'pf ' in prob_label.lower():
        return r'$|f\rangle$ state population'
    else:
        return prob_label
