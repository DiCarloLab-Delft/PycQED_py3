# some convenience tools
#
import logging
log = logging.getLogger(__name__)
import numpy as np
import os
import time
import datetime
import warnings
from copy import deepcopy
from collections import OrderedDict as od
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap as lscmap

import pandas as pd
from sklearn.mixture import GaussianMixture as GM

from pycqed.utilities.get_default_datadir import get_default_datadir
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.signal import argrelextrema
# to allow backwards compatibility with old a_tools code
from .tools.file_handling import *
from .tools.data_manipulation import *
from .tools.plotting import *
import colorsys as colors
from matplotlib import cm
from pycqed.analysis import composite_analysis as RA
from pycqed.measurement import hdf5_data


from scipy.stats import multivariate_normal

datadir = get_default_datadir()
print('Data directory set to:', datadir)


######################################################################
#     Filehandling tools
######################################################################


def nearest_idx(array, value):
    '''
    find the index of the value closest to the specified value.
    '''
    return np.abs(array-value).argmin()


def nearest_value(array, value):
    '''
    find the value in the array that is closest to the specified value.
    '''
    return array[nearest_idx(array, value)]


def verify_timestamp(timestamp):
    if len(timestamp) == 6:
        daystamp = time.strftime('%Y%m%d')
        tstamp = timestamp
    elif len(timestamp) == 14:
        daystamp = timestamp[:8]
        tstamp = timestamp[8:]
    elif len(timestamp) == 15:  # In case day and timestamp separted by _
        daystamp = timestamp[:8]
        tstamp = timestamp[9:]
    else:
        raise Exception("Cannot interpret timestamp '%s'" % timestamp)

    return daystamp, tstamp


def is_older(ts0, ts1, or_equal=False):
    '''
    returns True if timestamp ts0 is an earlier data than timestamp ts1,
    False otherwise.
    '''
    if ts0 is None or ts1 is None:
        return True
    else:

        dstamp0, tstamp0 = verify_timestamp(ts0)
        dstamp1, tstamp1 = verify_timestamp(ts1)
        if not or_equal:
            # print 'isolder', (dstamp0+tstamp0) < (dstamp1+tstamp1)
            return (dstamp0+tstamp0) < (dstamp1+tstamp1)
        else:
            return ((dstamp0+tstamp0) <= (dstamp1+tstamp1))


def is_equal(ts0, ts1, or_equal=False):
    '''
    returns True if timestamp ts0 is an the same data as timestamp ts1,
    False otherwise.
    '''
    dstamp0, tstamp0 = verify_timestamp(ts0)
    dstamp1, tstamp1 = verify_timestamp(ts1)

    return (dstamp0+tstamp0) == (dstamp1+tstamp1)


def return_last_n_timestamps(n, contains=''):
    timestamps = []
    for i in range(n):
        if i == 0:
            timestamp = latest_data(contains=contains,
                                    return_timestamp=True)[0]
            timestamps.append(timestamp)
        else:
            timestamps.append(latest_data(contains=contains,
                                          older_than=timestamps[-1],
                                          return_timestamp=True)[0])
    return timestamps


def latest_data(contains='', older_than=None, newer_than=None, or_equal=False,
                return_timestamp=False, raise_exc=True,
                folder=None, return_all=False):
    '''
    finds the latest taken data with <contains> in its name.
    returns the full path of the data directory.

    if older_than is not None, then the latest data that fits and that
    is older than the date given by the timestamp older_than is returned.
    if newer_than is not None, than the latest data that fits and that
    is newer than the date given by the timestamp newer_than is returned

    If no fitting data is found, an exception is raised.
    Except when you specifically ask not to to
    this in: raise_exc = False, then a 'False' is returned.
    return_all = True: returns all the folders that satisfy
        the requirements (Cristian)
    '''
    if (folder is None):
        search_dir = datadir
    else:
        search_dir = folder

    daydirs = os.listdir(search_dir)
    if len(daydirs) == 0:
        logging.warning('No data found in datadir')
        return None

    daydirs.sort()

    measdirs = []
    i = len(daydirs)-1

    while len(measdirs) == 0 and i >= 0:
        daydir = daydirs[i]
        # this makes sure that (most) non day dirs do not get searched
        # as they should start with a digit (e.g. YYYYMMDD)
        if daydir[0].isdigit():
            all_measdirs = [d for d in os.listdir(
                os.path.join(search_dir, daydir))]
            all_measdirs.sort()
            measdirs = []
            for d in all_measdirs:
                # this routine verifies that any output directory
                # is a 'valid' directory
                # (i.e, obeys the regular naming convention)
                _timestamp = daydir + d[:6]
                try:
                    dstamp, tstamp = verify_timestamp(_timestamp)
                except:
                    continue
                timestamp = dstamp+tstamp
                if contains in d:
                    if older_than is not None:
                        if not is_older(timestamp, older_than,
                                        or_equal=or_equal):
                            continue
                    if newer_than is not None:
                        if not is_older(newer_than, timestamp,
                                        or_equal=or_equal):
                            continue
                    measdirs.append(d)
        i -= 1
    if len(measdirs) == 0:
        if raise_exc is True:
            raise Exception('No data found.')
        else:
            return False
    else:
        measdirs.sort()
        if return_all:
            return search_dir, daydir, measdirs
        measdir = measdirs[-1]
        if return_timestamp is False:
            return os.path.join(search_dir, daydir, measdir)
        else:
            return str(daydir)+str(measdir[:6]), os.path.join(
                search_dir, daydir, measdir)


def data_from_time(timestamp, folder=None):
    '''
    returns the full path of the data specified by its timestamp in the
    form YYYYmmddHHMMSS.
    '''
    if (folder is None):
        folder = datadir
    daydirs = os.listdir(folder)
    if len(daydirs) == 0:
        raise Exception('No data in the data directory specified')

    daydirs.sort()
    daystamp, tstamp = verify_timestamp(timestamp)

    if not os.path.isdir(os.path.join(folder, daystamp)):
        raise KeyError("Requested day '%s' not found" % daystamp)

    measdirs = [d for d in os.listdir(os.path.join(folder, daystamp))
                if d[:6] == tstamp]
    if len(measdirs) == 0:
        raise KeyError("Requested data '%s_%s' not found"
                       % (daystamp, tstamp))
    elif len(measdirs) == 1:
        return os.path.join(folder, daystamp, measdirs[0])
    else:
        raise NameError('Timestamp is not unique: %s ' % (measdirs))


def measurement_filename(directory=os.getcwd(), file_id=None, ext='hdf5'):
    dirname = os.path.split(directory)[1]
    if file_id is None:
        if dirname[6:9] == '_X_':
            fn = dirname[0:7]+dirname[9:]+'.'+ext
        else:
            fn = dirname+'.'+ext

    if os.path.exists(os.path.join(directory, fn)):
        return os.path.join(directory, fn)
    else:
        logging.warning("Data path '%s' does not exist" %
                        os.path.join(directory, fn))
        return None


def get_start_stop_time(timestamp):
    '''
    Retrieves start and stop time from HDF5 file timestamp.
    '''
    from pycqed.analysis import measurement_analysis as MA
    ma = MA.MeasurementAnalysis(timestamp=timestamp)
    timestring_start = get_instrument_setting(
        ma, 'MC', 'measurement_begintime')
    timestring_stop = get_instrument_setting(
        ma, 'MC', 'measurement_endtime')
    date_start, time_start = timestring_start.split(' ')
    date_stop, time_stop = timestring_stop.split(' ')
    timestamp_start = date_start.replace(
        '-', '')+'_'+time_start.replace(':', '')
    timestamp_stop = date_stop.replace('-', '')+'_'+time_stop.replace(':', '')
    return timestamp_start, timestamp_stop


def get_data_from_timestamp_legacy(timestamps, param_names, TwoD=False, max_files=None):
    from pycqed.analysis import measurement_analysis as MA
    if max_files is not None:
        get_timestamps = timestamps[:max_files]
    else:
        get_timestamps = timestamps
    data = od([(param, []) for param in param_names])
    for timestamp in get_timestamps:
        ma = MA.MeasurementAnalysis(timestamp=timestamp)
        if TwoD:
            ma.get_naming_and_values_2D()
        else:
            ma.get_naming_and_values()
        for param in param_names:
            if '.' not in param:
                special_output = {'amp': 0, 'phase': 1, 'I': 2, 'Q': 3}
                special_output.update(
                    {'I_raw': 0, 'Q_raw': 1, 'I_cal': 2, 'Q_cal': 3})
                special_output.update({'I': 0, 'Q': 1})
                # print special_output
                if param in list(special_output.keys()):
                    data[param].append(
                        getattr(ma, 'measured_values')[special_output[param]])
                elif param in dir(ma):
                    data[param].append(getattr(ma, param))
                elif param in ma.data_file.get('Experimental Data', {}):
                    data[param].append(
                        np.double(ma.data_file['Experimental Data'][param]))
                else:
                    warnings.warn(
                        'This data file attribute does not exist or hasn''t been coded for extraction.')

            else:
                if param.split('.')[0] in ma.data_file.get(
                        'Instrument settings', {}):
                    data[param].append(eval(
                        ma.data_file['Instrument settings'][
                            param.split('.')[0]].attrs[param.split('.')[1]]))
                elif param.split('.')[0] in ma.data_file.get('Analysis', {}):
                    temp = ma.data_file['Analysis']
                    for ii in range(len(param.split('.'))-1):
                        temp = temp[param.split('.')[ii]]
                    data[param].append(temp.attrs[param.split('.')[-1]])
                elif param.split('.')[0] in list(ma.data_file.keys()):
                    temp = ma.data_file
                    for ii in range(len(param.split('.'))-1):
                        temp = temp[param.split('.')[ii]]
                    try:
                        data[param].append(eval(temp[param.split('.')[-1]]))
                    except KeyError:
                        data[param].append(temp[param.split('.')[-1]])
                else:
                    warnings.warn(
                        'This data file attribute does not exist or '
                        'hasn''t been coded for extraction.')
        ma.data_file.close()
    return data


def get_param_value_from_file(file_path, instr_name, param_name, h5mode='r+'):
    data_file = h5py.File(measurement_filename(file_path), h5mode)
    instr_settings = data_file['Instrument settings']
    if instr_name in list(instr_settings.keys()):
        if param_name in list(instr_settings[instr_name].attrs):
            param_val = eval(instr_settings[instr_name].attrs[
                                       param_name])
        else:
            raise KeyError('"{}" does not exist for instrument "{}"'.format(
                param_name, instr_name))
    else:
        raise KeyError('"{}" does not exist in "Instrument settings."'.format(
            instr_name))

    return param_val


def get_qb_channel_map_from_hdf(qb_names, file_path, value_names, h5mode='r+'):

    data_file = h5py.File(measurement_filename(file_path), h5mode)
    instr_settings = data_file['Instrument settings']
    channel_map = {}

    if 'raw' in value_names[0]:
        ro_type = 'raw w'
    elif 'digitized' in value_names[0]:
        ro_type = 'digitized w'
    elif 'lin_trans' in value_names[0]:
        ro_type = 'lin_trans w'
    else:
        ro_type = 'w'

    for qbn in qb_names:
        qbchs = [str(eval(instr_settings[qbn].attrs['acq_I_channel']))]
        # eval because strings are saved as representations
        ro_acq_weight_type = eval(instr_settings[qbn].attrs['acq_weights_type'])
        if ro_acq_weight_type in ['SSB', 'DSB', 'optimal_qutrit']:
            qbchs += [str(eval(instr_settings[qbn].attrs['acq_Q_channel']))]
        channel_map[qbn] = [ch for ch in value_names for nr in qbchs
                            if ro_type+nr in ch]

    all_values_empty = np.all([len(v) == 0 for v in channel_map.values()])
    if len(channel_map) == 0 or all_values_empty:
        raise ValueError('Did not find any channels. qb_channel_map is empty.')
    return channel_map


def get_qb_thresholds_from_file(qb_names, file_path, h5mode='r+'):
    data_file = h5py.File(measurement_filename(file_path), h5mode)
    instr_settings = data_file['Instrument settings']
    thresholds = {}
    for qbn in qb_names:
        ro_channel = eval(instr_settings[qbn].attrs['RO_acq_weight_function_I'])
        thresholds[qbn] = 1.5*eval(
            instr_settings['UHFQC'].attrs['quex_thres_{}_level'.format(
                ro_channel)])
    return thresholds


def get_data_from_ma_v1(ma, param_names):
    data = od([(param, None) for param in param_names])
    for param in param_names:
        if '.' not in param:
            special_output = {'amp': 0, 'phase': 1, 'I': 2, 'Q': 3}
            special_output.update(
                {'I_raw': 0, 'Q_raw': 1, 'I_cal': 2, 'Q_cal': 3})
            special_output.update({'I': 0, 'Q': 1})
            # print special_output
            if param in list(special_output.keys()):
                data[param] = getattr(
                    ma, 'measured_values')[special_output[param]]
            elif param in dir(ma):
                data[param] = getattr(ma, param)
            elif param in ma.data_file.get('Experimental Data', {}):
                data[param] = np.double(
                    ma.data_file['Experimental Data'][param])
            else:
                warnings.warn(
                    'This data file attribute does not exist or hasn''t been coded for extraction.')

        else:
            if param.split('.')[0] in ma.data_file.get(
                    'Instrument settings', {}):
                data[param] = eval(ma.data_file['Instrument settings'][
                    param.split('.')[0]].attrs[param.split('.')[1]])
            elif param.split('.')[0] in ma.data_file.get('Analysis', {}):
                temp = ma.data_file['Analysis']
                for ii in range(len(param.split('.'))-1):
                    temp = temp[param.split('.')[ii]]
                data[param] = temp.attrs[param.split('.')[-1]]
            elif param.split('.')[0] in list(ma.data_file.keys()):
                temp = ma.data_file
                for ii in range(len(param.split('.'))-1):
                    temp = temp[param.split('.')[ii]]
                try:
                    data[param] = eval(temp[param.split('.')[-1]])
                except KeyError:
                    data[param] = temp[param.split('.')[-1]]
            else:
                warnings.warn(
                    'This data file attribute does not exist or '
                    'hasn"t been coded for extraction.')
    return data


def get_data_from_ma_v2(ma, param_names, numeric_params=None):
    data = od([(param, None) for param in param_names])
    # print 'boo7', data['amp']
    for param in param_names:
        if param == 'all_data':
            data[param] = ma.measured_values
        elif param == 'exp_metadata':
            if 'Experimental Data' not in ma.data_file or \
               'Experimental Metadata' not in ma.data_file['Experimental Data']:
                warnings.warn('The data file does not contain '
                              'experimental metadata')
            else:
                data[param] = hdf5_data.read_dict_from_hdf5(
                    {}, ma.data_file['Experimental Data']['Experimental Metadata'])
        elif param == 'fit_params':
            temp = ma.data_file['Analysis']
            for key in list(temp.keys()):
                if 'Fitted Params' in key:
                    fit_key = key
            temp = temp[fit_key]
            data[param] = {key: temp[key].attrs['value']
                           for key in list(temp.keys()) if key != 'covar'}
            free_vars = 0
            for key in list(data[param].keys()):
                if temp[key].attrs['vary']:
                    free_vars += 1
            data[param].update({key+'_err': temp[key].attrs['stderr']
                                for key in list(temp.keys()) if key != 'covar'})
            data[param].update({'chi_squared': temp.attrs['chisqr']})

            # tmp_var is a temporary fix!
            # should be removed at some point
            try:
                tmp_var = eval(ma.data_file['Instrument settings'][
                    'MC'].attrs['detector_function_name'])
            except:
                tmp_var = None
            if tmp_var == 'TimeDomainDetector':
                temp2 = ma.data_file['Instrument settings']['TD_Meas']
                exec(
                    ('cal_zero = %s' % (eval(temp2.attrs['cal_zero_points']))),
                    locals())
                exec(
                    ('cal_one = %s' % (eval(temp2.attrs['cal_one_points']))),
                    locals())
                dofs = eval(temp2.attrs['NoSegments']) - \
                    len(cal_zero) - len(cal_one)
            else:
                dofs = len(ma.sweep_points)
            dofs -= free_vars+1
            data[param].update(
                {'chi_squared_reduced': temp.attrs['chisqr']/dofs})
            data[param].update({'chi_squared_dofs': dofs})
        elif '.' not in param:
            special_output = {'amp': 0, 'phase': 1, 'I': 2, 'Q': 3}
            special_output.update(
                {'I_raw': 0, 'Q_raw': 1, 'I_cal': 2, 'Q_cal': 3})
            special_output.update({'I': 0, 'Q': 1})
            # print special_output
            # print 'boo8', ma.measured_values[special_output[param]]
            if param in special_output:
                data[param] = ma.measured_values[special_output[param]]
            elif param in dir(ma):
                data[param] = getattr(ma, param)
            elif param in list(ma.data_file.get('Experimental Data', {}).keys()):
                data[param] = np.double(
                    ma.data_file['Experimental Data'][param])
            elif param in list(ma.data_file.get('Analysis', {}).keys()):
                data[param] = np.double(ma.data_file['Analysis'][param])
            else:
                warnings.warn(
                    'The data file attribute %s does not exist or '
                    'hasn"t been coded for extraction.' % (param))
        else:
            if param.split('.')[0] in list(ma.data_file.get(
                    'Instrument settings', {}).keys()):
                data[param] = eval(ma.data_file['Instrument settings'][
                    param.split('.')[0]].attrs[param.split('.')[1]])
            else:
                extract_param = True
                if param.split('.')[0] in list(ma.data_file.get(
                        'Analysis', {}).keys()):
                    temp = ma.data_file['Analysis']
                elif param.split('.')[0] in list(ma.data_file.keys()):
                    temp = ma.data_file
                else:
                    extract_param = False
                    warnings.warn(
                        'The data file attribute %s does not exist or '
                        'hasn"t been coded for extraction.' % (param))
                if extract_param:
                    for ii in range(len(param.split('.'))-1):
                        temp = temp[param.split('.')[ii]]
                    param_end = param.split('.')[-1]
                    if param_end in list(temp.attrs.keys()):
                        try:
                            data[param] = eval(temp.attrs[param_end])
                        except TypeError:
                            data[param] = temp.attrs[param_end]
                    elif param_end in list(temp.keys()):
                        data[param] = temp[param_end].value
        if numeric_params is not None:
            if param in numeric_params:
                data[param] = np.double(data[param])

    return data


def get_data_from_ma(ma, param_names, data_version=2, numeric_params=None):
    if data_version == 1:
        data = get_data_from_ma_v1(ma, param_names)
    elif data_version == 2:
        data = get_data_from_ma_v2(ma, param_names,
                                   numeric_params=numeric_params)
    return data


def append_data_from_ma(ma, param_names, data, data_version=2,
                        numeric_params=None):

    new_data = get_data_from_ma(ma, param_names, data_version=data_version,
                                numeric_params=numeric_params)
    for param in param_names:
        data[param].append(new_data[param])


def get_data_from_timestamp_list(timestamps,
                                 param_names,
                                 TwoD=False,
                                 TwoD_tuples=False,
                                 max_files=None,
                                 filter_no_analysis=False,
                                 numeric_params=None,
                                 ma_type='MeasurementAnalysis'):
    # dirty import inside this function to prevent circular import
    # FIXME: this function is at the base of the analysis v2 but relies
    # on the old analysis in the most dirty way. Also not completely clear
    # how the data extraction works here
    from pycqed.analysis import measurement_analysis as ma
    ma_func = getattr(ma, ma_type)

    if type(timestamps) is str:
        timestamps = [timestamps]
        single_timestamp = True
    else:
        single_timestamp = False
        if type(param_names) is list:
            data = od([(param, []) for param in param_names])
        elif type(param_names) is dict:
            data = od([(param, []) for param in param_names.values()])
        else:
            ValueError("Key 'param_names' is incorrect type.")

    if max_files is not None:
        get_timestamps = timestamps[:max_files]
    else:
        get_timestamps = timestamps

    remove_timestamps = []
    for timestamp in get_timestamps:
        try:
            ana = ma_func(timestamp=timestamp, auto=False, close_file=False)
        except Exception as e:
            try:
                ana.finish()
                del ana
            except Exception:
                pass
            logging.warning(e)
            remove_timestamps.append(timestamp)
        else:
            try:
                do_analysis = True
                if filter_no_analysis:
                    if 'Analysis' in ana.data_file.keys():
                        do_analysis = True
                    else:
                        do_analysis = False
                if do_analysis:
                    if TwoD_tuples:
                        print('Extracting 2D_tuples')
                        ana.get_naming_and_values_2D_tuples()
                    elif TwoD:
                        ana.get_naming_and_values_2D()
                        print('Extracting 2D')
                    else:
                        ana.get_naming_and_values()

                    if 'datasaving_format' in ana.data_file[
                            'Experimental Data'].attrs:
                        datasaving_format = ana.get_key('datasaving_format')
                    else:
                        print(
                            'Using legacy data loading, assuming old formatting')
                        datasaving_format = 'Version 1'

                    if datasaving_format == 'Version 1':
                        if single_timestamp:
                            data = get_data_from_ma(ana, param_names.values(),
                                                    data_version=1)
                        else:
                            append_data_from_ma(ana, param_names.values(), data,
                                                data_version=1)

                    elif datasaving_format == 'Version 2':
                        if single_timestamp:
                            data = get_data_from_ma(ana, param_names.values(),
                                                    data_version=2)
                        else:
                            append_data_from_ma(
                                ana, param_names.values(), data, data_version=2)

                else:
                    remove_timestamps.append(timestamp)
                    do_analysis = True
                ana.finish()
            except Exception as inst:
                logging.warning('Error "%s" when processing timestamp %s' %
                                (inst, timestamp))
                raise

    if len(remove_timestamps) > 0:
        for timestamp in remove_timestamps:
            get_timestamps.remove(timestamp)
        print('timestamps removed by filtering:', remove_timestamps)

    if type(param_names) is list:
        out_data = data
    elif type(param_names) is dict:
        out_data = od([(key, data[val]) for key, val in param_names.items()])

    if numeric_params is not None:
        for nparam in numeric_params:
            if nparam in out_data.keys():
                try:
                    out_data[nparam] = np.array(
                        [np.double(val) for val in out_data[nparam]])
                except ValueError as instance:
                    raise(instance)

    out_data['timestamps'] = get_timestamps

    return out_data


def convert_instr_str_list_to_numeric_array(string_list):
    return np.double(string_list[:])


def get_plot_title_from_folder(folder):
    measurementstring = os.path.split(folder)[1]
    timestamp = os.path.split(os.path.split(folder)[0])[1] \
        + '/' + measurementstring[:6]
    measurementstring = measurementstring[7:]
    default_plot_title = timestamp+'\n'+measurementstring
    return default_plot_title


def file_in_folder(folder, timestamp):
    all_files = [f for f in os.listdir(folder) if timestamp in f]
    all_files.sort()

    if (len(all_files) > 1):
        print(
            'More than one file satisfies the requirements! Import: '+all_files[0])

    return all_files[0]


def get_all_msmt_filepaths(folder, suffix='hdf5', pattern=''):
    filepaths = []
    suffixlen = len(suffix)

    for root, dirs, files in os.walk(folder):
        for f in files:
            if len(f) > suffixlen and f[-suffixlen:] == suffix and pattern in f:
                filepaths.append(os.path.join(root, f))

    filepaths = sorted(filepaths)

    return filepaths


def get_instrument_setting(analysis_object, instrument_name, parameter):
    instrument_settings = analysis_object.data_file['Instrument settings']
    instrument = instrument_settings[instrument_name]
    attr = eval(instrument.attrs[parameter])
    return attr


def compare_instrument_settings_timestamp(timestamp_a, timestamp_b):
    '''
    Takes two analysis objects as input and prints the differences between
    the instrument settings. Currently it only compares settings existing in
    object_a, this function can be improved to not care about the order of
    arguments.
    '''

    h5mode = 'r+'
    h5filepath = measurement_filename(get_folder(timestamp_a))
    analysis_object_a = h5py.File(h5filepath, h5mode)
    h5filepath = measurement_filename(get_folder(timestamp_b))
    analysis_object_b = h5py.File(h5filepath, h5mode)
    sets_a = analysis_object_a['Instrument settings']
    sets_b = analysis_object_b['Instrument settings']

    for ins_key in list(sets_a.keys()):
        print()
        try:
            sets_b[ins_key]

            ins_a = sets_a[ins_key]
            ins_b = sets_b[ins_key]
            print('Instrument "%s" ' % ins_key)
            diffs_found = False
            for par_key in list(ins_a.attrs.keys()):
                try:
                    ins_b.attrs[par_key]
                except KeyError:
                    print('Instrument "%s" does have parameter "%s"' % (
                        ins_key, par_key))

                if eval(ins_a.attrs[par_key]) == eval(ins_b.attrs[par_key]):
                    pass
                else:
                    print('    "%s" has a different value '
                          ' "%s" for %s, "%s" for %s' % (
                              par_key, eval(ins_a.attrs[par_key]), timestamp_a,
                              eval(ins_b.attrs[par_key]), timestamp_b))
                    diffs_found = True

            if not diffs_found:
                print('    No differences found')
        except KeyError:
            print('Instrument "%s" not present in second settings file'
                  % ins_key)


def compare_instrument_settings(analysis_object_a, analysis_object_b):
    '''
    Takes two analysis objects as input and prints the differences between the
    instrument settings. Currently it only compares settings existing in
    object_a, this function can be improved to not care about the order
    of arguments.
    '''
    sets_a = analysis_object_a.data_file['Instrument settings']
    sets_b = analysis_object_b.data_file['Instrument settings']

    for ins_key in list(sets_a.keys()):
        print()
        try:
            sets_b[ins_key]

            ins_a = sets_a[ins_key]
            ins_b = sets_b[ins_key]
            print('Instrument "%s" ' % ins_key)
            diffs_found = False
            for par_key in list(ins_a.attrs.keys()):
                try:
                    ins_b.attrs[par_key]
                except KeyError:
                    print('Instrument "%s" does have parameter "%s"' % (
                        ins_key, par_key))

                if eval(ins_a.attrs[par_key]) == eval(ins_b.attrs[par_key]):
                    pass
                else:
                    print('    "%s" has a different value '
                          ' "%s" for a, "%s" for b' % (
                              par_key, eval(ins_a.attrs[par_key]),
                              eval(ins_b.attrs[par_key])))
                    diffs_found = True

            if not diffs_found:
                print('    No differences found')
        except KeyError:
            print('Instrument "%s" not present in second settings file'
                  % ins_key)


def get_timestamps_in_range(timestamp_start, timestamp_end=None,
                            label=None, exact_label_match=False, folder=None):
    if folder is None:
        folder = datadir
    if not isinstance(label, list):
        label = [label]

    datetime_start = datetime_from_timestamp(timestamp_start)

    if timestamp_end is None:
        datetime_end = datetime.datetime.today()
    else:
        datetime_end = datetime_from_timestamp(timestamp_end)
    days_delta = (datetime_end.date() - datetime_start.date()).days
    all_timestamps = []
    for day in reversed(list(range(days_delta+1))):
        date = datetime_start + datetime.timedelta(days=day)
        datemark = timestamp_from_datetime(date)[:8]
        try:
            all_measdirs = [d for d in os.listdir(os.path.join(folder, datemark))]
        except FileNotFoundError:
            all_measdirs = []
        # Remove all hidden folders to prevent errors
        all_measdirs = [d for d in all_measdirs if not d.startswith('.')]

        if np.all([l is not None for l in label]):
            for each_label in label:
                if exact_label_match:
                    all_measdirs = [x for x in all_measdirs if label == x]
                else:
                    all_measdirs = [x for x in all_measdirs if each_label in x]
        if (date.date() - datetime_start.date()).days == 0:
            # Check if newer than starting timestamp
            timemark_start = timemark_from_datetime(datetime_start)
            all_measdirs = [dirname for dirname in all_measdirs
                            if int(dirname[:6]) >= int(timemark_start)]

        if (date.date() - datetime_end.date()).days == 0:
            # Check if older than ending timestamp
            timemark_end = timemark_from_datetime(datetime_end)
            all_measdirs = [dirname for dirname in all_measdirs if
                            int(dirname[:6]) <= int(timemark_end)]
        timestamps = ['{}_{}'.format(datemark, dirname[:6])
                      for dirname in all_measdirs]
        timestamps.reverse()
        all_timestamps += timestamps
    # Ensures the order of the timestamps is ascending
    all_timestamps.sort()
    return all_timestamps


def get_mean_df(label, starting_timestamp, ending_timestamp,
                return_raw_dataframes=False):
    '''
    Returns a dataframe containing the mean and standard error of mean (sem)
    of all datasets that match a certain label.
    Function assumes that the the datasets have identical sweep points.

    if return raw_dataframes
    '''
    # Import within function statement to prevent circular import
    from pycqed.analysis import measurement_analysis as MA
    timestamps = get_timestamps_in_range(timestamp_start=starting_timestamp,
                                         timestamp_end=ending_timestamp,
                                         label=label)
    for i, timestamp in enumerate(timestamps):
        ana = MA.MeasurementAnalysis(timestamp=timestamp, auto=False,
                                     close_file=True, close_fig=True)
        ana.get_naming_and_values()
        if i == 0:  # Initialize appropriate number of dataframes
            n_variables = len(ana.value_names)
            dataframes = [pd.DataFrame() for i in range(n_variables)]
        for j in range(len(ana.value_names)):
            dataframes[j][timestamp] = ana.measured_values[j]
        ana.finish()

    # Create the combined dataframe
    mean_df = pd.DataFrame()
    # Add sweep points to dataframe
    for i, par_name in enumerate(ana.parameter_names):
        if len(ana.parameter_names) != 1:
            mean_df[par_name] = ana.sweep_points[i]
        else:
            mean_df[par_name] = ana.sweep_points
    # Add the mean and sem to the dataframe
    for i, val_name in enumerate(ana.value_names):
        mean_df[val_name+'_mean'] = dataframes[i].mean(axis=1)
        mean_df[val_name+'_sem'] = dataframes[i].sem(axis=1)

    print('Found %s timestamps matching %s' % (len(timestamps), label))
    if return_raw_dataframes:
        return mean_df, dataframes
    return mean_df
######################################################################
#    Analysis tools
######################################################################


def get_folder(timestamp=None, older_than=None, label='',
               suppress_printing=True, **kw):
    if timestamp is not None:
        folder = data_from_time(timestamp)
        if not suppress_printing:
            print('loaded file from folder "%s" using timestamp "%s"' % (
                folder, timestamp))
    elif older_than is not None:
        folder = latest_data(label, older_than=older_than)
        if not suppress_printing:
            print('loaded file from folder "%s"using older_than "%s"' % (
                folder, older_than))
    else:
        folder = latest_data(label)
        if not suppress_printing:
            print('loaded file from folder "%s" using label "%s"' % (
                folder, label))
    return folder


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the
        signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are
        minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an
            odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    """
    if int(window_len) & 0x1 == 0:
        window_len += 1

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')

    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')

    # Cut edges of y since a mirror image is used
    edge = (window_len - 1) / 2
    edge = int(edge)
    return y[edge:-edge]


def find_second_peak(sweep_pts=None, data_dist_smooth=None,
                     key=None, peaks=None, percentile=20, optimize=False,
                     verbose=False):

    """
    Used for qubit spectroscopy analysis. Find the second gf/2 peak/dip based on
    the location of the tallest peak found, given in peaks, which is the result
    of peak_finder. The algorithm takes the index of this tallest peak and looks
    to the right and to the left of it for the tallest peaks in these new
    ranges. The resulting two new peaks are compared and the tallest/deepest
    one is chosen.

    Args:
        sweep_pts (array):      the sweep points array in your measurement
                                (typically frequency)
        data_dist_smooth (array):   the smoothed y data of your spectroscopy
                                    measurement, typically result of
                                    calculate_distance_ground_state for a qubit
                                    spectroscopy
        key (str):  the feature to search for, 'peak' or 'dip'
        peaks (dict):   the dict returned by peak_finder containing the
                        peaks/dips values for the tallest peak around which
                        this routine will search for second peak/dip
        percentile (int):   percentile of data defining background noise; gets
                            passed to peak_finder
        optimize (bool):    the peak_finder optimize parameter
        verbose (bool):     print detailed logging information

    Returns:
        f0  (float):                frequency of ge transition
        f0_gf_over_2 (float):       frequency of gf/2 transition
        kappa_guess (float):        guess for the kappa of the ge peak/dip
        kappa_guess_ef (float):     guess for the kappa of the gf/2 peak/dip
    """

    tallest_peak = peaks[key] #the ge freq
    tallest_peak_idx = peaks[key+'_idx']
    tallest_peak_width = peaks[key+'_width']
    if verbose:
        print('Largest '+key+' is at ', tallest_peak)

    # Calculate how many data points away from the tallest peak
    # to look left and right. Should be 50MHz away.
    freq_range = sweep_pts[-1]-sweep_pts[0]
    num_points = sweep_pts.size
    # n = int(50e6*num_points/freq_range)
    # m = int(50e6*num_points/freq_range)
    n = int(1e6*num_points/freq_range)
    m = int(1e6*num_points/freq_range)

    # Search for 2nd peak (f_ge) to the right of the first (tallest)
    while int(len(sweep_pts)-1) <= int(tallest_peak_idx+n) and n > 0:
        # Reduce n if outside of range
        n -= 1
    if (int(tallest_peak_idx+n)) == sweep_pts.size:
        # If n points to the right of tallest peak is the range edge:
        n = 0

    if not ((int(tallest_peak_idx+n)) >= sweep_pts.size):
        if verbose:
            print('Searching for the second {:} starting at {:} points to the '
                  'right of the largest in the range {:.5}-{:.5}'.format(
                  key,
                  n,
                  sweep_pts[int(tallest_peak_idx+n)],
                  sweep_pts[-1]))

        peaks_right = peak_finder(
            sweep_pts[int(tallest_peak_idx+n)::],
            data_dist_smooth[int(tallest_peak_idx+n)::],
            percentile=percentile,
            num_sigma_threshold=1,
            optimize=optimize,
            window_len=0,
            key=key)

        # The peak/dip found to the right of the tallest is assumed to be
        # the ge peak, which means that the tallest was in fact the gf/2 peak
        if verbose:
            print('Right '+key+' is at ', peaks_right[key])
        subset_right = data_dist_smooth[int(tallest_peak_idx+n)::]
        val_right = subset_right[peaks_right[key+'_idx']]
        f0_right = peaks_right[key]
        kappa_guess_right = peaks_right[key+'_width']
        f0_gf_over_2_right = tallest_peak
        kappa_guess_ef_right = tallest_peak_width
    else:
        if verbose:
            print('Right '+key+' is None')
        val_right = 0
        f0_right = 0
        kappa_guess_right = 0
        f0_gf_over_2_right = tallest_peak
        kappa_guess_ef_right = tallest_peak_width

    # Search for 2nd peak (f_gf/2) to the left of the first (tallest)
    while(int(tallest_peak_idx-m)<0 and m>0):
        # Reduce m if outside of range
        m -= 1
    if int(tallest_peak_idx-m) == 0:
        # If m points to the left of tallest peak is the range edge:
        m = 0
    if not (int(tallest_peak_idx-m) <= 0):
        if verbose:
            print('Searching for the second {:} starting at {:} points to the '
                  'left of the largest, in the range {:.5}-{:.5}'.format(
                  key,
                  m,
                  sweep_pts[0],
                  sweep_pts[int(tallest_peak_idx-m-1)]) )

        peaks_left = peak_finder(
            sweep_pts[0:int(tallest_peak_idx-m)],
            data_dist_smooth[0:int(tallest_peak_idx-m)],
            percentile=percentile,
            num_sigma_threshold=1,
            optimize=optimize,
            window_len=0,
            key=key)

        # The peak/dip found to the left of the tallest is assumed to be
        # the gf/2 peak, which means that the tallest was indeed the ge peak
        if verbose:
            print('Left '+key+' is at ', peaks_left[key])
        subset_left = data_dist_smooth[0:int(tallest_peak_idx-m)]
        val_left = subset_left[peaks_left[key+'_idx']]
        f0_left = tallest_peak
        kappa_guess_left = tallest_peak_width
        f0_gf_over_2_left = peaks_left[key]
        kappa_guess_ef_left = peaks_left[key+'_width']
    else:
        if verbose:
            print('Left '+key+' is None')
        val_left = 0
        f0_left = tallest_peak
        kappa_guess_left = tallest_peak_width
        f0_gf_over_2_left = 0
        kappa_guess_ef_left = 0

    if key == 'peak':
        compare_func = lambda x, y: x > y
        compare_func_inv = lambda x, y: x < y
    else:
        compare_func = lambda x, y: x < y
        compare_func_inv = lambda x, y: x > y

    # if np.abs(val_left) > np.abs(val_right):
    if compare_func(np.abs(val_left), np.abs(val_right)):
        # If peak on the left taller than peak on the right, then
        # the second peak is to the left of the tallest and it is indeed
        # the gf/2 peak, while the tallest is the ge peak.
        # if np.abs(f0_gf_over_2_left - tallest_peak) > 50e6:
        #     # If the two peaks found are separated by at least 50MHz,
        #     # then both the ge and gf/2 have been found.
        if verbose:
            print('Both largest (f0) and second-largest (f1) '+
                  key + 's have been found. '
                  'f0 was assumed to the LEFT of f1.')
        # else:
        #     # If not, then it is just some other signal.
        #     logging.warning('The second '+key+' was not found. Fitting to '
        #                             'the next largest '+key+' found.')

        f0 = f0_left
        kappa_guess = kappa_guess_left
        f0_gf_over_2 = f0_gf_over_2_left
        kappa_guess_ef = kappa_guess_ef_left

    # elif np.abs(val_left) < np.abs(val_right):
    elif compare_func_inv(np.abs(val_left), np.abs(val_right)):
        # If peak on the right taller than peak on the left, then
        # the second peak is to the right of the tallest and it is in fact
        # the ge peak, while the tallest is the gf/2 peak.
        # if np.abs(f0_right - tallest_peak) > 50e6:
        #     # If the two peaks found are separated by at least 50MHz,
        #     # then both the ge and gf/2 have been found.
        if verbose:
            print('Both largest (f0) and second-largest (f1) ' + key +
                  's have been found. '
                  'f0 was assumed to the RIGHT of f1.')
        # else:
        #     # If not, then it is just some other signal.
        #     logging.warning('The second '+key+' was not found. Fitting to '
        #                             'the next largest '+key+' found.')
        f0 = f0_right
        kappa_guess = kappa_guess_right
        f0_gf_over_2 = f0_gf_over_2_right
        kappa_guess_ef = kappa_guess_ef_right

    else:
        # If the peaks on the right and left are equal, or cannot be compared,
        # then there was probably no second peak, and only noise was found.
        logging.warning('Only f0 has been found.')
        f0 = tallest_peak
        kappa_guess = tallest_peak_width
        f0_gf_over_2 = tallest_peak
        kappa_guess_ef = tallest_peak_width

    return f0, f0_gf_over_2, kappa_guess, kappa_guess_ef

def peak_finder_v2(x, y, perc=90, window_len=11):
    '''
    Peak finder based on argrelextrema function from scipy
    only finds maximums, this can be changed to minimum by using -y instead of y
    '''
    smoothed_y = smooth(y, window_len=window_len)
    percval = np.percentile(smoothed_y, perc)
    filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
    array_peaks = argrelextrema(filtered_y, np.greater)
    peaks_x = x[array_peaks]
    sort_mask = np.argsort(y[array_peaks])[::-1]
    return peaks_x[sort_mask]

def peak_finder_v3(x, y, smoothing=True, window_len=31, perc=99.6, factor=1):
    '''
    Peak finder based on argrelextrema function from scipy
    only finds maximums, this can be changed to minimum by using factor=-1
    '''
    if smoothing:
        y = (smooth(y,window_len=window_len)-y)
    else:
        y = y - np.average(y)
    y = factor*y
    percval = np.percentile(y, perc)
    filtered_y = np.where(y>percval,y,percval)
    array_peaks = argrelextrema(filtered_y, np.greater)
    peaks_x = x[array_peaks]
    peaks_y = y[array_peaks]
    sort_mask = np.argsort(y[array_peaks])[::-1]
    return peaks_x, peaks_y, y

def cut_edges(array, window_len=11):
    array = array[(window_len//2):-(window_len//2)]
    return array

def peak_finder(x, y, percentile=20, num_sigma_threshold=5, window_len=11,
                key=None, optimize=False):
    """
    Uses look_for_peak_dips (the old peak_finder routine renamed) to find peaks/
    dips. If optimize==False, results from look_for_peak_dips is returned,
    and this routine is the same as the old peak_finder routine.
    If optimize==True, reruns look_for_peak_dips until tallest (for peaks)
    data point/lowest (for dips) data point has been found.

    :param x:                       x data
    :param y:                       y data
    :param percentile:              percentile of data defining background noise
    :param num_sigma_threshold:     number of std deviations above background
                                    where to look for data
    :param key:                     'peak' or 'dip'; tell look_for_peaks_dips
                                    to return only the peaks or the dips
    :param optimize:                re-run look_for_peak_dips until tallest
                                    (for peaks) data point/lowest (for dip) data
                                    point has been found
    :return:                        dict of peaks, dips, or both from
                                    look_for_peaks_dips
    """

    # First search for peaks/dips
    search_result = look_for_peaks_dips(x=x, y_smoothed=y, percentile=percentile,
                                        num_sigma_threshold=num_sigma_threshold,
                                        key=key, window_len=window_len)

    if optimize:
    # Rerun if the found peak/dip is smaller/larger than the largest/
    # smallest data point

        y_for_test = deepcopy(y)
        # Figure out if the data set has peaks or dips
        if key is not None:
            key_1 = key
        else:
            if search_result['dip'] is None:
                key_1 = 'peak'
            elif search_result['peak'] is None:
                key_1 = 'dip'
                y_for_test = -y_for_test
            elif np.abs(y[search_result['dip_idx']]) < \
                    np.abs(y[search_result['peak_idx']]):
                key_1 = 'peak'
            elif np.abs(y[search_result['dip_idx']]) > \
                 np.abs(y[search_result['peak_idx']]):
                key_1 = 'dip'
                y_for_test = -y_for_test
            else:
                logging.error('Cannot determine for sure if data set has peaks '
                              'or dips. Assume peaks.')
                key_1 = 'peak'

        if (y_for_test[search_result[key_1+'_idx']] < max(y_for_test)):
            while ((y_for_test[search_result[key_1+'_idx']] < max(y_for_test)) and
                    (num_sigma_threshold<100) and
                        (len(search_result[key_1+'s_idx'])>1)):

                search_result_1 = deepcopy(search_result)
                num_sigma_threshold += 1

                search_result = look_for_peaks_dips(x=x,y_smoothed=y,
                                    percentile=percentile,
                                    num_sigma_threshold=(num_sigma_threshold),
                                    key=key_1, window_len=window_len)

                if search_result[key_1+'_idx'] is None:
                    search_result = search_result_1
                    break

            print('Peak Finder: Optimal num_sigma_threshold after optimization '
                  'is ', num_sigma_threshold)

    return search_result

def look_for_peaks_dips(x, y_smoothed, percentile=20, window_len=11,
                        num_sigma_threshold=5, key=None):

    '''
    Peak finding algorithm designed by Serwan
    1 smooth data
    2 Define the threshold based on background data
    3

    key:    'peak' or 'dip'; return only the peaks or the dips
    '''

    # Smooth the data
    y_smoothed = smooth(y_smoothed, window_len=window_len)

    # Finding peaks
    # Defining the threshold

    percval = np.percentile(y_smoothed, percentile)
    y_background = y_smoothed[y_smoothed < percval]
    background_mean = np.mean(y_background)
    background_std = np.std(y_background)
    # Threshold is defined several sigma awy from the background
    threshold = background_mean + num_sigma_threshold * background_std

    thresholdlst = np.arange(y_smoothed.size)[y_smoothed > threshold]
    #datthreshold = y_smoothed[thresholdlst]

    if thresholdlst.size is 0:
        kk = 0
    else:
        kk = thresholdlst[0]
    inpeak = False
    peakranges = []
    peak_indices = []
    peak_widths = []
    if len(thresholdlst) != 0:
        for i in thresholdlst:
            if inpeak is False:
                inpeak = True
                peakfmin = kk
                peakfmax = kk
                kk = i+1
            else:
                if kk == i:
                    peakfmax = kk
                    kk += 1
                else:
                    inpeak = False
                    peakranges.append([peakfmin, peakfmax])

            peakranges.append([peakfmin, peakfmax])
        for elem in peakranges:
            try:
                if elem[0] != elem[1]:
                    peak_indices += [elem[0] +
                                     np.argmax(y_smoothed[elem[0]:elem[1]])]
                else:
                    peak_indices += [elem[0]]
                #peak_widths += [x[elem[1]] - x[elem[0]]]
            except:
                pass

        #eliminate duplicates
        peak_indices = np.unique(peak_indices)

        #Take an approximate peak width for each peak index
        for i,idx in enumerate(peak_indices):
            if (idx+i+1)<x.size and (idx-i-1)>=0:
                #ensure data points idx+i+1 and idx-i-1 are inside sweep pts
                peak_widths += [ (x[idx+i+1] - x[idx-i-1])/5 ]
            elif (idx+i+1)>x.size and (idx-i-1)>=0:
                peak_widths += [ (x[idx] - x[idx-i])/5 ]
            elif (idx+i+1)<x.size and (idx-i-1)<0:
                peak_widths += [ (x[idx+i] - x[idx])/5 ]
            else:
                peak_widths += [ 5e6 ]

        peaks = np.take(x, peak_indices)                   #Frequencies of peaks
        peak_vals = np.take(y_smoothed, peak_indices)           #values of peaks
        peak_index = peak_indices[np.argmax(peak_vals)]
        peak_width = peak_widths[np.argmax(peak_vals)]
        peak = x[peak_index]                          #Frequency of highest peak

    else:
        peak = None
        peak_vals = None
        peak_index = None
        peaks = []
        peak_width = None

    # Finding dip_index
    # Definining the threshold
    percval = np.percentile(y_smoothed, 100-percentile)
    y_background = y_smoothed[y_smoothed > percval]
    background_mean = np.mean(y_background)
    background_std = np.std(y_background)
    # Threshold is defined several sigma awy from the background
    threshold = background_mean - num_sigma_threshold * background_std

    thresholdlst = np.arange(y_smoothed.size)[y_smoothed < threshold]
    #datthreshold = y_smoothed[thresholdlst]

    if thresholdlst.size is 0:
        kk = 0
    else:
        kk = thresholdlst[0]
    indip = False
    dipranges = []
    dip_indices = []
    dip_widths = []
    if len(thresholdlst) != 0:
        for i in thresholdlst:
            if indip is False:
                indip = True
                dipfmin = kk
                dipfmax = kk
                kk = i+1
            else:
                if kk == i:
                    dipfmax = kk
                    kk += 1
                else:
                    indip = False
                    dipranges.append([dipfmin, dipfmax])

            dipranges.append([dipfmin, dipfmax])
        for elem in dipranges:
            try:
                if elem[0] != elem[1]:
                    dip_indices += [elem[0] +
                                    np.argmin(y_smoothed[elem[0]:elem[1]])]
                else:
                    dip_indices += [elem[0]]
                #dip_widths += [x[elem[1]] - x[elem[0]]]
            except:
                pass

        #eliminate duplicates
        dip_indices = np.unique(dip_indices)

        #Take an approximate dip width for each dip index
        for i,idx in enumerate(dip_indices):
            if (idx+i+1)<x.size and (idx-i-1)>=0:         #ensure data points
                                                          #idx+i+1 and idx-i-1
                                                          #are inside sweep pts
                dip_widths += [ (x[idx+i+1] - x[idx-i-1])/5 ]
            elif (idx+i+1)>x.size and (idx-i-1)>=0:
                dip_widths += [ (x[idx] - x[idx-i])/5 ]
            elif (idx+i+1)<x.size and (idx-i-1)<0:
                dip_widths += [ (x[idx+i] - x[idx])/5 ]
            else:
                dip_widths += [ 5e6 ]

        dips = np.take(x, dip_indices)
        dip_vals = np.take(y_smoothed, dip_indices)
        dip_index = dip_indices[np.argmin(dip_vals)]
        dip_width = dip_widths[np.argmax(dip_vals)]
        dip = x[dip_index]

    else:
        dip = None
        dip_vals = None
        dip_index = None
        dips = []
        dip_width = None

    if key == 'peak':
        return {'peak': peak, 'peak_idx': peak_index, 'peak_values':peak_vals,
                'peak_width': peak_width, 'peak_widths': peak_widths,
                'peaks': peaks, 'peaks_idx': peak_indices,
                'dip': None, 'dip_idx': None, 'dip_values':[],
                'dip_width': None, 'dip_widths': [],
                'dips': [], 'dips_idx': []}
    elif key == 'dip':
        return {'peak': None, 'peak_idx': None, 'peak_values':[],
                'peak_width': None, 'peak_widths': [],
                'peaks': [], 'peaks_idx': [],
                'dip': dip, 'dip_idx': dip_index, 'dip_values':dip_vals,
                'dip_width': dip_width, 'dip_widths': dip_widths,
                'dips': dips, 'dips_idx': dip_indices}
    else:
        return {'peak': peak, 'peak_idx': peak_index, 'peak_values':peak_vals,
                'peak_width': peak_width, 'peak_widths': peak_widths,
                'peaks': peaks, 'peaks_idx': peak_indices,
                'dip': dip, 'dip_idx': dip_index, 'dip_values':dip_vals,
                'dip_width': dip_width, 'dip_widths': dip_widths,
                'dips': dips, 'dips_idx': dip_indices}


def calculate_distance_ground_state(data_real, data_imag, percentile=70,
                                    normalize=False):
    ''' Calculates the distance from the ground state by assuming that
        for the largest part of the data, the system is in its ground state
    '''
    perc_real = np.percentile(data_real, percentile)
    perc_imag = np.percentile(data_imag, percentile)

    mean_real = np.mean(np.take(data_real,
                                np.where(data_real < perc_real)[0]))
    mean_imag = np.mean(np.take(data_imag,
                                np.where(data_imag < perc_imag)[0]))

    data_real_dist = data_real - mean_real
    data_imag_dist = data_imag - mean_imag

    data_dist = np.abs(data_real_dist + 1.j * data_imag_dist)
    if normalize:
        data_dist /= np.max(data_dist)
    return data_dist

# def rotate_data_to_zero(data_I, data_Q, NoCalPoints):


def zigzag(seq, sample_0, sample_1, nr_samples):
    '''
    Splits a sequence in two sequences, one containing the odd entries, the
    other containing the even entries.
    e.g. in-> [0,1,2,3,4,5] -> out0 = [0,2,4] , out1[1,3,5]
    '''
    return seq[sample_0::nr_samples], seq[sample_1::nr_samples]


def calculate_rotation_matrix(delta_I, delta_Q):
    '''
    Calculates a matrix that rotates the data to lie along the Q-axis.
    Input can be either the I and Q coordinates of the zero cal_point or
    the difference between the 1 and 0 cal points.
    '''

    angle = np.arctan2(delta_Q, delta_I)
    rotation_matrix = np.transpose(
        np.matrix([[np.cos(angle), -1*np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]))
    return rotation_matrix


def normalize_TD_data(data, data_zero, data_one):
    """
    Normalizes measured data to refernce signals for zero and one
    """
    return (data - data_zero) / (data_one - data_zero)


def normalize_data(data):
    print(
        'a_tools.normalize_data is deprecated, recommend using '
        'a_tools.normalize_data_v2()')
    return data / np.mean(data)


def normalize_data_v2(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def normalize_2D_data(data_2D):
    for k in range(data_2D.shape[1]):
        data_2D[:, k] /= np.mean(data_2D[:, k])
    return data_2D


def normalize_2D_data_on_elements(data_2D, elements):
    '''
    Normalizes every row in a 2D array by normalizing on the mean
    of the elements specified in row_elements
    '''
    for k in range(data_2D.shape[1]):
        data_2D[:, k] /= np.mean(data_2D[elements, k])
    return data_2D


def rotate_and_normalize_data_IQ(data, cal_zero_points=None, cal_one_points=None,
                                 zero_coord=None, one_coord=None, **kw):
    '''
    Rotates and normalizes data with respect to some reference coordinates.
    There are two ways to specify the reference coordinates.
        1. Explicitly defining the coordinates
        2. Specifying which elements of the input data correspond to zero
            and one
    Inputs:
        data (numpy array) : 2D dataset that has to be rotated and normalized
        zero_coord (tuple) : coordinates of reference zero
        one_coord (tuple) : coordinates of reference one
        cal_zero_points (range) : range specifying what indices in 'data'
                                  correspond to zero
        cal_one_points (range) : range specifying what indices in 'data'
                                 correspond to one
    '''
    # Extract zero and one coordinates
    if np.all([cal_zero_points==None, cal_one_points==None,
               zero_coord==None, one_coord==None]):
        # no cal points were used
        normalized_data = rotate_and_normalize_data_no_cal_points(data=data,
                                                                  **kw)
    elif np.all([cal_one_points==None, one_coord==None]) and \
            (not np.all([cal_zero_points==None, zero_coord==None])):
        # only 2 cal points used; both are I pulses
        I_zero = np.mean(data[0][cal_zero_points])
        Q_zero = np.mean(data[1][cal_zero_points])

        # Translate the data
        trans_data = [data[0] - I_zero, data[1] - Q_zero]

        # Least squares fitting to the line through the data, that also
        # intercepts the calibration point

        from lmfit.models import LinearModel

        x = trans_data[0]
        y = trans_data[1]

        linear_model = LinearModel()
        linear_model.set_param_hint('intercept',
                                    value=0,
                                    vary=False)
        linear_model.set_param_hint('slope')
        params = linear_model.make_params()
        fit_res = linear_model.fit(data=y,
                                   x=x,
                                   params=params)

        line_slope = fit_res.params['slope'].value
        line_intercept = fit_res.params['intercept'].value
        #finx the x, y coordinates of the projected points
        x_proj = (x+line_slope*y-line_slope*line_intercept)/(line_slope**2+1)
        y_proj = line_slope*(x_proj)+line_intercept

        #find the minimum (on the y axis) point on the line
        y_min_line = min(fit_res.best_fit)
        x_min_line = x[np.argmin(fit_res.best_fit)]

        #find x,y coordinates with respect to end of line
        x_data = np.abs(x_min_line - x_proj)
        y_data = y_proj-y_min_line

        #find distance from points on line to end of line
        rotated_data = np.sqrt(x_data**2+y_data**2)

        normalized_data = rotated_data

        # #normalize data
        # max_min_distance = max(rotated_data) - min(rotated_data)
        # normalized_data = (rotated_data - min(rotated_data))/max_min_distance

    else:
        # for 4
        if zero_coord is not None:
            I_zero = zero_coord[0]
            Q_zero = zero_coord[1]
        else:
            I_zero = np.mean(data[0][cal_zero_points])
            Q_zero = np.mean(data[1][cal_zero_points])
            zero_coord = (I_zero, Q_zero)

        if one_coord is not None:
            I_one = one_coord[0]
            Q_one = one_coord[1]
        else:
            I_one = np.mean(data[0][cal_one_points])
            Q_one = np.mean(data[1][cal_one_points])
            one_coord = (I_one, Q_one)

        # Translate the data
        trans_data = [data[0] - I_zero, data[1] - Q_zero]

        # Rotate the data
        M = calculate_rotation_matrix(I_one-I_zero, Q_one-Q_zero)
        outp = [np.asarray(elem)[0] for elem in M * trans_data]
        rotated_data_ch1 = outp[0]

        # Normalize the data
        one_zero_dist = np.sqrt((I_one-I_zero)**2 + (Q_one-Q_zero)**2)
        normalized_data = rotated_data_ch1/one_zero_dist

    return [normalized_data, zero_coord, one_coord]


def rotate_and_normalize_data_no_cal_points(data, **kw):

    """
    Rotates and projects data based on principal component analysis.
    (Source: http://www.cs.otago.ac.nz/cosc453/student_tutorials/
    principal_components.pdf)
    Assumes data has shape (2, nr_sweep_pts), ie the shape of
    MeasurementAnalysis.measured_values.
    """

    #translate each column in the data by its mean
    mean_x = np.mean(data[0])
    mean_y = np.mean(data[1])
    trans_x = data[0] - mean_x
    trans_y = data[1] - mean_y

    #compute the covariance 2x2 matrix
    cov_matrix = np.cov(np.array([trans_x, trans_y]))

    #find eigenvalues and eigenvectors of the covariance matrix
    [eigvals, eigvecs] = np.linalg.eig(cov_matrix)

    #compute the transposed feature vector
    row_feature_vector = np.array([(eigvecs[0, np.argmin(eigvals)],
                                    eigvecs[1, np.argmin(eigvals)]),
                                   (eigvecs[0, np.argmax(eigvals)],
                                    eigvecs[1, np.argmax(eigvals)])])
    #compute the row_data_trans matrix with (x,y) pairs in each column. Each
    #row is a dimension (row1 = x_data, row2 = y_data)
    row_data_trans = np.array([trans_x, trans_y])
    #compute final, projected data; only the first row is of interest (it is the
    #principal axis
    final_data = np.dot(row_feature_vector, row_data_trans)
    normalized_data = final_data[1, :]

    #normalize data
    # max_min_distance = np.sqrt(max(final_data[1,:])**2 +
    #                            min(final_data[1,:])**2)
    # normalized_data = final_data[1,:]/max_min_distance
    # max_min_difference = max(normalized_data -  min(normalized_data))
    # normalized_data = (normalized_data-min(normalized_data))/max_min_difference

    return normalized_data


def rotate_and_normalize_data_1ch(data, cal_zero_points=np.arange(-4, -2, 1),
                                  cal_one_points=np.arange(-2, 0, 1), **kw):
    '''
    Normalizes data according to calibration points
    Inputs:
        data (numpy array) : 1D dataset that has to be normalized
        cal_zero_points (range) : range specifying what indices in 'data'
                                  correspond to zero
        cal_one_points (range) : range specifying what indices in 'data'
                                 correspond to one
    '''
    # Extract zero and one coordinates
    I_zero = np.mean(data[cal_zero_points])
    I_one = np.mean(data[cal_one_points])
    # Translate the date
    trans_data = data - I_zero
    # Normalize the data
    one_zero_dist = I_one-I_zero
    normalized_data = trans_data/one_zero_dist

    return normalized_data


def predict_gm_proba_from_cal_points(X, cal_points):
    """
    For each point of the data array X, predicts the probability of being
    in the states of each cal_point respectively,
    in the limit of narrow gaussians.
    Args:
        X: Data (n, n_channels)
        cal_points: array of calpoints where each row is a different state and
        columns are number of channels (n_cal_points, n_channels)
    """
    def find_prob(p, s, mu):
        approx = 0
        for mu_i, p_i in zip(mu, p):
            approx += mu_i*p_i
        diff = np.abs(s - approx)
        return np.sum(diff)
    probas = []
    initial_guess = np.ones(cal_points.shape[0])/cal_points.shape[0]
    proba_bounds = Bounds(np.zeros(cal_points.shape[0]),
                          np.ones(cal_points.shape[0]))
    proba_sum_constr = LinearConstraint(np.ones(cal_points.shape[0]),
                                        [1.], [1.])
    for pt in X:
        opt_results = minimize(find_prob, initial_guess,
                               args=(pt, cal_points), method='SLSQP',
                               bounds=proba_bounds,
                               constraints=proba_sum_constr)
        probas.append(opt_results.x)
    return np.array(probas)


def predict_gm_proba_from_clf(X, clf_params):
    """
    Predict gaussian mixture posterior probabilities for single shots
    of different levels of a qudit.
    Args:
        X: Data (n_datapoints, n_channels)
        clf_params: dictionary with parameters for Gaussian Mixture classifier
            means_: array of means of each component of the GM
            covariances_: covariance matrix
            covariance_type: type of covariance matrix
            weights_: array of priors of being in each level. (n_levels,)
            precisions_cholesky_: array of precision_cholesky

            For more info see about parameters see :
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.
            GaussianMixture.html
    Returns: (n_datapoints, n_levels) array of posterior probability of being in
        each level

    """
    reqs_params = ['means_', 'covariances_', 'covariance_type',
                   'weights_', 'precisions_cholesky_']
    clf_params = deepcopy(clf_params)
    for r in reqs_params:
        assert r in clf_params, "Required Classifier parameter {} " \
                                "not given.".format(r)
    gm = GM(covariance_type=clf_params.pop('covariance_type'))
    for param_name, param_value in clf_params.items():
        setattr(gm, param_name, param_value)
    probas = gm.predict_proba(X)
    return probas

def datetime_from_timestamp(timestamp):
    try:
        if len(timestamp) == 14:
            return datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        elif len(timestamp) == 15:
            return datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except Exception as e:
        print('Invalid timestamp :"{}"'.format(timestamp))
        raise e


def timestamp_from_datetime(date):
    return datetime.datetime.strftime(date, "%Y%m%d_%H%M%S")


def datemark_from_datetime(date):
    return datetime.datetime.strftime(date, "%Y%m%d")


def timemark_from_datetime(date):
    return datetime.datetime.strftime(date, "%H%M%S")


def current_datetime(timestamp):
    return datetime.datetime.today()


def current_timestamp():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def current_datemark():
    return time.strftime('%Y%m%d', time.localtime())


def current_timemark():
    return time.strftime('%H%M%S', time.localtime())

######################################################################
#    Plotting tools
######################################################################


def color_plot(x, y, z, fig, ax, cax=None,
               show=False, normalize=False, log=False,
               transpose=False, add_colorbar=True, **kw):
    '''
    x, and y are lists, z is a matrix with shape (len(x), len(y))
    In the future this function can be overloaded to handle different
    types of input.
    Args:
        x (list):
            x data
        y (list):
            y data
        fig (Object):
            figure object
        log (string/bool):
            True/False for z axis scaling, or any string containing any
            combination of letters x, y, z for scaling of the according axis.
            Remember to set the labels correctly.
    '''

    norm = None
    try:
        if log is True or 'z' in log:
            norm = LogNorm()

        if 'y' in log:
            y = np.log10(y)

        if 'x' in log:
            x = np.log10(x)
    except TypeError:  # log is not iterable
        pass

    # calculate coordinates for corners of color blocks
    # x coordinates
    x_vertices = np.zeros(np.array(x.shape)+1)
    x_vertices[1:-1] = (x[:-1]+x[1:])/2.
    x_vertices[0] = x[0] - (x[1]-x[0])/2.
    x_vertices[-1] = x[-1] + (x[-1]-x[-2])/2.
    # y coordinates
    y_vertices = np.zeros(np.array(y.shape)+1)
    if len(y)>1:
        y_vertices[1:-1] = (y[:-1]+y[1:])/2.
        y_vertices[0] = y[0] - (y[1]-y[0])/2.
        y_vertices[-1] = y[-1] + (y[-1]-y[-2])/2.
    else:
        y_vertices += y[0]
        y_vertices[1] *= 1.00001
    cmap_chosen = kw.get('cmap_chosen', 'viridis')

    # This version (below) does not plot the last row, but it possibly fixes
    # an issue where it wouldn't plot at all on one computer
    # Above lines work as of 26/11/2015 on La Ferrari in both
    # MA.MeasurementAnalysis and MA.TwoD_Analysis
    # x_vertices = np.array(x)-(x[1]-x[0])/2.0  # Shift to ensure centre of cmap
    # y_vertices = np.array(y)-(y[1]-y[0])/2.0  # at right position

    x_grid, y_grid = np.meshgrid(x_vertices, y_vertices)
    # print x_grid.shape, y_grid.shape



    # # mgrid sets the grid points to start at x (or y) 0 and end at the
    # latest x (or y), the slice includes the edge point through slicing
    # tricks.
    # mgrid points correspond to the center of each sqaure in the cmap

    if normalize:
        z = normalize_data_v2(z, axis=1, order=2)

    cmap = plt.get_cmap(kw.pop('cmap', cmap_chosen))
    # CMRmap is our old default

    # Empty values in the array are filled with np.nan, this ensures
    # the plot limits are set correctly.
    clim = kw.get('clim', [np.nanmin(z), np.nanmax(z)])

    if transpose:
        print('Inverting x and y axis for colormap plot')
        colormap = ax.pcolormesh(y_grid.transpose(),
                                 x_grid.transpose(),
                                 z.transpose(),
                                 linewidth=0, rasterized=True,
                                 cmap=cmap, vmin=clim[0], vmax=clim[1])
    else:
        colormap = ax.pcolormesh(x_grid, y_grid, z, cmap=cmap, norm=norm,
                                 linewidth=0, rasterized=True,
                                 vmin=clim[0], vmax=clim[1])

    plot_title = kw.pop('plot_title', None)

    xlabel = kw.pop('xlabel', None)
    ylabel = kw.pop('ylabel', None)
    x_unit = kw.pop('x_unit', None)
    y_unit = kw.pop('y_unit', None)
    zlabel = kw.pop('zlabel', None)

    xlim = kw.pop('xlim', None)
    ylim = kw.pop('ylim', None)

    if plot_title is not None:
        ax.set_title(plot_title, y=1.05, fontsize=18)

    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')

    if transpose:
        ax.set_xlim(y_vertices[0], y_vertices[-1])
        ax.set_ylim(x_vertices[0], x_vertices[-1])
        if xlim is not None:
            ax.set_xlim(ylim)
        if ylim is not None:
            ax.set_ylim(xlim)
        set_xlabel(ax, ylabel, unit=y_unit)
        set_ylabel(ax, xlabel, unit=x_unit)

    else:
        ax.set_xlim(x_vertices[0], x_vertices[-1])
        ax.set_ylim(y_vertices[0], y_vertices[-1])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        set_xlabel(ax, xlabel, unit=x_unit)
        set_ylabel(ax, ylabel, unit=y_unit)

    if add_colorbar:
        if cax is None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes('right', size='10%', pad='5%')
        cbar = plt.colorbar(colormap, cax=cax, orientation='vertical')
        if zlabel is not None:
            cbar.set_label(zlabel)
        return fig, ax, colormap, cbar
    return fig, ax, colormap


def color_plot_slices(xvals, yvals, zvals, ax=None,
                      normalize=False, log=False,
                      save_name=None, **kw):
    """
    Originally by Nathan
    Display a color figure for something like a tracked DAC sweep.
    xvals should be a single vector with values for the primary sweep.
    yvals and zvals should be a list of arrays with the sweep points and measured values.
    """
    # create a figure and set of axes
    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

    # calculate coordinates for corners of color blocks
    # x coordinates
    xvals = np.array(xvals)
    xvertices = np.zeros(np.array(xvals.shape)+1)
    xvertices[1:-1] = (xvals[:-1]+xvals[1:])/2.
    xvertices[0] = xvals[0] - (xvals[1]-xvals[0])/2
    xvertices[-1] = xvals[-1] + (xvals[-1]-xvals[-2])/2
    # y coordinates
    yvertices = []
    for xx in range(len(xvals)):
        yvertices.append(np.zeros(np.array(yvals[xx].shape)+1))
        yvertices[xx][1:-1] = (yvals[xx][:-1]+yvals[xx][1:])/2.
        yvertices[xx][0] = yvals[xx][0] - (yvals[xx][1]-yvals[xx][0])/2
        yvertices[xx][-1] = yvals[xx][-1] + (yvals[xx][-1]-yvals[xx][-2])/2

    # various plot options
    # define colormap
    cmap = kw.pop('cmap', 'viridis')
    #clim = kw.pop('clim', [None, None])
    # normalized plot
    if normalize:
        for xx in range(len(xvals)):
            zvals[xx] /= np.mean(zvals[xx])
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx])/np.log(10)

    # add blocks to plot
    #hold = kw.pop('hold', False)
    for xx in range(len(xvals)):
        tempzvals = np.array([np.append(zvals[xx], np.array(0)),
                              np.append(zvals[xx], np.array(0))]).transpose()
        # im = ax.pcolor(xvertices[xx:xx+2],
        #                yvertices[xx],
        #                tempzvals, cmap=cmap)
    return ax


def linecut_plot(x, y, z, fig, ax,
                 xlabel=None,
                 y_name='', y_unit='', log=True,
                 zlabel=None, legend=True,
                 line_offset=0, **kw):
    '''
    Plots horizontal linecuts of a 2D plot.
    x and y must be 1D arrays.
    z must be a 2D array with shape(len(x),len(y)).
    '''
    colormap = plt.cm.get_cmap('RdYlBu')
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(
                              0, 0.9, len(y))])
    for i in range(len(y)):
        label = '{}: {:.4g} {}'.format(
            y_name, y[i], y_unit)
        ax.plot(x, z[:, i], label=label)
    if log:
        ax.set_yscale('log')
    if legend:
        ax.legend(loc=0, bbox_to_anchor=(1.1, 1))
    ax.set_position([0.1, 0.1, 0.5, 0.8])
    ax.set_ylabel(xlabel)
    ax.set_ylabel(zlabel)
    return ax


def color_plot_interpolated(x, y, z, ax=None,
                            num_points=300,
                            zlabel=None, cmap='viridis',
                            interpolation_method='linear',
                            vmin=None, vmax=None,
                            N_levels=30,
                            cax=None, cbar_orientation='vertical',
                            plot_cbar=True):
    """
    Plots a heatmap using z values at coordinates (x, y) using cubic
    interpolation.
    x: 1D array
    y: 1D array
    z: 1D array

    returns
        ax: (matplotlib axis object)
        CS: (mappable used for creating colorbar)


    """
    if ax is None:
        f, ax = plt.subplots()
    # define grid.
    xi = np.linspace(min(x), max(x), num_points)
    yi = np.linspace(min(y), max(y), num_points)
    # grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]),
                  method=interpolation_method)
    CS = ax.contour(xi, yi, zi, N_levels, linewidths=0.2, colors='k',
                    vmin=vmin, vmax=vmax)
    CS = ax.contourf(xi, yi, zi, N_levels, cmap=cmap, vmin=vmin, vmax=vmax)
    if plot_cbar:
        if cax is None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes('right', size='5%', pad='5%')
        cbar = plt.colorbar(CS, cax=cax, orientation=cbar_orientation)
        if zlabel is not None:
            cbar.set_label(zlabel)
        return ax, CS, cbar
    return ax, CS

def plot_errorbars(x, y, ax=None, err_bars=None, label=None, **kw):

    color = kw.pop('color', 'C0')
    # see https://matplotlib.org/users/dflt_style_changes.html for details about
    # the 'C#" color notation
    marker = kw.pop('marker', 'none')
    linewidth = kw.pop('linewidth', 2)
    markersize = kw.pop('markersize', 2)
    capsize = kw.pop('capsize', 2)
    capthick = kw.pop('capthick', 2)

    if ax is None:
        new_plot_created = True
        f, ax = plt.subplots()
    else:
        new_plot_created = False

    if err_bars is None:
        if (len(y.shape))==1:
            err_bars = np.std(y)/np.sqrt(y.size)
        else:
            err_bars = []
            for data in y:
                err_bars.append(np.std(data)/np.sqrt(data.size))
            err_bars = np.asarray(err_bars)
        if label is None:
            label = 'stderr'

    ax.errorbar( x, y, yerr=err_bars, label=label,
                 ecolor=color, fmt=marker,
                 elinewidth=linewidth, markersize=markersize,
                 capsize=capsize, capthick=capthick, )

    if new_plot_created:
        return f,ax
    else:
        return ax


######################################################################
#    Calculations tools
######################################################################


def calculate_transmon_transitions(EC, EJ, asym=0, reduced_flux=0,
                                   no_transitions=2, dim=None):
    '''
    Calculates transmon energy levels from the full transmon qubit Hamiltonian.
    '''
    if dim is None:
        dim = no_transitions*20

    EJphi = EJ*np.sqrt(asym**2 + (1-asym**2)*np.cos(np.pi*reduced_flux)**2)

    Ham = 4*EC*np.diag(np.arange(-dim, dim+1)**2) - EJphi/2 * \
        (np.eye(2*dim+1, k=+1) + np.eye(2*dim+1, k=-1))
    HamEigs = np.linalg.eigvalsh(Ham)
    HamEigs.sort()

    transitions = HamEigs[1:]-HamEigs[:-1]

    return transitions[:no_transitions]


def fit_EC_EJ(f01, f12):
    '''
    Calculates EC and EJ from f01 and f12 by numerical optimization.
    '''
    from scipy import optimize
    # initial guesses
    EC0 = f01-f12
    EJ0 = (f01+EC0)**2/(8*EC0)

    penaltyfn = lambda Es: calculate_transmon_transitions(*Es)-[f01, f12]
    (EC, EJ), success = optimize.leastsq(penaltyfn, (EC0, EJ0))
    return EC, EJ


def solve_quadratic_equation(a, b, c, verbose=False):
    '''
    returns solutions to the quadratic equation. Will raise an error if the
    solution is negative
    '''
    d = b**2-4*a*c
    if d < 0:
        if verbose:
            print("This equation has no real solution")
        return [np.NAN, np.NAN]
    elif d == 0:
        x = (-b+np.sqrt(b**2-4*a*c))/2*a
        if verbose:
            print("This equation has one solution: ", x)
        return [x, x]
    else:
        x1 = (-b+np.sqrt((b**2)-(4*(a*c))))/(2*a)
        x2 = (-b-np.sqrt((b**2)-(4*(a*c))))/(2*a)
        if verbose:
            print("This equation has two solutions: ", x1, " or", x2)
        return [x1, x2]


# def find_min(x, y, min_target=None, return_fit=False, perc=30):
#     from scipy.signal import argrelextrema
#     from lmfit.models import QuadraticModel
#     from functools import reduce
#     # filtering through percentiles
#     th_perc = np.percentile(y, perc)
#     mask = np.where(y < th_perc, True, False)

#     # function that multiplies all elements in vector
#     multiply_vec = lambda vec: reduce(lambda x, y: x*y, vec)

#     # function that multiplies all elements in vector from idx_min
#     if min_target is None:
#         idx_min = np.argmin(y)
#     else:
#         local_min_array = argrelextrema(y, np.less)[0]
#         idx_min = local_min_array[
#             np.argmin(np.abs(x[local_min_array]-min_target))]

#     mask_ii = lambda ii: multiply_vec(
#         mask[min(idx_min, ii):max(idx_min, ii+1)])
#     # function that returns mask_ii applied for all elements of the vector mask
#     continuous_mask = np.array(
#         [m for m in map(mask_ii, np.arange(len(mask)))], dtype=np.bool)
#     # doing the fit
#     my_fit_model = QuadraticModel()

#     # !!!!! @Ramiro the continuous mask does not seem to work (all False array)
#     # I used the regular mask for now. We should discuss this
#     # x_fit = x[continuous_mask]
#     # y_fit = y[continuous_mask]

#     x_fit = x[mask]
#     y_fit = y[mask]
#     my_fit_params = my_fit_model.guess(data=y_fit, x=x_fit)
#     my_fit_res = my_fit_model.fit(data=y_fit,
#                                   x=x_fit,
#                                   pars=my_fit_params)
#     x_min = -0.5*my_fit_res.best_values['b']/my_fit_res.best_values['a']
#     y_min = my_fit_model.func(x_min, **my_fit_res.best_values)

#     if return_fit:
#         return x_min, y_min, my_fit_res
#     else:
#         return x_min, y_min

def find_min(x, y, return_fit=False, perc=30):
    from lmfit.models import QuadraticModel
    from functools import reduce
    # filtering through percentiles
    th_perc = np.percentile(y, perc)
    mask = np.where(y < th_perc, True, False)
    # function that multiplies all elements in vector
    multiply_vec = lambda vec: reduce(lambda x, y: x*y, vec)
    # function that multiplies all elements in vector from idx_min
    idx_min = np.argmin(y)
    mask_ii = lambda ii: multiply_vec(
        mask[min(idx_min, ii):max(idx_min, ii+1)])
    # function that returns mask_ii applied for all elements of the vector mask
    continuous_mask = np.array(
        [m for m in map(mask_ii, np.arange(len(mask)))], dtype=np.bool)
    # doing the fit
    my_fit_model = QuadraticModel()
    x_fit = x[continuous_mask]
    y_fit = y[continuous_mask]
    my_fit_params = my_fit_model.guess(data=y_fit, x=x_fit)
    my_fit_res = my_fit_model.fit(data=y_fit,
                                  x=x_fit,
                                  pars=my_fit_params)
    x_min = -0.5*my_fit_res.best_values['b']/my_fit_res.best_values['a']
    y_min = my_fit_model.func(x_min, **my_fit_res.best_values)

    if return_fit:
        return x_min, y_min, my_fit_res
    else:
        return x_min, y_min


def get_color_order(i, max_num, cmap='viridis'):
    # take a blue to red scale from 0 to max_num
    # uses HSV system, H_red = 0, H_green = 1/3 H_blue=2/3
    print('It is recommended to use the updated function "get_color_cycle".')
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    return cmap((i/max_num) % 1)


def get_color_order_hsv(i, max_num):
    # take a blue to red scale from 0 to max_num
    # uses HSV system, H_red = 0, H_green = 1/3 H_blue=2/3
    return colors.hsv_to_rgb(2.*float(i)/(float(max_num)*3.), 1., 1.)


def get_color_list(max_num, cmap='viridis'):
    '''
    Return an array of max_num colors take in even spacing from the
    color map cmap.
    '''
    # Default matplotlib colormaps have a discrete set of colors
    if cmap == 'tab10':
        max_num = 10
    if cmap == 'tab20':
        max_num = 20

    if isinstance(cmap, str):
        try:
            cmap = cm.get_cmap(cmap)
        except ValueError:
            logging.warning('Using Vega10 as a fallback, upgrade matplotlib')
            cmap = cm.get_cmap('Vega10')
    return [cmap(i) for i in np.linspace(0.0, 1.0, max_num)]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = lscmap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,
                                            a=minval,
                                            b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def print_pars_table(n_ts=10, pars=None):
    '''
    Prints out a table containing the value for the indicated parameters from the last N timestamps.
    Input:
        n_ts (int), number of time-stamps to include in the table (rows).
        pars (list), list spanning the parameters of interest (columns).

    Every element on the list pars need to correspond to a parameter stored in the Instrument settings of the HDF5 data-files.
    Examples:
        pars = ['IVVI.dac1', 'IVVI.dac2', 'IVVI.dac3']
        pars = ['Qubit.f_RO', 'Qubit.RO_acq_integration_length', 'Qubit.RO_pulse_power']
        pars = ['Qubit.spec_pow', 'Qubit.RO_power_cw']
    '''
    ts_list = return_last_n_timestamps(n_ts)
    pdict = {}
    nparams = []
    for i,p in enumerate(pars):
        pdict.update({p:p})
    opt_dict = {'scan_label':'','exact_label_match':True}
    # print(ts_list)
    scans = RA.quick_analysis(t_start=ts_list[-1],t_stop=ts_list[0], options_dict=opt_dict,
                      params_dict_TD=pdict,numeric_params=nparams)

    pars_line = 'timestamp \t'
    for p in pars:
        pars_line = pars_line + p + '\t'
    print(pars_line)
    for i,ts in enumerate(scans.TD_timestamps):
        i_line = '%s \t'%ts
        for p in pars:
            i_line = i_line + scans.TD_dict[p][i]+'\t'
        print(i_line)