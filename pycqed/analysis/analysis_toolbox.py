# some convenience tools
#
import numpy as np
import logging
import os
import time
import datetime
import warnings
from copy import deepcopy
from collections import OrderedDict as od
from matplotlib import colors
import pandas as pd
from pycqed.utilities.get_default_datadir import get_default_datadir
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy.signal import argrelextrema
from scipy import optimize
# to allow backwards compatibility with old a_tools code
from .tools.data_manipulation import *
from .tools.plotting import *
import colorsys as colors
from matplotlib import cm
from pycqed.analysis import composite_analysis as RA

# import qutip as qp
# import qutip.metrics as qpmetrics

from matplotlib.colors import LogNorm
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel, set_cbarlabel,
                                            data_to_table_png,
                                            SI_prefix_and_scale_factor)
datadir = get_default_datadir()
print('Data directory set to:', datadir)


######################################################################
#     Filehandling tools
######################################################################


def nearest_idx(array, value):
    """
    find the index of the value closest to the specified value.
    """
    return np.abs(array - value).argmin()


def nearest_value(array, value):
    """
    find the value in the array that is closest to the specified value.
    """
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
    """
    returns True if timestamp ts0 is an earlier data than timestamp ts1,
    False otherwise.
    """
    if ts0 is None or ts1 is None:
        return True
    else:

        dstamp0, tstamp0 = verify_timestamp(ts0)
        dstamp1, tstamp1 = verify_timestamp(ts1)
        if not or_equal:
            # print 'isolder', (dstamp0+tstamp0) < (dstamp1+tstamp1)
            return (dstamp0 + tstamp0) < (dstamp1 + tstamp1)
        else:
            return ((dstamp0 + tstamp0) <= (dstamp1 + tstamp1))


def is_equal(ts0, ts1, or_equal=False):
    """
    returns True if timestamp ts0 is an the same data as timestamp ts1,
    False otherwise.
    """
    dstamp0, tstamp0 = verify_timestamp(ts0)
    dstamp1, tstamp1 = verify_timestamp(ts1)

    return (dstamp0 + tstamp0) == (dstamp1 + tstamp1)


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
    """
    finds the latest taken data with <contains> in its name.
    returns the full path of the data directory.

    if older_than is not None, then the latest data that fits and that
    is older than the date given by the timestamp older_than is returned.
    if newer_than is not None, than the latest data that fits and that
    is newer than the date given by the timestamp newer_than is returned

    If no fitting data is found, an exception is raised.
    Except when you specifically ask not to do
    this in: raise_exc = False, then a 'False' is returned.
    return_all = True: returns all the folders that satisfy
        the requirements (Cristian)
    """
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
    i = len(daydirs) - 1
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
                timestamp = dstamp + tstamp
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
            return str(daydir) + str(measdir[:6]), os.path.join(
                search_dir, daydir, measdir)


def get_datafilepath_from_timestamp(timestamp):
    """
    Return the full filepath of a datafile designated by a timestamp.

    Args:
        timestamp (str)
            formatted as "YYMMHH_hhmmss""
    Return:
        filepath (str)
            the full filepath of a datafile

    Note: there also exist two separate functions that are typically
    combined in analysis to achieve the same effect.
    These are "data_from_time" and "measurement_filename".

    This function is intended to replace both of these and be faster.

    """

    # Not only verifies but also decomposes the timestamp
    daystamp, tstamp = verify_timestamp(timestamp)

    daydir = os.listdir(os.path.join(datadir, daystamp))

    # Looking for the folder starting with the right timestamp
    measdir_names = [item for item in daydir if item.startswith(tstamp)]

    if len(measdir_names) > 1:
        raise ValueError('Timestamp is not unique')
    elif len(measdir_names) == 0:
        raise ValueError('No data at timestamp.')
    measdir_name = measdir_names[0]
    # Naming follows a standard convention
    data_fp = os.path.join(datadir, daystamp, measdir_name,
                           measdir_name + '.hdf5')
    return data_fp


def data_from_time(timestamp, folder=None):
    """
    returns the full path of the data specified by its timestamp in the
    form YYYYmmddHHMMSS.
    """
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
        # if dirname[6:9] == '_X_':
        #     fn = dirname[0:7] + dirname[9:] + '.' + ext
        # else:
        fn = dirname + '.' + ext
    if os.path.exists(os.path.join(directory, fn)):
        return os.path.join(directory, fn)
    else:
        logging.warning("Data path '%s' does not exist" %
                        os.path.join(directory, fn))
        return None


def get_start_stop_time(timestamp):
    """
    Retrieves start and stop time from HDF5 file timestamp.
    """
    from pycqed.analysis import measurement_analysis as MA
    ma = MA.MeasurementAnalysis(timestamp=timestamp)
    timestring_start = get_instrument_setting(
        ma, 'MC', 'measurement_begintime')
    timestring_stop = get_instrument_setting(ma, 'MC', 'measurement_endtime')
    date_start, time_start = timestring_start.split(' ')
    date_stop, time_stop = timestring_stop.split(' ')
    timestamp_start = date_start.replace(
        '-', '') + '_' + time_start.replace(':', '')
    timestamp_stop = date_stop.replace('-', '') + '_' + time_stop.replace(':',
                                                                          '')
    return timestamp_start, timestamp_stop


def get_data_from_timestamp_legacy(timestamps, param_names, TwoD=False,
                                   max_files=None):
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
                    data[param].append(ma.data_file['Instrument settings'][
                        param.split('.')[0]].attrs[
                        param.split('.')[1]])
                elif param.split('.')[0] in ma.data_file.get('Analysis', {}):
                    temp = ma.data_file['Analysis']
                    for ii in range(len(param.split('.')) - 1):
                        temp = temp[param.split('.')[ii]]
                    data[param].append(temp.attrs[param.split('.')[-1]])
                elif param.split('.')[0] in list(ma.data_file.keys()):
                    temp = ma.data_file
                    for ii in range(len(param.split('.')) - 1):
                        temp = temp[param.split('.')[ii]]
                    data[param].append(temp[param.split('.')[-1]])
                else:
                    warnings.warn(
                        'This data file attribute does not exist or hasn''t been coded for extraction.')
        ma.data_file.close()
    return data


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
            if param.split('.')[0] in ma.data_file.get('Instrument settings',
                                                       {}):
                data[param] = ma.data_file['Instrument settings'][
                    param.split('.')[0]].attrs[param.split('.')[1]]
            elif param.split('.')[0] in ma.data_file.get('Analysis', {}):
                temp = ma.data_file['Analysis']
                for ii in range(len(param.split('.')) - 1):
                    temp = temp[param.split('.')[ii]]
                data[param] = temp.attrs[param.split('.')[-1]]
            elif param.split('.')[0] in list(ma.data_file.keys()):
                temp = ma.data_file
                for ii in range(len(param.split('.')) - 1):
                    temp = temp[param.split('.')[ii]]
                data[param] = temp[param.split('.')[-1]]
            else:
                warnings.warn(
                    'This data file attribute does not exist or hasn''t been coded for extraction.')
    return data


def get_data_from_ma_v2(ma, param_names, numeric_params=None):
    data = od([(param, None) for param in param_names])
    for param in param_names:
        if param == 'all_data':
            data[param] = ma.measured_values
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
            data[param].update({key + '_err': temp[key].attrs['stderr']
                                for key in list(temp.keys()) if key != 'covar'})
            data[param].update({'chi_squared': temp.attrs['chisqr']})

            # tmp_var is a temporary fix!
            # should be removed at some point
            try:
                tmp_var = ma.data_file['Instrument settings'][
                    'MC'].attrs['detector_function_name']
            except Exception:
                tmp_var = None
            if tmp_var == 'TimeDomainDetector':
                temp2 = ma.data_file['Instrument settings']['TD_Meas']
                exec(('cal_zero = %s' % (temp2.attrs['cal_zero_points'])),
                     locals())
                exec(('cal_one = %s' % (temp2.attrs['cal_one_points'])),
                     locals())
                # noinspection PyUnresolvedReferences
                dofs = int(temp2.attrs['NoSegments']) - \
                    len(cal_zero) - len(cal_one)
            else:
                dofs = len(ma.sweep_points)
            dofs -= free_vars + 1
            data[param].update(
                {'chi_squared_reduced': temp.attrs['chisqr'] / dofs})
            data[param].update({'chi_squared_dofs': dofs})
        elif '.' not in param:
            special_output = {'amp': 0, 'phase': 1, 'I': 2, 'Q': 3}
            special_output.update(
                {'I_raw': 0, 'Q_raw': 1, 'I_cal': 2, 'Q_cal': 3})
            special_output.update({'I': 0, 'Q': 1})
            if param in special_output:
                data[param] = ma.measured_values[special_output[param]]
            elif param in dir(ma):
                data[param] = getattr(ma, param)
            elif param in list(
                    ma.data_file.get('Experimental Data', {}).keys()):
                data[param] = np.double(
                    ma.data_file['Experimental Data'][param])
            elif param in list(ma.data_file.get('Analysis', {}).keys()):
                data[param] = np.double(ma.data_file['Analysis'][param])
            else:
                warnings.warn(
                    'The data file attribute %s does not exist or hasn\'t '
                    'been coded for extraction.' % (param))
        else:
            if param.split('.')[0] in list(
                    ma.data_file.get('Instrument settings', {}).keys()):
                data[param] = ma.data_file['Instrument settings'][
                    param.split('.')[0]].attrs[param.split('.')[1]]
            else:
                extract_param = True
                if param.split('.')[0] in list(
                        ma.data_file.get('Analysis', {}).keys()):
                    temp = ma.data_file['Analysis']
                elif param.split('.')[0] in list(ma.data_file.keys()):
                    temp = ma.data_file
                else:
                    extract_param = False
                    print(ma.folder)
                    warnings.warn(
                        'The data file attribute %s does not exist or hasn\'t '
                        'been coded for extraction.' % (param))
                if extract_param:
                    for ii in range(len(param.split('.')) - 1):
                        temp = temp[param.split('.')[ii]]
                    param_end = param.split('.')[-1]
                    if param_end in list(temp.attrs.keys()):
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
                        numeric_params=None, filter_dict=None):
    """
    Extract data from an analysis object and appends it to lists in a dict.

    params
    ------
    ma:
        analysis class to be used to extract the data?
    param_names:
        parameters to extract.
    data: (dictionary)
        dictionary containing lists of data.
    numeric_params
        this parameter is ignored (TODO remove this)
    filter_dict:
        dictionary to use to filter, keys correpond to parameter names,
        values correspond to desired values of these params. If a Value is not
        equal to the filter param, no data from that dataset is loaded at all.

    Returns
    -------
    nothing, the data object is modified inside the function scope.

    """
    if filter_dict is not None:
        param_names_filter = list(param_names)+list(filter_dict.keys())
    else:
        param_names_filter = param_names

    new_data = get_data_from_ma(ma, param_names_filter, data_version=data_version,
                                numeric_params=numeric_params)

    if filter_dict is not None:
        for k, v in filter_dict.items():
            if new_data[k] != str(v):
                break
        else:
            for param in param_names:
                data[param].append(new_data[param])
    else:
        for param in param_names:
            data[param].append(new_data[param])


def get_data_from_timestamp_list(timestamps,
                                 param_names,
                                 TwoD=False,
                                 max_files=None,
                                 filter_no_analysis=False,
                                 numeric_params=None,
                                 filter_dict=None,
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
                    if TwoD:
                        ana.get_naming_and_values_2D()
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
                                                data_version=1, filter_dict=filter_dict)

                    elif datasaving_format == 'Version 2':
                        if single_timestamp:
                            data = get_data_from_ma(ana, param_names.values(),
                                                    data_version=2)
                        else:
                            append_data_from_ma(
                                ana, param_names.values(), data, data_version=2,
                                filter_dict=filter_dict)

                else:
                    remove_timestamps.append(timestamp)
                    do_analysis = True
                ana.finish()
            except KeyError as e:

                logging.warning('KeyError "%s" when processing timestamp %s' %
                                (e, timestamp))
                logging.warning(e)

            except Exception as e:
                logging.warning('Error "%s" when processing timestamp %s' %
                                (e, timestamp))
                raise(e)

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
                    raise (instance)

    out_data['timestamps'] = get_timestamps

    return out_data


def convert_instr_str_list_to_numeric_array(string_list):
    return np.double(string_list[:])


def get_plot_title_from_folder(folder):
    measurementstring = os.path.split(folder)[1]
    timestamp = os.path.split(os.path.split(folder)[0])[1] \
        + '/' + measurementstring[:6]
    measurementstring = measurementstring[7:]
    default_plot_title = timestamp + '\n' + measurementstring
    return default_plot_title


def file_in_folder(folder, timestamp):
    all_files = [f for f in os.listdir(folder) if timestamp in f]
    all_files.sort()

    if len(all_files) > 1:
        print('More than one file satisfies the requirements! Import: '
              + all_files[0])

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
    attr = instrument.attrs[parameter]
    return attr


def compare_instrument_settings_timestamp(timestamp_a, timestamp_b):
    """
    Takes two analysis objects as input and prints the differences between the
    instrument settings. Currently it only compares settings existing in
    object_a, this function can be improved to not care about the order of
    arguments.
    """

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

                if ins_a.attrs[par_key] == ins_b.attrs[par_key]:
                    pass
                else:
                    print('    "%s" has a different value '
                          ' "%s" for %s, "%s" for %s' % (
                              par_key, ins_a.attrs[par_key], timestamp_a,
                              ins_b.attrs[par_key], timestamp_b))
                    diffs_found = True

            if not diffs_found:
                print('    No differences found')
        except KeyError:
            print('Instrument "%s" not present in second settings file'
                  % ins_key)


def compare_instrument_settings(analysis_object_a, analysis_object_b):
    """
    Takes two analysis objects as input and prints the differences between the
    instrument settings. Currently it only compares settings existing in
    object_a, this function can be improved to not care about the order of
    arguments.
    """
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

                if ins_a.attrs[par_key] == ins_b.attrs[par_key]:
                    pass
                else:
                    print('    "%s" has a different value '
                          ' "%s" for a, "%s" for b' % (
                              par_key, ins_a.attrs[par_key],
                              ins_b.attrs[par_key]))
                    diffs_found = True

            if not diffs_found:
                print('    No differences found')
        except KeyError:
            print('Instrument "%s" not present in second settings file'
                  % ins_key)


def get_timestamps_in_range(timestamp_start, timestamp_end=None,
                            label=None, exact_label_match=False, folder=None):
    '''
    Input parameters:
        label: a string or list of strings to compare the experiment name to
        exact_label_match: 'True' : the label should exactly match the folder name
        (excluding "timestamp_"). 'False': the label must be a substring of the folder name


    '''
    if folder is None:
        folder = datadir

    datetime_start = datetime_from_timestamp(timestamp_start)
    if timestamp_end is None:
        datetime_end = datetime.datetime.today()
    else:
        datetime_end = datetime_from_timestamp(timestamp_end)
    days_delta = (datetime_end.date() - datetime_start.date()).days
    all_timestamps = []
    for day in reversed(list(range(days_delta + 1))):
        date = datetime_start + datetime.timedelta(days=day)
        datemark = timestamp_from_datetime(date)[:8]
        try:
            all_measdirs = [d for d in os.listdir(
                os.path.join(folder, datemark))]
        except FileNotFoundError:
            # Sometimes, when choosing multiples days, there is a day
            # with no measurements
            all_measdirs = []

        # Remove all hidden folders to prevent errors
        all_measdirs = [d for d in all_measdirs if not d.startswith('.')]

        if exact_label_match:
            if isinstance(label, str):
                label = [label]
            for each_label in label:
                # Remove 'hhmmss_' timestamp and check if exactly equals
                all_measdirs = [x for x in all_measdirs if each_label == x[7:]]
        else:
            if isinstance(label, str):
                label = [label]
            for each_label in label:
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
    if len(all_timestamps) == 0:
        raise ValueError('No matching timestamps found for label "{}"'.format(
            label))
    return all_timestamps


def get_mean_df(label, starting_timestamp, ending_timestamp,
                return_raw_dataframes=False):
    """
    Returns a dataframe containing the mean and standard error of mean (sem)
    of all datasets that match a certain label.
    Function assumes that the the datasets have identical sweep points.

    if return raw_dataframes
    """
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
        mean_df[val_name + '_mean'] = dataframes[i].mean(axis=1)
        mean_df[val_name + '_sem'] = dataframes[i].sem(axis=1)

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

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')

    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')

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

    tallest_peak = peaks[key]  # the ge freq
    tallest_peak_idx = peaks[key + '_idx']
    tallest_peak_width = peaks[key + '_width']
    if verbose:
        print('Largest ' + key + ' is at ', tallest_peak)

    # Calculate how many data points away from the tallest peak
    # to look left and right. Should be 50MHz away.
    freq_range = sweep_pts[-1] - sweep_pts[0]
    num_points = sweep_pts.size
    n = int(50e6 * num_points / freq_range)
    m = int(50e6 * num_points / freq_range)

    # Search for 2nd peak (f_ge) to the right of the first (tallest)
    while (int(len(sweep_pts) - 1) <= int(tallest_peak_idx + n) and
           n > 0):
        # Reduce n if outside of range
        n -= 1
    if (int(tallest_peak_idx + n)) == sweep_pts.size:
        # If n points to the right of tallest peak is the range edge:
        n = 0
    if not ((int(tallest_peak_idx + n)) >= sweep_pts.size):
        if verbose:
            print(
                'Searching for the gf/2 {:} {:} points to the right of the largest'
                ' in the range {:.5}-{:.5}'.format(
                    key,
                    n,
                    sweep_pts[int(tallest_peak_idx + n)],
                    sweep_pts[-1]))

        peaks_right = peak_finder(
            sweep_pts[int(tallest_peak_idx + n)::],
            data_dist_smooth[int(tallest_peak_idx + n)::],
            percentile=percentile,
            num_sigma_threshold=1,
            optimize=optimize,
            window_len=0,
            key=key)

        # The peak/dip found to the right of the tallest is assumed to be
        # the ge peak, which means that the tallest was in fact the gf/2 peak
        if verbose:
            print('Right ' + key + ' is at ', peaks_right[key])
        subset_right = data_dist_smooth[int(tallest_peak_idx + n)::]
        val_right = subset_right[peaks_right[key + '_idx']]
        f0_right = peaks_right[key]
        kappa_guess_right = peaks_right[key + '_width']
        f0_gf_over_2_right = tallest_peak
        kappa_guess_ef_right = tallest_peak_width
    else:
        if verbose:
            print('Right ' + key + ' is None')
        val_right = 0
        f0_right = 0
        kappa_guess_right = 0
        f0_gf_over_2_right = tallest_peak
        kappa_guess_ef_right = tallest_peak_width

    # Search for 2nd peak (f_gf/2) to the left of the first (tallest)
    while (int(tallest_peak_idx - m) < 0 and m > 0):
        # Reduce m if outside of range
        m -= 1
    if int(tallest_peak_idx - m) == 0:
        # If m points to the left of tallest peak is the range edge:
        m = 0
    if not (int(tallest_peak_idx - m) <= 0):
        if verbose:
            print('Searching for the gf/2 {:} {:} points to the left of the '
                  'largest, in the range {:.5}-{:.5}'.format(
                      key,
                      m,
                      sweep_pts[0],
                      sweep_pts[int(tallest_peak_idx - m - 1)]))

        peaks_left = peak_finder(
            sweep_pts[0:int(tallest_peak_idx - m)],
            data_dist_smooth[0:int(tallest_peak_idx - m)],
            percentile=percentile,
            num_sigma_threshold=1,
            optimize=optimize,
            window_len=0,
            key=key)

        # The peak/dip found to the left of the tallest is assumed to be
        # the gf/2 peak, which means that the tallest was indeed the ge peak
        if verbose:
            print('Left ' + key + ' is at ', peaks_left[key])
        subset_left = data_dist_smooth[0:int(tallest_peak_idx - m)]
        val_left = subset_left[peaks_left[key + '_idx']]
        f0_left = tallest_peak
        kappa_guess_left = tallest_peak_width
        f0_gf_over_2_left = peaks_left[key]
        kappa_guess_ef_left = peaks_left[key + '_width']
    else:
        if verbose:
            print('Left ' + key + ' is None')
        val_left = 0
        f0_left = tallest_peak
        kappa_guess_left = tallest_peak_width
        f0_gf_over_2_left = 0
        kappa_guess_ef_left = 0

    if np.abs(val_left) > np.abs(val_right):
        # If peak on the left taller than peak on the right, then
        # the second peak is to the left of the tallest and it is indeed
        # the gf/2 peak, while the tallest is the ge peak.
        if np.abs(f0_gf_over_2_left - tallest_peak) > 50e6:
            # If the two peaks found are separated by at least 50MHz,
            # then both the ge and gf/2 have been found.
            if verbose:
                print('Both f_ge and f_gf/2 ' + key + 's have been found. '
                                                      'f_ge was assumed to the LEFT of f_gf/2.')
        else:
            # If not, then it is just some other signal.
            logging.warning('The f_gf/2 ' + key + ' was not found. Fitting to '
                                                  'the next largest ' + key + ' found.')

        f0 = f0_left
        kappa_guess = kappa_guess_left
        f0_gf_over_2 = f0_gf_over_2_left
        kappa_guess_ef = kappa_guess_ef_left

    elif np.abs(val_left) < np.abs(val_right):
        # If peak on the right taller than peak on the left, then
        # the second peak is to the right of the tallest and it is in fact
        # the ge peak, while the tallest is the gf/2 peak.
        if np.abs(f0_right - tallest_peak) > 50e6:
            # If the two peaks found are separated by at least 50MHz,
            # then both the ge and gf/2 have been found.
            if verbose:
                print('Both f_ge and f_gf/2 have been found. '
                      'f_ge was assumed to the RIGHT of f_gf/2.')
        else:
            # If not, then it is just some other signal.
            logging.warning('The f_gf/2 ' + key + ' was not found. Fitting to '
                                                  'the next largest ' + key + ' found.')
        f0 = f0_right
        kappa_guess = kappa_guess_right
        f0_gf_over_2 = f0_gf_over_2_right
        kappa_guess_ef = kappa_guess_ef_right

    else:
        # If the peaks on the right and left are equal, or cannot be compared,
        # then there was probably no second peak, and only noise was found.
        logging.warning('Only f_ge has been found.')
        f0 = tallest_peak
        kappa_guess = tallest_peak_width
        f0_gf_over_2 = tallest_peak
        kappa_guess_ef = tallest_peak_width

    return f0, f0_gf_over_2, kappa_guess, kappa_guess_ef


def peak_finder_v2(x, y, perc=90, window_len=11):
    """
    Peak finder based on argrelextrema function from scipy
    only finds maximums, this can be changed to minimum by using -y instead of y
    """
    smoothed_y = smooth(y, window_len=window_len)
    percval = np.percentile(smoothed_y, perc)
    filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
    array_peaks = argrelextrema(filtered_y, np.greater)
    peaks_x = x[array_peaks]
    sort_mask = np.argsort(y[array_peaks])[::-1]
    return peaks_x[sort_mask]


def peak_finder_v3(x, y, smoothing=True, window_len=31, perc=99.6, factor=1):
    """
    Peak finder based on argrelextrema function from scipy
    only finds maximums, this can be changed to minimum by using factor=-1
    """
    if smoothing:
        y = (smooth(y, window_len=window_len) - y)
    else:
        y = y - np.average(y)
    y = factor * y
    percval = np.percentile(y, perc)
    filtered_y = np.where(y > percval, y, percval)
    array_peaks = argrelextrema(filtered_y, np.greater)
    peaks_x = x[array_peaks]
    peaks_y = y[array_peaks]
    sort_mask = np.argsort(y[array_peaks])[::-1]
    return peaks_x, peaks_y, y


def cut_edges(array, window_len=11):
    array = array[(window_len // 2):-(window_len // 2)]
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
    search_result = look_for_peaks_dips(x=x, y_smoothed=y,
                                        percentile=percentile,
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

        if y_for_test[search_result[key_1 + '_idx']] < max(y_for_test):
            while ((y_for_test[search_result[key_1 + '_idx']] < max(
                    y_for_test)) and
                   (num_sigma_threshold < 100) and
                   (len(search_result[key_1 + 's_idx']) > 1)):

                search_result_1 = deepcopy(search_result)
                num_sigma_threshold += 1

                search_result = look_for_peaks_dips(x=x, y_smoothed=y,
                                                    percentile=percentile,
                                                    num_sigma_threshold=(
                                                        num_sigma_threshold),
                                                    key=key_1,
                                                    window_len=window_len)

                if search_result[key_1 + '_idx'] is None:
                    search_result = search_result_1
                    break

            print('Peak Finder: Optimal num_sigma_threshold after optimization '
                  'is ', num_sigma_threshold)

    return search_result


def look_for_peaks_dips(x, y_smoothed, percentile=20, window_len=11,
                        num_sigma_threshold=5, key=None):
    """
    Peak finding algorithm designed by Serwan
    1 smooth data
    2 Define the threshold based on background data
    3

    key:    'peak' or 'dip'; return only the peaks or the dips
    """

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
                kk = i + 1
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
            except:
                pass

        # eliminate duplicates
        peak_indices = np.unique(peak_indices)

        # Take an approximate peak width for each peak index
        for i, idx in enumerate(peak_indices):
            if (idx + i + 1) < x.size and (idx - i - 1) >= 0:
                # ensure data points idx+i+1 and idx-i-1 are inside sweep pts
                peak_widths += [(x[idx + i + 1] - x[idx - i - 1]) / 5]
            elif (idx + i + 1) > x.size and (idx - i - 1) >= 0:
                peak_widths += [(x[idx] - x[idx - i]) / 5]
            elif (idx + i + 1) < x.size and (idx - i - 1) < 0:
                peak_widths += [(x[idx + i] - x[idx]) / 5]
            else:
                peak_widths += [5e6]

        peaks = np.take(x, peak_indices)  # Frequencies of peaks
        peak_vals = np.take(y_smoothed, peak_indices)  # values of peaks
        peak_index = peak_indices[np.argmax(peak_vals)]
        peak_width = peak_widths[np.argmax(peak_vals)]
        peak = x[peak_index]  # Frequency of highest peak

    else:
        peak = None
        peak_vals = None
        peak_index = None
        peaks = []
        peak_width = None

    # Finding dip_index
    # Definining the threshold
    percval = np.percentile(y_smoothed, 100 - percentile)
    y_background = y_smoothed[y_smoothed > percval]
    background_mean = np.mean(y_background)
    background_std = np.std(y_background)
    # Threshold is defined several sigma awy from the background
    threshold = background_mean - num_sigma_threshold * background_std

    thresholdlst = np.arange(y_smoothed.size)[y_smoothed < threshold]

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
                kk = i + 1
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
            except:
                pass

        # eliminate duplicates
        dip_indices = np.unique(dip_indices)

        # Take an approximate dip width for each dip index
        for i, idx in enumerate(dip_indices):
            if (idx + i + 1) < x.size and (
                    idx - i - 1) >= 0:  # ensure data points
                # idx+i+1 and idx-i-1
                # are inside sweep pts
                dip_widths += [(x[idx + i + 1] - x[idx - i - 1]) / 5]
            elif (idx + i + 1) > x.size and (idx - i - 1) >= 0:
                dip_widths += [(x[idx] - x[idx - i]) / 5]
            elif (idx + i + 1) < x.size and (idx - i - 1) < 0:
                dip_widths += [(x[idx + i] - x[idx]) / 5]
            else:
                dip_widths += [5e6]

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
        return {'peak': peak, 'peak_idx': peak_index, 'peak_values': peak_vals,
                'peak_width': peak_width, 'peak_widths': peak_widths,
                'peaks': peaks, 'peaks_idx': peak_indices,
                'dip': None, 'dip_idx': None, 'dip_values': [],
                'dip_width': None, 'dip_widths': [],
                'dips': [], 'dips_idx': []}
    elif key == 'dip':
        return {'peak': None, 'peak_idx': None, 'peak_values': [],
                'peak_width': None, 'peak_widths': [],
                'peaks': [], 'peaks_idx': [],
                'dip': dip, 'dip_idx': dip_index, 'dip_values': dip_vals,
                'dip_width': dip_width, 'dip_widths': dip_widths,
                'dips': dips, 'dips_idx': dip_indices}
    else:
        return {'peak': peak, 'peak_idx': peak_index, 'peak_values': peak_vals,
                'peak_width': peak_width, 'peak_widths': peak_widths,
                'peaks': peaks, 'peaks_idx': peak_indices,
                'dip': dip, 'dip_idx': dip_index, 'dip_values': dip_vals,
                'dip_width': dip_width, 'dip_widths': dip_widths,
                'dips': dips, 'dips_idx': dip_indices}


def calculate_distance_ground_state(data_real, data_imag, percentile=70,
                                    normalize=False):
    """ Calculates the distance from the ground state by assuming that
        for the largest part of the data, the system is in its ground state
    """
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


def zigzag(seq, sample_0, sample_1, nr_samples):
    """
    Splits a sequence in two sequences, one containing the odd entries, the
    other containing the even entries.
    e.g. in-> [0,1,2,3,4,5] -> out0 = [0,2,4] , out1[1,3,5]
    """
    return seq[sample_0::nr_samples], seq[sample_1::nr_samples]


def calculate_rotation_matrix(delta_I, delta_Q):
    """
    Calculates a matrix that rotates the data to lie along the Q-axis.
    Input can be either the I and Q coordinates of the zero cal_point or
    the difference between the 1 and 0 cal points.
    """

    angle = np.arctan2(delta_Q, delta_I)
    rotation_matrix = np.transpose(
        np.matrix([[np.cos(angle), -1 * np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]))
    return rotation_matrix


def normalize_TD_data(data, data_zero, data_one):
    """
    Normalizes measured data to refernce signals for zero and one
    """
    return (data - data_zero) / (data_one - data_zero)


def normalize_data(data):
    print(
        'a_tools.normalize_data is deprecated, recommend using a_tools.normalize_data_v2()')
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
    """
    Normalizes every row in a 2D array by normalizing on the mean
    of the elements specified in row_elements
    """
    for k in range(data_2D.shape[1]):
        data_2D[:, k] /= np.mean(data_2D[elements, k])
    return data_2D


def rotate_and_normalize_data(data, cal_zero_points=None, cal_one_points=None,
                              zero_coord=None, one_coord=None, **kw):
    """
    Rotates and normalizes data with respect to some reference coordinates.
    there are two ways to specify the reference coordinates.
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
    """
    # Extract zero and one coordinates

    if np.all([cal_zero_points == None, cal_one_points == None,
               zero_coord == None, one_coord == None]):
        # no cal points were used
        normalized_data = rotate_and_normalize_data_no_cal_points(data=data,
                                                                  **kw)
    elif np.all([cal_one_points == None, one_coord == None]) and \
            (not np.all([cal_zero_points == None, zero_coord == None])):
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
        # finx the x, y coordinates of the projected points
        x_proj = (x + line_slope * y - line_slope * line_intercept) / (
            line_slope ** 2 + 1)
        y_proj = line_slope * (x_proj) + line_intercept

        # find the minimum (on th ey axis) point on the line
        y_min_line = min(fit_res.best_fit)
        x_min_line = x[np.argmin(fit_res.best_fit)]

        # find x,y coordinates with respect to end of line
        x_data = np.abs(x_min_line - x_proj)
        y_data = y_proj - y_min_line

        # find distance from points on line to end of line
        rotated_data = np.sqrt(x_data ** 2 + y_data ** 2)

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
        M = calculate_rotation_matrix(I_one - I_zero, Q_one - Q_zero)
        outp = [np.asarray(elem)[0] for elem in M * trans_data]
        rotated_data_ch1 = outp[0]

        # Normalize the data
        one_zero_dist = np.sqrt((I_one - I_zero) ** 2 + (Q_one - Q_zero) ** 2)
        normalized_data = rotated_data_ch1 / one_zero_dist

    return [normalized_data, zero_coord, one_coord]


def rotate_and_normalize_data_no_cal_points(data, **kw):
    """
    Rotates and projects data based on principal component analysis.
    (Source: http://www.cs.otago.ac.nz/cosc453/student_tutorials/
    principal_components.pdf)
    Assumes data has shape (2, nr_sweep_pts), ie the shape of
    MeasurementAnalysis.measured_values.
    """

    # translate each column in the data by its mean
    mean_x = np.mean(data[0])
    mean_y = np.mean(data[1])
    trans_x = data[0] - mean_x
    trans_y = data[1] - mean_y

    # compute the covariance 2x2 matrix
    cov_matrix = np.cov(np.array([trans_x, trans_y]))

    # find eigenvalues and eigenvectors of the covariance matrix
    [eigvals, eigvecs] = np.linalg.eig(cov_matrix)

    # compute the transposed feature vector
    row_feature_vector = np.array([(eigvecs[0, np.argmin(eigvals)],
                                    eigvecs[1, np.argmin(eigvals)]),
                                   (eigvecs[0, np.argmax(eigvals)],
                                    eigvecs[1, np.argmax(eigvals)])])
    # compute the row_data_trans matrix with (x,y) pairs in each column. Each
    # row is a dimension (row1 = x_data, row2 = y_data)
    row_data_trans = np.array([trans_x, trans_y])
    # compute final, projected data; only the first row is of interest (it is the
    # principal axis
    final_data = np.dot(row_feature_vector, row_data_trans)
    normalized_data = final_data[1, :]

    # normalize data
    # max_min_distance = np.sqrt(max(final_data[1,:])**2 +
    #                            min(final_data[1,:])**2)
    # normalized_data = final_data[1,:]/max_min_distance
    # max_min_difference = max(normalized_data -  min(normalized_data))
    # normalized_data = (normalized_data-min(normalized_data))/max_min_difference

    return normalized_data


def normalize_data_v3(data, cal_zero_points=np.arange(-4, -2, 1),
                      cal_one_points=np.arange(-2, 0, 1), **kw):
    """
    Normalizes data according to calibration points
    Inputs:
        data (numpy array) : 1D dataset that has to be normalized
        cal_zero_points (range) : range specifying what indices in 'data'
                                  correspond to zero
        cal_one_points (range) : range specifying what indices in 'data'
                                 correspond to one
    """
    # Extract zero and one coordinates
    I_zero = np.mean(data[cal_zero_points])
    I_one = np.mean(data[cal_one_points])
    # Translate the date
    trans_data = data - I_zero
    # Normalize the data
    one_zero_dist = I_one - I_zero
    normalized_data = trans_data / one_zero_dist

    return normalized_data


def datetime_from_timestamp(timestamp: str):
    """
    Converst a timestamp instring in a datetime object.
    """
    try:
        if len(timestamp) == 14:
            return datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        elif len(timestamp) == 15:
            return datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        else:
            raise ValueError
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


def color_plot(x, y, z, fig=None, ax=None, cax=None,
               show=False, normalize=False, log=False,
               transpose=False, add_colorbar=True,
               xlabel='', ylabel='', zlabel='',
               x_unit='', y_unit='', z_unit='', **kw):
    """
    x, and y are lists, z is a matrix with shape (len(x), len(y))
    In the future this function can be overloaded to handle different
    types of input.
    Args:
        x (array [shape: n*1]):     x data
        y (array [shape: m*1]):     y data
        z (array [shape: n*m]):     z data
        fig (Object):
            figure object
        log (string/bool):
            True/False for z axis scaling, or any string containing any
            combination of letters x, y, z for scaling of the according axis.
            Remember to set the labels correctly.
    """
    if ax is None:
        fig, ax = plt.subplots()

    norm = kw.get('norm', None)
    if norm is None:
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
    x_vertices = np.zeros(np.array(x.shape) + 1)
    x_vertices[1:-1] = (x[:-1] + x[1:]) / 2.
    x_vertices[0] = x[0] - (x[1] - x[0]) / 2.
    x_vertices[-1] = x[-1] + (x[-1] - x[-2]) / 2.
    # y coordinates
    y_vertices = np.zeros(np.array(y.shape) + 1)
    y_vertices[1:-1] = (y[:-1] + y[1:]) / 2.
    y_vertices[0] = y[0] - (y[1] - y[0]) / 2.
    y_vertices[-1] = y[-1] + (y[-1] - y[-2]) / 2.
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

    title = kw.pop('title', None)

    xlabel = kw.get('xlabel', xlabel)
    ylabel = kw.get('ylabel', ylabel)
    zlabel = kw.get('zlabel', zlabel)
    x_unit = kw.get('x_unit', x_unit)
    y_unit = kw.get('y_unit', y_unit)
    z_unit = kw.get('z_unit', z_unit)
    cbarticks = kw.get('cbarticks', None)
    cbarextend = kw.get('cbarextend', 'neither')

    xlim = kw.pop('xlim', None)
    ylim = kw.pop('ylim', None)

    if title is not None:
        ax.set_title(title)

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
            cax = ax_divider.append_axes('right', size='5%', pad='2%')
        cbar = plt.colorbar(colormap, cax=cax, orientation='vertical',
            ticks=cbarticks, extend=cbarextend)
        if zlabel is not None:
            set_cbarlabel(cbar, zlabel, unit=z_unit)
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
    xvertices = np.zeros(np.array(xvals.shape) + 1)
    xvertices[1:-1] = (xvals[:-1] + xvals[1:]) / 2.
    xvertices[0] = xvals[0] - (xvals[1] - xvals[0]) / 2
    xvertices[-1] = xvals[-1] + (xvals[-1] - xvals[-2]) / 2
    # y coordinates
    yvertices = []
    for xx in range(len(xvals)):
        yvertices.append(np.zeros(np.array(yvals[xx].shape) + 1))
        yvertices[xx][1:-1] = (yvals[xx][:-1] + yvals[xx][1:]) / 2.
        yvertices[xx][0] = yvals[xx][0] - (yvals[xx][1] - yvals[xx][0]) / 2
        yvertices[xx][-1] = yvals[xx][-1] + (yvals[xx][-1] - yvals[xx][-2]) / 2

    # various plot options
    # define colormap
    cmap = kw.pop('cmap', 'viridis')
    # clim = kw.pop('clim', [None, None])
    # normalized plot
    if normalize:
        for xx in range(len(xvals)):
            zvals[xx] /= np.mean(zvals[xx])
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx]) / np.log(10)

    # add blocks to plot
    # hold = kw.pop('hold', False)
    for xx in range(len(xvals)):
        tempzvals = np.array([np.append(zvals[xx], np.array(0)),
                              np.append(zvals[xx], np.array(0))]).transpose()
        # im = ax.pcolor(xvertices[xx:xx+2],
        #                yvertices[xx],
        #                tempzvals, cmap=cmap)
    return ax


def linecut_plot(x, y, z, fig, ax,
                 xlabel=None, x_unit='',
                 y_name='', y_unit='', log=True,
                 zlabel=None, z_unit_linecuts='', legend=True,
                 line_offset=0, **kw):
    """
    Plots horizontal linecuts of a 2D plot.
    x and y must be 1D arrays.
    z must be a 2D array with shape(len(x),len(y)).
    """
    z_unit_linecuts = kw.pop("z_unit_linecuts", z_unit_linecuts)

    colormap = plt.cm.get_cmap('RdYlBu')
    ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(
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
    set_xlabel(ax, xlabel, x_unit)
    set_ylabel(ax, zlabel, z_unit_linecuts)
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


def plot_errorbars(x, y, ax=None, linewidth=2, markersize=2, marker='none'):
    if ax is None:
        new_plot_created = True
        f, ax = plt.subplots()
    else:
        new_plot_created = False

    standard_error = np.std(y) / np.sqrt(y.size)

    ax.errorbar(x, y, yerr=standard_error, ecolor='k', fmt=marker,
                linewidth=linewidth, markersize=markersize)

    if new_plot_created:
        return f, ax
    else:
        return


######################################################################
#    Calculations tools
######################################################################


def calculate_transmon_transitions(EC, EJ, asym=0, reduced_flux=0,
                                   no_transitions=2, dim=None, ng=0,
                                   return_injs=False):
    """
    Calculates transmon energy levels from the full transmon qubit Hamiltonian.
    """
    if dim is None:
        dim = no_transitions * 10

    EJphi = EJ * np.sqrt(
        asym ** 2 + (1 - asym ** 2) * np.cos(np.pi * reduced_flux) ** 2)
    Ham = 4 * EC * np.diag(
        np.arange(-dim - ng, dim - ng + 1) ** 2) - EJphi / 2 * \
        (np.eye(2 * dim + 1, k=+1) + np.eye(2 * dim + 1, k=-1))

    if return_injs:
        HamEigs, HamEigVs = np.linalg.eigh(Ham)
        # HamEigs.sort()
        transitions = HamEigs[1:] - HamEigs[:-1]
        charge_number_operator = np.diag(np.arange(-dim - ng, dim - ng + 1))
        injs = np.zeros([dim, dim])
        for i in range(dim):
            for j in range(dim):
                vect_i = np.matrix(HamEigVs[:, i])
                vect_j = np.matrix(HamEigVs[:, j])
                injs[i, j] = vect_i * (charge_number_operator * vect_j.getH())
        return transitions[:no_transitions], injs

    else:
        HamEigs = np.linalg.eigvalsh(Ham)
        HamEigs.sort()
        transitions = HamEigs[1:] - HamEigs[:-1]
        return transitions[:no_transitions]


def calculate_transmon_and_resonator_transitions_old(EC, EJ, f_r, g_01,
                                                     dim=None, ng=0, f_01=None,
                                                     f_12=None,
                                                     g_12_approximation=1):
    """
    Calculates transmon energy levels and resonator from the full transmon qubit Hamiltonian.
    """

    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01, f_12], injs = calculate_transmon_transitions(EC, EJ, asym=0,
                                                        reduced_flux=0,
                                                        no_transitions=2,
                                                        dim=dim, ng=ng,
                                                        return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    try:
        E00 = 0
        g_01 = np.abs(g_01)
        H1 = np.array([[f_01, g_01],
                       [g_01, f_r]])
        E10, E01 = np.linalg.eigvalsh(H1)
        g_12_transmon = abs(g_01 * injs[1, 2] / injs[0, 1])
        g_12_resonator = abs(g_01 * np.sqrt(2))
        # print('g_12_correction',g_12_transmon/g_12_resonator)
        H2 = np.array([[f_01 + f_12, g_12_transmon, 0],
                       [g_12_transmon, f_01 + f_r, g_12_resonator],
                       [0, g_12_resonator, 2 * f_r]])
        E20, E11, E02 = np.linalg.eigvalsh(H2)
        # print(EC/1e6, EJ/1e9, f_r/1e9, g_01/1e6, f_01/1e9, f_12/1e9)
    except np.linalg.LinAlgError:
        print(EC / 1e6, EJ / 1e9, f_r / 1e9,
              g_01 / 1e6, f_01 / 1e9, f_12 / 1e9)

    f_01_d = E10 - E00
    f_12_d = E20 - E10
    f_r_d = E01 - E00
    f_01_res_shifted = E11 - E01
    f_r_qubit_shifted = E11 - E10

    return f_01_d, f_12_d, f_r_d, f_01_res_shifted, f_r_qubit_shifted


def calculate_transmon_and_resonator_transitions(Ec, Ej, f_bus, gs, ng=0):
    """
    Calculate dressed qubit and bus frequencies based on several physical parameters
    Input:
    - Ec: transmon charging energy
    - Ej: transmon josephson energy
    - f_bus: the bare bus frequency (float or list of floats)
    - g: qubit-bus coupling (float or list of floats)
    Output:
    - f01_dressed: 01 transition of a dressed qubit
    - f12_dressed: 12 transition of a dressed qubit
    - f_bus_transitions: dressed bus transition with qubit in a ground state
    - photon_splittings: a qubit photon splitting (difference between f01_dressed with
        and without a photon in this resonator).
    - f_bus_shifted_transitions: dressed bus transition with qubit in an excited state

    Replacement for the previous calculate_transmon_and_resonator_transitions function,
    which was assuming only one bus. That one is temporarily kept as
    'calculate_transmon_and_resonator_transitions_old'. It is verified to produce
    the same result to 6 decimal places.
    """
    if isinstance(f_bus, float) or isinstance(f_bus, int):
        f_bus = [f_bus]
    if isinstance(gs, float) or isinstance(gs, int):
        gs = [gs]

    # max photons
    max_ph = 2

    # number of buses
    n_bus = len(f_bus)

    # qubit operators
    (f01, f12), injs = calculate_transmon_transitions(Ec, Ej, asym=0,
                                                      reduced_flux=0,
                                                      no_transitions=2,
                                                      ng=ng, dim=10,
                                                      return_injs=True)
    H_q = qp.Qobj(np.diag((0, f01, f01+f12)))
    H_q = qp.tensor(H_q, *[qp.qeye(max_ph+1)]*n_bus)
    a_q = qp.Qobj(np.diag((1, np.abs(injs[2, 1]/injs[1, 0])), k=1))
    a_q = qp.tensor(a_q, *[qp.qeye(max_ph+1)]*n_bus)

    # bus operators
    n_r_list = []
    a_r_list = []

    a_r_generating_list = [qp.destroy(
        max_ph+1)] + [qp.qeye(max_ph+1)]*(n_bus-1)
    for fb, g in zip(f_bus, gs):
        a_r = qp.tensor(qp.qeye(3), *a_r_generating_list)
        a_r_list.append(a_r)
        n_r_list.append(a_r.dag() * a_r)
        a_r_generating_list = a_r_generating_list[-1:]+a_r_generating_list[:-1]

    # full hamiltonian
    H = deepcopy(H_q)
    for fb, g, a_r, n_r in zip(f_bus, gs, a_r_list, n_r_list):
        H += fb*n_r
        H += g*(a_q*a_r.dag() + a_q.dag()*a_r)

    # eigenenergies and eigenstates
    ees, ess = H.eigenstates()

    # define bare qubit states
    qubit_g = qp.tensor(qp.fock(3, 0), *[qp.fock(max_ph+1, 0)]*n_bus)
    qubit_e = qp.tensor(qp.fock(3, 1), *[qp.fock(max_ph+1, 0)]*n_bus)
    qubit_f = qp.tensor(qp.fock(3, 2), *[qp.fock(max_ph+1, 0)]*n_bus)

    # define bare bus states
    bus_e_list = []
    bus_e_qubit_e_list = []
    bus_e_generating_list = [
        qp.fock(max_ph+1, 1)] + [qp.fock(max_ph+1, 0)]*(n_bus-1)
    for fb, g in zip(f_bus, gs):
        bus_e = qp.tensor(qp.fock(3, 0), *bus_e_generating_list)
        bus_e_list.append(bus_e)
        bus_e_qubit_e = qp.tensor(qp.fock(3, 1), *bus_e_generating_list)
        bus_e_qubit_e_list.append(bus_e_qubit_e)
        bus_e_generating_list = bus_e_generating_list[-1:] + \
            bus_e_generating_list[:-1]

    # find states with the largest overlap to specific state
    def largest_overlap_index(state, ess, ees):
        max_fid = 0
        best_index = 0
        for i, es in enumerate(ess):
            fid = qpmetrics.fidelity(state, es)**2
            if fid > max_fid:
                max_fid = fid
                best_index = i
            if max_fid > 0.5:
                break
        return i, max_fid, ees[i]

    # find states with the largest overlap to specific state
    def largest_overlap_energy(state, ess, ees):
        return largest_overlap_index(state, ess, ees)[2]

    # ground state energy
    E_g = largest_overlap_energy(qubit_g, ess, ees)
    # excited state energy
    E_e = largest_overlap_energy(qubit_e, ess, ees)
    # second_excited state energy
    E_f = largest_overlap_energy(qubit_f, ess, ees)

    # transitions of dressed states
    f01_dressed = E_e - E_g
    f12_dressed = E_f - E_e

    # bus transition with qubit in ground state
    f_bus_transitions = []
    # bus transition with qubit in excited state
    f_bus_shifted_transitions = []
    # qubit 01 transition with a photon in a bus
    f01_dressed_shifted = []

    # dressed bus transitions
    for bus_qubit_g, bus_qubit_e in zip(bus_e_list, bus_e_qubit_e_list):
        E_b_g = largest_overlap_energy(bus_qubit_g, ess, ees)
        E_b_e = largest_overlap_energy(bus_qubit_e, ess, ees)

        f_bus_transitions.append(E_b_g-E_g)
        f_bus_shifted_transitions.append(E_b_e-E_e)
        f01_dressed_shifted.append(E_b_e - E_b_g)

    if n_bus == 1:
        f_bus_transitions = f_bus_transitions[0]
        f_bus_shifted_transitions = f_bus_shifted_transitions[0]
        f01_dressed_shifted = f01_dressed_shifted[0]

    return f01_dressed, f12_dressed, f_bus_transitions, f01_dressed_shifted, f_bus_shifted_transitions


def fit_Ec_Ej_fbus_g(f01, f12, fbus, f01_shifted):
    '''
    Use the qubit anharmonicity and photon splitting to calculate
    Ec, Ej, resonator frequency, and coupling to resonator

    Input:
    - f01: qubit frequency
    - f12: e-f transition frequency
    - fbus: resonator frequency (bus of RO resonator; dressed i.e. as measured)
    - f01_shifted: qubit frequency in presence of a single photon in the resonator

    Output:
    tuple containing:
    - Ec: charging energy
    - Ej: josephson energy
    - fbus: bare resonator frequency (bus of RO resonator)
    - g: coupling between qubit and resonator
    '''

    # pack measured values into an array
    measured = np.array([f01, f12, fbus, f01_shifted])

    # guess initial parameters

    # Ec is smaller that f01-f12 because of a bus so let's use
    # f01_shifted-f12 as a guess
    Ec_guess = f01_shifted - f12
    # and let's use large Ej/Ec limit for Ej guess
    Ej_guess = (f01 + Ec_guess) ** 2 / (8 * Ec_guess)
    # detuning is roughly (fbus-f01) so I use it to guess g and fbus
    # in a dispersive regime
    g_guess = np.sqrt(np.abs(fbus-f01) * np.abs(f01-f01_shifted) * 2)
    fbus_guess = fbus+g_guess**2/(fbus-f01)

    # pack all quesses into an array
    guesses = np.array([Ec_guess, Ej_guess, fbus_guess, g_guess])

    # define a penalty function to minimize
    def penalty_function(params):
        # calculate frequencies based on parameters
        calc = np.array(calculate_transmon_and_resonator_transitions(*params))
        # remove last element which is the bus frequency for qubit in 1 state
        calc = calc[:-1]

        # calculate difference between calculated and measured frequencies
        errors = calc - measured
        return errors**2

    # optimize
    out = optimize.leastsq(penalty_function, guesses, full_output=1)

#     print('Measured frequencies: '+str(measured))
#     print('Frequencies after fitting: '+str(np.around(calculate_transmon_and_resonator_transitions(*out[0]), decimals=4)[:-1]))
    return tuple(out[0])


def calculate_transmon_RR_PF_transitions(EC, EJ, f_r, f_PF, g_1, J_1,
                                         dim=None, ng=0, f_01=None, f_12=None,
                                         g_12_approximation=1):
    """
    Calculates transmon energy levels and resonator from the full transmon qubit Hamiltonian.
    """

    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01, f_12], injs = calculate_transmon_transitions(EC, EJ, asym=0,
                                                        reduced_flux=0,
                                                        no_transitions=2,
                                                        dim=dim, ng=ng,
                                                        return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    try:
        E000 = 0
        g_1 = np.abs(g_1)
        J_1 = np.abs(J_1)
        H1 = np.array([[f_01, g_1, 0],
                       [g_1, f_r, J_1],
                       [0, J_1, f_PF]])
        E100, E010, E001 = np.linalg.eigvalsh(H1)
        g_2_trm = abs(g_1 * injs[1, 2] / injs[0, 1])
        g_2_r_trm = abs(g_1 * np.sqrt(2))
        g_2_r_PF = abs(J_1 * np.sqrt(2))
        # print('g_12_correction',g_2_trm/g_2_r_trm)
        H2 = np.zeros([6, 6])
        H2[0, 0] = f_01 + f_12
        H2[0, 1] = H2[1, 0] = g_2_trm

        H2[1, 1] = f_01 + f_r
        H2[1, 2] = H2[2, 1] = J_1
        H2[1, 3] = H2[3, 1] = g_2_r_trm

        H2[2, 2] = f_01 + f_PF
        H2[2, 4] = H2[4, 2] = g_1

        H2[3, 3] = 2 * f_r
        H2[3, 4] = H2[4, 3] = g_2_r_PF

        H2[4, 4] = f_r + f_PF
        H2[4, 5] = H2[5, 4] = g_2_r_PF

        H2[5, 5] = 2 * f_PF

        E200, E110, E101, E020, E011, E002 = np.linalg.eigvalsh(H2)
    except np.linalg.LinAlgError:
        print(EC / 1e6, EJ / 1e9, f_r / 1e9, g_1 / 1e6, f_01 / 1e9, f_12 / 1e9)

    f_q_01 = E100
    f_q_12 = E200 - E100

    f_r1 = E010
    f_r2 = E001

    f_disp_r1 = E110 - E100
    f_disp_r2 = E101 - E100

    f_nrsplt_r1 = E110 - E010
    f_nrsplt_r2 = E101 - E001

    return f_q_01, f_r1, f_r2, f_q_12, f_disp_r1, f_disp_r2, f_nrsplt_r1, f_nrsplt_r2


def calculate_transmon_RR_PF_bus_transitions(EC, EJ, f_r, f_PF, f_bus, g_trm_RR, g_RR_PF, g_trm_bus,
                                             dim=None, ng=0):
    """
    Calculates transmon and resonator energy levels and resonator from the full Hamiltonian.
    """

    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01, f_12], injs = calculate_transmon_transitions(EC, EJ, asym=0,
                                                        reduced_flux=0,
                                                        no_transitions=2,
                                                        dim=dim, ng=ng,
                                                        return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    g_trm_RR = np.abs(g_trm_RR)
    g_RR_PF = np.abs(g_RR_PF)
    H1 = np.array([[f_01, g_trm_RR, 0, g_trm_bus],
                   [g_trm_RR, f_r, g_RR_PF, 0],
                   [0, g_RR_PF, f_PF, 0],
                   [g_trm_bus, 0, 0, f_bus]])

    E1000, E0100, E0010, E0001 = np.linalg.eigvalsh(H1)
    g_2_trm_r = abs(g_trm_RR * injs[1, 2] / injs[0, 1])
    g_2_r_trm = abs(g_trm_RR * np.sqrt(2))
    g_2_r_PF = abs(g_RR_PF * np.sqrt(2))
    g_2_bus_trm = abs(g_trm_bus * np.sqrt(2))
    g_2_trm_bus = abs(g_trm_bus * injs[1, 2] / injs[0, 1])

    H2 = np.zeros([10, 10])
    H2[0, 0] = f_01 + f_12
    H2[0, 1] = H2[1, 0] = g_2_trm_r
    H2[0, 3] = H2[3, 0] = g_2_trm_bus

    H2[1, 1] = f_01 + f_r
    H2[1, 2] = H2[2, 1] = g_RR_PF
    H2[1, 4] = H2[4, 1] = g_2_r_trm
    H2[1, 6] = H2[6, 1] = g_trm_bus

    H2[2, 2] = f_01 + f_PF
    H2[2, 5] = H2[5, 2] = g_trm_RR
    H2[2, 8] = H2[8, 2] = g_trm_bus

    H2[3, 3] = f_bus + f_01
    H2[3, 6] = H2[6, 3] = g_trm_RR
    H2[3, 9] = H2[9, 3] = g_2_bus_trm

    H2[4, 4] = 2 * f_r
    H2[4, 5] = H2[5, 4] = g_2_r_PF

    H2[5, 5] = f_r + f_PF
    H2[5, 7] = H2[7, 5] = g_2_r_PF

    H2[6, 6] = f_r + f_bus
    H2[6, 8] = H2[8, 6] = g_RR_PF

    H2[7, 7] = 2 * f_PF

    H2[8, 8] = f_PF + f_bus

    H2[9, 9] = f_bus + f_bus

    E2000, E1100, E1010, E1001, E0200, E0110, E0101, E0020, E0011, E0002 = \
        np.linalg.eigvalsh(H2)

    f_q_01 = E1000
    f_q_12 = E2000 - E1000

    f_r1 = E0100
    f_r2 = E0010
    f_r3 = E0001

    f_disp_r1 = E1100 - E1000
    f_disp_r2 = E1010 - E1000
    f_disp_r3 = E1001 - E1000

    f_nrsplt_r1 = E1100 - E0100
    f_nrsplt_r2 = E1010 - E0010
    f_nrsplt_r3 = E1001 - E0010

    return f_q_01, f_r1, f_r2, f_q_12, f_disp_r1, f_disp_r2, f_nrsplt_r1, f_nrsplt_r2


def calculate_tr_bus_tr_bus_tr_transitions(EC1, EC2, EC3, EJ1, EJ2, EJ3, f_bus1, f_bus2, g1, g2, g3, g4,
                                           dim=None, ng=0):
    '''
    Calculates transmon energy levels and resonator from the full transmon qubit Hamiltonian.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)
    [f_01_2, f_12_2], injs = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)
    [f_01_3, f_12_3], injs = calculate_transmon_transitions(EC3, EJ3, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    g1 = np.abs(g1)
    g2 = np.abs(g2)
    g3 = np.abs(g3)
    g4 = np.abs(g4)
    H1 = np.array([[f_01_1,     0,      0,     g1,  0],
                   [0,     f_01_2,      0,     g2,  g3],
                   [0,          0, f_01_3,      0,  g4],
                   [g1,        g2,      0, f_bus1,  0],
                   [0,         g3,     g4,      0,  f_bus2]])

    E10000, E01000, E00100, E00010, E00001 = np.linalg.eigvalsh(H1)
    return E10000, E01000, E00100, E00010, E00001


def calculate_tr_bus_square(EC1, EC2, EC3, EC4, EJ1, EJ2, EJ3, EJ4, f_bus1_2,
                            f_bus2_3, f_bus3_4, f_bus4_1, g1_2, g2_3, g3_4, g4_1,
                            dim=None, ng=0):
    '''
    Calculates transmon energy levels and resonator from the full transmon qubit Hamiltonian.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)
    [f_01_2, f_12_2], injs = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)
    [f_01_3, f_12_3], injs = calculate_transmon_transitions(EC3, EJ3, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)
    [f_01_4, f_12_4], injs = calculate_transmon_transitions(EC4, EJ4, asym=0, reduced_flux=0,
                                                            no_transitions=2, dim=dim, ng=ng,
                                                            return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    H1 = np.array([[f_01_1,     0,     0,     0,    g1_2,       0,       0,    g4_1],
                   [0, f_01_2,     0,     0,    g1_2,    g2_3,       0,       0],
                   [0,     0, f_01_3,     0,       0,    g2_3,    g3_4,       0],
                   [0,     0,     0, f_01_4,       0,       0,    g3_4,   g4_1],
                   [g1_2,  g1_2,     0,     0, f_bus1_2,       0,       0,       0],
                   [0,  g2_3,  g2_3,     0,       0, f_bus2_3,       0,       0],
                   [0,     0,  g3_4,  g3_4,       0,       0, f_bus3_4,       0],
                   [g4_1,     0,     0,  g4_1,       0,       0,       0, f_bus4_1]])

    E10000000, E01000000, E00100000, E00010000, E00001000, E00000100, E00000010, E0000001 = np.linalg.eigvalsh(
        H1)
    return E10000000, E01000000, E00100000, E00010000, E00001000, E00000100, E00000010, E0000001


def calculate_tr_tr_tr_transitions(EC1, EC2, EC3, EJ1, EJ2, EJ3, g1_2, g2_3,
                                   dim=None, ng=0):
    '''
    Calculates transmon energy levels for three coupled transmons in the 1-excitation manifolc.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs1 = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_2, f_12_2], injs2 = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_3, f_12_3], injs3 = calculate_transmon_transitions(EC3, EJ3, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    g1_2 = np.abs(g1_2)
    g2_3 = np.abs(g2_3)
    H1 = np.array([[f_01_1,     g1_2,      0],
                   [g1_2,     f_01_2,      g2_3],
                   [0,          g2_3, f_01_3]])

    E1, E2, E3 = np.linalg.eigvalsh(H1)

    g1_2_2first = g1_2*injs1[1, 2]/injs1[0, 1]
    g1_2_2sec = g1_2*injs2[1, 2]/injs2[0, 1]
    g2_3_2first = g2_3*injs2[1, 2]/injs2[0, 1]
    g2_3_2sec = g2_3*injs3[1, 2]/injs3[0, 1]

    H2 = np.array([[f_01_1+f_12_1, g1_2_2first, 0, 0, 0, 0],
                   [g1_2_2first, f_01_1+f_01_2, g1_2_2sec, g2_3, 0, 0],
                   [0, g1_2_2sec, f_01_2+f_12_2, 0, g2_3_2first, 0],
                   [0, g2_3, 0, f_01_1+f_01_3, g1_2, 0],
                   [0, 0, g2_3_2first, g1_2, f_01_2+f_01_3, g2_3_2sec],
                   [0, 0, 0, 0, g2_3_2sec, f_01_3+f_12_3]])
    E4, E5, E6, E7, E8, E9 = np.linalg.eigvalsh(H2)

    return E1, E2, E3, E4, E5, E6, E7, E8, E9


def calculate_tr_tr_tr_bus_transitions(EC1, EC2, EC3, EJ1, EJ2, EJ3, fbus, g1_2, g2_3, g1_bus, g3_bus,
                                       dim=None, ng=0):
    '''
    Calculates transmon energy levels for three coupled transmons in the 1-excitation manifolc.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs1 = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_2, f_12_2], injs2 = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_3, f_12_3], injs3 = calculate_transmon_transitions(EC3, EJ3, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    g1_2 = np.abs(g1_2)
    g2_3 = np.abs(g2_3)
    H1 = np.array([[f_01_1,     g1_2,     0, g1_bus],
                   [g1_2,     f_01_2,  g2_3,     0],
                   [0,          g2_3, f_01_3, g3_bus],
                   [g1_bus,        0, g3_bus,  fbus]])

    E1, E2, E3, E4 = np.linalg.eigvalsh(H1)

    g1_2_2first = g1_2*injs1[1, 2]/injs1[0, 1]
    g1_2_2sec = g1_2*injs2[1, 2]/injs2[0, 1]
    g2_3_2first = g2_3*injs2[1, 2]/injs2[0, 1]
    g2_3_2sec = g2_3*injs3[1, 2]/injs3[0, 1]

    g1_bus_2first = g1_bus*injs1[1, 2]/injs1[0, 1]
    g1_bus_2sec = g1_bus*np.sqrt(2)
    g3_bus_2first = g3_bus*injs3[1, 2]/injs3[0, 1]
    g3_bus_2sec = g3_bus*np.sqrt(2)

    H2 = np.array([[f_01_1+f_12_1, g1_2_2first, 0, 0, 0, 0, g1_bus_2first, 0, 0, 0],
                   [g1_2_2first, f_01_1+f_01_2, g1_2_2sec,
                       g2_3, 0, 0, 0, g1_bus, 0, 0],
                   [0, g1_2_2sec, f_01_2+f_12_2, 0, g2_3_2first, 0, 0, 0, 0, 0],
                   [0, g2_3, 0, f_01_1+f_01_3, g1_2, 0, g3_bus, 0, g1_bus, 0],
                   [0, 0, g2_3_2first, g1_2, f_01_2 +
                       f_01_3, g2_3_2sec, 0, g3_bus, 0, 0],
                   [0, 0, 0, 0, g2_3_2sec, f_01_3+f_12_3, 0, 0, g3_bus_2first, 0],
                   [g1_bus_2first, 0, 0, g3_bus, 0, 0,
                       fbus+f_01_1, g1_2, 0, g1_bus_2sec],
                   [0, g1_bus, 0, 0, g3_bus, 0, g1_2, fbus+f_01_2, g2_3, 0],
                   [0, 0, 0, g1_bus, 0, g3_bus_2first, 0,
                       g2_3, fbus+f_01_3, g3_bus_2sec],
                   [0, 0, 0, 0, 0, 0, g1_bus_2sec, 0, g3_bus_2sec, fbus*2]])
    E5, E6, E7, E8, E9, E10, E11, E12, E13, E14 = np.linalg.eigvalsh(H2)

    return E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14


def calculate_tr_bus_tr_bus_transitions(EC1, EC3, EJ1, EJ3, fbus2, fbus4, g1_2, g2_3, g3_4, g4_1,
                                        dim=None, ng=0):
    '''
    Calculates transmon energy levels for three coupled transmons in the 1-excitation manifolc.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs1 = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_3, f_12_3], injs3 = calculate_transmon_transitions(EC3, EJ3, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    H1 = np.array([[f_01_1, g1_2, 0, g4_1],
                   [g1_2, fbus2, g2_3, 0],
                   [0, g2_3, f_01_3, g3_4],
                   [g4_1, 0, g3_4, fbus4]])

    E1, E2, E3, E4 = np.linalg.eigvalsh(H1)

    g1_2_2first = g1_2*injs1[1, 2]/injs1[0, 1]
    g1_2_2sec = g1_2*np.sqrt(2)

    g2_3_2first = g2_3*np.sqrt(2)
    g2_3_2sec = g2_3*injs3[1, 2]/injs3[0, 1]

    g3_4_2first = g3_4*injs3[1, 2]/injs3[0, 1]
    g3_4_2sec = g3_4*np.sqrt(2)

    g4_1_2first = g4_1*np.sqrt(2)
    g4_1_2sec = g4_1*injs1[1, 2]/injs1[0, 1]

    f_01_2 = fbus2
    f_12_2 = fbus2
    f_01_4 = fbus4
    f_12_4 = fbus4

    H2 = np.array([[f_01_1+f_12_1, g1_2_2first, 0, 0, 0, 0, g4_1_2first, 0, 0, 0],
                   [g1_2_2first, f_01_1+f_01_2, g1_2_2sec,
                       g2_3, 0, 0, 0, g4_1, 0, 0],
                   [0, g1_2_2sec, f_01_2+f_12_2, 0, g2_3_2first, 0, 0, 0, 0, 0],
                   [0, g2_3, 0, f_01_1+f_01_3, g1_2, 0, g3_4, 0, g4_1, 0],
                   [0, 0, g2_3_2first, g1_2, f_01_2 +
                       f_01_3, g2_3_2sec, 0, g3_4, 0, 0],
                   [0, 0, 0, 0, g2_3_2sec, f_01_3+f_12_3, 0, 0, g3_4_2first, 0],
                   [g4_1_2first, 0, 0, g3_4, 0, 0,
                       f_01_4+f_01_1, g1_2, 0, g4_1_2sec],
                   [0, g4_1, 0, 0, g3_4, 0, g1_2, f_01_4+f_01_2, g2_3, 0],
                   [0, 0, 0, g4_1, 0, g3_4_2first, 0,
                       g2_3, f_01_4+f_01_3, g3_4_2sec],
                   [0, 0, 0, 0, 0, 0, g4_1_2sec, 0, g3_4_2sec, f_01_4+f_01_2]])
    E5, E6, E7, E8, E9, E10, E11, E12, E13, E14 = np.linalg.eigvalsh(H2)

    return E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14


def calculate_tr_bus_tr(EC1, EC2, EJ1, EJ2, f_bus, g1, g2, dim=None, ng=0):
    '''
    Calculates energy levels for a transmon-resonator-transmon system from the full transmon qubit Hamiltonian.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs1 = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_2, f_12_2], injs2 = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    g1 = np.abs(g1)
    g2 = np.abs(g2)
    H1 = np.array([[f_01_1,     0,      g1],
                   [0,     f_01_2,      g2],
                   [g1,        g2,      f_bus]])

    E100, E010, E001 = np.linalg.eigvalsh(H1)
    g2_trm_r1 = abs(g1*injs1[1, 2]/injs1[0, 1])
    g2_r_trm1 = abs(g1*np.sqrt(2))
    g2_trm_r2 = abs(g2*injs2[1, 2]/injs2[0, 1])
    g2_r_trm2 = abs(g2*np.sqrt(2))
    H2 = np.zeros([6, 6])
    H2[0, 0] = f_01_1+f_12_1
    H2[0, 3] = H2[3, 0] = g2_trm_r1
    H2[1, 1] = f_01_1+f_01_2
    H2[1, 3] = H2[3, 1] = g2
    H2[1, 4] = H2[4, 1] = g1
    H2[2, 2] = f_01_2+f_12_2
    H2[2, 4] = H2[4, 2] = g2_trm_r2
    H2[3, 3] = f_01_1+f_bus
    H2[3, 5] = H2[5, 3] = g2_r_trm1
    H2[4, 4] = f_01_2+f_bus
    H2[4, 5] = H2[5, 4] = g2_r_trm2
    H2[5, 5] = f_bus+f_bus
    E200, E110, E020, E101, E011, E002 = np.linalg.eigvalsh(H2)
    ZZ1 = E110 - E010
    ZZ2 = E110 - E100
    return E100, E010, E001, E200, E110, E020, E101, E011, E002


def calculate_tr_tr(EC1, EC2, EJ1, EJ2, g1, dim=None, ng=0):
    '''
    Calculates energy levels for two directly coupled transmons from the full transmon qubit Hamiltonian.
    '''
    # calculate the bare transmon transitions, hardcoded to three levels only
    [f_01_1, f_12_1], injs1 = calculate_transmon_transitions(EC1, EJ1, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)
    [f_01_2, f_12_2], injs2 = calculate_transmon_transitions(EC2, EJ2, asym=0, reduced_flux=0,
                                                             no_transitions=2, dim=dim, ng=ng,
                                                             return_injs=True)

    # problem can be cut up in th 0, 1 and 2-excitation manifold with E_ij, i excitations in the qubit and j of the resonator
    # try:
    #E0000 = 0
    g1 = np.abs(g1)
    H1 = np.array([[f_01_1,     g1],
                   [g1,        f_01_2]])

    E10, E01 = np.linalg.eigvalsh(H1)
    g2_trm_r1 = abs(g1*injs1[1, 2]/injs1[0, 1])
    g2_trm_r2 = abs(g1*injs2[1, 2]/injs2[0, 1])
    H2 = np.zeros([3, 3])
    H2[0, 0] = f_01_1+f_12_1
    H2[0, 1] = H2[1, 0] = g2_trm_r1
    H2[1, 1] = f_01_1+f_01_2
    H2[1, 2] = H2[2, 1] = g2_trm_r2
    H2[2, 2] = f_01_2+f_12_2
    E20, E11, E02 = np.linalg.eigvalsh(H2)
    ZZ1 = E11 - E01
    ZZ2 = E11 - E10
    return E10, E01, E20, E11, E02


def fit_EC_EJ_g_f_res_ng(flux_01, f_01, flux_12, f_12, flux_r, f_r, ng=0,
                         asym=0):
    """
    Fits EC, EJ, g and f_res from a list of f01, f12 and f resonators
    as a function of thier respective flux settings by numerical optimization.
    for initial guess it takes the maximum of the inputs
    """
    # initial guesses
    g_01_ss_guess = 300e6
    EC_guess = np.max(f_01) - np.max(f_12)
    print('EC_guess', EC_guess / 1e6)
    EJmax_guess = (np.max(f_01) + EC_guess) ** 2 / (8 * EC_guess)

    f_r_guess = np.min(f_r)
    asym_guess = 0.1

    def g01(g_01_ss, EC, EJ, EJmax):
        return (EJ / EC) ** (1 / 4) / (EJmax / EC) ** (1 / 4) * g_01_ss

    def penaltyfn(params):
        EC, EJmax, f_r_bare, g_01_ss, asym = params
        # calculate f01s
        f01s = []
        for fl in flux_01:
            EJ = EJmax * np.sqrt(
                asym ** 2 + (1 - asym ** 2) * np.cos(np.pi * fl) ** 2)
            g_01 = g01(g_01_ss, EC, EJ, EJmax)
            f01s.append(
                calculate_transmon_and_resonator_transitions(EC, EJ, f_r_bare,
                                                             g_01, ng=ng)[0])
        f01s = np.array(f01s)
        # calculate f12s
        f12s = []
        for fl in flux_12:
            EJ = EJmax * np.sqrt(
                asym ** 2 + (1 - asym ** 2) * np.cos(np.pi * fl) ** 2)
            g_01 = g01(g_01_ss, EC, EJ, EJmax)
            f12s.append(
                calculate_transmon_and_resonator_transitions(EC, EJ, f_r_bare,
                                                             g_01, ng=ng)[1])
        f12s = np.array(f12s)
        # calculate f_rs
        f_rs = []
        for fl in flux_r:
            EJ = EJmax * np.sqrt(
                asym ** 2 + (1 - asym ** 2) * np.cos(np.pi * fl) ** 2)
            g_01 = g01(g_01_ss, EC, EJ, EJmax)
            f_rs.append(
                calculate_transmon_and_resonator_transitions(EC, EJ, f_r_bare,
                                                             g_01, ng=ng)[2])
        f_rs = np.array(f_rs)

        penalty_01 = f_01 - f01s
        penalty_12 = f_12 - f12s
        penalty_alpha = (f_01 - f_12) - (f01s - f12s)
        penalty_r = f_r - f_rs
        return np.concatenate(
            (penalty_01, penalty_12, penalty_alpha * 20, penalty_r * 15))

    (EC0, EJmax, fres, g_01_ss, asym), success = optimize.leastsq(
        penaltyfn,
        (EC_guess, EJmax_guess, f_r_guess, g_01_ss_guess, asym_guess))
    print(success)
    print('with', EC0, EJmax, fres, g_01_ss, asym)

    return EC0, EJmax, fres, g_01_ss, asym


def fit_EC_EJ(f01, f12):
    """
    Calculates EC and EJ from f01 and f12 by numerical optimization.
    """
    # initial guesses
    EC0 = f01 - f12
    EJ0 = (f01 + EC0) ** 2 / (8 * EC0)

    def penaltyfn(Es): return calculate_transmon_transitions(*Es) - [f01, f12]
    (EC, EJ), success = optimize.leastsq(penaltyfn, (EC0, EJ0))
    return EC, EJ


def solve_quadratic_equation(a, b, c, verbose=False):
    """
    returns solutions to the quadratic equation. Will raise an error if the
    solution is negative
    """
    d = b ** 2 - 4 * a * c
    if d < 0:
        if verbose:
            print("This equation has no real solution")
        return [np.NAN, np.NAN]
    elif d == 0:
        x = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
        if verbose:
            print("This equation has one solution: ", x)
        return [x, x]
    else:
        x1 = (-b + np.sqrt((b ** 2) - (4 * (a * c)))) / (2 * a)
        x2 = (-b - np.sqrt((b ** 2) - (4 * (a * c)))) / (2 * a)
        if verbose:
            print("This equation has two solutions: ", x1, " or", x2)
        return [x1, x2]


"""Chirp z-Transform.
As described in
Rabiner, L.R., R.W. Schafer and C.M. Rader.
The Chirp z-Transform Algorithm.
IEEE Transactions on Audio and Electroacoustics, AU-17(2):86--92, 1969
"""


def chirpz(x, A, W, M):
    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}
    """
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex, x.dtype) or np.issubdtype(np.float, x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x, dtype=np.complex)

    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))

    n = np.arange(N, dtype=float)
    y = np.power(A, -n) * np.power(W, n**2 / 2.) * x
    Y = np.fft.fft(y, L)

    v = np.zeros(L, dtype=np.complex)
    v[:M] = np.power(W, -n[:M]**2/2.)
    v[L-N+1:] = np.power(W, -n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)

    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W, k**2 / 2.)

    return g


def zoom_fft(t, y, fmin, fmax):
    m = len(t)
    Fs = 1./(t[1]-t[0])
    chirp_y = chirpz(y, M=m, A=np.exp(2*1j*np.pi*fmin/(Fs)),
                     W=np.exp(-2*1j*np.pi*(fmax-fmin)/(m*Fs)))
    chirp_x = fmin + np.arange(m)*(fmax-fmin)/m
    return [chirp_x, chirp_y]


def calculate_f_qubit_from_power_scan(f_bare, f_shifted, g_coupling=65e6, RWA=False):
    '''
    Inputs are in Hz
    f_bare: the resonator frequency without a coupled qubit
    f_shifted: the reso freq shifted due to coupling of a qwubit
    g_coupling: the coupling strengs
    Output:
    f_q: in Hz
    '''
    w_r = f_bare * 2 * np.pi
    w_shift = f_shifted * 2*np.pi
    g = 2*np.pi * g_coupling
    shift = (w_shift - w_r)/g**2
    # f_shift > 0 when f_qubit<f_res
    # For the non-RWA result (only dispersive approximation)
    if (RWA == False):
        w_q = -1/(shift) + np.sqrt(1/(shift**2)+w_r**2)
    # For the RWA approximation
    else:
        w_q = -1/shift + w_r
    return w_q/(2.*np.pi)


def calculate_g_coupling_from_frequency_shift(f_bare, f_shifted, f_qubit):
    w_r = 2*np.pi * f_bare
    w_shift = 2*np.pi * f_shifted
    w_q = 2*np.pi*f_qubit
    shift = w_shift-w_r
    rhs = 1./(w_q-w_r) + 1./(w_q+w_r)
    # rhs_RWA = 1./(w_q-w_r)
    return np.sqrt(np.abs(shift/rhs))/(2*np.pi)


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

    def multiply_vec(vec): return reduce(lambda x, y: x * y, vec)
    # function that multiplies all elements in vector from idx_min
    idx_min = np.argmin(y)

    def mask_ii(ii): return multiply_vec(
        mask[min(idx_min, ii):max(idx_min, ii + 1)])
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
    x_min = -0.5 * my_fit_res.best_values['b'] / my_fit_res.best_values['a']
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
    return cmap((i / max_num) % 1)


def get_color_order_hsv(i, max_num):
    # take a blue to red scale from 0 to max_num
    # uses HSV system, H_red = 0, H_green = 1/3 H_blue=2/3
    return colors.hsv_to_rgb(2. * float(i) / (float(max_num) * 3.), 1., 1.)


def get_color_list(max_num, cmap='viridis'):
    """
    Return an array of max_num colors take in even spacing from the
    color map cmap.
    """
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


def print_pars_table(n_ts=10, pars=None):
    """
    Prints out a table containing the value for the indicated parameters from the last N timestamps.
    Input:
        n_ts (int), number of time-stamps to include in the table (rows).
        pars (list), list spanning the parameters of interest (columns).

    Every element on the list pars need to correspond to a parameter stored in the Instrument settings of the HDF5 data-files.
    Examples:
        pars = ['IVVI.dac1', 'IVVI.dac2', 'IVVI.dac3']
        pars = ['Qubit.f_RO', 'Qubit.RO_acq_integration_length', 'Qubit.RO_pulse_power']
        pars = ['Qubit.spec_pow', 'Qubit.RO_power_cw']
    """
    ts_list = return_last_n_timestamps(n_ts)
    pdict = {}
    nparams = []
    for i, p in enumerate(pars):
        pdict.update({p: p})
    opt_dict = {'scan_label': '', 'exact_label_match': False}
    # print(ts_list)
    scans = RA.quick_analysis(t_start=ts_list[-1], t_stop=ts_list[0],
                              options_dict=opt_dict,
                              params_dict_TD=pdict, numeric_params=nparams)

    pars_line = 'timestamp \t'
    for p in pars:
        pars_line = pars_line + p + '\t'
    print(pars_line)
    for i, ts in enumerate(scans.TD_timestamps):
        i_line = '%s \t' % ts
        for p in pars:
            i_line = i_line + scans.TD_dict[p][i] + '\t'
        print(i_line)
