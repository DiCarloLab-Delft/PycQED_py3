import os
import numpy as np
import h5py
import json
import datetime
from pycqed.measurement import hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
import errno
import pycqed as pq
import sys
import glob
from os.path import dirname, exists
from os import makedirs
import logging
import subprocess
from functools import reduce  # forward compatibility for Python 3
import operator


def get_git_revision_hash():
    try:
        # Refers to the global qc_config
        PycQEDdir = pq.__path__[0]
        hash = subprocess.check_output(['git', 'rev-parse',
                                        '--short=10', 'HEAD'], cwd=PycQEDdir)
    except:
        logging.warning('Failed to get Git revision hash, using 00000 instead')
        hash = '00000'

    return hash


def str_to_bool(s):
    valid = {'true': True, 't': True, '1': True,
             'false': False, 'f': False, '0': False, }
    if s.lower() not in valid:
        raise KeyError('{} not a valid boolean string'.format(s))
    b = valid[s.lower()]
    return b


def bool_to_int_str(b):
    if b:
        return '1'
    else:
        return '0'


def int_to_bin(x, w, lsb_last=True):
    """
    Converts an integer to a binary string of a specified width
    x (int) : input integer to be converted
    w (int) : desired width
    lsb_last (bool): if False, reverts the string e.g., int(1) = 001 -> 100
    """
    bin_str = '{0:{fill}{width}b}'.format((int(x) + 2**w) % 2**w,
                                          fill='0', width=w)
    if lsb_last:
        return bin_str
    else:
        return bin_str[::-1]


def mopen(filename, mode='w'):
    if not exists(dirname(filename)):
        try:
            makedirs(dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    file = open(filename, mode='w')
    return file


def dict_to_ordered_tuples(dic):
    '''Convert a dictionary to a list of tuples, sorted by key.'''
    if dic is None:
        return []
    keys = dic.keys()
    # keys.sort()
    ret = [(key, dic[key]) for key in keys]
    return ret


def to_hex_string(byteval):
    '''
    Returns a hex representation of bytes for printing purposes
    '''
    return "b'" + ''.join('\\x{:02x}'.format(x) for x in byteval) + "'"


def load_settings_onto_instrument(instrument, load_from_instr=None,
                                  folder=None, label=None,
                                  timestamp=None, **kw):
    '''
    Loads settings from an hdf5 file onto the instrument handed to the
    function.
    By default uses the last hdf5 file in the datadirectory.
    By giving a label or timestamp another file can be chosen as the
    settings file.
    '''

    older_than = None
    instrument_name = instrument.name
    success = False
    count = 0
    while success is False and count < 10:
        try:
            if folder is None:
                folder = a_tools.get_folder(timestamp=timestamp,
                                            older_than=older_than, **kw)
            else:
                folder = folder
            filepath = a_tools.measurement_filename(folder)
            f = h5py.File(filepath, 'r')
            sets_group = f['Instrument settings']
            if load_from_instr is None:
                ins_group = sets_group[instrument_name]
            else:
                ins_group = sets_group[load_from_instr]
            print('Loaded Settings Successfully')
            success = True
        except:
            older_than = os.path.split(folder)[0][-8:] \
                + '_' + os.path.split(folder)[1][:6]
            folder = None
            success = False
        count += 1

    if not success:
        print('Could not open settings for instrument "%s"' % (
            instrument_name))
        return False

    for parameter, value in ins_group.attrs.items():
        if value != 'None':  # None is saved as string in hdf5
            if type(value) == str:
                if value == 'False':
                    try:
                        instrument.set(parameter, False)
                    except:
                        print('Could not set parameter: "%s" to "%s" for instrument "%s"' % (
                            parameter, value, instrument_name))
                else:
                    try:
                        instrument.set(parameter, float(value))
                    except Exception:
                        try:
                            instrument.set(parameter, value)
                        except:
                            try:
                                instrument.set(parameter, int(value))
                            except:
                                print('Could not set parameter: "%s" to "%s" for instrument "%s"' % (
                                    parameter, value, instrument_name))
            else:
                instrument.set(parameter, value)
    f.close()
    return True


def load_settings_onto_instrument_v2(instrument, load_from_instr: str=None,
                                     label: str='', filepath: str=None,
                                     timestamp: str=None):
    '''
    Loads settings from an hdf5 file onto the instrument handed to the
    function. By default uses the last hdf5 file in the datadirectory.
    By giving a label or timestamp another file can be chosen as the
    settings file.

    Args:
        instrument (instrument) : instrument onto which settings should be
            loaded
        load_from_instr (str) : optional name of another instrument from
            which to load the settings.
        label (str)           : label used for finding the last datafile
        filepath (str)        : exact filepath of the hdf5 file to load.
            if filepath is specified, this takes precedence over the file
            locating options (label, timestamp etc.).
        timestamp (str)       : timestamp of file in the datadir


    '''

    older_than = None
    folder = None
    instrument_name = instrument.name
    success = False
    count = 0
    # Will try multiple times in case the last measurements failed and
    # created corrupt data files.
    while success is False and count < 10:
        try:
            if filepath is None:
                folder = a_tools.get_folder(timestamp=timestamp, label=label,
                                            older_than=older_than)
                filepath = a_tools.measurement_filename(folder)

            f = h5py.File(filepath, 'r')
            snapshot = {}
            h5d.read_dict_from_hdf5(snapshot, h5_group=f['Snapshot'])

            if load_from_instr is None:
                ins_group = snapshot['instruments'][instrument_name]
            else:
                ins_group = snapshot['instruments'][load_from_instr]
            success = True
        except Exception as e:
            logging.warning(e)
            older_than = os.path.split(folder)[0][-8:] \
                + '_' + os.path.split(folder)[1][:6]
            folder = None
            success = False
        count += 1

    if not success:
        logging.warning('Could not open settings for instrument "%s"' % (
            instrument_name))
        return False

    for parname, par in ins_group['parameters'].items():
        try:
            if instrument.parameters[parname].has_set:
                instrument.set(parname, par['value'])
        except Exception as e:
            print('Could not set parameter: "{}" to "{}" '
                  'for instrument "{}"'.format(parname, par['value'],
                                               instrument_name))
            logging.warning(e)
    f.close()
    return True



def send_email(subject='PycQED needs your attention!',
               body='', email=None):
    # Import smtplib for the actual sending function
    import smtplib
    # Here are the email package modules we'll need
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if email is None:
        email = qt.config['e-mail']

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = subject
    family = 'serwan.asaad@gmail.com'
    msg['From'] = 'Lamaserati@tudelft.nl'
    msg['To'] = email
    msg.attach(MIMEText(body, 'plain'))

    # Send the email via our own SMTP server.
    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login('DCLabemail@gmail.com', 'DiCarloLab')
    s.sendmail(email, family, msg.as_string())
    s.quit()


def list_available_serial_ports():
    '''
    Lists serial ports

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of available serial ports

    Frunction from :
    http://stackoverflow.com/questions/12090503/
        listing-available-com-ports-with-python
    '''
    import serial
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(256)]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')

    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')

    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def add_suffix_to_dict_keys(inputDict, suffix):
    return {str(key)+suffix: (value) for key, value in inputDict.items()}


def execfile(path, global_vars=None, local_vars=None):
    """
    Args:
        path (str)  : filepath of the file to be executed
        global_vars : use globals() to use globals from namespace
        local_vars  : use locals() to use locals from namespace

    execfile function that existed in python 2 but does not exists in python3.
    """
    with open(path, 'r') as f:
        code = compile(f.read(), path, 'exec')
        exec(code, global_vars, local_vars)


def span_num(center: float, span: float, num: int, endpoint: bool=True):
    """
    Creates a linear span of points around center
    Args:
        center (float) : center of the array
        span   (float) : span the total range of values to span
        num      (int) : the number of points in the span
        endpoint (bool): whether to include the endpoint

    """
    return np.linspace(center-span/2, center+span/2, num, endpoint=endpoint)


def span_step(center: float, span: float, step: float, endpoint: bool=True):
    """
    Creates a range of points spanned around a center
    Args:
        center (float) : center of the array
        span   (float) : span the total range of values to span
        step   (float) : the stepsize between points in the array
        endpoint (bool): whether to include the endpoint in the span

    """
    # True*step/100 in the arange ensures the right boundary is included
    return np.arange(center-span/2, center+span/2+endpoint*step/100, step)


def gen_sweep_pts(start: float=None, stop: float=None,
                  center: float=0, span: float=None,
                  num: int=None, step: float=None, endpoint=True):
    """
    Generates an array of sweep points based on different types of input
    arguments.
    Boundaries of the array can be specified using either start/stop or
    using center/span. The points can be specified using either num or step.

    Args:
        start  (float) : start of the array
        stop   (float) : end of the array
        center (float) : center of the array
                         N.B. 0 is chosen as a sensible default for the span.
                         it is argued that no such sensible default exists
                         for the other types of input.
        span   (float) : span the total range of values to span

        num      (int) : number of points in the array
        step   (float) : the stepsize between points in the array
        endpoint (bool): whether to include the endpoint

    """
    if (start is not None) and (stop is not None):
        if num is not None:
            return np.linspace(start, stop, num, endpoint=endpoint)
        elif step is not None:
            # numpy arange does not natively support endpoint
            return np.arange(start, stop + endpoint*step/100, step)
        else:
            raise ValueError('Either "num" or "step" must be specified')
    elif (center is not None) and (span is not None):
        if num is not None:
            return span_num(center, span, num, endpoint=endpoint)
        elif step is not None:
            return span_step(center, span, step, endpoint=endpoint)
        else:
            raise ValueError('Either "num" or "step" must be specified')
    else:
        raise ValueError('Either ("start" and "stop") or '
                         '("center" and "span") must be specified')


def getFromDict(dataDict: dict, mapList: list):
    """
    get a value from a nested dictionary by specifying a list of keys

    Args:
        dataDict: nested dictionary to get the value from
        mapList : list of strings specifying the key of the item to get
    Returns:
        value from dictionary

    example:
        example_dict = {'a': {'nest_a': 5, 'nest_b': 8}
                        'b': 4}
        getFromDict(example_dict, ['a', 'nest_a']) -> 5
    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict: dict, mapList: list, value):
    """
    set a value in a nested dictionary by specifying the location using a list
    of key.

    Args:
        dataDict: nested dictionary to set the value in
        mapList : list of strings specifying the key of the item to set
        value   : the value to set

    example:
        example_dict = {'a': {'nest_a': 5, 'nest_b': 8}
                        'b': 4}
        example_dict_after = getFromDict(example_dict, ['a', 'nest_a'], 6)
        example_dict = {'a': {'nest_a': 6, 'nest_b': 8}
                        'b': 4}
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


class NumpyJsonEncoder(json.JSONEncoder):
    '''
    JSON encoder subclass that converts Numpy types to native python types
    for saving in JSON files.
    Also converts datetime objects to strings.
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return str(obj)
        else:
            return super().default(obJ)
