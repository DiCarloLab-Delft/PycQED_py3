"""
Module for handling HDF5 data within qcodes.
Based on hdf5 datawrapper from qtlab originally by Reinier Heeres and
Wolfgang Pfaff.

Contains:
- a data class (HDF5Data) which is essentially a wrapper of a h5py data
  object, adapted for usage with qcodes
- name generators in the style of qtlab Data objects
- functions to create standard data sets
"""

import os
import time
import h5py
import numpy as np
import logging


class DateTimeGenerator:
    """
    Class to generate filenames / directories based on the date and time.
    """

    def __init__(self):
        pass

    def create_data_dir(self, datadir: str, name: str=None, ts=None,
                        datesubdir: bool=True, timesubdir: bool=True):
        """
        Create and return a new data directory.

        Input:
            datadir (string): base directory
            name (string): optional name of measurement
            ts (time.localtime()): timestamp which will be used
                if timesubdir=True
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time

        Output:
            The directory to place the new file in
        """

        path = datadir
        if ts is None:
            ts = time.localtime()
        if datesubdir:
            path = os.path.join(path, time.strftime('%Y%m%d', ts))
        if timesubdir:
            tsd = time.strftime('%H%M%S', ts)
            timestamp_verified = False
            counter = 0
            # Verify if timestamp is unique by seeing if the folder exists
            while not timestamp_verified:
                counter += 1
                try:
                    measdirs = [d for d in os.listdir(path)
                                if d[:6] == tsd]
                    if len(measdirs) == 0:
                        timestamp_verified = True
                    else:
                        # if timestamp not unique, add one second
                        # This is quite a hack
                        ts = time.localtime((time.mktime(ts)+1))
                        tsd = time.strftime('%H%M%S', ts)
                    if counter >= 3600:
                        raise Exception()
                except OSError as err:
                    if 'cannot find the path specified' in str(err):
                        timestamp_verified = True
                    elif 'No such file or directory' in str(err):
                        timestamp_verified = True
                    else:
                        raise err
            if name is not None:
                path = os.path.join(path, tsd+'_'+name)
            else:
                path = os.path.join(path, tsd)

        return path, tsd

    def new_filename(self, data_obj, folder):
        """Return a new filename, based on name and timestamp."""
        path, tstr = self.create_data_dir(folder,
                                          name=data_obj._name,
                                          ts=data_obj._localtime)
        filename = '%s_%s.hdf5' % (tstr, data_obj._name)
        return os.path.join(path, filename)


class Data(h5py.File):

    def __init__(self, name: str, datadir: str):
        """
        Creates an empty data set including the file, for which the currently
        set file name generator is used.

        kwargs:
            name (string) : base name of the file
            datadir (string) : A folder will be created within the datadir
                using the standard timestamp structure
        """
        self._name = name

        self._localtime = time.localtime()
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        self.filepath = DateTimeGenerator().new_filename(
            self, folder=datadir)

        self.filepath = self.filepath.replace("%timemark", self._timemark)

        self.folder, self._filename = os.path.split(self.filepath)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        super(Data, self).__init__(self.filepath, 'a')
        self.flush()


def encode_to_utf8(s):
    """
    Required because h5py does not support python3 strings
    """
    # converts byte type to string because of h5py datasaving
    if isinstance(s, str):
        s = s.encode('utf-8')
    # If it is an array of value decodes individual entries
    elif isinstance(s, (np.ndarray, list, tuple)):
        s = [s.encode('utf-8') for s in s]
    return s


def write_dict_to_hdf5(data_dict: dict, entry_point, overwrite=False):
    """
    Args:
        data_dict (dict): dictionary to write to hdf5 file
        entry_point (hdf5 group.file) : location in the nested hdf5 structure
            where to write to.
    """
    for key, item in data_dict.items():
        # Basic types
        if isinstance(item, (str, float, int, bool, np.number,
                             np.float_, np.int_, np.bool_)):
            try:
                entry_point.attrs[key] = item
            except Exception as e:
                print('Exception occurred while writing'
                      ' {}:{} of type {}'.format(key, item, type(item)))
                logging.warning(e)
        elif isinstance(item, np.ndarray):
            try:
                entry_point.create_dataset(key, data=item)
            except RuntimeError:
                if overwrite:
                    del entry_point[key]
                    entry_point.create_dataset(key, data=item)
                else:
                    raise

        elif item is None:
            # as h5py does not support saving None as attribute
            # I create special string, note that this can create
            # unexpected behaviour if someone saves a string with this name
            entry_point.attrs[key] = 'NoneType:__None__'

        elif isinstance(item, dict):
            try:
                entry_point.create_group(key)
            except RuntimeError:
                if overwrite:
                    del entry_point[key]
                    entry_point.create_group(key)
                else:
                    raise 
            write_dict_to_hdf5(data_dict=item,
                               entry_point=entry_point[key],
                               overwrite=overwrite)
        elif isinstance(item, (list, tuple)):
            if len(item) > 0:
                elt_type = type(item[0])
                # Lists of a single type, are stored as an hdf5 dset
                if (all(isinstance(x, elt_type) for x in item) and
                        not isinstance(item[0], dict) and
                        not isinstance(item, tuple)):
                    if isinstance(item[0], (int, float,
                                            np.int32, np.int64)):
                        try:
                            entry_point.create_dataset(key, 
                                                       data=np.array(item))
                        except RuntimeError:
                            if overwrite:
                                del entry_point[key]
                                entry_point.create_dataset(key, 
                                                           data=np.array(item))
                            else:
                                raise
                        entry_point[key].attrs['list_type'] = 'array'

                    # strings are saved as a special dtype hdf5 dataset
                    elif isinstance(item[0], str):
                        dt = h5py.special_dtype(vlen=str)
                        data = np.array(item)
                        data = data.reshape((-1, 1))
                        try:
                            ds = entry_point.create_dataset(
                                key, (len(data), 1), dtype=dt)
                        except RuntimeError:
                            if overwrite:
                                del entry_point[key]
                                entry_point.create_dataset(
                                    key, (len(data), 1), dtype=dt)
                            else:
                                raise
                        ds.attrs['list_type'] = 'str'
                        ds[:] = data
                    else:
                        logging.warning(
                            'List of type "{}" for "{}":"{}" not '
                            'supported, storing as string'.format(
                                elt_type, key, item))
                        entry_point.attrs[key] = str(item)
                # Storing of generic lists/tuples
                else:
                    try:
                        entry_point.create_group(key)
                    except RuntimeError:
                        if overwrite:
                            del entry_point[key]
                            entry_point.create_group(key)
                        else:
                            raise
                    # N.B. item is of type list
                    list_dct = {'list_idx_{}'.format(idx): entry for
                                idx, entry in enumerate(item)}
                    group_attrs = entry_point[key].attrs
                    if isinstance(item, tuple):
                        group_attrs['list_type'] = 'generic_tuple'
                    else:
                        group_attrs['list_type'] = 'generic_list'
                    group_attrs['list_length'] = len(item)
                    write_dict_to_hdf5(
                        data_dict=list_dct,
                        entry_point=entry_point[key],
                        overwrite=overwrite)
            else:
                # as h5py does not support saving None as attribute
                entry_point.attrs[key] = 'NoneType:__emptylist__'

        else:
            logging.warning(
                'Type "{}" for "{}" (key): "{}" (item) at location {} '
                'not supported, '
                'storing as string'.format(type(item), key, item,
                                           entry_point))
            entry_point.attrs[key] = str(item)


def read_dict_from_hdf5(data_dict: dict, h5_group):
    """
    Reads a dictionary from an hdf5 file or group that was written using the
    corresponding "write_dict_to_hdf5" function defined above.

    Args:
        data_dict (dict):
                dictionary to which to add entries being read out.
                This argument exists because it allows nested calls of this
                function to add the data to an existing data_dict.
        h5_group  (hdf5 group):
                hdf5 file or group from which to read.
    """
    # if 'list_type' not in h5_group.attrs:
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            data_dict[key] = {}
            data_dict[key] = read_dict_from_hdf5(data_dict[key],
                                                 item)
        else:  # item either a group or a dataset
            if 'list_type' not in item.attrs:
                data_dict[key] = item.value
            elif item.attrs['list_type'] == 'str':
                # lists of strings needs some special care, see also
                # the writing part in the writing function above.
                list_of_str = [x[0] for x in item.value]
                data_dict[key] = list_of_str

            else:
                data_dict[key] = list(item.value)
    for key, item in h5_group.attrs.items():
        if isinstance(item, str):
            # Extracts "None" as an exception as h5py does not support
            # storing None, nested if statement to avoid elementwise
            # comparison warning
            if item == 'NoneType:__None__':
                item = None
            elif item == 'NoneType:__emptylist__':
                item = []
        data_dict[key] = item

    if 'list_type' in h5_group.attrs:
        if (h5_group.attrs['list_type'] == 'generic_list' or
                h5_group.attrs['list_type'] == 'generic_tuple'):
            list_dict = data_dict
            data_list = []
            for i in range(list_dict['list_length']):
                data_list.append(list_dict['list_idx_{}'.format(i)])

            if h5_group.attrs['list_type'] == 'generic_tuple':
                return tuple(data_list)
            else:
                return data_list
        else:
            raise NotImplementedError('cannot read "list_type":"{}"'.format(
                h5_group.attrs['list_type']))
    return data_dict
