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
    '''
    Class to generate filenames / directories based on the date and time.
    '''

    def __init__(self):
        pass

    def create_data_dir(self, datadir: str, name: str=None, ts=None,
                        datesubdir: bool=True, timesubdir: bool=True):
        '''
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
        '''

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
        '''Return a new filename, based on name and timestamp.'''
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
    '''
    Required because h5py does not support python3 strings
    '''
    # converts byte type to string because of h5py datasaving
    if type(s) == str:
        s = s.encode('utf-8')
    # If it is an array of value decodes individual entries
    elif type(s) == np.ndarray or list:
        s = [s.encode('utf-8') for s in s]
    return s


def write_dict_to_hdf5(data_dict: dict, entry_point):
    """
    Args:
        data_dict (dict): dictionary to write to hdf5 file
        entry_point (hdf5 group.file) : location in the nested hdf5 structure
            where to write to.
    """
    for key, item in data_dict.items():
        if isinstance(item, (str, bool, tuple, float, int)):
            entry_point.attrs[key] = item
        elif isinstance(item, np.ndarray):
            entry_point.create_dataset(key, data=item)
        elif item is None:
            # as h5py does not support saving None as attribute
            # I create special string, note that this can create
            # unexpected behaviour if someone saves a string with this name
            entry_point.attrs[key] = 'NoneType:__None__'

        elif isinstance(item, dict):
            entry_point.create_group(key)
            write_dict_to_hdf5(data_dict=item,
                               entry_point=entry_point[key])
        elif isinstance(item, list):
            if len(item) > 0:
                elt_type = type(item[0])
                if all(isinstance(x, elt_type) for x in item):
                    if isinstance(item[0], (int, float,
                                            np.int32, np.int64)):

                        entry_point.create_dataset(key,
                                                   data=np.array(item))
                        entry_point[key].attrs['list_type'] = 'array'
                    elif isinstance(item[0], str):
                        dt = h5py.special_dtype(vlen=str)
                        data = np.array(item)
                        data = data.reshape((-1, 1))
                        ds = entry_point.create_dataset(
                            key, (len(data), 1), dtype=dt)
                        ds[:] = data
                    elif isinstance(item[0], dict):
                        entry_point.create_group(key)
                        group_attrs = entry_point[key].attrs
                        group_attrs['list_type'] = 'dict'
                        base_list_key = 'list_idx_{}'
                        group_attrs['base_list_key'] = base_list_key
                        group_attrs['list_length'] = len(item)
                        for i, list_item in enumerate(item):
                            list_item_grp = entry_point[key].create_group(
                                base_list_key.format(i))
                            write_dict_to_hdf5(
                                data_dict=list_item,
                                entry_point=list_item_grp)
                    else:
                        logging.warning(
                            'List of type "{}" for "{}":"{}" not '
                            'supported, storing as string'.format(
                                elt_type, key, item))
                        entry_point.attrs[key] = str(item)
                else:
                    logging.warning(
                        'List of mixed type for "{}":"{}" not supported, '
                        'storing as string'.format(type(item), key, item))
                    entry_point.attrs[key] = str(item)
            else:
                # as h5py does not support saving None as attribute
                entry_point.attrs[key] = 'NoneType:__emptylist__'

        else:
            logging.warning(
                'Type "{}" for "{}" (key): "{}" (item) at location {} '
                'not supported, '
                'storing as string'.format(type(item), key, item, entry_point))
            entry_point.attrs[key] = str(item)


def read_dict_from_hdf5(data_dict, h5_group):
    if 'list_type' not in h5_group.attrs:
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):
                data_dict[key] = {}
                data_dict[key] = read_dict_from_hdf5(data_dict[key],
                                                     item)
            else:  # item either a group or a dataset
                if 'list_type' not in item.attrs:
                    data_dict[key] = item.value
                else:
                    data_dict[key] = list(item.value)
        for key, item in h5_group.attrs.items():
            if type(item) is str:
                # Extracts "None" as an exception as h5py does not support
                # storing None, nested if statement to avoid elementwise
                # comparison warning
                if item == 'NoneType:__None__':
                    item = None
                elif item == 'NoneType:__emptylist__':
                    item = []
            data_dict[key] = item
    elif h5_group.attrs['list_type'] == 'dict':
        # preallocate empty list
        list_to_be_filled = [None] * h5_group.attrs['list_length']
        base_list_key = h5_group.attrs['base_list_key']
        for i in range(h5_group.attrs['list_length']):
            list_to_be_filled[i] = {}
            read_dict_from_hdf5(
                data_dict=list_to_be_filled[i],
                h5_group=h5_group[base_list_key.format(i)])

        # THe error is here!, extract correctly but not adding to
        # data dict correctly
        data_dict = list_to_be_filled
    else:
        raise NotImplementedError('cannot read "list_type":"{}"'.format(
            h5_group.attrs['list_type']))
    return data_dict
