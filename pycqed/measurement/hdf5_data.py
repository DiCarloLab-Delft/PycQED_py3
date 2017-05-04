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
import logging
import time
import h5py
import numpy as np
import pycqed as pq
from uuid import getnode as get_mac


try:
    # This method is currently not working robustly
    qc_config
except NameError:
    try:
        # The datadict is the first (hard coded) fall back option
        from pycqed.init.config import setup_dict
        mac = get_mac()
        setup_name = setup_dict.mac_dict[str(mac)]
        qc_config = {'datadir': setup_dict.data_dir_dict[setup_name]}
    except Exception:
        # the default data directory location is the last fall back option
        # Stores data in the default data location (pycqed_py3/data/)
        datadir = os.path.join(os.path.dirname(pq.__file__), os.pardir, 'data')
        logging.warning('Setting datadir to default location')
        qc_config = {'datadir': datadir}


class DateTimeGenerator:
    '''
    Class to generate filenames / directories based on the date and time.
    '''

    def __init__(self):
        pass

    def create_data_dir(self, datadir, name=None, ts=None,
                        datesubdir=True, timesubdir=True):
        '''
        Create and return a new data directory.

        Input:
            datadir (string): base directory
            name (string): optional name of measurement
            ts (time.localtime()): timestamp which will be used if timesubdir=True
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

    def new_filename(self, data_obj, folder=None):
        '''Return a new filename, based on name and timestamp.'''
        if folder is None:
            folder = qc_config['datadir']

        path, tstr = self.create_data_dir(folder,
                                          name=data_obj._name,
                                          ts=data_obj._localtime)
        filename = '%s_%s.hdf5' % (tstr, data_obj._name)
        return os.path.join(path, filename)


class Data(h5py.File):

    # _data_list = data.Data._data_list
    _filename_generator = DateTimeGenerator()

    def __init__(self, name='None', filepath=None, *args, **kwargs):
        """
        Creates an empty data set including the file, for which the currently
        set file name generator is used.

        kwargs:
            name (string) : default is 'data' (%timemark is interpreted as
            its timemark)
            filepath (string) : path to the file. If this is a folder it will 
            generate a folder within it with the standard timestamp structure
        """
        # FIXME: the name generation here is a bit nasty
        # name = data.Data._data_list.new_item_name(self, name)
        self._name = name

        self._localtime = time.localtime()
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        if filepath is not None:
            if os.path.isdir(filepath):
                self.filepath = self._filename_generator.new_filename(self, folder=filepath)
            else:
                self.filepath = filepath
        else:
            self.filepath = self._filename_generator.new_filename(self)

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
