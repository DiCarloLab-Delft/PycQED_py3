# hdf5_data.py, handling data in hdf5 container
#
# Wolfgang Pfaff <wolfgangpfff@gmail.com>
# Reinier Heeres <reinier@heeres.eu>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module providing handling of HDF5 data within qtlab.

Contains:
- a data class (HDF5Data) which is essentially a wrapper of a h5py data
  object, adapted for usage with qtlab
- name generators in the style of qtlab Data objects
- functions to create standard data sets
"""

import os
import time
import h5py
from lib.config import get_config
config = get_config()
in_qtlab = config.get('qtlab', False)

import data


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
                        ts = time.localtime((time.mktime(ts)+1))
                        tsd = time.strftime('%H%M%S', ts)
                    if counter >= 10:
                        raise Exception()
                except OSError as e:
                    # if the day folder does not exist the timestamp is unique
                    if (
                            '[Error 3] The system cannot find the path'
                            not in str(e) and
                            '[Errno 2] No such file or directory:'
                            not in str(e)):
                        raise e
                    else:
                        timestamp_verified = True

            if name is not None:
                path = os.path.join(path, tsd+'_'+name)
            else:
                path = os.path.join(path, tsd)

        return path, tsd

    def new_filename(self, data_obj):
        '''Return a new filename, based on name and timestamp.'''
        path, tstr = self.create_data_dir(config['datadir'],
                                          name=data_obj._name,
                                          ts=data_obj._localtime)
        filename = '%s_%s.hdf5' % (tstr, data_obj._name)
        return os.path.join(path, filename)


class Data(h5py.File):

    _data_list = data.Data._data_list
    _filename_generator = DateTimeGenerator()

    def __init__(self, name='None', filepath=None, *args, **kwargs):
        """
        Creates an empty data set including the file, for which the currently
        set file name generator is used.

        kwargs:
            name (string) : default is 'data' (%timemark is interpreted as
            its timemark)
        """

        # FIXME: the name generation here is a bit nasty
        name = data.Data._data_list.new_item_name(self, name)
        self._name = name

        self._localtime = time.localtime()
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        if filepath:
            self.filepath = filepath
        else:
            self.filepath = self._filename_generator.new_filename(self)

        self.filepath = self.filepath.replace("%timemark", self._timemark)

        self.folder, self._filename = os.path.split(self.filepath)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        super(Data, self).__init__(self.filepath, 'a')
        self.flush()
