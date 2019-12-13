import os
import pytest
import pycqed as pq
from pycqed.analysis import analysis_toolbox as a_tools

datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
a_tools.datadir = datadir


def test_get_timestamps_in_nonexistent_folder_does_not_crash():
    a_tools.get_timestamps_in_range('20170412_000000', '20170414_000000',
                                    label='Rabi')


def test_get_timestamps_in_range():
    timestamps = a_tools.get_timestamps_in_range(
        '20170412_000000', '20170413_000000', label='Rabi')
    assert timestamps == ['20170412_183928', '20170412_185618']


def test_get_timestamps_in_range_no_matches_raises_ValueError():
    with pytest.raises(ValueError):

        a_tools.get_timestamps_in_range(
            '20170412_000000', '20170413_000000', label='fake_label')


def test_get_datafilepath_from_timestamp():
    timestamp = '20170412_183928'
    data_fp = a_tools.get_datafilepath_from_timestamp(timestamp)
    print(data_fp)
    assert data_fp == os.path.join(
        datadir, '20170412', '183928_Rabi-n1_q9', '183928_Rabi-n1_q9.hdf5')


def test_get_datafilepath_from_timestamp_raises_no_data():
    timestamp = '20170412_183929'
    with pytest.raises(ValueError):
        a_tools.get_datafilepath_from_timestamp(timestamp)
