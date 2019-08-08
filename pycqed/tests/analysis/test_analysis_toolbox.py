import os
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
