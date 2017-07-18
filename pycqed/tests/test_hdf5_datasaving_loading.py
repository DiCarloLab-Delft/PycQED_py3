import unittest
import h5py
from pycqed.measurement import hdf5_data as h5d
import numpy as np

from pycqed.measurement import measurement_control
from pycqed.instrument_drivers.physical_instruments.dummy_instruments \
    import DummyParHolder
from pycqed.utilities.get_default_datadir import get_default_datadir

from qcodes import station


class Test_HDF5(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        # set up a pulsar with some mock settings for the element
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=True, verbose=True)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        self.mock_parabola = DummyParHolder('mock_parabola')
        self.station.add_component(self.mock_parabola)
        self.mock_parabola_2 = DummyParHolder('mock_parabola_2')
        self.station.add_component(self.mock_parabola_2)
        self.datadir = get_default_datadir()

    def test_storing_and_loading_station_snapshot(self):
        """
        Stores and writes a station (instrument) snapshot.
        """

        self.mock_parabola_2.x(1)
        self.mock_parabola_2.y(2.245)
        self.mock_parabola_2.array_like(np.linspace(0, 11, 23))

        snap = self.station.snapshot(update=True)
        data_object = h5d.Data(name='test_object_snap', datadir=self.datadir)
        h5d.write_dict_to_hdf5(snap, data_object)
        data_object.close()
        filepath = data_object.filepath

        new_dict = {}
        opened_hdf5_file = h5py.File(filepath, 'r')
        h5d.read_dict_from_hdf5(new_dict, opened_hdf5_file)

        self.assertEqual(snap.keys(), new_dict.keys())
        self.assertEqual(snap['instruments'].keys(),
                         new_dict['instruments'].keys())
        mock_parab_pars = snap['instruments']['mock_parabola_2']['parameters']

        self.assertEqual(mock_parab_pars['x']['value'],
                         1)
        self.assertEqual(mock_parab_pars['y']['value'],
                         2.245)
        np.testing.assert_array_equal(
            mock_parab_pars['array_like']['value'],
            np.linspace(0, 11, 23))

    def test_writing_and_reading_dicts_to_hdf5(self):
        """
        Tests dumping some random dictionary to hdf5 and reading back the
        stored values. The input dictionary contains:
            - list of ints
            - list of floats
            - nested dict
            - 1D array
            - 2D array
            - list of mixed type
        """
        test_dict = {
            'list_of_ints': list(np.arange(5)),
            'list_of_floats': list(np.arange(5.1)),
            'weird_dict': {'a': 5},
            'dataset1': np.linspace(0, 20, 31),
            'dataset2': np.array([[2, 3, 4, 5],
                                  [2, 3, 1, 2]]),
            'list_of_mixed_type': ['hello', 4, 4.2]}

        data_object = h5d.Data(name='test_object', datadir=self.datadir)
        h5d.write_dict_to_hdf5(test_dict, data_object)
        data_object.close()
        filepath = data_object.filepath

        new_dict = {}
        opened_hdf5_file = h5py.File(filepath, 'r')
        h5d.read_dict_from_hdf5(new_dict, opened_hdf5_file)
        # objects are not identical but the string representation should be
        self.assertEqual(test_dict.keys(), new_dict.keys())
        self.assertEqual(test_dict['list_of_ints'], new_dict['list_of_ints'])
        self.assertEqual(test_dict['list_of_floats'], new_dict['list_of_floats'])
        self.assertEqual(test_dict['weird_dict'], new_dict['weird_dict'])
        self.assertEqual(test_dict.keys(), new_dict.keys())

    @unittest.skip('NotImplemented')
    def test_loading_settings_onto_instrument(self):
        pass

