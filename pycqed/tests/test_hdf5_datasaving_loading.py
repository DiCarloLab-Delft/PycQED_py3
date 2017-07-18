import unittest
import h5py
from pycqed.measurement import hdf5_data as h5d
import numpy as np
import pycqed.utilities.general as gen
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
            'MC', live_plot_enabled=False, verbose=True)
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

        The storing is not magic, it currently does not work for:
            - list of mixed type (stored as string)
        """
        test_dict = {
            'list_of_ints': list(np.arange(5)),
            'list_of_floats': list(np.arange(5.1)),
            'some_bool': True,
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
        self.assertEqual(test_dict['list_of_floats'],
                         new_dict['list_of_floats'])
        self.assertEqual(test_dict['weird_dict'], new_dict['weird_dict'])
        self.assertEqual(test_dict['some_bool'], new_dict['some_bool'])

    def test_loading_settings_onto_instrument(self):
        """
        Tests storing and reading of parameters.
        Tests for different types of parameters including
        - array
        - float
        - int
        - dict
        - bool
        """
        arr = np.linspace(12, 42, 11)
        self.mock_parabola.array_like(arr)
        self.mock_parabola.x(42.23)
        self.mock_parabola.y(2)
        self.mock_parabola.status(True)
        self.mock_parabola.dict_like({'a': {'b': [2, 3, 5]}})

        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points([0, 1])
        self.MC.set_detector_function(self.mock_parabola.skewed_parabola)
        self.MC.run('test_MC_snapshot_storing')
        self.mock_parabola.array_like(arr+5)
        self.mock_parabola.x(13)
        # Test that these are not the same as before the experiment
        np.testing.assert_array_equal(self.mock_parabola.array_like(),
                                      arr+5)
        self.assertEqual(self.mock_parabola.x(), 13)

        # Now load the settings from the last file
        gen.load_settings_onto_instrument_v2(self.mock_parabola,
                                             label='test_MC_snapshot_storing')

        # Test that these are not the same as before the experiment
        np.testing.assert_array_equal(self.mock_parabola.array_like(),
                                      arr)
        self.assertEqual(self.mock_parabola.x(), 42.23)

        # Loading it into another instrument to test for the other values
        gen.load_settings_onto_instrument_v2(
            self.mock_parabola_2,
            load_from_instr=self.mock_parabola.name,
            label='test_MC_snapshot_storing')
        self.assertEqual(self.mock_parabola_2.y(), 2)
        self.assertEqual(self.mock_parabola_2.status(), True)
        self.assertEqual(self.mock_parabola_2.dict_like(),
                         {'a': {'b': [2, 3, 5]}})


