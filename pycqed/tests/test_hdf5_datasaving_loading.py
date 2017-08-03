import os
import pycqed as pq
import unittest
import h5py
from pycqed.measurement import hdf5_data as h5d
import numpy as np
import pycqed.utilities.general as gen
from pycqed.measurement import measurement_control
from pycqed.instrument_drivers.physical_instruments.dummy_instruments \
    import DummyParHolder

from qcodes import station
from pycqed.analysis import analysis_toolbox as a_tools


class Test_HDF5(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.MC.datadir(self.datadir)
        a_tools.datadir = self.datadir
        self.station.add_component(self.MC)

        self.mock_parabola = DummyParHolder('mock_parabola')
        self.station.add_component(self.mock_parabola)
        self.mock_parabola_2 = DummyParHolder('mock_parabola_2')
        self.station.add_component(self.mock_parabola_2)

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

        """
        test_dict = {
            'list_of_ints': list(np.arange(5)),
            'list_of_floats': list(np.arange(5.1)),
            'some_bool': True,
            'weird_dict': {'a': 5},
            'dataset1': np.linspace(0, 20, 31),
            'dataset2': np.array([[2, 3, 4, 5],
                                  [2, 3, 1, 2]]),
            'list_of_mixed_type': ['hello', 4, 4.2, {'a': 5}, [4, 3]],

            'tuple_of_mixed_type': tuple(['hello', 4, 4.2, {'a': 5}, [4, 3]]),
            'a list of strings': ['my ', 'name ', 'is ', 'earl.'],
            'some_np_bool': np.bool(True),
            'list_of_dicts': [{'a': 5}, {'b': 3}],
            'some_int': 3,
            'some_float': 3.5,
            'some_np_int': np.int(3),
            'some_np_float': np.float(3.5)
        }

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

        self.assertEqual(test_dict['list_of_dicts'],
                         new_dict['list_of_dicts'])


        self.assertEqual(test_dict['list_of_mixed_type'],
                         new_dict['list_of_mixed_type'])
        self.assertEqual(test_dict['list_of_mixed_type'][0],
                         new_dict['list_of_mixed_type'][0])
        self.assertEqual(test_dict['list_of_mixed_type'][2],
                         new_dict['list_of_mixed_type'][2])

        self.assertEqual(test_dict['tuple_of_mixed_type'],
                         new_dict['tuple_of_mixed_type'])
        self.assertEqual(type(test_dict['tuple_of_mixed_type']),
                         type(new_dict['tuple_of_mixed_type']))
        self.assertEqual(test_dict['tuple_of_mixed_type'][0],
                         new_dict['tuple_of_mixed_type'][0])
        self.assertEqual(test_dict['tuple_of_mixed_type'][2],
                         new_dict['tuple_of_mixed_type'][2])

        self.assertEqual(test_dict['some_np_bool'],
                         new_dict['some_np_bool'])
        self.assertEqual(test_dict['some_int'], new_dict['some_int'])
        self.assertEqual(test_dict['some_np_float'], new_dict['some_np_float'])
        self.assertEqual(test_dict['a list of strings'],
                         new_dict['a list of strings'])
        self.assertEqual(test_dict['a list of strings'][0],
                         new_dict['a list of strings'][0])

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
