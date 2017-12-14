import unittest
import numpy as np
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
import pycqed.measurement.detector_functions as det
from pycqed.instrument_drivers.physical_instruments.dummy_instruments \
    import DummyParHolder

from qcodes import station


class Test_Detectors(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        self.mock_parabola = DummyParHolder('mock_parabola')
        self.station.add_component(self.mock_parabola)

    def test_function_detector_simple(self):

        def dummy_function(val_a, val_b):
            return val_a
        # Testing input of a simple dict
        d = det.Function_Detector(dummy_function, value_names=['a'],
                                  value_units=None,
                                  msmt_kw={'val_a': 5.5, 'val_b': 1})
        self.assertEqual(d.value_names, ['a'])
        self.assertEqual(d.value_units, ['a.u.'])

        self.MC.set_sweep_function(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(np.linspace(0, 10, 10))
        self.MC.set_detector_function(d)
        dat = self.MC.run()
        dset = dat["dset"]
        np.testing.assert_array_almost_equal(np.ones(10)*5.5, dset[:, 1])

    def test_function_detector_parameter(self):

        def dummy_function(val_a, val_b):
            return val_a+val_b
        # Testing input of a simple dict

        x = self.mock_parabola.x
        d = det.Function_Detector(dummy_function, value_names=['xvals+1'],
                                  value_units=['s'],
                                  msmt_kw={'val_a': x, 'val_b': 1})
        self.assertEqual(d.value_names, ['xvals+1'])
        self.assertEqual(d.value_units, ['s'])

        xvals = np.linspace(0, 10, 10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(xvals)
        self.MC.set_detector_function(d)
        dat = self.MC.run()
        dset = dat["dset"]
        np.testing.assert_array_almost_equal(xvals+1, dset[:, 1])

    def test_function_detector_dict_all_keys(self):

        def dummy_function(val_a, val_b):
            return {'a': val_a, 'b': val_b}
        # Testing input of a simple dict
        d = det.Function_Detector(dummy_function, value_names=['aa', 'b'],
                                  result_keys=['a', 'b'],
                                  value_units=['s', 's'],
                                  msmt_kw={'val_a': 5.5, 'val_b': 1})

        xvals = np.linspace(0, 10, 10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(xvals)
        self.MC.set_detector_function(d)
        dat = self.MC.run()
        dset = dat["dset"]
        np.testing.assert_array_almost_equal(np.ones(10)*5.5, dset[:, 1])
        np.testing.assert_array_almost_equal(np.ones(10)*1, dset[:, 2])
        self.assertEqual(np.shape(dset), (10, 3))

    def test_function_detector_dict_single_key(self):
        def dummy_function(val_a, val_b):
            return {'a': val_a, 'b': val_b}
        # Testing input of a simple dict
        d = det.Function_Detector(dummy_function, value_names=['aa'],
                                  result_keys=['a'],
                                  value_units=['s'],
                                  msmt_kw={'val_a': 5.5, 'val_b': 1})

        xvals = np.linspace(0, 10, 10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(xvals)
        self.MC.set_detector_function(d)
        dat = self.MC.run()
        dset = dat["dset"]
        np.testing.assert_array_almost_equal(np.ones(10)*5.5, dset[:, 1])
        self.assertEqual(np.shape(dset), (10, 2))

    def test_UHFQC_state_map(self):
        """
        Tests the statemap method of the UHFQC statistics detector
        """
        d = det.UHFQC_statistics_logging_det
        test_statemap = {'00': '00', '01': '10', '10': '10', '11': '00'}
        sm_arr = d.statemap_to_array(test_statemap)
        exp_sm_arr = np.array([0, 2, 2, 0], dtype=np.uint32)
        np.testing.assert_array_equal(sm_arr, exp_sm_arr)

        invalid_sm = {'01': '10'}
        with self.assertRaises(ValueError):
            d.statemap_to_array(invalid_sm)

    def test_multi_det_basics(self):
        def dummy_function_1(val_a, val_b):
            return val_a

        def dummy_function_2(val_a, val_b):
            return val_a + val_b

        # Testing input of a simple dict
        x = self.mock_parabola.x
        d0 = det.Function_Detector(dummy_function_1, value_names=['a'],
                                   value_units=['my_unit'],
                                   msmt_kw={'val_a': x, 'val_b': 1})
        d1 = det.Function_Detector(dummy_function_2, value_names=['b'],
                                   value_units=None,
                                   msmt_kw={'val_a': x, 'val_b': 1})

        dm = det.Multi_Detector([d0, d1], det_idx_suffix=False)
        self.assertEqual(dm.value_names, ['a', 'b'])
        self.assertEqual(dm.value_units, ['my_unit', 'a.u.'])

        dm_suffix = det.Multi_Detector([d0, d1], det_idx_suffix=True)
        self.assertEqual(dm_suffix.value_names, ['a_det0', 'b_det1'])
        self.assertEqual(dm_suffix.value_units,  ['my_unit', 'a.u.'])

        dh = det.Dummy_Detector_Hard()
        with self.assertRaises(ValueError):
            dm = det.Multi_Detector([dh, d0])

    def test_Multi_Detector_soft(self):
        def dummy_function_1(val_a, val_b):
            return val_a, val_b

        def dummy_function_2(val_a, val_b):
            return val_a + val_b

        # Testing input of a simple dict
        x = self.mock_parabola.x
        d0 = det.Function_Detector(dummy_function_1, value_names=['a', 'b'],
                                   value_units=['my_unit', 'a.u.'],
                                   msmt_kw={'val_a': x, 'val_b': 1})
        d1 = det.Function_Detector(dummy_function_2, value_names=['b'],
                                   value_units=None,
                                   msmt_kw={'val_a': x, 'val_b': 1})

        dm = det.Multi_Detector([d0, d1], det_idx_suffix=False)
        self.assertEqual(dm.value_names, ['a', 'b', 'b'])
        self.assertEqual(dm.value_units, ['my_unit', 'a.u.', 'a.u.'])

        dm_suffix = det.Multi_Detector([d0, d1], det_idx_suffix=True)
        self.assertEqual(dm_suffix.value_names, ['a_det0',
                                                 'b_det0', 'b_det1'])
        self.assertEqual(dm_suffix.value_units,  ['my_unit', 'a.u.', 'a.u.'])

        xvals = np.linspace(0, 10, 10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(xvals)
        self.MC.set_detector_function(dm)
        dat = self.MC.run("multi_detector")
        dset = dat["dset"]
        np.testing.assert_array_almost_equal(xvals, dset[:, 1])
        np.testing.assert_array_almost_equal(np.ones(len(xvals)), dset[:, 2])
        np.testing.assert_array_almost_equal(xvals+1, dset[:, 3])

    def test_Multi_Detector_hard(self):
        sweep_pts = np.linspace(0, 10, 5)
        d0 = det.Dummy_Detector_Hard()
        d1 = det.Dummy_Detector_Hard()
        dm = det.Multi_Detector([d0, d1])

        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(dm)
        dat = self.MC.run('Multi_hard')
        dset = dat['dset']
        x = dset[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y[0], dset[:, 1])
        np.testing.assert_array_almost_equal(y[1], dset[:, 2])
        np.testing.assert_array_almost_equal(y[0], dset[:, 3])
        np.testing.assert_array_almost_equal(y[1], dset[:, 4])

    @classmethod
    def tearDownClass(self):
        self.MC.close()
        self.mock_parabola.close()
        del self.station.components['MC']
        del self.station.components['mock_parabola']
