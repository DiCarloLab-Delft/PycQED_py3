import os
import pycqed as pq
import unittest
import numpy as np
import pycqed.analysis.analysis_toolbox as a_tools
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep, None_Sweep_idx
import pycqed.measurement.detector_functions as det
from pycqed.instrument_drivers.physical_instruments.dummy_instruments \
    import DummyParHolder
from pycqed.measurement.optimization import nelder_mead, SPSA
from pycqed.analysis import measurement_analysis as ma
from pycqed.utilities.get_default_datadir import get_default_datadir
from pycqed.measurement.hdf5_data import read_dict_from_hdf5

from qcodes import station


class Test_MeasurementControl(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=True, verbose=True)
        self.MC.station = self.station
        self.station.add_component(self.MC)

        self.mock_parabola = DummyParHolder('mock_parabola')
        self.station.add_component(self.mock_parabola)

    def setUp(self):
        self.MC.soft_avg(1)

    def test_soft_sweep_1D(self):

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        dat = self.MC.run('1D_soft')
        dset = dat["dset"]
        x = dset[:, 0]
        xr = np.arange(len(x))/15
        y = np.array([np.sin(xr/np.pi), np.cos(xr/np.pi)])
        y0 = dset[:, 1]
        y1 = dset[:, 2]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y[0, :])
        np.testing.assert_array_almost_equal(y1, y[1, :])

        # Test that the return dictionary has the right entries
        dat_keys = set(['dset', 'sweep_parameter_names',
                        'sweep_parameter_units',
                        'value_names', 'value_units'])
        self.assertEqual(dat_keys, set(dat.keys()))

        self.assertEqual(dat['sweep_parameter_names'], ['pts'])
        self.assertEqual(dat['sweep_parameter_units'], ['arb. unit'])
        self.assertEqual(dat['value_names'], ['I', 'Q'])
        self.assertEqual(dat['value_units'], ['mV', 'mV'])

    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.")
    def test_data_location(self):
        sweep_pts = np.linspace(0, 10, 30)
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        self.MC.run('datadir_test_file')
        # raises an error if the file is not found
        ma.MeasurementAnalysis(label='datadir_test_file')

        # change the datadir
        test_dir2 = os.path.abspath(os.path.join(
            os.path.dirname(pq.__file__), os.pardir, 'data_test_2'))
        self.MC.datadir(test_dir2)

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        self.MC.run('datadir_test_file_2')
        # raises an error if the file is not found
        with self.assertRaises(Exception):
            ma.MeasurementAnalysis(label='datadir_test_file_2')
        ma.a_tools.datadir = test_dir2
        # changing the dir makes it find the file now
        ma.MeasurementAnalysis(label='datadir_test_file_2')
        self.MC.datadir(get_default_datadir())

    def test_hard_sweep_1D(self):
        sweep_pts = np.linspace(0, 10, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Hard())
        dat = self.MC.run('1D_hard')
        dset = dat['dset']
        x = dset[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        y0 = dset[:, 1]
        y1 = dset[:, 2]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y[0])
        np.testing.assert_array_almost_equal(y1, y[1])
        d = self.MC.detector_function
        self.assertEqual(d.times_called, 1)

    def test_soft_sweep_2D(self):
        sweep_pts = np.linspace(0, 10, 30)
        sweep_pts_2D = np.linspace(0, 10, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        dat = self.MC.run('2D_soft', mode='2D')
        dset = dat["dset"]
        x = dset[:, 0]
        y = dset[:, 1]
        xr = np.arange(len(sweep_pts)*len(sweep_pts_2D))/15
        z = np.array([np.sin(xr/np.pi), np.cos(xr/np.pi)])
        z0 = dset[:, 2]
        z1 = dset[:, 3]

        x_tiled = np.tile(sweep_pts, len(sweep_pts_2D))
        y_rep = np.repeat(sweep_pts_2D, len(sweep_pts))
        np.testing.assert_array_almost_equal(x, x_tiled)
        np.testing.assert_array_almost_equal(y, y_rep)
        np.testing.assert_array_almost_equal(z0, z[0, :])
        np.testing.assert_array_almost_equal(z1, z[1, :])

    def test_soft_sweep_2D_function_calls(self):
        sweep_pts = np.arange(0, 30, 1)
        sweep_pts_2D = np.arange(0, 5, 1)
        s1 = None_Sweep_idx(sweep_control='soft')
        s2 = None_Sweep_idx(sweep_control='soft')
        self.MC.set_sweep_function(s1)
        self.MC.set_sweep_function_2D(s2)
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())

        self.assertEqual(s1.num_calls, 0)
        self.assertEqual(s2.num_calls, 0)
        self.MC.run('2D_soft', mode='2D')

        # Test that the 2D scan only gets called 5 times (when it changes)
        # The 1D value always changes and as such should always be called
        self.assertEqual(s1.num_calls, 30*5)
        self.assertEqual(s2.num_calls, 5)

    def test_hard_sweep_2D(self):
        """
        Hard inner loop, soft outer loop
        """
        sweep_pts = np.linspace(10, 20, 3)
        sweep_pts_2D = np.linspace(0, 10, 5)
        self.MC.live_plot_enabled(False)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Hard())
        dat = self.MC.run('2D_hard', mode='2D')
        dset = dat["dset"]
        x = dset[:, 0]
        y = dset[:, 1]
        z = self.data = [np.sin(x / np.pi), np.cos(x/np.pi)]
        z0 = dset[:, 2]
        z1 = dset[:, 3]

        x_tiled = np.tile(sweep_pts, len(sweep_pts_2D))
        y_rep = np.repeat(sweep_pts_2D, len(sweep_pts))
        np.testing.assert_array_almost_equal(x, x_tiled)
        np.testing.assert_array_almost_equal(y, y_rep)
        np.testing.assert_array_almost_equal(z0, z[0])
        np.testing.assert_array_almost_equal(z1, z[1])
        d = self.MC.detector_function
        self.assertEqual(d.times_called, 5)

        self.MC.live_plot_enabled(True)

    def test_many_shots_hard_sweep(self):
        """
        Tests acquiring more than the maximum number of shots for a hard
        detector by setting the number of sweep points high
        """
        sweep_pts = np.arange(50)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Shots_Detector(max_shots=5))
        dat = self.MC.run('man_shots')
        dset = dat["dset"]
        x = dset[:, 0]
        y = dset[:, 1]

        self.assertEqual(np.shape(dset), (len(sweep_pts), 2))
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y, sweep_pts)

        d = self.MC.detector_function
        self.assertEqual(d.times_called, 10)

    def test_soft_averages_hard_sweep_1D(self):
        sweep_pts = np.arange(50)
        self.MC.soft_avg(1)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Hard(noise=.4))
        noisy_dat = self.MC.run('noisy_dat')
        noisy_dset = noisy_dat["dset"]
        x = noisy_dset[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        yn_0 = abs(noisy_dset[:, 1] - y[0])
        yn_1 = abs(noisy_dset[:, 2] - y[1])

        d = self.MC.detector_function
        self.assertEqual(d.times_called, 1)

        self.MC.soft_avg(5000)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(d)
        avg_dat = self.MC.run('averaged_dat')
        avg_dset = avg_dat["dset"]
        yavg_0 = abs(avg_dset[:, 1] - y[0])
        yavg_1 = abs(avg_dset[:, 2] - y[1])

        np.testing.assert_array_almost_equal(x, sweep_pts)
        self.assertGreater(np.mean(yn_0), np.mean(yavg_0))
        self.assertGreater(np.mean(yn_1), np.mean(yavg_1))

        np.testing.assert_array_almost_equal(yavg_0, np.zeros(len(x)),
                                             decimal=2)
        np.testing.assert_array_almost_equal(yavg_1, np.zeros(len(x)),
                                             decimal=2)
        self.assertEqual(d.times_called, 5001)

    def test_soft_averages_hard_sweep_2D(self):
        self.MC.soft_avg(1)
        self.MC.live_plot_enabled(False)
        sweep_pts = np.arange(5)
        sweep_pts_2D = np.linspace(5, 10, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Hard(noise=.2))
        noisy_dat = self.MC.run('2D_hard', mode='2D')
        noisy_dset = noisy_dat["dset"]
        x = noisy_dset[:, 0]
        y = noisy_dset[:, 1]
        z = [np.sin(x / np.pi), np.cos(x/np.pi)]
        z0 = abs(noisy_dset[:, 2] - z[0])
        z1 = abs(noisy_dset[:, 3] - z[1])

        x_tiled = np.tile(sweep_pts, len(sweep_pts_2D))
        y_rep = np.repeat(sweep_pts_2D, len(sweep_pts))
        np.testing.assert_array_almost_equal(x, x_tiled)
        np.testing.assert_array_almost_equal(y, y_rep)

        d = self.MC.detector_function
        self.assertEqual(d.times_called, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.soft_avg(1000)
        avg_dat = self.MC.run('averaged_dat', mode='2D')
        avg_dset = avg_dat["dset"]
        x = avg_dset[:, 0]
        y = avg_dset[:, 1]
        zavg_0 = abs(avg_dset[:, 2] - z[0])
        zavg_1 = abs(avg_dset[:, 3] - z[1])

        np.testing.assert_array_almost_equal(x, x_tiled)
        self.assertGreater(np.mean(z0), np.mean(zavg_0))
        self.assertGreater(np.mean(z1), np.mean(zavg_1))

        np.testing.assert_array_almost_equal(zavg_0, np.zeros(len(x)),
                                             decimal=2)
        np.testing.assert_array_almost_equal(zavg_1, np.zeros(len(x)),
                                             decimal=2)

        self.assertEqual(d.times_called, 5*1000+5)
        self.MC.live_plot_enabled(True)

    def test_soft_sweep_1D_soft_averages(self):
        self.mock_parabola.noise(0)
        self.mock_parabola.x(0)
        self.mock_parabola.y(0)
        self.mock_parabola.z(0)

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.soft_avg(1)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(self.mock_parabola.parabola)
        dat = self.MC.run('1D_soft')
        dset = dat["dset"]
        x = dset[:, 0]
        y_exp = x**2
        y0 = dset[:, 1]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y_exp, decimal=5)

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.soft_avg(10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(self.mock_parabola.parabola)
        dat = self.MC.run('1D_soft')
        dset = dat["dset"]
        x = dset[:, 0]
        y_exp = x**2
        y0 = dset[:, 1]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y_exp, decimal=5)

    def test_adaptive_measurement_nelder_mead(self):
        self.MC.soft_avg(1)
        self.mock_parabola.noise(0)
        self.MC.set_sweep_functions(
            [self.mock_parabola.x, self.mock_parabola.y])
        self.MC.set_adaptive_function_parameters(
            {'adaptive_function': nelder_mead,
             'x0': [-50, -50], 'initial_step': [2.5, 2.5]})
        self.mock_parabola.noise(.5)
        self.MC.set_detector_function(self.mock_parabola.parabola)
        dat = self.MC.run('1D test', mode='adaptive')
        dset = dat["dset"]
        xf, yf, pf = dset[-1]
        self.assertLess(xf, 0.7)
        self.assertLess(yf, 0.7)
        self.assertLess(pf, 0.7)

    def test_adaptive_measurement_SPSA(self):
        self.MC.soft_avg(1)
        self.mock_parabola.noise(0)
        self.MC.set_sweep_functions(
            [self.mock_parabola.x, self.mock_parabola.y])
        self.MC.set_adaptive_function_parameters(
            {'adaptive_function': SPSA,
             'x0': [-50, -50],
             'a': (0.5)*(1+300)**0.602,
             'c': 0.2,
             'alpha': 1.,  # 0.602,
             'gamma': 1./6.,  # 0.101,
             'A': 300,
             'p': 0.5,
             'maxiter': 330})
        self.mock_parabola.noise(.5)
        self.MC.set_detector_function(self.mock_parabola.parabola)
        dat = self.MC.run('1D test', mode='adaptive')
        dset = dat["dset"]
        xf, yf, pf = dset[-1]
        self.assertLess(xf, 0.7)
        self.assertLess(yf, 0.7)
        self.assertLess(pf, 0.7)

    @classmethod
    def tearDownClass(self):
        self.MC.close()
        self.mock_parabola.close()
        del self.station.components['MC']
        del self.station.components['mock_parabola']

    def test_persist_mode(self):
        sweep_pts = np.linspace(0, 10, 5)
        self.MC.persist_mode(True)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Hard())
        dat = self.MC.run('1D_hard')
        dset = dat["dset"]
        x = dset[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        y0 = dset[:, 1]
        y1 = dset[:, 2]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y[0])
        np.testing.assert_array_almost_equal(y1, y[1])
        d = self.MC.detector_function
        self.assertEqual(d.times_called, 1)

        persist_dat = self.MC._persist_dat
        x_p = persist_dat[:, 0]
        y0_p = persist_dat[:, 1]
        y1_p = persist_dat[:, 2]
        np.testing.assert_array_almost_equal(x, x_p)
        np.testing.assert_array_almost_equal(y0, y0_p)
        np.testing.assert_array_almost_equal(y1, y1_p)

        self.MC.clear_persitent_plot()
        self.assertEqual(self.MC._persist_dat, None)

    def test_data_resolution(self):
        # This test will fail if the data is saved as 32 bit floats
        sweep_pts = [3e9+1e-3, 3e9+2e-3]
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        dat = self.MC.run('1D_soft')
        x = dat['dset'][:, 0]
        np.testing.assert_array_almost_equal(x, sweep_pts, decimal=5)

    def test_save_exp_metadata(self):
        metadata_dict = {
            'intParam': 1,
            'floatParam': 2.5e-3,
            'strParam': 'spam',
            'listParam': [1, 2, 3, 4],
            'arrayParam': np.array([4e5, 5e5]),
            'dictParam': {'a': 1, 'b': 2},
            'tupleParam': (3, 'c')
        }

        old_a_tools_datadir = a_tools.datadir
        a_tools.datadir = self.MC.datadir()

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Soft())
        self.MC.run('test_exp_metadata', exp_metadata=metadata_dict)
        a = ma.MeasurementAnalysis(label='test_exp_metadata', auto=False)

        a_tools.datadir = old_a_tools_datadir

        loaded_dict = read_dict_from_hdf5(
            {}, a.data_file['Experimental Data']['Experimental Metadata'])

        np.testing.assert_equal(metadata_dict, loaded_dict)
