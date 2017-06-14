import unittest
import numpy as np
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep, None_Sweep_idx
import pycqed.measurement.detector_functions as det
from pycqed.instrument_drivers.physical_instruments.dummy_instruments \
    import DummyParHolder
from pycqed.measurement.optimization import nelder_mead, SPSA

from qcodes import station


class Test_MeasurementControl(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.station = station.Station()
        # set up a pulsar with some mock settings for the element
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
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
        x = dat[:, 0]
        xr = np.arange(len(x))/15
        y = np.array([np.sin(xr/np.pi), np.cos(xr/np.pi)])
        y0 = dat[:, 1]
        y1 = dat[:, 2]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y[0, :])
        np.testing.assert_array_almost_equal(y1, y[1, :])

    def test_hard_sweep_1D(self):
        sweep_pts = np.linspace(0, 10, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(det.Dummy_Detector_Hard())
        dat = self.MC.run('1D_hard')
        x = dat[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        y0 = dat[:, 1]
        y1 = dat[:, 2]
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

        x = dat[:, 0]
        y = dat[:, 1]
        xr = np.arange(len(sweep_pts)*len(sweep_pts_2D))/15
        z = np.array([np.sin(xr/np.pi), np.cos(xr/np.pi)])
        z0 = dat[:, 2]
        z1 = dat[:, 3]

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
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Hard())
        dat = self.MC.run('2D_hard', mode='2D')
        x = dat[:, 0]
        y = dat[:, 1]
        z = self.data = [np.sin(x / np.pi), np.cos(x/np.pi)]
        z0 = dat[:, 2]
        z1 = dat[:, 3]

        x_tiled = np.tile(sweep_pts, len(sweep_pts_2D))
        y_rep = np.repeat(sweep_pts_2D, len(sweep_pts))
        np.testing.assert_array_almost_equal(x, x_tiled)
        np.testing.assert_array_almost_equal(y, y_rep)
        np.testing.assert_array_almost_equal(z0, z[0])
        np.testing.assert_array_almost_equal(z1, z[1])
        d = self.MC.detector_function
        self.assertEqual(d.times_called, 5)

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
        x = dat[:, 0]
        y = dat[:, 1]

        self.assertEqual(np.shape(dat), (len(sweep_pts), 2))
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
        x = noisy_dat[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        yn_0 = abs(noisy_dat[:, 1] - y[0])
        yn_1 = abs(noisy_dat[:, 2] - y[1])

        d = self.MC.detector_function
        self.assertEqual(d.times_called, 1)

        self.MC.soft_avg(5000)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(d)
        avg_dat = self.MC.run('averaged_dat')
        yavg_0 = abs(avg_dat[:, 1] - y[0])
        yavg_1 = abs(avg_dat[:, 2] - y[1])

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
        sweep_pts = np.arange(5)
        sweep_pts_2D = np.linspace(5, 10, 5)
        self.MC.set_sweep_function(None_Sweep(sweep_control='hard'))
        self.MC.set_sweep_function_2D(None_Sweep(sweep_control='soft'))
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_sweep_points_2D(sweep_pts_2D)
        self.MC.set_detector_function(det.Dummy_Detector_Hard(noise=.2))
        noisy_dat = self.MC.run('2D_hard', mode='2D')
        x = noisy_dat[:, 0]
        y = noisy_dat[:, 1]
        z = [np.sin(x / np.pi), np.cos(x/np.pi)]
        z0 = abs(noisy_dat[:, 2] - z[0])
        z1 = abs(noisy_dat[:, 3] - z[1])

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
        x = avg_dat[:, 0]
        y = avg_dat[:, 1]
        zavg_0 = abs(avg_dat[:, 2] - z[0])
        zavg_1 = abs(avg_dat[:, 3] - z[1])

        np.testing.assert_array_almost_equal(x, x_tiled)
        self.assertGreater(np.mean(z0), np.mean(zavg_0))
        self.assertGreater(np.mean(z1), np.mean(zavg_1))

        np.testing.assert_array_almost_equal(zavg_0, np.zeros(len(x)),
                                             decimal=2)
        np.testing.assert_array_almost_equal(zavg_1, np.zeros(len(x)),
                                             decimal=2)

        self.assertEqual(d.times_called, 5*1000+5)

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
        x = dat[:, 0]
        y_exp = x**2
        y0 = dat[:, 1]
        np.testing.assert_array_almost_equal(x, sweep_pts)
        np.testing.assert_array_almost_equal(y0, y_exp, decimal=5)

        sweep_pts = np.linspace(0, 10, 30)
        self.MC.soft_avg(10)
        self.MC.set_sweep_function(self.mock_parabola.x)
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(self.mock_parabola.parabola)
        dat = self.MC.run('1D_soft')
        x = dat[:, 0]
        y_exp = x**2
        y0 = dat[:, 1]
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
        xf, yf, pf = dat[-1]
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
        xf, yf, pf = dat[-1]
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
        x = dat[:, 0]
        y = [np.sin(x / np.pi), np.cos(x/np.pi)]
        y0 = dat[:, 1]
        y1 = dat[:, 2]
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
