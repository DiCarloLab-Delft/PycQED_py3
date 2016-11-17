import unittest
import numpy as np
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
import pycqed.measurement.detector_functions as det

from qcodes import station
station = station.Station()


class Test_MeasurementControl(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.MC = measurement_control.MeasurementControl(
            'MC', live_plot_enabled=False, verbose=False)
        self.MC.station = station
        station.add_component(self.MC)

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
        self.MC.set_detector_function(det.Dummy_Detector_Hard(noise=.5))
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

    def tearDown(self):
        self.MC.close()
        del station.components['MC']
