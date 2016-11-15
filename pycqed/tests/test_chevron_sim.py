import numpy as np
from unittest import TestCase

from pycqed.simulations import chevron_sim as chs


class TestChevronSim(TestCase):

    @classmethod
    def setUpClass(self):
        self.e_min = -0.0322
        self.e_max = 0.0322
        self.e_points = 20
        self.time_stop = 60
        self.time_step = 4
        self.bias_tee = lambda self, t, a, b, c: t*a**2+t*b+c

        # self.distortion = lambda t: self.lowpass_s(t, 2)
        self.distortion = lambda self, t: self.bias_tee(t, 0., 2e-5, 1)

        self.time_vec = np.arange(0., self.time_stop, self.time_step)
        self.freq_vec = np.linspace(self.e_min, self.e_max, self.e_points)

    def test_output_shape(self):
        """
        Trivial test that just checks if there is nothing that broke the
        chevron sims in a way that breaks it.
        """

        result = chs.chevron(2.*np.pi*(6.552 - 4.8),
                             self.e_min, self.e_max,
                             self.e_points,
                             np.pi*0.0385,
                             self.time_stop,
                             self.time_step,
                             self.distortion)
        self.assertEqual(np.shape(result),
                         (len(self.freq_vec), len(self.time_vec)+1))
