import numpy as np

from pycqed.simulations import chevron_sim as chs


class TestChevronSim:

    @classmethod
    def setup_class(cls):
        cls.e_min = -0.0322
        cls.e_max = 0.0322
        cls.e_points = 20
        cls.time_stop = 60
        cls.time_step = 4
        cls.bias_tee = lambda self, t, a, b, c: t * a ** 2 + t * b + c

        # self.distortion = lambda t: self.lowpass_s(t, 2)
        cls.distortion = lambda self, t: self.bias_tee(t, 0., 2e-5, 1)

        cls.time_vec = np.arange(0., cls.time_stop, cls.time_step)
        cls.freq_vec = np.linspace(cls.e_min, cls.e_max, cls.e_points)

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
        assert np.shape(result) == (len(self.freq_vec), len(self.time_vec)+1)
