import unittest
import os
import pycqed as pq


import pycqed.instrument_drivers.meta_instrument.distortions_corrector as dc
import pycqed.instrument_drivers.meta_instrument.kernel_object as ko


class Test_distorter(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.kernel_object = ko.DistortionKernel('kernel_object')
        self.dist_corr = dc.Dummy_distortion_corrector(self.kernel_object)
        test_datadir = os.path.join(pq.__path__[0], 'tests', 'test_output')
        self.kernel_object.kernel_dir(test_datadir)

    def test_measure_trace(self):
        self.dist_corr.measure_trace()
        self.dist_corr.plot_trace()

    def test_static_loop(self):
        # Setting some mock kernel that is used to generate the fake pulse
        self.kernel_object.corrections_length(50e-6)
        self.kernel_object.decay_length_1(30e-6)
        self.kernel_object.decay_tau_1(20e-6)
        self.kernel_object.decay_amp_1(.1)

        # This mocks calling all the methods from the interactive loop
        self.dist_corr.open_new_correction(8e-6, AWG_sampling_rate=1e9,
                                           name='Test_kernel_corr')

        self.dist_corr.measure_trace()
        self.dist_corr.plot_trace()
        self.dist_corr.fit_exp_model(30e-9, 5e-6)
        self.dist_corr.plot_fit()
        self.dist_corr.save_plot('fit_{}.png'.format(0))
        self.dist_corr.test_new_kernel()
        self.dist_corr.measure_trace()
        self.dist_corr.plot_trace()
        self.dist_corr.apply_new_kernel()
    # def test_interactive_loop(self):
    #     # FIXME: need some fancy mocking for this
    #     self.dist_corr.interactive_loop()
