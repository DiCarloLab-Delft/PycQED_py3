import unittest
import numpy as np

from pycqed.instrument_drivers.meta_instrument import kernel_object as ko

from pycqed.measurement import kernel_functions as kf

from qcodes import station

"""
Kernel TODO's:
 - Rename distortions class
 - path for saving the kernels should not be the notebook directory
 - test saving and loading
 - add calculate only if parameters changed option
 - Way to include RT corrections or any generic file
 - Change parameters to SI units
 - Automatically pick order of distortions to speed
   up convolutions
 - Add shortening of kernel if possible
"""


class Test_KernelObject(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('setting up')
        self.station = station.Station()
        self.k0 = ko.Distortion('k0')
        self.k1 = ko.Distortion('k1')
        self.station.add_component(self.k0)
        self.station.add_component(self.k1)

    def test_skin_kernel(self):
        print('skin_kernel')
        self.k0.skineffect_alpha(0.1)
        self.k0.skineffect_length(40e-9)
        kObj_skin = self.k0.get_skin_kernel()
        kf_skin = kf.skin_kernel(alpha=.1, length=40)
        np.testing.assert_array_equal(kObj_skin, kf_skin)

    def test_bounce_kernel(self):
        print('bounce_kernel')
        bl = 40e-9
        ba = .2
        bt = 12e-9
        self.k0.bounce_amp_1(ba)
        self.k0.bounce_tau_1(bt)
        self.k0.bounce_length_1(bl)
        kObj_bounce = self.k0.get_bounce_kernel_1()
        kf_bounce = kf.bounce_kernel(amp=ba, time=bt*1e9, length=bl*1e9)
        np.testing.assert_array_equal(kObj_bounce, kf_bounce)

    def test_decay_kernel(self):
        dA = 3
        dtau = 15e-9
        dl = 100e-9
        for i in [1, 2]:
            self.k0.set('decay_amp_{}'.format(i), dA)
            self.k0.set('decay_tau_{}'.format(i), dtau)
            self.k0.set('decay_length_{}'.format(i), dl)

        kObj_dec1 = self.k0.get_decay_kernel_1()
        kObj_dec2 = self.k0.get_decay_kernel_1()
        kf_dec = kf.decay_kernel(amp=dA, tau=dtau*1e9, length=dl*1e9)

        np.testing.assert_array_equal(kf_dec, kObj_dec1)
        np.testing.assert_array_equal(kf_dec, kObj_dec2)

    def test_config_changed_flag(self):
        print('config_changed_flag')
        self.k0.decay_amp_1(.9)
        self.assertEqual(self.k0.config_changed(), True)
        self.k0.kernel()
        self.assertEqual(self.k0.config_changed(), False)
        self.k0.decay_amp_1(.9)
        self.assertEqual(self.k0.config_changed(), False)
        self.k0.decay_amp_1(.91)
        self.assertEqual(self.k0.config_changed(), True)

    def test_kernel_loading(self):
        pass
        # FIXME: this preloaded kernel should be added to the repo and the test
        # restored
        # datadir = os.path.join(pq.__path__[0], 'tests', 'test_data',
        #                        'test_kernels')
        # self.k0.kernel_dir(datadir)
        # print(self.k0.kernel_dir())
        # self.k0.kernel_list(['precompiled_RT_20161206.txt'])
        # kernel = self.k0.kernel()

    # def test_convolve_kernels(self):
    #     kernel_list
    #     self.k0.convolve_kernel(kernel_list, length)

    def test_convolve_kernel(self):
        pass

    # def test_kernel_loading(self):
        # self.k0.corrections_length(50)  # ns todo rescale.
        # self.k0.kernel_to_cache()
        # self.k0.get_corrections_kernel()

    # def test_smart_loading(self):
    #     pass


class Test_Kernel_functions(unittest.TestCase):

    def test_bounce_kernel(self):
        pass

    def test_heaviside(self):
        hs = kf.heaviside(np.array([-1, -.5, 0, 1, 2]))
        np.testing.assert_array_equal(hs, [0, 0, 1, 1, 1])

    def test_square(self):
        sq = kf.square(np.arange(-2, 5), 3)
        np.testing.assert_array_equal(sq, [0, 0, 1, 1, 1, 0, 0])

    def test_skin_kernel(self):
        skin_kernel_test = kf.skin_kernel(alpha=.1, length=40)
        known_skin_vals = np.array([
            1.00540222e+00,  -1.59080709e-03,  -7.02241770e-04,
            -4.17894781e-04,  -2.84886822e-04,  -2.10146281e-04,
            -1.63242389e-04,  -1.31535177e-04,  -1.08919606e-04,
            -9.21203433e-05,  -7.92379832e-05,  -6.91027435e-05,
            -6.09587865e-05,  -5.42982090e-05,  -4.87683793e-05,
            -4.41176036e-05,  -4.01619210e-05,  -3.67640800e-05,
            -3.38198160e-05,  -3.12486520e-05,  -2.89875850e-05,
            -2.69866621e-05,  -2.52058216e-05,  -2.36126000e-05,
            -2.21804419e-05,  -2.08874370e-05,  -1.97153637e-05,
            -1.86489578e-05,  -1.76753461e-05,  -1.67836041e-05,
            -1.59644070e-05,  -1.52097526e-05,  -1.45127390e-05,
            -1.38673850e-05,  -1.32684847e-05,  -1.27114874e-05,
            -1.21924004e-05,  -1.17077070e-05,  -1.12542990e-05,
            -1.08294205e-05])
        np.testing.assert_array_almost_equal(
            skin_kernel_test, known_skin_vals, decimal=7)

    def test_poly_kernel(self):
        test_kernel = kf.poly_kernel([0, 0, 1], length=40)
        known_vals = np.zeros(40)
        known_vals[0] = 1

        np.testing.assert_array_almost_equal(
            test_kernel, known_vals, decimal=7)

        test_kernel = kf.poly_kernel([1, 0, 1], length=10)
        known_vals = np.arange(10)*2-1
        known_vals[0] = 1

        np.testing.assert_array_almost_equal(
            test_kernel, known_vals, decimal=7)
