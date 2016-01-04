from instrument import Instrument
import types
import numpy as np

class Bart_parameter_holder(Instrument):

    def __init__(self, name, channels=3):
        Instrument.__init__(self, name, tags=['positioner'])

        # Instrument parameters
        self.add_parameter('x',
            type=int,
            flags=Instrument.FLAG_GET)

        self.add_parameter('y',
            type=float,
            flags=Instrument.FLAG_GET)

    def _do_get_x(self):
        return self.x
    def set_x(self, x):
        self.x = x
    def _do_get_y(self):
        return self.y
    def set_y(self, y):
        self.y = y
    def _do_get_x1(self):
        return self.x1
    def _do_get_x2(self):
        return self.x2
    def _do_get_x3(self):
        return self.x3
    def _do_get_x4(self):
        return self.x4
    def _do_get_x5(self):
        return self.x5
    def _do_get_x6(self):
        return self.x6
    def _do_get_noise(self):
        return self.noise
    def set_x1(self, x1):
        self.x1 = x1
    def set_x2(self, x2):
        self.x2 = x2
    def set_x3(self, x3):
        self.x3 = x3
    def set_x4(self, x4):
        self.x4 = x4
    def set_x5(self, x5):
        self.x5 = x5
    def set_x6(self, x6):
        self.x6 = x6
    def set_noise(self, noise):
        self.noise = noise

    def measure_simple_parabola(self):
        return (self.x)**2

    def measure_noise_parabola(self):
        return (self.x)**2 + self.noise*np.random.rand(1)

    def measure_1D_sinc(self):
        return -np.sinc(self.x)

    def measure_simple_paraboloid(self):
        return (self.x)**2 + (self.y)**2

    def measure_noise_paraboloid(self):
        return (self.x)**2 + (self.y)**2 + self.noise*np.random.rand(1)

    def measure_6_params(self):
        return (self.x1)**2 + (self.x2)**2 + (self.x3)**2 + (self.x4)**2 + (self.x5)**2 + (self.x6)**2

    def measure_parabola_with_sine_modulation(self):
        return (self.x1-1)**2 + (self.y-2)**2 + 3 + np.sin(self.x) #Of eigenlijk die mooie sinc functie die je tekend  #zoekd ie random functie

    def measure_2D_sinc(self):
        return -(np.sinc(self.x+.2)+np.sinc(self.y-.5))+self.noise*np.random.rand(1)

    def measure_2D_gauss(self):
        mu, sigma = 0, 1
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(self.x-mu)**2/(2*sigma**2)) + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(self.y-mu)**2/(2*sigma**2)) + self.noise*np.random.rand(1)

    def lorentzian(self, x, p):
        numerator =  (p[0]**2 )
        denominator = (x-p[1])**2 + p[0]**2
        y = p[2]*(numerator/denominator)
        return y

    def measure_2D_lorentzian(self):
        p = [1,0,100]  # [hwhm, peak center, intensity]
        return -(self.lorentzian(self.x, p) + self.lorentzian(self.y, p) + self.noise*np.random.rand(1))

