from qcodes.instrument.base import Instrument


class VNA_Resonator_Node(Instrument):

    def __init__(self, name, freq=None, BW=1e3, span=10e6, power=-20, **kw):
        super().__init__(name, **kw)
        self.name = name
        self.freq = freq
        self.BW = BW
        self.span = span
        self.power = power

    def frequency_measurement(self):
        if self.nbr_points is None:
            self.nbr_points = int(1 + self.span / 1000)
        VNA_fct.acquire_single_linear_frequency_span(self.name,
                                                     center_freq=self.freq,
                                                     span=self.span,
                                                     nbr_points=self.nbr_points,
                                                     power=self.power,
                                                     bandwidth=self.BW)
        ma_obj = ma.Homodyne_Analysis(
            auto=True, label=self.name, fitting_model='hanger')
        self.freq = int(ma_obj.fit_results.best_values['f0']*1e9)
        return self.freq

    def power_measurement(self):
        if self.nbr_points is None:
            self.nbr_points = int(1 + self.span / 1000)
        VNA_fct.acquire_linear_frequency_span_vs_power('Test_2D',
                                                       center_freq=self.freq,
                                                       span=self.span,
                                                       nbr_points=self.nbr_points,
                                                       start_power=self.start_power,
                                                       stop_power=self.stop_power+self.step_power/2.,
                                                       step_power=self.step_power,
                                                       bandwidth=self.BW)
        return True

    def dac_measurement(self):
        if self.nbr_points is None:
            self.nbr_points = int(1 + self.span / 1000)
        VNA_fct.acquire_2D_linear_frequency_span_vs_param('Test_2D',
                                                          center_freq=self.freq,
                                                          span=self.span,
                                                          nbr_points=self.nbr_points,
                                                          parameter=self.self_dac,
                                                          sweep_vector=self.dac_range,
                                                          power=self.power,
                                                          bandwidth=self.BW)
        # parameter.set(0)
        return True

    def multi_dac_measurement(self):
        for dac in self.list_dacs:
            self.self_dac = dac
            self.dac_measurement()
