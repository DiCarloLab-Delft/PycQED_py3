import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis.fitting_models import LorentzFunc
import time
from pycqed.analysis import fitting_models as fm


class DummyParHolder(Instrument):
    """
    Holds dummy parameters which are get and set able as well as provides
    some basic functions that depends on these parameters for testing
    purposes.

    Located in physical instruments because it mimics a instrument that
    talks directly to the hardware.
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        for parname in ["x", "y", "z", "x0", "y0", "z0"]:
            self.add_parameter(
                parname,
                unit="m",
                parameter_class=ManualParameter,
                vals=vals.Numbers(),
                initial_value=0.,
            )

        # Instrument integer parameters
        for parname in ["x_int", "y_int", "z_int", "x0_int", "y0_int", "z0_int"]:
            self.add_parameter(
                parname,
                unit="m",
                parameter_class=ManualParameter,
                vals=vals.Ints(),
                initial_value=0,
            )

        self.add_parameter(
            "noise",
            unit="V",
            label="white noise amplitude",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter(
            "delay",
            unit="s",
            label="Sampling delay",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter("parabola", unit="V", get_cmd=self._measure_parabola)

        self.add_parameter("parabola_int", unit="V", get_cmd=self._measure_parabola_int)

        self.add_parameter("parabola_float_int", unit="V", get_cmd=self._measure_parabola_float_int)

        self.add_parameter(
            "parabola_list", unit="V", get_cmd=self._measure_parabola_list
        )

        self.add_parameter(
            "skewed_parabola", unit="V", get_cmd=self._measure_skewed_parabola
        )
        self.add_parameter(
            "cos_mod_parabola", unit="V", get_cmd=self._measure_cos_mod_parabola
        )

        self.add_parameter("lorentz_dip", unit="V", get_cmd=self._measure_lorentz_dip)

        self.add_parameter(
            "lorentz_dip_cos_mod", unit="V", get_cmd=self._measure_lorentz_dip_cos_mod
        )

        self.add_parameter(
            "array_like",
            unit="a.u.",
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
        )

        self.add_parameter(
            "nested_lists_like",
            unit="a.u.",
            parameter_class=ManualParameter,
            vals=vals.Lists(elt_validator=vals.Lists())
        )

        self.add_parameter(
            "dict_like", unit="a.u.", parameter_class=ManualParameter, vals=vals.Dict()
        )

        self.add_parameter(
            "complex_like", unit="a.u.", parameter_class=ManualParameter, vals=vals.ComplexNumbers()
        )

        self.add_parameter(
            "status", vals=vals.Anything(), parameter_class=ManualParameter
        )

    def get_idn(self):
        return "dummy"

    def _measure_lorentz_dip(self):
        time.sleep(self.delay())
        y0 = LorentzFunc(self.x(), -1, center=self.x0(), sigma=5)
        y1 = LorentzFunc(self.y(), -1, center=self.y0(), sigma=5)
        y2 = LorentzFunc(self.z(), -1, center=self.z0(), sigma=5)

        y = y0 + y1 + y2 + self.noise() * np.random.rand(1)
        return y

    def _measure_lorentz_dip_cos_mod(self):
        time.sleep(self.delay())
        y = self._measure_lorentz_dip()
        cos_val = np.cos(self.x() * 10 + self.y() * 10 + self.z() * 10) / 200
        return y + cos_val

    def _measure_parabola(self):
        time.sleep(self.delay())
        return (
            (self.x() - self.x0()) ** 2
            + (self.y() - self.y0()) ** 2
            + (self.z() - self.z0()) ** 2
            + self.noise() * np.random.rand(1)
        )

    def _measure_parabola_int(self):
        time.sleep(self.delay())
        return (
            (self.x_int() - self.x0_int()) ** 2
            + (self.y_int() - self.y0_int()) ** 2
            + (self.z_int() - self.z0_int()) ** 2
            + self.noise() * np.random.rand(1)
        )

    def _measure_parabola_float_int(self):
        time.sleep(self.delay())
        return (
            (self.x() - self.x0()) ** 2
            + (self.y() - self.y0()) ** 2
            + (self.z() - self.z0()) ** 2
            + (self.x_int() - self.x0_int()) ** 2
            + (self.y_int() - self.y0_int()) ** 2
            + (self.z_int() - self.z0_int()) ** 2
            + self.noise() * np.random.rand(1)
        )

    def _measure_parabola_list(self):
        # Returns same as measure parabola but then as a list of list
        # This corresponds to a natural format for e.g., the
        # UHFQC single int avg detector.
        # Where the outer list would be lenght 1 (seq of 1 segment)
        # with 1 entry (only one value logged)
        return np.array([self._measure_parabola()])

    def _measure_cos_mod_parabola(self):
        time.sleep(self.delay())
        cos_val = (
            np.cos(self.x() / 10 + self.y() / 10 + self.z() / 10) ** 2
        )  # ensures always larger than 1
        par = self._measure_parabola()
        n = self.noise() * np.random.rand(1)
        return cos_val * par + n + par / 10

    def _measure_skewed_parabola(self):
        """
        Adds a -x term to add a corelation between the parameters.
        """
        time.sleep(self.delay())
        return (self.x() ** 2 + self.y() ** 2 + self.z() ** 2) * (
            1 + abs(self.y() - self.x())
        ) + self.noise() * np.random.rand(1)


class DummyChevronAlignmentParHolder(Instrument):
    """
    Holds dummy parameters which are get and set able as well as provides
    some basic functions that depends on these parameters for testing
    purposes.

    Dedicated specifically for a Chevron Alignment testind and also a
    good example for testing adaptive sampling

    Located in physical instruments because it mimics a instrument that
    talks directly to the hardware.
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        self.add_parameter(
            "t",
            unit="s",
            label="Pulse duration",
            parameter_class=ManualParameter,
            vals=vals.Numbers(0., 500e-6),
            initial_value=10e-9,
        )

        self.add_parameter(
            "amp",
            unit="a.u.",
            label="Square pulse amplitude",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=.180,
        )

        self.add_parameter(
            "amp_center_1",
            unit="a.u.",
            label="Amplitude center of chevron on one left side",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=-.167,
        )

        self.add_parameter(
            "amp_center_2",
            unit="a.u.",
            label="Amplitude center of chevron on one right side",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=+.187,
        )

        self.add_parameter(
            "J2",
            unit="Hz",
            label="Coupling of interacting states",
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e6, 500e6),
            initial_value=12.5e6,
        )

        self.add_parameter(
            "detuning_swt_spt",
            unit="Hz",
            label="Detuning @ swtspt",
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e5, 100e9),
            initial_value=2.0e9,
        )

        self.add_parameter(
            "flux_bias",
            unit="A",
            label="Square pulse amplitude",
            # parameter_class=ManualParameter,
            vals=vals.Numbers(-5e-3, 5e-3),
            initial_value=180e-6,
            set_cmd=self._set_bias_and_center_amps,
        )

        self.add_parameter(
            "noise",
            unit="frac",
            label="Noise amplitude",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0.05,
        )

        self.add_parameter(
            "delay",
            unit="s",
            label="Sampling delay",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter(
            "frac_excited",
            unit="frac",
            vals=vals.Numbers(),
            get_cmd=self._measure_chevron_excited)

        self.add_parameter(
            "frac_ground",
            unit="frac",
            vals=vals.Numbers(),
            get_cmd=self._measure_chevron_ground)

    def get_idn(self):
        return "dummy chevron alignment"

    def _get_noise(self):
        return np.random.uniform(-self.noise(), self.noise())

    def _measure_chevron_excited(self):
        time.sleep(self.delay())

        population = fm.ChevronFunc(
            amp=self.amp(),
            amp_center_1=self.amp_center_1(),
            amp_center_2=self.amp_center_2(),
            J2=self.J2(),
            detuning_swt_spt=self.detuning_swt_spt(),
            t=self.t(),
        )
        population += self._get_noise()
        return population

    def _measure_chevron_ground(self):
        time.sleep(self.delay())

        population = fm.ChevronInvertedFunc(
            amp=self.amp(),
            amp_center_1=self.amp_center_1(),
            amp_center_2=self.amp_center_2(),
            J2=self.J2(),
            detuning_swt_spt=self.detuning_swt_spt(),
            t=self.t(),
        )
        population += self._get_noise()
        return population

    def _set_bias_and_center_amps(self, val):
        """
        Will be usefull for testing the ChevronAlignment analysis
        """
        poly_pos = np.poly1d([71.875, 0.164062])
        poly_neg = np.poly1d([53.125, -0.186563])

        self.amp_center_1(poly_neg(val))
        self.amp_center_2(poly_pos(val))

        return val
