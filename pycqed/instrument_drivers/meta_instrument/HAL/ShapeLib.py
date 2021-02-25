import logging
from typing import Sequence, Optional, Dict, Union, Callable, Any, List

from pycqed.measurement.waveform_control_CC import waveform as wf

from qcodes import Instrument
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals



log = logging.getLogger(__name__)




# FIXME: naming; shape/pulse/waveform/...

class Shape(InstrumentBase):
    def __init__(self, name: str):
        super().__init__(name)
        # FIXME: implement

    def wave(self, **kwargs: Any):
        raise RuntimeError("call overridden method")


"""
the QuantumDevice constructs a single object representing all instruments used based on a JSON configuration

    ->

    q*.shape.ef
    q*.shape.ge

"""


class QuantumDevice:
    def __init__(self):
        pass  # FIXME: implement

    def from_JSON(self, json: str):
        pass  # FIXME: implement

    def register_shape(self, shape: Shape):
        pass  # FIXME: implement




"""

Microwave shapes based om mw_lutman
register Shapes using decorator: @QuantumDevice.register_shape


"""

def theta_to_amp(theta: float, amp180: float):
    """
    Convert θ in deg to pulse amplitude based on a reference amp180.

    Note that all angles are mapped onto the domain [-180, 180) so that
    the minimum possible angle for each rotation is used.
    """
    # phase wrapped to [-180, 180)
    theta_wrap = ((-theta + 180) % 360 - 180) * -1
    amp = theta_wrap / 180 * amp180
    return amp


# parameters shared by several pulse types
# FIXME: split according to scope: channel/instrument/setup
#@QuantumDevice.register_shape
class WaveformParameters(Shape):
    def __init__(self, name: str):
        super().__init__(name)

        self.wf_func = wf.mod_gauss
        self.spec_func = wf.block_pulse

        # from
        self.add_parameter(
            'pulse_delay',
            unit='s',
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
            initial_value=0)

        # from Base_LutMan
        # FIXME: instrument property
        self.add_parameter(
            "sampling_rate",
            unit="Hz",
            vals=vals.Numbers(1, 100e10),
            initial_value=1e9,
            parameter_class=ManualParameter,
        )

        # from Base_MW_Lutman
        # FIXME: decide in HAL based on HW capabilities
        self.add_parameter(
            'cfg_sideband_mode',
            vals=vals.Enum('real-time', 'static'),
            initial_value='static',
            parameter_class=ManualParameter)



# GroundExcited
class ge(Shape):
    def __init__(self, name: str):
        super().__init__(name)

        # FIXME: remove "mw_" prefixes
        self.add_parameter(
            'mw_amp180',
            unit='frac',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=1.0)
        self.add_parameter(
            'mw_amp90_scale',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=0.5)
        self.add_parameter(
            'mw_motzoi',
            vals=vals.Numbers(-2, 2),
            parameter_class=ManualParameter,
            initial_value=0.0)
        self.add_parameter(
            'mw_gauss_width',
            vals=vals.Numbers(min_value=1e-9),
            unit='s',
            parameter_class=ManualParameter,
            initial_value=4e-9)
        self.add_parameter(
            'mw_phi',
            unit='deg',
            label='Phase of Rphi pulse',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=0)

    # based on generate_standard_waveforms()
    def wave(self, theta: float, phi: float, wp: WaveformParameters):
        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation  # FIXME
        else:
            f_modulation = 0

        if theta == 90:
            amp = self.mw_amp180 * self.mw_amp90_scale
        elif theta == -90:
            amp = - self.mw_amp180 * self.mw_amp90_scale
        else:
            amp = theta_to_amp(theta=theta, amp180=self.mw_amp180)
        return wp.wf_func(
            amp=amp,
            phase=phi,
            sigma_length=self.mw_gauss_width,
            f_modulation=f_modulation,
            sampling_rate=wp.sampling_rate,
            motzoi=self.mw_motzoi,
            delay=wp.pulse_delay)


# second excited
class ef(Shape):
    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            'mw_ef_modulation',
            unit='Hz',
            vals=vals.Numbers(),
            docstring=('Modulation frequency for driving pulses to the second excited-state.'),
            parameter_class=ManualParameter,
            initial_value=50.0e6)
        self.add_parameter(
            'mw_ef_amp180',
            unit='frac',
            docstring=('Pulse amplitude for pulsing the ef/12 transition'),
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=.2)

    # based on generate_standard_waveforms()
    def wave(self, theta: float, phi: float, wp: WaveformParameters):
        amp = theta_to_amp(theta=theta, amp180=self.mw_ef_amp180)
        return wp.wf_func(
            amp=amp,
            phase=phi,
            sigma_length=self.mw_gauss_width,  # FIXME
            f_modulation=self.mw_ef_modulation,
            sampling_rate=wp.sampling_rate,
            motzoi=0,
            delay=wp.pulse_delay)


"""

Flux shapes based om flux_lutman

"""



"""

Readout shapes based om ro_lutman

"""




