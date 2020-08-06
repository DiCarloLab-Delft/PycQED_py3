from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Validator, Strings

import numpy as np


class NP_NANs(Validator):
    is_numeric = True

    def __init__(self):
        self._valid_values = [np.nan]

    def __repr__(self):
        return "<nan>"

    def validate(self, value, context=""):
        try:
            if not np.isnan(value):
                raise ValueError("{} is not nan; {}".format(repr(value), context))
        except Exception:
            raise ValueError("{} is not nan; {}".format(repr(value), context))


class InstrumentParameter(ManualParameter):
    """
    Args:
        name (string): the name of the instrument that one wants to add.

        instrument (Optional[Instrument]): the "parent" instrument this
            parameter is attached to, if any.

        initial_value (Optional[string]): starting value, the
            only invalid value allowed, and None is only allowed as an initial
            value, it cannot be set later

        **kwargs: Passed to Parameter parent class
    """

    def get_instr(self):
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the instrument_name is the instrument that is the value
        # of this parameter.
        return self._instrument.find_instrument(instrument_name)

    def set_validator(self, vals):
        """
        Set a validator `vals` for this parameter.

        Args:
            vals (Validator):  validator to set
        """
        if vals is None:
            self.vals = Strings()
        elif isinstance(vals, Validator):
            self.vals = vals
        else:
            raise TypeError("vals must be a Validator")


class ConfigParameter(ManualParameter):
    # TODO: move this to qcodes as a pull request

    """
    Define one parameter that reflects a manual configuration setting.

    Args:
        name (string): the local name of this parameter

        instrument (Optional[Instrument]): the instrument this applies to,
            if any.

        initial_value (Optional[string]): starting value, the
            only invalid value allowed, and None is only allowed as an initial
            value, it cannot be set later

        **kwargs: Passed to Parameter parent class
    """

    def __init__(self, name, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._instrument = instrument
        # if the instrument does not have _config_changed attribute creates it
        if not hasattr(self._instrument, "_config_changed"):
            self._instrument._config_changed = True
        self._meta_attrs.extend(["instrument", "initial_value"])

        if initial_value is not None:
            self.validate(initial_value)
            self.cache.set(initial_value)

    def set_raw(self, value):
        """
        Validate and saves value.
        If the value is different from the latest value it sets the
        Args:
            value (any): value to validate and save
        """
        self.validate(value)
        if value != self.get_latest():
            self._instrument._config_changed = True
        self.cache.set(value)

    def get_raw(self):
        """ Return latest value"""
        return self.get_latest()
