"""
This module contains the functions to make the basic waveforms to construct
a Delft/Amsterdam houses pattern.

This is used in the quantum efficiency paper and as a showcase for the
cryoscope.
"""

import numpy as np

# Amsterdam houses functions


def ams_sc(unitlength: int, ams_sc_base, ams_sc_step):
    """
    staircase shaped house
    """
    ams_sc = ams_sc_base * np.ones(13 * unitlength) + np.concatenate(
        [
            0 * np.ones(unitlength),
            ams_sc_step * np.ones(unitlength),
            2 * ams_sc_step * np.ones(unitlength),
            3 * ams_sc_step * np.ones(unitlength),
            4 * ams_sc_step * np.ones(unitlength),
            5 * ams_sc_step * np.ones(unitlength),
            6 * ams_sc_step * np.ones(unitlength),
            5 * ams_sc_step * np.ones(unitlength),
            4 * ams_sc_step * np.ones(unitlength),
            3 * ams_sc_step * np.ones(unitlength),
            2 * ams_sc_step * np.ones(unitlength),
            ams_sc_step * np.ones(unitlength),
            0.0 * np.ones(unitlength),
        ]
    )
    return ams_sc


def ams_clock(unitlength: int, ams_clock_base, ams_clock_delta):
    ams_clock = ams_clock_base * np.ones(8 * unitlength) + np.concatenate(
        [
            np.linspace(0, ams_clock_delta, unitlength),
            ams_clock_delta * np.ones(6 * unitlength),
            np.linspace(ams_clock_delta, 0, unitlength),
        ]
    )
    return ams_clock


def ams_bottle(unitlength: int, ams_bottle_base, ams_bottle_delta):
    ams_bottle = ams_bottle_base * np.ones(8 * unitlength) + np.concatenate(
        [
            np.linspace(0, ams_bottle_delta, 3 * unitlength) ** 4
            / ams_bottle_delta ** 3,
            ams_bottle_delta * np.ones(2 * unitlength),
            np.linspace(ams_bottle_delta, 0, 3 * unitlength) ** 4
            / ams_bottle_delta ** 3,
        ]
    )
    return ams_bottle


def ams_bottle2(unitlength: int, ams_bottle_base, ams_bottle_delta):
    """
    Quite steep bottle (based on second order polynomial)
    """
    ams_bottle = ams_bottle_base * np.ones(7 * unitlength) + np.concatenate(
        [
            np.linspace(0, ams_bottle_delta, 3 * unitlength) ** 2
            / ams_bottle_delta ** 1,
            ams_bottle_delta * np.ones(1 * unitlength),
            np.linspace(ams_bottle_delta, 0, 3 * unitlength) ** 2
            / ams_bottle_delta ** 1,
        ]
    )
    return ams_bottle


def ams_bottle3(unitlength: int, ams_bottle_base, ams_bottle_delta):
    """
    Normal triangular rooftop
    """
    ams_bottle = ams_bottle_base * np.ones(13 * unitlength) + np.concatenate(
        [
            np.linspace(0, ams_bottle_delta, int(6.5 * unitlength)),
            np.linspace(ams_bottle_delta, 0, int(6.5 * unitlength)),
        ]
    )
    return ams_bottle


def ams_midup(unitlength: int, ams_midup_base, ams_midup_delta):
    ams_midup = ams_midup_base * np.ones(9 * unitlength) + np.concatenate(
        [
            0 * np.ones(3 * unitlength),
            ams_midup_delta * np.ones(3 * unitlength)
            + -0.03
            * np.linspace(-unitlength, unitlength, 3 * unitlength) ** 2
            / unitlength ** 2,
            0 * np.ones(3 * unitlength),
        ]
    )
    return ams_midup
