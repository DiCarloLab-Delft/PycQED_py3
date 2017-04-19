'''
Currently empty should contain the plotting tools portion of the
analysis toolbox
'''
import numpy as np
from pyqtgraph.units import SI_PREFIXES, UNITS as SI_UNITS


def set_xlabel(axis, label, unit=None, **kw):
    """
    Takes in an axis object and add a unit aware label to it.

    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_xlabel

    """
    xticks = axis.get_xticks()
    scale_factor, unit = SI_prefix_and_scale_factor(
        val=max(abs(xticks)), unit=unit)
    axis.set_xticklabels(xticks*scale_factor)
    axis.set_xlabel(label+' ({})'.format(unit), **kw)
    return axis


def set_ylabel(axis, label, unit=None, **kw):
    """
    Takes in an axis object and add a unit aware label to it.

    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_ylabel

    """
    yticks = axis.get_yticks()
    scale_factor, unit = SI_prefix_and_scale_factor(
        val=max(abs(yticks)), unit=unit)
    axis.set_yticklabels(yticks*scale_factor)
    axis.set_ylabel(label+' ({})'.format(unit), **kw)
    return axis


def SI_prefix_and_scale_factor(val, unit=None):
    """
    Takes in a value and unit and if applicable returns the proper SI prefix
    and prefix power
    """
    validtypes = (float, int, np.integer, np.floating)
    if unit in SI_UNITS and isinstance(val, validtypes):
        if val == 0:
            prefix_power = 0
        else:
            # The defined prefixes go down to -24 but this is below
            # the numerical precision of python
            prefix_power = np.clip(-15, (np.log10(abs(val))//3 * 3), 24)
        # Determine SI prefix, number 8 corresponds to no prefix
        SI_prefix_idx = int(prefix_power/3 + 8)
        prefix = SI_PREFIXES[SI_prefix_idx]
        # Convert the unit
        scale_factor = 10**-prefix_power
        unit = prefix+unit
    else:
        scale_factor = 1

    return scale_factor, unit


def annotate_point_pair(ax, text, xy_start, xy_end, xycoords='data',
                        text_offset=(-10, -5), arrowprops = None, **kw):
    '''
    Annotates two points by connecting them with an arrow.
    The annotation text is placed near the center of the arrow.

    Function copied from "http://stackoverflow.com/questions/14612637/
    plotting-distance-arrows-in-technical-drawing/32522399#32522399"
    Modified by Adriaan to allows specifying offset of text in two directions.
    '''

    if arrowprops is None:
        arrowprops = dict(arrowstyle='<->')

    assert isinstance(text, str)

    xy_text = ((xy_start[0] + xy_end[0])/2., (xy_start[1] + xy_end[1])/2.)
    arrow_vector = xy_end[0]-xy_start[0] + (xy_end[1] - xy_start[1]) * 1j
    arrow_angle = np.angle(arrow_vector)
    text_angle = arrow_angle - 0.5*np.pi

    ax.annotate(
            '', xy=xy_end, xycoords=xycoords,
            xytext=xy_start, textcoords=xycoords,
            arrowprops=arrowprops, **kw)

    label = ax.annotate(
        text,
        xy=xy_text,
        xycoords=xycoords,
        xytext=(text_offset[0] * np.cos(text_angle) +
                text_offset[1] * np.sin(text_angle),
                text_offset[0] * np.sin(text_angle) +
                text_offset[1] * np.cos(text_angle)),
        textcoords='offset points', **kw)
    return label
