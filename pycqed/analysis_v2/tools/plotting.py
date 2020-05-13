"""
Contains plotting tools developed after the implementation of analysis v2
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import logging
log = logging.getLogger(__name__)


def scatter_pnts_overlay(
        x,
        y,
        fig=None,
        ax=None,
        transpose=False,
        color='w',
        edgecolors='gray',
        linewidth=0.5,
        marker='.',
        s=None,
        c=None,
        alpha=1,
        setlabel=None,
        cmap=None,
        **kw):
    """
    Adds a scattered overlay of the provided data points
    x, and y are lists.
    Args:
        x (array [shape: n*1]):     x data
        y (array [shape: m*1]):     y data
        fig (Object):
            figure object
    """
    if ax is None:
        fig, ax = plt.subplots()

    if transpose:
        log.debug('Inverting x and y axis for non-interpolated points')
        ax.scatter(y, x, marker=marker,
                   color=color, edgecolors=edgecolors, linewidth=linewidth, s=s,
                   c=c, alpha=alpha, label=setlabel, cmap=cmap)
    else:
        ax.scatter(x, y, marker=marker,
                   color=color, edgecolors=edgecolors, linewidth=linewidth, s=s,
                   c=c, alpha=alpha, label=setlabel, cmap=cmap)

    return fig, ax


def contour_overlay(x, y, z, colormap="viridis",
                    transpose: bool = False,
                    contour_levels: list = [90, 180, 270],
                    vlim: tuple = (0, 360),
                    linestyles: str = 'dashed',
                    cyclic_data: bool = False,
                    return_contours_only: bool = False,
                    ax=None, fig=None, **kw):
    """
    x, and y are lists, z is a matrix with shape (len(x), len(y))
    N.B. The contour overaly suffers from artifacts sometimes

    Args:
        x (array [shape: n*1]):     x data
        y (array [shape: m*1]):     y data
        z (array [shape: n*m]):     z data for the contour
        colormap (matplotlib.colors.Colormap or str): colormap to be used
        vlim (tuple(vmin, vmax)): required for the colormap nomalization and
            for cyclic data
        cyclic_data (bool): when `True` assumes z data is cyclic at the
            boundaries specified by vlim and avoids contour artifacts
        fig (Object):
            figure object
    """
    ax_fig_are_None = ax is None and fig is None

    if ax is None:
        fig, ax = plt.subplots()

    vmin = vlim[0]
    vmax = vlim[-1]

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    linewidth = 2
    fontsize = 'smaller'

    if transpose:
        y_tmp = np.copy(y)
        y = np.copy(x)
        x = y_tmp
        z = np.transpose(z)

    if cyclic_data:
        # Avoid contour plot artifact for cyclic data by removing the
        # data half way to the cyclic boundary
        minz = (vmin + np.min(contour_levels)) / 2
        maxz = (vmax + np.max(contour_levels)) / 2
        z = np.copy(z)  # don't change the original data
        z[(z < minz) | (z > maxz)] = np.nan

    c = ax.contour(x, y, z,
                   levels=contour_levels, linewidths=linewidth, cmap=colormap,
                   norm=norm, linestyles=linestyles)
    ax.clabel(c, fmt='%.1f', inline='True', fontsize=fontsize)

    if return_contours_only:
        return fig, ax
    else:
        contours = c.allsegs
        if ax_fig_are_None:
            fig.clf()
            plt.close(fig)
            del fig
            del ax
        return contours


def annotate_pnts(txt, x, y,
                  textcoords='offset points',
                  ha='center',
                  va='center',
                  xytext=(0, 0),
                  bbox=dict(boxstyle='circle, pad=0.2', fc='white', alpha=0.7),
                  arrowprops=None,
                  transpose=False,
                  fig=None,
                  ax=None,
                  **kw):
    """
    A handy for loop for the ax.annotate

    See fluxing analysis on how it is used
    """
    if ax is None:
        fig, ax = plt.subplots()

    if transpose:
        y_tmp = np.copy(y)
        y = np.copy(x)
        x = y_tmp

    for i, text in enumerate(txt):
        ax.annotate(text,
                    xy=(x[i], y[i]),
                    textcoords=textcoords,
                    ha=ha,
                    va=va,
                    xytext=xytext,
                    bbox=bbox)
    return fig, ax
