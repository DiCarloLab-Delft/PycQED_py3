'''
Currently empty should contain the plotting tools portion of the
analysis toolbox
'''
import numpy as np


def annotate_point_pair(ax, text, xy_start, xy_end, xycoords='data',
                        text_offset=(-10, -5), arrowprops=None, **kw):
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


import numpy as np
import matplotlib.pyplot as plt
import colorsys as colors


def get_color_order(i, max_num):
    # take a blue to red scale from 0 to max_num
    # uses HSV system, H_red = 0, H_green = 1/3 H_blue=2/3
    return colors.hsv_to_rgb(2.*float(i)/(float(max_num)*3.), 1., 1.)


def get_color_from_cmap(i, max_num):
    pass


def flex_color_plot_vs_x(xvals, yvals, zvals, ax=None,
                         xwidth=None,
                         normalize=False, log=False,
                         save_name=None,
                         cmap='viridis',
                         clim=[None, None],
                         alpha=1,
                         **kw):
    """
    Display a color figure for something like a tracked DAC sweep.
    xvals should be a single vector with values for the primary sweep.
    yvals and zvals should be a list of arrays with the sweep points and
    measured values.
    """
    # create a figure and set of axes
    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

    # calculate coordinates for corners of color blocks
    # x coordinates
    if xwidth is None:
        xvals = np.array(xvals)
        xvertices = np.zeros(np.array(xvals.shape)+1)
        xvertices[1:-1] = (xvals[:-1]+xvals[1:])/2.
        xvertices[0] = xvals[0] - (xvals[1]-xvals[0])/2
        xvertices[-1] = xvals[-1] + (xvals[-1]-xvals[-2])/2
    else:
        xvertices = []
        for xval in xvals:
            xvertices.append(xval+np.array([-0.5, 0.5])*xwidth)
    # y coordinates
    yvertices = []
    for xx in range(len(xvals)):
        yvertices.append(np.zeros(np.array(yvals[xx].shape)+1))
        yvertices[xx][1:-1] = (yvals[xx][:-1]+yvals[xx][1:])/2.
        yvertices[xx][0] = yvals[xx][0] - (yvals[xx][1]-yvals[xx][0])/2
        yvertices[xx][-1] = yvals[xx][-1] + (yvals[xx][-1]-yvals[xx][-2])/2

    # normalized plot
    if normalize:
        for xx in range(len(xvals)):
            zvals[xx] /= np.mean(zvals[xx])
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx])/np.log(10)

    # add blocks to plot
    colormaps = []
    for xx in range(len(xvals)):
        tempzvals = np.array(
            [np.append(zvals[xx], np.array(0)),
             np.append(zvals[xx], np.array(0))]).transpose()
        if xwidth is None:
            colormaps.append(ax.pcolor(xvertices[xx:xx+2],
                                       yvertices[xx],
                                       tempzvals,
                                       cmap=cmap, vmin=clim[0], vmax=clim[1],
                                       alpha=alpha))
        else:
            colormaps.append(
                ax.pcolor(xvertices[xx], yvertices[xx], tempzvals, cmap=cmap,
                          alpha=alpha))
        #XX, YY = np.meshgrid(xvertices[xx:xx+2],yvertices[xx])
        # ax1.pcolor(XX,YY,tempzvals,cmap=cmap)

    return {'fig': ax.figure, 'ax': ax,
            'cmap': colormaps[0], 'cmaps': colormaps}


def flex_colormesh_plot_vs_xy(xvals, yvals, zvals, ax=None,
                              normalize=False, log=False,
                              save_name=None, **kw):
    """
    Add a rectangular block to a color plot using pcolormesh.
    xvals and yvals should be single vectors with values for the two sweep points.
    zvals should be a list of arrays with the measured values with shape (len(yvals), len(xvals)).
    """
    # create a figure and set of axes
    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

    # convert xvals and yvals to single dimension arrays
    xvals = np.squeeze(np.array(xvals))
    yvals = np.squeeze(np.array(yvals))

    # calculate coordinates for corners of color blocks
    # x coordinates
    xvertices = np.zeros(np.array(xvals.shape)+1)
    xvertices[1:-1] = (xvals[:-1]+xvals[1:])/2.
    xvertices[0] = xvals[0] - (xvals[1]-xvals[0])/2
    xvertices[-1] = xvals[-1] + (xvals[-1]-xvals[-2])/2
    # y coordinates
    yvertices = np.zeros(np.array(yvals.shape)+1)
    yvertices[1:-1] = (yvals[:-1]+yvals[1:])/2.
    yvertices[0] = yvals[0] - (yvals[1]-yvals[0])/2
    yvertices[-1] = yvals[-1] + (yvals[-1]-yvals[-2])/2

    xgrid, ygrid = np.meshgrid(xvertices, yvertices)

    # various plot options
    # define colormap
    cmap = plt.get_cmap(kw.pop('cmap', 'CMRmap'))
    clim = kw.pop('clim', [None, None])
    # normalized plot
    if normalize:
        zvals /= np.mean(zvals, axis=0)
# logarithmic plot
    if log:

        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx])/np.log(10)

    alpha = kw.pop('alpha', 1)

    # add blocks to plot
    # hold = kw.pop('hold',False)
    do_transpose = kw.pop('transpose', False)
    if do_transpose:
        colormap = ax.pcolormesh(ygrid.transpose(),
                                 xgrid.transpose(),
                                 zvals.transpose(),
                                 cmap=cmap, vmin=clim[0], vmax=clim[1])
    else:
        colormap = ax.pcolormesh(xgrid, ygrid, zvals, cmap=cmap,
                                 vmin=clim[0], vmax=clim[1])

    return {'fig': ax.figure, 'ax': ax, 'cmap': colormap}
