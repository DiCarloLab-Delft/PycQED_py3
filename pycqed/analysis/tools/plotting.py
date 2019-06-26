'''
Currently empty should contain the plotting tools portion of the
analysis toolbox
'''
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np

SI_PREFIXES = 'yzafpnμm kMGTPEZY'
SI_UNITS = 'm,s,g,W,J,V,A,F,T,Hz,Ohm,S,N,C,px,b,B,K,Bar,Vpeak,Vpp,Vp,Vrms'.split(',')


def set_axis_label(axis_type, axes, label, unit=None, **kw):
    """
        Takes in an axis object and add a unit aware label to it.

        Args:
            axis_type: a string, 'x', 'y' or 'z' specifying the axis
            axes: matplotlib axes object to set label on
            label: the desired label
            unit:  the unit
            **kw : keyword argument to be passed to matplotlib.set_xlabel

        """
    if axis_type.lower() == 'x':
        get_ticks = axes.get_xticks
        axis = axes.xaxis
        set_label = axes.set_xlabel
    elif axis_type.lower() == 'y':
        get_ticks = axes.get_yticks
        axis = axes.yaxis
        set_label = axes.set_ylabel
    elif axis_type.lower() == 'z':
        get_ticks = axes.get_zticks
        axis = axes.zaxis
        set_label = axes.set_zlabel
    else:
        raise KeyError('No axis named' + str(axis_type))

    if unit is not None and unit != '':
        ticks = get_ticks()

        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(ticks)), unit=unit)

        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: float('{:.4f}'.format(x*scale_factor)))

        axis.set_major_formatter(formatter)

        set_label(label+' ({})'.format(unit), **kw)
    else:
        set_label(label, **kw)
    return axes


def set_xlabel(axes, label, unit=None, **kw):
    set_axis_label('x', axes, label, unit=unit, **kw)


def set_ylabel(axes, label, unit=None, **kw):
    set_axis_label('y', axes, label, unit=unit, **kw)


def set_zlabel(axes, label, unit=None, **kw):
    set_axis_label('z', axes, label, unit=unit, **kw)


def SI_prefix_and_scale_factor(val, unit=None):
    """
    Takes in a value and unit and if applicable returns the proper
    scale factor and SI prefix.
    Args:
        val (float) : the value
        unit (str)  : the unit of the value
    returns:
        scale_factor (float) : scale_factor needed to convert value
        unit (str)           : unit including the prefix
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

    if unit is None:
        unit = ''  # to ensure proper return value

    return scale_factor, unit


def SI_val_to_msg_str(val: float, unit: str=None, return_type=str):
    """
    Takes in a value  with optional unit and returns a string tuple consisting
    of (value_str, unit) where the value and unit are rescaled according to
    SI prefixes.
    the value_str is of the type specified in return_type (str) by default.
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
        val = val*10**-prefix_power
        unit = prefix+unit

    value_str = return_type(val)
    # To ensure right type of return value
    if unit is None:
        unit = ''
    return value_str, unit


def data_to_table_png(data: list, filename: str, title: str='',
                      close_fig: bool=True):
    """
    Takes in a list of list containing the data to be
    put in a table and saves this as a png.
    """
    # Determine the shape of the table
    nrows, ncols = np.shape(data)
    hcell, wcell = 0.3, 2.
    hpad, wpad = 0.5, 0

    fig = plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
    ax = fig.add_subplot(111)
    ax.axis('off')
    # make the table
    table = ax.table(cellText=data,
                     loc='center')
    # rescale to make it more readable
    table.scale(1, 1.5)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=450)
    if close_fig:
        plt.close(fig)


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


def get_color_order(i, max_num, cmap='viridis'):
    # take a blue to red scale from 0 to max_num
    # uses HSV system, H_red = 0, H_green = 1/3 H_blue=2/3
    # return colors.hsv_to_rgb(2.*float(i)/(float(max_num)*3.), 1., 1.)
    print('It is recommended to use the updated function "get_color_cycle".')
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    return cmap((i/max_num) % 1)


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

        dx = abs(np.max(xvals)-np.min(xvals))/len(xvals)
        xvertices[1:-1] = (xvals[:-1]+xvals[1:])/2.
        xvertices[0] = xvals[0] - dx/2
        xvertices[-1] = xvals[-1] + dx/2
    else:
        xvertices = []
        for xval in xvals:
            xvertices.append(xval+np.array([-0.5, 0.5])*xwidth)
    # y coordinates
    yvertices = []
    for xx in range(len(xvals)):
        # Important to sort arguments in case unsorted (e.g., FFT freqs)
        sorted_yarguments = yvals[xx].argsort()
        yvals[xx] = yvals[xx][sorted_yarguments]
        zvals[xx] = zvals[xx][sorted_yarguments]

        yvertices.append(np.zeros(np.array(yvals[xx].shape)+1))
        yvertices[xx][1:-1] = (yvals[xx][:-1]+yvals[xx][1:])/2.
        yvertices[xx][0] = yvals[xx][0] - (yvals[xx][1]-yvals[xx][0])/2
        yvertices[xx][-1] = yvals[xx][-1] + (yvals[xx][-1]-yvals[xx][-2])/2

        # normalized plot
        if normalize:
            zvals[xx] /= np.mean(zvals[xx])
        # logarithmic plot
        if log:
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

    return {'fig': ax.figure, 'ax': ax,
            'cmap': colormaps[0], 'cmaps': colormaps}


def flex_colormesh_plot_vs_xy(xvals, yvals, zvals, ax=None,
                              normalize=False, log=False,
                              save_name=None, **kw):
    """
    Add a rectangular block to a color plot using pcolormesh.
    xvals and yvals should be single vectors with values for the
    two sweep points.
    zvals should be a list of arrays with the measured values with shape
    (len(yvals), len(xvals)).
    """

    xvals = np.array(xvals)
    yvals = np.array(yvals)
    zvals = np.array(zvals)

    # First, we need to sort the data as otherwise we get odd plotting
    # artefacts. An example is e.g., plotting a fourier transform
    sorted_x_arguments = xvals.argsort()
    xvals = xvals[sorted_x_arguments]
    sorted_y_arguments = yvals.argsort()
    yvals = yvals[sorted_y_arguments]
    zvals = zvals[:,  sorted_x_arguments]
    zvals = zvals[sorted_y_arguments, :]

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
    cmap = plt.get_cmap(kw.pop('cmap', 'viridis'))
    clim = kw.pop('clim', [None, None])
    # normalized plot
    if normalize:
        zvals /= np.mean(zvals, axis=0)
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx])/np.log(10)

    # add blocks to plot
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


def autolabel_barplot(ax, rects, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                '%.2f' % (height),
                ha='center', va='bottom', rotation=rotation)
