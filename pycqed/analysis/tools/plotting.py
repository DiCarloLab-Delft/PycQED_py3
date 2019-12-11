'''
Currently empty should contain the plotting tools portion of the
analysis toolbox
'''
import lmfit
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import matplotlib.colors as col
import hsluv
import logging
from matplotlib.patches import Rectangle, ConnectionPatch

golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
single_col_figsize = (3.39, golden_mean*3.39)
double_col_figsize = (6.9, golden_mean*6.9)
thesis_col_figsize = (12.2/2.54, golden_mean*12.2/2.54)


def set_xlabel(axis, label, unit=None, **kw):
    """
    Add a unit aware x-label to an axis object.

    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_xlabel

    """
    if unit is not None and unit != '':
        xticks = axis.get_xticks()
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(xticks)), unit=unit)
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:.4g}'.format(x*scale_factor))

        axis.xaxis.set_major_formatter(formatter)

        axis.set_xlabel(label+' ({})'.format(unit), **kw)
    else:
        axis.set_xlabel(label, **kw)
    return axis


def set_ylabel(axis, label, unit=None, **kw):
    """
    Add a unit aware y-label to an axis object.

    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_ylabel

    """
    if unit is not None and unit != '':
        yticks = axis.get_yticks()
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(yticks)), unit=unit)
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:.6g}'.format(x*scale_factor))

        axis.yaxis.set_major_formatter(formatter)

        axis.set_ylabel(label+' ({})'.format(unit), **kw)
    else:
        axis.set_ylabel(label, **kw)
    return axis


def set_cbarlabel(cbar, label, unit=None, **kw):
    """
    Add a unit aware z-label to a colorbar object

    Args:
        cbar: colorbar object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to cbar.set_label
    """
    if unit is not None and unit != '':
        zticks = cbar.get_ticks()
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(zticks)), unit=unit)
        cbar.set_ticks(zticks)
        cbar.set_ticklabels(zticks*scale_factor)
        cbar.set_label(label + ' ({})'.format(unit))

    else:
        cbar.set_label(label, **kw)
    return cbar


SI_PREFIXES = dict(zip(range(-24, 25, 3), 'yzafpnμm kMGTPEZY'))
SI_PREFIXES[0] = ""

# N.B. not all of these are SI units, however, all of these support SI prefixes
SI_UNITS = 'm,s,g,W,J,V,A,F,T,Hz,Ohm,S,N,C,px,b,B,K,Bar,Vpeak,Vpp,Vp,Vrms,$\Phi_0$,A/s'.split(
    ',')


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

    if unit in SI_UNITS:
        try:
            with np.errstate(all="ignore"):
                prefix_power = np.log10(abs(val))//3 * 3
                prefix = SI_PREFIXES[prefix_power]
                # Greek symbols not supported in tex
                if plt.rcParams['text.usetex'] and prefix == 'μ':
                    prefix = r'$\mu$'

            return 10 ** -prefix_power,  prefix + unit
        except (KeyError, TypeError):
            pass

    return 1, unit if unit is not None else ""


def SI_val_to_msg_str(val: float, unit: str=None, return_type=str):
    """
    Takes in a value  with optional unit and returns a string tuple consisting
    of (value_str, unit) where the value and unit are rescaled according to
    SI prefixes, IF the unit is an SI unit (according to the comprehensive list
    of SI units in this file ;).

    the value_str is of the type specified in return_type (str) by default.
    """

    sc, new_unit = SI_prefix_and_scale_factor(val, unit)
    try:
        new_val = sc*val
    except TypeError:
        return return_type(val), unit

    return return_type(new_val), new_unit


def format_lmfit_par(par_name: str, lmfit_par, end_char=''):
    """Format an lmfit par to a string of value with uncertainty."""
    val_string = par_name
    val_string += ': {:.4f}'.format(lmfit_par.value)
    if lmfit_par.stderr is not None:
        val_string += r'$\pm$' + '{:.4f}'.format(lmfit_par.stderr)
    else:
        val_string += r'$\pm$' + 'NaN'
    val_string += end_char
    return val_string


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


def plot_lmfit_res(fit_res, ax, plot_init: bool=False,
                   plot_numpoints: int=1000,
                   plot_kw: dict ={}, plot_init_kw: dict = {}, **kw):
    """
    Plot the result of an lmfit optimization.

    Args:
        fit_res:        lmfit result object.
        ax:             matplotlib axis object to plot on.
        plot_init:      if True plots the initial guess of the fit.
        plot_numpoints: number of points to use for interpolating the fit.
        plot_kw:        dictionary of options to pass to the plot of the fit.
        plot_init_kw    dictionary of options to pass to the plot of the
                        initial guess.
        **kw            **kwargs, unused only here to match call signature.

    Return:
        axis :          Returns matplotlib axis object on which the plot
                        was performed.

    """
    if hasattr(fit_res, 'model'):
        model = fit_res.model
        # Testing input
        if not (isinstance(model, lmfit.model.Model) or
                isinstance(model, lmfit.model.ModelResult)):
            raise TypeError(
                'The passed item in "fit_res" needs to be'
                ' a fitting model, but is {}'.format(type(model)))
        if len(model.independent_vars) == 1:
            independent_var = model.independent_vars[0]
        else:
            raise ValueError('Fit can only be plotted if the model function'
                             ' has one independent variable.')

        x_arr = fit_res.userkws[independent_var]
        xvals = np.linspace(np.min(x_arr), np.max(x_arr),
                            plot_numpoints)
        yvals = model.eval(fit_res.params,
                           **{independent_var: xvals})
        if plot_init:
            yvals_init = model.eval(fit_res.init_params,
                                    **{independent_var: xvals})

    else:  # case for the minimizer fit
        # testing input
        fit_xvals = fit_res.userkws
        if len(fit_xvals.keys()) == 1:
            independent_var = list(fit_xvals.keys())[0]
        else:
            raise ValueError('Fit can only be plotted if the model function'
                             ' has one independent variable.')

        x_arr = fit_res.userkws[independent_var]
        xvals = np.linspace(np.min(x_arr), np.max(x_arr),
                            plot_numpoints)
        fit_fn = fit_res.fit_fn
        yvals = fit_fn(**fit_res.params,
                       **{independent_var: xvals})
        if plot_init:
            yvals_init = fit_fn(**fit_res.init_params,
                                **{independent_var: xvals})
    # acutal plotting
    ax.plot(xvals, yvals, **plot_kw)
    if plot_init:
        ax.plot(xvals, yvals_init, **plot_init_kw)
    return ax


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

    **grid-orientation**
        The grid orientation for the zvals is the same as is used in
        ax.pcolormesh.
        Note that the column index corresponds to the x-coordinate,
        and the row index corresponds to y.
        This can be counterintuitive: zvals(y_idx, x_idx)
        and can be inconsistent with some arrays of zvals
        (such as a 2D histogram from numpy).
    """

    xvals = np.array(xvals)
    yvals = np.array(yvals)

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


def set_axeslabel_color(ax, color):
    '''
    Ad hoc function to set the labels, ticks, ticklabels and title to a color.

    This is useful when e.g., making a presentation on a dark background
    '''
    ax.tick_params(color=color, which='both')  # both major and minor ticks
    plt.setp(ax.get_xticklabels(), color=color)
    plt.setp(ax.get_yticklabels(), color=color)
    plt.setp(ax.yaxis.get_label(), color=color)
    plt.setp(ax.xaxis.get_label(), color=color)
    plt.setp(ax.title, color=color)


# generate custom colormaps
def make_segmented_cmap():
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap


def make_anglemap(N=256, use_hpl=True):
    h = np.ones(N)  # hue
    h[:N//2] = 11.6  # red
    h[N//2:] = 258.6  # blue
    s = 100  # saturation
    l = np.linspace(0, 100, N//2)  # luminosity
    l = np.hstack((l, l[::-1]))

    colorlist = np.zeros((N, 3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii, :] = hsluv.hpluv_to_rgb((h[ii], s, l[ii]))
        else:
            colorlist[ii, :] = hsluv.hsluv_to_rgb((h[ii], s, l[ii]))
    colorlist[colorlist > 1] = 1  # correct numeric errors
    colorlist[colorlist < 0] = 0
    return col.ListedColormap(colorlist)


hsluv_anglemap = make_anglemap(use_hpl=False)


def plot_fit(xvals, fit_res, ax, **plot_kws):
    """
    Evaluates a fit result at specified values to plot the fit.
    """
    model = fit_res.model
    independent_var = model.independent_vars[0]
    yvals = model.eval(fit_res.params, **{independent_var: xvals})
    ax.plot(xvals, yvals, **plot_kws)


def cmap_to_alpha(cmap):
    """
    Takes a cmap and makes the transparency of the cmap
    changes with each element.
    """
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = col.ListedColormap(my_cmap)
    return my_cmap


def cmap_first_to_alpha(cmap):
    """
    Makes the first element of a cmap transparant.
    """
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[0, -1] = 0
    my_cmap[1:, -1] = 1

    # Create new colormap
    my_cmap = col.ListedColormap(my_cmap)
    return my_cmap


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:

        fig_height = fig_width*golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              #               'text.fontsize': 8, # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def connected_zoombox(ax0, ins_ax,
                      corner_a=(1, 1), corner_b=(2, 2),
                      square_kws={}, line_kws={}):
    """
    Create a rectangle in ax0 corresponding to the ins_ax and connect corners.

    Parameters
    ----------
    ax0 : matplotlib axis
        The parent axis on which to draw the square and connecting lines.
    ins_ax : matplotlib axis
        The inset axis. The limits of this axis are taken to determine the
        location of the square.
    corner_a : tuple of ints
        Tuple of location codes used to determine what corners to connect.
            'upper right'     1
            'upper left'      2
            'lower left'      3
            'lower right'     4
    """
    x_ins = ins_ax.get_xlim()
    y_ins = ins_ax.get_ylim()

    # xy coordinates corresponding to counterclockwise locations.
    # this order is chosen to be consistent with ax.legend()
    xy1 = (x_ins[1], y_ins[1])  # upper right
    xy2 = (x_ins[0], y_ins[1])  # upper left
    xy3 = (x_ins[0], y_ins[0])  # lower left
    xy4 = (x_ins[1], y_ins[0])  # lower right
    xy_corners = [xy1, xy2, xy3, xy4]

    # ensures we have sensible defaults that can be overwritten
    def_line_kws = dict(
        color='grey',
        arrowstyle='-', zorder=0, lw=1.5, ls=':')
    def_line_kws.update(line_kws)

    conA = ConnectionPatch(xy_corners[corner_a[0]-1],
                           xy_corners[corner_a[1]-1],
                           'data', 'data',
                           axesA=ins_ax, axesB=ax0, **def_line_kws)
    ins_ax.add_artist(conA)

    conB = ConnectionPatch(xy_corners[corner_b[0]-1],
                           xy_corners[corner_b[1]-1],
                           'data', 'data',
                           axesA=ins_ax, axesB=ax0, **def_line_kws)
    ins_ax.add_artist(conB)

    def_sq_kws = dict(ec='k', lw=0.5, fill=0, zorder=4)
    def_sq_kws.update(square_kws)

    rect = Rectangle((x_ins[0], y_ins[0]),
                     x_ins[1]-x_ins[0], y_ins[1]-y_ins[0],
                     **def_sq_kws)
    ax0.add_patch(rect)


def restore_default_plot_params():
    """
    Restore the matplotlib rcParams to their default values
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
