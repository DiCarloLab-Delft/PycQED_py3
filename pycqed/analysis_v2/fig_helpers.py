import matplotlib
import numpy as np
from matplotlib.patches import Rectangle,ConnectionPatch

golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
single_col_figsize = (3.39, golden_mean*3.39)
double_col_figsize = (6.9, golden_mean*6.9)


def restore_default_plot_params():
    """
    Restores the matplotlib rcParams to their default values
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


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

def filipyfy():
	"""
	Set up matplotlib to how filip likes it
    """
	matplotlib.rcParams['figure.dpi']=300
	matplotlib.rcParams["mathtext.fontset"] = "stixsans"
	matplotlib.rcParams["font.family"] = "sans-serif"
	matplotlib.rcParams["font.sans-serif"] = "Arial"

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
