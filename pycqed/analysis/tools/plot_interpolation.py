import numpy as np
import logging
from scipy import interpolate


def areas(ip):
    p = ip.tri.points[ip.tri.vertices]
    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]) / 2
    return areas


def scale(points, xy_mean, xy_scale):
    points = np.asarray(points, dtype=float)
    return (points - xy_mean) / xy_scale


def unscale(points, xy_mean, xy_scale):
    points = np.asarray(points, dtype=float)
    return points * xy_scale + xy_mean


def calc_mean_and_scale(x, y):
    x_bounds = np.min(x), np.max(x)
    y_bounds = np.min(y), np.max(y)

    xy_mean = np.mean(x_bounds), np.mean(y_bounds)
    xy_scale = np.ptp(x_bounds), np.ptp(y_bounds)

    return xy_mean, xy_scale


class DegInterpolator:
    """
    """

    def __init__(self, pnts, z, **kw):
        phases = np.deg2rad(z)
        newdata_cos = np.cos(phases)
        newdata_sin = np.sin(phases)

        self.ip_cos = interpolate.LinearNDInterpolator(pnts, newdata_cos, **kw)
        self.ip_sin = interpolate.LinearNDInterpolator(pnts, newdata_sin, **kw)

    def __call__(self, x, y, **kw):

        data_cos = self.ip_cos(x, y, **kw).squeeze()
        data_sin = self.ip_sin(x, y, **kw).squeeze()

        z_out = np.rad2deg(np.arctan2(data_sin, data_cos)) % 360
        return z_out


class HeatmapInterpolator:
    """
    """

    def __init__(
        self, x, y, z, interp_method: str = "linear", rescale: bool = False, **kw
    ):
        """
        Args:
            rescale (bool): Rescales `x` and `y` data to (-0.5, 0.5) range.
            Useful when working small/large scales.
            If `True` you must take the input range into account when interpolating.
        """
        assert {interp_method} <= {"linear", "nearest", "deg"}

        points = np.column_stack((x, y))
        if rescale:
            xy_mean, xy_scale = calc_mean_and_scale(x, y)
            self.xy_mean, self.xy_scale = xy_mean, xy_scale

            scaled_pnts = scale(points, xy_mean=xy_mean, xy_scale=xy_scale)
            del points
        else:
            scaled_pnts = points

        if interp_method == "linear":
            ip = interpolate.LinearNDInterpolator(scaled_pnts, z, **kw)
        elif interp_method == "nearest":
            ip = interpolate.NearestNDInterpolator(scaled_pnts, z, **kw)
        elif interp_method == "deg":
            ip = DegInterpolator(scaled_pnts, z, **kw)

        self.rescale = rescale
        self.ip = ip

    def __call__(self, x, y, **kw):

        z_out = self.ip(x, y, **kw)
        return z_out.squeeze()

    def unscale(self, pnts):
        """
        For convenience, when using `rescale=True` this can be used to unscale points
        """
        return unscale(pnts, xy_mean=self.xy_mean, xy_scale=self.xy_scale)

    def scale(self, pnts):
        """
        For convenience, when using `rescale=True` this can be used to scale points
        """
        return scale(pnts, xy_mean=self.xy_mean, xy_scale=self.xy_scale)


def interpolate_heatmap(
    x,
    y,
    z=None,
    ip=None,
    n: int = None,
    interp_method: str = "linear",
    interp_grid_data: bool = True,
):
    """
    Args:
        x   (array): x data points
        y   (array): y data points
        z   (array): z data points, not used if `ip` provided
        ip  (HeatmapInterpolator): can be specified to avoid generating new
        interpolator, e.g. use same interpolator to plot quantities along contour
        n     (int): number of points for each dimension on the interpolated
            grid
        interp_method {"linear", "nearest", "deg"} determines what interpolation
            method is used.
        detect_grid (bool): Will make a few simple checks and not interpolate
            the data is already on a grid. This is convenient to be able to use
            same analysis

    Returns:
        x_grid : N*1 array of x-values of the interpolated grid
        y_grid : N*1 array of x-values of the interpolated grid
        z_grid : N*N array of z-values that form a grid.


    The output of this method can directly be used for
        plt.imshow(z_grid, extent=extent, aspect='auto')
        where the extent is determined by the min and max of the x_grid and
        y_grid.

    The output can also be used as input for
        ax.pcolormesh(x, y, Z,**kw)

    """

    # interpolation needs to happen on a rescaled grid, this is somewhat akin to an
    # assumption in the interpolation that the scale of the experiment is chosen sensibly.
    # N.B. even if interp_method == "nearest" the linear interpolation is used
    # to determine the amount of grid points. Could be improved.

    if n is None:
        points = np.column_stack((x, y))
        xy_mean, xy_scale = calc_mean_and_scale(x, y)
        scaled_pnts = scale(points, xy_mean=xy_mean, xy_scale=xy_scale)
        ip_for_areas = interpolate.LinearNDInterpolator(scaled_pnts, z)
        # Calculate how many grid points are needed.
        # factor from A=√3/4 * a² (equilateral triangle)
        all_areas = areas(ip_for_areas)
        area_min = all_areas[all_areas > 0.0].min()
        # N.B. a factor 4 was added as there were to few points for uniform
        # grid otherwise.
        n = int(0.658 / np.sqrt(area_min)) * 4
        n = max(n, 10)
        if n > 500:
            logging.debug("n: {} larger than 500. Clipped to 500.".format(n))
            n = 500

    unique_xs = np.unique(x)
    num_unique_xs = len(unique_xs)
    unique_ys = np.unique(y)
    num_unique_ys = len(unique_ys)

    if num_unique_xs * num_unique_ys == len(x) and not interp_grid_data:
        # Data is already on a grid, don't create larger interpolation grid
        x_lin = np.linspace(-0.5, 0.5, num_unique_xs)
        y_lin = np.linspace(-0.5, 0.5, num_unique_ys)
    else:
        x_lin = y_lin = np.linspace(-0.5, 0.5, n)

    if z is not None and ip is None:
        ip = HeatmapInterpolator(x, y, z, interp_method=interp_method, rescale=True)
    elif z is None and ip is None:
        raise ValueError("`z` values or an `ip` (interpolation object) must be provided!")

    x_reshaped = x_lin[:, None]
    y_reshaped = y_lin[None, :]
    z_grid = ip(x_reshaped, y_reshaped)

    # x and y grid points need to be rescaled from the linearly chosen points
    x_grid = np.array([x_lin, np.full(len(x_lin), np.min(y_lin))]).T
    x_grid = ip.unscale(x_grid).T[0]

    y_grid = np.array([np.full(len(y_lin), np.min(x_lin)), y_lin]).T
    y_grid = ip.unscale(y_grid).T[1]

    return x_grid, y_grid, z_grid.T
