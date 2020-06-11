"""
Contains tools for manipulation of 2D contours
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import logging

log = logging.getLogger(__name__)


def path_angles_2D(pnts, normalize: bool = None, degrees: bool = True):
    """
    Returns `len(pnts) - 2` angles between consecutive segments.

    Args:
        pnt (array): shape = (len(pnts), 2)
        normalize (bool): set `True` to normalize the `pnts` before calculating
            the angles
        degrees (bool): set `True` to return angles in degrees
    """
    if normalize:
        pnts_T = pnts.T
        min_xy = np.array([np.min(p) for p in pnts_T])
        max_xy = np.array([np.max(p) for p in pnts_T])
        pnts = (pnts - min_xy) / (max_xy - min_xy)

    a_pnts = pnts[:-2]
    b_pnts = pnts[1:-1]
    c_pnts = pnts[2:]

    ba_pnts = a_pnts - b_pnts
    bc_pnts = c_pnts - b_pnts

    angles = (
        (np.arctan2(*bc) - np.arctan2(*ba)) % (2 * np.pi)
        for ba, bc in zip(ba_pnts, bc_pnts)
    )

    if degrees:
        angles = (np.degrees(angle) for angle in angles)

    return angles


def distance_along_2D_contour(
    c_pnts,
    normalize_pnts_before_dist: bool = False,
    normalize_output_dist: bool = False,
):
    """
    Returns the cumulative distance along a 2D contour path

    Args:
        c_pnts (array): shape = (len(pnts), 2)
        normalize_pnts_before_dist (bool): normalizes each dimension (min, max)
            to (0, 1) range for the purpose of calculating the distance along
            the path, heterogeneous scales in each dimensions might give
            unexpected results if not using this option
        normalize_output_dist (bool): if `True` the returned distance will go
            from 0.0 to 1.0

    """

    # Normalize input points before calculating distance to avoid numerical
    # problems when the two dimensions have very different scales
    if normalize_pnts_before_dist:
        c_pnts_T = np.transpose(c_pnts)
        pnts_for_dist = []
        for pnts_dim in c_pnts_T:
            min_val = pnts_dim.min()
            pnts_for_dist.append((pnts_dim - min_val) / (pnts_dim.max() - min_val))
        pnts_for_dist = np.transpose(pnts_for_dist)
    else:
        pnts_for_dist = c_pnts

    # Linear length along the path on the 2D plane
    distance = np.cumsum(np.sqrt(np.sum(np.diff(pnts_for_dist, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)

    # Normalize to [0, 1] range
    if normalize_output_dist:
        distance /= distance[-1]

    return distance


def interp_2D_contour(
    c_pnts,
    interp_method: str = "slinear",
    normalize_pnts_before_dist: bool = True,
    normalize_interp_domain: bool = True,
):
    """
    Returns and `interp1d` along a 2D contour path according to `interp_method`

    Args:
        c_pnts (array): shape = (len(pnts), 2)
        interp_method (str): see `kind` argument of `scipy.interpolate.interp1d`
        normalize_pnts_before_dist (bool): normalizes each dimension (min, max)
            to (0, 1) range for the purpose of calculating the distance along
            the path, heterogeneous scales in each dimensions might give
            unexpected results if not using this option
        normalize_interp_domain (bool): if `True` the returned interpolator
            will accept distance along the path in the (0, 1) range
    """
    assert interp_method in {"slinear", "quadratic", "cubic"}

    distance = distance_along_2D_contour(
        c_pnts, normalize_pnts_before_dist, normalize_interp_domain
    )

    interpolator = interp1d(distance, c_pnts, kind=interp_method, axis=0)

    return interpolator


def interp_pnts_along_2D_contour(
    c_pnts,
    num_pnts: int,
    interp_method: str = "slinear",
    normalize_pnts_before_dist: bool = True,
):
    """
    Returns a list of 2D pnts interpolated along the segments of a 2D
    contour specified by `c_pnts`

    Args:
        c_pnts (array): shape = (len(pnts), 2)
        num_pnts (int): number of equidistant points to be generated
            along the normalized path of the contour
        interp_method (str): see `interp_2D_contour`
        normalize_pnts_before_dist (str): see `interp_2D_contour`
    """
    interp = interp_2D_contour(
        c_pnts, interp_method, normalize_pnts_before_dist=normalize_pnts_before_dist
    )
    pnts = interp(np.linspace(0, 1, num_pnts))
    return pnts


def simplify_2D_path(
    path, angle_thr: float = 3.0, cumulative: bool = True, normalize: bool = True
):
    """
    Removes redundant points along a 2D path according to a threshold angle
    between consecutive segments.
    Consecutive points are assumed to be connected segments.

    Args:
        path (array): shape = (len(path), 2)
        angle_thr (float):  tolerance angle in degrees
            - applied after normalizing points along each dimensions
            - points that deviate from a straight line more than
                `angle_thr` will be included in the output path
        cumulative (bool): if true the `angle_thr` is considered cumulative
            along the path. Gives better results.
        normalize (bool): see `path_angles_2D`

    """
    angles = np.fromiter(path_angles_2D(path, normalize=normalize), dtype=np.float64)
    dif_from_180 = angles - 180

    if cumulative:
        inlc = np.full(len(dif_from_180), False)

        cum_diff = 0
        for i, diff in enumerate(dif_from_180):
            cum_diff += diff
            if np.abs(cum_diff) > angle_thr:
                inlc[i] = True
                cum_diff = 0

        where_incl = np.where(inlc)
    else:
        where_incl = np.where(np.abs(dif_from_180) > angle_thr)

    # Include initial and final pnts
    path_out = np.concatenate(([path[0]], path[where_incl[0] + 1], [path[-1]]))

    return path_out


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    From: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def pnts_in_hull(pnts, hull):
    """
    Return the points in `pnts` that are also contained inside the hull
    """
    where = np.where(in_hull(pnts, hull))
    pnts_inside = pnts[where]
    return pnts_inside
