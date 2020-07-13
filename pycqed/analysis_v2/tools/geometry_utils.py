"""
Utilities for geometrical calculations
"""

import numpy as np
import pycqed.analysis_v2.tools.contours2d as c2d


def closest_pnt_on_segment(
    seg1_x, seg1_y, seg2_x, seg2_y, point_x, point_y, return_dist: bool = True
):
    """
    Determines the closes point on a line segment from a given point

    Inspired from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

    Args:
        (seg1_x, seg1_y): point(s) A of the segment
        (seg2_x, seg2_y): point(s) B of the segment

        (point_x, point_y): point(s) from which the distance is to be minimized
    """
    seg1_x = np.asarray(seg1_x)
    seg2_x = np.asarray(seg2_x)
    point_x = np.asarray(point_x)
    seg1_y = np.asarray(seg1_y)
    seg2_y = np.asarray(seg2_y)
    point_y = np.asarray(point_y)

    px = seg2_x - seg1_x
    py = seg2_y - seg1_y

    norm = px * px + py * py

    u = ((point_x - seg1_x) * px + (point_y - seg1_y) * py) / norm

    u[u > 1.0] = 1.0
    # np.inf is to cover the case of norm == 0 => division by zero
    u[(u < 0.0) | (u == np.inf)] = 0.0

    x = seg1_x + u * px
    y = seg1_y + u * py

    if return_dist:
        dx = x - point_x
        dy = y - point_y

        dist = np.sqrt(dx * dx + dy * dy)

        return x, y, dist
    else:
        return x, y


def closest_pnt_on_triangle(A_x, A_y, B_x, B_y, C_x, C_y, point_x, point_y):
    """
    Return the point with minimum distance calculated by `closest_pnt_on_segment`
    """

    A_x = np.asarray(A_x)
    B_x = np.asarray(B_x)
    C_x = np.asarray(C_x)
    point_x = np.asarray(point_x)
    A_y = np.asarray(A_y)
    B_y = np.asarray(B_y)
    C_y = np.asarray(C_y)
    point_y = np.asarray(point_y)

    from_seg_1 = closest_pnt_on_segment(A_x, A_y, B_x, B_y, point_x, point_y, return_dist=True)
    from_seg_2 = closest_pnt_on_segment(B_x, B_y, C_x, C_y, point_x, point_y, return_dist=True)
    from_seg_3 = closest_pnt_on_segment(C_x, C_y, A_x, A_y, point_x, point_y, return_dist=True)

    distances = np.array([from_seg_1[-1], from_seg_2[-1], from_seg_3[-1]])
    args_min = np.argmin(distances.T, axis=1)

    x = np.choose(args_min, (from_seg_1[0], from_seg_2[0], from_seg_3[0]))
    y = np.choose(args_min, (from_seg_1[1], from_seg_2[1], from_seg_3[1]))

    return x, y


def constrain_to_triangle(triangle, x, y):
    """
    If points (x, y) are outside the triangle defined by triangle
    then the points outside are projected onto the triangle sides

    Example:
        from pycqed.analysis_v2.tools import geometry_utils as geo

        fig, ax = plt.subplots(1, 1, dpi=120)

        cal_triangle = np.array([[0.72332126, 3.67366289],
               [4.10132008, 3.73165123],
               [5.62289489, 2.74094961]])
        x = np.random.uniform(cal_triangle.T[0].min(), cal_triangle.T[0].max(), 50)
        y = np.random.uniform(cal_triangle.T[1].min(), cal_triangle.T[1].max(), 50)

        ax.plot(cal_triangle[[0, 1, 2, 0]].T[0], cal_triangle[[0, 1, 2, 0]].T[1], "-", linewidth=1)
        ax.scatter(x, y)

        proj_x, proj_y = geo.constrain_to_triangle(cal_triangle, x, y)
        ax.scatter(proj_x, proj_y, s=markersize/10, label="Projected on triangle")
        ax.legend()
    """

    ouside_triangle = c2d.in_hull(np.array((x, y)).T, triangle) ^ 1
    x_corr = np.array(x)
    y_coor = np.array(y)

    if np.any(ouside_triangle):
        proj_x, proj_y = closest_pnt_on_triangle(*triangle.flatten(), x, y)
        where = np.where(ouside_triangle)
        x_corr[where] = proj_x[where]
        y_coor[where] = proj_y[where]

    return x_corr, y_coor
