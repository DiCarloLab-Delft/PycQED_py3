"""
measurement_control.py is becoming very large

this file is intended for small helpers to keep main file more clean
"""
from collections.abc import Iterable
from scipy.spatial import ConvexHull
import numpy as np


def scale_bounds(af_pars, x_scale=None):
    if x_scale is not None:
        for b_name in ["bounds", "dimensions"]:
            if b_name in af_pars.keys():
                # ND hull compatible with adaptive learners
                bounds = af_pars[b_name]
                if isinstance(bounds, ConvexHull):
                    vertices = bounds.points[bounds.vertices]
                    scale = np.array(x_scale)
                    scaled_vertices = vertices * scale
                    scaled_hull = ConvexHull(scaled_vertices)
                    af_pars[b_name] = scaled_hull

                # 1D
                elif not isinstance(bounds[0], Iterable):
                    scaled_bounds = tuple(b * x_scale for b in bounds)
                    af_pars[b_name] = scaled_bounds

                # ND
                elif isinstance(bounds[0], Iterable):
                    scaled_bounds = tuple(
                        tuple(b * scale for b in bounds_dim)for
                        bounds_dim, scale in zip(bounds, x_scale)
                    )
                    af_pars[b_name] = scaled_bounds

    return True
