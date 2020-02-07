from adaptive.learner import Learner1D
from adaptive.learner.learner1D import default_loss
import numpy as np
from functools import partial
import logging
import operator
import random

log = logging.getLogger(__name__)

# ######################################################################
# Utilities for adaptive.learner.learner1D
# ######################################################################


def mk_res_loss_func(
    default_loss_func, min_distance=0.0, max_distance=1.0, dist_is_norm=True
):
    min_distance_orig = min_distance
    max_distance_orig = max_distance

    def func(xs, values, *args, **kw):
        if not dist_is_norm:
            min_distance_used = min_distance_orig / kw["learner"]._scale[0]
            max_distance_used = max_distance_orig / kw["learner"]._scale[0]
        else:
            min_distance_used = min_distance_orig
            max_distance_used = max_distance_orig
        distance = abs(xs[1] - xs[0])
        if distance < min_distance_used:
            loss = 0.0  # don't keep splitting sufficiently small segments
        elif distance > max_distance_used:
            # maximally prioritize segments that are too large proportional to their size
            loss = 1.1 + distance
            # The line below was supposed to achieve the same result...
            # But there seems to be some issues do to divisions probably in the
            # `adaptive` package
            # loss = np.inf
        else:
            loss = default_loss_func(xs, values, *args, **kw)
        return loss

    if not dist_is_norm:
        func.needs_learner_access = True

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


def mk_non_uniform_res_loss_func(
    default_loss_func, n_points: int = 49, res_bounds=(0.5, 3.0)
):
    """
    This function is intended to allow for specifying the min and max
    simplex volumes in a more user friendly and not precise way.
    For a more precise way use the mk_res_loss_func to specify the
    simplex volume limits directly
    """
    # Learner1D normalizes the parameter space to unity
    normalized_domain_size = 1.0
    assert res_bounds[1] > res_bounds[0]
    uniform_resolution = normalized_domain_size / n_points
    min_distance = uniform_resolution * res_bounds[0]
    max_distance = uniform_resolution * res_bounds[1]
    func = mk_res_loss_func(
        default_loss_func, min_distance=min_distance, max_distance=max_distance
    )

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


# ######################################################################
# Learner1D wrappings to be able to access all learner data
# ######################################################################

class Learner1D_Optimize(Learner1D):
    """
    Does everything that the LearnerND does plus wraps it such that
    `mk_optimize_res_loss_func` can be used

    It also accepts using loss fucntions made by
    `mk_non_uniform_res_loss_func` and `mk_res_loss_func`
    inluding providing one of the loss functions from
    adaptive.learner.learnerND

    The resolution loss function in this doc are built such that some
    other loss function is used when the resolution boundaries are respected
    """

    def __init__(self, func, bounds, loss_per_interval=None):
        super(Learner1D_Optimize, self).__init__(func, bounds, loss_per_interval)
        # Keep the orignal learner behaviour but pass extra arguments to
        # the provided input loss function
        if hasattr(self.loss_per_interval, "needs_learner_access"):
            self.best_min = np.inf
            self.best_max = -np.inf
            # self.best_min_last = np.inf
            # self.best_max_last = -np.inf
            # Save the loss fucntion that requires the learner instance
            input_loss_per_interval = self.loss_per_interval
            self.loss_per_interval = partial(input_loss_per_interval, learner=self)
            self.last_best_max = -np.inf
            self.last_best_min = np.inf


# ######################################################################
# Loss and goal functions to be used with the Learner1D_Optimize
# ######################################################################


def mk_optimization_loss(
    minimize: bool = True, threshold: float = None, converge_at_local: bool = False,
    use_random: bool = True
):
    # This sign vs = version is crutial
    if converge_at_local:
        compare_op = operator.le if minimize else operator.ge
    else:
        compare_op = operator.lt if minimize else operator.gt

    # comp_with = "best_min" if minimize else "best_max"
    # This delayed use of the best seen value might help with noise
    # It delays the decision of not considering a local minima interesting
    # anymore
    comp_with = "last_best_min" if minimize else "last_best_max"

    moving_threshold = threshold is None
    if use_random:
        eval_non_interesting = lambda xs_: random.uniform(0, np.abs(xs_[0] - xs_[1]))
    else:
        eval_non_interesting = lambda xs_: np.abs(xs_[0] - xs_[1])

    def func(xs, values, learner, *args, **kw):
        best_optimal = getattr(learner, comp_with)
        comp_threshold = best_optimal if moving_threshold else threshold
        # and compare_op(best_optimal, threshold)

        values = np.array(values)
        scaled_threshold = comp_threshold / learner._scale[1]

        if np.any(compare_op(values, scaled_threshold)):
            # the default_loss of adaptive is never > 1.0 because it
            # normalizes both domains
            # adding the length of the segment makes intersting to sample
            # both sides of an interesting point
            loss = 1.1 + np.abs(xs[0] - xs[1])
        else:
            # The default loss is not very good here, it favors the gradient
            # loss = default_loss(xs, values)
            loss = eval_non_interesting(xs)

        return loss

    return func


def mk_optimization_loss_func(
    minimize=True,
    min_distance=0.0,
    max_distance=np.inf,
    dist_is_norm=False,
    threshold=None,
    converge_at_local=False,
    use_random=True
):
    """
    If you don't specify the threshold you must make use of
    make_optimization_goal_func!!!

    Otherwise the global optimization does not work!

    If you specify the threshold you can use make_threshold_goal_func
    """
    threshold_loss_func = mk_optimization_loss(
        minimize=minimize, threshold=threshold,
        converge_at_local=converge_at_local, use_random=use_random
    )

    func = mk_res_loss_func(
        threshold_loss_func,
        min_distance=min_distance,
        max_distance=max_distance,
        dist_is_norm=dist_is_norm,
    )

    func.needs_learner_access = True
    return func


def make_optimization_goal_func(minimize: bool = True):
    """
    The generated function alway returns True, but is required for the
    mk_optimization_loss_func to work!!!
    """
    update_from = "best_min" if minimize else "best_max"
    last_update_from = "last_best_min" if minimize else "last_best_max"
    optimal_selector = np.min if minimize else np.max
    # Should we keep it like this?
    # compare_op = operator.lt if minimize else operator.gt

    def goal(learner):
        if len(learner.data):
            values = list(learner.data.values())
            # values = np.array(list(learner.data.values()))
            # current_threshold = getattr(learner, update_from)
            # where = compare_op(values, current_threshold)
            # updateQ = np.sum(where) > 0
            # if updateQ:
            #     updated_optimal = np.sort(values[np.where(where)])[0]
            #     updated_optimal = optimal_selector(values[np.where(where)])
            #     setattr(learner, update_from, updated_optimal)
            setattr(learner, last_update_from, getattr(learner, update_from))
            updated_optimal = optimal_selector(values)
            setattr(learner, update_from, updated_optimal)
        return False

    return goal


def make_threshold_goal_func(
    threshold: float, max_pnts_beyond_threshold: int, minimize: bool = True
):
    compare_op = operator.lt if minimize else operator.gt

    def goal(learner):
        num_pnts = np.sum(
            compare_op(np.array(list(learner.data.items())).T[1], threshold)
        )
        return len(learner.data) and num_pnts >= max_pnts_beyond_threshold

    return goal


# ######################################################################
# Some old optimizer experiments based on Learner1D_Optimize
# ######################################################################

# It was supposed to work similar to LearnerND_Optimize but it didn't work
# For reference only, use the above stuff

# def mk_optimization_loss(minimize=True):
#     def func(xs, values, learner):
#         # Assumes 1D image domain
#         # Assumes values is numpy array
#         # The learner evaluates first the boundaries
#         # make sure the min max takes in account all data at the beggining
#         # of the sampling
#         if not learner.npoints > 2:
#             local_min = np.min(list(learner.data.values()))
#             local_max = np.max(list(learner.data.values()))
#         else:
#             local_min = np.min(values)
#             local_max = np.max(values)

#         learner.best_min = (
#             local_min if learner.best_min > local_min else learner.best_min
#         )
#         learner.best_max = (
#             local_max if learner.best_max < local_max else learner.best_max
#         )
#         values_domain_len = np.subtract(learner.best_max, learner.best_min, dtype=float)
#         if values_domain_len == 0:
#             # A better number precision check should be used
#             # This should avoid running into numerical problems at least
#             # when the values are exctly the same.
#             return 0.5
#         # Normalize to the values domain
#         # loss will always be positive
#         # This is important because the learner expect positive output
#         # from the loss function
#         if minimize:
#             loss = np.average((learner.best_max - values) / values_domain_len)
#         else:
#             loss = np.average((values - learner.best_min) / values_domain_len)
#         return loss

#     func.needs_learner_access = True
#     return func


# def mk_optimize_res_loss_func(n_points, res_bounds=(0.5, 3.0), minimize=True):
#     """
#     Creates a loss function that distributes sampling points over the
#     sampling domain in a more optimal way compared to uniform sampling
#     with the goal of finding the minima or maxima
#     It samples with an enforced resolution minimum and maximum.

#     Arguments:
#         n_points: budget of point available to sample
#         res_bounds: (res_boundss[0], res_boundss[1]) resolution in
#             units of uniform resolution
#             (0., np.inf) => infinitely small resolution allowed and no
#             minimum resolution imposed (i.e. don't force to explore the
#             full domain)
#             using (0., np.inf) will stuck the learner at the first optimal
#             it finds
#         minimize: (bool) False for maximize
#             Makes the learner get more "stuck" in regions with high gradients

#     Return: loss_per_simplex function to be used with LearnerND
#     """
#     opt_loss_func = mk_optimization_loss(minimize=minimize)

#     func = mk_non_uniform_res_loss_func(
#         opt_loss_func, n_points=n_points, res_bounds=res_bounds
#     )
#     func.needs_learner_access = True
#     return func


# def mk_optimize_distance_loss_func(
#     min_distance=0.0, max_distance=1.0, minimize=True
# ):
#     """
#     Creates a loss function that distributes sampling points over the
#     sampling domain in a more optimal way compared to uniform sampling
#     with the goal of finding the minima or maxima
#     It samples with an enforced resolution minimum and maximum.

#     Arguments:
#         minimize: (bool) False for maximize
#             Makes the learner get more "stuck" in regions with high gradients

#     Return: loss_per_simplex function to be used with LearnerND
#     """
#     opt_loss_func = mk_optimization_loss(minimize=minimize)

#     func = mk_res_loss_func(
#         opt_loss_func, min_distance=min_distance, max_distance=max_distance
#     )
#     func.needs_learner_access = True
#     return func
