import adaptive
from adaptive.learner import LearnerND
import numpy as np
from functools import partial
from collections.abc import Iterable
import logging

log = logging.getLogger(__name__)

log.error("`learnerND_optimize` is deprecated! Use `learnernND_minimize`.")

# ######################################################################
# Loss function utilities for adaptive.learner.learnerND
# ######################################################################

"""
NB: Only works with ND > 1 domain, and 1D image

Possible things to improve
- try resolution loss with the default losses of the adaptive package
- find how to calculate the extension of a simplex in each dimension
such that it would be possible to specify the resolution boundaries
per dimension
"""


def mk_res_loss_func(default_loss_func, min_volume=0.0, max_volume=1.0):
    # *args, **kw are used to allow for things like mk_target_func_val_loss_example
    def func(simplex, values, value_scale, *args, **kw):
        vol = adaptive.learner.learnerND.volume(simplex)
        if vol < min_volume:
            return 0.0  # don't keep splitting sufficiently small simplices
        elif vol > max_volume:
            return np.inf  # maximally prioritize simplices that are too large
        else:
            return default_loss_func(simplex, values, value_scale, *args, **kw)

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


def mk_non_uniform_res_loss_func(
    default_loss_func, n_points: int = 249, n_dim: int = 1, res_bounds=(0.5, 3.0)
):
    """
    This function is intended to allow for specifying the min and max
    simplex volumes in a more user friendly and not precise way.
    For a more precise way use the mk_res_loss_func to specify the
    simplex volume limits directly
    """
    # LearnerND normalizes the parameter space to unity
    normalized_domain_size = 1.0
    assert res_bounds[1] > res_bounds[0]
    pnts_per_dim = np.ceil(np.power(n_points, 1.0 / n_dim))  # n-dim root
    uniform_resolution = normalized_domain_size / pnts_per_dim
    min_volume = (uniform_resolution * res_bounds[0]) ** n_dim
    max_volume = (uniform_resolution * res_bounds[1]) ** n_dim
    func = mk_res_loss_func(
        default_loss_func, min_volume=min_volume, max_volume=max_volume
    )
    return func


# ######################################################################
# LearnerND wrappings to be able to access all learner data
# ######################################################################


class LearnerND_Optimize(LearnerND):
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

    def __init__(self, func, bounds, loss_per_simplex=None):
        super(LearnerND_Optimize, self).__init__(func, bounds, loss_per_simplex)
        # Keep the orignal learner behaviour but pass extra arguments to
        # the provided input loss function
        if hasattr(self.loss_per_simplex, "needs_learner_access"):
            self.best_min = np.inf
            self.best_max = -np.inf
            # Save the loss fucntion that requires the learner instance
            input_loss_per_simplex = self.loss_per_simplex
            self.loss_per_simplex = partial(input_loss_per_simplex, learner=self)


def mk_optimization_loss(minimize=True, use_grad=False):
    def func(simplex, values, value_scale, learner):
        # Assumes values is numpy array
        # The learner evaluate first the boundaries
        # make sure the min max takes in account all data at the beggining
        # of the sampling
        if not learner.bounds_are_done:
            local_min = np.min(list(learner.data.values()))
            local_max = np.max(list(learner.data.values()))
        else:
            local_min = np.min(values)
            local_max = np.max(values)

        learner.best_min = (
            local_min if learner.best_min > local_min else learner.best_min
        )
        learner.best_max = (
            local_max if learner.best_max < local_max else learner.best_max
        )
        values_domain_len = np.subtract(learner.best_max, learner.best_min, dtype=float)
        if values_domain_len == 0:
            # A better number precision check should be used
            # This should avoid running into numerical problems at least
            # when the values are exctly the same.
            return 0.5
        # Normalize to the values domain
        # loss will always be positive
        # This is important because the learner expect positive output
        # from the loss function
        if minimize:
            loss = np.average((learner.best_max - values) / values_domain_len)
        else:
            loss = np.average((values - learner.best_min) / values_domain_len)
        if use_grad:
            loss += np.std(values) / values_domain_len
        return loss

    func.needs_learner_access = True
    return func


def mk_optimize_res_loss_func(
    n_points, n_dim, res_bounds=(0.5, 3.0), minimize=True, use_grad=False
):
    """
    Creates a loss function that distributes sampling points over the
    sampling domain in a more optimal way compared to uniform sampling
    with the goal of finding the minima or maxima
    It samples with an enforced resolution minimum and maximum.

    Arguments:
        n_points: budget of point available to sample
        n_dim: domain dimension of the function to sample
        res_bounds: (res_boundss[0], res_boundss[1]) resolution in
            units of uniform resolution
            (0., np.inf) => infinitely small resolution allowed and no
            minimum resolution imposed (i.e. don't force to explore the
            full domain)
            using (0., np.inf) will stuck the learner at the first optimal
            it finds
        minimize: (bool) False for maximize
        use_grad: (bool) adds the std of the simplex's value to the loss
            Makes the learner get more "stuck" in regions with high gradients

    Return: loss_per_simplex function to be used with LearnerND
    """
    opt_loss_func = mk_optimization_loss(minimize=minimize, use_grad=use_grad)

    func = mk_non_uniform_res_loss_func(
        opt_loss_func, n_points=n_points, n_dim=n_dim, res_bounds=res_bounds
    )
    func.needs_learner_access = True
    return func


# ######################################################################
# Below is the first attempt, it works but the above one is more general
# ######################################################################


def mk_target_func_val_loss_example(val):
    """
    This is an attemp to force the learner to keep looking for better
    optimal points and not just being pushed away from the local optimal
    (when using this as the default_loss_func with mk_res_loss_func)
    It is constantly trying to find a better point than the best seen.

    NB: Didn't seem to work for me for the CZ simulations

    NB2: It is still a good example of how to use the LearnerND wrapper above
    such that the entire learner data is available without modifying the
    original LearnerND on any other way that might become very
    incompatible later
    """

    def target_func_val_loss(simplex, values, value_scale, learner):
        # Assumes values is numpy array
        loss_value = 1.0 / np.sum((values - val) ** 2)
        # Keep updating the widest range
        learner.best_min = (
            loss_value if learner.best_min > loss_value else learner.best_min
        )
        learner.best_max = (
            loss_value if learner.best_max < loss_value else learner.best_max
        )
        # downscore simplex to be minimum if it is not better than best seen loss
        return learner.best_min if loss_value < learner.best_max else loss_value

    return target_func_val_loss


"""
Possible improvement for the use of std
- Try including also the nearst points in the std and see if it works
 even better
"""


def mk_target_func_val_loss_times_std(val):
    def target_func_val_loss(simplex, values, value_scale):
        # Assumes values is numpy array
        loss_value = 1.0 / np.sum((values - val) ** 2) * np.std(values)
        return loss_value

    return target_func_val_loss


def mk_target_func_val_loss_plus_std(val):
    """
    This one is sensible to the gradient only a bit
    The mk_target_func_val_loss_times_std seemed to work better
    """

    def target_func_val_loss(simplex, values, value_scale):
        # Assumes values is numpy array
        loss_value = 1.0 / np.sum((values - val) ** 2) + np.std(values)
        return loss_value

    return target_func_val_loss


# ######################################################################


def mk_target_func_val_loss(val):
    def target_func_val_loss(simplex, values, value_scale):
        # Assumes values is numpy array
        loss_value = 1.0 / np.sum((values - val) ** 2)
        return loss_value

    return target_func_val_loss


def mk_target_val_res_loss_func(
    target_value, n_points, n_dim, res_bounds=(0.5, 3.0), default_loss_func="sum"
):
    if isinstance(default_loss_func, str):
        if default_loss_func == "times_std":
            default_func = mk_target_func_val_loss_times_std(target_value)
        elif default_loss_func == "plus_std":
            log.warning("times_std is probably better...")
            default_func = mk_target_func_val_loss_plus_std(target_value)
        elif default_loss_func == "needs_learner_example":
            default_func = mk_target_func_val_loss_example(target_value)
        elif default_loss_func == "sum":
            default_func = mk_target_func_val_loss(target_value)
        else:
            raise ValueError("Default loss function type not recognized!")
    func = mk_non_uniform_res_loss_func(
        default_func, n_points=n_points, n_dim=n_dim, res_bounds=res_bounds
    )
    if default_loss_func == "needs_learner_example":
        func.needs_learner_access = True
    return func


# ######################################################################
# Attempt to limit the resolution in each dimension
# ######################################################################


def mk_res_loss_per_dim_func(
    default_loss_func, min_distances=0.0, max_distances=np.inf
):
    """
    This function is intended to allow for specifying the min and max
    distance between points for adaptive sampling for each dimension or
    for all dimensions
    """
    # if min_distances is None and max_distances is not None:
    #     min_distances = np.full(np.size(max_distances), np.inf)
    # elif max_distances is None and min_distances is not None:
    #     max_distances = np.full(np.size(min_distances), 0.0)
    # else:
    #     raise ValueError("The min_distances or max_distances must be specified!")

    min_distances = np.asarray(min_distances)
    max_distances = np.asarray(max_distances)
    assert np.all(min_distances < max_distances)

    def func(simplex, values, value_scale, *args, **kw):
        learner = kw.pop("learner")
        verticesT = simplex.T
        max_for_each_dim = np.max(verticesT, axis=1)
        min_for_each_dim = np.min(verticesT, axis=1)
        diff = max_for_each_dim - min_for_each_dim
        if np.all(diff < min_distances):
            # don't keep splitting sufficiently small simplices
            loss = 0.0
        elif np.any(diff > max_distances):
            # maximally prioritize simplices that are too large in any dimension
            loss = np.inf
        else:
            if hasattr(default_loss_func, "needs_learner_access"):
                kw["learner"] = learner
            loss = default_loss_func(simplex, values, value_scale, *args, **kw)
        return loss

    func.needs_learner_access = True
    return func


def mk_optimize_res_loss_per_dim_func(
    bounds, min_distances=0.0, max_distances=np.inf, minimize=True, use_grad=False
):
    """
    It doesn't work well because I dind't realise soon enough that more
    control over how the learner splits the simlpices is necessary in order
    force it really limit the resolution in each dimension. :(

    The problem is that we either block the learner from splitting the
    simplex when the resolution limit is achieved in one dimention or
    all dimensions.

    Creates a loss function that distributes sampling points over the
    sampling domain in a more optimal way compared to uniform sampling
    with the goal of finding the minima or maxima
    It samples with an enforced resolution minimum and maximum.

    Arguments:
        bounds: (Iterable of tuples)
            Could also be retrieved from the learner, but that would
            make things slower.
        min_distances: (number or arraylike)
        max_distances: (number or arraylike)
        minimize: (bool) False for maximize

    Return: loss_per_simplex function to be used with LearnerND
    """
    log.warning("This function does not work very well.")
    opt_loss_func = mk_optimization_loss(minimize=minimize, use_grad=use_grad)
    bounds = np.array(bounds)
    domain_length = bounds.T[1] - bounds.T[0]
    min_distances = np.asarray(min_distances)
    max_distances = np.asarray(max_distances)
    min_distances = min_distances / domain_length
    max_distances = max_distances / domain_length

    func = mk_res_loss_per_dim_func(
        opt_loss_func, min_distances=min_distances, max_distances=max_distances
    )
    func.needs_learner_access = True
    return func


# ######################################################################
# Utilities for evaluating points before starting the runner
# ######################################################################


def evaluate_X(learner, X):
    """
    Evaluates the learner's sampling function at the given point
    or points.
    Can be used to evaluate some initial points that the learner will
    remember before running the runner.

    Arguments:
        learner: (BaseLearner) an instance of the learner
        X: single point or iterable of points
            A tuple is considered single point for a multi-variable
            domain.
    """
    if type(X) is tuple or not isinstance(X, Iterable):
        # A single-variable domain single point or
        # a multi-variable domain single point is given
        learner.tell(X, learner.function(X))
    else:
        # Several points are to be evaluated
        Y = [learner.function(Xi) for Xi in X]
        learner.tell_many(X, Y)
