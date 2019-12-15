import adaptive
from adaptive.learner import LearnerND
import numpy as np
from functools import partial
import logging

log = logging.getLogger('__name__')

# ######################################################################
# Loss function utilities for adaptive.learner.learnerND
# ######################################################################

"""
NB: Only tested with ND > 1 -> 1D

Possible things to improve
- try resolution loss with the default losses of the adaptive package
- find how to calculate the extension of a simplex in each dimension
such that it would be possible to specify the resolution boundaries
per dimension
"""


def mk_resolution_loss_func(default_loss_func, min_volume=0, max_volume=1):
    # *args, **kw are used to allow for things like mk_target_func_val_loss_example
    def func(simplex, values, value_scale, *args, **kw):
        vol = adaptive.learner.learnerND.volume(simplex)
        if vol < min_volume:
            return 0.  # don't keep splitting sufficiently small simplices
        elif vol > max_volume:
            return np.inf  # maximally prioritize simplices that are too large
        else:
            return default_loss_func(simplex, values, value_scale, *args, **kw)

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, 'nth_neighbors'):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


def mk_non_uniform_resolution_loss_func(
    default_loss_func, n_points: int = 249, n_dim: int = 1, res_bounds=(0.5, 3.0)
):
    """
    This function is intended to allow for specifying the min and max
    simplex volumes in a more user friendly and not precise way.
    For a more precise way use the mk_resolution_loss_func to specify the
    simplex volume limits directly
    """
    # LearnerND normalizes the parameter space to unity
    normalized_domain_size = 1.0
    assert res_bounds[0] > 0.
    assert res_bounds[1] > res_bounds[0]
    pnts_per_dim = np.ceil(np.power(n_points, 1.0 / n_dim))  # n-dim root
    uniform_resolution = normalized_domain_size / pnts_per_dim
    min_volume = (uniform_resolution * res_bounds[0]) ** n_dim
    max_volume = (uniform_resolution * res_bounds[1]) ** n_dim
    func = mk_resolution_loss_func(
        default_loss_func, min_volume=min_volume, max_volume=max_volume
    )
    return func

# ######################################################################
# LearnerND wrappings to be able to acces all learner data
# ######################################################################


class LearnerND_Optimize(LearnerND):
    """
    Does everything that the LearnerND does plus wraps it such that
    `mk_optimize_resolution_loss_func` can be used

    It also accepts using loss fucntions made by
    `mk_non_uniform_resolution_loss_func` and `mk_resolution_loss_func`
    inluding providing one of the loss functions from
    adaptive.learner.learnerND

    The resolution loss function in this doc are built such that some
    other loss function is used when the resolution boundaries are respected
    """
    def __init__(self, func, bounds, loss_per_simplex=None):
        super(LearnerND_Optimize, self).__init__(func, bounds, loss_per_simplex)
        # Keep the orignal learner behaviour but pass extra arguments to
        # the provided input loss function
        if hasattr(self.loss_per_simplex, 'needs_learner_access'):
            self.best_min = np.inf
            self.best_max = -np.inf
            # Save the loss fucntion that requires the learner instance
            input_loss_per_simplex = self.loss_per_simplex
            self.loss_per_simplex = partial(input_loss_per_simplex, learner=self)


def mk_optimize_resolution_loss_func(
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

    Return: loss_per_simplex function to be used with LearnerND
    """
    def loss(simplex, values, value_scale, learner):
        # Assumes values is numpy array
        # The learner evaluate fisrt the boundaries
        # make sure the min max takes in account all data at the beggining
        # of the sampling
        if not learner.bounds_are_done:
            local_min = np.min(list(learner.data.values()))
            local_max = np.max(list(learner.data.values()))
        else:
            local_min = np.min(values)
            local_max = np.max(values)

        learner.best_min = local_min if learner.best_min > local_min else learner.best_min
        learner.best_max = local_max if learner.best_max < local_max else learner.best_max
        values_domain_len = learner.best_max - learner.best_min
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

    func = mk_non_uniform_resolution_loss_func(
        loss, n_points=n_points, n_dim=n_dim, res_bounds=res_bounds
    )
    func.needs_learner_access = True
    return func


# ######################################################################
# Below is the firs attempt, it works but the above one is more general
# ######################################################################

def mk_target_func_val_loss_example(val):
    """
    This is an attemp to force the learner to keep looking for better
    optimal points and not just being pushed away from the local optimal
    (when using this as the default_loss_func with mk_resolution_loss_func)
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
        learner.best_min = loss_value if learner.best_min > loss_value else learner.best_min
        learner.best_max = loss_value if learner.best_max < loss_value else learner.best_max
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


def mk_target_val_resolution_loss_func(
        target_value, n_points, n_dim, res_bounds=(0.5, 3.0),
        default_loss_func='sum'):
    # log.error(default_loss_func)
    if isinstance(default_loss_func, str):
        if default_loss_func == 'times_std':
            default_func = mk_target_func_val_loss_times_std(target_value)
        elif default_loss_func == 'plus_std':
            log.warning('times_std is probably better...')
            default_func = mk_target_func_val_loss_plus_std(target_value)
        elif default_loss_func == 'needs_learner_example':
            default_func = mk_target_func_val_loss_example(target_value)
        elif default_loss_func == 'sum':
            default_func = mk_target_func_val_loss(target_value)
        else:
            raise ValueError('Default loss function type not recognized!')
    func = mk_non_uniform_resolution_loss_func(
        default_func,
        n_points=n_points,
        n_dim=n_dim,
        res_bounds=res_bounds,
    )
    if default_loss_func == 'needs_learner_example':
        func.needs_learner_access = True
    return func
