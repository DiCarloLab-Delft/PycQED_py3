"""
Author: Victor Negîrneac
Last update: 2020-02-15

Minimization toolbox for 1D domain functions.
Developed based on the `adaptive.Learner1D` from adaptive v0.10.0:
https://github.com/python-adaptive/adaptive/releases/tag/v0.10.0

I hope it survives any changes that the `adaptive` package might suffer
"""

from adaptive.learner import Learner1D
import numpy as np
from functools import partial
import logging
import operator
import random
from pycqed.utilities.general import get_module_name

log = logging.getLogger(__name__)

# ######################################################################
# Learner1D wrappings to be able to access all learner data
# ######################################################################


class Learner1D_Minimizer(Learner1D):
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
        # Sanity check that can save hours of debugging...
        assert bounds[1] > bounds[0]

        super().__init__(func, bounds, loss_per_interval)
        # Keep the orignal learner behaviour but pass extra arguments to
        # the provided input loss function
        if hasattr(self.loss_per_interval, "needs_learner_access"):
            # Save the loss function that requires the learner instance
            input_loss_per_interval = self.loss_per_interval
            self.loss_per_interval = partial(input_loss_per_interval, learner=self)

            if hasattr(input_loss_per_interval, "threshold"):
                self.threshold = input_loss_per_interval.threshold
            else:
                self.threshold = None

            self.compare_op = None
            if hasattr(input_loss_per_interval, "converge_below"):
                self.converge_below = input_loss_per_interval.converge_below
            else:
                self.converge_below = None

            self.moving_threshold = np.inf
            self.no_improve_count = 0

            if hasattr(input_loss_per_interval, "max_no_improve_in_local"):
                self.max_no_improve_in_local = (
                    input_loss_per_interval.max_no_improve_in_local
                )
                assert self.max_no_improve_in_local >= 2
            else:
                self.max_no_improve_in_local = 4

            if hasattr(input_loss_per_interval, "update_losses_after_no_improv"):
                self.update_losses_after_no_improv = (
                    input_loss_per_interval.update_losses_after_no_improv
                )
            else:
                self.update_losses_after_no_improv = True

            self.last_min = np.inf

            # State variable local vs "global search"
            # Note that all the segments that were considered interesting at
            # some point will be still have very high priority when this
            # variable is set back to False
            self.sampling_local_minima = False

        # Recompute all losses if the function scale changes i.e. a new best
        # min or max appeared
        # This happens in `adaptive.Learner1D.tell`
        self._recompute_losses_factor = 1

    def _recompute_all_losses(self):
        """
        This is the equivalent fucntion that exists in LearnernND for this
        purpuse.

        It is just a copy paste of a few lines from the `Learner1D.tell`

        It is used to recompute losses when the `Learner1D_Minimizer` is "done"
        with sampling a local minimum.
        """

        # NB: We are not updating the scale here as the `tell` method does
        # because we assume this method will be called only after sampling
        # `max_no_improve_in_local` points in the local minimum

        for interval in reversed(self.losses):
            self._update_interpolated_loss_in_interval(*interval)


# ######################################################################
# Utilities for adaptive.learner.learner1D
# ######################################################################


def mk_res_loss_func(
    default_loss_func, min_distance=0.0, max_distance=1.0, dist_is_norm=False
):
    min_distance_orig = min_distance
    max_distance_orig = max_distance

    # Wrappers to make it work with the default loss of `adaptive` package
    if get_module_name(default_loss_func, level=0) == "adaptive":
        def _default_loss_func(xs, values, *args, **kw):
            return default_loss_func(xs, values)
    else:
        def _default_loss_func(xs, values, *args, **kw):
            return default_loss_func(xs, values, *args, **kw)

    def func(xs, values, *args, **kw):
        if dist_is_norm:
            min_distance_used = min_distance_orig
            max_distance_used = max_distance_orig
        else:
            min_distance_used = min_distance_orig / kw["learner"]._scale[0]
            max_distance_used = max_distance_orig / kw["learner"]._scale[0]

        # `dist` is normalised 0 <= dist <= 1 because xs are scaled
        dist = abs(xs[1] - xs[0])
        if dist < min_distance_used:
            loss = 0.0  # don't keep splitting sufficiently small intervals
        elif dist > max_distance_used:
            # maximally prioritize intervals that are too large
            # the learner will compare all the segments that have inf loss based
            # on the distance between them
            loss = np.inf
        else:
            loss = _default_loss_func(xs, values, *args, **kw)
        return loss

    if not dist_is_norm:
        func.needs_learner_access = True

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


def mk_non_uniform_res_loss_func(
    default_loss_func, npoints: int = 49, res_bounds=(0.5, 3.0)
):
    """
    This function is intended to allow for specifying the min and max
    interval size in a more user friendly and not precise way.
    For a more precise way use the mk_res_loss_func to specify the
    interval size limits directly
    """
    # Learner1D normalizes the parameter space to unity
    normalized_domain_size = 1.0
    assert res_bounds[1] > res_bounds[0]
    uniform_resolution = normalized_domain_size / npoints
    min_distance = uniform_resolution * res_bounds[0]
    max_distance = uniform_resolution * res_bounds[1]
    func = mk_res_loss_func(
        default_loss_func,
        min_distance=min_distance,
        max_distance=max_distance,
        dist_is_norm=True,
    )

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


# ######################################################################
# Loss and goal functions to be used with the Learner1D_Minimizer
# ######################################################################


def mk_minimization_loss(
    threshold: float = None,
    converge_at_local: bool = False,
    randomize_global_search: bool = False,
    interval_weight: float = 5.0,
):
    assert interval_weight >= 0.0 and interval_weight <= 1000.0
    compare_op_start = operator.le if converge_at_local else operator.lt

    # `w` controls how "square" is the resulting function
    # more "square" => x needs to be lower in order for the interval_factor
    # to be lower
    w = interval_weight / 1000.0
    with np.errstate(divide="ignore"):
        A = np.divide(1.0, np.arctan(np.divide(1.0, w)))

    def interval_factor(vol):
        with np.errstate(divide="ignore"):
            out = A * np.arctan(np.divide(vol, w))
        return out

    w_not = 1.0 - w
    with np.errstate(divide="ignore"):
        A_not = np.divide(1.0, np.arctan(np.divide(1.0, w_not)))

    def close_to_optimal_factor(scale, dist):
        with np.errstate(divide="ignore"):
            out = A_not * np.arctan(np.divide(dist, scale * w_not))
        return out

    def func(xs, values, learner, *args, **kw):
        threshold_is_None = threshold is None
        comp_threshold = learner.moving_threshold if threshold_is_None else threshold
        compare_op = (
            compare_op_start if learner.compare_op is None else learner.compare_op
        )

        # `dist` is normalised 0 <= dist <= 1 because xs are scaled
        dist = np.abs(xs[0] - xs[1])

        # learner._scale[1] makes sure it is the biggest loss and is a
        # finite value such that `dist` can be added

        # `dist_best_val_in_interval` is the distance (>0) of the best
        # pnt (minimum) in the ineterval with respect to the maximum
        # seen ao far, in units of sampling function
        dist_best_val_in_interval = (
            learner._bbox[1][1] - np.min(values) * learner._scale[1]
        )

        if dist_best_val_in_interval == 0.0:
            # In case the function landscape is constant so far
            return dist

        values = np.array(values)
        scaled_threshold = comp_threshold / learner._scale[1]
        if np.any(compare_op(values, scaled_threshold)):
            # This interval is the most interesting because we are beyond the
            # threshold, set its loss to maximum

            if threshold_is_None:
                # We treat a moving threshold for a global minimization in a
                # different way than a fixed threshold

                # The `dist` is added to ensure that both sides of the best
                # point are sampled when the threshold is not moving, avoiding the
                # sampling to get stuck at one side of the best seen point
                loss = dist_best_val_in_interval + dist
            else:
                # This makes sure the sampling around the minimum beyond the
                # threshold is uniform

                # `scaled_threshold - np.min(values)` is added to ensure that,
                # from intervals with same length with a point that has a
                # function value beyond the fixed threshold, the points closer
                # to the best value are sampled first

                # `scaled_threshold - np.min(values)` is normalized
                # 0 <= scaled_threshold - np.min(values) <= 1
                side_weight = dist * (1.0 + scaled_threshold - np.min(values))
                loss = (learner._bbox[1][1] - comp_threshold) + side_weight
        else:
            # This interval is not interesting, but we bias our search towards
            # lower function values and make sure to not oversample by
            # taking into account the interval distance

            # Big loss => interesting point => difference from maximum function
            # value gives high loss
            loss = close_to_optimal_factor(learner._scale[1], dist_best_val_in_interval) * interval_factor(dist)

        if randomize_global_search:
            # In case the learner is not working well some biased random
            # sampling might help
            # [2020-02-14] Not tested much
            loss = random.uniform(0.0, loss)

        return loss

    return func


def mk_minimization_loss_func(
    threshold=None,
    converge_below=None,
    min_distance=0.0,
    max_distance=np.inf,
    dist_is_norm=False,
    converge_at_local=False,
    randomize_global_search=False,
    max_no_improve_in_local=4,
    update_losses_after_no_improv=True,
    interval_weight=50.,
):
    """
    If you don't specify the threshold you must make use of
    mk_minimization_goal_func!!!
    Otherwise the global optimization does not work!
    If you specify the threshold you must use mk_threshold_goal_func

    This tool is intended to be used for sampling continuous (possibly
    noisy) functions.
    """
    threshold_loss_func = mk_minimization_loss(
        threshold=threshold,
        converge_at_local=converge_at_local,
        randomize_global_search=randomize_global_search,
        interval_weight=interval_weight
    )

    func = mk_res_loss_func(
        threshold_loss_func,
        min_distance=min_distance,
        max_distance=max_distance,
        dist_is_norm=dist_is_norm,
    )

    func.needs_learner_access = True

    # This is inteded to accessed by the learner
    # Just to make life easier for the user
    func.threshold = threshold
    func.converge_at_local = converge_at_local
    func.max_no_improve_in_local = max_no_improve_in_local
    func.converge_below = converge_below
    func.update_losses_after_no_improv = update_losses_after_no_improv
    return func


def mk_minimization_goal_func():
    """
    The generated function alway returns False such that it can be chained with
    the user's stop condition e.g. `goal=lambda l: goal(l) or l.npoints > 100`,
    but is required for the mk_minimization_loss_func to work!!!
    This is required because it updates important variables for the loss
    function to work properly
    """

    def goal(learner):
        # No action if no points
        if len(learner.data):
            if len(learner.data) < 2:
                # First point, just take it as the threshold
                # Do it here to make sure calculation with the
                # `moving_threshold` don't run into numerical issues with inf
                learner.moving_threshold = learner._bbox[1][0]
            else:
                # Update second best minimum
                found_new_min = learner._bbox[1][0] < learner.last_min
                if found_new_min:
                    learner.moving_threshold = learner.last_min
                    # learner.second_min = learner.last_min
                    learner.no_improve_count = 1
                    learner.sampling_local_minima = True

                if learner.sampling_local_minima:
                    if learner.no_improve_count >= learner.max_no_improve_in_local:
                        # We decide to "get out of the local minimum"
                        learner.sampling_local_minima = False
                        # Reset count to minimum
                        learner.no_improve_count = 0
                        if learner.update_losses_after_no_improv:
                            # Update the threshold so that _recompute_all_losses
                            # has the desired effect
                            learner.moving_threshold = learner._bbox[1][0]

                            # Force update all losses such that the learner stops
                            # sampling points in the local minimum

                            # This has some computation overhead but should not
                            # happen too often as finding a new minimum is not
                            # expected to happen many times

                            # NB: this method does not exist in the original
                            # `Learner1D`
                            learner._recompute_all_losses()
                    else:
                        learner.no_improve_count += 1
                else:
                    # We are back in global search
                    # Now we can move the `moving_threshold` to latest minimum
                    learner.moving_threshold = learner._bbox[1][0]
            if (
                learner.converge_below is not None
                and learner.converge_below > learner._bbox[1][0]
            ):
                learner.compare_op = operator.le

            # Keep track of the last iteration best minimum to be used in the
            # next iteration
            learner.last_min = learner._bbox[1][0]
        return False

    return goal


def mk_min_threshold_goal_func(max_pnts_beyond_threshold: int):
    compare_op = operator.lt
    minimization_goal = mk_minimization_goal_func()

    def goal(learner):
        threshold = learner.threshold
        if threshold is None:
            raise ValueError(
                "You must specify a threshold argument in `mk_minimization_loss_func`!"
            )
        # This needs to be a func to avoid evaluating it if there is no data yet
        num_pnts = lambda: np.sum(
            compare_op(np.array(list(learner.data.items())).T[1], threshold)
        )
        return len(learner.data) and num_pnts() >= max_pnts_beyond_threshold

    return lambda l: minimization_goal(l) or goal(l)
