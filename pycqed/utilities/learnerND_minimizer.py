from adaptive.learner import LearnerND
from adaptive.learner.learnerND import volume
import numpy as np
from functools import partial
from collections.abc import Iterable
import logging
import operator
import random
import scipy

log = logging.getLogger(__name__)

"""
NB: Only works with ND (N > 1) domain, and 1D image

Possible things to improve
- find how to calculate the extension of a simplex in each dimension
such that it would be possible to specify the resolution boundaries
per dimension
"""

# ######################################################################
# LearnerND wrappings to be able to access all learner data
# ######################################################################


class LearnerND_Minimizer(LearnerND):
    """
    Does everything that the LearnerND does plus wraps it such that
    `mk_optimize_res_loss_func` can be used

    It also accepts using loss fucntions made by
    `mk_non_uniform_res_loss_func` and `mk_vol_limits_loss_func`
    inluding providing one of the loss functions from
    adaptive.learner.learnerND

    The resolution loss function in this doc are built such that some
    other loss function is used when the resolution boundaries are respected
    """

    def __init__(self, func, bounds, loss_per_simplex=None):
        super().__init__(func, bounds, loss_per_simplex)
        # Keep the orignal learner behaviour but pass extra arguments to
        # the provided input loss function
        if hasattr(self.loss_per_simplex, "needs_learner_access"):
            # Save the loss fucntion that requires the learner instance
            input_loss_per_simplex = self.loss_per_simplex
            self.loss_per_simplex = partial(input_loss_per_simplex, learner=self)

            if hasattr(input_loss_per_simplex, "threshold"):
                self.threshold = input_loss_per_simplex.threshold
            else:
                self.threshold = None

            self.compare_op = None
            if hasattr(input_loss_per_simplex, "converge_below"):
                self.converge_below = input_loss_per_simplex.converge_below
            else:
                self.converge_below = None

            self.moving_threshold = np.inf
            self.no_improve_count = 0

            if hasattr(input_loss_per_simplex, "max_no_improve_in_local"):
                self.max_no_improve_in_local = (
                    input_loss_per_simplex.max_no_improve_in_local
                )
                assert self.max_no_improve_in_local >= 2
            else:
                self.max_no_improve_in_local = 7

            self.last_min = np.inf

            # State variable local vs "global search"
            # Note that all the segments that were considered interesting at
            # some point will be still have very high priority when this
            # variable is set back to False
            self.sampling_local_minima = False

            # Compute the domain volume here to avoid the computation in each
            # call of the `mk_vol_limits_loss_func`
            self.vol_bbox = 1.
            for dim_bounds in self._bbox:
                self.vol_bbox *= (dim_bounds[1] - dim_bounds[0])

            self.hull_vol_factor = 1.
            if isinstance(bounds, scipy.spatial.ConvexHull):
                # In case an irregular shaped boundary is used
                self.hull_vol_factor = bounds.volume / self.vol_bbox

        # Recompute all losses if the function scale changes i.e. a new best
        # min or max appeared
        # This happens in `adaptive.LearnerND._update_range` which gets called
        # by `adaptive.LearnerND.tell`
        self._recompute_losses_factor = 1


# ######################################################################
# Loss function utilities for adaptive.learner.learnerND
# ######################################################################


def mk_vol_limits_loss_func(
    default_loss_func, min_volume=0.0, max_volume=1.0, vol_is_norm=False
):
    min_vol_orig = min_volume
    max_vol_orig = max_volume

    def func(simplex, values, value_scale, *args, **kw):

        if vol_is_norm:
            # We want to the normalization to be with respect to the
            # hull's volume in case the domain is a hull
            min_vol_used = min_vol_orig * kw["learner"].hull_vol_factor
            max_vol_used = max_vol_orig * kw["learner"].hull_vol_factor
        else:
            vol_bbox = kw["learner"].vol_bbox
            min_vol_used = min_vol_orig / vol_bbox
            max_vol_used = max_vol_orig / vol_bbox

        vol = volume(simplex)
        if vol < min_vol_used:
            return 0.0  # don't keep splitting sufficiently small simplices
        elif vol > max_vol_used:
            return np.inf  # maximally prioritize simplices that are too large
        else:
            return default_loss_func(simplex, values, value_scale, *args, **kw)

    # Preserve loss function atribute in case a loss function from
    # adaptive.learner.learnerND is given
    if hasattr(default_loss_func, "nth_neighbors"):
        func.nth_neighbors = default_loss_func.nth_neighbors
    return func


def mk_non_uniform_res_loss_func(
    default_loss_func, npoints: int = 249, ndim: int = 2,
    res_bounds=(0.5, 3.0)
):
    """
    This function is intended to allow for specifying the min and max
    simplex volumes in a more user friendly and not precise way.
    For a more precise way use the mk_vol_limits_loss_func to specify the
    simplex volume limits directly
    """
    # LearnerND normalizes the parameter space to unity
    normalized_domain_vol = 1.0
    assert res_bounds[1] > res_bounds[0]
    pnts_per_dim = np.ceil(np.power(npoints, 1.0 / ndim))  # n-dim root
    uniform_resolution = normalized_domain_vol / pnts_per_dim
    min_volume = (uniform_resolution * res_bounds[0]) ** ndim
    max_volume = (uniform_resolution * res_bounds[1]) ** ndim
    func = mk_vol_limits_loss_func(
        default_loss_func, min_volume=min_volume, max_volume=max_volume,
        vol_is_norm=True
    )
    return func

# ######################################################################
# Loss and goal functions to be used with the LearnerND_Minimizer
# ######################################################################


def mk_minimization_loss(
    threshold: float = None,
    converge_at_local: bool = False,
    randomize_global_search: bool = False,
):
    compare_op_start = operator.le if converge_at_local else operator.lt

    def func(simplex, values, value_scale, learner, *args, **kw):
        threshold_is_None = threshold is None
        comp_threshold = learner.moving_threshold if threshold_is_None else threshold
        compare_op = compare_op_start if learner.compare_op is None else learner.compare_op

        # `vol` is normalised 0 <= vol <= 1 because the domain is scaled to a
        # unit hypercube
        vol = volume(simplex)
        if not learner._scale:
            # In case the function landscape is constant so far
            return vol

        # learner._scale makes sure it is the biggest loss and is a
        # finite value such that `vol` can be added

        # `dist_best_val_in_simplex` is the distance (>0) of the best
        # pnt (minimum) in the simplex with respect to the maximum
        # seen ao far, in units of sampling function
        dist_best_val_in_simplex = (
            learner._max_value - np.min(values) * learner._scale
        )
        dist_all = np.average(learner._max_value - np.array(values) * learner._scale)

        # NB: this might have numerical issues, consider using
        # `learner._output_multiplier` if issues arise or keep the
        # cost function in a reasonable range
        scaled_threshold = comp_threshold / learner._scale
        if np.any(compare_op(values, scaled_threshold)):
            # This simplex is the most interesting because we are beyond the
            # threshold, set its loss to maximum

            if threshold_is_None:
                # We treat a moving threshold for a global minimization in a
                # different way than a fixed threshold

                # The `vol` is added to ensure that all simplices of the best
                # point are sampled when the threshold is not moving, avoiding
                # the sampling to get stuck in the initial simplex of the best
                # seen point
                loss = dist_best_val_in_simplex + vol
            else:
                # This makes sure the sampling around the minimum beyond the
                # threshold is uniform

                # `scaled_threshold - np.min(values)` is added to ensure that,
                # from simplices with same length with a point that has a
                # function value beyond the fixed threshold, the points closer
                # to the best value are sampled first

                # `scaled_threshold - np.min(values)` is normalized
                # 0 <= scaled_threshold - np.min(values) <= 1
                side_weight = vol * (1.0 + scaled_threshold - np.min(values))
                loss = (learner._max_value - comp_threshold) + side_weight
        else:
            # This simplex is not interesting, but we bias our search towards
            # lower function values and make sure to not oversample by
            # taking into account the simplex distance

            # Big loss => interesting point => difference from maximum function
            # value gives high loss
            # loss = dist_best_val_in_simplex * vol
            loss = dist_all * vol

        if randomize_global_search:
            # In case the learner is not working well some biased random
            # sampling might help
            # [2020-02-14] Not tested much
            loss = random.uniform(0.0, loss)

        return loss

    func.needs_learner_access = True
    return func


def mk_minimization_loss_func(
    threshold=None,
    converge_below=None,
    converge_at_local=False,
    randomize_global_search=False,
    max_no_improve_in_local=7,
    min_volume=0.0,
    max_volume=np.inf,
    vol_is_norm=False,
    bounds=None,
    npoints=None,
    res_bounds=(0.0, np.inf)
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
    )
    if bounds is None and npoints is None:
        func = mk_vol_limits_loss_func(
            threshold_loss_func,
            min_volume=min_volume,
            max_volume=max_volume,
            vol_is_norm=vol_is_norm,
        )
    else:
        if bounds is None or npoints is None:
            raise ValueError("Both `bounds` and `npoints` must be specified!")

        if isinstance(bounds, scipy.spatial.ConvexHull):
            ndim = bounds.ndim
        else:
            ndim = len(bounds)

        func = mk_non_uniform_res_loss_func(
            default_loss_func=threshold_loss_func, npoints=npoints,
            ndim=ndim, res_bounds=res_bounds)

    func.needs_learner_access = True

    # This is inteded to accessed by the learner
    # Just to make life easier for the user
    func.threshold = threshold
    func.converge_at_local = converge_at_local
    func.max_no_improve_in_local = max_no_improve_in_local
    func.converge_below = converge_below
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
                learner.moving_threshold = learner._min_value
            else:
                # Update second best minimum
                found_new_min = learner._min_value < learner.last_min
                if found_new_min:
                    learner.moving_threshold = learner.last_min
                    # learner.second_min = learner.last_min
                    learner.no_improve_count = 1
                    learner.sampling_local_minima = True

                if learner.sampling_local_minima:
                    if (
                        learner.no_improve_count >= learner.max_no_improve_in_local
                        and not found_new_min
                    ):
                        # "Getting out of the minima"
                        learner.sampling_local_minima = False
                        # Reset count to minimum
                        learner.no_improve_count = 0
                    else:
                        learner.no_improve_count += 1
                else:
                    # We are back in global search
                    # Now we can move the `moving_threshold` to latest minimum
                    learner.moving_threshold = learner._min_value
            if (learner.converge_below is not None and
                    learner.converge_below > learner._min_value):
                learner.compare_op = operator.le

            # Keep track of the last iteration best minimum to be used in the
            # next iteration
            learner.last_min = learner._min_value
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
            compare_op(np.fromiter(learner.data.values(), dtype=np.float64), threshold)
        )
        return len(learner.data) and num_pnts() >= max_pnts_beyond_threshold

    return lambda l: minimization_goal(l) or goal(l)

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
