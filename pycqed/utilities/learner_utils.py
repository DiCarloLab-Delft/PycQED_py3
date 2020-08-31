from collections.abc import Iterable
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.measurement.hdf5_data as h5d
import numpy as np
import logging

log = logging.getLogger(__name__)

# ######################################################################
# Utilities for evaluating points before starting the runner
# ######################################################################
def evaluate_X(learner, X, x_scale=None):
    """
    Evaluates the learner's sampling function at the given point
    or points.
    Can be used to evaluate some initial points that the learner will
    remember before running its own runner.

    Arguments:
        learner: (BaseLearner) an instance of the learner
        X: single point or iterable of points
            A tuple is considered single point for a multi-variable
            domain.
    """
    if type(X) is tuple or not isinstance(X, Iterable):
        # A single-variable domain single point or
        # a multi-variable domain single point is given
        X_scaled = scale_X(X, x_scale)
        learner.tell(X_scaled, learner.function(X_scaled))
    else:
        # Several points are to be evaluated
        X_scaled = [scale_X(Xi, x_scale) for Xi in X]
        Y = (learner.function(Xi) for Xi in X_scaled)

        learner.tell_many(X_scaled, Y)


def scale_X(X, x_scale=None):
    if x_scale is not None:
        if isinstance(X, Iterable):
            X_scaled = tuple(xi * scale for xi, scale in zip(X, x_scale))
        else:
            X_scaled = X * x_scale
    else:
        X_scaled = X

    return X_scaled


def tell_X_Y(learner, X, Y, x_scale=None):
    """
    NB: Telling the learner about two many points takes a significant
    time. Beyond 1000 points expect several minutes

    Tell the learner about the sampling function values at the given
    point or points.
    Can be used to avoid evaluating some initial points that the learner
    will remember before running on its own.

    Use case: avoid sampling the boundaries that the learner needs but
    we are not interested in.

    Arguments:
        learner: (BaseLearner) an instance of the learner
        X: single point or iterable of points
            A tuple is considered single point for a multi-variable
            domain.
        Y: scalar or iterable of scalars corresponding to each Xi
    """
    if type(X) is tuple or not isinstance(X, Iterable):
        # A single-variable domain single point or
        # a multi-variable domain single point is given
        X_scaled = scale_X(X, x_scale)
        learner.tell(X_scaled, Y)
    else:
        # Several points are told to the learner
        X_scaled = [scale_X(Xi, x_scale) for Xi in X]
        learner.tell_many(X_scaled, Y)


# ######################################################################
# pycqed especific
# ######################################################################


def prepare_learner_data_for_restore(
    timestamp: str, value_names: set = None
):
    """
    NB: Telling the learner about two many points takes a significant
    time. Beyond 1000 pnts expect several minutes

    Args:
        tell_multivariate_image: (bool) usually we only give the learner a
        scalar value (e.g. cost func), use this if you want to give it more

    Usage example:
        [1]:
            import adaptive
            adaptive.notebook_extension(_inline_js=False)

            from pycqed.utilities.learners_utils import (tell_X_Y,
                prepare_learner_data_for_restore)

            ts = "20200219_194452"
            dummy_f = lambda X, Y: 0.
            dict_for_learner = prepare_learner_data_for_restore(ts,
                value_names={"Cost func"})
            learner = adaptive.Learner2D(dummy_f, bounds=dict_for_learner["bounds"])
            X=dict_for_learner["X"]
            Y=dict_for_learner["Y"]
            tell_X_Y(learner, X, Y)

            def plot(l):
                plot = l.plot(tri_alpha=1.)
                return (plot + plot.Image + plot.EdgePaths).cols(2)
        [2]:
            %%opts Overlay [height=500 width=700]
            a_plot = plot(learner)
            a_plot
    """
    value_names_label = "value_names"
    sweep_parameter_names = "sweep_parameter_names"

    data_fp = a_tools.get_datafilepath_from_timestamp(timestamp)

    param_spec = {
        "data": ("Experimental Data/Data", "dset"),
        value_names_label: ("Experimental Data", "attr:" + value_names_label),
        sweep_parameter_names: ("Experimental Data", "attr:" + sweep_parameter_names),
    }
    raw_data_dict = h5d.extract_pars_from_datafile(data_fp, param_spec)

    # This should have been done in the `extract_pars_from_datafile`...
    raw_data_dict[value_names_label] = np.array(raw_data_dict[value_names_label], dtype=str)

    dim_domain = len(raw_data_dict[sweep_parameter_names])
    data_T = raw_data_dict["data"].T

    if dim_domain == 1:
        X = data_T[0][0]
        bounds = (np.min(X), np.max(X))
    else:
        X = data_T[:dim_domain]
        bounds = [(np.min(Xi), np.max(Xi)) for Xi in X]
        X = X.T  # Shaping for the learner

    if value_names is not None:
        img_idxs = np.where(
            [name in value_names for name in raw_data_dict[value_names_label]]
        )[0]
        Y = data_T[dim_domain + img_idxs]
    else:
        Y = data_T[dim_domain:]

    Y = Y[0] if len(Y) == 1 else Y.T

    return {"raw_data_dict": raw_data_dict, "bounds": bounds, "X": X, "Y": Y}
