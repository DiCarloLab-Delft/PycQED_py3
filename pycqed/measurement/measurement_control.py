import types
import logging
import time
import numpy as np
from collections.abc import Iterable
import operator
from scipy.optimize import fmin_powell
from pycqed.measurement import hdf5_data as h5d
from pycqed.utilities.general import (
    dict_to_ordered_tuples,
    delete_keys_from_dict,
    check_keyboard_interrupt,
    KeyboardFinish,
    flatten,
    get_git_revision_hash,
)
from pycqed.utilities.get_default_datadir import get_default_datadir
from pycqed.utilities.general import get_module_name

# Used for auto qcodes parameter wrapping
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_det
from pycqed.analysis.tools.data_manipulation import get_generation_means

from pycqed.analysis.tools.plot_interpolation import interpolate_heatmap

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.plots.colors import color_cycle

# Used for adaptive sampling
from adaptive import runner
from adaptive.learner import BaseLearner, Learner1D, Learner2D, LearnerND

# SKOptLearner Notes
# NB: This optimizer can be slow and is intended for very, very costly
# functions compared to the computation time of the optimizer itself

# NB2: One of the cool things is that it can do hyper-parameter
# optimizations e.g. if the parameters are integers

# NB3: The optimizer comes with several options and might require
# some wise choices for your particular case
from adaptive.learner import SKOptLearner

# Optimizer based on adaptive sampling
from pycqed.utilities.learner1D_minimizer import Learner1D_Minimizer
from pycqed.utilities.learnerND_minimizer import LearnerND_Minimizer
import pycqed.utilities.learner_utils as lu
from . import measurement_control_helpers as mch

from skopt import Optimizer  # imported for checking types

try:
    import msvcrt  # used on windows to catch keyboard input
except:
    print("Could not import msvcrt (used for detecting keystrokes)")

try:
    import PyQt5

    # For reference:
    # from pycqed.measurement import qcodes_QtPlot_monkey_patching
    # The line above was (and still is but keep rading) necessary
    # for the plotmon_2D to be able to set colorscales from
    # `qcodes_QtPlot_colors_override.py` and be able to set the
    # colorbar range when the plots are created
    # See also `MC.plotmon_2D_cmaps`, `MC.plotmon_2D_zranges` below
    # That line was moved into the `__init__.py` of pycqed so that
    # `QtPlot` can be imported from qcodes with all the modifications

    from qcodes.plots.pyqtgraph import QtPlot, TransformState
except Exception:
    print(
        "pyqtgraph plotting not supported, "
        'try "from qcodes.plots.pyqtgraph import QtPlot" '
        "to see the full error"
    )
    print("When instantiating an MC object," " be sure to set live_plot_enabled=False")

log = logging.getLogger(__name__)


def is_subclass(obj, test_obj):
    """
    Extra check to ensure test_obj is a class before
    checking for whether it is a subclass. Prevents Python from
    throwing an error if test_obj is not a class but a function
    """
    return isinstance(obj, type) and issubclass(obj, test_obj)


class MeasurementControl(Instrument):

    """
    New version of Measurement Control that allows for adaptively determining
    data points.
    """

    def __init__(
        self,
        name: str,
        plotting_interval: float = 3,
        datadir: str = get_default_datadir(),
        live_plot_enabled: bool = True,
        verbose: bool = True,
    ):
        super().__init__(name=name)

        self.add_parameter(
            "datadir",
            initial_value=datadir,
            vals=vals.Strings(),
            parameter_class=ManualParameter,
        )
        # Soft average does not work with adaptive measurements.
        self.add_parameter(
            "soft_avg",
            label="Number of soft averages",
            parameter_class=ManualParameter,
            vals=vals.Ints(1, int(1e8)),
            initial_value=1,
        )

        self.add_parameter(
            "plotting_max_pts",
            label="Maximum number of live plotting points",
            parameter_class=ManualParameter,
            vals=vals.Ints(1),
            initial_value=4000,
        )
        self.add_parameter(
            "verbose",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=verbose,
        )
        self.add_parameter(
            "live_plot_enabled",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=live_plot_enabled,
        )
        self.add_parameter(
            "plotting_interval",
            unit="s",
            vals=vals.Numbers(min_value=0.001),
            set_cmd=self._set_plotting_interval,
            get_cmd=self._get_plotting_interval,
        )
        self.add_parameter(
            "persist_mode",
            vals=vals.Bool(),
            parameter_class=ManualParameter,
            initial_value=True,
        )

        self.add_parameter(
            "on_progress_callback",
            vals=vals.Callable(),
            docstring="A callback to communicate progress. This should be a "
            "Callable accepting ints between 0 and 100 indicating percdone.",
            parameter_class=ManualParameter,
            initial_value=None,
        )

        self.add_parameter(
            "cfg_clipping_mode",
            vals=vals.Bool(),
            docstring="Clipping mode, when True ignores ValueErrors  when "
            "setting parameters. This can be useful when running optimizations",
            parameter_class=ManualParameter,
            initial_value=False,
        )

        self.add_parameter(
            "instrument_monitor",
            parameter_class=ManualParameter,
            initial_value=None,
            vals=vals.Strings(),
        )

        # pyqtgraph plotting process is reused for different measurements.
        if self.live_plot_enabled():
            self.create_plot_monitor()
        self.plotting_interval(plotting_interval)

        self.soft_iteration = 0  # used as a counter for soft_avg
        self._persist_dat = None
        self._persist_xlabs = None
        self._persist_ylabs = None

        # plotmon_2D colorbar color mapping and ranges
        # Change this to your preferences when using the plotmon_2D
        # This could be a parameter but it doesn't seem to be worth saving
        # See `choose_MC_cmap_zrange` in this file to know how this is used
        # e.g. self.plotmon_2D_cmaps = {"Phase": "anglemap45"}
        # see pycqed.measurment.qcodes_QtPlot_colors_override for more cmaps
        self.plotmon_2D_cmaps = {}
        # e.g. self.plotmon_2D_zranges = {"Phase": (0.0, 180.0)}
        self.plotmon_2D_zranges = {}

        # Flag used to create a specific plot trace for LearnerND_Minimizer
        # and Learner1D_Minimizer.
        self.Learner_Minimizer_detected = False
        self.CMA_detected = False

        # Setting this to true adds 5s to each experiment
        # If possible set to False as default but mind that for now many
        # Analysis rely on the old snapshot
        self.save_legacy_snapshot = True

    ##############################################
    # Functions used to control the measurements #
    ##############################################

    def run(
        self,
        name: str = None,
        exp_metadata: dict = None,
        mode: str = "1D",
        disable_snapshot_metadata: bool = False,
        **kw
    ):
        """
        Core of the Measurement control.

        Args:
            name (string):
                    Name of the measurement. This name is included in the
                    name of the data files.
            exp_metadata (dict):
                    Dictionary containing experimental metadata that is saved
                    to the data file at the location
                        file['Experimental Data']['Experimental Metadata']
                bins (list):
                    If bins is specified in exp_metadata this is used to
                    average data in specific bins for live plotting.
                    This is useful when it is required to take data in single
                    shot mode.
            mode (str):
                    Measurement mode. Can '1D', '2D', or 'adaptive'.
            disable_snapshot_metadata (bool):
                    Disables metadata saving of the instrument snapshot.
                    This can be useful for performance reasons.
                    N.B. Do not use this unless you know what you are doing!
                    Except for special cases instrument settings should always
                    be saved in the datafile.
                    This is an argument instead of a parameter because this
                    should always be explicitly diabled in order to prevent
                    accidentally leaving it off.

        """
        # Setting to zero at the start of every run, used in soft avg
        self.soft_iteration = 0

        if mode != "adaptive":
            # Certain adaptive visualization features leave undesired effects
            # on the plots of non-adaptive plots
            self.clean_previous_adaptive_run()

        self.set_measurement_name(name)
        self.print_measurement_start_msg()

        self.mode = mode
        # used in determining data writing indices (deprecated?)
        self.iteration = 0

        # used for determining data writing indices and soft averages
        self.total_nr_acquired_values = 0

        # needs to be defined here because of the with statement below
        return_dict = {}
        self.last_sweep_pts = None  # used to prevent resetting same value

        with h5d.Data(
            name=self.get_measurement_name(), datadir=self.datadir()
        ) as self.data_object:
            try:

                check_keyboard_interrupt()
                self.get_measurement_begintime()
                if not disable_snapshot_metadata:
                    self.save_instrument_settings(self.data_object)
                self.create_experimentaldata_dataset()

                self.plotting_bins = None
                if exp_metadata is not None:
                    self.save_exp_metadata(exp_metadata, self.data_object)
                    if "bins" in exp_metadata.keys():
                        self.plotting_bins = exp_metadata["bins"]

                if mode != "adaptive":
                    try:
                        # required for 2D plotting and data storing.
                        # try except because some swf get the sweep points in the
                        # prepare statement. This needs a proper fix
                        self.xlen = len(self.get_sweep_points())
                    except Exception:
                        self.xlen = 1
                if self.mode == "1D":
                    self.measure()
                elif self.mode == "2D":
                    self.measure_2D()
                elif self.mode == "adaptive":
                    self.measure_soft_adaptive()
                else:
                    raise ValueError('Mode "{}" not recognized.'.format(self.mode))
            except KeyboardFinish as e:
                print(e)
            result = self.dset[()]
            self.get_measurement_endtime()
            self.save_MC_metadata(self.data_object)  # timing labels etc

            return_dict = self.create_experiment_result_dict()

        self.finish(result)
        return return_dict

    def measure(self, *kw):
        if self.live_plot_enabled():
            self.initialize_plot_monitor()

        for sweep_function in self.sweep_functions:
            sweep_function.prepare()

        if (
            self.sweep_functions[0].sweep_control == "soft"
            and self.detector_function.detector_control == "soft"
        ):
            self.detector_function.prepare()
            self.get_measurement_preparetime()
            self.measure_soft_static()

        elif self.detector_function.detector_control == "hard":
            self.get_measurement_preparetime()
            sweep_points = self.get_sweep_points()

            while self.get_percdone() < 100:
                start_idx = self.get_datawriting_start_idx()
                if len(self.sweep_functions) == 1:
                    self.sweep_functions[0].set_parameter(sweep_points[start_idx])
                    self.detector_function.prepare(
                        sweep_points=self.get_sweep_points().astype(np.float64)
                    )
                    self.measure_hard()
                else:  # If mode is 2D
                    for i, sweep_function in enumerate(self.sweep_functions):
                        swf_sweep_points = sweep_points[:, i]
                        val = swf_sweep_points[start_idx]
                        sweep_function.set_parameter(val)
                    self.detector_function.prepare(
                        sweep_points=sweep_points[
                            start_idx : start_idx + self.xlen, 0
                        ].astype(np.float64)
                    )
                    self.measure_hard()
        else:
            raise Exception(
                "Sweep and Detector functions not "
                + "of the same type. \nAborting measurement"
            )
            print(self.sweep_function.sweep_control)
            print(self.detector_function.detector_control)

        check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon(force_update=True)
        if self.mode == "2D":
            self.update_plotmon_2D(force_update=True)
        elif self.mode == "adaptive":
            self.update_plotmon_adaptive(force_update=True)
        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()

        return

    def measure_soft_static(self):
        for self.soft_iteration in range(self.soft_avg()):
            for i, sweep_point in enumerate(self.sweep_points):
                self.measurement_function(sweep_point)

    def measure_soft_adaptive(self, method=None):
        """
        Uses the adaptive function and keywords for that function as
        specified in self.af_pars()
        """
        self.save_optimization_settings()

        # This allows to use adaptive samplers with distinct setting and
        # keep the data in the same dataset. E.g. sample a segment of the
        # positive axis and a segment of the negative axis
        multi_adaptive_single_dset = self.af_pars.get(
            "multi_adaptive_single_dset", False
        )
        if multi_adaptive_single_dset:
            af_pars_list = self.af_pars.pop("adaptive_pars_list")
        else:
            af_pars_list = [self.af_pars]

        for sweep_function in self.sweep_functions:
            sweep_function.prepare()
        self.detector_function.prepare()
        self.get_measurement_preparetime()

        # ######################################################################
        # BEGIN loop of points in extra dims
        # ######################################################################
        # Used to (re)initialize the plot monitor only between the iterations
        # of this for loop
        last_i_af_pars = -1

        Xs = self.af_pars.get("extra_dims_sweep_pnts", [None])
        for X in Xs:
            # ##################################################################
            # BEGIN loop of adaptive samplers with distinct settings
            # ##################################################################

            for i_af_pars, af_pars in enumerate(af_pars_list):
                # We detect the type of adaptive function here so that the right
                # adaptive plot monitor is initialized and configured
                self.Learner_Minimizer_detected = False
                self.CMA_detected = False

                # Used to update plots specific to this type of optimizers
                module_name = get_module_name(af_pars.get("adaptive_function", self))
                self.Learner_Minimizer_detected = (
                    self.Learner_Minimizer_detected
                    or (
                        module_name == "learner1D_minimizer"
                        and hasattr(af_pars.get("loss_per_interval", self), "threshold")
                    )
                    or (
                        module_name == "learnerND_minimizer"
                        and hasattr(af_pars.get("loss_per_simplex", self), "threshold")
                    )
                )

                self.CMA_detected = (
                    self.CMA_detected or module_name == "cma.evolution_strategy"
                )

                # Determines if the optimization will minimize or maximize
                self.minimize_optimization = af_pars.get("minimize", True)
                self.f_termination = af_pars.get("f_termination", None)

                self.adaptive_besteval_indxs = [0]

                if self.live_plot_enabled() and i_af_pars > last_i_af_pars:
                    self.initialize_plot_monitor_adaptive()
                last_i_af_pars = i_af_pars

                self.adaptive_function = af_pars.get("adaptive_function")

                if self.adaptive_function == "Powell":
                    self.adaptive_function = fmin_powell

                if len(Xs) > 1 and X is not None:
                    opt_func = lambda x: self.mk_optimization_function()(
                        flatten([x, X])
                    )
                else:
                    opt_func = self.mk_optimization_function()

                if is_subclass(self.adaptive_function, BaseLearner):
                    Learner = self.adaptive_function
                    mch.scale_bounds(af_pars=af_pars, x_scale=self.x_scale)

                    # Pass the right parameters two each type of learner
                    if issubclass(Learner, Learner1D):
                        self.learner = Learner(
                            opt_func,
                            bounds=af_pars["bounds"],
                            loss_per_interval=af_pars.get("loss_per_interval", None),
                        )
                    elif issubclass(Learner, Learner2D):
                        self.learner = Learner(
                            opt_func,
                            bounds=af_pars["bounds"],
                            loss_per_triangle=af_pars.get("loss_per_triangle", None),
                        )
                    elif issubclass(Learner, LearnerND):
                        self.learner = Learner(
                            opt_func,
                            bounds=af_pars["bounds"],
                            loss_per_simplex=af_pars.get("loss_per_simplex", None),
                        )
                    elif issubclass(Learner, SKOptLearner):
                        # NB: This learner expects the `optimization_function`
                        # to be scalar
                        # See https://scikit-optimize.github.io/modules/generated/skopt.optimizer.gp_minimize.html#skopt.optimizer.gp_minimize
                        self.learner = Learner(
                            opt_func,
                            dimensions=af_pars["dimensions"],
                            base_estimator=af_pars.get("base_estimator", "gp"),
                            n_initial_points=af_pars.get("n_initial_points", 10),
                            acq_func=af_pars.get("acq_func", "gp_hedge"),
                            acq_optimizer=af_pars.get("acq_optimizer", "auto"),
                            n_random_starts=af_pars.get("n_random_starts", None),
                            random_state=af_pars.get("random_state", None),
                            acq_func_kwargs=af_pars.get("acq_func_kwargs", None),
                            acq_optimizer_kwargs=af_pars.get(
                                "acq_optimizer_kwargs", None
                            ),
                        )
                    else:
                        raise NotImplementedError(
                            "Learner subclass type not supported."
                        )

                    if "X0Y0" in af_pars:
                        # Tell the learner points that are already evaluated
                        # Typically to avoid evaluating the boundaries
                        # Intended for `LearnerND` and derivatives there of
                        # NB: this points don't show up in the `dset`. They are
                        # stored only in the learner's memory
                        # NB: Put a significant number of points (e.g. ~100) on
                        # the boundaries to really avoid the learner going there
                        X0 = af_pars["X0Y0"]["X0"]
                        Y0 = af_pars["X0Y0"]["Y0"]

                        # For convenience we allows the user to specify a
                        # single Y0 value that will be the image for all the
                        # domain points in X0
                        if not isinstance(Y0, Iterable) or len(Y0) < len(X0):
                            Y0 = np.repeat([Y0], len(X0), axis=0)

                        lu.tell_X_Y(self.learner, X=X0, Y=Y0, x_scale=self.x_scale)

                    if "X0" in af_pars:
                        # Tell the learner the initial points if provided
                        lu.evaluate_X(self.learner, af_pars["X0"], x_scale=self.x_scale)

                    # N.B. the runner that is used is not an `adaptive.Runner` object
                    # rather it is the `adaptive.runner.simple` function. This
                    # ensures that everything runs in a single process, as is
                    # required by QCoDeS (May 2018) and makes things simpler.
                    self.runner = runner.simple(
                        learner=self.learner, goal=af_pars["goal"]
                    )

                    # Only save optimization results if the sampling is a single
                    # adaptive run
                    # Needs more elaborated developments
                    if not multi_adaptive_single_dset and Xs[0] is None:
                        # NB: If you reload the optimizer module, `issubclass` will fail
                        # This is because the reloaded class is a new distinct object
                        if issubclass(self.adaptive_function, SKOptLearner):
                            # NB: Having an optmizer that also complies with the adaptive
                            # interface breaks a bit the previous structure
                            # now there are many checks for this case
                            # Because this is also an optimizer we save the result
                            # Pass the learner because it contains all the points
                            self.save_optimization_results(
                                self.adaptive_function, self.learner
                            )
                        elif (
                            issubclass(self.adaptive_function, Learner1D_Minimizer)
                            or issubclass(self.adaptive_function, LearnerND_Minimizer)
                        ):
                            # Because this is also an optimizer we save the result
                            # Pass the learner because it contains all the points
                            self.save_optimization_results(
                                self.adaptive_function, self.learner
                            )

                elif isinstance(
                    self.adaptive_function, types.FunctionType
                ) or isinstance(self.adaptive_function, np.ufunc):
                    try:
                        # exists so it is possible to extract the result
                        # of an optimization post experiment
                        af_pars_copy = dict(af_pars)
                        non_used_pars = [
                            "adaptive_function",
                            "minimize",
                            "f_termination",
                        ]
                        for non_used_par in non_used_pars:
                            af_pars_copy.pop(non_used_par, None)
                        self.adaptive_result = self.adaptive_function(
                            self.mk_optimization_function(), **af_pars_copy
                        )
                    except StopIteration:
                        print("Reached f_termination: %s" % (self.f_termination))

                    if (
                        not multi_adaptive_single_dset
                        and Xs[0] is None
                        and hasattr(self, "adaptive_result")
                    ):
                        self.save_optimization_results(
                            self.adaptive_function, result=self.adaptive_result
                        )
                else:
                    raise Exception(
                        'optimization function: "%s" not recognized'
                        % self.adaptive_function
                    )
            # ##################################################################
            # END loop of adaptive samplers with distinct settings
            # ##################################################################

        # ######################################################################
        # END loop of points in extra dims
        # ######################################################################

        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()
        check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon(force_update=True)
        self.update_plotmon_adaptive(force_update=True)
        return

    def measure_hard(self):
        new_data = np.array(self.detector_function.get_values()).astype(np.float64).T

        ###########################
        # Shape determining block #
        ###########################

        datasetshape = self.dset.shape
        start_idx, stop_idx = self.get_datawriting_indices_update_ctr(new_data)

        new_datasetshape = (np.max([datasetshape[0], stop_idx]), datasetshape[1])
        self.dset.resize(new_datasetshape)
        len_new_data = stop_idx - start_idx
        if len(np.shape(new_data)) == 1:
            old_vals = self.dset[start_idx:stop_idx, len(self.sweep_functions)]
            new_vals = (new_data + old_vals * self.soft_iteration) / (
                1 + self.soft_iteration
            )

            self.dset[start_idx:stop_idx, len(self.sweep_functions)] = new_vals.astype(
                np.float64
            )
        else:
            old_vals = self.dset[start_idx:stop_idx, len(self.sweep_functions) :]
            new_vals = (new_data + old_vals * self.soft_iteration) / (
                1 + self.soft_iteration
            )

            self.dset[
                start_idx:stop_idx, len(self.sweep_functions) :
            ] = new_vals.astype(np.float64)
        sweep_len = len(self.get_sweep_points().T)

        ######################
        # DATA STORING BLOCK #
        ######################
        if sweep_len == len_new_data:  # 1D sweep
            self.dset[:, 0] = self.get_sweep_points().T.astype(np.float64)
        else:
            try:
                if len(self.sweep_functions) != 1:
                    relevant_swp_points = self.get_sweep_points()[
                        start_idx : start_idx + len_new_data :
                    ]
                    self.dset[
                        start_idx:, 0 : len(self.sweep_functions)
                    ] = relevant_swp_points.astype(np.float64)
                else:
                    self.dset[start_idx:, 0] = self.get_sweep_points()[
                        start_idx : start_idx + len_new_data :
                    ].T.astype(np.float64)
            except Exception:
                # There are some cases where the sweep points are not
                # specified that you don't want to crash (e.g. on -off seq)
                pass

        check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon()
        if self.mode == "2D":
            self.update_plotmon_2D_hard()
        self.iteration += 1
        self.print_progress(stop_idx)
        return new_data

    def measurement_function(self, x):
        """
        Core measurement function used for soft sweeps
        """
        if np.size(x) == 1:
            x = [x]
        if np.size(x) != len(self.sweep_functions):
            raise ValueError('size of x "%s" not equal to # sweep functions' % x)
        for i, sweep_function in enumerate(self.sweep_functions[::-1]):
            # If statement below tests if the value is different from the
            # last value that was set, if it is the same the sweep function
            # will not be called. This is important when setting a parameter
            # is either expensive (e.g., loading a waveform) or has adverse
            # effects (e.g., phase scrambling when setting a MW frequency.

            # x[::-1] changes the order in which the parameters are set, so
            # it is first the outer sweep point and then the inner.This
            # is generally not important except for specifics: f.i. the phase
            # of an agilent generator is reset to 0 when the frequency is set.
            swp_pt = x[::-1][i]
            # The value that was actually set. Returned by the sweep
            # function if known.
            set_val = None
            if self.iteration == 0:
                # always set the first point
                set_val = sweep_function.set_parameter(swp_pt)
            else:
                # start_idx -1 refers to the last written value
                prev_swp_pt = self.last_sweep_pts[::-1][i]
                if swp_pt != prev_swp_pt:
                    # only set if not equal to previous point
                    try:
                        set_val = sweep_function.set_parameter(swp_pt)
                    except ValueError as e:
                        if self.cfg_clipping_mode():
                            log.warning("MC clipping mode caught exception:")
                            log.warning(e)
                        else:
                            raise e
            if isinstance(set_val, float):
                # The Value in x is overwritten by the value that the
                # sweep function returns. This allows saving the value
                # that was actually set rather than the one that was
                # intended. This does require custom support from
                # a sweep function.
                x[-i - 1] = set_val

        # used for next iteration
        self.last_sweep_pts = x

        datasetshape = self.dset.shape

        vals = self.detector_function.acquire_data_point()
        start_idx, stop_idx = self.get_datawriting_indices_update_ctr(vals)
        # Resizing dataset and saving
        new_datasetshape = (np.max([datasetshape[0], stop_idx]), datasetshape[1])
        self.dset.resize(new_datasetshape)
        new_data = np.append(x, vals)

        old_vals = self.dset[start_idx:stop_idx, :]
        new_vals = (new_data + old_vals * self.soft_iteration) / (
            1 + self.soft_iteration
        )

        self.dset[start_idx:stop_idx, :] = new_vals.astype(np.float64)
        # update plotmon
        check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon()
        if self.mode == "2D":
            self.update_plotmon_2D()
        elif self.mode == "adaptive":
            self.update_plotmon_adaptive()
        self.iteration += 1
        if self.mode != "adaptive":
            self.print_progress(stop_idx)
        else:
            self.print_progress_adaptive()
        return vals

    def mk_optimization_function(self):
        """
        Returns a wrapper around the measurement function
        This construction is necessary to be able to run several adaptive
        samplers with distinct settings in the same dataset
        """

        def func(x):
            """
            A wrapper around the measurement function.
            It takes the following actions based on parameters specified
            in self.af_pars:
            - Rescales the function using the "x_scale" parameter, default is 1
            - Inverts the measured values if "minimize"==False
            - Compares measurement value with "f_termination" and raises an
            exception, that gets caught outside of the optimization loop, if
            the measured value is smaller than this f_termination.

            Measurement function with scaling to correct physical value
            """
            if self.x_scale is not None:
                x_ = np.array(x, dtype=np.float64)
                scale_ = np.array(self.x_scale, dtype=np.float64)
                # NB this division here might interfere with measurements
                # that involve integer values in `x`
                x = type(x)(x_ / scale_)

            vals = self.measurement_function(x)
            # This takes care of data that comes from a "single" segment of a
            # detector for a larger shape such as the UFHQC single int avg detector
            # that gives back data in the shape [[I_val_seg0, Q_val_seg0]]
            if len(np.shape(vals)) == 2:
                vals = np.array(vals)[:, 0]

            # to check if vals is an array with multiple values
            if isinstance(vals, Iterable):
                vals = vals[self.par_idx]

            if self.mode == "adaptive":
                # Keep track of the best seen points so far so that they can be
                # plotted as stars, need to be done before inverting `vals`
                col_indx = len(self.sweep_function_names) + self.par_idx
                comp_op = operator.lt if self.minimize_optimization else operator.gt
                if comp_op(vals, self.dset[self.adaptive_besteval_indxs[-1], col_indx]):
                    self.adaptive_besteval_indxs.append(len(self.dset) - 1)

            if self.minimize_optimization:
                if self.f_termination is not None:
                    if vals < self.f_termination:
                        raise StopIteration()
            else:
                # when maximizing interrupt when larger than condition before
                # inverting
                if self.f_termination is not None:
                    if vals > self.f_termination:
                        raise StopIteration()
                vals = np.multiply(-1, vals)

            return vals

        return func

    def finish(self, result):
        """
        Deletes arrays to clean up memory and avoid memory related mistakes
        """
        # this data can be plotted by enabling persist_mode
        self._persist_dat = result
        self._persist_xlabs = self.sweep_par_names
        self._persist_ylabs = self.detector_function.value_names

        for attr in [
            "TwoD_array",
            "dset",
            "sweep_points",
            "sweep_points_2D",
            "sweep_functions",
            "xlen",
            "ylen",
            "iteration",
            "soft_iteration",
        ]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    ###################
    # 2D-measurements #
    ###################

    def run_2D(self, name=None, **kw):
        self.run(name=name, mode="2D", **kw)

    def tile_sweep_pts_for_2D(self):
        self.xlen = len(self.get_sweep_points())
        self.ylen = len(self.sweep_points_2D)
        if np.size(self.get_sweep_points()[0]) == 1:
            # create inner loop pts
            self.sweep_pts_x = self.get_sweep_points()
            x_tiled = np.tile(self.sweep_pts_x, self.ylen)
            # create outer loop
            self.sweep_pts_y = self.sweep_points_2D
            y_rep = np.repeat(self.sweep_pts_y, self.xlen)
            # 2020-02-09, This does not preserve types, e.g. integer parameters
            # and rises validators exceptions
            if np.issubdtype(type(self.sweep_pts_x[0]), np.integer) or np.issubdtype(
                type(self.sweep_points_2D[0]), np.integer
            ):
                c = np.column_stack(
                    (x_tiled.astype(np.object), y_rep)
                )  # this preserves types
            else:
                c = np.column_stack((x_tiled, y_rep))
            self.set_sweep_points(c)
            self.initialize_plot_monitor_2D()
        return

    def measure_2D(self, **kw):
        """
        Sweeps over two parameters set by sweep_function and sweep_function_2D.
        The outer loop is set by sweep_function_2D, the inner loop by the
        sweep_function.

        Soft(ware) controlled sweep functions require soft detectors.
        Hard(ware) controlled sweep functions require hard detectors.
        """

        self.tile_sweep_pts_for_2D()
        self.measure(**kw)
        return

    def set_sweep_function_2D(self, sweep_function):
        # If it is not a sweep function, assume it is a qc.parameter
        # and try to auto convert it it
        if not isinstance(sweep_function, swf.Sweep_function):
            sweep_function = wrap_par_to_swf(sweep_function)

        if len(self.sweep_functions) != 1:
            raise KeyError(
                "Specify sweepfunction 1D before specifying sweep_function 2D"
            )
        else:
            self.sweep_functions.append(sweep_function)
            self.sweep_function_names.append(str(sweep_function.__class__.__name__))

    def set_sweep_points_2D(self, sweep_points_2D):
        self.sweep_functions[1].sweep_points = sweep_points_2D
        self.sweep_points_2D = sweep_points_2D

    ###########
    # Plotmon #
    ###########
    """
    There are three kinds of plotmons, the regular plotmon,
    the 2D plotmon (which does a heatmap) and the adaptive plotmon.
    """

    def create_plot_monitor(self):
        """
        Creates new PyQtGraph plotting monitor.
        Can also be used to recreate these when plotting has crashed.
        """
        if hasattr(self, "main_QtPlot"):
            del self.main_QtPlot
        if hasattr(self, "secondary_QtPlot"):
            del self.secondary_QtPlot

        self.main_QtPlot = QtPlot(
            window_title="Main plotmon of {}".format(self.name), figsize=(600, 400)
        )
        self.secondary_QtPlot = QtPlot(
            window_title="Secondary plotmon of {}".format(self.name), figsize=(600, 400)
        )

    def initialize_plot_monitor(self):
        if self.main_QtPlot.traces != []:
            self.main_QtPlot.clear()
        self.curves = []
        self.curves_mv_thresh = []
        xlabels = self.sweep_par_names
        xunits = self.sweep_par_units
        ylabels = self.detector_function.value_names
        yunits = self.detector_function.value_units

        j = 0
        if (
            self._persist_ylabs == ylabels and self._persist_xlabs == xlabels
        ) and self.persist_mode():
            persist = True
        else:
            persist = False
        for yi, ylab in enumerate(ylabels):
            for xi, xlab in enumerate(xlabels):
                if persist:  # plotting persist first so new data on top
                    yp = self._persist_dat[:, yi + len(self.sweep_function_names)]
                    xp = self._persist_dat[:, xi]
                    if len(xp) < self.plotting_max_pts():
                        self.main_QtPlot.add(
                            x=xp,
                            y=yp,
                            subplot=j + 1,
                            color=0.75,  # a grayscale value
                            pen=None,
                            symbol="o",
                            symbolSize=5,
                        )

                if self.mode == "adaptive":
                    kw = {"pen": None}
                else:
                    kw = {}
                self.main_QtPlot.add(
                    x=[0],
                    y=[0],
                    xlabel=xlab,
                    xunit=xunits[xi],
                    ylabel=ylab,
                    yunit=yunits[yi],
                    subplot=j + 1,
                    color=color_cycle[j % len(color_cycle)],
                    symbol="o",
                    symbolSize=5,
                    **kw
                )
                self.curves.append(self.main_QtPlot.traces[-1])

                if self.Learner_Minimizer_detected and yi == self.par_idx:
                    self.main_QtPlot.add(
                        x=[0],
                        y=[0],
                        xlabel=xlab,
                        xunit=xunits[xi],
                        ylabel=ylab,
                        yunit=yunits[yi],
                        subplot=j + 1,
                        color=color_cycle[3],
                        symbol="s",
                        symbolSize=3,
                    )
                    self.curves_mv_thresh.append(self.main_QtPlot.traces[-1])

                j += 1
            self.main_QtPlot.win.nextRow()

    def update_plotmon(self, force_update=False):
        # Note: plotting_max_pts takes precendence over force update
        if self.live_plot_enabled() and (
            self.dset.shape[0] < self.plotting_max_pts()
            or (self.plotting_bins is not None)
        ):
            i = 0
            try:
                time_since_last_mon_update = time.time() - self._mon_upd_time
            except Exception:
                # creates the time variables if they did not exists yet
                self._mon_upd_time = time.time()
                time_since_last_mon_update = 1e9
            try:
                if (
                    time_since_last_mon_update > self.plotting_interval()
                    or force_update
                ):

                    nr_sweep_funcs = len(self.sweep_function_names)
                    for y_ind in range(len(self.detector_function.value_names)):
                        for x_ind in range(nr_sweep_funcs):
                            x = self.dset[:, x_ind]
                            y = self.dset[:, nr_sweep_funcs + y_ind]

                            # used to average e.g., single shot measuremnts
                            # can be specified in MC.run(exp_metadata['bins'])
                            if self.plotting_bins is not None:
                                x = self.plotting_bins
                                if len(y) % len(x) != 0:
                                    # nan's are appended if shapes do not match
                                    missing_vals = missing_vals = int(
                                        len(x) - len(y) % len(x)
                                    )
                                    y_ext = np.concatenate(
                                        [y, np.ones(missing_vals) * np.nan]
                                    )
                                else:
                                    y_ext = y

                                y = np.nanmean(
                                    y_ext.reshape(
                                        (len(self.plotting_bins), -1), order="F"
                                    ),
                                    axis=1,
                                )

                            self.curves[i]["config"]["x"] = x
                            self.curves[i]["config"]["y"] = y
                            i += 1

                            if (
                                self.Learner_Minimizer_detected
                                and y_ind == self.par_idx
                            ):
                                min_x = np.min(x)
                                max_x = np.max(x)
                                threshold = (
                                    self.learner.moving_threshold
                                    if self.learner.threshold is None
                                    else self.learner.threshold
                                )
                                if threshold < np.inf:
                                    threshold = (
                                        threshold
                                        if self.minimize_optimization
                                        else -threshold
                                    )
                                    self.curves_mv_thresh[x_ind]["config"]["x"] = [
                                        min_x,
                                        max_x,
                                    ]
                                    self.curves_mv_thresh[x_ind]["config"]["y"] = [
                                        threshold,
                                        threshold,
                                    ]
                    self._mon_upd_time = time.time()
                    self.main_QtPlot.update_plot()
            except Exception as e:
                log.warning(e)

    def initialize_plot_monitor_2D(self):
        """
        Preallocates a data array to be used for the update_plotmon_2D command.

        Made to work with at most 2 2D arrays (as this is how the labview code
        works). It should be easy to extend this function for more vals.
        """
        if self.live_plot_enabled():
            self.time_last_2Dplot_update = time.time()
            n = len(self.sweep_pts_y)
            m = len(self.sweep_pts_x)
            self.TwoD_array = np.empty([n, m, len(self.detector_function.value_names)])
            self.TwoD_array[:] = np.NAN
            self.secondary_QtPlot.clear()
            slabels = self.sweep_par_names
            sunits = self.sweep_par_units
            zlabels = self.detector_function.value_names
            zunits = self.detector_function.value_units

            for j in range(len(self.detector_function.value_names)):
                cmap, zrange = self.choose_MC_cmap_zrange(zlabels[j], zunits[j])
                config_dict = {
                    "x": self.sweep_pts_x,
                    "y": self.sweep_pts_y,
                    "z": self.TwoD_array[:, :, j],
                    "xlabel": slabels[0],
                    "xunit": sunits[0],
                    "ylabel": slabels[1],
                    "yunit": sunits[1],
                    "zlabel": zlabels[j],
                    "zunit": zunits[j],
                    "subplot": j + 1,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)

    def update_plotmon_2D(self, force_update=False):
        """
        Adds latest measured value to the TwoD_array and sends it
        to the QC_QtPlot.
        """
        if self.live_plot_enabled():
            try:
                i = int((self.iteration) % (self.xlen * self.ylen))
                x_ind = int(i % self.xlen)
                y_ind = int(i / self.xlen)
                for j in range(len(self.detector_function.value_names)):
                    z_ind = len(self.sweep_functions) + j
                    self.TwoD_array[y_ind, x_ind, j] = self.dset[i, z_ind]
                self.secondary_QtPlot.traces[j]["config"]["z"] = self.TwoD_array[
                    :, :, j
                ]
                if (
                    time.time() - self.time_last_2Dplot_update
                    > self.plotting_interval()
                    or self.iteration == len(self.sweep_points)
                    or force_update
                ):
                    self.time_last_2Dplot_update = time.time()
                    self.secondary_QtPlot.update_plot()
            except Exception as e:
                log.warning(e)

    def initialize_plot_monitor_2D_interp(self, ld=0):
        """
        Initialize a 2D plot monitor for interpolated (adaptive) plots
        """
        if self.live_plot_enabled() and len(self.sweep_function_names) == 2:
            self.time_last_2Dplot_update = time.time()

            # self.secondary_QtPlot.clear()
            slabels = self.sweep_par_names
            sunits = self.sweep_par_units
            zlabels = self.detector_function.value_names
            zunits = self.detector_function.value_units

            self.im_plots = []
            self.im_plot_scatters = []
            self.im_plot_scatters_last = []
            self.im_plot_scatters_last_one = []

            for j in range(len(self.detector_function.value_names)):
                cmap, zrange = self.choose_MC_cmap_zrange(
                    # force the choice of clipped cmap because we are likely
                    # running an optimization
                    "cost" if self.mode == "adaptive" and j == 0 else zlabels[j],
                    zunits[j],
                )
                config_dict = {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": np.zeros([2, 2]),
                    "xlabel": slabels[0],
                    "xunit": sunits[0],
                    "ylabel": slabels[1],
                    "yunit": sunits[1],
                    "zlabel": zlabels[j],
                    "zunit": zunits[j],
                    "subplot": j + 1,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)
                self.im_plots.append(self.secondary_QtPlot.traces[-1])

                self.secondary_QtPlot.add(
                    x=[0],
                    y=[0],
                    pen=None,
                    color=1.0,
                    width=0,
                    symbol="o",
                    symbolSize=4,
                    subplot=j + 1,
                )
                self.im_plot_scatters.append(self.secondary_QtPlot.traces[-1])

                # Used to show the position of the last sampled points
                self.secondary_QtPlot.add(
                    x=[0],
                    y=[0],
                    # pen=None,
                    color=1.0,
                    width=0,
                    symbol="o",
                    symbolSize=4,
                    subplot=j + 1,
                )
                self.im_plot_scatters_last.append(self.secondary_QtPlot.traces[-1])
                self.secondary_QtPlot.add(
                    x=[0],
                    y=[0],
                    pen=None,
                    color=color_cycle[3],  # Make the last one red
                    width=0,
                    symbol="o",
                    symbolSize=7,  # and larger than the rest
                    subplot=j + 1,
                )
                self.im_plot_scatters_last_one.append(self.secondary_QtPlot.traces[-1])

    def update_plotmon_2D_interp(self, force_update=False):
        """
        Updates the interpolated 2D heatmap
        """
        if self.live_plot_enabled() and len(self.sweep_function_names) == 2:
            try:
                if (
                    time.time() - self.time_last_2Dplot_update
                    > self.plotting_interval()
                    # and avoid warning due to too little points
                    and len(self.dset) > 4
                ) or force_update:
                    # exists to force reset the x- and y-axis scale
                    new_sc = TransformState(0, 1, True)

                    x_vals = self.dset[:, 0]
                    y_vals = self.dset[:, 1]
                    for j in range(len(self.detector_function.value_names)):
                        z_ind = len(self.sweep_functions) + j
                        z_vals = self.dset[:, z_ind]

                        # Interpolate points
                        x_grid, y_grid, z_grid = interpolate_heatmap(
                            x_vals, y_vals, z_vals
                        )
                        # trace = self.secondary_QtPlot.traces[j]
                        trace = self.im_plots[j]
                        trace["config"]["x"] = x_grid
                        trace["config"]["y"] = y_grid
                        trace["config"]["z"] = z_grid
                        # force rescale the axes
                        trace["plot_object"]["scales"]["x"] = new_sc
                        trace["plot_object"]["scales"]["y"] = new_sc

                        # Mark all measured points on which the interpolation
                        # is based
                        trace = self.im_plot_scatters[j]
                        trace["config"]["x"] = x_vals
                        trace["config"]["y"] = y_vals
                        # Mark the last sampled points
                        pnts_num = 4
                        if len(x_vals) > pnts_num:
                            trace = self.im_plot_scatters_last[j]
                            trace["config"]["x"] = x_vals[-pnts_num:]
                            trace["config"]["y"] = y_vals[-pnts_num:]
                        trace = self.im_plot_scatters_last_one[j]
                        trace["config"]["x"] = x_vals[-1:]
                        trace["config"]["y"] = y_vals[-1:]

                    self.time_last_2Dplot_update = time.time()
                    self.secondary_QtPlot.update_plot()
            except Exception as e:
                log.warning(e)

    def initialize_plot_monitor_adaptive(self):
        """
        Uses the Qcodes plotting windows for plotting adaptive plot updates
        """
        if self.CMA_detected:
            self.initialize_plot_monitor_adaptive_cma()
            self.initialize_plot_monitor_2D_interp()

        else:
            self.initialize_plot_monitor()
            self.time_last_ad_plot_update = time.time()
            self.secondary_QtPlot.clear()
            self.initialize_plot_monitor_2D_interp()

            value_names = self.detector_function.value_names
            xlabels = self.sweep_par_names
            zunits = self.detector_function.value_units

            self.iter_traces = []
            self.iter_bever_traces = []
            self.iter_bever_x_traces = []

            # Because of a bug in QCoDes pytqtgraph backend we don't
            # want line plots and heatmaps in the same plotmon
            # this if statement prevents that from happening
            if len(self.sweep_functions) == 2:
                iter_plotmon = self.main_QtPlot
                iter_start_idx = len(self.sweep_functions) * len(value_names)
            else:
                iter_plotmon = self.secondary_QtPlot
                iter_start_idx = 0

            if (
                self._persist_ylabs == value_names
                and self._persist_xlabs == xlabels
                and self.persist_mode()
            ):
                persist = True
            else:
                persist = False

            # Add evolution of parameters over iterations
            xunits = self.sweep_par_units
            xlabels_num = len(xlabels)
            for k in range(xlabels_num):
                if persist:
                    yp = self._persist_dat[:, k]
                    xp = range(len(yp))
                    if len(xp) < self.plotting_max_pts():
                        iter_plotmon.add(
                            x=xp,
                            y=yp,
                            subplot=k + 1 + iter_start_idx,
                            color=0.75,  # a grayscale value
                            pen=None,
                            symbol="o",
                            symbolSize=5,
                        )
                iter_plotmon.add(
                    x=[0],
                    y=[0],
                    xlabel="iteration",
                    ylabel=xlabels[k],
                    yunit=xunits[k],
                    subplot=k + 1 + iter_start_idx,
                    symbol="o",
                    symbolSize=5,
                    color=color_cycle[2],
                )
                self.iter_traces.append(iter_plotmon.traces[-1])

                iter_plotmon.add(
                    x=[0],
                    y=[0],
                    xlabel="iteration",
                    subplot=k + 1 + iter_start_idx,
                    symbol="star",
                    symbolSize=12,
                    color=color_cycle[1],
                )
                self.iter_bever_x_traces.append(iter_plotmon.traces[-1])

            iter_plotmon.win.nextRow()

            zlables_num = len(value_names)
            for j in range(zlables_num):
                if persist:
                    yp = self._persist_dat[:, j + xlabels_num]
                    xp = range(len(yp))
                    if len(xp) < self.plotting_max_pts():
                        iter_plotmon.add(
                            x=xp,
                            y=yp,
                            subplot=xlabels_num + j + 1 + iter_start_idx,
                            color=0.75,  # a grayscale value
                            pen=None,
                            symbol="o",
                            symbolSize=5,
                        )

                iter_plotmon.add(
                    x=[0],
                    y=[0],
                    xlabel="iteration",
                    ylabel=value_names[j],
                    yunit=zunits[j],
                    subplot=xlabels_num + j + 1 + iter_start_idx,
                    symbol="o",
                    symbolSize=5,
                    color=color_cycle[j],
                )
                self.iter_traces.append(iter_plotmon.traces[-1])

                iter_plotmon.add(
                    x=[0],
                    y=[0],
                    xlabel="iteration",
                    subplot=xlabels_num + j + 1 + iter_start_idx,
                    symbol="star",
                    symbolSize=12,
                    color=color_cycle[1],
                )
                self.iter_bever_traces.append(iter_plotmon.traces[-1])

                # We want to plot a line that indicates the moving threshold
                # for the cost function when we use the `LearnerND_Minimizer` or
                # the `Learner1D_Minimizer` samplers
                if self.Learner_Minimizer_detected and j == self.par_idx:
                    iter_plotmon.add(
                        x=[0],
                        y=[0],
                        name="Thresh max priority pnts",
                        xlabel="iteration",
                        subplot=xlabels_num + j + 1 + iter_start_idx,
                        symbol="s",
                        symbolSize=3,
                        color=color_cycle[3],
                    )
                    self.iter_mv_threshold = iter_plotmon.traces[-1]

    def update_plotmon_adaptive(self, force_update=False):
        if self.CMA_detected:
            return self.update_plotmon_adaptive_cma(force_update=force_update)
        else:
            self.update_plotmon(force_update=force_update)
        if self.live_plot_enabled():
            try:
                if (
                    time.time() - self.time_last_ad_plot_update
                    > self.plotting_interval()
                    or force_update
                ):
                    sweep_functions_num = len(self.sweep_functions)
                    detector_function_num = len(self.detector_function.value_names)

                    # In case the dset is not complete yet
                    # besteval_idxs = np.array(self.adaptive_besteval_indxs)
                    len_dset = len(self.dset)
                    # besteval_idxs = besteval_idxs[besteval_idxs < len(self.dset)]
                    # Update parameters' iterations
                    for k in range(sweep_functions_num):
                        y = self.dset[:, k]
                        x = range(len_dset)
                        besteval_idxs = np.array(self.adaptive_besteval_indxs)
                        y_besteval = y[besteval_idxs]
                        self.iter_traces[k]["config"]["x"] = x
                        self.iter_traces[k]["config"]["y"] = y
                        self.iter_bever_x_traces[k]["config"]["x"] = besteval_idxs
                        self.iter_bever_x_traces[k]["config"]["y"] = y_besteval
                        self.time_last_ad_plot_update = time.time()
                    self.secondary_QtPlot.update_plot()

                    for j in range(detector_function_num):
                        y_ind = sweep_functions_num + j
                        y = self.dset[:, y_ind]
                        x = range(len_dset)
                        besteval_idxs = np.array(self.adaptive_besteval_indxs)
                        y_besteval = y[besteval_idxs]
                        iter_traces_idx = j + sweep_functions_num
                        self.iter_traces[iter_traces_idx]["config"]["x"] = x
                        self.iter_traces[iter_traces_idx]["config"]["y"] = y
                        self.iter_bever_traces[j]["config"]["x"] = besteval_idxs
                        self.iter_bever_traces[j]["config"]["y"] = y_besteval
                        if self.Learner_Minimizer_detected:
                            # We want just a line from the first pnt to the last
                            threshold = (
                                self.learner.moving_threshold
                                if self.learner.threshold is None
                                else self.learner.threshold
                            )
                            if threshold < np.inf:
                                threshold = (
                                    threshold
                                    if self.minimize_optimization
                                    else -threshold
                                )
                                self.iter_mv_threshold["config"]["x"] = [
                                    0,
                                    len_dset - 1,
                                ]
                                self.iter_mv_threshold["config"]["y"] = [
                                    threshold,
                                    threshold,
                                ]
                        self.time_last_ad_plot_update = time.time()
                    self.secondary_QtPlot.update_plot()
            except Exception as e:
                log.warning(e)
        self.update_plotmon_2D_interp(force_update=force_update)

    def initialize_plot_monitor_adaptive_cma(self):
        """
        Uses the Qcodes plotting windows for plotting adaptive plot updates
        """

        # new code
        if self.main_QtPlot.traces != []:
            self.main_QtPlot.clear()

        if self.secondary_QtPlot.traces != []:
            self.secondary_QtPlot.clear()

        self.curves = []
        self.curves_best_ever = []
        self.curves_distr_mean = []

        xlabels = self.sweep_par_names
        xunits = self.sweep_par_units
        ylabels = self.detector_function.value_names
        yunits = self.detector_function.value_units

        j = 0
        if (
            self._persist_ylabs == ylabels and self._persist_xlabs == xlabels
        ) and self.persist_mode():
            persist = True
        else:
            persist = False

        ##########################################
        # Main plotmon
        ##########################################
        for yi, ylab in enumerate(ylabels):
            for xi, xlab in enumerate(xlabels):
                if persist:  # plotting persist first so new data on top
                    yp = self._persist_dat[:, yi + len(self.sweep_function_names)]
                    xp = self._persist_dat[:, xi]
                    if len(xp) < self.plotting_max_pts():
                        self.main_QtPlot.add(
                            x=xp,
                            y=yp,
                            subplot=j + 1,
                            color=0.75,  # a grayscale value
                            symbol="o",
                            pen=None,  # makes it a scatter
                            symbolSize=5,
                        )

                self.main_QtPlot.add(
                    x=[0],
                    y=[0],
                    xlabel=xlab,
                    xunit=xunits[xi],
                    ylabel=ylab,
                    yunit=yunits[yi],
                    subplot=j + 1,
                    pen=None,
                    color=color_cycle[0],
                    symbol="o",
                    symbolSize=5,
                )
                self.curves.append(self.main_QtPlot.traces[-1])

                self.main_QtPlot.add(
                    x=[0],
                    y=[0],
                    xlabel=xlab,
                    xunit=xunits[xi],
                    ylabel=ylab,
                    yunit=yunits[yi],
                    subplot=j + 1,
                    color=color_cycle[2],
                    symbol="o",
                    symbolSize=5,
                )
                self.curves_distr_mean.append(self.main_QtPlot.traces[-1])

                self.main_QtPlot.add(
                    x=[0],
                    y=[0],
                    xlabel=xlab,
                    xunit=xunits[xi],
                    ylabel=ylab,
                    yunit=yunits[yi],
                    subplot=j + 1,
                    pen=None,
                    color=color_cycle[1],
                    symbol="star",
                    symbolSize=10,
                )
                self.curves_best_ever.append(self.main_QtPlot.traces[-1])

                j += 1
            self.main_QtPlot.win.nextRow()

        ##########################################
        # Secondary or Main plotmon
        ##########################################

        self.iter_traces = []
        self.iter_bever_traces = []
        self.iter_bever_x_traces = []
        self.iter_mean_traces = []

        # Use the secondary plot for iterations if not in 2D mode
        if len(self.sweep_functions) == 2:
            iter_plotmon = self.main_QtPlot
            plot_num = j
        else:
            iter_plotmon = self.secondary_QtPlot
            plot_num = 0

        # Add evolution of parameters over iterations
        xlabels_num = len(xlabels)
        for k in range(xlabels_num):
            if persist:
                yp = self._persist_dat[:, k]
                xp = range(len(yp))
                if len(xp) < self.plotting_max_pts():
                    iter_plotmon.add(
                        x=xp,
                        y=yp,
                        subplot=k + 1 + plot_num,
                        color=0.75,  # a grayscale value
                        pen=None,
                        symbol="o",
                        symbolSize=5,
                    )
            iter_plotmon.add(
                x=[0],
                y=[0],
                xlabel="iteration",
                ylabel=xlabels[k],
                yunit=xunits[k],
                subplot=k + 1 + plot_num,
                symbol="o",
                symbolSize=5,
                color=color_cycle[2],
            )
            self.iter_traces.append(iter_plotmon.traces[-1])

            iter_plotmon.add(
                x=[0],
                y=[0],
                xlabel="iteration",
                ylabel=xlabels[k],
                yunit=xunits[k],
                subplot=k + 1 + plot_num,
                symbol="star",
                symbolSize=12,
                color=color_cycle[1],
            )
            self.iter_bever_x_traces.append(iter_plotmon.traces[-1])

        iter_plotmon.win.nextRow()

        for j in range(len(self.detector_function.value_names)):
            if persist:
                yp = self._persist_dat[:, j + len(xlabels)]
                xp = range(len(yp))
                if len(xp) < self.plotting_max_pts():
                    iter_plotmon.add(
                        x=xp,
                        y=yp,
                        subplot=xlabels_num + plot_num + 1,
                        color=0.75,  # a grayscale value
                        symbol="o",
                        pen=None,  # makes it a scatter
                        symbolSize=5,
                    )

            iter_plotmon.add(
                x=[0],
                y=[0],
                name="Measured values",
                xlabel="iteration",
                x_unit="#",
                color=color_cycle[0],
                ylabel=ylabels[j],
                yunit=yunits[j],
                subplot=xlabels_num + plot_num + 1,
                symbol="o",
                symbolSize=5,
            )
            self.iter_traces.append(iter_plotmon.traces[-1])

            iter_plotmon.add(
                x=[0],
                y=[0],
                symbol="star",
                symbolSize=15,
                name="Best ever measured",
                color=color_cycle[1],
                xlabel="iteration",
                x_unit="#",
                ylabel=ylabels[j],
                yunit=yunits[j],
                subplot=xlabels_num + plot_num + 1,
            )
            self.iter_bever_traces.append(iter_plotmon.traces[-1])
            iter_plotmon.add(
                x=[0],
                y=[0],
                color=color_cycle[2],
                name="Generational mean",
                symbol="o",
                symbolSize=8,
                xlabel="iteration",
                x_unit="#",
                ylabel=ylabels[j],
                yunit=yunits[j],
                subplot=xlabels_num + plot_num + 1,
            )
            self.iter_mean_traces.append(iter_plotmon.traces[-1])
            plot_num += 1

        # required for the first update call to work
        self.time_last_ad_plot_update = time.time()

    def update_plotmon_adaptive_cma(self, force_update=False):
        """
        Special adaptive plotmon for
        """

        if self.live_plot_enabled():
            try:
                if (
                    time.time() - self.time_last_ad_plot_update
                    > self.plotting_interval()
                    or force_update
                ):
                    ##########################################
                    # Main plotmon
                    ##########################################
                    i = 0
                    nr_sweep_funcs = len(self.sweep_function_names)

                    # best_idx -1 as we count from 0 and best eval
                    # counts from 1.
                    best_index = int(self.opt_res_dset[-1, -1] - 1)

                    # Update parameters' iterations
                    best_evals_idx = (self.opt_res_dset[:, -1] - 1).astype(int)
                    sweep_functions_num = len(self.sweep_functions)
                    for k in range(sweep_functions_num):
                        y = self.dset[:, k]
                        x = range(len(y))
                        self.iter_traces[k]["config"]["x"] = x
                        self.iter_traces[k]["config"]["y"] = y

                        self.iter_bever_x_traces[k]["config"]["x"] = best_evals_idx
                        self.iter_bever_x_traces[k]["config"]["y"] = y[best_evals_idx]

                        self.time_last_ad_plot_update = time.time()

                    for j in range(len(self.detector_function.value_names)):
                        y_ind = nr_sweep_funcs + j

                        ##########################################
                        # Main plotmon
                        ##########################################
                        for x_ind in range(nr_sweep_funcs):

                            x = self.dset[:, x_ind]
                            y = self.dset[:, y_ind]

                            self.curves[i]["config"]["x"] = x
                            self.curves[i]["config"]["y"] = y

                            best_x = x[best_index]
                            best_y = y[best_index]
                            self.curves_best_ever[i]["config"]["x"] = [best_x]
                            self.curves_best_ever[i]["config"]["y"] = [best_y]
                            mean_x = self.opt_res_dset[:, 2 + x_ind]
                            # std_x is needed to implement errorbars on X
                            # std_x = self.opt_res_dset[:, 2+nr_sweep_funcs+x_ind]
                            # to be replaced with an actual mean
                            mean_y = self.opt_res_dset[:, 2 + 2 * nr_sweep_funcs]
                            mean_y = get_generation_means(self.opt_res_dset[:, 1], y)
                            # TODO: turn into errorbars
                            self.curves_distr_mean[i]["config"]["x"] = mean_x
                            self.curves_distr_mean[i]["config"]["y"] = mean_y
                            i += 1
                        ##########################################
                        # Secondary plotmon
                        ##########################################
                        # Measured value vs function evaluation
                        y = self.dset[:, y_ind]
                        x = range(len(y))
                        self.iter_traces[j + sweep_functions_num]["config"]["x"] = x
                        self.iter_traces[j + sweep_functions_num]["config"]["y"] = y

                        # generational means
                        gen_idx = self.opt_res_dset[:, 1]
                        self.iter_mean_traces[j]["config"]["x"] = gen_idx
                        self.iter_mean_traces[j]["config"]["y"] = mean_y

                        # This plots the best ever measured value vs iteration
                        # number of evals column
                        best_func_val = y[best_evals_idx]
                        self.iter_bever_traces[j]["config"]["x"] = best_evals_idx
                        self.iter_bever_traces[j]["config"]["y"] = best_func_val

                    self.main_QtPlot.update_plot()
                    self.secondary_QtPlot.update_plot()
                    self.update_plotmon_2D_interp(force_update=True)

                    self.time_last_ad_plot_update = time.time()

            except Exception as e:
                log.warning(e)

    def update_plotmon_2D_hard(self):
        """
        Adds latest datarow to the TwoD_array and send it
        to the QC_QtPlot.
        Note that the plotmon only supports evenly spaced lattices.
        """
        try:
            if self.live_plot_enabled():
                i = int((self.iteration) % self.ylen)
                y_ind = i
                for j in range(len(self.detector_function.value_names)):
                    z_ind = len(self.sweep_functions) + j
                    self.TwoD_array[y_ind, :, j] = self.dset[
                        i * self.xlen : (i + 1) * self.xlen, z_ind
                    ]
                    self.secondary_QtPlot.traces[j]["config"]["z"] = self.TwoD_array[
                        :, :, j
                    ]

                if (
                    time.time() - self.time_last_2Dplot_update
                    > self.plotting_interval()
                    or self.iteration == len(self.sweep_points) / self.xlen
                ):
                    self.time_last_2Dplot_update = time.time()
                    self.secondary_QtPlot.update_plot()
        except Exception as e:
            log.warning(e)

    def _set_plotting_interval(self, plotting_interval):
        if hasattr(self, "main_QtPlot"):
            self.main_QtPlot.interval = plotting_interval
            self.secondary_QtPlot.interval = plotting_interval
        self._plotting_interval = plotting_interval

    def _get_plotting_interval(self):
        return self._plotting_interval

    def clear_persitent_plot(self):
        self._persist_dat = None
        self._persist_xlabs = None
        self._persist_ylabs = None

    def update_instrument_monitor(self):
        if self.instrument_monitor() is not None:
            inst_mon = self.find_instrument(self.instrument_monitor())
            inst_mon.update()

    ##################################
    # Small helper/utility functions #
    ##################################

    def get_data_object(self):
        """
        Used for external functions to write to a datafile.
        This is used in time_domain_measurement as a hack and is not
        recommended.
        """
        return self.data_object

    def get_column_names(self):
        self.column_names = []
        self.sweep_par_names = []
        self.sweep_par_units = []

        for sweep_function in self.sweep_functions:
            self.column_names.append(
                sweep_function.parameter_name + " (" + sweep_function.unit + ")"
            )

            self.sweep_par_names.append(sweep_function.parameter_name)
            self.sweep_par_units.append(sweep_function.unit)

        for i, val_name in enumerate(self.detector_function.value_names):
            self.column_names.append(
                val_name + " (" + self.detector_function.value_units[i] + ")"
            )
        return self.column_names

    def create_experimentaldata_dataset(self):
        data_group = self.data_object.create_group("Experimental Data")
        self.dset = data_group.create_dataset(
            "Data",
            (0, len(self.sweep_functions) + len(self.detector_function.value_names)),
            maxshape=(
                None,
                len(self.sweep_functions) + len(self.detector_function.value_names),
            ),
            dtype="float64",
        )
        self.get_column_names()
        self.dset.attrs["column_names"] = h5d.encode_to_utf8(self.column_names)
        # Added to tell analysis how to extract the data
        data_group.attrs["datasaving_format"] = h5d.encode_to_utf8("Version 2")
        data_group.attrs["sweep_parameter_names"] = h5d.encode_to_utf8(
            self.sweep_par_names
        )
        data_group.attrs["sweep_parameter_units"] = h5d.encode_to_utf8(
            self.sweep_par_units
        )

        data_group.attrs["value_names"] = h5d.encode_to_utf8(
            self.detector_function.value_names
        )
        data_group.attrs["value_units"] = h5d.encode_to_utf8(
            self.detector_function.value_units
        )

    def create_experiment_result_dict(self):
        try:
            # only exists as an open dataset when running an
            # optimization
            opt_res_dset = self.opt_res_dset[()]
        except (ValueError, AttributeError) as e:
            opt_res_dset = None

        # Include best seen optimization
        opt_res = getattr(self, "opt_res", None)

        result_dict = {
            "dset": self.dset[()],
            "opt_res_dset": opt_res_dset,
            "sweep_parameter_names": self.sweep_par_names,
            "sweep_parameter_units": self.sweep_par_units,
            "value_names": self.detector_function.value_names,
            "value_units": self.detector_function.value_units,
            "opt_res": opt_res,
        }
        return result_dict

    def save_optimization_settings(self):
        """
        Saves the parameters used for optimization
        """
        opt_sets_grp = self.data_object.create_group("Optimization settings")
        param_list = dict_to_ordered_tuples(self.af_pars)
        for (param, val) in param_list:
            opt_sets_grp.attrs[param] = str(val)

    def save_cma_optimization_results(self, es):
        """
        This function is to be used as the callback when running cma.fmin.
        It get's handed an instance of an EvolutionaryStrategy (es).
        From here it extracts the results and stores these in the hdf5 file
        of the experiment.
        """
        # code extra verbose to understand what is going on
        generation = es.result.iterations
        evals = es.result.evaluations  # number of evals at start of each gen
        xfavorite = es.result.xfavorite  # center of distribution, best est
        stds = es.result.stds  # stds of distribution, stds of xfavorite
        fbest = es.result.fbest  # best ever measured
        xbest = es.result.xbest  # coordinates of best ever measured
        evals_best = es.result.evals_best  # index of best measurement

        if not self.minimize_optimization:
            fbest = -fbest

        results_array = np.concatenate(
            [[generation, evals], xfavorite, stds, [fbest], xbest, [evals_best]]
        )
        if not "optimization_result" in self.data_object["Experimental Data"].keys():
            opt_res_grp = self.data_object["Experimental Data"]
            self.opt_res_dset = opt_res_grp.create_dataset(
                "optimization_result",
                (0, len(results_array)),
                maxshape=(None, len(results_array)),
                dtype="float64",
            )

            # FIXME: Jan 2018, add the names of the parameters to column names
            self.opt_res_dset.attrs["column_names"] = h5d.encode_to_utf8(
                "generation, "
                + "evaluations, "
                + "xfavorite, " * len(xfavorite)
                + "stds, " * len(stds)
                + "fbest, "
                + "xbest, " * len(xbest)
                + "best evaluation,"
            )

        old_shape = self.opt_res_dset.shape
        new_shape = (old_shape[0] + 1, old_shape[1])
        self.opt_res_dset.resize(new_shape)
        self.opt_res_dset[-1, :] = results_array

    def save_optimization_results(self, adaptive_function, result):
        """
        Saves the result of an adaptive measurement (optimization) to
        the hdf5 file and adds it to self as well.

        Contains some hardcoded data reshufling based on known adaptive
        functions.
        """
        opt_res_grp = self.data_object.create_group("Optimization_result")

        if self.CMA_detected:
            res_dict = {
                "xopt": result[0],
                "fopt": result[1],
                "evalsopt": result[2],
                "evals": result[3],
                "iterations": result[4],
                "xmean": result[5],
                "stds": result[6],
                "stop": result[-3],
            }
            # entries below cannot be stored
            # 'cmaes': result[-2],
            # 'logger': result[-1]}
        elif is_subclass(adaptive_function, Optimizer):
            # result = learner
            # Because MC saves all the datapoints we save only the best point
            # for convenience
            opt_idx_selector = np.argmin if self.minimize_optimization else np.argmax
            opt_indx = opt_idx_selector(result.yi)
            res_dict = {"xopt": result.Xi[opt_indx], "fopt": result.yi[opt_indx]}
        elif (
            is_subclass(adaptive_function, Learner1D_Minimizer)
            or is_subclass(adaptive_function, LearnerND_Minimizer)
        ):
            # result = learner
            # Because MC saves all the datapoints we save only the best point
            # for convenience
            # Only works for a function that returns a scalar
            opt_idx_selector = np.argmin if self.minimize_optimization else np.argmax
            X = list(result.data.keys())
            Y = list(result.data.values())
            opt_indx = opt_idx_selector(Y)
            xopt = X[opt_indx]
            res_dict = {
                "xopt": np.array(xopt)
                if is_subclass(adaptive_function, LearnerND_Minimizer)
                or is_subclass(adaptive_function, Learner1D_Minimizer)
                else xopt,
                "fopt": Y[opt_indx],
            }
        elif adaptive_function.__module__ == "pycqed.measurement.optimization":
            res_dict = {"xopt": result[0], "fopt": result[1]}
        else:
            res_dict = {"opt": result}
        self.opt_res = res_dict
        h5d.write_dict_to_hdf5(res_dict, entry_point=opt_res_grp)

    def save_instrument_settings(self, data_object=None, *args):
        """
        Store the last known value of all parameters in the datafile.

        Datasaving is based on the snapshot of the QCoDeS station object.
        """
        if data_object is None:
            data_object = self.data_object
        if not hasattr(self, "station"):
            log.warning(
                "No station object specified, could not save", " instrument settings"
            )
        else:
            # This saves the snapshot of the entire setup
            snap_grp = data_object.create_group("Snapshot")
            snap = self.station.snapshot()
            exclude_keys = {
                "inter_delay",
                "post_delay",
                "vals",
                "instrument",
                "functions",
                "__class__",
                "raw_value",
                "instrument_name",
                "full_name",
                "val_mapping",
            }
            cleaned_snapshot = delete_keys_from_dict(
                # complex values are not supported in hdf5
                # converting to string avoids annoying warnings (but necessary
                # for other cases), maybe this should be done at the level of
                # `h5d.write_dict_to_hdf5` but would somewhat messy anyway as
                # there are a lot of checks related to saving and parsing
                # other types in `h5d.read_dict_from_hdf5`
                # `gen.load_settings_onto_instrument_v2` works properly as it
                # will try to evaluate a string if a parameter type is not str
                # but was saved as a string
                snap, keys=exclude_keys, types_to_str={complex})

            h5d.write_dict_to_hdf5(cleaned_snapshot, entry_point=snap_grp)

            if self.save_legacy_snapshot:
                # Below is old style saving of snapshot, exists for the sake of
                # preserving deprecated functionality
                set_grp = data_object.create_group("Instrument settings")
                inslist = dict_to_ordered_tuples(self.station.components)
                for (iname, ins) in inslist:
                    instrument_grp = set_grp.create_group(iname)
                    par_snap = ins.snapshot()["parameters"]
                    parameter_list = dict_to_ordered_tuples(par_snap)
                    for (p_name, p) in parameter_list:
                        try:
                            val = str(p["value"])
                        except KeyError:
                            val = ""
                        instrument_grp.attrs[p_name] = str(val)

    def save_MC_metadata(self, data_object=None, *args):
        """
        Save metadata on the MC (such as timings)
        """
        set_grp = data_object.create_group("MC settings")

        bt = set_grp.create_dataset("begintime", (9, 1))
        bt[:, 0] = np.array(time.localtime(self.begintime))
        pt = set_grp.create_dataset("preparetime", (9, 1))
        pt[:, 0] = np.array(time.localtime(self.preparetime))
        et = set_grp.create_dataset("endtime", (9, 1))
        et[:, 0] = np.array(time.localtime(self.endtime))

        set_grp.attrs["mode"] = self.mode
        set_grp.attrs["measurement_name"] = self.measurement_name
        set_grp.attrs["live_plot_enabled"] = self.live_plot_enabled()

    @classmethod
    def save_exp_metadata(self, metadata: dict, data_object):
        """
        Saves experiment metadata to the data file. The metadata is saved at
            file['Experimental Data']['Experimental Metadata']

        Args:
            metadata (dict):
                    Simple dictionary without nesting. An attribute will be
                    created for every key in this dictionary.
            data_object:
                    An open hdf5 data object.
        """
        if "Experimental Data" in data_object:
            data_group = data_object["Experimental Data"]
        else:
            data_group = data_object.create_group("Experimental Data")

        if "Experimental Metadata" in data_group:
            metadata_group = data_group["Experimental Metadata"]
        else:
            metadata_group = data_group.create_group("Experimental Metadata")

        h5d.write_dict_to_hdf5(metadata, entry_point=metadata_group)

    def get_percdone(self):
        percdone = (
            (self.total_nr_acquired_values)
            / (np.shape(self.get_sweep_points())[0] * self.soft_avg())
            * 100
        )
        return percdone

    def print_progress(self, stop_idx=None):
        if self.verbose():
            percdone = self.get_percdone()
            elapsed_time = time.time() - self.begintime
            progress_message = (
                "\r {percdone}% completed \telapsed time: "
                "{t_elapsed}s \ttime left: {t_left}s".format(
                    percdone=int(percdone),
                    t_elapsed=round(elapsed_time, 1),
                    t_left=round((100.0 - percdone) / (percdone) * elapsed_time, 1)
                    if percdone != 0
                    else "",
                )
            )
            if self.on_progress_callback() is not None:
                self.on_progress_callback()(percdone)
            if percdone != 100:
                end_char = ""
            else:
                end_char = "\n"
            print("\r", progress_message, end=end_char)

    def print_progress_adaptive(self):
        if self.verbose():
            acquired_points = self.dset.shape[0]

            elapsed_time = time.time() - self.begintime
            progress_message = (
                "\rAcquired {acquired_points} points, \telapsed time: "
                "{t_elapsed}s".format(
                    acquired_points=acquired_points, t_elapsed=round(elapsed_time, 1)
                )
            )
            end_char = ""
            print("\r", progress_message, end=end_char)

    def is_complete(self):
        """
        Returns True if enough data has been acquired.
        """
        acquired_points = self.dset.shape[0]
        total_nr_pts = np.shape(self.get_sweep_points())[0]
        if acquired_points < total_nr_pts:
            return False
        elif acquired_points >= total_nr_pts:
            if self.soft_avg() != 1 and self.soft_iteration == 0:
                return False
            else:
                return True

    def print_measurement_start_msg(self):
        if self.verbose():
            if len(self.sweep_functions) == 1:
                print("Starting measurement: %s" % self.get_measurement_name())
                print("Sweep function: %s" % self.get_sweep_function_names()[0])
                print("Detector function: %s" % self.get_detector_function_name())
            else:
                print("Starting measurement: %s" % self.get_measurement_name())
                for i, sweep_function in enumerate(self.sweep_functions):
                    print("Sweep function %d: %s" % (i, self.sweep_function_names[i]))
                print("Detector function: %s" % self.get_detector_function_name())

    def get_datetimestamp(self):
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def get_datawriting_start_idx(self):
        if self.mode == "adaptive":
            max_sweep_points = np.inf
        else:
            max_sweep_points = np.shape(self.get_sweep_points())[0]

        start_idx = int(self.total_nr_acquired_values % max_sweep_points)

        self.soft_iteration = int(self.total_nr_acquired_values // max_sweep_points)

        return start_idx

    def get_datawriting_indices_update_ctr(self, new_data, update: bool = True):
        """
        Calculates the start and stop indices required for
        storing a hard measurement.

        N.B. this also updates the "total_nr_acquired_values" counter.
        """

        # This is the case if the detector returns a simple float or int
        if len(np.shape(new_data)) == 0:
            xlen = 1
        # This is the case for a 1D hard detector or an N-D soft detector
        elif len(np.shape(new_data)) == 1:
            # Soft detector (returns values 1 by 1)
            if len(self.detector_function.value_names) == np.shape(new_data)[0]:
                xlen = 1
            else:  # 1D Hard detector (returns values in chunks)
                xlen = len(new_data)
        else:
            if self.detector_function.detector_control == "soft":
                # FIXME: this is an inconsistency that should not be there.
                xlen = np.shape(new_data)[1]
            else:
                # in case of an N-D Hard detector dataset
                xlen = np.shape(new_data)[0]

        start_idx = self.get_datawriting_start_idx()
        stop_idx = start_idx + xlen

        if update:
            # Sometimes one wants to know the start/stop idx without
            self.total_nr_acquired_values += xlen

        return start_idx, stop_idx

    ####################################
    # Non-parameter get/set functions  #
    ####################################

    def set_sweep_function(self, sweep_function):
        """
        Used if only 1 sweep function is set.
        """
        # If it is not a sweep function, assume it is a qc.parameter
        # and try to auto convert it it
        if not isinstance(sweep_function, swf.Sweep_function):
            sweep_function = wrap_par_to_swf(sweep_function)
        self.sweep_functions = [sweep_function]
        self.set_sweep_function_names([str(sweep_function.name)])

    def get_sweep_function(self):
        return self.sweep_functions[0]

    def set_sweep_functions(self, sweep_functions):
        """
        Used to set an arbitrary number of sweep functions.
        """
        sweep_function_names = []
        for i, sweep_func in enumerate(sweep_functions):
            # If it is not a sweep function, assume it is a qc.parameter
            # and try to auto convert it it
            if not hasattr(sweep_func, "sweep_control"):
                sweep_func = wrap_par_to_swf(sweep_func)
                sweep_functions[i] = sweep_func
            sweep_function_names.append(str(sweep_func.name))
        self.sweep_functions = sweep_functions
        self.set_sweep_function_names(sweep_function_names)

    def get_sweep_functions(self):
        return self.sweep_functions

    def set_sweep_function_names(self, swfname):
        self.sweep_function_names = swfname

    def get_sweep_function_names(self):
        return self.sweep_function_names

    def set_detector_function(self, detector_function, wrapped_det_control="soft"):
        """
        Sets the detector function. If a parameter is passed instead it
        will attempt to wrap it to a detector function.
        """
        if not hasattr(detector_function, "detector_control"):
            detector_function = wrap_par_to_det(detector_function, wrapped_det_control)
        self.detector_function = detector_function
        self.set_detector_function_name(detector_function.name)

    def get_detector_function(self):
        return self.detector_function

    def set_detector_function_name(self, dfname):
        self._dfname = dfname

    def get_detector_function_name(self):
        return self._dfname

    ################################
    # Parameter get/set functions  #
    ################################

    def get_git_hash(self):
        self.git_hash = get_git_revision_hash()
        return self.git_hash

    def get_measurement_begintime(self):
        self.begintime = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def get_measurement_endtime(self):
        self.endtime = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def get_measurement_preparetime(self):
        self.preparetime = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def set_sweep_points(self, sweep_points):
        self.sweep_points = np.array(sweep_points)
        # line below is because some sweep funcs have their own sweep points
        # attached
        # This is a mighty bad line! Should be adding sweep points to the
        # individual sweep funcs
        if len(np.shape(sweep_points)) == 1:
            self.sweep_functions[0].sweep_points = np.array(sweep_points)

    def get_sweep_points(self):
        if hasattr(self, "sweep_points"):
            return self.sweep_points
        else:
            return self.sweep_functions[0].sweep_points

    def set_adaptive_function_parameters(self, adaptive_function_parameters):
        """
        adaptive_function_parameters: Dictionary containing options for
            running adaptive mode.

        The following arguments are reserved keywords. All other entries in
        the dictionary get passed to the adaptive function in the measurement
        loop.

        Reserved keywords:
            "adaptive_function":    function
            "x_scale": (array)     rescales sweep parameters for
                adaptive function, defaults to None (no rescaling).
                Each sweep_function/parameter is rescaled by dividing by
                the respective component of x_scale.
            "minimize": True        Bool, inverts value to allow minimizing
                                    or maximizing
            "f_termination" None    terminates the loop if the measured value
                                    is smaller than this value
            "par_idx": 0            If a parameter returns multiple values,
                                    specifies which one to use.

        Common keywords (used in python nelder_mead implementation):
            "x0":                   list of initial values
            "initial_step"
            "no_improv_break"
            "maxiter"
        """
        self.af_pars = adaptive_function_parameters

        # x_scale is expected to be an array or list.
        self.x_scale = self.af_pars.pop("x_scale", None)
        self.par_idx = self.af_pars.pop("par_idx", 0)

        # [2020-03-07] these flags were moved in the loop in measure_soft_adaptive
        # # Determines if the optimization will minimize or maximize
        # self.minimize_optimization = self.af_pars.pop("minimize", True)
        # self.f_termination = self.af_pars.pop("f_termination", None)

        module_name = get_module_name(self.af_pars.get("adaptive_function", self))
        self.CMA_detected = module_name == "cma.evolution_strategy"

        # ensures the cma optimization results are saved during the experiment
        if self.CMA_detected and "callback" not in self.af_pars:
            self.af_pars["callback"] = self.save_cma_optimization_results

    def get_adaptive_function_parameters(self):
        return self.af_pars

    def set_measurement_name(self, measurement_name):
        if measurement_name is None:
            self.measurement_name = "Measurement"
        else:
            self.measurement_name = measurement_name

    def get_measurement_name(self):
        return self.measurement_name

    def set_optimization_method(self, optimization_method):
        self.optimization_method = optimization_method

    def get_optimization_method(self):
        return self.optimization_method

    def clean_previous_adaptive_run(self):
        """
        Performs a reset of variables and parameters used in the previous run
        that are not relevant or even conflicting with the current one.
        """
        self.learner = None
        self.Learner_Minimizer_detected = False
        self.CMA_detected = False
        self.af_pars = dict()

    def choose_MC_cmap_zrange(self, zlabel: str, zunit: str):
        cost_func_names = ["cost", "cost func", "cost function"]
        cmap = None
        zrange = None
        cmaps = self.plotmon_2D_cmaps
        zranges = self.plotmon_2D_zranges

        # WARNING!!! If this ever gives problems see `__init__.py` in `pycqed`
        # module folder

        if cmaps and zlabel in cmaps.keys():
            cmap = cmaps[zlabel]
        elif zunit == "%":
            cmap = "hot"
        elif zunit.lower() == "deg":
            cmap = "anglemap45"
        elif np.any(np.array(cost_func_names) == zlabel.lower()):
            cmap = (
                "inferno_clip_high"
                if hasattr(self, "minimize_optimization")
                and not self.minimize_optimization
                else "inferno_clip_low"
            )
        else:
            cmap = "viridis"

        if zranges and zlabel in zranges.keys():
            zrange = zranges[zlabel]
        elif zunit.lower() == "deg":
            zrange = (0.0, 360.0)

        return cmap, zrange

    ################################
    # Actual parameters            #
    ################################

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {
            "vendor": "PycQED",
            "model": "MeasurementControl",
            "serial": "",
            "firmware": "2.0",
        }
