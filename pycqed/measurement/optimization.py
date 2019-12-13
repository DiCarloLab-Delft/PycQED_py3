import copy
import numpy as np
import logging
import collections
from skopt import Optimizer
from adaptive.utils import cache_latest
from adaptive.notebook_integration import ensure_holoviews
from adaptive.learner.base_learner import BaseLearner

log = logging.getLogger(__name__)


def nelder_mead(fun, x0,
                initial_step=0.1,
                no_improve_thr=10e-6, no_improv_break=10,
                maxiter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                verbose=False):
    '''
    parameters:
        fun (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x0
        x0 (numpy array): initial position
        initial_step (float/np array): determines the stepsize to construct
            the initial simplex. If a float is specified it uses the same
            value for all parameters, if an array is specified it uses
            the specified step for each parameter.

        no_improv_thr,  no_improv_break (float, int): break after
            no_improv_break iterations with an improvement lower than
            no_improv_thr
        maxiter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        alpha (float): reflection coefficient
        gamma (float): expansion coefficient
        rho (float): contraction coefficient
        sigma (float): shrink coefficient
            For details on these parameters see Wikipedia page

    return: tuple (best parameter array, best score)


    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Implementation from https://github.com/fchollet/nelder-mead, edited by
    Adriaan Rol for use in PycQED.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    '''
    # init
    x0 = np.array(x0)  # ensures algorithm also accepts lists
    dim = len(x0)
    prev_best = fun(x0)
    no_improv = 0
    res = [[x0, prev_best]]
    if type(initial_step) is float:
        initial_step_matrix = np.eye(dim)*initial_step
    elif (type(initial_step) is list) or (type(initial_step) is np.ndarray):
        if len(initial_step) != dim:
            raise ValueError('initial_step array must be same lenght as x0')
        initial_step_matrix = np.diag(initial_step)
    else:
        raise TypeError('initial_step ({})must be list or np.array'.format(
                        type(initial_step)))

    for i in range(dim):
        x = copy.copy(x0)
        x = x + initial_step_matrix[i]
        score = fun(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after maxiter
        if maxiter and iters >= maxiter:
            # Conclude failure break the loop
            if verbose:
                print('max iterations exceeded, optimization failed')
            break
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            # Conclude success, break the loop
            if verbose:
                print('No improvement registered for {} rounds,'.format(
                      no_improv_break) + 'concluding succesful convergence')
            break

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = fun(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = fun(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = fun(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = fun(redx)
            nres.append([redx, score])
        res = nres

    # once the loop is broken evaluate the final value one more time as
    # verification
    fun(res[0][0])
    return res[0]


def SPSA(fun, x0,
         initial_step=0.1,
         no_improve_thr=10e-6, no_improv_break=10,
         maxiter=0,
         gamma=0.101, alpha=0.602, a=0.2, c=0.3, A=300,
         p=0.5, ctrl_min=0., ctrl_max=np.pi,
         verbose=False):
    '''
    parameters:
        fun (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x0
        x0 (numpy array): initial position


        no_improv_thr,  no_improv_break (float, int): break after
            no_improv_break iterations with an improvement lower than
            no_improv_thr
        maxiter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        alpha, gamma, a, c, A, (float): parameters for the SPSA gains
            (see refs for definitions)
        p (float): probability to get 1 in Bernoulli +/- 1 distribution
            (see refs for context)
        ctrl_min, ctrl_max (float/array): boundaries for the parameters.
            can be either a global boundary for all dimensions, or a
            numpy array containing the boundary for each dimension.
    return: tuple (best parameter array, best score)

    alpha, gamma, a, c, A and p, are parameters for the algorithm.
    Their function is described in the references below,
    and even optimal values have been discussed in the literature.


    Pure Python/Numpy implementation of the SPSA algorithm designed by Spall.
    Implementation from http://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF,
    edited by Ramiro Sagastizabal for use in PycQED.
    Reference: http://www.jhuapl.edu/SPSA/Pages/References-Intro.htm
    '''
    # init
    x0 = np.array(x0)  # ensures algorithm also accepts lists
    dim = len(x0)
    prev_best = fun(x0)
    no_improv = 0
    res = [[x0, prev_best]]

    x = copy.copy(x0)

    # SPSA iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after maxiter
        if maxiter and iters >= maxiter:
            # Conclude failure break the loop
            if verbose:
                print('max iterations exceeded, optimization failed')
            break
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            # Conclude success, break the loop
            if verbose:
                print('No improvement registered for {} rounds,'.format(
                      no_improv_break) + 'concluding succesful convergence')
            break
        # step 1
        a_k = a/(iters+A)**alpha
        c_k = c/iters**gamma
        # step 2
        delta = np.where(np.random.rand(dim) > p, 1, -1)
        # step 3
        x_plus = x+c_k*delta
        x_minus = x-c_k*delta
        y_plus = fun(x_plus)
        y_minus = fun(x_minus)
        # res.append([x_plus, y_plus])
        # res.append([x_minus, y_minus])
        # step 4
        gradient = (y_plus-y_minus)/(2.*c_k*delta)
        # step 5
        x = x-a_k*gradient
        x = np.where(x < ctrl_min, ctrl_min, x)
        x = np.where(x > ctrl_max, ctrl_max, x)
        score = fun(x)
        log.warning("SPSA: Evaluated gradient at x_minus={};x_plus={}".format(x_minus,
                                                                          x_plus))
        log.warning("SPSA: y_minus={};y_plus={}".format(y_plus,
                                                    y_minus))
        log.warning("SPSA: Gradient={}".format(gradient))
        log.warning("SPSA: Jump={};new_x={}".format(a_k*gradient, x))
        res.append([x, score])

    # once the loop is broken evaluate the final value one more time as
    # verification
    fun(res[0][0])
    return res[0]


class SKOptLearnerND(Optimizer, BaseLearner):
    """
    [Victor 2019-12-04]
    This is an modification of the original
    ``adaptive.learner.skopt_learner.SKOptLearner``
    It is here because the original one uses set() and was not
    compatible with the SKOpt optimizer that expects list()
    Original docstring below
    --------------------------------------------------------------------

    Learn a function minimum using ``skopt.Optimizer``.

    This is an ``Optimizer`` from ``scikit-optimize``,
    with the necessary methods added to make it conform
    to the ``adaptive`` learner interface.

    Parameters
    ----------
    function : callable
        The function to learn.
    **kwargs :
        Arguments to pass to ``skopt.Optimizer``.
    """

    def __init__(self, function, **kwargs):
        self.function = function
        self.pending_points = set()
        self.data = collections.OrderedDict()
        super().__init__(**kwargs)

    def tell(self, x, y, fit=True):
        if isinstance(x, collections.abc.Iterable):
            self.pending_points.discard(tuple(x))
            self.data[tuple(x)] = y
            super().tell(x, y, fit)
        else:
            self.pending_points.discard(x)
            self.data[x] = y
            super().tell([x], y, fit)

    def tell_pending(self, x):
        # 'skopt.Optimizer' takes care of points we
        # have not got results for.
        self.pending_points.add(tuple(x))

    def remove_unfinished(self):
        pass

    @cache_latest
    def loss(self, real=True):
        if not self.models:
            return np.inf
        else:
            model = self.models[-1]
            # Return the in-sample error (i.e. test the model
            # with the training data). This is not the best
            # estimator of loss, but it is the cheapest.
            return 1 - model.score(self.Xi, self.yi)

    def ask(self, n, tell_pending=True):
        if not tell_pending:
            raise NotImplementedError(
                "Asking points is an irreversible "
                "action, so use `ask(n, tell_pending=True`."
            )
        points = super().ask(n)
        # TODO: Choose a better estimate for the loss improvement.
        if self.space.n_dims > 1:
            return points, [self.loss() / n] * n
        else:
            return [p[0] for p in points], [self.loss() / n] * n

    @property
    def npoints(self):
        """Number of evaluated points."""
        return len(self.Xi)

    def plot(self, nsamples=200):
        hv = ensure_holoviews()
        if self.space.n_dims > 1:
            raise ValueError("Can only plot 1D functions")
        bounds = self.space.bounds[0]
        if not self.Xi:
            p = hv.Scatter([]) * hv.Curve([]) * hv.Area([])
        else:
            scatter = hv.Scatter(([p[0] for p in self.Xi], self.yi))
            if self.models:
                model = self.models[-1]
                xs = np.linspace(*bounds, nsamples)
                xsp = self.space.transform(xs.reshape(-1, 1).tolist())
                y_pred, sigma = model.predict(xsp, return_std=True)
                # Plot model prediction for function
                curve = hv.Curve((xs, y_pred)).opts(style=dict(line_dash="dashed"))
                # Plot 95% confidence interval as colored area around points
                area = hv.Area(
                    (xs, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma),
                    vdims=["y", "y2"],
                ).opts(style=dict(alpha=0.5, line_alpha=0))

            else:
                area = hv.Area([])
                curve = hv.Curve([])
            p = scatter * curve * area

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (bounds[1] - bounds[0])
        plot_bounds = (bounds[0] - margin, bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

    def _get_data(self):
        return [x[0] for x in self.Xi], self.yi

    def _set_data(self, data):
        xs, ys = data
        self.tell_many(xs, ys)

# ######################################################################
# Some utilities
# ######################################################################


def multi_targets_phase_offset(target, spacing, phase_name: str = None):
    """
    Intended to be used in cost functions that targets several phases
    at the same time equaly spaced

    Args:
        target(float): unit = deg, target phase for which the output
            will be zero

        spacing(float): unit = deg, spacing > 0, spacing to other phases
            for which the output will be zero

        phase_name(str): if specified a string version of the function
            will be returned, inteded for the construction of a custom
            cost lambda function including other terms, e.g. some
            other target phase using this funtion.
            NB: numpy needs to be defined as "np" in the file the string
            function will be executed
    """
    target = np.asarray(target)
    if phase_name is not None:
        string = 'np.min([({phase_name} - {target}) % {spacing}, (360 - ({phase_name} - {target})) % {spacing}])'
        return string.format(
            phase_name=phase_name,
            target=str(target),
            spacing=str(spacing))
    else:
        return lambda phase: np.min([(phase - target) % spacing, (360 - (phase - target)) % spacing], axis=0)
