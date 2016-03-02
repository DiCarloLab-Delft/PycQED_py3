import copy
import numpy as np


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
        alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

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
