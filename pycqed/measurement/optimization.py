import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from pycqed.analysis import machine_learning_toolbox as ml
except Exception:
    logger.warning('Machine learning packages not loaded. '
                   'Run from pycqed.analysis import machine_learning_toolbox to see errors.')

from sklearn.model_selection import GridSearchCV as gcv, train_test_split

from scipy.optimize import fmin_l_bfgs_b,fmin,minimize,fsolve

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
         p=0.5, ctrl_min=0.,ctrl_max=np.pi,
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
        res.append([x, score])

    # once the loop is broken evaluate the final value one more time as
    # verification
    fun(res[0][0])
    return res[0]

def generate_new_training_set(new_train_values, new_target_values,
                              training_grid=None, target_values=None):
    if training_grid is None:
        training_grid =new_train_values
        target_values = new_target_values

    else:
        if np.shape(new_train_values)[1] != np.shape(training_grid)[1] or \
           np.shape(new_target_values)[1] != np.shape(target_values)[1]:
            print('Shape missmatch between new training values and existing ones!'
                  ' Returning None.')
            return None,None

        training_grid = np.append(training_grid,new_train_values,axis=0)
        target_values = np.append(target_values,new_target_values,axis=0)

    return training_grid,target_values

def center_and_scale(X_in,y_in):
    '''
    Preprocessing of Data. Mainly transform the data to mean 0 and interval [-1,1]
    :param X: training data list of parameters (each equally long). Standing vector!
    :param y: validation data list of parameters (each equally long).Standing vector!
    :output:
        :X: rescaled and centered training data
        :y: rescaled and centered test data
        :input_feature_means: mean values of initial training data parameters
        :output_feature_means: mean values of initial validation data parameters
        :input_feature_ext: abs(max-min) of initial training data parameters
        :output_feature_ext: abs(max-min) of initial validation data parameters
    '''

    if not isinstance(X_in,np.ndarray):
        X_in = np.array(X_in)
    if X_in.ndim == 1:
        X_in.shape = (np.size(X_in),X_in.ndim)
        #X_in.reshape((np.size(X_in),X_in.ndim))
    if not isinstance(y_in,np.ndarray):
        y_in= np.array(y_in)
    if y_in.ndim == 1:
        #y_in.reshape((np.size(y_in),y_in.ndim))
        y_in.shape = (np.size(y_in),y_in.ndim)

    X = copy.deepcopy(X_in)
    y = copy.deepcopy(y_in)
    input_feature_means = np.zeros(np.size(X,1))       #saving means of training
    output_feature_means = np.zeros(np.size(y,1))     #and target features
    input_feature_ext= np.zeros(np.size(X,1))
    output_feature_ext = np.zeros(np.size(y,1))

    if  np.size(X,1)==1:
        input_feature_means= [np.mean(X)]
        input_feature_ext = [np.max(X) \
                             -np.min(X)]
        X -= input_feature_means  #offset to mean 0
        X /= input_feature_ext    #rescale to [-1,1]

    else:
        for it in range(np.size(X,1)):
            input_feature_means[it]= np.mean(X[:,it])
            input_feature_ext[it] = np.max(X[:,it]) \
                                    -np.min(X[:,it])
            X[:,it] -= input_feature_means[it]  #offset to mean 0
            X[:,it] /= input_feature_ext[it]    #rescale to [-1,1]
    if np.size(y,1) == 1:
        output_feature_means= [np.mean(y)]
        output_feature_ext = [np.max(y) \
                              -np.min(y)]
        y -= output_feature_means #offset to mean 0
        y /= output_feature_ext   #rescale to [-1,1]
    else:
        for it in range(np.size(y,1)):
            output_feature_means[it]= np.mean(y[:,it])
            output_feature_ext[it] = np.max(y[:,it]) \
                                     -np.min(y[:,it])
            y[:,it] -= output_feature_means[it] #offset to mean 0
            y[:,it] /= output_feature_ext[it]   #rescale to [-1,1]
    return X,y,\
           input_feature_means,input_feature_ext,\
           output_feature_means,output_feature_ext


def neural_network_opt(fun, training_grid, target_values = None,
                       estimator='GRNN_neupy',hyper_parameter_dict=None,
                       x_init = None):
    """
    parameters:
        fun:           Function that can be used to get data points if None,
                       target_values have to be provided instead.
        training_grid: The values on which to train the Neural Network. It
                       contains features as column vectors of length as the
                       number of datapoints in the training set.
        target_values: The target values measured during data acquisition by a
                       hard sweep over the traning grid.
        estimator: The estimator used to model the function mapping the
                   training_grid on the target_values.
        hyper_parameter_dict: if None, the default hyperparameters
                              of the selected estimator are used. Should contain
                              estimator dependent hyperparameters such as hidden
                              layer sizes for a neural network. See
                              <machine_learning_toolbox> for specific
                              information on available estimators.
        x_ini: Initial values for the minimization of the fitted function.
    output:
        optimal points where network is minimized.
        est: estimator instance representing the trained model. Consists of a
             predict(X) method, which computes the network response for a given
             input value X.
    """
    ###############################################################
    ###          create measurement data from test_grid         ###
    ###############################################################
    #get input dimension, training grid contains parameters as row(!!) vectors
    if len(np.shape(training_grid)) == 1:
        training_grid = np.transpose(np.array([training_grid]))
    n_samples = np.size(training_grid,0)
    print('Nr Samples: ', n_samples)
    n_features = np.size(training_grid,1)
    print('Nr Features: ', n_features)

    if fun is None:
        output_dim = np.size(target_values,1)
    else:
        #if the sweep is adaptive, acquire data points by applying fun
        first_value = fun(training_grid[0])
        output_dim = np.size(first_value)
        target_values = np.zeros((n_samples,output_dim))
        target_values[0,:] = first_value
        for i in range(1,n_samples):
            target_values[i,:]=fun(training_grid[i])

    #Preprocessing of Data. Mainly transform the data to mean 0 and interval [-1,1]
    training_grid_centered,target_values_centered,\
    input_feature_means,input_feature_ext,\
    output_feature_means,output_feature_ext \
                 = center_and_scale(training_grid,target_values)
    #Save the preprocessing information in order to be able to rescale the values later.
    pre_processing_dict ={'output': {'scaling': output_feature_ext,
                                     'centering':output_feature_means},
                          'input': {'scaling': input_feature_ext,
                                    'centering':input_feature_means}}

    ##################################################################
    ### initialize grid search cross val with hyperparameter dict. ###
    ###    and MLPR instance and fit a model functione to fun()     ###
    ##################################################################
    def mlpr():
        est = ml.MLP_Regressor_scikit(hyper_parameter_dict,
                                      output_dim=output_dim,
                                      n_feature=n_samples,
                                      pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered, np.ravel(target_values_centered))
        est.print_best_params()
        return est

    def dnnr():
        est = ml.DNN_Regressor_tf(hyper_parameter_dict,
                               output_dim=output_dim,
                               n_feature=n_features,
                               pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered,target_values_centered)
        return est

    def grnn():
        est = ml.GRNN_neupy(hyper_parameter_dict,
                            pre_proc_dict=pre_processing_dict)
        cv_est = ml.CrossValidationEstimator(hyper_parameter_dict,est)
        cv_est.fit(training_grid_centered,target_values_centered)
        return cv_est

    def polyreg():
        est = ml.Polynomial_Regression(hyper_parameter_dict,
                                       pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered,target_values_centered)
        return est

    estimators = {'MLP_Regressor_scikit': mlpr, #defines all current estimators currently implemented
                  'DNN_Regressor_tf': dnnr,
                  'GRNN_neupy': grnn,
                  'Polynomial_Regression_scikit': polyreg}

    est = estimators[estimator]()       #create and fit instance of the chosen estimator

    def estimator_wrapper(X):
        pred = est.predict([X])
        print('pred: ', pred)
        if output_dim == 1.:
            return np.abs(pred+1.)
        else:
            pred = pred[0]
            norm = 0.
            for it in range(len(pred)):
                print(it)
                if it == 0:
                    w = 1
                else:
                    w = 1
                norm += w*np.abs(pred[it] + 1.)
            output = norm
            print('norm: ', norm)
            print('')

            return output

    ###################################################################
    ###     perform gradient descent to minimize modeled landscape  ###
    ###################################################################
    if x_init is None:
        x_init = np.zeros(n_features)
        #The data is centered. No values above -1,1 should be encountered
        bounds=[(-1.,1.) for i in range(n_features)]
        res = fmin_l_bfgs_b(estimator_wrapper, x_init, bounds=bounds,
                            approx_grad=True)
    else:
        print('x_init minimizer:', x_init)
        for it in range(n_features):
            x_init[it] = (x_init[it]-input_feature_means[it])/input_feature_ext[it] # scale initial value
        bounds=[(-1.,0.5) for i in range(n_features)]
        res = fmin_l_bfgs_b(estimator_wrapper, x_init, approx_grad=True, bounds=bounds)
        # res = minimize(estimator_wrapper, x_init, method='Nelder-Mead')
        # res = [res.x]
        print('result:', res)
        # print(res)


    # result = res.x
    result = res[0]
    opti_flag = True

    for it in range(n_features):
        if not opti_flag:
            break
        if np.abs(result[it]) >= 2*np.std(training_grid_centered, 0)[it]:
            opti_flag = False
    if not opti_flag:
        print('optimization most likely failed. Results outside 2-sigma surrounding'
              'of at least one data feature mean value! Values will still be updated.')
    # Rescale values
    amp = est.predict([result])[0]
    print('amp: ', amp)
    if output_dim == 1.:
        amp = amp*output_feature_ext+output_feature_means
    else:
        for it in range(output_dim):
            amp[it] = amp[it]*output_feature_ext[it]+output_feature_ext[it]
    for it in range(n_features):
        result[it] = result[it]*input_feature_ext[it]+input_feature_means[it]
    print('minimization results: ', result, ' :: ', amp)

    return np.array(result), est,opti_flag




def gradient(fun,x,grid_spacing):
    """
    this computes the gradient of fun() using finite differences
    :param fun: function of which to compute the gradient
    :param x: evaluation point for the gradient
    :param grid_spacing: displacement in forward finite difference, has to be
                         provided for every feature in x
    :return: returns gradient value grad(fun)(x) computed by finite differences
    """

    grad = np.zeros(len(x))
    for iter in range(len(x)):
        x_h = np.array(x)
        #using forward difference here
        x_h[iter] += grid_spacing[iter]
        #compute finite difference
        grad[iter]=(fun(x_h)-fun(x))/grid_spacing[iter]
    return grad


def gradient_descent(fun, x_ini,grid_spacing,lamb_ini=1, max_iter=500 ):
    """
    :param fun: function to be minimized
    :param x_ini: initial point for the iteration 
    :param grid_spacing: spacing between points on the discretized evaluation 
                         grid. has to provide a spacing for every feature in x_ini.
    :param lamb_ini: initial learning rate. 
    :param tol: tolerance to determine convergence. 
    :param max_iter: maximum iterations permitted until gradient descent fails
    
    :return: input values that minimizes fun()

    Note: using Barzilai-Borwein adaptive step lengths for second derivative approx.
    """
    iter = 0
    #determine the tolerance as 1e-4 times the norm of the step size vector
    tol = np.linalg.norm(grid_spacing)*1e-4
    #perform first step
    x = np.array(x_ini)
    grad = gradient(fun,x,grid_spacing)
    s = -grad/lamb_ini         #go downhill
    diff = np.linalg.norm(s)   #defines the stepsize taken. If < tol, extremum was found
    x_new = np.array(x+s)

    while diff > tol and iter < max_iter:
        #central loop, makes gradient update and performs step with adaptive
        #learning rate.
        grad_new = gradient(fun,x_new,grid_spacing)
        y = grad_new - grad
        grad = np.array(grad_new)
        lamb = np.linalg.norm(y)**2/np.dot(y,s)
        s = -grad_new/lamb
        x = np.array(x_new)
        x_new = np.array(x+s)
        diff = np.linalg.norm(s)
        iter += 1
        if(diff <= tol and iter < max_iter):
            print("gradient descent finished after "+str(iter)+"iterations")
            print("minimum found at: "+str(x_new))
        elif(iter>=max_iter):
            print("Warning: gradient descent did not finish within \
                       max iterations!")
    return x_new,[diff,iter]
