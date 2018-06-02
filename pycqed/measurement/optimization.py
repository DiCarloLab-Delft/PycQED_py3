import copy
import math
import logging
import numpy as np

#from neupy.algorithms import GRNN as grnn
from sklearn.model_selection import GridSearchCV as gcv, train_test_split
from sklearn.neural_network import MLPRegressor as mlpr
import tensorflow as tf

from scipy.optimize import fmin

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

def center_and_scale(X,y):
    '''
    Preprocessing of Data. Mainly transform the data to mean 0 and interval [-1,1]
    :param X: training data list of parameters (each equally long)
    :param y: validation data list of parameters (each equally long)
    :output:
        :X: rescaled and centered training data
        :y: rescaled and centered test data
        :input_feature_means: mean values of initial training data parameters
        :output_feature_means: mean values of initial validation data parameters
        :input_feature_ext: abs(max-min) of initial training data parameters
        :output_feature_ext: abs(max-min) of initial validation data parameters
    '''
    input_feature_means = np.zeros(np.size(X,1))       #saving means of training
    output_feature_means = np.zeros(y.ndim)     #and target features
    input_feature_ext= np.zeros(np.size(X,1))
    output_feature_ext = np.zeros(y.ndim)

    for it in range(np.size(X,1)):
        input_feature_means[it]= np.mean(X[:,it])
        X[:,it] -= input_feature_means[it]  #offset to mean 0
        input_feature_ext[it] = np.max(X[:,it]) \
                                -np.min(X[:,it])
        X[:,it] /= input_feature_ext[it]    #rescale to [-1,1]
    for it in range(y.ndim):
        output_feature_means[it]= np.mean(y)
        y -= output_feature_means[it] #offset to mean 0
        output_feature_ext[it] = np.max(y) \
                                 -np.min(y)
        y /= output_feature_ext[it]   #rescale to [-1,1]

    return X,y,input_feature_means,input_feature_ext,\
           output_feature_means,output_feature_ext


def neural_network_opt(training_grid, target_values ,hidden_layer_sizes = [(5,)],
                       alphas= 0.0001, solver='lbfgs',estimator='MLPRegressor',
                       iters = 200, beta=1.,gamma=1.):
    """
    parameters:
        fun: Function to be optimized. So far this is only an optimization for
             multivariable functions with scalar return value due to gradient
             descent implementation

        training_grid: The values on which to train the Neural Network. It
                       contains features as column vectors of length as the
                       number of datapoints in the training set.

        hidden_layer_sizes: List of tuples containing the number of units for every
                            hidden layer of the network. E.g (5,5) would be a
                            network with two hidden layers with 5 units each.
                            Crossvalidation is used to determine the best
                            network architecture within this list.

        alphas: List of values for the learning parameter alpha (learning rate)

        solver: optimization function used for the gradient descent during the
                learning process. 'adam' is the default solver of MLPRegressor

    output: returns the optimized feature vector X, minimizing fun(X).
    """
    ###############################################################
    ###          create measurement data from test_grid         ###
    ###############################################################
    if not isinstance(hidden_layer_sizes,list):
        hidden_layer_sizes = [hidden_layer_sizes]

    if not isinstance(alphas,list):
        alphas = [alphas]
    #transform input into array
    training_grid = np.transpose(training_grid)
    target_values = np.transpose(target_values)
    #get input dimension, training grid contains parameters as row!! vectors
    n_samples = np.size(training_grid,0)
    n_features = np.size(training_grid,1)
    output_dim = target_values.ndim

    #Preprocessing of Data. Mainly transform the data to mean 0 and interval [0,1]
    training_grid,target_values,\
    input_feature_means,input_feature_ext,\
    output_feature_means,output_feature_ext \
        = center_and_scale(training_grid,target_values)

    ##################################################################
    ### initialize grid search cross val with hyperparameter dict. ###
    ###    and MLPR instance and fit a model functione to fun()     ###
    ##################################################################
    def mlpr():
        est = MLP_Regressor_scikit(hidden_layers=hidden_layer_sizes,
                                   output_dim=output_dim,
                                   n_feature=n_samples,
                                   alpha=alphas)
        est.fit(training_grid, target_values)
        est.print_best_params()
        return est

    def dnnr():
        est = DNN_Regressor_tf(hidden_layers=hidden_layer_sizes,
                               output_dim=output_dim,
                               n_feature=n_samples,
                               alpha=alphas,
                               iters = iters,
                               beta = beta)
        est.fit(training_grid,target_values)
        return est

    def grnn():
        est = GRNN_neupy(gamma=gamma)
        est.fit(training_grid,target_values)
        return est


    estimators = {'MLP_Regressor_scikit': mlpr, #defines all current estimators currently implemented
                  'DNN_Regressor_tf': dnnr,
                  'GRNN_neupy': grnn}

    est = estimators[estimator]()       #create and fit instance of the chosen estimator
    ###################################################################
    ###     perform gradient descent to minimize modeled landscape  ###
    ###################################################################
    x_ini = np.zeros(n_features)
    res = fmin(est.predict, x_ini, full_output=True)
    result = res[0]
    #Rescale values
    amp = res[1] * output_feature_ext + output_feature_means
    result[0] = result[0]*input_feature_ext[0]+input_feature_means[0]
    result[1] = result[1]*input_feature_ext[1]+input_feature_means[1]
    print('minimization results: ',result,'::',amp)

    ## testing plots
#     x_mesh = np.linspace(-1.,1,200)
#     y_mesh = np.linspace(-1.,1.,200)
#     Xm,Ym = np.meshgrid(x_mesh,y_mesh)
#     Zm = np.zeros_like(Xm)
#     for k in range(np.size(x_mesh)):
#         for l in range(np.size(y_mesh)):
#             Zm[k,l] = gridCV.predict([[Xm[k,l],Ym[k,l]]])
#     Zm = Zm*output_feature_ext + output_feature_means
#     Xm = Xm*input_feature_ext[0] +input_feature_means[0]
#     Ym = Ym*input_feature_ext[1] +input_feature_means[1]
#     import matplotlib.pyplot as plt
#     plt.figure()
#     levels = np.linspace(0,0.06,30)
#     CP = plt.contourf(Xm,Ym,Zm,levels,extend='both')
#     plt.plot(result[0],result[1],'co',label='network minimum')
#     plt.tick_params(axis='both',which='minor',labelsize=14)
#     plt.ylabel('ch2 offset [V]',fontsize=20)
#     plt.xlabel('ch1 offset [V]',fontsize=20)
# #    plt.xlim(-0.2,3)
#     cbar = plt.colorbar(CP)
#     cbar.ax.set_ylabel('Magnitude [V]',fontsize=20)
#     plt.show()

    return np.array(result),est
    # return [np.array(result), np.array(amp,dtype='float32')]
    #mght want to adapt alpha,ect.i


class MLP_Regressor_scikit:

    def __init__(self,hidden_layers=[10],output_dim=1,n_feature=1, alpha = 0.5,
                 activation = ['relu']):
        self._n_feature = n_feature
        self._hidden_layers = hidden_layers
        self._output_dim = output_dim
        self.alpha = alpha
        self.activation = activation
        self.mlpr_ = mlpr(solver='lbfgs')
        self.parameter_dict = {'hidden_layer_sizes': self._hidden_layers,
                          'alpha': self.alpha,
                          'activation': self.activation}
        self.gridCV = gcv(self.mlpr_,self.parameter_dict,cv=5)
        self.train_input = None
        self.train_valid = None
        self.bestParams = None
        self.score = None

    def fit(self, x_train, y_train):
        if self.train_input is None:
            self.train_input = x_train
            self.train_valid = y_train
        else:
            logging.warning('< MLP_Regressor_scikit > has already been trained!'
                            're-training estimator on new input data!')
        self.gridCV.fit(x_train, y_train)
        self.bestParams = self.gridCV.best_params_
        self.score = self.gridCV.best_score_


    def predict(self, x_pred):
        '''
        Has to be callable by scipy optimizers such as fmin(). I.e input has
        has to be wrapped to a list for the estimators predict method.
        '''
        return self.gridCV.predict([x_pred])

    def print_best_params(self):
        print("Best parameters: "+str(self.bestParams))
        print("Best CV score of ANN: "+str(self.score))

class DNN_Regressor_tf:
    '''
        alpha: learning rate for gradient descent
        beta: L1 regression multiplier. 0. --> regression disabled
    '''
    def __init__(self, hidden_layers=[10],output_dim=1, alpha = 0.5,
                 beta=1., n_feature = 1, iters = 200):

        self._n_feature = n_feature
        self._hidden_layers = hidden_layers
        self._output_dim = output_dim
        self._session = tf.Session()
        self.alpha = alpha
        self.beta = beta
        self.iters = iters

    def get_stddev(self,inp_dim, out_dim):
        std = 1.3 / math.sqrt(float(inp_dim) + float(out_dim))
        return std

    def network(self, x):
        x = tf.cast(x,tf.float32)
        hidden = []
        regularizer = tf.contrib.layers.l1_regularizer(self.beta)
        reg_terms = []
        #input layer. Input does not need to be transformed as we are regressing
        with tf.name_scope("input"):
            weights = tf.Variable(tf.truncated_normal([self._n_feature, self._hidden_layers[0]],
                                                      stddev=self.get_stddev(self._n_feature,
                                                                             self._hidden_layers[0])),
                                                      name='weights')
            biases = tf.Variable(tf.zeros([self._hidden_layers[0]]), name='biases')
            input_ = tf.matmul(x, weights) + biases
            reg_terms.append(tf.contrib.layers.apply_regularization(regularizer,[weights,biases]))

        #hidden layers
        for ind, size in enumerate(self._hidden_layers):
            if ind == len(self._hidden_layers) - 1: break
            with tf.name_scope("hidden{}".format(ind+1)):
                weights = tf.Variable(tf.truncated_normal([size, self._hidden_layers[ind+1]],
                                                          stddev=self.get_stddev(self._n_feature, self._hidden_layers[ind+1])), name='weights')
                biases = tf.Variable(tf.zeros([self._hidden_layers[ind+1]]), name='biases')
                inputs = input_ if ind == 0 else hidden[ind-1]
                hidden.append(tf.nn.relu(tf.matmul(inputs,weights)+biases,name="hidden{}".format(ind+1)))
                reg_terms.append(tf.contrib.layers.apply_regularization(regularizer,[weights,biases]))

    #output layer
        with tf.name_scope("output"):
            weights =  tf.Variable(tf.truncated_normal([self._hidden_layers[-1],self._output_dim],
                                                       stddev=self.get_stddev(self._hidden_layers[-1],self._output_dim)),name='weights')
            biases = tf.Variable(tf.zeros([self._output_dim]),name='biases')
            logits = tf.matmul(hidden[-1],weights)+biases               #regression model. Select linear act. fct.
            reg_terms.append(tf.contrib.layers.apply_regularization(regularizer,[weights,biases]))

        return logits, reg_terms

    def fit(self,x_train=None,y_train=None):
        if x_train is not None:
            logging.warning('< DNN_Regressor_tf > has already been trained!'
                            're-training estimator on new input data!')
        x = tf.placeholder(tf.float32, [None, self._n_feature])
        y = tf.placeholder(tf.float32, [None, self._output_dim])
        logits ,reg_terms = self.network(x)
        loss = self.loss(logits,y) + reg_terms
        train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

        self._x = x
        self._y = y
        self._logits = logits

        accuracy = self.evaluate(logits,y)  #used for learning curve creation

        init = tf.initialize_all_variables()
        self._session.run(init)

        # plt.figure()
        # plt.grid(True)
        # plt.title('learning curve')       ## Learning Curve plotting
        # plt.xlabel('learning epoch')
        # plt.ylabel('loss')
        learning_progress = []
        for i in range(self.iters):
            print('test')
            self._session.run(train_op,feed_dict={x:x_train, y: y_train})
            _acc = self._session.run(accuracy, feed_dict={self._x: x_train, self._y: y_train})
            learning_progress.append(_acc)
        self.learning_acc = learning_progress
        # plt.plot(range(self.iters),learning_progress,'go')
        # plt.show()

    def evaluate(self,logits_test,y_test):
        _accuracy = 1 - \
                    tf.reduce_sum(tf.square(y_test-logits_test))\
                   /tf.reduce_sum(tf.square(y_test-tf.reduce_mean(y_test)))
        return _accuracy


    def predict(self, samples):
        predictions = self._logits
        return self._session.run(predictions, {self._x: [samples]})

class GRNN_neupy:
    '''
    Generalized Regression Neural Network implementation from neupy
        gamma: scaling factor for the standard dev. input.
               1.--> use std (or -if None- the regular std dev of the input data)
    '''
    def __init__(self,std=None,gamma=1.,verbose =False):
        self._std = std
        self._gamma = gamma
        self._verbose = verbose
        self.score = None
        self._grnn = None

    def fit(self,x_train,y_train):
        if not isinstance(x_train,np.ndarray):
            x_train = np.array(x_train)
            if x_train.ndim == 1:
                x_train.reshape((np.size(x_train),x_train.ndim))
        if not isinstance(y_train,np.ndarray):
            y_train = np.array(y_train)
            if y_train.ndim == 1:
                y_train.reshape((np.size(y_train),y_train.ndim))
        if self._std is None:
            std_x = 0.
            for it in x_train.ndim:
                std_x+= np.std(x_train[:,it])
            std_x = self._gamma*std_x/x_train.ndim
            self._std = std_x
        self._grnn = grnn(self._std)
        self._grnn.train(x_train,y_train)

    def predict(self,samples):

        if not isinstance(samples,np.ndarray):
            x_train = np.array(samples)
        return self._grnn.predict(samples)

    def evaluate(self,x,y):

        pred = self._grnn.predict(x)
        acc = 1. - np.linalg.norm(pred-y)**2  \
                   /np.linalg.norm(y-np.mean(y,axis=0))
        return acc



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
