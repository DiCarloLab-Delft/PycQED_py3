import copy
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

#from neupy.algorithms import GRNN as grnn
from sklearn.neural_network import MLPRegressor as mlpr
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans as KMeans
from neupy.algorithms import GRNN as grnn
try:
	import tensorflow as tf
except Exception:
	logging.warning('Could not import tensorflow')


class Estimator(metaclass=ABCMeta):
    '''
    BASIC ESTIMATOR CLASS

    Defines the basic functionality of an estimator.
    -Independent of Regression or Classification,
     every estimator instance should hold a fitting method, a prediction method
     and a evaluate method.
    -Data input vectors should be standardized to be column vectors as
     most preimplemented classes use this as a standard.

    Paramters:
    -- pre_processing_dict: dictionary of data-specific information used for
       preprocessing. E.g the relevant information for data centering
       (mean values, data range). For the moment this is handled outside the
       classifiers as not every implementation (i.e tensorflow) has simple way
       of integrating this.
       -The data used for fitting every estimator should in general be centered
        (mean 0) and scaled to the unit interval ([0,1] or [-1,1]). For regression
        models this also holds for the target data.
       -In order to retrieve the correct output values distribution after fitting
        to the input distribution, it is then required to rescale the values obtained
        from the prediction method of the estimator using the values passed in the
        estimators pre_processing_dict field.
    -- score: The value obtained from evaluating a dataset obtained from the
       same distribution as the data used for training.(E.g by using a train-test-split)
    -- _name: The internal name of this estimator.
    -- _type: General type of the estimator. E.g Regression, classification
    '''
    def __init__(self,name='estimator',pre_proc_dict=None,type=None):
        self.pre_proc_dict = pre_proc_dict
        self.score = None
        self._name = name
        self._type = type

    @abstractmethod
    def fit(self,data,target):
        '''
        data: Data drawn from the distribution to be fitted by this estimator
        target: data from the target distribution in supervised models
        '''
        pass

    @abstractmethod
    def fit(self,data):
        '''
        data: Data drawn from the distribution to be fitted by this estimator
        '''
        pass

    @abstractmethod
    def predict(self,data):
        '''
        data: Data drawn from the fitted distribution to be predicted
        '''
        pass

    @abstractmethod
    def evaluate(self,data,target):
        '''
        -Evaluating the preformance of the estimator by comparing the predictions
         of data with the target values with some reasonable distance measure.

        data: Data drawn from the fitted distribution to be predicted
        target: The target values associated with the data input.
        '''
        pass

class MLP_Regressor_scikit(Estimator):
    '''
    Ordinary neural network implementation from scikit-learn.

    Hyperparameters for this estimator:
            regularization_coefficient: L1 regularization multiplier.
                                        0. --> regularization disabled.
            hidden_layers: network architecture. An input of [10,10] corresponds
                           to a network with two hidden layers with 10 nodes each.
                           additional to this, the network has input and output
                           layers corresponding to the number of input features
                           and output parameters. This is determined from the
                           input training data.
            activation_function: Activation function used throughout the network
                                 possible functions are: 'relu'(default)
                                                         'logistic'
                                                         'tanh' ...
    '''
    def __init__(self,hyper_parameter_dict,output_dim=1,n_feature=1,
                 pre_proc_dict=None):

        super().__init__(name='MLP_Regressor_scikit',pre_proc_dict=pre_proc_dict,
                         type='Regressor')
        self.hyper_parameter_dict = hyper_parameter_dict
        self.output_dim = output_dim
        self.n_feature = n_feature

        self.extract_hyper_params_from_dict()
        self.mlpr_ = mlpr(solver='lbfgs',
                          hidden_layer_sizes=self._hidden_layers,
                          activation=self.activation,
                          alpha=self.alpha,
                          max_iter=5000)
        self.score = -np.infty

    def extract_hyper_params_from_dict(self):
        self._hidden_layers= self.hyper_parameter_dict.pop('hidden_layers',[10])
        self._hidden_layers= tuple(self._hidden_layers)
        self.alpha = self.hyper_parameter_dict.pop('regularization_coefficient',0.5)
        self.activation = self.hyper_parameter_dict.pop('activation_function','relu')


    def fit(self, x_train, y_train):

        self.mlpr_.fit(x_train, y_train)
        self.score = self.evaluate(x_train,y_train)
        print('MLP_Regressor scikit trained with '+
              self.score+' accuracy on training set.')

    def predict(self, x_pred):
        '''
        Has to be callable by scipy optimizers such as fmin(). I.e input has
        has to be wrapped to a list for the estimators predict method.
        '''
        return self.mlpr_.predict(x_pred)

    def evaluate(self,x,y):
        self.score = self.mlpr_.score(x,y)
        return self.score

    def print_score(self):
        print("Training score of ANN: "+str(self.score))

class DNN_Regressor_tf(Estimator):
    '''
    Ordinary neural network implementation in Tensorflow.
        -loss function used: squared loss l(y,y_pred)=||y - y_pred||**2
        -accuracy measure: coefficient of determination:
         R_acc = 1-sum((y-y_pred)**2)/sum((y-mean(y))**2)
        -optimizer: tf.GradientDescentOptimizer

        Hyperparameters for this estimator:
            learning_rate: learning rate for gradient descent
            regularization_coefficient: L1 regularization multiplier.
                                        0. --> regularization disabled.
            hidden_layers: network architecture. An input of [10,10] corresponds
                           to a network with two hidden layers with 10 nodes each.
                           additional to this, the network has input and output
                           layers corresponding to the number of input features
                           and output parameters. This is defined by n_feature
                           and output_dim.
            learning_steps: Number of gradient descents performed. Can potentially
                            be used for early stopping applications.


    '''
    def __init__(self,hyper_parameter_dict,output_dim=1,n_feature = 1,
                 pre_proc_dict = None):

        super().__init__(name='DNN_Regressor_tf',pre_proc_dict=pre_proc_dict,
                         type='Regressor')
        self.hyper_parameter_dict=hyper_parameter_dict
        self._n_feature = n_feature
        self._output_dim = output_dim
        self._session = tf.Session()
        self.learning_acc = []
        self.extract_hyper_params_from_dict()

    def extract_hyper_params_from_dict(self):
        self._hidden_layers= self.hyper_parameter_dict.pop('hidden_layers',[10])
        self.alpha = self.hyper_parameter_dict.pop('learning_rate',0.5)
        self.beta = self.hyper_parameter_dict.pop('regularization_coefficient',0.)
        self.iters = self.hyper_parameter_dict.pop('learning_steps',200)

    def get_stddev(self, inp_dim, out_dim):
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
        # x_train,y_train, \
        x = tf.placeholder(tf.float32, [None, self._n_feature])
        y = tf.placeholder(tf.float32, [None, self._output_dim])
        logits, reg_terms = self.network(x)

        loss = self.loss(logits, y) + tf.reduce_sum(reg_terms)
        print(loss)
        print(self.alpha)
        train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

        self._x = x
        self._y = y
        self._logits = logits

        accuracy = self.evaluate_np(logits,y)  #used for learning curve creation

        init = tf.initialize_all_variables()
        self._session.run(init)

        # plt.figure()
        # plt.grid(True)
        # plt.title('learning curve')       ## Learning Curve plotting
        # plt.xlabel('learning epoch')
        # plt.ylabel('loss')
        for i in range(self.iters):
            self._session.run(train_op,feed_dict={x: x_train, y: y_train})
            _acc = self._session.run(accuracy, feed_dict={x: x_train, y: y_train})
            self.learning_acc.append([i, _acc])
        self.score = self.learning_acc[-1]
        # plt.plot(range(self.iters),learning_progress,'go')
        # plt.show()

    def loss(self, logits_test, y_test):
        _tmp = tf.square(y_test-logits_test)
        _loss = tf.reduce_sum(_tmp) / tf.reduce_sum(tf.square(y_test-tf.reduce_mean(y_test)))
        return _loss

    def evaluate_np(self,logits_test,y_test):
        _accuracy = 1 - \
                    tf.reduce_sum(tf.square(y_test-logits_test)) \
                    /tf.reduce_sum(tf.square(y_test-tf.reduce_mean(y_test)))
        return _accuracy

    def evaluate(self,logits_test=np.array([]),y_test=np.array([])):
        self.score = 1 - \
            np.linalg.norm((y_test-logits_test)**2) \
            /np.linalg.norm((y_test-np.linalg.norm(y_test)**2))
        return self.score


    def predict(self, samples):
        predictions = self._logits
        #return self._session.run(predictions, {self._x: [samples]})
        return self._session.run(predictions, {self._x: samples})

class Polynomial_Regression(Estimator):
    '''
    Estimator for a Polynomial regression with degre as a hyperparameter

        Hyperparameters for this estimator:
            polynomail_dimension: degree of the regression polynomial

    '''
    def __init__(self,hyper_parameter_dict,pre_proc_dict=None):

        super().__init__(name='Polynomial_Regression_scikit',
                         pre_proc_dict=pre_proc_dict,type='Regressor')
        self._polyReg = None
        self.extract_hyper_params_from_dict()

    def extract_hyper_params_from_dict(self):
        self.ndim = self.hyper_parameter_dict.pop('polynomial_dimension',1)

    def poly_features_transform(self,X):
        if X.ndim==1:
            data_shape = [len(X),1]
        else:
            data_shape = np.shape(X)
        #so far does not support fits with mixed polynomials
        A = np.zeros((data_shape[0],1+self.ndim*data_shape[1]))
        A[:,0] = np.ones((data_shape[0]))
        for it in range(self.ndim):
            A[:,1+it*data_shape[1]:1+(it+1)*data_shape[1]] = X**(it+1)
        return A

    def fit(self,x_train,y_train):
        self._polyReg = LinearRegression(fit_intercept=False)
        self._polyReg.fit(self.poly_features_transform(x_train),y_train)

        #x = np.transpose(np.array([np.linspace(-1,1,50)]))
        # plt.figure()
        # plt.plot(np.linspace(-1,1,50),
        #          self.predict(x)
        #          )
        # plt.plot(x_train[:,0], y_train, 'o')
        # plt.show()

    def predict(self,samples):
        if not isinstance(samples,np.ndarray):
            samples = np.array(samples)
        return self._polyReg.predict(self.poly_features_transform(samples))

    def evaluate(self,x,y):
        pred = self.predict(x)
        self.score= 1. - np.linalg.norm(pred-y)**2 \
                   / np.linalg.norm(y-np.mean(y,axis=0))**2
        return self.score


class GRNN_neupy(Estimator):
    '''
    Generalized Regression Neural Network implementation from neupy
    If the training data target values are multidimensional, for every paramter
    in the target data, a new GRNN is initialized and trained.
        Hyperparameters for this estimator:

            gamma: list of scaling factors for the standard dev. input.
                   1.--> use std (or -if None- the regular std dev of
                   the input data)
    '''
    def __init__(self,hyper_paramter_dict,verbose =False,
                 pre_proc_dict=None):

        super().__init__(name='GRNN_neupy',pre_proc_dict=pre_proc_dict,
                       type='Regressor')
        self.hyper_parameter_dict = hyper_paramter_dict
        self.extract_hyper_params_from_dict()
        self._verbose = verbose
        self._grnn = None

    def extract_hyper_params_from_dict(self):
        self._std= self.hyper_parameter_dict.pop('standard_deviations',None)
        self._gamma = self.hyper_parameter_dict.pop('std_scaling',[1.])
        if not isinstance(self._gamma,list):
            self._gamma = [self._gamma]

    def fit(self,x_train,y_train):
        if not isinstance(x_train,np.ndarray):
            x_train = np.array(x_train)
            if x_train.ndim == 1:
                x_train.reshape((np.size(x_train),x_train.ndim))
        if not isinstance(y_train,np.ndarray):
            y_train = np.array(y_train)
            if y_train.ndim == 1:
                y_train.reshape((np.size(y_train),y_train.ndim))
        if len(self._gamma) != y_train.ndim:
            logging.warning('Hyperparameter gamma contains only '
                            +str(len(self._gamma))+
                            ' values while there are '+str(y_train.ndim)+ ' output'
                            ' dimensions. Missing values are set to the value for'
                            'the first parameter!')
            while len(self._gamma) <= y_train.ndim:
                self._gamma.append(self._gamma[0])

        if self._std is None:
            std_x = 0.
            for it in range(x_train.ndim):
                std_x += np.std(x_train[:,it])
            self._std = std_x/x_train.ndim
        self._grnn = []
        for it in range(y_train.ndim):
            new_grnn =grnn(std=self._gamma[it]*self._std)
            print('GRNN initialized with std: ',self._std)
            new_grnn.train(x_train,y_train[:,it])
            self._grnn.append(new_grnn)

    def predict(self,samples):

        if not isinstance(samples,np.ndarray):
            samples = np.array(samples)
        predictions = []
        for it in range(len(self._grnn)):
            predictions.append(self._grnn[it].predict(samples))
        return np.array(predictions)

    def evaluate(self,x,y):
        pred = self.predict(x)
        self.score = 1. - np.linalg.norm(pred-y)**2 \
                   / np.linalg.norm(y-np.mean(y,axis=0))**2
        return self.score

class CrossValidationEstimator(Estimator):
    '''
    Estimator wrapper performing a n_fold Cross Validation

    Hyperparameters for this estimator:
        cv_n_fold : Number ov splittings of the training data
                    E.g: for cv_n_fold = 5. the training data is split into 5
                         equally sized partitions of which 4 are used for
                         training and one for validation. The roles of the sets
                         then switch until the estimator was tested on all
                         partitions
    '''



    def __init__(self,hyper_parameter_dict,estimator : Estimator):
        super().__init__(name='CV_Estimator_Wrapper',
                         type='Wrapper')
        self.estimator = estimator
        self.hyper_parameter_dict=hyper_parameter_dict
        self.extract_hyper_params_from_dict()
        self.pre_proc_dict = self.estimator.pre_proc_dict
        self.gen_error_emp = None
        self.batch_errors = None

    def extract_hyper_params_from_dict(self):
        self.n_fold= self.hyper_parameter_dict.pop('cv_n_fold',1)

    def fit(self,x_train,y_train):
        if not isinstance(x_train,np.ndarray):
            x_train = np.array(x_train)
            if x_train.ndim == 1:
                x_train.reshape((np.size(x_train),x_train.ndim))
        if not isinstance(y_train,np.ndarray):
            y_train = np.array(y_train)
            if y_train.ndim == 1:
                y_train.reshape((np.size(y_train),y_train.ndim))
        sample_number = np.shape(x_train)[0]
        if sample_number != np.shape(y_train)[0]:
            logging.error('training and target values have different first dimension'
                          '. Sample number missmatch.')
        reminder = sample_number % self.n_fold
        batch_size = int((sample_number-reminder)/self.n_fold)
        self.batch_errors = []
        for it in range(0,sample_number-reminder,batch_size):

            test_batch = x_train[it:it+batch_size-1,:]
            train_batch = np.concatenate((x_train[:it,:],x_train[it+batch_size:,:]),
                                         axis=0)
            test_target = y_train[it:it+batch_size-1,:]
            train_target = np.concatenate((y_train[:it,:],y_train[it+batch_size:,:]),
                                          axis=0)
            self.estimator.fit(train_batch,train_target)
            self.batch_errors.append(self.estimator.evaluate(test_batch,test_target))


        self.estimator.fit(x_train,y_train) #refit estimator on training data
        self.score = np.mean(self.batch_errors)
        # if self.score <= 0.8:
        #     logging.warning('Cross Validation finished with average score: ',self.score,
        #                     '. Most likely unstable predictions. Try larger training data set.')
        self.std_error_emp = np.std(self.batch_errors)

    def predict(self,data):
        if not isinstance(data,np.ndarray):
            data = np.array(data)
        return self.estimator.predict(data)

    def evaluate(self,x,y):
        return self.estimator.evaluate(x,y)

    def get_std_error(self):
        return self.std_error_emp



class K_means_scikit(Estimator):

    def __init__(self,hyper_parameter_dictionary,pre_proc_dict=None):

        super().__init__(name='K_means_scikit',type='Clustering',
                         pre_proc_dict=pre_proc_dict)
        self.hyper_parameter_dictionary = hyper_parameter_dictionary
        self.extract_hyper_params_from_dict()
        self.cluster_centers = None
        self.labels = None

    def extract_hyper_params_from_dict(self):
        self.n_clusters= self.hyper_parameter_dict.pop('cluster_number',1)

    def fit(self,X_train):

        if not isinstance(X_train,np.ndarray):
            x_train = np.array(X_train)
            if x_train.ndim == 1:
                x_train.reshape((np.size(X_train),X_train.ndim))
        self._kmeans = KMeans(n_clusters=self.n_clusters)
        self._kmeans.fit(X_train)
        self.labels = self._kmeans.labels_
        self.cluster_centers = self._kmeans.cluster_centers_

    def predict(self,data):
        return self._kmeans.predict(data)

    def evaluate(self,data):
        return self._kmeans.score(data)

    def get_labels(self):
        return self.labels