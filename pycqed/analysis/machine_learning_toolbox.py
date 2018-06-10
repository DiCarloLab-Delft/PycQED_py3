import copy
import math
import logging
import numpy as np
from abc import ABCMeta, abstractmethod

#from neupy.algorithms import GRNN as grnn
from sklearn.model_selection import GridSearchCV as gcv, train_test_split
from sklearn.neural_network import MLPRegressor as mlpr
#from neupy.algorithms import GRNN as grnn
import tensorflow as tf


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

    def __init__(self,hidden_layers=[10],output_dim=1,n_feature=1, alpha = 0.5,
                 activation = ['relu'],pre_proc_dict=None):

        super.__init__(name='MLP_Regressor_scikit',pre_proc_dict=pre_proc_dict,
                       type='Regressor')
        self._hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.n_feature = n_feature
        self.alpha = alpha
        self.activation = activation
        self.mlpr_ = mlpr(solver='lbfgs')
        self.parameter_dict = {'hidden_layer_sizes': self._hidden_layers,
                               'alpha': self.alpha,
                               'activation': self.activation}
        self.gridCV = gcv(self.mlpr_,self.parameter_dict,cv=5)
        self.bestParams = None

    def fit(self, x_train, y_train):

        self.gridCV.fit(x_train, y_train)
        self.bestParams = self.gridCV.best_params_
        self.score = self.gridCV.best_score_


    def predict(self, x_pred):
        '''
        Has to be callable by scipy optimizers such as fmin(). I.e input has
        has to be wrapped to a list for the estimators predict method.
        '''
        return self.gridCV.predict(x_pred)

    def evaluate(self,x,y):
        pred = self.gridCV.predict(x)
        acc = 1. - np.linalg.norm(pred-y)**2 \
                   / np.linalg.norm(y-np.mean(y,axis=0))**2
        return acc

    def print_best_params(self):
        print("Best parameters: "+str(self.bestParams))
        print("Best CV score of ANN: "+str(self.score))

class DNN_Regressor_tf(Estimator):
    '''
        alpha: learning rate for gradient descent
        beta: L1 regression multiplier. 0. --> regression disabled
        -loss function used: squared loss l(y,y_pred)=||y - y_pred||**2
        -accuracy measure: coefficient of determination:
         R_acc = 1-sum((y-y_pred)**2)/sum((y-mean(y))**2)
        -optimizer: tf.GradientDescentOptimizer
    '''
    def __init__(self, hidden_layers=[10],output_dim=1, alpha = 0.5,
                 beta=1., n_feature = 1, iters = 200, pre_proc_dict = None):

        super.__init__(name='DNN_Regressor_tf',pre_proc_dict=pre_proc_dict,
                       type='Regressor')
        self._n_feature = n_feature
        self._hidden_layers = hidden_layers
        self._output_dim = output_dim
        self._session = tf.Session()
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.learning_acc = []

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
        x_train,y_train, \
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
        for i in range(self.iters):
            self._session.run(train_op,feed_dict={x:x_train, y: y_train})
            _acc = self._session.run(accuracy, feed_dict={self._x: x_train, self._y: y_train})
            self.learning_acc.append(_acc)
        self.score = self.learning_acc[-1]
        # plt.plot(range(self.iters),learning_progress,'go')
        # plt.show()

    def evaluate(self,logits_test,y_test):
        _accuracy = 1 - \
                    tf.reduce_sum(tf.square(y_test-logits_test)) \
                    /tf.reduce_sum(tf.square(y_test-tf.reduce_mean(y_test)))
        return _accuracy


    def predict(self, samples):
        predictions = self._logits
        #return self._session.run(predictions, {self._x: [samples]})
        return self._session.run(predictions, {self._x: samples})

class GRNN_neupy(Estimator):
    '''
    Generalized Regression Neural Network implementation from neupy
        gamma: scaling factor for the standard dev. input.
               1.--> use std (or -if None- the regular std dev of the input data)
    '''
    def __init__(self,std=None,gamma=1.,verbose =False,
                 pre_proc_dict=None):

        super.__init__(name='MLP_Regressor_scikit',pre_proc_dict=pre_proc_dict,
                       type='Regressor')
        self._std = std
        self._gamma = gamma
        self._verbose = verbose
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
                std_x += np.std(x_train[:,it])
            std_x = self._gamma*std_x/x_train.ndim
            self._std = std_x
        else:
            self._std = self._gamma*self._std
        self._grnn = grnn(self._std)
        self._grnn.train(x_train,y_train)

    def predict(self,samples):

        if not isinstance(samples,np.ndarray):
            samples = np.array(samples)
        return self._grnn.predict(samples)

    def evaluate(self,x,y):
        pred = self._grnn.predict(x)
        acc = 1. - np.linalg.norm(pred-y)**2 \
                   / np.linalg.norm(y-np.mean(y,axis=0))**2
        return acc


    def evaluate(self,x,y):

        pred = self._grnn.predict(x)
        acc = 1. - np.linalg.norm(pred-y)**2 \
                   /np.linalg.norm(y-np.mean(y,axis=0))**2
        return acc
