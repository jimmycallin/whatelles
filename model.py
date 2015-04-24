"""
Most of the code taken form here: http://deeplearning.net/tutorial/
"""

import theano
import theano.tensor as T
import numpy as np


class NNPrediction():

    def __init__(self, config):
        self.config = config
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.n_epochs = config.get('n_epochs', 50)
        self.rng = np.random.RandomState(self.config.get('seed', 1))
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.batch_size = self.config.get('batch_size', 500)
        self.validation_improvement_threshold = self.config.get('validation_improvement_threshold', 0.995)
        self.min_iterations = self.config.get('min_iterations', 10000)
        self.index = T.lscalar()
        self.classifier = MLP(rng=self.rng,
                              input=self.x,
                              n_in=self.config['n_in'],
                              n_hiddens=self.config['n_hiddens'],
                              n_out=self.config['n_out'],
                              activation_function=self.config['activation_function'],
                              cost_function=self.config['cost_function'])

        self.cost = (self.classifier.calculate_cost(self.y)
                     + self.config.get('L1_reg', 0) * self.classifier.L1
                     + self.config.get('L2_reg', 0.0001) * self.classifier.L2_sqr)

        gparams = [T.grad(self.cost, param) for param in self.classifier.params]

        self.updates = [(param, param - self.learning_rate * gparam)
                        for param, gparam in zip(self.classifier.params, gparams)]

    def _initialize_train_model(self, train_set_x, train_set_y):

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        shared_x = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(train_set_y, dtype=theano.config.floatX),
                                 borrow=True)

        shared_y = T.cast(shared_y, 'int32')

        batch_interval = slice(self.index * self.batch_size, (self.index + 1) * self.batch_size)
        train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: shared_x[batch_interval],
                self.y: shared_y[batch_interval]
            }
        )

        return train_model

    def _initialize_test_model(self, train_set_x, train_set_y):
        shared_x = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(train_set_y, dtype=theano.config.floatX),
                                 borrow=True)

        shared_y = T.cast(shared_y, 'int32')
        batch_interval = slice(self.index * self.batch_size, (self.index + 1) * self.batch_size)
        test_model = theano.function(inputs=[self.index],
                                     outputs=self.classifier.errors(self.y),
                                     givens={self.x: shared_x[batch_interval],
                                             self.y: shared_y[batch_interval]})
        return test_model

    def train(self, train_set_x, train_set_y, validation_set_x=None, validation_set_y=None):
        """
        Trains the classifier given a training set.
        If given a validation set, validate the improvement of the model every :validation_frequency:th time.
            If no improvements have happened for a while, abort the training early.
        """

        train_model = self._initialize_train_model(train_set_x, train_set_y)

        validation_model = None
        if validation_set_x is not None and validation_set_y is not None:
            validation_model = self._initialize_test_model(validation_set_x, validation_set_y)
            best_error_rate = np.inf

        n_train_batches = len(train_set_x) // self.batch_size
        epoch = 0
        iteration = 0
        best_iteration = None
        break_early = False

        if validation_model is not None:
            patience = self.min_iterations
            validation_frequency = min(n_train_batches, patience // 2)
        else:
            patience = self.n_epochs * len(train_set_x)

        for epoch in range(self.n_epochs):
            if break_early:
                break
            print("Training epoch {}".format(epoch))
            for minibatch_index in range(n_train_batches):
                iteration = epoch * n_train_batches + minibatch_index
                train_model(minibatch_index)
                if validation_model is not None and (iteration + 1) % validation_frequency == 0:
                    error_rate = self._evaluate(validation_model, self.batch_size, len(validation_set_x))
                    print("Validation error rate: {}, epoch {}, minibatch {}".format(error_rate,
                                                                                     epoch,
                                                                                     minibatch_index))
                    if error_rate < best_error_rate:
                        if error_rate < best_error_rate * self.validation_improvement_threshold:
                            patience = max(patience, iteration * 2)
                        best_error_rate = error_rate
                        best_iteration = iteration

                if patience <= iteration:
                    break_early = True
                    break

        print("Finished training model.")
        if validation_model is not None:
            print("Best validation error rate: {} on iteration {}".format(best_error_rate, best_iteration))

    def predict(self, test_set_x, test_set_y):
        test_model = self._initialize_test_model(test_set_x, test_set_y)
        predictions = self._evaluate(test_model, self.batch_size, len(test_set_x))
        return predictions

    def _evaluate(self, test_model, batch_size, test_set_length):
        return np.mean([test_model(i) for i in range(test_set_length // batch_size)])


class MLP(object):

    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hiddens, n_out, activation_function, cost_function):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hiddens: list[int]
        :param n_hidden: list of number of hidden units for each hidden layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hidden_layers = []
        prev_layer_n = n_in
        prev_input = input
        for n_hidden in n_hiddens:
            hidden_layer = HiddenLayer(rng=rng,
                                       input=prev_input,
                                       n_in=prev_layer_n,
                                       n_out=n_hidden,
                                       activation=activation_function)
            self.hidden_layers.append(hidden_layer)
            prev_layer_n = n_hidden
            prev_input = hidden_layer.output

        self.log_regression_layer = LogisticRegression(input=self.hidden_layers[-1].output,
                                                       n_in=n_hiddens[-1],
                                                       n_out=n_out,
                                                       cost_function=cost_function)

        l1_hidden_layers = sum([abs(hl.W).sum() for hl in self.hidden_layers])
        l2_hidden_layers = sum([(hl.W ** 2).sum() for hl in self.hidden_layers])
        self.L1 = l1_hidden_layers + abs(self.log_regression_layer.W).sum()
        self.L2_sqr = l2_hidden_layers + (self.log_regression_layer.W ** 2).sum()
        self.errors = self.log_regression_layer.errors
        self.calculate_cost = self.log_regression_layer.calculate_cost
        self.params = [p for hl in self.hidden_layers for p in hl.params] + self.log_regression_layer.params


class HiddenLayer():

    def __init__(self,
                 rng,
                 input,
                 n_in,
                 n_out,
                 W=None,
                 b=None,
                 activation=T.tanh):
        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6 / (n_in + n_out)),
                                              high=np.sqrt(6 / (n_in + n_out)),
                                              size=(n_in, n_out)),
                                  dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                # Sigmoid demands a larger interval, according to [Xavier10].
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None else self.activation(lin_output))
        self.params = [self.W, self.b]


class LogisticRegression(object):

    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, cost_function):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type: cost_function: function
        :param cost_function: Cost function to use.

        """
        self.cost_function = cost_function

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                              dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def calculate_cost(self, y):
        return self.cost_function(self.p_y_given_x, y)

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class Reproduce(NNPrediction):

    def __init__(self, corpus):

        self.feature_matrix = self.featurify(corpus)

        self.config = {'n_in': self.feature_matrix.shape[1],
                       'n_hiddens': [7*20, 50],
                       'n_out': 10,
                       'activation_function': T.nnet.sigmoid,
                       'cost_function': LogisticRegression.cross_entropy,
                       'n_epochs': 1000,
                       'batch_size': 500}

        super(self.config)

    def featurify(self, corpus):
        pass


def negative_log_likelihood(y_pred, y):
    return -T.mean(T.log(y_pred)[T.arange(y.shape[0]), y])


def cross_entropy(y_pred, y):
    c_entrop = T.sum(T.nnet.categorical_crossentropy(y_pred, y))
    return c_entrop


config = {'n_in': 28 * 28,
          'n_hiddens': [500, 20],
          'n_out': 10,
          'activation_function': T.nnet.softmax,
          'cost_function': cross_entropy,
          'n_epochs': 1000,
          'batch_size': 500}
