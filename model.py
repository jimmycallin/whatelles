"""
Most of the code taken form here: http://deeplearning.net/tutorial/
"""

import theano
import theano.tensor as T
import numpy as np
import pandas as pd
import os
import re


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
                              cost_function=self.config['cost_function'],
                              vocab_size=self.config['vocab_size'],
                              embedding_dimensionality=self.config['embedding_dimensionality'],
                              no_embeddings=self.config['no_embeddings'])

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

    def __init__(self,
                 rng,
                 input,
                 n_in,
                 n_hiddens,
                 n_out,
                 activation_function,
                 cost_function,
                 vocab_size,
                 embedding_dimensionality,
                 no_embeddings):
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

        embedding_layer = EmbeddingLayer(rng=rng,
                                         input=input,
                                         vocab_size=vocab_size,
                                         embedding_dimensionality=embedding_dimensionality,
                                         no_embeddings=no_embeddings,
                                         embeddings=None,
                                         activation=activation_function)

        self.hidden_layers = [embedding_layer]
        prev_layer_n = embedding_layer.n_out
        prev_input = embedding_layer.output
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
                                                       n_in=self.hidden_layers[-1].n_out,
                                                       n_out=n_out,
                                                       cost_function=cost_function)

        l1_hidden_layers = sum([abs(hl.W).sum() for hl in self.hidden_layers])
        l2_hidden_layers = sum([(hl.W ** 2).sum() for hl in self.hidden_layers])
        self.L1 = l1_hidden_layers + abs(self.log_regression_layer.W).sum()
        self.L2_sqr = l2_hidden_layers + (self.log_regression_layer.W ** 2).sum()
        self.errors = self.log_regression_layer.errors
        self.calculate_cost = self.log_regression_layer.calculate_cost
        self.params = [p for hl in self.hidden_layers for p in hl.params] + self.log_regression_layer.params


class EmbeddingLayer():

    def __init__(self,
                 rng,
                 input,
                 vocab_size,
                 embedding_dimensionality,
                 no_embeddings,
                 embeddings=None,
                 activation=T.nnet.sigmoid):
        self.input = T.cast(input, 'int32')
        self.activation = activation
        batch_size = input.shape[0]
        self.embedding_dimensionality = embedding_dimensionality
        self.no_embeddings = no_embeddings
        self.n_out = no_embeddings * embedding_dimensionality

        if embeddings is None:
            embeddings = np.asarray(rng.uniform(low=-np.sqrt(6 / (vocab_size + embedding_dimensionality)),
                                                high=np.sqrt(6 / (vocab_size + embedding_dimensionality)),
                                                size=(vocab_size, embedding_dimensionality)),
                                    dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                # Sigmoid demands a larger interval, according to [Xavier10].
                embeddings *= 4

            embeddings = theano.shared(value=embeddings, name='embeddings', borrow=True)

        self.embeddings = embeddings

        # Replace all word indices in input with word embeddings
        emb_input = self.embeddings[self.input.flatten()]

        # Reshape to match original input (times embedding dimensionality on columns)
        self.W = emb_input.reshape((batch_size, no_embeddings * embedding_dimensionality))
        self.output = self.W if self.activation is None else self.activation(self.W)
        self.params = [self.embeddings]


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
        self.n_in = n_in
        self.n_out = n_out

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
    """
    NEXT TIME: connect self.train_x and self.train_y to NNPrediction.
    Featurify isn't entirely done. It should only do ngrams where one of the classes is the focus word.
    Good work today.
    Also, refactorize the hell out of this. most of the data loading should be done in evaluate.py
    """
    def __init__(self):
        self.config = {'n_hiddens': [50],
                       'embedding_dimensionality': 20,
                       'activation_function': T.nnet.sigmoid,
                       'cost_function': cross_entropy,
                       'n_epochs': 300,
                       'batch_size': 30,
                       'no_embeddings': 7,
                       'window_size': (3, 3),
                       'classes_filepath': 'resources/train/europarl/classes.csv',
                       'training_filepath': 'resources/train/europarl/data.csv',
                       'development_filepath': 'resources/test/teddev/data.csv'}

        self.no_words = 0
        self._word2id = dict()
        self.classes = self.load_classes(self.config['classes_filepath'])
        print("Loading training data... ")
        training_data = self.load_data(self.config['training_filepath'])
        self.train_x, self.train_y = self.featurify(training_data)
        print("Loading development_data data... ")
        development_data = self.load_data(self.config['development_filepath'])
        self.dev_x, self.dev_y = self.featurify(development_data)

        self.config['n_in'] = self.train_x.shape[0]
        self.config['n_out'] = len(self.classes)
        self.config['vocab_size'] = self.no_words

        super().__init__(self.config)

    def load_classes(self, classes_filepath):
        with open(classes_filepath) as f:
            return [line.split(",")[1].strip() for line in f]

    def featurify(self, data):
        """ Param data: dataframe of (class_labels,
                                      removed_words,
                                      source_sentence,
                                      target_sentence,
                                      source2target_alignment
                                      target2source_alignment)
            """
        x_matrix = []
        y_vector = []
        for (class_labels,
             removed_words,
             source_sentence,
             target_sentence,
             source2target_alignment,
             target2source_alignment) in data.itertuples(index=False):

            if not isinstance(class_labels, list):  # we're not dealing with these for now
                continue
            words = target_sentence.strip().lower().split()
            ngrams = self.ngramify(words, self.config['window_size'])
            for class_label, removed_word in zip(class_labels, removed_words):
                for ngram in ngrams:
                    if ngram[self.config['window_size'][0]].startswith("replace_"):
                        x_matrix.append([self.word2id(ngram_word) for ngram_word in ngram])
                        y_vector.append(self.classes.index(class_label))
                        continue
        return np.asarray(x_matrix, dtype=np.int32), np.asarray(y_vector, dtype=np.int32)

    def ngramify(self, words, window_size):
        for i, word in enumerate(words):
            if i - window_size[0] < 0 or i + window_size[1] + 1 >= len(words):
                continue
            left = i - window_size[0]
            right = i + window_size[1] + 1
            context = []
            for j in range(right - left):
                context.append(words[left + j])
            yield context

    def word2id(self, word):
        if word not in self._word2id:
            self._word2id[word] = self.no_words
            self.no_words += 1
        return self._word2id[word]

    def load_data(self, data_path):
        """
        Returns data as tuple of (class_labels,
                                  removed_words,
                                  source_sentence,
                                  target_sentence,
                                  source2target_alignment
                                  target2source_alignment)
        """
        data = pd.read_csv(data_path,
                           sep='\t',
                           header=None,
                           quoting=pd.io.parsers.csv.QUOTE_NONE)
        data.columns = ['class_labels', 'removed_words', 'source_sentence', 'target_sentence', 'word_alignment']
        data['class_labels'] = data['class_labels'].str.split(' ')
        data['removed_words'] = data['removed_words'].str.split(' ')
        data = data.join(data.word_alignment.apply(aggregate_alignments)
                         .apply(pd.Series)
                         .rename(columns={0: 'source2target_alignment', 1: 'target2source_alignment'})
                         ).drop('word_alignment', axis=1)
        return data

    def _test_featurify(self, corpus):
        f_matrix = np.asarray([[1, 2, 3], [6, 5, 2], [2, 1, 4]], dtype=np.int32)
        return f_matrix


def negative_log_likelihood(y_pred, y):
    return -T.mean(T.log(y_pred)[T.arange(y.shape[0]), y])


def cross_entropy(y_pred, y):
    c_entrop = T.sum(T.nnet.categorical_crossentropy(y_pred, y))
    return c_entrop


def aggregate_alignments(align_line):
    """
    Parse the alignment file.

    Return:
      - s2t: a dict mapping source position to target position
      - t2s: a dict mapping target position to source position
    """
    align_tokens = re.split('\s+', align_line.strip())
    s2t = {}
    t2s = {}
    # process alignments
    for align_token in align_tokens:
        if align_token == '':
            continue
        (src_pos, tgt_pos) = re.split('\-', align_token)
        src_pos = int(src_pos)
        tgt_pos = int(tgt_pos)
        if src_pos not in s2t:
            s2t[src_pos] = []
        s2t[src_pos].append(tgt_pos)

        if tgt_pos not in t2s:
            t2s[tgt_pos] = []
        t2s[tgt_pos].append(src_pos)

    return (s2t, t2s)
