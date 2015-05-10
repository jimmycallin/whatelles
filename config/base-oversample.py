import theano.tensor as T
from model import cross_entropy
from model import PronounPrediction

"""
Base model for the pronoun prediction, using oversampling of classes to even out the class distribution.
This returns a worse macro-fscore on the test data, but I suspect it return higher results for the final
test data, since that data has a more even class distribution.
Tested on commit: 09f44b17b75e47ea4862853aea297e737609eeef
"""


def get_configs():
    config = {'model': PronounPrediction,
              'n_hiddens': [50],
              'embedding_dimensionality': 20,
              'activation_function': T.tanh,
              'cost_function': cross_entropy,
              'min_iterations': 60000,
              'n_epochs': 1000,
              'batch_size': 200,
              'no_embeddings': 23,
              'window_size': (3, 3),
              'oversample_filepath': 'resources/train/europarl/data-with-doc.csv',
              'classes_filepath': 'resources/train/iwslt14/classes.csv',
              'training_filepath': ['resources/train/iwslt14/data-with-doc.csv'],
              'development_filepath': 'resources/train/devset_larger/data-with-doc.csv',
              'test_filepath': 'resources/test/teddev/data-with-doc.csv',
              'n_tags': 5}

    return [config]
