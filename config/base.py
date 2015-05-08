import theano.tensor as T
from model import cross_entropy
from model import PronounPrediction


def get_configs():
    config = {'model': PronounPrediction,
              'n_hiddens': [50],
              'embedding_dimensionality': 20,
              'activation_function': T.tanh,
              'cost_function': cross_entropy,
              'n_epochs': 1000,
              'batch_size': 100,
              'no_embeddings': 7,
              'window_size': (3, 3),
              'classes_filepath': 'resources/train/ncv9/classes.csv',
              'training_filepath': 'resources/train/ncv9/data.csv',
              'development_filepath': 'resources/train/devset_larger/data.csv',
              'test_filepath': 'resources/test/teddev/data.csv'}

    return [config]
