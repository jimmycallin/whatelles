import theano.tensor as T
from model import cross_entropy
from model import PronounPrediction

"""
FINAL
Base model for the pronoun prediction. This returns a macro-fscore of about 54%.
Tested on commit: 09f44b17b75e47ea4862853aea297e737609eeef
"""

# n_tags: 5 53.20
# n_tags: 2 51.31
# senast med tanh: 60.40
# testar nu med sigmoid

def get_configs():
    config = {'model': PronounPrediction,
              'n_hiddens': [50],
              'embedding_dimensionality': 50,
              'activation_function': T.tanh,
              'cost_function': cross_entropy,
              'min_iterations': 100000,
              'n_epochs': 1000,
              'batch_size': 100,
              'no_embeddings': 23,
              'window_size': (4, 4),
              'classes_filepath': 'resources/train/iwslt14/classes.csv',
              'training_filepath': ['resources/train/iwslt14/data-with-doc.csv'],
              'development_filepath': 'resources/test/teddev/data-with-doc.csv',
              'test_filepath': 'resources/test/discomt/data-with-doc.csv',
              'n_tags': 3}

    return [config]
