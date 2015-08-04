"""
This tests how assymetric window sizes affects the evaluation results on the development set.
"""

from config import base
import evaluate as e

config = base.get_config()
config['test_filepath'] = 'resources/test/teddev/data-with-doc.csv'

window_sizes = [(4, 0), (4, 1), (4, 2), (4, 3), (3, 4), (2, 4), (1, 4), (0, 4)]

for window_size in window_sizes:
    print("Running {}".format(window_size))
    config['window_size'] = window_size
    config['no_embeddings'] = 2 * (window_size[0] + window_size[1]) + 1 + config['n_tags'] * 2
    predictions = e.evaluate(config)
    test_data = e.load_data(config['test_filepath'])
    e.output(predictions, test_data, config['classes'],
             'results/base.dev.window_size.{}+{}.txt'.format(window_size[0], window_size[1]))
    print("Saving {}".format(window_size))
