"""
This evaluates how a differing symmetric window size affects the evaluation results.
"""

from config import base
import evaluate as e

config = base.get_config()
config['test_filepath'] = 'resources/test/teddev/data-with-doc.csv'

window_sizes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
for window_size in window_sizes:
    print("Running {}".format(window_size))
    config['window_size'] = window_size
    config['no_embeddings'] = 2 * (window_size[0] + window_size[1]) + 1 + config['n_tags'] * 2
    predictions = e.evaluate(config)
    test_data = e.load_data(config['test_filepath'])
    e.output(predictions, test_data, config['classes'],
             'results/base.dev.window_size.{}+{}.txt'.format(window_size[0], window_size[1]))
    print("Saving {}".format(window_size))
