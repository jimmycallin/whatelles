"""
This evaluates how the number of preceding POS tags affects the evaluation results.
"""

from config import base
import evaluate as e

config = base.get_configs()[0]
config['test_filepath'] = 'resources/test/teddev/data-with-doc.csv'

n_tags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_tag in n_tags:
    print("Running {}".format(n_tag))
    config['n_tags'] = n_tag
    config['no_embeddings'] = 2 * (config['window_size'][0] + config['window_size'][1]) + 1 + n_tag * 2
    predictions = e.evaluate(config)
    test_data = e.load_data(config['test_filepath'])
    e.output(predictions, test_data, config['classes'],
             'results/base.dev.n_tags.{}.txt'.format(n_tag))
    print("Saving {}".format(n_tag))
