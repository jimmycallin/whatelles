"""
This evaluates how the number of preceding POS tags affects the evaluation results.
"""

from config import base
import evaluate as e

config = base.get_config()
config['test_filepath'] = 'resources/test/teddev/data-with-doc.csv'

ignores = ['ignore_pos_tags', 'ignore_target_context', 'ignore_source_context']
n_embeddings = {'ignore_pos_tags': config['n_tags'] * 2,
                'ignore_target_context': config['window_size'][0] + config['window_size'][1],
                'ignore_source_context': config['window_size'][0] + config['window_size'][1] + 1}
for ignore in ignores:
    print("Running {}".format(ignore))
    config[ignore] = True
    config['no_embeddings'] = sum(n_embeddings[feature] for feature in n_embeddings if not config.get(feature, False))
    print("no_embeddings: {}".format(config['no_embeddings']))
    print(config)
    predictions = e.evaluate(config)
    test_data = e.load_data(config['test_filepath'])
    e.output(predictions, test_data, config['classes'],
             'results/base.dev.ignore.{}.txt'.format(ignore))
    print("Saving {}".format(ignore))
    config[ignore] = False
