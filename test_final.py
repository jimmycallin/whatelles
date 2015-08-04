"""
This runs the final configuration as reported in the paper.
"""

from config import base
import evaluate as e

config = base.get_config()
output_path = 'results/final.output.txt'
print("Running configuration: {}".format(config))

predictions = e.evaluate(config)
test_data = e.load_data(config['test_filepath'])
e.output(predictions, test_data, config['classes'],
         output_path)
print("Saved output to {}".format(output_path))
