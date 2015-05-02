import sys
from importlib import import_module

from data_utils import Sentence


def load_classes(classes_filepath):
    with open(classes_filepath) as f:
        return [line.split(",")[1].strip() for line in f]


def load_training_data(config):
    return load_data(config['training_filepath'])


def load_development_data(config):
    return load_data(config['development_filepath'])


def load_test_data(config):
    return load_data(config['test_filepath'])


def load_data(data_path):
    """
    Returns data as a list of data_utils.Sentence instances.
    """
    with open(data_path) as data:
        for line in data:
            (class_labels,
             removed_words,
             source_sentence,
             target_sentence,
             alignments) = line.split("\t")

            sentence = Sentence(source_sentence,
                                target_sentence,
                                class_labels.split(),
                                removed_words.split(),
                                alignments.split())
            yield sentence


def load_configs():
    configs = import_module(sys.argv[1]).get_configs()
    return configs


def train_model(config):
    model_parameters = {x: config[x] for x in config['model'].model_parameters if x in config}
    model = config['model'](model_parameters)
    training_data = load_training_data(config)
    development_data = load_development_data(config)
    model.train(training_data, development_data)
    return model


def test_model(model, test_data):
    print("Testing model...")
    return model.predict(test_data)


def evaluate(config):
    config['classes'] = load_classes(config['classes_filepath'])
    test_data = load_test_data(config)
    model = train_model(config)
    predictions = test_model(model, test_data)
    return predictions


def output(predictions, classes, test_path, output_path):
    """
    Output test according to test data template.
    This should be read by discoMT_scorer.pl.
    """
    pred_iter = iter(predictions)
    test_instances = []
    with open(test_path) as test_data:
        for line in test_data:
            (class_labels,
             removed_words,
             source_sentence,
             target_sentence,
             alignments) = [x.strip() for x in line.split('\t')]
            class_labels = class_labels.split()
            removed_words = removed_words.split()
            instances_predicted = []
            for _ in range(len(class_labels)):
                instances_predicted.append(classes[next(pred_iter)])

            test_instances.append([instances_predicted,
                                   removed_words, source_sentence, target_sentence, alignments])

    with open(output_path, 'w') as output:
        for line in test_instances:
            line_str = ""
            for column in line[:2]:
                line_str += " ".join(column) + "\t"
            line_str += "\t".join(line[2:])
            print(line_str)
            output.write(line_str + "\n")


if __name__ == '__main__':
    evaluate(sys.argv[1])
