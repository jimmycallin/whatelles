import sys
import fileinput
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


def load_data(data_paths):
    """
    Returns data as a list of data_utils.Sentence instances.
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    with fileinput.input(files=data_paths) as data:
        last_sentence = None
        for line in data:
            (class_labels,
             removed_words,
             source_sentence,
             target_sentence,
             alignments) = line.split("\t")

            sentence = Sentence(source_sentence,
                                target_sentence,
                                alignments.split(),
                                class_labels.split(),
                                removed_words.split(),
                                last_sentence)
            last_sentence = sentence
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


def output(predictions, sentences, class_labels, output_path):
    """
    Output test according to test data template.
    This should be read by discoMT_scorer.pl.
    """
    pred_iter = iter(predictions)
    with open(output_path, "w") as f:
        for sentence in sentences:
            removed_words = " ".join(sentence.removed_words)
            source_sentence = " ".join(sentence.source_sentence)
            target_sentence = " ".join(sentence.target_sentence)
            alignments = " ".join(sentence.alignments)
            predicted = []
            for _ in sentence.classes:
                predicted.append(class_labels[next(pred_iter)])
            predicted = " ".join(predicted)
            instance_output = "{}\t{}\t{}\t{}\t{}".format(predicted, removed_words,
                                                          source_sentence, target_sentence, alignments)
            f.write(instance_output + '\n')
            print(instance_output)


if __name__ == '__main__':
    evaluate(sys.argv[1])
