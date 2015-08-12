import sys
import fileinput
import itertools
from importlib import import_module

from data_utils import Sentence


def load_classes(classes_filepath):
    """
    Loads the file with class information. Usually called classes.csv
    """
    with open(classes_filepath) as f:
        return [line.split(",")[1].strip() for line in f]


def load_class_distributions(classes_filepath):
    """
    Load file with class information, keep the distribution data.
    """
    with open(classes_filepath) as f:
        return {line.split(",")[1].strip(): int(line.split(",")[0].strip()) for line in f}


def load_training_data(config):
    """
    Loads a generator with training data, either from data.csv or data-with-doc.csv
    Returns: A generator of data as a list of data_utils.Sentence instances.
    """
    if 'oversample_filepath' in config:
        class_distribution = load_class_distributions(config['classes_filepath'])
        oversample_from = oversample(load_data(config['oversample_filepath']), class_distribution)
        training_data = load_data(config['training_filepath'])
        return itertools.chain(training_data, oversample_from)
    else:
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
            (new_doc,
             class_labels,
             removed_words,
             source_sentence,
             target_sentence,
             alignments) = line.split("\t")

            if int(new_doc) == 1:
                last_sentence = None

            sentence = Sentence(source_sentence,
                                target_sentence,
                                alignments.split(),
                                class_labels.split(),
                                removed_words.split(),
                                last_sentence)
            last_sentence = sentence
            yield sentence


def oversample(oversample_from, current_class_distribution):
    """
    Oversample from a class distribution given a separate data corpus, so that all classes have about an equal
    distribution.
    """
    biggest_class, biggest_class_val = max(current_class_distribution.items(), key=lambda x: x[1])
    n_to_sample = {cl: biggest_class_val - n_instances for cl, n_instances in current_class_distribution.items()}
    after = current_class_distribution.copy()
    for instance in oversample_from:
        # if the class is in the instance, and n_to_sample is greater than 0:
        for cl in instance.classes:
            if n_to_sample[cl] > 0:
                n_to_sample[cl] -= 1
                after[cl] += 1
                yield instance
                break

        if all(val <= 0 for cl, val in n_to_sample.items()):
            raise StopIteration()
    print("Before oversampling: {}".format(current_class_distribution))
    print("After oversampling: {}".format(after))


def load_configs():
    configs = import_module(sys.argv[1]).get_configs()
    return configs


def train_model(config):
    model_parameters = {x: config[x] for x in config['model'].model_parameters if x in config}
    model = config['model'](model_parameters)
    training_data = load_training_data(config)
    if 'development_filepath' in config:
        development_data = load_development_data(config)
        model.train(training_data, development_data)
    else:
        model.train(training_data)
    return model


def test_model(model, test_data):
    print("Testing model...")
    return model.predict(test_data)


def evaluate(config):
    """
    Evaluate a model using the given config. The config should be dict like.
    """
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
            for _ in range(len(sentence.removed_words_target_indices)):
                predicted.append(class_labels[next(pred_iter)])
            predicted = " ".join(predicted)
            instance_output = "{}\t{}\t{}\t{}\t{}".format(predicted, removed_words,
                                                          source_sentence, target_sentence, alignments)
            f.write(instance_output + '\n')
            print(instance_output)


if __name__ == '__main__':
    evaluate(sys.argv[1])
