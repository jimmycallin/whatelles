import expy
import os
import sys
import pandas as pd
from importlib import import_module

classes = ('ce', 'elle', 'elles', 'il', 'ils', 'c\'', 'ça', 'cela', 'on', 'OTHER')

other = ('le', 'l\'', 'se', 's\'', 'y', 'en', 'qui', 'que', 'qu\'', 'tout', 'faire', 'ont',
         'fait', 'est', 'parler', 'comprendre', 'chose', 'choses', 'ne', 'pas', 'dessus', 'dedans')


def load_classes(config):
    return pd.read_csv(os.path.join(config['train_path'], 'classes.csv'), header=None)


def load_training_data(config):
    return load_data(config.pop('train_path'))


def load_test_data(config):
    return load_data(config.pop('test_path'))


def load_data(path):
    data = pd.read_csv(os.path.join(path, 'data.csv'), sep='\t', header=None)
    data.columns = ['class_labels', 'removed_words', 'source_sentence', 'target_sentence', 'word_alignment']
    data['class_labels'] = data['class_labels'].str.split(' ')
    data['removed_words'] = data['removed_words'].str.split(' ')
    return data


def get_project(config, test_data):
    return expy.Project(config.pop('project_name'), test_data=test_data)


def load_configs():
    configs = import_module(sys.argv[1]).get_configs()
    return configs


def train_model(config):
    if not os.path.isfile(config['model_store_path']):
        model_parameters = {x: config[x] for x in config['model'].model_parameters}
        model = config['model'](**model_parameters)
        print("Model not found at {}, \n building model from scratch.".format(config['model_store_path']))
        model.train(load_training_data(config))
        model.store(config['model_store_path'])
        print("Stored trained model at {}".format(config['model_store_path']))
        return model
    else:
        print("Loaded existing model from {}".format(config['store_path']))
        return config['model'].load(config['model_store_path'])


def test_model(model, test_data):
    print("Testing model...")
    return model.predict(test_data)


def evaluate(config):
    test_data = load_test_data(config)
    project = get_project(config, test_data)
    model = train_model(config)
    predictions = test_model(model, test_data)
    print("Storing experiment...")
    experiment = project.new_experiment(predictions,
                                        config,
                                        description=config.pop('description'),
                                        tags=config.pop('tags'))

    experiment.experiment_report(accuracy=True,
                                 precision=True,
                                 recall=True,
                                 f1_score=True)


def prepare_testing():
    sys.argv.append('config.test')


if __name__ == '__main__':
    evaluate(sys.argv[1])
