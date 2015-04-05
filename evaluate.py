import expy


def get_project(config, test_data):
    return expy.Project(config.pop('project_name'), test_data=test_data)


def load_data(config):
    pass


def evaluate(config):
    pass
