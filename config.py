"""Set params here"""

config = {
    'epochs': 10,
    'batch_size': 128,
    'routing_iterations': 3,
    'stddev': 0.01,
    'checkpoint_path': 'net-model/network',
    'log': 'tf_logs',
    'm_plus': 0.9,
    'm_minus': 0.1,
    'lambda': 0.5,
    'alpha': 0.0005
}


def get_from_config(key):
    """
    Returns the value of the given key from the config dict

    :param key: A string, the name of the param in the config dict
    :return: The value of the key in config
    """
    assert isinstance(key, str), str(key) + ' must be of type str.'
    assert key in config, str(key) + ' is not found in config. It must be one for these'
    return config[key]
