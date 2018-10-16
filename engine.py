# Contains a set of param
config = {
  # For training
  'batch_size': 128,
  'epoch': 50,
  'routing_iterations': 3,
  'stddev': 0.01,

  # For dataset
  'train_data_path': 'data/train.csv',
  'test_data_path' : 'data/test.csv',
  'num_of_threads': 4,

  # For margin loss
  'm_plus': 0.9,
  'm_minus': 0.1,
  'lambda': 0.5
}

def get_from_config(key):
  """Returns the value of the given key from the config dict

  :param key: A string, the name of the param in the config dict
  :return: The value of the key in config
  """

  assert isinstance(key, str), str(key) + ' must be of type str.'
  assert key in config, str(key) + ' is not found in config.'
  return config[key]

