config = {
  # For training
  'batch_size': 128,
  'epoch': 50,
  'routing_iterations': 3,
  'stddev': 0.01,

  # For dataset
  'train_data_path': 'data/train.csv',
  'test_data_path' : 'data/test.csv',
  'number_of_threads': 4,

  # For margin loss
  'm_plus': 0.9,
  'm_minus': 0.1,
  'lambda': 0.5
}

def get_from_config(key):
  assert isinstance(key, str), str(key) + ' must be of type str.'
  assert key in config, str(key) + ' is not found in config.'
  return config[key]

