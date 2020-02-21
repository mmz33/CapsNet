#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import namedtuple
from engine import Engine


# data from here: https://www.kaggle.com/c/digit-recognizer/data
train_csv = 'kaggle-mnist/train.csv'
test_csv = 'kaggle-mnist/test.csv'

############# Read data #############
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

num_train = train_data.shape[0]
val_size = 0.1
num_val = int(val_size * num_train)

train_images = train_data.iloc[num_val:, 1:].values.astype('float32') 
train_labels = train_data.iloc[num_val:, 0].values.astype('int32')

val_images = train_data.iloc[:num_val, 1:].values.astype('float32') 
val_labels = train_data.iloc[:num_val, 0].values.astype('int32')

test_images = test_data.values.astype('float32')

############# normalize #############
train_images /= 255.0
val_images /= 255.0
test_images /= 255.0

############# Reshape #############
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

############# convert to one-hot encoded #############
def to_one_hot(a):
    res = np.zeros((a.size, a.max() + 1))
    res[np.arange(a.size), a] = 1
    return res

train_labels = to_one_hot(train_labels)
val_labels = to_one_hot(val_labels)

############# Run test #############
Datasets = namedtuple('Datasets', ['train', 'valid', 'test'])
datasets = Datasets(train=(train_images, train_labels), valid=(val_images, val_labels), test=test_images)

# create engine
engine = Engine(datasets=datasets)
engine.init_engine(is_training=False)
engine.test_kaggle()
