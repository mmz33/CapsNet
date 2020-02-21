# Capsule Network (CapsNet)

This repository represents a TensorFlow implementation of capsule network. For more details about CapsNet, you can check my [blog post](https://mmz33.github.io/Capsule-Networks/). you can also find [here](https://github.com/mmz33/CapsNet/blob/master/report/capsnet_seminar_report.pdf) a detailed report about this topic.

## Files
- `dataset.py`: loads the MNIST dataset using keras API
- `capsnet.py`: represents CapsNet architecture and contains function to build it
- `capsule_layer.py`: represents CapsNet layers which are mainly PrimaryCaps and DigitCaps layers
- `engine.py`: it extracts parameters from the config, set up training and testing configuration, and implements them
- `config.py`: represents a dict of parameters with a getter function
- `main.py`: the main entry point
- `utils.py`: contians some helping functions
- `run_kaggle.py`: a script to run digit recognizer competition from kaggle

## Training

For training, run `python3 main.py --train`. In `config.py`, you can specify your hyperparameters. `checkpoint_path` is the location where models/checkpoints are saved. `log` is the location where TensorFlow summaries are saved to be used later in Tensorboard for example. 

## Testing

For testing, you just need to run `python3 main.py --test`. This will load the model corresponding to the latest saved checkpoint.

#### Kaggle Digit Recognizer

In addition, the code was tested on the test data provided in digit recognizer competition from Kaggle, which is also MNIST data. The score achieved was: `0.99500` which is `99.5%` accuracy.

## Tensorboard

#### TF Graph
![capsnet-tf-graph](https://user-images.githubusercontent.com/17355283/75074459-09bb8f00-54fc-11ea-9add-2e9830624da3.png)

#### Train

<img width="1322" alt="Screenshot 2020-02-21 at 22 51 21" src="https://user-images.githubusercontent.com/17355283/75074810-c3b2fb00-54fc-11ea-8494-12f8f63e2466.png">
