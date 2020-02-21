# Capsule Network

This repository represents a TensorFlow implementation of capsule network (CapsNet). For more details about CapsNet, you can check my [blog post](https://mmz33.github.io/Capsule-Networks/). you can also find [here](https://github.com/mmz33/CapsNet/blob/master/report/capsnet_seminar_report.pdf) a detailed report about this topic.

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

#### Train

<img width="1322" alt="Screenshot 2020-02-21 at 22 51 21" src="https://user-images.githubusercontent.com/17355283/75074810-c3b2fb00-54fc-11ea-8494-12f8f63e2466.png">


#### Reconstructed Images

##### During training
<img width="180" height="180" alt="train_3" src="https://user-images.githubusercontent.com/17355283/75075144-7b480d00-54fd-11ea-8b7c-7944defaea93.png"> <img width="180" height="180" alt="train_9" src="https://user-images.githubusercontent.com/17355283/75075457-235dd600-54fe-11ea-8dd7-fa6acc581fcc.png"> <img width="180" height="180" alt="train_0" src="https://user-images.githubusercontent.com/17355283/75075461-23f66c80-54fe-11ea-9561-473408ed2a58.png"> <img width="180" height="180" alt="train_8" src="https://user-images.githubusercontent.com/17355283/75075465-25c03000-54fe-11ea-8e9c-840665609289.png">

##### During validation

<img width="180" height="180" alt="valid_1" src="https://user-images.githubusercontent.com/17355283/75075578-6c158f00-54fe-11ea-98b1-a2c7cadf1396.png"> <img width="180" height="180" alt="valid_2" src="https://user-images.githubusercontent.com/17355283/75075579-6cae2580-54fe-11ea-8de5-d6465bc0296d.png"> <img width="180" height="180" alt="valid_8" src="https://user-images.githubusercontent.com/17355283/75075582-6f107f80-54fe-11ea-8810-19dd0ca7866f.png"> <img width="180" height="180" alt="valid_9" src="https://user-images.githubusercontent.com/17355283/75075584-6fa91600-54fe-11ea-8e95-bb3992e00e98.png">


#### TF Graph
![capsnet-tf-graph](https://user-images.githubusercontent.com/17355283/75074459-09bb8f00-54fc-11ea-9add-2e9830624da3.png)
