# Ad_Classification

Hockey vs Advertisement Classifier!

This repository contains the code used to collect data, preprocess data, train a PyTorch model, and make predictions on new images. Advertisements are distracting, and the objective of this project is to mute these advertisements when they come on and only unmute when the hockey game returns.

*Note: Unfortunately, I have not included the model weights due to it being trained on copyrighted material.*

## Motivation

https://www.youtube.com/watch?v=6U6-YJ1yjoQ

## Use Cases

The classifier has been set up to be executable on a [Mac](./mac_inference) or [Raspberry Pi](./pi_inference) (haven't bought an infrared remote for the Pi to control a TV yet), where it mutes the TV/laptop when advertisements are on.

## Demos

## Progress

Identified remaining issues:

Data quality -- some incorrectly labeled data, image duplicates, ads are not diverse enough

Data leakage -- images are augmented before the train/cv split

Duplicate code

### May 2024

I reduced image size to 224x224 and used a more standard architecture with 2 convolutional/relu layers followed by a max pooling. I added a second dense layer so that the network can learn a nonlinear transformation of the feature maps.

My previous work ran a grid search and then immediately saved the best model weights of the model with the best AUC. Because classifiers have so many tradeoffs between metrics, I decided to add a `PerformanceTracker` class so that I can compare models later.
