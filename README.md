# Ad_Classification

Sports vs Advertisement Classifier!

This repository contains the code used to collect data, preprocess data, train a PyTorch model, and make predictions on new images. Advertisements are distracting, and the objective of this project is to mute these advertisements when they are on and only unmute when the sports game returns.

## Demos

[See report here](https://api.wandb.ai/links/antonryoung02_org/g0l4czrq)

## Motivation

https://www.youtube.com/watch?v=6U6-YJ1yjoQ

## Use Cases

The classifier has been set up to be executable on an [Apple Mac](./mac_inference) where it mutes the TV/laptop when advertisements are on.

## Progress

### Sep-Dec 2024

Implemented MobileNet, ShuffletNet, and GhostNet architectures

Moved all model hyperparameters to a .yaml config file

Gathered baseball & basketball datasets and shifted to multiclass classification learning

Recreated Ads dataset with higher quality data

### July-August 2024

I stopped using copyright data to train my model. Instead, I gathered for commercial use sports images from Openverse

Added model conversions to coreML format for use within in a macOS application

Model inference on coreML for Mac M1 is < 1ms

### June 2024

I switched from pytorch to pytorch lightning for a more organized training procedure and better compatability with Weights & Biases.

I moved the data augmentation to occur after the train/val split to avoid data leakage, and I implemented a SqueezeNet architecture which led to better performance with fewer parameters.

### May 2024

I reduced image size to 224x224 and used a more standard architecture with 2 convolutional/relu layers followed by a max pooling. I added a second dense layer so that the network can learn a nonlinear transformation of the feature maps.

My previous commits ran a grid search and then immediately saved the best model weights of the model with the best AUC. Because classifiers have so many tradeoffs between metrics, I decided to add a `PerformanceTracker` class so that I can compare models later.
