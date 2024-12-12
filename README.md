# Ad_Classification

Sports vs Advertisement Classifier!

This repository contains the code used to collect data, preprocess data, train a PyTorch model, and make predictions on new images. Advertisements are distracting, and the objective of this project is to mute these advertisements when they are on and only unmute when the sports game returns.

## Implementation Features

### Researched and implemented efficient neural network architectures designed to run on low-memory, low-power devices 

[modeling/architectures directory](./modeling/architectures)

- Squeezenet: [https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)

- MobileNet: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

- ShuffleNet: [https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)

- GhostNet: [https://arxiv.org/abs/1911.11907](https://arxiv.org/abs/1911.11907)

### Designed flexible ML pipeline with Pytorch Lightning and integrated logging with Weights & Biases 

[modeling directory](./modeling)

- Model, training, and data augmentation hyperparameters are all subject to cross validation

- Different architectures and hyperparameter configurations are able to run seamlessly via changing keys of the config.yaml file

- Implemented custom logging classes to identify gradient problems, log worst model predictions, and log performance metrics of interest

- See Weights & Biases training report here [https://api.wandb.ai/links/antonryoung02_org/g0l4czrq](https://api.wandb.ai/links/antonryoung02_org/g0l4czrq)

### Curated 5-class dataset from creative commons image provider 

[openverse.com](https://openverse.org)

- Fetched and sorted through query results to gather over 12,000 representative images to classify Ad, Football, Basketball, Baseball, and Hockey images

### Developed MacOS menu bar application to handle model execution and sound control

- App allows for one-click start, screen share selection with Apple's [ScreenCaptureKit API](https://developer.apple.com/documentation/screencapturekit/), and user settings to customize prediction speed and moving average buffer size.

- Core logic is separated from the application and is thoroughly unit-tested

## Motivation

https://www.youtube.com/watch?v=6U6-YJ1yjoQ

## Use Cases

The classifier has been set up to be executable on an [Apple Mac](./mac_inference) where it mutes the TV/laptop when advertisements are on.

## Progress

### Sep-Dec 2024

Implemented MobileNet, ShuffletNet, and GhostNet architectures

Moved all pipeline hyperparameters to a .yaml config file

Gathered baseball & basketball datasets and shifted to multiclass classification learning

Recreated Ads dataset with higher quality data

### July-August 2024

I stopped using copyright data to train my model. Instead, I gathered for commercial use sports images from Openverse

Added model conversions to coreML format for use within in a macOS application (Model inference on coreML for Mac M1 is < 1ms)

### June 2024

I switched from pytorch to pytorch lightning for a more organized training procedure and better compatability with Weights & Biases.

I moved the data augmentation to occur after the train/val split to avoid data leakage, and I implemented a SqueezeNet architecture which led to better performance with fewer parameters.

### May 2024

I reduced image size to 224x224 and used a more standard architecture with 2 convolutional/relu layers followed by a max pooling. I added a second dense layer so that the network can learn a nonlinear transformation of the feature maps.

My previous commits ran a grid search and then immediately saved the best model weights of the model with the best AUC. Because classifiers have so many tradeoffs between metrics, I decided to add a `PerformanceTracker` class so that I can compare models later.
