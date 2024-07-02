# Ad_Classification

Hockey vs Advertisement Classifier!

This repository contains the code used to collect data, preprocess data, train a PyTorch model, and make predictions on new images. Advertisements are distracting, and the objective of this project is to mute these advertisements when they come on and only unmute when the hockey game returns.

*Note: Unfortunately, I have not included the model weights due to it being trained on copyrighted material.*

## Motivation

https://www.youtube.com/watch?v=6U6-YJ1yjoQ

## Use Cases

The classifier has been set up to be executable on an [Apple Mac](./mac_inference) where it mutes the TV/laptop when advertisements are on.

## Demos
[See report here](https://api.wandb.ai/links/antonryoung02_org/g0l4czrq)

## Progress

Identified remaining issues:

Data quality -- Some incorrectly labeled data and rare / irrelevant events. Still some concerns with data leakage due to images being captured 3 seconds apart

Model size -- I will quantize the model in float16 and try model pruning to further reduce the size and number of parameters

### June 2024

I switched to pytorch lightning for a more organized training procedure and better compatability with Weights & Biases. 

I moved the data augmentation to occur after the train/val split to avoid data leakage, and I implemented a SqueezeNet architecture which led to better performance with fewer parameters.

### May 2024

I reduced image size to 224x224 and used a more standard architecture with 2 convolutional/relu layers followed by a max pooling. I added a second dense layer so that the network can learn a nonlinear transformation of the feature maps.

My previous commits ran a grid search and then immediately saved the best model weights of the model with the best AUC. Because classifiers have so many tradeoffs between metrics, I decided to add a `PerformanceTracker` class so that I can compare models later.
