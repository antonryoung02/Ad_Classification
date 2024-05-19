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

My previous work ran a grid search and then immediately saved the best model weights of the model with the best AUC. Because classifiers have so many tradeoffs between metrics, I decided to add a `PerformanceTracker` class so that I can compare models later.

Scores of the 5 best-performing models:

| Accuracy  | Precision | Recall   | PR AUC   | F1       | Codename |
|-----------|-----------|----------|----------|----------|----------|
| 0.980695  | 0.975141  | 0.955617 | 0.990405 | 0.965163 | Model 22 |
| 0.977998  | 0.971966  | 0.948757 | 0.989825 | 0.960201 | Model 11 |
| 0.976676  | 0.966318  | 0.950937 | 0.991659 | 0.958095 | Model 27 |
| 0.975882  | 0.970463  | 0.942835 | 0.991292 | 0.955736 | Model 26 |
| 0.976094  | 0.965833  | 0.948476 | 0.989653 | 0.956875 | Model 10 |

Overall rankings among 32 models:

| Codename | Accuracy | Precision | Recall | PR AUC | F1  | Average Ranking |
|----------|----------|-----------|--------|--------|-----|-----------------|
| Model 22 | 1.0      | 2.0       | 1.0    | 3.0    | 1.0 | 1.6             |
| Model 11 | 2.0      | 5.0       | 4.0    | 4.0    | 2.0 | 3.4             |
| Model 27 | 3.0      | 9.0       | 3.0    | 1.0    | 3.0 | 3.8             |
| Model 26 | 5.0      | 6.0       | 8.0    | 2.0    | 5.0 | 5.2             |
| Model 10 | 4.0      | 10.0      | 5.0    | 5.0    | 4.0 | 5.6             |


