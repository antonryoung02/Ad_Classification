# Raspberry Pi Inference

This directory contains all files necessary to run the CNN on a raspberry pi.


## Specifications:

- Raspberry Pi 4 8gb 
- Camera module 2


## Setup:

### Install poetry 

`pip install poetry`

### Create and activate virtual environment

`poetry install`

`poetry shell`


## Running Inference:

Make sure libcamera module is activated, helpful setup tutorial here: https://www.youtube.com/watch?v=Z8cs1cRrc5A&t=602

- Navigate to `pi_inference`
- Run `./classify_script.sh`

Cancel script execution to end inference



