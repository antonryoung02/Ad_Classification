## Mac Inference

This directory contains the files necessary to run inference on an apple laptop.

## How it works

Every 2 seconds, the shell script takes a screenshot and saves the image at `IMAGE_PATH`. This file is then read in the python script, preprocessed, and a prediction is made. The prediction is written to `ad_signal.txt`, where the laptop is muted if the prediction is True.

Note: You should to disable the screencapture sound effect in `System Preferences > Sound > Alert Volume to 0`

## How to use

cd to `Ad_Classification`.

Make sure poetry is installed, then run `poetry install` to install packages and `poetry shell` to activate. 

`cd mac_inference`

Start the inference: `bash ./mac_inference.sh`

Interrupt the script with `CTRL C` to end execution
