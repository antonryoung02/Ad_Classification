#!/bin/bash

CLASSIFICATION=$1
SAVE_DIR="data_$CLASSIFICATION"

STOP_FILE="/Users/anton/Downloads/Coding/Ad_Classification/data_processing/stop_signal.txt"
LOG_FILE="/Users/anton/Downloads/Coding/Ad_Classification/data_processing/log.txt"
mkdir -p "$SAVE_DIR" >> "$LOG_FILE" 2>&1
rm -rf $STOP_FILE >> "$LOG_FILE" 2>&1

while true
do
    if [ -f "$STOP_FILE" ]; then
        break
    fi
    FILENAME="screenshot_$(date +%Y%m%d_%H%M%S).png"

    FILEPATH="/Users/anton/Downloads/Coding/Ad_Classification/$SAVE_DIR/$FILENAME"

    if screencapture -x "$FILEPATH" >> "$LOG_FILE" 2>&1; then
        echo ""
    else
        echo "Failed to save screenshot to $FILEPATH" >> "$LOG_FILE" 2>&1
    fi

    sleep 3
done