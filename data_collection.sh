#!/bin/bash

CLASSIFICATION=$1
SAVE_DIR="data_$CLASSIFICATION"

STOP_FILE="/Users/anton/Downloads/Coding/Ad_Classification/stop_signal.txt"
LOG_FILE="/Users/anton/Downloads/Coding/Ad_Classification/log.txt"
mkdir -p "$SAVE_DIR" >> "$LOG_FILE" 2>&1
rm -rf $STOP_FILE >> "$LOG_FILE" 2>&1

while true
do
    if [ -f "$STOP_FILE" ]; then
        echo "Stop signal received. Exiting script." >> "$LOG_FILE" 2>&1
        break
    fi
    # File name for the screenshot
    FILENAME="screenshot_$(date +%Y%m%d_%H%M%S).png"

    # Full path for the screenshot file
    FILEPATH="/Users/anton/Downloads/Coding/Ad_Classification/$SAVE_DIR/$FILENAME"

    # Taking the screenshot
    if screencapture -x "$FILEPATH" >> "$LOG_FILE" 2>&1; then
        echo "Screenshot saved to $FILEPATH" >> "$LOG_FILE" 2>&1
    else
        echo "Failed to save screenshot to $FILEPATH" >> "$LOG_FILE" 2>&1
    fi

    # Wait for 3 seconds before the next screenshot
    sleep 3
done