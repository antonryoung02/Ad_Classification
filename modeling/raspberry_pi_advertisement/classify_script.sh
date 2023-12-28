#!/bin/bash

touch /tmp/classify_script_running
trap "rm -f /tmp/classify_script_running" EXIT
python ./inference.py &

echo "Starting while loop"
FULL_PATH="/home/antonryoung02/raspberry_pi_advertisement/image.png"
while true; do
	echo "Capturing image; $FULL_PATH"
	libcamera-still -o "$FULL_PATH" > /dev/null 2>&1

	if [ -f "$FULL_PATH" ]; then
		echo "Saved successfully"
	else 
		echo "Failed to save image"
	fi

	sleep 3
done



