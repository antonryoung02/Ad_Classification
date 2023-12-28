#!/bin/bash

source .env

touch /tmp/classify_script_running
trap "rm -f /tmp/classify_script_running" EXIT
python ./inference.py &

while true; do
	echo "Capturing image; $IMAGE_PATH"
	libcamera-still -o "$IMAGE_PATH" > /dev/null 2>&1

	if [ -f "$IMAGE_PATH" ]; then
		echo "Saved successfully"
	else 
		echo "Failed to save image"
	fi

	sleep 3
done



