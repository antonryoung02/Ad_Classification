#!/bin/bash
IMAGE_PATH="./images/image.png"
AD_SIGNAL_PATH="./ad_signal.txt"
mkdir -p $(dirname "$IMAGE_PATH")
# Initialize an array to keep the last three ad_signal values
ad_signal_history=(0 0 0)

touch ./classify_script_running
trap "rm -f ./classify_script_running" EXIT
python ./inference.py &

while true; do
 
    screencapture "$IMAGE_PATH"

    if [ $? -eq 0 ]; then
        echo "Screenshot saved to $IMAGE_PATH"
    else 
        echo "Failed to capture screenshot"
    fi

    sleep 1

    if [ -f "$AD_SIGNAL_PATH" ]; then
        read -r ad_signal < "$AD_SIGNAL_PATH"

        if [ "$ad_signal" = "True" ]; then
            ad_signal_history=(1 "${ad_signal_history[@]:0:2}")
        else
            ad_signal_history=(0 "${ad_signal_history[@]:0:2}")
        fi

        #Rolling average
        sum=$((${ad_signal_history[0]} + ${ad_signal_history[1]} + ${ad_signal_history[2]}))
        prediction=$(echo "$sum / 3" | bc -l)
        #prediction="${ad_signal_history[0]}"

        if (( $(echo "$prediction > 0.5" | bc -l) )); then
            osascript -e 'set volume output muted true'
        else
            osascript -e 'set volume output muted false'
        fi
    else
        echo "Ad signal file not found."
    fi

    sleep 1
done
